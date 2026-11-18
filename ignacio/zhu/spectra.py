"""The E_peak of each episode measured two ways on the same spectra: their five models chosen by BIC, and our rule.

Their way follows Zhu et al. 2025 (section 2.3) and 2026 (section 2.2). NaI 10-900 keV without 30-40 keV, around the iodine K edge at
33 keV. BGO 0.3-40 MeV, or 0.3-10 MeV for a precursor. PGSTAT. Five models: Band, CPL, Band+BB, CPL+BB and PL+BB. The chosen model is the
one with the lowest BIC = -2 ln L + k ln N (k free parameters, N channels) among those "with all parameters well constrained". They sample
each fit with MCMC in XSPEC. I fit each model from many starts instead (the same minimiser as the rest of the project, with PGSTAT guarded
so it can't return NaN, see safe_pgstat), and give the chosen model's E_peak a profile-likelihood error. I read "well constrained" as:
every free parameter has a finite error from the covariance and sits away from its limits, and the amplitudes, E_peak and kT are known
to better than 100%.

Our way is grbtool.fitting.fit_window, the rule from notebook 14: Comptonized, Band or a two-break Band, chosen by Delta C, with NaI
8-900 keV, BGO 0.2-40 MeV and PSTAT. It runs as fit_window_ours, which is the same steps except that an error search that stops without an
answer gives a NaN error and a flag instead of losing the whole window (this happened in GRB 090618's main emission). Both ways run in the
same worker on the same source and background spectra, so a difference between the two E_peaks comes from the model choice and the
energy range, not from the data."""
import contextlib
import copy
import multiprocessing
import time
import warnings

import numpy as np
import pandas as pd
from gdt.core.spectra.fitting import SpectralFitter, SpectralFitterPgstat
from gdt.core.spectra.functions import Band, BlackBody, Comptonized, PowerLaw
from scipy.optimize import brentq

from grbtool.fitting import (BAND_SHAPES, CPL_SHAPES, FLUX_RANGE, MEASURED_DC, NAMES, PL_SHAPES, USABLE_SIGMA, OutOfTime, Stages,
                             all_spectra, choose_model, errors_of, fit_all_models, fit_flags, time_limit, window_data)
from spectral_fit_tools import channel_mask, covariance, cstat, energy_flux, fit_model, flux_with_error, scaled_starts
from .io import worker_start

SEARCH_FAILURES = (ValueError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError)

NAI_RANGE = (10.0, 900.0)           # keV
K_EDGE = (30.0, 40.0)               # keV, left out of the NaI spectra
BGO_RANGE = (300.0, 40000.0)
BGO_RANGE_PE = (300.0, 10000.0)
BB_SHARE = 0.1                      # a blackbody added to a fit starts with this share of the counts
MODEL_MINUTES = 5.0                 # for each model of one window
ERROR_MINUTES = 10.0                # for the profile error of the chosen E_peak
MODELS = ["band", "cpl", "band_bb", "cpl_bb", "pl_bb"]
EPEAK_INDEX = {"band": 1, "cpl": 1, "band_bb": 1, "cpl_bb": 1, "pl_bb": None}      # among the free parameters
SCALE_PARAMETERS = {"band": [0, 1], "cpl": [0, 1], "band_bb": [0, 1, 4, 5], "cpl_bb": [0, 1, 3, 4], "pl_bb": [0, 2, 3]}


TINY = 1e-12


def safe_pgstat(obs_counts, mod_rates, exposure, back_counts, back_var, back_exp):
    """GDT's pgstat (the profile Gaussian likelihood of the XSPEC manual), guarded so it never gives NaN.

    PGSTAT treats the background in each channel as Gaussian, with the variance from its fit, and profiles over it. Where that variance is
    much bigger than the background counts, as in the sparse high-energy channels after a cubic fit interpolated across a long burst (GRB
    090618's main emission has 25 counts of uncertainty on 8), the square root in the profiled background can go negative for some model
    values. GDT then returns NaN, which also stops the minimiser from comparing its starts. So I floor the square root's argument at zero and
    the logarithm's at TINY. Where GDT's value is finite the two are identical, and tests/test_zhu.py checks that."""
    mask = obs_counts > 0
    pg = np.zeros_like(obs_counts, dtype=float)
    pg[~mask] = exposure * mod_rates[~mask] + back_counts[~mask] * (exposure / back_exp) - back_var[~mask] * (exposure / back_exp) ** 2 / 2.0
    n, m, b, v = obs_counts[mask], mod_rates[mask], back_counts[mask], back_var[mask]
    q = exposure * v - back_exp * b + back_exp**2 * m
    r = exposure * v * m - n * v - back_exp * b * m
    d = np.sqrt(np.maximum(q**2 - 4.0 * back_exp**2 * r, 0.0))
    d[q < 0.0] *= -1.0
    f = (-q + d) / (2.0 * back_exp**2)
    denominator = -q + d
    f_alt = 2.0 * r / np.where(np.abs(denominator) > TINY, denominator, TINY)
    f = np.where(f < 0.0, f_alt, f)
    pg[mask] = (exposure * (m + f) - n * np.log(np.maximum(exposure * m + exposure * f, TINY)) + (b - back_exp * f) ** 2 / (2.0 * v)
                - n * (1.0 - np.log(n)))
    return -pg.sum()


class SafePgstatFitter(SpectralFitterPgstat):
    """SpectralFitterPgstat with safe_pgstat as its statistic."""

    def __init__(self, pha_list, bkgd_list, rsp_list, back_exp=None, **kwargs):
        SpectralFitter.__init__(self, pha_list, bkgd_list, rsp_list, safe_pgstat, **kwargs)
        self._back_exp = self._exposure if back_exp is None else back_exp


def zhu_data(burst, window, spectra, precursor=False):
    """Counts, background, responses and their channels for one window."""
    phas, baks, rsps, masks = window_data(burst, window, spectra, nai_range=NAI_RANGE, bgo_range=BGO_RANGE_PE if precursor else BGO_RANGE)
    dets = burst.detectors
    masks = [m & ~channel_mask(r, K_EDGE) if d.startswith("n") else m for m, r, d in zip(masks, rsps, dets)]
    return phas, baks, rsps, masks


def _fit(data, function, starts, seconds):
    """fit_model with PGSTAT and the polish, within a time limit. Leaves the best values in function.default_values."""
    with warnings.catch_warnings(action="ignore"), time_limit(seconds):
        return fit_model(*data, function, starts, statistic=SafePgstatFitter, polish=True)


def _with_blackbody(base_values, bb_amplitude):
    return [list(base_values) + [BB_SHARE * bb_amplitude, kt] for kt in (5.0, 15.0, 40.0, 100.0)]


def constrained(function, fit, scale):
    """Our reading of "all parameters well constrained": finite errors, away from the limits, scale parameters to better than 100%."""
    err = errors_of(fit)
    values = np.asarray(fit.parameters, float)
    free = np.flatnonzero(function.free)
    lo = np.asarray(function.min_values, float)[free]
    hi = np.asarray(function.max_values, float)[free]
    with np.errstate(invalid="ignore"):
        above_lo = np.where(np.isfinite(lo), values - lo > 1e-3 * (np.abs(lo) + 1.0), True)
        below_hi = np.where(np.isfinite(hi), hi - values > 1e-3 * (np.abs(hi) + 1.0), True)
    off_limits = above_lo & below_hi
    ok = np.isfinite(err) & (err > 0) & off_limits
    ok[scale] &= err[scale] < np.abs(values[scale])
    return bool(ok.all())


def profile_error_pg(data, function, fit, index, reach=10.0, delta=1.0, seconds=None):
    """(-, +) profile-likelihood error of free parameter `index`, with PGSTAT (spectral_fit_tools.profile_error does the same with PSTAT).
    Each side is searched on its own, with half the time each if seconds is given, so a side that fails or runs out of time still leaves the
    other side's error. Returns (minus, plus, notes), with NaN for a side that has no answer and a note such as
    "error_search_failed:minus:ValueError"."""
    best = list(function.default_values)
    k = np.flatnonzero(function.free)[index]
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(covariance(fit)[index, index])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.1 * abs(best[k]) + 0.1
    c_min = 2.0 * fit.statistic

    def delta_c(value):
        trial = copy.deepcopy(function)
        trial.free = [f and i != k for i, f in enumerate(function.free)]
        start = list(best)
        start[k] = value
        with warnings.catch_warnings(action="ignore"):
            return 2.0 * fit_model(*data, trial, [start], statistic=SafePgstatFitter).statistic - c_min - delta

    errors, notes = [], []
    for side, sign, bound in (("minus", -1, function.min_values[k]), ("plus", 1, function.max_values[k])):
        edge = best[k] + sign * reach * sigma
        edge = max(edge, bound) if sign < 0 else min(edge, bound)
        try:
            with (time_limit(seconds / 2) if seconds else contextlib.nullcontext()):
                if delta_c(edge) < 0:
                    errors.append(np.nan)          # the statistic doesn't rise by delta before the edge: no error on this side
                    notes.append(f"error_open:{side}")
                else:
                    errors.append(abs(brentq(delta_c, best[k], edge, xtol=1e-3 * sigma) - best[k]))
        except OutOfTime:
            errors.append(np.nan)
            notes.append(f"error_out_of_time:{side}")
        except SEARCH_FAILURES as error:
            errors.append(np.nan)
            notes.append(f"error_search_failed:{side}:{type(error).__name__}")
    return errors[0], errors[1], notes


def fit_zhu(burst, window, spectra, precursor=False, minutes=MODEL_MINUTES):
    """Their five models on one window, and the choice between them. Returns one row: every model's statistic, BIC and E_peak, and the choice."""
    data = zhu_data(burst, window, spectra, precursor)
    n_channels = int(sum(int(np.sum(m)) for m in data[3]))
    seconds = 60.0 * minutes
    row, fits, late = {"zhu_channels": n_channels}, {}, []

    def attempt(name, function, starts):
        try:
            fits[name] = (function, _fit(data, function, starts, seconds))
        except OutOfTime:
            late.append(name)

    attempt("cpl", Comptonized(), scaled_starts(Comptonized(), *data, CPL_SHAPES))
    band_starts = scaled_starts(Band(), *data, BAND_SHAPES)
    if "cpl" in fits:
        band_starts = [list(fits["cpl"][0].default_values[:3]) + [-3.0, 100.0]] + band_starts
    attempt("band", Band(), band_starts)
    attempt("pl", PowerLaw(), scaled_starts(PowerLaw(), *data, PL_SHAPES))
    bb_amplitude = scaled_starts(BlackBody(), *data, [[1.0, 30.0]])[0][0]
    for name, base in (("band_bb", "band"), ("cpl_bb", "cpl"), ("pl_bb", "pl")):
        if base in fits:
            function = (Band() if base == "band" else Comptonized() if base == "cpl" else PowerLaw()) + BlackBody()
            attempt(name, function, _with_blackbody(fits[base][0].default_values, bb_amplitude))

    candidates = []
    for name in MODELS:
        if name not in fits:
            continue
        function, fit = fits[name]
        k = int(np.sum(function.free))
        stat = 2.0 * fit.statistic
        bic = stat + k * np.log(n_channels)
        ok = constrained(function, fit, SCALE_PARAMETERS[name])
        i = EPEAK_INDEX[name]
        row.update({f"zhu_{name}_stat": stat, f"zhu_{name}_bic": bic, f"zhu_{name}_constrained": ok,
                    f"zhu_{name}_Epeak": fit.parameters[i] if i is not None else np.nan})
        if name.endswith("_bb"):
            row[f"zhu_{name}_kT"] = fit.parameters[-1]
        candidates.append((bic, name, ok))
    flags = [f"out_of_time:{name}" for name in late]
    usable = [c for c in candidates if c[2]]
    if not usable:
        flags.append("no_model_fully_constrained")
        usable = candidates
    if not usable:
        row.update(zhu_model=None, zhu_Epeak=np.nan, zhu_Epeak_err_lo=np.nan, zhu_Epeak_err_hi=np.nan, zhu_flags=" ".join(flags + ["no_fit"]))
        return row
    usable.sort()
    _, chosen, _ = usable[0]
    row["zhu_model"] = chosen
    row["zhu_dbic_next"] = usable[1][0] - usable[0][0] if len(usable) > 1 else np.nan
    function, fit = fits[chosen]
    i = EPEAK_INDEX[chosen]
    if i is None:
        # PL+BB has no E_peak of its own; the best model with one is quoted beside it
        flags.append("chosen_model_has_no_epeak")
        with_peak = [c for c in usable if EPEAK_INDEX[c[1]] is not None]
        if not with_peak:
            row.update(zhu_Epeak=np.nan, zhu_Epeak_err_lo=np.nan, zhu_Epeak_err_hi=np.nan, zhu_flags=" ".join(flags))
            return row
        _, chosen, _ = with_peak[0]
        function, fit = fits[chosen]
        i = EPEAK_INDEX[chosen]
        row["zhu_model_with_epeak"] = chosen
    function.default_values = list(function.default_values)
    lo, hi, notes = profile_error_pg(data, function, fit, i, seconds=60.0 * ERROR_MINUTES)
    flags += notes
    k = int(np.sum(function.free))
    row.update(zhu_Epeak=fit.parameters[i], zhu_Epeak_err_lo=lo, zhu_Epeak_err_hi=hi, zhu_flags=" ".join(flags),
               zhu_parameters=" ".join(f"{v:.6g}" for v in fit.parameters), zhu_stat=2.0 * fit.statistic, zhu_dof=n_channels - k)
    return row


class SafeStages(Stages):
    """grbtool's Stages, except that an error search that stops (the root finder given no change of sign, or a fit that returns NaN)
    gives NaN on both sides and is noted, instead of stopping the whole window. Everything else is the same."""

    def __init__(self, minutes):
        super().__init__(minutes)
        self.failed = []

    def errors(self, name, data, function, fit, k, **options):
        try:
            return super().errors(name, data, function, fit, k, **options)
        except SEARCH_FAILURES as error:
            self.failed.append(f"{name}:{type(error).__name__}")
            return np.nan, np.nan


def fit_window_ours(burst, window, spectra, seed=1, minutes=5.0):
    """Our rule on one window: grbtool.fitting.fit_window line for line, with SafeStages in place of Stages, so a failed error search
    leaves the fits and gives a NaN error with the flag error_search_failed. Returns the same row as fit_window."""
    data = window_data(burst, window, spectra)
    phas, baks, _, _ = data
    gross = sum(float(np.sum(p.data.counts)) for p in phas)
    net = gross - sum(float(np.sum(b.counts)) for b in baks)
    stages = SafeStages(minutes)
    fits = fit_all_models(data, stages)
    model, k = choose_model(fits)
    function, fit = fits[model]
    extra = [list(function.default_values[:3]) + [b, 100.0] for b in (-3.0, -6.0)] if model == "band" else []
    lo, hi = stages.errors("errors", data, function, fit, k, extra_starts=extra)
    err = errors_of(fit)
    try:
        flux, flux_err = flux_with_error(function, fit, FLUX_RANGE, seed=seed)
    except (np.linalg.LinAlgError, ValueError):
        flux, flux_err = energy_flux(function, fit.parameters, FLUX_RANGE), np.nan
    cpl, cpl_fit = fits["cpl"]
    band, band_fit = fits["band"]
    lb, lb_fit = fits["lowbreak"]
    pl, pl_fit = fits["pl"]
    bb, bb_fit = fits["bb"]
    sigma = net / np.sqrt(max(gross, 1.0))
    measured = sigma >= USABLE_SIGMA and cstat(pl_fit) - cstat(cpl_fit) > MEASURED_DC
    limit_lo, limit_hi = np.nan, np.nan
    if not measured:
        down, up = stages.errors("limits", data, cpl, cpl_fit, 1, reach=40.0, delta=2.71)
        limit_lo, limit_hi = cpl_fit.parameters[1] - down, cpl_fit.parameters[1] + up
    beta = {"lowbreak": 5, "band": 3}.get(model)
    flags = fit_flags(fits, model, stages.late)
    if stages.failed:
        flags = (flags + " " if flags else "") + " ".join(f"error_search_failed:{f}" for f in stages.failed)
    return dict(label=window.label, kind=window.kind, t_start=window.span[0], t_stop=window.span[1], live_duration=window.duration, pieces=len(window.pieces),
                gross_counts=gross, net_counts=net, sigma=sigma, usable=sigma >= USABLE_SIGMA, measured=measured, model=NAMES[model],
                Epeak=fit.parameters[k], Epeak_err_lo=lo, Epeak_err_hi=hi,
                alpha=fit.parameters[2], alpha_err=err[2], beta=fit.parameters[beta] if beta else np.nan, beta_err=err[beta] if beta else np.nan,
                flux=flux, flux_err=flux_err, cstat=cstat(fit), dof=fit.dof,
                Epeak_cpl=cpl_fit.parameters[1], Epeak_cpl_err=errors_of(cpl_fit)[1], Epeak_cpl_limit_lo=limit_lo, Epeak_cpl_limit_hi=limit_hi, alpha_cpl=cpl_fit.parameters[2], C_cpl=cstat(cpl_fit), dof_cpl=cpl_fit.dof,
                Epeak_band=band_fit.parameters[1], Epeak_band_err=errors_of(band_fit)[1], alpha_band=band_fit.parameters[2], beta_band=band_fit.parameters[3], C_band=cstat(band_fit), flux_band=energy_flux(band, list(band.default_values), FLUX_RANGE),
                Epeak_lowbreak=lb_fit.parameters[4], Eb=lb_fit.parameters[1], alpha1_lb=lb_fit.parameters[2], alpha2_lb=lb_fit.parameters[3], beta_lb=lb_fit.parameters[5], C_lowbreak=cstat(lb_fit),
                pl_index=pl_fit.parameters[1], C_pl=cstat(pl_fit), kT=bb_fit.parameters[1], C_bb=cstat(bb_fit), flux_cpl=energy_flux(cpl, list(cpl.default_values), FLUX_RANGE),
                fit_flags=flags, ran_out=" ".join(stages.late), **{f"seconds_{name}": seconds for name, seconds in stages.took.items()})


_SHARED = {}


def metadata(burst, window, ways):
    """What a fit row needs to be reproduced: the data, the energy ranges and channels, the statistics, and the code."""
    from sample_tools import FIT_RANGE_BGO, FIT_RANGE_NAI
    from .io import code_version
    pe = window.label == "PE"
    out = dict(detectors=" ".join(burst.detectors), pieces_s=" ".join(f"{a:.4f}:{b:.4f}" for a, b in window.pieces), code_version=code_version())
    if "ours" in ways:
        out.update(ours_statistic="PSTAT", ours_nai_keV=f"{FIT_RANGE_NAI[0]:g}-{FIT_RANGE_NAI[1]:g}", ours_bgo_keV=f"{FIT_RANGE_BGO[0]:g}-{FIT_RANGE_BGO[1]:g}",
                   ours_excluded="channel 0 and the overflow channel")
    if "zhu" in ways:
        bgo = BGO_RANGE_PE if pe else BGO_RANGE
        out.update(zhu_statistic="PGSTAT (guarded)", zhu_nai_keV=f"{NAI_RANGE[0]:g}-{NAI_RANGE[1]:g}", zhu_bgo_keV=f"{bgo[0]:g}-{bgo[1]:g}",
                   zhu_excluded=f"NaI {K_EDGE[0]:g}-{K_EDGE[1]:g} keV, channel 0 and the overflow channel")
    return out


def _worker(i):
    """The chosen ways on one window, inside a forked worker. A failure in one way doesn't lose the other."""
    burst, window, spectra, ways = _SHARED["burst"], _SHARED["windows"][i], _SHARED["spectra"], _SHARED["ways"]
    row = dict(label=window.label, t_start=window.span[0], t_stop=window.span[1], pieces=len(window.pieces), **metadata(burst, window, ways))
    if "ours" in ways:
        began = time.monotonic()
        try:
            ours = fit_window_ours(burst, window, spectra, minutes=_SHARED["minutes"])
            row.update({k: v for k, v in ours.items() if k not in row})
            row["fluence"] = row["flux"] * row["live_duration"] if np.isfinite(row.get("flux", np.nan)) else np.nan
        except Exception as error:
            row["ours_failed"] = f"{type(error).__name__}: {error}"
        row["seconds_ours"] = time.monotonic() - began
    if "zhu" in ways:
        began = time.monotonic()
        try:
            row.update(fit_zhu(burst, window, spectra, precursor=window.label == "PE", minutes=min(MODEL_MINUTES, _SHARED["minutes"])))
        except Exception as error:
            row["zhu_failed"] = f"{type(error).__name__}: {error}"
        row["seconds_zhu"] = time.monotonic() - began
    return row


def fit_episodes(burst, windows, workers=1, minutes=10.0, ways=("ours", "zhu")):
    """The table of fits for every window (label PE, ME, EE or WE), our rule and/or their way. The spectra are built once and shared with
    the workers by forking."""
    spectra, _ = all_spectra(burst, windows)
    _SHARED.update(burst=burst, windows=windows, spectra=spectra, minutes=minutes, ways=tuple(ways))
    if workers > 1 and len(windows) > 1:
        with multiprocessing.get_context("fork").Pool(min(workers, len(windows)), initializer=worker_start) as pool:
            rows = pool.map(_worker, range(len(windows)), chunksize=1)
    else:
        rows = [_worker(i) for i in range(len(windows))]
    return pd.DataFrame(rows).set_index("label")
