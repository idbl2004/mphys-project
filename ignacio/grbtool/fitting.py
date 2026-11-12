"""Fit every window with the same models, and choose the model with one rule. The rule is the one from notebook 14."""
import multiprocessing
import signal
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from gdt.core.background.primitives import BackgroundSpectrum
from gdt.core.data_primitives import EnergyBins, Gti
from gdt.core.spectra.fitting import SpectralFitterPstat
from gdt.core.spectra.functions import Band, BlackBody, Comptonized

from sample_tools import FIT_RANGE_BGO, FIT_RANGE_NAI, power_law
from spectral_fit_tools import (BandLowBreak, channel_mask, covariance, cstat, energy_flux, fit_model, flux_with_error, make_spectra,
                                profile_error, scaled_starts, window_response)

FLUX_RANGE = (10.0, 1000.0)     # keV
USABLE_SIGMA = 5.0              # net counts over the square root of the total, summed over detectors
MEASURED_DC = 9.0               # Comptonized against power law, 3 sigma for one extra parameter
TWO_BREAK_DC = 11.8             # two-break against Band, 3 sigma for the extra parameters (notebook 14)
STAGE_MINUTES = 30.0            # for each slow stage of one window: the starts of one model, the error search, the checks
POLISH_SECONDS = 300.0          # the polish gets at least this, even if the starts used up the stage's time

BAND_SHAPES = [[1.0, e, a, b, 100.0] for e in (30.0, 100.0, 300.0, 900.0, 2500.0) for a in (-0.6, -1.2) for b in (-2.6, -4.0, -6.0)]
CPL_SHAPES = [[1.0, e, a, 100.0] for e in (30.0, 100.0, 300.0, 900.0, 2500.0) for a in (-0.6, -1.2)]
PL_SHAPES = [[1.0, a, 100.0] for a in (-1.0, -1.5, -2.0)]
BB_SHAPES = [[1.0, t] for t in (5.0, 15.0, 40.0, 100.0)]


def lowbreak_shapes(epeak):
    return [[1.0, eb, a1, a2, ep, b, 2.0, 100.0] for eb in (20.0, 50.0, 100.0) for a1 in (-0.4, -0.7)
            for a2 in (-1.2, -1.5) for ep in (epeak, 2 * epeak) for b in (-2.5, -3.5)]


def all_spectra(burst, windows):
    """Source and background spectra of every stretch of every window, for each detector."""
    stretches = {(w.label, i): piece for w in windows for i, piece in enumerate(w.pieces)}
    spectra, chi2 = {}, {}
    for det in burst.detectors:
        spectra[det], chi2[det] = make_spectra(burst.data_dir, burst.trigger, det, stretches, burst.bkg_windows, burst.bkg_order)
    return spectra, chi2


def window_data(burst, window, spectra, detectors=None, nai_range=FIT_RANGE_NAI, bgo_range=FIT_RANGE_BGO):
    """Counts, background, response and channels for one window. Stretches of a window add up to one spectrum per detector."""
    phas, baks, rsps, masks = [], [], [], []
    for det in detectors or burst.detectors:
        parts = [spectra[det][window.label, i] for i in range(len(window.pieces))]
        pha, bak = parts[0]
        if len(parts) > 1:
            pha = type(pha).from_data(EnergyBins.sum([p.data for p, _ in parts]), gti=Gti.from_list(list(window.pieces)), trigger_time=pha.trigtime)
            bak = BackgroundSpectrum.sum([b for _, b in parts])
        rsp = window_response(burst.data_dir, burst.trigger, det, window.span)[0]
        phas.append(pha)
        baks.append(bak)
        rsps.append(rsp)
        masks.append(channel_mask(rsp, bgo_range if det.startswith("b") else nai_range))
    return phas, baks, rsps, masks


def errors_of(fit):
    try:
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.diag(covariance(fit)))
    except np.linalg.LinAlgError:
        return np.full(len(fit.parameters), np.nan)


class OutOfTime(Exception):
    """A slow stage used up its time."""


@contextmanager
def time_limit(seconds):
    """Raises OutOfTime after this long. Signals only reach the main thread, which is where each worker runs its fits."""
    def stop(*_):
        raise OutOfTime(f"took more than {seconds:.0f} s")
    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(seconds, 1e-3))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


class Stages:
    """The slow stages of one window, each with its own time: how long each one took, and which ran out."""

    def __init__(self, minutes=STAGE_MINUTES):
        self.seconds, self.took, self.late = 60.0 * minutes, {}, []

    def fit(self, name, data, function, starts):
        """fit_model with the polish, one start at a time, so running out of time keeps the best start so far. Given the time, it's the same fit as fit_model's.

        A window whose E_peak sits at the bottom of the band can use all of the minimiser's iterations on every start, which is about two minutes each (GRB 190829A).
        """
        began = time.monotonic()
        deadline = began + self.seconds
        best, values = None, None
        for start in starts:
            try:
                with time_limit(deadline - time.monotonic()):
                    fit = fit_model(*data, function, [start])
            except OutOfTime:
                self.late.append(name)
                break
            if best is None or fit.statistic < best.statistic:
                best, values = fit, np.array(function.default_values, dtype=float)
        if best is None:
            raise OutOfTime(f"{name}: no start finished in {self.seconds:.0f} s")
        phas, baks, rsps, masks = data
        polisher = SpectralFitterPstat(phas, baks, rsps, channel_masks=masks, method="L-BFGS-B")
        function.default_values = list(values)
        try:
            with warnings.catch_warnings(action="ignore"), time_limit(max(deadline - time.monotonic(), POLISH_SECONDS)):
                polisher.fit(function, options={"maxiter": 5000})
            if polisher.statistic < best.statistic:
                best = polisher
                values[np.array(function.free)] = best.parameters
        except OutOfTime:
            if name not in self.late:
                self.late.append(name)
        function.default_values = list(values)
        self.took[name] = time.monotonic() - began
        return best

    def errors(self, name, data, function, fit, k, **options):
        """The profile-likelihood error, which is NaN on both sides if it runs out of time."""
        began = time.monotonic()
        try:
            with warnings.catch_warnings(action="ignore"), time_limit(self.seconds):
                return profile_error(*data, function, fit, k, **options)
        except OutOfTime:
            self.late.append(name)
            return np.nan, np.nan
        finally:
            self.took[name] = time.monotonic() - began


def fit_all_models(data, stages=None):
    """Band, Comptonized, two-break, power law and blackbody fits to one window. Returns {name: (function, fit)}."""
    stages = stages or Stages()
    band = Band()
    band_fit = stages.fit("band", data, band, scaled_starts(band, *data, BAND_SHAPES))
    cpl = Comptonized()
    cpl_fit = stages.fit("cpl", data, cpl, scaled_starts(cpl, *data, CPL_SHAPES) + [list(band.default_values[:3]) + [100.0]])
    lb = BandLowBreak()
    A, epk, al, be = band_fit.parameters
    # the start from Band goes first, because it's the one to have tried if time runs out
    starts = [[A, 2.0, al, al, epk, min(be, -2.05), 2.0, 100.0]] + scaled_starts(lb, *data, lowbreak_shapes(cpl_fit.parameters[1]))
    lb_fit = stages.fit("two_break", data, lb, starts)
    pl = power_law()
    pl_fit = stages.fit("pl", data, pl, scaled_starts(pl, *data, PL_SHAPES))
    bb = BlackBody()
    bb_fit = stages.fit("bb", data, bb, scaled_starts(bb, *data, BB_SHAPES))
    return dict(band=(band, band_fit), cpl=(cpl, cpl_fit), lowbreak=(lb, lb_fit), pl=(pl, pl_fit), bb=(bb, bb_fit))


def choose_model(fits):
    """Two-break if its break is clean and it beats Band by Delta C > 11.8, Band if the tail is needed by Delta C > 9, otherwise Comptonized.

    A clean break sits above the bottom of the band and below half of E_peak, with its index off the limits and beta off -2.
    11.8 is 3 sigma for the extra parameters, and it's the threshold from notebook 14. Returns the model and the index of E_peak in its free parameters.
    """
    band, band_fit = fits["band"]
    lb, lb_fit = fits["lowbreak"]
    _, cpl_fit = fits["cpl"]
    eb, a1, _, epk_lb, b_lb = lb_fit.parameters[1:6]
    clean = (FIT_RANGE_NAI[0] < eb < 0.5 * epk_lb and lb.min_values[2] + 1e-3 < a1 < lb.max_values[2] - 1e-3
             and b_lb < -2.01 and cstat(band_fit) - cstat(lb_fit) > TWO_BREAK_DC)
    if clean:
        return "lowbreak", 4
    if cstat(cpl_fit) - cstat(band_fit) > MEASURED_DC and band_fit.parameters[3] < -2.01:
        return "band", 1
    return "cpl", 1


def fit_flags(fits, model, late):
    """What the table has to say about the fits. two_break_rejected: a two-break fits much better than Band but isn't a clean break, so the quoted model leaves something undescribed."""
    flags = []
    if model != "lowbreak" and cstat(fits["band"][1]) - cstat(fits["lowbreak"][1]) > TWO_BREAK_DC:
        flags.append("two_break_rejected")
    if late:
        flags.append("out_of_time")
    return " ".join(flags)


NAMES = dict(band="Band", cpl="Comptonized", lowbreak="BandLowBreak")


def fit_window(burst, window, spectra, seed=1, minutes=STAGE_MINUTES):
    """All the fits to one window and the row of the table. Errors come from the profile likelihood, on the chosen model."""
    data = window_data(burst, window, spectra)
    phas, baks, _, _ = data
    gross = sum(float(np.sum(p.data.counts)) for p in phas)
    net = gross - sum(float(np.sum(b.counts)) for b in baks)
    stages = Stages(minutes)
    fits = fit_all_models(data, stages)
    model, k = choose_model(fits)
    function, fit = fits[model]
    extra = [list(function.default_values[:3]) + [b, 100.0] for b in (-3.0, -6.0)] if model == "band" else []
    lo, hi = stages.errors("errors", data, function, fit, k, extra_starts=extra)
    err = errors_of(fit)
    try:
        flux, flux_err = flux_with_error(function, fit, FLUX_RANGE, seed=seed)
    except (np.linalg.LinAlgError, ValueError):     # no usable covariance: the flux is still known, its error isn't
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
    return dict(label=window.label, kind=window.kind, t_start=window.span[0], t_stop=window.span[1], live_duration=window.duration, pieces=len(window.pieces),
                gross_counts=gross, net_counts=net, sigma=sigma, usable=sigma >= USABLE_SIGMA, measured=measured, model=NAMES[model],
                Epeak=fit.parameters[k], Epeak_err_lo=lo, Epeak_err_hi=hi,
                alpha=fit.parameters[2], alpha_err=err[2], beta=fit.parameters[beta] if beta else np.nan, beta_err=err[beta] if beta else np.nan,
                flux=flux, flux_err=flux_err, cstat=cstat(fit), dof=fit.dof,
                Epeak_cpl=cpl_fit.parameters[1], Epeak_cpl_err=errors_of(cpl_fit)[1], Epeak_cpl_limit_lo=limit_lo, Epeak_cpl_limit_hi=limit_hi, alpha_cpl=cpl_fit.parameters[2], C_cpl=cstat(cpl_fit), dof_cpl=cpl_fit.dof,
                Epeak_band=band_fit.parameters[1], Epeak_band_err=errors_of(band_fit)[1], alpha_band=band_fit.parameters[2], beta_band=band_fit.parameters[3], C_band=cstat(band_fit), flux_band=energy_flux(band, list(band.default_values), FLUX_RANGE),
                Epeak_lowbreak=lb_fit.parameters[4], Eb=lb_fit.parameters[1], alpha1_lb=lb_fit.parameters[2], alpha2_lb=lb_fit.parameters[3], beta_lb=lb_fit.parameters[5], C_lowbreak=cstat(lb_fit),
                pl_index=pl_fit.parameters[1], C_pl=cstat(pl_fit), kT=bb_fit.parameters[1], C_bb=cstat(bb_fit), flux_cpl=energy_flux(cpl, list(cpl.default_values), FLUX_RANGE),
                fit_flags=fit_flags(fits, model, stages.late), ran_out=" ".join(stages.late), **{f"seconds_{name}": seconds for name, seconds in stages.took.items()})


_SHARED = {}


def _worker(i):
    window = _SHARED["windows"][i]
    try:
        row = fit_window(_SHARED["burst"], window, _SHARED["spectra"], minutes=_SHARED["minutes"])
        if _SHARED["extras"]:
            row.update(_SHARED["extras"](window, row))
        return row
    except Exception as error:      # one window failing doesn't lose the others, and the table says why
        return dict(label=window.label, kind=window.kind, t_start=window.span[0], t_stop=window.span[1], live_duration=window.duration, fit_failed=f"{type(error).__name__}: {error}")


def fit_windows(burst, windows, workers=1, spectra=None, extras=None, minutes=STAGE_MINUTES):
    """The table of fits. The spectra are built once, and the workers share them by forking. extras(window, row) adds to a row inside its worker."""
    if spectra is None:
        spectra, _ = all_spectra(burst, windows)
    _SHARED.update(burst=burst, windows=windows, spectra=spectra, extras=extras, minutes=minutes)
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            rows = pool.map(_worker, range(len(windows)), chunksize=1)
    else:
        rows = [_worker(i) for i in range(len(windows))]
    table = pd.DataFrame(rows).set_index("label")
    for column in ("Epeak", "model", "measured"):
        if column not in table:     # every window failed
            table[column] = np.nan
    return table
