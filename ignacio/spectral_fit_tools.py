"""
Spectral-fit helpers for the GBM spectra in notebooks 08-15.

Fits use SpectralFitterPstat: Poisson data with the polynomial background held
fixed. GDT's own Hessian gives errors 3-8x too small for these fits (notebook 11),
so the covariance is recomputed here with central differences.
"""
import copy
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from gdt.core.pha import Pha, Bak
from gdt.core.data_primitives import EnergyBins
from gdt.core.binning.unbinned import bin_by_edges
from gdt.core.background.primitives import BackgroundSpectrum
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.response import GbmRsp, GbmRsp2
from gdt.core.spectra.fitting import SpectralFitterPstat
from gdt.core.spectra.functions import Function

from lightcurve_tools import tte_path, fit_background


class BandLowBreak(Function):
    """Band function with a smooth extra break below the peak.

    N(E) = A (E/Epiv)^alpha2 exp(-E/E0) S(E)  below Ec = (alpha2 - beta) E0, E0 = Epeak / (2 + alpha2),
    and the Band power law with slope beta above Ec, continuous at Ec.
    S(E) = [1 + (Eb/E)^s]^((alpha2 - alpha1)/s) turns the slope from alpha1 below Eb into alpha2 above it;
    s sets the sharpness of the break (fixed). With Eb -> 0 or alpha1 = alpha2 it is the Band function.
    """
    nparams = 8
    param_list = [("A", "ph/s/cm^2/keV", "Amplitude"), ("Eb", "keV", "Low-energy break"),
                  ("alpha1", "", "Index below the break"), ("alpha2", "", "Index between break and peak"),
                  ("Epeak", "keV", "SED peak"), ("beta", "", "High-energy index"),
                  ("s", "", "Break sharpness"), ("Epiv", "keV", "Pivot energy")]
    default_values = [0.01, 50.0, -0.67, -1.5, 500.0, -3.0, 2.0, 100.0]
    delta_abs = [0.1] * 8
    delta_rel = [0.01] * 8
    min_values = [1e-10, 1.0, -1.9, -1.99, 1.0, -10.0, 0.5, 0.01]
    max_values = [np.inf, 5000.0, 2.0, 2.0, np.inf, -2.0001, 10.0, np.inf]
    free = [True, True, True, True, True, True, False, False]

    def eval(self, params, x):
        A, Eb, alpha1, alpha2, Epeak, beta, s, Epiv = params
        x = np.asarray(x, dtype=float)
        e0 = Epeak / (2.0 + alpha2)
        ec = (alpha2 - beta) * e0

        def shape(e):
            return (1.0 + (Eb / e) ** s) ** ((alpha2 - alpha1) / s)

        out = np.empty_like(x)
        low = x < ec
        out[low] = A * (x[low] / Epiv) ** alpha2 * np.exp(-x[low] / e0) * shape(x[low])
        out[~low] = (A * (ec / Epiv) ** alpha2 * np.exp(beta - alpha2) * shape(ec)
                     * (x[~low] / ec) ** beta)
        return out


def make_spectra(data_dir, burst_id, det, windows, bkg_windows, order):
    """Source PHA and background spectrum of one detector for each window.

    windows is {label: (t_start, t_stop)}. The background polynomial is fitted
    once and integrated over every window. Returns ({label: (pha, bkg)},
    background reduced chi2).
    """
    fitter, reduced_chi2 = fit_background(data_dir, burst_id, det, bkg_windows, order)
    tte = GbmTte.open(tte_path(data_dir, burst_id, det))
    spectra = {}
    for label, window in windows.items():
        # one bin with exactly these edges, so the exposure is the live time of the window
        pha = tte.to_phaii(bin_by_edges, np.array(window)).to_pha(time_ranges=[window])
        model = fitter.interpolate_bins(np.array([window[0]]), np.array([window[1]]))
        # GDT's own integrate_time would use the exposure of the 1.024 s fit bins,
        # so the background rate is multiplied by the source live time instead
        bkg = BackgroundSpectrum(np.ravel(model.rates), np.ravel(model.rate_uncertainty),
                                 pha.data.lo_edges, pha.data.hi_edges,
                                 np.full(pha.num_chans, pha.exposure))
        spectra[label] = (pha, bkg)
    tte.close()
    return spectra, reduced_chi2


def make_spectrum(data_dir, burst_id, det, window, bkg_windows, order):
    """Source PHA, background spectrum and background reduced chi2 for one window."""
    spectra, reduced_chi2 = make_spectra(data_dir, burst_id, det, {"window": window}, bkg_windows, order)
    pha, bkg = spectra["window"]
    return pha, bkg, reduced_chi2


def rsp2_path(data_dir, burst_id, det):
    """Newest CSPEC response file (.rsp2) for one detector."""
    return sorted(Path(data_dir).glob(f"glg_cspec_{det}_{burst_id}_v*.rsp2"))[-1]


def write_rsp(rsp, directory, filename):
    """Write a single-DRM response so that it reads back exactly.

    GDT 2.2.3 writes the full matrix but keeps the F_CHAN/N_CHAN row packing of
    the file the DRM came from, so rows packed differently there are read back
    shifted. Declaring one group per row, covering every channel, fixes that.
    """
    n_rows = rsp.num_ebins
    rsp._ngrp = np.ones(n_rows, dtype=np.int16)
    rsp._fchan = [np.array([1], dtype=np.int16)] * n_rows
    rsp._nchan = [np.array([rsp.num_chans], dtype=np.int16)] * n_rows
    rsp.write(directory, filename=filename, overwrite=True)


def window_response(data_dir, burst_id, det, window, bin_width=0.256):
    """Response for a time window, and how it was made.

    The .rsp2 file has one DRM every ~40 s. If the window sits inside one of
    them, that DRM is used. If it crosses a boundary, each 0.256 s bin of the
    window takes its nearest DRM, weighted by the counts in the bin (GDT's
    Rsp2.weighted, which GDT suggests for long accumulations).
    """
    rsp2 = GbmRsp2.open(rsp2_path(data_dir, burst_id, det))
    covers = np.flatnonzero((np.asarray(rsp2.tstart) <= window[0]) & (np.asarray(rsp2.tstop) >= window[1]))
    if covers.size:
        rsp, how = rsp2.extract_drm(index=covers[0]), "single DRM"
    else:
        edges = np.append(np.arange(window[0], window[1], bin_width), window[1])
        tte = GbmTte.open(tte_path(data_dir, burst_id, det))
        lightcurve = tte.to_phaii(bin_by_edges, edges).to_lightcurve()
        tte.close()
        rsp, how = rsp2.weighted(lightcurve), "counts-weighted"
    rsp2.close()
    return rsp, how


def load_spectra(spectrum_dir, burst_id, detectors, label="precursor"):
    """PHA, BAK and RSP files written by notebook 08."""
    stems = [spectrum_dir / f"{burst_id}_{det}_{label}" for det in detectors]
    return ([Pha.open(f"{s}.pha") for s in stems],
            [Bak.open(f"{s}.bak") for s in stems],
            [GbmRsp.open(f"{s}.rsp") for s in stems])


def channel_mask(rsp, energy_range):
    """Channels overlapping energy_range, without channel 0 and the overflow channel."""
    lo = np.asarray(rsp.ebounds.low_edges())
    hi = np.asarray(rsp.ebounds.high_edges())
    mask = (hi > energy_range[0]) & (lo < energy_range[1])
    mask[0] = mask[-1] = False
    return mask


def cstat(fitter):
    """C-stat as RMfit/XSPEC quote it; GDT stores -ln L."""
    return 2.0 * fitter.statistic


def fit_model(phas, baks, rsps, masks, function, starts,
              statistic=SpectralFitterPstat, method="SLSQP", polish=False):
    """Best of several fits started from `starts` (full parameter vectors).

    With polish=True the best fit is restarted with L-BFGS-B and kept if that
    lowers the C-stat: SLSQP can report success after a few steps when the
    gradient at the start is very large. The best-fit values are left in
    function.default_values.
    """
    free = np.array(function.free)
    best = None
    for start in starts:
        fitter = statistic(phas, baks, rsps, channel_masks=masks, method=method)
        function.default_values = list(start)
        with warnings.catch_warnings(action="ignore"):
            fitter.fit(function, options={"maxiter": 2000})
        if best is None or fitter.statistic < best.statistic:
            best, best_full = fitter, np.array(start, dtype=float)
    best_full[free] = best.parameters

    if polish:
        fitter = statistic(phas, baks, rsps, channel_masks=masks, method="L-BFGS-B")
        function.default_values = list(best_full)
        with warnings.catch_warnings(action="ignore"):
            fitter.fit(function, options={"maxiter": 5000})
        if fitter.statistic < best.statistic:
            best = fitter
            best_full[free] = best.parameters
    function.default_values = list(best_full)
    return best


def scaled_starts(function, phas, baks, rsps, masks, shapes):
    """Starting vectors whose amplitude (first parameter) reproduces the observed net counts.

    shapes are full parameter vectors; only their amplitude is replaced.
    Starting from the right normalisation keeps the minimiser away from the
    huge gradients a wrong amplitude gives.
    """
    net, starts = 0.0, []
    for pha, bak, mask in zip(phas, baks, masks):
        bkg_counts = bak.data.counts if isinstance(bak, Bak) else bak.counts
        net += np.sum((np.asarray(pha.data.counts) - np.asarray(bkg_counts))[mask])
    for shape in shapes:
        unit = [1.0] + list(shape[1:])
        model = sum(np.sum(np.asarray(rsp.drm.fold_spectrum(function.eval, unit))[mask]) * pha.exposure
                    for pha, rsp, mask in zip(phas, rsps, masks))
        starts.append([max(net, 1.0) / model] + list(shape[1:]))
    return starts


def covariance(fitter, step=1.0):
    """Covariance of the free parameters from a central-difference Hessian of -2 ln L.

    Steps are `step` times each parameter's conditional error (the diagonal of
    GDT's Hessian, which is fine); the result is stable for steps of 0.3-1.
    A step is capped at 10% of the parameter's value, so a parameter the data
    barely constrain (beta in faint windows) isn't stepped out of its range.
    """
    x = np.asarray(fitter.parameters, dtype=float)
    h = np.minimum(step / np.sqrt(np.abs(np.diag(fitter.hessian))), 0.1 * np.abs(x) + 1e-6)

    def m2lnl(params):
        return -2.0 * fitter._fold_model(fitter._function.fit_eval, params)

    n = x.size
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            corners = []
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                y = x.copy()
                y[i] += si * h[i]
                y[j] += sj * h[j]
                corners.append(m2lnl(y))
            hess[i, j] = hess[j, i] = (corners[0] - corners[1] - corners[2] + corners[3]) / (4 * h[i] * h[j])
    return np.linalg.inv(hess / 2.0)     # Fisher matrix = Hessian of -ln L


def profile_error(phas, baks, rsps, masks, function, fitter, index, reach=10.0, delta=1.0,
                  extra_starts=()):
    """(-, +) profile-likelihood error on free parameter `index`, others refitted.

    delta = 1 gives the 68% interval; delta = 2.71 gives a one-sided 95% limit.
    A side that doesn't reach delta within `reach` sigma (or the parameter
    bound) comes back as NaN: the data don't constrain it. Each refit starts
    from the best fit and from extra_starts (full vectors), so it can't stall
    in a local minimum that the full fit avoided.
    """
    best = list(function.default_values)
    k = np.flatnonzero(function.free)[index]
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(covariance(fitter)[index, index])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.1 * abs(best[k]) + 0.1          # covariance unusable here: search on a fixed scale
    c_min = cstat(fitter)

    def delta_c(value):
        trial = copy.deepcopy(function)
        trial.free = [f and i != k for i, f in enumerate(function.free)]
        starts = [list(best)] + [list(s) for s in extra_starts]
        for s in starts:
            s[k] = value
        c = cstat(fit_model(phas, baks, rsps, masks, trial, starts))
        if c < c_min - 0.01:
            warnings.warn(f"profiling found a fit {c_min - c:.2f} better than the best fit")
        return c - c_min - delta

    errors = []
    for sign, bound in ((-1, function.min_values[k]), (1, function.max_values[k])):
        edge = best[k] + sign * reach * sigma
        edge = max(edge, bound) if sign < 0 else min(edge, bound)
        if delta_c(edge) < 0:
            errors.append(np.nan)                    # not constrained within `reach` sigma
        else:
            root = brentq(delta_c, best[k], edge, xtol=1e-3 * sigma)
            errors.append(abs(root - best[k]))
    return tuple(errors)


def energy_flux(function, params, erange=(10.0, 1000.0)):
    """Energy flux in erg/cm2/s (GDT converts keV to erg)."""
    return function.integrate(params, erange, energy=True)


def flux_with_error(function, fitter, erange=(10.0, 1000.0), num_samples=2000, seed=1):
    """Flux and its error, propagated by sampling the parameters from covariance()."""
    rng = np.random.default_rng(seed)
    best = np.asarray(fitter.parameters, dtype=float)
    samples = rng.multivariate_normal(best, covariance(fitter), size=num_samples)
    free = np.array(function.free)
    inside = np.all((samples > np.array(function.min_values)[free])
                    & (samples < np.array(function.max_values)[free]), axis=1)
    fluxes = [energy_flux(function, s, erange) for s in samples[inside]]
    return energy_flux(function, best, erange), np.std(fluxes)


def simulate_pha(pha, bak, rsp, function, params, rng):
    """Poisson realisation of model (full parameter vector) + background, for checks only."""
    expected = np.asarray(rsp.drm.fold_spectrum(function.eval, params)) * pha.exposure \
        + np.asarray(bak.data.counts)
    fake = EnergyBins(rng.poisson(expected), pha.data.lo_edges, pha.data.hi_edges, pha.exposure)
    return Pha.from_data(fake, gti=pha.gti, trigger_time=pha.trigtime)
