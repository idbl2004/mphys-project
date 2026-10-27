"""
Shared functions for fitting the GBM count spectra made in 08_precursor_spectrum.ipynb.

Three things here differ from a fit done with GDT's defaults; the evidence for
each is in 09_precursor_band_fit.ipynb and 11_robustness_checks.ipynb:

* The statistic is GDT's SpectralFitterPstat (Poisson data, background model
  taken as known). With it the fit reproduces Dichiara's RMfit C-stat values;
  PG-stat gives the same parameters. GDT's SpectralFitterCstat is XSPEC's
  W-stat, which treats the background as a second Poisson measurement.
* GDT's default Band amplitude (0.01) is far below the best fit (1.6), and from
  there the minimiser can stop early, so every fit is restarted from several
  points and the best one kept.
* GDT's Hessian, and so its symmetric errors, came out ~5x too small here.
  In GDT 2.2.3 the nested finite differences in _mixed_pderiv share one
  parameter array, which shifts the off-diagonal terms (shown on a toy
  function in notebook 11). covariance() recomputes the Hessian with central
  differences; profile_error() gives profile-likelihood errors. Both agree
  with the scatter of simulated fits.
"""
import warnings

import numpy as np
from scipy.optimize import brentq
from gdt.core.pha import Pha, Bak
from gdt.core.data_primitives import EnergyBins
from gdt.core.binning.unbinned import bin_by_edges
from gdt.core.background.primitives import BackgroundSpectrum
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.missions.fermi.gbm.response import GbmRsp
from gdt.core.spectra.fitting import SpectralFitterPstat

from lightcurve_tools import tte_path, fit_background


def load_spectra(spectrum_dir, burst_id, detectors, label="precursor"):
    """PHA, BAK and RSP objects written by 08_precursor_spectrum.ipynb."""
    stems = [spectrum_dir / f"{burst_id}_{det}_{label}" for det in detectors]
    phas = [Pha.open(f"{s}.pha") for s in stems]
    baks = [Bak.open(f"{s}.bak") for s in stems]
    rsps = [GbmRsp.open(f"{s}.rsp") for s in stems]
    return phas, baks, rsps


def channel_mask(rsp, energy_range):
    """Channels to fit: every channel that overlaps energy_range.

    A channel is kept if any part of it is inside the range (as slice_energy
    does), so the real range is a little wider than the numbers given.
    Channel 0 and the overflow channel (last) are never real energy bins.
    """
    lo = np.asarray(rsp.ebounds.low_edges())
    hi = np.asarray(rsp.ebounds.high_edges())
    mask = (hi > energy_range[0]) & (lo < energy_range[1])
    mask[0] = mask[-1] = False
    return mask


def cstat(fitter):
    """Fit statistic on the scale RMfit / XSPEC print (GDT stores -ln L, not -2 ln L)."""
    return 2.0 * fitter.statistic


def fit_model(phas, baks, rsps, masks, function, starts):
    """Fit one model from several starting points; return the best fitter.

    starts is a list of full parameter vectors (fixed parameters included).
    """
    best = None
    for start in starts:
        fitter = SpectralFitterPstat(phas, baks, rsps, channel_masks=masks, method="SLSQP")
        function.default_values = list(start)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitter.fit(function, options={"maxiter": 2000})
        if best is None or fitter.statistic < best.statistic:
            best = fitter
    # leave the function holding the best-fit values, so it can be reused as a start
    full = np.array(function.default_values, dtype=float)
    full[np.array(function.free)] = best.parameters
    function.default_values = list(full)
    return best


def minus_two_log_like(fitter, free_params):
    """-2 ln L (the C-stat) at any free-parameter vector, without refitting."""
    return -2.0 * fitter._fold_model(fitter._function.fit_eval, np.asarray(free_params))


def covariance(fitter, step_fraction=1.0):
    """Covariance of the free parameters from a central-difference Hessian of -2 ln L.

    The step for each parameter is step_fraction times its conditional error
    (from GDT's diagonal, which is correct); results are stable for 0.3-1.
    """
    x = np.asarray(fitter.parameters, dtype=float)
    h = step_fraction / np.sqrt(np.abs(np.diag(fitter.hessian)))
    n = x.size
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            def shifted(si, sj):
                y = x.copy()
                y[i] += si * h[i]
                y[j] += sj * h[j]
                return minus_two_log_like(fitter, y)
            hess[i, j] = hess[j, i] = (shifted(1, 1) - shifted(1, -1)
                                       - shifted(-1, 1) + shifted(-1, -1)) / (4 * h[i] * h[j])
    # Fisher matrix is half the Hessian of -2 ln L
    return np.linalg.inv(hess / 2.0)


def profile_error(phas, baks, rsps, masks, function, fitter, index, delta=1.0, reach=10.0):
    """(minus, plus) 1-sigma error on free parameter `index` from the profile likelihood.

    The parameter is fixed at trial values and all the others are refitted; the
    error is where the C-stat has risen by `delta` (1.0 = 68% for one parameter).
    It can be asymmetric. For the Band fit it matches Dichiara's quoted errors.
    """
    free_idx = np.flatnonzero(function.free)[index]
    best_full = np.array(function.default_values, dtype=float)
    c_min = cstat(fitter)
    sigma = np.sqrt(covariance(fitter)[index, index])
    lo_bound, hi_bound = function.min_values[free_idx], function.max_values[free_idx]

    def rise(value):
        free = list(function.free)
        start = best_full.copy()
        start[free_idx] = value
        function.free = [f and k != free_idx for k, f in enumerate(free)]
        try:
            refit = fit_model(phas, baks, rsps, masks, function, [start])
        finally:
            function.free = free
            function.default_values = list(best_full)
        return cstat(refit) - c_min - delta

    centre = best_full[free_idx]
    errors = []
    for direction, bound in ((-1, lo_bound), (1, hi_bound)):
        edge = centre + direction * reach * sigma
        edge = max(edge, bound) if direction < 0 else min(edge, bound)
        if rise(edge) < 0:
            errors.append(np.nan)      # not constrained within reach
            continue
        root = brentq(rise, centre, edge, xtol=1e-3 * sigma)
        errors.append(abs(root - centre))
    return tuple(errors)


def energy_flux(function, params, erange=(10.0, 1000.0)):
    """Energy flux in erg cm^-2 s^-1 over erange (GDT already converts keV to erg)."""
    return function.integrate(params, erange, energy=True)


def flux_with_error(function, fitter, erange=(10.0, 1000.0), num_samples=2000, seed=1):
    """Energy flux and its 1-sigma error, sampling the parameters from covariance()."""
    rng = np.random.default_rng(seed)
    best = np.asarray(fitter.parameters, dtype=float)
    samples = rng.multivariate_normal(best, covariance(fitter), size=num_samples)
    lo = np.array(function.min_values)[np.array(function.free)]
    hi = np.array(function.max_values)[np.array(function.free)]
    samples = samples[np.all((samples > lo) & (samples < hi), axis=1)]
    fluxes = np.array([energy_flux(function, s, erange) for s in samples])
    return energy_flux(function, best, erange), np.std(fluxes)


def make_spectrum(data_dir, burst_id, det, window, bkg_windows, order):
    """Source PHA and background spectrum for one detector, as in notebook 08.

    Lets the robustness checks rebuild the spectrum with a different
    background order. Returns (pha, background_spectrum, background reduced chi2).
    """
    tte = GbmTte.open(tte_path(data_dir, burst_id, det))
    phaii = tte.to_phaii(bin_by_edges, np.array(window))
    tte.close()
    pha = phaii.to_pha(time_ranges=[window])
    fitter, reduced_chi2 = fit_background(data_dir, burst_id, det, bkg_windows, order)
    model = fitter.interpolate_bins(np.array([window[0]]), np.array([window[1]]))
    bkg = BackgroundSpectrum(np.asarray(model.rates).reshape(-1),
                             np.asarray(model.rate_uncertainty).reshape(-1),
                             pha.data.lo_edges, pha.data.hi_edges,
                             np.full(pha.num_chans, pha.exposure))
    return pha, bkg, reduced_chi2


def simulate_pha(pha, bak, rsp, function, params, rng):
    """A fake PHA: Poisson draw of (model folded through the response + background).

    params is the full parameter vector. The background is kept at its model
    value, which is what the known-background statistic assumes.
    """
    rates = np.asarray(rsp.drm.fold_spectrum(function.eval, params))
    expected = rates * pha.exposure + np.asarray(bak.data.counts)
    data = pha.data
    fake = EnergyBins(rng.poisson(expected), data.lo_edges, data.hi_edges, pha.exposure)
    return Pha.from_data(fake, gti=pha.gti, trigger_time=pha.trigtime)
