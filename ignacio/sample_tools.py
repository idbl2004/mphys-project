"""
Tools for running the spectral pipeline on the sample of bursts in notebook 18.

Nothing here changes the shared tools in lightcurve_tools.py and spectral_fit_tools.py, which are used as they are.
The catalogue's detectors and T90 set up each burst, so no burst needs tuning by hand. The precursor comes from
Bayesian blocks with one rule for every burst.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from gdt.core.binning.unbinned import bin_by_edges
from gdt.core.data_primitives import TimeBins
from gdt.core.spectra.functions import Band, BlackBody, Comptonized, PowerLaw
from gdt.misc.bayesian_blocks import bayesian_blocks
from gdt.missions.fermi.gbm.tte import GbmTte

from lightcurve_tools import background_in_band, fit_background, tte_path
from spectral_fit_tools import (channel_mask, covariance, cstat, energy_flux, fit_model, profile_error, rsp2_path,
                                scaled_starts, window_response)

FIT_RANGE_NAI = (8.0, 900.0)        # keV, as for the two known bursts
FIT_RANGE_BGO = (200.0, 40000.0)    # keV


def link_v00(data_dir):
    """The shared tools expect TTE files named _v00. Where only a later version exists, link it under that name."""
    data_dir = Path(data_dir)
    for path in sorted(data_dir.glob("glg_tte_*_v*.fit")):
        target = data_dir / f"{path.name.rsplit('_v', 1)[0]}_v00.fit"
        if not target.exists():
            target.symlink_to(path.name)


def burst_info(data_dir, trigger):
    """T90, the catalogue's detectors and fit, and the TTE time range."""
    data_dir = Path(data_dir)
    link_v00(data_dir)
    with fits.open(sorted(data_dir.glob(f"glg_bcat_all_{trigger}_v*.fit"))[-1]) as h:
        t90, t90_start = h[0].header["T90"], h[0].header["T90START"]
    with fits.open(sorted(data_dir.glob(f"glg_scat_all_{trigger}_flnc_comp_v*.fit"))[-1]) as h:
        names = h["DETECTOR DATA"].data["DETNAM"]
        detectors = [("b" if name.startswith("BGO") else "n") + format(int(name.split("_")[1]), "x") for name in names]
        channels = {d: tuple(int(c) for c in fit) for d, fit in zip(detectors, h["DETECTOR DATA"].data["FITCHAN"])}
        params = h["FIT PARAMS"].data
        catalogue = dict(interval=tuple(float(t) for t in params["TIMEBIN"][0]), Ep=float(params["PARAM1"][0][0]), Ep_err=float(params["PARAM1"][0][1]),
                         alpha=float(params["PARAM2"][0][0]), alpha_err=float(params["PARAM2"][0][1]))
    tte = GbmTte.open(tte_path(data_dir, trigger, detectors[-1]))
    tte_range = tuple(float(t) for t in tte.time_range)
    tte.close()
    angles = {}
    for det in detectors:
        with fits.open(rsp2_path(data_dir, trigger, det)) as h:
            trigtime = h[0].header["TRIGTIME"]
            angles[det] = [e.header["DET_ANG"] for e in h[2:] if e.header["TSTOP"] - trigtime > -1.0][0]
    return dict(t90=t90, t90_start=t90_start, detectors=detectors, channels=channels, catalogue=catalogue, tte_range=tte_range, angles=angles)


def background_windows(info, margin_pre=5.0, margin_post=30.0):
    """Before the burst, and from a margin after T90 to the end of the TTE. info["bkg_pre_stop"], if set, ends the first part."""
    start, stop = info["tte_range"]
    pre_stop = info.get("bkg_pre_stop", min(-margin_pre, info["t90_start"] - margin_pre))
    return [(start + 1.0, pre_stop), (info["t90_start"] + info["t90"] + margin_post, stop - 1.0)]


def net_light_curve(data_dir, trigger, detectors, band, edges, bkg_windows, order=3):
    """Summed counts, mean exposure and fitted background per bin of `detectors` in the energy band `band`."""
    counts, exposure, background = 0.0, 0.0, 0.0
    for det in detectors:
        tte = GbmTte.open(tte_path(data_dir, trigger, det))
        lc = tte.slice_energy(band).to_phaii(bin_by_edges, edges).to_lightcurve()
        tte.close()
        fitter, _ = fit_background(data_dir, trigger, det, bkg_windows, order)
        bkg, _ = background_in_band(fitter, band, edges)
        counts, exposure, background = counts + lc.counts, exposure + lc.exposure / len(detectors), background + bkg
    return np.asarray(counts, float), np.asarray(exposure, float), np.asarray(background, float)


def blocks_of(counts, exposure, background, edges, p0=0.05):
    """Bayesian blocks, with the excess in each block per second of live time and its significance.

    The fitted background can sit a little off over the long first block, and with so many counts that would look
    significant, so the excess is measured from the level of the first block, which is the time before the burst.
    """
    change_points = bayesian_blocks(TimeBins(counts, edges[:-1], edges[1:], exposure), p0=p0)
    rows = []
    for a, b in zip(change_points[:-1], change_points[1:]):
        rows.append(dict(start=edges[a], stop=edges[b], n=counts[a:b].sum(), live=exposure[a:b].sum(), bkg=background[a:b].sum()))
    blocks = pd.DataFrame(rows)
    blocks["rate"] = blocks.n / blocks.live
    offset = blocks.rate[0] - blocks.bkg[0] / blocks.live[0]
    blocks["excess"] = blocks.rate - blocks.bkg / blocks.live - offset
    blocks["sigma"] = blocks.excess * blocks.live / np.sqrt(np.maximum(blocks.n, 1.0))
    return blocks


def precursor_from_blocks(blocks, start_sigma=3.0, quiet_sigma=5.0, lull_fraction=0.25):
    """Precursor, gap and start of the main emission, with one rule for every burst.

    A block is quiet if it is below quiet_sigma and below lull_fraction of the highest excess so far. Runs of quiet blocks are
    looked for between the first significant block and the highest block of the burst. The gap is the first run, unless a later
    one is more than twice as long. The precursor is everything before the gap, and the main emission starts after it. Returns
    None if there's no quiet block there, which means no separate precursor by this rule.
    """
    excess, sigma = blocks.excess.values, blocks.sigma.values
    first = int(np.argmax(sigma > start_sigma))
    highest = int(np.argmax(excess))
    runs, current, peak = [], [], excess[first]
    for i in range(first + 1, highest):
        if sigma[i] < quiet_sigma and excess[i] < lull_fraction * peak:
            current.append(i)
        else:
            if current:
                runs.append(current)
            current = []
        peak = max(peak, excess[i])
    if current:
        runs.append(current)
    if not runs:
        return None
    length = lambda run: blocks.stop[run[-1]] - blocks.start[run[0]]
    gap = runs[0]
    for run in runs[1:]:
        if length(run) > 2.0 * length(gap):
            gap = run
    return dict(pre_start=float(blocks.start[first]), pre_stop=float(blocks.start[gap[0]]), main_start=float(blocks.stop[gap[-1]]), pre_peak=float(excess[first:gap[0]].max()),
                gap_excess=float(excess[gap].max()), gap_sigma=float(sigma[gap].max()), main_peak=float(excess[highest]))


def main_stop(blocks, main_start, fraction=0.1, min_duration=0.5):
    """End of the main emission: the first block after its peak, at least min_duration long, below `fraction` of the peak."""
    after = blocks[blocks.start >= main_start].reset_index(drop=True)
    peak_index = int(after.excess.idxmax())
    peak = after.excess[peak_index]
    for i in range(peak_index + 1, len(after)):
        if after.excess[i] < fraction * peak and after.stop[i] - after.start[i] >= min_duration:
            return float(after.start[i])
    return float(after.stop.iloc[-1])


def power_law():
    """GDT's power law has no upper bound on its index, and fits can run away, so bound it to a falling spectrum."""
    function = PowerLaw()
    function.max_values = [np.inf, -0.01, np.inf]
    return function


def fit_episode(data_dir, trigger, detectors, spectra, label, window):
    """Power law, Comptonized, Band and blackbody fits to one episode. Returns a dict of results, with profile errors on E_p."""
    responses = [window_response(data_dir, trigger, d, window)[0] for d in detectors]
    data = ([spectra[d][label][0] for d in detectors], [spectra[d][label][1] for d in detectors], responses,
            [channel_mask(r, FIT_RANGE_BGO if d.startswith("b") else FIT_RANGE_NAI) for r, d in zip(responses, detectors)])
    gross = {d: float(np.sum(spectra[d][label][0].data.counts)) for d in detectors}
    net = {d: gross[d] - float(np.sum(spectra[d][label][1].counts)) for d in detectors}

    pl = power_law()
    pl_fit = fit_model(*data, pl, scaled_starts(pl, *data, [[1.0, a, 100.0] for a in (-1.0, -1.5, -2.0)]), polish=True)
    cpl = Comptonized()
    cpl_fit = fit_model(*data, cpl, scaled_starts(cpl, *data, [[1.0, e, a, 100.0] for e in (50.0, 200.0, 800.0, 2000.0) for a in (-0.5, -1.0, -1.5)]), polish=True)
    band = Band()
    band_fit = fit_model(*data, band, scaled_starts(band, *data, [[1.0, e, a, b, 100.0] for e in (50.0, 200.0, 800.0) for a in (-0.5, -1.0) for b in (-2.5, -4.0)]), polish=True)
    bb = BlackBody()
    bb_fit = fit_model(*data, bb, scaled_starts(bb, *data, [[1.0, t] for t in (10.0, 30.0, 70.0, 150.0)]), polish=True)

    lo, hi = profile_error(*data, cpl, cpl_fit, 1, reach=20.0)
    lo95, hi95 = profile_error(*data, cpl, cpl_fit, 1, reach=40.0, delta=2.71)
    with np.errstate(invalid="ignore"):
        errors = np.sqrt(np.diag(covariance(cpl_fit)))
    return dict(t_start=window[0], t_stop=window[1], net_counts=sum(net.values()), gross_counts=sum(gross.values()), Ep=cpl_fit.parameters[1], Ep_err_lo=lo, Ep_err_hi=hi, Ep_limit_lo=cpl_fit.parameters[1] - lo95, Ep_limit_hi=cpl_fit.parameters[1] + hi95,
                alpha=cpl_fit.parameters[2], alpha_err=errors[2], C_cpl=cstat(cpl_fit), dof_cpl=cpl_fit.dof, C_pl=cstat(pl_fit), pl_index=pl_fit.parameters[1],
                Ep_band=band_fit.parameters[1], alpha_band=band_fit.parameters[2], beta_band=band_fit.parameters[3], C_band=cstat(band_fit), kT=bb_fit.parameters[1], C_bb=cstat(bb_fit),
                flux_cpl=energy_flux(cpl, list(cpl.default_values)))


def ccf_lag(low, high, bin_width, half_width=4, cut=False):
    """Lag of the low-energy light curve behind the high-energy one, in s, from the peak of their cross-correlation.

    Both are background-subtracted counts in the same bins. high covers the window and low the window plus the same number of
    extra bins on each side, so every trial lag sums over the same bins of high. With cut=True both cover only the window, and
    trial lags up to half its length use fewer and fewer bins. That pulls the peak towards zero, but it's the usual recipe.
    A positive lag means the low energies come later. The peak is refined with a parabola through the points within half_width
    bins of the highest one. NaN if the highest point is at the end of the range of lags, since that isn't a peak.
    """
    if cut:
        c = np.correlate(low, high, mode="full")
        lags = np.arange(len(c)) - (len(high) - 1)
        keep = np.abs(lags) <= len(high) // 2
        c, lags = c[keep], lags[keep]
    else:
        c = np.correlate(low, high, mode="valid")
        lags = np.arange(len(c)) - (len(c) - 1) // 2
    k = int(np.argmax(c))
    if not cut and (k < half_width or k >= len(c) - half_width):
        return np.nan
    near = np.arange(max(k - half_width, 0), min(k + half_width + 1, len(c)))
    a, b, _ = np.polyfit(lags[near], c[near], 2)
    peak = -b / (2 * a) if a < 0 else lags[k]
    return float(np.clip(peak, lags[near[0]], lags[near[-1]])) * bin_width


def lag_with_errors(low_counts, low_background, high_counts, high_background, bin_width, realisations=1000, seed=20, half_width=4, cut=False):
    """The lag from Poisson resamples of the observed counts. Returns its median and the distances to the 16th and 84th percentiles, in s.

    In a faint window the cross-correlation has several peaks, so the lag of the data themselves can fall outside that range.
    """
    rng = np.random.default_rng(seed)
    resampled = [ccf_lag(rng.poisson(low_counts) - low_background, rng.poisson(high_counts) - high_background, bin_width, half_width, cut) for _ in range(realisations)]
    down, middle, up = np.nanpercentile(resampled, [15.87, 50.0, 84.13])
    return middle, middle - down, up - middle
