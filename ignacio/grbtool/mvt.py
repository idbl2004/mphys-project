"""The MVT of a window from Rodrigo's Haar code, imported as it is from his repository, plus what tells you how far to trust it.

The conventions are his and Bala et al.'s (2026): 8-900 keV, NaI only, Poisson resamples with errors sqrt(counts), 300 realisations with
seed 2026, a first-order background, and SNR_MVT = the highest counts in a bin of width tau / sqrt(background rate x tau). The class
(robust, likely upper limit or upper limit) comes from Bala et al.'s validation curve. Set MVTTOOLS and BALA_CODE to the folders of the
two repositories if they aren't in the home folder.

This is only a first look, with one set of detectors and no made-to-measure check. The values to quote come from his own program, and
joint.py joins them to E_peak by label."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gdt.missions.fermi.gbm.tte import GbmTte

from lightcurve_tools import fit_background, tte_path
from .bursts import ALL_NAI, event_times

ENERGY = (8.0, 900.0)       # keV
REALISATIONS = 300
SEED = 2026
SET_TAU = 0.016             # s, the timescale at which the detectors are chosen
BACKGROUND_BIN = 1.024      # s, the bins of the background fit
BACKGROUND_SPAN = 50.0      # s on each side, as long as his intervals
PLATEAU_STEP = 1.2          # two bin widths agree if their medians are within one step of the ladder (his document of 4 Nov, step 8)
# Bala et al.'s intervals, which go with their validation curve. Their code holds them burst by burst, and I can't find a rule behind them.
BALA_BACKGROUND = {"bn230307656": [(-68.8, -18.8), (69.4, 119.4)], "bn211211549": [(-68.0, -18.0), (69.7, 119.7)]}
# SNR_MVT needed at 2, 6 and 20 ms, read off the figure of his injection test (run B3), the higher of the Gaussian and Norris pulses.
# Windows under 1 s use the set for a pulse like the precursor (0.384 s at 5000 counts/s).
THRESHOLD_MS = [2.0, 6.0, 20.0]
THRESHOLD_STANDARD = [135.0, 120.0, 65.0]
THRESHOLD_SHORT = [80.0, 52.0, 50.0]
CLASS_NAMES = {"Below 95% CI (Upper Limit)": "UL", "Within 95% CI (Likely Upper Limit)": "LUL", "Above 95% CI (Robust Measurement)": "robust"}


def code_folder():
    return Path(os.environ.get("MVTTOOLS", Path.home() / "rodrigo_MVTTools"))


def load_haar():
    """His haarMethods module, and the commit it comes from. It reads config.yaml from its own folder, so nothing depends on where this runs."""
    folder = code_folder()
    if not (folder / "analysisTools" / "haarMethods.py").exists():
        raise FileNotFoundError(f"no MVTTools in {folder}. Clone github.com/rodrigogleon/MVTTools there, or set MVTTOOLS.")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    import analysisTools.haarMethods as haarMethods
    commit = subprocess.run(["git", "-C", str(folder), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return haarMethods, commit


def light_curve(events, detectors, window, bin_width):
    """Summed counts of the detectors, with bin edges made the way his binDetectors makes them."""
    edges = np.arange(window[0], window[1] + bin_width, bin_width)
    return sum(np.histogram(events[d], bins=edges)[0] for d in detectors).astype(float)


def background_intervals(burst):
    """Bala et al.'s intervals where they exist. Otherwise up to 50 s on each side of the burst, the part of the spectral background intervals closest to it,
    because a straight line needs them to be short. On the GRB 230307A precursor the second choice gives SNR_MVT 123.0 against 121.6, and the same class."""
    if burst.trigger in BALA_BACKGROUND:
        return BALA_BACKGROUND[burst.trigger]
    (a, b), (c, d) = burst.bkg_windows[0], burst.bkg_windows[-1]
    return [(max(a, b - BACKGROUND_SPAN), b), (c, min(d, c + BACKGROUND_SPAN))]


def background_fits(burst, detectors=ALL_NAI):
    intervals = background_intervals(burst)
    return {d: fit_background(burst.data_dir, burst.trigger, d, intervals, 1)[0] for d in detectors}


def background_rate(fitter, window, band=ENERGY):
    """The mean background in counts per second over a window, in a band or over all channels.

    It's done as in his run_precursor_mvt.py: the fit is evaluated on its own 1.024 s bins, and each bin is weighted by how much of it lies
    inside the window. There's always one bin more than the window needs, with no weight, because GDT can't integrate a model over a single
    bin."""
    first, last = int(np.floor(window[0] / BACKGROUND_BIN)), int(np.ceil(window[1] / BACKGROUND_BIN)) + 1
    edges = BACKGROUND_BIN * np.arange(first, last + 1)
    model = fitter.interpolate_bins(edges[:-1], edges[1:])
    rates = np.asarray((model.integrate_energy(*band) if band else model.integrate_energy()).rates).ravel()
    weights = np.clip(np.minimum(edges[1:], window[1]) - np.maximum(edges[:-1], window[0]), 0.0, None)
    return float(np.sum(weights * rates) / weights.sum())


def validation_curve():
    """Bala et al.'s classifier and its curve, from their public code (GBM_MVT_paper, folder MVTfermi). None if it isn't there, and the table then has no class."""
    folder = Path(os.environ.get("BALA_CODE", Path.home() / "bala_GBM_MVT_paper" / "MVTfermi"))
    if not (folder / "clasify_mvt_value.py").exists():
        return None
    spec = importlib.util.spec_from_file_location("clasify_mvt_value", folder / "clasify_mvt_value.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL_FILE = str(folder / "mvt_snr_fit_model.npz")
    return module, module.load_model()


def merged_times(burst, detectors, window):
    """8-900 keV event times after GbmTte.merge, which keeps one photon per tick of 2 us.

    Only for the class, because the validation curve was calibrated this way. The MVT and the other SNR use all the photons, as Rodrigo does.
    """
    ttes = []
    for d in detectors:
        tte = GbmTte.open(tte_path(burst.data_dir, burst.trigger, d))
        ttes.append(tte.slice_time(window))
        tte.close()
    times = np.array(GbmTte.merge(ttes).slice_energy(ENERGY).data.times)
    return times[(times >= window[0]) & (times < window[1])]


def bala_class(curve, mvt_ms, times, background_all, window):
    """The class of an MVT, with the SNR defined as the curve was: merged detectors and an all-channel background."""
    module, (interpolators, mvt_range) = curve
    tau = mvt_ms / 1e3
    snr = float(np.histogram(times, bins=np.arange(window[0], window[1] + tau, tau))[0].max() / np.sqrt(background_all * tau))
    label, bounds = module.classify_mvt_value(mvt_ms, snr, interpolators, mvt_range)
    return CLASS_NAMES.get(label, label), snr, float(10 ** bounds[2])


def snr_mvt(events, rates, detectors, window, tau):
    edges = np.arange(window[0], window[1] + tau, tau)
    counts = sum(np.histogram(events[d], bins=edges)[0] for d in detectors)
    return float(counts.max() / np.sqrt(sum(rates[d] for d in detectors) * tau))


def best_set(events, rates, window, tau=SET_TAU):
    """Bala et al.'s rule: rank the detectors by their own SNR, add them one at a time, and keep the set with the highest SNR."""
    single = {d: snr_mvt(events, rates, [d], window, tau) for d in events}
    ranked = sorted(single, key=single.get, reverse=True)
    trials = [(snr_mvt(events, rates, ranked[:k], window, tau), ranked[:k]) for k in range(1, len(ranked) + 1)]
    return max(trials, key=lambda t: t[0])


def threshold(tau, duration):
    """The SNR_MVT needed to trust an MVT of tau. Outside 2-20 ms it takes the value at the end of that range."""
    needed = THRESHOLD_SHORT if duration < 1.0 else THRESHOLD_STANDARD
    return float(np.interp(np.log(1e3 * tau), np.log(THRESHOLD_MS), needed))


def measure(haar, events, detectors, window, bin_width, workers):
    """One row of his table: the MVT of the data themselves, and the distribution from the Poisson resamples."""
    counts = light_curve(events, detectors, window, bin_width)
    errors = np.sqrt(counts)
    (_, _, single, single_err, _, _, _) = haar.haarPowerMod(counts, errors, minimumBinSizeSeconds=bin_width)
    r = haar.estimateMvtDistribution(counts, errors, haar.haarPowerMod, binSizeSeconds=bin_width, nRealisations=REALISATIONS, nWorkers=workers, seed=SEED, resampleErrors=True)
    return dict(bin_ms=1e3 * bin_width, bins=len(counts), total_counts=float(counts.sum()), single_ms=1e3 * single, single_err_ms=1e3 * single_err, mvt_ms=1e3 * r["median"],
                mvt_p16_ms=1e3 * r["p16"], mvt_p84_ms=1e3 * r["p84"], n_valid=r["nValid"], n_limit=r["nLimit"], n_failed=r["nFailed"])


def mvt_table(burst, windows, bin_widths=(1e-4, 1e-5, 3e-6), workers=8, detectors=None, max_bins=300000, kinds=("precursor", "main", "tail", "given")):
    """The MVT of the chosen windows, from the widest bin down, with SNR_MVT and the class. All windows share one set of NaI detectors.

    The widest bin always runs. Narrower ones only run while the light curve has at most max_bins bins, because the run time goes up with the
    number of bins. The smallest bin that ran gives the value to quote, as in Bala et al.: the median if the class is robust, and otherwise
    the 84th percentile as an upper limit. plateau says whether it agrees with the next bin up within one step of the ladder. Without a plateau
    the value can only be an upper limit, and plateau is left empty if only one bin ran."""
    haar, commit = load_haar()
    curve = validation_curve()
    chosen_windows = [w for w in windows if w.kind in kinds]
    continuous = [w for w in chosen_windows if len(w.pieces) == 1]
    time_range = (min(w.span[0] for w in chosen_windows) - 1.0, max(w.span[1] for w in chosen_windows) + 1.0)
    events = event_times(burst, ALL_NAI, ENERGY, time_range=time_range)
    fits = background_fits(burst, list(events))
    main = next((w for w in windows if w.kind == "main"), windows[0])
    for_set = max(main.pieces, key=lambda p: p[1] - p[0])
    chosen = list(detectors) if detectors else best_set(events, {d: background_rate(fits[d], for_set) for d in events}, for_set)[1]
    rows = [dict(label=w.label, t_start=w.span[0], t_stop=w.span[1], note="cut in pieces, so no continuous light curve") for w in chosen_windows if len(w.pieces) > 1]
    for w in continuous:
        rates = {d: background_rate(fits[d], w.span) for d in chosen}
        background_all = sum(background_rate(fits[d], w.span, band=None) for d in chosen)
        times = merged_times(burst, chosen, w.span) if curve else None
        here = []
        for bin_width in sorted(bin_widths, reverse=True):
            if here and w.duration / bin_width > max_bins:
                break
            row = dict(label=w.label, t_start=w.span[0], t_stop=w.span[1], detectors=" ".join(chosen), code_commit=commit, **measure(haar, events, chosen, w.span, bin_width, workers))
            if row["n_valid"] > 0:
                tau = row["mvt_ms"] / 1e3
                row.update(snr_at_mvt=snr_mvt(events, rates, chosen, w.span, tau), snr_needed=threshold(tau, w.duration))
                row["above_injection_threshold"] = row["snr_at_mvt"] >= row["snr_needed"]
                if curve:
                    row["class"], row["snr_for_class"], row["snr_robust_above"] = bala_class(curve, row["mvt_ms"], times, background_all, w.span)
            here.append(row)
        last = here[-1]
        last["quoted"] = True
        if last["n_valid"] > 0:
            limit = last.get("class") in ("UL", "LUL")
            last["quoted_ms"], last["quoted_is_limit"] = (last["mvt_p84_ms"], True) if limit else (last["mvt_ms"], False)
        if len(here) > 1 and here[-2]["n_valid"] > 0 and last["n_valid"] > 0:
            wider, narrower = here[-2]["mvt_ms"], last["mvt_ms"]
            last["plateau"] = max(wider, narrower) / min(wider, narrower) <= PLATEAU_STEP
            if not last["plateau"]:
                last["quoted_ms"], last["quoted_is_limit"] = last["mvt_p84_ms"], True
        rows += here
    return pd.DataFrame(rows)
