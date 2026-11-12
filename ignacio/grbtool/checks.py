"""How far each window can be trusted: rates, telemetry limit, data gaps, live fraction, systematics, and the flags."""
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from gdt.core.spectra.functions import Band, Comptonized
from gdt.missions.fermi.gbm.response import GbmRsp2

from lightcurve_tools import aligned_bin_edges
from sample_tools import FIT_RANGE_BGO, FIT_RANGE_NAI
from spectral_fit_tools import BandLowBreak, channel_mask, cstat, fit_model, make_spectra, rsp2_path, scaled_starts, window_response
from .bursts import ALL_DETECTORS, event_times
from .fitting import errors_of, window_data

# Both rate rules were read off GRB 230307A, the only burst here that reaches them, so treat them as a warning and not as a measurement.
TELEMETRY_KCPS = 370.0      # all 14 detectors, 16 ms. The limit is 375 kHz. 16 ms bins reach 378 kcps in 3-7 s and at most 361 kcps elsewhere
HIGH_RATE_KCPS = 60.0       # any one fitted detector, 16 ms. It flags the same windows as the GBM team's pile-up intervals in GCN 33551
BIN_WIDTH = 0.016           # s
GAP_SIGNIFICANCE = 20.0     # a gap with rate x gap above this isn't a Poisson waiting time (notebook 15)


class Rates:
    """The rates of all 14 detectors over the burst, and the gaps in their merged event stream."""

    def __init__(self, burst, time_range):
        events = event_times(burst, ALL_DETECTORS, time_range=time_range)
        self.missing = [d for d in ALL_DETECTORS if d not in events]
        self.edges = aligned_bin_edges(time_range, BIN_WIDTH)
        widths = np.diff(self.edges)
        self.per_detector = {d: np.histogram(t, bins=self.edges)[0] / widths / 1e3 for d, t in events.items()}      # kcps, all channels, as recorded
        self.summed = sum(self.per_detector.values())
        centres = 0.5 * (self.edges[1:] + self.edges[:-1])
        merged = np.sort(np.concatenate(list(events.values())))
        local = np.interp(merged[:-1], centres, self.summed * 1e3)
        lengths = np.diff(merged)
        lost = lengths * local > GAP_SIGNIFICANCE
        self.gaps = pd.DataFrame({"t": merged[:-1][lost], "gap_ms": 1e3 * lengths[lost], "summed_kcps": local[lost] / 1e3})

    def telemetry_intervals(self, join=3.0, grid=0.5):
        """Stretches where the summed rate is at the limit, rounded outwards to the next half second. Bins at the limit less than `join` seconds apart are one stretch.

        In GRB 230307A the bins at the limit are at 3.25-3.30 s and 5.95-6.72 s, with the rate down to 82 kcps between them and gaps in the data at 4.80 and 5.83 s.
        One stretch gives 3-7 s, the interval the GBM team give for TTE losses. `join` is set by this burst, the only one here that reaches the limit.
        """
        at_limit = np.flatnonzero(self.summed >= TELEMETRY_KCPS)
        if not at_limit.size:
            return []
        groups = np.split(at_limit, np.flatnonzero(np.diff(self.edges[at_limit]) > join) + 1)
        intervals = [(float(np.floor(self.edges[g[0]] / grid) * grid), float(np.ceil(self.edges[g[-1] + 1] / grid) * grid)) for g in groups]
        merged = [intervals[0]]
        for a, b in intervals[1:]:
            if a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        return merged

    def in_pieces(self, pieces):
        """Mask of the 16 ms bins that lie in the given stretches of time."""
        centres = 0.5 * (self.edges[1:] + self.edges[:-1])
        return np.any([(centres > a) & (centres < b) for a, b in pieces], axis=0)


def window_flags(burst, window, rates, spectra):
    """Rates, gaps and live fraction of one window, and what they mean."""
    inside = rates.in_pieces(window.pieces)
    row, flags = {}, []
    if inside.any():
        peak = {d: float(rates.per_detector[d][inside].max()) for d in burst.detectors if d in rates.per_detector}
        bgo = [v for d, v in peak.items() if d.startswith("b")]
        nai = [v for d, v in peak.items() if d.startswith("n")]
        row.update(summed_max_kcps=float(rates.summed[inside].max()), nai_max_kcps=max(nai) if nai else np.nan, bgo_max_kcps=max(bgo) if bgo else np.nan)
        # notebook 15: E_peak moves by about 1% per 12 kcps in the BGO. Worked out for one shape, so it's a guide
        row["pileup_guide_pct"] = max(bgo) / 12.0 if bgo else np.nan
        if max(peak.values()) > HIGH_RATE_KCPS:
            flags.append("high_rate")
        if row["summed_max_kcps"] >= TELEMETRY_KCPS:
            flags.append("telemetry_limit")
    gaps = rates.gaps[np.any([(rates.gaps.t > a) & (rates.gaps.t < b) for a, b in window.pieces], axis=0)] if len(rates.gaps) else rates.gaps
    row.update(n_gaps=len(gaps), gap_ms=float(gaps.gap_ms.sum()) if len(gaps) else 0.0)
    if len(gaps):
        flags.append("data_gaps")
    live = [sum(float(spectra[d][window.label, i][0].exposure) for i in range(len(window.pieces))) / window.duration for d in burst.detectors]
    row["live_fraction_min"] = min(live)
    if min(live) < 0.8:
        flags.append("low_live_fraction")
    row["flags"] = " ".join(flags)
    return row


def start_vector(row):
    """The chosen model and its best-fit parameters, from a row of the table of fits. Returns (function, full vector, index of E_peak in the free parameters)."""
    if row.model == "BandLowBreak":
        return BandLowBreak(), [1.0, row.Eb, row.alpha1_lb, row.alpha2_lb, row.Epeak_lowbreak, min(row.beta_lb, -2.05), 2.0, 100.0], 4
    if row.model == "Band":
        return Band(), [1.0, row.Epeak_band, row.alpha_band, row.beta_band, 100.0], 1
    return Comptonized(), [1.0, row.Epeak_cpl, row.alpha_cpl, 100.0], 1


def _fit_again(row, data):
    """The chosen model again on changed data, from the best fit with E_peak moved either way, so a small change can't leave it where it started (notebook 16)."""
    function, start, k = start_vector(row)
    shapes = [list(start)] + [[v * f if i == k else v for i, v in enumerate(start)] for f in (0.8, 1.25)]
    if row.model == "Band":
        shapes += [[1.0, e, a, b, 100.0] for e in (30.0, 100.0, 300.0) for a in (-0.8, -1.2) for b in (-2.1, -3.0)]
    elif row.model == "Comptonized":
        shapes += [[1.0, e, a, 100.0] for e in (50.0, 100.0, 300.0, 1000.0) for a in (-0.6, -1.2)]
    fit = fit_model(*data, function, scaled_starts(function, *data, shapes), polish=True)
    return fit.parameters[k], cstat(fit)


def systematics(burst, window, row, spectra, spectra_order2):
    """How much E_peak moves with the BGO from 300 keV, a response interpolated to the middle of the window, and a background of order 2.

    The refit with nothing changed is the reference, so what's left is the change itself and not a difference between minimisers.
    """
    middle = 0.5 * (window.span[0] + window.span[1])
    reference, _ = _fit_again(row, window_data(burst, window, spectra))
    variants = {"bgo300": window_data(burst, window, spectra, bgo_range=(300.0, FIT_RANGE_BGO[1])), "order2": window_data(burst, window, spectra_order2)}
    phas, baks, _, _ = window_data(burst, window, spectra)
    interpolated = [GbmRsp2.open(rsp2_path(burst.data_dir, burst.trigger, d)).interpolate(middle) for d in burst.detectors]
    variants["interp_rsp"] = (phas, baks, interpolated, [channel_mask(r, FIT_RANGE_BGO if d.startswith("b") else FIT_RANGE_NAI) for r, d in zip(interpolated, burst.detectors)])
    out = {}
    for name, data in variants.items():
        out[f"dEp_{name}"] = _fit_again(row, data)[0] - reference
    out["Epeak_sys"] = float(np.sqrt(sum(v**2 for v in out.values())))
    return out


def local_background_check(burst, window):
    """A precursor's background from just before it, with a straight line. Returns E_peak (Comptonized) and the net counts that way."""
    stop = window.span[0] - 2.0
    start = max(stop - 38.0, burst.tte_range[0] + 1.0)
    if stop - start < 5.0:
        return dict(Epeak_cpl_local_bkg=np.nan, net_local_bkg=np.nan)
    spectra = {d: make_spectra(burst.data_dir, burst.trigger, d, {(window.label, i): p for i, p in enumerate(window.pieces)}, [(start, stop)], 1)[0] for d in burst.detectors}
    data = window_data(burst, window, spectra)
    cpl = Comptonized()
    fit = fit_model(*data, cpl, scaled_starts(cpl, *data, [[1.0, e, a, 100.0] for e in (30.0, 100.0, 300.0, 900.0, 2500.0) for a in (-0.6, -1.2)]), polish=True)
    net = sum(float(np.sum(p.data.counts)) - float(np.sum(b.counts)) for p, b in zip(data[0], data[1]))
    return dict(Epeak_cpl_local_bkg=fit.parameters[1], net_local_bkg=net)


def catalogue_check(burst):
    """The catalogue's own fit repeated: its detectors, channels and interval, and my background. If this works the data handling is right for this burst."""
    if not burst.catalogue or not burst.channels or not set(burst.detectors) <= set(burst.channels):
        return dict(cat_Ep=np.nan, cat_Ep_err=np.nan, my_Ep=np.nan, my_Ep_err=np.nan, ratio=np.nan, sigma=np.nan)
    interval = burst.catalogue["interval"]
    spectra = {d: make_spectra(burst.data_dir, burst.trigger, d, {"cat": interval}, burst.bkg_windows, burst.bkg_order)[0]["cat"] for d in burst.detectors}
    rsps = [window_response(burst.data_dir, burst.trigger, d, interval)[0] for d in burst.detectors]
    masks = [(np.arange(r.num_chans) >= burst.channels[d][0]) & (np.arange(r.num_chans) <= burst.channels[d][-1]) for r, d in zip(rsps, burst.detectors)]
    data = ([spectra[d][0] for d in burst.detectors], [spectra[d][1] for d in burst.detectors], rsps, masks)
    model = Comptonized()
    with warnings.catch_warnings(action="ignore"):
        fit = fit_model(*data, model, scaled_starts(model, *data, [[1.0, e, a, 100.0] for e in (100.0, 400.0, 1500.0) for a in (-0.7, -1.3)]), polish=True)
    my_err = errors_of(fit)[1]
    cat, cat_err = burst.catalogue["Ep"], burst.catalogue["Ep_err"]
    return dict(interval=f"{interval[0]:g} to {interval[1]:g}", cat_Ep=cat, cat_Ep_err=cat_err, my_Ep=fit.parameters[1], my_Ep_err=my_err, ratio=fit.parameters[1] / cat,
                sigma=(fit.parameters[1] - cat) / np.hypot(my_err, cat_err), my_alpha=fit.parameters[2], cat_alpha=burst.catalogue["alpha"])


def read_catalogue_background(burst):
    """The background intervals the GBM team used for T90, from the header of the catalogue file."""
    path = sorted(burst.data_dir.glob(f"glg_bcat_all_{burst.trigger}_v*.fit"))[-1]
    header = fits.getheader(path, 0)
    parse = lambda text: tuple(float(x) for x in text.strip("() ").split(","))
    return parse(header["LOBCKINT"]), parse(header["HIBCKINT"])
