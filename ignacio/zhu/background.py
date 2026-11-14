"""The light curve the episodes are found on, and its background.

The light curve is 8-1000 keV from the NaI detectors of the GBM catalogue for the burst (summed counts, mean live time), at the time
resolution Zhu et al. chose for that burst. The background is a polynomial fitted in time to each detector and channel over an interval
before and after the burst (lightcurve_tools.fit_background, as in every notebook since 05). Zhu et al. say only that they used GBM Data
Tools; the choices here are the project's and are the same in every episode mode, so differences between the modes come from the rules.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from gdt.core.binning.unbinned import bin_by_edges
from gdt.missions.fermi.gbm.tte import GbmTte

from grbtool.windows import set_background
from lightcurve_tools import aligned_bin_edges, background_in_band, fit_background, tte_path

BAND = (8.0, 1000.0)            # keV, their light-curve band
COARSE = 1.024                  # s, the bins used to find where the emission ends, for the background intervals
FINE = 0.064                    # s, a second search before the burst for short precursors, which COARSE dilutes ...
PRECURSOR_SIGMA = 4.0           # ... counted as emission from this significance: GRB 131216A's precursor, 106.6 s before its main
                                #   emission, is 4.8 sigma; the next weakest isolated block before any of 94 bursts (180402A, -87 s) is 3.1
EMISSION_SIGMA = 5.0            # a coarse block this significant (with the systematic) is emission, and the background intervals keep clear of it
CLEAR_MARGIN = 10.0             # s between the emission and a background interval
MIN_BACKGROUND = 20.0           # s, the background interval either side is made at least this long where the data allow ...
EDGE_MARGIN = 2.0               # s, ... but it never comes closer than this to the emission
MIN_POST = 10.0                 # s, the shortest interval after the burst worth adding when grbtool found none
MIN_INTERVAL = 4.0              # s, an interval shorter than this is dropped rather than fitted
COARSE_BACKGROUND = 0.256       # s, the fitted background is worked out on bins this wide and spread over finer ones


@dataclass
class LightCurve:
    """Counts, live time and fitted background per bin, summed over the NaI detectors, in their band."""
    edges: np.ndarray
    counts: np.ndarray
    exposure: np.ndarray
    background: np.ndarray
    tres: float


def background_counts(fitter, band, edges, step=COARSE_BACKGROUND):
    """Background counts in each bin from a fitted polynomial: lightcurve_tools.background_in_band, without its per-bin uncertainty.

    GDT works out the uncertainty of the polynomial in every bin, which takes about 10 s per detector and band for 17,000 bins of 4 ms and
    isn't used here. The polynomial is smooth over seconds, so it is evaluated on bins of `step` and spread over the fine bins through its
    running sum (tests/test_zhu.py checks the two agree).
    """
    if len(edges) > 2 and np.min(np.diff(edges)) >= step:
        return background_in_band(fitter, band, edges)[0]
    # at least two coarse bins: GDT can't integrate a single bin
    coarse = np.linspace(edges[0], edges[-1], max(2, int(np.ceil((edges[-1] - edges[0]) / step))) + 1)
    cumulative = np.r_[0.0, np.cumsum(background_in_band(fitter, band, coarse)[0])]
    return np.diff(np.interp(edges, coarse, cumulative))


def sides(burst):
    """The background intervals before the burst (those that end by the start of T90) and after it (the rest). An interval before
    always ends at least 2 s before T90 starts, and one after always starts after the last emission block, so this holds even when
    T90 runs past the end of the TTE data."""
    before = [w for w in burst.bkg_windows if w[1] <= burst.t90_start]
    return before, [w for w in burst.bkg_windows if w[1] > burst.t90_start]


def emission_span(burst):
    """From the end of the background interval before the burst to the start of the one after it. Where there is no interval on one side
    (the flags no_background_before_burst or no_background_after_burst), the span runs to 1 s inside that end of the TTE data instead."""
    before, after = sides(burst)
    start = before[-1][1] if before else burst.tte_range[0] + 1.0
    stop = after[0][0] if after else burst.tte_range[1] - 1.0
    return start, stop


def light_curve(burst, tres, span=None, detectors=None):
    """The light curve of a burst at time resolution tres, between the two background intervals (emission_span) unless a span is given.

    The same as sample_tools.net_light_curve (counts and live time from GDT, the background polynomial of each detector), with the
    background from background_counts."""
    if span is None:
        span = emission_span(burst)
    edges = np.round(aligned_bin_edges(span, tres), 9)
    edges = edges[(edges >= span[0] - 1e-9) & (edges <= span[1] + 1e-9)]
    detectors = detectors or burst.nai
    counts, exposure, background = 0.0, 0.0, 0.0
    for det in detectors:
        tte = GbmTte.open(tte_path(burst.data_dir, burst.trigger, det))
        lc = tte.slice_energy(BAND).to_phaii(bin_by_edges, edges).to_lightcurve()
        tte.close()
        fitter, _ = fit_background(burst.data_dir, burst.trigger, det, burst.bkg_windows, burst.bkg_order)
        counts = counts + np.asarray(lc.counts, float)
        exposure = exposure + np.asarray(lc.exposure, float) / len(detectors)
        background = background + background_counts(fitter, BAND, edges)
    return LightCurve(edges, np.asarray(counts, float), np.asarray(exposure, float), np.asarray(background, float), tres)


def clear_background(burst, rounds=5):
    """Background intervals that keep clear of all the emission, extended emission and early precursors included. Returns flags.

    grbtool's set_background starts the interval after the burst 30 s after T90 (or later if the fit improves), and ends the one before it
    5 s before T90 starts. T90 can miss a long weak tail: Zhu et al.'s whole-burst duration of GRB 090618 ends at about 195 s, and their
    230307A at about 133 s, both after grbtool's interval has begun. So the light curve is binned at COARSE over the whole TTE, split into
    blocks, and any block EMISSION_SIGMA above the fitted background pushes the intervals back to CLEAR_MARGIN from it. Before the
    burst's first emission the search is made again at FINE, where a block PRECURSOR_SIGMA above the background counts too, because a
    precursor of half a second is diluted in COARSE bins and can sit inside the interval before the burst (GRB 131216A's, 107 s before its
    main emission, and 131209B's, 16 s before), where it would be fitted as background. The fit is then
    redone without the emission, which can show more of a tail, and this repeats until nothing moves. Where the data start or end close to
    the burst, an interval is kept as long as the data allow but never closer than EDGE_MARGIN to the emission, and the burst is flagged.
    (Short GRB 130804A's data start 8 s before it, and an interval stretched to 20 s would have covered the burst.) Where grbtool found no
    room after the burst, an interval after the emission is added if at least MIN_POST is left, with a straight-line background.
    """
    from .episodes import blocks_of          # here, because episodes imports this module
    flags = set_background(burst)
    tte_start, tte_stop = burst.tte_range[0] + 1.0, burst.tte_range[1] - 1.0
    for _ in range(rounds):
        lc = light_curve(burst, COARSE, span=(tte_start, tte_stop))
        blocks = blocks_of(lc)
        loud = blocks[blocks.sigma >= EMISSION_SIGMA]
        if not loud.empty and float(loud.start.min()) - tte_start > 10 * FINE:
            early = blocks_of(light_curve(burst, FINE, span=(tte_start, float(loud.start.min())))).iloc[1:]   # the first block
            loud = pd.concat([loud, early[early.sigma >= PRECURSOR_SIGMA]])            # sits on the edge of the fit, as in find_episodes
        if loud.empty:
            break
        before, after = sides(burst)
        pre, post = (before[-1] if before else None), (after[0] if after else None)
        first_loud, last_loud = float(loud.start.min()), float(loud.stop.max())
        windows = []
        if pre is not None:
            pre_stop = min(pre[1], first_loud - CLEAR_MARGIN)
            if pre_stop - pre[0] < MIN_BACKGROUND:
                # as long as the data allow, but never closer than EDGE_MARGIN to the emission or to the start of T90
                pre_stop = min(pre[0] + MIN_BACKGROUND, first_loud - EDGE_MARGIN, burst.t90_start - EDGE_MARGIN)
                flags.append("emission_close_to_start_of_data")
            if pre_stop - pre[0] >= MIN_INTERVAL:
                windows.append((pre[0], pre_stop))
            else:
                flags.append("no_background_before_burst")
        else:
            flags.append("no_background_before_burst")
        if post is None and tte_stop - (last_loud + CLEAR_MARGIN) >= MIN_POST:
            post = (last_loud + CLEAR_MARGIN, tte_stop)            # set_background found no room 30 s after T90; there is after the emission
            flags.append("background_after_burst_added")
        if post is not None:
            post_start = max(post[0], last_loud + CLEAR_MARGIN)
            if post[1] - post_start < MIN_BACKGROUND:
                post_start = max(post[1] - MIN_BACKGROUND, last_loud + EDGE_MARGIN, burst.t90_start + burst.t90 + EDGE_MARGIN)
                flags.append("emission_close_to_end_of_data")
            if post[1] - post_start >= MIN_INTERVAL:
                windows.append((post_start, post[1]))
            else:
                flags.append("no_background_after_burst")
        if not windows:
            flags.append("no_background_interval_left")     # keep the last intervals that had room
            break
        if len(windows) == 1 or any(b - a < MIN_BACKGROUND for a, b in windows):
            burst.bkg_order = 1                                # short or one-sided intervals only hold a straight line
        if len(windows) == len(burst.bkg_windows) and np.allclose(np.ravel(windows), np.ravel(burst.bkg_windows)):
            break
        burst.bkg_windows = windows
    return sorted(set(flags))
