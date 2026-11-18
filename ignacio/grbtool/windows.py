"""Background intervals, precursor, gap, main emission and time slices, with one rule for every burst."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lightcurve_tools import aligned_bin_edges, fit_background
from sample_tools import blocks_of, main_stop, net_light_curve, precursor_from_blocks

BAND = (8.0, 1000.0)        # keV, the band for finding windows, as in notebooks 12 and 18
BIN_WIDTH = 0.016           # s
MAX_BINS = 15000            # the blocks take N^2 steps, so a very long burst gets wider bins
CHI2_DROP = 0.01            # a real drop in reduced chi2. Clean intervals differ by 0.005 at most in the ten bursts checked
CHI2_POOR = 1.0             # clean intervals are at 0.93-0.98


class WindowError(Exception):
    """The windows can't be found, with the reason."""


@dataclass
class Window:
    """One time window. pieces is more than one stretch when a bad interval is cut out of it."""
    label: str
    pieces: list
    kind: str                 # precursor, main, tail, slice, or given

    @property
    def span(self):
        return self.pieces[0][0], self.pieces[-1][1]

    @property
    def duration(self):
        return sum(b - a for a, b in self.pieces)


def post_burst_start(burst, pre, after_t90=30.0, step=30.0, min_length=60.0):
    """Where the background after the burst starts: 30 s after T90, and later if that interval still has the burst in it.

    T90 can end well before the emission does. GRB 230307A's tail is still there at 70-120 s, and 30 s after T90 is only 66 s. A tail inside the
    interval makes the polynomial a worse fit, and once it's left out the rest is still fitted by the same polynomial, so the reduced chi2
    falls. Moving 30 s later lowers it by 0.12 in 230307A's na and by 0.016 in 211211A's n2, and by at most 0.005 in the other eight bursts.
    Returns the start and the reduced chi2 of each detector."""
    stop = burst.tte_range[1] - 1.0
    post = burst.t90_start + burst.t90 + after_t90
    if stop - post < 20.0:
        return None, {d: fit_background(burst.data_dir, burst.trigger, d, [pre], 1)[1] for d in burst.detectors}
    quality = lambda start: {d: fit_background(burst.data_dir, burst.trigger, d, [pre, (start, stop)], burst.bkg_order)[1] for d in burst.detectors}
    chi2 = quality(post)
    while stop - (post + step) >= min_length:
        later = quality(post + step)
        if max(chi2[d] - later[d] for d in chi2) <= CHI2_DROP:
            break
        post, chi2 = post + step, later
    return post, chi2


def set_background(burst, first_emission=None, margin=5.0):
    """The background intervals: before the burst, ending a margin before anything happens, and after it to the end of the TTE. Returns flags."""
    start, stop = burst.tte_range
    pre_stop = max(min(-margin, burst.t90_start - margin, np.inf if first_emission is None else first_emission - margin), start + 6.0)
    pre = (start + 1.0, pre_stop)
    post, chi2 = post_burst_start(burst, pre)
    flags = []
    if post is None:
        burst.bkg_windows, burst.bkg_order = [pre], 1       # one side can only hold a straight line
        flags.append("background_before_burst_only")
    else:
        burst.bkg_windows = [pre, (post, stop - 1.0)]
        if stop - 1.0 - post < 60.0:
            flags.append("short_background_after_burst")
    if max(chi2.values()) > CHI2_POOR:
        flags.append("background_poor_fit")
    burst.bkg_chi2 = chi2
    return flags


def lull_precursor(blocks, start_sigma=3.0, lull_fraction=0.25):
    """The second try when there's no quiet gap: a drop below a quarter of the first peak, with a higher peak after it.

    GRB 230307A never goes back down to the background between its precursor and its main emission, but it does drop. The precursor window
    ends where the rate falls, and the main emission starts where the rate has passed half of the first peak again. A burst found this way
    gets the flag no_quiet_gap, because a first pulse of the main emission could look just the same."""
    excess, sigma = blocks.excess.values, blocks.sigma.values
    first = int(np.argmax(sigma > start_sigma))
    highest = int(np.argmax(excess))
    peak = excess[first]
    for i in range(first + 1, highest):
        if excess[i] < lull_fraction * peak:
            j = i
            while j + 1 < highest and excess[j + 1] < 0.5 * peak:
                j += 1
            return dict(pre_start=float(blocks.start[first]), pre_stop=float(blocks.start[i]), main_start=float(blocks.stop[j]), pre_peak=float(excess[first:i].max()),
                        gap_excess=float(excess[i:j + 1].max()), gap_sigma=float(sigma[i:j + 1].max()), main_peak=float(excess[highest]))
        peak = max(peak, excess[i])
    return None


def blocks_at(burst, bin_width):
    """Bayesian blocks on the NaI light curve, from before the burst to 15 s after T90, as in notebook 18."""
    t_end = burst.t90_start + burst.t90
    span = (max(burst.tte_range[0] + 1.0, min(-30.0, burst.t90_start - 20.0)), t_end + 15.0)
    edges = np.round(aligned_bin_edges(span, bin_width), 6)
    counts, exposure, background = net_light_curve(burst.data_dir, burst.trigger, burst.nai, BAND, edges, burst.bkg_windows, burst.bkg_order)
    return blocks_of(counts, exposure, background, edges)


def first_significant(blocks):
    above = blocks.sigma.values > 3.0
    if not above.any():
        raise WindowError("no block is 3 sigma above the first one, so the catalogue's detectors don't see the burst in this light curve")
    return float(blocks.start[int(np.argmax(above))])


def episodes(blocks):
    """The precursor if there is one, and the start of the main emission. Returns (dict or None, flags)."""
    found = precursor_from_blocks(blocks)
    if found is not None:
        return found, []
    found = lull_precursor(blocks)
    return found, ["no_quiet_gap"] if found is not None else ["no_precursor_found"]


def find_episodes(burst, background_given=False):
    """Set the background, then find the precursor and main emission. Returns (precursor or None, main, blocks table, flags).

    The background before the burst has to stop before the burst does anything, which can be well before T0, so it takes a first pass to find that.
    """
    bin_width = BIN_WIDTH
    t_end = burst.t90_start + burst.t90
    while (t_end + 45.0) / bin_width > MAX_BINS:
        bin_width *= 2.0
    flags = [] if background_given else set_background(burst)
    blocks = blocks_at(burst, bin_width)
    if not background_given:
        first = first_significant(blocks)
        before = burst.bkg_windows[0][1]
        if max(first - 5.0, burst.tte_range[0] + 6.0) < before - 0.5:
            flags = set_background(burst, first)
            blocks = blocks_at(burst, bin_width)
    found, more = episodes(blocks)
    flags += more
    start = found["main_start"] if found else first_significant(blocks)
    main = (start, main_stop(blocks, start))
    precursor = dict(found, window=(found["pre_start"], found["pre_stop"])) if found else None

    # The same rule on bins twice as wide. A precursor that moves says how far its edges can be trusted
    other, _ = episodes(blocks_at(burst, 2.0 * bin_width))
    if precursor and other:
        moved = max(abs(other["pre_start"] - found["pre_start"]), abs(other["pre_stop"] - found["pre_stop"]))
        precursor["window_other_bin"] = (other["pre_start"], other["pre_stop"])
        if moved > max(4.0 * bin_width, 0.1 * (found["pre_stop"] - found["pre_start"])):
            flags.append("precursor_moves_with_bin")
    elif bool(precursor) != bool(other):
        flags.append("precursor_found_at_one_bin_only")
    return precursor, main, blocks, flags


def end_of_emission(burst, start, bin_width=0.256, n_sigma=5.0):
    """The last time the burst is significant after `start`, before the background after the burst begins. From blocks on wider bins."""
    stop = burst.bkg_windows[1][0] if len(burst.bkg_windows) > 1 else burst.tte_range[1] - 1.0
    if stop - start < 2.0 * bin_width:
        return start
    edges = np.round(aligned_bin_edges((start, stop), bin_width), 6)
    edges = edges[(edges >= start) & (edges <= stop)]
    counts, exposure, background = net_light_curve(burst.data_dir, burst.trigger, burst.nai, BAND, edges, burst.bkg_windows, burst.bkg_order)
    net = counts - background
    # about 2 s at a time, measured from the fitted background, since there's no quiet first block here to take the level from as blocks_of does
    kernel = np.ones(min(8, len(net)))
    significant = np.flatnonzero(np.convolve(net, kernel, "same") / np.sqrt(np.maximum(np.convolve(counts, kernel, "same"), 1.0)) > n_sigma)
    return float(edges[significant[-1] + 1]) if significant.size else start


def cut_out(window, bad_intervals):
    """The stretches of a window that remain after cutting out the bad intervals."""
    pieces = [tuple(window)]
    for a, b in bad_intervals:
        kept = []
        for p, q in pieces:
            if b <= p or a >= q:
                kept.append((p, q))
                continue
            if a > p:
                kept.append((p, a))
            if b < q:
                kept.append((b, q))
        pieces = kept
    return pieces


def time_slices(burst, span, bad_intervals, target_net=10000.0, min_length=1.0, bin_width=0.016):
    """Consecutive slices with at least target_net net NaI counts and at least min_length in seconds. What's left at the end joins the last slice."""
    slices = []
    for a, b in cut_out(span, bad_intervals):
        edges = np.round(aligned_bin_edges((a, b), bin_width), 6)
        edges = np.unique(np.clip(edges, a, b))
        if len(edges) < 2:
            continue
        counts, _, background = net_light_curve(burst.data_dir, burst.trigger, burst.nai, BAND, edges, burst.bkg_windows, burst.bkg_order)
        net = np.cumsum(counts - background)
        start, reached, here = 0, 0.0, []
        for i in range(len(net)):
            if net[i] - reached >= target_net and edges[i + 1] - edges[start] >= min_length:
                here.append((float(edges[start]), float(edges[i + 1])))
                start, reached = i + 1, net[i]
        if here:
            here[-1] = (here[-1][0], float(edges[-1]))
        slices += here
    return slices


def windows_from_file(path):
    """Windows as they are in a shared windows file."""
    table = pd.read_csv(path, comment="#", index_col="label")
    return [Window(label, [(float(r.t_start), float(r.t_stop))], "precursor" if label == "pre" else "given") for label, r in table.iterrows()]


def automatic_windows(burst, bad_intervals, background_given=False):
    """Set the background, then the windows for the fits: the precursor, the main emission, what comes after it, and slices of the whole emission."""
    precursor, main, blocks, flags = find_episodes(burst, background_given)
    burst.flags += flags
    windows = []
    if precursor:
        windows.append(Window("pre", cut_out(precursor["window"], bad_intervals), "precursor"))
    windows.append(Window("main", cut_out(main, bad_intervals), "main"))
    end = end_of_emission(burst, main[1])
    if end - main[1] >= 1.0:
        windows.append(Window("tail", cut_out((main[1], end), bad_intervals), "tail"))
    windows = [w for w in windows if w.pieces]      # all of it inside a bad interval
    slices = time_slices(burst, (main[0], max(end, main[1])), bad_intervals)
    if len(slices) > 1:
        windows += [Window(f"s{i:02d}", [s], "slice") for i, s in enumerate(slices, 1)]
    return windows, precursor, blocks
