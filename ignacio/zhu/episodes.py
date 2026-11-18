"""Finding the episodes of a burst on its Bayesian blocks, under three rules that I keep apart (zhu2025, zhu2026 and project), with the
durations, the MVTs and their errors.

All three modes start from exactly the same things. The light curve is 8-1000 keV, summed over the NaI detectors the GBM catalogue uses
for the burst, and binned at the time resolution (Tres) Zhu et al. chose for that burst. The background is a polynomial in time for each
detector and channel, fitted before and after the burst (both in background.py). The blocks are Scargle et al.'s (2013) on the binned
counts, with the fitness N log(N/T) and a false-alarm rate p0 = 0.05, the value both papers use. For each block I keep its counts, its
background, its significance as a whole (with a 1% systematic on the background) and its mean excess per bin. So the modes only differ in
how they pick the precursor (PE), the quiet episode after it (QE1), the main emission (ME), the quiet episode before an extended emission
(QE2) and the extended emission (EE) out of the same blocks.

zhu2025 follows section 2.1 of Zhu et al. (2025). A quiet episode is one block at least 10 bins long, "a steady emission episode
occurring after the decay of the preceding episode and before the rapid rise of the subsequent episode", which I read as a block that is
lower than both of its neighbours. The QE1 has to be below 3 sigma. The PE "exhibits prominent pulse features", which I take to mean a
block at least 3 sigma above the background, and it "has an intensity weaker than that of the main emission", so its peak rate has to
be below the ME's.

zhu2026 follows section 2.1 of Zhu et al. (2026). It's the same, except that a QE only needs 8 bins and the PE's peak has to be 3 sigma
above the background. Then there are two cases. "When the QE is consistent with the background level" (I read this as below 2 sigma per
bin, the usual meaning of consistent), the PE just needs fewer counts than the ME. "Otherwise, the peak count of the ME must be
significantly higher than that of the PE", which I take as the ME's peak block being 3 sigma above the PE's. They also say that "the peak
count of the PE may be comparable to that of the ME" when the QE is at the background, so the ME's peak can't simply be the brightest
block. If the brightest block is in the first piece of emission, a QE at the background comes after it, and the brightest emission after
that is at least COMPARABLE_PEAK as high, then that later block is tried as the ME's peak. It's only kept if the rule then accepts a PE
(_bright_precursor_peak).

project is the precursor rule I wrote for notebook 18 (sample_tools.precursor_from_blocks), written again here (_project_runs) so that it
runs on the same blocks. A block counts as quiet if it is below 5 sigma and below a quarter of the highest rate so far, and a gap can be
several blocks long. The gap is the first run of quiet blocks, unless a later run is more than twice as long. I also ask a run to be at
least 8 bins long in total, like a 2026 QE.

Some things their words leave open. I made the same choice in every mode, and notebook 23 shows how each one was settled:
- The SNR of a QE is the mean excess per bin over the Poisson noise of one bin. Their own gaps in GRB 230307A, 211211A and 090618 come
  out at 4, 0 and 0 sigma per bin but 25, 0 and 1 sigma as whole blocks, and they quote 3.15 sigma for 230307A, so per bin is the
  reading that fits.
- When there are several quiet dips before the main peak, the zhu modes take the longest one. That gave back their T100,PE for 71% and
  their T100,QE1 for 86% of the 2025 bursts within 0.2 dex. Taking the nearest dip or the first one did worse.
- The QE2 is the first dip after the ME's peak that is at least 50 bins long and below 15% of that peak, with the rate at least doubling
  within the next 5 blocks. Without these numbers the rule would cut at short dips that their eye skips, like a 60 ms dip inside GRB
  211211A's main emission.
- T_MVT in the zhu modes is what their tables actually hold, "the rise time of the shortest pulse", measured from the start of the pulse
  to the centre of its brightest bin (brightest_rise). Every T_MVT they print is a whole number of half-bins, mostly odd ones, which is
  exactly what this reading gives, and notebook 23 checks how often it lands on their value. The project mode keeps half the shortest
  block whose two edges are certain at 5 sigma (Xiao et al. 2023, which both papers cite), or half the episode if none is, because a
  brightest bin inside a single block is just set by counting noise.
- The first block of the light curve never counts as emission, since it starts right where the background interval ends.
No burst gets a rule or an edge of its own."""
import multiprocessing
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from gdt.core.data_primitives import TimeBins
from gdt.misc.bayesian_blocks import bayesian_blocks

from .background import BAND, LightCurve, light_curve
from .io import worker_start

P0 = 0.05                       # false-alarm rate of a change point (both papers)
EMISSION_BLOCK_SIGMA = 3.0      # a block is emission if it is this far above the background as a whole
QUIET_SIGMA = 3.0               # a QE1 is below this per bin (both papers: "SNR below (around) 3 sigma")
AT_BACKGROUND_SIGMA = 2.0       # zhu2026: "the QE is consistent with the background level", per bin (1, 1.5 and 2 were compared in
                                #   notebook 23: 2 finds two more of their precursors and loses none)
ME_OVER_PE_SIGMA = 3.0          # zhu2026: "the peak count of the ME must be significantly higher than that of the PE"
COMPARABLE_PEAK = 0.5           # zhu2026: "the peak count of the PE may be comparable to that of the ME", read as the ME's at least half
                                #   the PE's (0.33 to 0.8 pick the same bursts among the first 76)
QE2_BINS = 50                   # the QE before an EE is at least this many bins long ...
EE_LULL_FRACTION = 0.15         # ... below this share of the ME's peak rate ...
EE_RISE = 2.0                   # ... and the rate rises again by at least this factor ...
EE_RISE_BLOCKS = 5              # ... within this many blocks
MVT_OVERLAP = 3e-7              # a block sets the MVT only if its two edges meet with less than this probability (5 sigma; Xiao et al. 2023)
BKG_SYSTEMATIC = 0.01           # the fitted background is good to about 1% (notebook 16 found a 3% dip once)
MAX_BINS = 120000               # beyond this the blocks take too long (about 80 s at 100,000 bins), so a resolution step that needs more is skipped
MODES = {
    "zhu2025": dict(qe_bins=10, pe_test="weaker", mvt="brightest"),
    "zhu2026": dict(qe_bins=8, pe_test="two_branch", mvt="brightest"),
    "project": dict(qe_bins=8, pe_test="project", mvt="certain"),
}
QE_BINS = MODES["zhu2026"]["qe_bins"]
PROJECT_QUIET_SIGMA = 5.0       # project mode, as in notebook 18
PROJECT_LULL = 0.25


def blocks_of(lc, counts=None, p0=P0):
    """Bayesian blocks of the light curve (or of other counts on the same bins), with each block's counts, background and significance.

    sigma is how far the whole block sits above the fitted background, in units of its noise. The noise is the Poisson noise plus
    BKG_SYSTEMATIC of the background, because over a long block a 1% error in the polynomial is worth several sigma of Poisson noise. snr_bin
    is the mean excess per bin divided by the Poisson noise of one bin, net / sqrt(bkg nbins). That's what I take "SNR below about 3 sigma" to
    mean for a quiet episode: Zhu et al.'s own gaps in GRB 230307A, 211211A and 090618 are about 4, 0 and 0 per bin but 25, 0 and 1 sigma as
    whole blocks, and they quote 3.15 sigma for 230307A. rate is the excess per second, which is what "peak count" compares."""
    counts = lc.counts if counts is None else counts
    cp = bayesian_blocks(TimeBins(counts, lc.edges[:-1], lc.edges[1:], lc.exposure), p0=p0)
    n = np.add.reduceat(counts, cp[:-1])
    bkg = np.add.reduceat(lc.background, cp[:-1])
    live = np.add.reduceat(lc.exposure, cp[:-1])
    start, stop = lc.edges[cp[:-1]], lc.edges[cp[1:]]
    blocks = pd.DataFrame(dict(start=start, stop=stop, i0=cp[:-1], i1=cp[1:], nbins=np.diff(cp), n=n, bkg=bkg, live=live))
    blocks["net"] = blocks.n - blocks.bkg
    blocks["sigma"] = blocks.net / np.sqrt(np.maximum(blocks.bkg, 1.0) + (BKG_SYSTEMATIC * blocks.bkg) ** 2)
    blocks["snr_bin"] = blocks.net / np.sqrt(np.maximum(blocks.bkg, 1.0) * blocks.nbins)
    blocks["rate"] = blocks.net / (blocks.stop - blocks.start)
    blocks["rate_err"] = np.sqrt(np.maximum(blocks.n, 1.0)) / (blocks.stop - blocks.start)
    return blocks




@dataclass
class EpisodeResult:
    """The episodes of one burst under one mode. Times in s from the trigger (None if the episode isn't there), block indices into the
    blocks table (first, last), and what the rule saw at each boundary it accepted."""
    burst: str = ""
    mode: str = "zhu2026"
    bin_width: float = np.nan
    p0: float = P0
    pe: tuple = None
    qe1: tuple = None
    me: tuple = None
    qe2: tuple = None
    ee: tuple = None
    we: tuple = None
    pe_blocks: tuple = None
    qe1_blocks: tuple = None
    me_blocks: tuple = None
    qe2_blocks: tuple = None
    ee_blocks: tuple = None
    pe_peak_rate: float = np.nan        # net counts/s of the PE's highest block
    pe_peak_sigma: float = np.nan       # that block's significance as a whole
    me_peak_rate: float = np.nan
    me_peak_sigma: float = np.nan
    qe1_snr_bin: float = np.nan         # mean excess per bin over one bin's noise
    qe1_dip: float = np.nan             # QE1 rate / PE peak rate
    qe1_recovery: float = np.nan        # ME peak rate / QE1 rate (inf if the QE1 is at or below the background)
    qe2_dip: float = np.nan             # QE2 rate / ME peak rate
    qe2_recovery: float = np.nan        # highest rate within EE_RISE_BLOCKS after the QE2 / QE2 rate
    mvt_pe: float = np.nan
    mvt_me: float = np.nan
    mvt_pe_bins: int = 0
    mvt_me_bins: int = 0
    mvt_pe_err: tuple = (np.nan, np.nan)
    mvt_me_err: tuple = (np.nan, np.nan)
    reasons: list = field(default_factory=list)
    flags: list = field(default_factory=list)

    def durations(self):
        span = lambda w: w[1] - w[0] if w else np.nan
        return dict(T100_PE=span(self.pe), T100_ME=span(self.me), T100_EE=span(self.ee), T100_WE=span(self.we), T100_QE1=span(self.qe1),
                    T100_QE2=span(self.qe2), TMVT_PE=self.mvt_pe, TMVT_ME=self.mvt_me, mvt_pe_bins=self.mvt_pe_bins, mvt_me_bins=self.mvt_me_bins,
                    TMVT_PE_lo=self.mvt_pe_err[0], TMVT_PE_hi=self.mvt_pe_err[1], TMVT_ME_lo=self.mvt_me_err[0], TMVT_ME_hi=self.mvt_me_err[1],
                    qe1_snr_bin=self.qe1_snr_bin, has_pe=self.pe is not None, has_ee=self.ee is not None)

    def record(self):
        """Everything, as plain values for a JSON file."""
        out = asdict(self)
        out.update(self.durations())
        return out


Episodes = EpisodeResult        # the name used before the modes were split


def _fitness(n, e, a, b):
    """Scargle et al. (2013, eq. 19) fitness N log(N/T) of the blocks [a, b) for arrays of a and b, from cumulative counts n and exposure e."""
    N, T = n[b] - n[a], e[b] - e[a]
    return np.where(N > 0, N * np.log(np.where(N > 0, N, 1.0) / np.where(T > 0, T, 1.0)), 0.0)


def ncp_prior(n_bins, p0=P0):
    """The Bayesian-blocks penalty for one more change point (Scargle et al. 2013, eq. 21, as in GDT's bayesian_blocks)."""
    return 4.0 - np.log(73.53 * p0 * n_bins ** -0.478)


def change_point_profile(counts, exposure, left, right, fixed, prior=0.0):
    """How likely a change point is at each place between the change points left and right, with the one at fixed held still.

    For each possible edge j, the light curve from left to right is split at j and at fixed, and the fitness of the blocks (their Poisson
    log-likelihood, up to a constant) gives the relative probability of j (Scargle et al. 2013, section 3.3). j is allowed to pass fixed,
    which just means the two change points swap over, and j == fixed means the block between them disappears. That case has one change point
    fewer, so it gets back the penalty (prior) that every change point costs in the blocks (ncp_prior). Returns the edges j and their
    probabilities."""
    n = np.r_[0.0, np.cumsum(counts[left:right])]
    e = np.r_[0.0, np.cumsum(exposure[left:right])]
    m, f = right - left, fixed - left
    j = np.arange(1, m)
    before, after = j < f, j > f
    fitness = _fitness(n, e, 0, f) + _fitness(n, e, f, m) + prior              # j == fixed: one change point fewer
    fitness = np.where(before, _fitness(n, e, 0, j) + _fitness(n, e, j, f) + _fitness(n, e, f, m), fitness)
    fitness = np.where(after, _fitness(n, e, 0, f) + _fitness(n, e, f, j) + _fitness(n, e, j, m), fitness)
    prob = np.exp(fitness - fitness.max())
    return left + j, prob / prob.sum()


def _percentile(edges, prob, q):
    return float(edges[min(np.searchsorted(np.cumsum(prob), q), len(edges) - 1)])


def mvt_of(lc, counts, blocks, first, last):
    """The MVT of blocks first..last in Xiao et al.'s (2023) sense: half the length of the shortest block whose two edges are certain.

    I go through the blocks from the shortest up. Each edge of a block gets a profile (change_point_profile) between the change points on
    either side, with the block's other edge held where it is. The chance that the left edge reaches or passes the right one, plus the chance
    that the right edge reaches or passes the left one, is the chance that the block isn't really there. The first block where that chance is
    below MVT_OVERLAP sets the MVT, and its error comes from the 16th and 84th percentiles of the two edges. Returns (MVT, its bins, (minus
    error, plus error), the overlap chance), with NaN if no block is certain enough."""
    cp = np.r_[blocks.i0.values, blocks.i1.values[-1]]
    prior = ncp_prior(len(counts))
    for k in sorted(range(first, last + 1), key=lambda k: blocks.nbins.values[k]):
        if k == 0 or k + 1 == len(blocks):
            continue                                            # an edge of the light curve, not a change point
        a, b = cp[k], cp[k + 1]
        edges_a, prob_a = change_point_profile(counts, lc.exposure, cp[k - 1], cp[k + 2], fixed=b, prior=prior)
        edges_b, prob_b = change_point_profile(counts, lc.exposure, cp[k - 1], cp[k + 2], fixed=a, prior=prior)
        overlap = float(prob_a[edges_a >= b].sum() + prob_b[edges_b <= a].sum())
        if overlap < MVT_OVERLAP:
            a_lo, a_hi = a - _percentile(edges_a, prob_a, 0.1587), _percentile(edges_a, prob_a, 0.8413) - a
            b_lo, b_hi = b - _percentile(edges_b, prob_b, 0.1587), _percentile(edges_b, prob_b, 0.8413) - b
            tres = lc.tres
            return 0.5 * (b - a) * tres, int(b - a), (0.5 * np.hypot(b_lo, a_hi) * tres, 0.5 * np.hypot(b_hi, a_lo) * tres), overlap
    return np.nan, 0, (np.nan, np.nan), np.nan


def _dips(rate):
    """Blocks whose rate is below both neighbours'."""
    dip = np.zeros(len(rate), bool)
    dip[1:-1] = (rate[1:-1] < rate[:-2]) & (rate[1:-1] < rate[2:])
    return dip




def _project_runs(blocks, rate, sigma, first, peak, min_bins):
    """The project rule's quiet runs (sample_tools.precursor_from_blocks, notebook 18) between the first emission block and the main peak.
    A run is a set of consecutive blocks below PROJECT_QUIET_SIGMA and below PROJECT_LULL of the highest rate so far. I also ask for at least
    min_bins in total, the same minimum as Zhu et al.'s 2026 QE, so that a block of one or two bins below the background doesn't count as a
    gap."""
    nbins = blocks.nbins.values
    runs, current, top = [], [], rate[first]
    for i in range(first + 1, peak):
        if sigma[i] < PROJECT_QUIET_SIGMA and rate[i] < PROJECT_LULL * top:
            current.append(i)
        else:
            if current:
                runs.append(current)
            current = []
        top = max(top, rate[i])
    if current:
        runs.append(current)
    return [r for r in runs if nbins[r].sum() >= min_bins]


def _quiet_dip_candidates(snr, long_dip, first, peak):
    return [i for i in np.flatnonzero(long_dip & (snr < QUIET_SIGMA)) if first < i < peak]


def _bright_precursor_peak(emission, long_dip, snr, rate, peak):
    """zhu2026 only: a candidate for the ME's peak when the PE is as bright as the ME, or brighter.

    I cut the emission at every dip that could be a QE. If the brightest block is in the first piece, a cut at the background (below
    AT_BACKGROUND_SIGMA per bin) follows it, and the brightest emission block after that cut is at least COMPARABLE_PEAK as high, then that
    block is the candidate. A spike followed by a much fainter tail, like a short burst with extended emission, keeps the spike as the ME.
    find_episodes only keeps the candidate if the rule then accepts a PE. Returns a block index (just peak when there is no candidate)."""
    loud = np.flatnonzero(emission)
    first, last = loud[0], loud[-1]
    cuts = [i for i in np.flatnonzero(long_dip & (snr < QUIET_SIGMA)) if first < i < last]
    if not cuts or peak > cuts[0]:
        return peak
    at_background = [i for i in cuts if i > peak and snr[i] < AT_BACKGROUND_SIGMA]
    if not at_background:
        return peak
    q = at_background[0]
    after = np.arange(q + 1, last + 1)
    after = after[emission[after]]
    if len(after) == 0:
        return peak
    candidate = int(after[np.argmax(rate[after])])
    return candidate if rate[candidate] >= COMPARABLE_PEAK * rate[peak] else peak


def find_episodes(lc, blocks, counts=None, mode="zhu2026", burst="", qe1_choice="longest", main_peak=None):
    """The episodes of one burst under one mode (see the module's docstring). Returns an EpisodeResult. main_peak (a block index) is
    used internally to try another block as the ME's peak (_bright_precursor_peak)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {list(MODES)}")
    settings = MODES[mode]
    out = EpisodeResult(burst=burst, mode=mode, bin_width=lc.tres)
    counts = lc.counts if counts is None else counts
    sigma, snr, rate, nbins = blocks.sigma.values, blocks.snr_bin.values, blocks.rate.values, blocks.nbins.values
    start, stop = blocks.start.values, blocks.stop.values
    emission = sigma >= EMISSION_BLOCK_SIGMA
    if emission[0]:
        # the light curve starts where the background interval ends, so an excess in its first block is more likely the fit than the burst
        # (GRB 111117A: 3.6 sigma over 1.4 s, ten seconds before the burst)
        emission[0] = False
        out.flags.append("excess_at_start_ignored")
    if not emission.any():
        out.flags.append("no_significant_block")
        return out
    if emission[-1]:
        out.flags.append("emission_reaches_background")          # the whole burst and the extended emission are lower limits
    loud = np.flatnonzero(emission)
    peak = loud[np.argmax(rate[loud])] if main_peak is None else int(main_peak)
    first, last = loud[0], loud[-1]
    long_dip = _dips(rate) & (nbins >= settings["qe_bins"])
    if settings["pe_test"] == "two_branch" and main_peak is None:
        moved = _bright_precursor_peak(emission, long_dip, snr, rate, peak)
        if moved != peak:
            trial = find_episodes(lc, blocks, counts, mode, burst, qe1_choice, main_peak=moved)
            if trial.pe is not None:
                trial.reasons.insert(0, f"me: the brightest block is in the first piece and a QE at the background follows; the brightest "
                                        f"emission after it peaks at {rate[moved] / rate[peak]:.2f} of it, so it is the ME's peak and the "
                                        f"first piece the PE")
                trial.flags.append("pe_peak_above_me_peak")
                return trial
            out.reasons.append("me: a later block was tried as the ME's peak (the first piece being brighter), but no PE was accepted with "
                               "it, so the brightest block stays the ME's peak")
    out.me_peak_rate, out.me_peak_sigma = float(rate[peak]), float(sigma[peak])

    # QE2 and EE first: they depend only on the main peak, and the zhu2026 test below compares the PE with the ME alone
    me_last = last
    for i in np.flatnonzero(_dips(rate) & (nbins >= max(settings["qe_bins"], QE2_BINS))):
        if i <= peak or i >= last:
            continue
        after = rate[i + 1:i + 1 + EE_RISE_BLOCKS]
        if rate[i] < EE_LULL_FRACTION * rate[peak] and after.max() >= EE_RISE * max(rate[i], 0.0) and emission[i + 1:].any():
            out.qe2, out.qe2_blocks = (float(start[i]), float(stop[i])), (int(i), int(i))
            out.ee, out.ee_blocks = (float(stop[i]), float(stop[last])), (int(i + 1), int(last))
            out.qe2_dip = float(rate[i] / rate[peak])
            out.qe2_recovery = float(after.max() / rate[i]) if rate[i] > 0 else np.inf
            out.reasons.append(f"qe2: first dip after the main peak of {nbins[i]} bins at {rate[i] / rate[peak]:.3f} of the peak, then x{out.qe2_recovery:.1f}")
            me_last = i - 1
            break

    # QE1 and PE
    pe_first = q1_first = q1_last = None
    if settings["pe_test"] == "project":
        runs = _project_runs(blocks, rate, sigma, first, peak, settings["qe_bins"])
        if runs:
            length = lambda run: stop[run[-1]] - start[run[0]]
            gap = runs[0]
            for run in runs[1:]:
                if length(run) > 2.0 * length(gap):
                    gap = run
            pe_first, q1_first, q1_last = first, gap[0], gap[-1]
            out.reasons.append(f"qe1: quiet run of {len(gap)} block(s) and {int(nbins[gap].sum())} bins, below {PROJECT_QUIET_SIGMA:g} sigma and "
                               f"{PROJECT_LULL:g} of the peak so far; the first of {len(runs)} run(s) unless a later one is twice as long")
        else:
            out.flags.append("no_quiet_gap_before_main")
    else:
        candidates = _quiet_dip_candidates(snr, long_dip, first, peak)
        if not candidates:
            out.flags.append("no_quiet_gap_before_main")
            near = [i for i in np.flatnonzero(long_dip) if first < i < peak and snr[i] < 2 * QUIET_SIGMA]
            if near:
                out.flags.append(f"nearest_miss_qe1_snr:{min(snr[near]):.2f}")
        if len(candidates) > 1:
            out.flags.append(f"quiet_gaps_before_main:{len(candidates)}")
        order = {"nearest": candidates[::-1], "first": candidates, "longest": sorted(candidates, key=lambda i: (-nbins[i], i))}[qe1_choice]
        reasons = []
        for rank, q1 in enumerate(order, 1):
            pe_peak = first + int(np.argmax(np.where(emission[first:q1], rate[first:q1], -np.inf)))
            if settings["pe_test"] == "weaker":
                passed, reason = rate[pe_peak] < rate[peak], "pe_not_weaker_than_me"
                why = "PE peak below the ME's"
            elif snr[q1] < AT_BACKGROUND_SIGMA:
                pe_net, me_net = blocks.net.values[first:q1].sum(), blocks.net.values[q1 + 1:me_last + 1].sum()
                passed, reason = pe_net < me_net, "pe_counts_not_below_me"
                why = "QE at the background and PE counts below the ME's"
            else:
                pull = (rate[peak] - rate[pe_peak]) / np.hypot(blocks.rate_err.values[peak], blocks.rate_err.values[pe_peak])
                passed, reason = pull > ME_OVER_PE_SIGMA, "me_peak_not_clearly_above_pe"
                why = f"ME peak {pull:.1f} sigma above the PE's"
            if passed:
                pe_first, q1_first, q1_last = first, q1, q1
                out.reasons.append(f"qe1: dip {rank} of {len(candidates)} in {qe1_choice}-first order, {nbins[q1]} bins at {snr[q1]:.2f} per bin; {why}")
                break
            reasons.append(reason)
        if candidates and pe_first is None:
            out.flags.extend(sorted(set(reasons)))
    me_first = first
    if pe_first is not None and q1_first > pe_first:
        pe_peak = pe_first + int(np.argmax(np.where(emission[pe_first:q1_first], rate[pe_first:q1_first], -np.inf)))
        out.pe, out.pe_blocks = (float(start[pe_first]), float(start[q1_first])), (int(pe_first), int(q1_first - 1))
        out.qe1, out.qe1_blocks = (float(start[q1_first]), float(stop[q1_last])), (int(q1_first), int(q1_last))
        out.pe_peak_rate, out.pe_peak_sigma = float(rate[pe_peak]), float(sigma[pe_peak])
        q1_rate = float(np.sum(blocks.net.values[q1_first:q1_last + 1]) / (stop[q1_last] - start[q1_first]))
        out.qe1_snr_bin = float(np.sum(blocks.net.values[q1_first:q1_last + 1]) / np.sqrt(max(np.sum(blocks.bkg.values[q1_first:q1_last + 1]), 1.0) * np.sum(nbins[q1_first:q1_last + 1])))
        out.qe1_dip = q1_rate / out.pe_peak_rate if out.pe_peak_rate > 0 else np.nan
        out.qe1_recovery = out.me_peak_rate / q1_rate if q1_rate > 0 else np.inf
        if settings["mvt"] == "brightest":
            out.mvt_pe, out.mvt_pe_bins = brightest_rise(lc, counts, blocks, pe_first, q1_first - 1)
        else:
            out.mvt_pe, out.mvt_pe_bins, out.mvt_pe_err, _ = mvt_of(lc, counts, blocks, pe_first, q1_first - 1)
            if not np.isfinite(out.mvt_pe):
                out.mvt_pe, out.mvt_pe_bins = 0.5 * (out.pe[1] - out.pe[0]), int(blocks.i0.values[q1_first] - blocks.i0.values[pe_first])
                out.flags.append("mvt_pe_is_half_the_precursor")
        me_first = q1_last + 1

    out.me, out.me_blocks = (float(start[me_first]), float(stop[me_last])), (int(me_first), int(me_last))
    if settings["mvt"] == "brightest":
        out.mvt_me, out.mvt_me_bins = brightest_rise(lc, counts, blocks, me_first, me_last)
    else:
        out.mvt_me, out.mvt_me_bins, out.mvt_me_err, _ = mvt_of(lc, counts, blocks, me_first, me_last)
        if not np.isfinite(out.mvt_me):
            out.mvt_me, out.mvt_me_bins = 0.5 * (out.me[1] - out.me[0]), int(blocks.i1.values[me_last] - blocks.i0.values[me_first])
            out.flags.append("mvt_me_is_half_the_main_emission")
    out.we = (out.pe[0] if out.pe else out.me[0], float(stop[last]))
    return out


def pulses(blocks, first, last):
    """The pulses among blocks first..last. A pulse is a block with a higher rate than both of its neighbours, taken out to the nearest block
    on each side that is lower than both of its own neighbours (or to the end of the range). Returns a table with each pulse's start, stop,
    peak block, rise (from its start to the middle of the peak block), and the length of the run of blocks above half the peak rate."""
    rate = blocks.rate.values
    start, stop = blocks.start.values, blocks.stop.values
    is_dip = lambda k: 0 < k < len(rate) - 1 and rate[k] < rate[k - 1] and rate[k] < rate[k + 1]
    rows = []
    for k in range(first, last + 1):
        if not (rate[k] > (rate[k - 1] if k > 0 else -np.inf) and rate[k] > (rate[k + 1] if k + 1 < len(rate) else -np.inf)):
            continue
        a = k
        while a - 1 >= first and not is_dip(a - 1):
            a -= 1
        b = k
        while b + 1 <= last and not is_dip(b + 1):
            b += 1
        lo = k
        while lo - 1 >= first and rate[lo - 1] >= 0.5 * rate[k]:
            lo -= 1
        hi = k
        while hi + 1 <= last and rate[hi + 1] >= 0.5 * rate[k]:
            hi += 1
        rows.append(dict(start=start[a], stop=stop[b], peak=k, rise=0.5 * (start[k] + stop[k]) - start[a], half_width=stop[hi] - start[lo]))
    return pd.DataFrame(rows, columns=["start", "stop", "peak", "rise", "half_width"])


def brightest_rise(lc, counts, blocks, first, last):
    """Zhu et al.'s T_MVT the way their tables have it: "the rise time of the shortest pulse" among blocks first..last, measured from the
    start of the pulse (pulses) to the centre of its brightest bin, or from the start of the blocks if none of them is a pulse.

    The reason for this reading is in their own numbers. Every T_MVT they print is a whole number of half-bins, and most are odd, which is what
    you get for a time from a bin edge to a bin centre. Inside one Bayesian block the rate is constant within the noise, so there the
    brightest bin is wherever the counting noise puts it, and the realisations in with_errors show how much that spreads. Returns (T_MVT,
    twice T_MVT in bins), so that T_MVT = bins x Tres / 2, the same as for the half-block readings."""
    p = pulses(blocks, first, last)
    if len(p):
        pulse = p.iloc[int(np.argmin((p.stop - p.start).values))]
        t0, t1 = float(pulse.start), float(pulse.stop)
    else:
        t0, t1 = float(blocks.start.values[first]), float(blocks.stop.values[last])
    inside = np.flatnonzero((lc.edges[:-1] >= t0 - 1e-9) & (lc.edges[1:] <= t1 + 1e-9))
    if len(inside) == 0:
        return np.nan, 0
    k = inside[int(np.argmax(np.asarray(counts, float)[inside]))]
    bins = 2 * int(k - inside[0]) + 1
    return 0.5 * bins * lc.tres, bins


def mvt_readings(lc, counts, blocks, first, last):
    """Five ways of reading "the rise time of the shortest pulse identified with Bayesian blocks", to compare with their table.
    - brightest: from the shortest pulse's start to the centre of its brightest bin (brightest_rise). This is the zhu modes' T_MVT.
    - certain: half the shortest block whose edges are certain at 5 sigma (mvt_of, Xiao et al. 2023). This is the project mode's T_MVT.
    - shortest: half the shortest block, without testing its edges.
    - rise: the rise time of the shortest pulse (pulses), from its start to the middle of its peak block.
    - half_width: half the shortest run of blocks above half a pulse's peak rate."""
    inside = blocks.iloc[first:last + 1]
    p = pulses(blocks, first, last)
    shortest_pulse = p.iloc[int(np.argmin((p.stop - p.start).values))] if len(p) else None
    return dict(brightest=brightest_rise(lc, counts, blocks, first, last)[0], certain=mvt_of(lc, counts, blocks, first, last)[0],
                shortest=0.5 * float((inside.stop - inside.start).min()),
                rise=float(shortest_pulse.rise) if shortest_pulse is not None else np.nan,
                half_width=0.5 * float(p.half_width.min()) if len(p) else np.nan)


def simulate(lc, blocks, rng):
    """Poisson counts from the block model of the light curve: each block's observed rate, spread over its bins in proportion to live time."""
    expected = np.empty(len(lc.counts))
    edges_index = np.searchsorted(lc.edges, blocks.start.values, side="left")
    for (a, b), n, live in zip(zip(edges_index, np.r_[edges_index[1:], len(lc.counts)]), blocks.n.values, blocks.live.values):
        expected[a:b] = n * lc.exposure[a:b] / live
    return rng.poisson(expected)




_SIM = {}


def _one_realisation(seed):
    lc, blocks, modes = _SIM["lc"], _SIM["blocks"], _SIM["modes"]
    counts = simulate(lc, blocks, np.random.default_rng(seed))
    new = blocks_of(lc, counts)
    return [dict(seed=seed, mode=m, **find_episodes(lc, new, counts, mode=m).durations()) for m in modes]


def with_errors(lc, blocks, realisations=40, seed=2026, workers=1, modes=tuple(MODES)):
    """Every number of every mode, measured again on light curves simulated from the burst's own block model.

    Each realisation draws Poisson counts from the blocks (simulate), finds new blocks, and applies each mode's rule to them. So the spread
    shows how much the numbers move with counting noise alone, for the burst's shape as the blocks see it. Realisation i uses seed + i, so the
    result doesn't depend on the number of workers. Returns a table with one row per realisation and mode."""
    _SIM.update(lc=lc, blocks=blocks, modes=modes)
    seeds = [seed + i for i in range(realisations)]
    if workers > 1:
        with multiprocessing.get_context("fork").Pool(workers, initializer=worker_start) as pool:
            rows = pool.map(_one_realisation, seeds, chunksize=1)
    else:
        rows = [_one_realisation(s) for s in seeds]
    return pd.DataFrame([r for group in rows for r in group])


def summarise(realisations, columns=("T100_PE", "T100_ME", "T100_EE", "T100_WE", "T100_QE1", "T100_QE2", "TMVT_PE", "TMVT_ME")):
    """Median and 16th/84th percentiles of each number over the realisations that find it, and how often a PE and an EE are found, per mode."""
    out = {}
    for mode, table in realisations.groupby("mode"):
        s = {"pe_found_fraction": float(table.has_pe.mean()), "ee_found_fraction": float(table.has_ee.mean())}
        for column in columns:
            values = table[column].dropna()
            if len(values):
                lo, mid, hi = np.percentile(values, [15.87, 50.0, 84.13])
                s.update({f"{column}_median": mid, f"{column}_lo": lo, f"{column}_hi": hi})
        out[mode] = s
    return out


def at_resolutions(burst, tres, factors=(0.5, 2.0), max_bins=MAX_BINS, modes=tuple(MODES)):
    """Every mode on the light curve binned at other multiples of their Tres. The rule's lengths in bins stay in bins, so they change in
    seconds too, as theirs would. Returns rows (factor, mode, numbers); a factor needing more than max_bins bins is skipped and noted."""
    rows = []
    for f in factors:
        lc = light_curve(burst, tres * f)
        if len(lc.counts) > max_bins:
            rows.append(dict(factor=f, skipped="too_many_bins"))
            continue
        blocks = blocks_of(lc)
        for m in modes:
            found = find_episodes(lc, blocks, mode=m)
            rows.append(dict(factor=f, tres=lc.tres, mode=m, flags=" ".join(found.flags), **found.durations()))
    return pd.DataFrame(rows)
