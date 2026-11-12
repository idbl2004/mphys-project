"""Spectral lags in their bands, at a time resolution chosen by a written rule, measured two ways.

Zhu et al. measure the lag of 25-50 keV behind 50-100 keV (tau32, 2025 and 2026) and behind 100-200 keV (tau42, 2025) from the peak of the
cross-correlation function (CCF), with a time resolution chosen for each episode, and leave a lag out "if the CCF cannot be reliably fitted
at a time resolution below 256 ms". Positive means the softer band comes later, as in long bursts.

Here the resolution is the finest of 4, 8, ... 256 ms (and no finer than their Tres) at which
  - both bands are at least MIN_SIGMA above the background in the episode,
  - the episode is at least MIN_BINS bins long, and
  - at least MIN_PEAK_FRACTION of the Poisson resamples have a real CCF peak (ccf_peak): the highest point at least 4 bins from either end
    of the range of lags, a parabola through the 9 points around it that opens downwards, and its top inside those 9 points.
That last test is our reading of "reliably fitted". It is made first on TRIAL_RESAMPLES as a quick screen, and then again on the 1000
resamples the quoted lag comes from, so a quoted lag always meets it on its own resamples. A lag that fails is not quoted: its status says
why (faint, too few bins, or no reliable peak up to 256 ms), and a lag that raises an error is stored as failed by run.py. The lag is
measured two ways at that resolution:
with the soft band running MARGIN_BINS past the episode, so every trial lag uses the same bins, and with both bands cut at the episode,
the usual recipe and probably theirs, which pulls lags towards zero. The value is the median of 1000 Poisson resamples with a real peak and
the errors their 16th and 84th percentiles; the cut version is judged by the same rule on its own resamples and has its own status.
"""
import numpy as np
from gdt.core.data_primitives import TimeBins
from gdt.misc.bayesian_blocks import bayesian_blocks
from scipy.signal import correlate

from grbtool.bursts import event_times
from grbtool.physics import MARGIN_BINS
from lightcurve_tools import aligned_bin_edges, fit_background
from .background import background_counts
from sample_tools import ccf_lag

BANDS = {"tau32": ((25.0, 50.0), (50.0, 100.0)), "tau42": ((25.0, 50.0), (100.0, 200.0))}      # (soft, hard), keV
RESOLUTIONS = tuple(2.0**k * 1e-3 for k in range(2, 9))         # 4 ms to 256 ms
MIN_SIGMA = 5.0
MIN_BINS = 8
MIN_PEAK_FRACTION = 0.9
TRIAL_RESAMPLES = 200           # for the "reliably fitted" test at each resolution; the quoted lag uses 1000


def ccf_lag_fft(low, high, bin_width, half_width=4, cut=False):
    """sample_tools.ccf_lag with the cut version's cross-correlation done by FFT, which gives the same lag (tests/test_zhu.py checks it).

    With both bands cut at the episode every trial lag is needed, and a direct correlation of a 20,000-bin episode takes about 0.3 s, so the
    1000 resamples of a long episode took minutes. The version with the soft band running past the episode only needs 2 MARGIN_BINS + 1
    trial lags and is left to ccf_lag.
    """
    if not cut:
        return ccf_lag(low, high, bin_width, half_width)
    c = correlate(low, high, mode="full", method="auto")      # scipy picks FFT when it is faster
    lags = np.arange(len(c)) - (len(high) - 1)
    keep = np.abs(lags) <= len(high) // 2
    c, lags = c[keep], lags[keep]
    k = int(np.argmax(c))
    near = np.arange(max(k - half_width, 0), min(k + half_width + 1, len(c)))
    a, b, _ = np.polyfit(lags[near], c[near], 2)
    peak = -b / (2 * a) if a < 0 else lags[k]
    return float(np.clip(peak, lags[near[0]], lags[near[-1]])) * bin_width


def ccf_peak(low, high, bin_width, half_width=4, cut=False):
    """The lag of sample_tools.ccf_lag (same CCF, same parabola), and whether it is a real peak: the highest point of the CCF is at least
    half_width bins from either end of the range of lags, the parabola through the points within half_width of it opens downwards, and
    its top lies strictly inside those points. ccf_lag only checks the first, and only for the version with the margin; a parabola that
    opens upwards, or whose top is clipped to the edge of the points, is a shoulder or a plateau rather than a peak. Returns (lag in s, real)."""
    if cut:
        c = correlate(low, high, mode="full", method="auto")
        lags = np.arange(len(c)) - (len(high) - 1)
        keep = np.abs(lags) <= len(high) // 2
        c, lags = c[keep], lags[keep]
    else:
        c = correlate(low, high, mode="valid", method="auto")
        lags = np.arange(len(c)) - (len(c) - 1) // 2
    k = int(np.argmax(c))
    if k < half_width or k >= len(c) - half_width:
        return np.nan, False
    near = np.arange(k - half_width, k + half_width + 1)
    a, b, _ = np.polyfit(lags[near], c[near], 2)
    if not a < 0:
        return float(lags[k]) * bin_width, False
    top = -b / (2 * a)
    return float(np.clip(top, lags[near[0]], lags[near[-1]])) * bin_width, bool(lags[near[0]] < top < lags[near[-1]])


def lag_with_errors(soft, soft_b, hard, hard_b, bin_width, realisations=1000, seed=20, half_width=4, cut=False):
    """The median lag of the Poisson resamples whose CCF has a real peak (ccf_peak), the distances to their 16th and 84th percentiles,
    in s, and the share of resamples with a real peak. NaNs if fewer than half of them have one."""
    rng = np.random.default_rng(seed)
    found = [ccf_peak(rng.poisson(soft) - soft_b, rng.poisson(hard) - hard_b, bin_width, half_width, cut) for _ in range(realisations)]
    real = np.array([lag for lag, ok in found if ok])
    share = real.size / realisations
    if real.size < 0.5 * realisations:
        return np.nan, np.nan, np.nan, share
    down, middle, up = np.percentile(real, [15.87, 50.0, 84.13])
    return middle, middle - down, up - middle, share


class EpisodeLags:
    """Photons of the burst's NaI detectors in every band, and their background fits, read once for all the lags of one burst."""

    def __init__(self, burst, bands=BANDS):
        self.burst, self.bands = burst, bands
        wanted = sorted({b for pair in bands.values() for b in pair})
        self.events = {band: event_times(burst, burst.nai, band) for band in wanted}
        self.fits = {d: fit_background(burst.data_dir, burst.trigger, d, burst.bkg_windows, burst.bkg_order)[0] for d in burst.nai}

    def curve(self, band, edges):
        """Counts and fitted background in each bin, summed over the detectors."""
        counts, background = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
        for d in self.burst.nai:
            counts += np.histogram(self.events[band][d], bins=edges)[0]
            background += background_counts(self.fits[d], band, edges)
        return counts, background

    def curves(self, name, window, bin_width):
        """Soft band over the window plus MARGIN_BINS each side, hard band over the window, and the significance of each in the window."""
        soft_band, hard_band = self.bands[name]
        edges = aligned_bin_edges(window, bin_width)
        wide = np.r_[edges[0] - bin_width * np.arange(MARGIN_BINS, 0, -1), edges, edges[-1] + bin_width * np.arange(1, MARGIN_BINS + 1)]
        soft, soft_b = self.curve(soft_band, wide)
        hard, hard_b = self.curve(hard_band, edges)
        inside = slice(MARGIN_BINS, MARGIN_BINS + len(edges) - 1)
        soft_sigma = (soft - soft_b)[inside].sum() / np.sqrt(max(soft[inside].sum(), 1.0))
        hard_sigma = (hard - hard_b).sum() / np.sqrt(max(hard.sum(), 1.0))
        return soft, soft_b, hard, hard_b, inside, soft_sigma, hard_sigma

    def peak_fraction(self, soft, soft_b, hard, hard_b, bin_width, realisations=TRIAL_RESAMPLES, seed=32):
        """Share of Poisson resamples whose CCF (soft band with the margin) has a real peak (ccf_peak)."""
        rng = np.random.default_rng(seed)
        return float(np.mean([ccf_peak(rng.poisson(soft) - soft_b, rng.poisson(hard) - hard_b, bin_width)[1] for _ in range(realisations)]))

    def measure(self, name, window, tres=0.0, realisations=1000):
        """The lag `name` (tau32 or tau42) of one episode, at the finest reliable resolution. Lags in ms; NaN with a reason if none is reliable."""
        tried = []
        for bin_width in RESOLUTIONS:
            if bin_width < tres - 1e-12:
                continue
            nbins = int(round((window[1] - window[0]) / bin_width))
            if nbins < MIN_BINS:
                tried.append(f"{1e3 * bin_width:.0f}ms:too_few_bins")
                break
            soft, soft_b, hard, hard_b, inside, soft_sigma, hard_sigma = self.curves(name, window, bin_width)
            if soft_sigma < MIN_SIGMA or hard_sigma < MIN_SIGMA:
                tried.append(f"{1e3 * bin_width:.0f}ms:faint")
                break                                   # coarser bins don't add counts
            fraction = self.peak_fraction(soft, soft_b, hard, hard_b, bin_width)
            if fraction < MIN_PEAK_FRACTION:
                tried.append(f"{1e3 * bin_width:.0f}ms:peak_in_{fraction:.3f}")
                continue
            lag, down, up, share = lag_with_errors(soft, soft_b, hard, hard_b, bin_width, realisations)
            if share < MIN_PEAK_FRACTION:                     # passed the quick screen but not on its own resamples
                tried.append(f"{1e3 * bin_width:.0f}ms:peak_in_{share:.3f}_of_{realisations}")
                continue
            cut, cut_down, cut_up, cut_share = lag_with_errors(soft[inside], soft_b[inside], hard, hard_b, bin_width, realisations, cut=True)
            cut_ok = cut_share >= MIN_PEAK_FRACTION
            data_lag, data_real = ccf_peak(soft - soft_b, hard - hard_b, bin_width)
            return dict(lag=name, status="measured", res_ms=1e3 * bin_width,
                        lag_ms=1e3 * lag, err_down_ms=1e3 * down, err_up_ms=1e3 * up, lag_data_ms=1e3 * data_lag, data_peak_real=data_real,
                        lag_cut_ms=1e3 * cut if cut_ok else np.nan, cut_down_ms=1e3 * cut_down if cut_ok else np.nan,
                        cut_up_ms=1e3 * cut_up if cut_ok else np.nan, cut_status="measured" if cut_ok else f"unusable:peak_in_{cut_share:.3f}",
                        peak_fraction=fraction, resamples_with_peak=share, cut_resamples_with_peak=cut_share, soft_sigma=soft_sigma,
                        hard_sigma=hard_sigma, tried=" ".join(tried))
        last = tried[-1] if tried else "none"
        status = ("unusable:faint" if last.endswith("faint") else "unusable:too_few_bins" if last.endswith("too_few_bins")
                  else "unusable:no_reliable_peak" if "peak_in" in last else "unusable")
        return dict(lag=name, status=status, res_ms=np.nan, lag_ms=np.nan, err_down_ms=np.nan, err_up_ms=np.nan, lag_data_ms=np.nan,
                    data_peak_real=False, lag_cut_ms=np.nan, cut_down_ms=np.nan, cut_up_ms=np.nan, cut_status=status, peak_fraction=np.nan,
                    resamples_with_peak=np.nan, cut_resamples_with_peak=np.nan, soft_sigma=np.nan, hard_sigma=np.nan, tried=" ".join(tried) or "none")


def known_shift_check(times, window, bin_width, shifts=(0.0, 0.006, 0.020), realisations=300, seed=3):
    """The lag code on light curves with a known answer. Returns {shift: (mean lag, scatter, error of the mean)} in s.

    A rate model is made from real photons (their Bayesian blocks at 4 ms, over the window and 1 s either side). Two light curves are drawn
    from it with independent Poisson noise, each with half the counts, the soft one delayed by the shift, and the lag is measured with
    sample_tools.ccf_lag. Over many draws the mean should be the shift. (Splitting the real photons into two halves instead leaves an
    offset of about a quarter of a bin, because the halves share each bin's photons and so are anticorrelated at exactly the true lag.)
    """
    fine = aligned_bin_edges((window[0] - 1.0, window[1] + 1.0), 0.004)
    counts = np.histogram(times, bins=fine)[0].astype(float)
    cp = bayesian_blocks(TimeBins(counts, fine[:-1], fine[1:], np.diff(fine)), p0=0.05)
    block_edges = fine[cp]
    cumulative = np.r_[0.0, np.cumsum(np.add.reduceat(counts, cp[:-1]))]
    expected = lambda edges, shift: 0.5 * np.diff(np.interp(edges - shift, block_edges, cumulative))
    edges = aligned_bin_edges(window, bin_width)
    wide = np.r_[edges[0] - bin_width * np.arange(MARGIN_BINS, 0, -1), edges, edges[-1] + bin_width * np.arange(1, MARGIN_BINS + 1)]
    rng = np.random.default_rng(seed)
    out = {}
    for shift in shifts:
        soft, hard = expected(wide, shift), expected(edges, 0.0)
        lags = np.array([ccf_lag(rng.poisson(soft).astype(float), rng.poisson(hard).astype(float), bin_width) for _ in range(realisations)])
        good = np.isfinite(lags)
        out[shift] = (float(np.mean(lags[good])), float(np.std(lags[good])), float(np.std(lags[good]) / np.sqrt(good.sum())))
    return out
