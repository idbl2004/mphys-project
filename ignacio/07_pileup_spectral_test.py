"""
Is there pulse pile-up in GRB 230307A, and is it what defines the 3-7 s
bad time interval?

06_pileup_deadtime_gaps.py showed that dead time is as severe at ~1.5-2 s and
~8-10 s as inside 3-7 s, so dead time alone doesn't explain why 3-7 s
was flagged. Pulse pile-up is the other candidate: two photons arriving
within ~1 us are recorded as one event with the summed energy. That
moves counts to higher energies and into the top (overflow) channel, and
it grows with the square of the rate.

The burst's own spectrum also changes with time, so a hardening in one
detector proves nothing. The idea here is to compare detectors: all NaI
see the same burst at different rates (na ~2.5x n8), and pile-up grows
with rate. For each NaI we take a spectral indicator divided by the same
indicator in n8 (the quietest), normalised to 1 in a reference stretch
where rates are moderate. BGO b1 is left out: its response differs from
a NaI's even more.

Two indicators, both background-subtracted:
- hardness = counts 300-850 keV / counts 10-50 keV
- overflow fraction = counts in the top channel / counts 10-850 keV
They are read two ways: summed over whole windows (3-7 s, busy times
outside it, reference) and as the slope of the bin-by-bin double ratio
against the detector's rate.

Limitation (only clear once I ran it): the double ratio only cancels the
burst's spectral evolution if both detectors respond to energy in the
same way. They don't: na faces the burst most directly and n8 at a larger
angle, so when the burst is hard na picks up relatively more high-energy
counts, with no pile-up at all. Because the burst is brightest when it is
hardest, rate and hardness rise together and the slope test is
confounded.

Result for GRB 230307A: n6/n8 stays at ~1 throughout. na/n8 rises to
~3 early on and falls back to ~1 by ~18 s, but at equal na rates
(e.g. ~36e3 counts/s at ~1.8 s and at ~13 s) the ratio is ~3.3 early and
~2.0 later: it follows the time (the burst getting softer), not the rate.
That points to the different detector responses, not pile-up. The size of
the effect also argues against pile-up: at a few percent of events piled,
mostly from photons below 100 keV, the 300-850 keV band cannot triple.
So this test does not show pile-up; a proper test needs the detector
responses (fit n8, predict na, look for a rate-dependent mismatch), which
belongs with the spectral fits.

Output: 07_pileup_spectral_test.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gdt.missions.fermi.gbm.tte import GbmTte

from settings import DATA_DIR, BURST_ID, BKG_WINDOWS, BAD_TIME_INTERVAL, FIGURE_DIR
from lightcurve_tools import (aligned_bin_edges, band_counts, fit_background,
                              background_in_band, tte_path)

REFERENCE_DET = "n8"                 # lowest rate of the three NaI
TEST_DETS = ["n6", "na"]
BIN_WIDTH = 0.256                    # s, enough counts per bin for ratios
TIME_RANGE = (0.0, 40.0)             # s
SOFT = (10.0, 50.0)                  # keV
HARD = (300.0, 850.0)                # keV, kept below the top channel
NORM_WINDOW = (18.0, 30.0)           # s, moderate rates: normalise ratios here
BUSY_OUTSIDE = [(1.4, 2.2), (7.8, 10.5)]   # s, deep dead time outside the BTI
MIN_COUNTS = 30                      # per bin and per piece, for the bin-by-bin plot


def top_channel_band(det):
    """Energy range that selects only the detector's top (overflow) channel."""
    tte = GbmTte.open(tte_path(DATA_DIR, BURST_ID, det))
    lo, hi = tte.ebounds.low_edges()[-1], tte.ebounds.high_edges()[-1]
    tte.close()
    return (lo + 1.0, hi - 1.0)      # stay inside so no neighbouring channel overlaps


def net_counts(det, band, edges, fitter):
    """Background-subtracted counts per bin and their 1-sigma error."""
    raw = band_counts(DATA_DIR, BURST_ID, det, band, edges)
    bkg, bkg_err = background_in_band(fitter, band, edges)
    return raw - bkg, np.sqrt(raw + bkg_err**2)


def ratio(a, a_err, b, b_err):
    r = a / b
    return r, np.abs(r) * np.sqrt((a_err / a)**2 + (b_err / b)**2)


def mask_of(windows, t):
    return np.any([(t > lo) & (t < hi) for lo, hi in windows], axis=0)


edges = aligned_bin_edges(TIME_RANGE, BIN_WIDTH)
t = 0.5 * (edges[:-1] + edges[1:])
windows = {
    "inside 3-7 s": mask_of([BAD_TIME_INTERVAL], t),
    "busy, outside 3-7 s": mask_of(BUSY_OUTSIDE, t),
    "reference": mask_of([NORM_WINDOW], t),
}

# Net counts (and variance) per bin for each piece of each indicator
parts = {}
for det in [REFERENCE_DET] + TEST_DETS:
    fitter, _ = fit_background(DATA_DIR, BURST_ID, det, BKG_WINDOWS)
    soft, soft_err = net_counts(det, SOFT, edges, fitter)
    hard, hard_err = net_counts(det, HARD, edges, fitter)
    top, top_err = net_counts(det, top_channel_band(det), edges, fitter)
    parts[det] = {
        "rate": band_counts(DATA_DIR, BURST_ID, det, (0.0, 1e5), edges) / BIN_WIDTH,
        "hardness": (hard, hard_err**2, soft, soft_err**2),
        "overflow": (top, top_err**2, soft + hard, soft_err**2 + hard_err**2),
    }


def indicator(det, name, sel):
    """Indicator from counts summed over the selected bins (robust at low counts)."""
    num, num_var, den, den_var = parts[det][name]
    return ratio(num[sel].sum(), np.sqrt(num_var[sel].sum()),
                 den[sel].sum(), np.sqrt(den_var[sel].sum()))


def double_ratio(det, name, sel):
    """(det / reference detector) in sel, divided by the same in the reference window."""
    here = ratio(*indicator(det, name, sel), *indicator(REFERENCE_DET, name, sel))
    norm = ratio(*indicator(det, name, windows["reference"]),
                 *indicator(REFERENCE_DET, name, windows["reference"]))
    return ratio(*here, *norm)


fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for det in [REFERENCE_DET] + TEST_DETS:
    axes[0].plot(t, parts[det]["rate"] / 1e3, lw=0.8, label=det)

for row, name in [(1, "hardness"), (2, "overflow")]:
    print(f"\n{name} double ratio (detector / {REFERENCE_DET}, =1 in "
          f"{NORM_WINDOW[0]:.0f}-{NORM_WINDOW[1]:.0f} s):")
    for det in TEST_DETS:
        for label, sel in windows.items():
            m, e = double_ratio(det, name, sel)
            print(f"  {det} {label:22s} {m:.3f} +/- {e:.3f}")

        # Bin by bin, keeping only bins with enough counts for a ratio to be Gaussian
        per_bin = [double_ratio(det, name, np.arange(len(t)) == i) for i in range(len(t))]
        dr, dr_err = np.array(per_bin).T
        enough = np.ones(len(t), dtype=bool)
        for d in (det, REFERENCE_DET):
            num, _, den, _ = parts[d][name]
            enough &= (num > MIN_COUNTS) & (den > MIN_COUNTS)
        good = enough & np.isfinite(dr) & (dr_err > 0)

        # Slope of the double ratio against this detector's rate: zero if no pile-up
        rate_k = parts[det]["rate"][good] / 1e3
        (slope, _), cov = np.polyfit(rate_k, dr[good], 1, w=1 / dr_err[good], cov="unscaled")
        slope_err = np.sqrt(cov[0, 0])
        print(f"  {det} slope vs rate: {slope:+.4f} +/- {slope_err:.4f} per 10^3 counts/s "
              f"({abs(slope) / slope_err:.1f} sigma from zero, {good.sum()} bins)")

        axes[row].errorbar(t[good], dr[good], yerr=dr_err[good], fmt=".", ms=3,
                           elinewidth=0.6, label=f"{det} / {REFERENCE_DET}")
    axes[row].axhline(1, color="black", lw=0.6)
    axes[row].set_ylabel(f"{name}\ndouble ratio")
    axes[row].legend(loc="upper right")

axes[0].set_ylabel("Recorded rate\n[10$^3$ counts/s]")
axes[0].legend(loc="upper right")
for ax in axes:
    ax.axvspan(*BAD_TIME_INTERVAL, color="grey", alpha=0.15)
    for lo, hi in BUSY_OUTSIDE:
        ax.axvspan(lo, hi, color="tab:orange", alpha=0.1)
    ax.axvspan(*NORM_WINDOW, color="tab:blue", alpha=0.05)
axes[-1].set_xlabel("Time since GBM trigger [s]")
axes[-1].set_xlim(*TIME_RANGE)
axes[0].set_title("Grey: 3-7 s BTI   orange: busy outside it   blue: normalisation")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "07_pileup_spectral_test.png", dpi=200)
print(f"\nsaved {FIGURE_DIR / '07_pileup_spectral_test.png'}")