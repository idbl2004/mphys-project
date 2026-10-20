"""
How badly is GRB 230307A affected by the detector saturating at 3-7 s?

Second step past the Fig. 1 reproduction (03_reproduce_dichiara_fig1.py).
Dichiara (and the GBM team) only flag 3-7 s as a bad time interval and
exclude it. Here two effects are measured for each detector, using all
energy channels, since every recorded photon costs the detector time:

1. Dead time. After each event the electronics are busy for 2.6 us
   (10 us for the overflow channel), so at high rates part of each bin is
   "dead" and photons arriving then are lost. GDT computes the live time
   of each bin from the recorded events; live fraction = live time / width.
2. Data gaps. If the TTE data stream overflows, whole stretches of events
   are dropped. At a rate r, a gap of length dt between consecutive events
   has probability exp(-r*dt), so any gap with r*dt > 20 (chance ~2e-9)
   is not Poisson noise but lost data. The search is repeated at 10 and
   30 to show the result doesn't hinge on that choice.

Pile-up proper (two photons arriving within ~1 us and being recorded as
one photon with the summed energy) also distorts the spectrum; that can't
be undone with counts alone and is why the interval stays excluded from
spectral fits. This script measures how much is lost, not how to fix it.

Output: 06_pileup_check.png (total rate, live fraction, gaps per detector).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.binning.unbinned import bin_by_edges

from settings import DATA_DIR, BURST_ID, BAD_TIME_INTERVAL, FIGURE_DIR
from lightcurve_tools import aligned_bin_edges, tte_path

DETECTORS = ["n6", "n8", "na", "b1"]
BIN_WIDTH = 0.016                # s
TIME_RANGE = (-5.0, 50.0)        # s
GAP_SIGNIFICANCE = 20.0          # flag gaps with rate * gap > this
ROBUSTNESS_LEVELS = [10.0, 20.0, 30.0]   # rerun the gap search at these

edges = aligned_bin_edges(TIME_RANGE, BIN_WIDTH)
centres = 0.5 * (edges[:-1] + edges[1:])
in_bti = (centres > BAD_TIME_INTERVAL[0]) & (centres < BAD_TIME_INTERVAL[1])
outside = ~in_bti & (centres > 0)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for det in DETECTORS:
    tte = GbmTte.open(tte_path(DATA_DIR, BURST_ID, det))
    times = np.asarray(tte.data.times)
    phaii = tte.to_phaii(bin_by_edges, edges)
    tte.close()

    counts = np.asarray(phaii.data.counts).sum(axis=1)
    live_fraction = np.asarray(phaii.data.exposure) / BIN_WIDTH
    rate = counts / BIN_WIDTH                       # recorded counts/s, all energies

    print(f"{det}: peak recorded rate {rate.max():.0f} counts/s at t = {centres[rate.argmax()]:.3f} s")

    # Gaps between consecutive events, judged against the local rate
    times = times[(times > TIME_RANGE[0]) & (times < TIME_RANGE[1])]
    gap_lengths = np.diff(times)
    gap_bins = np.clip(np.digitize(times[:-1], edges) - 1, 0, len(rate) - 1)
    # Take the busier of the bin before and after: the gap itself drags its own bin down
    local_rate = np.maximum(rate[gap_bins], rate[np.clip(gap_bins + 1, 0, len(rate) - 1)])
    # Robustness: how much do the results depend on the threshold?
    # A gap with rate*gap > 10 has chance e^-10 ~ 5e-5 per gap; with ~1e6
    # events in the burst a few such gaps are expected by chance at 10, none at 20.
    for level in ROBUSTNESS_LEVELS:
        flagged = local_rate * gap_lengths > level
        print(f"    threshold {level:4.0f}: {flagged.sum():3d} gaps, "
              f"total {1e3 * gap_lengths[flagged].sum():5.1f} ms, "
              f"expected by chance {len(gap_lengths) * np.exp(-level):.2g}")
    # Shortest gap the search can see at the typical rate inside 3-7 s
    typical_rate = np.median(rate[in_bti])
    print(f"    shortest detectable gap at {typical_rate:.0f} counts/s: "
          f"{1e3 * GAP_SIGNIFICANCE / typical_rate:.2f} ms")
    lost = local_rate * gap_lengths > GAP_SIGNIFICANCE
    gap_starts, gap_sizes = times[:-1][lost], gap_lengths[lost]
    lost_counts = np.sum(local_rate[lost] * gap_sizes)

    print(f"    live fraction: min {live_fraction.min():.3f} at t = {centres[live_fraction.argmin()]:.3f} s; "
          f"mean inside 3-7 s {live_fraction[in_bti].mean():.3f}, "
          f"after T0 outside it {live_fraction[outside].mean():.3f}")
    print(f"    counts lost to dead time inside 3-7 s: "
          f"{np.sum(counts[in_bti] * (1 / live_fraction[in_bti] - 1)):.0f} "
          f"({100 * (1 - live_fraction[in_bti].mean()):.1f}% of the time)")
    print(f"    data gaps: {len(gap_sizes)} found (5 longest listed), total {1e3 * gap_sizes.sum():.1f} ms, "
          f"~{lost_counts:.0f} counts missing")
    longest = np.argsort(gap_sizes)[::-1][:5]
    for i in sorted(longest, key=lambda i: gap_starts[i]):
        print(f"      gap at t = {gap_starts[i]:.4f} s lasting {1e3 * gap_sizes[i]:.2f} ms")

    line, = axes[0].plot(centres, rate / 1e3, lw=0.6, label=det)
    axes[1].plot(centres, live_fraction, lw=0.6, color=line.get_color())
    axes[2].vlines(gap_starts, 0, 1e3 * gap_sizes, color=line.get_color(), lw=1.5)

axes[0].set_ylabel("Recorded rate\n[10$^3$ counts/s]")
axes[0].legend(loc="upper right", ncol=4)
axes[1].set_ylabel("Live fraction")
axes[2].set_ylabel("Data gaps [ms]")
axes[2].set_xlabel("Time since GBM trigger [s]")
for ax in axes:
    ax.axvspan(*BAD_TIME_INTERVAL, color="grey", alpha=0.15)
axes[-1].set_xlim(*TIME_RANGE)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "06_pileup_check.png", dpi=200)
print(f"saved {FIGURE_DIR / '06_pileup_check.png'}")