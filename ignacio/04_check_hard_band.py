"""
Which detectors and channels did Dichiara use for the hard band of Fig. 1?

The edges 883 and 38367 keV turn out to be b1 channel boundaries, which
points to the band being built from b1. But b1 alone falls ~20% short of
the published peak (~600 cts/bin at 1.84 s). This script tries the
plausible combinations and prints the peak and pre-burst level of each.
"""
import numpy as np
from gdt.missions.fermi.gbm.tte import GbmTte

from settings import DATA_DIR, BURST_ID, BIN_WIDTH, HARD_BAND, PLOT_RANGE
from lightcurve_tools import aligned_bin_edges, tte_path

PEAK_WINDOW = (1.6, 2.0)   # s, the tall grey spike in Fig. 1
TOLERANCE = 1.0            # keV, to absorb rounding of the published edges

bin_edges = aligned_bin_edges(PLOT_RANGE, BIN_WIDTH)
centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])


def hard_counts(det, drop_top_channel=False):
    """Counts per bin using only channels lying fully inside HARD_BAND."""
    tte = GbmTte.open(tte_path(DATA_DIR, BURST_ID, det))
    low = np.asarray(tte.ebounds.low_edges())
    high = np.asarray(tte.ebounds.high_edges())
    inside = np.where((low >= HARD_BAND[0] - TOLERANCE) &
                      (high <= HARD_BAND[1] + TOLERANCE))[0]
    if drop_top_channel:
        inside = inside[inside < len(low) - 1]
    times = np.asarray(tte.data.times)
    channels = np.asarray(tte.data.channels).astype(int)
    tte.close()
    if len(inside) == 0:
        print(f"  {det}: no channels fully inside the band")
        return np.zeros(len(centres))
    print(f"  {det}: channels {inside[0]}-{inside[-1]} = "
          f"{low[inside[0]]:.1f}-{high[inside[-1]]:.1f} keV")
    keep = (channels >= inside[0]) & (channels <= inside[-1])
    counts, _ = np.histogram(times[keep], bins=bin_edges)
    return counts


print("Channels fully inside 883-38367 keV:")
curves = {det: hard_counts(det) for det in ["b0", "b1", "n6", "n8", "na"]}
print("NaI without their top channel:")
nai_no_top = sum(hard_counts(det, drop_top_channel=True) for det in ["n6", "n8", "na"])
nai = curves["n6"] + curves["n8"] + curves["na"]

options = {
    "b1": curves["b1"],
    "b0 + b1": curves["b0"] + curves["b1"],
    "b1 + NaI": curves["b1"] + nai,
    "b1 + NaI, no NaI top channel": curves["b1"] + nai_no_top,
}
in_window = (centres > PEAK_WINDOW[0]) & (centres < PEAK_WINDOW[1])
pre_burst = centres < -1.0
print("\nPaper (by eye): peak ~600 cts/bin near 1.84 s, pre-burst ~10-15")
for name, counts in options.items():
    i = counts[in_window].argmax()
    print(f"{name:30s} peak {counts[in_window][i]:4.0f} at "
          f"{centres[in_window][i]:.3f} s, pre-burst {counts[pre_burst].mean():4.1f}")