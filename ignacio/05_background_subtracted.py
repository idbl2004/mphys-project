"""
Background-subtracted light curves of GRB 230307A (bn230307656).

First step past the Fig. 1 reproduction (03_reproduce_dichiara_fig1.py).
Fig. 1 shows raw counts, so the background (~42 cts/bin soft, ~11 hard)
is still in it. Here the background is modelled and removed, using the
same recipe Dichiara use for their spectra: a 3rd-order polynomial in time
fitted to -89..-7 s and +162..+273 s, channel by channel.

Outputs
-------
05_background_fit_check.png
    Raw counts in 1.024 s bins over the whole fit range with the model on
    top and the fit windows shaded: shows whether the polynomial is a fair
    description of the quiet periods, and how it interpolates under the burst.
05_lightcurves_bkg_subtracted.png
    Net counts per 16 ms bin, same bands and detectors as Fig. 1, corrected
    for dead time (thin dark line: without the correction).

Dead-time correction: each detector's raw counts are divided by its live
fraction in that bin (06_pileup_deadtime_gaps.py: down to ~0.77 in b1, ~0.92 in na
during the precursor). The fitted background is already a rate per unit
live time, so after the correction both are on the same footing.
Error on each bin = sqrt(N / f^2 + sigma_bkg^2), with N raw counts and f
the live fraction: Poisson noise scaled by the correction, plus the
uncertainty of the background model.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from settings import (DATA_DIR, BURST_ID, BIN_WIDTH, SOFT_BAND, HARD_BAND,
                      SOFT_DETECTORS, HARD_DETECTORS, BKG_WINDOWS, BKG_ORDER,
                      BAD_TIME_INTERVAL, PRECURSOR, PLOT_RANGE, FIGURE_DIR)
from lightcurve_tools import (aligned_bin_edges, band_counts, fit_background,
                              background_in_band, live_fraction)

BANDS = {
    f"{SOFT_BAND[0]:.0f}-{SOFT_BAND[1]:.0f} keV":
        {"range": SOFT_BAND, "detectors": SOFT_DETECTORS, "colour": "red"},
    f"{HARD_BAND[0]:.0f}-{HARD_BAND[1]:.0f} keV":
        {"range": HARD_BAND, "detectors": HARD_DETECTORS, "colour": "grey"},
}
FINE_BIN = BIN_WIDTH
COARSE_BIN = 1.024      # s, for the fit check plot
FULL_RANGE = (-100.0, 280.0)   # s, covers both background windows

fine_edges = aligned_bin_edges(PLOT_RANGE, FINE_BIN)
coarse_edges = aligned_bin_edges(FULL_RANGE, COARSE_BIN)
fine_t = 0.5 * (fine_edges[:-1] + fine_edges[1:])
coarse_t = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])

# One background fit per detector, reused for every band and binning
fitters = {}
for det in sorted({d for band in BANDS.values() for d in band["detectors"]}):
    fitters[det], chi2 = fit_background(DATA_DIR, BURST_ID, det, BKG_WINDOWS, BKG_ORDER)
    print(f"{det}: background fit reduced chi2 = {chi2:.2f}")

results = {}
for name, band in BANDS.items():
    raw = np.zeros(len(fine_t))
    corrected = np.zeros(len(fine_t))
    corrected_var = np.zeros(len(fine_t))
    bkg = np.zeros(len(fine_t))
    bkg_var = np.zeros(len(fine_t))
    raw_coarse = np.zeros(len(coarse_t))
    bkg_coarse = np.zeros(len(coarse_t))
    for det in band["detectors"]:
        det_raw = band_counts(DATA_DIR, BURST_ID, det, band["range"], fine_edges)
        live = live_fraction(DATA_DIR, BURST_ID, det, fine_edges)
        raw += det_raw
        corrected += det_raw / live
        corrected_var += det_raw / live**2
        b, b_err = background_in_band(fitters[det], band["range"], fine_edges)
        bkg += b
        bkg_var += b_err**2   # detectors are fitted independently
        raw_coarse += band_counts(DATA_DIR, BURST_ID, det, band["range"], coarse_edges)
        bkg_coarse += background_in_band(fitters[det], band["range"], coarse_edges)[0]

    net = corrected - bkg
    net_err = np.sqrt(corrected_var + bkg_var)
    net_uncorrected = raw - bkg
    results[name] = dict(raw=raw, bkg=bkg, net=net, net_err=net_err,
                         net_uncorrected=net_uncorrected,
                         raw_coarse=raw_coarse, bkg_coarse=bkg_coarse)

    at_trigger = np.argmin(np.abs(fine_t))
    pre = fine_t < -1.0
    print(f"{name}: background at T0 = {bkg[at_trigger]:.1f} "
          f"+/- {np.sqrt(bkg_var[at_trigger]):.1f} cts/bin; "
          f"net counts before -1 s average {net[pre].mean():.1f} (should be ~0)")
    precursor = (fine_t > PRECURSOR[0]) & (fine_t < PRECURSOR[1])
    gain = net[precursor].sum() / net_uncorrected[precursor].sum() - 1
    print(f"{name}: precursor net counts {net[precursor].sum():.0f} "
          f"+/- {np.sqrt(np.sum(net_err[precursor]**2)):.0f} "
          f"(dead-time correction adds {100 * gain:.1f}%)")

# --- Check of the background fit -------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for ax, (name, band) in zip(axes, BANDS.items()):
    r = results[name]
    ax.step(coarse_t, r["raw_coarse"], where="mid", color=band["colour"], lw=0.8,
            label=f"{name} raw")
    ax.plot(coarse_t, r["bkg_coarse"], color="black", lw=1.2,
            label=f"poly{BKG_ORDER} background")
    for lo, hi in BKG_WINDOWS:
        ax.axvspan(lo, hi, color="tab:blue", alpha=0.1)
    # Zoom the y-axis on the background level; the burst goes off the top
    quiet = r["bkg_coarse"]
    ax.set_ylim(0.8 * quiet.min(), 1.6 * quiet.max())
    ax.set_ylabel(f"Counts per {COARSE_BIN} s")
    ax.legend(loc="upper right")
axes[-1].set_xlabel("Time since GBM trigger [s]")
axes[0].set_title("Shaded: background fit windows")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "05_background_fit_check.png", dpi=200)

# --- Background-subtracted light curves ------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for ax, (name, band) in zip(axes, BANDS.items()):
    r = results[name]
    ax.errorbar(fine_t, r["net"], yerr=r["net_err"], fmt="none",
                ecolor=band["colour"], elinewidth=0.4, alpha=0.4)
    ax.plot(fine_t, r["net"], color=band["colour"], lw=0.7, label=f"{name}, dead-time corrected")
    ax.plot(fine_t, r["net_uncorrected"], color="black", lw=0.3, alpha=0.6,
            label="without correction")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvspan(*BAD_TIME_INTERVAL, color="grey", alpha=0.15, label="BTI (pile-up)")
    ax.set_ylabel("Net counts [cts/bin]")
    ax.legend(loc="upper right")
axes[-1].set_xlim(*PLOT_RANGE)
axes[-1].set_xlabel("Time since GBM trigger [s]")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "05_lightcurves_bkg_subtracted.png", dpi=200)
print(f"saved 05_background_fit_check.png and 05_lightcurves_bkg_subtracted.png in {FIGURE_DIR}")