# make_lightcurve_4det.py: light curves of GRB 230307A in Dichiara's 4 detectors
import matplotlib
matplotlib.use("Agg")  # no display in WSL, save to file
import matplotlib.pyplot as plt
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.binning.unbinned import bin_by_time

# NaI detectors (n6, n8, na): 8-1000 keV. BGO (b1) sees higher energies: 200 keV - 40 MeV
detectors = {"n6": (8, 1000), "n8": (8, 1000), "na": (8, 1000), "b1": (200, 40000)}

# Load and bin each detector once (16 ms, as Dichiara), keep the light curves
lcs = {}
for det, erange in detectors.items():
    tte = GbmTte.open(f"data/fermiGBM/GRB230307A/current/glg_tte_{det}_bn230307656_v00.fit")
    lcs[det] = tte.to_phaii(bin_by_time, 0.016).to_lightcurve(energy_range=erange)

# Two figures: whole burst, and zoom on the precursor; 4 stacked panels each
for name, (t0, t1) in {"full": (-10, 60), "precursor": (-0.2, 0.5)}.items():
    fig, axes = plt.subplots(len(detectors), 1, sharex=True, figsize=(7, 9))
    for ax, det in zip(axes, detectors):
        lc, erange = lcs[det], detectors[det]
        ax.step(lc.centroids, lc.rates, where="mid", lw=0.7)
        ax.set_xlim(t0, t1)
        inside = (lc.centroids > t0) & (lc.centroids < t1)
        ax.set_ylim(0, 1.1 * lc.rates[inside].max())  # scale y to the visible window
        ax.set_ylabel("counts/s")
        kind = "BGO" if det.startswith("b") else "NaI"  # b = BGO detector, n = NaI
        ax.text(0.98, 0.85, f"{kind} {det}: {erange[0]}-{erange[1]} keV",
                transform=ax.transAxes, ha="right", fontweight="bold")
        if name == "precursor":
            ax.axvspan(-0.064, 0.320, alpha=0.2)  # Dichiara precursor window
    axes[-1].set_xlabel("Time since trigger (s)")
    axes[0].set_title(f"GRB 230307A, 16 ms bins, Dichiara detectors ({name})")
    fig.align_ylabels(axes)  # line up the y-labels so none get squashed
    fig.tight_layout()
    fig.savefig(f"ignacio/lc_4det_{name}.png", dpi=150)
    plt.close(fig)
    print("saved", f"ignacio/lc_4det_{name}.png")