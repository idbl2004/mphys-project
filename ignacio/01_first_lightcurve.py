# make_lightcurve.py: light curve of GRB 230307A from Fermi GBM TTE data (NaI n6)
import numpy as np
# tools: numpy (maths), matplotlib (plots), GDT (reads Fermi data)
import matplotlib
matplotlib.use("Agg")  # no display in WSL/Chandra, so just save plots o files
import matplotlib.pyplot as plt
from gdt.missions.fermi.gbm.tte import GbmTte #GDT = NASA´s Gamma-ray Data Tools; GbmTe reads GBM photon-by-photon (TTE) files. 
from gdt.core.binning.unbinned import bin_by_time #tunrs the raw photon list into equal time bins

# Load the photon list for detector n6
# raw data: list of every photon (arrival time + energy channel)
tte = GbmTte.open("data/fermiGBM/GRB230307A/current/glg_tte_n6_bn230307656_v00.fit") #Load the photon list (TTE) for detector n6, oneof the NaI detectors facing the burst
# bin photons into 16ms bins (Dichiara 2023): counts per time bin per enegry channel
# Count photons in 16 ms boxes (as Dichiara et al. 2023), then sum 8-1000 keV
phaii = tte.to_phaii(bin_by_time, 0.016)
lc = phaii.to_lightcurve(energy_range=(8, 1000)) # Make the light curve: for each 16ms time bin, add up the counts in all 
#channels between 8 and 1000kev ( the useful NaI range) and divide by the Bin width -> count rate (counts/s) vs time. 
#summing over energy gives the ligtht curve: summing over time intead eg the precursor window would give a spectrum 

# Make two plots: the whole burst, and a zoom on the precursor
# loop over two time windows: whole burst, and zoom on the precursor (Dichiara window +/- a bit)
for name, (t0, t1) in {"full": (-10, 60), "precursor": (-0.2, 0.5)}.items():
    fig, ax = plt.subplots()
    ax.step(lc.centroids, lc.rates, where="mid")
    ax.set_xlim(t0, t1)
    inside = (lc.centroids > t0) & (lc.centroids < t1)
    ax.set_ylim(0, 1.1 * lc.rates[inside].max())  # scale y-axis to what's visible
    ax.set_xlabel("Time since trigger (s)")
    ax.set_ylabel("Count rate (counts/s)")
    ax.set_title(f"GRB 230307A, NaI n6, 8-1000 keV, 16 ms bins ({name})")
    if name == "precursor":
        ax.axvspan(-0.064, 0.320, alpha=0.2, label="Dichiara precursor window")
        ax.legend()
    fig.savefig(f"ignacio/lc_n6_{name}.png", dpi=150)
    plt.close(fig)
    print("saved", f"ignacio/lc_n6_{name}.png")
    # Rest of the loop: draw count rate vs time as a step plot (one step per
    # 16 ms bin), zoom the x-axis to this window and rescale y to what's visible,
    # add labels and a title. On the precursor plot, shade Dichiara's precursor
    # window (-0.064 to 0.320 s) to compare. Saved each plot as a PNG in ignacio/.