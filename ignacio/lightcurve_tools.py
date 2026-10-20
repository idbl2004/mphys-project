"""
Shared functions for building Fermi/GBM light curves from TTE files.

Written for GRB 230307A but nothing here is specific to that burst: the
data folder, burst ID, detectors, bands and windows are all arguments.
"""
from pathlib import Path

import numpy as np
from gdt.missions.fermi.gbm.tte import GbmTte
from gdt.core.binning.unbinned import bin_by_time, bin_by_edges
from gdt.core.background.fitter import BackgroundFitter
from gdt.core.background.binned import Polynomial


def tte_path(data_dir, burst_id, det):
    return Path(data_dir) / f"glg_tte_{det}_{burst_id}_v00.fit"


def aligned_bin_edges(time_range, bin_width):
    """Bin edges covering time_range with one edge exactly on the trigger."""
    first_bin = np.floor(time_range[0] / bin_width)
    last_bin = np.ceil(time_range[1] / bin_width)
    return np.arange(first_bin, last_bin + 1) * bin_width


def band_counts(data_dir, burst_id, det, energy_band, bin_edges):
    """Raw counts per bin for one detector in one energy band."""
    tte = GbmTte.open(tte_path(data_dir, burst_id, det))
    photons_in_band = tte.slice_energy(energy_band)
    arrival_times = np.asarray(photons_in_band.data.times)
    tte.close()
    counts, _ = np.histogram(arrival_times, bins=bin_edges)
    return counts

def channels_used(data_dir, burst_id, det, energy_band):
    """Channels slice_energy keeps for a band, and the energies they really cover.

    slice_energy keeps every channel that overlaps the band, so the real
    band runs from the bottom of the lowest channel kept to the top of the
    highest one, which can be well outside the numbers asked for.
    """
    tte = GbmTte.open(tte_path(data_dir, burst_id, det))
    channels = np.asarray(tte.slice_energy(energy_band).data.channels)
    low_edges = np.asarray(tte.ebounds.low_edges())
    high_edges = np.asarray(tte.ebounds.high_edges())
    tte.close()
    lo_chan, hi_chan = int(channels.min()), int(channels.max())   # stored as floats
    return lo_chan, hi_chan, low_edges[lo_chan], high_edges[hi_chan], high_edges[-1]

def live_fraction(data_dir, burst_id, det, bin_edges):
    """Fraction of each bin the detector was able to record (1 = no dead time).

    Every recorded event, at any energy, makes the detector blind for
    2.6 us (10 us in the overflow channel); GDT takes this from the file
    and returns the live time of each bin as the PHAII exposure.
    """
    tte = GbmTte.open(tte_path(data_dir, burst_id, det))
    phaii = tte.to_phaii(bin_by_edges, bin_edges)
    tte.close()
    return np.asarray(phaii.data.exposure) / np.diff(bin_edges)


def fit_background(data_dir, burst_id, det, windows, order=3, fit_bin_width=1.024):
    """Fit the background of one detector; returns the fitter and its fit quality.

    Same idea as RMfit / Dichiara: a polynomial in time is fitted to each
    energy channel separately, using only the quiet windows either side of
    the burst. The fit is done on 1.024 s bins, because at 16 ms most
    channels have zero or one count per bin and a chi-squared fit would be
    biased. The model can then be evaluated on any bins, including 16 ms.
    """
    tte = GbmTte.open(tte_path(data_dir, burst_id, det))
    phaii = tte.to_phaii(bin_by_time, fit_bin_width, time_ref=0.0)
    tte.close()

    fitter = BackgroundFitter.from_phaii(phaii, Polynomial, time_ranges=windows)
    fitter.fit(order=order)
    # Fit quality summed over all channels; close to 1 means the polynomial
    # describes the quiet windows within the Poisson noise
    reduced_chi2 = np.sum(fitter.statistic) / np.sum(fitter.dof)
    return fitter, reduced_chi2


def background_in_band(fitter, energy_band, bin_edges):
    """Background counts per bin, and their 1-sigma error, in one energy band."""
    model = fitter.interpolate_bins(bin_edges[:-1], bin_edges[1:])
    in_band = model.integrate_energy(*energy_band)
    # Convert the fitted rate (counts/s) to counts per bin ourselves: GDT's own
    # .counts uses the exposure of the coarse 1.024 s fit bins, which is wrong
    # for 16 ms bins. This ignores dead time, which is <1% away from the burst.
    widths = np.diff(bin_edges)
    bkg_counts = np.squeeze(np.asarray(in_band.rates)) * widths
    bkg_error = np.squeeze(np.asarray(in_band.rate_uncertainty)) * widths
    return bkg_counts, bkg_error
