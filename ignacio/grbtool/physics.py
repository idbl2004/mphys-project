"""What the fits imply: k-correction, E_iso and L_iso, the Amati plane, the shattering-flare budget, and spectral lags. From notebooks 18-20."""
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck18, z_at_value
from scipy.integrate import quad

from lightcurve_tools import aligned_bin_edges, background_in_band, fit_background
from sample_tools import ccf_lag, lag_with_errors
from .bursts import event_times
from .checks import start_vector

FIDUCIAL = 1e50             # erg/s, the tidal energy rate of the resonant-shattering estimate
EFFICIENCY = 0.2            # fraction of it that comes out as the flare
LOW, HIGH = (25.0, 50.0), (100.0, 300.0)    # keV, the lag bands of Dichiara et al. (2023)
MARGIN_BINS = 25
# Minaev & Pozanenko (2020): log10(E_p,i / 100 keV) = a log10(E_iso / 1e51 erg) + b
TYPE_I = (0.41, 0.83)
TYPE_II = (0.43, -0.24)


def shape_of(row):
    """N(E) up to a constant, from the chosen model in a row of the table of fits."""
    function, params, _ = start_vector(row)
    return lambda e: function.eval(params, np.atleast_1d(np.asarray(e, float)))[0]


def k_correction(shape, z):
    """Energy in the rest-frame 1-10^4 keV over the energy in the observed 10-1000 keV."""
    upper = quad(lambda x: np.exp(2 * x) * shape(np.exp(x)), np.log(1.0 / (1 + z)), np.log(1e4 / (1 + z)), limit=300)[0]
    lower = quad(lambda x: np.exp(2 * x) * shape(np.exp(x)), np.log(10.0), np.log(1000.0), limit=300)[0]
    return upper / lower


def energetics(row, z):
    """L_iso, E_iso and rest-frame E_peak of one window."""
    d = Planck18.luminosity_distance(z).to(u.cm).value
    k = k_correction(shape_of(row), z)
    no_k = 4 * np.pi * d**2 * row.flux
    return dict(Epeak_rest=row.Epeak * (1 + z), k=k, L_iso=no_k * k, L_iso_no_k=no_k, E_iso=no_k * k * row.live_duration / (1 + z))


def amati_offsets(e_iso, ep_rest):
    """Distance from the two Amati tracks in dex of E_p,i. Positive means harder than the track."""
    y = np.log10(ep_rest / 100.0)
    x = np.log10(e_iso / 1e51)
    return dict(dex_from_type_I=y - (TYPE_I[0] * x + TYPE_I[1]), dex_from_type_II=y - (TYPE_II[0] * x + TYPE_II[1]))


def budget(l_iso):
    """What the precursor would need if it came from resonant shattering: tidal energy rate against the fiducial, and the crust field if it's the unipolar-inductor limit."""
    return dict(rate_over_fiducial=l_iso / EFFICIENCY / FIDUCIAL, B_min_G=1.2e15 * np.sqrt(l_iso / 3.6e50))


def z_for_budget(flux, multiple=1.0):
    """Redshift at which this flux needs `multiple` times the fiducial budget, with no k-correction. For bursts with no measured redshift."""
    distance = np.sqrt(multiple * FIDUCIAL * EFFICIENCY / (4 * np.pi * flux)) * u.cm
    return float(z_at_value(Planck18.luminosity_distance, distance.to("Mpc"), zmin=1e-4, zmax=10.0))


def lag_bin(duration, number):
    """About `number` bins across the window, on powers of 2 ms, and never below 4 ms."""
    return 1e-3 * max(4.0, 2.0**np.round(np.log2(1e3 * duration / number)))


class Lags:
    """Photons and background fits of one burst, read once, for the lag of 25-50 keV behind 100-300 keV."""

    def __init__(self, burst):
        self.burst = burst
        self.events = {band: event_times(burst, burst.nai, band) for band in (LOW, HIGH)}
        self.fits = {d: fit_background(burst.data_dir, burst.trigger, d, burst.bkg_windows, burst.bkg_order)[0] for d in burst.nai}

    def curve(self, band, edges):
        counts, background = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
        for d in self.burst.nai:
            counts += np.histogram(self.events[band][d], bins=edges)[0]
            background += background_in_band(self.fits[d], band, edges)[0]
        return counts, background

    def lag(self, window, bin_width, realisations=1000, half_width=4):
        """The lag both ways: with the low band running past the window, and with both cut at its edges. Usable if both bands are above 5 sigma."""
        edges = aligned_bin_edges(window, bin_width)
        wide = np.r_[edges[0] - bin_width * np.arange(MARGIN_BINS, 0, -1), edges, edges[-1] + bin_width * np.arange(1, MARGIN_BINS + 1)]
        low, low_b = self.curve(LOW, wide)
        high, high_b = self.curve(HIGH, edges)
        inside = slice(MARGIN_BINS, MARGIN_BINS + len(edges) - 1)
        low_sigma = (low - low_b)[inside].sum() / np.sqrt(max(low[inside].sum(), 1.0))
        high_sigma = (high - high_b).sum() / np.sqrt(max(high.sum(), 1.0))
        middle, down, up = lag_with_errors(low, low_b, high, high_b, bin_width, realisations, half_width=half_width)
        cut, cut_down, cut_up = lag_with_errors(low[inside], low_b[inside], high, high_b, bin_width, realisations, half_width=half_width, cut=True)
        return dict(bin_ms=1e3 * bin_width, low_sigma=low_sigma, high_sigma=high_sigma, usable=bool(low_sigma >= 5.0 and high_sigma >= 5.0), lag_ms=1e3 * middle, err_down=1e3 * down, err_up=1e3 * up,
                    lag_data_ms=1e3 * ccf_lag(low - low_b, high - high_b, bin_width, half_width), lag_cut_ms=1e3 * cut, cut_down=1e3 * cut_down, cut_up=1e3 * cut_up)


def lag_table(burst, windows):
    """Lags in the precursor and in the main emission. A window cut in pieces is skipped, since a lag needs continuous data."""
    chosen = [w for w in windows if w.kind in ("precursor", "main") and len(w.pieces) == 1]
    if not chosen:
        return pd.DataFrame()
    source = Lags(burst)
    rows = []
    for w in chosen:
        rows.append(dict(label=w.label, t_start=w.span[0], t_stop=w.span[1], **source.lag(w.span, lag_bin(w.duration, 64 if w.kind == "precursor" else 512))))
    return pd.DataFrame(rows).set_index("label")
