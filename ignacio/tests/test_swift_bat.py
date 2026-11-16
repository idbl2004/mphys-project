"""
Tests for instruments/swift_bat.py. Run from ignacio/ with: python -m pytest tests/test_swift_bat.py -q

They need HEASoft, the BAT calibration files and GRB 211211A's BAT data (see the module's docstring), and are skipped without them.
The check is that the chain run here gives the archive pipeline's own spectra and responses for the same windows.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instruments import swift_bat as B

HAVE_BAT = (B.HEADAS / "headas-init.sh").exists() and (B.CALDB / "data/swift/bat").is_dir() and any((B.DATA / "01088940000").rglob("*bevshsp_uf.evt.gz"))
bat_data = pytest.mark.skipif(not HAVE_BAT, reason="needs HEASoft, the BAT CALDB and GRB 211211A's BAT data")


@bat_data
@pytest.mark.slow
@pytest.mark.parametrize("product", ["bevas", "bevps"])
def test_the_chain_gives_the_archive_spectrum_and_response(product):
    check = B.against_archive(B.burst("GRB211211A"), product)
    assert abs(check["total_rate_ratio"] - 1) < 0.02            # the position differs slightly from the pipeline's
    assert check["worst_channel_sigma"] < 1.5
    assert check["worst_error_ratio"] < 0.03
    lo, hi = check["response_ratio_range"]
    assert 0.999 < lo and hi < 1.001


@bat_data
@pytest.mark.slow
def test_the_light_curve_peaks_in_the_main_emission():
    t, rate, error = B.light_curve(B.burst("GRB211211A"), 0.064)
    assert np.all(np.isfinite(rate)) and np.all(error > 0)       # bins without data are left out
    assert len(t) > 5000
    assert 0 < t[np.argmax(rate)] < 15                          # the brightest part of GRB 211211A is in its first 15 s


def test_the_gbm_offset_is_the_difference_of_the_two_trigger_times():
    bat = B.BatBurst("x", "00000000000", 0.0, "2021-12-11T13:09:59.634", 0.0, 0.0, Path("."), Path("."))
    assert B.gbm_offset(bat, "2021-12-11T13:09:59.651") == pytest.approx(-0.017, abs=1e-6)


def _orbit_files(folder, swift_km, fermi_km, ra=0.0, dec=90.0):
    """A Swift orbit file and a GBM trigdat file with the spacecraft standing still at the given positions (J2000, km)."""
    from astropy.io import fits
    (folder / "auxil").mkdir(parents=True)
    t = np.arange(-5.0, 6.0)
    sao = fits.BinTableHDU.from_columns([fits.Column("TIME", "D", array=t), fits.Column("POSITION", "3E", array=np.tile(swift_km, (len(t), 1)))])
    sao.name = "PREFILTER"
    fits.HDUList([fits.PrimaryHDU(), sao]).writeto(folder / "auxil" / "sw00000000000sao.fits.gz")
    rate = fits.BinTableHDU.from_columns([fits.Column("TIME", "D", array=t), fits.Column("ENDTIME", "D", array=t + 1.0),
                                          fits.Column("EIC", "3E", array=np.tile(fermi_km, (len(t), 1)))])
    rate.name = "EVNTRATE"
    fits.HDUList([fits.PrimaryHDU(), rate]).writeto(folder / "trigdat.fit")
    return B.BatBurst("x", "00000000000", 0.0, "2021-12-11T13:09:59.634", ra, dec, folder, folder), folder / "trigdat.fit"


def test_light_travel_is_positive_when_swift_is_nearer_the_burst(tmp_path):
    # the burst straight above the north pole (dec 90): Swift 7000 km nearer to it than Fermi sees its light 7000/c = 23.35 ms earlier
    bat, trigdat = _orbit_files(tmp_path, [0.0, 0.0, 7000.0], [0.0, 0.0, 0.0])
    assert B.light_travel(bat, trigdat, 0.0) == pytest.approx(7000.0 / 299792.458, rel=1e-6)


def test_light_travel_ignores_separations_across_the_line_of_sight(tmp_path):
    bat, trigdat = _orbit_files(tmp_path, [7000.0, 0.0, -100.0], [-7000.0, 300.0, 0.0])
    assert B.light_travel(bat, trigdat, 0.0) == pytest.approx(-100.0 / 299792.458, rel=1e-5)


def _diagonal_spectrum():
    """15-150 keV in 40 logarithmic channels, each photon bin read only by its own channel, errors of 1."""
    edges = np.geomspace(15.0, 150.0, 41)
    return dict(rate=np.zeros(40), error=np.ones(40), e_min=edges[:-1], e_max=edges[1:], e_lo=edges[:-1], e_hi=edges[1:], matrix=np.eye(40))


def test_expected_index_of_a_peak_far_above_the_band_is_alpha():
    from instruments.bat_compare import expected_index
    assert expected_index(_diagonal_spectrum(), lambda e: B.cutoff_power_law(e, 1.0, -1.5, 1e5)) == pytest.approx(-1.5, abs=0.01)


def test_expected_index_of_a_peak_inside_the_band_is_softer_than_alpha():
    from instruments.bat_compare import expected_index
    spec = _diagonal_spectrum()
    cpl = lambda e_peak: (lambda e: B.cutoff_power_law(e, 1.0, -1.0, e_peak))
    assert expected_index(spec, cpl(60.0)) < expected_index(spec, cpl(300.0)) < -1.0


def test_gbm_model_shapes_fold_like_bat_own_models():
    # GBM's fits are evaluated with GDT's functions; the same Comptonized through either gives the same index, and a Band whose peak
    # and break lie far above the band is its low-energy power law
    from gdt.core.spectra.functions import Band, Comptonized
    from instruments.bat_compare import _shape, expected_index
    spec = _diagonal_spectrum()
    gdt = expected_index(spec, _shape(Comptonized(), [1.0, 80.0, -0.8, 100.0]))
    assert gdt == pytest.approx(expected_index(spec, lambda e: B.cutoff_power_law(e, 1.0, -0.8, 80.0)), abs=1e-3)
    assert expected_index(spec, _shape(Band(), [1.0, 1e5, -1.2, -2.5, 100.0])) == pytest.approx(-1.2, abs=0.01)


def test_light_travel_follows_the_burst_direction(tmp_path):
    # a burst at RA 90, Dec 0 lies along +y: only the y separation counts
    bat, trigdat = _orbit_files(tmp_path, [5000.0, 3000.0, -2000.0], [-4000.0, -1000.0, 6000.0], ra=90.0, dec=0.0)
    assert B.light_travel(bat, trigdat, 0.0) == pytest.approx(4000.0 / 299792.458, rel=1e-5)


def test_the_offset_adds_the_light_travel_time_at_the_time_asked(monkeypatch):
    from instruments import bat_compare as C
    calls = {}
    monkeypatch.setattr(C.B, "burst", lambda name: "bat")
    monkeypatch.setattr(C.B, "gbm_offset", lambda bat, utc: 2.5)

    def travel(bat, trigdat, met, gbm_offset=0.0, at=0.0):
        calls.update(gbm_offset=gbm_offset, at=at)
        return 0.020
    monkeypatch.setattr(C.B, "light_travel", travel)
    import pandas as pd
    match = pd.DataFrame({"bat": ["GRBX"]}, index=["X"])
    bat, offset, travelled, note = C._bat_and_offset("X", match, "utc", "trigdat.fit", 0.0, at=7.0)
    assert offset == pytest.approx(2.52) and travelled == pytest.approx(0.020) and note == ""
    assert calls == dict(gbm_offset=2.5, at=7.0)                    # both spacecraft taken at the same instant, 7 s after the GBM trigger


def _realistic_spectrum():
    """Photon bins 10-200 keV, finer than 40 channels over 15-150 keV, each photon bin read by the channel holding its centre: a matrix
    that isn't square, with photon bins outside the band."""
    e_in = np.geomspace(10.0, 200.0, 121)
    chan = np.geomspace(15.0, 150.0, 41)
    centre = np.sqrt(e_in[:-1] * e_in[1:])
    matrix = np.zeros((120, 40))
    for i, c in enumerate(centre):
        j = np.searchsorted(chan, c) - 1
        if 0 <= j < 40:
            matrix[i, j] = 1.0
    return dict(rate=np.zeros(40), error=np.ones(40), e_min=chan[:-1], e_max=chan[1:], e_lo=e_in[:-1], e_hi=e_in[1:], matrix=matrix)


def test_expected_index_with_a_rectangular_response():
    from instruments.bat_compare import expected_index
    spec = _realistic_spectrum()
    assert expected_index(spec, lambda e: B.cutoff_power_law(e, 1.0, -1.3, 1e5)) == pytest.approx(-1.3, abs=0.02)


def test_two_break_model_without_its_break_folds_like_band():
    from gdt.core.spectra.functions import Band
    from spectral_fit_tools import BandLowBreak
    from instruments.bat_compare import _shape, expected_index
    spec = _realistic_spectrum()
    band = expected_index(spec, _shape(Band(), [1.0, 90.0, -0.9, -2.6, 100.0]))
    two_break = expected_index(spec, _shape(BandLowBreak(), [1.0, 30.0, -0.9, -0.9, 90.0, -2.6, 2.0, 100.0]))   # alpha1 = alpha2
    assert two_break == pytest.approx(band, abs=1e-3)


@pytest.mark.skipif(not __import__("instruments.bat_compare", fromlist=["MATCH"]).MATCH.exists(), reason="needs the BAT match table")
def test_a_burst_without_results_keeps_its_rows():
    from instruments import bat_compare as C
    out = C.episodes(["not_a_burst"])
    assert list(out.episode) == ["pe", "qe1", "me"] and out.bat_flags.str.startswith("no GBM episodes").all()
    assert {"bat_start", "d_start"} <= set(out.columns)
