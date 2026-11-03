"""
Tests for the shared functions. Run from ignacio/ with: python -m pytest tests -q

They need the GRB 230307A data and the saved precursor spectra, and take under a minute.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdt.core.spectra.functions import Band, Comptonized
from gdt.missions.fermi.gbm.response import GbmRsp

from settings import BKG_ORDER, BKG_WINDOWS, BURST_ID, CODE_DIR, DATA_DIR, FIT_RANGE_BGO, FIT_RANGE_NAI, PRECURSOR, SPECTRAL_DETECTORS
from lightcurve_tools import aligned_bin_edges
from pileup_tools import PULSE, record
from spectral_fit_tools import (BandLowBreak, channel_mask, cstat, energy_flux, fit_model, load_spectra, scaled_starts, window_response, write_rsp)


@pytest.fixture(scope="module")
def precursor_data():
    phas, baks, rsps = load_spectra(CODE_DIR / "spectra", BURST_ID, SPECTRAL_DETECTORS)
    masks = [channel_mask(r, FIT_RANGE_BGO if d.startswith("b") else FIT_RANGE_NAI) for r, d in zip(rsps, SPECTRAL_DETECTORS)]
    return phas, baks, rsps, masks


def test_bin_edges_have_one_on_the_trigger():
    edges = aligned_bin_edges((-1.0, 1.0), 0.016)
    assert np.isclose(np.abs(edges).min(), 0.0)
    assert np.allclose(np.diff(edges), 0.016)


def test_low_break_model_reduces_to_band():
    energies = np.logspace(0.5, 4.5, 200)
    band, two = Band(), BandLowBreak()
    a, ep, alpha, beta = 0.3, 500.0, -0.8, -2.7
    expected = band.eval([a, ep, alpha, beta, 100.0], energies)
    assert np.allclose(two.eval([a, 50.0, alpha, alpha, ep, beta, 2.0, 100.0], energies), expected, rtol=1e-10)   # same slope either side of the break
    assert np.allclose(two.eval([a, 1e-6, -0.2, alpha, ep, beta, 2.0, 100.0], energies), expected, rtol=1e-8)      # break far below the data


def test_nufnu_peak_is_at_epeak():
    energies = np.logspace(1, 4, 20000)
    for function, params in ((Band(), [1.0, 300.0, -0.7, -2.6, 100.0]), (Comptonized(), [1.0, 300.0, -0.7, 100.0])):
        nufnu = energies**2 * function.eval(params, energies)
        assert energies[np.argmax(nufnu)] == pytest.approx(300.0, rel=1e-3)


def test_response_written_reads_back_exactly(tmp_path):
    rsp = window_response(DATA_DIR, BURST_ID, "n6", PRECURSOR)[0]
    write_rsp(rsp, tmp_path, "check.rsp")
    again = GbmRsp.open(tmp_path / "check.rsp")
    assert np.allclose(np.asarray(again.drm.matrix), np.asarray(rsp.drm.matrix), rtol=1e-6, atol=1e-12)


def test_scaled_starts_reproduce_the_net_counts(precursor_data):
    phas, baks, rsps, masks = precursor_data
    band = Band()
    start = scaled_starts(band, phas, baks, rsps, masks, [[1.0, 170.0, -0.6, -3.0, 100.0]])[0]
    model = sum(np.sum(np.asarray(r.drm.fold_spectrum(band.eval, start))[m]) * p.exposure for p, r, m in zip(phas, rsps, masks))
    net = sum(np.sum((np.asarray(p.data.counts) - np.asarray(b.data.counts))[m]) for p, b, m in zip(phas, baks, masks))
    assert model == pytest.approx(net, rel=1e-6)


def test_cstat_by_hand_matches_gdt(precursor_data):
    phas, baks, rsps, masks = precursor_data
    band = Band()
    fit = fit_model(phas, baks, rsps, masks, band, scaled_starts(band, phas, baks, rsps, masks, [[1.0, 170.0, -0.6, -3.0, 100.0]]), polish=True)
    total = 0.0
    for pha, bak, rsp, mask in zip(phas, baks, rsps, masks):
        n = np.asarray(pha.data.counts, float)[mask]
        low, high = np.asarray(rsp.drm.photon_bins.low_edges()), np.asarray(rsp.drm.photon_bins.high_edges())
        photons = band.eval(list(band.default_values), np.sqrt(low * high)) * (high - low)
        mu = (photons @ np.asarray(rsp.drm.matrix))[mask] * pha.exposure + np.asarray(bak.data.counts)[mask]
        term = mu - n
        term[n > 0] += n[n > 0] * np.log(n[n > 0] / mu[n > 0])
        total += 2 * term.sum()
    assert total == pytest.approx(cstat(fit), abs=1e-3)
    assert fit.parameters[1] == pytest.approx(170.3, abs=1.0)                 # Dichiara's value, and notebook 09's


def test_energy_flux_against_the_integral():
    # With E_0 huge the Comptonized model is a power law, so the energy flux has a closed form, in erg/cm2/s
    index = -1.5
    function = Comptonized()
    params = [1.0, 1e9, index, 100.0]
    e1, e2 = 10.0, 1000.0
    exact = 100.0**(-index) * (e2**(index + 2) - e1**(index + 2)) / (index + 2) * 1.602176634e-9
    assert energy_flux(function, params, (e1, e2)) == pytest.approx(exact, rel=1e-3)


def test_pulse_shape_matches_the_published_numbers():
    t = np.arange(0.0, 6.0, 0.052)
    assert t[np.argmax(PULSE)] == pytest.approx(0.364, abs=0.03)
    assert PULSE.max() == pytest.approx(1.0, abs=0.01)
    assert PULSE.min() == pytest.approx(-0.7, abs=0.05)


def test_recorded_rate_follows_dead_time_at_low_rates():
    rng = np.random.default_rng(1)
    grid = np.logspace(np.log10(150.0), np.log10(5000.0), 500)
    draw = lambda n: rng.choice(grid, n)
    for rate in (0.01, 0.05):
        heights, total = record(draw, rate, 100.0, 1e9, 100000, rng)
        assert len(heights) / total == pytest.approx(rate / (1 + 2.6 * rate), rel=0.03)


def test_ccf_lag_gets_a_known_delay_back():
    from sample_tools import ccf_lag
    bin_width = 0.004
    t = np.arange(-1.0, 1.0, bin_width)
    pulse = lambda delay: 200.0 * np.exp(-0.5 * ((t - delay) / 0.03)**2)
    margin = 25
    high = pulse(0.0)[margin:-margin]
    for delay in (-0.01, 0.0, 0.006, 0.02):
        assert ccf_lag(pulse(delay), high, bin_width) == pytest.approx(delay, abs=2e-4)
