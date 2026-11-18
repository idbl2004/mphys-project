"""
Tests for instruments/bat_sample.py (Burlon et al.'s BAT precursors). Run from ignacio/ with: python -m pytest tests/test_bat_sample.py -q
The classification is tested on made-up fit results; the sample table on the file copied from the paper.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instruments import bat_sample as S


def fits(index, i_err, gain, alpha=-1.0, ep=80.0, ep_err=(10.0, 12.0)):
    pl = dict(params=np.array([0.05, index]), errors=[(i_err, i_err), (i_err, i_err)], chi2=100.0, dof=50)
    cpl = dict(params=np.array([0.05, alpha, ep]), errors=[(0.01, 0.01), (0.1, 0.1), ep_err], chi2=100.0 - gain, dof=49)
    return pl, cpl


def test_peak_only_when_cutoff_wins():
    assert S.classify(*fits(-1.3, 0.05, 25.0))["measured"]
    assert not S.classify(*fits(-1.3, 0.05, 5.0))["measured"]                       # not enough gain
    assert not S.classify(*fits(-1.3, 0.05, 25.0, alpha=-2.3))["measured"]           # index too soft for a peak
    assert not S.classify(*fits(-1.3, 0.05, 25.0, ep=200.0))["measured"]             # above the band
    assert not S.classify(*fits(-1.3, 0.05, 25.0, ep_err=(10.0, np.nan)))["measured"]  # one-sided


def test_index_places_peak():
    assert "above" in S.classify(*fits(-1.0, 0.1, 2.0))["note"]
    assert "below" in S.classify(*fits(-2.6, 0.1, 2.0))["note"]
    assert "allows a peak inside" in S.classify(*fits(-1.7, 0.1, 2.0))["note"]


def test_the_sample_is_their_table():
    t = S.read_sample()
    assert t.grb.nunique() == 15 and len(t) == 31 and (t.episode == "main").sum() == 15
    assert set(t.loc[t.grb == "GRB070306", "episode"]) == {"main", "pre1", "pre2"}
    pre = t[t.episode.str.startswith("pre")]
    assert (pre.t2_s > pre.t1_s).all() and (t.alpha < 0).all()
    assert t.loc[(t.grb == "GRB060729") & (t.episode == "pre1"), "alpha_minus"].isna().all()   # printed "+-20.75": left empty


def test_a_curved_fit_is_compared_with_our_curved_fit():
    """Where they quote E_peak, their alpha is the cut-off power law's: ours of the same model is compared, and the E_peaks."""
    pl, cpl = fits(-2.04, 0.04, 12.0, alpha=-1.25, ep=42.4, ep_err=(3.7, 3.5))
    row = dict(S.classify(pl, cpl), their_alpha=-1.30, their_alpha_minus=0.47, their_alpha_plus=0.47, their_epeak_keV=41.0,
               their_epeak_minus=8.5, their_epeak_plus=8.5)
    assert row["cpl_alpha"] == -1.25 and row["cpl_alpha_minus"] == 0.1
    assert abs(S.against_theirs(row, "cpl_alpha")) < 0.3                # -1.25 against -1.30 +- 0.29 (68%)
    assert abs(S.against_theirs(row)) > 2.4                             # the power law's -2.04 is not what they quote
    assert S.epeak_against_theirs(row) == pytest.approx((42.4 - 41.0) / np.hypot(8.5 / S.NINETY, 3.7))


def test_curved_rows_have_main_epeak():
    t = S.read_sample()
    curved = t[t.main_epeak_keV.notna()]
    assert sorted(curved.grb) == ["GRB050820A", "GRB060124", "GRB060714", "GRB061121"] and (curved.episode == "main").all()
    assert set(curved.loc[curved.source == "bat", "grb"]) == {"GRB060714"}


def test_fluence_and_rest_frame_energy():
    """For a photon index of -2 the energy per log energy is flat: the fluence is norm 50^2 ln(hi/lo) keV per cm^2 per s."""
    expected = 1e-3 * 2500 * np.log(10) * S.KEV * 10
    assert S.band_fluence(1e-3, -2.0, 15, 150, 10) == pytest.approx(expected, rel=1e-6)
    # the same flat spectrum carries the same energy in any decade, so the rest-frame band only moves the distance and the 1 + z
    from astropy import units as u
    from astropy.cosmology import Planck18
    d = Planck18.luminosity_distance(1.0).to(u.cm).value
    assert S.eiso_rest(1e-3, -2.0, 1.0, 10) == pytest.approx(4 * np.pi * d ** 2 * expected / 2, rel=1e-6)


def test_printed_redshifts():
    assert S.redshift("1.949") == 1.949 and S.redshift("2.09?") == 2.09 and S.redshift("<3.5") is None


def test_li_and_mao_spans():
    """Their pulse fits as spans: GRB 050318's precursor pulse starts at -1.80 s and peaks 2.45 s later, decaying with 1.32 s."""
    spans, info = S.limao_spans()
    assert len(info) == 52 and info.loc["GRB050726", "z"] == "<3.5"
    assert spans.loc["GRB050318", ("start", "pre")] == -1.80
    assert spans.loc["GRB050318", ("stop", "pre")] == pytest.approx(-1.80 + 2.45 + 3 * 1.32)
    assert (spans[("stop", "pre")] > spans[("start", "pre")]).all() and (spans[("start", "main")] > spans[("start", "pre")]).all()
