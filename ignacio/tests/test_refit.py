"""
Tests for zhu/refit.py: which fits that ran out of time are refitted, and the switch to one fixed response. Run from ignacio/ with:
    python -m pytest tests/test_refit.py -q -p no:cacheprovider
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zhu import refit as R

ROW = dict(model="Comptonized", C_cpl=100.0, C_band=100.0, C_lowbreak=100.0)


def test_late_chosen_model_counts():
    assert R.could_matter(dict(ROW, model="Band"), ["band"])
    assert R.could_matter(dict(ROW, model="BandLowBreak"), ["two_break"])
    assert R.could_matter(ROW, ["errors"])


def test_late_two_break_near_threshold():
    band = dict(ROW, model="Band")
    assert not R.could_matter(dict(band, C_lowbreak=97.0), ["two_break"])          # 3 short of 11.8: out of reach
    assert R.could_matter(dict(band, C_lowbreak=92.0), ["two_break"])              # 8: within reach
    assert not R.could_matter(dict(band, C_lowbreak=80.0), ["two_break"])          # past it already: turned down for its break


def test_late_band_near_threshold():
    assert not R.could_matter(dict(ROW, C_band=98.0), ["band"])
    assert R.could_matter(dict(ROW, C_band=94.0), ["band"])
    assert not R.could_matter(dict(ROW, C_band=85.0), ["band"])
    assert not R.could_matter(ROW, ["pl", "bb"])                                    # neither can be chosen


def test_the_fixed_response_is_put_back_afterwards():
    import grbtool.fitting as F
    own = F.window_response
    with R.response_fixed_at(0.0):
        assert F.window_response is not own
    assert F.window_response is own


@pytest.mark.skipif(not (R.SAMPLES["coppin"][0] / "160625B").is_dir(), reason="needs the measured samples in results/")
def test_each_late_window_is_listed_once_with_a_reason():
    t = R.late_windows()
    keys = t[["sample", "burst", "label", "t_start", "t_stop"]]
    assert not keys.duplicated().any() and set(t.why) <= set("ABCZ")
    assert (t.ours | t.zhu).all()
