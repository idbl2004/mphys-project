"""
Tests for the coppin package (Coppin et al.'s bursts measured with the zhu code). Run from ignacio/ with:
    python -m pytest tests/test_coppin.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coppin import sample as S


def test_times_from_trigger():
    """A made-up burst triggering at 12:00:00.000 UTC whose prompt emission starts 10 s later, and one across midnight."""
    table = pd.DataFrame(dict(grb=["GRBX", "GRBY"], trigger=["bnX", "bnY"], t_prompt_utc=["12:00:10.000", "00:00:05.000"],
                              t_precursor_s=[-12.5, -8.0], duration_s=[2.0, 1.0]))
    catalogue = pd.DataFrame(dict(trigger_time=[60000.5, 60000.0 + 86397.0 / 86400.0]), index=pd.Index(["bnX", "bnY"], name="trigger_name"))
    t = S.from_trigger(table, catalogue)
    assert t.prompt_start_s.tolist() == pytest.approx([10.0, 8.0], abs=1e-4)          # the second triggers at 23:59:57
    assert t.pre_start_s.tolist() == pytest.approx([-2.5, 0.0], abs=1e-4) and t.pre_stop_s.tolist() == pytest.approx([-0.5, 1.0], abs=1e-4)


def test_precursor_starts_agree():
    """Converted to the trigger, their first precursor starts within 0.2 s of our zhu2026 precursor for most of the bursts in both."""
    t = S.from_trigger(S.read_table(), S.read_catalogue()).sort_values("pre_start_s").groupby("trigger").first()
    ours = pd.read_csv(S.SHARED / "zhu_remeasured.csv", comment="#").set_index("trigger").PE_start_s.dropna()
    both = t.join(ours, how="inner")
    assert len(both) >= 15
    assert np.mean(np.abs(both.pre_start_s - both.PE_start_s) < 0.2) >= 0.7


def test_bin_rule():
    rule = S.tres_rule()
    assert rule[0] < 0                                                          # brighter bursts get finer bins
    middle = 1e-3 * 2.0 ** np.round(rule[1])                                    # at 1 ph/cm2/s the line is at its intercept
    assert S.tres_for(np.array([1e4, 1.0, 1e-3]), rule).tolist() == pytest.approx([0.004, middle, 0.128])


def test_sample_selection():
    s = S.build(write=False)
    assert len(s) > 50 and s.trigger.is_unique and (s.reason == "coppin2020").all()
    assert (s.t90 > S.LONG_T90_S).all() and not s.trigger.isin(S.zhu_triggers()).any()
    lead = s.era.map(S.TTE_LEAD_S)
    assert (s.pre_start_s >= -lead).all() and set(np.round(1e3 * s.Tres)) <= {4, 8, 16, 32, 64, 128}
