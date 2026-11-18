"""
Tests for grbtool's one way in to the samples and to any one burst (grbtool/samples.py, instruments/bat_sample.one_burst). Run from
ignacio/ with:
    python -m pytest tests/test_samples.py -q -p no:cacheprovider
The last ones read the measured samples in results/ and are skipped where those aren't present.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from grbtool import samples as S
from grbtool.bursts import BurstError

RESULTS = CODE_DIR / "results"


def test_an_unknown_sample_stops_with_the_list():
    for argv in (["nonsense"], []):
        with pytest.raises(SystemExit, match="zhu.coppin.burlon.limao"):
            S.run_sample(argv)


def test_the_swift_samples_take_fetch_or_measure():
    with pytest.raises(SystemExit, match="fetch.measure"):
        S.run_sample(["limao", "nonsense"])


def test_bin_rule_matches_samples():
    """Coppin et al.'s bursts were given their bins by the same rule (coppin.sample.build)."""
    sample = pd.read_csv(CODE_DIR / "shared" / "coppin_sample.csv", comment="#", index_col=0)
    for trigger, tres in zip(sample.trigger, sample.Tres):
        assert S.bin_for(trigger) == pytest.approx(tres)


def test_a_burst_with_no_peak_flux_asks_for_the_bin():
    with pytest.raises(BurstError, match="--tres"):
        S.bin_for("bn000000000")


def test_the_command_line_shows_the_new_options():
    out = subprocess.run([sys.executable, "-m", "grbtool", "--help"], cwd=CODE_DIR, capture_output=True, text=True).stdout
    assert "--zhu" in out and "--swift" in out and "sample" in out
    out = subprocess.run([sys.executable, "-m", "grbtool", "sample"], cwd=CODE_DIR, capture_output=True, text=True)
    assert out.returncode != 0 and "zhu|coppin|burlon|limao" in out.stderr


@pytest.mark.skipif(not (Path.home() / "swift_bat" / "summary_general.txt").exists(), reason="needs the BAT catalogue")
def test_bat_refuses_a_burst_it_did_not_see():
    from instruments.bat_sample import one_burst
    with pytest.raises(ValueError, match="BAT catalogue"):
        one_burst("GRB000000Z")


measured = pytest.mark.skipif(not (RESULTS / "coppin" / "160625B").is_dir() or not (RESULTS / "zhu" / "211211A").is_dir(),
                              reason="needs the measured samples in results/")


@measured
def test_the_soft_line_is_grb_230307a_ratio():
    from zhu.features import collect
    e = pd.read_csv(CODE_DIR / "shared" / "epeak_230307A.csv", comment="#").set_index("label")
    ratio = e.loc["pre", "Epeak_keV"] / collect("zhu2026", RESULTS / "zhu").loc["230307A", "Ep_ME_ours"]
    assert S.SOFT_RATIO == pytest.approx(ratio, abs=5e-5)


@measured
def test_the_summary_reads_a_burst_as_notebook_29_does():
    """The numbers notebook 29 prints for these bursts."""
    s = S.summary("160625B", RESULTS / "coppin")
    assert s["Ep_ratio"] == pytest.approx(0.054, abs=5e-4) and s["soft"] and not s["fast"]
    assert s["mvt_ratio"] == pytest.approx(10.79, abs=0.01)
    s = S.summary("211211A", RESULTS / "zhu")
    assert s["mvt_ratio"] == pytest.approx(7.0, abs=0.01) and s["mvt_ratio_err_dex"] == pytest.approx(0.448, abs=1e-3)
    s = S.summary("090929A", RESULTS / "coppin")
    assert s["EPI_zhu"] == pytest.approx(8.22, abs=0.01) and s["P_above_zhu"] == 1.0 and not s["soft"]
    assert "no precursor" in S.describe(S.summary("230307A", RESULTS / "zhu"))
