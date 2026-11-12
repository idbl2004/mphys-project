"""
Tests for the zhu package. Run from ignacio/ with: python -m pytest tests/test_zhu.py -q

Each rule is tested on a made-up light curve or table with a known answer. The tests marked as real data need GRB 211211A in
data/fermiGBM and are skipped without it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grbtool.bursts import DATA_ROOT
from sample_tools import ccf_lag
from zhu import classify
from zhu.catalogue import EPI_LINE, LITERATURE_Z, epi, read_table_2026, redshift_of
from zhu.background import LightCurve
from zhu.episodes import MODES, blocks_of, change_point_profile, find_episodes, mvt_of, summarise, with_errors
from zhu.lags import ccf_lag_fft, known_shift_check

HAVE_211211A = (DATA_ROOT / "GRB211211A" / "current").is_dir()
real_data = pytest.mark.skipif(not HAVE_211211A, reason="needs GRB 211211A in data/fermiGBM")

# A made-up burst on a 2000 counts/s background, 4 ms bins: precursor 0-0.2 s, quiet to 0.8 s, main emission in three steps to 4 s,
# a lull at 3% of the main peak to 5 s, then extended emission to 8 s.
TRES = 0.004
STEPS = [(0.0, 0.2, 6000.0), (0.8, 1.5, 20000.0), (1.5, 3.0, 40000.0), (3.0, 4.0, 15000.0), (4.0, 5.0, 1200.0), (5.0, 8.0, 4000.0)]


def made_up_burst(seed=1):
    edges = np.round(np.arange(-2.0, 10.0 + TRES / 2, TRES), 9)
    centres = 0.5 * (edges[1:] + edges[:-1])
    rate = np.full(len(centres), 2000.0)
    for a, b, r in STEPS:
        rate[(centres > a) & (centres < b)] += r
    counts = np.random.default_rng(seed).poisson(rate * TRES).astype(float)
    return LightCurve(edges, counts, np.full(len(counts), TRES), np.full(len(counts), 2000.0 * TRES), TRES)


@pytest.fixture(scope="module")
def burst_and_episodes():
    lc = made_up_burst()
    blocks = blocks_of(lc)
    return lc, blocks, find_episodes(lc, blocks)


def test_the_rule_finds_each_episode_of_a_made_up_burst(burst_and_episodes):
    _, _, found = burst_and_episodes
    for got, want in ((found.pe, (0.0, 0.2)), (found.qe1, (0.2, 0.8)), (found.me, (0.8, 4.0)), (found.qe2, (4.0, 5.0)), (found.ee, (5.0, 8.0))):
        assert got is not None
        assert abs(got[0] - want[0]) <= 3 * TRES and abs(got[1] - want[1]) <= 3 * TRES
    assert found.flags == []


def test_durations_are_whole_numbers_of_bins_and_mvts_whole_numbers_of_half_bins(burst_and_episodes):
    _, _, found = burst_and_episodes
    d = found.durations()
    for key in ("T100_PE", "T100_QE1", "T100_ME", "T100_EE", "T100_WE"):
        assert abs(d[key] / TRES - round(d[key] / TRES)) < 1e-6
    for key in ("TMVT_PE", "TMVT_ME"):
        assert abs(d[key] / (TRES / 2) - round(d[key] / (TRES / 2))) < 1e-6


def test_the_mvt_is_half_the_shortest_step(burst_and_episodes):
    _, _, found = burst_and_episodes
    assert found.mvt_pe == pytest.approx(0.1, abs=2 * TRES)       # the precursor is one 0.2 s block
    assert found.mvt_me == pytest.approx(0.35, abs=2 * TRES)      # the shortest step of the main emission is 0.7 s


def test_a_change_point_profile_peaks_at_the_step():
    counts = np.r_[np.full(100, 10.0), np.full(100, 30.0), np.full(100, 10.0)]
    edges, prob = change_point_profile(counts, np.ones(300), 0, 300, fixed=200)
    assert edges[np.argmax(prob)] == 100
    assert prob.sum() == pytest.approx(1.0)
    assert prob[edges >= 200].sum() < 1e-10       # the block between 100 and 200 is certainly there


def test_a_block_with_uncertain_edges_does_not_set_the_mvt():
    # a strong 20-bin step, and later one bin a little high (80 counts on 50, 4 sigma): the single bin is the shortest block, but its
    # edges aren't certain at 5 sigma, so the MVT comes from the step
    counts = np.full(400, 50.0)
    counts[100:120] = 150.0
    counts[200] = 80.0
    lc = LightCurve(np.arange(401) * TRES, counts, np.full(400, TRES), np.full(400, 40.0), TRES)
    blocks = pd.DataFrame(dict(i0=[0, 100, 120, 200, 201], i1=[100, 120, 200, 201, 400]))
    blocks["nbins"] = blocks.i1 - blocks.i0
    mvt, nbins, _, overlap = mvt_of(lc, counts, blocks, 0, 4)
    assert nbins == 20 and mvt == pytest.approx(10 * TRES) and overlap < 3e-7


def test_simulated_realisations_find_the_precursor_with_a_spread(burst_and_episodes):
    lc, blocks, _ = burst_and_episodes
    table = with_errors(lc, blocks, realisations=4)
    assert set(table["mode"]) == set(MODES) and len(table) == 4 * len(MODES)
    summary = summarise(table)["zhu2026"]
    assert summary["pe_found_fraction"] == 1.0
    assert summary["T100_PE_median"] == pytest.approx(0.2, abs=3 * TRES)


def test_a_burst_with_no_quiet_gap_has_no_precursor():
    lc = made_up_burst()
    centres = 0.5 * (lc.edges[1:] + lc.edges[:-1])
    filled = lc.counts + np.random.default_rng(2).poisson(np.where((centres > 0.2) & (centres < 0.8), 5000.0 * TRES, 0.0))
    found = find_episodes(lc, blocks_of(lc, filled), filled)
    assert found.pe is None
    assert "no_quiet_gap_before_main" in found.flags or "quiet_gap_relaxed" not in found.flags


def test_the_fft_cross_correlation_gives_the_same_lag():
    rng = np.random.default_rng(4)
    x = np.arange(3000)
    shape = 50 * np.exp(-((x - 1400) / 120.0) ** 2) + 80 * np.exp(-((x - 1700) / 40.0) ** 2)
    low = rng.poisson(np.roll(shape, 13) + 20) - 20.0
    high = rng.poisson(shape + 20) - 20.0
    assert ccf_lag_fft(low, high, 0.004, cut=True) == pytest.approx(ccf_lag(low, high, 0.004, cut=True), abs=1e-9)
    wide = np.r_[np.zeros(25), low, np.zeros(25)]        # the version with the soft band running past the window is passed straight on
    assert ccf_lag_fft(wide, high, 0.004) == ccf_lag(wide, high, 0.004)
    assert ccf_lag_fft(wide, high, 0.004) == pytest.approx(13 * 0.004, abs=0.004)


def test_the_lag_code_gets_a_known_shift_back():
    rng = np.random.default_rng(5)
    times = np.sort(np.r_[rng.normal(2.0, 0.3, 40000), rng.normal(3.5, 0.08, 30000), rng.uniform(0.0, 6.0, 20000)])
    for shift, (mean, _, error) in known_shift_check(times, (1.0, 5.0), 0.008, realisations=100).items():
        assert abs(mean - shift) < max(4 * error, 0.002)


def test_their_table_has_all_bursts_and_their_epi():
    table = read_table_2026()
    assert len(table) == 367
    assert table.Type.value_counts().to_dict() == {"II": 323, "I-S": 35, "I-L": 9}
    row = table.loc["161001A"]
    by_hand = np.log10(row.Ep_ME**2 / (row.T100_PE * row.T100_QE1**0.5 * row.TMVT_PE))
    assert row.EPI == pytest.approx(by_hand)
    assert row.EPI == pytest.approx(6.10, abs=0.01)
    assert epi(100.0, 1.0, 1.0, 1.0) == pytest.approx(4.0)


def test_redshifts_come_from_the_project_first():
    empty = pd.DataFrame(columns=["name", "z", "z_note", "photometric"])
    assert redshift_of("211211A", empty) == LITERATURE_Z["211211A"][0]
    found = pd.DataFrame([dict(name="090510", z=0.903, z_note="", photometric=False), dict(name="100101A", z=2.0, z_note="ph", photometric=True)])
    assert redshift_of("090510A", found) == pytest.approx(0.903)
    assert redshift_of("100101A", found) is None


def test_clusters_are_named_by_the_bursts_of_known_origin():
    names = ["211211A", "a", "b", "090618A", "c", "d", "e"]
    labels = np.array([0, 0, 1, 1, 2, -1, 0])
    named = classify.name_clusters(labels, names)
    assert list(named) == ["merger", "merger", "collapsar", "collapsar", "other", "other", "merger"]


def test_the_design_matrix_is_logged_and_z_scored():
    table = read_table_2026()
    x = classify.design_matrix(table)
    assert np.allclose(x.mean(), 0.0, atol=1e-9) and np.allclose(x.std(ddof=0), 1.0)
    raw = np.log10(table.T100_PE)
    assert np.allclose(x.T100_PE, (raw - raw.mean()) / raw.std(ddof=0))


def test_the_chance_of_lying_above_the_line():
    assert classify.probability_above(EPI_LINE, 0.2, 0.2) == pytest.approx(0.5)
    assert classify.probability_above(EPI_LINE + 0.2, 0.2, 0.2) == pytest.approx(0.8413, abs=1e-3)
    # uneven errors: the share above the measured value is the upper error's share
    assert classify.probability_above(6.0, 0.1, 0.3, line=6.0) == pytest.approx(0.75)


def test_the_threshold_sits_in_the_gap_between_two_clean_groups():
    table = pd.DataFrame(dict(EPI=np.r_[np.linspace(4, 5.9, 50), np.linspace(6.5, 8, 20)], Type=["II"] * 50 + ["I-S"] * 20))
    best, lo, hi, _ = classify.epi_threshold(table, boot=200)
    assert 5.9 < best < 6.5


@real_data
def test_their_channels_leave_out_the_k_edge_and_low_bgo():
    from grbtool.bursts import set_up
    from grbtool.fitting import all_spectra
    from grbtool.windows import Window
    from zhu.background import clear_background
    from zhu.spectra import zhu_data
    burst = set_up("GRB211211A", "bn211211549")
    clear_background(burst)
    window = Window("PE", [(-0.016, 0.408)], "precursor")
    spectra, _ = all_spectra(burst, [window])
    _, _, rsps, masks = zhu_data(burst, window, spectra, precursor=True)
    for det, rsp, mask in zip(burst.detectors, rsps, masks):
        lo, hi = np.asarray(rsp.ebounds.low_edges()), np.asarray(rsp.ebounds.high_edges())
        if det.startswith("n"):
            assert not np.any(mask & (hi > 30.0) & (lo < 40.0))
            assert lo[mask].min() < 11.0 and hi[mask].max() > 850.0
        else:
            assert hi[mask].min() > 300.0 and lo[mask].max() < 10000.0


@real_data
def test_the_quick_background_matches_the_full_one():
    from grbtool.bursts import set_up
    from lightcurve_tools import aligned_bin_edges, background_in_band, fit_background
    from zhu.background import background_counts, clear_background
    burst = set_up("GRB211211A", "bn211211549")
    clear_background(burst)
    fitter, _ = fit_background(burst.data_dir, burst.trigger, "n2", burst.bkg_windows, burst.bkg_order)
    edges = aligned_bin_edges((1.0, 11.0), 0.004)
    full = background_in_band(fitter, (8.0, 1000.0), edges)[0]
    quick = background_counts(fitter, (8.0, 1000.0), edges)
    assert np.max(np.abs(quick / full - 1)) < 1e-3
    assert quick.sum() == pytest.approx(full.sum(), rel=1e-5)
    # a window shorter than one coarse bin (GRB 090831A's precursor is 32 ms), and a single bin
    short = aligned_bin_edges((0.0, 0.032), 0.004)
    assert background_counts(fitter, (8.0, 1000.0), short).sum() == pytest.approx(background_in_band(fitter, (8.0, 1000.0), short)[0].sum(), rel=1e-3)
    assert len(background_counts(fitter, (8.0, 1000.0), np.array([0.0, 0.032]))) == 1


def test_a_fit_away_from_its_limits_with_finite_errors_counts_as_constrained(monkeypatch):
    from types import SimpleNamespace
    from gdt.core.spectra.functions import Comptonized
    from zhu import spectra
    function = Comptonized()
    fit = SimpleNamespace(parameters=np.array([0.1, 67.0, -1.15]))
    monkeypatch.setattr(spectra, "errors_of", lambda f: np.array([0.03, 6.2, 0.15]))
    assert spectra.constrained(function, fit, [0, 1])
    monkeypatch.setattr(spectra, "errors_of", lambda f: np.array([0.03, 90.0, 0.15]))      # E_peak known to worse than 100%
    assert not spectra.constrained(function, fit, [0, 1])
    fit.parameters = np.array([0.1, 67.0, -1.9])                                             # index at its lower limit
    monkeypatch.setattr(spectra, "errors_of", lambda f: np.array([0.03, 6.2, 0.15]))
    assert not spectra.constrained(function, fit, [0, 1])


def test_the_inspiral_time_matches_the_textbook_numbers():
    from zhu.physics import BNS, frequency_at, inspiral_time
    assert BNS == pytest.approx(1.219, abs=1e-3)                     # two 1.4 solar-mass stars
    assert inspiral_time(100.0) == pytest.approx(2.16, abs=0.02)      # course notes: 2.2 s at 100 Hz
    assert inspiral_time(188.0) == pytest.approx(0.40, abs=0.01)      # and 0.4 s at 188 Hz
    assert frequency_at(inspiral_time(150.0)) == pytest.approx(150.0)
    assert inspiral_time(100.0) / inspiral_time(200.0) == pytest.approx(2 ** (8 / 3))


def test_a_longer_delay_allows_a_lower_mode_frequency_and_a_redshift_raises_it():
    from zhu.physics import delay_limits
    ours = pd.DataFrame(dict(pe_start=[0.0, 0.0], me_start=[0.5, 2.0]), index=["a", "b"])
    out = delay_limits(ours)
    assert out.f_min_bns["a"] > out.f_min_bns["b"]
    with_z = delay_limits(ours, pd.Series({"a": 1.0}))
    assert with_z.delay_rest["a"] == pytest.approx(0.25)
    assert with_z.f_min_bns["a"] > out.f_min_bns["a"]
    assert with_z.f_min_nsbh["a"] < with_z.f_min_bns["a"]           # a heavier pair sweeps faster, so the same delay allows a lower frequency


def test_the_guarded_pgstat_is_gdts_where_gdts_is_defined_and_finite_where_it_is_not():
    from gdt.core.spectra.fitting import pgstat
    from zhu.spectra import safe_pgstat
    rng = np.random.default_rng(8)
    counts = rng.poisson(20.0, 100).astype(float)
    counts[:5] = 0.0
    model = rng.uniform(0.1, 2.0, 100)
    background = rng.uniform(5.0, 15.0, 100)
    variance = background.copy()
    args = (counts, model, 10.0, background, variance, 10.0)
    assert np.isfinite(pgstat(*args))
    assert safe_pgstat(*args) == pytest.approx(pgstat(*args), rel=1e-12)
    wild = (counts, model * 50, 10.0, background, (25.0 * np.ones(100)) ** 2, 10.0)     # variance far above the counts
    assert np.isfinite(safe_pgstat(*wild))


def test_two_batches_cannot_take_the_same_burst_and_a_stopped_one_lets_go(tmp_path):
    import subprocess
    from zhu import io, run
    held = run.claim("200101A", tmp_path)
    assert held is not None
    assert run.claim("200101A", tmp_path) is None                     # taken, even for a second claim from this process
    run.release(held)
    code = f"from zhu import run; h = run.claim('200101A', {str(tmp_path)!r}); print('held' if h else 'no', flush=True); import time; time.sleep(120)"
    child = subprocess.Popen([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        assert run.claim("200101A", tmp_path) is None                 # another process has it
    finally:
        child.kill()                                                  # stopped without letting go itself
        child.wait()
    again = run.claim("200101A", tmp_path)
    assert again is not None                                          # the kernel let go for it
    run.release(again)
    assert not run.finished("200101A", out_root=tmp_path)
    io.set_status("200101A", "complete", "", root=tmp_path)
    assert run.finished("200101A", out_root=tmp_path)
    io.set_status("200101A", "failed", "a bug", root=tmp_path)
    assert run.finished("200101A", out_root=tmp_path)                 # a failure waits for --retry-failed
    assert not run.finished("200101A", out_root=tmp_path, retry_failed=True, since="9999-01-01 00:00:00")
    assert run.finished("200101A", out_root=tmp_path, retry_failed=True, since="2000-01-01 00:00:00")     # failed again in this batch


def test_a_law_search_finds_the_law_that_made_the_labels():
    from zhu import laws
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=["a", "b", "c", "d"])
    y = (X.a - 0.5 * X.c + 0.1 * rng.normal(size=300)) > 0.2
    best = laws.search(X, y, max_terms=2).iloc[0]
    assert best.law == {"a": 1.0, "c": -0.5} or best.law == {"a": 2.0, "c": -1.0}
    assert best.auc > 0.97 and abs(best.threshold * (1 if best.law["a"] == 1.0 else 0.5) - 0.2) < 0.15
    folds, chosen = laws.nested_score(X, y, max_terms=2, repeats=2)
    assert folds.balanced_accuracy.mean() > 0.9


def test_the_best_threshold_is_in_the_gap_and_auc_counts_ties_half():
    from zhu import laws
    c, balanced = laws.best_threshold([1, 2, 3, 10, 11, 12], [0, 0, 0, 1, 1, 1])
    assert 3 < c < 10 and balanced == 1.0
    assert laws.auc([1, 1], [0, 1]) == 0.5
    assert np.allclose(laws.aucs(np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]), [0, 0, 1]), [1.0, 0.0])


def test_detectors_by_angle_take_the_closest_nais_and_the_bgo_on_their_side(monkeypatch):
    from zhu import run
    angles = {"n0": 80.0, "n1": 55.0, "n2": 30.0, "n3": 70.0, "n4": 90.0, "n5": 100.0, "n6": 45.0, "n7": 120.0, "n8": 130.0, "n9": 140.0, "na": 50.0, "nb": 59.0}
    monkeypatch.setattr(run, "_angle", lambda data_dir, trigger, det: angles[det])
    assert run.detectors_by_angle("GRBx", "bnx") == ["n2", "n6", "na", "b0"]
    angles["n2"] = 61.0
    assert run.detectors_by_angle("GRBx", "bnx") == ["n6", "na", "n1", "b1"]


def test_an_excess_in_the_first_block_is_not_a_precursor():
    lc = made_up_burst()
    edges = lc.edges[lc.edges >= -2.0]
    counts = lc.counts.copy()
    counts[:200] += np.random.default_rng(6).poisson(900.0 * TRES, 200)       # a small excess in the first 0.8 s of the light curve
    found = find_episodes(lc, blocks_of(lc, counts), counts)
    assert found.pe is not None and abs(found.pe[0] - 0.0) <= 3 * TRES



def gap_burst(gap_bins, seed=4):
    """A precursor at 0-0.2 s, a quiet gap of gap_bins bins, then a main emission of 1 s."""
    edges = np.round(np.arange(-2.0, 6.0 + TRES / 2, TRES), 9)
    centres = 0.5 * (edges[1:] + edges[:-1])
    rate = np.full(len(centres), 2000.0)
    gap_end = 0.2 + gap_bins * TRES
    rate[(centres > 0.0) & (centres < 0.2)] += 20000.0
    rate[(centres > gap_end) & (centres < gap_end + 1.0)] += 60000.0
    counts = np.random.default_rng(seed).poisson(rate * TRES).astype(float)
    return LightCurve(edges, counts, np.full(len(counts), TRES), np.full(len(counts), 2000.0 * TRES), TRES)


def test_the_two_papers_differ_in_the_shortest_quiet_gap():
    lc = gap_burst(9)                                   # a 36 ms gap: 9 bins, enough for 2026 (8) but not for 2025 (10)
    blocks = blocks_of(lc)
    assert find_episodes(lc, blocks, mode="zhu2026").pe is not None
    assert find_episodes(lc, blocks, mode="zhu2025").pe is None
    lc = gap_burst(30)
    blocks = blocks_of(lc)
    for mode in MODES:
        found = find_episodes(lc, blocks, mode=mode)
        assert found.pe is not None and found.mode == mode
        assert abs(found.qe1[1] - (0.2 + 30 * TRES)) <= 2 * TRES


def test_an_episode_result_records_its_blocks_and_reasons(burst_and_episodes):
    _, blocks, found = burst_and_episodes
    record = found.record()
    for key in ("mode", "bin_width", "p0", "pe_blocks", "qe1_blocks", "me_blocks", "qe2_blocks", "ee_blocks", "pe_peak_rate", "qe1_snr_bin",
                "qe1_dip", "qe1_recovery", "qe2_dip", "qe2_recovery", "reasons", "flags", "T100_PE"):
        assert key in record
    first, last = record["pe_blocks"]
    assert blocks.start[first] == pytest.approx(record["pe"][0]) and blocks.stop[last] == pytest.approx(record["pe"][1])
    assert record["qe2_dip"] < 0.15 and record["qe2_recovery"] >= 2.0
    assert any(r.startswith("qe1:") for r in record["reasons"]) and any(r.startswith("qe2:") for r in record["reasons"])
    with pytest.raises(ValueError):
        find_episodes(made_up_burst(), blocks, mode="zhu2027")


def test_each_burst_keeps_a_state_and_a_failure_is_recorded_not_skipped(tmp_path, monkeypatch):
    from grbtool.bursts import BurstError
    from zhu import io, run
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    monkeypatch.setattr(io, "RESULTS", tmp_path)
    io.set_stage("b1", "episodes", "done", "12 blocks", root=tmp_path, version="test")
    assert io.state("b1", tmp_path)["stages"]["episodes"]["status"] == "done"
    calls = []

    def fake_measure(name, row, workers, realisations, out_root):
        calls.append(name)
        if name == "nodata":
            io.set_status(name, "not_measurable", "no data", out_root)
            raise BurstError("no data")
        if name == "broken":
            raise RuntimeError("a bug")
        io.set_status(name, "complete", "", out_root)

    monkeypatch.setattr(run, "measure", fake_measure)
    sample = pd.DataFrame(dict(trigger=["t1", "t2", "t3"], reason=["z25"] * 3, Tres=[0.004] * 3), index=["good", "nodata", "broken"])
    failed = run.batch(sample, out_root=tmp_path)
    assert set(failed) == {"nodata", "broken"}
    table = io.states(sample.index, tmp_path)
    assert table.status.to_dict() == {"good": "complete", "nodata": "not_measurable", "broken": "failed"}
    assert "a bug" in table.reason["broken"]
    for name in sample.index:                            # no burst is left taken
        handle = run.claim(name, tmp_path)
        assert handle is not None
        run.release(handle)
    calls.clear()
    run.batch(sample, out_root=tmp_path)                 # a second run takes nothing: every burst has a final state
    assert calls == []
    run.batch(sample, out_root=tmp_path, retry_failed=True)
    assert calls == ["broken"]                           # unless asked to try the failures again, once


def test_quality_statuses_say_what_kind_of_number_each_is():
    from zhu import quality
    row = dict(T100_PE=0.2, pe_found_fraction=0.95, T100_WE=80.0, flags="emission_reaches_background", measured_PE="False",
               Ep_PE_cpl_limit_lo=40.0, Ep_PE_ours=60.0, Ep_ME_zhu=np.nan, tau32_ME_ms=np.nan, tau32_ME_status="unusable:faint")
    assert quality.timing_status(row, "T100_PE") == "measured"
    assert quality.timing_status(dict(row, pe_found_fraction=0.5), "T100_PE") == "unstable"
    assert quality.timing_status(row, "T100_WE") == "lower_limit"
    assert quality.timing_status(row, "TMVT_ME") == "missing"
    assert quality.epeak_status(row, "PE", "ours") == "limit"
    assert quality.epeak_status(row, "ME", "zhu") == "unusable"
    assert quality.lag_status(row, "tau32_ME") == "unusable:faint"
    assert "lower limits" in quality.meaning("emission_reaches_background")


def test_the_literature_table_keeps_each_numbers_source():
    from zhu.catalogue import read_literature
    lit = read_literature()
    assert {"paper", "where", "burst", "quantity", "value", "minus", "plus", "units", "definition", "source"} <= set(lit.columns)
    assert set(lit.source) <= {"tabulated", "text", "named in figure", "read from figure"}
    row = lit[(lit.burst == "090618A") & (lit.quantity == "start_ME")].iloc[0]
    assert float(row["value"]) == pytest.approx(49.992) and row["where"] == "Table 2 (printed rows)"
    ep = lit[(lit.burst == "090131A") & (lit.quantity == "Ep_ME_CPL+BB")].iloc[0]
    assert float(ep.value) == pytest.approx(68.15) and float(ep.minus) == pytest.approx(15.79)
    assert (lit[lit.quantity == "in_2025_sample"].source == "read from figure").sum() == 17
    assert len(lit[(lit["where"] == "Table 1 (machine-readable)") & (lit.quantity == "Ep_ME")]) == 367


HAVE_230307A = (DATA_ROOT / "GRB230307A" / "current").is_dir()


def _precursor_fit(folder, trigger, window):
    from grbtool.bursts import set_up
    from grbtool.fitting import all_spectra
    from grbtool.windows import Window
    from zhu.background import clear_background
    from zhu.spectra import fit_window_ours
    burst = set_up(folder, trigger)
    clear_background(burst)
    w = Window("PE", [window], "precursor")
    spectra, _ = all_spectra(burst, [w])
    return fit_window_ours(burst, w, spectra, minutes=5)


@pytest.mark.slow
@real_data
def test_grb211211a_precursor_spectrum_matches_notebook_14():
    # notebook 14: the precursor is Comptonized with E_peak 64-67 keV
    row = _precursor_fit("GRB211211A", "bn211211549", (-0.016, 0.408))
    assert row["model"] == "Comptonized" and row["measured"]
    assert 62.0 < row["Epeak"] < 70.0
    assert 3.0 < row["Epeak_err_lo"] < 9.0 and 3.0 < row["Epeak_err_hi"] < 10.0


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_230307A, reason="needs GRB 230307A in data/fermiGBM")
def test_grb230307a_precursor_band_peak_matches_notebook_09():
    # notebook 09: Band E_peak 170.3 +/- 5.3 keV with Dichiara's four detectors; with na and b1 alone notebook 12 found 0-14 keV higher
    row = _precursor_fit("GRB230307A", "bn230307656", (0.0, 0.4))
    assert row["measured"]
    assert 170.3 - 3 * 5.3 < row["Epeak_band"] < 170.3 + 3 * 5.3 + 14.0
    assert row["model"] in ("Band", "BandLowBreak")         # the two-break model wins in na, the low-energy break of open point 3


def test_a_column_that_carries_the_labels_passes_the_permutation_test_and_noise_does_not():
    from zhu import laws
    rng = np.random.default_rng(12)
    X = pd.DataFrame(dict(base=rng.normal(size=160), signal=rng.normal(size=160), noise=rng.normal(size=160)))
    y = (0.6 * X.base + X.signal + 0.5 * rng.normal(size=160)) > 0.4
    gain, _, p = laws.permutation_test(X, y, ["base"], "signal", permutations=30, repeats=2)
    assert gain > 0 and p < 0.05
    gain, _, p = laws.permutation_test(X, y, ["base", "signal"], "noise", permutations=30, repeats=2)
    assert p > 0.05
    labels = laws.label_table(["211211A", "090510A", "090618A"], pd.Series({"211211A": "I-L", "090510A": "I-S", "090618A": "II"}))
    assert labels.label_status.tolist() == ["confirmed", "candidate", "confirmed"]


def test_the_2026_count_test_uses_the_main_emission_alone():
    """A precursor with more counts than the main pulse but fewer than the main pulse and its extended emission together. The gap after
    the precursor is at the background, so the 2026 rule compares counts: the main emission's alone, not the extended emission's too."""
    edges = np.round(np.arange(-2.0, 9.0 + TRES / 2, TRES), 9)
    centres = 0.5 * (edges[1:] + edges[:-1])
    rate = np.full(len(centres), 2000.0)
    for a, b, r in [(0.0, 1.0, 10000.0), (1.5, 1.8, 25000.0), (3.0, 6.0, 3000.0)]:
        rate[(centres > a) & (centres < b)] += r
    counts = np.random.default_rng(8).poisson(rate * TRES).astype(float)
    lc = LightCurve(edges, counts, np.full(len(counts), TRES), np.full(len(counts), 2000.0 * TRES), TRES)
    blocks = blocks_of(lc)
    late = find_episodes(lc, blocks, mode="zhu2026")
    assert late.ee is not None and late.pe is None and "pe_counts_not_below_me" in late.flags
    assert find_episodes(lc, blocks, mode="zhu2025").pe is not None          # 2025 compares peaks, and the main peak is higher


def test_the_light_curve_runs_between_the_background_intervals_on_whichever_sides_there_are():
    from types import SimpleNamespace
    from zhu.background import emission_span
    burst = SimpleNamespace(t90_start=0.0, t90=10.0, tte_range=(-30.0, 300.0), bkg_windows=[(-25.0, -5.0), (50.0, 250.0)])
    assert emission_span(burst) == (-5.0, 50.0)
    burst.bkg_windows = [(50.0, 250.0)]                  # nothing before the burst: from 1 s after the data start
    assert emission_span(burst) == (-29.0, 50.0)
    burst.bkg_windows = [(-25.0, -5.0)]                  # nothing after: to 1 s before the data end
    assert emission_span(burst) == (-5.0, 299.0)


def test_a_ccf_peak_counts_only_if_it_is_a_real_top():
    from zhu.lags import ccf_peak
    t = np.arange(400.0)
    pulse = 200.0 * np.exp(-0.5 * ((t - 200.0) / 12.0) ** 2)
    late = np.r_[np.zeros(25), np.roll(pulse, 3), np.zeros(25)]
    lag, real = ccf_peak(late, pulse, 0.004)
    assert real and lag == pytest.approx(ccf_lag(late, pulse, 0.004)) and lag == pytest.approx(0.012, abs=0.002)
    beyond = np.r_[np.zeros(25), np.roll(pulse, 40), np.zeros(25)]       # a shift past the 25-bin margin: the CCF only rises
    assert ccf_peak(beyond, pulse, 0.004)[1] is False
    lag, real = ccf_peak(np.roll(pulse, 3), pulse, 0.004, cut=True)
    assert real and lag == pytest.approx(ccf_lag(np.roll(pulse, 3), pulse, 0.004, cut=True))


def test_epeak_draws_stay_positive_and_keep_uneven_errors():
    from zhu.features import draw_two_piece
    rng = np.random.default_rng(0)
    draws = 10 ** draw_two_piece(100.0, 30.0, 60.0, 40000, rng)
    assert draws.min() > 0
    low, high = np.percentile(draws, [15.87, 84.13])
    assert low == pytest.approx(70.0, rel=0.06) and high == pytest.approx(160.0, rel=0.06)
    assert draw_two_piece(100.0, np.nan, np.nan, 10, rng) is None             # no error: no draws, rather than draws with no spread
    assert draw_two_piece(np.nan, 10.0, 10.0, 10, rng) is None
    one_sided = 10 ** draw_two_piece(100.0, 150.0, 20.0, 40000, rng)          # minus error past zero: the lower side is open
    assert np.percentile(one_sided, 84.13) == pytest.approx(120.0, rel=0.05)


def _write_burst(folder, spectra_me=(1.0, 3.0), measured_pe=False):
    """A results folder for one made-up burst, as run.measure writes it."""
    import json
    folder.mkdir(parents=True)
    found = dict(pe=[0.0, 0.2], qe1=[0.2, 1.0], me=[1.0, 3.0], qe2=None, ee=None, we=[0.0, 3.0], T100_PE=0.2, T100_QE1=0.8, TMVT_PE=0.01,
                 T100_ME=2.0, TMVT_ME=0.005, T100_WE=3.0, flags=[], reasons=[])
    (folder / "episodes.json").write_text(json.dumps(dict(burst=folder.name, trigger="bn000000000", reason="type_I", Tres=0.004, nbins=1,
                                                          nblocks=1, data_gaps=0, bkg_flags=[], setup_flags=[], modes={"zhu2026": found})))
    (folder / "state.json").write_text(json.dumps(dict(burst=folder.name, status="complete", reason="", stages={})))
    pd.DataFrame([dict(label="PE", t_start=0.0, t_stop=0.2, Epeak=60.0, Epeak_err_lo=10.0, Epeak_err_hi=20.0, measured=measured_pe,
                       zhu_Epeak=55.0, zhu_Epeak_err_lo=np.nan, zhu_Epeak_err_hi=15.0),
                  dict(label="ME", t_start=spectra_me[0], t_stop=spectra_me[1], Epeak=500.0, Epeak_err_lo=50.0, Epeak_err_hi=60.0, measured=True,
                       zhu_Epeak=480.0, zhu_Epeak_err_lo=40.0, zhu_Epeak_err_hi=50.0)]).set_index("label").to_csv(folder / "spectra_zhu2026.csv")
    real = pd.DataFrame(dict(mode="zhu2026", has_pe=True, has_ee=False, T100_PE=0.2 * np.exp(np.linspace(-0.1, 0.1, 10)), T100_QE1=0.8,
                             TMVT_PE=0.01, T100_ME=2.0, TMVT_ME=0.005, T100_WE=3.0, T100_EE=np.nan, T100_QE2=np.nan))
    real.loc[[0, 1], ["T100_PE", "T100_QE1", "TMVT_PE"]] = np.nan          # two light curves that lost the precursor
    real.to_csv(folder / "realisations.csv", index=False)


def test_a_spectrum_is_read_only_on_the_window_it_was_fitted_to(tmp_path):
    from zhu.features import load_burst
    _write_burst(tmp_path / "b1", spectra_me=(1.0, 2.5))
    row = load_burst(tmp_path / "b1", "zhu2026")
    assert "no_spectrum_for_this_window:ME" in row["flags"] and "Ep_ME_zhu" not in row
    assert row["Ep_PE_zhu"] == 55.0 and "ep_error_one_sided:PE_zhu" in row["flags"]


def test_draws_of_a_bursts_numbers_skip_light_curves_that_lost_the_precursor(tmp_path):
    from zhu.features import epi_samples, input_draws, load_burst
    _write_burst(tmp_path / "b1")
    row = load_burst(tmp_path / "b1", "zhu2026")
    draws = input_draws(row, tmp_path / "b1", n=500)
    assert "Ep_PE" not in draws                          # our rule doesn't count the precursor E_peak as measured
    assert draws.T_PE.notna().all() and draws.T_PE.std() > 0
    assert np.log10(0.2) - 0.11 < draws.T_PE.min() and draws.T_PE.max() < np.log10(0.2) + 0.11
    assert epi_samples(row, tmp_path / "b1", n=500).size == 500
    _write_burst(tmp_path / "b2", measured_pe=True)
    assert "Ep_PE" in input_draws(load_burst(tmp_path / "b2", "zhu2026"), tmp_path / "b2", n=50)


def test_each_modes_spectra_are_on_its_own_windows_and_a_window_is_fitted_once(tmp_path, monkeypatch):
    from zhu import run, spectra
    calls = []

    def fake_fit(burst, windows, workers=1, minutes=10.0, ways=("ours", "zhu")):
        calls.append([(w.label, w.span, tuple(ways)) for w in windows])
        rows = []
        for w in windows:
            row = dict(label=w.label, t_start=w.span[0], t_stop=w.span[1], pieces=len(w.pieces), Epeak=100.0, measured=True, seconds_ours=1.0)
            if "zhu" in ways:
                row.update(zhu_Epeak=120.0, seconds_zhu=1.0)
            rows.append(row)
        return pd.DataFrame(rows).set_index("label")

    monkeypatch.setattr(spectra, "fit_episodes", fake_fit)
    same = dict(pe=[0.0, 0.2], me=[1.0, 3.0])
    record = dict(reason="type_I", bad_intervals=[[0.45, 0.65], [0.9, 1.2]],
                  modes={"zhu2026": same, "zhu2025": dict(same), "project": dict(pe=[0.5, 0.6], me=[1.0, 3.0])})
    run.stage_spectra(record, burst=None, out=tmp_path, workers=1)
    assert len(calls) == 1 and [c[0] for c in calls[0]] == ["PE", "ME"]          # zhu2025 and the project ME reuse those fits
    late = pd.read_csv(tmp_path / "spectra_zhu2026.csv", index_col=0)
    assert late.loc["ME", "t_start"] == 1.0 and late.loc["ME", "pieces"] == 1    # the episode's window, though 1.0-1.2 s was cut out
    assert pd.read_csv(tmp_path / "spectra_zhu2025.csv", index_col=0).equals(late)
    ours = pd.read_csv(tmp_path / "spectra_project.csv", index_col=0)
    assert ours.loc["PE", "fit_flags"] == "window_inside_bad_interval" and not np.isfinite(ours.loc["PE", "Epeak"])
    assert ours.loc["ME", "Epeak"] == 100.0 and not [c for c in ours.columns if c.startswith("zhu_") or c == "seconds_zhu"]

