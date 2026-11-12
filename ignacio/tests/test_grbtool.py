"""
Tests for grbtool. Run from ignacio/ with: python -m pytest tests/test_grbtool.py -q

The rules are tested on small made-up cases with a known answer. The tests marked as real data need GRB 230307A, and the MVT one
needs Rodrigo's repository, so it's skipped without it. They take a few minutes.
"""
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import BURST_ID, DATA_DIR
from sample_tools import precursor_from_blocks
from gdt.core.spectra.functions import Band
from spectral_fit_tools import fit_model, scaled_starts
from grbtool import fitting, joint
from grbtool.bursts import PRESETS, BurstError, find_trigger, set_up, event_times
from grbtool.checks import Rates, TELEMETRY_KCPS
from grbtool.fitting import BAND_SHAPES, OutOfTime, Stages, all_spectra, choose_model, fit_flags, fit_window, time_limit, window_data
from grbtool.pipeline import merged_flags
from grbtool.report import quick_look
from lightcurve_tools import fit_background
from grbtool.mvt import BALA_BACKGROUND, ENERGY, background_intervals, background_rate, bala_class, code_folder, light_curve, load_haar, merged_times, snr_mvt, threshold, validation_curve
from grbtool.physics import amati_offsets, budget, k_correction, lag_bin
from grbtool.windows import Window, cut_out, lull_precursor, windows_from_file


def test_cut_out_splits_a_window_around_a_bad_interval():
    assert cut_out((0.32, 18.0), [(3.0, 7.0)]) == [(0.32, 3.0), (7.0, 18.0)]
    assert cut_out((0.32, 2.0), [(3.0, 7.0)]) == [(0.32, 2.0)]
    assert cut_out((4.0, 5.0), [(3.0, 7.0)]) == []
    assert cut_out((5.0, 9.0), [(3.0, 7.0)]) == [(7.0, 9.0)]


def test_window_duration_leaves_out_what_was_cut():
    w = Window("x", [(0.32, 3.0), (7.0, 18.0)], "main")
    assert w.span == (0.32, 18.0)
    assert np.isclose(w.duration, 13.68)


def made_up_blocks(excess, sigma):
    starts = np.arange(len(excess), dtype=float)
    return pd.DataFrame(dict(start=starts, stop=starts + 1.0, excess=excess, sigma=sigma))


def test_a_dip_that_never_gets_quiet_is_found_only_by_the_second_rule():
    blocks = made_up_blocks([0.0, 100.0, 10.0, 30.0, 500.0, 200.0], [0.0, 30.0, 12.0, 15.0, 80.0, 40.0])
    assert precursor_from_blocks(blocks) is None
    found = lull_precursor(blocks)
    assert found["pre_start"] == 1.0
    assert found["pre_stop"] == 2.0
    assert found["main_start"] == 4.0


def test_a_burst_with_one_pulse_has_no_precursor():
    blocks = made_up_blocks([0.0, 20.0, 100.0, 500.0, 200.0], [0.0, 6.0, 20.0, 80.0, 40.0])
    assert lull_precursor(blocks) is None


def test_the_telemetry_rule_rounds_out_to_half_seconds():
    edges = np.arange(0.0, 10.0, 0.016)
    summed = np.full(len(edges) - 1, 200.0)
    summed[(edges[:-1] >= 3.25) & (edges[:-1] < 6.7)] = TELEMETRY_KCPS + 1.0
    assert Rates.telemetry_intervals(SimpleNamespace(edges=edges, summed=summed)) == [(3.0, 7.0)]


def test_hits_of_the_limit_close_together_are_one_stretch_and_far_apart_are_not():
    edges = np.arange(0.0, 40.0, 0.016)
    summed = np.full(len(edges) - 1, 200.0)
    for a, b in ((3.25, 3.3), (5.95, 6.72), (30.1, 30.3)):
        summed[(edges[:-1] >= a) & (edges[:-1] < b)] = TELEMETRY_KCPS + 1.0
    assert Rates.telemetry_intervals(SimpleNamespace(edges=edges, summed=summed)) == [(3.0, 7.0), (30.0, 30.5)]


def test_no_bad_interval_below_the_limit():
    edges = np.arange(0.0, 10.0, 0.016)
    assert Rates.telemetry_intervals(SimpleNamespace(edges=edges, summed=np.full(len(edges) - 1, TELEMETRY_KCPS - 1.0))) == []


def fake_fits(c_band, c_cpl, c_lb, band=(1.0, 300.0, -1.0, -3.0), lb=(1.0, 40.0, -0.5, -1.5, 300.0, -3.0)):
    fit = lambda c, p: SimpleNamespace(statistic=c / 2.0, parameters=np.array(p))
    limits = SimpleNamespace(min_values=[0, 0, -1.9], max_values=[0, 0, 2.0])
    return dict(band=(None, fit(c_band, band)), cpl=(None, fit(c_cpl, (1.0, 300.0, -1.0))), lowbreak=(limits, fit(c_lb, lb)))


def test_the_model_rule():
    assert choose_model(fake_fits(500.0, 500.5, 499.0))[0] == "cpl"
    assert choose_model(fake_fits(500.0, 520.0, 499.0))[0] == "band"
    assert choose_model(fake_fits(500.0, 520.0, 480.0)) == ("lowbreak", 4)
    # a break below the bottom of the band, or a slope at its limit, isn't a clean break however good the fit
    assert choose_model(fake_fits(500.0, 520.0, 480.0, lb=(1.0, 5.0, -0.5, -1.5, 300.0, -3.0)))[0] == "band"
    assert choose_model(fake_fits(500.0, 520.0, 480.0, lb=(1.0, 40.0, -1.9, -1.5, 300.0, -3.0)))[0] == "band"
    # beta at its limit means Band's peak isn't a peak
    assert choose_model(fake_fits(500.0, 520.0, 499.0, band=(1.0, 300.0, -1.0, -2.0)))[0] == "cpl"


def test_the_table_says_when_a_better_two_break_was_thrown_out():
    not_clean = fake_fits(500.0, 520.0, 480.0, lb=(1.0, 40.0, -1.9, -1.5, 300.0, -3.0))
    assert fit_flags(not_clean, choose_model(not_clean)[0], []) == "two_break_rejected"
    # nothing to say when the two-break isn't much better, or when it's the one quoted
    assert fit_flags(fake_fits(500.0, 520.0, 499.0), "band", []) == ""
    assert fit_flags(fake_fits(500.0, 520.0, 480.0), "lowbreak", []) == ""
    assert fit_flags(fake_fits(500.0, 520.0, 480.0), "lowbreak", ["band"]) == "out_of_time"


def test_the_flags_of_the_rates_the_fits_and_the_checks_end_up_in_one_column():
    table = pd.DataFrame(dict(flags=["data_gaps", "", np.nan], fit_flags=["two_break_rejected", "", np.nan], check_failed=[np.nan, "OutOfTime: late", np.nan]))
    assert merged_flags(table) == ["data_gaps two_break_rejected", "checks_failed", ""]
    assert merged_flags(table[["flags"]]) == ["data_gaps", "", ""]


def test_the_time_limit_stops_a_slow_step():
    with pytest.raises(OutOfTime):
        with time_limit(0.05):
            while True:
                pass


def test_the_time_limit_leaves_nothing_behind_after_a_quick_step():
    with time_limit(5.0):
        pass
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_no_start_finishing_in_time_is_an_error(monkeypatch):
    def never(*_):
        raise OutOfTime
    monkeypatch.setattr(fitting, "fit_model", never)
    with pytest.raises(OutOfTime):
        Stages().fit("band", (None, None, None, None), Band(), [[1.0, 300.0, -1.0, -3.0, 100.0]])


def made_up_table(**columns):
    base = dict(kind=["precursor", "main"], t_start=[-1.0, 5.0], t_stop=[0.0, 15.0], Epeak=[100.0, 500.0], Epeak_err_lo=[10.0, 20.0], Epeak_err_hi=[10.0, 20.0],
                model=["Comptonized", "BandLowBreak"], measured=[True, False])
    return pd.DataFrame(base | columns, index=["pre", "main"])


def test_quick_look_draws_with_and_without_a_precursor_and_a_failed_fit(tmp_path):
    burst = SimpleNamespace(name="GRB000000A", detectors=["n0", "b0"], redshift=None)
    blocks = pd.DataFrame(dict(start=[-5.0, -1.0, 0.0, 5.0, 15.0], stop=[-1.0, 0.0, 5.0, 15.0, 20.0], excess=[0.0, 1000.0, 0.0, 5000.0, 0.0], sigma=[0.0, 10.0, 0.0, 50.0, 0.0]))
    quick_look(burst, made_up_table(), blocks, [(7.0, 8.0)], tmp_path / "both.png")
    quick_look(burst, made_up_table().iloc[1:], None, [], tmp_path / "no_precursor_no_blocks.png")
    quick_look(burst, made_up_table(Epeak=[np.nan, np.nan]), blocks, [], tmp_path / "failed.png")
    for name in ("both", "no_precursor_no_blocks", "failed"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0


def test_k_correction_of_a_flat_spectrum_is_the_ratio_of_decades():
    # N(E) = E^-2 is flat in nuFnu, so the energy goes as the log of the range, whatever the redshift
    assert np.isclose(k_correction(lambda e: e**-2.0, 0.5), np.log(1e4) / np.log(100.0), rtol=1e-6)


def test_amati_offset_is_zero_on_the_track():
    out = amati_offsets(1e51, 100.0 * 10**0.83)
    assert np.isclose(out["dex_from_type_I"], 0.0)
    assert np.isclose(out["dex_from_type_II"], 0.83 + 0.24)


def test_the_budget_at_dichiara_luminosity():
    out = budget(3.6e50)
    assert np.isclose(out["rate_over_fiducial"], 18.0)
    assert np.isclose(out["B_min_G"], 1.2e15)


def test_lag_bins_are_powers_of_two_and_at_least_4_ms():
    assert lag_bin(0.1, 64) == 4e-3
    assert lag_bin(20.0, 512) == 32e-3


def test_mvt_threshold_stays_flat_outside_its_range():
    assert threshold(2e-3, 5.0) == pytest.approx(135.0)
    assert threshold(1e-3, 5.0) == pytest.approx(135.0)
    assert threshold(50e-3, 5.0) == pytest.approx(65.0)
    assert threshold(6e-3, 0.4) == pytest.approx(52.0)


def test_a_name_that_is_not_a_burst_stops():
    with pytest.raises(BurstError):
        find_trigger("not a burst")


# real data
@pytest.fixture(scope="module")
def burst():
    b = set_up("GRB230307A", BURST_ID, data_dir=DATA_DIR, detectors=PRESETS["GRB230307A"]["detectors"])
    b.bkg_windows = PRESETS["GRB230307A"]["bkg_windows"]
    return b


def test_set_up_from_the_catalogue():
    b = set_up("GRB230307A", BURST_ID, data_dir=DATA_DIR)
    assert b.detectors == ["na", "b1"]
    assert np.isclose(b.t90, 34.56, atol=0.01)
    assert b.redshift == 0.065


def test_the_tool_gives_back_the_table_for_the_precursor(burst):
    windows = {w.label: w for w in windows_from_file(PRESETS["GRB230307A"]["windows_file"])}
    spectra, _ = all_spectra(burst, [windows["pre"]])
    row = fit_window(burst, windows["pre"], spectra)
    table = pd.read_csv(PRESETS["GRB230307A"]["table"], index_col="label").loc["pre"]
    assert abs(row["Epeak_band"] - table.Epeak_keV) < 0.05 * 0.5 * (table.Epeak_err_lo + table.Epeak_err_hi)
    assert abs(row["C_band"] - table.cstat) < 0.01
    assert abs(row["Epeak_cpl"] - table.Epeak_cpl_keV) < 0.1 * row["Epeak_cpl_err"]


@pytest.fixture(scope="module")
def precursor_data(burst):
    windows = {w.label: w for w in windows_from_file(PRESETS["GRB230307A"]["windows_file"])}
    spectra, _ = all_spectra(burst, [windows["pre"]])
    return window_data(burst, windows["pre"], spectra)


def test_a_fit_given_the_time_is_the_fit_from_fit_model(precursor_data):
    starts = scaled_starts(Band(), *precursor_data, BAND_SHAPES[:6])
    expected = fit_model(*precursor_data, Band(), starts, polish=True)
    stages = Stages()
    fit = stages.fit("band", precursor_data, Band(), starts)
    assert np.allclose(fit.parameters, expected.parameters, rtol=1e-10)
    assert fit.statistic == pytest.approx(expected.statistic, rel=1e-12)
    assert stages.late == []


def test_running_out_of_time_keeps_the_best_start_and_says_so(precursor_data, monkeypatch):
    starts = scaled_starts(Band(), *precursor_data, BAND_SHAPES[:6])
    real, calls = fit_model, []

    def stops_at_the_second_start(*args):
        calls.append(1)
        if len(calls) > 1:
            raise OutOfTime
        return real(*args)
    monkeypatch.setattr(fitting, "fit_model", stops_at_the_second_start)
    stages = Stages()
    fit = stages.fit("band", precursor_data, Band(), starts)
    assert len(calls) == 2
    assert stages.late == ["band"]
    assert np.isfinite(fit.statistic)


def test_rates_find_the_bad_interval_and_the_gaps(burst):
    rates = Rates(burst, (-2.0, 50.0))
    assert rates.telemetry_intervals() == [(3.0, 7.0)]
    assert len(rates.gaps) == 12
    assert np.isclose(rates.summed.max(), 377.7, atol=0.2)


@pytest.mark.skipif(not (code_folder() / "analysisTools").exists(), reason="Rodrigo's repository isn't there")
def test_haar_step_on_the_precursor(burst):
    haar, _ = load_haar()
    detectors = ["n0", "n1", "n6", "n7", "n9", "na"]
    events = event_times(burst, detectors, ENERGY, time_range=(-1.0, 1.0))
    counts = light_curve(events, detectors, (-0.064, 0.320), 3e-6)
    assert counts.sum() == 14534
    assert 1e3 * haar.haarPowerMod(counts, np.sqrt(counts), minimumBinSizeSeconds=3e-6)[2] == pytest.approx(12.764, abs=0.01)


# The table from Rodrigo's program. The header is the one in his document of 4 Nov. The rows only test the format: pre is his example row with its blanks filled, the others are made up.
MVT_HEADER = "label,mvt_s,mvt_err_lo_s,mvt_err_hi_s,band_keV,upper_limit,class,snr_physical,snr_bala,snr_physical_ibkg,detectors,bin_width_s,valid_pct,trusted,flags,sys_note,script,mvt_commit\n"
MVT_PRE = "pre,0.01522,0.00683,0.00219,8-900,false,robust,121.6,111.1,123.4,n0 n1 n6 n7 n9 na,3e-06,42,untested,gap,,run_window_mvt.py,abc1234\n"
MVT_LIMIT = "m01,0.0040,,,8-900,true,UL,90,80,91,n0 n1,3e-06,20,untested,gap;pileup-na,,run_window_mvt.py,abc1234\n"
MVT_NA_ALONE = "m08,0.030,0.004,0.005,8-900,false,robust,70,65,71,na,3e-06,80,true,,,run_window_mvt.py,abc1234\n"


def mvt_file(tmp_path, text):
    path = tmp_path / "mvt.csv"
    path.write_text(text)
    return path


def test_his_table_joins_to_the_epeak_table_of_both_bursts(tmp_path):
    first = joint.read_mvt_table(mvt_file(tmp_path, MVT_HEADER + MVT_PRE + MVT_LIMIT))
    second = joint.read_mvt_table(mvt_file(tmp_path, MVT_HEADER + MVT_PRE))
    for name, table in (("230307A", first), ("211211A", second)):
        epeak = joint.read_epeak_table(joint.SHARED / f"epeak_{name}.csv")
        joined, left_out = joint.join_tables(epeak, table)
        assert list(joined.index) == list(table.index) and left_out["no_epeak"] == [] and left_out["no_value"] == []
        assert (joined.Epeak_total_hi >= joined.Epeak_err_hi).all()


def test_the_join_also_takes_a_results_table_of_the_tool_whose_flags_are_about_the_spectra(tmp_path):
    epeak = pd.DataFrame(dict(label=["pre"], t_start=[-0.064], t_stop=[0.32], Epeak=[170.3], Epeak_err_lo=[5.0], Epeak_err_hi=[5.0], flags=["no_quiet_gap"]))
    results = tmp_path / "windows.csv"
    epeak.to_csv(results, index=False)
    joined, _ = joint.join_tables(joint.read_epeak_table(results), joint.read_mvt_table(mvt_file(tmp_path, MVT_HEADER + MVT_PRE)))
    assert joined.loc["pre", "spectral_flags"] == "no_quiet_gap" and joined.loc["pre", "flags"] == "gap" and joined.loc["pre", "Epeak_keV"] == 170.3


def test_text_columns_stay_text_whatever_the_rows_hold(tmp_path):
    table = joint.read_mvt_table(mvt_file(tmp_path, MVT_HEADER + MVT_NA_ALONE))
    assert table.loc["m08", "detectors"] == "na" and table.loc["m08", "trusted"] == "true" and table.loc["m08", "flags"] == ""
    assert table.upper_limit.dtype == bool


@pytest.mark.parametrize("text, words", [
    ("# made by run.py\n" + MVT_HEADER + MVT_PRE, "# line"),
    (MVT_HEADER.replace(",", ", ") + MVT_PRE, "spaces"),
    (MVT_HEADER + MVT_PRE + MVT_PRE, "more than once"),
    (MVT_HEADER + MVT_PRE.replace(",false,", ",,"), "true or false"),
    (MVT_HEADER + MVT_LIMIT.replace(",,,8-900", ",0.001,0.001,8-900"), "empty errors"),
    (MVT_HEADER + MVT_PRE.replace("0.00683", ""), "both errors"),
    (MVT_HEADER + MVT_PRE.replace("0.00683", "-0.00683"), "negative"),
    (MVT_HEADER + MVT_PRE.replace("8-900", "8-1000"), "own file"),
    (MVT_HEADER + MVT_PRE.replace("0.00683", "..."), "isn't a number"),
    (MVT_HEADER.replace("mvt_s,", "mvt_ms,") + MVT_PRE, "no column"),
])
def test_a_table_that_breaks_the_format_is_refused(tmp_path, text, words):
    with pytest.raises(joint.TableError, match=words):
        joint.read_mvt_table(mvt_file(tmp_path, text))


def test_a_column_name_in_both_tables_is_refused(tmp_path):
    table = joint.read_mvt_table(mvt_file(tmp_path, MVT_HEADER.replace("mvt_commit", "commit") + MVT_PRE))
    with pytest.raises(joint.TableError, match="commit"):
        joint.join_tables(joint.read_epeak_table(joint.SHARED / "epeak_230307A.csv"), table)


def test_an_mvt_written_in_ms_is_refused_because_it_is_longer_than_its_window(tmp_path):
    table = joint.read_mvt_table(mvt_file(tmp_path, MVT_HEADER + MVT_PRE.replace("0.01522", "15.22")))
    with pytest.raises(joint.TableError, match="longer than its window"):
        joint.join_tables(joint.read_epeak_table(joint.SHARED / "epeak_230307A.csv"), table)


def test_what_does_not_join_is_listed(tmp_path):
    other = "bala_G,0.0022,,,8-900,true,UL,90,80,91,n0,3e-06,20,untested,,,run_window_mvt.py,abc1234\n"
    no_value = "m11,,,,8-900,true,UL,,,,n0,3e-06,0,untested,,,run_window_mvt.py,abc1234\n"
    joined, left_out = joint.join_tables(joint.read_epeak_table(joint.SHARED / "epeak_230307A.csv"), joint.read_mvt_table(mvt_file(tmp_path, MVT_HEADER + MVT_PRE + other + no_value)))
    assert list(joined.index) == ["pre", "m11"] and left_out["no_epeak"] == ["bala_G"] and left_out["no_value"] == ["m11"] and "m02" in left_out["no_mvt"]


def test_a_value_whose_check_has_not_run_is_marked_provisional_and_a_limit_is_not(tmp_path):
    joined, left_out = joint.join_tables(joint.read_epeak_table(joint.SHARED / "epeak_230307A.csv"), joint.read_mvt_table(mvt_file(tmp_path, MVT_HEADER + MVT_PRE + MVT_LIMIT + MVT_NA_ALONE)))
    assert left_out["provisional"] == ["pre"] and not joined.loc["m01", "provisional"] and not joined.loc["m08", "provisional"]


def test_the_axes_of_the_figure_hold_both_bursts(tmp_path):
    first = joint.read_epeak_table(joint.SHARED / "epeak_230307A.csv")
    second = joint.read_epeak_table(joint.SHARED / "epeak_211211A.csv")
    limit = "f2_01_10,0.002,,,8-900,true,UL,90,80,91,n1 n2 na,3e-06,20,untested,,,run_window_mvt.py,abc1234\n"
    faint = "f2_27_66,0.060,0.005,0.005,8-900,false,robust,90,80,91,n1 n2 na,3e-06,20,true,,,run_window_mvt.py,abc1234\n"
    joined, _ = joint.join_tables(second, joint.read_mvt_table(mvt_file(tmp_path, MVT_HEADER + limit + faint)))
    (x_lo, x_hi), (y_lo, y_hi) = joint.epeak_against_mvt([("GRB230307A", first, None, None), ("GRB211211A", second, joined, None)], tmp_path / "figure.png")
    assert x_lo < 1.5 and x_hi > 66 and y_lo < 29 and y_hi > 1380
    assert (tmp_path / "figure.png").exists()


@pytest.mark.skipif(not (Path(os.environ.get("BALA_CODE", Path.home() / "bala_GBM_MVT_paper" / "MVTfermi")) / "clasify_mvt_value.py").exists(), reason="Bala et al.'s code isn't there")
def test_snr_and_class_of_the_precursor_come_back_at_his_value(burst):
    """His document of 4 Nov gives 121.6, 111.1 and robust at 15.22 ms, with his six detectors and Bala et al.'s background intervals."""
    detectors, window, tau_ms = ["n0", "n1", "n6", "n7", "n9", "na"], (-0.064, 0.320), 15.22
    assert background_intervals(burst) == BALA_BACKGROUND[burst.trigger]
    events = event_times(burst, detectors, ENERGY, time_range=(-1.0, 1.0))
    fits = {d: fit_background(burst.data_dir, burst.trigger, d, background_intervals(burst), 1)[0] for d in detectors}
    rates = {d: background_rate(fits[d], window) for d in detectors}
    assert snr_mvt(events, rates, detectors, window, tau_ms / 1e3) == pytest.approx(121.6, abs=0.05)
    label, snr, _ = bala_class(validation_curve(), tau_ms, merged_times(burst, detectors, window), sum(background_rate(fits[d], window, band=None) for d in detectors), window)
    assert snr == pytest.approx(111.1, abs=0.05) and label == "robust"
    # A window inside one bin of the background fit has to run too, and a straight line changes little over it
    assert background_rate(fits["na"], (0.1, 0.5)) == pytest.approx(rates["na"], rel=1e-3)
