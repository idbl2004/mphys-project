"""Feature tables: each burst's measured numbers read back into one row, set against Zhu et al.'s, and EPI from our numbers with its
uncertainty.

collect(mode) reads every results/zhu/<burst>/ folder (see run.py for the files) into one row per burst for one episode mode: the
episodes, the spread of each number over the burst's simulated light curves, the numbers at half and twice their Tres, and the lags and
spectra measured on that mode's own windows (their way and our rule for zhu2026 and zhu2025, our rule for project). A lag or spectrum is
only read if its window is exactly the mode's window. against_theirs() puts their nine numbers beside ours. epi_samples() draws EPI from
our measurement: the precursor's times move together as in one simulated light curve, and E_p,ME is drawn from its profile-likelihood
interval. write_table() writes the final table with a quality status on every number (quality.py).
"""
import json

import numpy as np
import pandas as pd

from . import RESULTS
from .catalogue import EPI_LINE, epi
from .io import load_light_curve
from .quality import epeak_status, lag_status, timing_status

THEIR_NUMBERS = ["T100_PE", "T100_QE1", "TMVT_PE", "T100_ME", "TMVT_ME", "T100_WE", "Ep_ME", "tau32_ME_ms", "tau32_WE_ms"]
TIMES = ["T100_PE", "T100_QE1", "TMVT_PE", "T100_ME", "TMVT_ME", "T100_WE", "T100_EE", "T100_QE2"]


def _percentiles(values):
    values = pd.Series(values).dropna()
    if values.empty:
        return np.nan, np.nan
    return tuple(np.percentile(values, [15.87, 84.13]))


def spectra_file(folder, mode):
    """Each mode's fits are in its own file, on that mode's windows (run.stage_spectra)."""
    return folder / f"spectra_{mode}.csv"


def _same_window(row, span):
    return span is not None and abs(row["t_start"] - span[0]) < 1e-6 and abs(row["t_stop"] - span[1]) < 1e-6


def way_of(mode):
    """Which E_peak goes with an episode mode: their way for their modes, our rule for ours."""
    return "ours" if mode == "project" else "zhu"


def _usable_error(value, error, side):
    """An error that can be used in log10: positive and finite, and on the minus side smaller than the value (a minus error as large as
    the value says the lower side is open, and is treated as missing)."""
    if error is None or not np.isfinite(error) or error <= 0 or (side == "lo" and error >= value):
        return np.nan
    return float(error)


def draw_two_piece(value, lo, hi, n, rng):
    """Draws of a positive quantity with uneven minus and plus errors, as a two-piece normal in log10 (so draws stay positive): widths
    log10(value / (value - lo)) below and log10((value + hi) / value) above, joined at the value. If one error is missing the other is
    used for both sides (flag ep_error_one_sided); if both are, there are no draws and None is returned."""
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    lo, hi = _usable_error(value, lo, "lo"), _usable_error(value, hi, "hi")
    if not (np.isfinite(lo) or np.isfinite(hi)):
        return None
    lo, hi = (lo if np.isfinite(lo) else hi), (hi if np.isfinite(hi) else lo)
    s_lo = np.log10(value / (value - lo)) if lo < value else np.log10((value + hi) / value)
    s_hi = np.log10((value + hi) / value)
    above = rng.random(n) < s_hi / (s_lo + s_hi)
    width = np.abs(rng.normal(size=n))
    return np.log10(value) + np.where(above, width * s_hi, -width * s_lo)


def load_burst(folder, mode="zhu2026"):
    """Everything measured for one burst under one episode mode, as one flat dict (missing stages give NaN)."""
    record = json.loads((folder / "episodes.json").read_text())
    found = record["modes"][mode]
    row = {k: record.get(k) for k in ("burst", "trigger", "reason", "Tres", "nbins", "nblocks", "data_gaps")}
    row["name"] = record["burst"]
    row["mode"] = mode
    number = lambda v: np.nan if v is None else float(v)
    row.update({k: number(found.get(k)) for k in TIMES + ["TMVT_PE_lo", "TMVT_PE_hi", "TMVT_ME_lo", "TMVT_ME_hi", "mvt_pe_bins", "mvt_me_bins",
                                                         "qe1_snr_bin", "qe1_dip", "qe1_recovery", "qe2_dip", "qe2_recovery", "pe_peak_rate", "me_peak_rate"]})
    row["flags"] = " ".join(found.get("flags", []) + record.get("bkg_flags", []) + record.get("setup_flags", []))
    s = json.loads((folder / "state.json").read_text()) if (folder / "state.json").exists() else {}
    row["status"], row["status_reason"] = s.get("status", "unknown"), s.get("reason", "")
    row["reasons"] = " | ".join(found.get("reasons", []))
    row["bad_intervals"] = len(record.get("bad_intervals", []))
    for key in ("pe", "qe1", "me", "qe2", "ee", "we"):
        span = found.get(key)
        row[f"{key}_start"], row[f"{key}_stop"] = (span if span else (np.nan, np.nan))
    for reading, value in record.get("mvt_readings", {}).get(mode, {}).get("pe", {}).items():
        row[f"TMVT_PE_{reading}"] = value
    for reading, value in record.get("mvt_readings", {}).get(mode, {}).get("me", {}).items():
        row[f"TMVT_ME_{reading}"] = value
    path = folder / "realisations.csv"
    if path.exists():
        real = pd.read_csv(path)
        real = real[real["mode"] == mode]
        row["realisations"] = len(real)
        row["pe_found_fraction"] = real.has_pe.mean() if len(real) else np.nan
        row["ee_found_fraction"] = real.has_ee.mean() if len(real) else np.nan
        for q in TIMES:
            row[f"{q}_p16"], row[f"{q}_p84"] = _percentiles(real[q]) if len(real) else (np.nan, np.nan)
    path = folder / "resolutions.csv"
    if path.exists():
        res = pd.read_csv(path)
        res = res[res.get("mode", pd.Series(dtype=str)) == mode] if "mode" in res else res.iloc[0:0]
        for _, r in res.iterrows():
            tag = {0.5: "half", 2.0: "double"}.get(float(r.factor), f"x{r.factor}")
            for q in TIMES:
                row[f"{q}_{tag}"] = r.get(q, np.nan)
    path = folder / "lags.csv"
    if path.exists():
        lags = pd.read_csv(path)
        lags = lags[lags["mode"] == mode] if "mode" in lags else lags.iloc[0:0]
        for _, r in lags.iterrows():
            key = f"{r.lag}_{r.label}"
            if not _same_window(r, found.get(r.label.lower())):
                row[f"{key}_status"] = "no_lag_for_this_window"
                continue
            row.update({f"{key}_ms": r.get("lag_ms", np.nan), f"{key}_down": r.get("err_down_ms", np.nan), f"{key}_up": r.get("err_up_ms", np.nan),
                        f"{key}_cut_ms": r.get("lag_cut_ms", np.nan), f"{key}_cut_down": r.get("cut_down_ms", np.nan), f"{key}_cut_up": r.get("cut_up_ms", np.nan),
                        f"{key}_res_ms": r.get("res_ms", np.nan), f"{key}_status": r.get("status", ""),
                        f"{key}_cut_status": r.get("cut_status", "")})
    path = spectra_file(folder, mode)
    if path.exists():
        spectra = pd.read_csv(path, index_col=0)
        for label, r in spectra.iterrows():
            span = found.get(str(label).lower())
            if span is None:
                continue
            if not _same_window(r, span):
                row["flags"] += f" no_spectrum_for_this_window:{label}"
                continue
            row.update({f"Ep_{label}_zhu": r.get("zhu_Epeak", np.nan), f"Ep_{label}_zhu_lo": r.get("zhu_Epeak_err_lo", np.nan),
                        f"Ep_{label}_zhu_hi": r.get("zhu_Epeak_err_hi", np.nan), f"model_{label}_zhu": r.get("zhu_model", None),
                        f"zhu_flags_{label}": r.get("zhu_flags", None)})
            row.update({f"Ep_{label}_ours": r.get("Epeak", np.nan), f"Ep_{label}_ours_lo": r.get("Epeak_err_lo", np.nan),
                        f"Ep_{label}_ours_hi": r.get("Epeak_err_hi", np.nan), f"model_{label}_ours": r.get("model", None),
                        f"measured_{label}": r.get("measured", np.nan), f"sigma_{label}": r.get("sigma", np.nan),
                        f"Ep_{label}_cpl_limit_lo": r.get("Epeak_cpl_limit_lo", np.nan), f"Ep_{label}_cpl_limit_hi": r.get("Epeak_cpl_limit_hi", np.nan),
                        f"flux_{label}": r.get("flux", np.nan), f"fluence_{label}": r.get("fluence", np.nan), f"fit_flags_{label}": r.get("fit_flags", None)})
            for way in ("zhu", "ours"):
                value = float(row.get(f"Ep_{label}_{way}", np.nan) or np.nan)
                errors = [np.isfinite(_usable_error(value, float(row.get(f"Ep_{label}_{way}_{side}", np.nan) or np.nan), side)) for side in ("lo", "hi")]
                if np.isfinite(value) and sum(errors) == 1:
                    row["flags"] += f" ep_error_one_sided:{label}_{way}"
    row["flags"] = row["flags"].strip()
    return row


def _measured(folder):
    path = folder / "episodes.json"
    return path.exists() and "modes" in json.loads(path.read_text())


def collect(mode="zhu2026", root=RESULTS, names=None):
    """One row per measured burst under one episode mode. With names (e.g. the whole sample), every one of them gets a row: a burst that
    wasn't measured has only its status and the reason (from state.json), so nothing drops out of a table without a line saying why."""
    folders = [root / n for n in names] if names is not None else sorted(f for f in root.iterdir() if f.is_dir())
    rows = []
    for folder in folders:
        if _measured(folder):
            rows.append(load_burst(folder, mode))
        elif names is not None:
            s = json.loads((folder / "state.json").read_text()) if (folder / "state.json").exists() else {}
            rows.append(dict(name=folder.name, burst=folder.name, mode=mode, status=s.get("status", "not_started"), status_reason=s.get("reason", ""), flags=""))
    return pd.DataFrame(rows).set_index("name")


def against_theirs(ours, table):
    """Each of their nine numbers beside ours. Times and E_p are compared in dex (log10 ours/theirs), lags in ms (ours minus theirs)."""
    both = ours.join(table[THEIR_NUMBERS + ["Type", "EPI"]].add_prefix("their_"), how="left")
    out = pd.DataFrame(index=both.index)
    out["Type"] = both.their_Type
    pairs = {q: q for q in THEIR_NUMBERS[:6]}
    pairs.update({"Ep_ME": "Ep_ME_zhu", "tau32_ME_ms": "tau32_ME_cut_ms", "tau32_WE_ms": "tau32_WE_cut_ms"})
    for q, mine in pairs.items():
        theirs = both[f"their_{q}"]
        value = both.get(mine, pd.Series(np.nan, index=both.index))
        out[f"{q}_theirs"], out[f"{q}_ours"] = theirs, value
        out[f"{q}_diff"] = value - theirs if q.startswith("tau") else np.log10(value / theirs)
    return out


def epi_samples(row, folder, n=4000, way="zhu", seed=11):
    """EPI drawn from our measurement of one burst. way='zhu' uses E_p,ME from their model choice, 'ours' from our rule.

    The durations and MVT of the precursor come from the burst's own light curve, moved by the deviations one simulated realisation shows
    from the median of all of them (in log10), so the three move together. E_p,ME is drawn from a two-piece normal with its profile errors.
    Returns the samples (empty if the precursor or E_p is missing)."""
    base = np.array([row.get("T100_PE", np.nan), row.get("T100_QE1", np.nan), row.get("TMVT_PE", np.nan)], float)
    if not np.all(np.isfinite(base) & (base > 0)):
        return np.array([])
    rng = np.random.default_rng(seed)
    log_ep = draw_two_piece(float(row.get(f"Ep_ME_{way}", np.nan) or np.nan), row.get(f"Ep_ME_{way}_lo", np.nan), row.get(f"Ep_ME_{way}_hi", np.nan), n, rng)
    if log_ep is None:
        return np.array([])
    picks = _timing_deviations(folder, row.get("mode", "zhu2026"), ["T100_PE", "T100_QE1", "TMVT_PE"], n, rng)
    if picks is None:
        return np.array([])
    times = 10 ** (np.log10(base) + picks)
    return epi(10 ** log_ep, times[:, 0], times[:, 1], times[:, 2])


MIN_COMPLETE = 5            # simulated light curves that must have all the numbers asked for before their spread is used


def _timing_deviations(folder, mode, columns, n, rng):
    """n rows of log10 deviations of the given numbers from their median over the burst's simulated light curves, each row from one
    simulated light curve in which every one of the numbers was found (so they move together). None if fewer than MIN_COMPLETE such
    light curves exist: the spread isn't known, and no draws are made rather than draws with no spread."""
    path = folder / "realisations.csv"
    if not path.exists():
        return None
    real = pd.read_csv(path)
    real = real[real["mode"] == mode][columns]
    real = real[(real > 0).all(axis=1)].dropna()
    if len(real) < MIN_COMPLETE:
        return None
    logs = np.log10(real.values)
    return (logs - np.median(logs, axis=0))[rng.integers(0, len(logs), n)]


def epi_summary(ours, root=RESULTS, way="zhu"):
    """EPI from our numbers for every burst: the value from the measurement itself, the 16th and 84th percentiles of the draws, and the
    chance of lying above their line."""
    rows = []
    for name, row in ours.iterrows():
        value = epi(row.get(f"Ep_ME_{way}", np.nan), row.T100_PE, row.T100_QE1, row.TMVT_PE) if np.isfinite(row.get("T100_PE", np.nan) or np.nan) else np.nan
        samples = epi_samples(row, root / name, way=way)
        if samples.size:
            p16, p84 = np.percentile(samples, [15.87, 84.13])
            above = float(np.mean(samples > EPI_LINE))
        else:
            p16 = p84 = above = np.nan
        rows.append(dict(GRB=name, EPI=value, EPI_p16=p16, EPI_p84=p84, P_above=above))
    return pd.DataFrame(rows).set_index("GRB").add_suffix(f"_{way}")


def input_draws(row, folder, n=2000, seed=13):
    """Draws of a burst's numbers within their errors, as log10 values named as in laws.NAMES. The timing numbers move together: each
    draw takes the deviations from the median that one simulated light curve shows, using only light curves in which every timing number
    the burst has was found (a light curve that lost the precursor says nothing about the precursor's spread). The E_peaks are drawn from
    two-piece normals in log10 with their profile errors (draw_two_piece); E_p,ME is their way for their modes and our rule for the
    project mode, and E_p,PE is drawn only where our rule counts it as measured. A number that can't be drawn (missing, no error, or too
    few complete light curves) is left out, and a law that needs it skips the burst. Returns a DataFrame with n rows."""
    rng = np.random.default_rng(seed)
    mode = row.get("mode", "zhu2026")
    timing = {"T100_PE": "T_PE", "T100_QE1": "T_QE1", "TMVT_PE": "MVT_PE", "T100_ME": "T_ME", "TMVT_ME": "MVT_ME", "T100_WE": "T_WE"}
    present = [c for c in timing if np.isfinite(float(row.get(c, np.nan) or np.nan)) and row.get(c) > 0]
    out = {}
    picks = _timing_deviations(folder, mode, present, n, rng) if present else None
    if picks is not None:
        for i, long in enumerate(present):
            out[timing[long]] = np.log10(row[long]) + picks[:, i]
    for column, short in ((f"Ep_ME_{way_of(mode)}", "Ep_ME"), ("Ep_PE_ours", "Ep_PE")):
        if short == "Ep_PE" and str(row.get("measured_PE", "")) != "True":
            continue
        draws = draw_two_piece(float(row.get(column, np.nan) or np.nan), row.get(f"{column}_lo", np.nan), row.get(f"{column}_hi", np.nan), n, rng)
        if draws is not None:
            out[short] = draws
    return pd.DataFrame(out)


TABLE_COLUMNS = {
    "trigger": "trigger", "reason": "sample_group", "status": "status", "status_reason": "status_reason", "reasons": "episode_reasons", "Tres": "Tres_s",
    "pe_start": "PE_start_s", "pe_stop": "PE_stop_s", "me_start": "ME_start_s", "me_stop": "ME_stop_s", "ee_start": "EE_start_s", "ee_stop": "EE_stop_s",
    "T100_PE": "T100_PE_s", "T100_PE_p16": "T100_PE_p16", "T100_PE_p84": "T100_PE_p84",
    "T100_QE1": "T100_QE1_s", "T100_QE1_p16": "T100_QE1_p16", "T100_QE1_p84": "T100_QE1_p84",
    "T100_ME": "T100_ME_s", "T100_ME_p16": "T100_ME_p16", "T100_ME_p84": "T100_ME_p84",
    "T100_WE": "T100_WE_s", "T100_WE_p16": "T100_WE_p16", "T100_WE_p84": "T100_WE_p84",
    "TMVT_PE": "TMVT_PE_s", "TMVT_PE_p16": "TMVT_PE_p16", "TMVT_PE_p84": "TMVT_PE_p84",
    "TMVT_ME": "TMVT_ME_s", "TMVT_ME_p16": "TMVT_ME_p16", "TMVT_ME_p84": "TMVT_ME_p84",
    "Ep_ME_zhu": "Ep_ME_their_way_keV", "Ep_ME_zhu_lo": "Ep_ME_their_way_minus", "Ep_ME_zhu_hi": "Ep_ME_their_way_plus", "model_ME_zhu": "model_ME_their_way",
    "Ep_ME_ours": "Ep_ME_our_rule_keV", "Ep_ME_ours_lo": "Ep_ME_our_rule_minus", "Ep_ME_ours_hi": "Ep_ME_our_rule_plus", "model_ME_ours": "model_ME_our_rule",
    "measured_PE": "PE_spectrum_measured", "Ep_PE_ours": "Ep_PE_our_rule_keV", "Ep_PE_ours_lo": "Ep_PE_our_rule_minus", "Ep_PE_ours_hi": "Ep_PE_our_rule_plus",
    "Ep_PE_cpl_limit_lo": "Ep_PE_95_low_keV", "Ep_PE_cpl_limit_hi": "Ep_PE_95_high_keV", "Ep_PE_zhu": "Ep_PE_their_way_keV",
    "tau32_ME_ms": "tau32_ME_ms", "tau32_ME_down": "tau32_ME_minus", "tau32_ME_up": "tau32_ME_plus", "tau32_ME_cut_ms": "tau32_ME_cut_ms",
    "tau32_WE_ms": "tau32_WE_ms", "tau32_WE_cut_ms": "tau32_WE_cut_ms", "tau32_PE_ms": "tau32_PE_ms", "tau32_PE_down": "tau32_PE_minus", "tau32_PE_up": "tau32_PE_plus",
    "EPI_zhu": "EPI", "EPI_p16_zhu": "EPI_p16", "EPI_p84_zhu": "EPI_p84", "P_above_zhu": "P_EPI_above_6.2", "EPI_ours": "EPI_our_rule_Ep",
    "flags": "flags",
}


def write_table(ours, path):
    """The measurements as one table, one row per burst, with plain column names and units, and a header saying what each part is."""
    out = ours[[c for c in TABLE_COLUMNS if c in ours.columns]].rename(columns=TABLE_COLUMNS)
    for q in ("T100_PE", "T100_QE1", "TMVT_PE", "T100_ME", "TMVT_ME", "T100_WE"):
        out[f"{q}_status"] = [timing_status(r, q) for _, r in ours.iterrows()]
    out["Ep_ME_their_way_status"] = [epeak_status(r, "ME", "zhu") for _, r in ours.iterrows()]
    out["Ep_PE_our_rule_status"] = [epeak_status(r, "PE", "ours") for _, r in ours.iterrows()]
    out["tau32_ME_status"] = [lag_status(r, "tau32_ME") for _, r in ours.iterrows()]
    out["tau32_ME_cut_status"] = [lag_status(r, "tau32_ME", "cut") for _, r in ours.iterrows()]
    out["tau32_PE_status"] = [lag_status(r, "tau32_PE") for _, r in ours.iterrows()]
    out["episode_mode"] = ours["mode"]
    unfinished = ours["status"].ne("complete")              # a burst still running, failed or not measurable: its numbers aren't final
    for column in [c for c in out.columns if c.endswith("_status") and c != "status"]:
        out.loc[unfinished, column] = ours.loc[unfinished, "status"]
    header = ("# Zhu et al.'s bursts measured again (ignacio/zhu, notebooks 23-25). Times in s from the trigger; _p16/_p84 are the 16th and 84th\n"
              "# percentiles over 40 light curves simulated from each burst's blocks. E_peak in keV with profile-likelihood errors, their way (their\n"
              "# models, BIC, PGSTAT, their energy ranges) and by our rule (notebook 14). Ep_PE_95_low/high: the 95% interval of the Comptonized\n"
              "# E_peak where the precursor's E_peak isn't measured. Lags in ms, soft behind hard; _cut_ms with both bands cut at the window.\n"
              "# EPI from E_p,ME their way; p16/p84 and P_EPI_above_6.2 from 4000 draws (features.epi_samples). *_status: see zhu/quality.py.\n")
    with open(path, "w") as f:
        f.write(header)
        out.to_csv(f, float_format="%.6g")
    return out


def detector_test(names, table, out_path):
    """The rule with the catalogue NaIs and with every NaI within 60 degrees of the burst (the choice of Xiao et al. 2023), for a few
    bursts. It reads the photons again, so it takes a few minutes per burst, and the table is saved to out_path."""
    from grbtool.bursts import DATA_ROOT, _angle, set_up
    from .catalogue import data_name
    from .background import clear_background, light_curve
    from .episodes import blocks_of, find_episodes
    rows = []
    for name in names:
        row = table.loc[name]
        burst = set_up(data_name(name, DATA_ROOT), row.trigger)
        clear_background(burst)
        angles = {d: _angle(burst.data_dir, burst.trigger, d) for d in [f"n{i:x}" for i in range(12)]}
        near = sorted([d for d, a in angles.items() if a is not None and a < 60], key=lambda d: angles[d])
        for label, detectors in (("catalogue", burst.nai), ("within 60 deg", near)):
            lc = light_curve(burst, row.Tres, detectors=detectors)
            found = find_episodes(lc, blocks_of(lc))
            d = found.durations()
            rows.append(dict(GRB=name, detectors=label, which=" ".join(detectors), **{k: d[k] for k in ("T100_PE", "T100_QE1", "TMVT_PE", "T100_ME", "TMVT_ME", "T100_WE")}))
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def boundary_cause(name, row, table, root=RESULTS):
    """Why our precursor or gap differs from theirs, sorted from what the rule recorded and the saved blocks: no precursor, other gap
    (their gap matches another of our long quiet dips within 20%), same gap other start, edge, or other."""
    import re
    flags = str(row.get("flags", ""))
    if not np.isfinite(row.get("T100_PE", np.nan)):
        miss = re.search(r"nearest_miss_qe1_snr:([\d.]+)", flags)
        return "no precursor" + (f" (closest dip {miss.group(1)} sigma per bin)" if miss else "")
    theirs_qe1 = table.T100_QE1[name]
    if abs(np.log10(row["T100_QE1"] / theirs_qe1)) < np.log10(1.2):
        return "same gap, other start"
    _, blocks = load_light_curve(root / name)
    long_dips = blocks[(blocks.nbins >= 8) & (blocks.snr_bin < 3) & (blocks.start < row.get("me_start", np.inf))]
    if np.any(np.abs(np.log10((long_dips.stop - long_dips.start) / theirs_qe1)) < np.log10(1.2)):
        return "other gap"
    if "excess_at_start_ignored" in flags or "emission_reaches_background" in flags:
        return "edge"
    return "other"


def paper_vs_ours(ours, table, root=RESULTS):
    """One row per burst and number: their value, ours (zhu2026 episodes, E_p their way, lags cut at the window), the difference (dex for
    times and E_p, ms for lags), our 68% interval, whether they agree (theirs inside our interval, or within 0.05 dex / 1 sigma of a lag),
    our number's status, and the known reason when they don't."""
    rows = []
    for name, r in ours.iterrows():
        cause = None
        for q in ("T100_PE", "T100_QE1", "TMVT_PE", "T100_ME", "TMVT_ME", "T100_WE", "Ep_ME", "tau32_ME_ms", "tau32_WE_ms", "EPI"):
            theirs = table.loc[name, q]
            if q == "Ep_ME":
                value, lo, hi = r.get("Ep_ME_zhu", np.nan), r.get("Ep_ME_zhu", np.nan) - r.get("Ep_ME_zhu_lo", np.nan), r.get("Ep_ME_zhu", np.nan) + r.get("Ep_ME_zhu_hi", np.nan)
                status = epeak_status(r, "ME", "zhu")
            elif q.startswith("tau"):
                key = q.replace("_ms", "")
                value = r.get(f"{key}_cut_ms", np.nan)
                lo, hi = value - r.get(f"{key}_cut_down", np.nan), value + r.get(f"{key}_cut_up", np.nan)
                status = lag_status(r, key, "cut")
            elif q == "EPI":
                value, lo, hi = r.get("EPI_zhu", np.nan), r.get("EPI_p16_zhu", np.nan), r.get("EPI_p84_zhu", np.nan)
                status = "measured" if np.isfinite(value) else "missing"
            else:
                value, lo, hi = r.get(q, np.nan), r.get(f"{q}_p16", np.nan), r.get(f"{q}_p84", np.nan)
                status = timing_status(r, q)
            linear = q.startswith("tau") or q == "EPI"
            diff = value - theirs if linear else (np.log10(value / theirs) if np.isfinite(value) and value > 0 and theirs > 0 else np.nan)
            agrees = bool(np.isfinite(value) and ((lo <= theirs <= hi) or (not linear and abs(diff) < 0.05)))
            reason = ""
            if not agrees and np.isfinite(value):
                if q in ("T100_PE", "T100_QE1", "TMVT_PE", "EPI"):
                    cause = cause or boundary_cause(name, r, table, root)
                    reason = cause if cause != "same gap, other start" or q != "TMVT_PE" else "MVT reading (their pulse choice)"
                    if q == "TMVT_PE" and cause in ("same gap, other start",):
                        reason = "MVT reading (their pulse choice)"
                elif q == "TMVT_ME":
                    reason = "MVT reading (their pulse choice)"
                elif q in ("T100_ME", "T100_WE"):
                    reason = "extended-emission split or faint tail" + (" (tail runs into the background interval)" if "emission_reaches_background" in str(r.get("flags", "")) else "")
                elif q == "Ep_ME":
                    reason = f"model or window (ours: {r.get('model_ME_zhu', '')})"
                else:
                    reason = "lag resolution or window"
            elif not np.isfinite(value):
                reason = status
            rows.append(dict(burst=name, number=q, paper=theirs, ours=value, difference=diff, ours_low=lo, ours_high=hi, agrees=agrees,
                             our_status=status, reason=reason))
    return pd.DataFrame(rows)
