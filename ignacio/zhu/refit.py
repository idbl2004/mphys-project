"""Fits run again on chosen windows of a burst that is already measured, with the burst set up exactly as the batch set it up (the
same detectors, background and telemetry cuts).

This covers three things. First, the windows whose fits ran out of time in the batch (late_windows, run_late). To measure 220 bursts in
days, the batch gave each slow step of a fit 5 minutes (run.SPECTRUM_MINUTES) and Zhu et al.'s error search 10. A fit that runs out keeps
its best start so far and says so, in ran_out (our rule) or zhu_flags (their way). Here those fits get 30 minutes each, the default of
python -m grbtool, and the new E_peak goes next to the batch's. Second, any list of windows at all, like a paper's own time slices
(fit_windows), optionally all with the response of one time instead of each window's own. Third, catalogue_peak repeats the GBM
catalogue's own fit of each burst's peak-flux second, to check my data handling against the GBM team's where both sides use the same
response files.

From ignacio/:
    python -m zhu.refit late                          the windows whose fits ran out and could change an E_peak the notebooks use
    python -m zhu.refit late --run --workers 3        refit them, into results/time_limit_check/refits.csv (notebook 33)
    python -m zhu.refit catalogue --workers 3         the catalogue's peak-flux fits repeated, into results/catalogue_check/peak_seconds.csv"""
import argparse
import json
import multiprocessing
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import CODE_DIR

SAMPLES = {"zhu": (CODE_DIR / "results" / "zhu", CODE_DIR / "shared" / "zhu_sample.csv"),
           "coppin": (CODE_DIR / "results" / "coppin", CODE_DIR / "shared" / "coppin_sample.csv")}
OUT = CODE_DIR / "results" / "time_limit_check"
LONG_MINUTES = 30.0
MODES = ("zhu2026", "project")          # the modes whose E_peaks the notebooks use
KEYS = {"Comptonized": "cpl", "Band": "band", "BandLowBreak": "two_break", "PowerLaw": "pl"}


def could_matter(r, late):
    """Whether something that ran out of time in our rule's fits could change the E_peak our rule gives. That's the case if the chosen model
    or its error search ran out, or if a model that ran out was close enough to the threshold that would have chosen it (two-break: Band's C
    minus its C between 5 and 11.8; Band against a chosen Comptonized: between 4 and 9). A model further from its threshold, or past it but
    turned down for another reason (a break that isn't clean), can't change the choice."""
    chosen = KEYS.get(r.get("model"), "")
    if chosen in late or "errors" in late:
        return True
    if "two_break" in late and chosen != "two_break" and 5.0 < r["C_band"] - r["C_lowbreak"] <= 11.8:
        return True
    return "band" in late and chosen == "cpl" and 4.0 < r["C_cpl"] - r["C_band"] <= 9.0


def late_windows(samples=SAMPLES, modes=MODES):
    """One row per window (sample, burst, label, edges) in which something ran out that could matter, with the batch's values, the modes
    that use the window, which way needs refitting, and why: A the chosen model ran out, B its error search, C a model that could have
    been chosen, Z something in their way."""
    rows = {}
    for sample, (root, table) in samples.items():
        triggers = pd.read_csv(table, comment="#", index_col=0).trigger
        for name in triggers.index:
            for mode in modes:
                path = root / str(name) / f"spectra_{mode}.csv"
                if not path.exists():
                    continue
                for _, r in pd.read_csv(path).iterrows():
                    if r["label"] not in ("PE", "ME") or not np.isfinite(r.get("t_start", np.nan)):
                        continue
                    ours_late = str(r["ran_out"]).split() if pd.notna(r.get("ran_out")) else []
                    zhu_late = [f for f in str(r["zhu_flags"]).split() if "out_of_time" in f] if pd.notna(r.get("zhu_flags")) else []
                    ours_late = ours_late if could_matter(r, ours_late) else []
                    if not ours_late and not zhu_late:
                        continue
                    key = (sample, str(name), r["label"], round(float(r["t_start"]), 6), round(float(r["t_stop"]), 6))
                    row = rows.setdefault(key, dict(sample=sample, burst=str(name), trigger=triggers[name], label=r["label"],
                                                    t_start=float(r["t_start"]), t_stop=float(r["t_stop"]), modes=[], ours=False, zhu=False))
                    row["modes"].append(mode)
                    chosen = KEYS.get(r.get("model"), "")
                    why = ("A" if chosen in ours_late else "B" if "errors" in ours_late else "C") if ours_late else "Z"
                    row["why"] = min(row.get("why", "Z"), why)
                    if ours_late:
                        row["ours"] = True
                        row.update(old_model=r.get("model"), old_Epeak=r.get("Epeak"), old_lo=r.get("Epeak_err_lo"), old_hi=r.get("Epeak_err_hi"),
                                   old_ran_out=" ".join(ours_late))
                    if zhu_late or "zhu_Epeak" in r and pd.notna(r.get("zhu_Epeak")):
                        row["zhu"] = row["zhu"] or bool(zhu_late)
                        row.update(old_zhu_model=r.get("zhu_model"), old_zhu_Epeak=r.get("zhu_Epeak"), old_zhu_lo=r.get("zhu_Epeak_err_lo"),
                                   old_zhu_hi=r.get("zhu_Epeak_err_hi"), old_zhu_late=" ".join(zhu_late))
    table = pd.DataFrame(rows.values())
    table["modes"] = table.modes.map(lambda m: " ".join(sorted(set(m))))
    return table


def set_up_as_batch(name, trigger, root):
    """The burst as the batch set it up, checked against the background intervals its episodes were found with."""
    from .background import clear_background
    from .run import start_burst
    record = json.loads((root / name / "episodes.json").read_text())
    burst, _ = start_burst(name, trigger)
    clear_background(burst)
    if not np.allclose(np.ravel(burst.bkg_windows), np.ravel(record["bkg_windows"])):
        raise RuntimeError(f"{name}: the background is now {burst.bkg_windows}, the batch's was {record['bkg_windows']}")
    return burst, record


def _refit_burst(group, minutes=LONG_MINUTES):
    """Every late window of one burst, fitted again with more time."""
    from grbtool.fitting import all_spectra
    from grbtool.windows import Window, cut_out
    from . import spectra as S
    S.ERROR_MINUTES = minutes                   # their error search, read when fit_zhu runs
    sample, name = group["sample"].iloc[0], group["burst"].iloc[0]
    began = time.monotonic()
    try:
        burst, record = set_up_as_batch(name, group["trigger"].iloc[0], SAMPLES[sample][0])
        windows = [Window(f"{r.label}_{i}", cut_out((r.t_start, r.t_stop), record["bad_intervals"]), "precursor" if r.label == "PE" else "main")
                   for i, r in enumerate(group.itertuples())]
        spectra, _ = all_spectra(burst, windows)
    except Exception as error:
        return [dict(sample=sample, burst=name, label=r.label, t_start=r.t_start, t_stop=r.t_stop, failed=f"{type(error).__name__}: {error}"[:300])
                for r in group.itertuples()]
    out = []
    for r, w in zip(group.itertuples(), windows):
        row = dict(sample=sample, burst=name, label=r.label, t_start=r.t_start, t_stop=r.t_stop)
        if r.ours:
            t0 = time.monotonic()
            try:
                f = S.fit_window_ours(burst, w, spectra, minutes=minutes)
                row.update({f"new_{k}": f[k] for k in ("model", "Epeak", "Epeak_err_lo", "Epeak_err_hi", "Epeak_band", "Epeak_cpl", "Epeak_lowbreak",
                                                        "C_band", "C_cpl", "C_lowbreak", "ran_out", "fit_flags", "measured")})
            except Exception as error:
                row["new_failed"] = f"{type(error).__name__}: {error}"[:300]
            row["new_seconds"] = time.monotonic() - t0
        if r.zhu:
            t0 = time.monotonic()
            try:
                z = S.fit_zhu(burst, w, spectra, precursor=r.label == "PE", minutes=minutes)
                row.update({f"new_{k}": z.get(k) for k in ("zhu_model", "zhu_Epeak", "zhu_Epeak_err_lo", "zhu_Epeak_err_hi", "zhu_flags")})
            except Exception as error:
                row["new_zhu_failed"] = f"{type(error).__name__}: {error}"[:300]
            row["new_zhu_seconds"] = time.monotonic() - t0
        out.append(row)
    print(f"{sample} {name}: {len(group)} windows in {time.monotonic() - began:.0f} s", flush=True)
    return out


def run_late(workers=3, out=OUT / "refits.csv"):
    """Refit every late window, the bursts with a late chosen model first; the table is rewritten after each burst, so a stopped run
    keeps what it did."""
    table = late_windows()
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = sorted((g for _, g in table.groupby(["sample", "burst"], sort=False)), key=lambda g: (g.why.min(), -len(g)))
    rows = []
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        for part in pool.imap_unordered(_refit_burst, groups):
            rows += part
            table.merge(pd.DataFrame(rows), on=["sample", "burst", "label", "t_start", "t_stop"], how="left").to_csv(out, index=False)
    return pd.read_csv(out)


@contextmanager
def response_fixed_at(t):
    """Every window gets the response covering time t (s from the trigger) instead of its own: what a fit that ignores the spacecraft's
    rotation since t uses. The workers forked inside inherit it."""
    import grbtool.fitting as F
    own = F.window_response
    F.window_response = lambda data_dir, trigger, det, window, bin_width=0.256: own(data_dir, trigger, det, (t, t + 0.1), bin_width)
    try:
        yield
    finally:
        F.window_response = own


@contextmanager
def responses_from(folder):
    """The response files are read from folder instead of the burst's data folder: an earlier version of them, say, as a paper used.
    The workers forked inside inherit it."""
    from pathlib import Path
    import spectral_fit_tools as T
    own = T.rsp2_path
    T.rsp2_path = lambda data_dir, trigger, det: sorted(Path(folder).glob(f"glg_cspec_{det}_{trigger}_v*.rsp2"))[-1]
    try:
        yield
    finally:
        T.rsp2_path = own


def previous_responses(trigger, version, detectors, folder):
    """A burst's response files of an earlier version, from the archive's previous/ folder (the version a published fit used), into
    folder. Files already there are kept."""
    import subprocess
    from pathlib import Path
    from grbtool.bursts import ARCHIVE
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for det in detectors:
        name = f"glg_cspec_{det}_{trigger}_{version}.rsp2"
        if not (folder / name).exists():
            url = f"{ARCHIVE}/20{trigger[2:4]}/{trigger}/previous/{name}"
            subprocess.run(["curl", "-sSfL", "-m", "300", "-o", str(folder / f"{name}.part"), url], check=True)
            (folder / f"{name}.part").rename(folder / name)
    return folder


def fit_windows(name, trigger, root, windows, response_at=None, responses=None, workers=3, minutes=None):
    """Our rule (the batch's settings) on any windows [(label, start, stop), ...] of a measured burst; with response_at, every window
    gets the response of that time instead of its own; with responses, the response files come from that folder. Returns the table of
    fits, one row per label."""
    from contextlib import nullcontext
    from grbtool.windows import Window, cut_out
    from .run import SPECTRUM_MINUTES
    from .spectra import fit_episodes
    burst, record = set_up_as_batch(name, trigger, root)
    ws = [Window(label, cut_out((a, b), record["bad_intervals"]), "main") for label, a, b in windows]
    with response_fixed_at(response_at) if response_at is not None else nullcontext(), \
            responses_from(responses) if responses is not None else nullcontext():
        return fit_episodes(burst, ws, workers=workers, minutes=minutes or SPECTRUM_MINUTES, ways=("ours",))


def catalogue_peak(name, trigger, root):
    """The GBM catalogue's own Comptonized fit of the burst's peak-flux second (its pflx spectrum: the same interval, detectors and
    channels), repeated with the batch's set-up and grbtool.checks.catalogue_check, with the response versions on each side: where
    both use the same responses this says whether the data handling reproduces the GBM team's."""
    import copy
    from astropy.io import fits
    from grbtool.bursts import _detector_name
    from grbtool.checks import catalogue_check
    from spectral_fit_tools import rsp2_path
    out = dict(burst=name, trigger=trigger)
    try:
        burst, _ = set_up_as_batch(name, trigger, root)
        path = sorted(burst.data_dir.glob(f"glg_scat_all_{trigger}_pflx_comp_v*.fit"))
        if not path:
            return dict(out, note="no peak-flux fit in the catalogue files")
        with fits.open(path[-1]) as h:
            data, p = h["DETECTOR DATA"].data, h["FIT PARAMS"].data[0]
            detectors = [_detector_name(n) for n in data["DETNAM"]]
            theirs = {d: str(f).rsplit("_", 1)[-1].split(".")[0] for d, f in zip(detectors, data["RSPFILE"])}
            channels = {d: tuple(int(c) for c in fit) for d, fit in zip(detectors, data["FITCHAN"])}
            interval = tuple(float(t) for t in p["TIMEBIN"])
            catalogue = dict(interval=interval, Ep=float(p["PARAM1"][0]), Ep_err=0.5 * float(p["PARAM1"][1] + p["PARAM1"][2]),
                             alpha=float(p["PARAM2"][0]), alpha_err=0.5 * float(p["PARAM2"][1] + p["PARAM2"][2]))
        missing = [d for d in detectors if not sorted(burst.data_dir.glob(f"glg_cspec_{d}_{trigger}_v*.rsp2"))]
        if missing:
            return dict(out, detectors=" ".join(detectors), note=f"no response file here for {' '.join(missing)}")
        ours = {d: rsp2_path(burst.data_dir, trigger, d).name.rsplit("_", 1)[-1].split(".")[0] for d in detectors}
        out.update(interval_s=f"{interval[0]:.3f} {interval[1]:.3f}", detectors=" ".join(detectors), their_responses=" ".join(sorted(set(theirs.values()))),
                   our_responses=" ".join(sorted(set(ours.values()))), same_responses=theirs == ours)
        if not (burst.tte_range[0] <= interval[0] and interval[1] <= burst.tte_range[1]):
            return dict(out, note=f"the interval is outside the TTE data ({burst.tte_range[0]:.0f} to {burst.tte_range[1]:.0f} s)")
        peak = copy.copy(burst)
        peak.catalogue, peak.channels, peak.detectors = catalogue, channels, detectors
        out.update(catalogue_check(peak))
    except Exception as error:
        out["note"] = f"{type(error).__name__}: {error}"[:200]
    return out


def _peak_worker(args):
    return catalogue_peak(*args)


def run_catalogue_peaks(workers=3, out=CODE_DIR / "results" / "catalogue_check" / "peak_seconds.csv"):
    """catalogue_peak for every measured burst of both samples; one row each."""
    jobs = []
    for sample, (root, table) in SAMPLES.items():
        for name, trigger in pd.read_csv(table, comment="#", index_col=0).trigger.items():
            if (root / str(name) / "episodes.json").exists():
                jobs.append((str(name), trigger, root))
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for row in pool.imap_unordered(_peak_worker, jobs):
            rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False)
    return pd.read_csv(out)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m zhu.refit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["late", "catalogue"])
    parser.add_argument("--run", action="store_true", help="fit them, not just list them")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    if args.task == "catalogue":
        run_catalogue_peaks(args.workers)
        return
    if not args.run:
        table = late_windows()
        with pd.option_context("display.width", 250, "display.max_rows", 300):
            print(table[["sample", "burst", "label", "t_start", "t_stop", "modes", "why", "ours", "zhu", "old_model", "old_Epeak", "old_ran_out",
                         "old_zhu_model", "old_zhu_Epeak", "old_zhu_late"]].round(2).to_string())
        print(f"{len(table)} windows in {table.groupby(['sample', 'burst']).ngroups} bursts; why: {table.why.value_counts().sort_index().to_dict()}")
        return
    run_late(args.workers)


if __name__ == "__main__":
    main()
