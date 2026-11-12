"""Run the steps in order for one burst and write the tables."""
import json
import subprocess
import time
from datetime import date
from pathlib import Path

import pandas as pd

from . import CODE_DIR
from .bursts import DATA_ROOT, PRESETS, BurstError, download, find_trigger, set_up
from .checks import Rates, catalogue_check, local_background_check, systematics, window_flags
from .fitting import STAGE_MINUTES, all_spectra, fit_windows, time_limit
from .physics import amati_offsets, budget, energetics, lag_table, z_for_budget
from .report import epeak_vs_mvt, joint_table, quick_look
from .windows import BIN_WIDTH, automatic_windows, blocks_at, windows_from_file

RESULTS = CODE_DIR / "results"


def _git(*args):
    return subprocess.run(["git", *args], cwd=CODE_DIR, capture_output=True, text=True).stdout.strip()


def local_trigger(name):
    """The trigger from the files already downloaded for this burst, if there are any."""
    found = sorted((DATA_ROOT / name / "current").glob("glg_bcat_all_bn*_v*.fit"))
    return found[-1].name.split("_")[3] if found else None


def burst_summary(burst, table):
    """Energetics of the precursor and of the main emission, Amati offsets, and the budget a precursor would need. Only what the redshift allows."""
    z = burst.redshift
    out = dict(burst=burst.name, trigger=burst.trigger, z=z, T90=burst.t90, T90_start=burst.t90_start, detectors=" ".join(burst.detectors), flags=" ".join(burst.flags))
    pre = table.loc["pre"] if "pre" in table.index else None
    main = table.loc["main"] if "main" in table.index else None
    if pre is not None and "Epeak" in pre and pd.notna(pre.Epeak):
        out.update(pre_start=pre.t_start, pre_stop=pre.t_stop, pre_sigma=pre.sigma, pre_usable=bool(pre.usable), pre_measured=bool(pre.measured), pre_model=pre.model, pre_Ep=pre.Epeak,
                   pre_Ep_lo=pre.Epeak_err_lo, pre_Ep_hi=pre.Epeak_err_hi, pre_Ep_cpl=pre.Epeak_cpl, pre_limit_hi=pre.Epeak_cpl_limit_hi, pre_flux=pre.flux)
        if main is not None and pd.notna(main.Epeak):
            out["pre_over_main_Ep_cpl"] = pre.Epeak_cpl / main.Epeak_cpl
            out["pre_over_main_Ep_best"] = pre.Epeak / main.Epeak
        if z is None:
            out.update(z_at_fiducial=z_for_budget(pre.flux, 1.0), z_at_18_times=z_for_budget(pre.flux, 18.0))
    if main is not None and pd.notna(main.Epeak):
        out.update(main_start=main.t_start, main_stop=main.t_stop, main_model=main.model, main_Ep=main.Epeak, main_Ep_lo=main.Epeak_err_lo, main_Ep_hi=main.Epeak_err_hi, main_flux=main.flux)
    if z is None:
        return out
    for label, row in (("pre", pre), ("main", main)):
        if row is None or pd.isna(row.Epeak):
            continue
        e = energetics(row, z)
        out.update({f"{label}_{k}": v for k, v in e.items()})
        out.update({f"{label}_{k}": v for k, v in amati_offsets(e["E_iso"], e["Epeak_rest"]).items()})
    if pre is not None and pd.notna(pre.Epeak):
        out.update({f"pre_no_k_{k}": v for k, v in budget(out["pre_L_iso_no_k"]).items()})
        out.update({f"pre_k_{k}": v for k, v in budget(out["pre_L_iso"]).items()})
    return out


def merged_flags(table):
    """The flags from the rates, from the fits and from the checks, in one column."""
    parts = [table["flags"].fillna("")]
    if "fit_flags" in table:
        parts.append(table.fit_flags.fillna(""))
    if "check_failed" in table:
        parts.append(table.check_failed.notna().map({True: "checks_failed", False: ""}))
    return [" ".join(word for word in row if word) for row in zip(*parts)]


def analyse(name, trigger=None, redshift=None, detectors=None, windows_file=None, bkg_windows=None, bad_intervals=None, get_data=True, workers=4, with_checks=True, with_lags=True,
            with_mvt=False, mvt_bins=(1e-4, 1e-5, 3e-6), mvt_workers=8, mvt_max_bins=3e5, mvt_slices=False, out_dir=None, preset=False, only=None, fit_minutes=STAGE_MINUTES, log=print):
    """The whole analysis of one burst. Returns the folder with the results."""
    if preset:
        p = PRESETS[name]
        trigger, detectors, windows_file, bkg_windows, bad_intervals = p["trigger"], p["detectors"], p["windows_file"], p["bkg_windows"], p["bad_intervals"]
    trigger = trigger or local_trigger(name)
    if trigger is None:
        if not get_data:
            raise BurstError(f"no data for {name} and downloading is off")
        trigger = find_trigger(name)
        log(f"{name} is trigger {trigger}")
    data_dir = DATA_ROOT / name / "current"
    if get_data and not list(data_dir.glob(f"glg_bcat_all_{trigger}_v*.fit")):
        log(f"downloading {trigger}")
        download(name, trigger)
    burst = set_up(name, trigger, detectors=detectors, redshift=redshift)
    log(f"{name}: T90 {burst.t90:.1f} s from {burst.t90_start:.1f} s, detectors {' '.join(burst.detectors)}, z {burst.redshift}")
    if bkg_windows:
        burst.bkg_windows = [tuple(w) for w in bkg_windows]

    # rates first, because the bad intervals decide the windows
    t_end = burst.t90_start + burst.t90
    rates = Rates(burst, (max(burst.tte_range[0] + 1.0, min(-30.0, burst.t90_start - 20.0)), t_end + 60.0))
    if rates.missing:
        burst.flags.append("no_tte:" + "+".join(rates.missing))
    if bad_intervals is None:
        bad_intervals = rates.telemetry_intervals()
    if bad_intervals:
        burst.flags.append("bad_interval:" + "+".join(f"{a:g}to{b:g}" for a, b in bad_intervals))
    log(f"highest summed rate {rates.summed.max():.0f} kcps, bad intervals {bad_intervals}, {len(rates.gaps)} data gaps")

    precursor, blocks = None, None
    if windows_file:
        windows = windows_from_file(windows_file)
        if not burst.bkg_windows:
            raise BurstError("a windows file needs background intervals too")
        blocks = blocks_at(burst, BIN_WIDTH)    # only for the quick look
    else:
        windows, precursor, blocks = automatic_windows(burst, bad_intervals, background_given=bool(bkg_windows))
    if only:
        windows = [w for w in windows if w.label in only]
    log(f"background {[tuple(round(x, 1) for x in w) for w in burst.bkg_windows]}, order {burst.bkg_order}; {len(windows)} windows; flags {burst.flags}")

    spectra, chi2 = all_spectra(burst, windows)
    burst.bkg_chi2 = chi2
    if with_checks:
        order2 = {d: all_spectra_one(burst, windows, d) for d in burst.detectors}

    def extras(window, row):
        """Run inside each worker, so the systematics share the worker's time."""
        out = {}
        if with_checks:
            began = time.monotonic()
            try:
                with time_limit(60.0 * fit_minutes):
                    out.update(systematics(burst, window, pd.Series(row), spectra, order2))
                    if window.kind == "precursor":
                        out.update(local_background_check(burst, window))
            except Exception as error:
                out["check_failed"] = f"{type(error).__name__}: {error}"
            out["seconds_checks"] = time.monotonic() - began
        return out

    table = fit_windows(burst, windows, workers=workers, spectra=spectra, extras=extras, minutes=fit_minutes)
    flags = pd.DataFrame({w.label: window_flags(burst, w, rates, spectra) for w in windows}).T
    table = table.join(flags)
    table["flags"] = merged_flags(table)
    table = table.drop(columns="fit_flags", errors="ignore")
    if "ran_out" in table and table.ran_out.fillna("").any():
        log("ran out of time: " + ", ".join(f"{label} ({stages})" for label, stages in table.ran_out.dropna().items() if stages))
    if "fit_failed" in table:
        log("fits failed: " + ", ".join(f"{label} ({why})" for label, why in table.fit_failed.dropna().items()))
    if burst.redshift is not None:
        for label in table.index[table.Epeak.notna()]:
            for key, value in energetics(table.loc[label], burst.redshift).items():
                table.loc[label, key] = value

    out_dir = Path(out_dir) if out_dir else RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "windows.csv", float_format="%.6g")
    rates.gaps.to_csv(out_dir / "gaps.csv", index=False, float_format="%.6g")
    if blocks is not None:
        blocks.to_csv(out_dir / "blocks.csv", index=False, float_format="%.6g")
    quick_look(burst, table, blocks, bad_intervals, out_dir / "quick_look.png")

    summary = burst_summary(burst, table)
    if with_checks:
        summary["catalogue_check"] = catalogue_check(burst)
    if with_lags:
        lags = lag_table(burst, windows)
        lags.to_csv(out_dir / "lags.csv", float_format="%.5g")
        for label, row in lags.iterrows():
            summary.update({f"{label}_lag_ms": row.lag_ms, f"{label}_lag_down": row.err_down, f"{label}_lag_up": row.err_up, f"{label}_lag_usable": bool(row.usable)})
    if with_mvt:
        from .mvt import mvt_table
        kinds = ("precursor", "main", "tail", "given", "slice") if mvt_slices else ("precursor", "main", "tail", "given")
        timing = mvt_table(burst, windows, bin_widths=mvt_bins, workers=mvt_workers, max_bins=mvt_max_bins, kinds=kinds)
        timing.to_csv(out_dir / "mvt.csv", index=False, float_format="%.5g")
        joint = joint_table(table, timing)
        joint.to_csv(out_dir / "joint.csv", float_format="%.6g")
        epeak_vs_mvt(burst, joint, out_dir / "epeak_vs_mvt.png")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    provenance = dict(burst=name, trigger=trigger, data_dir=str(data_dir), run_date=date.today().isoformat(), commit=_git("rev-parse", "--short", "HEAD") + ("+uncommitted" if _git("status", "--porcelain", ".") else ""),
                      detectors=burst.detectors, angles=burst.angles, background=burst.bkg_windows, background_order=burst.bkg_order, background_chi2=burst.bkg_chi2,
                      bad_intervals=bad_intervals, windows={w.label: w.pieces for w in windows}, window_source=str(windows_file) if windows_file else "automatic",
                      precursor_window_other_bin=precursor.get("window_other_bin") if precursor else None, flags=burst.flags)
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    log(f"wrote {out_dir}")
    return out_dir


def all_spectra_one(burst, windows, det):
    """The spectra of one detector with a background of order 2, for the systematics."""
    from spectral_fit_tools import make_spectra
    stretches = {(w.label, i): piece for w in windows for i, piece in enumerate(w.pieces)}
    return make_spectra(burst.data_dir, burst.trigger, det, stretches, burst.bkg_windows, 2)[0]
