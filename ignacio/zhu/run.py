"""Running the measurements: data downloads, one burst from start to end, and the whole sample in batches that can run side by side and
be stopped and restarted.

From ignacio/:
    python -m zhu.run prefetch z25 type_I       download the data of the bursts in these groups (all groups if none are named)
    python -m zhu.run burst 211211A              measure one burst and write results/zhu/211211A/
    python -m zhu.run batch --workers 3          measure every burst of the sample that isn't finished; several batches can run at once
    python -m zhu.run batch --retry-failed       the same, and try again the bursts whose code failed in an earlier batch
    python -m zhu.run status                     one line per burst: its state, why it stopped if it did, and the code version used

One burst goes through five stages. Each one writes its own files in results/zhu/<burst>/ and a line in state.json (io.py):
- episodes: the light curve at their Tres, its Bayesian blocks, and the episodes under all three modes (episodes.json, blocks.csv,
  light_curve.npz, quick_look.png);
- errors: every mode again on 40 light curves simulated from the blocks (realisations.csv);
- resolutions: every mode at half and twice their Tres (resolutions.csv);
- lags: tau32 and tau42 of each mode's episodes, with a window shared by two modes measured once (lags.csv);
- spectra: each mode's PE and ME, their way and our rule for zhu2026 and zhu2025 and our rule alone for project, with a shared window
  fitted once (spectra_zhu2026.csv, spectra_zhu2025.csv, spectra_project.csv).
A stage whose files are already there is skipped, so a stopped batch loses at most the stage it was in. When the episodes are made again,
every later file and stage record goes with them. A burst that can't be measured (no data, no usable response, or emission before the data
start) is marked not_measurable with the reason, and a burst whose code fails is marked failed with the error. Neither stops the batch,
and neither is left without a line saying why."""
import argparse
import fcntl
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import RESULTS, SHARED
from grbtool.bursts import DATA_ROOT, BurstError, _angle, download, set_up
from grbtool.checks import Rates
from grbtool.windows import Window, cut_out
from .background import clear_background, emission_span, light_curve
from .catalogue import data_name
from .episodes import MODES, P0, at_resolutions, blocks_of, find_episodes, mvt_readings, with_errors
from .io import HELD, clear_stages, code_version, now, read_json, set_stage, set_status, state, states, write_csv, write_json

SAMPLE = SHARED / "zhu_sample.csv"
GROUPS = ["z25", "type_I", "supernova", "near_line", "random"]
REALISATIONS = 40
RESOLUTION_FACTORS = (0.5, 2.0)
RESOLUTION_MAX_BINS = 60000         # half their Tres is skipped above this many bins, where the blocks take several minutes
WITH_EE_WE = False                  # spectra of the extended and whole emission too (for the 29 of 2025); this run does PE and ME only
SPECTRUM_MINUTES = 5.0              # time for each slow step of one window's fits (see grbtool.fitting.Stages)
WAYS = {"zhu2026": ("ours", "zhu"), "zhu2025": ("ours", "zhu"), "project": ("ours",)}      # spectral fits made on each mode's windows
LATER = ("errors", "resolutions", "lags", "spectra")
LATER_FILES = ("realisations.csv", "resolutions.csv", "lags.csv", "spectra_zhu2026.csv", "spectra_zhu2025.csv", "spectra_project.csv",
               "spectra.csv")


def read_sample(groups=None):
    """The bursts we measure (written by catalogue.build_sample), in the order of GROUPS, optionally only some groups."""
    sample = pd.read_csv(SAMPLE, comment="#", index_col=0)
    groups = groups or GROUPS
    sample = sample[sample.reason.isin(groups)]
    order = {g: i for i, g in enumerate(GROUPS)}
    return sample.iloc[sorted(range(len(sample)), key=lambda i: order[sample.reason.iloc[i]])]


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def prefetch(sample, workers=3):
    """Download the data of every burst in the sample that isn't complete yet. Returns the names that failed."""
    def one(name, trigger):
        folder = data_name(name, DATA_ROOT)
        t0 = time.time()
        download(folder, trigger)
        return folder, time.time() - t0

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        jobs = {pool.submit(one, name, row.trigger): name for name, row in sample.iterrows()}
        for done, job in enumerate(as_completed(jobs), 1):
            name = jobs[job]
            try:
                folder, seconds = job.result()
                log(f"{done}/{len(jobs)} {name}: {folder} ready ({seconds:.0f} s)")
            except Exception as error:
                failed.append(name)
                log(f"{done}/{len(jobs)} {name}: FAILED {error}")
    return failed


def detectors_by_angle(folder, trigger, max_angle=60.0, most=3):
    """The detectors for a burst the GBM catalogue has no spectral fit for: the NaIs within max_angle of it, the closest `most` of them,
    and the BGO on the same side of the spacecraft (b0 faces n0-n5, b1 faces n6-nb)."""
    data_dir = DATA_ROOT / folder / "current"
    angles = {d: _angle(data_dir, trigger, d) for d in [f"n{i:x}" for i in range(12)]}
    near = sorted([d for d, a in angles.items() if a is not None and a < max_angle], key=lambda d: angles[d])[:most]
    if not near:
        raise BurstError(f"no NaI detector within {max_angle:g} deg of {trigger}")
    return near + ["b0" if int(near[0][1], 16) < 6 else "b1"]


def single_responses(folder, trigger):
    """For bursts whose archive has only one response matrix per detector (.rsp, as for many bursts of 2011-2016) instead of the
    time-resolved .rsp2 that grbtool downloads. I fetch each detector's CSPEC .rsp and save a copy named ..._single.rsp2. Both files have the
    same layout (a primary header, the energy bounds, then the matrices), so the shared code reads it as a .rsp2 with one matrix covering the
    whole burst. Returns the detectors done."""
    import re
    import subprocess
    from grbtool.bursts import ARCHIVE, _listing
    url = f"{ARCHIVE}/20{trigger[2:4]}/{trigger}/current/"
    names = sorted(set(re.findall(r'href="(glg_cspec_(?:n[0-9ab]|b[01])_' + trigger + r'_v\d+\.rsp)"', _listing(url))))
    out = DATA_ROOT / folder / "current"
    done = []
    for name in names:
        target = out / name
        if not target.exists():
            result = subprocess.run(["curl", "-sSfL", "--retry", "3", "-m", "300", "-o", str(target), url + name], capture_output=True, text=True)
            if result.returncode != 0:
                target.unlink(missing_ok=True)
                raise BurstError(f"couldn't download {name}: {result.stderr.strip()[:100]}")
        copy = out / name.replace(".rsp", "_single.rsp2")
        if not copy.exists():
            copy.write_bytes(target.read_bytes())
        done.append(name.split("_")[2])
    return done


def start_burst(name, trigger):
    """set_up with the catalogue's detectors. If the archive has only single-matrix responses, they are fetched first (single_responses);
    if the catalogue has no fit for the burst, the detectors are chosen by angle. Returns (burst, notes): a list of what was done."""
    folder = data_name(name, DATA_ROOT)
    notes = []
    for _ in range(3):
        try:
            if any("detectors_by_angle" in n for n in notes):
                detectors = detectors_by_angle(folder, trigger)
                return set_up(folder, trigger, detectors=detectors), notes
            return set_up(folder, trigger), notes
        except BurstError as error:
            if "no NaI detector with a response" in str(error) and not any("single_responses" in n for n in notes):
                notes.append(f"single_responses:{len(single_responses(folder, trigger))}")
            elif "no catalogue fit" in str(error) and not any("detectors_by_angle" in n for n in notes):
                notes.append("detectors_by_angle")
            else:
                raise
    raise BurstError(f"couldn't set up {name}")


def quick_look(lc, blocks, found, path):
    """The light curve with its blocks and the episodes shaded, to look at by eye."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    we = found["we"] or found["me"]
    pad = 0.15 * (we[1] - we[0]) + 5 * lc.tres
    keep = (lc.edges[:-1] >= we[0] - pad) & (lc.edges[1:] <= we[1] + pad)
    factor = max(1, int(keep.sum() // 600))
    edges = lc.edges[:-1][keep][::factor]
    net = np.add.reduceat((lc.counts - lc.background)[keep], np.arange(0, keep.sum(), factor)) / (factor * lc.tres)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.step(edges, net / 1e3, where="post", color="0.55", lw=0.7, label=f"8-1000 keV, {1e3 * factor * lc.tres:g} ms bins")
    inside = blocks[(blocks.stop > we[0] - pad) & (blocks.start < we[1] + pad)]
    ax.step(np.r_[inside.start.values, inside.stop.values[-1]], np.r_[inside.rate.values, inside.rate.values[-1]] / 1e3, where="post",
            color="k", lw=1.0, label="Bayesian blocks")
    colours = dict(pe="#2a78d6", qe1="#bbbbbb", me="#eb6834", qe2="#bbbbbb", ee="#1baf7a")     # as in grbtool.report
    for key, colour in colours.items():
        if found[key]:
            ax.axvspan(*found[key], color=colour, alpha=0.18 if key.startswith("qe") else 0.25, lw=0, label=key.upper())
    ax.set_xlim(we[0] - pad, we[1] + pad)
    ax.set_xlabel("time since trigger (s)")
    ax.set_ylabel("net rate (kcounts/s)")
    ax.legend(fontsize=7, ncol=4, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def claim(name, root=RESULTS):
    """Take a burst for this process, with an exclusive lock (flock) on results/zhu/<burst>/.lock. The kernel lets go of the lock when the
    process ends, however it ends, so a stopped batch never leaves a burst taken, and two processes can never both hold it. Returns the open
    lock file, which has to stay open while the burst is measured and is passed to release() afterwards, or None if another process already
    has the burst. The file itself is never deleted, because deleting it while another process opens it could give two locks on two
    different files."""
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    handle = open(folder / ".lock", "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    HELD.append(handle)
    return handle


def release(handle):
    if handle in HELD:
        HELD.remove(handle)
    fcntl.flock(handle, fcntl.LOCK_UN)
    handle.close()


def _windows(found, labels=("PE", "ME", "EE", "WE")):
    return {label: tuple(found[label.lower()]) for label in labels if found.get(label.lower())}


def stage_episodes(name, row, burst, bkg_flags, setup_flags, out):
    """The light curve, its blocks and every mode's episodes; writes episodes.json and the files the later stages read."""
    t0 = time.monotonic()
    rates = Rates(burst, emission_span(burst))
    lc = light_curve(burst, row.Tres)
    blocks = blocks_of(lc)
    modes, readings = {}, {}
    for mode in MODES:
        found = find_episodes(lc, blocks, mode=mode, burst=name)
        modes[mode] = found.record()
        index = lambda x: int(np.searchsorted(blocks.start.values, x - 1e-9))
        readings[mode] = {key: mvt_readings(lc, lc.counts, blocks, index(span[0]), index(span[1]) - 1)
                          for key, span in (("pe", found.pe), ("me", found.me)) if span}
    record = dict(burst=name, trigger=row.trigger, reason=row.reason, Tres=row.Tres, nai=burst.nai, detectors=burst.detectors,
                  bkg_windows=burst.bkg_windows, bkg_order=burst.bkg_order, bkg_flags=bkg_flags, setup_flags=setup_flags,
                  emission_span=emission_span(burst), bad_intervals=rates.telemetry_intervals(),
                  data_gaps=int(len(rates.gaps)), nbins=int(len(lc.counts)), nblocks=int(len(blocks)), p0=P0,
                  band_keV=[8.0, 1000.0], modes=modes, mvt_readings=readings, code_version=code_version(), seconds=time.monotonic() - t0)
    write_csv(out / "blocks.csv", blocks, index=False)
    np.savez_compressed(out / "light_curve.tmp.npz", edges=lc.edges, counts=lc.counts, exposure=lc.exposure, background=lc.background, tres=lc.tres)
    os.replace(out / "light_curve.tmp.npz", out / "light_curve.npz")
    write_json(out / "episodes.json", record)
    if modes["zhu2026"].get("me"):
        quick_look(lc, blocks, modes["zhu2026"], out / "quick_look.png")
    return lc, blocks


def _overlap(span, intervals):
    """Seconds of a window inside the given intervals."""
    return float(sum(max(0.0, min(span[1], b) - max(span[0], a)) for a, b in intervals))


def stage_lags(record, burst, out):
    """tau32 and tau42 of each mode's episodes. A window shared by two modes is measured once and written for both. A lag that raises an
    error is stored as failed with the error. bad_s is how much of the window is at the telemetry limit (the lag is measured anyway; the
    column says where to be careful)."""
    from .lags import EpisodeLags
    source = EpisodeLags(burst)
    rows, done = [], {}
    for mode in MODES:
        for label, span in _windows(record["modes"][mode]).items():
            for lag in ("tau32", "tau42"):
                key = (label, lag, round(span[0], 6), round(span[1], 6))
                if key not in done:
                    try:
                        done[key] = source.measure(lag, span, record["Tres"])
                    except Exception as error:
                        done[key] = dict(lag=lag, status=f"failed:{type(error).__name__}: {error}"[:200])
                rows.append(dict(mode=mode, label=label, t_start=span[0], t_stop=span[1], bad_s=_overlap(span, record["bad_intervals"]), **done[key]))
    write_csv(out / "lags.csv", pd.DataFrame(rows), index=False)


def stage_spectra(record, burst, out, workers):
    """Spectra of each mode's PE and ME (and of the 2025 bursts' EE and WE if WITH_EE_WE is set). Their way and our rule run on the
    zhu2026 and zhu2025 windows, and only our rule on the project windows (WAYS). A window is fitted once per run: a mode whose window is
    exactly one that was already fitted copies that row (for the project mode, only our rule's part). Nothing is taken from an earlier run. The
    intervals at the telemetry limit are cut out of each window, and a window that lies completely inside them gets a row with no fit and the
    flag window_inside_bad_interval. t_start and t_stop are the episode's window, and pieces_s are the stretches actually fitted."""
    from .spectra import fit_episodes
    bad = record["bad_intervals"]
    done = {}
    for mode, ways in WAYS.items():
        labels = ("PE", "ME") + (("EE", "WE") if WITH_EE_WE and record["reason"] == "z25" and mode != "project" else ())
        windows = _windows(record["modes"][mode], labels)
        todo = []
        for label, span in windows.items():
            key = (label, round(span[0], 6), round(span[1], 6))
            if key in done and set(ways) <= set(done[key]["ways"].split()):
                continue
            window = Window(label, cut_out(span, bad), "precursor" if label == "PE" else "main")
            if window.pieces:
                todo.append((key, span, window))
            else:
                done[key] = dict(label=label, t_start=span[0], t_stop=span[1], pieces=0, ways="", fit_flags="window_inside_bad_interval")
        if todo:
            fitted = fit_episodes(burst, [w for _, _, w in todo], workers=min(workers, len(todo)), minutes=SPECTRUM_MINUTES, ways=ways)
            for (key, span, _), (_, r) in zip(todo, fitted.reset_index().iterrows()):
                r = r.to_dict()
                r.update(t_start=span[0], t_stop=span[1], ways=" ".join(ways))
                done[key] = r
        rows = []
        for label, span in windows.items():
            r = dict(done[(label, round(span[0], 6), round(span[1], 6))])
            if "zhu" not in ways:
                r = {k: v for k, v in r.items() if not str(k).startswith("zhu_") and k != "seconds_zhu"}
                r["ways"] = "ours" if r.get("ways") else ""
            rows.append(r)
        table = pd.DataFrame(rows).set_index("label") if rows else pd.DataFrame(columns=["label", "t_start", "t_stop"]).set_index("label")
        write_csv(out / f"spectra_{mode}.csv", table)


def measure(name, row, workers=3, realisations=REALISATIONS, redo=(), out_root=RESULTS):
    """Every stage for one burst. row has trigger, Tres and reason (a row of the sample). Each stage records itself in state.json."""
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    set_status(name, "in_progress", "", out_root)
    try:
        burst, notes = start_burst(name, row.trigger)
    except BurstError as error:
        (out / "not_measurable.txt").write_text(str(error) + "\n")
        set_status(name, "not_measurable", error, out_root)
        raise
    (out / "not_measurable.txt").unlink(missing_ok=True)
    setup_flags = list(burst.flags) + list(notes)
    bkg_flags = clear_background(burst)
    record = read_json(out / "episodes.json", {})
    same = len(record.get("bkg_windows", [])) == len(burst.bkg_windows) and np.allclose(np.ravel(record.get("bkg_windows", [])), np.ravel(burst.bkg_windows))
    if "modes" in record and "episodes" not in redo and not same:
        raise RuntimeError(f"the background intervals are now {burst.bkg_windows}, not the {record['bkg_windows']} the episodes were found "
                           "with; redo the episodes")
    lc = blocks = None
    if "episodes" in redo or "modes" not in record:
        # everything after the episodes depends on them, so their files and stage records go too
        for later in LATER_FILES:
            (out / later).unlink(missing_ok=True)
        clear_stages(name, LATER, out_root)
        lc, blocks = stage_episodes(name, row, burst, bkg_flags, setup_flags, out)
        record = read_json(out / "episodes.json")
        set_stage(name, "episodes", "done", f"{record['nblocks']} blocks", out_root)
        log(f"{name}: episodes in {record['seconds']:.0f} s; " + "; ".join(f"{m}: {' '.join(record['modes'][m]['flags']) or 'no flags'}" for m in MODES))
    if not any(record["modes"][m].get("me") for m in MODES):
        for stage in LATER:
            set_stage(name, stage, "skipped", "no main emission under any mode", out_root)
        set_status(name, "complete", "no main emission found under any mode", out_root)
        log(f"{name}: no main emission found, stopping here")
        return record

    if "errors" in redo or not (out / "realisations.csv").exists():
        t0 = time.monotonic()
        if lc is None:
            from .io import load_light_curve
            lc, blocks = load_light_curve(out)
        write_csv(out / "realisations.csv", with_errors(lc, blocks, realisations, workers=workers), index=False)
        set_stage(name, "errors", "done", f"{realisations} realisations", out_root)
        log(f"{name}: {realisations} realisations in {time.monotonic() - t0:.0f} s")

    if "resolutions" in redo or not (out / "resolutions.csv").exists():
        t0 = time.monotonic()
        span = emission_span(burst)
        factors = tuple(f for f in RESOLUTION_FACTORS if (span[1] - span[0]) / (row.Tres * f) <= RESOLUTION_MAX_BINS)
        skipped = [f for f in RESOLUTION_FACTORS if f not in factors]
        table = at_resolutions(burst, row.Tres, factors)
        if skipped:
            table = pd.concat([table, pd.DataFrame([dict(factor=f, skipped="too_many_bins") for f in skipped])], ignore_index=True)
        write_csv(out / "resolutions.csv", table, index=False)
        set_stage(name, "resolutions", "done", f"factors {factors}, skipped {skipped}", out_root)
        log(f"{name}: resolutions in {time.monotonic() - t0:.0f} s")

    if "lags" in redo or not (out / "lags.csv").exists():
        t0 = time.monotonic()
        stage_lags(record, burst, out)
        set_stage(name, "lags", "done", "", out_root)
        log(f"{name}: lags in {time.monotonic() - t0:.0f} s")

    if "spectra" in redo or not all((out / f"spectra_{mode}.csv").exists() for mode in WAYS):
        t0 = time.monotonic()
        stage_spectra(record, burst, out, workers)
        set_stage(name, "spectra", "done", "", out_root)
        log(f"{name}: spectra in {time.monotonic() - t0:.0f} s")
    set_status(name, "complete", "", out_root)
    log(f"{name}: done in {time.monotonic() - began:.0f} s")
    return record


def finished(name, out_root=RESULTS, retry_failed=False, since=None):
    """A burst is finished when its state says complete, not_measurable or failed. With retry_failed a failed burst isn't finished,
    unless it failed again after `since` (the time this batch began), so one batch tries it once."""
    s = state(name, out_root)
    status = s.get("status")
    if status == "failed" and retry_failed:
        return since is not None and s.get("time", "") >= since
    return status in ("complete", "not_measurable", "failed")


def batch(sample, workers=3, realisations=REALISATIONS, out_root=RESULTS, retry_failed=False):
    """measure() for every burst of the sample that isn't finished and isn't taken by another batch, in the sample's order. Several batches
    can run side by side over the same sample: each takes a burst with claim() and checks again that it isn't finished before starting."""
    since = now()
    failed = {}
    for i, (name, row) in enumerate(sample.iterrows(), 1):
        if finished(name, out_root, retry_failed, since):
            continue
        handle = claim(name, out_root)
        if handle is None:
            continue
        try:
            if finished(name, out_root, retry_failed, since):     # another batch finished it between the first check and the claim
                continue
            log(f"--- {i}/{len(sample)} {name} ({row.reason})")
            try:
                measure(name, row, workers, realisations, out_root=out_root)
            except BurstError as error:
                failed[name] = str(error)
                if state(name, out_root).get("status") != "not_measurable":
                    set_status(name, "failed", f"BurstError: {error}", out_root)
                log(f"{name}: can't be measured: {error}")
            except Exception as error:
                failed[name] = f"{type(error).__name__}: {error}"
                set_status(name, "failed", failed[name], out_root)
                log(f"{name}: FAILED {failed[name]}\n{traceback.format_exc()}")
        finally:
            release(handle)
    log(f"batch finished, {len(failed)} not measured or failed here: {failed}")
    return failed


def measure_named(names, sample, workers=3, realisations=REALISATIONS, redo=(), out_root=RESULTS):
    """measure() for the bursts named, one after another, each taken with claim(). A failure is recorded in the burst's state, as batch()
    records it, and the next name is measured."""
    for name in names:
        handle = claim(name, out_root)
        if handle is None:
            log(f"{name} is being measured by another process; left alone")
            continue
        try:
            measure(name, sample.loc[name], workers, realisations, tuple(redo), out_root=out_root)
        except BurstError as error:
            if state(name, out_root).get("status") != "not_measurable":
                set_status(name, "failed", f"BurstError: {error}", out_root)
            log(f"{name}: can't be measured: {error}")
        except Exception as error:
            set_status(name, "failed", f"{type(error).__name__}: {error}", out_root)
            log(f"{name}: FAILED {type(error).__name__}: {error}\n{traceback.format_exc()}")
        finally:
            release(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["prefetch", "burst", "batch", "status"])
    parser.add_argument("names", nargs="*", help=f"burst names for 'burst', or some of {GROUPS} for 'prefetch' and 'batch'")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--realisations", type=int, default=REALISATIONS)
    parser.add_argument("--redo", nargs="*", default=[], help="stages to run again even if their files exist")
    parser.add_argument("--retry-failed", action="store_true", help="batch: try again the bursts whose code failed in an earlier batch")
    args = parser.parse_args(argv)
    if args.task == "prefetch":
        sample = read_sample(args.names)
        log(f"downloading {len(sample)} bursts with {args.workers} at a time")
        failed = prefetch(sample, args.workers)
        log(f"finished; failed: {failed or 'none'}")
    elif args.task == "burst":
        measure_named(args.names, read_sample(), args.workers, args.realisations, args.redo)
    elif args.task == "status":
        table = states(read_sample(args.names).index)
        print(table.to_string())
        print(table.status.value_counts().to_string())
    else:
        batch(read_sample(args.names), args.workers, args.realisations, retry_failed=args.retry_failed)


if __name__ == "__main__":
    sys.exit(main())
