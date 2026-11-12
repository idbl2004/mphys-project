"""Files: where each burst's results live, its state, the code version that made them, and reading them back.

Each burst has a folder results/zhu/<burst>/ with one file per stage (see run.py) and a state.json saying, for every stage, whether it is
done, failed (with the error) or skipped (with the reason), when, and with which code. A burst is never dropped without a line saying why.
Files are written to a temporary name and renamed, so a stopped run never leaves half a file behind.
"""
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import CODE_DIR, RESULTS

STAGES = ("episodes", "errors", "resolutions", "lags", "spectra")


def burst_dir(name, root=RESULTS):
    return Path(root) / name


def jsonable(value):
    """numpy values, tuples and NaN made safe for JSON: NaN becomes null, and an infinite value the string "inf" or "-inf" (a recovery
    ratio after a gap at or below the background is infinite), which float() reads back."""
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        return float(value) if np.isfinite(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(data), indent=1))
    os.replace(tmp, path)


def write_csv(path, table, **options):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(tmp, **options)
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


_VERSION = None


def code_version():
    """The commit the code was run from (with +changes if the zhu package differs from it) and a short hash of the zhu package, worked out
    once when a process first asks, so every result of one run carries the version of the code that run loaded, even if the files are
    edited while it runs."""
    global _VERSION
    if _VERSION is None:
        _VERSION = _code_version_now()
    return _VERSION


def _code_version_now():
    package = sorted((CODE_DIR / "zhu").glob("*.py"))
    digest = hashlib.sha1(b"".join(p.read_bytes() for p in package)).hexdigest()[:10]
    try:
        commit = subprocess.run(["git", "-C", str(CODE_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        changed = subprocess.run(["git", "-C", str(CODE_DIR), "status", "--porcelain", "--", "zhu"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit, changed = "unknown", ""
    return f"git:{commit}{'+changes' if changed else ''} zhu:{digest}"


def now():
    """The time to the microsecond, as text that sorts in time order."""
    return datetime.now().isoformat(sep=" ", timespec="microseconds")


def state(name, root=RESULTS):
    return read_json(burst_dir(name, root) / "state.json", {"burst": name, "status": "not_started", "stages": {}})


def set_stage(name, stage, status, detail="", root=RESULTS, version=None):
    """Record one stage of one burst: status is done, failed or skipped."""
    s = state(name, root)
    s["stages"][stage] = dict(status=status, detail=str(detail)[:500], time=time.strftime("%Y-%m-%d %H:%M:%S"), code=version or code_version())
    s["status"] = "in_progress"
    burst_dir(name, root).mkdir(parents=True, exist_ok=True)
    write_json(burst_dir(name, root) / "state.json", s)
    return s


def set_status(name, status, reason="", root=RESULTS):
    """The burst as a whole: complete, not_measurable (with the reason), failed (with the error) or in_progress, and when."""
    s = state(name, root)
    s["status"], s["reason"], s["time"] = status, str(reason)[:500], now()
    burst_dir(name, root).mkdir(parents=True, exist_ok=True)
    write_json(burst_dir(name, root) / "state.json", s)
    return s


def clear_stages(name, stages, root=RESULTS):
    """Forget these stages of a burst: their files are about to be made again (run.measure does this when the episodes are redone)."""
    s = state(name, root)
    for stage in stages:
        s["stages"].pop(stage, None)
    burst_dir(name, root).mkdir(parents=True, exist_ok=True)
    write_json(burst_dir(name, root) / "state.json", s)
    return s


def states(names, root=RESULTS):
    """One row per burst: its overall status and reason, the status of each stage, and the code version(s) its stages were run with
    (more than one means the stages come from different versions of the code)."""
    rows = []
    for name in names:
        s = state(name, root)
        versions = sorted({v.get("code", "") for v in s["stages"].values()} - {""})
        rows.append(dict(burst=name, status=s.get("status"), reason=s.get("reason", ""),
                         **{stage: s["stages"].get(stage, {}).get("status", "") for stage in STAGES}, code=" | ".join(versions)))
    return pd.DataFrame(rows).set_index("burst")


def load_light_curve(folder):
    """The light curve and blocks saved by run.measure for one burst."""
    from .background import LightCurve
    saved = np.load(Path(folder) / "light_curve.npz")
    lc = LightCurve(saved["edges"], saved["counts"], saved["exposure"], saved["background"], float(saved["tres"]))
    return lc, pd.read_csv(Path(folder) / "blocks.csv")
