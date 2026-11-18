"""Downloads that survive HEASARC's slow long transfers. Each file is fetched with curl, resuming from whatever is already on disk, and a
transfer that stays slower than MIN_RATE for STALL_S seconds is dropped and started again on a fresh connection. I added this on 16 Nov
2026, when a fresh connection got about 500 kB/s while transfers that had been open for minutes had slowed down to a few kB/s. A file only
counts as complete when its size matches the archive's Content-Length, the same test grbtool.bursts.download uses (so that one then skips
these files).

From ignacio/:   python -m coppin.fetch --workers 4      the GBM files of every burst in shared/coppin_sample.csv"""
import argparse
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from grbtool.bursts import ARCHIVE, DATA_ROOT, WANTED, _listing
from zhu.catalogue import data_name

from . import SAMPLE

MIN_RATE = 30000        # bytes per second
STALL_S = 20
TRIES = 60


def remote_size(url):
    head = subprocess.run(["curl", "-sIL", "-m", "60", url], capture_output=True, text=True).stdout
    sizes = re.findall(r"(?i)content-length:\s*(\d+)", head)
    return int(sizes[-1]) if sizes else None


def fetch_file(url, target):
    """True once target has the archive's size."""
    target = Path(target)
    size = None
    for _ in range(TRIES):
        size = size or remote_size(url)
        if size is not None and target.exists():
            if target.stat().st_size == size:
                return True
            if target.stat().st_size > size:
                target.unlink()
        subprocess.run(["curl", "-sSfL", "-C", "-", "--speed-limit", str(MIN_RATE), "--speed-time", str(STALL_S), "-m", "1800",
                        "-o", str(target), url], capture_output=True)
    return size is not None and target.exists() and target.stat().st_size == size


def fetch_burst(name, trigger):
    """Every file grbtool.bursts.download would take for this burst. Returns the files that didn't complete."""
    url = f"{ARCHIVE}/20{trigger[2:4]}/{trigger}/current/"
    names = sorted(f for f in set(re.findall(r'href="(glg_[^"]+)"', _listing(url))) if WANTED.search(f))
    out = DATA_ROOT / data_name(name, DATA_ROOT) / "current"
    out.mkdir(parents=True, exist_ok=True)
    return [f for f in names if not fetch_file(url + f, out / f)]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="coppin.fetch")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    sample = pd.read_csv(SAMPLE, comment="#", index_col=0)
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(fetch_burst, name, row.trigger): name for name, row in sample.iterrows()}
        for done, job in enumerate(as_completed(jobs), 1):
            name = jobs[job]
            try:
                bad = job.result()
                print(time.strftime("%H:%M:%S"), f"{done}/{len(jobs)} {name}:", "complete" if not bad else f"INCOMPLETE {bad}", flush=True)
            except Exception as error:
                print(time.strftime("%H:%M:%S"), f"{done}/{len(jobs)} {name}: FAILED {error}", flush=True)


if __name__ == "__main__":
    main()
