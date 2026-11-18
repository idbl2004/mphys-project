"""python -m coppin sample | prefetch | batch | status (see the package's docstring)."""
import argparse
import sys

import pandas as pd

from . import RESULTS, SAMPLE, CODE_DIR

sys.path.insert(0, str(CODE_DIR))


def read_sample():
    return pd.read_csv(SAMPLE, comment="#", index_col=0)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="coppin")
    parser.add_argument("part", choices=["sample", "prefetch", "batch", "status"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.part == "sample":
        from .sample import build
        s = build()
        print(f"{len(s)} bursts written to {SAMPLE}; Tres (ms):", (1e3 * s.Tres).round().astype(int).value_counts().sort_index().to_dict())
        return
    from zhu import run
    sample = read_sample()
    if args.part == "prefetch":
        failed = run.prefetch(sample, workers=args.workers)
        print("failed:", failed or "none")
    elif args.part == "batch":
        RESULTS.mkdir(parents=True, exist_ok=True)
        run.batch(sample, workers=args.workers, out_root=RESULTS, retry_failed=args.retry_failed)
    else:
        from zhu.io import states
        print(states(sample.index, RESULTS).to_string())


if __name__ == "__main__":
    main()
