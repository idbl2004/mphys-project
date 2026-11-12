"""python -m grbtool GRB180728A --z 0.117"""
import argparse
import os

# Each worker fits one window, so the linear algebra stays on one thread
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

parser = argparse.ArgumentParser(prog="grbtool", description="E_peak, MVT and the rest of the analysis for any GBM burst, with one set of rules.")
parser.add_argument("burst", help="name like GRB180728A")
parser.add_argument("--trigger", help="GBM trigger like bn180728728, if the name isn't enough")
parser.add_argument("--z", type=float, help="redshift, for the energetics")
parser.add_argument("--detectors", nargs="+", help="instead of the catalogue's")
parser.add_argument("--windows", help="csv with label,t_start,t_stop, instead of finding them")
parser.add_argument("--bkg", nargs=4, type=float, metavar=("A", "B", "C", "D"), help="background intervals A to B and C to D, instead of finding them")
parser.add_argument("--bad-interval", nargs=2, type=float, action="append", metavar=("A", "B"), help="time to leave out, instead of finding it from the rates")
parser.add_argument("--preset", action="store_true", help="the hand choices of notebooks 12 and 14, to test the tool against their tables")
parser.add_argument("--only", nargs="+", help="fit only these windows")
parser.add_argument("--workers", type=int, default=4, help="windows fitted at once")
parser.add_argument("--no-download", action="store_true")
parser.add_argument("--no-checks", action="store_true", help="skip the systematics and the catalogue check")
parser.add_argument("--no-lags", action="store_true")
parser.add_argument("--mvt", action="store_true", help="also a first look at the MVT, with Rodrigo's Haar code. The values to quote come from his own program, through grbtool.joint")
parser.add_argument("--mvt-bins", nargs="+", type=float, default=[0.1, 0.01, 0.003], help="bin widths in ms")
parser.add_argument("--mvt-workers", type=int, default=8)
parser.add_argument("--mvt-max-bins", type=float, default=3e5, help="narrower bins only while the light curve has at most this many bins")
parser.add_argument("--mvt-slices", action="store_true", help="also the time slices, which takes much longer")
parser.add_argument("--fit-minutes", type=float, default=30.0, help="time for each slow stage of one window. A fit that runs out keeps its best start so far, and the table says so")
parser.add_argument("--out", help="folder for the results")
args = parser.parse_args()

from .bursts import BurstError
from .pipeline import analyse
from .windows import WindowError

try:
    analyse(args.burst, trigger=args.trigger, redshift=args.z, detectors=args.detectors, windows_file=args.windows, bkg_windows=[args.bkg[:2], args.bkg[2:]] if args.bkg else None,
            bad_intervals=[tuple(b) for b in args.bad_interval] if args.bad_interval else None, get_data=not args.no_download, workers=args.workers, with_checks=not args.no_checks,
            with_lags=not args.no_lags, with_mvt=args.mvt, mvt_bins=[1e-3 * b for b in args.mvt_bins], mvt_workers=args.mvt_workers, mvt_max_bins=args.mvt_max_bins, mvt_slices=args.mvt_slices,
            out_dir=args.out, preset=args.preset, only=args.only, fit_minutes=args.fit_minutes)
except (BurstError, WindowError) as error:
    raise SystemExit(f"{args.burst}: {error}")
