"""E_peak and the MVT of the same windows, joined by label, for one burst or several. Nothing here measures an MVT.

The MVT table comes from Rodrigo's program, one csv per burst, in the format agreed on 4 Nov 2026. This reads it, refuses a table that breaks the format
instead of guessing, joins it to the E_peak table, and draws E_peak against MVT. The axes come from the data, because the two bursts together run from
about 2 ms to 60 ms and from about 40 keV to 1600 keV.

Run it from ignacio/ with:  python -m grbtool.joint GRB230307A GRB211211A
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullFormatter

from . import CODE_DIR
from .report import PLAIN

SHARED = CODE_DIR / "shared"
RESULTS = CODE_DIR / "results"
TIMING = RESULTS / "timing"     # his tables wait here, out of git, until he clears them for the repo
REQUIRED = ["label", "mvt_s", "mvt_err_lo_s", "mvt_err_hi_s", "band_keV", "upper_limit"]
TEXT_COLUMNS = ["label", "band_keV", "upper_limit", "class", "detectors", "trusted", "flags", "sys_note", "script", "mvt_commit"]
BAND = "8-900"
BURST_COLOURS = ["#eb6834", "#2a78d6", "#1baf7a", "#7a4fbf", "#b8860b", "#c2185b"]
ARROW = 0.25                    # length of an upper-limit arrow, as a share of the value
SHOW_LABELS_UP_TO = 8           # rows of one burst. With more than this only the precursor gets a label
TICKS = [m * 10.0 ** k for k in range(-1, 5) for m in (1, 2, 5)]
# Dichiara et al. (2023) Table 1, 8-1000 keV. They go on the figure in grey, and never into a joined table.
PUBLISHED = {"GRB230307A": pd.DataFrame({"mvt_ms": [17.0, 12.0, 15.0, 62.0], "err_ms": [2.0, np.nan, 2.0, 4.0], "upper_limit": [False, True, False, False]},
                                        index=["pre", "m01", "t1_07_18", "t1_18_40"])}


class TableError(Exception):
    """A table breaks the agreed format, and the message says where. Nothing is joined."""


def read_mvt_table(path, band=BAND):
    """His table, checked against every rule of the format. upper_limit comes back as a bool, and trusted, flags and detectors as text, whatever the rows hold."""
    path = Path(path)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows or not rows[0]:
        raise TableError(f"{path.name} is empty")
    if any(row and row[0].startswith("#") for row in rows):
        raise TableError(f"{path.name} has a # line. The table has one header line and no comments")
    spaced = [cell for row in rows for cell in row if cell != cell.strip()]
    if spaced:
        raise TableError(f"{path.name} has spaces around a value, as in {spaced[0]!r}")
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        raise TableError(f"{path.name} has no column {', '.join(missing)}")
    table = pd.read_csv(path, dtype={c: str for c in TEXT_COLUMNS if c in rows[0]})
    repeated = sorted(table.label[table.label.duplicated()])
    if repeated:
        raise TableError(f"{path.name} has {', '.join(repeated)} more than once. Each label gets one row, and other runs of a window go in their own file")
    table = table.set_index("label")
    limit = table.upper_limit.str.lower().map({"true": True, "false": False})
    if limit.isna().any():
        raise TableError(f"{path.name}: upper_limit has to be true or false, and isn't in {', '.join(table.index[limit.isna()])}")
    table["upper_limit"] = limit.astype(bool)
    for column in REQUIRED[1:4]:
        values = pd.to_numeric(table[column], errors="coerce")
        if (table[column].notna() & values.isna()).any():
            raise TableError(f"{path.name}: {column} isn't a number in {', '.join(table.index[table[column].notna() & values.isna()])}")
        table[column] = values
    measured = ~table.upper_limit
    no_value = measured & (table.mvt_s.isna() | table.mvt_err_lo_s.isna() | table.mvt_err_hi_s.isna())
    if no_value.any():
        raise TableError(f"{path.name}: a measured value needs mvt_s and both errors, and {', '.join(table.index[no_value])} don't have them")
    if (table.upper_limit & (table.mvt_err_lo_s.notna() | table.mvt_err_hi_s.notna())).any():
        raise TableError(f"{path.name}: an upper limit has empty errors, and {', '.join(table.index[table.upper_limit & (table.mvt_err_lo_s.notna() | table.mvt_err_hi_s.notna())])} don't")
    if (table.mvt_s <= 0).any() or (table.mvt_err_lo_s < 0).any() or (table.mvt_err_hi_s < 0).any():
        raise TableError(f"{path.name}: mvt_s has to be above zero, and the errors are distances from it, so they can't be negative")
    if band and (table.band_keV != band).any():
        raise TableError(f"{path.name}: this table is for {band} keV, and {', '.join(table.index[table.band_keV != band])} aren't. Other bands go in their own file")
    for column in ("flags", "sys_note"):
        if column in table:
            table[column] = table[column].fillna("")
    if "trusted" in table:
        table["trusted"] = table.trusted.str.lower()
    return table


def read_epeak_table(path):
    """An E_peak table, from shared/ or from a results folder of the tool, under one set of names. The tool's own flags are about the spectra, so they get their own name."""
    table = pd.read_csv(path, comment="#").set_index("label")
    return table.rename(columns={"Epeak": "Epeak_keV", "flags": "spectral_flags"})


def join_tables(epeak, mvt):
    """An inner join by label, with E_peak's errors put together. Returns the table, and what didn't make it: nothing is dropped without being listed."""
    clash = sorted(set(epeak.columns) & set(mvt.columns))
    if clash:
        raise TableError(f"both tables have a column called {', '.join(clash)}. Rename it in one of them")
    joined = epeak.join(mvt, how="inner")
    too_long = joined.index[joined.mvt_s > joined.t_stop - joined.t_start]
    if len(too_long):
        raise TableError(f"the MVT of {', '.join(too_long)} is longer than its window. mvt_s is in seconds")
    systematic = joined.Epeak_sys.fillna(0.0) if "Epeak_sys" in joined else 0.0
    joined["Epeak_total_lo"] = np.hypot(joined.Epeak_err_lo.fillna(0.0), systematic)
    joined["Epeak_total_hi"] = np.hypot(joined.Epeak_err_hi.fillna(0.0), systematic)
    left_out = dict(no_epeak=sorted(set(mvt.index) - set(epeak.index)), no_mvt=[label for label in epeak.index if label not in mvt.index], no_value=list(joined.index[joined.mvt_s.isna()]))
    # A value whose made-to-measure check hasn't run yet. His rule sends these only as provisional. An upper limit is never checked, so it isn't one of them.
    joined["provisional"] = (~joined.upper_limit) & (joined.trusted == "untested") if "trusted" in joined else False
    left_out["provisional"] = list(joined.index[joined.provisional])
    return joined, left_out


def _range(low, high, margin=1.3):
    """Axis limits that hold every value, with a margin."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    return np.nanmin(low[low > 0]) / margin, np.nanmax(high) * margin


def _point(ax, x, y, y_lo, y_hi, x_lo, x_hi, is_limit, colour, marker, filled=True):
    style = dict(fmt=marker, color=colour, ms=6, lw=1.2, mfc=colour if filled else "white")
    if is_limit:
        ax.errorbar(x, y, xerr=ARROW * x, yerr=[[y_lo], [y_hi]], xuplims=True, **style)
    else:
        ax.errorbar(x, y, xerr=[[x_lo], [x_hi]], yerr=[[y_lo], [y_hi]], **style)


def epeak_against_mvt(bursts, path):
    """E_peak against MVT. bursts is a list of (name, E_peak table, joined table or None, instrument fractions or None).

    Colour is the burst, a square is its precursor, and an open marker is provisional, with its made-to-measure check still to run. E_peak's bars are statistical and systematic, and the pale bars add the instrument term where there is one.
    Dichiara et al.'s MVT is in grey, at my E_peak for the same window. Returns the axis limits.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    xs, ys = [], []
    for (name, epeak, joined, instrument), colour in zip(bursts, BURST_COLOURS):
        published = PUBLISHED.get(name)
        if published is not None:
            for label, row in published.iterrows():
                e = epeak.loc[label]
                lo, hi = np.hypot(e.Epeak_err_lo, e.Epeak_sys), np.hypot(e.Epeak_err_hi, e.Epeak_sys)
                _point(ax, row.mvt_ms, e.Epeak_keV, lo, hi, row.err_ms, row.err_ms, row.upper_limit, "0.6", "s" if label == "pre" else "o", filled=False)
                xs += [row.mvt_ms * (1 - ARROW) if row.upper_limit else row.mvt_ms - row.err_ms, row.mvt_ms + np.nan_to_num(row.err_ms)]
                ys += [e.Epeak_keV - lo, e.Epeak_keV + hi]
        if joined is None or joined.empty:
            continue
        shown = joined[joined.mvt_s.notna()]
        for label, row in shown.iterrows():
            x = 1e3 * row.mvt_s
            marker = "s" if label == "pre" else "o"
            if instrument is not None and label in instrument.index:
                pale = np.hypot(row.Epeak_total_hi, row.Epeak_keV * instrument[label])
                ax.errorbar(x, row.Epeak_keV, yerr=pale, fmt="none", ecolor=colour, alpha=0.3, lw=4)
                ys += [row.Epeak_keV - pale, row.Epeak_keV + pale]
            _point(ax, x, row.Epeak_keV, row.Epeak_total_lo, row.Epeak_total_hi, 1e3 * row.mvt_err_lo_s, 1e3 * row.mvt_err_hi_s, row.upper_limit, colour, marker, filled=not row.provisional)
            if label == "pre" or len(shown) <= SHOW_LABELS_UP_TO:
                ax.annotate(label, (x, row.Epeak_keV), textcoords="offset points", xytext=(6, 5), fontsize=8)
            xs += [x * (1 - ARROW) if row.upper_limit else x - 1e3 * row.mvt_err_lo_s, x + np.nan_to_num(1e3 * row.mvt_err_hi_s)]
            ys += [row.Epeak_keV - row.Epeak_total_lo, row.Epeak_keV + row.Epeak_total_hi]
        ax.plot([], [], "o", color=colour, label=name[:3] + " " + name[3:])
    ax.plot([], [], "s", color="0.3", label="square: precursor")
    if any(joined is not None and "provisional" in joined and joined.provisional.any() for _, _, joined, _ in bursts):
        ax.plot([], [], "o", color="0.3", mfc="white", label="open: provisional, check not run yet")
    if any(name in PUBLISHED for name, *_ in bursts):
        ax.plot([], [], "o", color="0.6", mfc="white", label="Dichiara et al. (2023) Table 1, 8-1000 keV")
    limits = None
    if xs:
        limits = _range(xs, xs), _range(ys, ys)
        ax.set(xscale="log", yscale="log", xlim=limits[0], ylim=limits[1])
        for axis, (low, high) in zip((ax.xaxis, ax.yaxis), limits):
            axis.set_major_locator(FixedLocator([t for t in TICKS if low <= t <= high]))
            axis.set_major_formatter(PLAIN)
            axis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="Minimum variability timescale [ms]", ylabel="E$_{peak}$ [keV]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return limits


def _files_for(name, results=RESULTS):
    """The E_peak table and the MVT table of a burst. Its table in shared/ if it has one, otherwise the results folder of the tool. His table may be in results/timing, or in shared/ once he has cleared it."""
    short = name.replace("GRB", "")
    epeak = SHARED / f"epeak_{short}.csv"
    if not epeak.exists():
        epeak = Path(results) / name / "windows.csv"
    mvt = next((p for p in (TIMING / f"mvt_{short}.csv", SHARED / f"mvt_{short}.csv") if p.exists()), None)
    return epeak, mvt


def main():
    parser = argparse.ArgumentParser(prog="grbtool.joint", description="Join E_peak to the MVT table from Rodrigo's program, and draw them.")
    parser.add_argument("bursts", nargs="+", help="names like GRB230307A")
    parser.add_argument("--out", default=str(RESULTS / "joint"), help="folder for the results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bursts = []
    for name in args.bursts:
        epeak_file, mvt_file = _files_for(name)
        epeak = read_epeak_table(epeak_file)
        instrument_file = SHARED / f"epeak_{name.replace('GRB', '')}_instrument_check.csv"
        instrument = pd.read_csv(instrument_file).set_index("label").Epeak_instrument_frac if instrument_file.exists() else None
        joined = None
        if mvt_file is None:
            print(f"{name}: no MVT table yet")
        else:
            joined, left_out = join_tables(epeak, read_mvt_table(mvt_file))
            joined.to_csv(out / f"joint_{name}.csv", float_format="%.6g")
            print(f"{name}: {len(joined)} windows joined, from {mvt_file}")
            for key, text in (("no_epeak", "in his table but not in the E_peak table"), ("no_mvt", "with no MVT yet"), ("no_value", "with no MVT value, so not drawn"), ("provisional", "provisional, with the check still to run")):
                if left_out[key]:
                    print(f"  {text}: {', '.join(left_out[key])}")
        bursts.append((name, epeak, joined, instrument))
    epeak_against_mvt(bursts, out / "epeak_vs_mvt.png")
    print("wrote", out)


if __name__ == "__main__":
    main()
