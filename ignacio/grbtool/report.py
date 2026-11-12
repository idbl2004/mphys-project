"""A quick look at one burst, to see by eye whether the rules did something sensible: where the windows fell, and E_peak with time."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

COLOURS = {"precursor": "#2a78d6", "main": "#eb6834", "tail": "#1baf7a", "given": "0.55"}
MARKERS = {"Band": "s", "BandLowBreak": "D", "Comptonized": "^"}
PLAIN = FuncFormatter(lambda value, _: f"{value:g}")
MIN_RANGE = 5.0     # the E_peak axis covers at least this factor, so small differences don't look large


def _time_range(windows, blocks):
    """From the first thing that happens to the last, with a margin."""
    starts, stops = list(windows.t_start), list(windows.t_stop)
    if blocks is not None and (blocks.sigma > 3.0).any():
        active = blocks[blocks.sigma > 3.0]
        starts.append(active.start.min())
        stops.append(active.stop.max())
    left, right = min(starts), max(stops)
    margin = 0.05 * (right - left)
    return left - margin, right + margin


def _rate(ax, blocks):
    if blocks is not None:
        ax.stairs(blocks.excess.values / 1e3, np.append(blocks.start.values, blocks.stop.values[-1]), color="0.2", lw=0.8, label="net rate, Bayesian blocks", zorder=3)
    ax.axhline(0.0, color="0.7", lw=0.6)


def _precursor_zoom(ax, window, shown, blocks, bad_intervals):
    """The precursor at its own time scale, with the height set by the precursor and not by what comes after it."""
    width = window.t_stop - window.t_start
    left, right = window.t_start - 1.5 * width, window.t_stop + 1.5 * width
    _rate(ax, blocks)
    for _, row in shown.iterrows():
        ax.axvspan(row.t_start, row.t_stop, color=COLOURS[row.kind], alpha=0.18, lw=0)
    for a, b in bad_intervals or []:
        ax.axvspan(a, b, facecolor="none", edgecolor="0.5", hatch="///", lw=0)
    ax.set_xlim(left, right)
    if blocks is not None:
        inside = blocks[(blocks.stop > window.t_start) & (blocks.start < window.t_stop)]
        before = blocks[(blocks.stop > left) & (blocks.start < window.t_start)]
        low = min(0.0, before.excess.min() if len(before) else 0.0)
        ax.set_ylim(low / 1e3, 1.5 * max(inside.excess.max(), 1.0) / 1e3 if len(inside) else None)
    ax.set(xlabel="Time since GBM trigger [s]", ylabel="Net rate [kcps]")
    ax.set_title("The precursor up close", fontsize=9, loc="left")


def quick_look(burst, table, blocks, bad_intervals, path):
    """Net rate with the windows on top, E_peak below on the same time axis, and the precursor up close. Hollow points aren't measured: the Comptonized doesn't beat a power law there."""
    shown = table[table.kind.isin(COLOURS)]
    precursor = table[table.kind == "precursor"]
    rows = 3 if len(precursor) else 2
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3.0 * rows + 0.8), gridspec_kw={"height_ratios": [1, 1.2, 0.8][:rows]})
    whole, bottom = axes[:2]
    bottom.sharex(whole)

    _rate(whole, blocks)
    given_named = False
    for label, row in shown.iterrows():
        name = f"{label}: {row.t_start:.2f} to {row.t_stop:.2f} s"
        if row.kind == "given":
            name, given_named = (None if given_named else "windows given by hand"), True
        whole.axvspan(row.t_start, row.t_stop, color=COLOURS[row.kind], alpha=0.1 if row.kind == "given" else 0.18, lw=0, label=name)
    for i, (a, b) in enumerate(bad_intervals or []):
        for ax in (whole, bottom):
            ax.axvspan(a, b, facecolor="none", edgecolor="0.5", hatch="///", lw=0, label="left out" if i == 0 and ax is whole else None)
    whole.set_ylabel("Net rate, NaI, 8-1000 keV [kcps]")
    whole.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False, borderaxespad=0.2)
    whole.set_xlim(*_time_range(shown if len(shown) else table, blocks))

    points = table[(table.kind.isin(["precursor", "slice", "given"])) | ((table.kind == "main") & ~(table.kind == "slice").any())]
    points = points[points.Epeak.notna()]
    any_loose = False
    for model, marker in MARKERS.items():
        part = points[points.model == model]
        if part.empty:
            continue
        kept = part[part.measured.astype(bool)]
        loose = part[~part.measured.astype(bool)]
        any_loose = any_loose or len(loose) > 0
        for chosen, face in ((kept, "#2a78d6"), (loose, "none")):
            if len(chosen):
                bottom.errorbar(0.5 * (chosen.t_start + chosen.t_stop), chosen.Epeak, xerr=0.5 * (chosen.t_stop - chosen.t_start), yerr=[chosen.Epeak_err_lo.fillna(0.0), chosen.Epeak_err_hi.fillna(0.0)],
                                fmt=marker, ms=5, mfc=face, mec="#2a78d6", ecolor="#2a78d6", elinewidth=0.8, label=model if face != "none" or kept.empty else None)
    if any_loose:
        bottom.plot([], [], "o", mfc="none", mec="#2a78d6", ms=5, label="hollow: not measured")
    bottom.set(yscale="log", ylabel="E$_{peak}$ [keV], chosen model", xlabel="Time since GBM trigger [s]")
    bottom.grid(alpha=0.25, lw=0.5)
    if len(points):
        bottom.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False, borderaxespad=0.2)
        low, high = bottom.get_ylim()
        if high / low < MIN_RANGE:
            middle = np.sqrt(low * high)
            low, high = middle / np.sqrt(MIN_RANGE), middle * np.sqrt(MIN_RANGE)
            bottom.set_ylim(low, high)
        bottom.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        bottom.yaxis.set_major_formatter(PLAIN)
        bottom.yaxis.set_minor_formatter(NullFormatter())

    if len(precursor):
        _precursor_zoom(axes[2], precursor.iloc[0], shown, blocks, bad_intervals)
    fig.suptitle(f"{burst.name}, {' '.join(burst.detectors)}" + (f", z = {burst.redshift}" if burst.redshift else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def joint_table(table, timing):
    """E_peak and MVT of the same windows, joined by label. The MVT is the value to quote, at the smallest bin that ran."""
    if "quoted" not in timing:
        return table.iloc[0:0]
    quoted = timing[timing.quoted.fillna(False).astype(bool)].set_index("label")
    columns = ["detectors", "bin_ms", "mvt_ms", "mvt_p16_ms", "mvt_p84_ms", "n_valid", "n_limit", "n_failed", "snr_at_mvt", "snr_needed", "above_injection_threshold", "class", "quoted_ms", "quoted_is_limit", "plateau"]
    keep = ["kind", "t_start", "t_stop", "model", "Epeak", "Epeak_err_lo", "Epeak_err_hi", "Epeak_sys", "Epeak_band", "Epeak_cpl", "measured", "flags"]
    return table[[c for c in keep if c in table]].join(quoted[[c for c in columns if c in quoted]].rename(columns={"detectors": "mvt_detectors"}), how="inner")


def epeak_vs_mvt(burst, joint, path):
    """E_peak against MVT, one point per window, with upper limits on the MVT as arrows. E_peak bars are statistical and systematic together."""
    joint = joint[joint.quoted_ms.notna()] if "quoted_ms" in joint else joint.iloc[0:0]
    if joint.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sys_error = joint.Epeak_sys.fillna(0.0) if "Epeak_sys" in joint else 0.0
    for label, row in joint.iterrows():
        colour, marker = (COLOURS["precursor"], "s") if row.kind == "precursor" else (COLOURS["main"], "o")
        extra = sys_error[label] if "Epeak_sys" in joint else 0.0
        y_error = [[np.hypot(np.nan_to_num(row.Epeak_err_lo), extra)], [np.hypot(np.nan_to_num(row.Epeak_err_hi), extra)]]
        if bool(row.quoted_is_limit):
            ax.errorbar(row.quoted_ms, row.Epeak, xerr=0.25 * row.quoted_ms, yerr=y_error, xuplims=True, fmt=marker, color=colour, ms=6)
        else:
            ax.errorbar(row.quoted_ms, row.Epeak, xerr=[[row.mvt_ms - row.mvt_p16_ms], [row.mvt_p84_ms - row.mvt_ms]], yerr=y_error, fmt=marker, color=colour, ms=6)
        ax.annotate(label, (row.quoted_ms, row.Epeak), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.plot([], [], "s", color=COLOURS["precursor"], label="precursor")
    ax.plot([], [], "o", color=COLOURS["main"], label="other windows")
    ax.set(xscale="log", yscale="log", xlabel="Minimum variability timescale [ms]", ylabel="E$_{peak}$ [keV], chosen model", title=burst.name)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(PLAIN)
        axis.set_minor_formatter(PLAIN)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
