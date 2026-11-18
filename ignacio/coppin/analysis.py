"""Coppin et al.'s precursors against ours, and the precursor's spectrum against its main emission, for any set of measured bursts.

table(root, triggers) gives one row per burst: Coppin et al.'s first precursor (in times from the GBM trigger), ours under the zhu2026 rule
(found or not, and its edges), whether the two start together, and our E_peak of the precursor and of the main emission with their
status. Two precursor starts count as agreeing when they differ by less than STARTS_AGREE_S or two light-curve bins, whichever is bigger.
Their Bayesian blocks and ours run on different light curves (theirs on all the NaI with 1000 s of background, ours on the catalogue's NaI
at the bin from coppin.sample.tres_rule), so a block edge can easily move by a bin or two."""
import numpy as np
import pandas as pd

from . import RESULTS, SHARED
from .sample import from_trigger, read_catalogue, read_table

STARTS_AGREE_S = 0.2


def theirs():
    """Coppin et al.'s first precursor of each burst, from the trigger, with the number of precursors they list."""
    t = from_trigger(read_table(), read_catalogue())
    first = t.sort_values("pre_start_s").groupby("trigger").first()
    first["n_precursors"] = t.groupby("trigger").size()
    return first[["grb", "pre_start_s", "pre_stop_s", "duration_s", "prompt_start_s", "n_precursors", "t90", "flux_64"]]


def table(root=RESULTS, names=None):
    """One row per burst measured in root (zhu.run's layout), joined to Coppin et al.'s precursor by trigger."""
    from zhu.features import collect
    ours = collect("zhu2026", root, names)
    ours = ours[ours.get("status", "complete") == "complete"] if "status" in ours else ours
    c = theirs()
    rows = []
    for name, o in ours.iterrows():
        trigger = o.get("trigger")
        if trigger not in c.index:
            continue
        k = c.loc[trigger]
        tres = float(o.get("Tres", np.nan))
        found = bool(pd.notna(o.get("pe_start")))
        row = dict(name=name, trigger=trigger, Tres_s=tres, their_start_s=k.pre_start_s, their_stop_s=k.pre_stop_s,
                   their_n=int(k.n_precursors), found=found)
        if found:
            row.update(our_start_s=o.pe_start, our_stop_s=o.pe_stop, delta_start_s=o.pe_start - k.pre_start_s,
                       delta_stop_s=o.pe_stop - k.pre_stop_s)
            row["starts_agree"] = bool(abs(row["delta_start_s"]) <= max(STARTS_AGREE_S, 2 * tres))
        measured_pe, measured_me = str(o.get("measured_PE")) == "True", str(o.get("measured_ME")) == "True"
        row.update(Ep_PE_keV=o.get("Ep_PE_ours") if measured_pe else np.nan, Ep_PE_measured=measured_pe,
                   Ep_PE_95_lo_keV=np.nan if measured_pe else o.get("Ep_PE_cpl_limit_lo"),
                   Ep_PE_95_hi_keV=np.nan if measured_pe else o.get("Ep_PE_cpl_limit_hi"),
                   Ep_ME_keV=o.get("Ep_ME_ours") if measured_me else np.nan, Ep_ME_measured=measured_me)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.set_index("name")
        out["Ep_ratio"] = out.Ep_PE_keV / out.Ep_ME_keV
    return out


def summary(t):
    """Counts: precursors found by our rule, starts that agree, E_peak measured."""
    found = t[t.found]
    return dict(bursts=len(t), found=int(t.found.sum()), starts_agree=int(found.starts_agree.sum()),
                median_abs_delta_start_s=float(found.delta_start_s.abs().median()) if len(found) else np.nan,
                Ep_PE_measured=int(t.Ep_PE_measured.sum()), Ep_both_measured=int((t.Ep_PE_measured & t.Ep_ME_measured).sum()),
                median_ratio=float(t.Ep_ratio.median()) if t.Ep_ratio.notna().any() else np.nan)


def trigger_met(name, trigger):
    """The GBM trigger time (MET) of a burst, read from the header of one of its TTE files."""
    from astropy.io import fits
    from grbtool.bursts import DATA_ROOT
    from zhu.catalogue import data_name
    tte = sorted((DATA_ROOT / data_name(name, DATA_ROOT)).rglob(f"glg_tte_n*_{trigger}_v*.fit"))
    with fits.open(tte[0]) as f:
        return float(f[0].header["TRIGTIME"])


def windows(root=RESULTS, mode="zhu2026"):
    """The precursor and main-emission windows of every Coppin burst in which our rule finds a precursor, in the format of
    shared/windows_zhu_sample.csv, so that the MVT can be measured on exactly the windows of our E_peak."""
    import json
    from zhu.io import states
    sample = pd.read_csv(SHARED / "coppin_sample.csv", comment="#", index_col=0)
    status = states(sample.index, root)
    rows = []
    for name in sample.index[status.status == "complete"]:
        record = json.loads((root / name / "episodes.json").read_text())
        found = record["modes"][mode]
        if not found.get("pe") or not found.get("me"):
            continue
        t0 = trigger_met(name, sample.loc[name, "trigger"])
        for label, key in (("pre", "pe"), ("main", "me")):
            a, b = found[key]
            limit = [f"{x:.1f}-{y:.1f}" for x, y in record.get("bad_intervals", []) if x < b and y > a]
            rows.append(dict(burst=f"GRB{name}", trigger=sample.loc[name, "trigger"], T0_met=f"{t0:.6f}", label=label,
                             t_start=round(a, 6), t_stop=round(b, 6), Tres_s=record["Tres"], detectors=" ".join(record["nai"]),
                             source=f"{mode} rule, code {record['code_version']}", telemetry_limit_inside=" ".join(limit),
                             background_flags=" ".join(record.get("bkg_flags", [])), flags=" ".join(found.get("flags", []))))
    return pd.DataFrame(rows)
