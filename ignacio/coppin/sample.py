"""The Coppin et al. sample: their precursors in times from the GBM trigger, the bursts kept, and the light-curve bin of each."""
import numpy as np
import pandas as pd

from . import CATALOGUE, SAMPLE, SHARED, TABLE

LONG_T90_S = 2.0                # long bursts only, by the catalogue's T90
TTE_LEAD_S = {"before_2013": 20.0, "from_2013": 125.0}
# Triggered TTE data start about 29 s before the trigger until 2012 and about 135 s from 2013 (measured on the 111 bursts of the zhu
# sample: 24-33 s and 131-139 s, with a few shorter files). A precursor starting later than -20 s (-125 s) leaves room for a background
# interval before it. The data of each burst decide in the end: zhu.run marks a burst whose emission starts before its data
# not_measurable.


def read_table():
    """Coppin et al.'s precursors: GRB name, Fermi trigger, start of the prompt emission (UTC), the precursor's start from it, its
    duration (s)."""
    return pd.read_csv(TABLE, comment="#", sep=r"\s+", names=["grb", "trigger", "t_prompt_utc", "t_precursor_s", "duration_s"])


def read_catalogue():
    return pd.read_csv(CATALOGUE, comment="#").set_index("trigger_name")


def from_trigger(table, catalogue):
    """Each precursor's start and stop in s from the GBM trigger: the prompt emission's start (UTC, same day as the trigger unless the
    difference is more than half a day) minus the trigger time, plus their times."""
    t = table.join(catalogue, on="trigger")
    seconds = t.t_prompt_utc.map(lambda s: sum(float(x) * f for x, f in zip(s.split(":"), (3600.0, 60.0, 1.0))))
    offset = seconds - (t.trigger_time % 1.0) * 86400.0
    offset = offset.where(offset > -43200.0, offset + 86400.0).where(offset < 43200.0, offset - 86400.0)
    t["prompt_start_s"] = offset
    t["pre_start_s"] = offset + t.t_precursor_s
    t["pre_stop_s"] = t.pre_start_s + t.duration_s
    return t


def zhu_table():
    """Zhu et al.'s Table 1 with each burst's GBM trigger name (bn...)."""
    z = pd.read_csv(SHARED / "zhu2026_table1.csv", comment="#", dtype={"fermi_id": str})
    z["trigger"] = "bn" + z.fermi_id.astype(str).str.replace("bn", "")
    return z


def zhu_triggers():
    z = zhu_table()
    return set(z.trigger[z.fermi_id.notna()])


def tres_rule(catalogue=None):
    """A rule for the light-curve bin, since Coppin et al.'s bursts aren't in Zhu et al.'s table and Zhu et al. don't say how they chose
    theirs. Their Tres follows the burst's brightness: on 343 bursts it goes with the catalogue's 64 ms peak flux (Spearman -0.57) and hardly
    at all with T90. So I fit log2(Tres / 1 ms) as a straight line in log10 of the 64 ms peak flux, using their long bursts. Cross-validated,
    it gives their bin for 56% of bursts and one within a factor of 2 for 95%, where a fixed 16 ms would give 50% and 90%. Returns (slope,
    intercept)."""
    catalogue = read_catalogue() if catalogue is None else catalogue
    z = zhu_table().join(catalogue, on="trigger")
    z = z[(z.t90 > LONG_T90_S) & (z.flux_64 > 0)]
    slope, intercept = np.polyfit(np.log10(z.flux_64), np.log2(1e3 * z.Tres), 1)
    return float(slope), float(intercept)


def tres_for(flux_64, rule):
    """The bin for a burst of this 64 ms peak flux: the rule's step, rounded, within Zhu et al.'s 4-128 ms."""
    step = np.clip(np.round(rule[0] * np.log10(flux_64) + rule[1]), 2, 7)
    return 1e-3 * 2.0 ** step


def build(write=True):
    """The sample: Coppin et al.'s long bursts that aren't in Zhu et al.'s table and whose first precursor the TTE data probably cover,
    one row per burst (the first precursor gives the times; the number of precursors is kept), in the format zhu.run reads."""
    catalogue = read_catalogue()
    t = from_trigger(read_table(), catalogue)
    counts = t.groupby("trigger").size()
    first = t.sort_values("pre_start_s").groupby("trigger").first()
    first["n_precursors"] = counts
    first["in_zhu_table"] = first.index.isin(zhu_triggers())
    first["era"] = np.where(first.index.str[2:4].astype(int) >= 13, "from_2013", "before_2013")
    first["tte_lead_needed_s"] = first.era.map(TTE_LEAD_S)
    first["tte_likely_covers_precursor"] = first.pre_start_s >= -first.tte_lead_needed_s
    keep = (~first.in_zhu_table) & (first.t90 > LONG_T90_S) & first.tte_likely_covers_precursor & (first.flux_64 > 0)
    rule = tres_rule(catalogue)
    s = first[keep].copy()
    s["Tres"] = tres_for(s.flux_64, rule)
    s["reason"] = "coppin2020"
    s.index.name = "trigger"
    s = s.reset_index()
    s["name"] = s.grb.str.replace("GRB", "", regex=False)
    s = s.set_index("name").sort_values("trigger")
    columns = ["trigger", "Tres", "reason", "t90", "t90_start", "flux_64", "fluence", "n_precursors", "prompt_start_s", "pre_start_s",
               "pre_stop_s", "era"]
    s = s[columns]
    s[["prompt_start_s", "pre_start_s", "pre_stop_s"]] = s[["prompt_start_s", "pre_start_s", "pre_stop_s"]].round(3)   # their times are to the ms
    if write:
        with open(SAMPLE, "w") as f:
            f.write("# Coppin et al. (2020, Phys. Rev. D 102, 103014) bursts with a precursor that Zhu et al.'s 2026 table doesn't have: long "
                    f"(T90 > {LONG_T90_S:g} s), the first precursor\n# probably inside the TTE data (from -20 s before 2013, -125 s after). "
                    "Times in s from the GBM trigger: prompt_start_s and pre_start_s/pre_stop_s are their\n# prompt-emission start and first "
                    "precursor (coppin.sample.from_trigger). Tres: the light-curve bin from the 64 ms peak flux, log2(Tres/ms) = "
                    f"{rule[0]:.3f} log10(flux_64) + {rule[1]:.3f},\n# rounded and kept within 4-128 ms (coppin.sample.tres_rule, fitted "
                    "to Zhu et al.'s long bursts). Written by coppin.sample.build.\n")
            s.to_csv(f)
    return s
