"""Does the tool give back what the notebooks gave? Compares a results folder with the tables in shared/."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import CODE_DIR

SHARED = CODE_DIR / "shared"


def against_table(results, table_file):
    """Band and Comptonized E_peak, Band C-stat and flux, window by window. Differences are in units of the table's error on E_peak."""
    mine = pd.read_csv(Path(results) / "windows.csv", index_col="label")
    table = pd.read_csv(table_file, index_col="label")
    rows = []
    for label in table.index.intersection(mine.index):
        m, t = mine.loc[label], table.loc[label]
        error = 0.5 * (t.Epeak_err_lo + t.Epeak_err_hi)
        rows.append(dict(label=label, model_table=t.model, model_rule=m.model, Ep_table=t.Epeak_keV, Ep_rule=m.Epeak, Ep_err_table=error, Ep_err_rule=0.5 * (m.Epeak_err_lo + m.Epeak_err_hi),
                         Ep_band=m.Epeak_band, Ep_cpl_table=t.Epeak_cpl_keV, Ep_cpl=m.Epeak_cpl, dCpl_pct=100 * (m.Epeak_cpl / t.Epeak_cpl_keV - 1),
                         C_table=t.cstat, C_rule=m.cstat, C_band=m.C_band, C_cpl=m.C_cpl,
                         dC_cpl_band_table=t.dC_cpl_minus_band, dC_cpl_band=m.C_cpl - m.C_band, flux_table=t.flux_10_1000keV_erg_cm2_s, flux_rule=m.flux, flux_band=m.flux_band, sys_table=t.Epeak_sys, sys_rule=m.get("Epeak_sys", np.nan)))
    return pd.DataFrame(rows).set_index("label")


def against_sample(names, results_root=None):
    """The precursor and main emission of the sample bursts against notebook 18: windows, Comptonized E_peak, significance and whether it counts as measured."""
    fits = pd.read_csv(SHARED / "sample_fits.csv")
    fits = fits[fits.episode.isin(["pre", "main"])]
    rows = []
    for name in names:
        path = Path(results_root or CODE_DIR / "results") / name / "windows.csv"
        if not path.exists():
            continue
        mine = pd.read_csv(path, index_col="label")
        for _, ref in fits[fits.burst == name].iterrows():
            if ref.episode not in mine.index:
                rows.append(dict(burst=name, episode=ref.episode, note="not found by the tool"))
                continue
            m = mine.loc[ref.episode]
            error = 0.5 * (ref.Ep_err_lo + ref.Ep_err_hi)
            same = abs(m.t_start - ref.t_start) < 1e-3 and abs(m.t_stop - ref.t_stop) < 5e-3
            rows.append(dict(burst=name, episode=ref.episode, same_window=same, nb18_window=f"{ref.t_start:g} to {ref.t_stop:g}", tool_window=f"{m.t_start:g} to {m.t_stop:g}", Ep_nb18=ref.Ep, err_nb18=error, Ep_tool=m.Epeak_cpl,
                             diff_in_sigma=(m.Epeak_cpl - ref.Ep) / error if error > 0 else np.nan, C_nb18=ref.C_cpl, C_tool=m.C_cpl, sigma_nb18=ref.sigma, sigma_tool=m.sigma, measured_nb18=ref.measured, measured_tool=m.measured,
                             rule_model=m.model, rule_Ep=m.Epeak, sys=m.get("Epeak_sys", np.nan), flags=m.get("flags", "")))
    return pd.DataFrame(rows)


def main(argv):
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    if argv and argv[0] == "--sample":
        print(against_sample(argv[1:]).round(2).to_string(index=False))
        return
    for name in argv:
        table = against_table(CODE_DIR / "results" / "preset" / name, SHARED / f"epeak_{name[3:]}.csv")
        same_model = table.model_table == table.model_rule
        same = table[same_model]
        print(f"\n{name}: {len(table)} windows, the rule picks the table's model in {same_model.sum()}")
        print(table[["model_table", "model_rule", "Ep_table", "Ep_rule", "Ep_err_table", "Ep_err_rule", "Ep_band", "Ep_cpl_table", "Ep_cpl", "C_table", "C_rule", "C_band", "C_cpl", "sys_table", "sys_rule"]].round(2).to_string())
        print("Comptonized E_peak against the table, largest difference: "
              f"{table.dCpl_pct.abs().max():.3f}% in {table.dCpl_pct.abs().idxmax()}")
        if len(same):
            print(f"same model: E_peak differs by at most {(same.Ep_rule - same.Ep_table).abs().div(same.Ep_err_table).max():.3f} of its error, C-stat by at most {(same.C_rule - same.C_table).abs().max():.3f}, "
                  f"flux by at most {100 * (same.flux_rule / same.flux_table - 1).abs().max():.3f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
