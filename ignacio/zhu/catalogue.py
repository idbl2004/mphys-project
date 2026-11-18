"""Their tables, the sample we measure, their index EPI, and redshifts."""
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import SHARED

TABLE_2026 = SHARED / "zhu2026_table1.csv"
REDSHIFTS = SHARED / "redshifts_greiner.csv"
EPI_LINE = 6.2              # Zhu et al. 2026: Type I above, Type II below
EPI_UNSURE = (6.0, 6.4)     # where they found both kinds

# The 29 three-episode bursts of Zhu et al. 2025. 12 are labelled in their figures. The other 17 are unlabelled stars in their
# T100,PE against T_wt1 figure: reading the star centres off the vector file and looking for the nearest row of the 2026 table
# gives one burst each, all 0.004 dex away and the next nearest at least 0.012 dex (notebook 22 repeats the matching).
SAMPLE_2025 = ["090131A", "090618A", "090831A", "091010A", "110825A", "111228A", "140206B", "140810A", "160509A", "170228A", "170409A",
               "180605A", "180731A", "181117A", "190114C", "200101A", "200311A", "200914A", "210606B", "211019A", "211023A", "211211A",
               "220305A", "220624A", "230307A", "231104A", "240403A", "241029A", "241030A"]
KILONOVA = ["211211A", "230307A"]                    # GRB 060614 too, but GBM didn't see it
SUPERNOVA = ["090618A", "111228A", "180728A", "190114C", "190829A", "211023A"]
CONTROL_SEED = 20261111
# Redshifts the project already uses, with their sources. They come first, because the Greiner table has none for the two kilonova bursts.
LITERATURE_Z = {"230307A": (0.065, "Dichiara et al. 2023"), "211211A": (0.076, "Gompertz et al. 2023; Rastinejad et al. 2022"),
                "180728A": (0.117, "Starling et al. 2026"), "190829A": (0.0785, "notebook 18"), "190114C": (0.4245, "notebook 18"),
                "090618A": (0.54, "notebook 18"), "111228A": (0.716, "notebook 18"), "211023A": (0.390, "notebook 18")}

# The two rows of Zhu et al. 2025's Table 2 printed in the paper (the rest is only in the journal's machine-readable file): start and
# end of each episode in s from the trigger, their Tres in s, and T_MVT of each episode.
BOUNDARIES_2025 = {
    "090131A": dict(Tres=0.004, PE=(-0.276, 0.316), ME=(2.096, 9.78), EE=(22.476, 84.04), WE=(-0.276, 84.04),
                    TMVT=dict(PE=0.102, ME=0.974, EE=0.594, WE=0.102)),
    "090618A": dict(Tres=0.008, PE=(-0.632, 38.504), ME=(49.992, 99.144), EE=(107.616, 194.648), WE=(-0.632, 194.648),
                    TMVT=dict(PE=1.644, ME=4.284, EE=7.244, WE=1.644)),
}

# The same two bursts in their 2025 spectral tables (Tables 3-4): E_p in keV with its minus and plus errors for each model, and their BIC.
# Their BIC values are offset by a constant, but the differences between models match PGSTAT + k ln N. The 2026 table's E_p,ME is the
# lowest-BIC model here (CPL+BB for 090131, Band+BB for 090618).
SPECTRA_2025 = {
    ("090131A", "PE"): {"band": (1707.19, 950.91, 1073.06, 1.89), "cpl": (1769.41, 1030.29, 1041.45, -0.34), "band_bb": (1855.68, 1492.06, 1259.22, 8.10),
                        "cpl_bb": (1031.17, 641.54, 641.19, 5.22), "pl_bb": (None, None, None, -0.58)},
    ("090618A", "PE"): {"band": (161.38, 17.42, 19.19, -37.04), "cpl": (203.96, 15.44, 15.51, -17.92), "band_bb": (166.36, 23.89, 21.80, -28.09),
                        "cpl_bb": (320.39, 76.34, 80.76, -24.44), "pl_bb": (None, None, None, 54.27)},
    ("090131A", "ME"): {"band": (48.67, 10.32, 10.15, 296.34), "cpl": (53.87, 5.65, 5.63, 295.33), "band_bb": (62.00, 15.25, 15.64, 289.63),
                        "cpl_bb": (68.15, 15.79, 16.26, 288.60)},
    ("090618A", "ME"): {"band": (172.84, 6.86, 6.77, 129.03), "cpl": (224.99, 5.49, 5.45, 291.95), "band_bb": (170.16, 12.99, 12.14, 118.49),
                        "cpl_bb": (298.95, 19.08, 19.46, 145.55)},
}

COLUMNS_2026 = [("GRB", 0, 7), ("fermi_id", 8, 17), ("Tres", 18, 23), ("T100_PE", 24, 31), ("T100_ME", 32, 39), ("T100_WE", 40, 47),
                ("T100_QE1", 48, 55), ("TMVT_PE", 56, 62), ("TMVT_ME", 63, 69), ("Ep_ME", 70, 77), ("tau32_ME_ms", 78, 85),
                ("tau32_WE_ms", 86, 93), ("T90", 94, 100), ("Type", 101, 104)]


def epi(ep_me, t100_pe, t100_qe1, tmvt_pe):
    """EPI = log10[(E_p,ME/keV)^2 / (T100,PE T100,QE1^(1/2) T_MVT,PE)], times in seconds (Zhu et al. 2026, eq. 2)."""
    return np.log10(np.asarray(ep_me, float) ** 2 / (np.asarray(t100_pe, float) * np.sqrt(np.asarray(t100_qe1, float)) * np.asarray(tmvt_pe, float)))


def parse_mrt(path):
    """Their machine-readable Table 1 (fixed-width text from the journal) as a table. Used once, to write TABLE_2026."""
    lines = [line.rstrip("\n") for line in open(path) if line[:2].isdigit()]
    table = pd.DataFrame([{name: line[a:b].strip() for name, a, b in COLUMNS_2026} for line in lines])
    for name, _, _ in COLUMNS_2026[2:-1]:
        table[name] = pd.to_numeric(table[name])
    return table


def read_table_2026(path=TABLE_2026):
    """Zhu et al. 2026, Table 1, indexed by burst name, with the trigger and their EPI added."""
    table = pd.read_csv(path, comment="#", dtype={"fermi_id": str}).set_index("GRB")
    table["trigger"] = ["bn" + i if isinstance(i, str) and i else None for i in table.fermi_id]
    table["merger"] = table.Type.str.startswith("I-")
    table["EPI"] = epi(table.Ep_ME, table.T100_PE, table.T100_QE1, table.TMVT_PE)
    return table


def build_sample(table, near_line=5.0, n_random=20, seed=CONTROL_SEED):
    """The bursts I measure, each with the reason it's in. z25: the 29 bursts of the 2025 paper. type_I: every other GBM burst they call I-S
    or I-L. supernova: the bursts with a supernova. near_line: Type II bursts with their EPI at least near_line, where a different measurement
    could move a burst across 6.2. random: a seeded draw from the other Type II, so the controls aren't only the hard cases."""
    gbm = table[table.trigger.notna()]
    reasons = {}
    for name in SAMPLE_2025:
        reasons[name] = "z25"
    for name in gbm.index[gbm.merger]:
        reasons.setdefault(name, "type_I")
    for name in SUPERNOVA:
        reasons.setdefault(name, "supernova")
    for name in gbm.index[(gbm.Type == "II") & (gbm.EPI >= near_line)]:
        reasons.setdefault(name, "near_line")
    rest = gbm.index[(gbm.Type == "II") & ~gbm.index.isin(list(reasons))]
    for name in np.random.default_rng(seed).choice(sorted(rest), size=n_random, replace=False):
        reasons[name] = "random"
    sample = gbm.loc[list(reasons), ["trigger", "Type", "Tres", "EPI"]].copy()
    sample["reason"] = [reasons[n] for n in sample.index]
    return sample


def parse_greiner(path):
    """Redshifts from a saved copy of J. Greiner's GRB table (www.mpe.mpg.de/~jcg/grbgen.html).

    Its names have no letter when a burst was alone that day, and an S added after short bursts. Only plain numbers are kept as
    redshifts. A value marked as photometric (ph) is kept apart, and anything with '?', '<' or '>' is left out.
    """
    text = open(path, errors="ignore").read()
    rows = []
    for row in re.findall(r"<TR VALIGN=\"TOP\">(.*?)(?=<TR|</TABLE>)", text, re.S | re.I):
        cells = [html.unescape(re.sub(r"<.*?>", "", c)).replace("\xa0", " ").strip() for c in re.findall(r"<TD.*?>(.*?)</TD>", row, re.S | re.I)]
        if len(cells) < 10 or not re.match(r"\d{6}", cells[0]):
            continue
        name = cells[0][:-1] if cells[0].endswith("S") and len(cells[0]) > 6 else cells[0]
        z_text = cells[-1]
        number = re.match(r"^\s*(\d+\.\d+|\d+)\s*([a-z]*)\s*$", z_text)
        if not number:
            continue
        rows.append(dict(name=name, z=float(number.group(1)), z_note=number.group(2), photometric="ph" in number.group(2)))
    return pd.DataFrame(rows).drop_duplicates("name")


def redshift_of(name, redshifts):
    """The redshift of one of their burst names (like 090618A): the project's own value if there is one, else the Greiner table, else None. Photometric ones are skipped."""
    if name in LITERATURE_Z:
        return LITERATURE_Z[name][0]
    found = redshifts.set_index("name")
    for candidate in (name, name[:-1] if name.endswith("A") else None):
        if candidate and candidate in found.index and not found.loc[candidate, "photometric"]:
            return float(found.loc[candidate, "z"])
    return None


def data_name(name, root):
    """The folder name for a burst's data. Folders from earlier notebooks are named without a final A (GRB090618), so those are reused."""
    for candidate in (f"GRB{name}", f"GRB{name[:-1]}" if name.endswith("A") else None):
        if candidate and (Path(root) / candidate / "current").is_dir():
            return candidate
    return f"GRB{name}"


# ---------------------------------------------------------------------------------------------------------------------------------------
# The literature table: every published number this study uses, with where it comes from. Built once with build_literature_table() from
# their Table 1 (the ApJ machine-readable file, already in TABLE_2026) and the LaTeX source of the 2025 paper (arXiv:2506.08675), and saved
# as LITERATURE; the notebooks read the saved file.

LITERATURE = SHARED / "zhu_literature_values.csv"
PAPER_2025 = "Zhu et al. 2025, MNRAS 541, 3236 (arXiv:2506.08675)"
PAPER_2026 = "Zhu et al. 2026, ApJ 1003, 142 (arXiv:2604.19182)"
DEFINITIONS_2026 = {
    "Tres": ("s", "time resolution of the light curve the Bayesian blocks were run on, chosen per burst"),
    "T100_PE": ("s", "duration of the precursor emission, from Bayesian-block edges"),
    "T100_ME": ("s", "duration of the main emission"),
    "T100_WE": ("s", "duration of the whole emission, from the start of the PE to the end of the last episode"),
    "T100_QE1": ("s", "duration of the quiescent episode between PE and ME"),
    "TMVT_PE": ("s", "rise time of the shortest pulse in the PE identified with Bayesian blocks (minimum variability timescale)"),
    "TMVT_ME": ("s", "the same for the ME"),
    "Ep_ME": ("keV", "nuFnu peak energy of the ME spectrum, lowest-BIC model with all parameters constrained"),
    "tau32_ME_ms": ("ms", "spectral lag of 25-50 keV behind 50-100 keV in the ME, from the CCF peak"),
    "tau32_WE_ms": ("ms", "the same for the whole emission"),
    "T90": ("s", "GBM catalogue T90"),
    "Type": ("", "their class from HDBSCAN on t-SNE/UMAP maps of the nine numbers: I-S, I-L or II"),
}
METHOD_2025 = [
    ("p0", 0.05, "", "false-positive rate of a Bayesian-block change point", "Section 2.1 text"),
    ("band_keV", "8-1000", "keV", "light-curve energy band (GBM Data Tools)", "Section 2.1 text"),
    ("qe_min_bins", 10, "bins", "a quiescent episode lasts at least 10 times the time resolution", "Section 2.1 text"),
    ("qe1_snr_max", 3.0, "sigma", "SNR of the quiescent episode between PE and ME (3.15 allowed for GRB 230307A)", "Section 2.1 text"),
    ("lag_bands_keV", "tau32: 50-100 vs 25-50; tau42: 100-200 vs 25-50", "keV", "spectral-lag bands", "Section 2.1 text"),
    ("lag_max_resolution", 0.256, "s", "no lag if the CCF can't be reliably fitted at 256 ms or finer", "Section 2.1 text"),
    ("nai_keV", "10-900, without 30-40", "keV", "NaI spectral range (iodine K edge removed)", "Section 2.2 text"),
    ("bgo_keV", "300-40000; 300-10000 for PE", "keV", "BGO spectral range", "Section 2.2 text"),
    ("statistic", "PGSTAT, MCMC 1e5 iterations, last 50%", "", "spectral fitting in PyXspec", "Section 2.2 text"),
    ("models", "Band, CPL, Band+BB, CPL+BB, PL+BB", "", "spectral models; lowest BIC = -2lnL + k lnN with all parameters constrained", "Section 2.2 text"),
    ("tsne_perplexity", 5, "", "t-SNE perplexity on 12 log-scaled, z-scored numbers", "Section 3 text"),
    ("umap", "n_neighbors 5, min_dist 0.001", "", "UMAP settings", "Section 3 text"),
]
METHOD_2026 = [
    ("qe_min_bins", 8, "bins", "a QE lasts at least 8 times the time resolution and is a single Bayesian block", "Section 2.1 text"),
    ("qe1_snr_max", 3.0, "sigma", "the QE is 'relatively stable emission close to the background, i.e., SNR below around 3 sigma'", "Section 2.1 text"),
    ("pe_peak_sigma", 3.0, "sigma", "the PE peak exceeds the background by at least 3 sigma (when the QE isn't at the background)", "Section 2.1 text"),
    ("tsne_perplexity", 6, "", "t-SNE perplexity on 9 numbers (log of all but the two lags), z-scored", "Section 3 text"),
    ("umap", "n_neighbors 6, min_dist 1e-4", "", "UMAP settings", "Sections 2.3 and 3 text"),
    ("clustering", "HDBSCAN (scikit-learn); settings not given", "", "clusters on the maps", "Section 2.3 text"),
    ("stability", "perplexity and n_neighbors 5, 6, 7 and random initialisations: consistent", "", "their stability check", "Section 3 text"),
    ("EPI_line", "6.2 +/- 0.2", "", "Type I above, Type II below; 6.0-6.4 is a mixed zone (three Type II and one Type I)", "Section 4 and Conclusions"),
    ("I_L_rule", "T100,ME > 2 s and T100,WE > 10 s", "s", "Type I-L among Type I", "Section 3 text and Appendix C"),
]
# Section 3 of the 2025 paper, the six Type IL candidates (durations rounded to 0.01 s in the text)
TEXT_2025 = {
    "090831A": dict(T100_PE=0.03, T100_ME=0.28, T100_EE=76.28, T100_WE=80.0, T_wt1=0.24, tau42_ME_ms=(41.71, 3.02), tau42_WE_ms=(30.52, 1.99), TMVT_WE=0.014, TMVT_ME=0.062),
    "170228A": dict(T100_PE=0.32, T100_ME=28.51, T100_EE=81.18, T100_WE=111.87, T_wt1=1.31, tau42_ME_ms=(72.65, 4.12), tau42_WE_ms=(80.15, 10.47), TMVT_WE=0.016, TMVT_ME=0.144),
    "180605A": dict(T100_PE=0.47, T100_ME=0.46, T100_EE=27.85, T100_WE=30.43, T_wt1=0.29, tau42_ME_ms=(-1.33, 1.36), tau42_WE_ms=(21.25, 1.27), TMVT_WE=0.068, TMVT_PE=0.108),
    "200311A": dict(T100_PE=0.21, T100_ME=17.14, T100_EE=47.73, T100_WE=69.22, T_wt1=0.3, tau42_ME_ms=(15.84, 1.97), tau42_WE_ms=(-0.73, 2.96), TMVT_WE=0.056, TMVT_PE=0.072),
    "200914A": dict(T100_PE=0.54, T100_ME=21.25, T100_EE=35.76, T100_WE=62.14, T_wt1=1.23, tau42_ME_ms=(162.63, 48.37), tau42_WE_ms=(76.39, 22.53), TMVT_PE=0.040, TMVT_ME=0.248),
    "211019A": dict(T100_PE=0.76, T100_ME=13.59, T100_EE=54.44, T100_WE=70.54, T_wt1=1.21, tau42_ME_ms=(70.52, 5.03), tau42_WE_ms=(84.89, 1.01), TMVT_PE=0.020, TMVT_ME=0.076),
}
LABELLED_2025 = ["090618A", "111228A", "190114C", "211023A", "211211A", "230307A", "090831A", "170228A", "180605A", "200311A", "200914A", "211019A"]


def _value_with_errors(cell):
    """'$-1.36_{-0.12}^{+0.12}$' -> (-1.36, 0.12, 0.12); '345.52/298' -> (345.52, nan, nan) with dof kept apart; '--' -> NaN."""
    cell = cell.strip().strip("$").replace("\\\\", "").strip()
    m = re.match(r"^(-?[\d.]+)_\{-([\d.]+)\}\^\{\+([\d.]+)\}$", cell)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = re.match(r"^(-?[\d.]+)\s*\\pm\s*([\d.]+)$", cell)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(2))
    try:
        return float(cell), np.nan, np.nan
    except ValueError:
        return np.nan, np.nan, np.nan


def build_literature_table(tex_2025, path=LITERATURE):
    """Every published number used here, one row each: paper, where, burst, quantity, value, minus/plus error, units, definition, and how
    it was obtained (tabulated, printed in the text, or read from a figure). tex_2025 is the LaTeX source of the 2025 paper."""
    rows = []
    add = lambda **r: rows.append(dict(dict(burst="", minus=np.nan, plus=np.nan, units="", definition="", source="tabulated"), **r))
    table = read_table_2026()
    for name, r in table.iterrows():
        for quantity, (units, definition) in DEFINITIONS_2026.items():
            add(paper=PAPER_2026, where="Table 1 (machine-readable)", burst=name, quantity=quantity, value=r[quantity], units=units, definition=definition)
    for quantity, value, units, definition, where in METHOD_2026:
        add(paper=PAPER_2026, where=where, quantity=quantity, value=value, units=units, definition=definition, source="text")
    for quantity, value, units, definition, where in METHOD_2025:
        add(paper=PAPER_2025, where=where, quantity=quantity, value=value, units=units, definition=definition, source="text")
    text = open(tex_2025, errors="ignore").read()
    # Table 2 (temporal properties): the two rows printed in the paper
    i = text.find("The temporal properties of three-episode GRBs")
    block = text[i:text.find("\\end{table", i)]
    burst = None
    for line in block.splitlines():
        cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
        if len(cells) != 12 or cells[4] not in ("PE", "ME", "EE", "WE"):
            continue
        if cells[0]:
            burst = cells[0] + "A"
            add(paper=PAPER_2025, where="Table 2 (printed rows)", burst=burst, quantity="Tres", value=float(cells[1]) / 1e3, units="s", definition="time resolution")
            add(paper=PAPER_2025, where="Table 2 (printed rows)", burst=burst, quantity="T_wt1", value=float(cells[2]), units="s", definition="quiescent episode PE-ME")
            add(paper=PAPER_2025, where="Table 2 (printed rows)", burst=burst, quantity="T_wt2", value=float(cells[3]), units="s", definition="quiescent episode ME-EE")
        ep = cells[4]
        for k, (q, units, d) in {5: ("start", "s", "episode start from the trigger"), 6: ("stop", "s", "episode end"), 7: ("T100", "s", "episode duration"),
                                 8: ("TMVT", "s", "minimum variability timescale"), 9: ("tau_res", "ms", "time resolution of the lag"),
                                 10: ("tau32", "ms", "lag 25-50 behind 50-100 keV"), 11: ("tau42", "ms", "lag 25-50 behind 100-200 keV")}.items():
            v, lo, hi = _value_with_errors(cells[k])
            add(paper=PAPER_2025, where="Table 2 (printed rows)", burst=burst, quantity=f"{q}_{ep}", value=v, minus=lo, plus=hi, units=units, definition=d)
    # Tables 3-6 (spectral fits): the two rows printed in each
    for episode, caption in (("PE", "precursor emissions"), ("ME", "main emissions"), ("EE", "extended emissions"), ("WE", "whole emissions")):
        i = text.find(f"The spectral fitting results for {caption}")
        block = text[i:text.find("\\end{table", i)]
        burst = None
        for line in block.splitlines():
            cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
            if len(cells) < 10 or cells[1] not in ("Band", "CPL", "Band+BB", "CPL+BB", "PL+BB"):
                continue
            burst = (cells[0] + "A") if cells[0] else burst
            model = cells[1]
            for k, (q, units) in {2: ("alpha", ""), 3: ("beta", ""), 4: ("Ep", "keV"), 5: ("kT", "keV"), 6: ("flux_BB", "1e-7 erg/cm2/s"),
                                  7: ("flux_total", "1e-7 erg/cm2/s"), 9: ("BIC", "")}.items():
                v, lo, hi = _value_with_errors(cells[k])
                if np.isfinite(v):
                    add(paper=PAPER_2025, where=f"Table {3 + ['PE', 'ME', 'EE', 'WE'].index(episode)} (printed rows)", burst=burst,
                        quantity=f"{q}_{episode}_{model}", value=v, minus=lo, plus=hi, units=units, definition=f"{model} fit to the {episode}")
            stat, dof = cells[8].split("/")
            add(paper=PAPER_2025, where=f"Table {3 + ['PE', 'ME', 'EE', 'WE'].index(episode)} (printed rows)", burst=burst,
                quantity=f"pgstat_{episode}_{model}", value=float(stat), units="", definition=f"PGSTAT of the {model} fit, {dof} degrees of freedom")
    for burst, values in TEXT_2025.items():
        for q, v in values.items():
            value, err = (v if isinstance(v, tuple) else (v, np.nan))
            add(paper=PAPER_2025, where="Section 3 text", burst=burst, quantity=q, value=value, minus=err, plus=err,
                units="ms" if "tau" in q else "s", definition="as printed in the text (durations rounded to 0.01 s)", source="text")
    for burst in SAMPLE_2025:
        how = "named in their figures" if burst in LABELLED_2025 else "unnamed marker in their T100,PE-T_wt1 figure, matched to the 2026 table"
        add(paper=PAPER_2025, where="named in the text and figures" if burst in LABELLED_2025 else "Figure 6 (marker positions, T100,PE-T_wt1 plane)", burst=burst, quantity="in_2025_sample",
            value=1, definition=how, source="named in figure" if burst in LABELLED_2025 else "read from figure")
    out = pd.DataFrame(rows)[["paper", "where", "burst", "quantity", "value", "minus", "plus", "units", "definition", "source"]]
    out.to_csv(path, index=False)
    return out


def read_literature(path=LITERATURE):
    return pd.read_csv(path)
