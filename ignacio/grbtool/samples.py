"""Every sample the project measured, and Zhu et al.'s rules or Swift/BAT on any single burst, all through python -m grbtool.

    python -m grbtool sample zhu prefetch|batch|status|burst [names]      Zhu et al.'s 111 bursts (zhu.run)
    python -m grbtool sample coppin sample|prefetch|batch|status          Coppin et al.'s 103 (coppin)
    python -m grbtool sample burlon fetch|measure [names]                 Burlon et al.'s 15 Swift bursts (instruments.bat_sample)
    python -m grbtool sample limao fetch|measure [names]                  Li & Mao's 52 Swift bursts (instruments.bat_sample)
    python -m grbtool GRB160625B --zhu       one GBM burst by Zhu et al.'s rules, the way the samples were measured
    python -m grbtool GRB211211A --swift     one burst with BAT, the way Li & Mao's were

Each sample still runs its own module's command, unchanged. This just gives them one way in. The one-burst versions use the same steps
and settings as the samples, so a burst measured here can be put next to them."""
import numpy as np
import pandas as pd

from . import CODE_DIR

ANY = CODE_DIR / "results" / "any"
SAMPLES = ("zhu", "coppin", "burlon", "limao")
SOFT_RATIO = 0.1876     # GRB 230307A's precursor E_peak over its main emission's (notebook 29): a precursor at or below it is "soft"


def run_sample(argv):
    """python -m grbtool sample <name> <part> [names]: the sample's own command."""
    if not argv or argv[0] not in SAMPLES:
        raise SystemExit("python -m grbtool sample zhu|coppin|burlon|limao <part> [names] (see grbtool/samples.py)")
    name, rest = argv[0], list(argv[1:])
    if name == "zhu":
        from zhu.run import main
        return main(rest)
    if name == "coppin":
        from coppin.__main__ import main
        return main(rest)
    from instruments import bat_sample as S
    part, names = (rest[0] if rest else ""), rest[1:] or None
    if part == "fetch":
        S.fetch(names or (sorted(S.read_sample().grb.unique()) if name == "burlon" else sorted(S.limao_spans()[1].index)))
    elif part == "measure":
        (S.measure if name == "burlon" else S.limao)(names)
    else:
        raise SystemExit(f"python -m grbtool sample {name} fetch|measure [names]")
    return 0


def bin_for(trigger):
    """The light-curve bin (s) for a burst: the step coppin.sample's rule gives for its 64 ms peak flux in the GBM catalogue, the rule
    fitted to Zhu et al.'s own choices, since they don't say how they chose them."""
    from coppin.sample import read_catalogue, tres_for, tres_rule
    from .bursts import BurstError
    catalogue = read_catalogue()
    if trigger not in catalogue.index or not catalogue.loc[trigger, "flux_64"] > 0:
        raise BurstError(f"{trigger} has no 64 ms peak flux in shared/fermigbrst_extract.csv to choose the bin from; give it with --tres")
    return float(tres_for(catalogue.loc[trigger, "flux_64"], tres_rule(catalogue)))


def zhu_one(name, trigger=None, tres=None, workers=3, download=True, out_root=ANY):
    """Zhu et al.'s rules on any GBM burst, the way their sample and Coppin et al.'s were measured: every stage of zhu.run (the episodes
    and their errors from simulated light curves, the half and twice bins, the lags, the spectra both ways) into results/any/<burst>/.
    tres is the light-curve bin in s (default: bin_for). Returns the summary."""
    from zhu.run import measure, prefetch
    from .bursts import find_trigger
    from .pipeline import local_trigger
    short = name[3:] if name.startswith("GRB") else name
    trigger = trigger or local_trigger(f"GRB{short}") or find_trigger(name)
    row = pd.Series(dict(trigger=trigger, Tres=tres or bin_for(trigger), reason="any"), name=short)
    if download and prefetch(pd.DataFrame([row]), workers=1):
        raise SystemExit(f"{name}: the download failed")
    measure(short, row, workers=workers, out_root=out_root)
    return summary(short, out_root)


def summary(short, root=ANY, mode="zhu2026"):
    """One burst under Zhu et al.'s 2026 rules, read the way notebooks 25 and 29 read the samples: the episodes, T_MVT, E_peak both ways,
    EPI their way with the chance of it lying above their line, and the precursor against the main emission. I call the precursor soft if its
    E_peak ratio (our rule) is at or below GRB 230307A's, and fast if its T_MVT ratio isn't shown to be above 1 (its log minus its error is at
    most 0). EPI is left out if their main-emission E_peak comes from a model with an unconstrained parameter, as in notebook 29."""
    from zhu.features import collect, epi_summary
    features = collect(mode, root, [short])
    f = features.loc[short]
    if f.get("status") != "complete" or not np.isfinite(f.get("me_start", np.nan)):
        return dict(burst=short, note=f"no main emission found: {f.get('status')} {f.get('status_reason', '')}".strip())
    out = dict(burst=short, Tres_s=f["Tres"], flags=f["flags"])
    for key in ("pe", "qe1", "me"):
        out[f"{key}_start"], out[f"{key}_stop"] = f[f"{key}_start"], f[f"{key}_stop"]
    for key in ("TMVT_PE", "TMVT_PE_p16", "TMVT_PE_p84", "TMVT_ME", "TMVT_ME_p16", "TMVT_ME_p84", "T100_PE", "T100_QE1",
                "Ep_PE_ours", "Ep_ME_ours", "Ep_ME_zhu", "Ep_ME_zhu_lo", "Ep_ME_zhu_hi", "model_ME_zhu"):
        out[key] = f.get(key, np.nan)
    measured = str(f.get("measured_PE")) == "True" and str(f.get("measured_ME")) == "True"
    out["Ep_ratio"] = f.Ep_PE_ours / f.Ep_ME_ours if measured else np.nan
    half = lambda lo, hi: 0.5 * (np.log10(hi) - np.log10(lo))
    out["mvt_ratio"] = f.TMVT_PE / f.TMVT_ME
    out["mvt_ratio_err_dex"] = np.hypot(half(f.TMVT_PE_p16, f.TMVT_PE_p84), half(f.TMVT_ME_p16, f.TMVT_ME_p84))
    out["soft"] = bool(out["Ep_ratio"] <= SOFT_RATIO)
    out["fast"] = bool(np.log10(out["mvt_ratio"]) - out["mvt_ratio_err_dex"] <= 0)
    constrained = ("no_model_fully_constrained" not in str(f.get("zhu_flags_ME", ""))
                   and np.isfinite(f.get("Ep_ME_zhu_lo", np.nan)) and np.isfinite(f.get("Ep_ME_zhu_hi", np.nan)))
    if np.isfinite(f.get("T100_PE", np.nan)) and constrained:
        out.update(epi_summary(features, root).loc[short].to_dict())
    else:
        out["EPI_note"] = "no precursor" if not np.isfinite(f.get("T100_PE", np.nan)) else "main-emission E_peak not constrained their way"
    return out


def describe(s):
    """The summary as a few lines of text."""
    if "note" in s:
        return f"GRB {s['burst']}: {s['note']}"
    lines = [f"GRB {s['burst']} by Zhu et al.'s 2026 rules, bin {1e3 * s['Tres_s']:g} ms" + (f" (flags: {s['flags']})" if s["flags"] else "")]
    if not np.isfinite(s["pe_start"]):
        lines.append(f"  no precursor; main emission {s['me_start']:.2f} to {s['me_stop']:.2f} s")
        return "\n".join(lines)
    lines.append(f"  precursor {s['pe_start']:.2f} to {s['pe_stop']:.2f} s, quiet {s['qe1_start']:.2f} to {s['qe1_stop']:.2f} s, "
                 f"main emission {s['me_start']:.2f} to {s['me_stop']:.2f} s")
    lines.append(f"  T_MVT: precursor {s['TMVT_PE']:.3f} s ({s['TMVT_PE_p16']:.3f} to {s['TMVT_PE_p84']:.3f}), main emission {s['TMVT_ME']:.3f} s "
                 f"({s['TMVT_ME_p16']:.3f} to {s['TMVT_ME_p84']:.3f})")
    lines.append(f"  E_peak (our rule): precursor {s['Ep_PE_ours']:.0f} keV, main emission {s['Ep_ME_ours']:.0f} keV; "
                 + (f"ratio {s['Ep_ratio']:.3f}, {'soft' if s['soft'] else 'not soft'} (GRB 230307A: {SOFT_RATIO:.3f})" if np.isfinite(s["Ep_ratio"])
                    else "no ratio (one of them isn't measured)"))
    lines.append(f"  T_MVT ratio {s['mvt_ratio']:.2f} (+-{s['mvt_ratio_err_dex']:.2f} dex): "
                 + ("not shown to be slower than the main emission" if s["fast"] else "slower than the main emission by this reading"))
    if "EPI_zhu" in s:
        lines.append(f"  EPI (their way) {s['EPI_zhu']:.2f} ({s['EPI_p16_zhu']:.2f} to {s['EPI_p84_zhu']:.2f}), chance above their line "
                     f"{s['P_above_zhu']:.2f}")
    else:
        lines.append(f"  no EPI: {s['EPI_note']}")
    return "\n".join(lines)


def bat_one(name, z=None, tres=None):
    """Swift/BAT on one burst (instruments.bat_sample.one_burst, at bin tres in s, 256 ms by default), as a few lines of text."""
    from instruments.bat_sample import TRES_BAT, one_burst
    tres = tres or TRES_BAT
    r = one_burst(name, z=z, tres=tres)
    lines = [f"{name} with BAT (bin {1e3 * tres:g} ms, our rule zhu2026)" + (f", flags: {r['flags']}" if r["flags"] else "")]
    for key, label in (("pe", "precursor"), ("me", "main emission")):
        if np.isfinite(r[f"{key}_start"]):
            index = f"power-law index {r[f'{key}_index']:.2f} +- {r[f'{key}_err']:.2f}" if f"{key}_index" in r else r.get(f"{key}_note", "no fit")
            lines.append(f"  {label} {r[f'{key}_start']:.2f} to {r[f'{key}_stop']:.2f} s from the BAT trigger: {index}")
        else:
            lines.append(f"  no {label} found")
    if "softer_by" in r:
        lines.append(f"  precursor softer by {r['softer_by']:.2f} ({r['softer_sigma']:.1f} sigma)")
    return "\n".join(lines)
