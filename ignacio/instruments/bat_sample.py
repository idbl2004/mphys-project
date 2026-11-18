"""Swift/BAT bursts that GBM didn't see. The main one is Burlon et al.'s (2008) sample of BAT precursors in bursts with a redshift
(shared/burlon2008_precursors.csv), measured with the BAT layer (swift_bat.py) on the windows their Table 1 gives. I measure them the way
notebook 27 measures BAT precursors (bat_compare.precursors): a spectrum with its response averaged over the window, a power law and a
cut-off power law fitted with chi-squared, and an E_peak quoted only where the cut-off power law earns it.

Burlon et al. fitted the same two models and found that a power law is enough for every precursor (the cut-off doesn't improve the fit at
3 sigma), so they give no precursor E_peak, and their power-law indices are the check that our spectra agree with theirs. Where they do
quote a main pulse's E_peak (in a footnote to their Table 1), the alpha next to it is the cut-off power law's, not a power-law index. That's
GRB 060714's main pulse from BAT (alpha -1.30, E_peak 41 keV) and the Konus-Wind ones, and there I compare our cut-off power law's index and
E_peak with theirs instead. Their main pulses of GRB 050820A and 060124 come from Konus-Wind, because BAT's event data stop before them.

The same layer also runs on Li & Mao's (2022) bigger sample (shared/limao2022_precursors.csv), using nothing from their paper except the
list and where to look (limao). Our rule finds the precursor and the main emission in BAT's light curve, and each gets a power law.
one_burst does the same for any burst BAT saw, with nothing from a paper at all.

From ignacio/ (all take burst names like GRB050315; with none, every burst of the sample):
    python -m instruments.bat_sample fetch [names]      the files the chain needs, into ~/swift_bat/data (public, HEASARC)
    python -m instruments.bat_sample measure [names]    Burlon et al.'s windows, merged into results/bat_sample/burlon2008.csv
    python -m instruments.bat_sample limao [names]      Li & Mao's bursts with our windows, merged into results/bat_sample/limao2022.csv
    python -m instruments.bat_sample one GRB211211A     any burst, merged into results/bat_sample/any_bat.csv
python -m grbtool sample burlon|limao fetch|measure and python -m grbtool GRB211211A --swift run the same things."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import swift_bat as B
from .bat_compare import CPL_GAIN

CODE_DIR = Path(__file__).resolve().parents[1]
SAMPLE = CODE_DIR / "shared" / "burlon2008_precursors.csv"
LIMAO = CODE_DIR / "shared" / "limao2022_precursors.csv"
TRES_BAT = 0.256                # s: these precursors last seconds, so the light curve is binned more coarsely than GBM's
OUT = CODE_DIR / "results" / "bat_sample"
HEASARC = "https://heasarc.gsfc.nasa.gov/FTP"
NINETY = 1.645                  # their errors are 90%; ours are 68%
KEV = 1.602176634e-9            # erg


def read_sample():
    return pd.read_csv(SAMPLE, comment="#")


NEEDED = (("bat/event", "bevshsp_uf.evt.gz"), ("bat/hk", "bdecb.hk.gz"), ("auxil", "sat.fits.gz"))   # all that swift_bat.prepare reads


def fetch(names=None):
    """The three files of observation 00<trigger>000 that the chain reads (events, the detector housekeeping, the attitude), in the folder
    layout of instruments.bat_data.fetch. Each file is written under a temporary name and renamed only when it is complete."""
    from coppin.fetch import fetch_file             # resumable, restarting slow transfers on a fresh connection (coppin.fetch says why)
    catalogue = B.catalogue()
    for name in names or sorted(read_sample().grb.unique()):
        row = catalogue[name]
        obs = f"{row['trigger']:08d}000"
        month = row["utc"][:7].replace("-", "_")
        for folder, suffix in NEEDED:
            target = B.DATA / obs / folder / f"sw{obs}{suffix}"
            if target.exists() or (B.DATA / obs / obs / folder / target.name).exists():   # either layout swift_bat.burst reads
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            if fetch_file(f"{HEASARC}/swift/data/obs/{month}/{obs}/{folder}/{target.name}", part):
                part.replace(target)
            else:
                print(name, obs, f"couldn't fetch {target.name}", flush=True)
        print(name, obs, "->", B.DATA / obs, flush=True)


def classify(pl, cpl):
    """E_peak and the power-law index of one window, with the same rule as bat_compare.precursors. E_peak is quoted if the cut-off power law
    gains more than CPL_GAIN in chi-squared, its index is above -2, and its E_peak is inside BAT's band with both errors found. Otherwise the
    power law's photon index says which side of the band E_peak is on, or that it can't be told."""
    gain = pl["chi2"] - cpl["chi2"]
    ep, (ep_minus, ep_plus) = cpl["params"][2], cpl["errors"][2]
    measured = bool(gain > CPL_GAIN and cpl["params"][1] > -2.0 and B.BAND[0] < ep < B.BAND[1] and np.isfinite(ep_minus) and np.isfinite(ep_plus))
    index, (i_minus, i_plus) = pl["params"][1], pl["errors"][1]
    gamma = -index
    if measured:
        note = "E_peak measured by BAT"
    elif np.isfinite(i_minus) and gamma + 2 * i_minus < 1.3:
        note = "E_peak above BAT's band (photon index below 1.3)"
    elif np.isfinite(i_plus) and gamma - 2 * i_plus > 2.3:
        note = "E_peak below BAT's band (photon index above 2.3)"
    else:
        note = "E_peak not measured by BAT (its photon index allows a peak inside the band)"
    (a_minus, a_plus) = cpl["errors"][1]
    return dict(pl_norm=pl["params"][0], pl_index=index, pl_minus=i_minus, pl_plus=i_plus, pl_chi2=pl["chi2"], dof=pl["dof"], cpl_gain=gain, measured=measured,
                Ep_keV=ep if measured else np.nan, Ep_minus=ep_minus if measured else np.nan, Ep_plus=ep_plus if measured else np.nan,
                cpl_alpha=cpl["params"][1], cpl_alpha_minus=a_minus, cpl_alpha_plus=a_plus, note=note)


def band_fluence(norm, index, lo, hi, seconds):
    """Energy fluence (erg cm^-2) between lo and hi keV over `seconds` of the power law norm (E / 50 keV)^index photons cm^-2 s^-1 keV^-1
    (swift_bat.power_law)."""
    from scipy.integrate import quad
    return quad(lambda e: e * norm * (e / 50.0) ** index, lo, hi)[0] * KEV * seconds


def eiso_rest(norm, index, z, seconds, band=(15.0, 150.0)):
    """Isotropic energy (erg) emitted in `band` (keV) of the burst's rest frame, as Burlon et al. define theirs: 4 pi d_L^2 S / (1 + z),
    S the fluence in the observed band that is `band` in the rest frame (the power law carried below 15 keV where z needs it), with the
    project's cosmology (Planck18, as grbtool.physics)."""
    from astropy import units as u
    from astropy.cosmology import Planck18
    d = Planck18.luminosity_distance(z).to(u.cm).value
    return 4 * np.pi * d ** 2 * band_fluence(norm, index, band[0] / (1 + z), band[1] / (1 + z), seconds) / (1 + z)


def against_theirs(row, ours="pl_index"):
    """Our index against their alpha, in sigma (their 90% errors brought to 68%): the power law's (pl_index) or, where their alpha is the
    cut-off power law's, ours of the same model (cpl_alpha)."""
    theirs = 0.5 * (row["their_alpha_minus"] + row["their_alpha_plus"]) / NINETY
    error = 0.5 * (row[f"{'pl' if ours == 'pl_index' else 'cpl_alpha'}_minus"] + row[f"{'pl' if ours == 'pl_index' else 'cpl_alpha'}_plus"])
    return (row[ours] - row["their_alpha"]) / np.hypot(theirs, error)


def epeak_against_theirs(row):
    """Our cut-off power law's E_peak against their footnote's, in sigma (their 90% errors brought to 68%)."""
    theirs = 0.5 * (row["their_epeak_minus"] + row["their_epeak_plus"]) / NINETY
    error = row["Ep_plus"] if row["Ep_keV"] < row["their_epeak_keV"] else row["Ep_minus"]
    return (row["Ep_keV"] - row["their_epeak_keV"]) / np.hypot(theirs, error)


def measure(names=None):
    """Every window of the sample (their precursors and main pulses), one row each; a window that can't be measured keeps its row with the
    reason in note."""
    catalogue = B.catalogue()
    rows = []
    for name, windows in read_sample().groupby("grb", sort=True):
        if names and name not in names:
            continue
        c = catalogue[name]
        try:
            bat = B.burst(name)
        except FileNotFoundError as error:
            rows += [dict(grb=name, episode=w.episode, note=f"no BAT data: {error}") for w in windows.itertuples()]
            continue
        for w in windows.itertuples():
            curved = bool(np.isfinite(w.main_epeak_keV))      # their alpha is then the cut-off power law's
            row = dict(grb=name, episode=w.episode, t1_s=w.t1_s, t2_s=w.t2_s, z=w.z, pcode=c["pcode"], their_alpha=w.alpha,
                       their_alpha_minus=w.alpha_minus, their_alpha_plus=w.alpha_plus, their_source=w.source,
                       their_alpha_from="cut-off power law" if curved else "power law", their_epeak_keV=w.main_epeak_keV,
                       their_epeak_minus=w.main_epeak_minus, their_epeak_plus=w.main_epeak_plus, their_eiso_erg=w.eiso_erg)
            if not (c["events_from"] <= w.t1_s and w.t2_s <= c["events_to"]):
                rows.append(dict(row, note=f"BAT's event data ({c['events_from']:g} to {c['events_to']:g} s) don't cover the window"))
                continue
            try:
                spec = B.spectrum_during_slew(bat, w.t1_s, w.t2_s, f"burlon_{w.episode}")
                pl = B.fit(spec, B.power_law, [0.05, -1.5])
                cpl = B.fit(spec, B.cutoff_power_law, [0.05, -1.0, 100.0])
            except Exception as error:                  # recorded, not fatal
                rows.append(dict(row, note=f"no BAT fit: {type(error).__name__}: {error}"))
                continue
            row.update(classify(pl, cpl))
            row["fluence_15_150"] = band_fluence(row["pl_norm"], row["pl_index"], 15.0, 150.0, w.t2_s - w.t1_s)
            row["eiso_15_150_rest"] = eiso_rest(row["pl_norm"], row["pl_index"], w.z, w.t2_s - w.t1_s)
            if w.source == "bat":                       # their energies of Konus-Wind windows are in other bands
                row["eiso_vs_theirs"] = row["eiso_15_150_rest"] / w.eiso_erg
            if w.source == "bat" and np.isfinite(w.alpha_minus):
                row["index_vs_theirs_sigma"] = against_theirs(row, "cpl_alpha" if curved else "pl_index")
            if curved and row["measured"]:
                row["epeak_vs_theirs_sigma"] = epeak_against_theirs(row)
            rows.append(row)
            ours = f"cut-off index {row['cpl_alpha']:.2f}, E_peak {row['Ep_keV']:.1f} keV" if curved else f"index {row['pl_index']:.2f}"
            print(f"{name} {w.episode}: {ours} (theirs {w.alpha:.2f}); {row['note']}", flush=True)
    return merge_into(OUT / "burlon2008.csv", pd.DataFrame(rows))


def merge_into(path, table):
    """Rows of the bursts in table replace their earlier rows in the CSV at path; the other bursts' rows stay. Locked, so that
    several processes can measure different bursts side by side."""
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and not table.empty:
            before = pd.read_csv(path)
            table = pd.concat([before[~before.grb.isin(table.grb)], table], ignore_index=True)
        keys = ["grb", "episode"] if "episode" in table else ["grb"]
        table = table.sort_values(keys, kind="stable").reset_index(drop=True) if not table.empty else table
        table.to_csv(path, index=False)
    return table


def redshift(z):
    """A printed redshift as a number: None for an upper limit ('<3.5'), the value for an uncertain one ('2.09?')."""
    z = str(z).strip()
    return None if z.startswith("<") else float(z.rstrip("?"))


def limao_spans():
    """Li & Mao's pulse fits as the span of each burst's precursor and main burst (s from the BAT trigger): from the first pulse's start
    t_s to about 5% of the last one's peak (t_s + tau_p + 3 tau_d), with each burst's redshift (as printed), type and T90."""
    t = pd.read_csv(LIMAO, comment="#", dtype={"z": str})
    t["stop"] = t.t_s + t.tau_p + 3 * t.tau_d
    t["kind"] = np.where(t.component.str.startswith("p"), "pre", "main")
    spans = t.groupby(["grb", "kind"]).agg(start=("t_s", "min"), stop=("stop", "max")).unstack()
    info = t.groupby("grb").agg(z=("z", "first"), type=("type", "first"), t90_s=("t90_s", "first"))
    return spans, info


def window_fit(bat, start, stop, label, z):
    """A power law in BAT's band over one window: its index and error, the 15-150 keV fluence and, where the redshift is known, the
    luminosity in the rest-frame 15-150 keV band and the rest-frame duration."""
    pl = B.fit(B.spectrum_during_slew(bat, start, stop, label), B.power_law, [0.05, -1.5])
    norm, index = pl["params"]
    seconds = stop - start
    out = dict(index=index, err=0.5 * (pl["errors"][1][0] + pl["errors"][1][1]), fluence_15_150=band_fluence(norm, index, 15.0, 150.0, seconds))
    if z is not None:
        out.update(L_rest_15_150=eiso_rest(norm, index, z, seconds) / (seconds / (1 + z)), duration_rest_s=seconds / (1 + z))
    return out


def limao(names=None, tres=TRES_BAT):
    """Li & Mao's bursts through the BAT pipeline. Our rule (zhu2026) finds the precursor and the main emission in BAT's mask-weighted light
    curve, and each one is fitted with a power law (window_fit). Their pulse fits are only used to set the stretch searched (30 s either side of
    their pulses, inside the event data) and to say whether our precursor is theirs, i.e. whether it overlaps the span of their precursor
    pulses. One row per burst, and a burst that can't be done keeps its row with the reason in note."""
    spans, info = limao_spans()
    catalogue = B.catalogue()
    rows = []
    for name in names or sorted(info.index):
        cat = catalogue[name]
        pre0, pre1 = spans.loc[name, ("start", "pre")], spans.loc[name, ("stop", "pre")]
        main0, main1 = spans.loc[name, ("start", "main")], spans.loc[name, ("stop", "main")]
        z = redshift(info.loc[name, "z"])
        row = dict(grb=name, z_printed=info.loc[name, "z"], z=z, type=info.loc[name, "type"], their_pre_start=pre0, their_pre_stop=pre1,
                   their_main_start=main0, their_main_stop=main1)
        try:
            bat = B.burst(name)
            span = (max(cat["events_from"], pre0 - 30), min(cat["events_to"], max(main1, pre1) + 30))
            around = main0 if main0 < cat["events_to"] else pre0
            found, blocks = B.episodes(bat, tres, 0.0, span, around=around)
            f = found["zhu2026"]
        except Exception as error:                      # recorded, not fatal
            rows.append(dict(row, note=f"no episodes: {type(error).__name__}: {error}"[:150]))
            merge_into(OUT / "limao2022.csv", pd.DataFrame([rows[-1]]))
            continue
        row.update(pe_start=f.pe[0] if f.pe else np.nan, pe_stop=f.pe[1] if f.pe else np.nan, me_start=f.me[0] if f.me else np.nan,
                   me_stop=f.me[1] if f.me else np.nan, blocks=len(blocks), flags=" ".join(f.flags))
        row["found"] = bool(f.pe) and f.pe[0] < pre1 and pre0 < f.pe[1]
        if row["found"] and f.me:
            for key, (a, b) in (("pe", f.pe), ("me", f.me)):
                try:
                    row.update({f"{key}_{k}": v for k, v in window_fit(bat, a, b, f"limao_{key}", z).items()})
                except Exception as error:
                    row[f"{key}_note"] = f"{type(error).__name__}: {error}"[:100]
        rows.append(row)
        merge_into(OUT / "limao2022.csv", pd.DataFrame([row]))     # saved burst by burst, so a stopped run loses nothing
        print(f"{name}: precursor {'found' if row['found'] else 'not found'}"
              + (f", index {row.get('pe_index', np.nan):.2f} against the main emission's {row.get('me_index', np.nan):.2f}" if row["found"] else ""), flush=True)
    return pd.read_csv(OUT / "limao2022.csv")


def one_burst(name, z=None, tres=TRES_BAT):
    """Any burst BAT saw, measured the way Li & Mao's are (limao) but with nothing from a paper. Our rule (zhu2026) runs on BAT's light curve
    over all of its event data (around the BAT trigger, if the data have gaps), and the precursor and main emission it finds each get a power
    law. softer_by is the main emission's index minus the precursor's, so it's positive when the precursor is softer, as in notebooks 30 and
    32. Returns the row, which is also merged into results/bat_sample/any_bat.csv."""
    cat = B.catalogue().get(name)
    if cat is None:
        raise ValueError(f"{name} isn't in the BAT catalogue ({B.CATALOGUE}): BAT didn't trigger on it, or the name is written differently")
    if not (np.isfinite(cat["events_from"]) and np.isfinite(cat["events_to"])):
        raise ValueError(f"{name}: the BAT catalogue gives no span for its event data")
    fetch([name])
    bat = B.burst(name)
    found, blocks = B.episodes(bat, tres, 0.0, (cat["events_from"], cat["events_to"]), around=0.0)
    f = found["zhu2026"]
    row = dict(grb=name, z=z, tres_s=tres, pcode=cat["pcode"], pe_start=f.pe[0] if f.pe else np.nan, pe_stop=f.pe[1] if f.pe else np.nan,
               me_start=f.me[0] if f.me else np.nan, me_stop=f.me[1] if f.me else np.nan, blocks=len(blocks), flags=" ".join(f.flags))
    for key, span in (("pe", f.pe), ("me", f.me)):
        if span:
            try:
                row.update({f"{key}_{k}": v for k, v in window_fit(bat, span[0], span[1], f"any_{key}", z).items()})
            except Exception as error:                  # recorded, not fatal
                row[f"{key}_note"] = f"{type(error).__name__}: {error}"[:100]
    if np.isfinite(row.get("pe_index", np.nan)) and np.isfinite(row.get("me_index", np.nan)):
        row["softer_by"] = row["me_index"] - row["pe_index"]
        row["softer_sigma"] = row["softer_by"] / np.hypot(row["pe_err"], row["me_err"])
    merge_into(OUT / "any_bat.csv", pd.DataFrame([row]))
    return row


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    part, names = (argv[0] if argv else ""), argv[1:] or None
    if part == "fetch":
        fetch(names)
    elif part == "measure":
        measure(names)
    elif part == "limao":
        limao(names)
    elif part == "one" and names:
        for name in names:
            print(one_burst(name))
    else:
        raise SystemExit("fetch, measure, limao or one <names>")


if __name__ == "__main__":
    main()
