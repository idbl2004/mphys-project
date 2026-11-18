"""BAT against GBM for the bursts both of them saw: the BAT chain checked against the BAT catalogue, the episode edges from BAT's light
curve against ours from GBM's, and BAT's spectrum of the precursor that GBM defines. Results go to results/bat/.

    python -m instruments.bat_compare validate GRB211211A GRB190114C ...     BAT catalogue names
    python -m instruments.bat_compare episodes 211211A 190114C ...           sample names (needs the zhu batch)
    python -m instruments.bat_compare precursors 211211A 190114C ...

Times are put on GBM's axis using the difference of the two trigger times and the light-travel time between Swift and Fermi (up to about
45 ms, see swift_bat.light_travel). I work that out at the time being compared, which matters for precursors that only last tens of ms.
Every burst is handled on its own. A burst that can't be compared (no GBM episodes, no BAT data, or a tool that fails) keeps its rows with
the reason, so one burst never costs the rest of the table.

For every precursor BAT has, I take GBM's best-supported fit of the same window (the lowest AIC among the Comptonized, Band and two-break
models), fold it through BAT's response and fit it with a power law. That gives the index BAT should measure if the two instruments agree
(bat_pl_index_from_gbm). tension_sigma compares it with BAT's own index, allowing for GBM's model choice and statistics, and it is left out
where GBM doesn't measure the precursor's E_peak. That way the instruments are only compared where BAT's band can say something, and no
E_peak is forced on BAT.

BAT's band is 15-150 keV, so I only report an E_peak when the Comptonized fit (alpha > -2, so that it has a peak) puts it inside the band
with finite profile errors and beats the power law by Delta chi2 > 9 (3 sigma for one parameter). Otherwise the power-law photon index
Gamma is given. Sakamoto et al. (2009, ApJ 693, 922) find that for 1.3 < Gamma < 2.3 the peak can be anywhere from about 30 to 220 keV, so I
only say the peak is above the band if Gamma is below 1.3 by more than 2 sigma, or below the band if Gamma is above 2.3 by more than 2
sigma. Nothing is forced, and a fit that fails is recorded, not dropped."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import swift_bat as B

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
OUT = CODE_DIR / "results" / "bat"
MATCH = Path.home() / "swift_bat" / "sample_in_bat_catalogue.csv"
CPL_GAIN = 9.0          # delta chi2 the Comptonized must gain over the power law before its E_peak is quoted


def _catalogue_table(path):
    rows = {}
    for line in open(path):
        if line.startswith("#") or "|" not in line:
            continue
        f = [x.strip() for x in line.split("|")]
        rows[f[0]] = f
    return rows


def validate(names):
    """Power-law fits of the 1-s peak and T100 windows against the catalogue's (T100 with the response averaged over the window)."""
    peak = _catalogue_table(Path.home() / "swift_bat/catalogue_1s/pow.txt")
    t100 = _catalogue_table(Path.home() / "swift_bat/catalogue_T100/summary_pow_parameters.txt")
    rows = []
    for name in names:
        bat = B.burst(name)
        for which, table in (("1-s peak", peak), ("T100", t100)):
            r = table.get(name)
            if r is None:
                continue
            start, stop = float(r[14]), float(r[15])
            if which == "T100":
                spec = B.spectrum_during_slew(bat, start, stop, "check_T100")
            else:
                spec = B.read_spectrum(*B.spectrum(bat, start, stop, "check_peak"))
            f = B.fit(spec, B.power_law, [float(r[5]), float(r[2])])
            rows.append(dict(burst=name, window=which, start=start, stop=stop, index=f["params"][1], index_minus=f["errors"][1][0],
                             index_plus=f["errors"][1][1], norm=f["params"][0], chi2=f["chi2"], dof=f["dof"], catalogue_index=float(r[2]),
                             catalogue_low=float(r[3]), catalogue_high=float(r[4]), catalogue_norm=float(r[5]), catalogue_chi2=float(r[8]),
                             catalogue_dof=int(float(r[9]))))
    return pd.DataFrame(rows)


def _gbm(name):
    """The burst's zhu batch record, its GBM trigger time (UTC to the microsecond, and MET) and GBM's trigdat file, which holds Fermi's
    position. Raises FileNotFoundError if the batch has no episodes for the burst (GRB 111228A: its data start too late)."""
    from zhu import RESULTS
    from zhu.run import read_sample
    from astropy.io import fits
    from astropy.time import Time, TimeDelta
    from grbtool.bursts import DATA_ROOT
    from zhu.catalogue import data_name
    path = RESULTS / name / "episodes.json"
    if not path.exists():
        raise FileNotFoundError(f"the zhu batch has no episodes for {name}")
    record = json.loads(path.read_text())
    trigger = read_sample().loc[name, "trigger"]
    folder = DATA_ROOT / data_name(name, DATA_ROOT)
    tte = sorted(folder.rglob(f"glg_tte_n*_{trigger}_v*.fit"))[0]
    met = fits.getheader(tte, 0)["TRIGTIME"]
    utc = Time("2001-01-01T00:00:00", scale="utc") + TimeDelta(met, format="sec")
    utc.precision = 6                                    # isot to the microsecond (the default rounds to 1 ms)
    trigdat = sorted(folder.rglob(f"glg_trigdat_all_{trigger}_v*.fit"))
    return record, utc.isot, (trigdat[-1] if trigdat else None), met


def _bat_and_offset(name, match, utc, trigdat, met, at=0.0):
    """The BAT burst, the seconds to add to a time from the BAT trigger to put it on GBM's axis for the same light, the light-travel part of
    that, and a note. The offset is the difference of the trigger times (gbm_offset) plus the light-travel time between the two spacecraft at
    `at` s after the GBM trigger (light_travel). If either spacecraft's position can't be read, the light-travel time is left out and the note
    says so. Raises FileNotFoundError if there are no BAT data for the burst."""
    bat = B.burst(match.loc[name, "bat"])
    shift = B.gbm_offset(bat, utc)
    try:
        if trigdat is None:
            raise FileNotFoundError("no GBM trigdat file")
        travel = B.light_travel(bat, trigdat, met, gbm_offset=shift, at=at)
    except (FileNotFoundError, KeyError, OSError) as error:
        return bat, shift, np.nan, f"light_travel_left_out:{type(error).__name__}"
    return bat, shift + travel, travel, ""


def _no_bat_data(name, match):
    """Why a burst of the BAT catalogue has no BAT data here, with the catalogue's trigger method and comment."""
    row = B.catalogue().get(match.loc[name, "bat"])
    obsid = f"observation {row['trigger']:08d}000 " if row else ""
    for line in B.CATALOGUE.read_text().splitlines():
        f = [x.strip() for x in line.split("|")]
        if f and f[0] == match.loc[name, "bat"] and len(f) > 17:
            return f"no BAT data: {obsid}not found (the catalogue: '{f[15]}', '{f[17]}')"
    return f"no BAT data: {obsid}not found"


EPISODE_COLUMNS = ["burst", "episode", "bat_pcode", "gbm_start", "gbm_stop", "bat_start", "bat_stop", "gbm_tres", "bat_tres", "offset_s",
                   "light_travel_s", "bat_flags"]


def episodes(names):
    """zhu2026 episodes from BAT's light curve (bins of max(Tres, 16 ms)) against ours from GBM, on GBM's time axis. To get there I add the
    difference of the trigger times and the light-travel time between the spacecraft at the start of the main emission, so the same light has
    the same time. There are three rows per burst (pe, qe1, me), and a burst that can't be compared keeps its rows with the reason in
    bat_flags."""
    match = pd.read_csv(MATCH).set_index("burst")
    rows = []
    for name in names:
        try:
            record, utc, trigdat, met = _gbm(name)
        except (FileNotFoundError, KeyError) as error:
            rows += [dict(burst=name, episode=key, bat_flags=f"no GBM episodes: {error}") for key in ("pe", "qe1", "me")]
            continue
        gbm, tres = record["modes"]["zhu2026"], record["Tres"]
        edges = lambda key, i: gbm[key][i] if gbm.get(key) else np.nan
        base = lambda key: dict(burst=name, episode=key, gbm_start=edges(key, 0), gbm_stop=edges(key, 1), gbm_tres=tres)
        if not gbm.get("me"):
            rows += [dict(base(key), bat_flags="GBM found no main emission under the zhu2026 rule") for key in ("pe", "qe1", "me")]
            continue
        try:
            bat, offset, travel, travel_note = _bat_and_offset(name, match, utc, trigdat, met, at=gbm["me"][0])
        except FileNotFoundError:
            rows += [dict(base(key), bat_flags=_no_bat_data(name, match)) for key in ("pe", "qe1", "me")]
            continue
        bat_tres = max(tres, 0.016)
        pcode = B.catalogue()[bat.name]["pcode"]
        we = gbm.get("we") or gbm["me"]
        span = (we[0] - 20.0, min(we[1] + 20.0, we[0] + 400.0))
        try:
            found, _ = B.episodes(bat, bat_tres, offset, span, around=gbm["me"][0])
            b, flags = found["zhu2026"], " ".join(found["zhu2026"].flags)
        except Exception as error:                      # a HEASoft tool or the blocks failing is recorded, not fatal
            b, flags = None, f"{type(error).__name__}: {error}"
        for key in ("pe", "qe1", "me"):
            s = getattr(b, key) if b is not None else None
            rows.append(dict(base(key), bat_pcode=pcode, bat_start=s[0] if s else np.nan, bat_stop=s[1] if s else np.nan, bat_tres=bat_tres,
                             offset_s=offset, light_travel_s=travel, bat_flags=" ".join(x for x in (flags, travel_note) if x)))
    out = pd.DataFrame(rows, columns=EPISODE_COLUMNS)
    out["d_start"], out["d_stop"] = out.bat_start - out.gbm_start, out.bat_stop - out.gbm_stop
    return out


AIC_SUPPORT = 10.0      # GBM models within this much AIC of the best are kept in the prediction's range (Burnham & Anderson 2002:
                        #   beyond 10 a model has essentially no support)


def _shape(function, params):
    """A photon-spectrum shape from one of GBM's fitted models, evaluated with GDT's own function (as the fit was), on any array of
    energies (keV)."""
    def shape(e):
        e = np.asarray(e, float)
        return function.eval(list(params), e.ravel()).reshape(e.shape)
    return shape


def expected_index(spec, shape):
    """The power-law index BAT would measure for a photon spectrum of this shape (one of GBM's fits). The shape is folded through the
    spectrum's own response and fitted with a power law on the same channels, with the same errors and no noise added. The overall scale
    doesn't matter, because it only rescales the power law's normalisation, so the shape alone sets the index. Raises ValueError if the shape
    can't be folded (if it isn't finite and positive across the band)."""
    from scipy.optimize import least_squares
    use = (spec["e_min"] >= B.BAND[0] - 1e-6) & (spec["e_max"] <= B.BAND[1] + 1e-6) & (spec["error"] > 0)
    model = B._photons(lambda e: shape(e), [], spec["e_lo"], spec["e_hi"]) @ spec["matrix"]
    if not (np.all(np.isfinite(model[use])) and np.mean(model[use]) > 0):
        raise ValueError("the shape isn't finite and positive in BAT's band")
    model = model / np.mean(model[use])

    def residuals(p):
        return (model[use] - (B._photons(B.power_law, p, spec["e_lo"], spec["e_hi"]) @ spec["matrix"])[use]) / spec["error"][use]
    return float(least_squares(residuals, [1.0, -1.5], bounds=B.power_law.bounds, x_scale="jac").x[1])


def gbm_shapes(name):
    """GBM's fits of the zhu2026 precursor window that can be folded into BAT's band (our rule fits all three models, whichever one it then
    quotes): {model: (shape, C-stat, free parameters)} for the Comptonized, Band and two-break models, the Comptonized shapes at the corners of
    its 68% errors (for GBM's statistical spread), and a note. The corners are only used when GBM measures the precursor's E_peak, since
    otherwise the spread isn't defined and neither is the tension. alpha's error is only used when the Comptonized is the quoted model,
    because that's the only case where it is stored. alpha is kept above -1.99 so that the shape still has a peak."""
    from gdt.core.spectra.functions import Band, Comptonized
    from spectral_fit_tools import BandLowBreak
    from zhu import RESULTS
    path = RESULTS / name / "spectra_zhu2026.csv"
    if not path.exists():
        return {}, [], "no GBM spectra"
    fits = pd.read_csv(path).set_index("label")
    if "PE" not in fits.index:
        return {}, [], "no GBM fit of the precursor"
    r = fits.loc["PE"]
    finite = lambda *keys: all(np.isfinite(float(r.get(k, np.nan))) for k in keys)
    models = {}
    if finite("Epeak_cpl", "alpha_cpl", "C_cpl"):
        models["Comptonized"] = (_shape(Comptonized(), [1.0, r["Epeak_cpl"], r["alpha_cpl"], 100.0]), float(r["C_cpl"]), 3)
    if finite("Epeak_band", "alpha_band", "beta_band", "C_band"):
        models["Band"] = (_shape(Band(), [1.0, r["Epeak_band"], r["alpha_band"], r["beta_band"], 100.0]), float(r["C_band"]), 4)
    if finite("Eb", "alpha1_lb", "alpha2_lb", "Epeak_lowbreak", "beta_lb", "C_lowbreak"):
        models["two-break"] = (_shape(BandLowBreak(), [1.0, r["Eb"], r["alpha1_lb"], r["alpha2_lb"], r["Epeak_lowbreak"], r["beta_lb"], 2.0,
                                                      100.0]), float(r["C_lowbreak"]), 6)
    corners, note = [], ""
    if str(r.get("measured")) != "True":
        note = "GBM doesn't measure the precursor's E_peak, so GBM's spread and the tension are left out"
    elif "Comptonized" in models and finite("Epeak_cpl_err"):
        a_err = float(r["alpha_err"]) if r["model"] == "Comptonized" and finite("alpha_err") else 0.0
        if a_err == 0.0:
            note = "GBM's spread from E_peak alone (alpha's error is stored only when the Comptonized is the quoted model)"
        for a in {max(r["alpha_cpl"] - a_err, -1.99), r["alpha_cpl"] + a_err}:
            for e in {max(r["Epeak_cpl"] - r["Epeak_cpl_err"], 1.0), r["Epeak_cpl"] + r["Epeak_cpl_err"]}:
                corners.append(_shape(Comptonized(), [1.0, e, a, 100.0]))
    return models, corners, note


def _from_gbm(name, spec, index, i_minus, i_plus):
    """What BAT should measure if GBM is right: GBM's best-supported model (lowest AIC = C + 2k) folded into BAT's band. The range covers
    every model within AIC_SUPPORT of it, GBM's statistical spread comes from the Comptonized's corners, and the tension puts both and
    BAT's error on the side the prediction lies together in quadrature. Returns the row's columns."""
    models, corners, note = gbm_shapes(name)
    predicted = {}
    for model, (shape, _, _) in models.items():
        try:
            predicted[model] = expected_index(spec, shape)
        except ValueError:
            pass
    if not predicted:
        return dict(from_gbm_note=note or "no GBM model could be folded into BAT's band")
    aic = {m: models[m][1] + 2 * models[m][2] for m in predicted}
    best = min(aic, key=aic.get)
    kept = [m for m in aic if aic[m] - aic[best] < AIC_SUPPORT]
    low, high = min(predicted[m] for m in kept), max(predicted[m] for m in kept)
    spread = []
    for corner in corners:
        try:
            spread.append(expected_index(spec, corner))
        except ValueError:
            pass
    stat = 0.5 * (max(spread) - min(spread)) if spread else np.nan
    expected = predicted[best]
    side = i_plus if expected > index else i_minus
    both = np.sqrt(side ** 2 + (0.5 * (high - low)) ** 2 + stat ** 2)
    return dict(from_gbm_model=best, bat_pl_index_from_gbm=expected, from_gbm_low=low, from_gbm_high=high, from_gbm_stat=stat,
                from_gbm_all=" ".join(f"{m}:{predicted[m]:.3f}" for m in predicted), from_gbm_note=note,
                tension_sigma=(expected - index) / both if np.isfinite(both) and both > 0 else np.nan)


def precursors(names):
    """BAT's spectrum of the precursor that GBM defines (zhu2026 rule), fitted with a power law and a Comptonized, next to GBM's E_peak and
    the index GBM's fit predicts in BAT's band (the _from_gbm columns). The response is averaged over the window, as the catalogue does for its
    T100 (swift_bat.spectrum_during_slew), which for a window of 5 s or less is just the single response. Times go onto BAT's axis with the
    light-travel time at the middle of the window. A burst that can't be compared keeps a row with the reason in note."""
    from zhu.features import collect
    ours = collect("zhu2026")
    match = pd.read_csv(MATCH).set_index("burst")
    rows = []
    for name in names:
        row = dict(burst=name, gbm_Ep=ours.loc[name, "Ep_PE_ours"] if name in ours.index else np.nan,
                   gbm_measured=ours.loc[name, "measured_PE"] if name in ours.index else None)
        try:
            record, utc, trigdat, met = _gbm(name)
        except (FileNotFoundError, KeyError) as error:
            rows.append(dict(row, note=f"no GBM episodes: {error}"))
            continue
        pe = record["modes"]["zhu2026"].get("pe")
        row["gbm_window"] = pe
        if not pe:
            rows.append(dict(row, note="no precursor in GBM under the zhu2026 rule"))
            continue
        try:
            bat, offset, travel, travel_note = _bat_and_offset(name, match, utc, trigdat, met, at=0.5 * (pe[0] + pe[1]))
        except FileNotFoundError:
            rows.append(dict(row, note=_no_bat_data(name, match)))
            continue
        cat = B.catalogue()[bat.name]
        row.update(offset_s=offset, light_travel_s=travel, bat_pcode=cat["pcode"])
        start, stop = pe[0] - offset, pe[1] - offset
        if not (cat["events_from"] <= start and stop <= cat["events_to"]):
            rows.append(dict(row, note=f"BAT's event data ({cat['events_from']:g} to {cat['events_to']:g} s) don't cover the precursor"))
            continue
        try:
            # averaged over 5 s slices when the window is longer: some precursors (GRB 190829A, 241030A) run into Swift's slew
            spec = B.spectrum_during_slew(bat, start, stop, "precursor")
            pl = B.fit(spec, B.power_law, [0.05, -1.5])
            cpl = B.fit(spec, B.cutoff_power_law, [0.05, -1.0, 100.0])
        except Exception as error:                      # recorded, not fatal
            rows.append(dict(row, note=f"no BAT fit: {type(error).__name__}: {error}"))
            continue
        gain = pl["chi2"] - cpl["chi2"]
        ep, (ep_minus, ep_plus) = cpl["params"][2], cpl["errors"][2]
        measured = bool(gain > CPL_GAIN and cpl["params"][1] > -2.0 and B.BAND[0] < ep < B.BAND[1] and np.isfinite(ep_minus)
                        and np.isfinite(ep_plus))
        index, (i_minus, i_plus) = pl["params"][1], pl["errors"][1]
        gamma = -index                                                      # photon index, N(E) ~ E^-Gamma
        try:
            row.update(_from_gbm(name, spec, index, i_minus, i_plus))
        except Exception as error:
            row.update(from_gbm_note=f"prediction failed: {type(error).__name__}: {error}")
        if measured:
            note = "E_peak measured by BAT"
        elif np.isfinite(i_minus) and gamma + 2 * i_minus < 1.3:
            note = "E_peak above BAT's band (photon index below 1.3)"
        elif np.isfinite(i_plus) and gamma - 2 * i_plus > 2.3:
            note = "E_peak below BAT's band (photon index above 2.3)"
        else:
            note = "E_peak not measured by BAT (its photon index allows a peak inside the band)"
        rows.append(dict(row, bat_pl_index=index, bat_pl_minus=i_minus, bat_pl_plus=i_plus, bat_pl_chi2=pl["chi2"], bat_dof=pl["dof"],
                         bat_cpl_gain=gain, bat_Ep=ep if measured else np.nan, bat_Ep_minus=ep_minus if measured else np.nan,
                         bat_Ep_plus=ep_plus if measured else np.nan, note=" ".join(x for x in (note, travel_note) if x)))
    return pd.DataFrame(rows)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    task, names = argv[0], argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    table = {"validate": validate, "episodes": episodes, "precursors": precursors}[task](names)
    path = OUT / {"validate": "validation.csv", "episodes": "episodes_vs_gbm.csv", "precursors": "precursor_spectra.csv"}[task]
    table.to_csv(path, index=False)             # one call writes the whole table: give every burst in one call
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
