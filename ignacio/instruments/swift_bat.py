"""Swift/BAT for the bursts that both GBM and BAT saw: the standard BAT chain, run with HEASoft's own tools, and light curves and spectra
with responses for any time window.

BAT is a coded-mask instrument. Each event gets a weight from the shadow the mask casts on its detector for the burst's position
(batmaskwtevt), and the weighted sums are background-subtracted rates per fully illuminated detector. These aren't Poisson counts, so BAT
spectra are fitted with chi-squared, and Bayesian blocks on BAT light curves need the Gaussian fitness. The band is 15-150 keV, so an
E_peak above it can only be a limit.

The chain follows the BAT analysis threads (and the archive pipeline, which tests/test_swift_bat.py reproduces):
- batdetmask finds the detectors that were on (housekeeping bdecb), on top of the global bad-detector map;
- batbinevt makes a detector-plane image of all the events, and bathotpix then flags the hot pixels;
- batmaskwtevt gives every event its mask weight for the burst's position, using the attitude file, and also writes the ray-tracing file
  that batupdatephakw needs to put the burst's position into each spectrum;
- batbinevt then makes mask-weighted light curves and 80-channel spectra for any window;
- batupdatephakw, batphasyserr and batdrmgen add the position keywords, the standard systematic errors and the response of each spectrum.

The data go in ~/swift_bat/data/<obsid>/ from the HEASARC archive (bat/ and auxil/ of observation 00<trigger>000), with the calibration
in ~/caldb."""
import gzip
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

HEADAS = Path(os.environ.get("HEADAS_BAT", Path.home() / "miniforge3/envs/MVTEnvHEA/heasoft"))
CALDB = Path(os.environ.get("CALDB_BAT", Path.home() / "caldb"))
DATA = Path(os.environ.get("BAT_DATA", Path.home() / "swift_bat/data"))
WORK = Path(os.environ.get("BAT_WORK", Path.home() / "swift_bat/work"))
CATALOGUE = Path(os.environ.get("BAT_CATALOGUE", Path.home() / "swift_bat/summary_general.txt"))
BAND = (15.0, 150.0)             # keV, where BAT's response is calibrated for GRB spectra
SPECTRAL_BINS = "CALDB:80"       # the standard 80 channels


@lru_cache(maxsize=1)
def heasoft_env():
    """The environment HEASoft's init script sets, plus the calibration database and a parameter-file folder of this process's own
    (two processes sharing one would overwrite each other's parameters)."""
    out = subprocess.run(["bash", "-c", f"export HEADAS={HEADAS}; . $HEADAS/headas-init.sh; env -0"], capture_output=True, check=True)
    env = dict(item.split("=", 1) for item in out.stdout.decode().split("\0") if "=" in item)
    pfiles = Path(tempfile.mkdtemp(prefix="bat_pfiles_"))
    env.update(CALDB=str(CALDB), CALDBCONFIG=str(CALDB / "software/tools/caldb.config"),
               CALDBALIAS=str(CALDB / "software/tools/alias_config.fits"), PFILES=f"{pfiles};{HEADAS / 'syspfiles'}",
               HEADASNOQUERY="1", HEADASPROMPT="/dev/null")
    return env


def run(tool, *args, **params):
    """One HEASoft tool with positional arguments and name=value parameters; raises with its output if it fails."""
    command = [tool, *map(str, args)] + [f"{k}={v}" for k, v in params.items()]
    done = subprocess.run(command, env=heasoft_env(), capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{tool} failed ({done.returncode}):\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    return done.stdout


def catalogue():
    """The BAT GRB catalogue's summary table (name, trigger id, trigger time in MET and UTC, ground position, T90, the span of the event
    data from the trigger, and the partial coding: the fraction of the mask's pattern the burst's direction illuminates; mask weighting
    is less reliable for a burst near the edge of the field, below about 0.1)."""
    number = lambda x: float(x) if x not in ("", "N/A") else np.nan        # some bursts lack a ground position or a T90
    rows = []
    for line in CATALOGUE.read_text().splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        f = [x.strip() for x in line.split("|")]
        if not f[1].isdigit():
            continue                                                       # bursts without a BAT trigger id
        rows.append(dict(name=f[0], trigger=int(f[1]), met=number(f[2]), utc=f[3], ra=number(f[4]), dec=number(f[5]), t90=number(f[8]),
                         events_from=number(f[12]), events_to=number(f[13]), pcode=number(f[14])))
    return {r["name"]: r for r in rows}


@dataclass
class BatBurst:
    name: str               # BAT catalogue name, e.g. GRB211211A
    obsid: str              # 00<trigger>000
    trigger_met: float      # BAT trigger time, Swift MET (s)
    trigger_utc: str
    ra: float               # ground position (deg)
    dec: float
    data: Path              # the folder holding bat/ and auxil/
    work: Path

    def file(self, part, suffix):
        hits = sorted((self.data / part).rglob(f"sw{self.obsid}{suffix}"))
        if not hits:
            raise FileNotFoundError(f"no sw{self.obsid}{suffix} under {self.data / part}")
        return hits[0]


def burst(name):
    """A burst of the BAT catalogue whose data are under DATA."""
    row = catalogue()[name]
    obsid = f"{row['trigger']:08d}000"
    tops = [p for p in (DATA / obsid, DATA / obsid / obsid) if (p / "bat").is_dir()]
    if not tops:
        raise FileNotFoundError(f"no BAT data for {name} under {DATA / obsid}")
    work = WORK / name
    work.mkdir(parents=True, exist_ok=True)
    return BatBurst(name, obsid, row["met"], row["utc"], row["ra"], row["dec"], tops[-1], work)


def gbm_offset(bat, gbm_trigger_utc):
    """Seconds to add to a time from the BAT trigger to make it a time from the GBM trigger."""
    return float((Time(bat.trigger_utc, scale="utc") - Time(gbm_trigger_utc, scale="utc")).sec)


C_KM_S = 299792.458


def light_travel(bat, gbm_trigdat, gbm_trigger_met, gbm_offset=0.0, at=0.0):
    """How many seconds earlier the burst's light reaches Swift than Fermi, `at` seconds after the GBM trigger: (r_Swift - r_Fermi) . n / c,
    where n is the unit vector to the burst (BAT's position).

    Both positions are taken at that one moment. Fermi's comes from GBM's trigdat file (EVNTRATE: EIC, J2000, km) at gbm_trigger_met + at, and
    Swift's from its orbit file (auxil/sao: POSITION, J2000, km, every second) at bat.trigger_met - gbm_offset + at, with gbm_offset from
    gbm_offset(). The two satellites can be about 14,000 km apart, so this reaches about 45 ms, and it drifts by up to about 0.04 ms a second as
    they move, which is why I work it out at the time being compared. Add it, together with gbm_offset, to a time from the BAT trigger to put
    it on GBM's axis for the same light. It's good to about 0.1 ms, since trigdat only gives Fermi's position once per time bin of up to a few
    seconds."""
    with fits.open(bat.file("auxil", "sao.fits.gz")) as f:
        t, pos = f[1].data["TIME"], np.asarray(f[1].data["POSITION"], float)
        r_swift = np.array([np.interp(bat.trigger_met - gbm_offset + at, t, pos[:, k]) for k in range(3)])
    with fits.open(gbm_trigdat) as f:
        e = f["EVNTRATE"].data
        t, pos = np.asarray(e["TIME"], float), np.asarray(e["EIC"], float)
        order = np.argsort(t)
        r_fermi = np.array([np.interp(gbm_trigger_met + at, t[order], pos[order, k]) for k in range(3)])
    ra, dec = np.radians(bat.ra), np.radians(bat.dec)
    n = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    return float((r_swift - r_fermi) @ n / C_KM_S)


def prepare(bat):
    """The chain up to mask-weighted events: returns the paths of the events, the detector quality map and the ray-tracing file.
    Each step is skipped if its output is already in the burst's work folder."""
    w = bat.work
    events, detmask, dpi, qmap, aux = w / "events.evt", w / "detmask.fits", w / "all.dpi", w / "qmap.fits", w / "raytrace.fits"
    if not events.exists():
        with gzip.open(bat.file("bat", "bevshsp_uf.evt.gz"), "rb") as src, open(events, "wb") as dst:
            shutil.copyfileobj(src, dst)
    if not detmask.exists():
        run("batdetmask", date=bat.trigger_met, outfile=detmask, detmask=bat.file("bat", "bdecb.hk.gz"), clobber="yes", chatter=1)
    if not qmap.exists():
        run("batbinevt", infile=events, outfile=dpi, outtype="DPI", timedel=0, timebinalg="u", energybins="-", weighted="no",
            outunits="counts", clobber="yes", chatter=1)
        run("bathotpix", infile=dpi, outfile=qmap, detmask=detmask, clobber="yes", chatter=1)
    if not aux.exists():
        run("batmaskwtevt", infile=events, attitude=bat.file("auxil", "sat.fits.gz"), ra=bat.ra, dec=bat.dec, detmask=qmap,
            auxfile=aux, clobber="yes", chatter=1)
    return events, qmap, aux


def light_curve(bat, tres, band=BAND):
    """The mask-weighted rate in band (keV) at bins of tres (s). Returns (time from the BAT trigger at the bin centres, rate, error), in counts
    per second per fully illuminated detector, with the background already taken out by the mask weighting. Bins with no events at all are
    left out. Those have zero rate and zero error, and they're where the event data don't reach, e.g. after about +390 s for GRB 211211A."""
    events, qmap, _ = prepare(bat)
    out = bat.work / f"lc_{tres * 1e3:g}ms_{band[0]:g}-{band[1]:g}keV.lc"
    if not out.exists():
        run("batbinevt", infile=events, outfile=out, outtype="LC", timedel=tres, timebinalg="u", energybins=f"{band[0]:g}-{band[1]:g}",
            weighted="yes", outunits="rate", detmask=qmap, clobber="yes", chatter=1)
    with fits.open(out) as f:
        d, h = f[1].data, f[1].header
        # TIME marks the point TIMEPIXR of the way through each bin (0 for batbinevt, i.e. the bin's start); return bin centres
        centre = (0.5 - float(h.get("TIMEPIXR", 0.0))) * tres
        t = np.asarray(d["TIME"], float) + centre - bat.trigger_met
        rate, error = np.asarray(d["RATE"], float).ravel(), np.asarray(d["ERROR"], float).ravel()
    covered = error > 0
    return t[covered], rate[covered], error[covered]


def spectrum(bat, start, stop, label):
    """The 80-channel mask-weighted spectrum of [start, stop] (s from the BAT trigger) with its standard systematic errors, position
    keywords and response. Returns (pha path, rsp path)."""
    events, qmap, aux = prepare(bat)
    pha, rsp = bat.work / f"{label}.pha", bat.work / f"{label}.rsp"
    run("batbinevt", infile=events, outfile=pha, outtype="PHA", timedel=0, timebinalg="u", tstart=f"{bat.trigger_met + start:.6f}",
        tstop=f"{bat.trigger_met + stop:.6f}", energybins=SPECTRAL_BINS, weighted="yes", outunits="rate", detmask=qmap, clobber="yes", chatter=1)
    run("batupdatephakw", pha, aux, chatter=1)
    run("batphasyserr", pha, "CALDB", chatter=1)
    run("batdrmgen", infile=pha, outfile=rsp, hkfile="NONE", clobber="yes", chatter=1)
    return pha, rsp


def against_archive(bat, product="bevas"):
    """Our spectrum and response for the window of one of the archive pipeline's spectra (bevas: the pipeline's main interval, bevps:
    the peak), set against theirs: ratio of total rates, largest channel difference in sigma, largest error ratio, response ratio."""
    archive_pha, archive_rsp = bat.file("bat", f"{product}.pha.gz"), bat.file("bat", f"{product}.rsp.gz")
    with fits.open(archive_pha) as f:
        start, stop = f[1].header["TSTART"] - bat.trigger_met, f[1].header["TSTOP"] - bat.trigger_met
        theirs = f[1].data
    pha, rsp = spectrum(bat, start, stop, f"archive_{product}")
    with fits.open(pha) as f:
        ours = f[1].data
    rate_o, rate_t = np.ravel(ours["RATE"]), np.ravel(theirs["RATE"])
    err_t = np.where(np.ravel(theirs["STAT_ERR"]) > 0, np.ravel(theirs["STAT_ERR"]), np.inf)
    with fits.open(rsp) as a, fits.open(archive_rsp) as b:
        area = lambda m: np.array([np.sum(r) for r in m["SPECRESP MATRIX"].data["MATRIX"]])
        ratio = area(a) / np.where(area(b) > 0, area(b), np.nan)
    return dict(window=(start, stop), total_rate_ratio=float(rate_o.sum() / rate_t.sum()),
                worst_channel_sigma=float(np.max(np.abs(rate_o - rate_t) / err_t)),
                worst_error_ratio=float(np.max(np.abs(np.ravel(ours["STAT_ERR"]) / err_t * np.isfinite(err_t) - np.isfinite(err_t)))),
                response_ratio_range=(float(np.nanmin(ratio)), float(np.nanmax(ratio))))


# ---------------------------------------------------------------- fitting
# BAT spectra are mask-weighted rates with Gaussian errors, so the fit is chi-squared: the model photon spectrum is integrated over each
# input energy bin of the response, folded through the matrix, and compared with the rates over 15-150 keV, with the statistical error
# and the standard systematic fraction (batphasyserr) added in quadrature.

def read_spectrum(pha, rsp):
    """Rates, errors (statistical and systematic in quadrature), channel edges and the dense response (input bins x channels)."""
    with fits.open(pha) as f:
        d = f[1].data
        rate = np.ravel(d["RATE"]).astype(float)
        error = np.hypot(np.ravel(d["STAT_ERR"]).astype(float), np.ravel(d["SYS_ERR"]).astype(float) * rate)
    with fits.open(rsp) as f:
        m, e = f["SPECRESP MATRIX"].data, f["EBOUNDS"].data
        n_chan = len(e)
        matrix = np.zeros((len(m), n_chan))
        for i, row in enumerate(m):
            k = 0
            for first, n in zip(np.atleast_1d(row["F_CHAN"]), np.atleast_1d(row["N_CHAN"])):
                values = np.atleast_1d(row["MATRIX"])[k:k + n]
                matrix[i, first:first + n] = values
                k += n
        return dict(rate=rate, error=error, e_min=np.asarray(e["E_MIN"], float), e_max=np.asarray(e["E_MAX"], float),
                    e_lo=np.asarray(m["ENERG_LO"], float), e_hi=np.asarray(m["ENERG_HI"], float), matrix=matrix)


def _photons(model, params, e_lo, e_hi, steps=8):
    """Photon flux (photons cm^-2 s^-1) in each input energy bin, integrated with the trapezoid rule on a log grid."""
    grid = np.exp(np.linspace(np.log(e_lo), np.log(e_hi), steps + 1))     # (steps+1, bins)
    return np.trapezoid(model(grid, *params), grid, axis=0)


def power_law(e, norm, index):
    """photons cm^-2 s^-1 keV^-1, norm at 50 keV (the BAT catalogue's pivot)"""
    return norm * (e / 50.0) ** index


def cutoff_power_law(e, norm, alpha, e_peak):
    """Comptonized: E^alpha exp(-(2+alpha) E / E_peak), norm at 50 keV"""
    return norm * (e / 50.0) ** alpha * np.exp(-(2.0 + alpha) * (e - 50.0) / e_peak)


# the ranges a fit may explore: a positive norm, an index that keeps (2 + alpha) > 0 for the Comptonized, and E_peak from 1 keV to 100 MeV
power_law.bounds = (np.array([0.0, -6.0]), np.array([np.inf, 3.0]))
cutoff_power_law.bounds = (np.array([0.0, -1.99, 1.0]), np.array([np.inf, 1.9, 1e5]))    # E_peak is a peak only for alpha > -2


def fit(spec, model, start, band=BAND):
    """Least-squares (chi-squared) fit of model to the channels inside band, within the model's bounds; returns the best parameters,
    chi2, dof and 68% errors from the profile of chi2 (delta chi2 = 1) for each parameter. An error that reaches a bound before chi2 has
    risen by 1 is NaN: the parameter is only limited on that side."""
    from scipy.optimize import least_squares, brentq
    use = (spec["e_min"] >= band[0] - 1e-6) & (spec["e_max"] <= band[1] + 1e-6) & (spec["error"] > 0)
    lo, hi = getattr(model, "bounds", (np.full(len(start), -np.inf), np.full(len(start), np.inf)))

    def residuals(p):
        predicted = _photons(model, p, spec["e_lo"], spec["e_hi"]) @ spec["matrix"]
        r = (spec["rate"][use] - predicted[use]) / spec["error"][use]
        return np.where(np.isfinite(r), r, 1e6)

    def inner(edge, sign):
        with np.errstate(invalid="ignore", over="ignore"):
            return np.where(np.isfinite(edge), edge + sign * 1e-9 * (np.abs(edge) + 1), edge)
    start = np.clip(np.asarray(start, float), inner(lo, 1), inner(hi, -1))
    best = least_squares(residuals, start, bounds=(lo, hi), x_scale="jac", max_nfev=20000)
    chi2 = float(np.sum(best.fun ** 2))
    errors = []
    for i in range(len(start)):
        others_lo, others_hi = np.delete(lo, i), np.delete(hi, i)

        def worse(value):
            fixed = lambda q: residuals(np.insert(q, i, value))
            q0 = np.clip(np.delete(best.x, i), inner(others_lo, 1), inner(others_hi, -1))
            q = least_squares(fixed, q0, bounds=(others_lo, others_hi), x_scale="jac", max_nfev=5000)
            return float(np.sum(q.fun ** 2)) - chi2 - 1.0

        step = max(abs(best.x[i]) * 0.02, 1e-3)
        limits = []
        for direction, edge in ((-1, lo[i]), (1, hi[i])):
            far, tries, crossed = best.x[i], 0, False
            while tries < 40:
                far = far + direction * step
                if (direction < 0 and far <= edge) or (direction > 0 and far >= edge):
                    far = edge + (1e-9 if direction < 0 else -1e-9) * (abs(edge) + 1)
                    crossed = worse(far) > 0
                    break
                if worse(far) > 0:
                    crossed = True
                    break
                step *= 1.5
                tries += 1
            limits.append(brentq(worse, best.x[i], far) if crossed else np.nan)
        errors.append((best.x[i] - limits[0], limits[1] - best.x[i]))
    return dict(params=best.x, chi2=chi2, dof=int(use.sum() - len(start)), errors=errors)


def spectrum_during_slew(bat, start, stop, label, slice_s=5.0):
    """The spectrum of [start, stop] with a response averaged over the window, for windows in which Swift slews and the burst moves across
    the field. There's one response per slice of slice_s seconds, weighted by the counts (15-150 keV) that slice contributes, which is what the
    BAT catalogue does. Returns the read_spectrum dict of the whole window with the averaged matrix in place of the single one."""
    pha, rsp = spectrum(bat, start, stop, label)
    whole = read_spectrum(pha, rsp)
    edges = np.arange(start, stop, slice_s)
    edges = np.append(edges, stop) if stop - edges[-1] > 1e-6 else edges
    if len(edges) <= 2:
        return whole
    matrices, weights = [], []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        p, r = spectrum(bat, a, b, f"{label}_slice{i:03d}")
        part = read_spectrum(p, r)
        use = (part["e_min"] >= BAND[0]) & (part["e_max"] <= BAND[1])
        weights.append(max(float(np.sum(part["rate"][use])) * (b - a), 0.0))
        matrices.append(part["matrix"])
        for path in (p, r):
            Path(path).unlink(missing_ok=True)
    weights = np.asarray(weights)
    if weights.sum() <= 0:
        return whole
    whole["matrix"] = np.tensordot(weights / weights.sum(), np.asarray(matrices), axes=1)
    return whole


# ---------------------------------------------------------------- episodes
# Zhu et al.'s episode rules (zhu.episodes.find_episodes) on BAT's light curve, to compare the episode boundaries of the bursts both
# instruments saw. Only the blocks differ: mask-weighted rates have Gaussian errors, so the Bayesian blocks use the 'measures' fitness
# (Scargle et al. 2013, section 3.3). The rates per detector are scaled by the number of good detectors to whole-array counts; a common
# scale changes nothing in the blocks or the significances, it only makes the numbers comparable in size with GBM's. One difference stays:
# BAT's errors include the burst's own counting noise, while GBM's significances use the background's alone, so a BAT block of the same
# brightness is a little less significant. Fermi and Swift are up to about 14,000 km apart, so a burst's photons reach them up to about 45 ms
# apart depending on its direction; bat_compare adds that light-travel time (light_travel) to the offset between the two time axes.

def blocks_gaussian(edges, counts, noise, p0=0.05):
    """Bayesian blocks of counts with Gaussian noise on contiguous bins, as a table with the columns zhu.episodes.blocks_of gives."""
    import pandas as pd
    from astropy.stats import bayesian_blocks
    centres = 0.5 * (edges[1:] + edges[:-1])
    change = bayesian_blocks(centres, counts, noise, fitness="measures", p0=p0)
    cp = np.unique(np.r_[0, np.searchsorted(centres, change[1:-1]), len(counts)])
    net = np.add.reduceat(counts, cp[:-1])
    variance = np.add.reduceat(noise ** 2, cp[:-1])
    start, stop = edges[cp[:-1]], edges[cp[1:]]
    blocks = pd.DataFrame(dict(start=start, stop=stop, i0=cp[:-1], i1=cp[1:], nbins=np.diff(cp), n=net, bkg=variance,
                               live=stop - start, net=net))
    blocks["sigma"] = blocks.net / np.sqrt(blocks.bkg)
    blocks["snr_bin"] = blocks.net / np.sqrt(blocks.bkg * blocks.nbins)
    blocks["rate"] = blocks.net / (blocks.stop - blocks.start)
    blocks["rate_err"] = np.sqrt(blocks.bkg) / (blocks.stop - blocks.start)
    return blocks


def episodes(bat, tres, offset, span, around=None, modes=("zhu2025", "zhu2026")):
    """The episodes of the BAT light curve under Zhu et al.'s rules, at time resolution tres, over span (s from the GBM trigger), with times
    counted from the GBM trigger (the offset is gbm_offset plus light_travel). If the BAT data have gaps inside span, I only use the unbroken
    stretch that contains the time `around` (the middle of span by default), and the result's flags say so. Returns {mode: EpisodeResult} and
    the blocks."""
    from zhu.background import LightCurve
    from zhu.episodes import find_episodes
    t, rate, error = light_curve(bat, tres)
    t = t + offset
    keep = (t - tres / 2 >= span[0]) & (t + tres / 2 <= span[1])
    t, rate, error = t[keep], rate[keep], error[keep]
    cut = None
    breaks = np.flatnonzero(np.diff(t) > 1.5 * tres)
    if len(breaks):
        around = 0.5 * (span[0] + span[1]) if around is None else around
        starts, stops = np.r_[0, breaks + 1], np.r_[breaks + 1, len(t)]
        inside = [(a, b) for a, b in zip(starts, stops) if t[a] - tres <= around <= t[b - 1] + tres]
        a, b = inside[0] if inside else max(zip(starts, stops), key=lambda ab: ab[1] - ab[0])
        cut = (float(t[a] - tres / 2), float(t[b - 1] + tres / 2))
        t, rate, error = t[a:b], rate[a:b], error[a:b]
    if len(t) < 10:
        raise ValueError(f"{bat.name}: too little BAT data in {span} at {tres} s")
    out_lc = bat.work / f"lc_{tres * 1e3:g}ms_{BAND[0]:g}-{BAND[1]:g}keV.lc"
    with fits.open(out_lc) as f:
        good = float(f[1].header.get("NGOODPIX", 1.0))
    edges = np.r_[t - tres / 2, t[-1] + tres / 2]
    counts, noise = rate * tres * good, error * tres * good
    blocks = blocks_gaussian(edges, counts, noise)
    lc = LightCurve(edges, counts, np.full(len(counts), tres), np.zeros(len(counts)), tres)
    found = {mode: find_episodes(lc, blocks, mode=mode, burst=bat.name) for mode in modes}
    if cut is not None:
        for result in found.values():
            result.flags.append(f"bat_data_cut_to:{cut[0]:.2f}..{cut[1]:.2f}")
    return found, blocks
