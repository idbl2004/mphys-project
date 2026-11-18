"""Find a burst, get its files and set it up from the GBM catalogue, with no tuning by hand."""
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.io import fits

from gdt.missions.fermi.gbm.tte import GbmTte

from . import CODE_DIR
from lightcurve_tools import tte_path
from sample_tools import link_v00
from spectral_fit_tools import rsp2_path

ARCHIVE = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers"
WANTED = re.compile(r"_tte_|_bcat_|_tcat_|_trigdat_|_scat_|_cspec_.*\.(pha|rsp2)$")
DATA_ROOT = CODE_DIR.parent / "data" / "fermiGBM"
ALL_NAI = [f"n{i}" for i in range(10)] + ["na", "nb"]
ALL_DETECTORS = ALL_NAI + ["b0", "b1"]

# The redshifts used in notebooks 18 and 19. Anything else needs --z, and without one the energetics are left out.
KNOWN_REDSHIFTS = {"GRB230307A": 0.065, "GRB211211A": 0.076, "GRB180728A": 0.117, "GRB190829A": 0.0785, "GRB190114C": 0.4245,
                   "GRB090618": 0.54, "GRB111228A": 0.716, "GRB211023A": 0.390}

# The two bursts analysed by hand. Only used with preset=True, to test the tool against the tables from notebooks 12 and 14.
PRESETS = {
    "GRB230307A": dict(trigger="bn230307656", detectors=["n6", "n8", "na", "b1"], bkg_windows=[(-89.0, -7.0), (162.0, 273.0)],
                       windows_file=CODE_DIR / "shared" / "windows_230307A.csv", table=CODE_DIR / "shared" / "epeak_230307A.csv", bad_intervals=[(3.0, 7.0)]),
    "GRB211211A": dict(trigger="bn211211549", detectors=["n2", "na", "b0"], bkg_windows=[(-130.0, -10.0), (150.0, 450.0)],
                       windows_file=CODE_DIR / "shared" / "windows_211211A.csv", table=CODE_DIR / "shared" / "epeak_211211A.csv", bad_intervals=[]),
}


class BurstError(Exception):
    """The burst can't be set up, with the reason. The tool stops rather than guess."""


@dataclass
class Burst:
    name: str
    trigger: str
    data_dir: Path
    t90: float
    t90_start: float
    detectors: list
    angles: dict
    catalogue: dict
    channels: dict
    tte_range: tuple
    bkg_windows: list = field(default_factory=list)
    bkg_order: int = 3
    redshift: float = None
    flags: list = field(default_factory=list)
    bkg_chi2: dict = field(default_factory=dict)

    @property
    def nai(self):
        return [d for d in self.detectors if d.startswith("n")]


def _listing(url):
    result = subprocess.run(["curl", "-sL", "-m", "90", url], capture_output=True, text=True)
    if result.returncode != 0:
        raise BurstError(f"couldn't read {url}: {result.stderr.strip()[:100]}")
    return result.stdout


def _describe(trigger, listing):
    """A trigger with its T90 and fluence from the catalogue file, so a person can tell which one a burst name means."""
    url = f"{ARCHIVE}/20{trigger[2:4]}/{trigger}/current/"
    name = sorted(re.findall(r'href="(glg_bcat_all_[^"]+\.fit)"', listing))[-1]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / name
        subprocess.run(["curl", "-sSfL", "-m", "120", "-o", str(path), url + name], capture_output=True)
        header = fits.getheader(path, 0)
    return f"{trigger} (T90 {header['T90']:.1f} s, fluence {header['FLU']:.2e} erg/cm2)"


def find_trigger(name):
    """The GBM trigger for a name like GRB180728A or GRB180728728.

    The letter comes from the order in which all instruments reported bursts that day, and GBM's files don't record it. So if the day has
    several GBM bursts this stops and lists them, because choosing one would be a guess.
    """
    found = re.fullmatch(r"(?:GRB|bn)?(\d{6})(\d{3})", name)
    if found:
        return f"bn{found.group(1)}{found.group(2)}"
    found = re.fullmatch(r"GRB(\d{6})[A-Z]*", name)
    if not found:
        raise BurstError(f"{name} isn't a GRB name like GRB180728A. Give the trigger with --trigger.")
    day = found.group(1)
    triggers = sorted(set(re.findall(rf"bn{day}\d{{3}}", _listing(f"{ARCHIVE}/20{day[:2]}/"))))
    listings = {t: _listing(f"{ARCHIVE}/20{day[:2]}/{t}/current/") for t in triggers}
    bursts = [t for t in triggers if "glg_bcat_all" in listings[t]]
    if len(bursts) == 1:
        return bursts[0]
    if not bursts:
        raise BurstError(f"no GBM burst on 20{day}: triggers that day were {triggers or 'none'}")
    raise BurstError(f"several GBM bursts on 20{day}: " + "; ".join(_describe(t, listings[t]) for t in bursts) + ". Say which with --trigger.")


def download(name, trigger, root=DATA_ROOT):
    """TTE, responses and catalogue files, skipping any already complete. Stops if a file can't be fetched."""
    url = f"{ARCHIVE}/20{trigger[2:4]}/{trigger}/current/"
    names = sorted(f for f in set(re.findall(r'href="(glg_[^"]+)"', _listing(url))) if WANTED.search(f))
    if not names:
        raise BurstError(f"nothing to download at {url}")
    out = Path(root) / name / "current"
    out.mkdir(parents=True, exist_ok=True)
    for f in names:
        target = out / f
        head = subprocess.run(["curl", "-sIL", "-m", "60", url + f], capture_output=True, text=True).stdout
        sizes = re.findall(r"(?i)content-length:\s*(\d+)", head)
        if target.exists() and sizes and target.stat().st_size == int(sizes[-1]):
            continue
        result = subprocess.run(["curl", "-sSfL", "--retry", "3", "-m", "600", "-o", str(target), url + f], capture_output=True, text=True)
        if result.returncode != 0 or (sizes and target.stat().st_size != int(sizes[-1])):
            target.unlink(missing_ok=True)
            raise BurstError(f"couldn't download {f}: {result.stderr.strip()[:100]}")
    return out


def _angle(data_dir, trigger, det):
    """The angle between the detector and the burst, from the response header, at the first response after the burst starts. None if
    there's no response. Some responses from 2010-2012 have no DET_ANG. For those I work the angle out from the burst's spacecraft azimuth and
    elevation (SRC_AZ, SRC_EL) and the detector's pointing (_angle_from_az_el), which gives DET_ANG back to within 0.1 deg on the responses
    that have both."""
    try:
        path = rsp2_path(data_dir, trigger, det)
    except IndexError:
        return None
    with fits.open(path) as h:
        trigtime = h[0].header["TRIGTIME"]
        header = [e.header for e in h[2:] if e.header["TSTOP"] - trigtime > -1.0][0]
        if "DET_ANG" in header:
            return float(header["DET_ANG"])
        return _angle_from_az_el(det, header["SRC_AZ"], header["SRC_EL"])


def _angle_from_az_el(det, src_az, src_el):
    """Angle (deg) between detector det's axis and a source at spacecraft azimuth src_az and elevation src_el (deg), with the detector
    pointings of GDT's GbmDetectors."""
    import astropy.units as u
    from gdt.missions.fermi.gbm.detectors import GbmDetectors
    d = GbmDetectors.from_str(det)
    az, zen = d.azimuth.to_value(u.rad), d.zenith.to_value(u.rad)
    s_az, s_zen = np.radians(src_az), np.radians(90.0 - src_el)
    cos = np.sin(zen) * np.sin(s_zen) * np.cos(az - s_az) + np.cos(zen) * np.cos(s_zen)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _detector_name(name):
    return ("b" if name.startswith("BGO") else "n") + format(int(name.split("_")[1]), "x")


def set_up(name, trigger, data_dir=None, detectors=None, redshift=None):
    """Read T90, detectors and the catalogue's fit, and check the files are all there and cover the burst."""
    data_dir = Path(data_dir) if data_dir else DATA_ROOT / name / "current"
    if not data_dir.is_dir():
        raise BurstError(f"no data in {data_dir}")
    link_v00(data_dir)
    flags = []
    bcat = sorted(data_dir.glob(f"glg_bcat_all_{trigger}_v*.fit"))
    if not bcat:
        raise BurstError(f"no glg_bcat file for {trigger}")
    with fits.open(bcat[-1]) as h:
        t90, t90_start = float(h[0].header["T90"]), float(h[0].header["T90START"])

    catalogue, channels, catalogue_detectors = None, {}, None
    scat = sorted(data_dir.glob(f"glg_scat_all_{trigger}_flnc_comp_v*.fit"))
    if scat:
        with fits.open(scat[-1]) as h:
            catalogue_detectors = [_detector_name(n) for n in h["DETECTOR DATA"].data["DETNAM"]]
            channels = {d: tuple(int(c) for c in fit) for d, fit in zip(catalogue_detectors, h["DETECTOR DATA"].data["FITCHAN"])}
            p = h["FIT PARAMS"].data
            catalogue = dict(interval=tuple(float(t) for t in p["TIMEBIN"][0]), Ep=float(p["PARAM1"][0][0]), Ep_err=float(p["PARAM1"][0][1]),
                             alpha=float(p["PARAM2"][0][0]), alpha_err=float(p["PARAM2"][0][1]))
    chosen = list(detectors) if detectors else catalogue_detectors
    if not chosen:
        raise BurstError(f"no catalogue fit for {trigger}, so no detectors to use. Give them with --detectors.")

    angles = {d: _angle(data_dir, trigger, d) for d in chosen}
    missing = [d for d, a in angles.items() if a is None]
    if missing:
        flags.append("no_response:" + "+".join(missing))
        chosen = [d for d in chosen if d not in missing]
    if not [d for d in chosen if d.startswith("n")]:
        raise BurstError(f"no NaI detector with a response among {chosen + missing}")
    if not [d for d in chosen if d.startswith("b")]:
        flags.append("no_bgo")
    if not any(angles[d] < 60.0 for d in chosen if d.startswith("n")):
        flags.append("no_nai_within_60deg")

    try:
        tte = GbmTte.open(tte_path(data_dir, trigger, chosen[0]))
    except (IndexError, FileNotFoundError):
        raise BurstError(f"no TTE file for {chosen[0]} of {trigger}")
    tte_range = tuple(float(t) for t in tte.time_range)
    tte.close()
    # A burst whose T90 starts before the TTE does has no background before it, and its first emission isn't in the data
    if t90_start - 5.0 <= tte_range[0] + 1.0:
        raise BurstError(f"T90 starts at {t90_start:.1f} s and the TTE only at {tte_range[0]:.1f} s, so the first emission and the background before it aren't in the data")
    if redshift is None:
        redshift = KNOWN_REDSHIFTS.get(name)
    return Burst(name=name, trigger=trigger, data_dir=data_dir, t90=t90, t90_start=t90_start, detectors=chosen, angles={d: angles[d] for d in chosen},
                 catalogue=catalogue, channels=channels, tte_range=tte_range, redshift=redshift, flags=flags)


def event_times(burst, detectors=ALL_DETECTORS, band=None, time_range=None):
    """Sorted event times per detector, for every detector with a TTE file."""
    out = {}
    for det in detectors:
        try:
            tte = GbmTte.open(tte_path(burst.data_dir, burst.trigger, det))
        except (IndexError, FileNotFoundError):
            continue
        data = tte.slice_energy(band) if band else tte
        times = np.asarray(data.data.times)
        tte.close()
        if time_range:
            times = times[(times > time_range[0]) & (times < time_range[1])]
        out[det] = np.sort(times)
    return out
