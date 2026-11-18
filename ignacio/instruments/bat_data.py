"""Everything instruments/swift_bat.py needs from outside the repository, fetched the same way every time. It's all public, from HEASARC
and the BAT GRB catalogue at NASA GSFC. Run from ignacio/:

    python -m instruments.bat_data caldb        the BAT calibration database (about 70 MB), into ~/caldb
    python -m instruments.bat_data catalogue    the BAT GRB catalogue's summary, 1-s peak and T100 power-law tables, into ~/swift_bat
    python -m instruments.bat_data match        which bursts of Zhu et al.'s sample BAT saw (sample_in_bat_catalogue.csv)
    python -m instruments.bat_data fetch        the bat/ and auxil/ folders of each matched burst's observation, into ~/swift_bat/data

HEASoft itself (with the BAT tools) lives in the MVTEnvHEA environment, and swift_bat.py finds it at ~/miniforge3/envs/MVTEnvHEA/heasoft."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import swift_bat as B

CODE_DIR = Path(__file__).resolve().parents[1]
HOME = Path.home() / "swift_bat"
HEASARC = "https://heasarc.gsfc.nasa.gov/FTP"
BATCAT = "https://swift.gsfc.nasa.gov/results/batgrbcat/summary_cflux"
CALDB_FILE = "goodfiles_swift_bat_20230607.tar.Z"          # the BAT calibration release used for every result here


def _get(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sSf", "-o", str(path), url], check=True)


def caldb():
    """The BAT calibration files and the CALDB index files, as swift_bat.heasoft_env expects them."""
    root = B.CALDB
    for name in ("caldb.config", "alias_config.fits"):
        _get(f"{HEASARC}/caldb/software/tools/{name}", root / "software" / "tools" / name)
    archive = root / CALDB_FILE
    _get(f"{HEASARC}/caldb/data/swift/bat/{CALDB_FILE}", archive)
    subprocess.run(f"gzip -dc '{archive}' | tar -xf - -C '{root}'", shell=True, check=True)
    print("CALDB in", root)


def catalogue():
    """The catalogue's general table and the power-law fits of its 1-s peak and T100 windows."""
    _get(f"{BATCAT}/summary_general_info/summary_general.txt", HOME / "summary_general.txt")
    _get(f"{BATCAT}/summary_1s_peak/summary_pow_parameters.txt", HOME / "catalogue_1s" / "pow.txt")
    _get(f"{BATCAT}/summary_T100/summary_pow_parameters.txt", HOME / "catalogue_T100" / "summary_pow_parameters.txt")
    print("catalogue tables in", HOME)


def match(minutes=3.0):
    """Bursts of the sample whose GBM trigger time (from the trigger name, to 0.001 day) is within `minutes` of a BAT trigger."""
    sys.path.insert(0, str(CODE_DIR))
    from zhu.catalogue import read_table_2026
    from zhu.run import read_sample
    sample, table = read_sample(), read_table_2026()
    rows = []
    for line in B.CATALOGUE.read_text().splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        f = [x.strip() for x in line.split("|")]
        utc = pd.to_datetime(f[3], errors="coerce") if len(f) > 8 else pd.NaT
        if f[1].isdigit() and not pd.isna(utc):            # rows without a trigger id or a time ('N/A') can't be matched
            rows.append(dict(bat=f[0], trig=int(f[1]), utc=utc, bat_t90=f[8]))
    bat = pd.DataFrame(rows)
    out = []
    for name, row in sample.iterrows():
        d = row.trigger[2:8]
        gbm = pd.Timestamp(f"20{d[:2]}-{d[2:4]}-{d[4:6]}") + pd.Timedelta(days=int(row.trigger[8:11]) / 1000)
        gap = (bat.utc - gbm).abs()
        i = gap.idxmin()
        if gap[i] < pd.Timedelta(minutes=minutes):
            out.append(dict(burst=name, Type=table.loc[name, "Type"], group=row.reason, bat=bat.loc[i, "bat"], trig=bat.loc[i, "trig"],
                            month=bat.loc[i, "utc"].strftime("%Y_%m"), bat_t90=bat.loc[i, "bat_t90"], dt_s=gap[i].total_seconds()))
    table_out = pd.DataFrame(out)
    table_out.to_csv(HOME / "sample_in_bat_catalogue.csv", index=False)
    print(len(table_out), "of", len(sample), "sample bursts are in the BAT catalogue")
    return table_out


def fetch(names=None):
    """The bat/ and auxil/ folders of observation 00<trigger>000 for each matched burst (or the BAT names given)."""
    matched = pd.read_csv(HOME / "sample_in_bat_catalogue.csv")
    if names:
        matched = matched[matched.bat.isin(names)]
    for row in matched.itertuples():
        obs = f"{row.trig:08d}000"
        month = getattr(row, "month", None)
        if not isinstance(month, str):
            cat = B.catalogue()[row.bat]
            month = cat["utc"][:7].replace("-", "_")
        target = B.DATA / obs
        for part in ("bat", "auxil"):
            subprocess.run(["wget", "-q", "-e", "robots=off", "-r", "-np", "-nH", "--cut-dirs=6", "-R", "index.html*", "-P", str(target),
                            f"{HEASARC}/swift/data/obs/{month}/{obs}/{part}/"], check=False)
        print(row.bat, obs, "->", target)


if __name__ == "__main__":
    task = sys.argv[1]
    {"caldb": caldb, "catalogue": catalogue, "match": match, "fetch": lambda: fetch(sys.argv[2:] or None)}[task]()
