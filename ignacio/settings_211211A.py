"""
Settings for GRB 211211A (bn211211549), with the same names as settings.py.

Used by 14_grb211211a.ipynb to run the GRB 230307A pipeline on a second burst.
Detectors are the ones the GBM catalogue uses for this burst. The background
windows were chosen from the light curve in notebook 14 (flat before the burst,
back to the pre-burst level by ~150 s).
"""
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data/fermiGBM/GRB211211A/current"
FIGURE_DIR = CODE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

BURST_ID = "bn211211549"
TRIGGER_MET = 660921004.65092              # s, GBM trigger time

SOFT_BAND = (7.0, 99.0)                    # keV
SOFT_DETECTORS = ["n2", "na"]
SPECTRAL_DETECTORS = ["n2", "na", "b0"]    # as in the GBM catalogue fits

BKG_WINDOWS = [(-130.0, -10.0), (150.0, 450.0)]   # s
BKG_ORDER = 3

FIT_RANGE_NAI = (8.0, 900.0)               # keV, same as for GRB 230307A
FIT_RANGE_BGO = (200.0, 40000.0)           # keV
FLUX_RANGE = (10.0, 1000.0)                # keV

WINDOWS_FILE = CODE_DIR / "shared" / "windows_211211A.csv"   # written by notebook 14
