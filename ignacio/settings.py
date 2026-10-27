"""
Settings shared by every script in this folder.

Everything specific to GRB 230307A lives here, so analysing another burst
should only need this file changed (plus the precursor and BTI times,
which have to be read off its own light curve first).
Values marked "Dichiara" are from Dichiara et al. (2023, ApJL 954, L29).
"""
from pathlib import Path

# Paths are built from this file's location, so the scripts run from any folder
CODE_DIR = Path(__file__).resolve().parent            # .../MVTTools/ignacio
DATA_DIR = CODE_DIR.parent / "data/fermiGBM/GRB230307A/current"
FIGURE_DIR = CODE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

BURST_ID = "bn230307656"

# Light curves (Dichiara Fig. 1)
BIN_WIDTH = 0.016                          # s
SOFT_BAND = (7.0, 99.0)                    # keV
HARD_BAND = (883.0, 38367.0)               # keV, b1 channel edges
SOFT_DETECTORS = ["n6", "n8", "na"]        # BGO has no channels below ~150 keV
HARD_DETECTORS = ["n6", "n8", "na", "b1"]  # my inference, see 04_check_hard_band.py
PLOT_RANGE = (-5.0, 50.0)                  # s

# Background: poly3 per channel on the quiet stretches (Dichiara Sec. 2)
BKG_WINDOWS = [(-89.0, -7.0), (162.0, 273.0)]   # s
BKG_ORDER = 3

BAD_TIME_INTERVAL = (3.0, 7.0)             # s, pile-up (Dichiara)
PRECURSOR = (-0.064, 0.320)                # s, precursor spectrum (Dichiara)

# Spectral fits (09, 10). Channels overlapping these ranges are fitted; these
# are the ranges that reproduce Dichiara's Band fit, with 484 d.o.f. against their
# 480 (my inference, the paper does not give them, see 09_precursor_band_fit.ipynb)
SPECTRAL_DETECTORS = ["n6", "n8", "na", "b1"]
FIT_RANGE_NAI = (8.0, 900.0)               # keV
FIT_RANGE_BGO = (200.0, 40000.0)           # keV
FLUX_RANGE = (10.0, 1000.0)                # keV, Dichiara Table 2