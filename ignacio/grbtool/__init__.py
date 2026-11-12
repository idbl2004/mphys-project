"""
grbtool: give it a GBM burst and it returns E_peak, the MVT and the rest of the analysis in one table.

It joins up the steps developed in the numbered scripts and notebooks of this folder, and imports their tools as
they are. Run it from ignacio/ with:  python -m grbtool GRB180728A --z 0.117

Steps, one module each:
    bursts      find the trigger, download the files, set the burst up from the GBM catalogue
    windows     background intervals, precursor, gap, main emission and time slices, with one rule for every burst
    fitting     power law, Comptonized, Band, two-break and blackbody in each window, and one rule for the model
    checks      rates, telemetry limit, data gaps, live fraction, and the flags
    physics     k-correction, E_iso, L_iso, Amati plane, shattering-flare budget and spectral lags
    mvt         a first look at the MVT with Rodrigo's Haar code, on the same windows
    pipeline    runs them in order and writes the tables
    joint       joins E_peak to the MVT table from Rodrigo's program by label, and draws them
"""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
