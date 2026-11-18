"""grbtool: give it a GBM burst and it gives back E_peak, the MVT and the rest of the analysis in one table.

It joins up the steps I developed in the numbered scripts and notebooks of this folder, and imports their tools as they are. Run it from
ignacio/ with:  python -m grbtool GRB180728A --z 0.117

The steps, one module each:
- bursts: find the trigger, download the files, and set the burst up from the GBM catalogue;
- windows: the background intervals, the precursor, the gap, the main emission and time slices, with one rule for every burst;
- fitting: power law, Comptonized, Band, two-break and blackbody fits in each window, and one rule for picking the model;
- checks: rates, the telemetry limit, data gaps, live fraction, and the flags;
- physics: k-correction, E_iso, L_iso, the Amati plane, the shattering-flare budget and spectral lags;
- mvt: a first look at the MVT with Rodrigo's Haar code, on the same windows;
- pipeline: runs them in order and writes the tables;
- joint: joins E_peak to the MVT table from Rodrigo's program by label, and draws them;
- samples: every sample of the project, and one burst measured the way the samples were (--zhu, --swift)."""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
