"""coppin: a second sample of GBM bursts with a precursor, from Coppin, de Vries & van Eijndhoven (2020, Phys. Rev. D 102, 103014),
measured with the same code as Zhu et al.'s sample (the zhu package). The point is to get the precursor's E_peak, and windows for the MVT,
for bursts that no precursor-classification study has measured.

Their search (2684 GBM bursts from 2008-2019, Bayesian blocks on the summed NaI with a 30 Hz threshold in at least two detectors) found
244 precursors in 217 bursts. Their table (shared/coppin2020_precursor_table.txt, from the IceCube GRBweb page) gives each precursor's start
relative to the start of the prompt emission, and its duration. sample.py turns those into times from the GBM trigger using the GBM
catalogue (shared/fermigbrst_extract.csv), keeps the long bursts that aren't in Zhu et al.'s table and whose precursor the TTE data
probably cover, and gives each one a light-curve bin from a rule fitted to Zhu et al.'s own choices, since they don't say how they chose
theirs. After that, the episodes, their errors, the half and twice bins, the lags and the spectra are just zhu.run unchanged, into
results/coppin/.

From ignacio/:
    python -m coppin sample               write shared/coppin_sample.csv
    python -m coppin prefetch             download the data of the sample
    python -m coppin batch --workers 3    measure every burst that isn't finished (several batches can run side by side)
    python -m coppin status               one line per burst"""
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
SHARED = CODE_DIR / "shared"
RESULTS = CODE_DIR / "results" / "coppin"
TABLE = SHARED / "coppin2020_precursor_table.txt"
CATALOGUE = SHARED / "fermigbrst_extract.csv"
SAMPLE = SHARED / "coppin_sample.csv"
