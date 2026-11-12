"""
zhu: the bursts of Zhu et al. (2025, MNRAS 541, 3236) and Zhu et al. (2026, ApJ 1003, 142), measured again with our own tools.

They split each burst into a precursor (PE), main emission (ME) and sometimes extended emission (EE) with Bayesian blocks, describe it
with a few numbers (durations, the gap after the precursor, an "MVT", lags, E_p of the main emission), cluster the bursts with t-SNE and
UMAP, and condense the result into one index, EPI. This package does the same steps with their definitions, adds an error bar to each
number, adds the precursor's own E_p, and tests the classification. Nothing in the shared tools or in grbtool is changed: they are
imported as they are.

Modules, in the order they're used:
    catalogue   their tables (and the few 2025 rows printed in the paper), the literature values with where each was read, the sample,
                EPI, redshifts
    io          each burst's results folder, its state.json (every stage done, failed or skipped, with the code version), safe writes
    background  background intervals clear of the burst, and the light curve with its background
    episodes    Bayesian blocks, the episode rule in three modes (zhu2025, zhu2026, project), the MVT with its error, and the rule again
                on simulated light curves and at other time bins
    lags        spectral lags in their bands at a resolution chosen by a written rule, with a check on a known shift
    spectra     their five models and BIC choice, next to our own rule, on the same spectra
    quality     what each flag means, and the status (measured, limit, lower limit, unstable...) every quoted number carries
    run         one burst from the data to its tables, a batch that can be stopped and restarted, and the downloads
    features    our numbers read back into one row per burst and mode, set against theirs, and EPI with its uncertainty
    classify    their clustering repeated on their table and tested, and EPI's line with an uncertainty
    physics     the precursor-to-main delay as an inspiral time, the size the variability allows
    laws        the search for the simplest classification law (a product of powers of measured numbers) and its honest score

From ignacio/:  python -m zhu.run burst 211211A      (one burst)
                python -m zhu.run batch z25           (a group of the sample; see zhu/run.py)
Tests:          python -m pytest tests/test_zhu.py -q
"""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
SHARED = CODE_DIR / "shared"
RESULTS = CODE_DIR / "results" / "zhu"
