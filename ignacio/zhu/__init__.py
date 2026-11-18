"""zhu: the bursts of Zhu et al. (2025, MNRAS 541, 3236) and Zhu et al. (2026, ApJ 1003, 142), measured again with my own tools.

Zhu et al. split each burst into a precursor (PE), a main emission (ME) and sometimes an extended emission (EE), using Bayesian blocks.
They describe every burst with a handful of numbers: the durations, the quiet gap after the precursor, an "MVT", spectral lags and the
E_peak of the main emission. Then they cluster the bursts with t-SNE and UMAP and boil the result down to one index, EPI. I wanted to
redo those steps with their own definitions, put an error bar on every number, add the precursor's own E_peak (which they never use),
and then test their classification. The shared tools and grbtool are imported as they are, and nothing in them was changed for this.

The modules, roughly in the order a burst goes through them:
- catalogue: their tables (and the few 2025 rows printed only in the paper), every literature value with where I read it, the sample,
  EPI and the redshifts.
- io: each burst's results folder and its state.json, which records every stage as done, failed or skipped, with the code version.
- background: background intervals that stay clear of the burst, and the light curve with its background.
- episodes: the Bayesian blocks, the episode rule in three modes (zhu2025, zhu2026, project), the MVT with its error, and the rule run
  again on simulated light curves and at other time bins.
- lags: spectral lags in their bands, with the time resolution picked by a written rule and a check on a known shift.
- spectra: their five models and BIC choice, next to our rule, on the same spectra.
- quality: what each flag means, and the status (measured, limit, lower limit, unstable...) that every quoted number carries.
- run: one burst from the data to its tables, a batch that can be stopped and restarted, and the downloads.
- refit: fits run again with more time, or on other windows, and the GBM catalogue's own fits repeated.
- features: my numbers read back into one row per burst and mode, set against theirs, and EPI with its uncertainty.
- classify: their clustering repeated on their own table and tested, and EPI's line with an uncertainty.
- physics: the precursor-to-main delay read as an inspiral time, and the size the variability allows.
- laws: the search for the simplest classification law (a product of powers of measured numbers) and a fair score for it.

From ignacio/:  python -m zhu.run burst 211211A     (one burst)
                python -m zhu.run batch z25          (one group of the sample, see run.py)
Tests:          python -m pytest tests/test_zhu.py -q"""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
SHARED = CODE_DIR / "shared"
RESULTS = CODE_DIR / "results" / "zhu"
