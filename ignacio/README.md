# GRB 230307A: light curves, precursor and time-resolved spectra (MPhys, Ignacio)

Reproduction of Dichiara et al. (2023, ApJL 954, L29) from Fermi/GBM TTE data, Fig. 1 to Table 2, then checks
that go beyond the paper, the same pipeline run on GRB 211211A, and a first run on a sample of other bursts. Scripts 01-07 run from the repo root
(`python ignacio/03_reproduce_dichiara_fig1.py`); notebooks 08-20 run from `ignacio/` with the MVTEnv kernel, in
the order given under Run times. Figures go to `ignacio/figures/` (01 and 02 write theirs to `ignacio/`, and those aren't tracked). Burst, data folder, detectors, bands and time windows are in
`settings.py` (`settings_211211A.py` for GRB 211211A).

| Script / notebook | What it does | Key result |
|---|---|---|
| `01_first_lightcurve.py` | n6 light curve, 16 ms, 8-1000 keV | double-peaked precursor |
| `02_four_detectors.py` | same, summed over n6, n8, na, b1 | |
| `03_reproduce_dichiara_fig1.py` | Fig. 1: 7-99 keV and 883-38367 keV | soft 42 pre-burst, peak 604 at 6.104 s; hard 11, peak 579 at 1.832 s |
| `04_check_hard_band.py` | which detectors make the hard band | all four detectors match the paper; b1 alone is ~20% short |
| `05_background_subtracted.py` | poly3 background, dead-time corrected net counts | precursor 5995 +/- 86 counts soft, 37 +/- 18 hard (not detected) |
| `06_pileup_deadtime_gaps.py` | live fraction and data gaps per detector | b1 live fraction down to 0.77; ~2 ms gaps at 4.80, 5.83, 6.85 s in all four detectors |
| `07_pileup_spectral_test.py` | NaI hardness ratios vs rate, b1 left out | sees nothing, but it isn't a sensitive test (see 15) |
| `08_precursor_spectrum.ipynb` | which detectors see the burst (angles, effective area, counts); precursor count spectra, poly3 background per channel, responses; writes `spectra/` | na is the only NaI with signal below 40 keV (n6 and n8 see the burst 134 and 94 deg off axis); net 1623, 883, 6184, 2829 counts |
| `09_precursor_band_fit.ipynb` | Band fit, n6 alone and joint; which choices move the result away from GDT's defaults | E_peak 170.3 +/- 5.3 keV, alpha -0.63 +/- 0.04, beta -2.95 +/- 0.10 |
| `10_precursor_flux_and_models.ipynb` | 10-1000 keV flux, nuFnu plot (Fig. 3), BB / Comptonized / Band+BB, where the models' nuFnu peaks | flux 3.32 +/- 0.04 x 10^-5 erg/cm^2/s; all of Table 2's precursor rows reproduced |
| `11_robustness_checks.ipynb` | background convention, P/PG/W-stat, response, background order, channel cuts, GDT errors, Band+BB | stable to <1 keV except W-stat (+2.7 keV), the K-edge cut (+3 keV) and the channel cuts (E_peak 152-170 keV over cuts giving 480 dof); GDT's errors 3-8x too small |
| `12_time_resolved_spectra.ipynb` | Band fits in Dichiara's 12 spectral windows and the 2 extra MVT windows; errors, flux, systematics, detector checks, Comptonized vs Band, low-energy shape, GBM catalogue check, precursor structure; writes `shared/epeak_230307A.csv` | E_peak within 0.8 sigma of Table 2 in all 12 windows; catalogue Comptonized fits reproduced; precursor = two pulses with the same E_peak (172 +/- 8 and 169 +/- 8 keV) but alpha softening from -0.43 to -0.77 (3.8 sigma), and no quiet gap before the main emission |
| `13_epeak_mvt_merge.ipynb` | joins E_peak to MVT by window label; Dichiara Fig. 2; first E_peak vs MVT plot, with the instrument uncertainty from 15 | uses Dichiara's Table 1 MVT until `shared/mvt_230307A.csv` exists |
| `14_grb211211a.ipynb` | the same pipeline on GRB 211211A, only the settings changed; comparison with Troja et al. (2022) and, slice by slice, Gompertz et al. (2023) | precursor 64-67 keV (Comptonized), then a quiet 0.7-0.9 s gap; E_peak within 2 sigma of Gompertz et al. in 18 of 20 slices and their low-energy break in all 12 where both find one; extended emission agrees with Troja et al. |
| `15_bright_emission_checks.ipynb` | how much GBM's trouble with this burst matters: the GBM team's flags, the summed rate against the telemetry limit, a simulation of pile-up from GBM's pulse shape (a check), GBM against GECAM in the same 149 bins, detectors compared in one energy band, the precursor against the GECAM table; writes `shared/epeak_230307A_instrument_check.csv` | 3-7 s is where the summed rate reaches the 375 kHz limit; pile-up moves E_peak up by at most ~4% and not at all in the precursor; GBM's E_p is ~10% off GECAM's in the bright phase and ~20% in the precursor, following the counts in a bin and not the rate |
| `16_grb211211a_checks.ipynb` | GRB 211211A: systematics, rates and simulated pile-up (a check), and the low-energy break with n2 and na separately; writes `shared/epeak_211211A_systematics.csv` | systematics 0.45-56 keV, always below the statistical error; pile-up under 2%; the break agrees in both detectors (within 1.6 sigma in 11 clean slices) and moves from 66 to ~17 keV, so it's real in this burst |
| `18_sample_precursors.ipynb` | the pipeline on merger candidates and collapsars with a supernova, with one rule for the windows (`sample_tools.py`): fits against the catalogue and Tan et al., rates, signal for an MVT, permutation tests by class, shattering-flare budget | precursor E_p doesn't separate the classes in this sample (p = 0.34, or 0.29 by rank with limits); the ratio to the main emission is only a hint; a factor of 3 would need ~14 bursts per group |
| `19_energetics.ipynb` | k-corrections, E_iso, L_iso, the Amati plane, and the energy budget of a resonant shattering flare for both known bursts | precursors 3.63e50 and 1.84e49 erg/s in 10-1000 keV (4.05e50 and 2.14e49 with the k-correction); 211211A's needs 0.92 times the fiducial tidal rate, 230307A's 18 times |
| `20_spectral_lags.ipynb` | spectral lags from the cross-correlation, tested with real photons and a known shift; Dichiara's and Troja's values; the sample | Dichiara's values and Troja's precursor value recovered; cutting both light curves at a short window returns only 60% of a shift; 230307A's precursor is 7.9 (-2.4/+2.9) ms when that's avoided; precursor lags follow the duration |

`lightcurve_tools.py` holds the light-curve functions; `spectral_fit_tools.py` the spectral-fit ones (including
the `BandLowBreak` model used in 12 and 14); `pileup_tools.py` the simulation of GBM's electronics used in 15 and 16; `sample_tools.py` the set-up of a new burst, the precursor rule and the lag code, used in 18 and 20. Tests of these are in `tests/` (`python -m pytest tests -q`, from `ignacio/`; pytest is in the environment file), and `environment_MVTEnv.yml` has the environment with exact versions. Notebook 17, the spectral half of the join with the MVT, goes in with the MVT table.

## Zhu et al. 2025 and 2026 measured again (`zhu/`, notebooks 22-26)

The bursts of Zhu et al. (2025, MNRAS 541, 3236; 2026, ApJ 1003, 142) measured again from the GBM photons with their episode rules written
as code, so every number has an error bar and a stated rule. All of it is in the package `zhu/`; the notebooks only read its results and
show, check and interpret them.

| Module | What it does |
|---|---|
| `catalogue.py` | their Table 1, the sample of 111 bursts, EPI, redshifts, and the literature table (`shared/zhu_literature_values.csv`: every published number used, with paper, table or figure, units, definition and whether it was tabulated, printed in the text or read from a figure) |
| `background.py` | the 8-1000 keV light curve of the catalogue's NaI detectors, and background intervals kept clear of all the emission |
| `episodes.py` | Bayesian blocks (p0 = 0.05) and the episodes under three rules kept apart: `zhu2025`, `zhu2026` and `project` (notebook 18's rule); T_MVT as their tables have it (the rise of the shortest pulse to its brightest bin) in the zhu modes, half the shortest certain block in the project mode, and three more readings for comparison; the rules again on 40 simulated light curves and at half and twice their time bin |
| `lags.py` | tau32 and tau42 in their bands, at a resolution chosen by a written rule, with a stored status when a lag isn't quoted and a check on light curves with a known shift |
| `spectra.py` | their way (five models, BIC, PGSTAT, their energy ranges) and our rule (notebook 14) on the same spectra, every row with its detectors, ranges, statistic and code version |
| `run.py` | one burst from the data to its tables, and batches that share the sample (each burst taken with a lock the system releases if a process stops) and can be stopped and restarted; `state.json` in each burst's folder says what each stage did, when, with which code version and, if it stopped, why |
| `io.py`, `quality.py` | the files and states, and what each flag means and the status of each number (measured, unstable, lower limit, limit, unusable, unconstrained, missing) |
| `features.py` | each burst's numbers read back into one row per mode, set against theirs, EPI with its uncertainty, and the final table of every burst |
| `classify.py`, `laws.py` | their clustering repeated and tested; whether the precursor's E_peak adds anything; the search for a compact law with nested cross-validation |
| `physics.py` | the precursor-to-main delay read as an inspiral time, and the size the variability allows |

| Notebook | What it shows |
|---|---|
| `22_zhu_table_and_classification.ipynb` | their table alone: EPI and its line, their t-SNE/UMAP/HDBSCAN repeated, then tested over seeds and settings, without a map and with each number left out |
| `23_zhu_episodes.ipynb` | their bursts measured again: the checks on every burst, the episodes and durations under the three rules against their printed edges and Table 1, what the rule changes of 13 Nov did, which quiet gap follows a precursor, what their T_MVT is and what that means for EPI, the detector choice, the error bars, every disagreement with its cause, and the threshold of a quiet episode (why GRB 230307A has no precursor under the rule as written) |
| `24_zhu_spectra_and_lags.ipynb` | E_p of the main emission fitted their way and by our rule, against their table and the GBM catalogue's own fits; the precursor's E_p (measured, or a 95% interval); Tan et al.'s three Type IL candidates; the lags |
| `25_zhu_epi_and_physics.ipynb` | EPI from our own numbers with its uncertainty and what it is made of; the precursor's E_p against EPI; the delay from precursor to main emission as an inspiral time; the size of the emitting region; energetics and the Amati plane |
| `26_zhu_classification_laws.ipynb` | confirmed labels kept apart from candidates; the three populations (kilonova mergers, merger candidates, supernova collapsars); whether E_p,PE or E_p,PE/E_p,ME adds to EPI (nested cross-validation, permutation tests, bootstrap, limits at both ends) and whether it adds anything once E_p,ME is known; which of EPI's terms matter; the compact-law search, with its sensitivity to the least certain labels and to GBM's energy scale; how many confirmed mergers a test would need |

To reproduce the whole analysis, from `ignacio/`:

1. `conda env create -f environment_MVTEnv.yml` and use that environment (MVTEnv).
2. `python -m zhu.run prefetch` downloads the data of the 111 bursts (about 7 GB, to `data/fermiGBM/`).
3. `OMP_NUM_THREADS=1 python -m zhu.run batch --workers 3`, as several copies side by side if there are cores to spare; `python -m zhu.run status`
   shows each burst's state. A stopped batch picks up where it left off; a burst whose code failed is left alone until a batch is run with
   `--retry-failed`. The laptop must not sleep while it runs.
4. Run notebooks 22 to 26 in order (`jupyter execute --inplace 2X_*.ipynb`). 22 needs only their table; 23-26 read `results/zhu/`
   (23's before/after section also reads `results/zhu_12Nov_rules/`, the batch under the rules of 12 Nov, and skips it if absent).
5. `python -m pytest tests/test_zhu.py -q` runs the tests (`-m "not slow"` leaves out the two that refit real spectra, a few minutes each).

The literature table was built once from the arXiv source of the 2025 paper with `catalogue.build_literature_table`; the 2026 numbers come
from the journal's machine-readable Table 1 (`shared/zhu2026_table1.csv`).

## Swift/BAT and the instrument layer (`instruments/`, notebook 27)

The same burst seen by two instruments: Fermi/GBM and Swift/BAT through the same verbs (`burst`, `light_curve`, `episodes`, `spectrum`),
each with its own implementation, since the two record and calibrate photons differently (the table in `fermi_gbm.py` sets out how).

| Module | What it does |
|---|---|
| `fermi_gbm.py` | GBM through those verbs, calling the validated code of `grbtool/` and `zhu/` exactly as the batch does (its test checks it gives the batch's episodes) |
| `swift_bat.py` | the standard BAT chain with HEASoft's own tools (detector mask, hot pixels, mask weighting, spectra with responses, averaged over the slew when the burst moves across the field); chi-squared fits with profile errors; Bayesian blocks with the Gaussian fitness; the light-travel time between Swift and Fermi |
| `bat_compare.py` | the chain against the BAT GRB catalogue, the episodes from BAT's light curve against GBM's, and BAT's spectrum of the precursor GBM defines, beside the index GBM's fit predicts in BAT's band |
| `bat_data.py` | everything BAT needs from outside the repository (calibration files, catalogue tables, the event data of the bursts both instruments saw) |

| Notebook | What it shows |
|---|---|
| `27_bat_check.ipynb` | the BAT chain against the catalogue; episode edges from BAT against GBM's; the precursors in BAT's band against what GBM's fits predict there |
| `00_demonstration.ipynb` | the whole project once, in order, on one burst and then on the sample, every step calling the package or reading a saved table |

## Reproducing the results

Everything from the GBM and BAT photons to the tables and figures is code in this repository; nothing is edited by hand. The expensive
part (the batch, hours) writes `results/zhu/` and `results/bat/`; the notebooks only read those and run in minutes, so their saved outputs
can be read without running anything.

**Environment.** `conda env create -f environment_MVTEnv.yml` (every package at its exact version). Swift/BAT needs HEASoft 6.37.1 with
the BAT tools (the conda `heasoft` package, in a second environment, MVTEnvHEA) and the BAT calibration release of 7 June 2023
(`python -m instruments.bat_data caldb`).

**Data and where it comes from.**

| What | Source | How it gets here |
|---|---|---|
| GBM TTE, responses and catalogue files of each burst | HEASARC Fermi GBM burst archive | `python -m zhu.run prefetch` (about 7 GB, into `data/fermiGBM/`) |
| Zhu et al. (2026) Table 1 | the journal's machine-readable table | `shared/zhu2026_table1.csv` |
| Zhu et al. (2025) printed values | the arXiv source of the paper | `shared/zhu_literature_values.csv` (`catalogue.build_literature_table`) |
| Tan et al. (2025) Tables 1 and 2 | the paper, transcribed | `shared/tan2025_type_il_*.csv` |
| Redshifts | J. Greiner's GRB table, with the project's own values first | `shared/redshifts_greiner.csv` |
| BAT event, attitude and housekeeping files | HEASARC Swift archive, observation 00<trigger>000 | `python -m instruments.bat_data fetch` (into `~/swift_bat/data/`) |
| BAT GRB catalogue (general table, 1-s peak and T100 power-law fits) | NASA GSFC BAT GRB catalogue | `python -m instruments.bat_data catalogue` |

**Configuration.** The rules and thresholds are named constants at the top of each module (`zhu/episodes.py`, `zhu/background.py`,
`zhu/spectra.py`, `zhu/lags.py`, `instruments/bat_compare.py`), each with the reason for its value. The sample of 111 bursts is
`shared/zhu_sample.csv` (`catalogue.build_sample`).

**Random seeds.** Fixed everywhere, so a rerun gives the same numbers: the sample's random controls 20261111; the 40 simulated light curves
per burst 2026; E_p flux errors 1; lag resamples 20 and 32; EPI draws 11 and the input draws 13; cross-validation 5 (law search), 7
(feature sets), 11 (permutation tests), bootstrap 3 and 9; t-SNE, UMAP and HDBSCAN seeds are listed in notebook 22.

**Code versions.** Every row the batch writes carries `git:<commit> zhu:<hash of the package's files>` (the `code_version` column of
each burst's tables, and its `state.json`); the final batch ran every burst on commit 4305eb5. The tables the notebooks write
(`shared/zhu_remeasured.csv`, `shared/zhu_paper_vs_ours.csv`) are made from those results by the notebooks of the commit they are in.

**Commands, from `ignacio/`, with the time each took on this laptop (i7-1255U, 12 threads, 7 GB in WSL):**

1. `python -m zhu.run prefetch` (about 7 GB; the time depends on the connection).
2. `OMP_NUM_THREADS=1 python -m zhu.run batch --workers 3`, six copies side by side. A burst takes a median of 8 minutes and at most
   about 55, almost all of it in the spectral fits: about 19 hours of batch time for the 111 bursts, or about 3.2 hours with six batches.
   A stopped batch picks up where it left off, and the laptop must not sleep while it runs. `python -m zhu.run status` shows each
   burst's state. Each model fit has 5 minutes and the profile error of the chosen E_peak 10; a window that needs more is recorded as
   such, keeps the best fit found, and can be fitted again with more time (`zhu.run.measure_named(..., redo=('spectra',))` with
   `SPECTRUM_MINUTES` and `zhu.spectra.MODEL_MINUTES` raised).
3. Notebooks 22-27 and 00 in order (`jupyter execute --inplace <notebook>`): 23-27 and 00 take about 6 minutes together (26 about 4
   of them), 22 a few minutes.
4. BAT: `python -m instruments.bat_data caldb`, `catalogue`, `match` and `fetch` (once), then `python -m instruments.bat_compare validate
   <BAT names>` (about 2 hours for 15 bursts, mostly the responses made every 5 s during slews), `episodes <names>` and
   `precursors <names>` (about 20 minutes together for 19 bursts), and notebook 27.
5. `python -m pytest tests -q`: 131 tests in about 11 minutes; `-m "not slow"` leaves out the six that refit real spectra or run the
   BAT chain.

**Rodrigo's MVT (Chandra).** The Haar-wavelet MVT is Rodrigo's code (MVTTools, branch haar-fix, commit 74b8ff2), run by him on Bath's
Chandra cluster, with his own scripts, on the windows in `shared/`. Its outputs are his and are not in this repository until he has
cleared them.

## Run times

On my laptop (i7-1255U, 7 GB RAM in WSL), each notebook uses one core and under 1 GB: 08-10 take 2-3 min each,
11 about 20 min the first time (the 1000 simulated spectra are then saved in `spectra/sim_band_null.npz` and
reloaded), 12 about 35 min, 13 under a minute, 14 about 1.5-2 h (three models with profile errors in 23
windows), 15 about 30 min, 16 about 30 min, 18 about 25 min, 19 a few minutes and 20 about 3 min. 08 has to run before 09-12, 12 before 13, 14 and 15 (they read its E_peak table), 15
before 13, 14 before 16, 12 and 14 before 18 and 19, and 18 before 20; otherwise they can run side by side.

## Where things live

- Code: `~/mphys_project/ignacio/` (git, branch `precursor-spectrum`, github.com/idbl2004/mphys-project)
- Data: `~/mphys_project/data/fermiGBM/<burst>/current/` (TTE, responses, catalogue files, from the HEASARC archive); only GRB 230307A's is in the repo
- Spectra written by the notebooks: `spectra/` (precursor), `spectra/time_resolved/`, `spectra/time_resolved_211211A/`
- Shared with the MVT analysis: `shared/` (windows files, E_peak tables, format in `shared/README.md`)
- Figures: `figures/`, regenerated by rerunning the scripts and notebooks
- Never commit `config.yaml` or the edited files in `fermiDataTools/`

## Open points

1. The 0.32-3 s flux is ~9% above Dichiara's with the same spectral shape. Their Table 2 value looks like a misprint,
   1.196 for 1.296: the C-stat they quote only fits 1.296, and GECAM agrees (12, section 9).
2. alpha is 0.025-0.032 harder than Dichiara's in 13-29 s (2.2-2.6 sigma on the combined errors), not explained
   by the analysis choices tested.
3. The Band function doesn't describe the NaI data down to 8 keV in the bright windows: starting the NaI at
   40 keV moves E_peak by up to ~50 keV, and a two-break model improves the C-stat by 23-325 (12, section 12).
   na is the only one of the twelve NaI detectors with real effective area below 40 keV for this burst (08), so
   GBM alone can't tell a low-energy break from na's calibration. na is also 12-25% below GECAM's Comptonized fits in
   15-40 keV (15, section 9), which fits either explanation.
4. Detector choice: Dichiara's n6 and n8 see the burst 134 and 94 deg off axis, far outside the 60 deg the GBM
   catalogue allows, and the GBM team recommend only na and b1 for this burst (GCN 33551). With na and b1 alone E_peak
   comes out 0-14 keV higher, up to 1.8 sigma (12, section 11).
5. Band+BB for the precursor: Delta C = 27 for 2 parameters, ~17 without na's 30-40 keV channels (11).
6. GRB 211211A: its E_peak depends strongly on the model in 0-26 s (Band against the
   two-break model, a factor of ~2), so 14 takes the two-break E_peak where that model is clean and clearly
   better, while 12 keeps Band for 230307A to compare with Dichiara. Its precursor flux is 1.5-2.7 times
   Troja et al.'s, with the same E_peak.
7. The errors in the E_peak table are the precision, not the accuracy. GBM and GECAM differ by ~10% in the bright phase
   and ~20% in the precursor, and I can't say which is closer (15, sections 6, 7 and 10).
8. m01, m02, m03 and t1_07_18 overlap the GBM team's pile-up intervals (2.5-7.5 s in na, 2.5-11 s in b1). The simulated
   effect on E_peak is up to ~4% and isn't corrected in the table (15, section 5). Unexplained: why the b1 interval starts
   at 2.5 s when b1 reaches 87 kcps at ~1.8 s, and why the long data gaps start just after CSPEC bin edges (15, section 2).
9. The sample is small. With 5 mergers and candidates against 4 collapsars, 18 only rules out a large difference in precursor E_p, and candidates aren't confirmed mergers. 200914A's main emission is 2 sigma above Tan et al.'s, and the two longest bursts are 12-18% above the catalogue, unexplained.
