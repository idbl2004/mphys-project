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
the `BandLowBreak` model used in 12 and 14); `pileup_tools.py` the simulation of GBM's electronics used in 15 and 16; `sample_tools.py` the set-up of a new burst, the precursor rule and the lag code, used in 18 and 20. Tests of these are in `tests/` (`python -m pytest tests -q`, from `ignacio/`; pytest isn't in the environment, so `pip install pytest` first), and `environment_MVTEnv.yml` has the environment with exact versions. Notebook 17, the spectral half of the join with the MVT, goes in with the MVT table.

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
