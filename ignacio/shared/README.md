# Shared files (E_peak and MVT)

Files that the E_peak and MVT analyses both read, so that the two can be joined window by window.

## Conventions

- **T0** = GBM trigger time: 699896651.671248 s MET for GRB 230307A (bn230307656), 660921004.65092 s MET for
  GRB 211211A (bn211211549). All times are in seconds from T0.
- **Windows** are joined by their `label`, never by their times.
- For GRB 230307A, **3-7 s** has TTE packet losses and no window overlaps it (Dichiara's choice). The GBM team also
  report pile-up in na at 2.5-7.5 s and in b1 at 2.5-11 s (GCN 33551), and advise using only na and b1 for this burst.
  m01, m02, m03 and `t1_07_18` overlap those intervals: see `notes` in the E_peak table and
  `epeak_230307A_instrument_check.csv`.

## `windows_230307A.csv`

One row per time window. Lines starting with `#` are comments.

| Column | Meaning |
|---|---|
| `label` | `pre` (precursor), `m01`-`m11` (main emission), `t1_07_18`, `t1_18_40` |
| `t_start`, `t_stop` | window edges, s from T0 |
| `dichiara_table` | which table of Dichiara et al. (2023) the window comes from: `2` (spectra), `1` (MVT), `1+2` (both) |

The 12 Table 2 windows are where E_peak can be checked against the paper. The four Table 1 windows (`pre`,
`m01`, `t1_07_18`, `t1_18_40`) are where MVT can be checked against the paper. Their Fig. 2 plots both.

## `epeak_230307A.csv`

Written by `12_time_resolved_spectra.ipynb`. One row per window, same labels as the windows file. Joint Band
fit of n6, n8, na and b1 (NaI 8-900 keV, BGO 200 keV-40 MeV), Poisson likelihood with the background fixed.

| Column | Meaning |
|---|---|
| `label`, `t_start`, `t_stop`, `dichiara_table` | as in the windows file |
| `model` | spectral model the row refers to (`Band`) |
| `Epeak_keV` | peak of nuFnu, observer frame |
| `Epeak_err_lo`, `Epeak_err_hi` | 68% statistical errors (profile likelihood, Delta C = 1) |
| `Epeak_sys` | systematic error: BGO lower edge, response and background order, added in quadrature |
| `Epeak_NaI40_keV` | E_peak with the NaI fitted from 40 keV instead of 8 keV: a check of the Band shape at low energies, not part of `Epeak_sys` |
| `alpha`, `alpha_err_lo`, `alpha_err_hi`, `alpha_sys` | low-energy index, same conventions |
| `beta`, `beta_err_lo`, `beta_err_hi` | high-energy index, 68% errors |
| `beta_upper95` | 95% upper limit on beta where its steep side isn't constrained (then `beta_err_lo` is empty) |
| `flux_10_1000keV_erg_cm2_s`, `flux_err` | mean energy flux in the window, 10-1000 keV |
| `cstat`, `dof` | fit statistic and degrees of freedom |
| `Epeak_cpl_keV` | E_peak from a Comptonized (cutoff power-law) fit to the same data |
| `dC_cpl_minus_band` | C-stat(Comptonized) - C-stat(Band); above ~9 the Band tail is needed |
| `Epeak_rest_keV` | E_peak x (1 + z), z = 0.065 |
| `notes` | anything that needs care in that window |
| `notebook`, `commit`, `run_date` | where the row came from |

Notebook 12 also refits every window with na and b1 alone, the only detectors within 60 deg of this burst and the
ones the GBM team recommend (section 11). That shift is reported there, not in this file.

## `moradi2024_table1.csv`

Table 1 of Moradi et al. (2024, arXiv:2410.17189) as published: time-resolved fits of GRB 230307A from GECAM, joined with
GBM only before T0 + 0.65 s and after T0 + 17.5 s. Used in notebooks 12 and 15 as an independent check, never as a result.

| Column | Meaning |
|---|---|
| `t_start`, `t_stop` | time bin, s from T0 |
| `model` | `band`, `cpl` (Comptonized) or `pl` |
| `alpha`, `beta`, `Ep_keV` | fit parameters, each with `_err_hi` and `_err_lo` |
| `flux_erg_cm2_s` | energy flux at 6-8000 keV |
| `photon_flux`, `photon_flux_err_hi`, `photon_flux_err_lo` | photon flux, as in their table |
| `cstat`, `dof` | their fit statistic and degrees of freedom |

## `epeak_230307A_instrument_check.csv`

Written by `15_bright_emission_checks.ipynb`. One row per window, kept apart from `epeak_230307A.csv` so that file stays
the reproduction of Dichiara's analysis.

| Column | Meaning |
|---|---|
| `label`, `t_start`, `t_stop`, `Epeak_keV` | as in the E_peak table |
| `share_na_pileup`, `share_b1_pileup` | share of the window inside the GBM team's pile-up interval for that detector |
| `b1_max_kcps` | highest recorded rate in b1, 16 ms bins |
| `pileup_Epeak_ratio_sim`, `pileup_dalpha_sim` | simulated effect of pile-up on E_peak (ratio) and alpha (difference); a check, not a correction |
| `gecam_Ep_keV`, `gecam_share_of_bins_without_gbm` | energy-weighted mean E_p of the bins of `moradi2024_table1.csv` inside the window, and the share of them with no GBM data |
| `Epeak_over_gecam` | `Epeak_keV` over that mean: a rough check, since it compares one Band fit with an average of 0.1 s Comptonized fits |
| `Epeak_instrument_frac` | fractional uncertainty on E_peak from the instruments disagreeing: 0.10, and 0.20 for the precursor (my rounding of notebook 15, sections 6 and 10) |

## `epeak_mvt_230307A.csv`

Written by `13_epeak_mvt_merge.ipynb`: the rows of `epeak_230307A.csv` that have an MVT value, with the MVT
columns joined on. `mvt_source` says where the MVT came from (Dichiara et al. 2023, Table 1, until
`mvt_230307A.csv` exists), `Epeak_err` is the mean statistical error and the systematic in quadrature, and
`Epeak_instrument_keV` is the instrument uncertainty from notebook 15, kept separate.

## GRB 211211A: `windows_211211A.csv` and `epeak_211211A.csv`

Written by `14_grb211211a.ipynb`. Windows: `pre` (precursor, edges from Bayesian blocks), `g00_02` to `g60_70`
(the time slices of Gompertz et al. 2023, so E_peak can be compared slice by slice) and `f2_01_10`, `f2_27_66`
(the GRB 211211A bars in Dichiara's Fig. 2, edges read off the figure). The `source` column says which.

The E_peak table has the same columns as for GRB 230307A, with three differences:
- `model` can change from window to window: `BandLowBreak` where its two-break fit is clean and better than Band by
  Delta C > 11.8; otherwise `Band` if its tail is needed (Delta C > 9 over Comptonized) and its beta is clear of the -2 limit; otherwise `Comptonized`. `notes` says why.
- alpha and beta errors are symmetric, from the covariance matrix.
- `dichiara_table`, `Epeak_sys`, `Epeak_NaI40_keV`, `alpha_sys`, `beta_upper95` and `Epeak_rest_keV` are empty. The
  systematics are in `epeak_211211A_systematics.csv` instead.

## `epeak_211211A_systematics.csv`

Written by `16_grb211211a_checks.ipynb`. One row per window, kept apart from `epeak_211211A.csv`.

| Column | Meaning |
|---|---|
| `t_start`, `t_stop`, `model`, `Epeak_keV` | as in the E_peak table |
| `Epeak_sys`, `alpha_sys` | the three shifts below in quadrature, and the same for alpha |
| `dEpeak_BGO_300keV`, `dEpeak_interpolated_response`, `dEpeak_background_order2` | the shift from each choice |
| `b0_max_kcps`, `n2_max_kcps`, `na_max_kcps` | highest recorded rate, 16 ms bins |
| `pileup_Epeak_band_ratio_sim` | simulated effect of pile-up on Band's E_peak; a check, and empty where b0 stays under 20 kcps |

## `energetics_known_bursts.csv`

Written by `19_energetics.ipynb`: k-correction, L_iso and E_iso for every window of both known bursts.

## The sample: `windows_sample.csv`, `sample_fits.csv`, `sample_precursors.csv`, `sample_snr.csv`, `sample_budget.csv`, `sample_lags.csv`

Written by `18_sample_precursors.ipynb`, except `sample_lags.csv` from `20_spectral_lags.ipynb`.

- `windows_sample.csv`: `burst`, `label` (`<burst>_pre`, `<burst>_main`), `t_start`, `t_stop`, in s from each burst's own GBM trigger.
- `sample_fits.csv`: one row per episode. Comptonized `Ep` with 68% errors and 95% limits, `alpha`, power-law index, Band and blackbody fits, `net_counts`, `sigma`, `flux_cpl` (10-1000 keV), and `usable` (at least 5 sigma) and `measured` (also Delta C > 9 over a power law).
- `sample_precursors.csv`: one row per burst, including the two known ones. `Ep_pre`, `Ep_main`, `Ep_main_best` (Band where it wins by Delta C > 9), `ratio`, `ratio_best`, `T_pre`, `z`, `limit` (95% upper limit on `Ep_pre`).
- `sample_snr.csv`: SNR_MVT of each precursor at 4-64 ms (`snr_*`), and the same in a quiet stretch before the burst (`quiet_*`).
- `sample_budget.csv`: the tidal energy rate a precursor needs, against the fiducial 1e50 erg/s, and for bursts with no redshift the z at which it would need 1 and 18 times that.
- `sample_lags.csv`: the lag of 25-50 keV behind 100-300 keV (`lag_ms`, with the low band running past the window; `lag_cut_ms`, with both cut at it) and the significance in each band.

## MVT file (format to agree)

My proposal, so the merge in `13_epeak_mvt_merge.ipynb` can read it directly: `mvt_230307A.csv`, keyed by the
same `label`, with `mvt_s`, `mvt_err_lo_s`, `mvt_err_hi_s` and `band_keV`. An upper limit would have its
value in `mvt_s`, empty errors, and `upper_limit` set to true.
