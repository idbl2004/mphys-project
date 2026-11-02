# Shared files (E_peak and MVT)

Files that the E_peak and MVT analyses both read, so that the two can be joined window by window.

## Conventions

- **T0** = GBM trigger time: 699896651.671248 s MET for GRB 230307A (bn230307656), 660921004.65092 s MET for
  GRB 211211A (bn211211549). All times are in seconds from T0.
- **Windows** are joined by their `label`, never by their times.
- For GRB 230307A, **3-7 s** is the bad time interval (pile-up) and no window overlaps it.

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

Notebook 12 also refits every window with na and b1 alone, the only detectors within 60 deg of this burst
(section 11). That shift is reported there, not in this file.

## `epeak_mvt_230307A.csv`

Written by `13_epeak_mvt_merge.ipynb`: the rows of `epeak_230307A.csv` that have an MVT value, with the MVT
columns joined on. `mvt_source` says where the MVT came from (Dichiara et al. 2023, Table 1, until
`mvt_230307A.csv` exists), and `Epeak_err` is the mean statistical error and the systematic in quadrature.

## GRB 211211A: `windows_211211A.csv` and `epeak_211211A.csv`

Written by `14_grb211211a.ipynb`. Windows: `pre` (precursor, edges from Bayesian blocks), `g00_02` to `g60_70`
(the time slices of Gompertz et al. 2023, so E_peak can be compared slice by slice) and `f2_01_10`, `f2_27_66`
(the GRB 211211A bars in Dichiara's Fig. 2, edges read off the figure). The `source` column says which.

The E_peak table has the same columns as for GRB 230307A, with three differences:
- `model` can change from window to window: `Band`, or where Band's beta runs into its -2 limit, `BandLowBreak`
  (when its two-break fit is clean) or `Comptonized`. `notes` says why.
- alpha and beta errors are symmetric, from the covariance matrix.
- `dichiara_table`, `Epeak_sys`, `Epeak_NaI40_keV`, `alpha_sys`, `beta_upper95` and `Epeak_rest_keV` are empty:
  those checks haven't been run for this burst.

## MVT file (format to agree)

My proposal, so the merge in `13_epeak_mvt_merge.ipynb` can read it directly: `mvt_230307A.csv`, keyed by the
same `label`, with `mvt_s`, `mvt_err_lo_s`, `mvt_err_hi_s` and `band_keV`. An upper limit would have its
value in `mvt_s`, empty errors, and `upper_limit` set to true.
