"""What each flag means, and a quality status for every number a burst gets.

A number is only quoted with its status:
  measured     found by the rule, with an error from the simulated light curves or the profile likelihood
  unstable     found in the data, but in fewer than STABLE_SHARE of the simulated light curves, so the rule's choice is fragile
  lower_limit  the episode runs into the background interval, so its end (and the durations that use it) can only be larger
  limit        a spectrum not counted as measured: the 95% interval of the Comptonized E_peak is given instead of a value
  unusable     not enough signal (a lag with no real peak, a spectrum that couldn't be fitted)
  unconstrained  a best-fit E_peak whose profile-likelihood search found no error on either side
  missing      the episode or quantity isn't there for this burst
"""
import numpy as np

STABLE_SHARE = 0.8


def _finite(value):
    try:
        return value is not None and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _episode_missing(row, label):
    """True if the row says the burst has no such episode under its mode: its start time is there, and empty. (A row without the
    column says nothing about it.)"""
    key = f"{label.lower()}_start"
    return key in row and not _finite(row[key])

FLAGS = {
    # episodes (episodes.find_episodes)
    "no_significant_block": "no block of the light curve is 3 sigma above the background",
    "excess_at_start_ignored": "the first block of the light curve was above 3 sigma; it is taken as a background misfit, not emission",
    "pe_peak_above_me_peak": "zhu2026: the precursor peaks as high as the main emission or higher, which the 2026 paper allows when the gap is at the background; the main emission's peak is the brightest block after the gap",
    "emission_reaches_background": "the last emission block ends where the background interval starts: T100,WE and T100,EE (or T100,ME if there is no EE) are lower limits",
    "no_background_before_burst": "the data start too soon before the emission for an interval before it; the background is fitted after it only",
    "no_background_after_burst": "the data end too soon after the emission for an interval after it",
    "no_background_interval_left": "neither side had room for an interval clear of the emission; the last intervals with room were kept",
    "window_inside_bad_interval": "the whole window lies in an interval at the telemetry limit, so it has no spectrum",
    "no_spectrum_for_this_window": "this mode's window differs from every fitted window, so no spectrum is quoted for it",
    "ep_error_one_sided": "the E_peak error is known on one side only; that side's width is used for both in the draws",
    "no_quiet_gap_before_main": "no block before the main peak passes the mode's quiet test, so no precursor under this mode",
    "quiet_gaps_before_main": "more than one quiet dip before the main peak; the mode's choice (the longest) was used",
    "nearest_miss_qe1_snr": "the quietest long dip before the main peak, in sigma per bin, when none was below 3",
    "pe_not_weaker_than_me": "zhu2025: the precursor's peak isn't below the main emission's",
    "pe_counts_not_below_me": "zhu2026: the gap is at the background but the precursor has as many counts as the main emission",
    "me_peak_not_clearly_above_pe": "zhu2026: the gap isn't at the background and the main peak isn't 3 sigma above the precursor's",
    "mvt_pe_is_half_the_precursor": "no block in the precursor has edges certain at 5 sigma, so T_MVT,PE is half the precursor",
    "mvt_me_is_half_the_main_emission": "no block in the main emission has edges certain at 5 sigma, so T_MVT,ME is half the main emission",
    # background (background.clear_background, grbtool.windows.set_background)
    "background_before_burst_only": "there was no room for a background interval after the burst; the polynomial is a straight line",
    "background_after_burst_added": "grbtool found no room 30 s after T90, but there was room after the emission; a straight line is fitted",
    "emission_close_to_start_of_data": "the data start less than 30 s before the emission, so the interval before the burst is short",
    "emission_close_to_end_of_data": "the data end soon after the emission, so the interval after the burst is short",
    "background_poor_fit": "the background polynomial fits its intervals with reduced chi2 above grbtool's limit",
    "short_background_after_burst": "the interval after the burst is under 60 s",
    "single_responses": "the archive has one response matrix per detector for this burst, used for the whole burst",
    "detectors_by_angle": "the catalogue has no fit for this burst, so the NaIs within 60 degrees (at most 3) and their BGO are used",
    # spectra (spectra.py and grbtool.fitting)
    "out_of_time": "a fit or error search used up its time; the best start found so far is kept",
    "error_search_failed": "the profile-likelihood search for the E_peak error stopped without an answer on that side; that side is left blank",
    "error_out_of_time": "the profile-likelihood search on that side of E_peak used up its time; that side is left blank",
    "error_open": "the fit statistic doesn't rise by 1 within 10 sigma (or the parameter's bound) on that side: no error there",
    "two_break_rejected": "a two-break model fits much better than Band but its break isn't clean, so part of the spectrum is undescribed",
    "no_model_fully_constrained": "their way: none of the five models has every parameter constrained; the lowest BIC is used anyway",
    "chosen_model_has_no_epeak": "their way: PL+BB has the lowest BIC; the best model with an E_peak is quoted beside it",
}


def meaning(flag):
    """The meaning of a flag; flags with a value after a colon share the meaning of their name."""
    return FLAGS.get(flag.split(":")[0], "(no description)")


def timing_status(row, quantity):
    """The status of one timing number of a burst (a row from features.collect)."""
    value = row.get(quantity, np.nan)
    if value is None or not np.isfinite(value):
        return "missing"
    reaches = "emission_reaches_background" in str(row.get("flags", ""))
    if reaches and (quantity in ("T100_WE", "T100_EE") or (quantity == "T100_ME" and not np.isfinite(row.get("T100_EE", np.nan) or np.nan))):
        return "lower_limit"
    found = row.get("ee_found_fraction" if quantity in ("T100_EE", "T100_QE2") else "pe_found_fraction", 1.0)
    if quantity in ("T100_PE", "T100_QE1", "TMVT_PE", "T100_EE", "T100_QE2") and np.isfinite(found) and found < STABLE_SHARE:
        return "unstable"
    return "measured"


def epeak_status(row, label, way="ours"):
    """The status of a spectrum's E_peak: measured, limit (our rule: not measured, with its 95% interval), unusable, unconstrained,
    missing (no such episode) or not_fitted (their way isn't fitted on the project mode's windows)."""
    if _episode_missing(row, label):
        return "missing"
    if way == "zhu" and row.get("mode") == "project":
        return "not_fitted"
    value = row.get(f"Ep_{label}_{way}", np.nan)
    if way == "ours" and str(row.get(f"measured_{label}", "")) != "True":
        lo = row.get(f"Ep_{label}_cpl_limit_lo", np.nan)
        return "limit" if lo is not None and np.isfinite(lo) else "unusable"
    if value is None or not np.isfinite(value):
        return "unusable"
    lo, hi = row.get(f"Ep_{label}_{way}_lo", np.nan), row.get(f"Ep_{label}_{way}_hi", np.nan)
    if not (np.isfinite(lo if lo is not None else np.nan) or np.isfinite(hi if hi is not None else np.nan)):
        return "unconstrained"                 # a best fit with no error on either side: the value isn't quoted as a measurement
    return "measured"


def lag_status(row, name, version="margin"):
    """A lag's status (name like tau32_ME): measured if a value was quoted, otherwise what the lag stage said about it, or missing if
    the episode isn't there. version "cut" is the lag with both bands cut at the window, which has its own test and status."""
    cut = version == "cut"
    if _finite(row.get(f"{name}_cut_ms" if cut else f"{name}_ms", np.nan)):
        return "measured"
    if _episode_missing(row, name.split("_")[-1]):
        return "missing"
    said = row.get(f"{name}_cut_status" if cut else f"{name}_status")
    return said if isinstance(said, str) and said and said != "measured" else "unusable"
