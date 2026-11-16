"""Fermi/GBM through the same verbs as instruments/swift_bat.py (burst, light_curve, episodes, spectrum), so the two instruments can be
compared burst by burst. Nothing here is new analysis: each function calls the GBM code already validated in notebooks 12-26 (grbtool and
the zhu package) the way the batch (zhu/run.py) does, so a GBM number got here is the number in results/zhu.

How the two instruments differ, and what each module does about it:

                    Fermi/GBM (this file)                                  Swift/BAT (swift_bat.py)
  data              time-tagged events (TTE): each photon's time (2 us)    the burst's event file: each photon's time, energy and
                    and one of 128 energy channels, 12 NaI and 2 BGO       detector, on 32,768 CdZnTe detectors under a coded mask
  time zero         the GBM trigger time                                   the BAT trigger time; gbm_offset() and light_travel()
                                                                           put a BAT time on GBM's axis for the same light
  background        a polynomial in time for each detector and channel,    mask weighting: each event is weighted by whether the burst
                    fitted before and after the burst, clear of all the    could reach its detector through the mask, which removes
                    emission (zhu.background.clear_background)             the background without a fit
  statistics        Poisson counts on a modelled background: PGSTAT or     weighted rates with Gaussian errors: chi-squared, and the
                    C-stat fits, and Poisson Bayesian blocks               Gaussian fitness for Bayesian blocks
  response          the catalogue's response for each detector             batdrmgen for the window, averaged over the slew when the
                    (RSP2, interpolated to the window's time)              burst moves across the field
  band              NaI 8-900 keV and BGO 0.3-40 MeV fitted together       15-150 keV, so most main-emission peaks are limits
  systematics       E_peak about 10% off GECAM's in the bright phase       the standard systematic error vector (batphasyserr)
                    and 20% in the precursor (notebook 15), and
                    pile-up near the telemetry limit
  time resolution   Zhu et al.'s bin for each burst (4-64 ms here)         bins of at least 16 ms, since mask-weighted bins at 4 ms are
                                                                           often empty

Run from ignacio/ (or put it on sys.path), with GBM data in data/fermiGBM (python -m zhu.run prefetch).
"""
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from grbtool.windows import Window, cut_out                                        # noqa: E402
from zhu import RESULTS                                                            # noqa: E402
from zhu.background import BAND, clear_background                                  # noqa: E402
from zhu.background import light_curve as _light_curve                             # noqa: E402
from zhu.episodes import blocks_of, find_episodes                                  # noqa: E402
from zhu.io import read_json                                                       # noqa: E402
from zhu.run import read_sample, start_burst                                       # noqa: E402


def burst(name):
    """A burst of the sample set up as the batch does it: the catalogue's detectors (by angle if the catalogue has no fit), background
    intervals clear of all the emission. Returns (burst, row of the sample, flags)."""
    row = read_sample().loc[name]
    gbm, notes = start_burst(name, row.trigger)
    flags = list(gbm.flags) + list(notes) + list(clear_background(gbm))
    return gbm, row, flags


def light_curve(gbm, tres, span=None, detectors=None):
    """8-1000 keV counts of the NaIs (summed), their live time and the fitted background, in bins of tres (zhu.background.light_curve)."""
    return _light_curve(gbm, tres, span=span, detectors=detectors)


def episodes(gbm, tres, mode="zhu2026", name=""):
    """The episodes of one rule (zhu2025, zhu2026 or project) on the light curve at tres. Returns (light curve, EpisodeResult)."""
    lc = light_curve(gbm, tres)
    return lc, find_episodes(lc, blocks_of(lc), mode=mode, burst=name)


def spectrum(gbm, start, stop, label="ME", bad_intervals=None, ways=("ours", "zhu"), minutes=None):
    """Fits of one window, the stretches inside bad_intervals cut out: our rule (power law, Comptonized, Band, blackbody and the two-break
    model, chosen by delta C) and their way (PGSTAT, lowest BIC). One row per window, as the batch writes it. By default the batch's own
    choices: the intervals where the summed rate reaches GBM's telemetry limit (worked out as the batch does, which reads the photons
    again) and its time per slow fitting step (zhu.run.SPECTRUM_MINUTES)."""
    from grbtool.checks import Rates
    from zhu.background import emission_span
    from zhu.run import SPECTRUM_MINUTES
    from zhu.spectra import fit_episodes
    if bad_intervals is None:
        bad_intervals = Rates(gbm, emission_span(gbm)).telemetry_intervals()
    window = Window(label, cut_out((start, stop), list(bad_intervals)), "precursor" if label == "PE" else "main")
    return fit_episodes(gbm, [window], workers=1, minutes=SPECTRUM_MINUTES if minutes is None else minutes, ways=ways)


def saved(name, mode="zhu2026", root=RESULTS):
    """What the batch saved for a burst: the episodes of one mode (a dict, None if the burst wasn't measured) and that mode's spectra
    (a DataFrame indexed by window label, None if there are none). Reads results/zhu/<name>/ only."""
    folder = root / name
    record = read_json(folder / "episodes.json", {})
    path = folder / f"spectra_{mode}.csv"
    spectra = pd.read_csv(path).set_index("label") if path.exists() else None
    return record.get("modes", {}).get(mode), spectra
