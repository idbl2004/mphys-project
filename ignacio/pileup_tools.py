"""
Monte Carlo of how a GBM detector records photons at high rates.

The pulse shape and the timing are the ones in Chaplin et al. (2013, NIM A 717, 21; arXiv:1211.6592). Every detection
adds E * f(t) to the voltage, and f(t) has a positive peak at 0.36 us followed by a negative lobe (-70% of the peak at
~1.6 us). The voltage is sampled every 0.104 us. An event is recorded at a maximum above threshold once the voltage
has fallen for 4 samples, with the height of the maximum, and the detector is then dead for 21 more samples (2.6 us in
total, 10 us after an overflow event). So two detections close together can add up (peak pile-up), and a detection on
the tail of an earlier one comes out low or not at all (tail pile-up).

Voltage is taken to be proportional to energy. Used in notebook 15 as a check of how much pile-up should matter, not to
correct the data.
"""
import numpy as np

SAMPLE = 0.104           # us between voltage samples
BUFFER, DEAD = 4, 21     # samples: 25 x 0.104 us = 2.6 us
DEAD_OVERFLOW = 92       # samples: 96 x 0.104 us = 10 us
SUBSAMPLE = 2            # arrival times are kept on a grid of SAMPLE / SUBSAMPLE
MAX_GAP = 13.0           # us. Pulses further apart than this can't affect each other, so longer gaps are cut short
BATCH = 20000            # detections simulated in one go, which keeps the memory small at any rate

_t = np.arange(0.0, 6.0, SAMPLE / SUBSAMPLE)
PULSE = 0.41023 * (26.0 * _t**1.27 - 31.0 * _t**3.5) * np.exp(-2.6 * _t)


def _record_batch(energies, rate, threshold, overflow, rng):
    """Recorded heights for one batch of detections at `rate` (per us)."""
    gaps = np.minimum(rng.exponential(1.0 / rate, len(energies)), MAX_GAP)
    index = (np.cumsum(gaps) / (SAMPLE / SUBSAMPLE)).astype(int)
    train = np.bincount(index, weights=energies)
    voltage = np.convolve(train, PULSE)[::SUBSAMPLE]

    middle = voltage[1:-BUFFER - 1]
    is_maximum = (middle > voltage[:-BUFFER - 2]) & (middle >= voltage[2:-BUFFER]) & (middle > threshold)
    peaks = np.flatnonzero(is_maximum) + 1
    falling = np.ones(len(peaks), bool)
    for i in range(BUFFER - 1):
        falling &= voltage[peaks + i + 2] < voltage[peaks + i + 1]
    peaks = peaks[falling]
    heights = voltage[peaks]

    keep = np.zeros(len(peaks), bool)
    dead_until = -1
    for i, (peak, height) in enumerate(zip(peaks.tolist(), heights.tolist())):
        if peak >= dead_until:
            keep[i] = True
            dead_until = peak + BUFFER + (DEAD_OVERFLOW if height >= overflow else DEAD)
    return heights[keep]


def record(draw_energies, rate, threshold, overflow, num_detections, rng):
    """Simulate num_detections detections at `rate` (per us).

    draw_energies(n) returns n energies in keV. Events above `overflow` keV get the longer dead time. Returns the
    recorded heights and the time simulated in us.
    """
    batches = max(1, int(np.ceil(num_detections / BATCH)))
    heights = [_record_batch(draw_energies(BATCH), rate, threshold, overflow, rng) for _ in range(batches)]
    return np.concatenate(heights), batches * BATCH / rate


def channel_sampler(rates, lo, hi, rng):
    """draw_energies for a spectrum given as rates per channel: log-uniform inside each channel."""
    prob = rates / rates.sum()

    def draw(n):
        channel = rng.choice(len(prob), n, p=prob)
        return lo[channel] * (hi[channel] / lo[channel])**rng.random(n)
    return draw
