"""
The bins of the counts-weighted response (spectral_fit_tools.response_bin_edges). Run from ignacio/ with:
python -m pytest tests/test_response_edges.py -q

A window that crosses a DRM boundary and is a whole number of 0.256 s bins used to get its end twice, and GDT's histogram then
stopped. That happened to the main emissions of GRB 110725A and GRB 130318A in the Coppin sample (73 and 158 bins). The last test
needs the GRB 230307A data.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from settings import BURST_ID, DATA_DIR, SPECTRAL_DETECTORS
from spectral_fit_tools import response_bin_edges, window_response

WIDTH = 0.256


def old_edges(window):
    """The edges as window_response made them before 19 Nov 2026."""
    return np.append(np.arange(window[0], window[1], WIDTH), window[1])


def windows():
    """Windows on the 8 ms grid of the light curves, some a whole number of bins long and some not."""
    for start in np.round(np.arange(-5.0, 5.0, 0.008), 3):
        for length in (0.1, 1.0, 7.0, 13.568, 18.688, 40.448):
            yield start, round(start + length, 3)


def test_edges_increase_and_span_the_window():
    for window in windows():
        edges = response_bin_edges(window, WIDTH)
        assert np.all(np.diff(edges) > 0)
        assert edges[0] == window[0] and edges[-1] == window[1]
        assert np.allclose(edges[:-1], window[0] + WIDTH * np.arange(len(edges) - 1))


def test_edges_unchanged_where_they_worked():
    repeated = 0
    for window in windows():
        old = old_edges(window)
        if np.all(np.diff(old) > 1e-6):
            assert np.array_equal(response_bin_edges(window, WIDTH), old)
        else:
            repeated += 1
    assert repeated > 0          # the grid does hold windows whose old edges repeated the end


def test_the_windows_that_failed():
    for window, bins in (((15.808, 34.496), 73), ((79.136, 119.584), 158), ((49.416, 62.984), 53)):
        assert not np.all(np.diff(old_edges(window)) > 0)
        assert len(response_bin_edges(window, WIDTH)) == bins + 1


def test_window_response_across_a_drm_boundary():
    # 20.008 to 33.576 s is 53 bins and crosses GRB 230307A's DRM boundary at 30.4 s; the old edges repeated its end
    window = (20.008, 33.576)
    assert not np.all(np.diff(old_edges(window)) > 0)
    det = next(d for d in SPECTRAL_DETECTORS if d.startswith("n"))
    rsp, how = window_response(DATA_DIR, BURST_ID, det, window)
    assert how == "counts-weighted"
    assert np.all(np.isfinite(rsp.drm.matrix))
