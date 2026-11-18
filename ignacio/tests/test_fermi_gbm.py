"""instruments/fermi_gbm.py gives the numbers the batch saved: it calls the same code, so the episodes of a burst found through it are
those in results/zhu. Skipped where the burst's data or the batch's results are not on this machine."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from instruments import fermi_gbm as G                       # noqa: E402

NAME = "090131A"
HAVE_RESULTS = (G.RESULTS / NAME / "episodes.json").exists() and (G.RESULTS / NAME / "spectra_zhu2026.csv").exists()
HAVE_DATA = (G.CODE_DIR.parent / "data" / "fermiGBM" / f"GRB{NAME}").exists()
results = pytest.mark.skipif(not HAVE_RESULTS, reason=f"needs results/zhu/{NAME} from the batch")


@results
def test_saved_gives_the_batch_episodes_and_spectra():
    episodes, spectra = G.saved(NAME)
    assert episodes["mode"] == "zhu2026" and episodes["has_pe"]
    assert {"PE", "ME"} <= set(spectra.index)
    assert spectra.loc["PE", "t_start"] == pytest.approx(episodes["pe"][0])
    assert spectra.loc["ME", "t_stop"] == pytest.approx(episodes["me"][1])


def test_saved_of_a_burst_never_measured_is_empty():
    assert G.saved("not_a_burst") == (None, None)


@pytest.mark.slow
@pytest.mark.skipif(not (HAVE_RESULTS and HAVE_DATA), reason=f"needs GRB {NAME} in data/fermiGBM and its batch results")
def test_adapter_episodes_match_batch():
    from zhu.io import code_version, read_json
    saved_code = read_json(G.RESULTS / NAME / "episodes.json", {}).get("code_version", "")
    if saved_code.split("zhu:")[-1] != code_version().split("zhu:")[-1]:
        pytest.skip(f"results/zhu/{NAME} was made with other code ({saved_code}); rerun the batch to compare")
    gbm, row, _ = G.burst(NAME)
    lc, found = G.episodes(gbm, row.Tres, "zhu2026", NAME)
    saved, _ = G.saved(NAME)
    assert np.allclose(found.pe, saved["pe"]) and np.allclose(found.me, saved["me"])
    assert found.durations()["T100_QE1"] == pytest.approx(saved["T100_QE1"])
    assert np.allclose(np.diff(lc.edges), row.Tres)                 # the light curve is binned at their time bin
