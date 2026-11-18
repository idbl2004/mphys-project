"""
Test for the DET_ANG fallback in grbtool/bursts.py: some responses of 2010-2012 have no DET_ANG, and the angle then comes from the burst's
spacecraft azimuth and elevation.
Run from ignacio/ with: python -m pytest -p no:cacheprovider tests/test_angle_fallback.py -q
"""
import sys
from pathlib import Path

import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grbtool.bursts import DATA_ROOT, _angle_from_az_el

BURSTS = {"GRB230307A": "bn230307656", "GRB211211A": "bn211211549"}


@pytest.mark.parametrize("burst", sorted(BURSTS))
def test_angle_from_az_el(burst):
    """Every response of the two bursts has DET_ANG, SRC_AZ and SRC_EL: the angle worked out from the last two is DET_ANG to 0.1 deg."""
    folder = DATA_ROOT / burst / "current"
    files = sorted(folder.glob(f"glg_cspec_n*_{BURSTS[burst]}_v*.rsp2"))
    if not files:
        pytest.skip(f"no responses of {burst} here")
    checked = 0
    for path in files:
        det = path.name.split("_")[2]
        with fits.open(path) as h:
            for e in h[2:]:
                assert _angle_from_az_el(det, e.header["SRC_AZ"], e.header["SRC_EL"]) == pytest.approx(e.header["DET_ANG"], abs=0.1)
                checked += 1
    assert checked > 100
