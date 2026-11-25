"""Shared fixtures. The tests use a blackbody star, so they run without the NewEra spectral cube; the tests that need the cube skip without it.

Run from the repository root:  pytest          (about 30 s)
                                pytest -m "not slow"
"""
import contextlib
import io
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Utilities"))      # also without `pip install -e .`
with contextlib.redirect_stdout(io.StringIO()):  # mute fspiderman's "looking for fspidermanrc file"
    import fspiderman  # noqa: F401

import newera_utils
from HItargets_utils import filter_lambda, filter_loc, hi_planets_by_name
from hyperillumination_phasecurve_utils import FittingConfig


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def toi2431b():
    """The paper's primary target (a/R* = 2.08)."""
    return hi_planets_by_name("TOI-2431 b")


@pytest.fixture(scope="session")
def lrs():
    """MIRI/LRS: the band edges [m] and the throughput file."""
    return filter_lambda("MIRI/LRS"), filter_loc("MIRI/LRS")


@pytest.fixture(scope="session")
def cfg_lrs(toi2431b, lrs):
    """TOI-2431 b in MIRI/LRS at the baseline settings, with a blackbody star."""
    (l1, l2), path = lrs
    return FittingConfig(planet=toi2431b, filter_path=path, l1=l1, l2=l2, sigma_1min=62.9, stellar_model="blackbody")


@pytest.fixture
def newera_cube():
    if not newera_utils.CUBE_FLUX.exists():
        pytest.skip("needs the NewEra spectral cube (inputs/stellar_model/NewEra/build_newera.ipynb)")
