"""The structure of the package: its layers, the band and planet registries, and how a configuration takes its band."""
import ast
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import hyperphase.bands as bands
from hyperphase.bands import band_of, filter_lambda, filter_loc, observed_band, register_band
from hyperphase.model import FittingConfig, with_planet
from hyperphase.targets import CUSTOM, hi_planets_by_name, register_planet

PACKAGE = Path(__file__).resolve().parents[1] / "hyperphase"
# A module imports only from its own layer or the layers before it.
LAYERS = [{"defaults", "paths", "_util", "bands", "parallel"},
          {"planet", "orbit", "roche", "irradiance", "stellar", "newera", "transit", "phasecurve", "validation"},
          {"targets", "noise", "model", "paper"},
          {"bias", "retrieval"},
          {"plotting"}]


def layer(module):
    top = module.split(".")[1]
    return next(i for i, group in enumerate(LAYERS) if top in group)


def module_imports():
    """{module: (the hyperphase modules it imports at module level, the top-level packages it imports at module level)}"""
    out = {}
    for f in PACKAGE.rglob("*.py"):
        name = ".".join(("hyperphase",) + f.relative_to(PACKAGE).with_suffix("").parts).removesuffix(".__init__")
        own, other = set(), set()
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, ast.ImportFrom) and node.module:
                (own if node.module.startswith("hyperphase.") else other).add(node.module)
            elif isinstance(node, ast.Import):
                other |= {a.name for a in node.names}
        out[name] = (own, {m.split(".")[0] for m in other})
    return out


def test_modules_import_only_from_their_own_or_earlier_layers():
    for module, (own, _) in module_imports().items():
        if module == "hyperphase": continue
        for imported in own:
            assert layer(imported) <= layer(module), f"{module} (layer {layer(module)}) imports {imported} (layer {layer(imported)})"
        if module.startswith("hyperphase.bias."):
            assert not any(m.startswith("hyperphase.retrieval") for m in own), f"{module} imports the retrievals"


def test_only_plotting_imports_matplotlib_at_module_level():
    for module, (_, other) in module_imports().items():
        if not module.startswith("hyperphase.plotting"):
            assert not other & {"matplotlib", "PIL"}, f"{module} imports {sorted(other & {'matplotlib', 'PIL'})} at module level"


STELLAR_MODEL_DEFAULTS = """
import inspect, sys
sys.path.insert(0, sys.argv[1])
from hyperphase.model import FittingConfig
from hyperphase.phasecurve import full_PhaseCurve
print(FittingConfig.__dataclass_fields__["stellar_model"].default, inspect.signature(full_PhaseCurve).parameters["stellar_model"].default)
"""


def test_the_default_stellar_model_follows_the_environment(root):
    """HYPERPHASE_STELLAR_MODEL sets the default star of every function and configuration, and an unknown name raises. The defaults are bound
    at import, so each value gets a process of its own."""
    def run(value):
        return subprocess.run([sys.executable, "-c", STELLAR_MODEL_DEFAULTS, str(root)], cwd=root, capture_output=True, text=True, timeout=300,
                              env={**os.environ, "HYPERPHASE_STELLAR_MODEL": value})
    assert run("blackbody").stdout.split()[-2:] == ["blackbody", "blackbody"]
    assert "HYPERPHASE_STELLAR_MODEL" in run("bb").stderr


def test_a_registered_band_works_like_a_shipped_one():
    l1, l2 = filter_lambda("MIRI/LRS")
    register_band("test/LRS-red", 8e-6, l2, filter_loc("MIRI/LRS"), overwrite=True)
    assert np.allclose(filter_lambda("test/LRS-red"), (8e-6, l2))
    assert filter_loc("test/LRS-red") == filter_loc("MIRI/LRS")
    with pytest.raises(KeyError):
        register_band("test/LRS-red", 8e-6, l2, filter_loc("MIRI/LRS"))
    assert observed_band(filter_loc("MIRI/LRS"), *filter_lambda("MIRI/LRS"), "55 Cnc e") == tuple(filter_lambda("MIRI/LRS", "55 Cnc e"))   # the file stays MIRI/LRS's
    for table in (bands.FILTER_LAMBDA_UM, bands.FULL_LAMBDA_UM, bands.FILTER_FILES, bands.PANDEXO_MODES, bands.FILTER_LAMBDA_M, bands.FULL_LAMBDA_M):
        del table["test/LRS-red"]


def test_a_registered_planet_is_found_by_name(toi2431b):
    twin = replace(toi2431b, name="Test-1 b", incl=85.)
    register_planet(twin, source="a test")
    try:
        assert hi_planets_by_name("test 1b") is twin
        with pytest.raises(KeyError):
            register_planet(twin)
    finally:
        del CUSTOM["Test-1 b"]


def test_a_config_keeps_its_band_and_with_planet_takes_the_planets_own_part(toi2431b):
    """55 Cnc e saturates the blue end of MIRI/LRS (inputs/noise/partial_bands.csv). A FittingConfig keeps the band it is given; with_planet and
    the name-based constructors give the planet its own part of the band."""
    cnc = hi_planets_by_name("55 Cnc e")
    lrs, part = filter_lambda("MIRI/LRS"), filter_lambda("MIRI/LRS", cnc)
    assert part[0] > lrs[0]
    cfg = FittingConfig(planet=toi2431b, filter_path=filter_loc("MIRI/LRS"), l1=lrs[0], l2=lrs[1], stellar_model="blackbody")
    assert (replace(cfg, planet=cnc).l1, replace(cfg, planet=cnc).l2) == (lrs[0], lrs[1])
    moved = with_planet(cfg, cnc)
    assert np.allclose((moved.l1, moved.l2), part) and band_of(moved) == "MIRI/LRS"
    assert np.allclose((with_planet(moved, toi2431b).l1, with_planet(moved, toi2431b).l2), lrs)
    assert observed_band(cfg.filter_path, 6e-6, 9e-6, cnc) == (6e-6, 9e-6)   # not a standard band: kept
