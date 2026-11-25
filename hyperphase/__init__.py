"""HyperPhase: phase curves of rocky exoplanets close enough to their star that it can no longer be treated as a point source.

The package is layered; a module imports only from the layers above it in this list:
- defaults, paths, bands: the baseline settings, where the data live, the observing bands;
- planet, orbit, roche, irradiance, stellar, newera, transit, phasecurve: the forward model (no pandas, no matplotlib);
- targets, noise, model: planet parameters, noise levels, and the model of the fits (FittingConfig, model_pcurve);
- bias, retrieval: the noiseless analysis and the nested-sampling retrievals;
- plotting: the figures.
paper holds the configuration of the paper and validation the alternative irradiance integrators.

The most used names are available from the top level (hyperphase.full_PhaseCurve, ...); they are imported on first use.
"""
from importlib import import_module


_API = {
    "PlanetParams": "planet",
    "hi_planets_by_name": "targets", "register_planet": "targets", "load_planets": "targets",
    "filter_lambda": "bands", "filter_loc": "bands", "register_band": "bands",
    "insolation": "irradiance", "surface_temperature": "irradiance",
    "full_PhaseCurve": "phasecurve", "thermal_PhaseCurve": "phasecurve", "reflected_PhaseCurve": "phasecurve", "nightside_flux": "phasecurve",
    "T_bright": "stellar",
    "FittingConfig": "model", "model_pcurve": "model", "with_planet": "model",
    "rms_1min": "noise",
}
__all__ = sorted(_API)


def __getattr__(name):
    if name in _API: return getattr(import_module(f"{__name__}.{_API[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_API))
