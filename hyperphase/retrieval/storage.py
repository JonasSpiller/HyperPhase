"""Saving and finding stored retrievals: dataset tags, file lookup, loading (also of runs saved before the package existed) and
the check that a stored run matches the current forward model."""
import glob
import os
import pickle
import re
import sys
from dataclasses import fields

import numpy as np
import pandas as pd

from hyperphase.bands import joint_name, observed_band, resolve_instrument
from hyperphase.bias.detectors import detector_cfgs
from hyperphase.defaults import DEFAULT_A, DEFAULT_A_B, DEFAULT_N_BINS, DEFAULT_N_ORBITS
from hyperphase.model import ConstantBandAlbedo, FittingConfig
from hyperphase.paths import FILTER_DIR
from hyperphase.retrieval.likelihood import make_logl
from hyperphase.retrieval.sampler import SamplerConfig
from hyperphase.stellar import stellar_spidergrid


############################################################################################################################################
#####   Loading stored fits
############################################################################################################################################

# Old flat module -> the package modules that now hold its classes, for runs pickled before the code was a package
# (their pickles name e.g. dynesty_utils.NestedFit, hyperillumination_phasecurve_utils.FittingConfig, hyperillumination_utils.PlanetParams).
_LEGACY_MODULES = {
    "dynesty_utils": ("hyperphase.retrieval.sampler", "hyperphase.retrieval.priors", "hyperphase.retrieval.likelihood"),
    "hyperillumination_phasecurve_utils": ("hyperphase.model",),
    "hyperillumination_utils": ("hyperphase.planet",),
}


class _Unpickler(pickle.Unpickler):
    """pickle.Unpickler that finds the classes of runs saved before the package existed under their new modules."""
    def find_class(self, module, name):
        for new in _LEGACY_MODULES.get(module, ()):
            mod = __import__(new, fromlist=[name])
            if hasattr(mod, name): return getattr(mod, name)
        return super().find_class(module, name)


def load_fit(path):
    """Read back a NestedFit written by run_nested_fit (scfg.outfile), also one saved before the package existed (_LEGACY_MODULES)."""
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "rb") as f: fit = _Unpickler(f).load()
    fit.meta["file"] = path   # set only after pickling, so the file never holds its own path
    _relocate_filter_paths(fit)
    return fit


def _relocate_filter_paths(fit):
    """Point each cfg's filter_path at the file of the same name in FILTER_DIR. The pickle stores the path of the machine that ran the
    fit, and the forward model reads the file. A stored path that exists is kept, unless it is the same file under another spelling
    (a symlink, or another case): it then takes the canonical spelling, by which observed_band looks the band up."""
    if fit.meta.get("cfg") is None: return
    for c in detector_cfgs(fit.meta["cfg"]):
        path = getattr(c, "filter_path", None)
        if not isinstance(path, str): continue
        local = FILTER_DIR / os.path.basename(path)
        if local.exists() and (not os.path.exists(path) or os.path.samefile(path, local)):
            c.filter_path = str(local)


def load_fits(paths):
    """Several saved fits as {insolation_model: NestedFit}, from a glob pattern or an iterable of paths; a repeated model raises."""
    paths = sorted(glob.glob(os.path.expanduser(paths))) if isinstance(paths, str) else list(paths)
    if not paths: raise FileNotFoundError("no fit files matched")
    out = {}
    for p in paths:
        f = load_fit(p)
        m = f.meta["insolation_model"]
        if m in out: raise ValueError(f"two files for insolation_model={m!r}: {out[m].meta.get('file')} and {p}")
        out[m] = f
    return out


############################################################################################################################################
#####   Dataset tags: the file names of stored runs, and finding a stored dataset again
############################################################################################################################################

# A dataset is one simulated observation (planet, instrument, albedos, binning, noise, seed) sampled with every model at one pfrac; its tag is
# everything the models share, and a result file is tag + "_{insolation_model}_priors-<sampled names>.pkl". The free parameters come LAST, so one
# glob on the tag picks up every model of the dataset, while runs with different free parameters cannot overwrite each other. A joint fit joins
# the noise levels of its detectors with "+". pfrac ends the tag: the same data sampled with another pfrac is another run.
DATASET_TAG = "{planet}_{instrument}_{A_label}_AB{A_B:g}_nbins{n_bins}_norbits{n_orbits}_rms1min{rms}_sigmascale{sigma_scaling:g}_seed{seed}_pfrac{pfrac:g}"
_RESULT_FILE_RE = re.compile(r"^(?P<planet>[^_]+)_(?P<instrument>.+?)_(?P<A_label>(?:fixed|scaled)A[\d.]+)_AB(?P<A_B>[\d.]+)_nbins(?P<n_bins>\d+)_norbits(?P<n_orbits>\d+)"
                             r"_rms1min(?P<rms>[\d.+]+)_sigmascale(?P<sigma_scaling>[\d.]+)_seed(?P<seed>\d+)_pfrac(?P<pfrac>[\d.]+)"
                             r"_(?P<insolation_model>hyperillum|naive_redist|naive)_priors-(?P<priors>.+)\.pkl$")
DATASET_COLUMNS = ["planet", "instrument", "A_B", "n_bins", "n_orbits", "seed", "pfrac", "sigma_scaling", "A_label", "rms", "models", "subfolder", "tag"]


def _planet_name(planet):
    """File-name spelling of a planet: its PlanetParams name or the string given, without spaces ("TOI-2431 b" -> "TOI-2431b")."""
    return (planet.name if hasattr(planet, "name") else str(planet)).replace(" ", "")


def _instrument_name(instrument):
    """File-name spelling of an instrument: the band ("MIRI/LRS" -> "MIRI-LRS"), or the joint alias / 'joint-<labels>' of a detector tuple."""
    dets = resolve_instrument(instrument)
    return (joint_name(dets) if isinstance(dets, tuple) else dets).replace("/", "-")


def dataset_tag(planet, instrument, sigmas=None, *, A_label=None, A_B=DEFAULT_A_B, n_bins=DEFAULT_N_BINS, n_orbits=DEFAULT_N_ORBITS, sigma_scaling=1., seed=42, pfrac=SamplerConfig.pfrac):
    """The tag of a dataset (DATASET_TAG).

    planet     : a PlanetParams or its name.
    instrument : a band, a joint alias or a detector tuple.
    sigmas     : sigma_1min [ppm] per detector; None puts the glob wildcard in their place (find_dataset).
    A_label    : the band albedo's label (e.g. 'fixedA0.1'); None: that of ConstantBandAlbedo(DEFAULT_A).
    """
    rms = "*" if sigmas is None else "+".join(f"{s:.1f}" for s in np.atleast_1d(sigmas))
    return DATASET_TAG.format(planet=_planet_name(planet), instrument=_instrument_name(instrument), A_label=A_label or ConstantBandAlbedo(DEFAULT_A).label, A_B=A_B,
                              n_bins=n_bins, n_orbits=n_orbits, rms=rms, sigma_scaling=sigma_scaling, seed=seed, pfrac=pfrac)


def parse_result_file(path):
    """The fields of a result file's name as a dict (dataset_tag's keywords, 'rms', 'insolation_model', 'priors', 'tag'); None for
    a file that is not a result file."""
    name = os.path.basename(path)
    m = _RESULT_FILE_RE.match(name)
    if not m: return None
    d = m.groupdict()
    d.update(A_B=float(d["A_B"]), n_bins=int(d["n_bins"]), n_orbits=int(d["n_orbits"]), sigma_scaling=float(d["sigma_scaling"]), seed=int(d["seed"]), pfrac=float(d["pfrac"]))
    d["tag"] = name[:m.start("insolation_model") - 1]
    return d


def list_datasets(results_dir):
    """Every dataset in `results_dir` and its subfolders as a DataFrame (DATASET_COLUMNS), one row per folder and tag; subfolder '' is
    the main folder. Files that are not result files are ignored."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*.pkl")) + glob.glob(os.path.join(results_dir, "*", "*.pkl"))):
        d = parse_result_file(f)
        if d is None: continue
        sub = os.path.relpath(os.path.dirname(f), results_dir)
        d["subfolder"] = "" if sub == "." else sub
        rows.setdefault((d["subfolder"], d["tag"]), {**{k: d[k] for k in DATASET_COLUMNS if k != "models"}, "models": []})["models"].append(d["insolation_model"])
    df = pd.DataFrame(list(rows.values()), columns=DATASET_COLUMNS)
    df["models"] = [", ".join(m) for m in df["models"]]
    return df.sort_values(["subfolder", "planet", "instrument", "A_label", "A_B", "n_bins", "n_orbits", "sigma_scaling", "seed", "pfrac"], ignore_index=True)


def find_dataset(results_dir, planet, instrument, subfolder=None, **settings):
    """The ONE stored dataset of `planet` and `instrument` with these settings (dataset_tag's keywords; unset ones are the baseline),
    whatever its noise levels, as a stem: f"{results_dir}/{stem}_*.pkl" finds its files. Raises when none or several match.

    subfolder : runs whose difference from the baseline is not in the tag sit in a subfolder of `results_dir`: "null_test" (data drawn
                from the point-source model) or "free_geometry" (geometry sampled); the stem is then 'subfolder/tag'.
    """
    pattern = dataset_tag(planet, instrument, None, **settings)
    folder = os.path.join(results_dir, subfolder) if subfolder else results_dir
    parsed = [parse_result_file(f) for f in glob.glob(os.path.join(folder, pattern + "_*.pkl"))]
    tags = sorted({d["tag"] for d in parsed if d})
    if len(tags) == 1: return os.path.join(subfolder, tags[0]) if subfolder else tags[0]
    have = list_datasets(results_dir)
    have = have[have.planet == _planet_name(planet)].drop(columns="tag")
    msg = (f"{len(tags) or 'no'} dataset(s) in {folder} match {pattern}" + ("".join("\n   " + t for t in tags)) +
           f"\n{results_dir} holds for {_planet_name(planet)}:\n" + (have.to_string(index=False) if len(have) else "   (nothing)"))
    raise (ValueError if tags else FileNotFoundError)(msg)


def check_datasets(results_dir, datasets):
    """Compare a list of datasets, {key: find_dataset keywords}, with `results_dir`: prints the entries without a run and the runs not
    on the list, and returns {key: stem} of the entries found."""
    found = {}
    for key, kw in datasets.items():
        try: found[key] = find_dataset(results_dir, **kw)
        except FileNotFoundError: print(f"no run for {key!r}")
    have = list_datasets(results_dir)
    for stem in sorted({os.path.join(sub, tag) if sub else tag for sub, tag in zip(have.subfolder, have.tag)} - set(found.values())): print(f"run not listed: {stem}")
    return found


def check_fit_current(fit, cfg=None, rtol=1e-6):
    """Is a saved fit still a fit of the current forward model? Returns the relative difference between the pickled stellar grid and
    the one the current code builds from the same inputs (max over detectors); 0 means identical. inf where the planet's observed band
    itself changed (bands.PARTIAL_BANDS). Prints a warning above rtol or on a band change; raises for a cfg from an older FittingConfig."""
    cfg = cfg if cfg is not None else fit.meta["cfg"]
    rel_max, band_changed = 0., False
    for c in detector_cfgs(cfg):
        missing = [f.name for f in fields(FittingConfig) if not hasattr(c, f.name)]
        if missing: raise ValueError(f"the saved FittingConfig predates the field(s) {missing}: this fit ({fit.meta.get('file', '?')}) was run with an older forward model and has to be re-run")
        band = observed_band(c.filter_path, c.l1, c.l2, c.planet)   # the band the current code observes this planet in
        if not np.allclose(band, (c.l1, c.l2), rtol=0., atol=1e-12):
            print(f"WARNING [{fit.meta['insolation_model']}]: {fit.meta.get('file', 'this fit')} was run over {1e6 * c.l1:.2f}-{1e6 * c.l2:.2f} um, but the current code observes "
                  f"{c.planet.name} in {1e6 * band[0]:.2f}-{1e6 * band[1]:.2f} um of this band (bands.PARTIAL_BANDS). Re-run the fit before quoting it.", file=sys.stderr, flush=True)
            band_changed = True
        now = stellar_spidergrid((c.l1, c.l2), c.filter_path, c.planet.logg, c.planet.feh, stellar_model=c.stellar_model)
        then = c.stellar_grid
        rel = np.inf if (np.shape(now[1]) != np.shape(then[1]) or not np.allclose(now[0], then[0])) else float(np.max(np.abs(np.asarray(now[1], float) / np.asarray(then[1], float) - 1.)))   # inf: different T_eff nodes
        rel_max = max(rel_max, rel)
    if rel_max > rtol:
        print(f"WARNING [{fit.meta['insolation_model']}]: the stellar grid of the current code differs from the one this fit was run with by up to {100 * rel_max:.3g} % "
              f"(stellar model, filter or band changed since {fit.meta.get('file', 'the run')}). Its samples are not the posterior of the current likelihood: re-run the fit before quoting it.", file=sys.stderr, flush=True)
    return np.inf if band_changed else rel_max


def logl_from_fit(fit, cfg=None, **model_kw):
    """The likelihood (GaussianLogL or JointGaussianLogL) a saved fit was run with, rebuilt from the pickle alone: its cfg (with the
    stellar grid of the run), data, labels and model keywords. Pass `cfg` for a fit whose meta has none. Runs check_fit_current;
    the geometry-only caches are rebuilt, so the call takes a second or two."""
    cfg = cfg if cfg is not None else fit.meta.get("cfg")
    if cfg is None: raise ValueError("this fit has no cfg in its meta - pass cfg=... (or re-run the fit)")
    check_fit_current(fit, cfg)
    kw = {**fit.meta.get("model_kw", {}), **model_kw}
    return make_logl(cfg, fit.meta["data"], fit.meta["insolation_model"], fit.pset, labels=fit.meta.get("labels"), fixed=fit.meta.get("fixed"), **kw)
