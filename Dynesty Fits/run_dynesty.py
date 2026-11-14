import os
import sys
import time
import traceback


def _repo_root(start):
    """The thesis folder: the first parent of `start` that holds Utilities/. Copies of this script (e.g. the series in "more runs/" and its done/ folder) then run from any depth."""
    d = os.path.abspath(start)
    while not os.path.isdir(os.path.join(d, "Utilities")):
        parent = os.path.dirname(d)
        if parent == d: raise FileNotFoundError(f"no Utilities/ folder above {start}")
        d = parent
    return d

REPO_ROOT = _repo_root(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "Utilities"))
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):   # mute "looking for spidermanrc..." in every process
    import fspiderman as sp
from HItargets_utils import hi_planets_by_name, filter_loc, filter_lambda, rms_1min, NoNoiseLevelError, resolve_instrument, detector_label, joint_name, JOINT_INSTRUMENTS, DEFAULT_A_B, DEFAULT_A, DEFAULT_U_BOL, DEFAULT_U, DEFAULT_N_ORBITS, DEFAULT_N_BINS, A_B_BOUNDS, F_BOUNDS, OFFSET_BOUNDS
from hyperillumination_phasecurve_utils import FittingConfig, ConstantBandAlbedo, ScaledBandAlbedo   # ScaledBandAlbedo(1.): A = A_B, the A_B-coupled alternative (only within the physical prior A_B_BOUNDS, where A >= 0)
from dynesty_utils import *


####################################################################################################
### Input
####################################################################################################

# Planet data
planet = hi_planets_by_name("TOI-2431 b") # "TOI-2431 b", "K2-141 b", "TOI-2260 b", "K2-157 b", KOI-1843.03

u, u_bol = DEFAULT_U, DEFAULT_U_BOL   # transit and irradiance limb darkening (baseline settings, HItargets_utils.py)

A_B_true = DEFAULT_A_B   # injected Bond albedo (baseline settings, HItargets_utils.py)

A_model = ConstantBandAlbedo(DEFAULT_A)   # band-integrated albedo A of the reflected light, independent of A_B

# Filters to be used: one dataset and one set of fits PER ENTRY, exactly as if the script were run once per entry (every other input is shared).
# An entry is a band name, or a JOINT fit of several detectors - one shared (A_B, F), one baseline per detector, as in the bias analysis (JWST_RETRIEVALS) - spelled either as the
# tuple of band names, ("JWST_NIRSpec.G395H_NRS1", "JWST_NIRSpec.G395H_NRS2"), or as its alias in HItargets_utils.JOINT_INSTRUMENTS, "JWST_NIRSpec.G395H_joint". The joint dataset
# draws its detectors from ONE noise stream, so its first detector's data are exactly the single-detector dataset of the same seed.
# "MIRI/F1280W" etc. have no PandExo mode: rms_1min takes sigma_1min from CUSTOM_SIGMA_1MIN in HItargets_utils.py. A band with no noise level for this planet
# (MIRI imaging for every planet but TOI-2431 b) is skipped with a warning, and the other entries still run.
instruments = ("JWST_NIRSpec.G395H_NRS1",)   # e.g. ("JWST_NIRSpec.G395H_NRS1", "JWST_NIRSpec.G395H_NRS2", "JWST_NIRSpec.G395H_joint", "MIRI/LRS", "MIRI/F1280W")

INSOL_TRUTH = "hyperillum"   # insolation model the simulated data are drawn from (no redistribution in the truth: any fitted F is bias)
F_TRUE = None                # heat redistribution of the truth, only for INSOL_TRUTH = "naive_redist" (None otherwise); the fitted F is then compared with it
MODELS = ("hyperillum", "naive", "naive_redist")  # insolation models fitted to those data; each one is sampled with its OWN pset (see PSETS)

seed = 42

n_bins = DEFAULT_N_BINS # number of bins in one orbit
n_orbits = DEFAULT_N_ORBITS # number of orbits observed

sigma_1min_scaling = 1. # scale the noise level

# Priors (uniform, the shared ranges of HItargets_utils); ParameterSet fixes the column order of theta and of res.samples.
# One baseline per detector: "offset" for a single band, "offset_<label>" per detector of a joint fit (labels: HItargets_utils.detector_label, e.g. NRS1, NRS2).
_A_B = Parameter("A_B", Uniform(*A_B_BOUNDS), r"$A_B$")
_F   = Parameter("F",   Uniform(*F_BOUNDS), r"$F$")

def offset_param(label=None):
    return Parameter("offset" if label is None else f"offset_{label}", Uniform(*OFFSET_BOUNDS), "offset [ppm]" if label is None else f"offset {label} [ppm]")

def psets_for(labels=None):
    """{model: ParameterSet} for a single band (labels=None) or the detectors `labels` of a joint fit; each model is sampled with its OWN set (only naive_redist has F)."""
    offsets = (offset_param(),) if labels is None else tuple(offset_param(l) for l in labels)
    return {"hyperillum":   ParameterSet((_A_B, *offsets)),
            "naive":        ParameterSet((_A_B, *offsets)),
            "naive_redist": ParameterSet((_A_B, *offsets, _F))}

# Sampler. pfrac=0.8 -> effort on the posterior; pfrac=0 -> all effort on lnZ (for the Bayes factor)
sampler = dict(dynamic=True, pfrac=0.8, nlive=500, nlive_batch=250, maxbatch=10, dlogz=0.01, nprocs=12)


####################################################################################################
####################################################################################################
####################################################################################################


OUTDIR = os.path.join(REPO_ROOT, "Dynesty Fits", "dynesty_results")   # every run, also one of the series in "more runs/", saves here


def run_instrument(instrument):
    """The whole pipeline for ONE entry of `instruments` - a band, or a joint fit of several detectors - with every other input taken from the module-level settings above: noise
    level(s), simulated data, the fit of every model in MODELS, summary and Bayes factors. Returns {model: NestedFit}. Running the script with several entries is identical to
    running it once per entry. Raises NoNoiseLevelError when a band has no noise level for the planet (the caller skips the entry)."""
    dets = resolve_instrument(instrument)
    joint = isinstance(dets, tuple)
    bands = list(dets) if joint else [dets]
    labels = [detector_label(b) for b in bands] if joint else None
    name = joint_name(bands) if joint else dets   # the instrument part of the file names

    # Noise level per band, straight from PandExo (served from the cache), or CUSTOM_SIGMA_1MIN where PandExo has no mode
    sigmas = [sigma_1min_scaling * rms_1min(planet, b, filter_lambda(b)) for b in bands]

    # context for simulating data functions: one FittingConfig per band; a tuple of them is a detector set (fitted jointly)
    cfgs = [FittingConfig(planet=planet, filter_path=filter_loc(b), l1=filter_lambda(b)[0], l2=filter_lambda(b)[1], A_=A_model,
                          sigma_1min=s_, n_orbits=n_orbits, n_bins=n_bins, reflected=True, u_bol=u_bol, u=u) for b, s_ in zip(bands, sigmas)]
    cfg = tuple(cfgs) if joint else cfgs[0]

    print(f"\n{planet.name}   a/Rs = {planet.R/planet.r2:.3f}, Rp/Rs = {planet.r1/planet.r2:.4f}, T_eff = {planet.T2_eff:.0f} K" + ("   JOINT fit of " + " + ".join(labels) if joint else ""))
    for b, c in zip(bands, cfgs):
        print(f"   [{b}, {c.l1*1e6:.2f}-{c.l2*1e6:.2f} um]   noise level (1 min) = {c.sigma_1min:.1f} ppm   ->   1-sigma per bin (over all orbits) = {c.sigma_bin:.1f} ppm")
    print(f"period = {planet.period*24*60:.1f} min -> {n_bins} bins of {planet.period*24*60/n_bins:.1f} min")

    # SIMULATED OBSERVATION. Reproducible: the same inputs and seed give the same data (the same seed in every band draws the same standardised noise, scaled by each band's sigma_bin;
    # a joint dataset continues one stream across its detectors, so its first detector's data equal that band's single dataset).
    data = simulate_data(cfg, A_B_true, insolation_model=INSOL_TRUTH, seed=seed, plotting=False, labels=labels, **({} if F_TRUE is None else dict(F=F_TRUE)))

    # LogLikelihoods. Same data for every model, each with its own pset (a JointGaussianLogL for a detector set)
    PSETS = psets_for(labels)
    missing = [m for m in MODELS if m not in PSETS]
    if missing: raise ValueError(f"no pset for model(s) {missing}; PSETS has {tuple(PSETS)}")
    logls = {m: make_logl(cfg, data, m, PSETS[m], labels=labels) for m in MODELS}

    # Dynesty. `tag` is everything the models share - it is the DATASET (dynesty_utils.DATASET_TAG), which plot_dynesty.ipynb finds again with find_dataset.
    # The free parameters differ per model, so "priors-..." goes LAST, after {insolation_model} (which run_nested_fit fills in): one glob on tag then picks up
    # every model of this dataset whatever it was sampled with, while runs with different free parameters still cannot overwrite each other.
    tag = dataset_tag(planet, instrument, sigmas, A_label=A_model.label, A_B=A_B_true, n_bins=n_bins, n_orbits=n_orbits, sigma_scaling=sigma_1min_scaling, seed=seed, pfrac=sampler["pfrac"])

    def outfile_of(pset):
        return os.path.join(OUTDIR, tag + "_{insolation_model}" + f"_priors-{'+'.join(pset.names)}.pkl")   # {insolation_model} stays a placeholder for run_nested_fit

    t0 = time.perf_counter()
    fits = {}
    print(f"\ndataset tag: {tag}")
    for m, L in logls.items():
        print(f"\n=== {m} " + "=" * 70)
        scfg = SamplerConfig(seed=seed, outfile=outfile_of(PSETS[m]), **sampler)
        fits[m] = run_nested_fit(L, PSETS[m], scfg)

    # Summary. The injected value of every parameter any model samples (every baseline is 0): without redistribution in the truth the injected F is 0, so the posterior's distance
    # from it IS the bias (posterior_summary only quotes a bias for parameters a model actually samples, so one base serves all three).
    base = dict(A_B=A_B_true, F=0. if data["truth"]["F"] is None else data["truth"]["F"], **{p_.name: 0. for p_ in PSETS["hyperillum"].params if p_.name.startswith("offset")})
    print("\n" + "=" * 78)
    for f in fits.values():
        posterior_summary(f, base)

    # Bayes factors, all against the truth's own model: "can these data tell hyperillumination from the naive alternative,
    # and from the naive alternative that is allowed to hide the difference in a redistribution F?"
    for den in ("naive", "naive_redist"):
        if all(m in fits for m in ("hyperillum", den)):
            bayes_factor(fits, "hyperillum", den, K=200)

    print(f"\n{name}: {(time.perf_counter() - t0)/60:.1f} min")
    for m, f in fits.items():
        print(f"   {m:11s} -> {f.meta.get('file', '(not saved)')}")
    return fits


# Everything below spawns the dynesty pool. multiprocessing uses SPAWN here, so each worker re-imports this file - without the guard it would re-run the whole script (PandExo, the fit, ...) in every worker.
if __name__ == "__main__":
    t_all = time.perf_counter()
    results, skipped, failed = {}, {}, {}
    for k, instrument in enumerate(instruments, 1):
        print("\n" + "#" * 100 + f"\n### {instrument if isinstance(instrument, str) else joint_name(instrument)}   ({k} / {len(instruments)})\n" + "#" * 100)
        try:
            results[instrument] = run_instrument(instrument)
        except NoNoiseLevelError as e:   # no PandExo mode and no CUSTOM_SIGMA_1MIN entry for this planet: nothing to simulate, so the band is skipped and the others still run
            skipped[instrument] = str(e)
            print(f"\nWARNING: {instrument} skipped for {planet.name} - no noise level: {e}", file=sys.stderr, flush=True)   # printed, not warnings.warn: HItargets_utils silences every warning at import (warnings.filterwarnings('ignore'))
        except Exception as e:   # any other failure of one band must not cost the other bands their run - as separate runs would not
            failed[instrument] = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    print("\n" + "=" * 100 + f"\ntotal {(time.perf_counter() - t_all)/60:.1f} min: {len(results)} of {len(instruments)} instrument(s) run" + (f", {len(skipped)} skipped" if skipped else "") + (f", {len(failed)} FAILED" if failed else ""))
    nm = lambda ins: ins if isinstance(ins, str) else joint_name(ins)
    for instrument, fits in results.items():
        print(f"   {nm(instrument):26s} " + ", ".join(f"{m} lnZ = {f.meta['lnZ']:.3f}" for m, f in fits.items()))
    for instrument in skipped:
        print(f"   {nm(instrument):26s} SKIPPED: no noise level for {planet.name} (no PandExo mode, no CUSTOM_SIGMA_1MIN entry)")
    for instrument, err in failed.items():
        print(f"   {nm(instrument):26s} FAILED: {err}")
    if failed: sys.exit(1)
