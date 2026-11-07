import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "Utilities"))
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):   # mute "looking for spidermanrc..." in every process
    import fspiderman as sp
from HItargets_utils import hi_planets_by_name, filter_loc, filter_lambda, run_pandexo, rms_1min, DEFAULT_A_B, DEFAULT_A, DEFAULT_U_B, DEFAULT_U, DEFAULT_N_ORBITS, DEFAULT_N_BINS
from hyperillumination_phasecurve_utils import FittingConfig, ConstantBandAlbedo, ScaledBandAlbedo
from dynesty_utils import *


####################################################################################################
### Input
####################################################################################################

# Planet data
planet = hi_planets_by_name("TOI-2431 b") # "TOI-2431 b", "K2-141 b", "TOI-2260 b", "K2-157 b", KOI-1843.03

u, u_B = DEFAULT_U, DEFAULT_U_B   # transit and irradiance limb darkening (baseline settings, HItargets_utils.py)

A_B_true = DEFAULT_A_B   # injected Bond albedo (baseline settings, HItargets_utils.py)

A_model = ConstantBandAlbedo(DEFAULT_A)   # band-integrated albedo A of the reflected light, independent of A_B

# Filter to be used
instrument = "JWST_NIRSpec.G395H_NRS1" # "MIRI/LRS" #  "MIRI/F1280W" (no PandExo mode: rms_1min takes sigma_1min from CUSTOM_SIGMA_1MIN in HItargets_utils.py)

INSOL_TRUTH = "hyperillum"   # insolation model the simulated data are drawn from (F = 0 in the truth: any fitted F is bias)
MODELS = ("hyperillum", "naive", "naive_redist") # ("hyperillum", "naive", "naive_redist")   # insolation models fitted to those data; each one is sampled with its OWN pset (see PSETS)

seed = 42

n_bins = DEFAULT_N_BINS # number of bins in one orbit
n_orbits = DEFAULT_N_ORBITS # number of orbits observed

sigma_1min_scaling = 1. # scale the noise level

# Priors (uniform); ParameterSet fixes the column order of theta and of res.samples.
_A_B    = Parameter("A_B",    Uniform(0.0, 0.8), r"$A_B$")
_OFFSET = Parameter("offset", Uniform(-50., 50.), "offset [ppm]")
_F      = Parameter("F",      Uniform(0.0, 0.5), r"$F$")
PSETS = {"hyperillum":   ParameterSet((_A_B, _OFFSET)),
         "naive":        ParameterSet((_A_B, _OFFSET)),
         "naive_redist": ParameterSet((_A_B, _OFFSET, _F))}

# Sampler. pfrac=0.8 -> effort on the posterior; pfrac=0 -> all effort on lnZ (for the Bayes factor)
sampler = dict(dynamic=True, pfrac=0.8, nlive=500, nlive_batch=250, maxbatch=10, dlogz=0.01, nprocs=8)


####################################################################################################
####################################################################################################
####################################################################################################


# Everything below spawns the dynesty pool. multiprocessing uses SPAWN here, so each worker re-imports this file - without the guard it would re-run the whole script (PandExo, the fit, ...) in every worker.
if __name__ == "__main__":

    OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynesty_results")

    lrs_range = filter_lambda(instrument)
    l1, l2 = lrs_range
    filter_path = filter_loc(instrument)

    # Noise level, straight from PandExo
    sigma_1min = sigma_1min_scaling * rms_1min(planet, instrument, lrs_range)

    # context for simulating data functions
    cfg = FittingConfig(planet=planet, filter_path=filter_path, l1=l1, l2=l2, A_=A_model,
                        sigma_1min=sigma_1min, n_orbits=n_orbits, n_bins=n_bins, reflected=True, u_B=u_B, u=u)

    print(f"\n{planet.name}  [{instrument}, {l1*1e6:.2f}-{l2*1e6:.2f} um]   a/Rs = {planet.R/planet.r2:.3f}, Rp/Rs = {planet.r1/planet.r2:.4f}, T_eff = {planet.T2_eff:.0f} K")
    print(f"Noise Level (1min) = {sigma_1min:.1f} ppm")
    print(f"period = {planet.period*24*60:.1f} min -> {n_bins} bins of {planet.period*24*60/n_bins:.1f} min")
    print(f"1-sigma per bin (over all orbits) = {cfg.sigma_bin:.1f} ppm")

    # SIMULATED OBSERVATION. Reproducible: the same inputs and seed give the same data.
    data = simulate_data(cfg, A_B_true, insolation_model=INSOL_TRUTH, seed=seed, plotting=False)

    # LogLikelihoods. Same data for every model, each with its own pset
    missing = [m for m in MODELS if m not in PSETS]
    if missing: raise ValueError(f"no pset for model(s) {missing}; PSETS has {tuple(PSETS)}")
    logls = {m: GaussianLogL.from_pset(cfg, data, m, PSETS[m]) for m in MODELS}

    # Dynesty. `tag` is everything the models share - it is the DATASET, and it is what plot_dynesty.ipynb is pasted with.
    # The free parameters differ per model, so "priors-..." goes LAST, after {insolation_model} (which run_nested_fit fills in):
    # one glob on tag then picks up every model of this dataset whatever it was sampled with, while runs with different free
    # parameters still cannot overwrite each other.
    tag = (f"{planet.name}_{instrument}_{A_model.label}_AB{A_B_true:g}"
           f"_nbins{n_bins}_norbits{n_orbits}_rms1min{sigma_1min:.1f}_sigmascale{sigma_1min_scaling:g}_seed{seed}").replace("/", "-").replace(" ", "")

    def outfile_of(pset):
        return os.path.join(OUTDIR, tag + "_{insolation_model}" + f"_priors-{'+'.join(pset.names)}.pkl")   # {insolation_model} stays a placeholder for run_nested_fit

    t0 = time.perf_counter()
    fits = {}
    print(f"\ndataset tag: {tag}")
    for m, L in logls.items():
        print(f"\n=== {m} " + "=" * 70)
        scfg = SamplerConfig(seed=seed, outfile=outfile_of(PSETS[m]), **sampler)
        fits[m] = run_nested_fit(L, PSETS[m], scfg)

    # Summary. The truth has no redistribution, so the injected F is 0 and the posterior's distance from it IS the bias
    # (posterior_summary only quotes a bias for parameters a model actually samples, so one base serves all three).
    base = dict(A_B=A_B_true, offset=0., F=0.)
    print("\n" + "=" * 78)
    for f in fits.values():
        posterior_summary(f, base)

    # Bayes factors, all against the truth's own model: "can these data tell hyperillumination from the naive alternative,
    # and from the naive alternative that is allowed to hide the difference in a redistribution F?"
    for den in ("naive", "naive_redist"):
        if all(m in fits for m in ("hyperillum", den)):
            bayes_factor(fits, "hyperillum", den, K=200)

    print(f"\ntotal {(time.perf_counter() - t0)/60:.1f} min")
    for m, f in fits.items():
        print(f"   {m:11s} -> {f.meta.get('file', '(not saved)')}")
