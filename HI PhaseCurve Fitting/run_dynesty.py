import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "Utilities"))
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):   # mute "looking for spidermanrc..." in every process
    import fspiderman as sp
from HItargets_utils import hi_planets_by_name, filter_loc, filter_lambda, run_pandexo, rms_1min
from hyperillumination_phasecurve_utils import FittingConfig, ConstantBandAlbedo, ScaledBandAlbedo
from dynesty_utils import *


####################################################################################################
### Input
####################################################################################################

# Planet data
planet = hi_planets_by_name("K2-141 b") # "TOI-2431 b"

u = [0.15, 0.08]  # Band-integrated (MIR) quadratic stellar limb darkening coefficients [u1, u2] (placeholders from grey Eddington atmosphere, not from K2-141 data, will be set by theoretical stellar atmosphere models)
u_B = 0.6  # Bolometric linear stellar limb darkening coefficient (placeholder, from Sadh_Gavassino_2026)

A_B_true = 3/2 * 0.28 # 0.42 (Ag = 0.28 geometric albedo in the optical from Zieba 2022, placeholder)

# ASSUMPTION (!): Model for band-integrated albedo in MIRI/LRS band (MIR)
A_model = ConstantBandAlbedo(.1) # Band-integrated albedo of rocky surface fixed to A=0.1 (small reflected part) in MIR
# A_model = ScaledBandAlbedo(1.)

# Filter to be used
instrument, instrument_pandexo = "MIRI/LRS", "MIRI LRS"

INSOL_TRUTH = "hyperillum"   # insolation model the simulated data are drawn from
MODELS = ("naive", "hyperillum")   # insolation models fitted to those data

seed = 42

n_bins = 27 # number of bins in one orbit
n_orbits = 3 # number of orbits observed (3 orbits has been done)

sigma_1min_scaling = 1. # scale the noise level

# Priors (uniform); ParameterSet fixes the column order of theta and of res.samples
pset = ParameterSet((Parameter("A_B",    Uniform(0.0, 0.8), r"$A_B$"),
                     Parameter("offset", Uniform(-50., 50.), "offset [ppm]")))

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
    sigma_1min = sigma_1min_scaling * rms_1min(run_pandexo(planet.name, instrument_pandexo, t14_h=planet.T14), planet.T14, lrs_range)
    stellar_grid = sp.stellar_grid.gen_grid(l1, l2, logg=4.5, response=filter_path)

    # context for simulating data functions
    cfg = FittingConfig(planet=planet, stellar_grid=stellar_grid, filter_path=filter_path, l1=l1, l2=l2, A_=A_model,
                        sigma_1min=sigma_1min, n_orbits=n_orbits, n_bins=n_bins, reflected=True, u_B=u_B, u=u)

    print(f"\n{planet.name}  [{instrument}, {l1*1e6:.2f}-{l2*1e6:.2f} um]   a/Rs = {planet.R/planet.r2:.3f}, Rp/Rs = {planet.r1/planet.r2:.4f}, T_eff = {planet.T2_eff:.0f} K")
    print(f"Noise Level (1min) = {sigma_1min:.1f} ppm")
    print(f"period = {planet.period*24*60:.1f} min -> {n_bins} bins of {planet.period*24*60/n_bins:.1f} min")
    print(f"1-sigma per bin (over all orbits) = {cfg.sigma_bin:.1f} ppm")

    # SIMULATED OBSERVATION
    data = simulate_data(cfg, A_B_true, insolation_model=INSOL_TRUTH, seed=seed, plotting=False)

    # LogLikelihoods. Same data, same priors -> the two lnZ are comparable
    logls = {m: GaussianLogL.from_pset(cfg, data, m, pset) for m in MODELS}

    # Dynesty. run_nested_fit fills {insolation_model} in, so one SamplerConfig serves both models without them overwriting each other
    tag = (f"{planet.name}_{instrument}_{A_model.label}_AB{A_B_true:g}"
           f"_priors-{'+'.join(pset.names)}"   # which parameters were sampled, so runs with different free parameters cannot overwrite each other
           f"_nbins{n_bins}_norbits{n_orbits}_rms1min{sigma_1min}_sigmascale{sigma_1min_scaling:g}_seed{seed}").replace("/", "-").replace(" ", "")
    scfg = SamplerConfig(seed=seed, outfile=os.path.join(OUTDIR, tag + "_{insolation_model}.pkl"), **sampler)

    t0 = time.perf_counter()
    fits = {}
    for m, L in logls.items():
        print(f"\n=== {m} " + "=" * 70)
        fits[m] = run_nested_fit(L, pset, scfg)

    # Summary
    base = dict(A_B=A_B_true, offset=0.)   # the injected truth
    print("\n" + "=" * 78)
    for f in fits.values():
        posterior_summary(f, base)
    if all(m in fits for m in ("naive", "hyperillum")):
        bayes_factor(fits, "hyperillum", "naive", K=200)

    print(f"\ntotal {(time.perf_counter() - t0)/60:.1f} min")
    for m, f in fits.items():
        print(f"   {m:11s} -> {f.meta.get('file', '(not saved)')}")
