"""Write the run files of every nested-sampling retrieval used in the paper.

Each run file is a copy of ../run_dynesty.py with its input lines set (planet, instruments, seed, n_bins, n_orbits, A_B_true, INSOL_TRUTH, the sampler's
pfrac and nprocs) and a one-line description on top; the free-geometry runs also sample the orbital geometry (see FREE_GEOMETRY below). Everything else (models,
priors, sampler settings, forward-model resolution) is whatever run_dynesty.py says, so re-run this script after changing run_dynesty.py. run_all.sh runs
the files side by side and moves each finished one to done/.

    python make_runs.py                 # write the run files that are not in done/ yet (pending files whose content is unchanged are left alone,
                                        # so this is safe while run_all.sh runs)
    python make_runs.py --all           # every run file, also those in done/ (run_all.sh then runs them again)
    python make_runs.py --outdir DIR    # save the results to DIR instead of retrievals/dynesty_results, where 03_retrieval_series.ipynb reads them
                                        # (01_retrieval_example.ipynb with RESULTS_DIR = "dynesty_results")

A run saves its fits under its dataset tag and overwrites a stored run with the same tag, so use --outdir to rerun into a fresh folder.

The runs (249 datasets with three fits each, in 132 run files):
  retrievals          every paper planet in the joint NIRSpec/G395H fit and in MIRI/LRS (55 Cnc e: MIRI/LRS only), seeds 40-69 for TOI-2431 b and K2-141 b
                      and 40-43 for the other planets; at seed 42 also NRS1 and NRS2 alone (both reference planets) and the MIRI imaging filters (TOI-2431 b)
  observing time      TOI-2431 b: 3 and 10 orbits in both modes, 2 orbits (one visit) in MIRI/LRS
  settings            TOI-2431 b: 20 and 36 phase bins, true albedo 0.4 and 0.05; both reference planets: NRS1, NRS2 and MIRI/LRS in one fit
  free geometry       both reference planets, seeds 61-64, and the corner-plot realization of TOI-2431 b (MIRI/LRS, seed 44); saved in free_geometry/
  null test           TOI-2431 b with data drawn from the point-source model, seeds 43-45, both modes; saved in null_test/
  sampler weighting   TOI-2431 b at pfrac = 0 and 1, against the baseline pfrac = 0.8

nprocs is set to the value each stored run used. It changes the individual samples (dynesty batches its proposals by the number of workers), not their
statistics; set it to whatever suits the machine. The stored runs took about 620 hours of sampling in total, about 2,000 core-hours at their nprocs
(about a week on 16 cores with run_all.sh).
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, os.pardir, "run_dynesty.py")

JOINT, LRS = "JWST_NIRSpec.G395H_joint", "MIRI/LRS"   # the two primary modes (JOINT: NRS1 + NRS2 in one fit, bands.JOINT_INSTRUMENTS)
NRS1, NRS2 = "JWST_NIRSpec.G395H_NRS1", "JWST_NIRSpec.G395H_NRS2"
IMAGING = ("MIRI/F1280W", "MIRI/F1500W", "MIRI/F1800W")   # noise levels for TOI-2431 b only (inputs/noise/custom_sigma_1min.csv)
MODES = (JOINT, LRS)
THREE_DETECTORS = ((NRS1, NRS2, LRS),)   # one entry: a joint fit of the three detectors, one A_B (and F) and one baseline per detector

# The paper's sample (paper.PAPER_PLANETS), {file stem: planet}
PLANETS = {"toi2431b": "TOI-2431 b", "toi2260b": "TOI-2260 b", "k2141b": "K2-141 b", "k2131b": "K2-131 b", "toi561b": "TOI-561 b",
           "kepler78b": "Kepler-78 b", "toi1444b": "TOI-1444 b", "toi1798_02": "TOI-1798.02", "hd80653b": "HD 80653 b", "cnc55e": "55 Cnc e"}
TARGET, OTHER = "TOI-2431 b", "K2-141 b"   # the two reference planets
MODES_OF = {"55 Cnc e": (LRS,)}            # 55 Cnc e saturates NIRSpec/G395H (inputs/noise/unobservable.csv)
SEEDS_OF = {TARGET: range(40, 70), OTHER: range(40, 70)}
SEEDS = range(40, 44)                      # the other planets

RUNS = {}   # file stem -> (description, settings of make_run)


def add(stem, description, **settings):
    if stem in RUNS: raise ValueError(f"make_runs.py: run file {stem} defined twice")
    RUNS[stem] = (description, settings)


def bands_text(instruments):
    names = {JOINT: "joint NIRSpec", LRS: "MIRI/LRS", NRS1: "NRS1", NRS2: "NRS2"}
    return " + ".join("NRS1 + NRS2 + MIRI/LRS in one fit" if isinstance(b, tuple) else names.get(b, b) for b in instruments)


# Retrievals. nprocs as in the stored runs: 12 dynesty workers per fit, one process for seeds 50-69 (run sixteen side by side; one fit on many workers
# scales poorly, see run_all.sh) and three for 55 Cnc e (its four seeds side by side).
for stem, name in PLANETS.items():
    for seed in SEEDS_OF.get(name, SEEDS):
        bands = MODES_OF.get(name, MODES)
        if seed == 42 and name in (TARGET, OTHER): bands += (NRS1, NRS2)
        if seed == 42 and name == TARGET: bands += IMAGING
        nprocs = 3 if name == "55 Cnc e" else 1 if seed >= 50 else None
        add(f"{stem}_seed{seed}", f"{name}, seed {seed}: {bands_text(bands)}", planet=name, instruments=bands, seed=seed, nprocs=nprocs)

# Observing time
add("toi2431b_norbits3_seed42", f"{TARGET}, 3 orbits, seed 42: {bands_text(MODES)}", planet=TARGET, instruments=MODES, seed=42, n_orbits=3)
for seed in (42, 43):
    add(f"toi2431b_norbits10_seed{seed}", f"{TARGET}, 10 orbits, seed {seed}: {bands_text(MODES)}", planet=TARGET, instruments=MODES, seed=seed, n_orbits=10)
for seed in range(43, 48):
    add(f"toi2431b_norbits2_seed{seed}", f"{TARGET}, 2 orbits (one visit), seed {seed}: MIRI/LRS", planet=TARGET, instruments=(LRS,), seed=seed, n_orbits=2)

# Settings
for n_bins in (20, 36):
    add(f"toi2431b_nbins{n_bins}_seed42", f"{TARGET}, {n_bins} phase bins, seed 42: {bands_text(MODES)}", planet=TARGET, instruments=MODES, seed=42, n_bins=n_bins)
add("toi2431b_AB0.4_seed42", f"{TARGET}, true A_B = 0.4, seed 42: {bands_text(MODES)}", planet=TARGET, instruments=MODES, seed=42, A_B_true=0.4)
add("toi2431b_AB0.05_seed43", f"{TARGET}, true A_B = 0.05 (the prior edge A_B = 0 cuts the point-source posteriors), seed 43: {bands_text(MODES)}",
    planet=TARGET, instruments=MODES, seed=43, A_B_true=0.05)
for stem, name in (("toi2431b", TARGET), ("k2141b", OTHER)):
    add(f"{stem}_3det_seed43", f"{name}, seed 43: {bands_text(THREE_DETECTORS)}", planet=name, instruments=THREE_DETECTORS, seed=43)

# Free geometry: nprocs = 2 (two cores per run, the files run side by side), except the corner-plot run (12 workers)
for stem, name in (("toi2431b", TARGET), ("k2141b", OTHER)):
    for seed in range(61, 65):
        add(f"{stem}_freegeom_seed{seed}", f"{name}, free geometry, seed {seed}: {bands_text(MODES)}", planet=name, instruments=MODES, seed=seed, nprocs=2, free_geometry=True)
add("toi2431b_freegeom_seed44", f"{TARGET}, free geometry, seed 44 (the realization of the corner plots): MIRI/LRS", planet=TARGET, instruments=(LRS,), seed=44, free_geometry=True)

# Null test: data drawn from the point-source model, which the extended-source model then fits as the misspecified one
for seed in (43, 44, 45):
    add(f"toi2431b_null_seed{seed}", f"{TARGET}, null test (point-source truth), seed {seed}: {bands_text(MODES)}", planet=TARGET, instruments=MODES, seed=seed, truth="naive")

# Sampler weighting: the evidences at pfrac = 0 (all effort on lnZ) and 1 (all on the posterior) against those of the retrievals at 0.8
for seed in range(40, 50):
    bands = MODES if seed in (43, 44) else (JOINT,)
    add(f"toi2431b_pfrac0_seed{seed}", f"{TARGET}, pfrac = 0, seed {seed}: {bands_text(bands)}", planet=TARGET, instruments=bands, seed=seed, pfrac=0.)
for seed in range(43, 47):
    bands = MODES if seed in (45, 46) else (JOINT,)
    add(f"toi2431b_pfrac1_seed{seed}", f"{TARGET}, pfrac = 1, seed {seed}: {bands_text(bands)}", planet=TARGET, instruments=bands, seed=seed, pfrac=1.)


# The free-geometry runs: inserted into run_dynesty.py before its sampler settings
FREE_GEOMETRY = r'''# Free geometry (added by make_runs.py): the orbital geometry is sampled in every model, with Gaussian priors centred on the injected values with the published
# uncertainties (paper.GEOMETRY_SIGMA), truncated at 5 sigma and the inclination also at 90 deg. The parameters are those the planet's source fitted
# (paper.GEOMETRY_PARAMS): a/R*, Rp/R* and the impact parameter b for TOI-2431 b, with i = arccos(b / (a/R*)), and a/R*, Rp/R* and i for K2-141 b.
import numpy as np
from hyperphase.paper import GEOMETRY_PARAMS, GEOMETRY_SIGMA
from hyperphase.retrieval.priors import TruncNormal
_GEOMETRY_TRUE = {"a_rs": planet.R / planet.r2, "rp_rs": planet.r1 / planet.r2, "incl": planet.incl, "b": planet.R / planet.r2 * np.cos(np.radians(planet.incl))}
_GEOMETRY_LABEL = {"a_rs": r"$a/R_\star$", "rp_rs": r"$R_p/R_\star$", "incl": r"$i$ [deg]", "b": r"$b$"}

def _geometry_param(name):
    mu, sd = _GEOMETRY_TRUE[name], GEOMETRY_SIGMA[planet.name][name]
    return Parameter(name, TruncNormal(mu, sd, mu - 5 * sd, min(mu + 5 * sd, 90.) if name == "incl" else mu + 5 * sd), _GEOMETRY_LABEL[name])

_GEOM = tuple(_geometry_param(n.split("|")[0]) for n in GEOMETRY_PARAMS[planet.name])   # "a_rs|b": a/R* at fixed b
_psets_fixed_geometry = psets_for

def psets_for(labels=None):
    """psets_for above with the geometry appended to every model's ParameterSet."""
    return {m: ParameterSet((*ps.params, *_GEOM)) for m, ps in _psets_fixed_geometry(labels).items()}

'''
FREE_SAMPLER = '''# Free geometry: every likelihood call rebuilds the geometry, so the sampler stops earlier: dlogz = 0.1 for the baseline run (the batches set the final
# precision) and n_effective = 5000 instead of dynesty's 10,000 (a third fewer calls; lnZ error about 0.12 instead of 0.09).
'''
FREE_LOGL_KW = '''
# Likelihood resolution per model (model_pcurve keywords). With the geometry sampled, the extended-source insolation map is rebuilt at every call: at 150 x 75
# instead of 300 x 150 the call costs a quarter less and ln L moves by <= 0.01. The point-source maps cost about 1 ms and stay at the default.
LOGL_KW = {"hyperillum": dict(n_lo=150, n_la=75)}
'''


def _sub(src, pattern, repl, what):
    new, n = re.subn(pattern, repl, src, count=1, flags=re.M)
    if n != 1: raise ValueError(f"make_runs.py: could not find the {what} line in run_dynesty.py (pattern {pattern!r}); update the pattern")
    return new


def make_run(description, planet, instruments, seed=None, n_bins=None, n_orbits=None, A_B_true=None, pfrac=None, nprocs=None, free_geometry=False, truth=None,
             outdir=None):
    if free_geometry and truth is not None: raise ValueError("make_runs.py: a run is either a free-geometry run or a null test, not both")
    src = open(BASE).read()
    src = _sub(src, r'^planet = hi_planets_by_name\("[^"]+"\)', f'planet = hi_planets_by_name("{planet}")', "planet")
    src = _sub(src, r'^instruments = \([^\n]*?\)(?=\s*(#|$))', "instruments = " + repr(tuple(instruments)), "instruments")
    if seed is not None:     src = _sub(src, r'^seed = \S+', f"seed = {seed}", "seed")
    if n_bins is not None:   src = _sub(src, r'^n_bins = \S+', f"n_bins = {n_bins}", "n_bins")
    if n_orbits is not None: src = _sub(src, r'^n_orbits = \S+', f"n_orbits = {n_orbits}", "n_orbits")
    if A_B_true is not None: src = _sub(src, r'^A_B_true = \S+', f"A_B_true = {A_B_true}", "A_B_true")
    if truth is not None:    src = _sub(src, r'^INSOL_TRUTH = "[^"]+"', f'INSOL_TRUTH = "{truth}"', "INSOL_TRUTH")
    if pfrac is not None:    src = _sub(src, r'^(sampler = dict\([^\n]*?pfrac=)[0-9.]+', rf"\g<1>{pfrac:g}", "sampler pfrac")
    if nprocs is not None:   src = _sub(src, r'^(sampler = dict\([^\n]*?nprocs=)[0-9]+', rf"\g<1>{nprocs}", "sampler nprocs")
    if free_geometry:
        src = _sub(src, r'^(?=# Sampler\.)', FREE_GEOMETRY.replace("\\", "\\\\"), "sampler comment")
        src = _sub(src, r'^(sampler = dict\([^\n]*?)dlogz=[0-9.]+', FREE_SAMPLER + r"\g<1>dlogz=0.1, n_effective=5000", "sampler dlogz")
        src = _sub(src, r'^(sampler = dict\([^\n]*\n)', r"\g<1>" + FREE_LOGL_KW, "sampler")
        src = _sub(src, r'make_logl\(cfg, data, m, PSETS\[m\], labels=labels\)', "make_logl(cfg, data, m, PSETS[m], labels=labels, **LOGL_KW.get(m, {}))", "make_logl")
        src = _sub(src, r'^(    base = dict\([^\n]*\n)', r"\g<1>    base.update({p_.name: p_.prior.mu for p_ in _GEOM})   # the injected geometry\n", "summary base")
    root = repr(os.path.abspath(outdir)) if outdir else 'REPO_ROOT, "retrievals", "dynesty_results"'
    subfolder = ', "free_geometry"' if free_geometry else ', "null_test"' if truth is not None else ""
    src = _sub(src, r'^OUTDIR = [^\n]*', f"OUTDIR = os.path.join({root}{subfolder})   # set by make_runs.py", "OUTDIR")
    return f"# {description}   (written by make_runs.py from run_dynesty.py; run by run_all.sh, moved to done/ when finished)\n" + src


if __name__ == "__main__":
    args = sys.argv[1:]
    rewrite_done = "--all" in args
    outdir = None
    if "--outdir" in args:
        if args.index("--outdir") + 1 >= len(args): sys.exit("make_runs.py: --outdir needs a folder")
        outdir = args[args.index("--outdir") + 1]
    written, unchanged, done = [], [], []
    for stem, (description, settings) in RUNS.items():
        if not rewrite_done and os.path.exists(os.path.join(HERE, "done", stem + ".py")):
            done.append(stem); continue
        path, content = os.path.join(HERE, stem + ".py"), make_run(description, outdir=outdir, **settings)
        if os.path.exists(path) and open(path).read() == content:
            unchanged.append(stem); continue
        open(path, "w").write(content)
        written.append(stem)
        print(f"{stem + '.py':32s} {description}")
    stale = sorted(f for f in os.listdir(HERE) if f.endswith(".py") and f != "make_runs.py" and f[:-3] not in RUNS)
    print(f"\n{len(written)} of {len(RUNS)} run files written to {HERE}" + (f"; {len(unchanged)} pending and unchanged" if unchanged else "") +
          (f"; {len(done)} already in done/ left alone (--all rewrites them too)" if done else ""))
    if stale: print(f"WARNING: run files not defined here, which run_all.sh would run too: {', '.join(stale)}", file=sys.stderr)
