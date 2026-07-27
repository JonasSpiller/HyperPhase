# Nightside temperatures of tidally locked rocky exoplanets

MSc thesis (ETH Zürich / Harvard) on **exoplanet phase-curve modeling**. The scientific goal is to quantify how two commonly under-modeled effects bias the
day–night temperature contrasts inferred from observed phase curves, and to map the
conditions under which they drive **atmospheric collapse**:

- **Hyper-illumination** — treating the host star as an *extended* source rather than a
  point, which reshapes the terminator insolation of short-period planets.
- **Eccentricity-driven tidal heating** — an internal energy source that warms the
  nightside independently of redistribution.

The reference system used throughout for validation is **K2-141b** (Malavolta et al.
2018; Barragán et al. 2018), a short-period rocky planet with a measured secondary-eclipse
depth of 142 ppm.

*Supervised by Robin Wordsworth (Department of Earth & Planetary Sciences, Harvard University) Sebastian Zieba (Center for Astrophysics | Harvard & Smithsonian) and Sascha Quanz (Institute for Particle and Astrophysics, ETH Zurich).*

## Current status

- **Forward model** (`Utilities/`): a hyper-illumination reflected + thermal phase-curve
  model built on SPIDERMAN, with a custom insolation → temperature-map → `direct_T`
  pipeline (`hyperillumination_phasecurve_utils.py`), orbital helpers (`kepler_utils.py`),
  and a fast split-evaluation `fast_lightcurve` (exact occultation + interpolated phase
  modulation, ~5× faster at unchanged accuracy).
- **Bias analysis** (`bias_analysis.py`, notebooks in `Hyperillumination/`): fit a *naive*
  point-source model to a hyper-illumination "truth" curve and measure the resulting bias
  in the recovered Bond albedo, mapped over `(a/R★, T_eff)` and `(a/R★, R_p/R★)`.
- **Speed & accuracy studies** of the SPIDERMAN forward model, and a locally **patched,
  accelerated SPIDERMAN** build (see below), all validated against K2-141b.

Repository layout: `Utilities/` (functions), `Hyperillumination/` (analysis notebooks),
`SPIDERMAN_fixes/` (the SPIDERMAN patches + their documentation), `Basic PhaseCurve Test/`,
`Filters/` (instrument throughputs), `Literature/`.

---

# Using the improved SPIDERMAN

The thermal forward model relies on **SPIDERMAN** (Louden & Kreidberg 2018,
[github.com/tomlouden/SPIDERMAN](https://github.com/tomlouden/SPIDERMAN)). Upstream
SPIDERMAN (last commit Aug 2020) has two problems for this project:

1. it does **not build or run** on a modern scientific-Python stack, and
2. its `lightcurve()` C routine is the dominant **runtime bottleneck** of the fitting and
   MCMC workloads.

I therefore run a **patched build**. The patches live in [`SPIDERMAN_fixes/`](SPIDERMAN_fixes/)
as two independent unified diffs against a pristine upstream clone, each with a companion
write-up:

| patch | what it does |
|---|---|
| `spiderman-version.patch` | compatibility — makes upstream build & run on modern Python/numpy/matplotlib/setuptools |
| `spiderman-speed.patch` | performance — ~4–5× faster `lightcurve()`, output bit-identical|

The three subsections below reproduce, from scratch, a plain `spiderman` install with (1) only the
version fixes, (2) only the speed patch, or (3) **both** (what I use).

### Requirements (all three recipes)

- A **C compiler**
- An activated Python environment with **`numpy`, `setuptools`, `wheel`** installed.
  Everything is built with `pip install . --no-build-isolation` so the C extension links
  against *your environment's own* numpy.
- A checkout of **this repository**, so the patch files are available locally. In the
  commands below, set:
  ```bash
  REPO=/path/to/this/repo        # the folder that contains SPIDERMAN_fixes/
  ```
- Tested with: Python 3.11, numpy 2.4.6, matplotlib 3.10, setuptools 82 (Apple Silicon / clang).

> After any of these installs, **do not** run `pip install spiderman-package` from PyPI
> again — it silently overwrites the patched build with the broken wheel.

## 1. Version fixes only (`spiderman-version.patch`)

Stock `spiderman-package` 1.0.3 predates modern scientific Python. The
PyPI wheel's precompiled C extension is **ABI-incompatible with numpy ≥ 2** (any
non-constant `direct_T` brightness grid yields garbage temperatures or segfaults inside
the bicubic-spline routines), and building the Aug-2020 source from scratch fails on a
current stack for four independent reasons.

This patch (6 small changes, none touching the physics) makes it **build and
run on a modern stack**, tested on **Python 3.11 / numpy 2.4.6 / matplotlib 3.10 /
setuptools 82**:

- `setup.py`: `license` must be a string, not a list (modern setuptools calls `.split()` on it);
- `setup.py`: drop the unused `import numpy.distutils.misc_util` (`numpy.distutils` was removed in numpy 2.0 and breaks setuptools ≥ 72);
- `c_src/_web.c`, `c_src/_heron.c`: `NPY_IN_ARRAY → NPY_ARRAY_IN_ARRAY` (39×; the old flag alias is gone from numpy 2.x headers);
- `spiderman/plot.py`: `matplotlib._png.read_png → matplotlib.image.imread` (private module removed in matplotlib 3.3);
- `spiderman/read_model.py`: `np.array([LO,LA,T]) → np.array([LO,LA,T], dtype=object)` (ragged arrays need an explicit object dtype since numpy 1.24);
- `spiderman/stellar_grid.py`: de-list the boolean mask `[(…)] → (…)`.

**Install from scratch:**
```bash
REPO=/path/to/this/repo
git clone https://github.com/tomlouden/SPIDERMAN.git
cd SPIDERMAN
git apply "$REPO/SPIDERMAN_fixes/spiderman-version.patch"
pip uninstall -y spiderman-package          # remove any broken PyPI install (ok if absent)
pip install . --no-build-isolation
python -c "import spiderman; print('OK', spiderman.__file__)"
```

## 2. Speed patch only (`spiderman-speed.patch`)

Four C-level changes, none of which touch the physics — they only remove redundant work in the inner
per-node loops:

- **1.** binary-search the ascending brightness grid instead of two O(*n*) linear scans (`util.c`);
- **2.** precompute the 16 bicubic coefficients **once per grid cell** instead of once per (segment × time-step), and drop the per-call heap churn (`bicubic.c`, `web.c`, `generate.c`);
- **3.** reentrant mapping routines that write into stack buffers instead of `malloc`/`free` (`generate.c`, `orthographic.c`);
- **4.** direct interval index for the uniform blackbody spline grid, no search (`blackbody.c`, `spline.c`).

Net effect: **`lightcurve()` runs ~4–5× faster** (measured 4.56× for a `direct_T`,
`N_LAYERS=60`, `len(t)=1000` curve), with output **bit-for-bit identical**
except a ≤ 0.04 ppm float-rounding difference on the bicubic path at sharp terminators
(≈ 25× below the `N_LAYERS` truncation error there, negligible vs. JWST noise).

**Install from scratch**:
```bash
REPO=/path/to/this/repo
git clone https://github.com/tomlouden/SPIDERMAN.git
cd SPIDERMAN
git apply "$REPO/SPIDERMAN_fixes/spiderman-speed.patch"
pip uninstall -y spiderman-package
pip install . --no-build-isolation
python -c "import spiderman; print('OK', spiderman.__file__)"
```

## 3. Both version + speed patches (recommended — the build I use)

The two patches are independent and **compose cleanly**; apply the version fixes first so
the tree builds on a modern stack, then the speed patch. The result is a modern-compatible,
~4–5× faster, drop-in `spiderman`.

**Install from scratch:**
```bash
REPO=/path/to/this/repo
git clone https://github.com/tomlouden/SPIDERMAN.git
cd SPIDERMAN
git apply "$REPO/SPIDERMAN_fixes/spiderman-version.patch"   # 1. modern-stack compatibility
git apply "$REPO/SPIDERMAN_fixes/spiderman-speed.patch"     # 2. ~4-5x speedup
pip uninstall -y spiderman-package
pip install . --no-build-isolation
python -c "import spiderman; print('OK', spiderman.__file__)"
```

> [!IMPORTANT]
> **This code base uses two SPIDERMAN builds side by side**, imported under different names — the patched one does the work, the version-only one is kept purely as a benchmark baseline:

| import | built via | role in this repository |
|---|---|---|
| **`fspiderman`** | recipe 3 — version **+ speed** patches | used throughout the pipeline |
| **`spiderman`** | recipe 1 — version fixes **only** | used only in [`SpeedAccuracy_ThermalPhaseCurve.ipynb`](Hyperillumination/SpeedAccuracy_ThermalPhaseCurve.ipynb) to benchmark speed & accuracy against `fspiderman` |

<sub>To keep both importable at once, `fspiderman` is built under its own package name rather than as a drop-in `spiderman`.</sub>

> [!TIP]
> **To run this model pipeline**, you don't need the two-build setup above. Install one patched `spiderman` (recipe 1 or 3), then make the `fspiderman` imports resolve to it — either way works:

- **change the imports** — replace `import fspiderman as sp` with `import spiderman as sp`; or
- **alias `fspiderman` to your patched `spiderman`** so the code runs unchanged:
  ```bash
  SITE=$(python -c "import spiderman, os; print(os.path.dirname(os.path.dirname(spiderman.__file__)))")
  printf 'import sys, spiderman\nsys.modules[__name__] = spiderman\n' > "$SITE/fspiderman.py"
  python -c "import fspiderman as sp; print('OK ->', sp.__file__)"
  ```

<sub>With a single build, `import spiderman` and `import fspiderman` point to the same library, so the benchmark in `SpeedAccuracy_ThermalPhaseCurve.ipynb` compares identical builds — irrelevant for running the pipeline.</sub>