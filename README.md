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
  and a fast split-evaluation `fast_lightcurve` (interpolated phase modulation, exact
  ingress/egress, totality skipped analytically — up to ~20× faster at bit-identical
  accuracy). Three insolation models: `hyperillum` (extended star), `naive` (point source) and
  `naive_redist` (point source plus Kreidberg & Loeb 2016 heat redistribution `F`, evaluated with
  SPIDERMAN's analytic Louden map; see `Hyperillumination/test_thermal_redist.ipynb`).
- **Bias analysis** (`simple_bias_analysis.py`, notebooks in `Hyperillumination/`): fit a *naive*
  point-source model to a hyper-illumination "truth" curve and measure the resulting bias
  in the recovered Bond albedo, mapped over `(a/R★, T_eff)` and `(a/R★, R_p/R★)`.
- **Speed & accuracy studies** of the SPIDERMAN forward model, run on my patched fork **fSPIDERMAN**
  (see below), all validated against K2-141b.

Repository layout: `Utilities/` (functions), `Hyperillumination/` (analysis notebooks),
`Basic PhaseCurve Test/`,
`Filters/` (instrument throughputs), `Literature/`.

---

# SPIDERMAN

The thermal forward model runs on **fSPIDERMAN** ([github.com/JonasSpiller/fSPIDERMAN](https://github.com/JonasSpiller/fSPIDERMAN)),
my fork of SPIDERMAN (Louden & Kreidberg 2018) with compatibility fixes for a modern scientific-Python stack, a ~4.5× faster
`lightcurve()`, a memory-leak fix, a clamp of the `direct_T` bicubic interpolation and a corrected stellar band-flux integral. Every
patch is documented in the fork's README. It installs as `fspiderman` (`pip install git+https://github.com/JonasSpiller/fSPIDERMAN.git`)
and coexists with an unmodified upstream `spiderman`, which this repository uses only as the benchmark baseline in
[`SpeedAccuracy_ThermalPhaseCurve.ipynb`](Hyperillumination/SpeedAccuracy_ThermalPhaseCurve.ipynb).
