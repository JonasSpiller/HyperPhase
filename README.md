# HyperPhase

Phase curves of rocky exoplanets that orbit so close to their star that it can no longer be treated as a point source.
At $2 < a/R_\star < 3$ the star subtends an angular radius of 20°–30° as seen from the planet and illuminates two thirds to three quarters of its
surface, so a bare rock emits from beyond the classical terminator. HyperPhase computes the irradiance of the extended stellar disk
("hyper-illumination"), the resulting surface-temperature maps, and thermal and reflected phase curves in the JWST bands. It then quantifies
what the standard point-source model, fitted with a free Bond albedo and heat redistribution, makes of such a phase curve: a spurious heat
redistribution and an albedo biased low. The code covers noiseless fits with Fisher uncertainties and predicted Bayes factors, robustness to
nuisance parameters, and nested-sampling retrievals of simulated observations.

<p align="center">
  <img src="docs/temperature_maps_TOI-2431b.png" width="560" alt="Surface temperature maps of TOI-2431 b for a point source and an extended star">
</p>

*Surface temperature of the bare rock TOI-2431 b ($`a/R_\star = 2.08`$, Bond albedo 0.2) for a point-source star (top) and the extended star
(bottom), seen edge-on at quadrature (left) and at transit (right). The extended star heats a ring beyond the terminator, which is visible even when
the nightside faces the observer.*

This repository accompanies the paper **Spiller, Zieba & Wordsworth (in prep.), *Hyper-Illumination in Ultra-Short-Period Rocky Planets:
How Finite-Star Geometry Biases Albedos and Heat Redistribution Inferred from JWST Phase Curves***.

## Installation

```bash
git clone --depth 1 https://github.com/JonasSpiller/HyperPhase.git   # the current version only, without the history (about 25 MB instead of 300 MB)
cd HyperPhase
conda env create -f environment.yml    # Python 3.11 with the package versions of the paper; builds fSPIDERMAN and runs pip install -e .
conda activate hyperphase
```

The thermal phase curves use [fSPIDERMAN](https://github.com/JonasSpiller/fSPIDERMAN), a fork of SPIDERMAN (Louden & Kreidberg 2018) with
fixes for numpy ≥ 2, a faster light curve, a corrected band integration, a temperature-map interpolation that keeps a 0 K nightside at
0 K, and errors that raise Python exceptions where SPIDERMAN ends the process (a planet inside its star). It is a C extension, so building it
needs a C compiler. To use an existing environment instead, install fSPIDERMAN from its repository and this repository with `pip install -e .`
(editable: the package `hyperphase` reads its data from `inputs/` of the repository; the environment variable `HYPERPHASE_DATA` points it
elsewhere, see `hyperphase/paths.py`).

Data that are not in the repository:

- **Stellar spectra.** By default (`stellar_model="newera"`) the planet-to-star contrast is normalised by NewEra model spectra
  (Hauschildt et al. 2025) at each host's $T_{\rm eff}$, $\log g$ and [Fe/H]. This needs a 113 MB spectral cube that is not in the
  repository: build it once with `inputs/stellar_model/NewEra/build_newera.ipynb` from the NewEra JWST product
  ([doi:10.25592/uhhfdm.18108](https://doi.org/10.25592/uhhfdm.18108), about 12 GB). Without the cube, use a blackbody star: pass
  `stellar_model="blackbody"` to a call, or set the environment variable `HYPERPHASE_STELLAR_MODEL=blackbody` before starting Python or
  Jupyter, which makes it the default of every function and notebook. The albedo biases stay the same, but the uncertainties and the
  evidence scale with the stellar flux in the band, so the numbers differ from the stored notebook outputs (for TOI-2431 b in MIRI/LRS
  the uncertainty of $A_B$ is about 9 % larger and the expected $\ln B$ about 15 % lower). The stored retrievals in
  `retrievals/example_runs/` were run with the NewEra star: without the cube their likelihoods are rebuilt from the stellar grid saved
  in each pickle, while the last section of `retrievals/01_retrieval_example.ipynb` (derived quantities) and
  `validation/blackbody_vs_newera.ipynb` need the cube.
- **Noise levels.** `inputs/noise/pandexo_cache.csv` holds the PandExo precision of the planets and bands of the paper, computed with
  PandExo and Pandeia 2026.7. Without PandExo installed these values are used as they are; with another version installed they are recomputed.
  Other planets and bands run PandExo, which needs the Pandeia reference data (`pandeia_refdata`) and the CDBS files (`PYSYN_CDBS`). Their
  results go to `~/.cache/hyperphase/pandexo_cache.csv` (or the file `HYPERPHASE_PANDEXO_CACHE` names), which is read before the shipped
  cache, so the shipped cache stays as it is.

## Quickstart

```python
import matplotlib.pyplot as plt
from hyperphase.bands import filter_lambda, filter_loc
from hyperphase.phasecurve import full_PhaseCurve
from hyperphase.targets import hi_planets_by_name

planet = hi_planets_by_name("TOI-2431 b")   # the paper's parameters of TOI-2431 b
band = "MIRI/LRS"
for model, label in (("hyperillum", "extended source"), ("naive", "point source")):
    # a bare rock with Bond albedo 0.2; stellar_model="blackbody" runs without the NewEra stellar spectra (see Installation)
    t, flux, depth = full_PhaseCurve(planet, filter_lambda(band), filter_path=filter_loc(band),
                                     insolation_model=model, A_B=0.2, stellar_model="blackbody")
    plt.plot(t / planet.period, (flux - 1) * 1e6, label=f"{label}: occultation depth {depth:.0f} ppm")
plt.ylim(-50, 250); plt.xlabel("orbital phase"); plt.ylabel("relative flux [ppm]"); plt.legend(); plt.show()
```

The insolation models are `"hyperillum"` (extended star), `"naive"` (point source) and `"naive_redist"` (point source with the heat
redistribution $F$ of Kreidberg & Loeb 2016, passed as `F=`). The numbered notebooks in `forward_model/`, starting with
`01_irradiance.ipynb`, show the irradiance, the temperature maps and the thermal and reflected curves step by step.

**Your own planet or band.** Every function takes the planet and the band directly, so nothing has to be added to a table:

```python
from hyperphase.planet import PlanetParams
from hyperphase.bands import register_band
from hyperphase.targets import register_planet
from hyperphase.phasecurve import full_PhaseCurve

planet = PlanetParams(name="My planet", period=0.3, incl=85., ecc=0., w=90., r1=0.015, r2=1., R=2.5, T2_eff=4500., a_AU=0.008, T14=1.0,
                      logg=4.6, feh=0.)   # r1, r2, R in any consistent units (here stellar radii); T14 in hours
t, flux, depth = full_PhaseCurve(planet, (5e-6, 12e-6), insolation_model="hyperillum", A_B=0.2, stellar_model="blackbody")   # a 5-12 um top-hat
register_planet(planet)   # optional: hi_planets_by_name("My planet") now finds it; a table in the format of inputs/planets/ loads with load_planets
register_band("NIRCam/F444W", 3.9e-6, 5.0e-6, "/path/to/f444w_throughput.dat")   # optional: filter_lambda / filter_loc then know the band
```

Single values of a catalogued planet change with `dataclasses.replace(hi_planets_by_name("TOI-2431 b"), incl=75.)`, and the baseline settings
per call (`A_B=`, `u_bol=`, ...) or per configuration (`dataclasses.replace(cfg, u_bol=0.6)`). The noise levels are the exception: `rms_1min`
reads its PandExo inputs from the archive by the planet's name, so a planet the archive does not have raises `NoNoiseLevelError`, and a
modified catalogued planet keeps the catalogued noise. Give such a configuration a `sigma_1min` [ppm] of your own.

## Conventions

- **Models.** `"hyperillum"` is the extended star of the paper, `"naive"` the point source, and `"naive_redist"` the point source with the heat
  redistribution $F$ of Kreidberg & Loeb (2016). `"hyperillum_redist"`, the extended star with $F$, serves as a simulated truth and can be sampled in the
  retrievals (`MODELS` of `run_dynesty.py`); the noiseless fits use the two point-source models.
- **Geometry.** `PlanetParams` names the planet radius `r1`, the stellar radius `r2` and the semi-major axis `R`, in any consistent units: only
  $R_p/R_\star$ = `r1/r2` and $a/R_\star$ = `R/r2` enter. `a_AU` is the semi-major axis in AU, for the light-travel time.
- **Albedos and emissivities.** `A_B` is the Bond albedo of the energy balance, `A` the band-integrated albedo of the reflected light (the field
  `A_` of `FittingConfig`, a function of `A_B`). `emissivity_bol` enters the energy balance, $\varepsilon_{\rm bol}\,\sigma T^4 = (1 - A_B)\,I$,
  and `emissivity` scales the thermal flux in the band.
- **Limb darkening.** `u_bol` is the bolometric linear limb darkening of the irradiance, `u = (u1, u2)` the quadratic limb darkening of the transit.
- **Units.** Times in days from mid-transit, wavelengths in m, irradiance in W m⁻², temperatures in K. Light curves are relative to the star:
  `full_PhaseCurve` returns 1 + planet − transit, `model_pcurve` the same minus 1 in ppm. Occultation depths are in ppm, noise levels in ppm
  (the precision of a one-minute integration).
- **"Joint"** means two things: the fit of $A_B$ and $F$ together (`fit_redist`, the `_J` columns of the tables), and the fit of several detectors
  with one $A_B$ (`JointGaussianLogL`, the mode `"JWST_NIRSpec.G395H_joint"` of NRS1 and NRS2).
- **Baseline settings** (albedos, emissivities, limb darkening, orbits, bins, map resolution, stellar model) and the fit ranges are defined once,
  in `hyperphase/defaults.py` (`DEFAULT_*`, `A_B_BOUNDS`, ...), and are the default of every function argument. The paper's planet sample, the
  published uncertainties of its reference planets and its fit list are in `hyperphase/paper.py` (`PAPER_PLANETS`, `GEOMETRY_SIGMA`, ...).
- **Data tables.** The curated planet parameters (`inputs/planets/curated_planets.csv`, with the published source of each value, taking precedence over
  the archive), the bands (`inputs/filters/bands.csv`) and the per-planet noise exceptions (`inputs/noise/`: bands without a PandExo mode, modes a
  planet cannot be observed in, bands observable only in part) are CSV files, read when the package is imported.

## Repository layout

| Folder | Contents |
|---|---|
| `hyperphase/` | The package, in layers (a module imports only from its own layer or the layers before it). Settings and data: `defaults` (baseline settings, fit ranges), `paths` (where the data live), `bands` (observing bands). Forward model: `planet` (`PlanetParams`), `irradiance` (extended- and point-source irradiance, heat redistribution, temperature maps), `stellar` (Planck functions, stellar spectra, SPIDERMAN's stellar grid, brightness temperatures, limb darkening), `newera` (the NewEra cube), `orbit`, `roche`, `transit` (batman), `phasecurve` (thermal and reflected phase curves). Catalogue and fit model: `targets` (archive and curated planets), `noise` (PandExo), `model` (`FittingConfig`, `model_pcurve`). Analyses: `bias` (noiseless fits, Fisher uncertainties, evidence, nuisance parameters, maps, per-planet tables, sweeps), `retrieval` (simulated data, priors, likelihoods, dynesty, stored runs and their predictions). Figures: `plotting`. Also `paper` (the paper's configuration) and `validation` (alternative irradiance integrators). |
| `forward_model/` | The forward model, in notebooks numbered in reading order: irradiance and illumination zones, thermal, reflected and full phase curves, heat redistribution, the nightside signal. |
| `bias_analysis/` | The noiseless analysis, in notebooks numbered in reading order: albedo bias and spurious redistribution for single planets, over $(a/R_\star, T_{\rm eff})$ and for the planet sample; predicted Bayes factors; nuisance parameters; planets at their Roche limit; unconfirmed TESS candidates. |
| `retrievals/` | Nested-sampling retrievals: `run_dynesty.py` (one run), `paper_runs/` (every retrieval of the paper), `example_runs/` (three stored retrievals of TOI-2431 b), and numbered notebooks that inspect a stored run, validate the sampler and compare the retrievals with their predictions. |
| `validation/` | Checks of the model: the irradiance against InstellCa (Sadh & Gavassino 2026), the speed and accuracy of the thermal light curves (fSPIDERMAN against SPIDERMAN), and the blackbody against the NewEra star. |
| `inputs/` | What the model reads but does not compute, each with the notebooks that made it: |
| `inputs/filters/` | Instrument throughputs (`inputs/filters/Spiderman/`) and the band table `bands.csv`. |
| `inputs/planets/` | The curated planet parameters, `curated_planets.csv`. |
| `inputs/stellar_model/` | NewEra spectral cube and limb-darkening coefficients. |
| `inputs/noise/` | PandExo noise levels and their cache, and the per-planet exceptions. |
| `inputs/exoplanet_archive/` | Download of the archive's PSCompPars table, derived columns, and the snapshot used. |
| `docs/` | Which notebook computes each figure and table of the paper ([`docs/paper_figures.md`](docs/paper_figures.md)), and the image of this README. |
| `tests/` | The test suite (see below). |

## Tests

```bash
pytest                 # about 30 s on 16 cores
pytest -m "not slow"   # without the brute-force check of the fit with redistribution
```

The tests check the irradiance against the closed form of the absorbed power, an adaptive quadrature and a brute-force
integration over the stellar surface; that heat redistribution conserves energy; SPIDERMAN's occultation depth and nightside flux against a
direct integral of the temperature map, the fast light curves against direct SPIDERMAN evaluation, and that a planet inside its star raises an
error instead of ending the process; the noiseless fits against a brute-force χ² grid; that the stored retrievals reproduce their
likelihood; and the structure of the package (its layers, the band and planet registries, the default stellar model). They use a
blackbody star, so they run without the NewEra cube; the tests that need the cube (whether the stored retrievals were run with the
current stellar grid) or PandExo 2026.7 are skipped otherwise.

## Reproducing the retrievals

The paper's nested-sampling retrievals (249 simulated datasets, three models each) are defined in `retrievals/paper_runs/`:

```bash
cd retrievals/paper_runs
python make_runs.py      # writes one run file per planet, seed and setting
./run_all.sh             # runs them side by side on all cores
```

They take about 2,000 core-hours in total and save to `retrievals/dynesty_results/`, which is not part of the repository. The paper's
retrievals use the NewEra star, so they need the spectral cube (see Installation). `retrievals/03_retrieval_series.ipynb` reads them
from there. The three stored retrievals in `retrievals/example_runs/` can be read with `hyperphase.retrieval.storage.load_fits`.

## Limitations

- The planet is a bare rock in instantaneous radiative equilibrium: uniform Bond albedo and emissivity, no thermal inertia, no internal heat.
  Heat transport enters only as the uniform-nightside redistribution $F$ of Kreidberg & Loeb (2016).
- Orbits are circular: the irradiance uses a fixed star–planet distance, and the fast light-curve evaluation times the occultation for $e = 0$.
  A phase curve of a planet with $e \neq 0$ (some archive planets keep their published eccentricity) raises a warning.
- The star's limb darkening is linear for the irradiance. The reflected light is grey and Lambertian, and its penumbral irradiance uses the
  bolometric `u_bol` rather than the limb darkening of the band.
- The planet emits as a blackbody scaled by the band emissivity, without spectral features of the surface.
- Sweeps over the stellar $T_{\rm eff}$ keep the host's $\log g$ and [Fe/H], so the swept stars are hypothetical; their significances shift by a few
  per cent below 3000 K.

## Data sources and acknowledgements

- Planet and stellar parameters: NASA Exoplanet Archive, Planetary Systems Composite Parameters table (doi:10.26133/NEA13), downloaded on
  2026 August 4. This research has made use of the NASA Exoplanet Archive, which is operated by the California Institute of Technology,
  under contract with the National Aeronautics and Space Administration under the Exoplanet Exploration Program. The derived columns and the
  curated parameters in `inputs/planets/curated_planets.csv` are not archive values; each entry cites its source.
- Unconfirmed TESS candidates (`bias_analysis/07_tess_candidates.ipynb`): the TOI table of ExoFOP, downloaded on 2026 October 20, with the
  host stars of the TESS Input Catalog v8 (Stassun et al. 2019). This research has made use of the Exoplanet Follow-up Observation Program
  (ExoFOP; DOI: 10.26134/ExoFOP5) website, which is operated by the California Institute of Technology, under contract with the National
  Aeronautics and Space Administration under the Exoplanet Exploration Program.
- Filter throughputs: SVO Filter Profile Service (Rodrigo et al. 2012, 2020, 2024). This research has made use of the SVO Filter Profile
  Service "Carlos Rodrigo", funded by MCIN/AEI/10.13039/501100011033/ through grant PID2023-146210NB-I00. The MIRI/LRS throughput is taken
  from the Pandeia reference data (Pontoppidan et al. 2016).
- Noise levels: PandExo (Batalha et al. 2017) and Pandeia (Pontoppidan et al. 2016), version 2026.7.
- Stellar spectra: NewEra (Hauschildt et al. 2025, A&A 698, A47).
- Limb darkening: ExoTiC-LD (Grant & Wakeford 2024).
- Light curves: SPIDERMAN (Louden & Kreidberg 2018) through fSPIDERMAN, and batman (Kreidberg 2015). Nested sampling: dynesty (Speagle 2020).
- Irradiance comparison: InstellCa (Sadh & Gavassino 2026), run from a modified copy (see License).

## Citation

If you use this code, please cite the paper above (see also `CITATION.cff`) and the packages it builds on, in particular SPIDERMAN.

## License

MIT, see `LICENSE`. Two files build on the code of others, also under the MIT License. `validation/InstellCa_3.0_exported.py` is a modified
copy of InstellCa, Copyright (c) 2021 Mradumay137; its header lists the changes. `validation/spiderman_numpy2.patch` is a patch to SPIDERMAN
(Louden & Kreidberg 2018), Copyright (c) 2016 Tom Louden, whose `LICENSE.txt` is the MIT License (its `setup.py` names the GNU GPL v3); the
patch's header reproduces that licence.
