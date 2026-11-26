# The paper's figures and tables in this repository

Each figure and table of Spiller, Zieba & Wordsworth (in prep.), with the notebook that computes it. The notebooks draw their figures at
notebook size and set the planet, mode and settings in their first cells. Where these differ from the figure in the paper, the entry says
what to set. § refers to a notebook's numbered sections, a quoted heading to a notebook whose sections are not numbered.

The nested-sampling retrievals (Section 4.3, Appendix E) need data that are not in the repository: they read the 249 retrievals defined in
`retrievals/paper_runs/` (see [Reproducing the retrievals](../README.md#reproducing-the-retrievals)). The observation of the two corner
plots is shipped in `retrievals/example_runs/`.

## Section 2: Hyper-illumination geometry

| Item | Shows | Notebook |
|---|---|---|
| Figure, Sec. 2 | The geometry in a cross section through star and planet | A drawing, not made in code. |
| Figure, Sec. 2.3 | Equilibrium temperature of TOI-2431 b against the angle from the substellar point | [`forward_model/01_irradiance.ipynb`](../forward_model/01_irradiance.ipynb) §1 |
| Figure, Sec. 2.3 | Temperature maps at quadrature and at transit | [`forward_model/01_irradiance.ipynb`](../forward_model/01_irradiance.ipynb) §6 (also the image of the README) |

## Section 3: Methods

| Item | Shows | Notebook |
|---|---|---|
| Table, Sec. 3.1 | Baseline parameters | The `DEFAULT_*` settings of [`hyperphase/defaults.py`](../hyperphase/defaults.py), each with its rationale. |
| Table, Sec. 3.2 | The planets of this study | `PAPER_PLANETS` in [`hyperphase/paper.py`](../hyperphase/paper.py) and the curated parameters in [`inputs/planets/curated_planets.csv`](../inputs/planets/curated_planets.csv), with the published source of each value; the ESM from [`inputs/exoplanet_archive/ExoArchive_customcols.py`](../inputs/exoplanet_archive/ExoArchive_customcols.py) |

## Section 4: Results

| Item | Shows | Notebook |
|---|---|---|
| Figure, Sec. 4.1 | The phase-curve signature: both models, the residuals of the noiseless fits, and the share of each phase bin in $\chi^2_{\min}$ | [`bias_analysis/04_evidence_prediction.ipynb`](../bias_analysis/04_evidence_prediction.ipynb) §1, for the fit without redistribution. The residuals of the fit with redistribution: [`bias_analysis/02_redistribution_bias.ipynb`](../bias_analysis/02_redistribution_bias.ipynb) §2. |
| Figure, Sec. 4.2 | Albedo bias and spurious redistribution in every band, for TOI-2431 b and K2-141 b | [`bias_analysis/03_bias_maps.ipynb`](../bias_analysis/03_bias_maps.ipynb) §2.3 ($A_B$ in each mode, with the joint fits), §2.6 ($F$ in each mode) and §2.7 (NRS1, NRS2 and MIRI/LRS in one fit); set `retrieval_planet` to the planet. The nightside panel: [`forward_model/06_nightside_temperature.ipynb`](../forward_model/06_nightside_temperature.ipynb) §3. |
| Figure, Sec. 4.3 | Retrievals of thirty simulated observations of TOI-2431 b | [`retrievals/03_retrieval_series.ipynb`](../retrievals/03_retrieval_series.ipynb) §7. Needs the paper's retrievals. |
| Figure, Sec. 4.4 | Robustness to nuisance parameters | [`bias_analysis/05_nuisance_robustness.ipynb`](../bias_analysis/05_nuisance_robustness.ipynb) §5 (both fits with the same nuisance parameters) and §6 (the tidal deformation with a uniform prior on $h_2$) |
| Figure, Sec. 4.5 | Bias maps over $a/R_\star$ and the host's $T_{\rm eff}$ in MIRI/LRS | [`bias_analysis/03_bias_maps.ipynb`](../bias_analysis/03_bias_maps.ipynb) §1.1 (without redistribution) and §2.1 (with redistribution), as shipped (`instrument = "MIRI/LRS"`) |
| Table, Sec. 4.5 | Bias, redistribution, nightside temperature and expected evidence of the paper's planets in the two primary modes | [`bias_analysis/03_bias_maps.ipynb`](../bias_analysis/03_bias_maps.ipynb) §1.2, §2.2 and §2.6, run once with `instrument = "JWST_NIRSpec.G395H_joint"` and once with `"MIRI/LRS"`. Expected evidence: [`bias_analysis/04_evidence_prediction.ipynb`](../bias_analysis/04_evidence_prediction.ipynb) §2. Nightside temperatures in MIRI/LRS: [`forward_model/06_nightside_temperature.ipynb`](../forward_model/06_nightside_temperature.ipynb) §2. |

## Section 5: Discussion

| Item | Shows | Notebook |
|---|---|---|
| Table, Sec. 5.1 | The results at the observing time of the JWST phase curves executed so far | Not in a notebook: the fits of the table of Section 4.5, with `n_orbits` set to each visit's exposure time divided by the orbital period. |

## Appendix A: The extended-source irradiance

| Item | Shows | Notebook |
|---|---|---|
| Figure, App. A.1 | Integration geometry: a cross section (a) and the star seen from the surface (b, c) | (a) is a drawing; (b, c) are not in a notebook. The integral itself is `insolation_hyperillum` in [`hyperphase/irradiance.py`](../hyperphase/irradiance.py). |
| Figure, App. A.1 | Edge of the fully illuminated zone and terminator against $a/R_\star$ | [`forward_model/01_irradiance.ipynb`](../forward_model/01_irradiance.ipynb) §3 |
| Table, App. A.5 | Published treatments of the extended-source irradiance | From the literature. |
| Figure, App. A.5 | Comparison with InstellCa and with the small-disk approximation of Kopal (1954) | [`validation/instellca_comparison.ipynb`](../validation/instellca_comparison.ipynb) §6; its §1 runs InstellCa. |

## Appendix E: Nested-sampling retrievals

| Item | Shows | Notebook |
|---|---|---|
| Table, App. E | Retrievals of thirty simulated observations of the two reference planets | [`retrievals/03_retrieval_series.ipynb`](../retrievals/03_retrieval_series.ipynb) §7. Needs the paper's retrievals. |
| Figure, App. E | Posteriors of one simulated observation of TOI-2431 b in MIRI/LRS | [`retrievals/01_retrieval_example.ipynb`](../retrievals/01_retrieval_example.ipynb) §2, as shipped (`results = "TOI-2431b LRS seed44"`, a stored retrieval in `example_runs/`) |

## Appendix G: Bandpass, fixed parameters, and the Roche limit

| Item | Shows | Notebook |
|---|---|---|
| Figure, App. G.2 | Dependence on the true $A_B$, $a/R_\star$, the inclination and $u_{\rm bol}$ | [`bias_analysis/01_albedo_bias.ipynb`](../bias_analysis/01_albedo_bias.ipynb) §3–5 (without redistribution) and [`bias_analysis/02_redistribution_bias.ipynb`](../bias_analysis/02_redistribution_bias.ipynb) §3–5 (with redistribution). The $a/R_\star$ column, with the planet moved along its host's Kepler relation, is not in a notebook; the maps of Section 4.5 show the dependence on $a/R_\star$ for the reference geometry. |