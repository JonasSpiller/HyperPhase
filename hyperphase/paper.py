"""The configuration of the paper (Spiller, Zieba & Wordsworth, in prep.): its planet sample, the published uncertainties of its two reference
planets, and the fit list of its per-instrument analyses.

The general modules do not import this file; the notebooks that reproduce the paper do."""
from hyperphase.bands import JWST_BANDS


# The planets of the paper (its table of targets): the confirmed rocky planets (Rp < targets.ROCKY_MAX_RADE) with a/R* < 3 and ESM > 4 (Kempton et
# al. 2018; the archive's value), ordered by a/R* (the curated values). The a/R* cut keeps the stellar half-angle above 19.5 deg; the ESM cut keeps
# the planets whose A_B the per-planet figures can constrain at all (K2-157 b, K2-223 b, KOI-1843.03, K2-183 b, K2-301 c, Kepler-990 c fail it with
# ESM 0.7-1.9; K2-106 b with 3.0). Every member is in the curated table (inputs/planets/curated_planets.csv), so the sample does not move with
# archive updates. Dropped on the a/R* cut: TOI-6255 b, TOI-431 b, GJ 367 b (3.10-3.35, ESM 15-22; three of the five GO 8864 targets) and TOI-6324 b
# (4.29). targets.best_hi_planets is NOT this set: it ranks by (1 - Rp/R*)/(a/R*) and has no brightness cut. 55 Cnc e is added outside the rule as
# the common reference of the hyper-illumination literature (Carter et al. 2024, Sadh & Gavassino 2026, Sadh 2026) and of the lava-world
# observations: it fails both the radius cut (1.87 R_E) and the a/R* cut (3.52). Its host (K = 4.0) saturates NIRSpec/G395H, so it has no noise
# level there (inputs/noise/unobservable.csv): MIRI/LRS only, and that only over 6.7-10.5 um, since it saturates the blue end of the band too
# (inputs/noise/partial_bands.csv).
PAPER_PLANETS = ("TOI-2431 b", "TOI-2260 b", "K2-141 b", "K2-131 b", "TOI-561 b", "Kepler-78 b", "TOI-1444 b", "TOI-1798.02", "HD 80653 b", "55 Cnc e")
PAPER_OUTSIDE_RULE = ("55 Cnc e",)   # the members of PAPER_PLANETS that the selection rule does not pick

# The fit list of the paper's per-instrument analyses: every JWST band singly plus one joint fit of the two NIRSpec/G395H arrays (a tuple inside the
# list = one joint fit of those detectors: shared A_B / F, a baseline per detector). A list, not a tuple, so a tuple always means a joint fit.
JWST_RETRIEVALS = JWST_BANDS[:2] + [tuple(JWST_BANDS[:2])] + JWST_BANDS[2:]

# Published 1-sigma uncertainties of the two reference planets, for every analysis that does not hold the geometry fixed: the nuisance fits
# (bias.nuisance, bias_analysis/05_nuisance_robustness.ipynb; the paper's nuisance-robustness figure, Sec. 4.4) and the free-geometry retrievals
# (retrievals/paper_runs/). GEOMETRY_PARAMS: the geometry as each planet's source FITTED it, in the column names of bias.nuisance.nuisance_problem.
# Tas et al. fit the impact parameter b and the stellar density of TOI-2431 b and derive i = arccos(b / (a/R*)), so its geometry is (a/R* at fixed
# b, Rp/R*, b); Malavolta et al. fit a/R* and i of K2-141 b directly. GEOMETRY_SIGMA: the Gaussian priors on these parameters (1 sigma, centred on
# the curated values, independent: correlations ignored). "a_rs" also serves "a_rs|b". For TOI-2431 b, incl is derived (mostly the errors of b and
# a/R* combined) and only sets the sin^2 i error of the ellipsoidal prior.
GEOMETRY_PARAMS = {"TOI-2431 b": ("a_rs|b", "rp_rs", "b"), "K2-141 b": ("a_rs", "rp_rs", "incl")}
GEOMETRY_SIGMA = {
    "TOI-2431 b": dict(a_rs=0.04,       # Tas et al. 2026 (A&A 707, A238): a/R* = 2.05 +- 0.04 from the fitted stellar density (2.08 adopted from a and R*)
                       rp_rs=0.00033,   # Tas et al. 2026 Tab. 2: 0.02131 +0.00033/-0.00032
                       b=0.032,         # Tas et al. 2026 Tab. 2: b = 0.573 +0.031/-0.033
                       incl=1.0),       # Tas et al. 2026 Tab. 2 (FCO fit): i = 74 +1.01/-0.96 deg
    "K2-141 b":   dict(a_rs=0.057,      # Malavolta et al. 2018 (AJ 155, 107) Tab. 3: 2.292 +0.053/-0.060, symmetrised
                       rp_rs=0.00046,   # Malavolta et al. 2018 Tab. 3: 0.02037 +- 0.00046
                       incl=3.15),      # Malavolta et al. 2018 Tab. 3: i = 86.3 +2.7/-3.6 deg (> 82.6, posterior peaked at 90), symmetrised; a dynesty run truncates it at 90 deg, the linear fits do not
}

# PLANET_MASSES: M_p [M_Earth] and its 1 sigma, M_star [M_sun] and its 1 sigma. They set the expected amplitudes of the ellipsoidal terms
# (bias.nuisance.ellipsoidal_prior, deformation_prior).
PLANET_MASSES = {"TOI-2431 b": (6.2, 1.6, 0.661, 0.0225),    # Tas et al. 2026 (A&A 707, A238) Tab. 1-2 (FCO fit; M_star +0.021/-0.024 symmetrised)
                 "K2-141 b":   (5.08, 0.41, 0.708, 0.028)}   # Malavolta et al. 2018 (AJ 155, 107) Tab. 2-3

# Fluid Love number h2 of the reference planets' tidal deformation (bias.nuisance.deformation_limits; the row "+ planetary deformation" of the
# paper's nuisance-robustness figure, Sec. 4.4): 2 for rocky planets (Correia 2014, A&A 570, L5, after Yoder 1995; Earth's moment of inertia in his
# Darwin-Radau relation gives 1.93). The range a fit may use is bias.nuisance.DEFORMATION_H2_RANGE.
DEFORMATION_H2_TRUE = 2.
