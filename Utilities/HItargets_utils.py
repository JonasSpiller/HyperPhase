import os, sys, re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from astropy import constants as const
import matplotlib.pyplot as plt

###########################################################################################################################
#####
#####   Baseline Settings
#####
###########################################################################################################################

# The one set of model settings of the repo: the default of every function argument and FittingConfig field, and what every notebook imports.
# Only sweeps and comparisons with other codes deviate from them, on purpose.

# Bond albedo A_B, which sets the temperature map: a bare-rock value at the upper end of what airless rocky planets show.
# Those literature values neglect hyperillumination, so for the lava planets they are likely somewhat low.
DEFAULT_A_B = 0.2

# Band-integrated albedo A of the reflected light (not a geometric albedo), independent of A_B: molten and glassy silicates reflect weakly (Essack et
# al. 2020).
DEFAULT_A = 0.1

# Bolometric emissivity emissivity_bol of the thermal emission, which sets the temperature map together with A_B: emissivity_bol sigma T^4 = (1-A_B)
# S. Only (1-A_B)/emissivity_bol enters, so emissivity_bol is exactly degenerate with A_B: with emissivity_bol fixed to 1, A_B is really the effective
# albedo 1 - (1-A_B)/emissivity_bol. A_B (weighted by the stellar spectrum) and emissivity_bol (weighted by the planet's thermal spectrum) sample
# different wavelengths and are independent; 1 assumes a surface that is black in the thermal infrared.
DEFAULT_EMISSIVITY_BOL = 1.

# Band-integrated emissivity of the thermal emission in the observing band (the thermal counterpart of A): the thermal planet flux is linear in it.
# Deliberately independent of A, although Kirchhoff's law ties them in the same band (emissivity = 1 - A for an opaque surface): A is a free
# phenomenological reflectance. Coupling the two (and emissivity_bol) through spectral emissivity data of hot silicates is a possible extension.
DEFAULT_EMISSIVITY = 1.

# Bolometric linear limb darkening u_bol of the irradiance: a model-atmosphere value instead of the grey Eddington value of 0.6.
# For TOI-2431 b, the primary target, the bolometric profile gives 0.49 (PHOENIX) and 0.50 (MPS-ATLAS) (stellar_model/LimbDarkening.ipynb).
DEFAULT_U_BOL = 0.5

# Quadratic limb darkening (u1, u2) of the batman transit. The transit is identical in both irradiation models and cancels exactly in every
# comparison, so the value only needs to be plausible: the median of ExoTiC-LD fits over the hosts of the twelve PAPER_PLANETS of 2026-11-02, in the
# band throughputs of the forward model (stellar_model/LimbDarkening.ipynb). Not re-derived for the sample of 2026-11-08 (hosts of 4100-5960 K, as
# before): the value does not enter any result.
DEFAULT_U = (0.09, 0.14)

# Orbits observed and phase-folded. The periods are only 4-7 h, so five orbits are roughly 1-1.5 days of JWST time (about 27 h for TOI-2431 b, 34 h
# for K2-141 b): ambitious, but within the range of a large program.
DEFAULT_N_ORBITS = 5

# Bins per orbit of the phase-folded light curve: 12 min for TOI-2431 b, 15 min for K2-141 b. The width is derived from the period, so the bins tile
# the orbit exactly.
DEFAULT_N_BINS = 27

# Reference Rp/R* of the (a/R*, T_eff) bias maps (simple_bias_analysis.reference_planet; Bias_Maps.ipynb, HypotheticalExtremePlanet.ipynb,
# For_Paper/Plots_Sec4.5.ipynb): the median of the
# PAPER_PLANETS (0.0158 for the nine of the selection rule, 0.0160 with 55 Cnc e; they span 0.012-0.021). The bias barely depends on it, because the
# mismatch and the response to A_B both scale with the planet's flux: between 0.0075 and 0.03 it changes by less than 3 % (A_B-only fit) and 5 % (with
# F) (For_Paper/Plots_Sec4.5.ipynb, res-maps-code). Map outputs from before 2026-11-14 used 0.015.
DEFAULT_MAP_RP_RS = 0.016

# Parameter ranges, shared by the least-squares bias analysis (simple_bias_analysis) and the nested sampling (dynesty_utils, run_dynesty.py).
# A_B_BOUNDS, F_BOUNDS: the physical ranges of A_B and of the Kreidberg & Loeb (2016) redistribution F. They are the uniform priors of every dynesty
# fit, and their flat-prior std defines an "unconstrained" fit (a Fisher sigma at or above it: the data add nothing to the physical range). F_BOUNDS
# is also the least-squares range of F.
A_B_BOUNDS = (0., 1.)
F_BOUNDS = (0., 0.5)
# A_B_FIT_BOUNDS: the least-squares range of A_B, the default of every fit in simple_bias_analysis. It extends below 0 so that the bias of the
# point-source model stays defined where it needs more absorbed energy than the star delivers to a point (A_B,fit < 0, which an observer's physical
# prior would cut off at 0). Only (1-A_B) enters the temperature, so a negative A_B is a valid model evaluation. The lower end is never reached in
# practice (the deepest fit found is about -1.2: true A_B = 0 near the contact limit); a fit there, or at A_B = 1, still counts as railed. Pass
# bounds=A_B_BOUNDS to fit within the physical range instead (railed at A_B = 0, as before 2026-11-08).
A_B_FIT_BOUNDS = (-2., 1.)
# Uniform prior of the free baseline offset of the dynesty fits [ppm]; the least-squares fits profile the offset out analytically.
OFFSET_BOUNDS = (-50., 50.)

# Imported only AFTER the baseline settings: hyperillumination_utils imports them from here.
from hyperillumination_utils import PlanetParams

###########################################################################################################################
#####
#####   Planetary Parameters
#####
###########################################################################################################################


# ==========================================================================
# SETTINGS
# ==========================================================================

# Aggregate table from ExoArchive_customcols.py. The newest match is used.
INPUT_GLOB = "NASAExoArchive_*_aggregate_short.csv"

# Reject companions comparable in size to their star (white dwarf systems, unconverged grazing fits).
MAX_R1_OVER_R2 = 0.3

# Rocky planets: Rp < ROCKY_MAX_RADE Earth radii. The cut is on the ABSOLUTE radius pl_rade, not on Rp/Rs: a giant around an evolved star has a small
# ratio (Kepler-91 b, Rp/Rs = 0.022 but 15.3 R_Earth) and would otherwise pass. A planet without pl_rade is dropped, as it cannot be confirmed rocky.
# Single definition, shared by best_hi_planets and simple_bias_analysis.rocky_sample / overlay_planets.
ROCKY_MAX_RADE = 1.7

# The planets of the paper (tab:targets): the confirmed rocky planets (Rp < ROCKY_MAX_RADE) with a/R* < 3 and ESM > 4 (Kempton et al. 2018; the
# archive's value), ordered by a/R* (the CUSTOM values, 2026-11-08). The a/R* cut keeps the stellar half-angle above 19.5 deg; the ESM cut keeps the
# planets whose A_B the per-planet figures can constrain at all (K2-157 b, K2-223 b, KOI-1843.03, K2-183 b, K2-301 c, Kepler-990 c fail it with ESM
# 0.7-1.9; K2-106 b with 3.0). Every member has a CUSTOM entry, so the sample does not move with archive updates. Dropped on the a/R* cut: TOI-6255 b,
# TOI-431 b, GJ 367 b (3.10-3.35, ESM 15-22; three of the five GO 8864 targets) and TOI-6324 b (4.29). best_hi_planets is NOT this set: it ranks by (1
# - Rp/R*)/(a/R*) and has no brightness cut. 55 Cnc e (2026-11-18) is added outside the rule as the common reference of the hyper-illumination
# literature (Carter et al. 2024, Sadh & Gavassino 2026, Sadh 2026) and of the lava-world observations: it fails both the radius cut (1.87 R_E) and
# the a/R* cut (3.52). Its host (K = 4.0) saturates NIRSpec/G395H, so it has no noise level there (UNOBSERVABLE): MIRI/LRS only, and that only over
# 6.7-10.5 um, since it saturates the blue end of the band too (PARTIAL_BANDS).
PAPER_PLANETS = ("TOI-2431 b", "TOI-2260 b", "K2-141 b", "K2-131 b", "TOI-561 b", "Kepler-78 b", "TOI-1444 b", "TOI-1798.02", "HD 80653 b", "55 Cnc e")
PAPER_OUTSIDE_RULE = ("55 Cnc e",)   # the members of PAPER_PLANETS that the selection rule does not pick (For_Paper/Numbers_Sec3.ipynb checks the others against the archive selection)

# The Roche limits drawn on every (a/R*, T_eff) map (simple_bias_analysis.draw_roche_frontier, main-sequence hosts) and used in
# HypotheticalExtremePlanet.ipynb, as {label: (iron mass fraction, rho_0 / rho_mean)} of a rock/iron planet of radius ROCHE_R_P. The radius and iron
# fraction give the mean density (astro_utils.rocky_mean_density, Fortney et al. 2007 eq. 8), and with rho_0 / rho_mean the Roche period
# (astro_utils.roche_period_h, Rappaport et al. 2013 eq. 5); simple_bias_analysis.roche_periods evaluates them.
# - "Earth-like": Earth's iron fraction 0.33 and Earth's central concentration 13.1 / 5.51 = 2.37 (PREM). The wall for a typical rocky planet.
# - "iron-rich": 70 % iron, the most that collisional mantle stripping leaves in planets above ~1 M_E (Marcus et al. 2010, as used by Rappaport et al.
#   2013), with rho_0 / rho_mean = 2.5, the top of Rappaport's range 1-2.5 (a larger ratio survives closer in, so this is the permissive end). The
#   innermost line: left of it no rocky planet of radius ROCHE_R_P survives around a main-sequence host (hatched on the maps).
# ROCHE_R_P = 1.5 R_E, typical of PAPER_PLANETS (TOI-2431 b and K2-141 b are 1.5 R_E). Compression makes larger planets denser and shifts the limits
# inwards: the Earth-like P_R is ~4.7 / ~4.0 / ~3.7 h at 1.0 / 1.5 / 1.7 R_E, i.e. a/R* of the line -5 / +12 % at 1.7 / 1.0 R_E against 1.5 R_E.
ROCHE_R_P = 1.5   # [R_E]
ROCHE_CASES = {"Earth-like": (0.33, 2.37), "iron-rich (70 % Fe)": (0.70, 2.5)}

# Argument of periastron where the archive has none. 90 deg is the transit convention and is irrelevant at e = 0.
DEFAULT_W_DEG = 90.0

AU_IN_R_SUN = float(const.au / const.R_sun)

# ==========================================================================
# Custom parameters
# ==========================================================================

# Planets with hand-curated parameters, each from a PUBLISHED source so that every value in the paper's target table can be cited. The rule: a/R*,
# Rp/R* and i are the published fit; where no paper fits a/R*, the ratio of the paper's own tabulated a and R* (TOI-2431 b, TOI-2260 b, Kepler-1323 b)
# or of its tabulated stellar density (TOI-1798.02); an inclination without a published value is acos(b / (a/R*)) from the published b, or 90 deg
# where nothing is published (stated per entry); T14 is published where it exists and otherwise follows from the published geometry (it only enters
# the PandExo in-transit integration count). Stellar parameters are published spectroscopic values or a named catalogue (TICv8, Fulton & Petigura
# 2018, Berger et al. 2020). r2 = 1, so R is a/R*. T14 in hours. a_AU = (a/R*) x R* of the adopted star: it only feeds the light-travel-time
# correction, and it lets apply_custom recover R* and Rp for the archive frame. Close-in rocky planets without a citable, physically consistent
# geometry are deliberately NOT here: Kepler-780 b, -1340 b, -1368 b, -1547 b, -1566 b, -879 c. CUSTOM takes precedence over the archive everywhere:
# hi_planets_by_name returns these objects, and apply_custom writes them into every archive frame.
CUSTOM = {
    # ---- the closest-in rocky planets with a well-constrained geometry and a bright host (the paper's sample is PAPER_PLANETS) ---------------------
    "TOI-2431 b":       PlanetParams(name="TOI-2431 b", period=0.22419578, incl=74.0, ecc=0.0, w=90.0, r1=0.02131, r2=1.0, R=2.08, T2_eff=4109.0, a_AU=0.00630, T14=0.748, logg=4.68, feh=-0.02),   # Tas et al. 2026 (A&A 707, A238) Tab. 1-2 (FCO fit): P, Rp/R*, i = 74, b = 0.573, a = 0.0063 AU, R* = 0.651, star. No a/R* in the paper: 2.08 = a / R* (Coy et al. 2026 Tab. 5 print 2.08 +- 0.05; the fitted rho* = 3.25 would give 2.05). T14 from the geometry (none published; ExoFOP's 0.29 h is wrong)
    "TOI-2260 b":       PlanetParams(name="TOI-2260 b", period=0.3524728, incl=69.7, ecc=0.0, w=90.0, r1=0.0158, r2=1.0, R=2.22, T2_eff=5534.0, a_AU=0.0097, T14=0.87, logg=4.62, feh=0.22),       # Giacalone et al. 2022 (AJ 163, 99) Tab. 2-3: P, b = 0.77 +- 0.04, a = 0.0097 AU, Rp = 1.62, R* = 0.94, M* = 0.99, star. No fitted a/R*, Rp/R*, i: a/R* = a / R*, Rp/R* = Rp / R*, i = acos(b / (a/R*)). T14 from the geometry. No RV mass; the archive's i = 90 is a placeholder
    "K2-141 b":         PlanetParams(name="K2-141 b", period=0.2803244, incl=86.3, ecc=0.0, w=90.0, r1=0.02037, r2=1.0, R=2.292, T2_eff=4599.0, a_AU=0.00726, T14=0.94, logg=4.62, feh=-0.06),     # Malavolta et al. 2018 (AJ 155, 107) Tab. 2-3 (R* 0.681); T14 Barragán et al. 2018. Zieba et al. 2022 (joint K2+Spitzer) get a/R* = 2.365, Rp/R* = 0.02061, i = 85.2
    "TOI-561 b":        PlanetParams(name="TOI-561 b", period=0.4465697, incl=87.0, ecc=0.0, w=90.0, r1=0.01519, r2=1.0, R=2.683, T2_eff=5372.0, a_AU=0.01052, T14=1.308, logg=4.50, feh=-0.40),   # Piotto et al. 2024 (MNRAS 535, 2763) Tab. 2; star Lacedelli et al. 2022 (MNRAS 511, 4551) Tab. 1 (R* 0.843): metal-poor, alpha-enhanced. JWST G395H gives Rp/R* = 0.0146 (Boucher et al. 2026)
    "TOI-6255 b":       PlanetParams(name="TOI-6255 b", period=0.23818244, incl=74.4, ecc=0.0, w=90.0, r1=0.0267, r2=1.0, R=3.10, T2_eff=3421.0, a_AU=0.00533, T14=0.36, logg=4.85, feh=-0.14),    # Dai et al. 2024 (AJ 168, 101) Tab. 1-2 (R* 0.370); b = 0.84. T14 from the geometry (none published; ExoFOP's 0.48 h is not from the fit). P/P_Roche = 1.13: ellipsoidal variations of tens of ppm expected
    "TOI-6324 b":       PlanetParams(name="TOI-6324 b", period=0.2792210, incl=82.36, ecc=0.0, w=90.0, r1=0.03217, r2=1.0, R=4.29, T2_eff=3247.0, a_AU=0.00585, T14=0.435, logg=4.93, feh=-0.32),  # geometry Gomez Barrientos et al. 2025 (AJ 170, 148) Tab. 3 (a/R* 4.29 +0.61/-0.69, b 0.57, i 82.36 +- 2.94, TESS Rp/R*); star and P Lee et al. 2025 (ApJL 983, L36) Tab. 1 (R* 0.293). Lee's own fit (a/R* 3.17, i 74.6) rests on a stellar density mislabelled by the factor rho_sun (10.7 solar = 15.1 g/cm^3). T14 from the geometry
    "TOI-431 b":        PlanetParams(name="TOI-431 b", period=0.490054945, incl=84.7, ecc=0.0, w=90.0, r1=0.0161, r2=1.0, R=3.35, T2_eff=4850.0, a_AU=0.01139, T14=1.10, logg=4.60, feh=0.20),     # Smith et al. 2026 (arXiv:2609.03029) Tab. 1-2, joint TESS+RV fit with Osborn's density prior; star Osborn et al. 2021 (MNRAS 507, 2782) Tab. 2 (R* 0.731). The archive's a/R* = 3.42 +- 1.71 (Kokori et al. 2023) is a catalogue value, not a fit
    # ---- 55 Cnc e: in PAPER_PLANETS outside the selection rule (common reference; 2026-11-18) ------------------------------------------------------
    "55 Cnc e":         PlanetParams(name="55 Cnc e", period=0.7365474, incl=83.59, ecc=0.0, w=90.0, r1=0.0182, r2=1.0, R=3.52, T2_eff=5172.0, a_AU=0.01544, T14=1.532, logg=4.43, feh=0.35),          # Bourrier et al. 2018 (A&A 619, A1) Tab. 4: P, a/R* 3.52 +- 0.01, Rp/R* 0.0182 +- 0.0002 (HST/STIS), i 83.59 +0.47/-0.44 (b = 0.39), a 0.01544 AU (e = 0.05 +- 0.03 set to 0); star Tab. 1 (R* 0.943 von Braun et al. 2011; T_eff, log g, [Fe/H] Yee et al. 2017). Rp = 1.87 R_E > ROCKY_MAX_RADE. T14 from the geometry (none in Bourrier)
    "GJ 367 b":         PlanetParams(name="GJ 367 b", period=0.3219225, incl=79.89, ecc=0.0, w=90.0, r1=0.01399, r2=1.0, R=3.327, T2_eff=3522.0, a_AU=0.00709, T14=0.629, logg=4.776, feh=-0.01),  # Goffo et al. 2023 (ApJL 955, L3) Tab. 1-2 (R* 0.458; e = 0.06 +0.07/-0.04 set to 0; b = 0.584); a/R* = 3.327 as adopted from Goffo by Zhang et al. 2024 (ApJL 961, L44) Sect. 2 (Lee et al. 2026, arXiv:2606.18355: 3.33 +- 0.17, b = 0.68, Rp/R* = 0.0148)
    # ---- other close-in rocky planets: well-constrained geometry and host --------------------------------------------------------------------------
    "K2-157 b":         PlanetParams(name="K2-157 b", period=0.3652575, incl=89.2, ecc=0.0, w=90.0, r1=0.00996, r2=1.0, R=2.405, T2_eff=5334.0, a_AU=0.00962, T14=1.161, logg=4.45, feh=0.02),        # Castro-Gonzalez et al. 2025 (A&A 699, A344) Tab. B.5, Tab. 1 (R* 0.860); a/R* from the stellar priors; i = 89.2 is a flat-prior median (b = 0.03 +- 0.38)
    "Kepler-78 b":      PlanetParams(name="Kepler-78 b", period=0.35500745, incl=75.2, ecc=0.0, w=90.0, r1=0.01473, r2=1.0, R=2.7, T2_eff=5058.0, a_AU=0.00939, T14=0.813, logg=4.55, feh=-0.18),     # Howard et al. 2013 (Nature 503, 381) Tab. 1: a/R* 2.7 +- 0.2, i 75.2 +2.6/-2.1, depth 217 ppm (Rp/R* = sqrt), T14 0.813; R* = 0.7475 Dai et al. 2019 (ApJ 883, 79) Tab. 1 (Gaia); T_eff, log g, [Fe/H] Pepe et al. 2013 (Nature 503, 377). No short-cadence data
    "K2-131 b":         PlanetParams(name="K2-131 b", period=0.3693038, incl=85.0, ecc=0.0, w=90.0, r1=0.0204, r2=1.0, R=2.66, T2_eff=5120.0, a_AU=0.00940, T14=1.027, logg=4.62, feh=-0.04),         # Dai et al. 2017 (AJ 154, 226) Tab. 7: a/R* 2.66 +0.36/-0.18, i 85 +10/-9, Rp/R* 0.0204 (Adams et al. 2021 give 0.0172: EVEREST depth 29 % shallower), Tab. 5 log g; T_eff, [Fe/H], R* = 0.7595 Bonomo et al. 2023 Tab. 1; T14 Singh et al. 2022 Tab. 4
    "TOI-1444 b":       PlanetParams(name="TOI-1444 b", period=0.4702694, incl=82.0, ecc=0.0, w=90.0, r1=0.01410, r2=1.0, R=2.73, T2_eff=5430.0, a_AU=0.01152, T14=1.283, logg=4.49, feh=0.13),        # Dai et al. 2021 (AJ 162, 62) Tab. 1-2 (R* 0.907; b = 0.38 +- 0.14); T14 MacDougall et al. 2023 as in Polanski et al. 2024 (their Rp/R* 0.01428)
    "K2-223 b":         PlanetParams(name="K2-223 b", period=0.505608, incl=78.1, ecc=0.0, w=90.0, r1=0.00687, r2=1.0, R=2.79, T2_eff=5858.0, a_AU=0.01362, T14=1.205, logg=4.56, feh=0.18),           # Jankowski et al. 2026 (arXiv:2609.01721) Tab. C.2, Tab. 1 (R* 1.05 +- 0.10); b = 0.57 +0.15/-0.21; Rp/R* 15 % below Adams/Mayo/Livingston (0.0080-0.0083); SNR 10, sub-Earth
    "EPIC 220674823 b": PlanetParams(name="EPIC 220674823 b", period=0.57130782, incl=88.5, ecc=0.0, w=90.0, r1=0.0148, r2=1.0, R=2.858, T2_eff=5532.0, a_AU=0.01320, T14=1.599, logg=4.40, feh=0.10),  # = K2-106 b. Palethorpe et al. 2026 (arXiv:2603.14552) Tab. 5, Tab. 1 (R* 0.993); log g Guenther et al. 2024. Rp/R* 2 sigma below Guenther (0.0155), Adams (0.0157), Rodriguez-Martinez (0.0160): Rp 1.61 vs 1.68-1.73
    "TOI-1798.02":      PlanetParams(name="TOI-1798.02", period=0.4378146, incl=83.7, ecc=0.0, w=90.0, r1=0.01622, r2=1.0, R=2.905, T2_eff=5106.0, a_AU=0.01069, T14=1.285, logg=4.572, feh=0.073),    # MacDougall et al. 2023 (AJ 166, 33) star (R* 0.791, rho* 2.42 g/cm^3) and transit posteriors (Rp/R*, b = 0.32 +0.31/-0.22, T14) as in Polanski et al. 2024 (ApJS 272, 32). No fitted a/R* exists: 2.905 from rho* and P; i = acos(b / (a/R*)). Confirmed only via the RV mass
    "HD 80653 b":       PlanetParams(name="HD 80653 b", period=0.71957926, incl=81.7, ecc=0.0, w=90.0, r1=0.01242, r2=1.0, R=2.973, T2_eff=5959.0, a_AU=0.01658, T14=1.725, logg=4.342, feh=0.255),    # = K2-312 b. Naponiello et al. 2026 (arXiv:2607.19325) Tab. 1-2 (R* 1.199), joint K2+TESS+HARPS-N; Frustagli et al. 2020 within 1 sigma. The archive's T_eff = 5838 matches no paper
    "Kepler-407 b":     PlanetParams(name="Kepler-407 b", period=0.669310092, incl=87.19, ecc=0.0, w=90.0, r1=0.01034, r2=1.0, R=3.193, T2_eff=5487.0, a_AU=0.01544, T14=1.627, logg=4.30, feh=0.35),  # transit Kepler Q1-Q17 DR25 KOI table (Thompson et al. 2018; a/R* 3.193 +- 0.093, consistent with Kepler III from the star, 3.09-3.22); star Brinkman et al. 2025 (arXiv:2410.00213) Tab. 1 (R* 1.04). Kepler eclipse 6 +- 2 ppm (Singh et al. 2022)
    "WASP-47 e":        PlanetParams(name="WASP-47 e", period=0.789592, incl=85.98, ecc=0.0, w=90.0, r1=0.01461, r2=1.0, R=3.205, T2_eff=5552.0, a_AU=0.01695, T14=1.899, logg=4.344, feh=0.38),       # Vanderburg et al. 2017 (AJ 154, 237) Tab. 3 (R* 1.137), rho* free (pinned by planets b, d). Rp = 1.81 +- 0.03 R_Earth > ROCKY_MAX_RADE, rho_p 6.3-8.1 g/cm^3. The archive's 3.112 is Crossfield's a over Nascimbeni's R*
    # ---- other close-in rocky planets, less well constrained: published fits of lower quality (KOI pipeline / Adams et al. 2021 without density
    # priors), faint hosts, statistical validation ----
    "KOI-1843.03":      PlanetParams(name="KOI-1843.03", period=0.1768913, incl=72.0, ecc=0.0, w=90.0, r1=0.01225, r2=1.0, R=1.9, T2_eff=3584.0, a_AU=0.00398, T14=0.65, logg=4.80, feh=0.0),          # = Kepler-974 c. Rappaport et al. 2013 (ApJL 773, L15) Tab. 1: a/R* 1.9 +0.3/-0.5, i 72 +12/-20, depth 150 ppm (Rp/R* = sqrt), T14 0.65; star as adopted there (Dressing & Charbonneau 2013: R* 0.45, T_eff 3584, log g 4.80). 30-min cadence only; Gaia-era star (TICv8, R* 0.538) would give a/R* 2.00. Tidally distorted (axis ratio >= 1.3)
    "K2-183 b":         PlanetParams(name="K2-183 b", period=0.469285, incl=90.0, ecc=0.0, w=90.0, r1=0.0117, r2=1.0, R=2.5, T2_eff=5519.0, a_AU=0.01128, T14=1.541, logg=4.43, feh=0.09),             # Adams et al. 2021 (PSJ 2, 152) Tab. 2: a/R* 2.5 +0.3/-0.4, i 90.1 +20/-10, Rp/R* 0.0117 (free fit, no density prior); T14 Kruse et al. 2019; star Livingston et al. 2018b (AJ 156, 277; Gaia, R* 0.97: Adams used Mayo's 0.87). Kepler III from the star gives 2.56
    "EPIC 206042996 c": PlanetParams(name="EPIC 206042996 c", period=0.354884, incl=89.9, ecc=0.0, w=90.0, r1=0.0223, r2=1.0, R=2.6, T2_eff=4055.0, a_AU=0.00815, T14=1.07, logg=4.58, feh=0.0),       # = K2-301 c. Adams et al. 2021 Tab. 2: a/R* 2.6 +- 0.5, i 89.9 +20/-10, Rp/R* 0.0223; T14 Vanderburg et al. 2016 (b = 0.25); star TICv8 (Stassun et al. 2019; R* 0.674, the EPIC 0.50 Adams used is wrong, so Rp ~ 1.63 not 1.2). Kp 16, validated only via the multiplicity boost, no imaging
    "Kepler-990 c":     PlanetParams(name="Kepler-990 c", period=0.53835407, incl=82.87, ecc=0.0, w=90.0, r1=0.01264, r2=1.0, R=2.693, T2_eff=5577.0, a_AU=0.01202, T14=1.5045, logg=4.40, feh=-0.11), # transit Kepler DR25 KOI table (a/R* 2.693 +- 0.554, consistent with Kepler III from the star, 2.76); star Fulton & Petigura 2018 (AJ 156, 264) Tab. 2 (R* 0.960), log g TICv8. Ks 12.9
    "EPIC 220554210 c": PlanetParams(name="EPIC 220554210 c", period=0.705310, incl=90.0, ecc=0.0, w=90.0, r1=0.0146, r2=1.0, R=3.0, T2_eff=5523.0, a_AU=0.01311, T14=1.704, logg=4.47, feh=0.15),     # = K2-282 c. Adams et al. 2021 Tab. 2: a/R* 3.0 +0.3/-0.5, i 91.3 +- 10 (= 88.7; 90 adopted); Rp/R* 0.0146 Heller et al. 2019 (A&A 627, A66) Tab. 1 (b = 0.25); T14 Kruse et al. 2019; star Livingston et al. 2018b (spectroscopic, R* 0.94). Planet b's density and the star both favour a/R* ~ 3.4-3.5
    "Kepler-1323 b":    PlanetParams(name="Kepler-1323 b", period=0.929906166, incl=88.59, ecc=0.0, w=90.0, r1=0.00956, r2=1.0, R=3.37, T2_eff=6063.0, a_AU=0.01911, T14=1.743, logg=4.354, feh=0.00), # a/R* = a / R* of Fulton & Petigura 2018 Tab. 2-3 (a 0.01911 AU, R* 1.221; also T_eff, [Fe/H]); the DR25 KOI fit (4.14 +- 0.15) is the b = 0 branch of the degeneracy and inconsistent with the star; Rp/R*, i, T14 DR25 KOI; log g Berger et al. 2020. Note: i = 88.59 with a/R* 3.37 overpredicts the measured T14 (2.0 vs 1.74 h). SNR 80, Ks 11.1
    "EPIC 201427007 b": PlanetParams(name="EPIC 201427007 b", period=0.72091, incl=88.5, ecc=0.0, w=90.0, r1=0.0142, r2=1.0, R=3.1, T2_eff=5383.0, a_AU=0.01567, T14=1.83, logg=4.33, feh=0.31),       # = K2-359 b. Adams et al. 2021 Tab. 2 (only fit): a/R* 3.1 +0.3/-0.6, i 88.5 +- 10, Rp/R* 0.0142; star TICv8 (R* 1.087, not EPIC 0.94). T14 from the geometry (none published). Rp ~ 1.68 (rocky cut not secure); Kp 14.2, no follow-up
    "EPIC 212624936 b": PlanetParams(name="EPIC 212624936 b", period=0.578657, incl=89.7, ecc=0.0, w=90.0, r1=0.0114, r2=1.0, R=3.0, T2_eff=5627.0, a_AU=0.01273, T14=1.52, logg=4.52, feh=-0.39),     # = K2-361 b. Adams et al. 2021 Tab. 2 (only fit; FPP 0.004, SNR 13): a/R* 3.0 +0.4/-0.6, i 90.3 +- 10 (= 89.7), Rp/R* 0.0114; star TICv8 (R* 0.9125); [Fe/H] is the EPIC value (+- 0.25). T14 from the geometry (none published). No follow-up
}


# Field overrides for ARCHIVE planets (a planet in CUSTOM never reaches them). T14 in HOURS, matching PlanetParams.
OVERRIDES = {
}

# Published 1-sigma uncertainties of the two reference planets, for every analysis that does not hold the geometry fixed: the nuisance fits of
# simple_bias_analysis (bias_analysis/Nuisance_Robustness.ipynb), fig:res_nuisance (paper_plot_utils.nuisance_robustness) and the free-geometry
# dynesty runs. Defined once, here. GEOMETRY_PARAMS: the geometry as each planet's source FITTED it, in the column names of
# simple_bias_analysis.nuisance_problem. Tas et al. fit the impact parameter b and the stellar density of TOI-2431 b and derive i = arccos(b /
# (a/R*)), so its geometry is (a/R* at fixed b, Rp/R*, b); Malavolta et al. fit a/R* and i of K2-141 b directly. GEOMETRY_SIGMA: the Gaussian priors
# on these parameters (1 sigma, centred on the adopted values of CUSTOM, independent: correlations ignored). "a_rs" also serves "a_rs|b". For TOI-2431
# b, incl is derived (mostly the errors of b and a/R* combined) and only sets the sin^2 i error of the ellipsoidal prior.
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
# (simple_bias_analysis.ellipsoidal_prior, deformation_prior). ELLIPSOIDAL_U_RANGE, ELLIPSOIDAL_G_RANGE: the bracket of the linear limb-darkening and
# of the gravity-darkening coefficient of the star in the infrared that alpha_ell of the stellar ellipsoidal term is evaluated over (NOT from a table
# yet; K dwarfs).
PLANET_MASSES = {"TOI-2431 b": (6.2, 1.6, 0.661, 0.0225),    # Tas et al. 2026 (A&A 707, A238) Tab. 1-2 (FCO fit; M_star +0.021/-0.024 symmetrised)
                 "K2-141 b":   (5.08, 0.41, 0.708, 0.028)}   # Malavolta et al. 2018 (AJ 155, 107) Tab. 2-3
M_EARTH_SUN = 3.0035e-6   # M_Earth / M_Sun (IAU 2015 nominal values)
ELLIPSOIDAL_U_RANGE, ELLIPSOIDAL_G_RANGE = (0.1, 0.3), (0.1, 0.4)
# Fluid Love number h2 of the planets' tidal deformation (simple_bias_analysis.deformation_limits; the row "+ planetary deformation" of
# fig:res_nuisance): the true value, 2 for rocky planets (Correia 2014, A&A 570, L5, after Yoder 1995; Earth's moment of inertia in his Darwin-Radau
# relation gives 1.93), and the range a fit may use, from all mass at the centre (1) to a homogeneous body (2.5).
DEFORMATION_H2_TRUE, DEFORMATION_H2_RANGE = 2., (1., 2.5)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _archive():
    """The aggregate table, loaded once, with the ranking metric and validity mask attached."""
    path = sorted((Path(__file__).resolve().parent.parent / "exoplanet_archive").glob(INPUT_GLOB))[-1]
    df = apply_custom(pd.read_csv(path, low_memory=False))   # CUSTOM / OVERRIDES take precedence over the archive, also in the ranking metric below
    df["hyperillum_metric"] = (df["pl_ratror"] - 1.0) / df["a_over_r2"]   # (r1 - r2) / R
    df["hyperillum_valid"] = (df["pl_ratror"] < MAX_R1_OVER_R2) & (df["a_over_r2"] > 1.0 + df["pl_ratror"]) # & df["st_teff"].notna()
    df["_key"] = df["pl_name"].str.lower().str.replace(r"[\s\-_]", "", regex=True)
    return df

def _normalise(name):
    """To accept naming inconsistencies"""
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")

# Keyed on the normalised name so loose spellings hit, exactly as hi_planets_by_name does for the archive:
# without this an entry written 'TOI 2431 b' silently misses the table's 'TOI-2431 b'.
# Built at import, so edit CUSTOM / OVERRIDES in this file rather than at runtime.
CUSTOM_N = {_normalise(k): v for k, v in CUSTOM.items()}
OVERRIDES_N = {_normalise(k): v for k, v in OVERRIDES.items()}
_both = set(CUSTOM_N) & set(OVERRIDES_N)
if _both: raise KeyError(f"planet(s) in both CUSTOM and OVERRIDES: {sorted(_both)}. A CUSTOM planet never reaches OVERRIDES in hi_planets_by_name, so put every change into its CUSTOM entry.")


# Archive columns that a PlanetParams field (CUSTOM entry or OVERRIDES value) replaces, so that every consumer of the archive table sees the curated
# values.
_FIELD_COLUMNS = {"period": "pl_orbper", "incl": "pl_orbincl", "ecc": "pl_orbeccen", "w": "pl_orblper", "T2_eff": "st_teff", "logg": "st_logg", "feh": "st_met", "a_AU": "pl_orbsmax", "T14": "pl_trandur"}
R_EARTH_IN_R_SUN = float(const.R_earth / const.R_sun)

# A copy of an archive table with CUSTOM and OVERRIDES written into it: CUSTOM / OVERRIDES always take precedence over the archive, not only in
# hi_planets_by_name but also in the sample cuts and background scatter of the bias maps (simple_bias_analysis.rocky_sample / overlay_planets /
# planet_archive_value) and in best_hi_planets. A CUSTOM planet replaces pl_orbper, pl_orbincl, pl_orbeccen, pl_orblper, pl_ratror (r1/r2), pl_rprs2,
# a_over_r2 and pl_ratdor (R/r2; e = 0 for all of them), st_teff, st_logg, st_met, pl_orbsmax (a_AU), pl_trandur (T14), st_rad = a_AU / (a/R*) (CUSTOM
# entries set a_AU = (a/R*) x R* of the adopted star for exactly this) and pl_rade = Rp/R* x st_rad. An OVERRIDES entry replaces only the columns of
# the fields it gives. Columns derived from these in ExoArchive_customcols.py (pl_Teq, the occultation depths, ESM, ...) are NOT recomputed. A CUSTOM
# planet the table does not have is left out (nothing to patch), with no note: the table is what it is. Rows are matched on the normalised name.
def apply_custom(df):
    df = df.copy()
    key = df["pl_name"].map(_normalise)
    for name, pl in CUSTOM.items():
        rows = key == _normalise(name)
        if not rows.any(): continue
        for field, col in _FIELD_COLUMNS.items():
            if col in df: df.loc[rows, col] = getattr(pl, field)
        rp_rs, a_rs = pl.r1 / pl.r2, pl.R / pl.r2
        for col, val in (("pl_ratror", rp_rs), ("pl_rprs2", rp_rs ** 2), ("a_over_r2", a_rs), ("pl_ratdor", a_rs)):
            if col in df: df.loc[rows, col] = val
        r_star = pl.a_AU * AU_IN_R_SUN / a_rs   # [R_sun]
        if "st_rad" in df: df.loc[rows, "st_rad"] = r_star
        if "pl_rade" in df: df.loc[rows, "pl_rade"] = rp_rs * r_star / R_EARTH_IN_R_SUN
    for name, fields in OVERRIDES.items():
        rows = key == _normalise(name)
        if not rows.any(): continue
        for field, val in fields.items():
            col = _FIELD_COLUMNS.get(field)
            if field in ("r1", "r2") and "pl_ratror" in df:
                r1 = fields.get("r1", df.loc[rows, "pl_ratror"].iloc[0]); r2 = fields.get("r2", 1.0)
                df.loc[rows, "pl_ratror"] = r1 / r2
                if "pl_rprs2" in df: df.loc[rows, "pl_rprs2"] = (r1 / r2) ** 2
            elif field == "R":
                for col in ("a_over_r2", "pl_ratdor"):
                    if col in df: df.loc[rows, col] = val / fields.get("r2", 1.0)
            elif col is not None and col in df:
                df.loc[rows, col] = val
    return df


_NOTED = set()   # (planet, note) pairs already printed, so each stellar-parameter fallback is reported once per session

def _note_once(name, msg):
    if (name, msg) not in _NOTED:
        _NOTED.add((name, msg)); print(f"{name}: {msg}")

# log g and [Fe/H] of the host from an archive row (only the stellar SED uses them, astro_utils.stellar_spectrum), with fallbacks that are reported
# instead of silent:
# - no st_logg: log g = log10(G M / R^2) with M the main-sequence mass at st_teff (astro_utils.main_sequence_M_star) and R = st_rad (the main-sequence
#   radius where st_rad is missing too); the archive has no stellar-mass column. Where st_logg IS known this reproduces it to +0.003 dex (median;
#   16-84 %: +-0.04, M dwarfs +-0.06-0.09), while a fixed 4.5 is off by -0.28 dex for
#   M dwarfs (log g 4.5 instead of 5.0 shifts the LRS stellar flux by ~6 % at 2600 K). Without st_teff, 4.5 (such a host has no stellar spectrum
#     anyway).
# - no st_met: solar [Fe/H] = 0, the standard assumption (k changes by <= 0.8 % per 0.1 dex; stellar_model/NewEra/build_newera.ipynb).
# Keys in `skip` (values given in OVERRIDES) are neither derived nor reported.
def _stellar_logg_feh(row, skip=()):
    name, out = row["pl_name"], {}
    if "logg" not in skip:
        logg = row["st_logg"]
        if pd.isna(logg):
            from astro_utils import main_sequence_M_star, main_sequence_R_star   # imported here: astro_utils imports this module
            T, R = row["st_teff"], row["st_rad"]
            if pd.notna(T):
                src = "st_rad" if pd.notna(R) else "the main-sequence radius"
                R = R if pd.notna(R) else float(main_sequence_R_star(T))
                logg = float(np.log10(const.G.cgs.value * float(main_sequence_M_star(T)) * const.M_sun.cgs.value / (R * const.R_sun.cgs.value) ** 2))
                _note_once(name, f"no st_logg in the archive; log g = {logg:.2f} from the main-sequence mass at T_eff = {T:.0f} K and {src}")
            else:
                logg = 4.5
                _note_once(name, "no st_logg and no st_teff in the archive; log g = 4.5 assumed")
        out["logg"] = float(logg)
    if "feh" not in skip:
        feh = row["st_met"]
        if pd.isna(feh):
            feh = 0.0
            _note_once(name, "no st_met in the archive; solar [Fe/H] = 0 assumed for the stellar spectrum")
        out["feh"] = float(feh)
    return out

def _params_from_row(row):
    """One PlanetParams from an archive row, with r1, r2, R in stellar radii and OVERRIDES applied."""
    overrides = OVERRIDES_N.get(row["_key"], {})   # row["_key"] is the normalised pl_name
    fields = dict(
        name=row["pl_name"],
        period=row["pl_orbper"],                        # days
        incl=row["pl_orbincl"],                         # deg
        ecc=0.0 if pd.isna(row["pl_orbeccen"]) else row["pl_orbeccen"],
        w=DEFAULT_W_DEG if pd.isna(row["pl_orblper"]) else row["pl_orblper"],
        r1=row["pl_ratror"],                            # stellar radii
        r2=1.0,                                         # stellar radii, by construction
        R=row["a_over_r2"],                             # stellar radii
        T2_eff=row["st_teff"],  # K
        **_stellar_logg_feh(row, skip=overrides),  # logg [log cgs], feh [dex]: archive values, or reported fallbacks
        # a_AU only feeds the light-travel-time correction, so reconstruct it from R and the stellar radius where pl_orbsmax is missing.
        a_AU=row["pl_orbsmax"] if pd.notna(row["pl_orbsmax"]) else row["a_over_r2"] * row["st_rad"] / AU_IN_R_SUN,
        T14=row["pl_trandur"],                   # hours
    )
    fields.update(overrides)
    return PlanetParams(**fields)


# --------------------------------------------------------------------------
# Planet Parameter Functions
# --------------------------------------------------------------------------

def best_hi_planets(n_planets, max_rade=ROCKY_MAX_RADE):
    """The n_planets ROCKY archive planets (pl_rade < max_rade Earth radii) most suitable for hyper-illumination, as {name: PlanetParams}, ordered strongest first.
    max_rade=None drops the radius cut (giant planets included)."""
    df = _archive()
    ok = df["hyperillum_valid"]
    if max_rade is not None: ok = ok & (df["pl_rade"] < max_rade)   # NaN radius -> False: cannot be confirmed rocky
    selected = df[ok].nsmallest(n_planets, "hyperillum_metric")
    return {row["pl_name"]: CUSTOM.get(row["pl_name"], _params_from_row(row)) for _, row in selected.iterrows()}

def _archive_row(name):
    """The archive row of one planet by loose name; KeyError (not pandas' IndexError) when the archive does not have it."""
    rows = _archive().pipe(lambda d: d[d["_key"] == _normalise(name)])
    if rows.empty: raise KeyError(f"{name!r} not found in {INPUT_GLOB} (nor in CUSTOM)")
    return rows.iloc[0]

def hi_planets_by_name(names):
    """The PlanetParams of one planet, or {name: PlanetParams} for a list of names. Accepts loose spellings ('TOI-2431b'). CUSTOM entries are returned
    whether or not they exist in the archive. An unknown name raises KeyError."""
    if isinstance(names, str):
        custom = CUSTOM_N.get(_normalise(names))
        if custom is not None:
            out = custom
        else:
            out = _params_from_row(_archive_row(names))
    else:
        out = {}
        for name in names:
            custom = CUSTOM_N.get(_normalise(name))
            if custom is not None:
                out[custom.name] = custom
                continue
            row = _archive_row(name)
            out[row["pl_name"]] = _params_from_row(row)
    return out

def best_hi_planets_df(n_planets, max_rade=ROCKY_MAX_RADE):
    """Returns df of n_planets most suitable (rocky, see best_hi_planets) planets for hyper-illumination, including simple metrics."""
    planets = best_hi_planets(n_planets, max_rade=max_rade)
    df = pd.DataFrame([[name, *params.args, params.T14, params.terminator_penumbra(unit="deg"), params.terminator_nightside(unit="deg"), params.fraction_illuminated()*100, params.substellar_boost()*100] for name, params in planets.items()],
                        columns=["Name", "Period [d]", "Incl [°]", "Ecc", "w [°]", "Rp/Rs", "Rs/Rs", "a/Rs", "Ts_eff [K]", "a [AU]", "T14 [h]", "gamma_f [°]", "gamma_t [°]", "f_illum [%]", "sub_boost [%]"])
    return df[["Name", "Period [d]", "Incl [°]", "Ecc", "w [°]", "Rp/Rs", "Rs/Rs", "a/Rs", "a [AU]", "Ts_eff [K]", "T14 [h]", "gamma_f [°]", "gamma_t [°]", "f_illum [%]", "sub_boost [%]"]]


def archive_densities(name):
    """(rho_star, rho_p) [g/cm^3] of an archive planet by loose name: the host from st_logg and st_rad (rho = 3 g / (4 pi G R), the archive has no mass column), the planet from
    pl_bmasse and pl_rade. NaN where the archive lacks a value; KeyError for a planet the archive does not have (CUSTOM entries are not consulted)."""
    row = _archive_row(name)
    g = 10. ** row["st_logg"]                                                   # [cm/s^2]
    R = row["st_rad"] * const.R_sun.cgs.value
    m_p, r_p = row["pl_bmasse"] * const.M_earth.cgs.value, row["pl_rade"] * const.R_earth.cgs.value
    return 3. * g / (4. * np.pi * const.G.cgs.value * R), m_p / (4. / 3. * np.pi * r_p ** 3)




###########################################################################################################################
#####
#####   Wavelength Filters
#####
###########################################################################################################################

# return throughput
@lru_cache(maxsize=None)
def _read_filter_file(filter_path):
    # Cached disk read
    d = np.loadtxt(filter_path)
    o = np.argsort(d[:, 0])
    return d[o, 0], d[o, 1]

def load_filter(lambda_range, filter_path=None, n=300):
    l1, l2 = float(lambda_range[0]), float(lambda_range[1])
    if filter_path is None:
        lam = np.linspace(l1, l2, n)
        return lam, np.ones_like(lam)
    lam, w = _read_filter_file(str(filter_path))
    mask = (lam >= l1) & (lam <= l2)

    lam_out = lam[mask]
    w_out = w[mask]   # the files are plain throughputs T(lambda) [electrons per incident photon], the SVO / Pandeia convention; the photon weighting lambda/(hc) is applied by the consumer (band_avg_planck here, get_filter in fspiderman)
    return lam_out, w_out


FILTER_DIR = Path(__file__).resolve().parent.parent / "filters" / "Spiderman"


# Used wavelength ranges [um] for observations -- what to integrate over
FILTER_LAMBDA_UM = {
    "MIRI/F1280W":   [11.6, 14.0],   # MIRI imaging TSO. Below the CO2 band, so combined with F1500W it tests for a CO2 atmosphere
    "MIRI/F1500W":   [13.5, 16.6],   # MIRI imaging TSO. Used on TRAPPIST-1 (15 um CO2 band), long wavelengths suit a cold nightside
    "MIRI/F1800W":   [16.5, 19.5],   # MIRI imaging TSO. MIRI's thermal background climbs steeply past ~15 um
    "MIRI/LRS":      [5.0, 10.5],    # LRS runs to ~12 um, but the red end is trimmed where the throughput drops
    "Spitzer/IRAC2": [4.0, 5.0],     # Spitzer IRAC channel 2 (4.5 um), to compare with the measured occultation depth fp = 142 ppm (Zieba 2022)
    "JWST_NIRSpec.G395H_NRS1": [2.9, 3.7], # two arrays of NIRSpec/G395H
    "JWST_NIRSpec.G395H_NRS2": [3.8, 5.2] # two arrays of NIRSpec/G395H
}

# Full filter ranges [um] -- the extent of the throughput files
FULL_LAMBDA_UM = {
    "MIRI/F1280W":   [11.0654, 14.5816],
    "MIRI/F1500W":   [12.7876, 17.4925],
    "MIRI/F1800W":   [15.5725, 20.8505],
    "MIRI/LRS":      [4.8900, 10.6100],   # 5.0-12.0 um and 5.0-14.0 um versions also exist in FILTER_DIR
    "Spitzer/IRAC2": [3.6955, 5.1008],
    "JWST_NIRSpec.G395H_NRS1": [2.8530, 3.7363], # two arrays of NIRSpec/G395H
    "JWST_NIRSpec.G395H_NRS2": [3.7960, 5.2039] # two arrays of NIRSpec/G395H
}

# SPIDERMAN throughput files
FILTER_FILES = {
    "MIRI/F1280W":   "JWST_MIRI.F1280W.spiderman.dat",
    "MIRI/F1500W":   "JWST_MIRI.F1500W.spiderman.dat",
    "MIRI/F1800W":   "JWST_MIRI.F1800W.spiderman.dat",
    "MIRI/LRS":      "MIRI_LRS.5.0-10.5um.spiderman.dat",
    "Spitzer/IRAC2": "Spitzer_IRAC.I2.spiderman.dat",
    "JWST_NIRSpec.G395H_NRS1": "JWST_NIRSpec.G395H_NRS1.spiderman.dat",
    "JWST_NIRSpec.G395H_NRS2": "JWST_NIRSpec.G395H_NRS2.spiderman.dat"
}

# PandExo mode names (jdi.load_mode_dict) -- None where PandExo has no mode
PANDEXO_MODES = {
    "MIRI/F1280W":   None,           # MIRI imaging TSO is not in PandExo: rms_1min uses CUSTOM_SIGMA_1MIN instead
    "MIRI/F1500W":   None,
    "MIRI/F1800W":   None,
    "MIRI/LRS":      "MIRI LRS",
    "Spitzer/IRAC2": None,
    "JWST_NIRSpec.G395H_NRS1": "NIRSpec G395H", # one PandExo mode covers both arrays,
    "JWST_NIRSpec.G395H_NRS2": "NIRSpec G395H"  # rms_1min_pandexo then bins to the array's filter_lambda range
}


# The JWST bands of the paper, by band centre, and the fit list of the per-instrument analyses: every band singly plus one joint fit of the two
# NIRSpec/G395H arrays (a tuple inside the list = one joint fit of those detectors: shared A_B / F, a baseline per detector). Lists, not tuples, so a
# tuple always means a joint fit.
JWST_BANDS = ["JWST_NIRSpec.G395H_NRS1", "JWST_NIRSpec.G395H_NRS2", "MIRI/LRS", "MIRI/F1280W", "MIRI/F1500W", "MIRI/F1800W"]
JWST_RETRIEVALS = JWST_BANDS[:2] + [tuple(JWST_BANDS[:2])] + JWST_BANDS[2:]
_bad = set(JWST_BANDS) - set(FILTER_LAMBDA_UM)
if _bad: raise KeyError(f"JWST_BANDS uses unknown instrument(s) {sorted(_bad)}; valid keys are {list(FILTER_LAMBDA_UM)}")

# Joint detector fits by name, for the `instruments` of run_dynesty.py: the alias stands for the detector tuple (the spelling JWST_RETRIEVALS uses);
# both spellings are accepted by resolve_instrument, and joint_name turns the tuple back into the alias for file names.
JOINT_INSTRUMENTS = {"JWST_NIRSpec.G395H_joint": tuple(JWST_BANDS[:2])}

def resolve_instrument(instrument):
    """An instrument entry -> a band name (str: a single-detector fit) or the tuple of band names of a joint fit (given as a tuple, or as one of the JOINT_INSTRUMENTS aliases)."""
    if isinstance(instrument, str) and instrument in JOINT_INSTRUMENTS: return JOINT_INSTRUMENTS[instrument]
    dets = tuple(instrument) if isinstance(instrument, (list, tuple)) else (instrument,)
    bad = [d for d in dets if d not in FILTER_LAMBDA_UM]
    if bad: raise KeyError(f"unknown instrument(s) {bad}; valid: {list(FILTER_LAMBDA_UM)} or a joint alias {list(JOINT_INSTRUMENTS)}")
    if len(dets) < 1 or len(set(dets)) != len(dets): raise ValueError(f"a joint fit needs distinct detectors, got {dets}")
    return dets if isinstance(instrument, (list, tuple)) else instrument

def detector_label(instrument):
    """Short label of a band for parameter names and panel titles: the part after the last '_' or '/' ("JWST_NIRSpec.G395H_NRS1" -> "NRS1", "MIRI/LRS" -> "LRS", "MIRI/F1280W" -> "F1280W")."""
    return re.split(r"[_/]", instrument)[-1]

def joint_name(detectors):
    """File-name spelling of a joint fit: its JOINT_INSTRUMENTS alias if there is one, else 'joint-' + the detector labels joined by '+'."""
    for alias, dets in JOINT_INSTRUMENTS.items():
        if tuple(dets) == tuple(detectors): return alias
    return "joint-" + "+".join(detector_label(d) for d in detectors)

# Converted once at import
FILTER_LAMBDA_M = {name: 1e-6 * np.array(r) for name, r in FILTER_LAMBDA_UM.items()}
FULL_LAMBDA_M = {name: 1e-6 * np.array(r) for name, r in FULL_LAMBDA_UM.items()}

def filter_lambda(instrument, planet=None):
    """The wavelength range [m] used for observations as a 2-element array: the band's FILTER_LAMBDA_UM range or, for a planet (PlanetParams or name) with a PARTIAL_BANDS entry on
    this instrument, the part of it that planet can be observed in (its host saturates the rest)."""
    if planet is not None:
        part = PARTIAL_BANDS_N.get(_normalise(getattr(planet, "name", planet)), {}).get(instrument)
        if part is not None: return part.copy()
    return FILTER_LAMBDA_M[instrument]

def full_lambda(instrument):
    """Full wavelength range [m] of the throughput file as a 2-element array."""
    return FULL_LAMBDA_M[instrument]

def filter_loc(instrument):
    """Path [str] to the throughput file of an instrument (a FILTER_LAMBDA_UM key) in filters/Spiderman/."""
    return str(FILTER_DIR / FILTER_FILES[instrument])

def pandexo_mode(instrument):
    """The PandExo mode name (as jdi.load_mode_dict expects it) for an instrument key of FILTER_LAMBDA_UM."""
    mode = PANDEXO_MODES[instrument]
    if mode is None:
        raise ValueError(f"PandExo has no mode for {instrument!r}; rms_1min takes such noise levels from CUSTOM_SIGMA_1MIN.")
    return mode


def instruments():
    """The instrument keys, for iterating over all of them."""
    return list(FILTER_FILES)



###########################################################################################################################
#####
#####   Pandexo (Noise Calculation) - inspired by the PandExo notebook of Sebastian Zieba
#####
###########################################################################################################################


# PandExo inputs and the aggregate columns they come from. Every one of these is required:
# a missing value raises, and the escape hatch is to pass it as a keyword override.
PANDEXO_COLUMNS = {
    "period":    "pl_orbper",   # days (not used by PandExo itself, reported for sanity)
    "rp_rearth": "pl_rade",     # Earth radii
    "rs_rsun":   "st_rad",      # solar radii
    "teff":      "st_teff",     # K
    "logg":      "st_logg",     # log surface gravity, cgs
    "metal":     "st_met",      # log Fe/H
    "t14_h":     "pl_trandur",  # transit duration T14, hours
}

# Reference wavelength of the normalisation band [micron], see exo_dict['star']['ref_wave'].
REF_WAVE_UM = {"sy_kmag": 2.22, "sy_jmag": 1.25}

# Keys pandexo_params returns, and therefore the keys accepted as overrides.
PANDEXO_PARAM_KEYS = (*PANDEXO_COLUMNS, "mag", "ref_wave")

# sigma_1min [ppm/min] from the S/N of one integration and its duration: rms_int = 1e6 / snr_int ppm, scaled to one minute by sqrt(t_int / 60 s).
def sigma_1min_from_integration(snr_int, t_int_s):
    return 1e6 / snr_int * np.sqrt(t_int_s / 60.)


# Custom noise levels sigma_1min [ppm/min] for instruments PandExo has no mode for (MIRI imaging, ...): {planet name: {instrument (FILTER_LAMBDA_UM
# key): sigma_1min}}. rms_1min falls back to these only where PandExo cannot supply a value. Planet names are matched loosely ("TOI-2431b" ==
# "TOI-2431 b").
CUSTOM_SIGMA_1MIN = {
    "TOI-2431 b": {
        "MIRI/F1280W": sigma_1min_from_integration(989.38, 3.59),    # 3.59 s/int  -> 247.2 ppm/min
        "MIRI/F1500W": sigma_1min_from_integration(1135.44, 5.39),   # 5.39 s/int  -> 264.0 ppm/min
        "MIRI/F1800W": sigma_1min_from_integration(1196.43, 13.48),  # 13.48 s/int -> 396.2 ppm/min
    },
}
_bad = {ins for d in CUSTOM_SIGMA_1MIN.values() for ins in d} - set(FILTER_LAMBDA_UM)
if _bad: raise KeyError(f"CUSTOM_SIGMA_1MIN uses unknown instrument(s) {sorted(_bad)}; valid keys are {list(FILTER_LAMBDA_UM)}")
CUSTOM_SIGMA_1MIN_N = {_normalise(k): v for k, v in CUSTOM_SIGMA_1MIN.items()}

# Modes in which a planet cannot be observed: {planet name: {instrument (FILTER_LAMBDA_UM key): reason}}. rms_1min raises NoNoiseLevelError for them
# before PandExo runs, exactly as for an instrument without any noise level: planet_bias_table fits them without a Fisher sigma (VERDICT_NO_NOISE),
# joint fits that need the detector and the false-positive tables leave the planet out, run_dynesty.py skips the entry, and paper_plot_utils.ns_series
# expects no runs (observable). Planet names are matched loosely.
_G395H_SAT_55CNC = "the host (K = 4.0) saturates NIRSpec/G395H; NIRCam grism was the only non-saturating near-infrared option (Patel et al. 2024, A&A 690, A159; Hu et al. 2024 used NIRCam F444W)"
UNOBSERVABLE = {
    "55 Cnc e": {"JWST_NIRSpec.G395H_NRS1": _G395H_SAT_55CNC, "JWST_NIRSpec.G395H_NRS2": _G395H_SAT_55CNC},
}
_bad = {ins for d in UNOBSERVABLE.values() for ins in d} - set(FILTER_LAMBDA_UM)
if _bad: raise KeyError(f"UNOBSERVABLE uses unknown instrument(s) {sorted(_bad)}; valid keys are {list(FILTER_LAMBDA_UM)}")
UNOBSERVABLE_N = {_normalise(k): v for k, v in UNOBSERVABLE.items()}

def observable(planet_name, instrument):
    """False if any detector of `instrument` (a band, a detector tuple or a JOINT_INSTRUMENTS alias) is in UNOBSERVABLE for the planet."""
    dets = resolve_instrument(instrument)
    return not any(d in UNOBSERVABLE_N.get(_normalise(planet_name), {}) for d in ([dets] if isinstance(dets, str) else dets))

# Bands a planet can be observed in only in part: {planet name: {instrument (FILTER_LAMBDA_UM key): [lo, hi] um}}. The host saturates the rest of the
# band in the first group (PandExo's full_saturation_mask: unrecoverable), so the white-light curve is the sum of the clean pixels, and the forward
# model must integrate the same range as the noise is computed over. filter_lambda(instrument, planet) returns this range; FittingConfig.__post_init__
# applies it to every config of the planet on the instrument's standard band (observed_band), so instrument_cfg, planet_bias_table, run_dynesty.py and
# every replace(cfg, planet=...) get it; rms_1min bins the PandExo spectrum over it. Planet names are matched loosely; a planet renamed around the
# same host (simple_bias_analysis.moved_planet, the Roche cases) does not inherit the entry, although the saturation belongs to the host. A partial
# band changes the forward model: dynesty runs made before the entry was added are flagged by dynesty_utils.check_fit_current.
PARTIAL_BANDS = {
    # PandExo (MIRI/LRS, ngroup optimised to the minimum of 2): 31 of the 166 pixels of 5.0-10.5 um fully saturated, 5.02-6.67 um; the first clean
    # pixel is at 6.71 um, and the noise of 6.7-10.5 um equals that of the clean pixels of 5.0-10.5 um. Hu et al. 2024 (Nature 630, 609;
    # arXiv:2405.04744, Methods), with 5 groups: 5.0-6.0 um saturated in the 2nd group (dropped), 6.0-7.7 um partially saturated and recovered; their
    # spectrum and white light curve start at 6.278 um (Methods). Integrating 5.0-10.5 um instead lowers the occultation depth from 136 to 129 ppm and
    # <ln B> from 8.7 to 6.5 (A_B = 0.2, 5 orbits)
    "55 Cnc e": {"MIRI/LRS": [6.7, 10.5]},
}
_bad = {ins for d in PARTIAL_BANDS.values() for ins in d} - set(FILTER_LAMBDA_UM)
if _bad: raise KeyError(f"PARTIAL_BANDS uses unknown instrument(s) {sorted(_bad)}; valid keys are {list(FILTER_LAMBDA_UM)}")
for _pl, _d in PARTIAL_BANDS.items():
    for _ins, (_lo, _hi) in _d.items():
        if not FILTER_LAMBDA_UM[_ins][0] <= _lo < _hi <= FILTER_LAMBDA_UM[_ins][1]: raise ValueError(f"PARTIAL_BANDS[{_pl!r}][{_ins!r}] = {[_lo, _hi]} is not inside the band {FILTER_LAMBDA_UM[_ins]}")
        if _ins in UNOBSERVABLE.get(_pl, {}): raise ValueError(f"{_pl!r} on {_ins!r} is both UNOBSERVABLE and in PARTIAL_BANDS")
PARTIAL_BANDS_N = {_normalise(k): {ins: 1e-6 * np.array(r, float) for ins, r in v.items()} for k, v in PARTIAL_BANDS.items()}

# The band [m] a FittingConfig observes `planet` (PlanetParams or name) in, from its throughput file and band edges: where (l1, l2) is a STANDARD band
# of the instrument of filter_path (its FILTER_LAMBDA_UM range, or any planet's PARTIAL_BANDS range of it), the planet's own filter_lambda(instrument,
# planet); any other (l1, l2) - a top-hat, a band-edge sweep - as given. FittingConfig.__post_init__ calls it, so a config moved to another planet
# (replace(cfg, planet=...)) takes that planet's band. Returns (l1, l2).
def observed_band(filter_path, l1, l2, planet):
    if filter_path is None or not (np.isfinite(l1) and np.isfinite(l2)): return l1, l2
    ins = _INSTRUMENT_OF_FILE.get(os.path.realpath(str(filter_path)))
    if ins is None: return l1, l2
    standard = [FILTER_LAMBDA_M[ins]] + [d[ins] for d in PARTIAL_BANDS_N.values() if ins in d]
    if not any(np.allclose((l1, l2), b, rtol=0., atol=1e-12) for b in standard): return l1, l2
    lo, hi = filter_lambda(ins, planet)
    return float(lo), float(hi)

_INSTRUMENT_OF_FILE = {os.path.realpath(str(FILTER_DIR / f)): ins for ins, f in FILTER_FILES.items()}

# Calculate the rms noise per minute of observation
def rms_1min_pandexo(result, T14, lambda_range, visualize=False, extra=False):
    """The uncertainty per transit time interval is sqrt(2) times smaller than the transit precision itself.
    Or differently said, lets say you know that the precision (or rms) of 1 hour of observations is 10 ppm, and your transit has a duration of 1 hour, the precision youll measure the transit depth is 10 * sqrt(2) ppm because you always measure the transit relative to the baseline.
    This example assumes in transit time = out of transit time. More precisely its
            rms_transit = rms_integration * sqrt(1 / Nin + 1 / Nout)
    , with Nin=Nout being the number of integrations in transit and rms_integration the precision per integration.
    """
    l1, l2 = lambda_range*1e6 # wavelength range in um
    rms_transit_bin = binning(l1, l2, result, visualize=visualize)
    integration_duration = result['timing']['Time/Integration incl reset (sec)']  # [sec]
    transit_duration = T14 * 60 * 60  # [sec]; T14 is in hours
    integrations_per_transit = transit_duration / integration_duration  # Nin
    rms_per_integration = rms_transit_bin / np.sqrt(2/integrations_per_transit)  # frac = 1 and therefore Nin = Nout = integrations_per_transit (for this to work it is important to keep exo_dict['observation']['noccultations']=1, exo_dict['observation']['baseline_unit']='frac', exo_dict['observation']['baseline']=1)
    rms_per_minute = rms_per_integration * np.sqrt(integration_duration / 60)

    if extra: return {"Precision Transit": rms_transit_bin, "Integration Duration": integration_duration, "RMS_1min_pandexo": rms_per_minute}
    else: return rms_per_minute


def pandexo_params(planet_name, **overrides):
    """Everything PandExo needs about one planet, taken from the INPUT_GLOB aggregate table by name. A required value that is neither in the table nor overridden raises ValueError.
    """
    bad = set(overrides) - set(PANDEXO_PARAM_KEYS)
    if bad:
        raise TypeError(f"unknown override(s) {sorted(bad)}; valid keys are {list(PANDEXO_PARAM_KEYS)}")

    row = _archive().pipe(lambda d: d[d["_key"] == _normalise(planet_name)])
    if row.empty:
        raise KeyError(f"{planet_name!r} not found in {INPUT_GLOB}")
    row = row.iloc[0]

    params = {key: row[col] for key, col in PANDEXO_COLUMNS.items()}
    params["name"] = row["pl_name"]

    # normalisation magnitude: K if the archive has it, else J
    mag_col = next((c for c in ("sy_kmag", "sy_jmag") if pd.notna(row[c])), None)
    params["mag"] = row[mag_col] if mag_col else np.nan
    params["ref_wave"] = REF_WAVE_UM[mag_col] if mag_col else np.nan
    params["mag_band"] = mag_col if mag_col else "override"

    params.update(overrides)

    missing = [k for k in PANDEXO_PARAM_KEYS if pd.isna(params[k])]
    if missing:
        where = ", ".join(f"{k} (column {PANDEXO_COLUMNS.get(k, 'sy_kmag/sy_jmag')})" for k in missing)
        raise ValueError(f"{params['name']}: {INPUT_GLOB} has no value for {where}. "
                         f"Pass it explicitly, e.g. pandexo_params({planet_name!r}, {missing[0]}=<value>).")
    return params


# The settings of run_pandexo that do not come from the planet. Also part of the PandExo cache key (see rms_1min), so change them here only.
PANDEXO_FIXED = dict(sat_level=60, sat_unit="%", noccultations=1, R=None, baseline_unit="frac", baseline=1, noise_floor=0,
                     star_type="phoenix", planet_type="constant", ngroup="optimize", background="ecliptic", background_level="high")


def run_pandexo(planet_name, instrument, verbose=True, **overrides):
    """PandExo noise estimate for one archive planet, identified by name alone. instrument is a FILTER_LAMBDA_UM key; the PandExo mode is looked up via pandexo_mode.
    Any PANDEXO_PARAM_KEYS entry can be overridden by keyword.
    """
    import pandexo.engine.justdoit as jdi   # imported lazily: PandExo is optional, and a module-level import would break every other notebook that imports this file

    mode = pandexo_mode(instrument)
    p = pandexo_params(planet_name, **overrides)

    if verbose:   # the full set of values actually used, once per call, each tagged with where it came from
        src = lambda k: "override" if k in overrides else PANDEXO_COLUMNS.get(k, p["mag_band"])
        unit = {"period": "d", "rp_rearth": "R_earth", "rs_rsun": "R_sun", "teff": "K", "logg": "log cgs",
                "metal": "log Fe/H", "t14_h": "h", "mag": f"mag ({p['mag_band']})", "ref_wave": "um"}
        print(f"PandExo inputs for {p['name']}  [instrument={instrument} (PandExo: {mode})]")
        for k in PANDEXO_PARAM_KEYS:
            print(f"    {k:10s} = {float(p[k]):<12.5g} {unit[k]:<16s} <- {src(k)}")
        print("    fixed: " + ", ".join(f"{k}={v}" for k, v in PANDEXO_FIXED.items()) + "\n")

    fx = PANDEXO_FIXED
    exo_dict = jdi.load_exo_dict()
    exo_dict['observation']['sat_level'] = fx["sat_level"]       # saturation level in percent of full well
    exo_dict['observation']['sat_unit'] = fx["sat_unit"]
    exo_dict['observation']['noccultations'] = fx["noccultations"]   # number of transits
    exo_dict['observation']['R'] = fx["R"]                       # None: no fixed binning -- bin afterwards with binning() instead of redoing the calculation
    exo_dict['observation']['baseline_unit'] = fx["baseline_unit"]   # 'frac' -> baseline is the in/out time ratio
    exo_dict['observation']['baseline'] = fx["baseline"]
    exo_dict['observation']['noise_floor'] = fx["noise_floor"]   # fixed level [ppm], or a path to a wavelength-dependent solution

    exo_dict['star']['type'] = fx["star_type"]          # phoenix or user (if you have your own)
    exo_dict['star']['mag'] = p['mag']
    exo_dict['star']['ref_wave'] = p['ref_wave']  # J = 1.25, H = 1.6, K = 2.22 micron
    exo_dict['star']['temp'] = p['teff']
    exo_dict['star']['metal'] = p['metal']        # log Fe/H
    exo_dict['star']['logg'] = p['logg']          # log surface gravity, cgs
    exo_dict['star']['radius'] = p['rs_rsun']
    exo_dict['star']['r_unit'] = 'R_sun'

    exo_dict['planet']['type'] = fx["planet_type"]      # constant: fixed transit depth
    exo_dict['planet']['transit_duration'] = p['t14_h'] * 3600.
    exo_dict['planet']['td_unit'] = 's'
    exo_dict['planet']['radius'] = p['rp_rearth']
    exo_dict['planet']['r_unit'] = 'R_earth'
    exo_dict['planet']['f_unit'] = 'rp^2/r*^2'    # primary transit

    inst_dict = jdi.load_mode_dict(mode)
    inst_dict["configuration"]["detector"]["ngroup"] = fx["ngroup"]   # optimize: picks the maximum that stays under sat_level
    inst_dict['background'] = fx["background"]
    inst_dict['background_level'] = fx["background_level"]

    result = jdi.run_pandexo(exo_dict, inst_dict, save_file=False)

    if verbose: print(result['warning'])

    return result



# PandExo's transit-depth error (jwst.py, var_tot) from the in/out-of-transit electron counts N and their variances V; t_ratio = on-source time out /
# in of transit.
def pandexo_depth_error(N_in, N_out, V_in, V_out, t_ratio):
    return t_ratio * np.sqrt(V_in / N_out**2 + N_in**2 * V_out / N_out**4)


# White-light transit-depth precision [ppm] of the band (lam_start, lam_end) [micron] from a PandExo result run with R = None (run_pandexo): sum the
# electrons and variances of the band's pixels, then apply PandExo's depth error to the sums. This is how PandExo's own engine bins when R is set, and
# the noise of the photon-weighted band depth SPIDERMAN computes. Saturated pixels are left out. Not sqrt(sum sigma_i^2) / N: that is the error of an
# equal-weight mean of the pixel depths, 1.4-1.6 times larger on MIRI/LRS.
def binning(lam_start, lam_end, result, visualize=True):
    raw, spec, timing = result["RawData"], result["FinalSpectrum"], result["timing"]
    wave, error = np.asarray(raw["wave"]), np.asarray(spec["error_w_floor"])
    N_in, N_out, V_in, V_out = (np.asarray(raw[k], float) for k in ("electrons_in", "electrons_out", "var_in", "var_out"))
    t_ratio = timing["On Source Time Out of Transit"] / timing["On Source Time In Transit"]

    # PandExo sets the error of a fully saturated pixel to NaN (mask_fully_saturated_final_spectrum), so the self-check and the sums below use the
    # other pixels only
    saturated = np.asarray(spec.get("full_saturation_mask", np.zeros(wave.size)), bool)
    if saturated.shape != wave.shape: raise ValueError(f"binning(): full_saturation_mask has shape {saturated.shape}, the spectrum {wave.shape}")

    # self-check: the formula must reproduce PandExo's own per-pixel errors (fails e.g. with a noise floor, which PandExo applies per binned point and
    # this sum would drop)
    if not np.allclose(pandexo_depth_error(N_in, N_out, V_in, V_out, t_ratio)[~saturated], error[~saturated], rtol=1e-6):
        raise ValueError("binning(): the summed-electron error does not reproduce PandExo's per-pixel errors (noise floor set?)")

    in_band = (lam_start < wave) & (wave < lam_end)
    use = in_band & ~saturated
    if in_band.any() and not use.any(): raise NoNoiseLevelError(f"every PandExo pixel in {lam_start:g}-{lam_end:g} um is fully saturated: no noise level in this band")   # not observable
    if not use.any(): raise ValueError(f"no usable PandExo pixels in {lam_start:g}-{lam_end:g} um")
    if (in_band & saturated).any():   # the noise then belongs to the unsaturated part of the band, while the forward model integrates all of it unless the planet has a PARTIAL_BANDS entry
        lo, hi = wave[in_band & saturated].min(), wave[in_band & saturated].max()
        print(f"binning(): {int((in_band & saturated).sum())} of {int(in_band.sum())} PandExo pixels in {lam_start:g}-{lam_end:g} um are fully saturated "
              f"({lo:.2f}-{hi:.2f} um) and left out of the noise; give the planet the clean range in HItargets_utils.PARTIAL_BANDS, so that the forward model integrates the same band", file=sys.stderr)
    rms_transit = 1e6 * pandexo_depth_error(N_in[use].sum(), N_out[use].sum(), V_in[use].sum(), V_out[use].sum(), t_ratio)

    if visualize:
        plt.errorbar(wave, spec["spectrum"], yerr=error, fmt=".", color="0.7", label="PandExo pixels")
        plt.errorbar(wave[use], np.asarray(spec["spectrum"])[use], yerr=error[use], fmt=".", label=f"band, white light: {rms_transit:.2f} ppm")
        plt.xlabel("wavelength [micron]"); plt.ylabel("transit depth"); plt.legend()
        plt.show()
    return rms_transit


# Raised by rms_1min when neither PandExo nor CUSTOM_SIGMA_1MIN can supply a noise level, so callers can skip exactly that case (and not PandExo /
# archive errors).
class NoNoiseLevelError(ValueError):
    pass


# PandExo cache: sigma_1min of every PandExo run made through rms_1min, in a CSV next to the PandExo notebook. A run takes seconds, a lookup
# milliseconds. The key is everything the value depends on: the PandExo inputs (pandexo_params: planet, radii, T_eff, log g, [Fe/H], T14,
# normalisation magnitude and band), the PandExo mode and the band binned over, PANDEXO_FIXED, PANDEXO_NOISE_VERSION and the installed PandExo /
# Pandeia versions. Any change (an archive update, an override, a new PandExo) is a miss: rms_1min runs PandExo and adds the row, so a value computed
# from other inputs is never served. The file is locked while it is read or written (fcntl), so parallel kernels and joblib workers can share it. To
# force a rerun: rms_1min(..., use_cache=False), which replaces the row, or delete rows / the file.
PANDEXO_CACHE = Path(__file__).resolve().parent.parent / "stellar_model" / "Pandexo" / "pandexo_cache.csv"
PANDEXO_NOISE_VERSION = 2   # version of the noise computation from a PandExo result (binning, rms_1min_pandexo); bump it when that changes. 2: summed electrons (2026-11-06)
_PANDEXO_KEY_PARAMS = ("rp_rearth", "rs_rsun", "teff", "logg", "metal", "t14_h", "mag", "ref_wave")   # pandexo_params without the period, which PandExo does not use
_PANDEXO_TEXT = ("planet", "instrument", "mode", "mag_band", "settings", "pandexo_version", "pandeia_version", "created")   # read as text, so "2026.10" stays a version

def _package_version(dist):
    from importlib.metadata import version, PackageNotFoundError
    try: return version(dist)
    except PackageNotFoundError: return "n/a"

def _pandexo_key(p, instrument, band):
    """Cache key of one sigma_1min: p from pandexo_params, band [m]."""
    return dict(planet=p["name"], instrument=instrument, mode=pandexo_mode(instrument), band_lo_um=1e6 * float(band[0]), band_hi_um=1e6 * float(band[1]),
                **{k: float(p[k]) for k in _PANDEXO_KEY_PARAMS}, mag_band=p["mag_band"],
                settings=f"noise v{PANDEXO_NOISE_VERSION}; " + ", ".join(f"{k}={v}" for k, v in PANDEXO_FIXED.items()),
                pandexo_version=_package_version("pandexo.engine"), pandeia_version=_package_version("pandeia.engine"))

def _pandexo_match(df, key):
    """Rows of the cache table that match key: numbers to 1e-9 relative, text exactly. A column the table lacks matches nothing."""
    m = np.ones(len(df), bool)
    for k, v in key.items():
        if k not in df: return np.zeros(len(df), bool)
        m &= np.isclose(pd.to_numeric(df[k], errors="coerce"), v, rtol=1e-9, atol=0.) if isinstance(v, float) else (df[k].astype(str) == str(v)).to_numpy()
    return m

def _read_pandexo_cache(f):
    f.seek(0)
    return pd.read_csv(f, dtype={c: str for c in _PANDEXO_TEXT}) if os.fstat(f.fileno()).st_size else pd.DataFrame()

def _lock(f, exclusive):
    """Lock the open cache file for parallel kernels and workers (fcntl, POSIX only; no lock on Windows)."""
    try: import fcntl
    except ImportError: return
    fcntl.flock(f, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

def pandexo_cache_lookup(key):
    """sigma_1min [ppm/min] of the newest cached row matching key, or None."""
    if not PANDEXO_CACHE.exists(): return None
    with open(PANDEXO_CACHE) as f:
        _lock(f, exclusive=False)
        df = _read_pandexo_cache(f)
    m = _pandexo_match(df, key)
    return float(df.loc[m, "sigma_1min"].iloc[-1]) if m.any() else None

def _pandexo_cache_store(entries):
    """Add (key, values) entries to the cache, replacing rows with the same key; read-modify-write under an exclusive lock."""
    PANDEXO_CACHE.touch(exist_ok=True)
    with open(PANDEXO_CACHE, "r+") as f:
        _lock(f, exclusive=True)
        df = _read_pandexo_cache(f)
        for key, _ in entries:
            if len(df): df = df[~_pandexo_match(df, key)]
        new = pd.DataFrame([{**key, **values} for key, values in entries])
        df = pd.concat([df, new], ignore_index=True) if len(df) else new
        f.seek(0); f.truncate()
        df.to_csv(f, index=False)


def rms_1min(planet, instrument, lrs_range=None, use_cache=True, **overrides):
    """Noise level sigma_1min [ppm/min] of one planet on one instrument: from the PandExo cache (PANDEXO_CACHE) if it has this exact configuration,
    else run_pandexo + rms_1min_pandexo, stored in the cache. instrument is a FILTER_LAMBDA_UM key (the PandExo mode is looked up via pandexo_mode),
    lrs_range [m] the band the PandExo spectrum is binned over (default: filter_lambda(instrument, planet), i.e. the planet's PARTIAL_BANDS range
    where it has one). overrides go to run_pandexo (verbose, mag, ...); the transit duration is planet.T14 unless t14_h overrides it. A new run also
    stores every other detector of the same PandExo mode at its own band, from the same result (one NIRSpec G395H run gives NRS1 and NRS2).
    use_cache=False reruns PandExo and replaces the cached row. Where PandExo has no mode for the instrument, the value comes from CUSTOM_SIGMA_1MIN
    (lrs_range and overrides then do not apply); if that has none either, a NoNoiseLevelError says where to add one. A mode listed in UNOBSERVABLE for
    the planet, and a band whose PandExo pixels are all fully saturated, raise NoNoiseLevelError as well (the latter is not cached: PandExo reruns on
    the next call)."""
    reason = UNOBSERVABLE_N.get(_normalise(planet.name), {}).get(instrument)
    if reason: raise NoNoiseLevelError(f"no noise level for {planet.name!r} on {instrument!r}: {reason} (HItargets_utils.UNOBSERVABLE)")
    if PANDEXO_MODES[instrument] is None:
        custom = CUSTOM_SIGMA_1MIN_N.get(_normalise(planet.name), {})
        if instrument not in custom:
            raise NoNoiseLevelError(f"no noise level for {planet.name!r} on {instrument!r}: PandExo has no mode for this instrument and CUSTOM_SIGMA_1MIN (HItargets_utils.py) has no entry "
                             f"for this planet/instrument (it has {sorted(custom) or 'none'} for this planet). Add one there, or pass sigma_1min by hand.")
        if overrides.pop("verbose", False): print(f"{planet.name} [{instrument}]: sigma_1min = {custom[instrument]:.1f} ppm/min from CUSTOM_SIGMA_1MIN (no PandExo mode)")
        return custom[instrument]
    if lrs_range is None: lrs_range = filter_lambda(instrument, planet)   # the planet's own part of the band where its host saturates the rest (PARTIAL_BANDS)
    overrides.setdefault("t14_h", planet.T14)
    verbose = overrides.pop("verbose", True)
    p = pandexo_params(planet.name, **overrides)   # exactly the inputs run_pandexo would use: the cache key
    if use_cache:
        cached = pandexo_cache_lookup(_pandexo_key(p, instrument, lrs_range))
        if cached is not None:
            if verbose: print(f"{p['name']} [{instrument}]: sigma_1min = {cached:.2f} ppm/min from the PandExo cache ({PANDEXO_CACHE.name})")
            return cached
    result = run_pandexo(planet.name, instrument, verbose=verbose, **overrides)
    created, entries, value = datetime.now().isoformat(timespec="seconds"), [], None
    siblings = [(d, filter_lambda(d, planet)) for d, m in PANDEXO_MODES.items() if m == PANDEXO_MODES[instrument] and d != instrument]
    for ins, band in [(instrument, lrs_range)] + siblings:
        try:
            out = rms_1min_pandexo(result, overrides["t14_h"], band, extra=True)
        except ValueError:
            if ins == instrument: raise
            continue   # a sibling band the run does not cover
        entries.append((_pandexo_key(p, ins, band), dict(sigma_1min=float(out["RMS_1min_pandexo"]), transit_precision_ppm=float(out["Precision Transit"]),
                                                         integration_s=float(out["Integration Duration"]), created=created)))
        if ins == instrument: value = float(out["RMS_1min_pandexo"])
    _pandexo_cache_store(entries)
    return value


# sigma_1min [ppm] for a whole sample of archive planets on ONE instrument: rms_1min per target, keyed by PlanetParams.name. For a joint fit (the
# detector tuple or a JOINT_INSTRUMENTS alias) the value per planet is {detector: ppm/min}, the form planet_bias_table takes for a detector set.
# Served from the PandExo cache where possible; only a planet's first run on a PandExo mode takes seconds. A target the archive cannot supply (no T14,
# no magnitude) or that PandExo fails on raises (KeyError / ValueError from pandexo_params, or PandExo's own error) rather than yielding NaN: a NaN
# sigma_1min would silently corrupt the fits downstream (planet_bias_table). Only planets with no noise level at all (instrument without a PandExo
# mode and no CUSTOM_SIGMA_1MIN entry; for a joint fit: on any of its detectors, since the fit weights them by their noise) are left out of the dict,
# with a note (possibly all of them: then {}). planet_bias_table fits them without a noise level (no Fisher sigma); the false-positive tables, which
# need one, leave them out.
def rms_1min_pandexo_sample(names, instrument, verbose=False, **overrides):
    dets = resolve_instrument(instrument)
    joint = not isinstance(dets, str)
    out, missing = {}, []
    for nm in names:
        pl = hi_planets_by_name(nm)
        try:
            out[pl.name] = {d: rms_1min(pl, d, verbose=verbose, **overrides) for d in dets} if joint else rms_1min(pl, dets, verbose=verbose, **overrides)
        except NoNoiseLevelError:
            missing.append(pl.name)
    if missing:
        print(f"no noise level on {joint_name(dets) if joint else dets} (no PandExo mode and no CUSTOM_SIGMA_1MIN entry, or not observable: UNOBSERVABLE) for {len(missing)} of {len(missing) + len(out)} planets: {', '.join(missing)}")
    return out


# sigma_1min [ppm/min] of ONE planet on several instruments: {instrument: rms_1min(...)}, PandExo (cache) or CUSTOM_SIGMA_1MIN; an instrument with
# neither is left out silently (NoNoiseLevelError), so callers test `ins in levels`. A planet the archive cannot supply PandExo inputs for raises, as
# in rms_1min_pandexo_sample.
def noise_levels(planet, instruments, **overrides):
    out = {}
    for ins in instruments:
        try: out[ins] = rms_1min(planet, ins, verbose=False, **overrides)
        except NoNoiseLevelError: pass
    return out
