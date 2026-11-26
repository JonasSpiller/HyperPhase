"""Baseline settings of the model and the parameter ranges of the fits.

Every function argument and FittingConfig field that has one of these settings defaults to the value defined here; only sweeps and comparisons
with other codes deviate from them. Override them per call (full_PhaseCurve(..., A_B=0.3)) or per configuration (dataclasses.replace(cfg,
u_bol=0.6)): they are bound as default arguments when the modules are imported, so assigning a new value to a constant here at runtime does not
change those defaults."""
import os


############################################################################################################################################
#####   Baseline Settings
############################################################################################################################################

# Bond albedo A_B, which sets the temperature map: a moderate value for a bare rock.
DEFAULT_A_B = 0.2

# Band-integrated albedo A of the reflected light (not a geometric albedo), independent of A_B: molten and glassy silicates reflect weakly
# (Essack et al. 2020).
DEFAULT_A = 0.1

# Bolometric emissivity in emissivity_bol sigma T^4 = (1-A_B) S, degenerate with A_B (see irradiance.BBtemp_from_insolation). 1 assumes a
# surface that is black in the thermal infrared.
DEFAULT_EMISSIVITY_BOL = 1.

# Band-integrated emissivity in the observing band (the thermal counterpart of A); the thermal planet flux is linear in it. Deliberately
# independent of A, although Kirchhoff's law ties them in the same band (emissivity = 1 - A for an opaque surface): A is a free
# phenomenological reflectance.
DEFAULT_EMISSIVITY = 1.

# Bolometric linear limb darkening u_bol of the irradiance: a model-atmosphere value instead of the grey Eddington value of 0.6.
# For TOI-2431 b, the primary target, the bolometric profile gives 0.49 (PHOENIX) and 0.50 (MPS-ATLAS) (inputs/stellar_model/limb_darkening.ipynb).
DEFAULT_U_BOL = 0.5

# Quadratic limb darkening (u1, u2) of the batman transit. The transit is identical in both irradiation models and cancels in every
# comparison, so the value only needs to be plausible: the median of ExoTiC-LD fits (PHOENIX) for the hosts of PAPER_PLANETS in NRS1 and NRS2,
# where PHOENIX covers the band (inputs/stellar_model/limb_darkening.ipynb).
DEFAULT_U = (0.10, 0.14)

# Orbits observed and phase-folded. The periods are only 4-7 h, so five orbits are roughly 1-1.5 days of JWST time (about 27 h for TOI-2431 b,
# 34 h for K2-141 b): ambitious, but within the range of a large program.
DEFAULT_N_ORBITS = 5

# Bins per orbit of the phase-folded light curve: 12 min for TOI-2431 b, 15 min for K2-141 b. The width is derived from the period, so the
# bins tile the orbit exactly.
DEFAULT_N_BINS = 27

# Reference Rp/R* of the (a/R*, T_eff) bias maps (bias.maps.reference_planet): the median of the PAPER_PLANETS (they span 0.012-0.021). The
# bias barely depends on it, because the mismatch and the response to A_B both scale with the planet's flux: between 0.0075 and 0.03 it
# changes by less than 3 % (A_B-only fit) and 5 % (with F).
DEFAULT_MAP_RP_RS = 0.016

# Parameter ranges, shared by the least-squares bias analysis (bias) and the nested sampling (retrieval).
# A_B_BOUNDS, F_BOUNDS: the physical ranges of A_B and of the Kreidberg & Loeb (2016) redistribution F. They are the uniform priors of every
# dynesty fit and set the "unconstrained" verdict (see bias.fisher.flat_prior_sigma). F_BOUNDS is also the least-squares range of F.
A_B_BOUNDS = (0., 1.)
F_BOUNDS = (0., 0.5)

# A_B_FIT_BOUNDS: the least-squares range of A_B, the default of every fit in bias. It extends below 0 so that the bias of the point-source
# model stays defined where that model needs more absorbed energy than the star delivers to a point (A_B,fit < 0). Only (1-A_B) enters the
# temperature, so a negative A_B is a valid model evaluation. The deepest fit found is about -1.2 (true A_B = 0 near the contact limit), so
# the lower end is never reached; a fit there, or at A_B = 1, still counts as railed. Pass bounds=A_B_BOUNDS to fit within the physical range.
A_B_FIT_BOUNDS = (-2., 1.)

# Uniform prior of the free baseline offset of the dynesty fits [ppm]; the least-squares fits profile the offset out analytically.
OFFSET_BOUNDS = (-50., 50.)
# Stellar SED: "newera" (the cube of newera), an stsynphot grid ("phoenix", "ck04models", "k93models") or "blackbody". The environment variable
# HYPERPHASE_STELLAR_MODEL sets another default, e.g. "blackbody" to run without the NewEra cube; set it before hyperphase is imported.
STELLAR_MODELS = ("newera", "blackbody", "phoenix", "ck04models", "k93models")
STELLAR_MODEL = os.environ.get("HYPERPHASE_STELLAR_MODEL", "newera")
if STELLAR_MODEL not in STELLAR_MODELS: raise ValueError(f"HYPERPHASE_STELLAR_MODEL = {STELLAR_MODEL!r}: must be one of {STELLAR_MODELS}")

# Resolution (longitude, latitude nodes) of the insolation / temperature map of every forward-model evaluation; 300/150 differs from 600/300
# by ~0.001 ppm (convergence study: validation/speed_accuracy.ipynb).
DEFAULT_N_LO, DEFAULT_N_LA = 300, 150
