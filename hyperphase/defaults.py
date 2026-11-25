"""Baseline settings of the model and the parameter ranges of the fits.

Every function argument and FittingConfig field that has one of these settings defaults to the value defined here. Override them per call
(full_PhaseCurve(..., A_B=0.3)) or per configuration (dataclasses.replace(cfg, u_bol=0.6)): they are bound as default arguments when the
modules are imported, so assigning a new value to a constant here at runtime does not change those defaults."""


############################################################################################################################################
#####   Baseline Settings
############################################################################################################################################

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
# For TOI-2431 b, the primary target, the bolometric profile gives 0.49 (PHOENIX) and 0.50 (MPS-ATLAS) (inputs/stellar_model/limb_darkening.ipynb).
DEFAULT_U_BOL = 0.5

# Quadratic limb darkening (u1, u2) of the batman transit. The transit is identical in both irradiation models and cancels exactly in every
# comparison, so the value only needs to be plausible: the median of ExoTiC-LD fits over the hosts of an earlier, twelve-planet version of the sample,
# in the band throughputs of the forward model (inputs/stellar_model/limb_darkening.ipynb). Not re-derived for the final sample (its hosts span the
# same 4100-5960 K): the value does not enter any result.
DEFAULT_U = (0.09, 0.14)

# Orbits observed and phase-folded. The periods are only 4-7 h, so five orbits are roughly 1-1.5 days of JWST time (about 27 h for TOI-2431 b, 34 h
# for K2-141 b): ambitious, but within the range of a large program.
DEFAULT_N_ORBITS = 5

# Bins per orbit of the phase-folded light curve: 12 min for TOI-2431 b, 15 min for K2-141 b. The width is derived from the period, so the bins tile
# the orbit exactly.
DEFAULT_N_BINS = 27

# Reference Rp/R* of the (a/R*, T_eff) bias maps (bias.maps.reference_planet; 03_bias_maps.ipynb, 06_roche_limit.ipynb): the median
# of the PAPER_PLANETS (0.0158 for the nine of the selection rule, 0.0160 with 55 Cnc e; they span 0.012-0.021). The bias barely depends on it,
# because the mismatch and the response to A_B both scale with the planet's flux: between 0.0075 and 0.03 it changes by less than 3 % (A_B-only fit)
# and 5 % (with F).
DEFAULT_MAP_RP_RS = 0.016

# Parameter ranges, shared by the least-squares bias analysis (bias) and the nested sampling (retrieval, retrievals/run_dynesty.py).
# A_B_BOUNDS, F_BOUNDS: the physical ranges of A_B and of the Kreidberg & Loeb (2016) redistribution F. They are the uniform priors of every dynesty
# fit, and their flat-prior std defines an "unconstrained" fit (a Fisher sigma at or above it: the data add nothing to the physical range). F_BOUNDS
# is also the least-squares range of F.
A_B_BOUNDS = (0., 1.)
F_BOUNDS = (0., 0.5)

# A_B_FIT_BOUNDS: the least-squares range of A_B, the default of every fit in bias. It extends below 0 so that the bias of the
# point-source model stays defined where it needs more absorbed energy than the star delivers to a point (A_B,fit < 0, which an observer's physical
# prior would cut off at 0). Only (1-A_B) enters the temperature, so a negative A_B is a valid model evaluation. The lower end is never reached in
# practice (the deepest fit found is about -1.2: true A_B = 0 near the contact limit); a fit there, or at A_B = 1, still counts as railed. Pass
# bounds=A_B_BOUNDS to fit within the physical range instead (railed at A_B = 0).
A_B_FIT_BOUNDS = (-2., 1.)

# Uniform prior of the free baseline offset of the dynesty fits [ppm]; the least-squares fits profile the offset out analytically.
OFFSET_BOUNDS = (-50., 50.)
STELLAR_MODEL = "newera"    # stellar SED: "newera" (the cube of newera), an stsynphot grid ("phoenix", "ck04models", "k93models") or "blackbody"

# Resolution (longitude, latitude nodes) of the insolation / temperature map of every forward-model evaluation; 300/150 differs from 600/300 by ~0.001
# ppm (see model_times).
DEFAULT_N_LO, DEFAULT_N_LA = 300, 150
