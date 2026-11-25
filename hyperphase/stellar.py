"""The star: Planck functions, stellar spectra (NewEra or blackbody), the in-band stellar grid of SPIDERMAN, brightness temperatures and limb darkening."""
import contextlib
import io
import warnings
from functools import lru_cache

import numpy as np
from astropy import constants as const
from scipy.integrate import quad
from scipy.optimize import brentq
with contextlib.redirect_stdout(io.StringIO()):   # mute "looking for spidermanrc..." in every process
    import fspiderman as sp

from hyperphase.bands import filter_lambda, filter_loc, full_lambda, load_filter
from hyperphase.defaults import STELLAR_MODEL
from hyperphase.newera import newera_spectrum
from hyperphase.paths import LD_DIR


H_PLANCK = const.h.value
C_LIGHT = const.c.value
K_BOLTZ = const.k_B.value
WIEN_B = 2.897771955e-3   # Wien's displacement constant [m K]
STEFAN_BOLTZMANN = (2.0 * np.pi**5 * K_BOLTZ**4) / (15.0 * H_PLANCK**3 * C_LIGHT**2) # Stefan-Boltzmann constant
SED_FLUX_TOL = 0.02         # max |pi int(I dlambda) / (sigma T_eff^4) - 1| accepted from a model spectrum (catches wrong units and a truncated wavelength range)


def T_bright(thermal_depth, planet, lambda_range, filter_path=None, stellar_model=STELLAR_MODEL):
    """Brightness temperature [K] of a thermal planet flux [ppm] in the band lambda_range [m] (filter_path: throughput file, None for a top-hat). The
    occultation depth gives the dayside value; it is a blackbody temperature, below the surface temperature for emissivity < 1.

    Solves (r1/r2)^2 S_planet(T) = depth S_star, where S_star is SPIDERMAN's own stellar normalisation (its cubic spline through stellar_spidergrid
    at T2_eff, spidergrid_value) and S_planet the same band integral of a blackbody (band_planck: same samples, response and sum_flux). So
    T_bright inverts SPIDERMAN's depths exactly whatever the stellar model and band weighting (a top-hat is energy-weighted there, like SPIDERMAN's
    filter=False); the remaining ~0.15 K of a round trip is SPIDERMAN's own planet quadrature. Raises (check_teff_nodes) outside the T_eff nodes and
    ValueError (brentq) for a flux below the band flux at 10 K."""
    grid = stellar_spidergrid(lambda_range, filter_path, planet.logg, planet.feh, stellar_model=stellar_model)
    check_teff_nodes(grid, planet.T2_eff)
    tgt = 1e-6 * thermal_depth * spidergrid_value(grid, planet.T2_eff) / (planet.r1 / planet.r2) ** 2
    return brentq(lambda T: band_planck(lambda_range, filter_path, T) - tgt, 10., 9000.)


# --------------------------------------------------------------------------
# Planck
# --------------------------------------------------------------------------
def planck_ratio(rprs2, Ts, Tp, wavelength_m):
    """
    Occultation depth in ppm for a blackbody planet: F_p/F_s = (r1/r2)² · B_λ(T_p)/B_λ(T_s)
    B_λ(T) = 2hc²/λ⁵ · 1/(exp(hc/λkT) − 1)
    rprs2 = (r1/r2)^2
    """
    x_s = H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZ * Ts)
    x_p = H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZ * Tp)
    return rprs2 * 1e6 * (np.exp(x_s) - 1.0) / (np.exp(x_p) - 1.0)


def planck_ratio_logderiv(Tp, wavelength_m):
    """d ln(occ_depth) / d ln(Tp), for propagating a temperature uncertainty into an occultation depth."""
    x_p = H_PLANCK * C_LIGHT / (wavelength_m * K_BOLTZ * Tp)
    return x_p * np.exp(x_p) / (np.exp(x_p) - 1.0)


# Wien's displacement law
def wien_lambda_max(T):
    return WIEN_B / T # max. wavelength [in m]


# Spectral radiance [W m^-2 sr^-1 m^-1] of a blackbody at temperature T, per unit wavelength [m] (Planck function): B_lambda(T) = (2 h c^2 / lambda^5)
# / (exp(h c / (lambda k_B T)) - 1)
def planck_spectrum(wavelengths, T):
    wavelengths = np.asarray(wavelengths, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        exponent = H_PLANCK * C_LIGHT / (wavelengths * K_BOLTZ * T)
        prefactor = 2.0 * H_PLANCK * C_LIGHT ** 2 / wavelengths ** 5
        B = prefactor / np.expm1(exponent)
    B = np.where((wavelengths <= 0) | (exponent > 700), 0.0, B)
    return B if B.ndim > 0 else float(B)


# Percentage of a blackbody's total emitted flux that lies at wavelengths shorter than `lambda_cutoff`, at temperature T.
# lambda_cutoff in m
def planck_fraction_below(T, lambda_cutoff):
    numerator, _ = quad(lambda wl: planck_spectrum(wl, T), 0.0, lambda_cutoff)
    denominator = STEFAN_BOLTZMANN * T ** 4 / np.pi
    return 100.0 * numerator / denominator


# Throughput-weighted mean spectral radiance [W m^-2 sr^-1 m^-1] of a blackbody at temperature T, averaged across the band.
def band_avg_planck(T, lambda_range, filter_path=None, n=300):
    """
    filter_path=None  -> tophat over lambda_range, photon-weighted (SPIDERMAN's filter=False is energy-weighted, so this does NOT match it; T_bright does not use this function)
    filter_path given -> real throughput on the file's own grid
    """
    lam, w = load_filter(lambda_range, filter_path=filter_path, n=n)

    # R [counts/photon] -> [counts/energy], so that weighting B_lambda (an energy spectral radiance) below yields a photon-integrated band average.
    # Same convention as fspiderman.get_filter (README of github.com/JonasSpiller/fSPIDERMAN, "6. Photon-counting response weighting"). Unused by the
    # pipeline: T_bright integrates through hpu.band_planck.
    w =  w * lam / (H_PLANCK * C_LIGHT)

    B_lambda = planck_spectrum(lam, T)

    return np.trapezoid(B_lambda * w, lam) / np.trapezoid(w, lam)


# --------------------------------------------------------------------------
# Stellar SED
# --------------------------------------------------------------------------

# Model spectrum on its native grid: wavelength [m], surface intensity F_lambda/pi [W m^-2 sr^-1 m^-1] (what B_lambda is for a blackbody). The same
# stsynphot catalog PandExo uses ($PYSYN_CDBS/grid/<stellar_model>). grid_to_spec interpolates linearly between the 8 bracketing models (logg, then
# [Fe/H], then T_eff) and raises ParameterOutOfBounds outside the grid or on a neighbour without data. Checked against sigma T_eff^4, which fails on
# wrong units or a truncated wavelength range. Cached: treat the arrays as read-only.
@lru_cache(maxsize=None)
def stellar_spectrum(T_eff, logg, feh, stellar_model=STELLAR_MODEL):
    if stellar_model == "newera":   # no sigma T^4 check: the JWST product starts at 0.6 um; validated against full-range models in inputs/stellar_model/NewEra/build_newera.ipynb
        wave, I = newera_spectrum(float(T_eff), float(logg), float(feh))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # stsynphot warns about optional reference files it cannot find on import; imported only here, as it needs $PYSYN_CDBS
            import stsynphot
            from synphot import units as syn_units
        s = stsynphot.grid_to_spec(stellar_model, float(T_eff), float(feh), float(logg))   # argument order: (gridname, t_eff, metallicity, log_g)
        w = s.waveset                                                    # the model's own wavelengths [A]
        wave, I = w.to_value("m"), s(w, flux_unit=syn_units.FLAM).value * 1e7 / np.pi   # A -> m; FLAM [erg s^-1 cm^-2 A^-1] -> W m^-3; /pi: surface flux -> surface intensity
        ratio = np.pi * np.trapezoid(I, wave) / (STEFAN_BOLTZMANN * T_eff**4)
        if abs(ratio - 1.) > SED_FLUX_TOL: raise ValueError(f"{stellar_model} (T_eff={T_eff:.0f} K, logg={logg:.2f}, [Fe/H]={feh:+.2f}): pi int(I dlambda) / sigma T_eff^4 = {ratio:.4f}")
    for a in (wave, I): a.setflags(write=False)   # cached and shared by every caller
    return wave, I


# Surface intensity [W m^-2 sr^-1 m^-1] of the star on lam [m]: planck_spectrum for "blackbody", else the model spectrum interpolated.
def stellar_intensity(lam, T_eff, logg, feh, stellar_model=STELLAR_MODEL):
    if stellar_model == "blackbody": return planck_spectrum(lam, T_eff)
    wave, I = stellar_spectrum(T_eff, logg, feh, stellar_model)
    if wave[0] > np.min(lam) or wave[-1] < np.max(lam): raise ValueError(f"{stellar_model} spectrum covers {wave[0]*1e6:.2f}-{wave[-1]*1e6:.2f} um, band needs {np.min(lam)*1e6:.2f}-{np.max(lam)*1e6:.2f} um")
    return np.interp(lam, wave, I)


# lam [m] refined with the model's own samples inside [min(lam), max(lam)], so interpolating the star onto it loses no spectral structure (the catalog
# grid can be coarser or finer than a filter file). Unchanged for "blackbody".
def band_wavelengths(lam, T_eff, logg, feh, stellar_model=STELLAR_MODEL):
    lam = np.asarray(lam, float)
    if stellar_model == "blackbody": return lam
    wave = stellar_spectrum(T_eff, logg, feh, stellar_model)[0]
    return np.union1d(lam, wave[(wave > lam.min()) & (wave < lam.max())])


# --------------------------------------------------------------------------
# Limb darkening (ExoTiC-LD)
# --------------------------------------------------------------------------

# Quadratic limb darkening of the transit, fitted with ExoTiC-LD (Grant & Wakeford 2024) to the specific intensities I(mu) of its stellar grids and
# band-weighted with the throughput files of the forward model (ExoTiC-LD intensities are photon rates, the throughputs electrons per photon: the
# fspiderman weighting). The NewEra cube cannot do this, it has no angular dependence. Model files are downloaded on first use into LD_DATA. exotic_ld
# is imported inside quad_ld, so the rest of this module does not depend on it. Used by inputs/stellar_model/limb_darkening.ipynb.
LD_DATA = str(LD_DIR)   # inputs/stellar_model/ExoTiC-LD (gitignored, like NewEra/): model cache, filled on first use
LD_GRID_TEFF_MIN = {"phoenix": 2300., "mps1": 3500., "mps2": 3500., "kurucz": 3500., "stagger": 4000.}   # [K] lower T_eff edge of each ExoTiC-LD grid; below it ExoTiC-LD silently falls back to the nearest model
LD_MU_MIN = 0.1     # ExoTiC-LD default; the steep drop of spherical models at the very limb is not fitted
A_PER_M = 1e10      # ExoTiC-LD takes wavelengths in Angstrom


# Quadratic (u1, u2) of one host star (PlanetParams: T2_eff, logg, feh) per band (band keys of bands), and the model I(mu)/I(1) each
# fit was made to. Returns (rows, profiles): one dict per band (ld_model, planet, T_eff, logg, feh, band, u1, u2, in_grid, covered) and {band: (mu,
# I_mu)}. in_grid = False: T_eff below the grid (nearest-model fallback). covered = False: the band extends beyond the wavelength range of the stellar
# model (ExoTiC-LD then integrates only the overlap; no overlap gives NaN and a printed note).
def quad_ld(planet, bands, ld_model, interpolate_type="nearest", mu_min=LD_MU_MIN, ld_data_path=LD_DATA):
    from exotic_ld import StellarLimbDarkening
    sld = StellarLimbDarkening(M_H=planet.feh, Teff=planet.T2_eff, logg=planet.logg, ld_model=ld_model,
                               ld_data_path=ld_data_path, interpolate_type=interpolate_type, verbose=0)
    wl_model = sld.stellar_wavelengths.min(), sld.stellar_wavelengths.max()
    rows, profiles = [], {}
    for band in bands:
        lam, T = load_filter(full_lambda(band), filter_loc(band))          # [m], electrons per photon, as in the forward model
        wl_band = filter_lambda(band, planet) * A_PER_M                     # the planet's own part of the band (bands.PARTIAL_BANDS)
        try:
            u1, u2 = sld.compute_quadratic_ld_coeffs(wl_band, "custom", custom_wavelengths=lam * A_PER_M, custom_throughput=T, mu_min=mu_min)
            profiles[band] = (sld.mus.copy(), sld.I_mu.copy())              # I_mu is set by the call above
        except ValueError as err:                                           # no overlap with the model's wavelength range
            print(f"{ld_model}, {planet.name}, {band}: {err}")
            u1 = u2 = np.nan
        rows.append(dict(ld_model=ld_model, planet=planet.name, T_eff=planet.T2_eff, logg=planet.logg, feh=planet.feh, band=band, u1=u1, u2=u2,
                         in_grid=planet.T2_eff >= LD_GRID_TEFF_MIN[ld_model], covered=wl_model[0] <= wl_band[0] and wl_band[1] <= wl_model[1]))
    return rows, profiles


# Bolometric intensity profile I_bol(mu)/I_bol(1) of one host star (PlanetParams: T2_eff, logg, feh) from an ExoTiC-LD grid, to compare with the
# linear law of u_bol (the irradiance). ExoTiC-LD intensities are photon rates per Angstrom, so the energy weight is 1/lambda (constants cancel in the
# normalisation); integrated over the whole wavelength range of the grid on its own samples. Returns (mu ascending, I_bol(mu)/I_bol(1), u_lin, f_out):
# u_lin is the least-squares linear coefficient over mu >= mu_min, f_out the fraction of a blackbody's sigma T_eff^4 outside the grid's wavelength
# range (what the profile misses; PHOENIX stops at 5.5 um, MPS-ATLAS reaches 160 um).
def bolometric_ld_profile(planet, ld_model, interpolate_type="nearest", mu_min=LD_MU_MIN, ld_data_path=LD_DATA):
    from exotic_ld import StellarLimbDarkening
    sld = StellarLimbDarkening(M_H=planet.feh, Teff=planet.T2_eff, logg=planet.logg, ld_model=ld_model,
                               ld_data_path=ld_data_path, interpolate_type=interpolate_type, verbose=0)
    wl, mu = sld.stellar_wavelengths, sld.mus                                                 # [A]
    I_bol = np.trapezoid(sld.stellar_intensities / wl[:, None], wl, axis=0)                   # photons -> energy
    o = np.argsort(mu)
    mu, I_bol = mu[o], I_bol[o] / I_bol[o][-1]
    x = -(1. - mu[mu >= mu_min])
    u_lin = np.dot(x, I_bol[mu >= mu_min] - 1.) / np.dot(x, x)
    T = planet.T2_eff
    f_out = 1. - (planck_fraction_below(T, wl.max() / A_PER_M) - planck_fraction_below(T, wl.min() / A_PER_M)) / 100.
    return mu, I_bol, u_lin, f_out


############################################################################################################################################
#####   SPIDERMAN's stellar grid
############################################################################################################################################

# SPIDERMAN stellar grid [teffs, totals]: totals = in-band surface intensity of the star (stellar_intensity) at each T_eff node,
# integrated exactly like fspiderman's gen_grid (sum_flux over the closed band [l1, l2], response R*lambda/hc from get_filter), on the filter's
# samples refined with the model's. SPIDERMAN cubic-splines totals at p.T_s and extrapolates silently outside the nodes (check_teff_nodes). The grid
# belongs to ONE star (logg, feh) and ONE band, which is why FittingConfig derives it. [K] the whole NewEra cube range on a UNIFORM 100 K grid:
# SPIDERMAN's spline (spline.c) swaps the interval widths of its tridiagonal rows, which only matters for non-uniform knots, so the nodes must stay
# uniform. Every node is a NewEra model except 7100 K (the grid steps by 200 K above 7000 K), which newera_spectrum interpolates linearly in flux
# between 7000 and 7200 K.
TEFF_NODES = np.arange(2300., 7201., 100.)


def stellar_spidergrid(lambda_range, filter_path, logg, feh, teffs=TEFF_NODES, stellar_model=STELLAR_MODEL):
    l1, l2 = map(float, lambda_range)
    return _stellar_spidergrid(l1, l2, filter_path or False, float(logg), float(feh), tuple(float(T) for T in teffs), stellar_model)


# The band integration shared by the star (stellar_spidergrid) and the planet (band_planck): the filter's own samples (or gen_grid's top-hat sampling)
# plus the band edges, and the per-energy response from fspiderman.get_filter (False for a top-hat). Cached and read-only, since every caller shares
# the arrays.
@lru_cache(maxsize=None)
def _band_setup(l1, l2, filter_path):
    filt = sp.get_filter(filter_path) if filter_path else False
    base = np.union1d(np.asarray(filt[0], float) if filt else np.linspace(l1, l2, 1000), [l1, l2])
    base = base[(base >= l1) & (base <= l2)]
    for a in [base] + (list(filt) if filt else []): a.setflags(write=False)
    return base, filt


@lru_cache(maxsize=None)   # hashable arguments only, hence the wrapper above
def _stellar_spidergrid(l1, l2, filter_path, logg, feh, teffs, stellar_model):
    if len(teffs) < 3 or np.any(np.diff(teffs) <= 0): raise ValueError("teffs: >= 3 strictly increasing T_eff nodes (SPIDERMAN's cubic spline)")
    base, filt = _band_setup(l1, l2, filter_path)
    totals = []
    for T in teffs:
        lam = band_wavelengths(base, T, logg, feh, stellar_model)
        totals.append(sp.stellar_grid.sum_flux(lam, stellar_intensity(lam, T, logg, feh, stellar_model), l1, l2, filt))
    grid = [np.array(teffs), np.array(totals)]
    for a in grid: a.setflags(write=False)   # cached and shared by every FittingConfig of this star and band
    return grid


# In-band intensity of a blackbody at T [K], integrated exactly like the stellar grid's totals (same samples, response and sum_flux). The planet side
# of T_bright.
def band_planck(lambda_range, filter_path, T):
    l1, l2 = map(float, lambda_range)
    base, filt = _band_setup(l1, l2, filter_path or False)
    return sp.stellar_grid.sum_flux(base, planck_spectrum(base, T), l1, l2, filt)


# The stellar surface intensity SPIDERMAN uses for a T_s: its cubic spline through the grid (spline_cubic_set with ibcbeg = ibcend = 0, "quadratic
# over the first/last interval", and spline_cubic_val in c_src/spline.c), replicated line by line, including the swapped interval widths of the
# interior rows (a2[i] = A(i,i-1) holds t[i+1]-t[i]), so that T_bright inverts SPIDERMAN's depths exactly. Interpolating the spectrum in T instead
# differs from this by up to ~3e-3 below 3500 K, where the NewEra node integrals are not smooth.
def spidergrid_value(stellar_grid, T_s):
    t, y = (np.asarray(a, float) for a in stellar_grid); n = t.size
    A, b = np.zeros((n, n)), np.zeros(n)
    A[0, 0], A[0, 1] = 1., -1.
    A[-1, -2], A[-1, -1] = -1., 1.
    for i in range(1, n - 1):
        b[i] = (y[i+1] - y[i]) / (t[i+1] - t[i]) - (y[i] - y[i-1]) / (t[i] - t[i-1])
        A[i, i-1], A[i, i], A[i, i+1] = (t[i+1] - t[i]) / 6., (t[i+1] - t[i-1]) / 3., (t[i] - t[i-1]) / 6.
    ypp = np.linalg.solve(A, b)
    i = int(np.clip(np.searchsorted(t, T_s, side="right") - 1, 0, n - 2))   # the interval spline_cubic_val's bisection picks
    dt, h = T_s - t[i], t[i+1] - t[i]
    return float(y[i] + dt * ((y[i+1] - y[i]) / h - (ypp[i+1] / 6. + ypp[i] / 3.) * h + dt * (0.5 * ypp[i] + dt * ((ypp[i+1] - ypp[i]) / (6. * h)))))


def check_teff_nodes(stellar_grid, T_s):
    T_nodes = stellar_grid[0]
    if not T_nodes[0] <= T_s <= T_nodes[-1]: raise ValueError(f"T_s = {T_s:.0f} K outside the stellar grid nodes [{T_nodes[0]:.0f}, {T_nodes[-1]:.0f}] K: SPIDERMAN's spline would extrapolate silently")
