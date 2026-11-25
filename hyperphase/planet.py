"""PlanetParams: the planet, its orbit and its host star."""
from dataclasses import dataclass, fields

import numpy as np
from astropy import constants as const

from hyperphase.defaults import DEFAULT_A_B, DEFAULT_EMISSIVITY, DEFAULT_EMISSIVITY_BOL
from hyperphase.irradiance import insolation_fullyilluminated_hyperillum, naive_insolation


H_PLANCK = const.h.value
C_LIGHT = const.c.value
K_B = const.k_B.value


@dataclass(frozen=True)
class PlanetParams:
    """A planet and its star. r1 is the planet radius, r2 the stellar radius and R the semi-major axis, in any consistent units: only the ratios
    r1/r2 and R/r2 enter. a_AU is the semi-major axis in AU, for the light-travel-time correction. targets.hi_planets_by_name builds one for
    a known planet."""
    name: str                # Exoplanet Name
    period: float       # orbital period [days]
    incl:   float       # inclination [deg]
    ecc:    float       # eccentricity
    w:      float       # argument of periastron [deg]
    r1:     float       # planet radius   [consistent units]
    r2:     float       # stellar radius  [consistent units]
    R:      float       # semi-major axis [consistent units]
    T2_eff: float       # stellar Teff [K]
    a_AU: float         # semi-major axis [AU] (for light travel time correction)
    T14: float  # Transit Duration [h]
    logg: float = 4.5  # stellar surface gravity [log10 cgs], for the stellar spectrum and limb darkening
    feh: float = 0.0  # stellar metallicity [Fe/H] [dex], for the stellar spectrum and limb darkening

    @property
    def args(self):
        return (self.period, self.incl, self.ecc, self.w,
                self.r1, self.r2, self.R, self.T2_eff, self.a_AU)

    def terminator_penumbra(self, unit="rad"):
        """Angle from the substellar point where the fully illuminated zone ends, arccos((r1 + r2)/R), in unit "rad" or "deg"."""
        gp = np.arccos((self.r1 + self.r2) / self.R)
        return np.degrees(gp) if unit == "deg" else gp

    def terminator_nightside(self, unit="rad"):
        """Angle from the substellar point where the penumbra ends and the nightside begins, arccos((r1 - r2)/R), in unit "rad" or "deg"."""
        gp = np.arccos((self.r1 - self.r2) / self.R)
        return np.degrees(gp) if unit == "deg" else gp

    def fraction_illuminated(self, extra=False):
        """Fraction of the surface that receives starlight (Carter 2019); extra=True splits it into the fully illuminated zone and the penumbra."""
        fraction_fullyilluminated = (1. - (self.r1 + self.r2) / self.R) / 2.
        fraction_penumbra = self.r2 / self.R
        if extra:
            return {"fully illuminated": fraction_fullyilluminated, "penumbra": fraction_penumbra}
        return fraction_fullyilluminated + fraction_penumbra

    def substellar_boost(self, insolation=True):
        """Fractional excess of the substellar insolation (insolation=True) or temperature over the point-source value. A distance effect only: the
        substellar point is r1 closer to the star, so the insolation boost is (R/(R - r1))^2 - 1 ~ 2 r1/R."""
        if insolation: return (insolation_fullyilluminated_hyperillum(0, self)/naive_insolation(0, self)) - 1
        else: return (insolation_fullyilluminated_hyperillum(0, self)/naive_insolation(0, self))**0.25 - 1

    def approx_occ_depth(self, eff_wavelength_range, A_B=DEFAULT_A_B, f=2/3, emissivity_bol=DEFAULT_EMISSIVITY_BOL, emissivity=DEFAULT_EMISSIVITY, A_g=0.0, n_wl=400):
        """Approximate (point-source) occultation depth [ppm] of a grey planet around a blackbody star: emissivity (r1/r2)^2 <B(T_day)> / <B(T2_eff)>
        + A_g (r1/R)^2, with <.> the photon-weighted top-hat average over eff_wavelength_range [m]. f is the redistribution factor of Cowan & Agol
        (2011): 1/4 full redistribution, 1/2 uniform dayside, 2/3 no redistribution (disk-integrated), 1 substellar."""
        wl_min, wl_max = eff_wavelength_range # [m]

        # Point-source value ignoring hyper-illumination.
        T1_day = self.T2_eff * np.sqrt(self.r2 / self.R) * (f * (1.0 - A_B) / emissivity_bol) ** 0.25

        wl = np.linspace(wl_min, wl_max, n_wl)
        B = lambda T: (2 * H_PLANCK * C_LIGHT ** 2 / wl ** 5) / np.expm1(H_PLANCK * C_LIGHT / (wl * K_B * T))

        contrast = np.trapezoid(wl * B(T1_day), wl) / np.trapezoid(wl * B(self.T2_eff), wl)
        return 1e6 * (emissivity * (self.r1 / self.r2) ** 2 * contrast + A_g * (self.r1 / self.R) ** 2)

    def summary(self, print_params=False, eff_wavelength_range=None):
        print(f"=== {self.name} ===")
        if print_params:
            for f in fields(self):
                print(f"{f.name:>8} = {getattr(self, f.name)}")
            print("--- derived ---")
        print(f"rp = r1/r2           = {self.r1 / self.r2:.4f}")
        print(f"a  = R/r2            = {self.R / self.r2:.3f}")
        print(f"terminator_penumbra  = {self.terminator_penumbra('deg'):.1f} deg")
        print(f"terminator_nightside = {self.terminator_nightside('deg'):.1f} deg")
        illuminated = self.fraction_illuminated(extra=True)
        print(f"fraction_illuminated = {100*self.fraction_illuminated():.1f} %  (fully illuminated: {100*illuminated['fully illuminated']:.1f} %, penumbra: {100*illuminated['penumbra']:.1f} %)")
        print(f"substellar_boost     = {100*self.substellar_boost():.5f} % (insolation), {100*self.substellar_boost(insolation=False):.5f} % (temperature)")
        if eff_wavelength_range is not None:
            print(f"approx_occ_depth     = {self.approx_occ_depth(eff_wavelength_range):.1f} ppm")
