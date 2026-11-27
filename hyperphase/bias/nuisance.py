"""Robustness to nuisance parameters: the free geometry, the stellar effective temperature and the ellipsoidal terms, as one linear least-squares
problem around the noiseless fit.

The fixed-geometry fit assumes the geometry and no other phase-curve terms; a real analysis marginalises over them, and each nuisance is one more
column of the same problem. Linearised around the noiseless point-source fit, whitened and with each column's mean removed per detector block (the
free baseline): y = r_perp (evidence_residual), X = [j_A, (j_F,) j_nu1, ...] with j = d(model) / d(theta) / sigma_bin. A Gaussian prior sigma_p on
a nuisance shift adds the penalty (delta_nu / sigma_p)^2, a flat prior none:
    delta = argmin |y - X delta|^2 + sum (delta_nu / sigma_p)^2,   Cov = (X'X + Lambda)^-1,   sigma_A_B = sqrt(Cov_AA),   chi2_min' = the minimum.
If the extended-source model carries the same nuisances with the same priors, <ln B> = chi2_min' / 2 to first order (the determinant terms
cancel).

The columns (nuisance_problem):
- "a_rs", "rp_rs", "incl": central differences of the point-source model, whose transit and occultation constrain the geometry too.
- "b", "a_rs|b": the impact parameter and a/R* at fixed b (i = arccos(b / (a/R*))), for a source that fitted b and derived i; from the a_rs and
  incl columns by the chain rule. A prior on "a_rs" also applies to "a_rs|b".
- "stellar:<k>": the star's ellipsoidal variation, -cos(4 pi phi) [ppm] in every bin (the star stays visible in occultation), one free amplitude
  per detector. Free per band, it can follow whatever it absorbs: the worst case.
- "stellar_shared": the same term with one amplitude for all detectors, scaled by band ratios (stellar_shared_column). A_ell depends on the band
  only through alpha_ell(u, g) (ellipsoidal_amplitude), by ~10 % across the infrared, so this is its physical form, and with the prior of
  ellipsoidal_prior, centred on the truth (no ellipsoidal term in truth or model), the realistic one.
- "planetary": the planet's tidal deformation (planetary_area_factor), linearised: F_p(phi) sin^2 i sin^2(2 pi phi), with F_p the point-source
  planet flux at the fitted A_B. One amplitude for all detectors (it is geometric); the term vanishes in occultation (before the mean is
  removed), where the residual does not, so it resembles the residual less than the stellar term.
- "T_s": the stellar T_eff [K] at fixed log g and [Fe/H] (step TEFF_STEP). The planet's temperature scales with it, so it acts like
  (1 - A_B)^(1/4), up to the part that cancels against the stellar band flux in F_p / F_star.
redist=True linearises the fit with redistribution around its own solution (A_B_fit, F_fit > 0; extra column j_F): the A_B-only fit sits at the
F = 0 edge, where the response to F switches on too flat to linearise."""
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from hyperphase.bands import band_of
from hyperphase.bias.detectors import is_detector_set
from hyperphase.bias.evidence import evidence_residual
from hyperphase.bias.fisher import whitened_jacobian
from hyperphase.bias.fits import _chi2_problem, binned, fit_naive, fit_redist
from hyperphase.defaults import A_B_FIT_BOUNDS, F_BOUNDS
from hyperphase.model import model_pcurve, model_times


############################################################################################################################################
#####   Physical ranges of the ellipsoidal terms
############################################################################################################################################
M_EARTH_SUN = 3.0035e-6   # M_Earth / M_Sun (IAU 2015 nominal values)

# Brackets of the star's infrared linear limb-darkening (u) and gravity-darkening (g) coefficients over which alpha_ell is evaluated
# (ellipsoidal_prior; K dwarfs, not from a table).
ELLIPSOIDAL_U_RANGE, ELLIPSOIDAL_G_RANGE = (0.1, 0.3), (0.1, 0.4)

# Range of the fluid Love number h2 of a rocky planet's tidal deformation, from all mass at the centre (1) to a homogeneous body (2.5); rocky planets
# have h2 ~ 2 (Correia 2014, A&A 570, L5, after Yoder 1995).
DEFORMATION_H2_RANGE = (1., 2.5)


############################################################################################################################################
#####   Robustness to nuisance parameters: one linear least-squares problem
############################################################################################################################################

# finite-difference steps of the geometry columns [a/R*, Rp/R*, deg] (nuisance_problem(steps=...))
NUISANCE_STEPS = dict(a_rs=0.005, rp_rs=1e-4, incl=0.2)
TEFF_STEP = 10.   # finite-difference step of the "T_s" column [K]
NUISANCE_LABELS = {"a_rs": r"$a/R_\star$", "rp_rs": r"$R_p/R_\star$", "incl": r"$i$", "b": r"$b$", "a_rs|b": r"$a/R_\star$ (fixed $b$)", "stellar": "stellar ellipsoidal (free per band)", "stellar_shared": "stellar ellipsoidal (shared)", "planetary": "planetary deformation", "T_s": r"$T_{\rm eff}$"}


def _geometry_model(cfg, A_B_, param, value, kw, insolation_model="naive", F=None):
    """The whitened, mean-removed binned point-source model (F for "naive_redist") at a perturbed geometry or stellar temperature: the building block
    of a column. The caches insol and refl_unit belong to the unperturbed system and are dropped."""
    kw = {k: v for k, v in kw.items() if k not in ("insol", "refl_unit")}
    if param == "incl":
        cfg = [replace(c, planet=replace(c.planet, incl=value)) for c in cfg] if is_detector_set(cfg) else replace(cfg, planet=replace(cfg.planet, incl=value))
    else:
        kw = {**kw, param: value}
    resid, _, sigma = _chi2_problem(cfg, None, insolation_model, **kw)
    return -resid(A_B_, F) / sigma


def nuisance_problem(cfg, truth, A_B_fit=None, priors=None, steps=NUISANCE_STEPS, bounds=A_B_FIT_BOUNDS, stellar_ratios=None, redist=False, F_fit=None, F_bounds=F_BOUNDS, **kw):
    """Everything the nuisance fits of one configuration share: y = r_perp of the noiseless fit, the columns of A_B (and F) and every nuisance column
    (see the module docstring), whitened and mean-removed per detector block. redist=False: around the A_B-only fit (A_B_fit=None fits it);
    redist=True: around the fit with redistribution (fit_redist unless A_B_fit and F_fit are given). priors: {name: Gaussian sigma} (a missing
    name is flat). stellar_ratios: see _stellar_ratios. Returns dict(A_B_fit, chi2, y, j_A, columns, priors, values [the system's geometry and
    T_s], sigma, blocks, phase, stellar_ratios), with redist also F_fit and j_F."""
    model = "naive_redist" if redist else "naive"
    if redist and (A_B_fit is None or F_fit is None): A_B_fit, F_fit = fit_redist(cfg, truth, bounds=bounds, F_bounds=F_bounds, **kw)[:2]
    F = float(F_fit) if redist else None
    ev = evidence_residual(cfg, truth, model, A_B_fit=A_B_fit, F=F, bounds=bounds, **kw)
    A = ev["A_B_fit"]
    cfgs = list(cfg) if is_detector_set(cfg) else [cfg]
    pl = cfgs[0].planet
    values = dict(a_rs=pl.R / pl.r2, rp_rs=pl.r1 / pl.r2, incl=pl.incl, T_s=pl.T2_eff)
    steps = {"T_s": TEFF_STEP, **steps}
    cols = {}
    for name, v in values.items():
        h = steps[name]
        cols[name] = (_geometry_model(cfg, A, name, v + h, kw, model, F) - _geometry_model(cfg, A, name, v - h, kw, model, F)) / (2. * h)
    # b and a/R* at fixed b from the incl column [per deg], by the chain rule on i = arccos(b / a) [deg]:
    #     di/db = -(180/pi) / (a sin i),   di/da|_b = (180/pi) b / (a^2 sin i)
    a0, i0 = values["a_rs"], np.radians(values["incl"])
    values.update({"b": a0 * np.cos(i0), "a_rs|b": a0})
    cols["b"] = -np.degrees(1.) / (a0 * np.sin(i0)) * cols["incl"]
    cols["a_rs|b"] = cols["a_rs"] + np.degrees(1.) * values["b"] / (a0 ** 2 * np.sin(i0)) * cols["incl"]
    priors = dict(priors or {})
    if "a_rs" in priors: priors.setdefault("a_rs|b", priors["a_rs"])   # one prior on a/R*, whichever of i and b is held
    planetary = []
    for k, (c, blk) in enumerate(zip(cfgs, ev["blocks"])):
        t_ = model_times(c.planet.period, kw.get("n_t") or 50 * c.n_bins)
        ph = 2. * np.pi * t_ / c.planet.period
        ell = -binned(c, np.cos(2. * ph))                                    # -cos(4 pi phi), bin-averaged [per ppm of amplitude]
        col = np.zeros(ev["r_perp"].size)
        col[blk] = (ell - ell.mean()) / c.sigma_bin
        cols[f"stellar:{k + 1}"] = col
        fp = model_pcurve(replace(c, transit=False), model, A, F=F, **kw)[1]   # the fitted model's planet flux [ppm], zero in occultation
        # d[F_p A(phi) / A(0)] / d delta at delta = 0 (planetary_area_factor)
        d = binned(c, fp * np.sin(np.radians(c.planet.incl)) ** 2 * np.sin(ph) ** 2)
        planetary.append((d - d.mean()) / c.sigma_bin)
    cols["planetary"] = np.concatenate(planetary)
    ratios = _stellar_ratios(cfgs, stellar_ratios)
    cols["stellar_shared"] = stellar_shared_column(cols, ratios)
    J = whitened_jacobian(cfg, model, A, F=F, bounds=bounds, F_bounds=F_bounds, **kw)
    out = dict(A_B_fit=A, chi2=ev["chi2"], y=ev["r_perp"], j_A=J[:, 0], columns=cols, priors=priors, values=values, sigma=ev["sigma"], blocks=ev["blocks"], phase=ev["phase"], stellar_ratios=ratios)
    if redist: out.update(F_fit=F, j_F=J[:, 1])
    return out


def _stellar_ratios(cfgs, stellar_ratios):
    """Band ratios of the shared stellar term, one per detector: None -> 1 for all (achromatic); {band key: ratio} -> each detector's band (1 where
    missing); a sequence -> as given, in detector order."""
    if stellar_ratios is None: return np.ones(len(cfgs))
    r = np.array([float(stellar_ratios.get(band_of(c), 1.)) for c in cfgs]) if isinstance(stellar_ratios, dict) else np.asarray(stellar_ratios, float).ravel()
    if r.size != len(cfgs) or not np.all(np.isfinite(r) & (r > 0.)): raise ValueError(f"stellar_ratios: {len(cfgs)} positive values needed, got {r}")
    return r


def stellar_shared_column(columns, ratios):
    """The stellar ellipsoidal column with one amplitude for all detectors: the columns "stellar:<k>" weighted by `ratios` (each band's alpha_ell
    relative to the reference) and summed; each is zero outside its own block."""
    names = sorted((c for c in columns if c.startswith("stellar:")), key=lambda c: int(c.split(":")[1]))
    r = np.asarray(ratios, float).ravel()
    if r.size != len(names): raise ValueError(f"{len(names)} detectors but {r.size} ratios")
    return np.sum([rk * columns[n] for rk, n in zip(r, names)], axis=0)


def planetary_area_factor(delta, incl, phi):
    """Projected area of the tidally deformed planet relative to mid-transit, A(phi) / A(0) (Akinsanmi et al. 2024, A&A 682, A15, eqs. 2 and 11;
    Leconte et al. 2011): the triaxial ellipsoid of Correia (2014), a1 = a2 (1 + 3q), a3 = a2 (1 - q), a1 the semi-axis toward the star, a2 the
    one along the orbit, a3 the polar one, with amplitude delta = (a1 - a2) / a2 = 3q (xi in the paper), at the orbital angle phi [rad,
    0 = mid-transit] and inclination incl [deg]. To first order 1 + delta sin^2 i sin^2 phi, the "planetary" column of nuisance_problem."""
    q = delta / 3.
    a1, a2, a3 = 1. + 3. * q, 1., 1. - q
    ci2, si2 = np.cos(np.radians(incl)) ** 2, np.sin(np.radians(incl)) ** 2
    area = lambda ph: np.sqrt(a1 ** 2 * a2 ** 2 * ci2 + a3 ** 2 * si2 * (a1 ** 2 * np.sin(ph) ** 2 + a2 ** 2 * np.cos(ph) ** 2))
    return area(np.asarray(phi, float)) / area(0.)


def nuisance_names(problem, nuisances):
    """The columns a nuisance selection stands for: "stellar" expands to one amplitude per detector."""
    out = []
    for n in nuisances:
        out += [c for c in problem["columns"] if c.startswith("stellar:")] if n == "stellar" else [n]
    return out


def nuisance_solve(problem, nuisances=(), priors=None, time_factor=1., limits=None):
    """One nuisance fit of `problem` (the A_B-only fit, or the fit with redistribution if it has j_F) with the columns of `nuisances` added (() = the
    fixed-geometry fit). priors: {name: sigma} overriding the problem's per name (None = flat). time_factor: the observing time relative to the
    problem's; the data rows scale with its square root but the prior rows do not, so a fit with priors does not scale as sqrt(time). limits:
    {name: (lo, hi)} hard limits on a nuisance's shift from its true value (bounded least squares); a nuisance that ends on a limit is held there
    (sigma_post 0, listed in on_limit) and the covariance is that of the others. Returns the shifted A_B_fit (and F_fit), sigma_A_B (and sigma_F),
    chi2, lnB_mean = chi2 / 2 (an expected ln B only for the A_B-only fit), per nuisance its shift (delta), sigma_post and
    shrinkage = sigma_post / sigma_prior (NaN for a flat prior), the augmented residual r_perp (without the prior rows) and cov."""
    names = nuisance_names(problem, nuisances)
    pri = {**problem["priors"], **(priors or {})}
    J = [problem["j_A"]] + ([problem["j_F"]] if "j_F" in problem else [])
    k, s = len(J), np.sqrt(time_factor)
    X, y = s * np.column_stack(J + [problem["columns"][n] for n in names]), s * problem["y"]
    prior = np.array([np.nan] * k + [np.nan if pri.get(n) is None else pri[n] for n in names], float)
    lam = np.where(np.isfinite(prior), 1. / prior ** 2, 0.)
    free = np.ones(X.shape[1], bool)
    if limits:   # bounded least squares, the priors as penalty rows
        lo = np.array([-np.inf] * k + [limits.get(n, (-np.inf, np.inf))[0] for n in names])
        hi = np.array([np.inf] * k + [limits.get(n, (-np.inf, np.inf))[1] for n in names])
        delta = lsq_linear(np.vstack([X, np.diag(np.sqrt(lam))]), np.concatenate([y, np.zeros(len(lam))]), bounds=(lo, hi), method="bvls").x
        free = ~(np.isclose(delta, lo) | np.isclose(delta, hi))
    cov = np.zeros((X.shape[1], X.shape[1]))
    cov[np.ix_(free, free)] = np.linalg.inv(X[:, free].T @ X[:, free] + np.diag(lam[free]))
    if not limits: delta = cov @ (X.T @ y)
    r = y - X @ delta
    chi2 = float(r @ r + np.sum(lam * delta ** 2))
    sd = np.sqrt(np.diag(cov))
    out = dict(sigma_A_B=float(sd[0]), delta_A_B=float(delta[0]), A_B_fit=float(problem["A_B_fit"] + delta[0]), chi2=chi2, lnB_mean=0.5 * chi2, names=names,
               delta=dict(zip(names, delta[k:])), sigma_post=dict(zip(names, sd[k:])), shrinkage=dict(zip(names, sd[k:] / prior[k:])), r_perp=r, cov=cov,
               on_limit=[n for n, f in zip(names, free[k:]) if not f])
    if k == 2: out.update(F_fit=float(problem["F_fit"] + delta[1]), sigma_F=float(sd[1]))
    return out


def nuisance_both_fits(problem, problem_redist, A_B_true, nuisances=(), priors=None, time_factor=1., limits=None, uniform=None):
    """Both point-source fits under the same nuisances: `problem` (A_B-only) and `problem_redist` (nuisance_problem(..., redist=True)), solved by
    nuisance_solve. Returns bias, sigma, bias_sigma and lnB_mean = chi2' / 2 of the A_B-only fit; bias_J, sigma_J, bias_sigma_J and F_fit of the
    fit with redistribution; z_F = sqrt(chi2'(A_B) - chi2'(A_B, F)), the significance of the spurious F with the nuisances in both fits; and the
    solutions fit, fit_J. Where the shifted solution has F <= 0, the fit with redistribution is the A_B-only fit (F_fit = 0, z_F = 0). limits: the
    hard limits of nuisance_solve, in both fits (the best fits); uniform: a flat prior instead (nuisance_both_fits_uniform)."""
    if uniform:
        if limits: raise ValueError("nuisance_both_fits: give a nuisance either `limits` (best fit) or `uniform` (posterior), not both")
        return nuisance_both_fits_uniform(problem, problem_redist, A_B_true, nuisances, priors, time_factor, uniform)
    a, f = nuisance_solve(problem, nuisances, priors, time_factor, limits), nuisance_solve(problem_redist, nuisances, priors, time_factor, limits)
    if f["F_fit"] <= 0.: f = {**a, "F_fit": 0., "sigma_F": np.nan}
    bias, bias_J = a["A_B_fit"] - A_B_true, f["A_B_fit"] - A_B_true
    return dict(bias=bias, sigma=a["sigma_A_B"], bias_sigma=abs(bias) / a["sigma_A_B"], lnB_mean=0.5 * a["chi2"], bias_J=bias_J, sigma_J=f["sigma_A_B"], bias_sigma_J=abs(bias_J) / f["sigma_A_B"],
                F_fit=f["F_fit"], z_F=float(np.sqrt(max(a["chi2"] - f["chi2"], 0.))), fit=a, fit_J=f)


def bounded_posterior(sol, name, lo, hi, peak=None, n_nodes=96):
    """Posterior of the linear nuisance problem when the nuisance `name` has a flat prior on its shift within (lo, hi), exact for the linear-Gaussian
    problem. `sol`: the nuisance_solve result with that nuisance free (no prior, no limits). The likelihood (with the Gaussian priors of the other
    nuisances) is Gaussian in all parameters, so the shift x of `name` has the marginal N(x_hat, C_xx), which the prior cuts to (lo, hi), and
    given x every other parameter p is N(p_hat + beta_p (x - x_hat), C_pp - beta_p C_px), beta_p = C_px / C_xx. peak: where the likelihood peaks,
    (A_B[, F], every nuisance shift); default: the solution of `sol`. Returns mass (the share of N(x_hat, C_xx) within (lo, hi): the factor by
    which the range cuts the evidence against an unbounded prior of the same density), x_mean and x_sd of the shift, median and sd of A_B (and F;
    medians by root finding on the mixture), and delta, sigma_post (posterior mean and sd of every nuisance shift)."""
    from scipy.stats import norm, truncnorm
    from scipy.optimize import brentq
    k = 2 if "F_fit" in sol else 1
    ix = k + sol["names"].index(name)
    C = sol["cov"]
    full = np.array([sol["A_B_fit"]] + ([sol["F_fit"]] if k == 2 else []) + [sol["delta"][n] for n in sol["names"]] if peak is None else peak, float)
    if full.size != C.shape[0]: raise ValueError(f"bounded_posterior: peak needs {C.shape[0]} values, got {full.size}")
    x0 = full[ix]
    sx = np.sqrt(C[ix, ix])
    a, b = (lo - x0) / sx, (hi - x0) / sx
    tn = truncnorm(a, b, loc=x0, scale=sx)
    xm, xsd = float(tn.mean()), float(tn.std())
    beta = C[:, ix] / C[ix, ix]
    mean = full + beta * (xm - x0)
    sd = np.sqrt(np.maximum(np.diag(C) - beta * C[:, ix], 0.) + beta ** 2 * xsd ** 2)
    u, w = np.polynomial.legendre.leggauss(n_nodes)   # Gauss-Legendre nodes in the cumulative probability of the truncated x
    xs, w = tn.ppf(0.5 * (u + 1.)), 0.5 * w
    median = []
    for j in range(k):
        sc = np.sqrt(max(C[j, j] - beta[j] * C[j, ix], 0.))
        m = full[j] + beta[j] * (xs - x0)
        median.append(brentq(lambda v: float(np.sum(w * norm.cdf((v - m) / sc))) - 0.5, mean[j] - 10. * sd[j], mean[j] + 10. * sd[j], xtol=1e-12))
    return dict(mass=float(norm.cdf(b) - norm.cdf(a)), x_mean=xm, x_sd=xsd, median=median, sd=list(sd[:k]),
                delta=dict(zip(sol["names"], mean[k:])), sigma_post=dict(zip(sol["names"], sd[k:])))


def nuisance_both_fits_uniform(problem, problem_redist, A_B_true, nuisances, priors, time_factor, uniform):
    """nuisance_both_fits with a flat prior on the shift of one nuisance, uniform = {name: (lo, hi)}: what a retrieval with that prior gives, exact
    for the linear problem (bounded_posterior). The best fit with the range as hard limits ends on a limit wherever the unbounded fit lies beyond
    it; the posterior does not, it spreads over the part of the range where the likelihood is not yet small. The range then cuts off likelihood of
    the point-source model, while the extended-source model, which peaks at the truth inside the range, keeps most of its own, so the Occam
    factors do not cancel and chi2'/2 of the bounded best fit is not the expected ln B:
        <ln B> = chi2'_free / 2 + ln(mass_ext / mass_ps),
    with chi2'_free the fit with the nuisance flat and unbounded, and mass each model's likelihood share within the range. bias: the posterior
    median of A_B of the point-source fit minus that of the extended-source model (which moves off the truth where the range is asymmetric about
    it); sigma: the posterior sd of A_B. The same for the fit with redistribution (bias_J, sigma_J; F_fit its posterior median). z_F stays the
    likelihood-ratio statistic of the best fits within the limits; where the best fit with redistribution has F <= 0 it is the A_B-only fit.
    Returns the keys of nuisance_both_fits (fit, fit_J with posterior medians, sds and means), plus best (nuisance_both_fits with the range as
    hard limits) and posterior (bounded_posterior of ps, ps_J and ext)."""
    (name, (lo, hi)), = _one_uniform(uniform)
    pri = {**(priors or {}), name: None}   # flat: the range is the prior
    best = nuisance_both_fits(problem, problem_redist, A_B_true, nuisances, pri, time_factor, limits=uniform)
    u = nuisance_uniform(problem, A_B_true, nuisances, uniform, priors, time_factor)
    fit, ps, ext = u["fit"], u["posterior"]["ps"], u["posterior"]["ext"]
    if best["F_fit"] <= 0.: fit_J, ps_J = {**fit, "F_fit": 0., "sigma_F": np.nan}, ps
    else:
        f = nuisance_solve(problem_redist, nuisances, pri, time_factor)
        ps_J = bounded_posterior(f, name, lo, hi); fit_J = _posterior_solution(f, ps_J)
    bias_J = fit_J["A_B_fit"] - ext["median"][0]
    return dict(bias=u["bias"], sigma=u["sigma"], bias_sigma=u["bias_sigma"], lnB_mean=u["lnB_mean"],
                bias_J=bias_J, sigma_J=fit_J["sigma_A_B"], bias_sigma_J=abs(bias_J) / fit_J["sigma_A_B"], F_fit=fit_J["F_fit"], z_F=best["z_F"], fit=fit, fit_J=fit_J,
                best=best, posterior=dict(ps=ps, ps_J=ps_J, ext=ext))


def nuisance_uniform(problem, A_B_true, nuisances, uniform, priors=None, time_factor=1.):
    """The A_B-only half of nuisance_both_fits_uniform (see there): the posterior of `problem` with a flat prior on the shift of one nuisance,
    uniform = {name: (lo, hi)}. Returns bias, sigma, bias_sigma, lnB_mean, fit (the posterior solution), free (the solve with that nuisance flat
    and unbounded) and posterior (bounded_posterior of ps and ext)."""
    (name, (lo, hi)), = _one_uniform(uniform)
    a = nuisance_solve(problem, nuisances, {**(priors or {}), name: None}, time_factor)
    ps, ext = bounded_posterior(a, name, lo, hi), bounded_posterior(a, name, lo, hi, peak=[A_B_true] + [0.] * len(a["names"]))
    fit = _posterior_solution(a, ps)
    bias = fit["A_B_fit"] - ext["median"][0]
    return dict(bias=bias, sigma=fit["sigma_A_B"], bias_sigma=abs(bias) / fit["sigma_A_B"], lnB_mean=0.5 * a["chi2"] + float(np.log(ext["mass"] / ps["mass"])),
                fit=fit, free=a, posterior=dict(ps=ps, ext=ext))


def _one_uniform(uniform):
    if len(uniform) != 1: raise ValueError(f"uniform: one nuisance with a flat bounded prior, got {list(uniform)}")
    return uniform.items()


def _posterior_solution(s, p):
    """The nuisance_solve result s with the posterior p of bounded_posterior in place of the best fit (shrinkage from the posterior sd)."""
    prior_sd = {n: s["sigma_post"][n] / s["shrinkage"][n] for n in s["names"]}
    return {**s, "A_B_fit": p["median"][0], "sigma_A_B": p["sd"][0], "delta": p["delta"], "sigma_post": p["sigma_post"], "on_limit": [], "posterior": True,
            "shrinkage": {n: p["sigma_post"][n] / prior_sd[n] for n in s["names"]}, **({"F_fit": p["median"][1], "sigma_F": p["sd"][1]} if len(p["median"]) == 2 else {})}


def nuisance_table(problem, A_B_true, rows):
    """Nuisance table of one configuration, one row per entry of `rows` ({label: tuple of nuisances, or dict(nuisances=..., uniform={name: (lo, hi)})
    for a flat bounded prior on one of them, nuisance_uniform}): sigma_A_B, bias_A_B, bias_sigma, chi2 and lnB_mean, and per nuisance: for the
    geometry its shrinkage sigma_post / sigma_prior ("shrink <name>") or, with a flat prior, its fitted shift and sigma ("<name>", "sig <name>");
    for an amplitude its fitted value and sigma (stellar in ppm, planetary as delta = (a1 - a2) / a2), plus the shrinkage with a prior. A row with
    a flat bounded prior gives posterior quantities (bias the difference of the posterior medians, chi2 that of the unbounded solve)."""
    out = []
    for label, nuis in rows.items():
        if isinstance(nuis, dict):
            u = nuisance_uniform(problem, A_B_true, nuis["nuisances"], nuis["uniform"])
            s = u["fit"]
            row = {"row": label, "sigma_A_B": u["sigma"], "bias_A_B": u["bias"], "bias_sigma": u["bias_sigma"], "chi2": u["free"]["chi2"], "lnB_mean": u["lnB_mean"]}
        else:
            s = nuisance_solve(problem, nuis)
            row = {"row": label, "sigma_A_B": s["sigma_A_B"], "bias_A_B": s["A_B_fit"] - A_B_true, "bias_sigma": abs(s["A_B_fit"] - A_B_true) / s["sigma_A_B"], "chi2": s["chi2"], "lnB_mean": s["lnB_mean"]}
        for n in s["names"]:
            shrink = s["shrinkage"][n]
            if n in problem["values"]:   # geometry: shrinkage, or shift and sigma under a flat prior
                if np.isfinite(shrink): row[f"shrink {n}"] = shrink
                else: row[n], row[f"sig {n}"] = s["delta"][n], s["sigma_post"][n]
            else:   # amplitude (zero in the truth): value and sigma, plus shrinkage with a prior
                row[n], row[f"sig {n}"] = s["delta"][n], s["sigma_post"][n]
                if np.isfinite(shrink): row[f"shrink {n}"] = shrink
        out.append(row)
    return pd.DataFrame(out)


def nuisance_fit_nonlinear(cfg, truth, nuisances, priors=None, A_B_start=None, bounds=A_B_FIT_BOUNDS, stellar_ratios=None, redist=False, F_bounds=F_BOUNDS, limits=None, **kw):
    """Nonlinear check of nuisance_solve: scipy least_squares over (A_B, [F,] the nuisances), the point-source model re-evaluated at every step (the
    planetary term with the exact planetary_area_factor), each detector's baseline removed by its mean and the Gaussian priors as residual rows
    (theta - theta_0) / sigma_p. With redist the fit starts at the noiseless solution of fit_redist, where F > 0: a start at F = 0 would stay
    there (see fit_redist). Sigmas from the Jacobian at the solution. limits: as in nuisance_solve, as solver bounds. Returns dict(A_B_fit,
    sigma_A_B, chi2, values, sigma_post, nfev, on_limit), with redist also F_fit and sigma_F."""
    from scipy.optimize import least_squares
    model = "naive_redist" if redist else "naive"
    cfgs = list(cfg) if is_detector_set(cfg) else [cfg]
    truths = list(truth) if is_detector_set(cfg) else [truth]
    pl = cfgs[0].planet
    start = dict(a_rs=pl.R / pl.r2, rp_rs=pl.r1 / pl.r2, incl=pl.incl, T_s=pl.T2_eff)
    start.update({"b": start["a_rs"] * np.cos(np.radians(pl.incl)), "a_rs|b": start["a_rs"]})   # b: i follows from b and a/R* (see nuisance_problem)
    names = []
    for n in nuisances: names += [f"stellar:{k + 1}" for k in range(len(cfgs))] if n == "stellar" else [n]
    priors = dict(priors or {})
    if "a_rs" in priors: priors.setdefault("a_rs|b", priors["a_rs"])
    ratios = _stellar_ratios(cfgs, stellar_ratios)
    yb = [binned(c, t) for c, t in zip(cfgs, truths)]
    # for the solver's scaling only: a typical step per parameter (stellar amplitudes: 1 ppm)
    typical = {"a_rs": 0.04, "a_rs|b": 0.04, "rp_rs": 3e-4, "incl": 1., "b": 0.03, "T_s": 50., "planetary": 0.02}
    k0 = 2 if redist else 1   # the first nuisance in theta = (A_B, [F,] nuisances)
    scale = np.array([0.05] + [0.005] * redist + [priors.get(n, typical.get(n, 1.)) for n in names])

    def residuals(theta):
        p = dict(zip(names, theta[k0:]))
        F = float(np.clip(theta[1], *F_bounds)) if redist else None
        geom = {k: p[k] for k in ("a_rs", "rp_rs", "T_s") if k in p}
        if "a_rs|b" in p: geom["a_rs"] = p["a_rs|b"]
        out = []
        for k, (c, y) in enumerate(zip(cfgs, yb)):
            if "incl" in p: c = replace(c, planet=replace(c.planet, incl=p["incl"]))
            elif "b" in p: c = replace(c, planet=replace(c.planet, incl=float(np.degrees(np.arccos(p["b"] / geom.get("a_rs", start["a_rs"]))))))
            t_, m = model_pcurve(c, model, theta[0], F=F, **geom, **kw)
            ph = 2. * np.pi * t_ / c.planet.period
            if "planetary" in p:
                fp = model_pcurve(replace(c, transit=False), model, theta[0], F=F, **geom, **kw)[1]
                # the exact projected-area factor, not its linearisation
                m = m + fp * (planetary_area_factor(p["planetary"], c.planet.incl, ph) - 1.)
            if f"stellar:{k + 1}" in p: m = m - p[f"stellar:{k + 1}"] * np.cos(2. * ph)
            if "stellar_shared" in p: m = m - p["stellar_shared"] * ratios[k] * np.cos(2. * ph)
            r = y - binned(c, m)
            out.append((r - r.mean()) / c.sigma_bin)
        out.append(np.array([(p[n] - start.get(n, 0.)) / priors[n] for n in names if n in priors]))   # amplitudes: prior centred on 0, the truth
        return np.concatenate(out)

    if redist: x0 = list(fit_redist(cfg, truth, bounds=bounds, F_bounds=F_bounds, **kw)[:2])
    else: x0 = [A_B_start if A_B_start is not None else fit_naive(cfg, truth, bounds=bounds, **kw)[0]]
    theta0 = np.array(x0 + [start.get(n, 0.) for n in names])
    solver_bounds = {}
    if limits:   # the limits are shifts from the true value (the start of each nuisance)
        lo = np.array([-np.inf] * k0 + [start.get(n, 0.) + limits.get(n, (-np.inf, np.inf))[0] for n in names])
        hi = np.array([np.inf] * k0 + [start.get(n, 0.) + limits.get(n, (-np.inf, np.inf))[1] for n in names])
        solver_bounds = dict(bounds=(lo, hi))
        theta0 = np.clip(theta0, lo, hi)   # a start outside the limits moves onto the nearest one
    res = least_squares(residuals, theta0, x_scale=scale, diff_step=1e-3, xtol=1e-10, ftol=1e-12, gtol=1e-12, **solver_bounds)
    free = np.ones(len(theta0), bool)
    # a parameter on a limit is held there (covariance of the others); the solver stops just inside, so within 0.1 % of the range counts as on it
    if limits:
        width = np.where(np.isfinite(hi - lo), hi - lo, 0.)
        free = (res.x - lo > 1e-3 * width) & (hi - res.x > 1e-3 * width)
    sd = np.zeros(len(theta0))
    sd[free] = np.sqrt(np.diag(np.linalg.inv(res.jac[:, free].T @ res.jac[:, free])))
    out = dict(A_B_fit=float(res.x[0]), sigma_A_B=float(sd[0]), chi2=float(2. * res.cost), values=dict(zip(names, res.x[k0:])), sigma_post=dict(zip(names, sd[k0:])), nfev=int(res.nfev),
               on_limit=[n for n, fr in zip(names, free[k0:]) if not fr])
    if redist: out.update(F_fit=float(np.clip(res.x[1], *F_bounds)), sigma_F=float(sd[1]))
    return out


def ellipsoidal_amplitude(planet, mass_ratio, u, g):
    """Expected semi-amplitude [ppm] of the star's ellipsoidal variation, A_ell = alpha_ell (M_p / M_star) (R_star / a)^3 sin^2 i with
    alpha_ell = 0.15 (15 + u) (1 + g) / (3 - u) (Morris & Naftilan 1993; Shporer 2017, PASP 129, 072001, eqs. 7-8). u, g: the band's linear limb-
    and gravity-darkening coefficients; mass_ratio = M_p / M_star."""
    alpha = 0.15 * (15. + u) * (1. + g) / (3. - u)
    return 1e6 * alpha * mass_ratio * (planet.r2 / planet.R) ** 3 * np.sin(np.radians(planet.incl)) ** 2


def ellipsoidal_prior(planet, mass_ratio, frac_q, u_range, g_range, sig_a_rs=0., sig_incl=0.):
    """Prior of the shared stellar term: the expected semi-amplitude A [ppm] (the mean of ellipsoidal_amplitude at the two ends of the (u, g)
    bracket) and its fractional 1-sigma error, the quadrature sum of those of M_p / M_star (frac_q), (R_star / a)^3 (3 sig_a_rs / (a/R*)),
    sin^2 i (2 sig_incl / tan i, sig_incl in deg) and alpha_ell (half the bracket over its middle, taken as 1 sigma: generous for a bracket).
    Errors independent (the M_star - a/R* correlation through the stellar density is ignored). Returns dict(A, frac, sigma = frac * A [ppm],
    parts {source: fractional error})."""
    lo, hi = ellipsoidal_amplitude(planet, mass_ratio, u_range[0], g_range[0]), ellipsoidal_amplitude(planet, mass_ratio, u_range[1], g_range[1])
    parts = {"M_p / M_star": float(frac_q), "(R_star / a)^3": 3. * sig_a_rs / (planet.R / planet.r2), "sin^2 i": 2. * np.radians(sig_incl) / np.tan(np.radians(planet.incl)),
             "alpha_ell (u, g)": (hi - lo) / (hi + lo)}
    A, frac = 0.5 * (lo + hi), float(np.sqrt(sum(v ** 2 for v in parts.values())))
    return dict(A=A, frac=frac, sigma=frac * A, parts=parts)


def deformation_amplitude(planet, mass_ratio, h2=1.):
    """Expected amplitude delta = (a1 - a2) / a2 of the planet's tidal deformation (the "planetary" column; a1 the semi-axis toward the star, a2
    the one along the orbit, a3 the polar one; xi in the paper) for the fluid Love number h2: the tidal and the centrifugal potential displace
    the surface of a synchronously rotating body in equilibrium by 7/6, -1/3 and -5/6 (in units of h2 zeta R_p, zeta = (M_star / M_p)
    (R_p / a)^3) along a1, a2, a3, so to first order delta = 1.5 h2 zeta. mass_ratio = M_p / M_star."""
    return 1.5 * h2 / mass_ratio * (planet.r1 / planet.R) ** 3


def deformation_prior(planet, mass_ratio, h2_range=DEFORMATION_H2_RANGE):
    """Prior of the planetary term: h2 within h2_range (DEFORMATION_H2_RANGE) bounds the amplitude, so it cannot be fitted freely. Returns dict(lo,
    hi [the amplitudes at the ends of h2_range], sigma [the std of a uniform distribution between them, the Gaussian width that stands for the
    range in the linear fits]). The errors of the mass ratio and the geometry (10-30 % of the amplitude) are small against the range and left out."""
    lo, hi = (deformation_amplitude(planet, mass_ratio, h2) for h2 in h2_range)
    return dict(lo=lo, hi=hi, sigma=(hi - lo) / np.sqrt(12.))


def deformation_limits(planet, mass_ratio, h2_true, h2_range=DEFORMATION_H2_RANGE):
    """(lo, hi) limits on the shift of the planetary amplitude from its true value, for a planet with Love number h2_true fitted with any h2 of
    h2_range: amplitude(h2_range) - amplitude(h2_true). The column is linear in the amplitude, so only this shift enters and the simulated truth
    can stay a sphere. For nuisance_solve(limits=) (best fit) or nuisance_both_fits(uniform=) (posterior; see nuisance_both_fits_uniform)."""
    lo, hi = (deformation_amplitude(planet, mass_ratio, h2) - deformation_amplitude(planet, mass_ratio, h2_true) for h2 in h2_range)
    return lo, hi
