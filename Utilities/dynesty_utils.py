import numpy as np
from dataclasses import dataclass, field
from hyperillumination_phasecurve_utils import model_pcurve  # forward model
from simple_bias_analysis import binned # exact binning


# ---------------------------------------------------------------------------
#  Simulated observation
# ---------------------------------------------------------------------------

def simulate_data(cfg, A_B_true, insolation_model="hyperillum", seed=42, noise=True, offset=0.,
                  k_truth=200, n_layers=80, n_lo=600, n_la=300, **model_kw):
    """Simulated phase-folded observation: binned(truth) + N(0, sigma_bin).

    cfg          : FittingConfig — the only mandatory input; supplies planet, filter, stellar_grid, A_(A_B), sigma_1min, n_orbits, n_bins, u_B, reflected.
    A_B_true     : true Bond albedo; defaults to cfg.A_B_true.
    noise        : False -> the noiseless ("Asimov") dataset. Use it for bias maps; do NOT use it for lnZ / Bayes factors, where chi2 = 0 at the truth inflates Z.
    offset       : constant baseline injected into the data [ppm]; set non-zero to check that the fitted nuisance offset is actually recovered.
    k_truth      : sub-samples per bin for the truth (n_t = k_truth * n_bins, must tile).
    """

    # high-resolution truth (thermal + reflected, since cfg.reflected=True)
    t_hi, y_hi = model_pcurve(cfg, insolation_model, A_B_true, n_t=k_truth * cfg.n_bins, n_layers=n_layers, n_lo=n_lo, n_la=n_la, **model_kw)

    # reduce to the observed bins
    y_true = binned(cfg, y_hi) + offset

    # adding noise
    s = cfg.sigma_bin
    rng = np.random.default_rng(seed)
    y_obs = y_true + rng.normal(0., s, cfg.n_bins) if noise else y_true.copy()

    t=(np.arange(cfg.n_bins) + 0.5) * cfg.planet.period / cfg.n_bins
    sigma = np.full(cfg.n_bins, s)

    return {"t": t, "y_obs": y_obs, "sigma":sigma}