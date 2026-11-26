"""Simulated observations: the noiseless truth at high resolution plus Gaussian noise, for one band or a detector set."""

import numpy as np

from hyperphase.bias.detectors import is_detector_set
from hyperphase.bias.fits import binned
from hyperphase.model import model_pcurve


############################################################################################################################################
#####   Simulate Observations
############################################################################################################################################

# Resolution of the simulated truth, well above the likelihood's defaults, so that a truth-vs-model residual is model bias, not grid
# resolution (the numerical floor: prediction.evidence_reference). k_truth: samples per bin (n_t = k_truth * n_bins).
TRUTH_KW = dict(k_truth=200, n_layers=80, n_lo=600, n_la=300)


def truth_model_kw(data, cfg):
    """The model_pcurve keywords the truth of `data` was computed with (n_t, n_layers, n_lo, n_la, and F for a redistribution truth),
    from data["truth"]["model_kw"]; data without this record are assumed to have TRUTH_KW."""
    tr = dict(data.get("truth") or {})
    kw = dict(tr.get("model_kw") or dict(n_t=TRUTH_KW["k_truth"] * cfg.n_bins, n_layers=TRUTH_KW["n_layers"], n_lo=TRUTH_KW["n_lo"], n_la=TRUTH_KW["n_la"]))
    if tr.get("F") is not None: kw.setdefault("F", tr["F"])
    return kw


def detector_data(data):
    """A simulate_data output -> the list of per-detector data dicts (data["detectors"] of a joint dataset, else the dict itself)."""
    return list(data["detectors"]) if "detectors" in data else [data]


def _show_simulated_data(*args, **kw):
    """plotting.retrieval.plot_simulated_data, shown: imported here, so that simulating data does not need matplotlib."""
    import matplotlib.pyplot as plt
    from hyperphase.plotting.retrieval import plot_simulated_data
    plot_simulated_data(*args, **kw); plt.show()


def simulate_data(cfg, A_B_true, insolation_model="hyperillum", seed=42, noise=True, offset=0.,
                  k_truth=TRUTH_KW["k_truth"], n_layers=TRUTH_KW["n_layers"], n_lo=TRUTH_KW["n_lo"], n_la=TRUTH_KW["n_la"], plotting=True, labels=None, _rng=None, **model_kw):
    """Simulated phase-folded observation: binned(truth) + N(0, sigma_bin), the truth at the resolution TRUTH_KW.

    cfg      : FittingConfig, or a detector set (same planet and binning): one dataset per detector from the same truth and ONE noise
               stream, so the first detector's draws equal its single-band dataset of the same seed. The per-detector dicts are in
               data["detectors"], next to the concatenated "t", "y_obs", "sigma".
    noise    : False gives the noiseless (Asimov) dataset: each lnZ is inflated (chi2 = 0 at the truth), but ln B is the expected value,
               ~chi2_min / 2.
    offset   : constant baseline added to the data [ppm].
    k_truth  : samples per bin of the truth (n_t = k_truth * n_bins).
    labels   : detector names for the plot of a detector set.
    model_kw : further model_pcurve keywords of the truth, e.g. F for a redistribution model (irradiance.REDIST_MODELS).
    data["truth"] records A_B, model, F, offset, seed and the truth's model_pcurve settings, and travels into NestedFit.meta.
    """
    if is_detector_set(cfg):
        rng = np.random.default_rng(seed)   # one stream for all detectors
        parts = [simulate_data(c, A_B_true, insolation_model, seed=seed, noise=noise, offset=offset, k_truth=k_truth, n_layers=n_layers, n_lo=n_lo, n_la=n_la, plotting=False, _rng=rng, **model_kw) for c in cfg]
        data = {k: np.concatenate([d[k] for d in parts]) for k in ("t", "y_obs", "sigma")}
        data["truth"] = dict(parts[0]["truth"])
        data["detectors"] = parts
        if plotting: _show_simulated_data(cfg, A_B_true, data, insolation_model=insolation_model, labels=labels)
        return data

    # high-resolution truth
    model_kw_truth = dict(n_t=k_truth * cfg.n_bins, n_layers=n_layers, n_lo=n_lo, n_la=n_la, **model_kw)
    t_hi, y_hi = model_pcurve(cfg, insolation_model, A_B_true, **model_kw_truth)

    # reduce to the observed bins
    y_true = binned(cfg, y_hi) + offset

    # noise (a stream handed in by the detector-set branch continues across detectors)
    s = cfg.sigma_bin
    rng = np.random.default_rng(seed) if _rng is None else _rng
    y_obs = y_true + rng.normal(0., s, cfg.n_bins) if noise else y_true.copy()

    t=(np.arange(cfg.n_bins) + 0.5) * cfg.planet.period / cfg.n_bins
    sigma = np.full(cfg.n_bins, s)

    # the injected truth, with the settings to re-evaluate it
    F_true = model_kw.get("F")
    truth = dict(A_B=float(A_B_true), insolation_model=insolation_model, F=None if F_true is None else float(F_true), offset=float(offset), seed=int(seed), noise=bool(noise), model_kw=model_kw_truth)
    data = {"t": t, "y_obs": y_obs, "sigma":sigma, "truth": truth}

    if plotting: _show_simulated_data(cfg, A_B_true, data, insolation_model=insolation_model)

    return data
