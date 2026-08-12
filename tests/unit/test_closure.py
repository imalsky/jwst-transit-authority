"""Closure tests: the noise model against synthetic Poisson counts, the
covariance metric against Monte Carlo, and (opt-in, slow) the autodiff
Jacobian against finite differences of the full forward model.

Default runs stay numpy-only; the FD test needs JWST_TOOL_RUN_SLOW=1 and
runs the real forward model three times (~5-10 min)."""
import os

import numpy as np
import pytest

from jwst_tool import binning, noise as noise_mod


def test_poisson_count_closure():
    """Empirical bin mean/variance of Poisson-count depth estimates must
    close with the analytic depth_error_bins variance."""
    rng = np.random.default_rng(11)
    n_pix = 300
    wl = np.sort(rng.uniform(3.0, 4.0, n_pix))
    flux = 4e3 * (1.1 + np.sin(4.0 * wl))          # e-/s per pixel
    t_int = 20.0                                    # one integration cycle (s)
    n_in, n_out = 120, 150                          # integrations in the window
    # pure-photon pandeia surrogate: sigma of a 1-integration rate estimate
    noise_1int = np.sqrt(flux / t_int)
    mode_result = dict(wl=wl.tolist(), flux=flux.tolist(),
                       noise_1int=noise_1int.tolist(), t_cycle_s=t_int)

    edges = np.array([3.0, 3.25, 3.5, 3.75, 4.0])
    op = binning.build_operator(wl, flux, edges, wl_lo=2.9, wl_hi=4.1)
    nz = noise_mod.depth_error_bins(mode_result, edges,
                                    t_in_s=n_in * t_int, t_out_s=n_out * t_int,
                                    n_transits=1, floor_spec=None, op=op)

    depth_true = 0.012 * (1.0 + 0.2 * np.sin(6.0 * wl))
    lam_in = flux * t_int * n_in * (1.0 - depth_true)
    lam_out = flux * t_int * n_out
    n_mc = 3000
    c_in = rng.poisson(lam_in, size=(n_mc, n_pix))
    c_out = rng.poisson(lam_out, size=(n_mc, n_pix))
    d_hat = 1.0 - (c_in / n_in) / (c_out / n_out)
    est = np.stack([binning.bin_values(op, d) for d in d_hat])

    d_true_bin = binning.bin_values(op, depth_true)
    se = np.sqrt(nz["var_phot"] / n_mc)
    # mean closes to the true binned depth (ratio-estimator bias ~1/counts,
    # negligible at ~1e7 counts/pixel); variance closes to the analytic value
    assert np.all(np.abs(est.mean(axis=0) - d_true_bin) < 5.0 * se + 1e-7)
    assert np.allclose(est.var(axis=0), nz["var_phot"], rtol=0.15)


@pytest.mark.skipif(os.environ.get("JWST_TOOL_RUN_SLOW") != "1",
                    reason="slow: 3 full VULCAN-JAX+ExoJAX forward runs "
                           "(~5-10 min, JAX required); set JWST_TOOL_RUN_SLOW=1")
def test_jacobian_row_matches_finite_difference():
    """The cached FD Jacobian row must agree with an independent smaller-step
    (h = 2 K) central difference: different step, different cache entries.

    Loose gate on purpose: chemistry certifies at yconv 1e-2, so shape
    correlation plus ~15% scale is what steady-state uniqueness guarantees."""
    from jwst_tool import forward

    def quiet(_s):
        return None

    T0 = 1560.0
    base = dict(planet="wasp39b", tp_mode="guillot",
                Tirr=T0, fisher_params=["Tirr"], use_photo=True)
    if forward.load_result(base) is None:
        forward.run_model(base, log=quiet)
    m0 = forward.load_result(base)
    names = [str(x) for x in m0["jac_names"]]
    row = np.asarray(m0["jac"][names.index("Tirr")])

    h = 2.0                                          # K, small vs T ~ 1000 K
    d = {}
    for s, tag in ((+h, "p"), (-h, "m")):
        p = dict(base, Tirr=T0 + float(s), fisher_params=[])
        if forward.load_result(p) is None:
            forward.run_model(p, log=quiet)
        d[tag] = np.asarray(forward.load_result(p)["depth"])
    fd = (d["p"] - d["m"]) / (2.0 * h)

    corr = np.corrcoef(row, fd)[0, 1]
    scale = float(np.dot(row, fd) / np.dot(fd, fd))
    assert corr > 0.99
    assert scale == pytest.approx(1.0, abs=0.15)
