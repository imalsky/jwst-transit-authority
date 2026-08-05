"""Correlated-noise scenario tier: the floor-budget covariance builder, the
full-covariance metric in the score and Fisher forecast, and the per-segment
slope nuisances the conservative scenario profiles."""
import numpy as np
import pytest

from jwst_tool import detect, fisher, noise as noise_mod


def _grid(n=60, seed=0):
    rng = np.random.default_rng(seed)
    wl = np.sort(rng.uniform(3.0, 5.0, n))
    var_phot = rng.uniform(0.5, 2.0, n) * 1e-10
    floor = rng.uniform(0.5, 1.5, n) * 1e-5
    return wl, var_phot, floor


# --- build_cov ----------------------------------------------------------------

def test_random_scenario_is_diagonal_fast_path():
    wl, var_phot, floor = _grid()
    assert noise_mod.build_cov(wl, var_phot, floor, "random") is None


def test_unknown_scenario_raises():
    wl, var_phot, floor = _grid()
    with pytest.raises(KeyError):
        noise_mod.build_cov(wl, var_phot, floor, "optimistic")


@pytest.mark.parametrize("scenario", ["moderate", "conservative"])
def test_cov_psd_and_scenario_invariant_totals(scenario):
    """C is PD and its diagonal equals max(var_phot, floor^2) exactly:
    scenarios re-allocate the floor EXCESS, never the per-bin total.
    (atol=0: the default atol is vacuous on ~1e-10 variances.)"""
    wl, var_phot, floor = _grid(seed=3)
    C = noise_mod.build_cov(wl, var_phot, floor, scenario)
    assert C.shape == (wl.size, wl.size)
    assert np.allclose(C, C.T, atol=0)
    assert np.linalg.eigvalsh(C).min() > 0
    assert np.allclose(np.diag(C), np.maximum(var_phot, floor ** 2),
                       rtol=1e-12, atol=0)
    assert not np.allclose(np.diag(C), var_phot + floor ** 2,
                           rtol=1e-3, atol=0)   # quadrature is GONE


# --- detection_significance with cov -----------------------------------------

def test_cov_diagonal_matches_sigma_path():
    """An explicitly diagonal C must reproduce the sigma fast path exactly."""
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.5, 2.0, 40) * 1e-5
    signal = rng.normal(0.0, 3e-5, 40)
    seg = np.array([0] * 20 + [1] * 20)
    steps = detect._segment_rows(seg)
    a = detect.detection_significance(signal, sigma, nuisance=steps)
    b = detect.detection_significance(signal, sigma, nuisance=steps,
                                      cov=np.diag(sigma ** 2))
    assert b == pytest.approx(a, rel=1e-10)


def test_smooth_signal_penalized_by_correlated_floor():
    """A smooth template loses S/N under a smooth-correlated floor; a narrow
    feature is much less affected."""
    wl, _vp, _fl = _grid(n=80, seed=5)
    var_phot = np.full(wl.size, 1e-11)
    floor = np.full(wl.size, 2e-5)
    sigma = np.maximum(np.sqrt(var_phot), floor)
    C = noise_mod.build_cov(wl, var_phot, floor, "conservative")
    smooth = 6e-5 * np.sin(np.pi * (wl - wl.min()) / (wl.max() - wl.min()))
    narrow = 6e-5 * np.exp(-0.5 * ((wl - 4.0) / 0.02) ** 2)
    ratio_smooth = (detect.detection_significance(smooth, sigma, cov=C)
                    / detect.detection_significance(smooth, sigma))
    ratio_narrow = (detect.detection_significance(narrow, sigma, cov=C)
                    / detect.detection_significance(narrow, sigma))
    assert ratio_smooth < 0.55
    assert ratio_narrow > 2.0 * ratio_smooth


# --- slope nuisances ----------------------------------------------------------

def test_per_segment_slope_profiled_out():
    """A per-segment linear-in-ln-wl trend profiles to ~0 with the slope rows
    supplied, and NOT without them."""
    wl = np.linspace(3.0, 5.0, 40)
    seg = np.array([0] * 20 + [1] * 20)
    lnl = np.log(wl)
    signal = np.where(seg == 0, 2e-4 * (lnl - lnl[:20].mean()),
                      -3e-4 * (lnl - lnl[20:].mean()))
    err = np.full(wl.size, 1e-5)
    steps = detect._segment_rows(seg)
    slopes = detect._slope_rows(seg, wl)
    assert len(slopes) == 2                      # every segment gets one
    without = detect.detection_significance(signal, err, nuisance=steps)
    assert without > 5.0
    with_slopes = detect.detection_significance(signal, err,
                                                nuisance=steps + slopes)
    assert with_slopes == pytest.approx(0.0, abs=1e-6)


def test_narrow_feature_survives_slope_profiling():
    # a bump centered in a segment is the case slope freedom cannot absorb
    wl = np.linspace(3.0, 5.0, 40)
    seg = np.array([0] * 20 + [1] * 20)
    signal = 8e-5 * np.exp(-0.5 * ((wl - 4.5) / 0.05) ** 2)
    err = np.full(wl.size, 1e-5)
    nuis = detect._segment_rows(seg) + detect._slope_rows(seg, wl)
    raw = detect.detection_significance(signal, err, marginalize_offset=False)
    prof = detect.detection_significance(signal, err, nuisance=nuis)
    assert prof > 0.75 * raw


# --- Fisher with cov + slopes --------------------------------------------------

def _fake_result(J, sigma, seg=None, cov=None, slope_rows=None):
    nb = sigma.size
    return dict(jac_bins=np.asarray(J), sigma=np.asarray(sigma),
                seg=np.asarray(seg if seg is not None else np.zeros(nb, int)),
                cov=cov,
                slope_rows=(np.zeros((0, nb)) if slope_rows is None
                            else np.asarray(slope_rows)))


def test_fisher_cov_diagonal_matches_sigma_path():
    rng = np.random.default_rng(2)
    nb = 30
    sigma = rng.uniform(0.5, 2.0, nb) * 1e-5
    J = rng.normal(size=(3, nb)) * 1e-4          # rows: two free + lnR0
    names = ["lnZ", "lnKzz"]
    a = fisher.mode_forecast(_fake_result(J, sigma), names)
    b = fisher.mode_forecast(_fake_result(J, sigma, cov=np.diag(sigma ** 2)),
                             names)
    for n in names:
        assert b[n] == pytest.approx(a[n], rel=1e-8)


def test_fisher_correlated_floor_inflates_smooth_parameter():
    """A parameter whose Jacobian is spectrally smooth must get a WORSE
    forecast under the correlated scenario at identical per-bin totals."""
    wl, _vp, _fl = _grid(n=50, seed=7)
    var_phot = np.full(wl.size, 1e-11)
    floor = np.full(wl.size, 2e-5)
    sigma = np.maximum(np.sqrt(var_phot), floor)
    C = noise_mod.build_cov(wl, var_phot, floor, "conservative")
    smooth_row = 1e-4 * np.sin(np.pi * (wl - wl.min()) / (wl.max() - wl.min()))
    lnr0_row = np.full(wl.size, -2e-4)
    J = np.stack([smooth_row, lnr0_row])
    names = ["lnZ"]
    diag_sig = fisher.mode_forecast(_fake_result(J, sigma), names)["lnZ"]
    corr_sig = fisher.mode_forecast(_fake_result(J, sigma, cov=C), names)["lnZ"]
    assert corr_sig > 1.5 * diag_sig


def test_fisher_slope_rows_absorb_slope_like_parameter():
    """A parameter whose response IS a linear trend becomes unconstrained
    once the slope nuisances are marginalized."""
    wl = np.linspace(3.0, 5.0, 40)
    seg = np.zeros(wl.size, int)
    sigma = np.full(wl.size, 1e-5)
    lnl = np.log(wl)
    trend = (lnl - lnl.mean()) / np.sqrt(np.mean((lnl - lnl.mean()) ** 2))
    J = np.stack([1e-4 * trend, np.full(wl.size, -2e-4)])   # free + lnR0
    names = ["lnZ"]
    free = fisher.mode_forecast(_fake_result(J, sigma, seg=seg), names)["lnZ"]
    assert np.isfinite(free)
    slopes = np.stack(detect._slope_rows(seg, wl))
    absorbed = fisher.mode_forecast(
        _fake_result(J, sigma, seg=seg, slope_rows=slopes), names)["lnZ"]
    assert absorbed == np.inf


def test_combined_forecast_accepts_mixed_scenarios():
    """combined_forecast must assemble modes whose noise models differ
    (diagonal + covariance blocks, with and without slope rows)."""
    rng = np.random.default_rng(4)
    wl1, vp1, fl1 = _grid(n=25, seed=8)
    sigma1 = np.maximum(np.sqrt(vp1), fl1)
    C1 = noise_mod.build_cov(wl1, vp1, fl1, "moderate")
    J1 = rng.normal(size=(3, 25)) * 1e-4
    r1 = _fake_result(J1, sigma1, cov=C1)
    wl2 = np.linspace(1.0, 2.5, 20)
    sigma2 = np.full(20, 1e-5)
    seg2 = np.array([0] * 10 + [1] * 10)
    J2 = rng.normal(size=(3, 20)) * 1e-4
    r2 = _fake_result(J2, sigma2, seg=seg2,
                      slope_rows=np.stack(detect._slope_rows(seg2, wl2)))
    names = ["lnZ", "lnKzz"]
    out = fisher.combined_forecast([r1, r2], names)
    assert set(out) == set(names)
    assert all(np.isfinite(v) and v > 0 for v in out.values())
