"""Operator tests for the count-space measurement operator: constant-depth
conservation, estimator/variance consistency (Monte Carlo), model-vs-noise
same-estimator identity, Jacobian linearity, gap handling, strict grid
validation, and the native-R LSF (flux-weighted) blur."""
import numpy as np
import pytest

from jwst_tool import binning, noise as noise_mod

# `np.trapezoid` is NumPy 2.0+; `np.trapz` is the 1.26 spelling (deprecated in
# 2.0). The package floor is NumPy>=1.26, so bind whichever exists.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _fake_pixels(rng, n=800, wl0=1.0, wl1=5.0):
    wl = np.sort(rng.uniform(wl0, wl1, n))
    flux = 1e3 * (1.2 + np.sin(3.0 * wl)) * rng.uniform(0.8, 1.2, n)
    return wl, flux


def test_constant_depth_conservation():
    rng = np.random.default_rng(0)
    wl, flux = _fake_pixels(rng)
    edges = noise_mod.make_bins(1.1, 4.9, 80.0)
    op = binning.build_operator(wl, flux, edges, wl_lo=0.9, wl_hi=5.1)
    wl_model = np.linspace(0.9, 5.1, 3000)
    d0 = 0.0123
    binned = binning.bin_model(op, wl_model, np.full(wl_model.size, d0))
    # exact up to float64 cumsum roundoff (sequential sum over the model grid)
    assert np.allclose(binned, d0, rtol=0, atol=1e-12)


def test_pl_antideriv_exact_on_audit_counterexample():
    """The antiderivative of a piecewise-linear model is piecewise-QUADRATIC;
    linearly interpolating the cumulative integral is wrong between nodes.
    Model y=x on [0,1]: average over [0.1,0.2] = 0.15 (np.interp gave 0.5)."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    icum = binning._pl_cumint(x, y)
    ia = binning._pl_antideriv(np.array([0.1]), x, y, icum)[0]
    ib = binning._pl_antideriv(np.array([0.2]), x, y, icum)[0]
    assert ia == pytest.approx(0.005, rel=0, abs=1e-15)
    assert ib == pytest.approx(0.02, rel=0, abs=1e-15)
    assert (ib - ia) / (0.2 - 0.1) == pytest.approx(0.15, rel=0, abs=1e-14)
    # endpoints reproduce the exact node integrals
    assert binning._pl_antideriv(x, x, y, icum) == pytest.approx(icum, abs=1e-15)


def test_bin_model_exact_for_linear_submodel():
    """A globally-linear model on a COARSE grid with pixel cells BETWEEN nodes
    must bin to machine precision (exact cell average = midpoint value)."""
    wl_model = np.array([2.0, 3.0, 4.0])          # coarse: only 2 intervals
    slope, intercept = 1.5, 0.7
    y = slope * wl_model + intercept
    rng = np.random.default_rng(11)
    wl_pix = np.sort(rng.uniform(2.05, 3.95, 60))  # cells sit between nodes
    flux = rng.uniform(0.5, 2.0, wl_pix.size)
    edges = np.array([2.0, 3.0, 4.0])
    op = binning.build_operator(wl_pix, flux, edges, wl_lo=2.0, wl_hi=4.0)
    got = binning.bin_model(op, wl_model, y)
    mid = 0.5 * (op["cell_lo"] + op["cell_hi"])
    d_pix = slope * mid + intercept               # exact per-cell average
    ref = np.zeros(op["wl_center"].size)
    den = np.zeros(op["wl_center"].size)
    np.add.at(ref, op["pix_bin"], op["pix_w"] * d_pix)
    np.add.at(den, op["pix_bin"], op["pix_w"])
    ref /= den
    assert np.allclose(got, ref, rtol=0, atol=1e-14)


def test_model_binning_is_linear():
    """Binned Jacobian == Jacobian of the binned model (operator linearity)."""
    rng = np.random.default_rng(1)
    wl, flux = _fake_pixels(rng)
    edges = noise_mod.make_bins(1.1, 4.9, 60.0)
    op = binning.build_operator(wl, flux, edges, wl_lo=0.9, wl_hi=5.1)
    wl_model = np.linspace(0.9, 5.1, 2000)
    a = 1e-2 * (1 + np.sin(8 * wl_model))
    b = 1e-3 * np.cos(5 * wl_model)
    lhs = binning.bin_model(op, wl_model, 2.0 * a + 3.0 * b)
    rhs = 2.0 * binning.bin_model(op, wl_model, a) + 3.0 * binning.bin_model(op, wl_model, b)
    assert np.allclose(lhs, rhs, rtol=1e-12)


def test_estimator_mc_mean_and_variance():
    """The reported (bin value, bin variance) must be the mean and variance of
    the actual count-weighted estimator, checked by Monte Carlo."""
    rng = np.random.default_rng(2)
    wl, flux = _fake_pixels(rng, n=400, wl0=2.0, wl1=3.0)
    edges = np.array([2.0, 2.3, 2.6, 3.0])
    op = binning.build_operator(wl, flux, edges, wl_lo=1.9, wl_hi=3.1)
    wl_model = np.linspace(1.9, 3.1, 1500)
    depth = 0.01 * (1.0 + 0.3 * np.sin(6 * wl_model))
    sig_pix = 5e-4 * rng.uniform(0.5, 2.0, wl.size)

    d_true_bin = binning.bin_model(op, wl_model, depth)
    var_bin = binning.bin_variance(op, sig_pix ** 2)

    # simulate the estimator: per-pixel depth draws, count-weighted bin means
    d_pix_expect = np.interp(wl, wl_model, depth)
    n_mc = 4000
    draws = d_pix_expect[None, :] + sig_pix[None, :] * rng.standard_normal((n_mc, wl.size))
    est = np.stack([binning.bin_values(op, d) for d in draws])
    mc_mean, mc_var = est.mean(axis=0), est.var(axis=0)

    # mean matches the binned model within 5 MC standard errors (+ the O(h^2)
    # cell-average vs point-sample difference, < 1e-6 on this smooth model)
    se = np.sqrt(var_bin / n_mc)
    assert np.all(np.abs(mc_mean - d_true_bin) < 5.0 * se + 1e-6)
    assert np.allclose(mc_var, var_bin, rtol=0.15)


def test_noise_and_model_share_bins_and_estimator():
    """depth_error_bins with the same operator returns bins aligned 1:1 with the
    binned model, and count-space variance (not inverse-variance)."""
    rng = np.random.default_rng(3)
    wl, flux = _fake_pixels(rng, n=300, wl0=2.0, wl1=4.0)
    noise = np.sqrt(flux) * rng.uniform(0.9, 1.1, wl.size)
    mode_result = dict(wl=wl.tolist(), flux=flux.tolist(),
                       noise_1int=noise.tolist(), t_cycle_s=20.0)
    edges = noise_mod.make_bins(2.05, 3.95, 50.0)
    op = binning.build_operator(wl, flux, edges, wl_lo=1.9, wl_hi=4.1)
    nz = noise_mod.depth_error_bins(mode_result, edges, 3600.0, 3600.0, 1, 0.0, op=op)
    assert nz["wl_center"].shape == op["wl_center"].shape
    var_pix = noise_mod.pixel_depth_variance(mode_result, 3600.0, 3600.0, 1)
    assert np.allclose(nz["var_phot"], binning.bin_variance(op, var_pix))
    # count-space variance >= inverse-variance combination (equality iff
    # weights are exactly ivar-optimal)
    inv = np.zeros(op["wl_center"].size)
    np.add.at(inv, op["pix_bin"], 1.0 / var_pix[op["pix_idx"]])
    assert np.all(nz["var_phot"] >= 1.0 / inv - 1e-18)


def test_detector_gap_does_not_leak_model():
    """Pixels at a detector-gap edge must not average the model across the gap
    (cell half-widths are capped)."""
    wl = np.concatenate([np.linspace(2.0, 2.5, 100), np.linspace(3.5, 4.0, 100)])
    flux = np.full(wl.size, 1e3)
    edges = np.array([2.0, 2.6, 3.4, 4.0])
    op = binning.build_operator(wl, flux, edges, wl_lo=1.9, wl_hi=4.1)
    # a model that is 0 on the blue side, 1 in the gap, 0 on the red side:
    wl_model = np.linspace(1.9, 4.1, 5000)
    d = ((wl_model > 2.6) & (wl_model < 3.4)).astype(float)
    binned = binning.bin_model(op, wl_model, d)
    # kept bins are the two detector sides; gap-only bin has no pixels
    assert op["keep"].tolist() == [True, False, True]
    assert np.all(binned < 0.02)


def test_full_sat_pixels_excluded_via_valid_mask():
    rng = np.random.default_rng(4)
    wl, flux = _fake_pixels(rng, n=200, wl0=2.0, wl1=3.0)
    valid = np.ones(wl.size, bool)
    valid[:50] = False
    edges = np.array([2.0, 2.5, 3.0])
    op = binning.build_operator(wl, flux, edges, wl_lo=1.9, wl_hi=3.1, valid=valid)
    assert not np.isin(op["pix_idx"], np.where(~valid)[0]).any()


def test_rebinning_nested_vs_direct():
    """Binning pixels directly to coarse bins equals count-weighted
    recombination of fine bins built from the same pixels."""
    rng = np.random.default_rng(5)
    wl, flux = _fake_pixels(rng, n=1000, wl0=2.0, wl1=4.0)
    coarse = np.geomspace(2.0, 4.0, 9)
    fine = np.geomspace(2.0, 4.0, 33)          # 4 fine bins per coarse bin
    wl_model = np.linspace(1.9, 4.1, 4000)
    d = 0.01 * (1 + 0.2 * np.sin(7 * wl_model))
    op_c = binning.build_operator(wl, flux, coarse, wl_lo=1.9, wl_hi=4.1)
    op_f = binning.build_operator(wl, flux, fine, wl_lo=1.9, wl_hi=4.1)
    d_c = binning.bin_model(op_c, wl_model, d)
    d_f = binning.bin_model(op_f, wl_model, d)
    wsum_f = np.zeros(op_f["wl_center"].size)
    np.add.at(wsum_f, op_f["pix_bin"], op_f["pix_w"])
    # recombine fine bins into coarse with their pixel-count weights
    fmap = np.digitize(op_f["wl_center"], coarse) - 1
    num = np.zeros(len(coarse) - 1)
    den = np.zeros(len(coarse) - 1)
    np.add.at(num, fmap, wsum_f * d_f)
    np.add.at(den, fmap, wsum_f)
    assert np.allclose(num[den > 0] / den[den > 0], d_c, rtol=1e-12)


def test_segment_ids_split_at_detector_gap_only():
    """NRS1|NRS2-style gap makes two segments; a smooth grid makes one."""
    wl = np.concatenate([np.linspace(2.87, 3.72, 300),   # NRS1
                         np.linspace(3.82, 5.18, 400)])   # NRS2 (0.10 um gap)
    seg = binning.segment_ids(wl)
    assert seg[:300].tolist() == [0] * 300
    assert seg[300:].tolist() == [1] * 400
    # smooth single-detector grid -> one segment
    assert binning.segment_ids(np.linspace(1.0, 5.0, 1000)).max() == 0
    # order-independence: shuffle in, segment ids follow the pixels
    rng = np.random.default_rng(7)
    perm = rng.permutation(wl.size)
    seg_p = binning.segment_ids(wl[perm])
    assert np.array_equal(seg_p, seg[perm])


def test_bin_segments_assigns_majority_segment():
    wl = np.concatenate([np.linspace(2.87, 3.72, 300),
                         np.linspace(3.82, 5.18, 400)])
    flux = np.full(wl.size, 1e3)
    edges = noise_mod.make_bins(2.9, 5.15, 100.0)
    op = binning.build_operator(wl, flux, edges, wl_lo=2.8, wl_hi=5.2)
    seg_pix = binning.segment_ids(wl)
    seg_bin = binning.bin_segments(op, seg_pix)
    # every kept bin is entirely in one detector, so bins are a monotonic
    # block of 0s then 1s, both present
    assert set(seg_bin.tolist()) == {0, 1}
    assert np.all(np.diff(seg_bin) >= 0)


def test_smooth_to_native_r_conserves_flux_and_noops_at_high_r():
    wl = np.linspace(4.0, 12.0, 20000)
    y = 0.01 + 1e-3 * np.sin(50 * wl)              # structured depth
    # MIRI-like low native R -> blur changes the model, conserves the mean
    r_lo = np.full(wl.size, 80.0)
    y_lo = binning.smooth_to_native_r(wl, y, wl, r_lo, 5.0, 11.0)
    assert not np.allclose(y_lo, y)                # something happened
    band = (wl >= 5.5) & (wl <= 10.5)             # interior (no edge effects)
    assert abs(_trapz(y_lo[band], wl[band]) - _trapz(y[band], wl[band])) \
        < 1e-3 * abs(_trapz(y[band], wl[band]))
    # very high native R (>> model sampling) -> kernel unresolved -> exact no-op
    r_hi = np.full(wl.size, 1e5)
    y_hi = binning.smooth_to_native_r(wl, y, wl, r_hi, 5.0, 11.0)
    assert np.array_equal(y_hi, y)


def test_smooth_flux_weight_none_equals_constant_weight():
    """weight=None and a constant weight are the same operator (the flat blur):
    L[F d]/L[F] = L[d] when F is constant."""
    wl = np.linspace(4.0, 12.0, 20000)
    y = 0.01 + 1e-3 * np.sin(50 * wl)
    r = np.full(wl.size, 80.0)
    flat = binning.smooth_to_native_r(wl, y, wl, r, 5.0, 11.0)
    ones = binning.smooth_to_native_r(wl, y, wl, r, 5.0, 11.0,
                                      weight=np.full(wl.size, 3.7))
    assert np.allclose(flat, ones, atol=0, rtol=1e-12)


def test_smooth_flux_weight_preserves_constant_depth():
    """A flat transit depth must survive the flux-weighted LSF unchanged for
    ANY stellar-flux weight: L[F c]/L[F] = c, unbiased even under a strong
    stellar gradient."""
    wl = np.linspace(4.0, 12.0, 20000)
    c = 0.0123
    y = np.full(wl.size, c)
    r = np.full(wl.size, 60.0)
    F = 1.0 + 5.0 * (wl - wl[0]) / (wl[-1] - wl[0])       # 6x throughput gradient
    F *= 1.0 + 0.4 * np.exp(-0.5 * ((wl - 8.0) / 0.05) ** 2)   # + a stellar line
    out = binning.smooth_to_native_r(wl, y, wl, r, 5.0, 11.0, weight=F)
    band = (wl >= 5.5) & (wl <= 10.5)
    assert np.allclose(out[band], c, atol=1e-12)


def test_smooth_flux_weight_matches_direct_ratio_reference():
    """The flux-weighted native-R blur must equal an independent brute-force
    evaluation of d_obs = L[F d]/L[F] on a dense grid, and must DIFFER from
    the flat blur L[d] where the stellar flux has structure."""
    wl = np.linspace(4.0, 12.0, 40000)
    lnw = np.log(wl)
    d = 0.01 + 3e-3 * np.exp(-0.5 * ((wl - 8.0) / 0.10) ** 2)   # planetary feature
    F = 1.0 + 0.8 * np.exp(-0.5 * ((wl - 8.05) / 0.06) ** 2)    # nearby stellar line
    R = 70.0
    r = np.full(wl.size, R)
    got = binning.smooth_to_native_r(wl, d, wl, r, 5.5, 10.5, weight=F)
    flat = binning.smooth_to_native_r(wl, d, wl, r, 5.5, 10.5)

    # independent reference: full Gaussian-in-ln(lambda) weighted means (the
    # untruncated continuous ratio; the implementation truncates at 4 sigma on a
    # uniform ln grid, so a ~5 ppm discretization residual is expected and OK)
    s = 1.0 / (2.3548 * R)
    band = (wl >= 6.5) & (wl <= 9.5)      # interior, away from working-band edges
    idx = np.where(band)[0][::50]         # subsample for a cheap O(n_sub * n) loop
    ref = np.empty(idx.size)
    for j, i in enumerate(idx):
        k = np.exp(-0.5 * ((lnw - lnw[i]) / s) ** 2)
        ref[j] = np.sum(k * F * d) / np.sum(k * F)
    err_weighted = np.max(np.abs(got[idx] - ref))
    err_flat = np.max(np.abs(flat[idx] - ref))
    assert err_weighted < 1e-5                     # tracks the true ratio (~5 ppm)
    assert err_flat > 1e-4                          # flat blur is ~120 ppm biased
    assert err_weighted < err_flat / 10.0           # flux weighting >10x closer


def test_degenerate_wavelength_pixels_flagged():
    """A pileup of near-duplicate wavelengths (the G395H red-edge artifact)
    must be flagged; smooth dispersion gradients must not."""
    base = np.linspace(2.0, 4.0, 500)
    pile = 3.0 + np.arange(300) * 4e-6            # 300 samples within 1.2e-3 um
    wl = np.concatenate([base, pile])
    bad = binning.degenerate_wl_mask(wl)
    assert bad[500:].all()
    assert not bad[:499].any()
    # a smooth 5x dispersion change across the band is NOT degenerate
    t = np.linspace(0, 1, 800)
    wl_smooth = 5.0 + np.cumsum(0.001 * (1 + 4 * t))
    assert not binning.degenerate_wl_mask(wl_smooth).any()


# --- strict grid validation ---------------------------------------------------
# Invalid grids must raise loudly at the entry point (never degrade silently),
# and exact duplicates must count as degenerate.

def test_exact_duplicates_are_degenerate():
    """Duplicate-majority grid: the zero-gap pixels must be flagged even
    though the ALL-gap median is zero."""
    wl = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 3.0])
    bad = binning.degenerate_wl_mask(wl)
    assert bad[:4].all()          # zero spectral support
    assert not bad[4:].any()      # the real pixels survive
    # fully-duplicated grid: everything degenerate, never all-False
    assert binning.degenerate_wl_mask(np.full(10, 2.5)).all()


def test_duplicates_do_not_split_segments():
    wl = np.array([1.0, 1.0, 1.0, 1.0, 1.001, 1.002])
    assert (binning.segment_ids(wl) == 0).all()
    # all-duplicate grid: one segment, no crash on a zero median gap
    assert (binning.segment_ids(np.full(5, 2.0)) == 0).all()


def test_nonfinite_wavelengths_raise_everywhere():
    wl_nan = np.array([1.0, np.nan, 1.2])
    wl_inf = np.array([1.0, np.inf, 1.2])
    w = np.ones(3)
    edges = np.array([0.9, 1.3])
    for bad in (wl_nan, wl_inf):
        with pytest.raises(ValueError, match="non-finite"):
            binning.degenerate_wl_mask(bad)
        with pytest.raises(ValueError, match="non-finite"):
            binning.segment_ids(bad)
        with pytest.raises(ValueError, match="non-finite"):
            binning.build_operator(bad, w, edges)


def test_invalid_weights_and_edges_raise():
    wl = np.linspace(1.0, 2.0, 10)
    w = np.ones(10)
    edges = np.array([1.0, 1.5, 2.0])
    with pytest.raises(ValueError, match="weights"):
        binning.build_operator(wl, np.where(np.arange(10) == 3, np.nan, w), edges)
    with pytest.raises(ValueError, match="weights"):
        binning.build_operator(wl, np.where(np.arange(10) == 3, -1.0, w), edges)
    with pytest.raises(ValueError, match="shape"):
        binning.build_operator(wl, np.ones(9), edges)
    for bad_edges in (np.array([2.0, 1.0]),          # descending
                      np.array([1.5]),               # single edge
                      np.array([1.0, np.nan, 2.0]),  # non-finite
                      np.array([1.0, 1.0, 2.0])):    # zero-width bin
        with pytest.raises(ValueError, match="edges"):
            binning.build_operator(wl, w, bad_edges)
    with pytest.raises(ValueError, match="valid mask"):
        binning.build_operator(wl, w, edges, valid=np.ones(9, bool))


def test_all_invalid_operator_raises_actionably():
    wl = np.linspace(1.0, 2.0, 10)
    edges = np.array([1.0, 1.5, 2.0])
    # every pixel zero-weight
    with pytest.raises(ValueError, match="no usable pixel"):
        binning.build_operator(wl, np.zeros(10), edges)
    # every pixel masked invalid
    with pytest.raises(ValueError, match="no usable pixel"):
        binning.build_operator(wl, np.ones(10), edges, valid=np.zeros(10, bool))
    # no overlap between pixels and bin edges
    with pytest.raises(ValueError, match="no usable pixel"):
        binning.build_operator(wl, np.ones(10), np.array([5.0, 6.0]))


def test_bin_model_rejects_bad_model_grids():
    wl = np.linspace(1.0, 2.0, 10)
    op = binning.build_operator(wl, np.ones(10), np.array([1.0, 1.5, 2.0]),
                                wl_lo=0.9, wl_hi=2.1)
    good_wl = np.linspace(0.9, 2.1, 50)
    good_y = np.full(50, 0.01)
    with pytest.raises(ValueError, match="ascending"):
        binning.bin_model(op, good_wl[::-1], good_y)
    dup = good_wl.copy()
    dup[10] = dup[9]
    with pytest.raises(ValueError, match="ascending"):
        binning.bin_model(op, dup, good_y)
    with pytest.raises(ValueError, match="non-finite model value"):
        binning.bin_model(op, good_wl,
                          np.where(np.arange(50) == 7, np.nan, good_y))


def test_smooth_rejects_bad_weight():
    wl = np.geomspace(1.0, 1.4, 400)
    y = np.full(400, 0.01)
    wl_r = np.array([1.0, 1.4])
    r = np.array([50.0, 50.0])
    bad_w = np.ones(400)
    bad_w[5] = np.nan
    with pytest.raises(ValueError, match="weight"):
        binning.smooth_to_native_r(wl, y, wl_r, r, 1.05, 1.35, weight=bad_w)
    with pytest.raises(ValueError, match="weight"):
        binning.smooth_to_native_r(wl, y, wl_r, r, 1.05, 1.35,
                                   weight=np.ones(399))


def test_smooth_clamped_model_span_preserves_constant_at_band_edge():
    """When the model grid spans EXACTLY the band, the working-grid edge clip
    can collapse the final cell to zero width; pre-fix that cell's 0/1e-300
    average filled the constant pad, dragging the weight=None blur ~49% low at
    the band edge. Both invariants must hold at machine precision across a
    band_hi sweep that toggles the zero-width alignment."""
    c, R = 0.0123, 60.0
    for band_hi in np.linspace(10.90, 11.10, 9):
        wl = np.linspace(5.0, band_hi, 3000)   # model exactly spans the band
        y = np.full(wl.size, c)
        r = np.full(wl.size, R)
        flat = binning.smooth_to_native_r(wl, y, wl, r, 5.0, band_hi)
        ones = binning.smooth_to_native_r(wl, y, wl, r, 5.0, band_hi,
                                          weight=np.full(wl.size, 3.7))
        assert np.max(np.abs(flat - c)) < 1e-12, band_hi
        assert np.allclose(flat, ones, atol=0, rtol=1e-12), band_hi


# --- LSF weight validation on both the active and the no-op paths -------------

_WL_R = np.linspace(2.5, 4.0, 50)
_R_CURVE = np.full(50, 100.0)


def _lsf(wl_model, weight):
    y = np.full_like(wl_model, 0.01)
    return binning.smooth_to_native_r(wl_model, y, _WL_R, _R_CURVE,
                                      3.0, 3.5, weight=weight)


# fine grid -> kernel resolved (LSF active); coarse grid -> kernel unresolved
# by the model grid, which is the no-op early return
_ACTIVE = np.linspace(3.0, 3.5, 4000)
_NOOP = np.linspace(3.0, 3.5, 12)


def test_lsf_active_path_is_actually_active_and_noop_path_is_actually_noop():
    """Guard the fixture itself: the two grids must exercise DIFFERENT paths,
    or the validation test below would be vacuous."""
    y_a = 0.01 + 1e-3 * np.sin(400.0 * _ACTIVE)
    out_a = binning.smooth_to_native_r(_ACTIVE, y_a, _WL_R, _R_CURVE, 3.0, 3.5)
    assert not np.allclose(out_a, y_a)          # blurred
    y_n = 0.01 + 1e-3 * np.sin(400.0 * _NOOP)
    out_n = binning.smooth_to_native_r(_NOOP, y_n, _WL_R, _R_CURVE, 3.0, 3.5)
    assert np.array_equal(out_n, y_n)           # returned unchanged


@pytest.mark.parametrize("grid", [_ACTIVE, _NOOP], ids=["active", "noop"])
@pytest.mark.parametrize(
    "make_bad",
    [lambda g: np.full_like(g, np.nan),
     lambda g: np.full_like(g, -1.0),
     lambda g: np.full_like(g, np.inf),
     lambda g: np.ones(g.size + 1)],
    ids=["nan", "negative", "inf", "wrong-shape"])
def test_lsf_weight_validated_on_both_paths(grid, make_bad):
    """An invalid stellar-flux weight raises whether or not the kernel is
    resolved; the diagnosis must not depend on the instrument mode."""
    with pytest.raises(ValueError, match="weight"):
        _lsf(grid, make_bad(grid))


@pytest.mark.parametrize("grid", [_ACTIVE, _NOOP], ids=["active", "noop"])
def test_lsf_valid_weight_preserves_a_flat_depth_on_both_paths(grid):
    out = _lsf(grid, np.full_like(grid, 2.5))
    assert np.allclose(out, 0.01, rtol=0, atol=1e-15)


# --- native-R table ordering (MIRI LRS dispersion-order pixel grid) -----------
#
# The pandeia pixel grid is DISPERSION order, not wavelength order: MIRI LRS
# ships it descending (13.86 -> 5.02 um). The kernel width is read with
# np.interp, which requires increasing sample points and silently returns the
# end value everywhere on a descending table -- so a raw pass-through blurred
# the whole 5-12 um band at R(red end) = 42 instead of R = 42..208.

def test_smooth_refuses_a_descending_native_r_table():
    """A descending wl_r must RAISE, not silently return R(last) everywhere."""
    wl = np.geomspace(5.0, 12.0, 4000)
    y = np.full(wl.size, 0.01)
    wl_r = np.linspace(12.0, 5.0, 200)               # dispersion order
    r = np.linspace(208.0, 42.0, 200)                # paired with it
    with pytest.raises(ValueError, match="STRICTLY ASCENDING"):
        binning.smooth_to_native_r(wl, y, wl_r, r, 5.2, 11.8)
    # ... and the same table sorted is accepted
    binning.smooth_to_native_r(wl, y, wl_r[::-1], r[::-1], 5.2, 11.8)


def test_smooth_validates_the_r_curve_itself():
    wl = np.geomspace(5.0, 12.0, 2000)
    y = np.full(wl.size, 0.01)
    wl_r = np.geomspace(5.0, 12.0, 50)
    with pytest.raises(ValueError, match="r_curve"):
        binning.smooth_to_native_r(wl, y, wl_r, np.full(49, 100.0), 5.2, 11.8)
    with pytest.raises(ValueError, match="r_curve"):
        binning.smooth_to_native_r(wl, y, wl_r, np.full(50, 0.0), 5.2, 11.8)
    with pytest.raises(ValueError, match="r_curve"):
        bad = np.full(50, 100.0)
        bad[7] = np.nan
        binning.smooth_to_native_r(wl, y, wl_r, bad, 5.2, 11.8)
