"""Detection-score and noise-math tests: offset/segment profiling in the
matched-template score, loud input validation, transit-count validation, and
no-floor detection-limit semantics."""
import numpy as np
import pytest

from jwst_tool import detect, forward, instruments as ins, noise as noise_mod


def test_offset_profiled_out():
    """A pure constant depth signal carries no distinguishing information once
    the offset is profiled (it is the offset), and a SINGLE bin with a free
    offset has no shape information: score 0, never |s|/sigma (recheck P2-D).
    Without profiling the score is the plain quadrature sum."""
    sig = np.full(20, 3e-5)
    err = np.full(20, 1e-5)
    assert detect.detection_significance(sig, err, marginalize_offset=True) == \
        pytest.approx(0.0, abs=1e-9)
    raw = detect.detection_significance(sig, err, marginalize_offset=False)
    assert raw == pytest.approx(np.sqrt(np.sum((sig / err) ** 2)))
    # one bin + free offset -> 0; offset disabled -> |s|/sigma
    s1 = detect.detection_significance(np.array([3e-4]), np.array([1e-4]),
                                       marginalize_offset=True)
    assert s1 == pytest.approx(0.0, abs=1e-9)
    s2 = detect.detection_significance(np.array([3e-4]), np.array([1e-4]),
                                       marginalize_offset=False)
    assert s2 == pytest.approx(3.0, rel=1e-12)
    # an in-span signal scores EXACTLY 0 under any positive weights: the
    # projection's rounding residual (up to ~4e-8 relative) is snapped, so
    # "score > 0" is a real test of identifiability, never a coin flip
    rng = np.random.default_rng(0)
    for _ in range(50):
        err = 10.0 ** rng.uniform(-5, -3, sig.size)
        assert detect.detection_significance(sig, err) == 0.0


def test_segment_step_profiled_out_but_real_feature_survives():
    """A per-detector STEP must profile to ~0 once segment offsets are
    supplied (otherwise a calibration step reads as a molecular detection),
    while a localized band (not flat, not a step) keeps most of its S/N
    under the same offset+step profiling."""
    seg = np.array([0] * 20 + [1] * 20)
    err = np.full(seg.size, 1e-5)
    steps = detect._segment_rows(seg)
    step_signal = np.where(seg == 0, 2e-5, 9e-5)   # different level per detector
    # offset alone cannot remove a two-level step
    only_off = detect.detection_significance(step_signal, err,
                                             marginalize_offset=True)
    assert only_off > 5.0
    # offset + segment step removes it entirely
    with_seg = detect.detection_significance(step_signal, err, nuisance=steps,
                                             marginalize_offset=True)
    assert with_seg == pytest.approx(0.0, abs=1e-6)
    # a narrow SO2-like feature is barely touched by the same projection
    wl = np.linspace(3.0, 5.0, 40)
    feature = 8e-5 * np.exp(-0.5 * ((wl - 4.05) / 0.05) ** 2)
    raw = detect.detection_significance(feature, err, marginalize_offset=False)
    prof = detect.detection_significance(feature, err, nuisance=steps,
                                         marginalize_offset=True)
    assert prof > 0.8 * raw


def test_noise_inflation_scales_variance():
    """noise_inflation multiplies sigma (variance by its square) and averages
    down with transits like the photon term."""
    rng = np.random.default_rng(0)
    wl = np.sort(rng.uniform(3.0, 5.0, 300))
    flux = np.full(wl.size, 1e3)
    noise = np.full(wl.size, 30.0)
    mode_result = dict(wl=wl.tolist(), flux=flux.tolist(),
                       noise_1int=noise.tolist(), t_cycle_s=20.0)
    edges = noise_mod.make_bins(3.05, 4.95, 60.0)
    a = noise_mod.depth_error_bins(mode_result, edges, 3600.0, 3600.0, 1, 0.0)
    b = noise_mod.depth_error_bins(mode_result, edges, 3600.0, 3600.0, 1, 0.0,
                                   noise_inflation=1.2)
    assert np.allclose(b["var_phot"], a["var_phot"] * 1.2 ** 2)


def _lsf_mode_inputs(depth_baseline):
    """Minimal evaluate_mode inputs for a low-R mode where the native-R blur
    is active: PRISM-like R_native=100, narrow Jacobian feature."""
    wl_pix = np.linspace(1.0, 2.0, 600)
    flux = np.full(wl_pix.size, 1e6)
    mode_result = dict(
        wl=wl_pix.tolist(), flux=flux.tolist(),
        noise_1int=np.full(wl_pix.size, 1e3).tolist(),
        t_cycle_s=10.0, r_native=np.full(wl_pix.size, 100.0).tolist(),
        n_full_sat=np.zeros(wl_pix.size).tolist(),
        n_part_sat=np.zeros(wl_pix.size).tolist(),
        ngroup=10, sat_frac=0.5, saturated=False)
    wl_model = np.linspace(0.95, 2.05, 4000)
    jac_row = 1e-3 * np.exp(-0.5 * ((wl_model - 1.5) / 0.002) ** 2)
    model = dict(wl_um=wl_model, depth=depth_baseline(wl_model),
                 mols=["H2O"], jac=[jac_row], jac_names=["p0"])
    return mode_result, model


def test_jacobian_lsf_does_not_depend_on_baseline_shape():
    """The LSF is a linear operator on every vector; whether the BASELINE is a
    fixed point of the blur (e.g. exactly flat) must not decide whether
    Jacobian rows are smoothed. Same binned Jacobian for flat and broad-bump
    baselines, and both differ from the unsmoothed no-r_native case."""
    mr_flat, model_flat = _lsf_mode_inputs(lambda wl: np.zeros(wl.size))
    mr_bump, model_bump = _lsf_mode_inputs(
        lambda wl: 5e-3 * np.exp(-0.5 * ((wl - 1.5) / 0.2) ** 2))
    kw = dict(target_mol=None, R_bin=200.0, t_in_s=3600.0, t_out_s=3600.0,
              n_transits=1, floor_spec=None)
    r_flat = detect.evaluate_mode("nirspec_prism", mr_flat, model_flat, **kw)
    r_bump = detect.evaluate_mode("nirspec_prism", mr_bump, model_bump, **kw)
    assert np.allclose(r_flat["jac_bins"][0], r_bump["jac_bins"][0],
                       rtol=0, atol=1e-15)
    # and the blur genuinely acts on the narrow feature: an identical setup
    # with no r_native (no blur) must differ by many ppm at the feature
    mr_none, model_none = _lsf_mode_inputs(lambda wl: np.zeros(wl.size))
    mr_none["r_native"] = None
    r_none = detect.evaluate_mode("nirspec_prism", mr_none, model_none, **kw)
    assert np.max(np.abs(r_none["jac_bins"][0] - r_flat["jac_bins"][0])) > 5e-6


def test_duplicate_pixel_wavelengths_are_masked_not_raised():
    """Exact-duplicate pixel wavelengths (the G395H red-edge pileup) are
    dropped as degenerate by the measurement operator, never rejected."""
    mr, model = _lsf_mode_inputs(lambda wl: np.zeros(wl.size))
    mr["wl"][10] = mr["wl"][9]
    r = detect.evaluate_mode(
        "nirspec_prism", mr, model, target_mol=None, R_bin=200.0,
        t_in_s=3600.0, t_out_s=3600.0, n_transits=1, floor_spec=None)
    assert r["n_pix_degenerate_dropped"] == 2
    assert np.isfinite(r["median_sigma_ppm"])


def test_lsf_width_is_applied_as_r_over_width(monkeypatch):
    """detect blurs with R_refdata / instruments.LSF_WIDTH: bit-identical to
    handing the operator the divided curve with the table off, and broader
    than the unscaled blur on a narrow Jacobian feature."""
    key = "miri_lrs"
    wl_pix = np.linspace(5.0, 7.0, 600)
    r_nat = np.full(wl_pix.size, 100.0)

    def inputs(r_native):
        mode_result = dict(
            wl=wl_pix.tolist(), flux=np.full(wl_pix.size, 1e6).tolist(),
            noise_1int=np.full(wl_pix.size, 1e3).tolist(), t_cycle_s=10.0,
            r_native=r_native.tolist(), n_full_sat=np.zeros(wl_pix.size).tolist(),
            n_part_sat=np.zeros(wl_pix.size).tolist(),
            ngroup=10, sat_frac=0.5, saturated=False)
        wl_model = np.linspace(4.9, 7.1, 4000)
        jac = 1e-3 * np.exp(-0.5 * ((wl_model - 6.0) / 0.005) ** 2)
        model = dict(wl_um=wl_model, depth=np.zeros(wl_model.size),
                     mols=["H2O"], jac=[jac], jac_names=["p0"])
        return mode_result, model

    kw = dict(target_mol=None, R_bin=200.0, t_in_s=3600.0, t_out_s=3600.0,
              n_transits=1, floor_spec=None)
    with_table = detect.evaluate_mode(key, *inputs(r_nat), **kw)["jac_bins"][0]
    width = r_nat / ins.lsf_r(key, wl_pix, r_nat)
    assert width.min() > 1.0
    monkeypatch.setattr(ins, "LSF_WIDTH", {})
    divided = detect.evaluate_mode(key, *inputs(r_nat / width), **kw)["jac_bins"][0]
    unscaled = detect.evaluate_mode(key, *inputs(r_nat), **kw)["jac_bins"][0]
    assert np.array_equal(with_table, divided)
    assert with_table.max() < 0.85 * unscaled.max()


def test_r_bin_beyond_model_resolution_refuses():
    """Bins finer than the model's own grid would report interpolated
    structure the model does not contain (the pixel-level high-R-grating
    ask); refused loudly, never silently interpolated. At or below the
    model's resolving power the band mean is sound and the run proceeds."""
    kw = dict(target_mol=None, t_in_s=3600.0, t_out_s=3600.0,
              n_transits=1, floor_spec=None)
    mr, model = _lsf_mode_inputs(lambda wl: np.zeros(wl.size))
    with pytest.raises(ValueError, match="resolving power"):
        detect.evaluate_mode("nirspec_prism", mr, model, R_bin=20000.0, **kw)
    r = detect.evaluate_mode("nirspec_prism", mr, model, R_bin=200.0, **kw)
    assert np.isfinite(r["median_sigma_ppm"])


# --- fail-fast input validation ----------------------------------------------

def test_detection_significance_rejects_bad_inputs():
    good_s = np.array([3e-4, 1e-4, 2e-4])
    good_sig = np.full(3, 1e-4)
    # baseline still works
    assert np.isfinite(detect.detection_significance(good_s, good_sig))
    for signal, sigma in (
            (np.array([[1.0, 2.0]]), np.array([1.0, 1.0])),   # 2-D signal
            (np.array([1e-4, np.nan]), np.full(2, 1e-4)),     # non-finite
            (good_s, np.array([1e-4, 0.0, 1e-4])),            # sigma = 0
            (good_s, np.array([1e-4, np.nan, 1e-4])),         # non-finite
            (good_s, np.full(2, 1e-4))):                      # shape
        with pytest.raises(ValueError):
            detect.detection_significance(signal, sigma)
    with pytest.raises(ValueError):                           # nuisance shape
        detect.detection_significance(good_s, good_sig,
                                      nuisance=[np.ones(2)])


def _mr(n_pix=10):
    return dict(flux=np.full(n_pix, 1e4), noise_1int=np.full(n_pix, 1e2),
                t_cycle_s=10.0)


def test_n_transits_and_window_validation():
    """A non-positive-integer transit count raises in the variance AND the
    scalers, never floored into a different (optimistic) count; a transit
    window shorter than one integration cycle raises rather than silently
    pretending one integration fits; integer-valued counts are accepted and
    scale variance exactly as 1/N; both entry points share ONE validator so
    the rules cannot drift apart."""
    short = dict(wl=[3.0, 3.1], flux=[1e3, 1e3], noise_1int=[30.0, 30.0],
                 t_cycle_s=100.0)
    with pytest.raises(ValueError):
        noise_mod.pixel_depth_variance(short, t_in_s=50.0, t_out_s=3600.0,
                                       n_transits=1)
    # bad counts refused by every entry point
    scaler = dict(var_phot=np.full(4, 1e-8), floor=np.zeros(4),
                  n_transits_eval=1, wl=np.linspace(3, 4, 4))
    assert detect.sigma_at_transits(scaler, 3).shape == (4,)   # valid baseline
    for bad in (2.7, 0, -1, 0.5, -2.5, "3", None):
        with pytest.raises((ValueError, TypeError)):
            noise_mod.pixel_depth_variance(_mr(), 3600.0, 3600.0, bad)
        with pytest.raises((ValueError, TypeError)):
            detect.sigma_at_transits(scaler, bad)
    # integer-valued counts accepted, variance scales exactly as 1/N
    v1 = noise_mod.pixel_depth_variance(_mr(), 3600.0, 3600.0, 1)
    for good in (3, np.int64(4), 5.0):
        vn = noise_mod.pixel_depth_variance(_mr(), 3600.0, 3600.0, good)
        assert np.allclose(vn * int(good), v1, rtol=0, atol=0)
    # depth_error_bins records the validated count and refuses non-integers
    edges = np.linspace(3.0, 5.0, 6)
    mr = dict(wl=np.linspace(3.0, 5.0, 200).tolist(),
              flux=np.full(200, 5e3).tolist(),
              noise_1int=np.full(200, 70.0).tolist(), t_cycle_s=20.0)
    out = noise_mod.depth_error_bins(mr, edges, 3600.0, 3600.0, 4, None)
    assert out["n_transits"] == 4
    with pytest.raises(ValueError):
        noise_mod.depth_error_bins(mr, edges, 3600.0, 3600.0, 4.5, None)


# --- no-floor detection-limit semantics ---------------------------------------

def _result(floor_ppm: float, n=60, signal_ppm=150.0) -> dict:
    wl = np.linspace(3.0, 5.0, n)
    bump = signal_ppm * 1e-6 * np.exp(
        -0.5 * ((np.log(wl) - np.log(4.0)) / 0.10) ** 2)
    return dict(wl=wl, depth=0.02 + bump, depth_wo=np.full(n, 0.02),
                floor=np.full(n, floor_ppm * 1e-6),
                var_phot=np.full(n, 300e-6) ** 2, n_transits_eval=1,
                seg=np.zeros(n, int))


def test_no_floor_limit_semantics():
    """With no floor, sigma averages down without bound: the score is
    STRICTLY monotone in transits and the limit is inf, so a beyond-cap
    target reads 'ran out of transits' (inf limit), never 'a systematic
    caps it'. (The floored ceiling/short-circuit and floored-monotone
    semantics are pinned in test_transits_window.py.)"""
    r = _result(0.0)
    sc = [detect.detection_significance(
              np.asarray(r["depth"]) - np.asarray(r["depth_wo"]),
              detect.sigma_at_transits(r, n),
              nuisance=detect._result_nuisance(r))
          for n in (1, 2, 5, 20, 100, 500)]
    assert np.all(np.diff(sc) > 0.0)
    # no floor + tiny signal: unreachable but the limit is still inf
    tt = detect.transits_to_target(_result(0.0, signal_ppm=2.0), 5.0)
    assert not tt["reachable"] and tt["sig_inf"] == float("inf")


def test_projected_score_uses_headline_nuisance_space():
    """The event forecast must use the same local nuisance projection as the
    collaborator-facing score (rather than reverting to calibration-only),
    and the projected score refuses a result without named Jacobian rows."""
    r = _result(100.0)
    signal = np.asarray(r["depth"]) - np.asarray(r["depth_wo"])
    r["jac_bins"] = signal[None, :]
    r["jac_names"] = ["lnR0"]
    raw = detect.transits_to_target(r, 1.0)
    projected = detect.transits_to_target(r, 1.0, projected=True)
    assert raw["reachable"]
    assert not projected["reachable"]
    assert projected["sig_inf"] == pytest.approx(0.0, abs=1e-6)
    r2 = _result(100.0)
    r2["sigma"] = detect.sigma_at_transits(r2, 1)
    with pytest.raises(ValueError, match="jac_bins and jac_names"):
        detect.detection_score(r2, projected=True)

    # The label names the rows PRESENT, never the rows the statistic could use:
    # the shipped defaults free lnZ and dlnCO and carry only the appended lnR0.
    assert detect.metric_label({"sigma_detect_proj": float("nan")}) \
        == "calibration profiled"
    lab = lambda names: detect.metric_label(
        {"sigma_detect_proj": 1.0, "jac_names": names})
    assert lab(["lnZ", "dlnCO", "lnR0"]) == "calibration + R0 projected"
    assert lab(["Tirr", "log_kappa_cloud", "lnR0"]) == \
        "calibration + T-P + cloud + R0 projected"


# --- MIRI LRS ships its pixel grid in DISPERSION order (descending) -----------

def _miri_descending_inputs(descending):
    """MIRI-LRS-shaped evaluate_mode inputs with a real R(lambda) ramp.

    ``descending=True`` reproduces the worker payload exactly: pandeia returns
    the LRS grid in dispersion order (13.86 -> 5.02 um) with r_native paired
    element-wise, so wl/flux/noise/r_native are all reversed together. The
    binned depth must not depend on that ordering.
    """
    wl_asc = np.linspace(5.0, 12.0, 372)
    # native R ramps ~42 -> ~208 across the LRS band (jwst_miri_p750l_disp)
    r_asc = np.linspace(42.0, 208.0, wl_asc.size)
    # throughput falls steeply to the red, like the real extracted count rate
    flux_asc = 1e6 * np.exp(-(wl_asc - 5.0) / 2.0)
    sl = slice(None, None, -1) if descending else slice(None)
    mode_result = dict(
        wl=wl_asc[sl].tolist(), flux=flux_asc[sl].tolist(),
        noise_1int=np.full(wl_asc.size, 1e3)[sl].tolist(),
        t_cycle_s=10.0, r_native=r_asc[sl].tolist(),
        n_full_sat=np.zeros(wl_asc.size).tolist(),
        n_part_sat=np.zeros(wl_asc.size).tolist(),
        ngroup=10, sat_frac=0.5, saturated=False)
    # a structured depth so the kernel WIDTH matters (a flat depth is a fixed
    # point of any blur and would make this test vacuous)
    wl_model = np.linspace(4.9, 12.1, 8000)
    depth = 0.02 + 1e-3 * np.sin(60.0 * np.log(wl_model))
    model = dict(wl_um=wl_model, depth=depth, mols=["H2O"])
    return mode_result, model


def test_miri_lsf_uses_local_native_r_not_the_dispersion_order_end_value():
    """The native-R blur must read R(lambda), not R at the end of the payload.

    The MIRI LRS pixel grid arrives DESCENDING; np.interp requires ascending
    sample points and silently returns the last table value everywhere on a
    reversed one, so passing the raw grid blurred the entire 5-12 um band at
    the red-end R (~42) instead of R = 42..208. Both orderings describe the
    same instrument, so both must give the same binned depth.
    """
    kw = dict(target_mol=None, R_bin=100.0, t_in_s=3600.0, t_out_s=3600.0,
              n_transits=1, floor_spec=None)
    mr_desc, model_desc = _miri_descending_inputs(descending=True)
    mr_asc, model_asc = _miri_descending_inputs(descending=False)
    r_desc = detect.evaluate_mode("miri_lrs", mr_desc, model_desc, **kw)
    r_asc = detect.evaluate_mode("miri_lrs", mr_asc, model_asc, **kw)
    assert np.allclose(r_desc["wl"], r_asc["wl"], rtol=0, atol=1e-12)
    # ordering is a payload convention, never a physics change
    assert np.allclose(r_desc["depth"], r_asc["depth"], rtol=0, atol=1e-12), (
        "binned depth changed with the pandeia grid ORDER: the native-R blur "
        "is reading the resolving-power table out of order (max diff "
        f"{np.max(np.abs(r_desc['depth'] - r_asc['depth'])) * 1e6:.1f} ppm)")
    # guard the fixture: the blur must genuinely depend on R here, or the
    # equality above would hold for the wrong reason
    mr_flat, model_flat = _miri_descending_inputs(descending=False)
    mr_flat["r_native"] = np.full(len(mr_flat["wl"]), 42.0).tolist()
    r_flat = detect.evaluate_mode("miri_lrs", mr_flat, model_flat, **kw)
    assert np.max(np.abs(r_flat["depth"] - r_asc["depth"])) > 1e-6, \
        "fixture is insensitive to R(lambda); the ordering test proves nothing"


def test_worker_warnings_and_the_ramp_advisory_reach_the_evaluated_result():
    """The worker computes a per-mode `warnings` dict; evaluate_mode was
    dropping it, so nothing downstream could ever show it. The short-ramp
    advisory rides the same dict rather than opening a second channel."""
    m = ins.MODES["miri_lrs"]
    wl_pix = np.linspace(m["wl_min"] * 1.001, m["wl_max"] * 0.999, 300)
    mr = dict(wl=wl_pix.tolist(), flux=np.full(wl_pix.size, 1e6).tolist(),
              noise_1int=np.full(wl_pix.size, 1e3).tolist(),
              t_cycle_s=20.0, r_native=None,
              n_full_sat=np.zeros(wl_pix.size).tolist(),
              n_part_sat=np.zeros(wl_pix.size).tolist(),
              ngroup=5, sat_frac=0.5, saturated=False,
              warnings={"full_saturated": "96 pixels saturated",
                        "bad_waveref": "out of range, using 4.36 to select "
                                       "diagnostic planes instead"})
    wl_model = np.linspace(m["wl_min"] * 0.95, m["wl_max"] * 1.05, 2000)
    model = dict(wl_um=wl_model, depth=np.full(wl_model.size, 0.01),
                 mols=["X"], wo_mols=[])
    kw = dict(target_mol=None, R_bin=100.0, t_in_s=3600.0, t_out_s=3600.0,
              n_transits=1, floor_spec=None)
    w = detect.evaluate_mode("miri_lrs", mr, model, **kw)["warnings"]
    assert w["full_saturated"] == "96 pixels saturated"   # forwarded
    assert "bad_waveref" not in w, \
        "pandeia housekeeping about 2D diagnostic planes is not an observing risk"
    # the advisory reaches the result; its threshold table is pinned directly
    # on ins.ngroup_advisory in test_instruments_registry
    assert "5-group" in w["ngroup"]                 # 2-5 groups on MIRI


def _emission_inputs(mode_key, thin_below_um):
    """evaluate_mode inputs for an emission model whose column bottom is thin
    only shortward of `thin_below_um`, carrying a negligible share of the
    1-15 um energy."""
    m = ins.MODES[mode_key]
    lo, hi = max(1.02, m["wl_min"]), min(14.8, m["wl_max"])
    wl_pix = np.linspace(lo, hi, 400)
    mode_result = dict(
        wl=wl_pix.tolist(), flux=np.full(wl_pix.size, 1e6).tolist(),
        noise_1int=np.full(wl_pix.size, 1e3).tolist(),
        t_cycle_s=10.0, r_native=np.full(wl_pix.size, 400.0).tolist(),
        n_full_sat=np.zeros(wl_pix.size).tolist(),
        n_part_sat=np.zeros(wl_pix.size).tolist(),
        ngroup=10, sat_frac=0.5, saturated=False)
    wl = np.geomspace(1.0, 15.0, 800)
    thin = wl < thin_below_um
    fp = np.where(thin, 1e-4, 1.0)          # the leak carries almost no energy
    depth = 1e-3 * np.ones(wl.size)
    model = dict(
        wl_um=wl, depth=depth, mols=["H2O", "CO2"], science_mode="emission",
        wo_mols=np.array(["CO2"]), depth_wo=depth[None, :] * 0.99,
        emis_thin_flux_frac_wo=np.array([0.0]),
        emis_tau_bottom=np.where(thin, 0.5, 50.0), fp_flux=fp,
        emis_tau_bottom_wo=np.where(thin, 0.5, 50.0)[None, :],
        fs_flux=np.ones(wl.size), emis_depth_norm=np.array([1.0]))
    return mode_result, model


def test_a_globally_safe_emission_column_is_refused_on_a_band_that_only_sees_the_leak():
    """The 1-15 um certificate cannot certify a narrower observing band. A
    column thin only shortward of 1.9 um leaks 0.004% of the planet's total
    emission -- far under the 1% gate -- but G140H measures 1.0-1.83 um,
    where the leak is ALL of it. The wide mode over the same model must still
    pass, so this is a band restriction, not a tightened threshold."""
    kw = dict(R_bin=100.0, t_in_s=3600.0, t_out_s=3600.0, n_transits=1,
              floor_spec=None)

    mr, model = _emission_inputs("nirspec_g140h", 1.9)
    # the model as a whole passes the unchanged band-global gate ...
    nu = 1e4 / model["wl_um"]
    assert forward.thin_flux_fraction(
        model["emis_tau_bottom"], model["fp_flux"], nu) \
        < forward.EMIS_THIN_FLUX_FRAC
    # ... and is still refused on the band that only sees the leak
    with pytest.raises(ValueError, match="optically thin"):
        detect.evaluate_mode("nirspec_g140h", mr, model, None, **kw)

    mr_w, model_w = _emission_inputs("nirspec_prism", 1.9)
    assert detect.evaluate_mode("nirspec_prism", mr_w, model_w, None,
                                **kw)["n_bins"] > 10

    # a wholly thick column passes everywhere, target stage included
    mr_ok, model_ok = _emission_inputs("nirspec_g140h", 0.0)
    assert detect.evaluate_mode("nirspec_g140h", mr_ok, model_ok, "CO2",
                                **kw)["n_bins"] > 5

    # missing or malformed evidence refuses; it never defaults to safe
    for key, bad in (("emis_tau_bottom", None),
                     ("fp_flux", np.zeros(3)),
                     ("emis_tau_bottom", np.full(model_ok["wl_um"].size, -1.0)),
                     ("emis_depth_norm", np.array([0.0])),
                     ("depth_wo", np.full_like(model_ok["depth_wo"], np.nan))):
        broken = dict(model_ok)
        broken.pop(key) if bad is None else broken.__setitem__(key, bad)
        with pytest.raises(ValueError, match="missing|malformed"):
            detect.evaluate_mode("nirspec_g140h", mr_ok, broken, "CO2", **kw)

    # A physical refusal must use the configuration-named exclusion path,
    # including when the cached global certificate already refuses it.
    globally_bad = dict(model_ok, emis_thin_flux_frac_wo=np.array([0.02]))
    with pytest.raises(detect.ModeUnusable, match="optically thin"):
        detect.evaluate_mode("nirspec_g140h", mr_ok, globally_bad, "CO2", **kw)


def test_emission_gate_cannot_be_diluted_by_masked_pixels_or_band_margins():
    kw = dict(R_bin=100.0, t_in_s=3600.0, t_out_s=3600.0, n_transits=1,
              floor_spec=None)
    for mode in ("nirspec_prism", "nirspec_g140h"):
        mr, model = _emission_inputs(mode, 0.0)
        wl = model["wl_um"]
        if mode == "nirspec_prism":
            pix = np.asarray(mr["wl"])
            mr["n_full_sat"] = ((pix > 2.0) & (pix < 4.0)).astype(int).tolist()
            mr.pop("r_native")
            bright_unmeasured = (wl > 2.1) & (wl < 3.9)
        else:
            bright_unmeasured = (wl < 1.01) | (wl > 1.85)
        model["fp_flux"] = np.where(bright_unmeasured, 1e8, 1.0)
        model["emis_tau_bottom"] = np.where(bright_unmeasured, 50.0, 0.5)
        with pytest.raises(detect.ModeUnusable, match="optically thin"):
            detect.evaluate_mode(mode, mr, model, None, **kw)


def test_nircam_data_excess_rides_the_warnings_channel():
    m = ins.MODES["nircam_f444w"]
    wl_pix = np.linspace(m["wl_min"] * 1.001, m["wl_max"] * 0.999, 300)
    mr = dict(wl=wl_pix.tolist(), flux=np.full(wl_pix.size, 1e6).tolist(),
              noise_1int=np.full(wl_pix.size, 1e3).tolist(),
              t_cycle_s=20.0, r_native=None,
              n_full_sat=np.zeros(wl_pix.size).tolist(),
              n_part_sat=np.zeros(wl_pix.size).tolist(),
              ngroup=100, sat_frac=0.5, saturated=False)
    wl_model = np.linspace(m["wl_min"] * 0.95, m["wl_max"] * 1.05, 2000)
    model = dict(wl_um=wl_model, depth=np.full(wl_model.size, 0.01),
                 mols=["X"], wo_mols=[])
    kw = dict(target_mol=None, R_bin=100.0, n_transits=1, floor_spec=None)
    # the parity artifact's visit: 27.8 GB, above the 15 GB recommendation
    w = detect.evaluate_mode("nircam_f444w", mr, model,
                             t_in_s=2.8 * 3600.0, t_out_s=2.819 * 3600.0, **kw)["warnings"]
    assert "27.8 GB" in w["data_excess"] and "above" in w["data_excess"]
    # a one-hour visit stays under PandExo's 5 GB flag: silent
    w = detect.evaluate_mode("nircam_f444w", mr, model,
                             t_in_s=1800.0, t_out_s=1800.0, **kw)["warnings"]
    assert "data_excess" not in w
