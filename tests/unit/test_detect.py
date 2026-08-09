"""Detection-score and noise-math tests: offset/segment profiling in the
matched-template score, loud input validation, transit-count validation, and
no-floor detection-limit semantics."""
import numpy as np
import pytest

from jwst_tool import detect, noise as noise_mod


def test_constant_offset_profiled_out():
    """A pure constant depth signal carries no distinguishing information once
    the offset is profiled (it is the offset)."""
    sig = np.full(20, 3e-5)
    err = np.full(20, 1e-5)
    assert detect.detection_significance(sig, err, marginalize_offset=True) == \
        pytest.approx(0.0, abs=1e-9)
    # without profiling it is just the quadrature sum
    raw = detect.detection_significance(sig, err, marginalize_offset=False)
    assert raw == pytest.approx(np.sqrt(np.sum((sig / err) ** 2)))


def test_segment_step_profiled_out():
    """A per-detector STEP must profile to ~0 once segment offsets are
    supplied; otherwise a calibration step reads as a molecular detection."""
    seg = np.array([0] * 15 + [1] * 15)
    signal = np.where(seg == 0, 2e-5, 9e-5)        # different level per detector
    err = np.full(seg.size, 1e-5)
    steps = detect._segment_rows(seg)
    # offset alone cannot remove a two-level step
    only_off = detect.detection_significance(signal, err, marginalize_offset=True)
    assert only_off > 5.0
    # offset + segment step removes it entirely
    with_seg = detect.detection_significance(signal, err, nuisance=steps,
                                             marginalize_offset=True)
    assert with_seg == pytest.approx(0.0, abs=1e-6)


def test_real_feature_survives_offset_and_step():
    """A localized band (not flat, not a step) keeps most of its S/N under
    offset+step profiling."""
    seg = np.array([0] * 20 + [1] * 20)
    wl = np.linspace(3.0, 5.0, 40)
    signal = 8e-5 * np.exp(-0.5 * ((wl - 4.05) / 0.05) ** 2)   # narrow SO2-like
    err = np.full(wl.size, 1e-5)
    steps = detect._segment_rows(seg)
    raw = detect.detection_significance(signal, err, marginalize_offset=False)
    prof = detect.detection_significance(signal, err, nuisance=steps,
                                         marginalize_offset=True)
    assert prof > 0.8 * raw       # a real feature is barely touched


def test_pixel_variance_raises_on_subcycle_window():
    """A transit window shorter than one integration cycle must raise, never
    silently pretend one integration fits."""
    mode_result = dict(wl=[3.0, 3.1], flux=[1e3, 1e3],
                       noise_1int=[30.0, 30.0], t_cycle_s=100.0)
    with pytest.raises(ValueError, match="shorter than one integration"):
        noise_mod.pixel_depth_variance(mode_result, t_in_s=50.0, t_out_s=3600.0,
                                       n_transits=1)
    with pytest.raises(ValueError, match="n_transits"):
        noise_mod.pixel_depth_variance(mode_result, t_in_s=3600.0,
                                       t_out_s=3600.0, n_transits=0)


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


def test_one_bin_offset_profiles_to_zero():
    """With a free constant offset a single bin has no shape information: the
    score must be 0, never |s|/sigma."""
    s = detect.detection_significance(np.array([3e-4]), np.array([1e-4]),
                                      marginalize_offset=True)
    assert s == pytest.approx(0.0, abs=1e-9)
    # consistency: two identical bins were already 0; one bin now matches
    s2 = detect.detection_significance(np.array([3e-4, 3e-4]),
                                       np.array([1e-4, 1e-4]),
                                       marginalize_offset=True)
    assert s2 == pytest.approx(0.0, abs=1e-9)
    # with the offset explicitly disabled the one-bin score is |s|/sigma
    s3 = detect.detection_significance(np.array([3e-4]), np.array([1e-4]),
                                       marginalize_offset=False)
    assert s3 == pytest.approx(3.0, rel=1e-12)


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


def test_short_ramp_below_recommended_minimum_is_flagged_not_demoted():
    """A ramp below ngroup_warn_below (PRISM: 2) gets a disclosure warning
    naming both group counts; at or above the threshold there is no such
    warning. The row is never marked saturated by the warning."""
    kw = dict(target_mol=None, R_bin=200.0, t_in_s=3600.0, t_out_s=3600.0,
              n_transits=1, floor_spec=None)
    mr, model = _lsf_mode_inputs(lambda wl: np.zeros(wl.size))
    mr["ngroup"] = 1
    r = detect.evaluate_mode("nirspec_prism", mr, model, **kw)
    hits = [w for w in r["warnings"]
            if "below the STScI-recommended minimum" in w]
    assert len(hits) == 1
    assert "1 groups per integration" in hits[0] and "of 2" in hits[0]
    assert r["saturated"] is False

    mr2, model2 = _lsf_mode_inputs(lambda wl: np.zeros(wl.size))
    mr2["ngroup"] = 2
    r2 = detect.evaluate_mode("nirspec_prism", mr2, model2, **kw)
    assert not [w for w in r2["warnings"]
                if "below the STScI-recommended minimum" in w]


# --- fail-fast input validation ----------------------------------------------

def test_detection_significance_rejects_bad_inputs():
    good_s = np.array([3e-4, 1e-4, 2e-4])
    good_sig = np.full(3, 1e-4)
    # baseline still works
    assert np.isfinite(detect.detection_significance(good_s, good_sig))
    with pytest.raises(ValueError, match="signal"):
        detect.detection_significance(np.array([[1.0, 2.0]]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="signal"):
        detect.detection_significance(np.array([1e-4, np.nan]), np.full(2, 1e-4))
    with pytest.raises(ValueError, match="sigma"):
        detect.detection_significance(good_s, np.array([1e-4, 0.0, 1e-4]))
    with pytest.raises(ValueError, match="sigma"):
        detect.detection_significance(good_s, np.array([1e-4, np.nan, 1e-4]))
    with pytest.raises(ValueError, match="sigma"):
        detect.detection_significance(good_s, np.full(2, 1e-4))       # shape
    with pytest.raises(ValueError, match="nuisance row"):
        detect.detection_significance(good_s, good_sig,
                                      nuisance=[np.ones(2)])
    # covariance: non-square, non-finite, non-symmetric, non-PD all raise
    with pytest.raises(ValueError, match="cov shape"):
        detect.detection_significance(good_s, good_sig, cov=np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        detect.detection_significance(good_s, good_sig,
                                      cov=np.array([[1.0, 2.0, 0.0],
                                                    [0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 1.0]]) * 1e-8)
    with pytest.raises(ValueError, match="positive-definite"):
        detect.detection_significance(good_s, good_sig, cov=-np.eye(3) * 1e-8)


def test_sigma_and_cov_at_transits_reject_bad_n():
    result = dict(n_transits_eval=1, var_phot=np.full(4, 1e-8),
                  floor=np.zeros(4), wl=np.linspace(3, 4, 4), scenario="random")
    assert detect.sigma_at_transits(result, 3).shape == (4,)   # valid
    for bad in (0, -2, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            detect.sigma_at_transits(result, bad)
        with pytest.raises(ValueError, match="positive integer"):
            detect.cov_at_transits(result, bad)


# --- transit-count validation and no-floor limits ----------------------------

def _mr(n_pix=10):
    return dict(flux=np.full(n_pix, 1e4), noise_1int=np.full(n_pix, 1e2),
                t_cycle_s=10.0)


def _scaler_result(n_pix=10):
    return dict(var_phot=np.full(n_pix, 1e-8), floor=np.zeros(n_pix),
                n_transits_eval=1)


@pytest.mark.parametrize("bad", [2.7, 0, -1, 0.5, -2.5, "3", None])
def test_bad_n_transits_refused_by_every_entry_point(bad):
    """A non-positive-integer count raises in the variance AND the scalers,
    never floored into a different (optimistic) transit count."""
    with pytest.raises((ValueError, TypeError)):
        noise_mod.pixel_depth_variance(_mr(), 3600.0, 3600.0, bad)
    with pytest.raises((ValueError, TypeError)):
        detect.sigma_at_transits(_scaler_result(), bad)


@pytest.mark.parametrize("good", [1, 3, np.int64(4), 5.0])
def test_integer_valued_n_transits_accepted_and_scales_as_1_over_n(good):
    v1 = noise_mod.pixel_depth_variance(_mr(), 3600.0, 3600.0, 1)
    vn = noise_mod.pixel_depth_variance(_mr(), 3600.0, 3600.0, good)
    assert np.allclose(vn * int(good), v1, rtol=0, atol=0)


def test_detect_and_noise_share_one_validator():
    """One shared validator, not two copies that can drift apart."""
    assert detect._n_transits is noise_mod.n_transits_int


def test_depth_error_bins_records_validated_count():
    edges = np.linspace(3.0, 5.0, 6)
    mr = dict(wl=np.linspace(3.0, 5.0, 200).tolist(),
              flux=np.full(200, 5e3).tolist(),
              noise_1int=np.full(200, 70.0).tolist(), t_cycle_s=20.0)
    out = noise_mod.depth_error_bins(mr, edges, 3600.0, 3600.0, 4, None)
    assert out["n_transits"] == 4
    with pytest.raises(ValueError):
        noise_mod.depth_error_bins(mr, edges, 3600.0, 3600.0, 4.5, None)


def _result(scenario: str, floor_ppm: float, n=60, signal_ppm=150.0) -> dict:
    wl = np.linspace(3.0, 5.0, n)
    bump = signal_ppm * 1e-6 * np.exp(
        -0.5 * ((np.log(wl) - np.log(4.0)) / 0.10) ** 2)
    return dict(wl=wl, depth=0.02 + bump, depth_wo=np.full(n, 0.02),
                floor=np.full(n, floor_ppm * 1e-6),
                var_phot=np.full(n, 300e-6) ** 2, n_transits_eval=1,
                scenario=scenario, seg=np.zeros(n, int),
                slope_rows=np.zeros((0, n)))


@pytest.mark.parametrize("scenario", ["random", "conservative"])
def test_no_floor_gives_infinite_detect_limit_not_1e26(scenario):
    """With no floor, sigma averages down without bound: the limit is inf and
    the finite clip absurdity never reaches a user-facing surface."""
    tt = detect.transits_to_target(_result(scenario, 0.0), 8.0)
    assert tt["sig_inf"] == float("inf")
    assert tt["reachable"] and tt["n"] is not None
    # no floor -> no floor EXCESS -> no correlated term -> monotone scan, so
    # no reachability WINDOW is reported even under a correlated preset
    assert tt["n_last"] is None


@pytest.mark.parametrize("scenario", ["random", "conservative"])
def test_no_floor_correlated_covariance_is_absent_and_score_is_monotone(
        scenario):
    r = _result(scenario, 0.0)
    assert all(detect.cov_at_transits(r, n) is None for n in (1, 5, 50, 500))
    sc = [detect.detection_significance(
              np.asarray(r["depth"]) - np.asarray(r["depth_wo"]),
              detect.sigma_at_transits(r, n),
              nuisance=detect._result_nuisance(r),
              cov=detect.cov_at_transits(r, n))
          for n in (1, 2, 5, 20, 100, 500)]
    assert np.all(np.diff(sc) > 0.0)


@pytest.mark.parametrize("scenario", ["random", "conservative"])
def test_floored_limit_is_finite_and_unchanged_by_the_fix(scenario):
    """The floored path keeps its exact previous behavior."""
    tt = detect.transits_to_target(_result(scenario, 100.0), 8.0)
    assert np.isfinite(tt["sig_inf"]) and 0.0 < tt["sig_inf"] < 100.0
    assert not tt["reachable"]          # target above the floor-capped limit


def test_no_floor_unreachable_target_still_reports_inf_limit():
    """Beyond-cap targets are unreachable with sig_inf = inf: 'ran out of
    transits', not 'a systematic caps it'."""
    r = _result("random", 0.0, signal_ppm=2.0)
    tt = detect.transits_to_target(r, 5.0)
    assert not tt["reachable"] and tt["sig_inf"] == float("inf")
