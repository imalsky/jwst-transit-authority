"""Detection-score and noise-math tests: offset/segment profiling in the
matched-template score, loud input validation, transit-count validation, and
no-floor detection-limit semantics."""
import numpy as np
import pytest

from jwst_tool import detect, instruments as ins, noise as noise_mod


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


def _miri_mode_inputs(ngroup):
    """Minimal evaluate_mode inputs inside the MIRI LRS band (5-12 um)."""
    wl_pix = np.linspace(5.0, 7.0, 300)
    mode_result = dict(
        wl=wl_pix.tolist(), flux=np.full(wl_pix.size, 1e6).tolist(),
        noise_1int=np.full(wl_pix.size, 1e3).tolist(),
        t_cycle_s=10.0, r_native=None,
        n_full_sat=np.zeros(wl_pix.size).tolist(),
        n_part_sat=np.zeros(wl_pix.size).tolist(),
        ngroup=ngroup, sat_frac=0.5, saturated=False)
    wl_model = np.linspace(4.9, 7.1, 2000)
    model = dict(wl_um=wl_model, depth=np.zeros(wl_model.size),
                 mols=["H2O"], jac=[np.zeros(wl_model.size)],
                 jac_names=["p0"])
    return mode_result, model


def test_ramp_and_budget_disclosures():
    """Short ramps and incomplete searches are DISCLOSED, never demoted.

    A ramp below ngroup_warn_below (PRISM: 2) gets a warning naming both
    group counts and never marks the row saturated; a worker payload with
    ramp_search_complete=False carries a budget warning (absent field, older
    payloads, stays silent); a 2-group MIRI selection is MIRI's shortest
    permitted ramp and gets a DISTINCT operational warning telling the user
    to confirm approval requirements in APT."""
    kw = dict(target_mol=None, R_bin=200.0, t_in_s=3600.0, t_out_s=3600.0,
              n_transits=1, floor_spec=None)
    mr, model = _lsf_mode_inputs(lambda wl: np.zeros(wl.size))
    mr["ngroup"] = 1
    r = detect.evaluate_mode("nirspec_prism", mr, model, **kw)
    hits = [w for w in r["warnings"]
            if "below this mode's STScI-recommended ramp" in w]
    assert len(hits) == 1
    assert "1 group(s) per integration" in hits[0]
    # the NIRSpec-specific reason, paraphrasing jwst-docs "NIRSpec Detector
    # Recommended Strategies" (the minimum recommended ramp is 2 groups; 1 is
    # for very bright BOTS targets). It must NOT be a generic short-ramp note.
    assert "very bright BOTS target" in hits[0]
    assert r["saturated"] is False
    # at the threshold: no short-ramp warning
    mr2, model2 = _lsf_mode_inputs(lambda wl: np.zeros(wl.size))
    mr2["ngroup"] = 2
    r2 = detect.evaluate_mode("nirspec_prism", mr2, model2, **kw)
    assert not [w for w in r2["warnings"]
                if "below this mode's STScI-recommended ramp" in w]
    assert not any("calculation budget" in w for w in r2["warnings"])
    # incomplete ramp search is disclosed
    mr3, model3 = _lsf_mode_inputs(lambda wl: np.zeros(wl.size))
    mr3["ramp_search_complete"] = False
    r3 = detect.evaluate_mode("nirspec_prism", mr3, model3, **kw)
    assert any("calculation budget" in w for w in r3["warnings"])
    # MIRI floor ramp gets its own operational warning
    kw_miri = dict(kw, R_bin=100.0)
    mr4, model4 = _miri_mode_inputs(ngroup=2)
    r4 = detect.evaluate_mode("miri_lrs", mr4, model4, **kw_miri)
    hits4 = [w for w in r4["warnings"] if w.startswith("MIRI floor ramp")]
    assert len(hits4) == 1 and "approval" in hits4[0]
    # 3 groups: still below the recommended 6, but NOT the floor warning
    mr5, model5 = _miri_mode_inputs(ngroup=3)
    r5 = detect.evaluate_mode("miri_lrs", mr5, model5, **kw_miri)
    assert not [w for w in r5["warnings"] if w.startswith("MIRI floor ramp")]
    assert any("below this mode's STScI-recommended ramp" in w
               for w in r5["warnings"])


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
    with pytest.raises(ValueError, match="shorter than one integration"):
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
    # one shared validator, not two copies that can drift apart
    assert detect._n_transits is noise_mod.n_transits_int
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
    assert r_desc["lsf_applied"] and r_asc["lsf_applied"]
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


@pytest.mark.parametrize(
    "mode_key,t_cycle,expect",
    [("nirspec_prism", 1600.0, True),    # above STScI's 1500 s NIRSpec advice
     ("nirspec_prism", 1400.0, False),
     ("miri_lrs", 2100.0, True),         # above APT's 2000 s MIRI hard limit
     ("miri_lrs", 1900.0, False),
     ("niriss_soss", 5000.0, False),     # no entry: the 30-group cap binds
     ("nircam_f444w", 5000.0, False)])   # no entry: the 100-group cap binds
def test_over_long_integrations_are_disclosed(mode_key, t_cycle, expect):
    """The ramp search maximizes groups, so on a faint target it can run past
    what STScI advises (NIRSpec: 1,500 s) or APT allows (MIRI: 2,000 s). The
    tool never shortens a ramp for the user, so it must say so instead.
    NIRISS and NIRCam have no limit entry -- their group caps bind first."""
    kw = dict(target_mol=None, R_bin=100.0, t_in_s=3.0e5, t_out_s=3.0e5,
              n_transits=1, floor_spec=None)
    m = ins.MODES[mode_key]
    wl_pix = np.linspace(m["wl_min"] * 1.001, m["wl_max"] * 0.999, 600)
    mr = dict(wl=wl_pix.tolist(), flux=np.full(wl_pix.size, 1e6).tolist(),
              noise_1int=np.full(wl_pix.size, 1e3).tolist(),
              t_cycle_s=t_cycle, r_native=None,
              n_full_sat=np.zeros(wl_pix.size).tolist(),
              n_part_sat=np.zeros(wl_pix.size).tolist(),
              ngroup=int(1e4), sat_frac=0.79, saturated=False)
    wl_model = np.linspace(m["wl_min"] * 0.95, m["wl_max"] * 1.05, 4000)
    model = dict(wl_um=wl_model, depth=np.full(wl_model.size, 0.01),
                 mols=[], wo_mols=[])
    r = detect.evaluate_mode(mode_key, mr, model, **kw)
    hits = [w for w in r["warnings"] if "integration is" in w]
    assert bool(hits) is expect, (mode_key, t_cycle, hits)
    if expect:
        limit = ins.INT_LENGTH_LIMIT_S[m["instrument"]]
        assert f"{limit[0]:.0f} s {limit[1]}" in hits[0]


def test_a_secondary_order_declares_its_shared_readout_and_saturation():
    """SOSS orders 1 and 2 are ONE readout of SUBSTRIP256, so the order-2 row
    must say that selecting both costs one observation, and that its
    saturation verdict comes from the brighter order-1 trace (Pandeia's
    fraction_saturation is scene-wide). Order 1 gets neither note."""
    kw = dict(target_mol=None, R_bin=100.0, t_in_s=3600.0, t_out_s=3600.0,
              n_transits=1, floor_spec=None)

    def _inputs(mode_key, saturated):
        m = ins.MODES[mode_key]
        wl_pix = np.linspace(m["wl_min"] * 1.001, m["wl_max"] * 0.999, 400)
        mr = dict(wl=wl_pix.tolist(), flux=np.full(wl_pix.size, 1e6).tolist(),
                  noise_1int=np.full(wl_pix.size, 1e3).tolist(),
                  t_cycle_s=20.0, r_native=None,
                  n_full_sat=np.zeros(wl_pix.size).tolist(),
                  n_part_sat=np.zeros(wl_pix.size).tolist(),
                  ngroup=5, sat_frac=1.4 if saturated else 0.5,
                  saturated=saturated)
        wl_model = np.linspace(m["wl_min"] * 0.95, m["wl_max"] * 1.05, 4000)
        return mr, dict(wl_um=wl_model, depth=np.full(wl_model.size, 0.01),
                        mols=[], wo_mols=[])

    def _notes(mode_key, saturated):
        mr, model = _inputs(mode_key, saturated)
        return list(detect.evaluate_mode(mode_key, mr, model, **kw)["warnings"])

    shared = "shares one detector readout with order 1"
    verdict = "set by the brighter order-1 trace"
    assert any(shared in w for w in _notes("niriss_soss_ord2", False))
    assert not any(verdict in w for w in _notes("niriss_soss_ord2", False))
    sat2 = _notes("niriss_soss_ord2", True)
    assert any(shared in w for w in sat2) and any(verdict in w for w in sat2)
    ord1 = _notes("niriss_soss", True)
    assert not any(shared in w or verdict in w for w in ord1)
