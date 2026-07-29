"""2026-07-28 pre-release science-audit regression suite.

Three fail-fast/honesty defects found by independent numerical re-derivation of
the detection, noise, and measurement layers, each pinned here:

  * ONE transit-count validator: ``noise.pixel_depth_variance`` floored a
    fractional ``n_transits`` (2.7 -> 2, reported under the 2.7 label) while
    ``detect.sigma_at_transits`` already refused it, so the same input raised
    or was silently accepted depending on the entry point;
  * ``binning.smooth_to_native_r`` validated its stellar-flux ``weight`` only
    on the ACTIVE path, so a NaN/negative weight raised on MIRI LRS / PRISM and
    passed silently on the high-R gratings where the kernel is unresolved;
  * with NO noise floor set, the ``sig_inf`` "N->infinity limit" was computed
    from the 1e-30 floor clip and surfaced as a ~1e26 sigma ceiling in the GUI,
    and the correlated branch reported a meaningless reachability window
    (verified: with no floor there is no floor EXCESS, so ``build_cov`` returns
    None at every finite N and the score is monotone).
"""
import numpy as np
import pytest

from jwst_tool import binning, detect, fisher, noise as noise_mod


# --- one transit-count validator ------------------------------------------

def _mr(n_pix=10):
    return dict(flux=np.full(n_pix, 1e4), noise_1int=np.full(n_pix, 1e2),
                t_cycle_s=10.0)


def _scaler_result(n_pix=10):
    return dict(var_phot=np.full(n_pix, 1e-8), floor=np.zeros(n_pix),
                n_transits_eval=1)


@pytest.mark.parametrize("bad", [2.7, 0, -1, 0.5, -2.5, "3", None])
def test_bad_n_transits_refused_by_every_entry_point(bad):
    """A count that is not a positive integer raises in the variance AND in the
    scalers -- never floored into a different (optimistic) transit count."""
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
    """Not two copies that can drift apart again."""
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


# --- LSF weight validation on the no-op paths -----------------------------

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
    """An invalid stellar-flux weight raises regardless of whether the kernel
    is resolved -- the diagnosis must not depend on the instrument mode."""
    with pytest.raises(ValueError, match="weight"):
        _lsf(grid, make_bad(grid))


@pytest.mark.parametrize("grid", [_ACTIVE, _NOOP], ids=["active", "noop"])
def test_lsf_valid_weight_preserves_a_flat_depth_on_both_paths(grid):
    out = _lsf(grid, np.full_like(grid, 2.5))
    assert np.allclose(out, 0.01, rtol=0, atol=1e-15)


# --- no-floor limit semantics ---------------------------------------------

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
    """With no floor, sigma averages down without bound: the limit is inf, and
    the finite absurdity (~1e26 sigma from the 1e-30 clip) never reaches a
    user-facing surface."""
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
    """A target that needs more than N_TRANSITS_CAP transits is unreachable
    with sig_inf = inf -- 'ran out of transits', not 'a systematic caps it'."""
    r = _result("random", 0.0, signal_ppm=2.0)
    tt = detect.transits_to_target(r, 5.0)
    assert not tt["reachable"] and tt["sig_inf"] == float("inf")


def _fisher_result(scenario: str, floor_ppm: float, n=60) -> dict:
    wl = np.linspace(3.0, 5.0, n)
    r = _result(scenario, floor_ppm, n=n)
    rng = np.random.default_rng(0)
    jac = np.vstack([1e-4 * np.sin(2.0 * wl), 1e-4 * np.cos(3.0 * wl),
                     1e-4 * rng.normal(size=n)])
    r["jac_bins"] = jac
    r["jac_names"] = ["lnZ", "lnKzz", "lnR0"]
    r["sigma"] = np.maximum(np.sqrt(r["var_phot"]), r["floor"])
    r["cov"] = None
    return r


@pytest.mark.parametrize("scenario", ["random", "conservative"])
def test_fisher_no_floor_limit_is_zero_not_1e_minus_26(scenario):
    """Precision improves without bound with no floor, so the display-unit
    limit is exactly 0.0 and the scan stays monotone."""
    r = _fisher_result(scenario, 0.0)
    names = ["lnZ", "lnKzz", "lnR0"]
    tt = fisher.transits_to_target(r, names, "lnZ", 1e9,
                                   detect.sigma_at_transits)
    assert tt["sig_inf"] == 0.0
    assert tt["reachable"] and tt["n_last"] is None


@pytest.mark.parametrize("scenario", ["random", "conservative"])
def test_fisher_floored_limit_stays_finite_and_positive(scenario):
    r = _fisher_result(scenario, 100.0)
    names = ["lnZ", "lnKzz", "lnR0"]
    tt = fisher.transits_to_target(r, names, "lnZ", 1e9,
                                   detect.sigma_at_transits)
    assert np.isfinite(tt["sig_inf"]) and tt["sig_inf"] > 0.0
