"""External-audit regression suite: three confirmed defects, pinned.

  * the minimum noise floor has exact PandExo semantics: sigma_final =
    max(sigma_random, floor) on the FINAL bins -- no quadrature, no
    sqrt(R/100) rescaling, no averaging below the floor with added transits;
  * Fisher rank detection is invariant under per-parameter unit rescaling;
  * the nuisance projection is invariant under nuisance-row rescaling.
"""
import numpy as np
import pytest

from jwst_tool import binning, detect, fisher, noise as noise_mod


def _mode_result(n_pix=200, seed=0):
    rng = np.random.default_rng(seed)
    wl = np.sort(rng.uniform(3.0, 5.0, n_pix))
    flux = 5e3 * (1.2 + np.cos(3.0 * wl))
    return dict(wl=wl.tolist(), flux=flux.tolist(),
                noise_1int=np.sqrt(flux / 20.0).tolist(), t_cycle_s=20.0)


def _bins(mode_result, edges, floor_spec, **kw):
    return noise_mod.depth_error_bins(mode_result, edges, 3600.0, 3600.0, 1,
                                      floor_spec, **kw)


# --- no editorial floor is ever applied implicitly ----------------------------

def test_floor_spec_is_a_required_argument_everywhere():
    """No API may supply a floor the caller did not ask for.

    The 15-40 ppm per-mode values are planning conventions, not calibrations;
    a default anywhere would let them reach a headline result silently.
    """
    import inspect

    from jwst_tool import detect

    for fn in (noise_mod.depth_error_bins, detect.evaluate_mode):
        p = inspect.signature(fn).parameters["floor_spec"]
        assert p.default is inspect.Parameter.empty, (
            f"{fn.__qualname__} grew a default floor_spec ({p.default!r})")


def test_registry_floors_are_named_as_suggestions_not_defaults():
    """The registry key must read as illustrative at every call site."""
    from jwst_tool import instruments as ins

    for key, m in ins.MODES.items():
        assert "floor_ppm_suggested" in m, key
        assert "floor_ppm" not in m, (
            f"{key}: bare `floor_ppm` reads as an applied default")
        assert 0.0 < m["floor_ppm_suggested"] <= 200.0, key


def test_no_module_applies_a_registry_floor_by_itself():
    """`floor_ppm_suggested` may only be READ by the GUI as a prefill.

    Checks the parsed AST, not raw text, so documenting the rule in a
    docstring or comment does not trip it -- only a real string reference.
    """
    import ast
    import pathlib

    src = pathlib.Path(noise_mod.__file__).parent
    for mod in ("noise.py", "detect.py", "fisher.py", "binning.py"):
        tree = ast.parse((src / mod).read_text(), str(src / mod))
        refs = [n for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and n.value == "floor_ppm_suggested"]
        assert not refs, (
            f"{mod}:{refs[0].lineno} reads the registry's suggested floor; only "
            "app.py may, and only to prefill a widget the user then owns")


def test_default_calculation_is_the_pandeia_random_sigma():
    """Acceptance: with no floor chosen, sigma is the Pandeia random sigma."""
    mr = _mode_result()
    edges = np.geomspace(3.0, 5.0, 12)
    nz = _bins(mr, edges, None)
    assert np.array_equal(nz["sigma"], np.sqrt(nz["var_phot"]))
    assert np.all(nz["floor"] == 0.0)
    # and an explicitly selected floor still uses the PandExo hard minimum
    chosen_ppm = 40.0
    nz_floor = _bins(mr, edges, chosen_ppm)
    assert np.array_equal(
        nz_floor["sigma"],
        np.maximum(np.sqrt(nz_floor["var_phot"]), chosen_ppm * 1e-6))


# --- floor semantics (PandExo convention) -------------------------------------

def test_no_floor_returns_random_errors_exactly():
    mr = _mode_result()
    edges = np.geomspace(3.0, 5.0, 12)
    nz = _bins(mr, edges, None)
    assert np.array_equal(nz["sigma"], np.sqrt(nz["var_phot"]))
    assert np.all(nz["floor"] == 0.0)
    nz0 = _bins(mr, edges, 0.0)          # scalar zero == no floor
    assert np.array_equal(nz0["sigma"], nz["sigma"])


def test_constant_floor_is_hard_max_not_quadrature():
    mr = _mode_result()
    edges = np.geomspace(3.0, 5.0, 12)
    free = _bins(mr, edges, None)
    ppm = float(np.median(free["sigma"]) * 1e6)   # floor at the median sigma
    nz = _bins(mr, edges, ppm)
    assert np.array_equal(nz["sigma"],
                          np.maximum(np.sqrt(nz["var_phot"]), ppm * 1e-6))
    # bins already noisier than the floor are untouched (max, not quadrature:
    # quadrature would inflate them by up to sqrt(2))
    above = free["sigma"] > ppm * 1e-6
    assert above.any() and (~above).any()
    assert np.array_equal(nz["sigma"][above], free["sigma"][above])
    assert np.all(nz["sigma"][~above] == ppm * 1e-6)


def test_floor_not_rescaled_by_binning_r():
    """The entered constant floor must arrive unchanged at EVERY binning R --
    the retired convention scaled it by sqrt(R/100) for finer bins."""
    mr = _mode_result()
    for R in (50, 100, 200, 400):
        edges = noise_mod.make_bins(3.0, 5.0, R)
        nz = _bins(mr, edges, 20.0)
        assert np.all(nz["floor"] == 20.0 * 1e-6)


def test_wavelength_table_interpolation_and_edge_extension():
    table = np.array([[3.5, 10.0], [4.0, 30.0], [4.5, 20.0]])
    wl = np.array([3.0, 3.5, 3.75, 4.25, 4.5, 5.0])
    floor = noise_mod.resolve_floor(wl, table)
    assert floor == pytest.approx(
        np.array([10.0, 10.0, 20.0, 25.0, 20.0, 20.0]) * 1e-6)


def test_wavelength_table_unsorted_rows_are_sorted():
    table = np.array([[4.5, 20.0], [3.5, 10.0], [4.0, 30.0]])
    wl = np.linspace(3.0, 5.0, 7)
    ref = noise_mod.resolve_floor(
        wl, np.array([[3.5, 10.0], [4.0, 30.0], [4.5, 20.0]]))
    assert np.array_equal(noise_mod.resolve_floor(wl, table), ref)


@pytest.mark.parametrize("bad", [
    -5.0,                                           # negative scalar
    float("nan"),                                   # non-finite scalar
    np.array([[3.5, 10.0], [4.0, -1.0]]),           # negative floor value
    np.array([[3.5, 10.0], [4.0, np.inf]]),         # non-finite table
    np.array([[3.5, 10.0], [3.5, 20.0]]),           # duplicate wavelength
    np.array([[3.5, 10.0]]),                        # single row
    np.array([3.5, 10.0, 4.0]),                     # wrong shape
])
def test_invalid_floor_specs_raise(bad):
    with pytest.raises(ValueError):
        noise_mod.resolve_floor(np.array([3.0, 4.0]), bad)


def test_transits_approach_floor_from_above():
    """sigma(N) decreases monotonically toward the floor, never below it."""
    mr = _mode_result()
    edges = np.geomspace(3.0, 5.0, 10)
    nz = _bins(mr, edges, 30.0)
    result = dict(var_phot=nz["var_phot"], floor=nz["floor"], n_transits_eval=1)
    prev = detect.sigma_at_transits(result, 1)
    for n in (2, 5, 20, 100, 10000):
        cur = detect.sigma_at_transits(result, n)
        assert np.all(cur <= prev + 1e-30)
        assert np.all(cur >= nz["floor"])
        prev = cur
    assert np.allclose(detect.sigma_at_transits(result, 10 ** 9), nz["floor"],
                       rtol=1e-3, atol=0)


def test_detect_and_noise_share_the_final_sigma():
    """The detection score consumes the same clamped sigma noise quotes."""
    mr = _mode_result()
    wl = np.asarray(mr["wl"])
    edges = np.geomspace(3.0, 5.0, 10)
    op = binning.build_operator(wl, np.asarray(mr["flux"]), edges)
    nz = _bins(mr, edges, 25.0, op=op)
    sig = detect.detection_significance(np.full(nz["sigma"].size, 1e-4),
                                        nz["sigma"], marginalize_offset=False)
    assert sig == pytest.approx(
        np.sqrt(np.sum((1e-4 / nz["sigma"]) ** 2)), rel=1e-12)


# --- Fisher unit-rescaling invariance ------------------------------------------

def test_fisher_rank_and_sigmas_invariant_under_unit_rescaling():
    """Rescaling parameters over 1e-12..1e12 must not change sigmas, rank, or
    the constrained subspace (the raw-eigenvalue threshold did)."""
    rng = np.random.default_rng(0)
    J0 = rng.standard_normal((5, 60)) * np.array(
        [1e-4, 1.0, 1e3, 1e-7, 5e2])[:, None]
    J0[4] = J0[3] * 3.0                       # one exact degeneracy pair
    s = np.full(60, 1e-4)
    F0 = (J0 / s[None, :] ** 2) @ J0.T
    d0 = {}
    base = fisher._marg_sigmas(F0, 5, diag=d0)
    assert np.isinf(base[3]) and np.isinf(base[4])       # the degenerate pair
    assert np.all(np.isfinite(base[:3]))
    for trial in range(25):
        f = 10.0 ** rng.uniform(-12, 12, 5)
        Js = J0 * f[:, None]
        Fs = (Js / s[None, :] ** 2) @ Js.T
        ds = {}
        sig = fisher._marg_sigmas(Fs, 5, diag=ds) * f    # back to raw units
        assert ds["fisher_rank"] == d0["fisher_rank"]
        assert np.array_equal(np.isinf(sig), np.isinf(base))
        m = np.isfinite(base)
        assert np.allclose(sig[m], base[m], rtol=1e-7, atol=0)


def test_fisher_multi_dim_null_space_invariant_under_rescaling():
    """A 2-D null space stays invariant under rescaling, which requires the
    basis-invariant subspace projection in the null-overlap test."""
    rng = np.random.default_rng(4)
    J0 = rng.standard_normal((6, 80)) * np.array(
        [1.0, 1e3, 1e-5, 1.0, 1e2, 1e-3])[:, None]
    J0[1] = 3.0 * J0[0]          # degeneracy A: params 0,1 unconstrained
    J0[3] = -2.0 * J0[2]         # degeneracy B: params 2,3 unconstrained
    s = np.full(80, 2e-4)
    F0 = (J0 / s[None, :] ** 2) @ J0.T
    d0 = {}
    base = fisher._marg_sigmas(F0, 6, diag=d0)
    assert d0["fisher_rank"] == 4                        # 6 params, 2 null dirs
    assert np.all(np.isinf(base[:4])) and np.all(np.isfinite(base[4:]))
    for _ in range(15):
        f = 10.0 ** rng.uniform(-9, 9, 6)
        Js = J0 * f[:, None]
        Fs = (Js / s[None, :] ** 2) @ Js.T
        ds = {}
        sig = fisher._marg_sigmas(Fs, 6, diag=ds) * f
        assert ds["fisher_rank"] == d0["fisher_rank"]
        assert np.array_equal(np.isinf(sig), np.isinf(base))
        m = np.isfinite(base)
        assert np.allclose(sig[m], base[m], rtol=1e-6, atol=0)


def test_fisher_zero_response_parameter_reads_inf():
    F = np.diag([4.0, 0.0])
    sig = fisher._marg_sigmas(F, 2)
    assert sig[0] == pytest.approx(0.5)
    assert np.isinf(sig[1])


# --- nuisance-row rescaling invariance -----------------------------------------

def test_detection_score_invariant_under_nuisance_row_rescaling():
    """The profiled score depends only on the SPAN of the nuisance rows; the
    raw-eigenvalue threshold silently dropped a valid down-scaled row."""
    rng = np.random.default_rng(7)
    n = 60
    wl = np.geomspace(3.0, 5.0, n)
    sigma = np.full(n, 1e-4)
    signal = 5e-4 * np.exp(-0.5 * ((wl - 4.0) / 0.1) ** 2)
    seg = (wl >= 4.0).astype(int)
    rows = detect._segment_rows(seg) + detect._slope_rows(seg, wl)
    base = detect.detection_significance(signal, sigma, nuisance=rows)
    C = noise_mod.build_cov(wl, sigma ** 2 * 0.25, sigma, "conservative")
    base_cov = detect.detection_significance(signal, sigma, nuisance=rows,
                                             cov=C)
    for trial in range(25):
        f = 10.0 ** rng.uniform(-12, 12, len(rows))
        scaled = [r * fi for r, fi in zip(rows, f)]
        got = detect.detection_significance(signal, sigma, nuisance=scaled)
        assert got == pytest.approx(base, rel=1e-9)
        got_cov = detect.detection_significance(signal, sigma,
                                                nuisance=scaled, cov=C)
        assert got_cov == pytest.approx(base_cov, rel=1e-9)


def test_detection_score_invariant_under_nuisance_basis_rotation():
    """ANY nonsingular remix of the nuisance rows, not just per-row rescaling,
    must leave the profiled score unchanged."""
    rng = np.random.default_rng(9)
    n = 60
    wl = np.geomspace(3.0, 5.0, n)
    sigma = np.full(n, 1e-4)
    signal = 5e-4 * np.exp(-0.5 * ((wl - 4.0) / 0.1) ** 2)
    seg = (wl >= 4.0).astype(int)
    rows = detect._segment_rows(seg) + detect._slope_rows(seg, wl)
    R = np.stack(rows)
    base = detect.detection_significance(signal, sigma, nuisance=rows)
    C = noise_mod.build_cov(wl, sigma ** 2 * 0.25, sigma, "conservative")
    base_cov = detect.detection_significance(signal, sigma, nuisance=rows, cov=C)
    trials = 0
    while trials < 20:
        M = rng.standard_normal((len(rows), len(rows)))
        if abs(np.linalg.det(M)) < 1e-3:            # keep M well-conditioned
            continue
        trials += 1
        mixed = list(M @ R)                          # arbitrary basis of the span
        assert detect.detection_significance(signal, sigma, nuisance=mixed) \
            == pytest.approx(base, rel=1e-7)
        assert detect.detection_significance(signal, sigma, nuisance=mixed,
                                             cov=C) == pytest.approx(base_cov,
                                                                     rel=1e-7)


def test_zero_nuisance_row_is_ignored():
    rng = np.random.default_rng(3)
    signal = rng.normal(0, 1e-4, 30)
    sigma = np.full(30, 1e-4)
    a = detect.detection_significance(signal, sigma)
    b = detect.detection_significance(signal, sigma,
                                      nuisance=[np.zeros(30)])
    assert b == pytest.approx(a, rel=1e-12)


# --- fail-fast on invalid public noise/covariance inputs ----------------------
# noise_inflation is squared, so -1 silently acted as +1; build_cov accepted
# negative variances and NaN/negative wavelengths.

def _tiny_mode_result(n=50):
    return dict(wl=np.linspace(3.0, 4.0, n).tolist(),
                flux=np.full(n, 1e6).tolist(),
                noise_1int=np.full(n, 1e3).tolist(),
                t_cycle_s=10.0)


def test_invalid_noise_inflation_raises():
    mr = _tiny_mode_result()
    edges = np.geomspace(3.0, 4.0, 12)
    for bad in (-1.0, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="noise_inflation"):
            noise_mod.depth_error_bins(mr, edges, 3600.0, 3600.0, 1,
                                       floor_spec=None, noise_inflation=bad)


def test_pixel_variance_rejects_bad_inputs():
    good = _tiny_mode_result()
    with pytest.raises(ValueError, match="t_cycle"):
        noise_mod.pixel_depth_variance(dict(good, t_cycle_s=0.0),
                                       3600.0, 3600.0, 1)
    with pytest.raises(ValueError, match="windows"):
        noise_mod.pixel_depth_variance(good, float("nan"), 3600.0, 1)
    bad_flux = dict(good)
    bad_flux["flux"] = np.where(np.arange(50) == 3, np.nan,
                                np.full(50, 1e6)).tolist()
    with pytest.raises(ValueError, match="flux"):
        noise_mod.pixel_depth_variance(bad_flux, 3600.0, 3600.0, 1)
    bad_noise = dict(good)
    bad_noise["noise_1int"] = np.where(np.arange(50) == 3, -1.0,
                                       np.full(50, 1e3)).tolist()
    with pytest.raises(ValueError, match="noise_1int"):
        noise_mod.pixel_depth_variance(bad_noise, 3600.0, 3600.0, 1)


def test_build_cov_rejects_bad_inputs():
    wl = np.linspace(3.0, 4.0, 10)
    var = np.full(10, 1e-9)
    fl = np.full(10, 5e-5)
    for kw, msg in ((dict(wl_center=np.where(np.arange(10) == 2, -1.0, wl)),
                     "wavelengths"),
                    (dict(var_phot=np.where(np.arange(10) == 2, -1e-9, var)),
                     "var_phot"),
                    (dict(var_phot=np.where(np.arange(10) == 2, np.nan, var)),
                     "var_phot"),
                    (dict(floor=np.where(np.arange(10) == 2, np.nan, fl)),
                     "floor")):
        args = dict(wl_center=wl, var_phot=var, floor=fl)
        args.update(kw)
        with pytest.raises(ValueError, match=msg):
            noise_mod.build_cov(args["wl_center"], args["var_phot"],
                                args["floor"], "moderate")
    with pytest.raises(ValueError, match="1-D"):
        noise_mod.build_cov(wl, var[:5], fl, "moderate")


def test_make_bins_rejects_bad_inputs():
    for args in ((-1.0, 4.0, 100.0), (4.0, 3.0, 100.0),
                 (3.0, float("nan"), 100.0), (3.0, 4.0, 0.0),
                 (3.0, 4.0, float("inf"))):
        with pytest.raises(ValueError):
            noise_mod.make_bins(*args)
