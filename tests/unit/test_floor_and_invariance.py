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

def test_no_editorial_floor_is_ever_applied_implicitly():
    """No API may supply a floor the caller did not ask for.

    The 15-40 ppm per-mode values are planning conventions, not calibrations;
    a default anywhere would let them reach a headline result silently. Three
    pins: floor_spec is a REQUIRED argument at every entry point; the
    registry key reads as a suggestion at every call site; and no
    computational module references the suggested floor (checked on the
    parsed AST, not raw text, so documenting the rule in a docstring or
    comment does not trip it -- only a real string reference)."""
    import ast
    import inspect
    import pathlib

    from jwst_tool import instruments as ins

    for fn in (noise_mod.depth_error_bins, detect.evaluate_mode):
        p = inspect.signature(fn).parameters["floor_spec"]
        assert p.default is inspect.Parameter.empty, (
            f"{fn.__qualname__} grew a default floor_spec ({p.default!r})")
    for key, m in ins.MODES.items():
        assert "floor_ppm_suggested" in m, key
        assert "floor_ppm" not in m, (
            f"{key}: bare `floor_ppm` reads as an applied default")
        assert 0.0 < m["floor_ppm_suggested"] <= 200.0, key
    src = pathlib.Path(noise_mod.__file__).parent
    for mod in ("noise.py", "detect.py", "fisher.py", "binning.py"):
        tree = ast.parse((src / mod).read_text(), str(src / mod))
        refs = [n for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and n.value == "floor_ppm_suggested"]
        assert not refs, (
            f"{mod}:{refs[0].lineno} reads the registry's suggested floor; only "
            "app.py may, and only to prefill a widget the user then owns")


def test_pixel_variance_matches_independent_unequal_window_oracle():
    """Literal hand calculation: whole integration counts, unequal baselines,
    and multiple transits all enter exactly once."""
    mode = {
        "flux": [1000.0, 4000.0],
        "noise_1int": [20.0, 40.0],
        "t_cycle_s": 90.0,
    }
    actual = noise_mod.pixel_depth_variance(
        mode, t_in_s=370.0, t_out_s=730.0, n_transits=3)
    n_in, n_out = 4, 8
    expected = np.array([
        (20.0 / 1000.0) ** 2 * (1.0 / n_in + 1.0 / n_out) / 3.0,
        (40.0 / 4000.0) ** 2 * (1.0 / n_in + 1.0 / n_out) / 3.0,
    ])
    assert np.array_equal(actual, expected)


# --- floor semantics (PandExo convention) -------------------------------------

def test_floor_is_hard_max_on_final_bins():
    """Exact PandExo semantics on the FINAL bins. With no floor chosen, sigma
    is the Pandeia random sigma exactly (scalar zero == no floor). A chosen
    floor is a hard max, not quadrature (quadrature would inflate
    above-floor bins by up to sqrt(2)). The entered constant floor arrives
    unchanged at EVERY binning R -- never rescaled by sqrt(R/100) for finer
    bins."""
    mr = _mode_result()
    edges = np.geomspace(3.0, 5.0, 12)
    nz = _bins(mr, edges, None)
    assert np.array_equal(nz["sigma"], np.sqrt(nz["var_phot"]))
    assert np.all(nz["floor"] == 0.0)
    nz0 = _bins(mr, edges, 0.0)          # scalar zero == no floor
    assert np.array_equal(nz0["sigma"], nz["sigma"])
    # hard max at a floor that splits the bins
    ppm = float(np.median(nz["sigma"]) * 1e6)   # floor at the median sigma
    nzf = _bins(mr, edges, ppm)
    assert np.array_equal(nzf["sigma"],
                          np.maximum(np.sqrt(nzf["var_phot"]), ppm * 1e-6))
    above = nz["sigma"] > ppm * 1e-6
    assert above.any() and (~above).any()
    assert np.array_equal(nzf["sigma"][above], nz["sigma"][above])
    assert np.all(nzf["sigma"][~above] == ppm * 1e-6)
    # no sqrt(R/100) rescaling at any binning R
    for R in (50, 100, 400):
        nzr = _bins(mr, noise_mod.make_bins(3.0, 5.0, R), 20.0)
        assert np.all(nzr["floor"] == 20.0 * 1e-6)


def test_wavelength_table_floor_resolution_and_validation():
    """Linear interpolation with constant edge extension; unsorted rows are
    sorted rather than corrupting the interpolation; invalid specs raise."""
    table = np.array([[3.5, 10.0], [4.0, 30.0], [4.5, 20.0]])
    wl = np.array([3.0, 3.5, 3.75, 4.25, 4.5, 5.0])
    floor = noise_mod.resolve_floor(wl, table)
    assert floor == pytest.approx(
        np.array([10.0, 10.0, 20.0, 25.0, 20.0, 20.0]) * 1e-6)
    wl2 = np.linspace(3.0, 5.0, 7)
    shuffled = np.array([[4.5, 20.0], [3.5, 10.0], [4.0, 30.0]])
    assert np.array_equal(noise_mod.resolve_floor(wl2, shuffled),
                          noise_mod.resolve_floor(wl2, table))
    for bad in (-5.0,                                   # negative scalar
                float("nan"),                           # non-finite scalar
                np.array([[3.5, 10.0], [4.0, -1.0]]),   # negative floor value
                np.array([[3.5, 10.0], [4.0, np.inf]]), # non-finite table
                np.array([[3.5, 10.0], [3.5, 20.0]]),   # duplicate wavelength
                np.array([[3.5, 10.0]]),                # single row
                np.array([3.5, 10.0, 4.0])):            # wrong shape
        with pytest.raises(ValueError):
            noise_mod.resolve_floor(np.array([3.0, 4.0]), bad)


def test_floor_is_consumed_downstream_unchanged():
    """sigma(N) decreases monotonically toward the floor and NEVER below it
    (no averaging below the floor with added transits), and the detection
    score consumes the same clamped sigma noise quotes."""
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
    # the detection score reads the clamped sigma, not a recomputed one
    wl = np.asarray(mr["wl"])
    op = binning.build_operator(wl, np.asarray(mr["flux"]), edges)
    nz2 = _bins(mr, edges, 25.0, op=op)
    sig = detect.detection_significance(np.full(nz2["sigma"].size, 1e-4),
                                        nz2["sigma"], marginalize_offset=False)
    assert sig == pytest.approx(
        np.sqrt(np.sum((1e-4 / nz2["sigma"]) ** 2)), rel=1e-12)


# --- Fisher unit-rescaling invariance ------------------------------------------

def test_fisher_rank_and_sigmas_invariant_under_unit_rescaling():
    """Rescaling parameters over 1e-12..1e12 must not change sigmas, rank, or
    the constrained subspace (a raw-eigenvalue threshold does). A 2-D null
    space stays invariant too, which requires the basis-invariant subspace
    projection in the null-overlap test. A parameter with zero response
    reads inf, finite ones invert exactly."""
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
    # multi-dimensional (2-D) null space, same invariance
    rng4 = np.random.default_rng(4)
    K0 = rng4.standard_normal((6, 80)) * np.array(
        [1.0, 1e3, 1e-5, 1.0, 1e2, 1e-3])[:, None]
    K0[1] = 3.0 * K0[0]          # degeneracy A: params 0,1 unconstrained
    K0[3] = -2.0 * K0[2]         # degeneracy B: params 2,3 unconstrained
    s6 = np.full(80, 2e-4)
    G0 = (K0 / s6[None, :] ** 2) @ K0.T
    e0 = {}
    base6 = fisher._marg_sigmas(G0, 6, diag=e0)
    assert e0["fisher_rank"] == 4                        # 6 params, 2 null dirs
    assert np.all(np.isinf(base6[:4])) and np.all(np.isfinite(base6[4:]))
    for _ in range(15):
        f = 10.0 ** rng4.uniform(-9, 9, 6)
        Ks = K0 * f[:, None]
        Gs = (Ks / s6[None, :] ** 2) @ Ks.T
        es = {}
        sig6 = fisher._marg_sigmas(Gs, 6, diag=es) * f
        assert es["fisher_rank"] == e0["fisher_rank"]
        assert np.array_equal(np.isinf(sig6), np.isinf(base6))
        m = np.isfinite(base6)
        assert np.allclose(sig6[m], base6[m], rtol=1e-6, atol=0)
    # zero-response parameter
    sig0 = fisher._marg_sigmas(np.diag([4.0, 0.0]), 2)
    assert sig0[0] == pytest.approx(0.5)
    assert np.isinf(sig0[1])


# --- nuisance-row rescaling invariance -----------------------------------------

def _slope_row(wl: np.ndarray) -> np.ndarray:
    """A centered, unit-RMS linear-in-ln(lambda) row: a second nuisance
    direction beyond the segment offsets, so the remix tests exercise a
    multi-row span."""
    r = np.log(np.asarray(wl, float))
    r = r - r.mean()
    return r / np.sqrt(np.mean(r ** 2))


def test_detection_score_invariant_under_nuisance_remix():
    """The profiled score depends only on the SPAN of the nuisance rows: it
    must be invariant under 24-decade per-row rescaling (a raw-eigenvalue
    threshold silently drops a valid down-scaled row) AND under an
    arbitrary nonsingular remix of the rows, not just per-row rescaling.
    An all-zero row is ignored, never an error or a changed score."""
    rng = np.random.default_rng(7)
    n = 60
    wl = np.geomspace(3.0, 5.0, n)
    sigma = np.full(n, 1e-4)
    signal = 5e-4 * np.exp(-0.5 * ((wl - 4.0) / 0.1) ** 2)
    seg = (wl >= 4.0).astype(int)
    rows = detect._segment_rows(seg) + [_slope_row(wl)]
    base = detect.detection_significance(signal, sigma, nuisance=rows)
    for trial in range(25):
        f = 10.0 ** rng.uniform(-12, 12, len(rows))
        scaled = [r * fi for r, fi in zip(rows, f)]
        got = detect.detection_significance(signal, sigma, nuisance=scaled)
        assert got == pytest.approx(base, rel=1e-9)
    R = np.stack(rows)
    trials = 0
    while trials < 20:
        M = rng.standard_normal((len(rows), len(rows)))
        if abs(np.linalg.det(M)) < 1e-3:            # keep M well-conditioned
            continue
        trials += 1
        mixed = list(M @ R)                          # arbitrary basis of the span
        assert detect.detection_significance(signal, sigma, nuisance=mixed) \
            == pytest.approx(base, rel=1e-7)
    # an all-zero nuisance row changes nothing
    sig_a = detect.detection_significance(signal, sigma)
    sig_b = detect.detection_significance(signal, sigma,
                                          nuisance=[np.zeros(n)])
    assert sig_b == pytest.approx(sig_a, rel=1e-12)


# --- fail-fast on invalid public noise inputs ---------------------------------

def _tiny_mode_result(n=50):
    return dict(wl=np.linspace(3.0, 4.0, n).tolist(),
                flux=np.full(n, 1e6).tolist(),
                noise_1int=np.full(n, 1e3).tolist(),
                t_cycle_s=10.0)


def test_public_noise_inputs_fail_fast():
    """noise_inflation must be finite and positive (it is squared, so -1
    silently acted as +1); pixel_depth_variance and make_bins validate
    shapes, signs and finiteness loudly."""
    mr = _tiny_mode_result()
    edges = np.geomspace(3.0, 4.0, 12)
    for bad in (-1.0, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            noise_mod.depth_error_bins(mr, edges, 3600.0, 3600.0, 1,
                                       floor_spec=None, noise_inflation=bad)
    bad_flux = dict(mr)
    bad_flux["flux"] = np.where(np.arange(50) == 3, np.nan,
                                np.full(50, 1e6)).tolist()
    bad_noise = dict(mr)
    bad_noise["noise_1int"] = np.where(np.arange(50) == 3, -1.0,
                                       np.full(50, 1e3)).tolist()
    for res, t_in in ((dict(mr, t_cycle_s=0.0), 3600.0),   # t_cycle
                      (mr, float("nan")),                  # transit window
                      (bad_flux, 3600.0),                  # flux
                      (bad_noise, 3600.0)):                # noise_1int
        with pytest.raises(ValueError):
            noise_mod.pixel_depth_variance(res, t_in, 3600.0, 1)
    for args in ((-1.0, 4.0, 100.0), (4.0, 3.0, 100.0),
                 (3.0, float("nan"), 100.0), (3.0, 4.0, 0.0),
                 (3.0, 4.0, float("inf"))):
        with pytest.raises(ValueError):
            noise_mod.make_bins(*args)
