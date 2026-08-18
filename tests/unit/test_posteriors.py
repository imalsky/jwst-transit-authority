"""Fisher-Gaussian posterior curves + named mode combinations.

Pins: the honesty labeling on every record, the no-fake-finite-curve rule for
null Fisher directions, fail-fast input validation, the display-unit
conventions (dlnCO's absolute-C/O transform), the single-mode-combo ==
mode_forecast identity, and the saturated-mode exclusion policy (same as the
"All usable modes" row, disclosed per combo, loud on zero usable).
"""
import numpy as np
import pytest

from jwst_tool import fisher, instruments, posteriors


def _result(seed=0, n_free=2, n_bins=40, saturated=False):
    """Synthetic evaluate_mode-like result: jac rows [free..., lnR0]."""
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((n_free + 1, n_bins))
    return dict(jac_bins=J, sigma=np.full(n_bins, 1e-4),
                seg=np.zeros(n_bins, int), saturated=saturated)


def _registry_keys(n):
    return list(instruments.MODES)[:n]


def _mode_result(seed, key, n_bins=30):
    r = _result(seed=seed, n_bins=n_bins)
    r["mode_key"] = key
    r["depth"] = np.full(n_bins, 0.021)
    return r


FREE = ["lnZ", "dlnCO"]
CENTERS = {"lnZ": 1.0, "dlnCO": 0.55}   # display units: dex, absolute C/O
CO = 0.55


# --- marginalized_posteriors -------------------------------------------------

def test_curves_match_fisher_and_are_honestly_labeled():
    """The statistical core in one place: honesty labels, sigma equal to the
    Fisher forecast, the display-unit transforms (dlnCO reported as absolute
    C/O with the zero-clipped Gaussian curve family), a caller grid used verbatim,
    and a LIST of results combining via combined_forecast."""
    r = _result()
    out = posteriors.marginalized_posteriors(r, FREE, CENTERS, co_eval=CO)
    assert out["kind"] == posteriors.FORECAST_KIND
    for phrase in ("Fisher", "Cramer-Rao", "Not a sampled posterior"):
        assert phrase in out["label"], phrase

    sig = fisher.mode_forecast(r, FREE)
    for name in FREE:
        rec = out["params"][name]
        want = fisher.display_sigma(name, sig[name], co_eval=CO)
        assert rec["constrained"]
        assert rec["sigma_display"] == pytest.approx(want, rel=1e-12)
        assert rec["center"] == CENTERS[name]
        # Ordinary display coordinates stay Gaussian (pdf integrates to
        # ~1 on the +/-5-sigma grid); absolute C/O is the same Gaussian
        # clipped at the physical boundary C/O = 0.
        theta, pdf = rec["theta"], rec["pdf"]
        # np.trapezoid is NumPy 2.0+; np.trapz was removed in NumPy 2.4
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        if name == "dlnCO":
            assert rec["curve_family"] == "gaussian_truncated_positive"
            assert np.all(theta >= 0.0)
            assert theta.min() == pytest.approx(max(CO - 5 * want, 0.0))
            assert theta.max() == pytest.approx(CO + 5 * want)
            assert theta[np.argmax(pdf)] == pytest.approx(CO, rel=1e-2)
            # clipping can only REMOVE mass, never add it
            assert _trapz(pdf, theta) <= 1.0 + 1e-4
        else:
            assert _trapz(pdf, theta) == pytest.approx(1.0, abs=1e-4)
            assert rec["curve_family"] == "gaussian"
            assert theta[np.argmax(pdf)] == pytest.approx(
                CENTERS[name], abs=1e-12)
            assert theta.min() == pytest.approx(CENTERS[name] - 5 * want)
            assert theta.max() == pytest.approx(CENTERS[name] + 5 * want)

    # sigma_display(dlnCO) = C/O * sigma_internal (the fisher-table rule)
    assert out["params"]["dlnCO"]["sigma_display"] == pytest.approx(
        CO * sig["dlnCO"], rel=1e-12)
    assert out["params"]["dlnCO"]["unit"] == "C/O ratio"

    # a caller-supplied grid is used verbatim
    grid = np.linspace(-2.0, 4.0, 33)
    out2 = posteriors.marginalized_posteriors(
        r, FREE, CENTERS, co_eval=CO, grids={"lnZ": grid})
    assert np.array_equal(out2["params"]["lnZ"]["theta"], grid)

    # a list of results combines via combined_forecast
    rs = [_result(seed=1), _result(seed=2)]
    outc = posteriors.marginalized_posteriors(rs, FREE, CENTERS, co_eval=CO)
    sigc = fisher.combined_forecast(rs, FREE)
    assert outc["n_modes"] == 2
    for name in FREE:
        want = fisher.display_sigma(name, sigc[name], co_eval=CO)
        assert outc["params"][name]["sigma_display"] == pytest.approx(
            want, rel=1e-12)

    # a very broad C/O curve is clipped at the physical boundary: the grid
    # starts exactly at 0 and the peak stays at the center
    curve = posteriors.truncated_gaussian_curve(0.55, 1.0)
    assert curve["theta"].min() == 0.0
    assert np.all(curve["theta"] >= 0.0)
    assert curve["theta"][np.argmax(curve["pdf"])] == pytest.approx(
        0.55, abs=0.06)


def test_null_directions_are_flagged_never_faked():
    """Duplicate Jacobian rows: both unconstrained directions must come back
    with NO curve (never a finite Gaussian) from marginalized_posteriors AND
    with recovered=False / None shifts (never a number) from mock_recovery."""
    (k1,) = _registry_keys(1)
    rng = np.random.default_rng(3)
    row = rng.standard_normal(50)
    J = np.stack([row, row, rng.standard_normal(50)])   # [p0, p1, lnR0]
    r = dict(jac_bins=J, sigma=np.full(50, 1e-4), mode_key=k1,
             depth=np.full(50, 0.02))
    out = posteriors.marginalized_posteriors(
        r, ["p0", "p1"], {"p0": 0.0, "p1": 0.0})
    real = posteriors.mock_realization([r], seed=0)
    rec = posteriors.mock_recovery([r], ["p0", "p1"], real)
    for name in ("p0", "p1"):
        p = out["params"][name]
        assert not p["constrained"]
        assert p["theta"] is None and p["pdf"] is None
        assert np.isinf(p["sigma_display"])
        assert rec["recovered"][name] is False
        assert rec["delta"][name] is None
        assert rec["delta_display"][name] is None


def test_input_validation_raises():
    """Fail-fast: non-finite sigma/Jacobian/centers, missing centers or
    co_eval, a free_names list that does not match the [free..., lnR0] row
    layout, an unconstrained gaussian_curve, and a non-positive C/O grid all
    raise."""
    r = _result()
    bad_sigma = dict(r, sigma=np.where(np.arange(40) == 5, np.nan, r["sigma"]))
    with pytest.raises(ValueError, match="finite"):
        posteriors.marginalized_posteriors(bad_sigma, FREE, CENTERS,
                                           co_eval=CO)
    bad_jac = dict(r, jac_bins=r["jac_bins"].copy())
    bad_jac["jac_bins"][0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        posteriors.marginalized_posteriors(bad_jac, FREE, CENTERS, co_eval=CO)
    with pytest.raises(ValueError, match="centers"):
        posteriors.marginalized_posteriors(r, FREE, {"lnZ": 1.0}, co_eval=CO)
    with pytest.raises(ValueError, match="finite"):
        posteriors.marginalized_posteriors(
            r, FREE, {"lnZ": np.nan, "dlnCO": 0.5}, co_eval=CO)
    # dlnCO without co_eval raises, exactly like display_sigma
    with pytest.raises(ValueError, match="co_eval"):
        posteriors.marginalized_posteriors(r, FREE, CENTERS)
    with pytest.raises(ValueError, match="rows"):
        posteriors.marginalized_posteriors(r, ["lnZ"], {"lnZ": 1.0})
    with pytest.raises(ValueError, match="unconstrained"):
        posteriors.gaussian_curve(0.0, np.inf)
    with pytest.raises(ValueError, match="center"):
        posteriors.truncated_gaussian_curve(-0.1, 1.0)


# --- named combinations ------------------------------------------------------

def test_combo_forecast_identity_saturation_policy_and_additivity():
    """combo_forecast's contract in one place.

    A single-mode combo equals mode_forecast (the pinned upstream identity
    must survive the wrapper). Saturated modes are excluded AND disclosed,
    with the forecast equal to the usable subset alone (the exclusion is
    real), and zero usable modes raise rather than reporting nothing.
    Fisher information is additive over independent modes, so a superset
    combination can only tighten a marginalized sigma -- pinned on
    ``combo_forecast`` directly since ``compare_combos`` (a thin wrapper
    with no production consumer) was removed on 2026-08-14; this invariant
    is pinned nowhere else. The with-centers path attaches constrained
    posterior curves."""
    k1, k2 = _registry_keys(2)

    # single-mode identity
    r = _result(seed=4)
    rec = posteriors.combo_forecast("just one", [k1], {k1: r}, FREE,
                                    co_eval=CO)
    sig = fisher.mode_forecast(r, FREE)
    for name in FREE:
        want = fisher.display_sigma(name, sig[name], co_eval=CO)
        assert rec["sigma_marginalized_display"][name] == pytest.approx(
            want, rel=1e-12)
    assert rec["kind"] == posteriors.FORECAST_KIND
    assert rec["posteriors"] is None and rec["posteriors_note"]

    # saturation policy: exclude, disclose, forecast = usable subset alone
    results = {k1: _result(seed=5), k2: _result(seed=6, saturated=True)}
    rec = posteriors.combo_forecast("mixed", [k1, k2], results, FREE,
                                    co_eval=CO)
    assert rec["usable_modes"] == [k1]
    assert [e["mode_key"] for e in rec["excluded"]] == [k2]
    assert "saturated" in rec["excluded"][0]["reason"]
    sig = fisher.combined_forecast([results[k1]], FREE)
    for name in FREE:
        want = fisher.display_sigma(name, sig[name], co_eval=CO)
        assert rec["sigma_marginalized_display"][name] == pytest.approx(
            want, rel=1e-12)
    with pytest.raises(ValueError, match="no usable mode"):
        posteriors.combo_forecast("dead", [k1],
                                  {k1: _result(saturated=True)}, FREE,
                                  co_eval=CO)

    # additivity + the with-centers path
    results = {k1: _result(seed=7), k2: _result(seed=8)}
    one = posteriors.combo_forecast("A", [k1], results, FREE, co_eval=CO)
    both = posteriors.combo_forecast("A + B", [k1, k2], results, FREE,
                                     centers=CENTERS, co_eval=CO)
    assert one["name"] == "A" and both["name"] == "A + B"
    assert (both["sigma_marginalized_display"]["lnZ"]
            <= one["sigma_marginalized_display"]["lnZ"] * (1 + 1e-12))
    post = both["posteriors"]
    assert post is not None and post["n_modes"] == 2
    assert post["params"]["lnZ"]["constrained"]
    single = fisher.mode_forecast(results[k1], FREE)["lnZ"]
    assert post["sigma_marginalized"]["lnZ"] <= single * (1 + 1e-12)


def test_combo_input_validation_raises():
    k1, k2 = _registry_keys(2)
    with pytest.raises(ValueError, match="unknown mode key"):
        posteriors.combo_forecast("bad", ["not_a_mode"],
                                  {"not_a_mode": _result()}, FREE, co_eval=CO)
    with pytest.raises(ValueError, match="duplicate"):
        posteriors.combo_forecast("dup", [k1, k1], {k1: _result()}, FREE,
                                  co_eval=CO)
    with pytest.raises(ValueError, match="no result"):
        posteriors.combo_forecast("missing", [k1, k2], {k1: _result()}, FREE,
                                  co_eval=CO)


# --- mock-observation layer --------------------------------------------------

def test_mock_realization_contract():
    """The full mock_realization contract: same seed identical, new seed
    different, depth_mock exactly depth + noise; a mode's stream depends
    only on (seed, mode_key), so adding or reordering other modes never
    changes it; the record discloses the seed-scheme version and numpy
    version (archival reproducibility: default_rng bitstreams are not
    contractually version-stable); and there are NO side effects -- global
    numpy RNG state untouched, the caller's result dicts (the noiseless
    depths and sigmas that feed scores and CSVs) byte-identical."""
    k1, k2 = _registry_keys(2)
    rs = [_mode_result(1, k1), _mode_result(2, k2)]
    depth_before = np.asarray(rs[0]["depth"]).copy()
    sigma_before = np.asarray(rs[0]["sigma"]).copy()
    np.random.seed(123)
    state_before = np.random.get_state()[1].copy()

    a = posteriors.mock_realization(rs, seed=7)
    b = posteriors.mock_realization(rs, seed=7)
    c = posteriors.mock_realization(rs, seed=8)
    assert a["seed"] == 7 and a["kind"] == posteriors.MOCK_KIND
    for i, k in enumerate((k1, k2)):
        assert np.array_equal(a["modes"][k]["noise"], b["modes"][k]["noise"])
        assert not np.array_equal(a["modes"][k]["noise"],
                                  c["modes"][k]["noise"])
        assert np.array_equal(a["modes"][k]["depth_mock"],
                              np.asarray(rs[i]["depth"])
                              + a["modes"][k]["noise"])
    alone = posteriors.mock_realization([rs[0]], seed=7)
    reordered = posteriors.mock_realization([rs[1], rs[0]], seed=7)
    for real in (alone, reordered):
        assert np.array_equal(real["modes"][k1]["noise"],
                              a["modes"][k1]["noise"]), \
            "a mode's draw must depend only on (seed, mode_key)"

    assert a["seed_scheme"] == posteriors.SEED_SCHEME
    assert a["numpy_version"] == np.__version__
    assert posteriors.SEED_SCHEME.startswith("v1:")

    assert np.array_equal(np.random.get_state()[1], state_before)
    assert np.array_equal(np.asarray(rs[0]["depth"]), depth_before)
    assert np.array_equal(np.asarray(rs[0]["sigma"]), sigma_before)
    assert a["modes"][k1]["depth_mock"] is not rs[0]["depth"]


def test_mock_layer_validates_loudly():
    """Refusals: bad seed, empty selection, non-positive sigma, missing
    mode_key, a recovery against a realization that does not cover the
    results, and a crc32 stream collision between mode keys (which would
    silently share noise draws; today's registry has none)."""
    import zlib
    k1, k2 = _registry_keys(2)
    r = _mode_result(1, k1)
    with pytest.raises(ValueError, match="seed"):
        posteriors.mock_realization([r], seed=-1)
    with pytest.raises(ValueError, match="empty"):
        posteriors.mock_realization([], seed=0)
    bad = dict(r, sigma=np.zeros_like(np.asarray(r["sigma"])))
    with pytest.raises(ValueError, match="finite and > 0"):
        posteriors.mock_realization([bad], seed=0)
    with pytest.raises(ValueError, match="mode_key"):
        posteriors.mock_realization([{"depth": r["depth"],
                                      "sigma": r["sigma"]}], seed=0)
    r2 = _mode_result(32, k2)
    real_one = posteriors.mock_realization([r], seed=0)
    with pytest.raises(ValueError, match="no draw for mode"):
        posteriors.mock_recovery([r, r2], FREE, real_one)
    with pytest.raises(ValueError, match="mock_realization record"):
        posteriors.mock_recovery([r], FREE, {"seed": 0})
    # today's registry is collision-free; a genuine crc32 collision (classic
    # colliding pair) is refused loudly
    posteriors._assert_mode_streams_distinct(instruments.MODES.keys())
    a, b = "plumless", "buckeroo"
    assert zlib.crc32(a.encode()) == zlib.crc32(b.encode())
    with pytest.raises(ValueError, match="collide"):
        posteriors._assert_mode_streams_distinct([a, b])


def test_mock_recovery_zero_mean_fisher_covariance_and_labels():
    """Monte Carlo pin of the linearization identities: over realizations,
    E[delta_theta] = 0 and Cov[delta_theta] = (F^-1)_free on the SAME
    stacked nuisance-augmented system as combined_forecast. Also pins the
    record's honesty labels and the display transform (mirrors
    fisher.display_sigma: dlnCO -> C/O, lnZ -> dex)."""
    k1, k2 = _registry_keys(2)
    rs = [_mode_result(11, k1), _mode_result(12, k2)]
    n_f = len(FREE)

    # expected covariance: free-block of the inverse joint Fisher matrix
    n_tot, system = fisher._combined_system(rs, n_f)
    F = np.zeros((n_tot, n_tot))
    for Jg, r in system:
        F += (Jg / np.asarray(r["sigma"])[None, :] ** 2) @ Jg.T
    cov_want = np.linalg.inv(F)[:n_f, :n_f]

    n_draw = 4000
    deltas = np.empty((n_draw, n_f))
    for i in range(n_draw):
        real = posteriors.mock_realization(rs, seed=i)
        rec = posteriors.mock_recovery(rs, FREE, real, co_eval=CO)
        deltas[i] = [rec["delta"][n] for n in FREE]
    mean = deltas.mean(axis=0)
    cov = np.cov(deltas.T)
    sig = np.sqrt(np.diag(cov_want))
    # E[delta] = 0 within MC error (~sig/sqrt(N), take 5x margin)
    assert np.all(np.abs(mean) < 5.0 * sig / np.sqrt(n_draw)), mean
    # Cov[delta] = F^-1 within MC error (relative, correlation-aware)
    assert np.allclose(cov, cov_want, rtol=0.15,
                       atol=0.15 * float(np.max(np.abs(cov_want))))
    # and the marginal widths match the reported marginalized sigmas
    marg = fisher.combined_forecast(rs, FREE)
    for j, name in enumerate(FREE):
        assert deltas[:, j].std() == pytest.approx(marg[name], rel=0.1)

    # labels + display transform on a single-mode recovery
    r = _mode_result(21, k1)
    real = posteriors.mock_realization([r], seed=5)
    rec = posteriors.mock_recovery([r], FREE, real, co_eval=CO)
    assert rec["kind"] == posteriors.MOCK_KIND
    assert "not a retrieval" in rec["label"]
    assert rec["seed"] == 5
    assert rec["recovered"] == {n: True for n in FREE}
    assert rec["delta_display"]["dlnCO"] == pytest.approx(
        CO * rec["delta"]["dlnCO"], rel=1e-12)
    assert rec["delta_display"]["lnZ"] == pytest.approx(
        rec["delta"]["lnZ"] / np.log(10.0), rel=1e-12)
    assert rec["units"]["dlnCO"] == "C/O ratio"


def test_scoring_modules_never_reference_the_mock_layer():
    """Structural pin of the ONE-DIRECTIONAL rule (not "display only": the
    draw IS fitted by mock_recovery, whose recovered-parameter shift the GUI
    overlays on the posterior panels). The scoring, forecast and export
    modules must carry NO reference to the mock-observation layer -- no
    `posteriors` import, no mock_realization/mock_recovery/depth_mock name
    -- so the draw can never flow back into a score, a cache, or a noiseless
    export. Checks the parsed AST, not raw text, so this docstring cannot
    trip it."""
    import ast
    import pathlib

    src = pathlib.Path(posteriors.__file__).parent
    banned = {"mock_realization", "mock_recovery", "depth_mock",
              "MOCK_KIND", "MOCK_LABEL"}
    for mod in ("detect.py", "fisher.py", "noise.py", "forward.py",
                "binning.py", "share_config.py"):
        tree = ast.parse((src / mod).read_text(), str(src / mod))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                mod_name = getattr(node, "module", "") or ""
                assert "posteriors" not in names and \
                    "posteriors" not in mod_name, (
                    f"{mod}:{node.lineno} imports posteriors -- the mock "
                    "layer must never enter a scoring module")
            hit = None
            if isinstance(node, ast.Name) and node.id in banned:
                hit = node.id
            elif isinstance(node, ast.Attribute) and node.attr in banned:
                hit = node.attr
            assert hit is None, (
                f"{mod}:{node.lineno} references {hit!r} -- the mock draw "
                "must never enter scores, caches, or result exports (it is "
                "fitted only by mock_recovery, consumed in app.py)")


def test_mock_draw_uses_the_same_sigma_as_the_reported_error_bars():
    """The mock draw must be a realization of the SAME noise model whose
    error bars and forecasts are displayed beside it: no scale factor, and
    the drawn scatter must match the reported sigma. A scale knob that moved
    the dots without moving the forecast was removed in 0.29.4 -- this pins
    the invariant so it cannot come back silently."""
    import inspect

    # the API must not accept a scale/multiplier on the draw
    params = set(inspect.signature(posteriors.mock_realization).parameters)
    assert params == {"results", "seed"}, params

    r = dict(_result(n_bins=400))
    r["mode_key"] = "nirspec_prism"
    r["sigma"] = np.full(400, 3.0e-4)
    r["depth"] = np.full(400, 0.021)
    draws = []
    for sd in range(24):
        m = posteriors.mock_realization([r], seed=sd)["modes"]["nirspec_prism"]
        # depth_mock is exactly depth + noise, and sigma is REPORTED unscaled
        assert np.allclose(m["depth_mock"], r["depth"] + m["noise"])
        assert np.array_equal(m["sigma"], r["sigma"])
        draws.append(m["noise"] / r["sigma"])
    z = np.concatenate(draws)
    # standardized residuals ~ N(0,1): a scale factor of s would show up here
    # as std = s. 9600 samples -> std known to ~0.7%.
    assert abs(float(z.std()) - 1.0) < 0.05, float(z.std())
    assert abs(float(z.mean())) < 0.05, float(z.mean())
