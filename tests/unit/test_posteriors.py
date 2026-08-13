"""Fisher-Gaussian posterior curves + named mode combinations.

Pins: the honesty labeling on every record, the no-fake-finite-curve rule for
null Fisher directions, fail-fast input validation, the display-unit
conventions (dlnCO's absolute-C/O transform), the single-mode-combo ==
mode_forecast identity, and the saturated-mode exclusion policy (same as the
"ALL USABLE (combined)" row, disclosed per combo, loud on zero usable).
"""
import numpy as np
import pytest

from jwst_tool import fisher, posteriors


def _result(seed=0, n_free=2, n_bins=40, saturated=False):
    """Synthetic evaluate_mode-like result: jac rows [free..., lnR0]."""
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((n_free + 1, n_bins))
    return dict(jac_bins=J, sigma=np.full(n_bins, 1e-4),
                seg=np.zeros(n_bins, int), saturated=saturated)


FREE = ["lnZ", "dlnCO"]
CENTERS = {"lnZ": 1.0, "dlnCO": 0.55}   # display units: dex, absolute C/O
CO = 0.55


# --- marginalized_posteriors -------------------------------------------------

def test_labels_are_honest_and_present():
    out = posteriors.marginalized_posteriors(_result(), FREE, CENTERS,
                                             co_eval=CO)
    assert out["kind"] == posteriors.FORECAST_KIND
    for phrase in ("Fisher", "Cramer-Rao", "Not a sampled posterior"):
        assert phrase in out["label"], phrase
    # per-record labeling rides the combo records too (checked below)


def test_curve_matches_fisher_sigma_and_normalizes():
    r = _result()
    out = posteriors.marginalized_posteriors(r, FREE, CENTERS, co_eval=CO)
    sig = fisher.mode_forecast(r, FREE)
    for name in FREE:
        rec = out["params"][name]
        want = fisher.display_sigma(name, sig[name], co_eval=CO)
        assert rec["constrained"]
        assert rec["sigma_display"] == pytest.approx(want, rel=1e-12)
        assert rec["center"] == CENTERS[name]
        # pdf: peaks at the center, integrates to ~1 on the +/-5 sigma grid
        theta, pdf = rec["theta"], rec["pdf"]
        assert theta[np.argmax(pdf)] == pytest.approx(CENTERS[name], abs=1e-12)
        assert np.trapz(pdf, theta) == pytest.approx(1.0, abs=1e-4)
        assert theta.min() == pytest.approx(CENTERS[name] - 5 * want)
        assert theta.max() == pytest.approx(CENTERS[name] + 5 * want)


def test_dlnco_display_transform_is_absolute_co():
    """sigma_display(dlnCO) = C/O * sigma_internal (the fisher-table rule)."""
    r = _result()
    sig_int = fisher.mode_forecast(r, FREE)["dlnCO"]
    out = posteriors.marginalized_posteriors(r, FREE, CENTERS, co_eval=CO)
    assert out["params"]["dlnCO"]["sigma_display"] == pytest.approx(
        CO * sig_int, rel=1e-12)
    assert out["params"]["dlnCO"]["unit"] == "C/O ratio"
    # and dlnCO without co_eval raises, exactly like display_sigma
    with pytest.raises(ValueError, match="co_eval"):
        posteriors.marginalized_posteriors(r, FREE, CENTERS)


def test_null_direction_is_flagged_never_a_fake_curve():
    """Duplicate Jacobian rows: both parameters must come back unconstrained
    with NO curve -- never a finite Gaussian."""
    rng = np.random.default_rng(3)
    row = rng.standard_normal(50)
    J = np.stack([row, row, rng.standard_normal(50)])   # [p0, p1, lnR0]
    r = dict(jac_bins=J, sigma=np.full(50, 1e-4))
    out = posteriors.marginalized_posteriors(
        r, ["p0", "p1"], {"p0": 0.0, "p1": 0.0})
    for name in ("p0", "p1"):
        rec = out["params"][name]
        assert not rec["constrained"]
        assert rec["theta"] is None and rec["pdf"] is None
        assert np.isinf(rec["sigma_display"])


def test_caller_grid_is_used_verbatim():
    r = _result()
    grid = np.linspace(-2.0, 4.0, 33)
    out = posteriors.marginalized_posteriors(
        r, FREE, CENTERS, co_eval=CO, grids={"lnZ": grid})
    assert np.array_equal(out["params"]["lnZ"]["theta"], grid)


def test_nonfinite_inputs_raise():
    r = _result()
    bad_sigma = dict(r, sigma=np.where(np.arange(40) == 5, np.nan, r["sigma"]))
    with pytest.raises(ValueError, match="finite"):
        posteriors.marginalized_posteriors(bad_sigma, FREE, CENTERS, co_eval=CO)
    bad_jac = dict(r, jac_bins=r["jac_bins"].copy())
    bad_jac["jac_bins"][0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        posteriors.marginalized_posteriors(bad_jac, FREE, CENTERS, co_eval=CO)
    with pytest.raises(ValueError, match="centers"):
        posteriors.marginalized_posteriors(r, FREE, {"lnZ": 1.0}, co_eval=CO)
    with pytest.raises(ValueError, match="finite"):
        posteriors.marginalized_posteriors(
            r, FREE, {"lnZ": np.nan, "dlnCO": 0.5}, co_eval=CO)


def test_row_count_mismatch_raises():
    """free_names must match the Jacobian's [free..., lnR0] layout."""
    r = _result(n_free=2)
    with pytest.raises(ValueError, match="rows"):
        posteriors.marginalized_posteriors(r, ["lnZ"], {"lnZ": 1.0})


def test_gaussian_curve_refuses_inf_sigma():
    with pytest.raises(ValueError, match="unconstrained"):
        posteriors.gaussian_curve(0.0, np.inf)


def test_list_of_results_uses_combined_forecast():
    rs = [_result(seed=1), _result(seed=2)]
    out = posteriors.marginalized_posteriors(rs, FREE, CENTERS, co_eval=CO)
    sig = fisher.combined_forecast(rs, FREE)
    assert out["n_modes"] == 2
    for name in FREE:
        want = fisher.display_sigma(name, sig[name], co_eval=CO)
        assert out["params"][name]["sigma_display"] == pytest.approx(
            want, rel=1e-12)


# --- named combinations ------------------------------------------------------

def _registry_keys(n):
    from jwst_tool import instruments as ins
    return list(ins.MODES)[:n]


def test_single_mode_combo_equals_mode_forecast():
    """The pinned upstream identity must survive the combo wrapper."""
    (key,) = _registry_keys(1)
    r = _result(seed=4)
    rec = posteriors.combo_forecast("just one", [key], {key: r}, FREE,
                                    co_eval=CO)
    sig = fisher.mode_forecast(r, FREE)
    for name in FREE:
        want = fisher.display_sigma(name, sig[name], co_eval=CO)
        assert rec["sigma_marginalized_display"][name] == pytest.approx(
            want, rel=1e-12)
    assert rec["kind"] == posteriors.FORECAST_KIND
    assert rec["posteriors"] is None and rec["posteriors_note"]


def test_combo_unknown_mode_key_raises():
    with pytest.raises(ValueError, match="unknown mode key"):
        posteriors.combo_forecast("bad", ["not_a_mode"],
                                  {"not_a_mode": _result()}, FREE, co_eval=CO)


def test_combo_saturated_modes_excluded_and_disclosed():
    k1, k2 = _registry_keys(2)
    results = {k1: _result(seed=5), k2: _result(seed=6, saturated=True)}
    rec = posteriors.combo_forecast("mixed", [k1, k2], results, FREE,
                                    co_eval=CO)
    assert rec["usable_modes"] == [k1]
    assert [e["mode_key"] for e in rec["excluded"]] == [k2]
    assert "saturated" in rec["excluded"][0]["reason"]
    # the forecast equals the usable subset alone (exclusion is real)
    sig = fisher.combined_forecast([results[k1]], FREE)
    for name in FREE:
        want = fisher.display_sigma(name, sig[name], co_eval=CO)
        assert rec["sigma_marginalized_display"][name] == pytest.approx(
            want, rel=1e-12)


def test_combo_all_saturated_raises():
    (key,) = _registry_keys(1)
    with pytest.raises(ValueError, match="no usable mode"):
        posteriors.combo_forecast("dead", [key],
                                  {key: _result(saturated=True)}, FREE,
                                  co_eval=CO)


def test_combo_with_centers_attaches_curves():
    k1, k2 = _registry_keys(2)
    results = {k1: _result(seed=7), k2: _result(seed=8)}
    rec = posteriors.combo_forecast("pair", [k1, k2], results, FREE,
                                    centers=CENTERS, co_eval=CO)
    post = rec["posteriors"]
    assert post is not None and post["n_modes"] == 2
    assert post["params"]["lnZ"]["constrained"]
    # combined sigma tighter than (or equal to) either single mode
    single = fisher.mode_forecast(results[k1], FREE)["lnZ"]
    assert post["sigma_marginalized"]["lnZ"] <= single * (1 + 1e-12)


def test_combo_duplicate_keys_and_missing_results_raise():
    k1, k2 = _registry_keys(2)
    with pytest.raises(ValueError, match="duplicate"):
        posteriors.combo_forecast("dup", [k1, k1], {k1: _result()}, FREE,
                                  co_eval=CO)
    with pytest.raises(ValueError, match="no result"):
        posteriors.combo_forecast("missing", [k1, k2], {k1: _result()}, FREE,
                                  co_eval=CO)


# --- mock-observation layer --------------------------------------------------

def _mode_result(seed, key, n_free=2, n_bins=30):
    r = _result(seed=seed, n_free=n_free, n_bins=n_bins)
    r["mode_key"] = key
    r["depth"] = np.full(n_bins, 0.021)
    return r


def test_mock_realization_is_seeded_and_reproducible():
    k1, k2 = _registry_keys(2)
    rs = [_mode_result(1, k1), _mode_result(2, k2)]
    a = posteriors.mock_realization(rs, seed=7)
    b = posteriors.mock_realization(rs, seed=7)
    c = posteriors.mock_realization(rs, seed=8)
    assert a["seed"] == 7 and a["kind"] == posteriors.MOCK_KIND
    for k in (k1, k2):
        assert np.array_equal(a["modes"][k]["noise"], b["modes"][k]["noise"])
        assert not np.array_equal(a["modes"][k]["noise"],
                                  c["modes"][k]["noise"])
        assert np.array_equal(
            a["modes"][k]["depth_mock"],
            np.asarray(rs[[k1, k2].index(k)]["depth"])
            + a["modes"][k]["noise"])


def test_mock_realization_per_mode_stream_is_selection_independent():
    """A mode's draw depends only on (seed, mode_key): adding or reordering
    other modes must not change it."""
    k1, k2 = _registry_keys(2)
    r1, r2 = _mode_result(1, k1), _mode_result(2, k2)
    alone = posteriors.mock_realization([r1], seed=3)
    both = posteriors.mock_realization([r2, r1], seed=3)
    assert np.array_equal(alone["modes"][k1]["noise"],
                          both["modes"][k1]["noise"])


def test_mock_realization_never_touches_global_numpy_state():
    (k1,) = _registry_keys(1)
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    posteriors.mock_realization([_mode_result(1, k1)], seed=0)
    assert np.array_equal(np.random.get_state()[1], before)


def test_mock_realization_validates_loudly():
    (k1,) = _registry_keys(1)
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


def test_mock_recovery_zero_mean_and_fisher_covariance():
    """Monte Carlo pin of the linearization identities: over realizations,
    E[delta_theta] = 0 and Cov[delta_theta] = (F^-1)_free on the SAME
    stacked nuisance-augmented system as combined_forecast."""
    k1, k2 = _registry_keys(2)
    rs = [_mode_result(11, k1), _mode_result(12, k2)]
    results = {k1: rs[0], k2: rs[1]}
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
        rec = posteriors.mock_recovery(list(results.values()), FREE, real,
                                       co_eval=CO)
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


def test_mock_recovery_display_transform_and_labels():
    (k1,) = _registry_keys(1)
    r = _mode_result(21, k1)
    real = posteriors.mock_realization([r], seed=5)
    rec = posteriors.mock_recovery([r], FREE, real, co_eval=CO)
    assert rec["kind"] == posteriors.MOCK_KIND
    assert "not a retrieval" in rec["label"]
    assert rec["seed"] == 5
    assert rec["recovered"] == {n: True for n in FREE}
    # display transform mirrors fisher.display_sigma (linear): dlnCO -> C/O
    assert rec["delta_display"]["dlnCO"] == pytest.approx(
        CO * rec["delta"]["dlnCO"], rel=1e-12)
    assert rec["delta_display"]["lnZ"] == pytest.approx(
        rec["delta"]["lnZ"] / np.log(10.0), rel=1e-12)
    assert rec["units"]["dlnCO"] == "C/O ratio"


def test_mock_recovery_null_direction_has_no_recovered_value():
    """Duplicate Jacobian rows: the unconstrained directions must come back
    recovered=False with None shifts -- never a number."""
    (k1,) = _registry_keys(1)
    rng = np.random.default_rng(6)
    row = rng.standard_normal(30)
    J = np.stack([row, row, rng.standard_normal(30)])
    r = dict(jac_bins=J, sigma=np.full(30, 1e-4), mode_key=k1,
             depth=np.full(30, 0.02))
    real = posteriors.mock_realization([r], seed=0)
    rec = posteriors.mock_recovery([r], ["p0", "p1"], real)
    for name in ("p0", "p1"):
        assert rec["recovered"][name] is False
        assert rec["delta"][name] is None
        assert rec["delta_display"][name] is None


def test_mock_recovery_requires_matching_realization():
    k1, k2 = _registry_keys(2)
    r1, r2 = _mode_result(31, k1), _mode_result(32, k2)
    real_one = posteriors.mock_realization([r1], seed=0)
    with pytest.raises(ValueError, match="no draw for mode"):
        posteriors.mock_recovery([r1, r2], FREE, real_one)
    with pytest.raises(ValueError, match="mock_realization record"):
        posteriors.mock_recovery([r1], FREE, {"seed": 0})


def test_mock_realization_never_mutates_the_inputs():
    """The draw is a NEW array layered on top: the caller's result dicts
    (the noiseless depths and sigmas that feed scores and CSVs) must come
    back byte-identical."""
    (k1,) = _registry_keys(1)
    r = _mode_result(41, k1)
    depth_before = np.asarray(r["depth"]).copy()
    sigma_before = np.asarray(r["sigma"]).copy()
    real = posteriors.mock_realization([r], seed=2)
    assert np.array_equal(np.asarray(r["depth"]), depth_before)
    assert np.array_equal(np.asarray(r["sigma"]), sigma_before)
    assert real["modes"][k1]["depth_mock"] is not r["depth"]


def test_scoring_modules_never_reference_the_mock_layer():
    """Structural pin of the display-only rule: the scoring/forecast/export
    modules must carry NO reference to the mock-observation layer -- no
    `posteriors` import, no mock_realization/mock_recovery/depth_mock name.
    Only app.py may consume the mock layer, and only for display and the
    clearly-named mock CSV. Checks the parsed AST, not raw text, so this
    docstring cannot trip it.
    """
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
                f"{mod}:{node.lineno} references {hit!r} -- mock draws are "
                "display-only and must never enter scores, caches, or "
                "result exports")


def test_compare_combos_orders_and_rejects_duplicates():
    k1, k2 = _registry_keys(2)
    results = {k1: _result(seed=9), k2: _result(seed=10)}
    recs = posteriors.compare_combos(
        {"A": [k1], "A + B": [k1, k2]}, results, FREE, co_eval=CO)
    assert [r["name"] for r in recs] == ["A", "A + B"]
    # more modes cannot loosen the marginalized bound
    assert (recs[1]["sigma_marginalized_display"]["lnZ"]
            <= recs[0]["sigma_marginalized_display"]["lnZ"] * (1 + 1e-12))
    with pytest.raises(ValueError, match="duplicate combo names"):
        posteriors.compare_combos([("A", [k1]), ("A", [k2])], results, FREE,
                                  co_eval=CO)
