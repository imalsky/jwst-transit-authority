"""Canonical parameters: leave-one-out sets, T-P files, emission gating, the key set and the opacity gate."""

import numpy as np
import pytest
from param_helpers import W39B_REFERENCE, _p, _pf, _table

from jwst_tool import chem_build, fisher, forward, jacobian, model_cache, model_defs, planets, run_params, tp_tables


def test_wo_mols_canonical_form():
    """Default = every RT molecule in fold order; a subset canonicalizes to fold order deduped; unknown molecules
    refuse; [] is the constrain goal's valid empty set; a saved payload round-trips.
    """
    cp = forward.canonical_params(_p())
    assert cp["wo_mols"] == forward.MOLECULES
    cp = forward.canonical_params(_p(extra_mols=["HCN", "C2H2"]))
    assert cp["wo_mols"] == [*forward.MOLECULES, "C2H2", "HCN"]
    cp = forward.canonical_params(_p(wo_mols=["SO2", "H2O", "SO2"]))
    assert cp["wo_mols"] == ["H2O", "SO2"]
    assert forward.canonical_params(_p(wo_mols=[]))["wo_mols"] == []
    with pytest.raises(ValueError):
        forward.canonical_params(_p(wo_mols=["HCN"]))  # not in the RT set
    assert forward.canonical_params(cp)["wo_mols"] == cp["wo_mols"]


def test_wo_mols_separates_cache_keys():
    """A constrain run must never serve a detect run's cache slot."""
    k_all = forward.params_key(_p())
    k_none = forward.params_key(_p(wo_mols=[]))
    k_sub = forward.params_key(_p(wo_mols=["H2O"]))
    assert len({k_all, k_none, k_sub}) == 3
    assert forward.params_key(_p(wo_mols=forward.MOLECULES)) == k_all


@pytest.mark.parametrize(
    ("kw", "n_wo_stages"),
    [
        (dict(), 1),
        (dict(wo_mols=[]), 0),
        (dict(science_mode="emission", tp_mode="guillot", Tirr=1560.0), 1),
        (dict(science_mode="emission", wo_mols=[], tp_mode="guillot", Tirr=1560.0), 0),
    ],
)
def test_progress_stages_mirror_the_solve_sequence(kw, n_wo_stages):
    """No removed-molecule stage for [], exactly one batch stage otherwise."""
    lines = []
    advance, _fin = forward._make_progress(forward.canonical_params(_p(**kw)), lines.append)

    def run_every_stage():
        for _ in range(100):
            advance()

    with pytest.raises(IndexError):
        run_every_stage()
    assert sum("removed-molecule" in s for s in lines) == n_wo_stages


def test_consumers_index_removed_spectra_by_wo_mols():
    """depth_wo aligns with the model's wo_mols, never mols, and a target with no removed spectrum refuses loudly."""
    from jwst_tool import detect

    n = 8
    order = np.arange(n)
    model = {
        "science_mode": "transmission",
        "wo_mols": np.array(["CO2"], dtype="U8"),
        "depth_wo": np.arange(n, dtype=float)[None, :] * 1e-3,
    }
    mols = ["H2O", "CO2", "CO"]
    got = detect._removed_spectrum(model, mols, "CO2", order)
    assert np.array_equal(got, model["depth_wo"][0])  # NOT mols.index -> 1
    assert detect._removed_spectrum(model, mols, None, order) is None
    with pytest.raises(ValueError, match="wo_mols"):
        detect._removed_spectrum(model, mols, "CO", order)  # constrain cache
    with pytest.raises(ValueError, match="extra_mols"):
        detect._removed_spectrum(model, mols, "HCN", order)  # not in RT set
    with pytest.raises(KeyError):
        detect._removed_spectrum({"depth_wo": model["depth_wo"]}, mols, "CO2", order)  # malformed payload
    unnamed = {k: v for k, v in model.items() if k != "science_mode"}
    with pytest.raises(KeyError, match="science_mode"):
        detect._removed_spectrum(unnamed, mols, "CO2", order)  # read as transmission, it skips the emission gate

    emission = {**model, "science_mode": "emission", "emis_thin_flux_frac_wo": np.array([0.0])}
    assert np.array_equal(detect._removed_spectrum(emission, mols, "CO2", order), model["depth_wo"][0])
    with pytest.raises(ValueError, match="emis_thin_flux_frac_wo"):
        detect._removed_spectrum({"wo_mols": model["wo_mols"], "science_mode": "emission"}, mols, "CO2", order)
    for malformed in (np.array([np.nan]), np.array([0.0, 0.0]), np.array([1.1])):  # non-finite, wrong shape, > 1
        with pytest.raises(ValueError, match="emis_thin_flux_frac_wo"):
            detect._removed_spectrum(
                {"wo_mols": model["wo_mols"], "science_mode": "emission", "emis_thin_flux_frac_wo": malformed},
                mols,
                "CO2",
                order,
            )


def test_wasp39b_reference_cache_key_and_table_bytes_are_stable():
    # The key hashes every canonical parameter: if ANY default feeding the
    # reference run changes, this trips even when the pins above still pass.
    # Re-pinning it follows a _VERSION bump (the version is a canonical key):
    # record what moved the spectrum, and whether the canonical
    # parameter SET moved with it.
    # The G395H SO2 significance behind this configuration is in notes
    # "Composition, C/O bounds and networks".
    assert forward.params_key(forward.canonical_params(dict(planet="wasp39b", tp_mode="file"))) == "6b7ef08447c10fdc"
    # ... and the bare DEFAULT run is that same atmosphere
    assert forward.params_key(forward.canonical_params(dict(planet="wasp39b"))) == "6b7ef08447c10fdc"
    # the sha1 pin is only meaningful re-derived from the file the run
    # actually reads -- this catches the table itself being swapped
    path = tp_tables.shipped_tp_file("wasp39b")
    assert path.name == "atm_W39b_evening_TP_Kzz.txt"
    tab = tp_tables.read_tp_table(path)
    assert tab["Kzz"] is not None, "the reference table must carry its Kzz column"
    import hashlib

    assert hashlib.sha1(path.read_bytes()).hexdigest()[:16] == W39B_REFERENCE["tp_file_sha1"]


# --- tp_mode="file" ---------------------------------------------------------


def test_file_mode_content_addressing_hygiene_and_bad_tables(tmp_path):
    p1 = _table(tmp_path, name="a.txt")
    p2 = _table(tmp_path, name="b.txt")  # same content, other path
    p3 = _table(tmp_path, name="c.txt", tmax=1500.0)  # different content
    cp1 = forward.canonical_params(_pf(p1))
    assert cp1["tp_file_sha1"] and len(cp1["tp_file_sha1"]) == 16
    assert forward.params_key(_pf(p1)) == forward.params_key(_pf(p2))
    assert forward.params_key(_pf(p1)) != forward.params_key(_pf(p3))
    # a tabulated profile has NO T-P Fisher rows
    assert model_defs.TP_PARAM_NAMES["file"] == []
    cp = forward.canonical_params(_pf(p1, fisher_params=["lnZ", "lnKzz"]))
    assert cp["fisher_params"] == ["lnKzz", "lnZ"]
    with pytest.raises(ValueError):
        forward.canonical_params(_pf(p1, fisher_params=["Tirr"]))
    # parametric T-P knobs are zeroed in file mode (cache hygiene) ...
    cp = forward.canonical_params(_pf(p1, Tirr=1560.0))
    assert cp["Tirr"] == cp["Tint"] == cp["log_kappa"] == cp["log_gamma"] == 0.0
    # ... and outside file mode the file identity is empty
    cp_g = forward.canonical_params(_p())
    assert cp_g["tp_file"] == "" and cp_g["tp_file_sha1"] == ""
    # malformed tables are rejected loudly: non-monotonic pressures, a
    # profile outside the modelable window, too few rows, no Pressure
    # column, and a path this machine does not hold
    nocol = tmp_path / "nocol.txt"
    nocol.write_text("#(dyne/cm2) (K)\nPress\tT\n1e6\t1000\n1e5\t900\n1e4\t800\n1e3\t700\n")
    for path in (
        _table(tmp_path, scramble=True, name="scram.txt"),
        _table(tmp_path, tmax=3300.0, name="hot.txt"),
        _table(tmp_path, rows=3, name="short.txt"),
        nocol,
        tmp_path / "missing.txt",
    ):
        with pytest.raises(ValueError):
            forward.canonical_params(_pf(path))

    # canonical_params is idempotent in file mode too
    # The GUI hands the SUBPROCESS canonical_params(params) as its params
    # file, so canonicalization must be idempotent (given a resolvable path).
    p = _table(tmp_path)
    cp = forward.canonical_params(_pf(p))
    cp2 = forward.canonical_params({**cp, "tp_file_path": str(p)})
    assert cp2 == cp


# --- kzz_mode ---------------------------------------------------------------


def test_kzz_modes_validate_and_zero_inert_knobs(tmp_path):
    with pytest.raises(ValueError):
        forward.canonical_params(_p(kzz_mode="file"))
    no_kzz = _table(tmp_path, kzz=False, name="nokzz.txt")
    with pytest.raises(ValueError):
        forward.canonical_params(_pf(no_kzz, kzz_mode="file"))
    cp = forward.canonical_params(_pf(_table(tmp_path), kzz_mode="file"))
    assert cp["kzz_const"] == cp["kzz_kmax"] == cp["kzz_plev"] == 0.0
    assert cp["kzz_kdeep"] == 0.0
    # parametric modes: only the active mode's knobs survive (cache hygiene)
    cp = forward.canonical_params(_p(kzz_mode="Pfunc", kzz_kmax=1.0e5, kzz_plev=0.1))
    assert cp["kzz_kmax"] == 1.0e5 and cp["kzz_plev"] == 0.1
    assert cp["kzz_const"] == 0.0 and cp["kzz_kdeep"] == 0.0
    cp = forward.canonical_params(_p(kzz_mode="JM16", kzz_kdeep=1.0e6))
    assert cp["kzz_kdeep"] == 1.0e6
    assert cp["kzz_const"] == cp["kzz_kmax"] == cp["kzz_plev"] == 0.0
    for bad in (
        dict(kzz_mode="Pfunc", kzz_kmax=1.0e15, kzz_plev=0.1),
        dict(kzz_mode="Pfunc", kzz_kmax=1.0e5, kzz_plev=1.0e5),
        dict(kzz_mode="JM16", kzz_kdeep=1.0),
        dict(kzz_mode="scale"),
    ):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(**bad))


# --- removed boundary fluxes / escape / settling -----------------------------


def test_boundary_flux_escape_and_settling_are_removed():
    """The four keys are gone from the payload (model version 63), and a request that names one, with any value, is
    refused by name rather than silently ignored.
    """
    cp0 = forward.canonical_params(_p())
    assert not {"use_settling", "diff_esc", "top_flux", "bot_flux"} & set(cp0)
    for named in (
        dict(use_settling=False),
        dict(use_settling=True),
        dict(diff_esc=[]),
        dict(top_flux=[["H2O", 1.0e8]]),
        dict(bot_flux=[["SO2", 1.0e9, 0.1]]),
    ):
        with pytest.raises(ValueError, match=rf"\['{next(iter(named))}'\] were removed"):
            forward.canonical_params(_p(**named))


# --- cloud deck -------------------------------------------------------------


def test_cloud_fisher_rows_require_their_deck(tmp_path):
    cp = forward.canonical_params(_p(cloud_on=True, fisher_params=["lnZ", "log_kappa_cloud", "alpha_cloud"]))
    assert set(cp["fisher_params"]) == {"alpha_cloud", "lnZ", "log_kappa_cloud"}
    with pytest.raises(ValueError):
        forward.canonical_params(_p(fisher_params=["log_kappa_cloud"]))
    # the deck stays freeable in file mode (no parametric T-P required)
    cp3 = forward.canonical_params(_pf(_table(tmp_path), cloud_on=True, fisher_params=["lnZ", "log_kappa_cloud"]))
    assert {"log_kappa_cloud"} <= set(cp3["fisher_params"])


def test_every_freeable_param_has_display_metadata():
    # Any parameter that can be freed in the Fisher forecast is also a valid
    # constraint goal and a Jacobian row, so it must carry a display label,
    # unit, symbol, and an FD step -- a missing entry KeyErrors the GUI/run.
    freeable = (
        set(model_defs.CHEM_PARAM_NAMES)
        | set(model_defs.CLOUD_FISHER_PARAMS)
        | {p for ns in model_defs.TP_PARAM_NAMES.values() for p in ns}
    )
    for m in (fisher.PARAM_LABELS, fisher.PARAM_UNITS, fisher.PARAM_SYMBOLS, model_defs.FD_STEPS):
        assert not (freeable - set(m)), sorted(freeable - set(m))


# --- emission mode (science_mode) ------------------------------------------


def test_emission_mode_gating_star_params_and_hygiene(tmp_path):
    with pytest.raises(ValueError):
        forward.canonical_params(_p(science_mode="reflection"))
    cp = forward.canonical_params(_p(science_mode="emission", tp_mode="guillot", Tirr=1560.0))
    assert cp["science_mode"] == "emission"
    # star defaults come from the planet registry
    assert cp["star_teff"] == 5485.0 and cp["star_logg"] == 4.5
    # transmission zeroes the star identity (it is noise-side only there)
    cp_t = forward.canonical_params(_p())
    assert cp_t["star_teff"] == cp_t["star_logg"] == cp_t["star_feh"] == 0.0
    # rayleigh + chord scheme are transmission-only: normalized in emission
    cp_e = forward.canonical_params(
        _p(science_mode="emission", tp_mode="guillot", Tirr=1560.0, use_rayleigh=True, rt_integration="trapezoid")
    )
    assert cp_e["use_rayleigh"] is False
    assert cp_e["rt_integration"] == "simpson"
    with pytest.raises(ValueError):
        forward.canonical_params(_p(science_mode="emission", tp_mode="guillot", Tirr=1560.0, star_teff=10000.0))
    # emission works with a tabulated profile too
    cp_f = forward.canonical_params(_pf(_table(tmp_path), science_mode="emission"))
    assert cp_f["science_mode"] == "emission"
    # The default follows the STRUCTURE, not the geometry -- a measured
    # table caps the column at its own bottom (7.6 bar for the shipped
    # tables), parametric profiles get the round 10 bar. Emission needs no
    # deeper default: measured, 7.6 bar is optically thick (tau 30-370) in
    # every instrument window.
    assert cp_f["p_btm_bar"] == model_defs.P_BTM_FILE_BAR
    assert cp_t["p_btm_bar"] == model_defs.P_BTM_PARAMETRIC_BAR
    # a deeper column is still reachable, and still gated on the table covering
    with pytest.raises(ValueError):
        forward.canonical_params(_pf(_table(tmp_path), science_mode="emission", p_btm_bar=100.0))
    cp_d = forward.canonical_params(
        _pf(_table(tmp_path, name="deep.txt", p_lo_log10=8.1), science_mode="emission", p_btm_bar=100.0)
    )
    assert cp_d["p_btm_bar"] == 100.0
    # the two geometries can never share a cache entry
    assert forward.params_key(_p(science_mode="emission", tp_mode="guillot", Tirr=1560.0)) != forward.params_key(
        _p(tp_mode="guillot", Tirr=1560.0)
    )


# --- composition FD stencils + the dayside emission default ----------------


def test_composition_fd_stencil_envelope():
    """FD Fisher rows solve the chemistry at every stencil point, so a
    baseline whose stencil leaves the validated range refuses (met=100 ->
    122x solar, co=0.12 -> C/O 0.098). The dlnCO stencil steps one-sided away
    from the network's C/O ceiling instead of across it. AD rows take no
    stencil and are exempt here (the dlnCO AD row has its own run-time margin
    gate).
    """
    forward.canonical_params(_p(met_x_solar=80.0, fisher_params=["lnZ"]))
    for kw, fp in (({"met_x_solar": 100.0}, "lnZ"), ({"met_x_solar": 0.1}, "lnZ"), ({"co_ratio": 0.12}, "dlnCO")):
        with pytest.raises(ValueError, match="stencil"):
            forward.canonical_params(_p(fisher_params=[fp], **kw))
    h0 = model_defs.FD_STEPS["dlnCO"]
    hi = model_defs.CO_MAX["sncho"]
    assert model_defs.fd_stencil("dlnCO", 0.55, hi) == ((1, -1, 2, -2), h0)
    assert model_defs.fd_stencil("dlnCO", 0.89, hi) == ((-1, -2, -4), h0 / 2)
    assert model_defs.fd_stencil("lnZ", 0.89, hi) == ((1, -1, 2, -2), model_defs.FD_STEPS["lnZ"])
    # the flip follows the network's photolysis-on ceiling, not a literal 1
    hi25 = model_defs.CO_MAX["sncho2025"]
    assert model_defs.fd_stencil("dlnCO", 2.0, hi25) == ((1, -1, 2, -2), h0)
    assert model_defs.fd_stencil("dlnCO", 9.0, hi25) == ((-1, -2, -4), h0 / 2)
    for co in (0.89, 0.95):
        forward.canonical_params(_p(co_ratio=co, fisher_params=["dlnCO"]))
    # both schemes' Richardson rows are exact on a cubic (sign and
    # coefficients of the one-sided stencil included); the one-sided row is
    # third order -- on a quartic its error is -f''''(x) h^3 / 3 exactly, so
    # halving h divides it by 8 -- while the central row stays exact
    x0, h = 0.3, 0.1

    def f(x):
        return x**3 - 2.0 * x

    def g(x):
        return x**4

    for offs in ((1, -1, 2, -2), (-1, -2, -4)):
        j1, j2 = jacobian.fd_estimates(offs, {s: f(x0 + s * h) for s in offs}, f(x0), h)
        assert abs((4.0 * j1 - j2) / 3.0 - (3.0 * x0**2 - 2.0)) < 1e-12
    err = {}
    for offs, hh in (((-1, -2, -4), h), ((-1, -2, -4), h / 2), ((1, -1, 2, -2), h)):
        j1, j2 = jacobian.fd_estimates(offs, {s: g(x0 + s * hh) for s in offs}, g(x0), hh)
        err[offs, hh] = abs((4.0 * j1 - j2) / 3.0 - 4.0 * x0**3)
    assert abs(err[(-1, -2, -4), h] / err[(-1, -2, -4), h / 2] - 8.0) < 1e-6
    assert err[(1, -1, 2, -2), h] < 1e-12
    # no stencil under AD; and without fisher_params the value is legal
    forward.canonical_params(_p(met_x_solar=100.0, fisher_params=["lnZ"], jac_method="ad"))
    forward.canonical_params(_p(met_x_solar=100.0))


def test_emission_defaults_to_the_dayside_temperature():
    """An eclipse sees the DAYSIDE; a published equilibrium temperature assumes
    full redistribution, which is a planet-average profile.

    Measured on HD 189733 b against the published MIRI/LRS eclipse spectrum
    (Inglis et al. 2024): the planet-average default puts the eclipse depth
    1.35x LOW (chi2 per point 934); the dayside default lands within 3% of the
    data (chi2 per point 31), the largest single effect in the emission path.
    """
    for key in planets.PLANETS:
        t = forward.canonical_params(dict(planet=key, tp_mode="guillot"))["Tirr"]
        e = forward.canonical_params(dict(planet=key, tp_mode="guillot", science_mode="emission"))["Tirr"]
        # 2**0.25 in temperature, on the 10 K rounding grid the widget uses
        assert e > t, key
        assert e == pytest.approx(t * 2**0.25, abs=10.0), (key, t, e)
    # an explicit T_irr still wins in both geometries
    for mode in ("transmission", "emission"):
        cp = forward.canonical_params(dict(planet="wasp39b", tp_mode="guillot", science_mode=mode, Tirr=1234.0))
        assert cp["Tirr"] == 1234.0


# --- unknown-key rejection -------------------------------------------------


def test_unknown_keys_refuse_with_a_hint_and_output_round_trips():
    # Unknown keys must refuse, never drop: a silently ignored
    # {"mode": "emission"} scores a TRANSMISSION spectrum against eclipse
    # data (chi2/N ~ 5e5).
    with pytest.raises(ValueError, match="science_mode"):
        forward.canonical_params(_p(mode="emission"))
    with pytest.raises(ValueError):
        forward.canonical_params(_p(totally_made_up=1))
    # ... while share_config validates a SAVED canonical payload by feeding
    # it back in, so every output key (echo fields included) is accepted
    cp = forward.canonical_params(dict(planet="wasp39b"))
    assert set(cp) <= run_params._KNOWN_PARAM_KEYS
    cp2 = forward.canonical_params(dict(cp))
    assert cp2 == cp


def test_param_keys_read_matches_the_source():
    # _PARAM_KEYS_READ is re-derived from the AST of canonical_params and the
    # helpers it calls, so adding a parameter without updating the allowlist
    # fails here instead of silently rejecting the new key at run time.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(run_params) + inspect.getsource(tp_tables))
    run_params_tree = ast.parse(inspect.getsource(run_params))
    funcs = {node.name for node in ast.walk(run_params_tree) if isinstance(node, ast.FunctionDef)} | {"resolve_tp_file"}
    keys = set()

    class V(ast.NodeVisitor):
        def visit_Call(self, node):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "get"
                and isinstance(f.value, ast.Name)
                and f.value.id == "params"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
            self.generic_visit(node)

        def visit_Subscript(self, node):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "params"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                keys.add(node.slice.value)
            self.generic_visit(node)

        def visit_Compare(self, node):
            if (
                len(node.comparators) == 1
                and isinstance(node.ops[0], ast.In)
                and isinstance(node.comparators[0], ast.Name)
                and node.comparators[0].id == "params"
                and isinstance(node.left, ast.Constant)
                and isinstance(node.left.value, str)
            ):
                keys.add(node.left.value)
            self.generic_visit(node)

    scanned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in funcs:
            scanned.add(node.name)
            V().visit(node)
    assert scanned == funcs, f"missing helper(s): {funcs - scanned}"
    assert keys == set(run_params._PARAM_KEYS_READ)


# --- ExoMolOP no-table gate ------------------------------------------------


def test_molecule_list_invariants():
    """Menu/default consistency, none of it enforced by the code itself."""
    import re

    # the preselected set must be a selectable subset with tables
    assert set(forward.EXTRA_MOLECULES_DEFAULT) <= set(forward.EXTRA_MOLECULES)
    assert not (set(forward.EXTRA_MOLECULES_DEFAULT) & forward._NO_EXOMOLOP_TABLE)
    # S_MOLECULES must catch EVERY sulfur-bearing species in either list, or
    # network="ncho" offers one the sulfur-free network cannot produce.
    # `S(?![a-z])` so SiO-style names never read as sulfur.
    s_bearing = {m for m in list(forward.MOLECULES) + forward.EXTRA_MOLECULES if re.search(r"S(?![a-z])", m)}
    assert s_bearing == set(model_defs.S_MOLECULES), (
        f"S_MOLECULES misses {sorted(s_bearing - set(model_defs.S_MOLECULES))} "
        f"/ over-lists {sorted(set(model_defs.S_MOLECULES) - s_bearing)}"
    )
    assert not (set(forward.MOLECULES) & set(forward.EXTRA_MOLECULES))

    # ... and one with no published table refuses at the API
    # ExoMolOP publishes no CS2/C2H6 k-table: the run must refuse at the
    # API with the pointed reason, not minutes later inside the RT build.
    for mol in sorted(forward._NO_EXOMOLOP_TABLE):
        with pytest.raises(ValueError, match="no published ExoMolOP k-table"):
            forward.canonical_params(_p(extra_mols=[mol]))


def test_no_exomolop_table_set_matches_the_installed_tables():
    # Data-gated cross-check: every OTHER extra must have a table installed,
    # and the frozenset's species must genuinely lack one. Skips without the
    # data root so light CI stays green.
    import importlib.util

    if importlib.util.find_spec("vulcan_forward") is None:
        pytest.skip("vulcan_forward not installed")
    from vulcan_forward import exomolop, paths as _fp

    try:
        _fp.data_root()
    except RuntimeError:
        pytest.skip("VULCAN_FORWARD_DATA not configured")
    have = set(exomolop.available())
    if not have:
        pytest.skip("no ExoMolOP tables installed")
    for mol in forward._NO_EXOMOLOP_TABLE:
        assert mol not in have, (
            f"{mol} now HAS an ExoMolOP table -- remove it from "
            "forward._NO_EXOMOLOP_TABLE and re-enable it in the GUI default"
        )
    missing = (set(forward.MOLECULES) | set(forward.EXTRA_MOLECULES)) - forward._NO_EXOMOLOP_TABLE - have
    assert not missing, (
        f"species {sorted(missing)} have no installed ExoMolOP table; fetch "
        "them or add them to forward._NO_EXOMOLOP_TABLE with a reason"
    )


def test_the_chemistry_radius_anchor_is_the_hydrostatic_identity():
    """The catalogue radius applies at p_ref_bar; chemistry anchors near 1 bar.
    Converting between them is exact for a constant-T, constant-mu column,
    where du/dlnP is constant and r(P) = 1/(1/R_ref + C ln(P/P_ref)). Checked
    against that closed form, then against the two invariants that matter: GM
    is unchanged (only the LEVEL Rp attaches to moves), and the catalogue
    gravity comes back at p_ref_bar.
    """
    kb, m_u = 1.380649e-16, 1.6605390666e-24
    p = np.geomspace(7.6, 1e-7, 400)
    T, mu = np.full(p.size, 1200.0), np.full(p.size, 2.3)
    rp, gs, p_ref, p_anchor = 9.1438e9, 422.0, 1.0e-3, 0.9233
    gm = gs * rp**2
    r = chem_build.radius_at_anchor(p, T, mu, rp, gs, p_ref, p_anchor)

    c = kb * 1200.0 / (2.3 * m_u * gm)
    assert r == pytest.approx(1.0 / (1.0 / rp + c * np.log(p_anchor / p_ref)), rel=1e-9)
    assert r < rp  # deeper is smaller
    # GM is level-free, so the catalogue gravity is recovered where it was
    # quoted
    back = chem_build.radius_at_anchor(p, T, mu, r, gm / r**2, p_anchor, p_ref)
    assert back == pytest.approx(rp, rel=1e-9)
    assert gm / back**2 == pytest.approx(gs, rel=1e-9)
    # the same pressure is a no-op, and outside the column refuses
    assert chem_build.radius_at_anchor(p, T, mu, rp, gs, p_ref, p_ref) == pytest.approx(rp, rel=1e-12)
    with pytest.raises(ValueError, match="inside the column"):
        chem_build.radius_at_anchor(p, T, mu, rp, gs, 50.0, p_anchor)

    # At the allowed 1-bar bottom the engine's anchor can be the exterior
    # interface. Integrate the remaining half layer at bottom-layer T/mu.
    p_edge = np.geomspace(1.0, 1e-7, 100)
    anchor = chem_build.chem_radius_anchor_bar(p_edge, 1e6, False)
    assert anchor > p_edge.max()
    assert chem_build.radius_at_anchor(p_edge, T[:100], mu[:100], rp, gs, p_ref, anchor) == pytest.approx(
        1.0 / (1.0 / rp + c * np.log(anchor / p_ref)), rel=1e-9
    )


def test_a_gravity_anchor_iterates_to_its_tolerance_and_is_refused_only_at_the_cap(monkeypatch):
    """A stable fixed point that needs more than three passes is used; a map still moving at the cap is refused,
    naming the gravity before and after its last pass."""
    from types import SimpleNamespace

    from jwst_tool import picaso_engine

    tol, cap = model_defs.ANCHOR_TOL, model_defs.ANCHOR_MAX_PASSES
    monkeypatch.setattr(chem_build, "build_tp", lambda _cp, g: (lambda _tp, _p: np.full(1, g),))
    cp = {"gs_cgs": 400.0}

    def guillot(step):
        return picaso_engine._guillot_gravity(cp, np.zeros(4), np.ones(1), np.ones(1), lambda T, _mu: step(float(T[0])))

    # contraction 0.5 towards 500: the 3rd pass still moves g by > tol, a later one settles
    calls = []
    g = guillot(lambda g: calls.append(g) or 500.0 + 0.5 * (g - 500.0))
    assert len(calls) > model_defs.ANCHOR_MIN_PASSES and abs(g - 500.0) / 500.0 < 2.0 * tol
    # a map that keeps moving g by 5 % per pass is refused at the cap, with distinct before and after values
    with pytest.raises(
        RuntimeError, match=rf"did not converge in {cap} passes: the last moved g from ([0-9.]+) to (?!\1)"
    ):
        guillot(lambda g: 1.05 * g)
    # the VULCAN chemistry anchor, on a radius map that keeps shrinking
    radii = iter(9.0e9 * 0.95**k for k in range(100))
    monkeypatch.setattr(chem_build, "chem_radius_anchor_bar", lambda *_a: 1.0)
    monkeypatch.setattr(chem_build, "radius_at_anchor", lambda *_a: next(radii))
    probe = SimpleNamespace(y0=np.ones((2, 2)), species_masses=np.ones(2), p_bar=np.ones(2))
    cp_v = {"gs_cgs": 400.0, "p_ref_bar": 1e-3, "rt_ptop_bar": 1e-7, "p_btm_bar": 10.0}
    with pytest.raises(RuntimeError, match=f"chemistry gravity anchor did not converge in {cap} passes"):
        chem_build._radius_anchor(cp_v, np.zeros(7), probe, 9.5e9, object(), False, lambda *_a: None)


def test_the_climate_anchor_refusal_names_its_last_pass(monkeypatch):
    from types import SimpleNamespace

    from jwst_tool import picaso_engine

    monkeypatch.setattr(chem_build, "build_tp", lambda _cp, g: (lambda _tp, p: np.full(np.size(p), 1000.0),))
    cp = {
        "gs_cgs": 400.0,
        "star_teff": 5000.0,
        "rstar_rsun": 1.0,
        "orbit_au": 0.05,
        "Tint": 200.0,
        "met_x_solar": 1.0,
        "co_ratio": 0.5,
    }
    col = SimpleNamespace(T_base=np.ones(2), y=np.ones((2, 1)), species_masses=np.ones(1))
    gs = iter(400.0 * 1.05**k for k in range(1, 100))
    with pytest.raises(RuntimeError, match=r"the last moved g from ([0-9.]+) to (?!\1)[0-9.]+ cm/s2 .* change T_int"):
        picaso_engine._climate_gravity(
            cp, np.ones(2), 2, 1.0, lambda *_a: col, lambda *_a: {}, lambda *_a: next(gs), lambda *_a: None
        )


def test_an_unstated_interior_temperature_runs_at_the_default_and_says_so():
    """T_int defaults to the registry's published value (WASP-107 b, Sing+2024, 460 K), else 300 K. A run that reads it
    without the request stating it records that for its caveat, under a key the chemistry key ignores; a saved payload
    maps to itself, and a flag naming a T_int other than the default is refused."""
    assert planets.TINT_DEFAULT_K == 300.0
    for planet in planets.PLANETS:
        for engine in (dict(tp_mode="guillot"), dict(chem_provider="picaso")):
            cp = forward.canonical_params(dict(planet=planet, **engine))
            assert (cp["Tint"], cp["tint_from_default"]) == (460.0 if planet == "wasp107b" else 300.0, True)
            assert forward.canonical_params(cp) == cp
            assert model_defs.TINT_RANGE[0] <= cp["Tint"] <= model_defs.TINT_RANGE[1]
    unstated = dict(planet="wasp39b", tp_mode="guillot")
    stated = dict(unstated, Tint=300.0)
    assert forward.canonical_params(stated)["tint_from_default"] is False
    assert forward.params_key(stated) != forward.params_key(unstated)
    assert model_cache.chem_key(stated) == model_cache.chem_key(unstated)
    assert forward.canonical_params(dict(planet="wasp39b"))["tint_from_default"] is False  # a table reads no T_int
    assert forward.canonical_params(dict(planet="wasp107b", tp_mode="guillot", Tint=120.0))["Tint"] == 120.0
    with pytest.raises(ValueError, match="tint_from_default=True names the default T_int, 300 K, but Tint=250"):
        forward.canonical_params(dict(stated, Tint=250.0, tint_from_default=True))


def test_the_run_network_is_handed_to_the_engine(monkeypatch):
    """vulcan3 freezes its network at the first import: the run's pair goes through the engine's select_network, which
    refuses another pair later in the process, never through the environment directly."""
    vulcan_chem = pytest.importorskip("vulcan_forward.vulcan_chem")
    calls = []
    monkeypatch.setattr(vulcan_chem, "select_network", lambda *pair: calls.append(pair))
    assert chem_build._select_network({"network": "ncho"}, lambda *_a: None) == model_defs.NETWORKS["ncho"]
    assert calls == [model_defs.NETWORKS["ncho"]]
