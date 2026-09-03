"""Pure-Python validation of forward.canonical_params (no chemistry stack).

The contract: structure is a Guillot profile, an explicit content-hash-keyed
table; no profile is ever silently substituted and
unknown parameter keys refuse. Condensation is detection-only. The WASP-39 b
reference state and its cache key are pinned below; re-measure against the
literature before moving either.
"""
import math

import numpy as np
import pytest

from jwst_tool import fisher, forward, planets


def _p(**kw):
    base = dict(planet="wasp39b", tp_mode="guillot",
                kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return base


def _table(tmp_path, kzz=True, name="prof.txt", tmin=800.0, tmax=1400.0,
           rows=8, scramble=False, p_lo_log10=6.9):
    # p_lo_log10: bottom pressure exponent. 6.9 (7.9e6 dyn/cm^2) clears the
    # 7.6 bar transmission column; an EMISSION run needs 100 bar, so pass 8.1.
    P = np.logspace(p_lo_log10, -1.0, rows)   # dyne/cm^2, descending (deep first)
    if scramble:
        P = P.copy()
        P[2], P[3] = P[3], P[2]               # break monotonicity
    T = np.linspace(tmax, tmin, rows)
    lines = ["#(dyne/cm2) (K)" + (" (cm2/s)" if kzz else ""),
             "Pressure\tTemp" + ("\tKzz" if kzz else "")]
    for i in range(rows):
        row = f"{P[i]:.6e}\t{T[i]:.1f}"
        if kzz:
            row += "\t1.0e9"
        lines.append(row)
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def _pf(path, **kw):
    base = dict(planet="wasp39b", tp_mode="file",
                tp_file=forward.TP_FILE_UPLOAD, tp_file_path=str(path),
                kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return base


def test_condensation_detection_only_with_full_refusal_matrix():
    # use_condense is canonical (default False); a detection-only
    # condensing run (photo + moldiff on, no Fisher) is ACCEPTED. The pinned
    # reservoir is not a reproducible function of the parameters, so
    # condensation + Fisher refuses under EVERY jac_method (FD included);
    # photo-off has no certifiable steady state; the growth term IS the
    # molecular-diffusion coefficient, so moldiff-off refuses too.
    assert forward.canonical_params(_p())["use_condense"] is False
    assert forward.canonical_params(
        _p(use_condense=True))["use_condense"] is True
    for jm in ("fd", "ad"):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(use_condense=True, jac_method=jm,
                                        fisher_params=["lnZ"]))
    # message must stay misread-proof: ~0.91 is a RELATIVE ERROR (tangent
    # ~91% wrong), never a 0.91 agreement ratio
    with pytest.raises(ValueError) as ei:
        forward.canonical_params(_p(use_condense=True, fisher_params=["lnZ"]))
    msg = str(ei.value)
    assert "91% wrong" in msg and "not a 9% mismatch" in msg
    with pytest.raises(ValueError):
        forward.canonical_params(_p(use_condense=True, use_photo=False))
    with pytest.raises(ValueError):
        forward.canonical_params(_p(use_condense=True, use_moldiff=False))


def test_conden_cfg_is_the_certified_recipe():
    # the S8 channel ships the certified convergence recipe: whole-column
    # pin, mtol_conv floor, sulfur-allotrope conver_ignore, trun_min bound
    c = forward.CONDEN_CFG
    assert c["condense_sp"] == ["S8"]
    assert c["fix_species"] == ["S8", "S8_l_s"]
    assert c["fix_species_from_coldtrap_lev"] is False
    assert c["mtol_conv"] == 1.0e-15
    assert {"S", "S2", "S3", "S4"} <= set(c["conver_ignore"])
    assert c["trun_min"] == c["stop_conden_time"]


def test_removed_modes_refuse_never_substitute():
    # no GCM profile may ever be silently substituted, and retired modes
    # refuse rather than defaulting -- for WASP-39b like every other planet
    with pytest.raises(ValueError):
        forward.canonical_params(dict(planet="wasp39b", tp_mode="baseline"))
    with pytest.raises(ValueError):
        forward.canonical_params(_p(kzz_mode="scale", kzz_x=1.0))
    assert all("has_gcm_baseline" not in pd for pd in planets.PLANETS.values())
    with pytest.raises(ValueError):
        forward.canonical_params(_p(tp_mode="isothermal"))
    # its old T_iso companion key is unknown now, and unknown keys REFUSE
    with pytest.raises(ValueError):
        forward.canonical_params(_p(T_iso=1100.0))


def test_shipped_tables_gate_defaults_and_are_never_substituted():
    """Maintainer decision: a planet whose bundled measured T-P/Kzz
    table is VERIFIED end-to-end defaults to it; every other planet defaults
    to analytic Guillot + constant Kzz and carries a written tp_table_note.
    Each planet resolves to ITS OWN table; a planet without one refuses
    loudly rather than borrowing another's atmosphere."""
    seen = {}
    for key in planets.PLANETS:
        cp = forward.canonical_params(dict(planet=key))
        if forward.shipped_tp_table_is_default(key):
            assert cp["tp_mode"] == "file", key
            assert cp["tp_file"] == forward.TP_FILE_SHIPPED, key
            assert cp["tp_file_sha1"], key            # content-addressed
            assert cp["kzz_mode"] == "file", key      # table carries Kzz
            assert cp["kzz_const"] == 0.0, key        # inert once tabulated
        else:
            assert cp["tp_mode"] == "guillot", key
            assert cp["kzz_mode"] == "const", key
            assert planets.PLANETS[key]["tp_table_note"], key
        name = forward.shipped_tp_table_name(key)
        if name:
            seen[key] = name
        else:
            with pytest.raises(ValueError):
                forward.canonical_params(dict(planet=key, tp_mode="file"))
    assert len(set(seen.values())) == len(seen)       # no shared table
    # the verified set is exactly WASP-39 b today; HD 189733 b ships a good
    # profile the solver does NOT certify at defaults, so it must stay
    # selectable-but-not-default -- choosing it explicitly still resolves
    assert [k for k in planets.PLANETS
            if forward.shipped_tp_table_is_default(k)] == ["wasp39b"]
    assert forward.shipped_tp_table_name("hd189733b")
    # its table carries a 6000 K thermosphere above 0.1 dyn/cm^2, outside the
    # modelable window, so file mode needs a model top at or below 1e-7 bar
    cp = forward.canonical_params(dict(planet="hd189733b", tp_mode="file",
                                       rt_ptop_bar=1.0e-7))
    assert cp["tp_mode"] == "file" and cp["kzz_mode"] == "file"
    with pytest.raises(ValueError):
        forward.canonical_params(dict(planet="hd189733b", tp_mode="file",
                                      rt_ptop_bar=1.0e-8))
    assert forward._default_tp_mode(dict(planet="wasp39b")) == "file"


def test_guillot_default_tirr_follows_planet_and_custom_system():
    """T_irr default = sqrt(2) * T_eq of the SELECTED planet on the GUI's
    20 K step grid (the widget step); a bare constant makes API and GUI
    defaults diverge. The custom planet derives T_eq from the entered star
    and orbit (T_eq = Teff sqrt(Rstar/2a)) -- as does every registry planet, so
    a refreshed stellar parameter can never leave a stale T_eq behind."""
    for key, p in planets.PLANETS.items():
        teq = planets.system_teq(p["star"]["teff"], p["rstar_rsun"],
                                 p["orbit_au"])
        expect = min(max(round(teq * math.sqrt(2.0) / 20.0) * 20.0,
                         800.0), 2500.0)
        cp = forward.canonical_params(dict(planet=key, tp_mode="guillot"))
        assert cp["Tirr"] == expect, key
    # per-planet values, pinned explicitly
    assert forward.default_tirr("wasp39b") == 1640.0
    assert forward.default_tirr("hd209458b") == 2040.0
    sys_cool = dict(star_teff=3300.0, rstar_rsun=0.30, orbit_au=0.05)
    teq = planets.system_teq(**sys_cool)                    # ~350 K
    expect = min(max(round(teq * math.sqrt(2.0) / 20.0) * 20.0, 800.0), 2500.0)
    cp = forward.canonical_params(dict(planet="custom", tp_mode="guillot",
                                       **sys_cool))
    assert cp["Tirr"] == expect
    # a different entered system must move the default
    cp_hot = forward.canonical_params(dict(
        planet="custom", tp_mode="guillot",
        star_teff=6100.0, rstar_rsun=1.2, orbit_au=0.03))
    assert cp_hot["Tirr"] != cp["Tirr"]
    # an explicit Tirr always wins over the derived default
    cp_exp = forward.canonical_params(dict(planet="custom", tp_mode="guillot",
                                           Tirr=1234.0, **sys_cool))
    assert cp_exp["Tirr"] == 1234.0


def test_tp_table_gates_are_grid_scoped_and_require_the_bottom(tmp_path):
    """Two table gates. (1) The T-window gate judges the profile the ENGINE
    evaluates (re-gridded onto CHEM_P_SPAN_DYN), not every raw row: a table
    extending past the grid (a hot thermosphere) is fine if the in-grid part
    is modelable. (2) A table stopping above the chemistry-grid bottom is
    REFUSED: the engine would clamp-extend the last tabulated T
    isothermally over the quench region."""
    def _write(name, T):
        p = tmp_path / name
        p.write_text("#(dyne/cm2) (K)\nPressure Temp\n"
                     + "\n".join(f"{pv:.6e} {t:.2f}" for pv, t in zip(P, T)))
        return p

    lo, hi = forward.CHEM_P_SPAN_DYN
    # extends both ways past the grid, in-grid profile is a modelable 900 K
    P = np.array([lo / 100, lo, hi, hi * 100])
    T = np.array([5000.0, 900.0, 900.0, 5000.0])       # out of window only outside
    assert forward._read_tp_table(_write("ok.txt", T))["T"].size == 4
    # in-grid profile itself breaches the ceiling -> refused
    T_bad = np.array([5000.0, 2990.0, 900.0, 5000.0])
    with pytest.raises(ValueError):
        forward._read_tp_table(_write("bad.txt", T_bad))
    # a 1 bar bottom is too shallow; the standard fixture (past P_b) passes
    Ps = np.logspace(6.0, -1.0, 8)
    shallow = tmp_path / "shallow.txt"
    shallow.write_text("#(dyne/cm2) (K)\nPressure\tTemp\n" + "\n".join(
        f"{Ps[i]:.6e}\t{1200.0 - 40.0 * i:.1f}" for i in range(8)) + "\n")
    with pytest.raises(ValueError):
        forward.canonical_params(_pf(shallow))
    forward.canonical_params(_pf(_table(tmp_path)))


def test_resolution_knobs_defaults_ranges_and_refusals():
    # there is no fidelity "quality" tier: nz/yconv are explicit and the RT
    # layer count is derived, not cache-keyed; the spectral grid is the
    # k-tables' own (no knob)
    cp = forward.canonical_params(_p())
    assert cp["nz"] == forward.NZ_DEFAULT == 100
    assert cp["yconv_cri"] == forward.YCONV_DEFAULT == 1.0e-2
    assert "quality" not in cp        # retired
    assert "art_nlayer" not in cp     # locked to nz in run_model, not cache-keyed
    # explicit in-range values accepted; the yconv ladder reaches its 1e-4
    # floor
    cp = forward.canonical_params(_p(nz=150, yconv_cri=1.0e-4))
    assert (cp["nz"], cp["yconv_cri"]) == (150, 1.0e-4)
    assert forward.YCONV_RANGE == (1.0e-4, 1.0e-2)
    for bad in (dict(nz=40), dict(nz=200), dict(yconv_cri=1.0),
                dict(yconv_cri=5.0e-5)):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(**bad))
    # the removed line-by-line / Mie keys refuse BY NAME (an old config says
    # what was dropped), never fall through to the generic unknown-key text
    for key in sorted(forward._REMOVED_PARAM_KEYS):
        with pytest.raises(ValueError, match="removed in 0.48.0"):
            forward.canonical_params(_p(**{key: 1}))


def test_extra_molecules_resolve_in_engine_and_unknown_refused():
    # Shami Tsai request: CS2 photochemical sulfur + the CH4-photolysis
    # hydrocarbons. Import-light: vulcan_forward.constants is pure constants.
    from vulcan_forward import constants as _vfc
    assert "C2H4" in forward.EXTRA_MOLECULES
    # CS2/C2H6 have no published ExoMolOP k-table: refused, never offered
    assert forward._NO_EXOMOLOP_TABLE == {"CS2", "C2H6"}
    assert not (forward._NO_EXOMOLOP_TABLE & set(forward.EXTRA_MOLECULES))
    # EVERY extra must resolve in the shared engine's molecule table -- a
    # token listed here but absent there would only fail at run time.
    for mol in forward.EXTRA_MOLECULES:
        spec = _vfc.MOLECULES[mol]
        assert spec["molmass"] > 0
        assert spec["vulcan"]
    # an out-of-set RT molecule is refused loudly, naming the INJECTABLE
    # molecule_table route (never "edit a constant inside another package")
    with pytest.raises(ValueError, match="molecule_table"):
        forward.canonical_params(_p(extra_mols=["PH3"]))


def test_vm_mol_pinned_explicitly_and_zeroed_without_moldiff():
    # the tool must PIN the vm_mol scheme in the canonical params (cache-
    # keyed), never inherit the upstream YAML default; False = the validated
    # baseline chemistry
    cp = forward.canonical_params(_p())
    assert cp["use_vm_mol"] is False
    assert forward.canonical_params(_p(use_vm_mol=True))["use_vm_mol"] is True
    # inert when moldiff is off (engine gates use_vm on both): zeroed so it
    # cannot fragment the cache into two identical setups
    on = forward.canonical_params(_p(use_moldiff=False, use_vm_mol=True))
    off = forward.canonical_params(_p(use_moldiff=False, use_vm_mol=False))
    assert on["use_vm_mol"] is False
    assert forward.params_key(on) == forward.params_key(off)


def test_composition_structural_path_baseline_and_ranges():
    # CO_BASELINE must be the network cfg's C_H/O_H (~0.549), never the
    # FastChem EQ-init ratio (0.458), which only seeds the initial guess;
    # run_model additionally cross-checks the live cfg
    assert abs(forward.CO_BASELINE - 0.00295 / 0.00537) < 1e-12
    assert abs(forward.CO_BASELINE - 0.549) < 1e-3
    # Composition is ONE structural path -- co_ratio (absolute N_C/N_O)
    # and met_x_solar go straight into the cfg elemental abundances; there
    # are no differential composition knobs in the canonical params
    # The DEFAULT co_ratio is CO_DEFAULT (the baseline rounded onto the
    # widgets' 0.05 grid); CO_BASELINE stays the display baseline + cross-check
    cp = forward.canonical_params(_p())
    assert cp["co_ratio"] == forward.CO_DEFAULT == 0.55
    assert "dco" not in cp and "co_baseline" not in cp
    # High C/O inside the bound is the same path, with NO detection-only
    # restriction: FD Fisher rows are certified re-solves, valid at any
    # baseline inside the bound
    cp = forward.canonical_params(_p(co_ratio=0.8, met_x_solar=30.0,
                                     fisher_params=["lnZ", "dlnCO"]))
    assert cp["co_ratio"] == 0.8 and cp["met_x_solar"] == 30.0
    for bad in (dict(co_ratio=0.05), dict(co_ratio=2.5),
                dict(met_x_solar=0.05), dict(met_x_solar=150.0)):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(**bad))


def test_fisher_names_and_jac_method_matrix():
    # FD Jacobians are certified re-solves -- no photo-on tangent
    # regime, so FD Fisher works photo-off; unknown rows refuse loudly
    cp = forward.canonical_params(_p(use_photo=False, fisher_params=["lnZ"]))
    assert cp["fisher_params"] == ["lnZ"] and cp["use_photo"] is False
    with pytest.raises(ValueError):
        forward.canonical_params(_p(fisher_params=["lnFoo"]))
    with pytest.raises(ValueError):
        forward.canonical_params(_p(fisher_params=["Tint_cl"]))  # climate-only
    # jac_method is canonical -- certified FD by default, unknown refused
    assert forward.canonical_params(
        _p(fisher_params=["lnKzz"]))["jac_method"] == "fd"
    with pytest.raises(ValueError):
        forward.canonical_params(_p(fisher_params=["lnKzz"],
                                    jac_method="magic"))
    # the warm-jvp AD rows are validated only photo-on; they cover EVERY
    # requested row, composition directions included (the C-rich b_z corner
    # refuses at run time), so comp-only selections KEEP 'ad'
    cp = forward.canonical_params(_p(fisher_params=["lnZ", "dlnCO"],
                                     jac_method="ad"))
    assert cp["jac_method"] == "ad"
    with pytest.raises(ValueError):
        forward.canonical_params(_p(fisher_params=["lnKzz"], jac_method="ad",
                                    use_photo=False))
    # with no Jacobian requested the knob is inert -- normalized to 'fd' so
    # it cannot fragment the cache key, and photo-off is then fine
    assert forward.canonical_params(_p(jac_method="ad"))["jac_method"] == "fd"
    assert forward.canonical_params(
        _p(jac_method="ad", use_photo=False))["jac_method"] == "fd"


def test_rt_knobs_defaults_validation_and_cache_key():
    # two ExoJAX RT knobs are canonical (cache-keyed)
    cp = forward.canonical_params(_p())
    assert cp["rt_ptop_bar"] == 1.0e-7
    assert cp["rt_integration"] == "simpson"
    cp = forward.canonical_params(_p(rt_ptop_bar=1.0e-6,
                                     rt_integration="trapezoid"))
    assert (cp["rt_ptop_bar"], cp["rt_integration"]) == (1.0e-6, "trapezoid")
    for bad in (dict(rt_ptop_bar=1.0e-5), dict(rt_ptop_bar=1.0e-10),
                dict(rt_integration="euler")):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(**bad))
    # a LIVE RT knob must change the cache key (different physics)
    k0 = forward.params_key(_p())
    assert forward.params_key(_p(rt_ptop_bar=1.0e-8)) != k0
    assert forward.params_key(_p(rt_integration="trapezoid")) != k0


def test_chem_key_separates_chemistry_from_rt_only_edits():
    """The chemistry-level cache key is invariant under every stripped
    RT/observable-only key (an RT edit must reuse the solved column) and
    moves with chemistry-relevant parameters (a wrong hit would corrupt
    every downstream product). Every stripped key is a canonical key."""
    cp = forward.canonical_params(_p())
    assert set(forward.CHEM_IRRELEVANT_PARAMS) <= set(cp)
    k0 = forward.chem_key(_p())
    for kw in (dict(cloud_on=True, log_kappa_cloud=-1.0),
               dict(p_ref_bar=0.05),
               dict(wo_mols=["H2O"]),
               dict(fisher_params=["lnZ"], jac_method="ad"),
               dict(star_teff=5300.0),
               dict(rt_integration="trapezoid")):
        assert forward.chem_key(_p(**kw)) == k0, kw
        # the flat key must still see live RT knobs (pinned above); the
        # chem key must not
    # rt_ptop_bar is dual-use: the chemistry grid top follows it
    for kw in (dict(met_x_solar=5.0), dict(co_ratio=0.3),
               dict(p_btm_bar=50.0), dict(nz=110), dict(rt_ptop_bar=1.0e-8)):
        assert forward.chem_key(_p(**kw)) != k0, kw


# --- WASP-39 b reference state: DO NOT let this drift ------------------------
# The REFERENCE configuration (tp_mode="file", the shipped evening-terminator
# table) is the one measured against the published JWST detection, and also
# the DEFAULT. The guard anchors to the EXPLICIT file-mode
# config so it protects the reference even if the default moves again.
# Re-measure against the literature before updating expected values.
W39B_REFERENCE = {
    "tp_mode": "file",                      # measured evening-terminator table
    "tp_file": "shipped",
    "tp_file_sha1": "1a4ce744e65205d8",     # exact profile bytes (T AND Kzz)
    "kzz_mode": "file",                     # mixing from the table, not a stand-in
    "kzz_const": 0.0,                       # inert once tabulated
    "met_x_solar": 10.0,                    # Tsai+2023 10x solar
    "co_ratio": 0.55,                       # CO_DEFAULT (maintainer):
                                            # the cfg C_H/O_H 0.549348 rounded
                                            # onto the 0.05 widget grid; the
                                            # 0.12% composition shift awaits
                                            # the SO2 re-measure noted below
    "use_photo": True,                      # SO2 is photochemical; non-negotiable
    "sl_angle_deg": 83.0,                   # Tsai+2023 terminator slant
    "use_vm_mol": False,                    # validated pre-flip baseline
    "nz": 100,
}


def test_wasp39b_reference_state_is_the_literature_validated_one():
    # tp_mode="file" is the ONLY explicit key: everything else must still
    # default to the validated values, or the reference run has drifted.
    cp = forward.canonical_params(dict(planet="wasp39b", tp_mode="file"))
    for key, want in W39B_REFERENCE.items():
        assert cp[key] == want, (
            f"WASP-39 b reference {key}: {cp[key]!r} != {want!r}. This "
            "changes the atmosphere behind the published-detection agreement "
            "-- re-measure G395H SO2 against Alderson+2023 / Tsai+2023 "
            "before updating W39B_REFERENCE.")


def test_network_semantics():
    """The kinetics-network contract, at the numpy level.

    (a) default sncho, carried in the canonical dict; unknown value refused.
    (b) ncho removes exactly the sulfur species from the RT sets: SO2 from
        the base set, H2S/CS2/OCS from the extras; an explicit sulfur extra
        or wo entry is refused, never dropped.
    (c) compatibility: refused with use_condense (the certified recipe
        condenses S8).
    (d) cache honesty: sncho and ncho are different keys.
    """
    # (a)
    cp = forward.canonical_params(_p())
    assert cp["network"] == "sncho"
    with pytest.raises(ValueError):
        forward.canonical_params(_p(network="chon"))
    # (b)
    cpn = forward.canonical_params(_p(network="ncho"))
    assert cpn["wo_mols"] == [m for m in forward.MOLECULES if m != "SO2"]
    assert forward.active_molecules(cpn) == cpn["wo_mols"]
    assert not set(forward.active_molecules(
        forward.canonical_params(
            _p(network="ncho",
               extra_mols=["HCN", "NH3"])))) & forward._S_MOLECULES
    with pytest.raises(ValueError):
        forward.canonical_params(_p(network="ncho", extra_mols=["H2S"]))
    with pytest.raises(ValueError):
        forward.canonical_params(_p(network="ncho", wo_mols=["SO2"]))
    # (c) condensation is sncho-only: ncho has no S8, and sncho2025's added
    #     C3/H2CS species are outside CONDEN_CFG's tuned conver_ignore list
    for net in ("ncho", "sncho2025"):
        with pytest.raises(ValueError, match="network='sncho'"):
            forward.canonical_params(_p(network=net, use_condense=True))
    # (d) every network is its own cache key, spectrum AND chemistry level
    keys = {n: forward.params_key(forward.canonical_params(_p(network=n)))
            for n in forward.NETWORKS}
    chem = {n: forward.chem_key(forward.canonical_params(_p(network=n)))
            for n in forward.NETWORKS}
    assert len(set(keys.values())) == len(set(chem.values())) == len(forward.NETWORKS)
    # (e) sncho2025 keeps sulfur, so it offers the same molecules as sncho
    assert (forward.active_molecules(forward.canonical_params(_p(network="sncho2025")))
            == forward.active_molecules(cp))


@pytest.mark.parametrize("network", sorted(forward.CO_MAX))
def test_co_bound_is_per_network_and_shared_by_every_surface(network):
    """One inclusive C/O bound per network, and the API, the GUI widget and
    share_config all read it -- the three used to disagree (API < 1.0, widget
    and share file <= 0.95), so an API run at C/O 0.97 could not be restored.

    The permissive bounds are the highest value SAMPLED in the 2026-09-02
    survey, not a measured ceiling: the convergence certificate is what
    refuses an individual case that does not converge.
    """
    import ast
    import importlib.util
    from pathlib import Path

    from jwst_tool import share_config

    lo, hi = forward.co_bounds(network)
    assert lo == forward.CO_MIN
    # both ends inclusive, and nothing past the top
    for co in (lo, hi):
        assert forward.canonical_params(
            _p(network=network, co_ratio=co))["co_ratio"] == co
    for co in (lo * 0.5, hi * 1.01):
        with pytest.raises(ValueError, match="outside"):
            forward.canonical_params(_p(network=network, co_ratio=co))
    # share_config refuses exactly what the API refuses, for this network
    key = "n0_{}".format
    state = {key("network"): network, key("co"): hi * 1.01}
    with pytest.raises(ValueError, match="co_ratio"):
        share_config._check_widget_ranges(state, key, key, "transmission")
    state[key("co")] = hi
    share_config._check_widget_ranges(state, key, key, "transmission")
    # the GUI widget's ceiling is forward.CO_MAX[<network>], not a literal
    src = Path(importlib.util.find_spec("jwst_tool.app").origin).read_text()
    co_calls = [n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "number_input"
                and any(kw.arg == "key" and isinstance(kw.value, ast.Call)
                        and kw.value.args
                        and getattr(kw.value.args[0], "value", None) == "co"
                        for kw in n.keywords)]
    assert len(co_calls) == 1
    assert [ast.unparse(a) for a in co_calls[0].args[1:3]] == [
        "forward.CO_MIN", "_co_max"]
    assert "forward.CO_MAX[st.session_state.get(K('network'), 'sncho')]" in [
        ast.unparse(n.value) for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "_co_max" for t in n.targets)]


def test_wo_mols_end_to_end_semantics():
    """The leave-one-out contract, end to end at the numpy level.

    (a) canonical form: default = every RT molecule in fold order; a subset
        canonicalizes to fold order deduped; unknown molecules refuse; [] is
        the constrain goal's valid empty set.
    (b) cache honesty: [] / subset / default are three different keys -- a
        constrain run must never serve a detect run's cache slot.
    (c) the progress contract mirrors the real solve sequence: no removed-
        molecule stage for [], exactly one batch stage otherwise.
    (d) consumers index depth_wo by the model's wo_mols, never by mols, and
        refuse loudly when the target has no removed spectrum.
    (e) the GUI asks for the TARGET only on detect and [] on constrain
        (the score reads one row, and the all-molecule block dominates a cold
        run) -- the all-molecule batch stays reachable through the API's
        wo_mols=None default.
    """
    # (a) canonicalization
    cp = forward.canonical_params(_p())
    assert cp["wo_mols"] == forward.MOLECULES
    cp = forward.canonical_params(_p(extra_mols=["HCN", "C2H2"]))
    assert cp["wo_mols"] == forward.MOLECULES + ["C2H2", "HCN"]
    cp = forward.canonical_params(_p(wo_mols=["SO2", "H2O", "SO2"]))
    assert cp["wo_mols"] == ["H2O", "SO2"]          # fold order, deduped
    assert forward.canonical_params(_p(wo_mols=[]))["wo_mols"] == []
    with pytest.raises(ValueError):
        forward.canonical_params(_p(wo_mols=["HCN"]))   # not in the RT set
    # a saved canonical payload round-trips unchanged (share_config path)
    assert forward.canonical_params(cp)["wo_mols"] == cp["wo_mols"]
    # (b) cache keys
    k_all = forward.params_key(_p())
    k_none = forward.params_key(_p(wo_mols=[]))
    k_sub = forward.params_key(_p(wo_mols=["H2O"]))
    assert len({k_all, k_none, k_sub}) == 3
    assert forward.params_key(_p(wo_mols=forward.MOLECULES)) == k_all
    # (c) progress stages mirror the solve sequence
    for kw, n_wo_stages in ((dict(), 1), (dict(wo_mols=[]), 0),
                            (dict(science_mode="emission",
                                  tp_mode="guillot", Tirr=1560.0), 1),
                            (dict(science_mode="emission", wo_mols=[],
                                  tp_mode="guillot", Tirr=1560.0), 0)):
        lines = []
        advance, _fin = forward._make_progress(
            forward.canonical_params(_p(**kw)), lambda s, _l=lines: _l.append(s))
        while True:
            try:
                advance()
            except IndexError:
                break
        assert sum("removed-molecule" in s for s in lines) == n_wo_stages, kw
    # (d) consumers follow wo_mols alignment
    from jwst_tool import detect
    n = 8
    order = np.arange(n)
    model = {"wo_mols": np.array(["CO2"], dtype="U8"),
             "depth_wo": np.arange(n, dtype=float)[None, :] * 1e-3}
    mols = ["H2O", "CO2", "CO"]
    got = detect._removed_spectrum(model, mols, "CO2", order)
    assert np.array_equal(got, model["depth_wo"][0])     # NOT mols.index -> 1
    assert detect._removed_spectrum(model, mols, None, order) is None
    with pytest.raises(ValueError, match="wo_mols"):
        detect._removed_spectrum(model, mols, "CO", order)   # constrain cache
    with pytest.raises(ValueError, match="extra_mols"):
        detect._removed_spectrum(model, mols, "HCN", order)  # not in RT set
    with pytest.raises(KeyError):
        detect._removed_spectrum({"depth_wo": model["depth_wo"]},
                                 mols, "CO2", order)     # malformed payload
    # (e) the GUI wiring: target-only on detect, [] on constrain
    from pathlib import Path
    app_src = (Path(__file__).resolve().parents[2] / "src" / "jwst_tool"
               / "app.py").read_text()
    assert 'wo_mols=([target_mol] if goal == "detect" else [])' in app_src, \
        "the GUI must ask for the target-only removed spectrum on detect"


def test_wasp39b_reference_cache_key_and_table_bytes_are_stable():
    # The key hashes every canonical parameter: if ANY default feeding the
    # reference run changes, this trips even when the pins above still pass.
    # Re-pinning this key is a _VERSION bump: state what moved the spectrum
    # in notes.md, which carries the per-version history of both.
    # NOT RE-MEASURED at v30/v31, required before quoting this key as a
    # science result: the default-geometry median depth (19,712 ppm at
    # v28-v29) and the G395H SO2 significance (2.89 at v27, BELOW the
    # published 4.5-4.8 -- that gap is real and open). Both need a full
    # run; SO2 also needs the pandeia backend. Full history: notes.md.
    # v42 re-pin: yconv_min joined the canonical key set at its cfg default
    # 0.1, so the key moved with the payload while the physics did not.
    assert forward.params_key(forward.canonical_params(
        dict(planet="wasp39b", tp_mode="file"))) == "bb46d19dceec08f9"
    # ... and the bare DEFAULT run is that same atmosphere
    assert forward.params_key(forward.canonical_params(
        dict(planet="wasp39b"))) == "bb46d19dceec08f9"
    # the sha1 pin is only meaningful re-derived from the file the run
    # actually reads -- this catches the table itself being swapped
    path = forward._shipped_tp_file("wasp39b")
    assert path.name == "atm_W39b_evening_TP_Kzz.txt"
    tab = forward._read_tp_table(path)
    assert tab["Kzz"] is not None, "the reference table must carry its Kzz column"
    import hashlib
    assert hashlib.sha1(path.read_bytes()).hexdigest()[:16] == \
        W39B_REFERENCE["tp_file_sha1"]


# --- tp_mode="file" ---------------------------------------------------------

def test_file_mode_content_addressing_hygiene_and_bad_tables(tmp_path):
    p1 = _table(tmp_path, name="a.txt")
    p2 = _table(tmp_path, name="b.txt")               # same content, other path
    p3 = _table(tmp_path, name="c.txt", tmax=1500.0)  # different content
    cp1 = forward.canonical_params(_pf(p1))
    assert cp1["tp_file_sha1"] and len(cp1["tp_file_sha1"]) == 16
    assert forward.params_key(_pf(p1)) == forward.params_key(_pf(p2))
    assert forward.params_key(_pf(p1)) != forward.params_key(_pf(p3))
    # a tabulated profile has NO T-P Fisher rows
    assert forward.TP_PARAM_NAMES["file"] == []
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
    nocol.write_text("#(dyne/cm2) (K)\nPress\tT\n1e6\t1000\n1e5\t900\n"
                     "1e4\t800\n1e3\t700\n")
    for path in (_table(tmp_path, scramble=True, name="scram.txt"),
                 _table(tmp_path, tmax=3300.0, name="hot.txt"),
                 _table(tmp_path, rows=3, name="short.txt"),
                 nocol, tmp_path / "missing.txt"):
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
    cp = forward.canonical_params(_p(kzz_mode="Pfunc", kzz_kmax=1.0e5,
                                     kzz_plev=0.1))
    assert cp["kzz_kmax"] == 1.0e5 and cp["kzz_plev"] == 0.1
    assert cp["kzz_const"] == 0.0 and cp["kzz_kdeep"] == 0.0
    cp = forward.canonical_params(_p(kzz_mode="JM16", kzz_kdeep=1.0e6))
    assert cp["kzz_kdeep"] == 1.0e6
    assert cp["kzz_const"] == cp["kzz_kmax"] == cp["kzz_plev"] == 0.0
    for bad in (dict(kzz_mode="Pfunc", kzz_kmax=1.0e15, kzz_plev=0.1),
                dict(kzz_mode="Pfunc", kzz_kmax=1.0e5, kzz_plev=1.0e5),
                dict(kzz_mode="JM16", kzz_kdeep=1.0),
                dict(kzz_mode="scale")):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(**bad))


# --- removed boundary fluxes / escape / settling -----------------------------

def test_boundary_flux_escape_and_settling_are_removed():
    """Cache-neutral removal: the four keys stay in the canonical payload
    pinned off, so no cached spectrum's key moved, and any enabling value
    is refused instead of silently ignored."""
    cp0 = forward.canonical_params(_p())
    assert cp0["use_settling"] is False and cp0["diff_esc"] == []
    assert cp0["top_flux"] == [] and cp0["bot_flux"] == []
    for enabling in (dict(use_settling=True),
                     dict(diff_esc=["H2"]),
                     dict(top_flux=[["H2O", 1.0e8]]),
                     dict(bot_flux=[["SO2", 1.0e9, 0.1]])):
        with pytest.raises(ValueError, match="removed from this tool"):
            forward.canonical_params(_p(**enabling))


# --- cloud deck -------------------------------------------------------------

def test_cloud_fisher_rows_require_their_deck(tmp_path):
    cp = forward.canonical_params(_p(cloud_on=True,
                                     fisher_params=["lnZ",
                                                    "log_kappa_cloud",
                                                    "alpha_cloud"]))
    assert set(cp["fisher_params"]) == {"alpha_cloud", "lnZ",
                                        "log_kappa_cloud"}
    with pytest.raises(ValueError):
        forward.canonical_params(_p(fisher_params=["log_kappa_cloud"]))
    # the deck stays freeable in file mode (no parametric T-P required)
    cp3 = forward.canonical_params(_pf(
        _table(tmp_path), cloud_on=True,
        fisher_params=["lnZ", "log_kappa_cloud"]))
    assert {"log_kappa_cloud"} <= set(cp3["fisher_params"])


def test_every_freeable_param_has_display_metadata():
    # Any parameter that can be freed in the Fisher forecast is also a valid
    # constraint goal and a Jacobian row, so it must carry a display label,
    # unit, symbol, and an FD step -- a missing entry KeyErrors the GUI/run.
    freeable = (set(forward.CHEM_PARAM_NAMES) | set(forward.CLOUD_FISHER_PARAMS)
                | {p for ns in forward.TP_PARAM_NAMES.values() for p in ns})
    for m in (fisher.PARAM_LABELS, fisher.PARAM_UNITS, fisher.PARAM_SYMBOLS,
              forward.FD_STEPS):
        assert not (freeable - set(m)), sorted(freeable - set(m))


# --- emission mode (science_mode) ------------------------------------------

def test_emission_mode_gating_star_params_and_hygiene(tmp_path):
    with pytest.raises(ValueError):
        forward.canonical_params(_p(science_mode="reflection"))
    cp = forward.canonical_params(_p(science_mode="emission",
                                     tp_mode="guillot", Tirr=1560.0))
    assert cp["science_mode"] == "emission"
    # star defaults come from the planet registry
    assert cp["star_teff"] == 5485.0 and cp["star_logg"] == 4.5
    # transmission zeroes the star identity (it is noise-side only there)
    cp_t = forward.canonical_params(_p())
    assert cp_t["star_teff"] == cp_t["star_logg"] == cp_t["star_feh"] == 0.0
    # rayleigh + chord scheme are transmission-only: normalized in emission
    cp_e = forward.canonical_params(_p(science_mode="emission",
                                       tp_mode="guillot", Tirr=1560.0,
                                       use_rayleigh=True,
                                       rt_integration="trapezoid"))
    assert cp_e["use_rayleigh"] is False
    assert cp_e["rt_integration"] == "simpson"
    with pytest.raises(ValueError):
        forward.canonical_params(_p(science_mode="emission",
                                    tp_mode="guillot", Tirr=1560.0,
                                    star_teff=10000.0))
    # emission works with a tabulated profile too
    cp_f = forward.canonical_params(_pf(_table(tmp_path),
                                        science_mode="emission"))
    assert cp_f["science_mode"] == "emission"
    # The default follows the STRUCTURE, not the geometry -- a measured
    # table caps the column at its own bottom (7.6 bar for the shipped
    # tables), parametric profiles get the round 10 bar. Emission needs no
    # deeper default: measured, 7.6 bar is optically thick (tau 30-370) in
    # every instrument window.
    assert cp_f["p_btm_bar"] == forward.P_BTM_FILE_BAR
    assert cp_t["p_btm_bar"] == forward.P_BTM_PARAMETRIC_BAR
    # a deeper column is still reachable, and still gated on the table covering
    with pytest.raises(ValueError):
        forward.canonical_params(_pf(_table(tmp_path),
                                     science_mode="emission", p_btm_bar=100.0))
    cp_d = forward.canonical_params(
        _pf(_table(tmp_path, name="deep.txt", p_lo_log10=8.1),
            science_mode="emission", p_btm_bar=100.0))
    assert cp_d["p_btm_bar"] == 100.0
    # the two geometries can never share a cache entry
    assert (forward.params_key(_p(science_mode="emission", tp_mode="guillot",
                                  Tirr=1560.0))
            != forward.params_key(_p(tp_mode="guillot", Tirr=1560.0)))


# --- composition FD stencils + the dayside emission default ----------------

def test_composition_fd_stencil_envelope():
    """FD Fisher rows solve the chemistry at every stencil point, so a
    baseline whose stencil leaves the validated range refuses (met=100 ->
    122x solar, co=0.12 -> C/O 0.098). The dlnCO stencil steps one-sided away
    from the network's C/O ceiling instead of across it. AD rows take no
    stencil and are exempt here (the dlnCO AD row has its own run-time margin
    gate)."""
    forward.canonical_params(_p(met_x_solar=80.0, fisher_params=["lnZ"]))
    for kw, fp in (({"met_x_solar": 100.0}, "lnZ"),
                   ({"met_x_solar": 0.1}, "lnZ"),
                   ({"co_ratio": 0.12}, "dlnCO")):
        with pytest.raises(ValueError, match="stencil"):
            forward.canonical_params(_p(fisher_params=[fp], **kw))
    h0 = forward.FD_STEPS["dlnCO"]
    hi = forward.CO_MAX["sncho"]
    assert forward.fd_stencil("dlnCO", 0.55, hi) == ((1, -1, 2, -2), h0)
    assert forward.fd_stencil("dlnCO", 0.89, hi) == ((-1, -2, -4), h0 / 2)
    assert forward.fd_stencil("lnZ", 0.89, hi) == ((1, -1, 2, -2),
                                                   forward.FD_STEPS["lnZ"])
    # the flip follows the network's ceiling, not a literal C/O = 1
    hi25 = forward.CO_MAX["sncho2025"]
    assert forward.fd_stencil("dlnCO", 2.0, hi25) == ((1, -1, 2, -2), h0)
    assert forward.fd_stencil("dlnCO", 9.0, hi25) == ((-1, -2, -4), h0 / 2)
    for co in (0.89, 0.95):
        forward.canonical_params(_p(co_ratio=co, fisher_params=["dlnCO"]))
    # both schemes' Richardson rows are exact on a cubic (sign and
    # coefficients of the one-sided stencil included); the one-sided row is
    # third order -- on a quartic its error is -f''''(x) h^3 / 3 exactly, so
    # halving h divides it by 8 -- while the central row stays exact
    x0, h = 0.3, 0.1
    f = lambda x: x ** 3 - 2.0 * x
    g = lambda x: x ** 4
    for offs in ((1, -1, 2, -2), (-1, -2, -4)):
        j1, j2 = forward.fd_estimates(offs, {s: f(x0 + s * h) for s in offs},
                                      f(x0), h)
        assert abs((4.0 * j1 - j2) / 3.0 - (3.0 * x0 ** 2 - 2.0)) < 1e-12
    err = {}
    for offs, hh in (((-1, -2, -4), h), ((-1, -2, -4), h / 2),
                     ((1, -1, 2, -2), h)):
        j1, j2 = forward.fd_estimates(offs, {s: g(x0 + s * hh) for s in offs},
                                      g(x0), hh)
        err[offs, hh] = abs((4.0 * j1 - j2) / 3.0 - 4.0 * x0 ** 3)
    assert abs(err[(-1, -2, -4), h] / err[(-1, -2, -4), h / 2] - 8.0) < 1e-6
    assert err[(1, -1, 2, -2), h] < 1e-12
    # no stencil under AD; and without fisher_params the value is legal
    forward.canonical_params(_p(met_x_solar=100.0, fisher_params=["lnZ"],
                                jac_method="ad"))
    forward.canonical_params(_p(met_x_solar=100.0))


def test_emission_defaults_to_the_dayside_temperature():
    """An eclipse sees the DAYSIDE; a published equilibrium temperature assumes
    full redistribution, which is a planet-average profile.

    This was the single largest error in the emission path. Measured on
    HD 189733 b against the published MIRI/LRS eclipse spectrum (Inglis et al.
    2024), switching only this default moved the eclipse depth from 1.35x LOW
    to within 3% of the data, and chi2 per point from 934 to 31.
    """
    for key in planets.PLANETS:
        t = forward.canonical_params(
            dict(planet=key, tp_mode="guillot"))["Tirr"]
        e = forward.canonical_params(
            dict(planet=key, tp_mode="guillot", science_mode="emission"))["Tirr"]
        # 2**0.25 in temperature, on the 10 K rounding grid the widget uses
        assert e > t, key
        assert e == pytest.approx(t * 2 ** 0.25, abs=10.0), (key, t, e)
    # an explicit T_irr still wins in both geometries
    for mode in ("transmission", "emission"):
        cp = forward.canonical_params(dict(planet="wasp39b", tp_mode="guillot",
                                           science_mode=mode, Tirr=1234.0))
        assert cp["Tirr"] == 1234.0


# --- unknown-key rejection -------------------------------------------------

def test_unknown_keys_refuse_with_a_hint_and_output_round_trips():
    # The bug this pins: a validation driver passed {"mode": "emission"}, the
    # key was silently ignored, and a TRANSMISSION spectrum was scored against
    # eclipse data (chi2/N ~ 5e5). Unknown keys must refuse, never drop.
    with pytest.raises(ValueError, match="science_mode"):
        forward.canonical_params(_p(mode="emission"))
    with pytest.raises(ValueError):
        forward.canonical_params(_p(totally_made_up=1))
    # ... while share_config validates a SAVED canonical payload by feeding
    # it back in, so every output key (echo fields included) is accepted
    cp = forward.canonical_params(dict(planet="wasp39b"))
    assert set(cp) <= forward._KNOWN_PARAM_KEYS
    cp2 = forward.canonical_params(dict(cp))
    assert cp2 == cp


def test_param_keys_read_matches_the_source():
    # _PARAM_KEYS_READ is re-derived from the AST of canonical_params and the
    # helpers it calls, so adding a parameter without updating the allowlist
    # fails here instead of silently rejecting the new key at run time.
    import ast
    import inspect

    src = inspect.getsource(forward)
    tree = ast.parse(src)
    funcs = {"canonical_params", "_resolve_tp_file", "default_p_btm_bar",
             "_default_tp_mode"}
    keys = set()

    class V(ast.NodeVisitor):
        def visit_Call(self, node):
            f = node.func
            if (isinstance(f, ast.Attribute) and f.attr == "get"
                    and isinstance(f.value, ast.Name) and f.value.id == "params"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)
            self.generic_visit(node)

        def visit_Subscript(self, node):
            if (isinstance(node.value, ast.Name) and node.value.id == "params"
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                keys.add(node.slice.value)
            self.generic_visit(node)

        def visit_Compare(self, node):
            if (len(node.comparators) == 1 and isinstance(node.ops[0], ast.In)
                    and isinstance(node.comparators[0], ast.Name)
                    and node.comparators[0].id == "params"
                    and isinstance(node.left, ast.Constant)
                    and isinstance(node.left.value, str)):
                keys.add(node.left.value)
            self.generic_visit(node)

    scanned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in funcs:
            scanned.add(node.name)
            V().visit(node)
    assert scanned == funcs, f"missing helper(s): {funcs - scanned}"
    assert keys == set(forward._PARAM_KEYS_READ)


# --- ExoMolOP no-table gate ------------------------------------------------


def test_molecule_list_invariants():
    """Menu/default consistency, none of it enforced by the code itself."""
    import re
    # the preselected set must be a selectable subset with tables
    assert set(forward.EXTRA_MOLECULES_DEFAULT) <= set(forward.EXTRA_MOLECULES)
    assert not (set(forward.EXTRA_MOLECULES_DEFAULT)
                & forward._NO_EXOMOLOP_TABLE)
    # _S_MOLECULES must catch EVERY sulfur-bearing species in either list, or
    # network="ncho" offers one the sulfur-free network cannot produce.
    # `S(?![a-z])` so SiO-style names never read as sulfur.
    s_bearing = {m for m in list(forward.MOLECULES) + forward.EXTRA_MOLECULES
                 if re.search(r"S(?![a-z])", m)}
    assert s_bearing == set(forward._S_MOLECULES), (
        f"_S_MOLECULES misses {sorted(s_bearing - set(forward._S_MOLECULES))} "
        f"/ over-lists {sorted(set(forward._S_MOLECULES) - s_bearing)}")
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
            "forward._NO_EXOMOLOP_TABLE and re-enable it in the GUI default")
    missing = (set(forward.MOLECULES) | set(forward.EXTRA_MOLECULES)) \
        - forward._NO_EXOMOLOP_TABLE - have
    assert not missing, (
        f"species {sorted(missing)} have no installed ExoMolOP table; fetch "
        "them or add them to forward._NO_EXOMOLOP_TABLE with a reason")
