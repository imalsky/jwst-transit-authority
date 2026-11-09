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
from param_helpers import W39B_REFERENCE, _p, _pf, _table

from jwst_tool import forward, model_cache, model_defs, picaso_engine, planets, run_params, tp_tables


def test_condensation_detection_only_with_full_refusal_matrix():
    # use_condense is canonical (default False); a detection-only
    # condensing run (photo + moldiff on, no Fisher) is ACCEPTED. The pinned
    # reservoir is not a reproducible function of the parameters, so
    # condensation + Fisher refuses under EVERY jac_method (FD included);
    # photo-off has no certifiable steady state; the growth term IS the
    # molecular-diffusion coefficient, so moldiff-off refuses too. Condensation
    # runs on sncho only (test_network_semantics), so every case names it.
    assert forward.canonical_params(_p())["use_condense"] is False
    assert forward.canonical_params(_p(use_condense=True, network="sncho"))["use_condense"] is True
    for jm in ("fd", "ad"):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(use_condense=True, network="sncho", jac_method=jm, fisher_params=["lnZ"]))
    with pytest.raises(ValueError):
        forward.canonical_params(_p(use_condense=True, network="sncho", use_photo=False))
    with pytest.raises(ValueError):
        forward.canonical_params(_p(use_condense=True, network="sncho", use_moldiff=False))


def test_shipped_tables_gate_defaults_and_are_never_substituted():
    """A planet whose bundled measured T-P/Kzz
    table is VERIFIED end-to-end defaults to it; every other planet defaults
    to analytic Guillot + constant Kzz and carries a written tp_table_note.
    Each planet resolves to ITS OWN table; a planet without one refuses
    loudly rather than borrowing another's atmosphere.
    """
    seen = {}
    for key in planets.PLANETS:
        cp = forward.canonical_params(dict(planet=key))
        if tp_tables.shipped_tp_table_is_default(key):
            assert cp["tp_mode"] == "file", key
            assert cp["tp_file"] == model_defs.TP_FILE_SHIPPED, key
            assert cp["tp_file_sha1"], key  # content-addressed
            assert cp["kzz_mode"] == "file", key  # table carries Kzz
            assert cp["kzz_const"] == 0.0, key  # inert once tabulated
        else:
            assert cp["tp_mode"] == "guillot", key
            assert cp["kzz_mode"] == "const", key
            assert planets.PLANETS[key]["tp_table_note"], key
        name = tp_tables.shipped_tp_table_name(key)
        if name:
            seen[key] = name
        else:
            with pytest.raises(ValueError):
                forward.canonical_params(dict(planet=key, tp_mode="file"))
    assert len(set(seen.values())) == len(seen)  # no shared table
    # the verified set is exactly WASP-39 b; HD 189733 b ships a good
    # profile the solver does NOT certify at defaults, so it must stay
    # selectable-but-not-default -- choosing it explicitly still resolves
    assert [k for k in planets.PLANETS if tp_tables.shipped_tp_table_is_default(k)] == ["wasp39b"]
    assert tp_tables.shipped_tp_table_name("hd189733b")
    # its table carries a 6000 K thermosphere above 0.1 dyn/cm^2, outside the
    # modelable window, so file mode needs a model top at or below 1e-7 bar
    cp = forward.canonical_params(dict(planet="hd189733b", tp_mode="file", rt_ptop_bar=1.0e-7))
    assert cp["tp_mode"] == "file" and cp["kzz_mode"] == "file"
    with pytest.raises(ValueError):
        forward.canonical_params(dict(planet="hd189733b", tp_mode="file", rt_ptop_bar=1.0e-8))
    assert run_params.default_tp_mode(dict(planet="wasp39b")) == "file"
    # a retired (GCM) mode refuses, never substitutes a profile
    with pytest.raises(ValueError):
        forward.canonical_params(dict(planet="wasp39b", tp_mode="baseline"))


def test_gray_cloud_pressure_and_opacity_are_one_knob():
    """The gray cloud's two coordinates differ by a constant in log space, so
    a Fisher width in dex means the same in either.

    The anchor is vulcan-forward's own closed form, which its
    tests/test_cloud_deck.py gates the RT against (Heng & Kitzmann 2017):
    uniformly mixed kappa gives slant tau = kappa (P/g) sqrt(2 pi Rp/H), so
    WASP-39 b at kappa = 0.1 cm^2/g and 896.6 K blocks down to 1.53e-4 bar.
    """
    g, rp = 422.0, 1.279
    assert planets.gray_cloud_log_kappa(1.53e-4, g, rp, 896.6) == pytest.approx(-1.0, abs=2.2e-3)

    teq = planets.system_teq(5485.0, 0.932, 0.04828)
    levels = (-6.0, -3.0, 0.0, 1.0)
    kappas = [planets.gray_cloud_log_kappa(10.0**lp, g, rp, teq) for lp in levels]
    # one decade of pressure IS one decade of opacity, in both directions
    assert kappas[0] - kappas[1] == pytest.approx(3.0)
    for lp, lk in zip(levels, kappas, strict=True):
        assert lk + lp == pytest.approx(kappas[0] + levels[0])
    # a 1-bar deck needs log10 kappa below -4, a range the pressure-entered
    # gray mode reaches directly
    assert planets.gray_cloud_log_kappa(1.0, g, rp, teq) < -4.0
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="cloud top pressure"):
            planets.gray_cloud_log_kappa(bad, g, rp, teq)


def test_guillot_default_tirr_follows_planet_and_custom_system():
    """T_irr default = sqrt(2) * T_eq of the SELECTED planet on the GUI's
    20 K step grid (the widget step); a bare constant makes API and GUI
    defaults diverge. The custom planet derives T_eq from the entered star
    and orbit (T_eq = Teff sqrt(Rstar/2a)) -- as does every registry planet, so
    a refreshed stellar parameter can never leave a stale T_eq behind.
    """
    for key, p in planets.PLANETS.items():
        teq = planets.system_teq(p["star"]["teff"], p["rstar_rsun"], p["orbit_au"])
        expect = min(max(round(teq * math.sqrt(2.0) / 20.0) * 20.0, 800.0), 2500.0)
        cp = forward.canonical_params(dict(planet=key, tp_mode="guillot"))
        assert cp["Tirr"] == expect, key
    # per-planet values, pinned explicitly
    assert model_defs.default_tirr("wasp39b") == 1640.0
    assert model_defs.default_tirr("hd209458b") == 2040.0
    sys_cool = dict(star_teff=3300.0, rstar_rsun=0.30, orbit_au=0.05)
    teq = planets.system_teq(**sys_cool)  # ~350 K
    expect = min(max(round(teq * math.sqrt(2.0) / 20.0) * 20.0, 800.0), 2500.0)
    cp = forward.canonical_params(dict(planet="custom", tp_mode="guillot", **sys_cool))
    assert cp["Tirr"] == expect
    # a different entered system must move the default
    cp_hot = forward.canonical_params(
        dict(planet="custom", tp_mode="guillot", star_teff=6100.0, rstar_rsun=1.2, orbit_au=0.03)
    )
    assert cp_hot["Tirr"] != cp["Tirr"]
    # an explicit Tirr always wins over the derived default
    cp_exp = forward.canonical_params(dict(planet="custom", tp_mode="guillot", Tirr=1234.0, **sys_cool))
    assert cp_exp["Tirr"] == 1234.0


def test_tp_table_gates_are_grid_scoped_and_require_the_bottom(tmp_path):
    """Two table gates. (1) The T-window gate judges the profile the ENGINE
    evaluates (re-gridded onto CHEM_P_SPAN_DYN), not every raw row: a table
    extending past the grid (a hot thermosphere) is fine if the in-grid part
    is modelable. (2) A table stopping above the chemistry-grid bottom is
    REFUSED: the engine would clamp-extend the last tabulated T
    isothermally over the quench region.
    """

    def _write(name, T):
        p = tmp_path / name
        p.write_text(
            "#(dyne/cm2) (K)\nPressure Temp\n" + "\n".join(f"{pv:.6e} {t:.2f}" for pv, t in zip(P, T, strict=True))
        )
        return p

    lo, hi = model_defs.CHEM_P_SPAN_DYN
    # extends both ways past the grid, in-grid profile is a modelable 900 K
    P = np.array([lo / 100, lo, hi, hi * 100])
    T = np.array([5000.0, 900.0, 900.0, 5000.0])  # out of window only outside
    assert tp_tables.read_tp_table(_write("ok.txt", T))["T"].size == 4
    # in-grid profile itself breaches the ceiling -> refused
    T_bad = np.array([5000.0, 2990.0, 900.0, 5000.0])
    with pytest.raises(ValueError):
        tp_tables.read_tp_table(_write("bad.txt", T_bad))
    # a 1 bar bottom is too shallow; the standard fixture (past P_b) passes
    Ps = np.logspace(6.0, -1.0, 8)
    shallow = tmp_path / "shallow.txt"
    shallow.write_text(
        "#(dyne/cm2) (K)\nPressure\tTemp\n"
        + "\n".join(f"{Ps[i]:.6e}\t{1200.0 - 40.0 * i:.1f}" for i in range(8))
        + "\n"
    )
    with pytest.raises(ValueError):
        forward.canonical_params(_pf(shallow))
    forward.canonical_params(_pf(_table(tmp_path)))


def test_resolution_knobs_defaults_ranges_and_refusals():
    # nz/yconv are explicit and the RT layer count is derived, not
    # cache-keyed; the spectral grid is the k-tables' own (no knob)
    cp = forward.canonical_params(_p())
    assert cp["nz"] == model_defs.NZ_DEFAULT == 100
    assert cp["yconv_cri"] == model_defs.YCONV_DEFAULT == 1.0e-2
    assert "art_nlayer" not in cp  # locked to nz in run_model, not cache-keyed
    # explicit in-range values accepted; the yconv ladder reaches its 1e-4
    # floor
    cp = forward.canonical_params(_p(nz=150, yconv_cri=1.0e-4))
    assert (cp["nz"], cp["yconv_cri"]) == (150, 1.0e-4)
    for bad in (dict(nz=40), dict(nz=200), dict(yconv_cri=1.0), dict(yconv_cri=5.0e-5)):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(**bad))
    # the removed line-by-line / Mie keys refuse BY NAME (an old config says
    # what was dropped), never fall through to the generic unknown-key text
    for key in sorted(run_params._REMOVED_PARAM_KEYS):
        with pytest.raises(ValueError, match="removed in 0.48.0"):
            forward.canonical_params(_p(**{key: 1}))


def test_extra_molecules_resolve_in_engine_and_unknown_refused():
    # The extras cover photochemical sulfur and the CH4-photolysis
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
    # Composition is ONE structural path -- co_ratio (absolute N_C/N_O)
    # and met_x_solar go straight into the cfg elemental abundances; there
    # are no differential composition knobs in the canonical params
    # The DEFAULT co_ratio is CO_DEFAULT (the cfg's C_H/O_H, 0.549, rounded
    # onto the widgets' 0.05 grid)
    cp = forward.canonical_params(_p())
    assert cp["co_ratio"] == model_defs.CO_DEFAULT == 0.55
    assert "dco" not in cp and "co_baseline" not in cp
    # High C/O inside the bound is the same path, with NO detection-only
    # restriction: FD Fisher rows are certified re-solves, valid at any
    # baseline inside the bound
    cp = forward.canonical_params(_p(co_ratio=0.8, met_x_solar=24.0, fisher_params=["lnZ", "dlnCO"]))
    assert cp["co_ratio"] == 0.8 and cp["met_x_solar"] == 24.0
    for bad in (
        dict(co_ratio=0.05),
        dict(co_ratio=2.5, network="sncho"),
        dict(co_ratio=12.0),
        dict(met_x_solar=0.05),
        dict(met_x_solar=150.0),
    ):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(**bad))


def test_fisher_names_and_jac_method_matrix():
    # FD Jacobians are certified re-solves -- no photo-on tangent
    # regime, so FD Fisher works photo-off; unknown rows refuse loudly
    cp = forward.canonical_params(_p(use_photo=False, fisher_params=["lnZ"]))
    assert cp["fisher_params"] == ["lnZ"] and cp["use_photo"] is False
    with pytest.raises(ValueError):
        forward.canonical_params(_p(fisher_params=["lnFoo"]))
    # jac_method is canonical -- certified FD by default, unknown refused
    assert forward.canonical_params(_p(fisher_params=["lnKzz"]))["jac_method"] == "fd"
    with pytest.raises(ValueError):
        forward.canonical_params(_p(fisher_params=["lnKzz"], jac_method="magic"))
    # the warm-jvp AD rows are validated only photo-on; they cover the
    # CHEMISTRY rows, composition directions included (the C-rich b_z corner
    # refuses at run time), so comp-only selections KEEP 'ad'
    cp = forward.canonical_params(_p(fisher_params=["lnZ", "dlnCO"], jac_method="ad"))
    assert cp["jac_method"] == "ad"
    with pytest.raises(ValueError):
        forward.canonical_params(_p(fisher_params=["lnKzz"], jac_method="ad", use_photo=False))
    # ... but the photo-off refusal covers CHEMISTRY rows only: a cloud-deck
    # row is an RT-only jvp on the converged column, photolysis-independent,
    # so an all-RT-only request keeps 'ad' photo-off
    cp = forward.canonical_params(
        _p(fisher_params=["log_kappa_cloud"], jac_method="ad", use_photo=False, cloud_on=True)
    )
    assert cp["jac_method"] == "ad"
    assert cp["fisher_params"] == ["log_kappa_cloud"]
    # one chemistry row in the set is enough to refuse the whole request
    with pytest.raises(ValueError, match="chemistry rows"):
        forward.canonical_params(
            _p(fisher_params=["log_kappa_cloud", "lnZ"], jac_method="ad", use_photo=False, cloud_on=True)
        )
    # with no Jacobian requested the knob is inert -- normalized to 'fd' so
    # it cannot fragment the cache key, and photo-off is then fine
    assert forward.canonical_params(_p(jac_method="ad"))["jac_method"] == "fd"
    assert forward.canonical_params(_p(jac_method="ad", use_photo=False))["jac_method"] == "fd"


_PIC = dict(planet="wasp39b", chem_provider="picaso")


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(science_mode="emission", fisher_params=["lnZ", "dlnCO"]),
        dict(tp_mode="guillot", fisher_params=["Tirr"]),
        dict(tp_mode="guillot", picaso_quench=True, kzz_const=1.0e10),
        dict(tp_mode="file"),
    ],
)
def test_picaso_payloads_are_idempotent_and_carry_their_own_keys(kw):
    cp = forward.canonical_params({**_PIC, **kw})
    assert forward.canonical_params(cp) == cp
    assert cp["picaso_stack"] == picaso_engine.PICASO_STACK and cp["network"] == ""
    assert cp["use_photo"] is False and cp["kzz_mode"] == "const"
    climate = cp["tp_mode"] == "picaso_climate"
    assert (cp["rfacv"] > 0) == climate and (cp["star_teff"] > 0) == (climate or cp["science_mode"] == "emission")
    assert (cp["kzz_const"] > 0) == cp["picaso_quench"]
    # the PICASO keys never reach a VULCAN payload
    assert not {"rfacv", "picaso_quench", "picaso_stack"} & set(forward.canonical_params(_p()))


def test_picaso_lnz_stencil_turns_one_sided_at_the_metallicity_cap():
    # the 10x default keeps its lnZ row; VULCAN's stencil is unchanged
    assert model_defs.fd_stencil("lnZ", 10.0, 0.99, model_defs.PICASO_MET_RANGE[1])[0] == (-1, -2, -4)
    assert model_defs.fd_stencil("lnZ", 10.0, 0.99)[0] == (1, -1, 2, -2)
    forward.canonical_params({**_PIC, "fisher_params": ["lnZ"]})


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        (dict(use_photo=True), "VULCAN-JAX settings"),
        (dict(network="sncho"), "VULCAN-JAX settings"),
        (dict(kzz_mode="JM16"), "VULCAN-JAX settings"),
        (dict(kzz_x=2.0), "VULCAN-JAX settings"),
        (dict(jac_method="ad", fisher_params=["lnZ"]), "finite differences"),
        (dict(picaso_quench=True, fisher_params=["lnKzz"]), "staircase"),
        (dict(rfacv=0.0), "rfacv"),
        (dict(Tint=40.0), "Tint"),
        (dict(fisher_params=["Tint"]), "not a PICASO climate Fisher row"),
        (dict(p_btm_bar=200.0), "100 bar"),
        (dict(extra_mols=["N2O"]), "equilibrium species"),
        (dict(chem_provider="vulcan", tp_mode="picaso_climate"), "chem_provider='picaso'"),
        (dict(chem_provider="vulcan", rfacv=0.5), "PICASO settings"),
        (dict(chem_provider="exo"), "unknown chem_provider"),
        (dict(met_x_solar=20.0), "PICASO engine's range"),
        (dict(tp_mode="file", picaso_quench=True), "quench search needs it"),
    ],
)
def test_picaso_refusals(kw, msg):
    with pytest.raises(ValueError, match=msg):
        forward.canonical_params({**_PIC, **kw})


def test_rt_knobs_defaults_validation_and_cache_key():
    # two ExoJAX RT knobs are canonical (cache-keyed)
    cp = forward.canonical_params(_p())
    assert cp["rt_ptop_bar"] == 1.0e-7
    assert cp["rt_integration"] == "simpson"
    cp = forward.canonical_params(_p(rt_ptop_bar=1.0e-6, rt_integration="trapezoid"))
    assert (cp["rt_ptop_bar"], cp["rt_integration"]) == (1.0e-6, "trapezoid")
    for bad in (dict(rt_ptop_bar=1.0e-5), dict(rt_ptop_bar=1.0e-10), dict(rt_integration="euler")):
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
    every downstream product). Every stripped key is a canonical key.
    """
    cp = forward.canonical_params(_p())
    assert set(model_cache.CHEM_IRRELEVANT_PARAMS) <= set(cp)
    k0 = model_cache.chem_key(_p())
    for kw in (
        dict(cloud_on=True, log_kappa_cloud=-1.0),
        dict(wo_mols=["H2O"]),
        dict(fisher_params=["lnZ"], jac_method="ad"),
        dict(star_teff=5300.0),
        dict(rt_integration="trapezoid"),
    ):
        assert model_cache.chem_key(_p(**kw)) == k0, kw
        # the flat key must still see live RT knobs (pinned above); the
        # chem key must not
    # rt_ptop_bar is dual-use: the chemistry grid top follows it. p_ref_bar is
    # the pressure the catalogue radius/gravity apply at, so it sets the
    # chemistry's own radius anchor and moves the converged column.
    for kw in (
        dict(met_x_solar=5.0),
        dict(co_ratio=0.3),
        dict(p_btm_bar=50.0),
        dict(nz=110),
        dict(rt_ptop_bar=1.0e-8),
        dict(p_ref_bar=0.05),
    ):
        assert model_cache.chem_key(_p(**kw)) != k0, kw


def test_data_revision_keys_the_cache_without_moving_the_unset_keys(monkeypatch):
    """A corrected opacity table changes no model parameter, so the deployment's
    dataset revision is what tells the cache the science data moved. It keys
    BOTH caches (the mount is synced wholesale, so a chemistry miss is the safe
    direction), and an unset value keys nothing -- a local tree that only
    changes by hand keeps every key it already has, including the pinned
    WASP-39 b reference below.
    """
    monkeypatch.delenv("JWST_TOOL_DATA_REVISION", raising=False)
    k0, c0 = forward.params_key(_p()), model_cache.chem_key(_p())
    assert "data_revision" not in forward.canonical_params(_p())
    for blank in ("", "   "):  # whitespace is not a revision
        monkeypatch.setenv("JWST_TOOL_DATA_REVISION", blank)
        assert (forward.params_key(_p()), model_cache.chem_key(_p())) == (k0, c0)
    monkeypatch.setenv("JWST_TOOL_DATA_REVISION", "a" * 40)
    k1, c1 = forward.params_key(_p()), model_cache.chem_key(_p())
    assert k1 != k0 and c1 != c0
    monkeypatch.setenv("JWST_TOOL_DATA_REVISION", "b" * 40)
    assert forward.params_key(_p()) != k1 and model_cache.chem_key(_p()) != c1
    monkeypatch.setenv("JWST_TOOL_DATA_REVISION", "a" * 40)
    assert (forward.params_key(_p()), model_cache.chem_key(_p())) == (k1, c1)
    # ... and canonical_params must accept its OWN output: the key is appended
    # from the environment, so every re-canonicalization (an uncached worker
    # run, a share_config reload) feeds it straight back in; refusing it
    # raises "unknown parameter key(s) ['data_revision']" on the deployment
    # only.
    cp = forward.canonical_params(_p())
    assert cp["data_revision"] == "a" * 40
    assert forward.canonical_params(cp) == cp
    from jwst_tool import share_config

    share = share_config.build_share(cp, goal={}, observation={})
    assert share["canonical_params"]["data_revision"] == "a" * 40
    assert share_config.widget_state(share, "n0_{}".format)["n0_planet"] == "wasp39b"


def test_wasp39b_reference_state_is_the_literature_validated_one():
    # tp_mode="file" is the ONLY explicit key: everything else must still
    # default to the validated values, or the reference run has drifted.
    cp = forward.canonical_params(dict(planet="wasp39b", tp_mode="file"))
    for key, want in W39B_REFERENCE.items():
        assert cp[key] == want, (
            f"WASP-39 b reference {key}: {cp[key]!r} != {want!r}. This "
            "changes the atmosphere behind the published-detection agreement "
            "-- re-measure G395H SO2 against Alderson+2023 / Tsai+2023 "
            "before updating W39B_REFERENCE."
        )


def test_network_semantics():
    """The kinetics-network contract, at the numpy level.

    (a) default sncho2025, carried in the canonical dict; unknown value refused.
    (b) ncho removes exactly the sulfur species from the RT sets: SO2 from
        the base set, H2S/CS2/OCS from the extras; an explicit sulfur extra
        or wo entry is refused, never dropped.
    (c) compatibility: refused with use_condense (the certified recipe
        condenses S8).
    (d) cache honesty: sncho and ncho are different keys.
    """
    # (a)
    cp = forward.canonical_params(_p())
    assert cp["network"] == "sncho2025"
    with pytest.raises(ValueError):
        forward.canonical_params(_p(network="chon"))
    # (b)
    cpn = forward.canonical_params(_p(network="ncho"))
    assert cpn["wo_mols"] == [m for m in forward.MOLECULES if m != "SO2"]
    assert model_defs.active_molecules(cpn) == cpn["wo_mols"]
    assert (
        not set(model_defs.active_molecules(forward.canonical_params(_p(network="ncho", extra_mols=["HCN", "NH3"]))))
        & model_defs.S_MOLECULES
    )
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
    keys = {n: forward.params_key(forward.canonical_params(_p(network=n))) for n in model_defs.NETWORKS}
    chem = {n: model_cache.chem_key(forward.canonical_params(_p(network=n))) for n in model_defs.NETWORKS}
    assert len(set(keys.values())) == len(set(chem.values())) == len(model_defs.NETWORKS)
    # (e) sncho2025 keeps sulfur, so it offers the same molecules as sncho
    assert model_defs.active_molecules(forward.canonical_params(_p(network="sncho"))) == model_defs.active_molecules(cp)


@pytest.mark.parametrize("network", sorted(model_defs.CO_MAX))
@pytest.mark.parametrize("use_photo", [True, False])
def test_co_bound_is_per_network_and_shared_by_every_surface(network, use_photo):
    """One inclusive C/O bound per (network, photolysis), and the API and
    share_config both read it: two separate bounds can disagree (API < 1.0,
    share file <= 0.95), and then an API run at C/O 0.97 cannot be restored.

    Photolysis is the second gate axis: the pinned network's wall above
    C/O 1.02 is photochemical, and photolysis OFF all three networks converge
    at C/O 10 (121-229 CLI steps). The permissive bounds
    are the highest value SAMPLED, not a measured ceiling: the convergence
    certificate is what refuses an individual case.

    The GUI widget deliberately does NOT track the two axes -- it spans the
    whole range and lets canonical_params refuse -- because a ceiling that
    moved with them would clamp a stored C/O silently.
    """
    import ast
    import importlib.util
    from pathlib import Path

    lo, hi = model_defs.co_bounds(network, use_photo)
    assert lo == model_defs.CO_MIN
    # photolysis off lifts every network to the same permissive ceiling
    assert hi == (model_defs.CO_MAX[network] if use_photo else model_defs.CO_MAX_PHOTO_OFF)
    # both ends inclusive, and nothing past either
    _q = {"network": network, "use_photo": use_photo}
    for co in (lo, hi):
        assert forward.canonical_params(_p(co_ratio=co, **_q))["co_ratio"] == co
    with pytest.raises(ValueError, match="below the modelled minimum"):
        forward.canonical_params(_p(co_ratio=lo * 0.5, **_q))
    with pytest.raises(ValueError, match="too high for this setup") as e:
        forward.canonical_params(_p(co_ratio=hi * 1.01, **_q))
    # an escape is named only when one exists: at the permissive ceiling all
    # three are already in force, so suggesting them would be false advice
    assert ("Try the" in str(e.value)) is (hi < model_defs.CO_MAX_PHOTO_OFF)
    # the widget spans the WHOLE range, so nothing silently clamps a stored
    # value when the network or the photolysis switch moves
    src = Path(importlib.util.find_spec("jwst_tool.gui_atmosphere").origin).read_text()
    co_calls = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "number_input"
        and any(
            kw.arg == "key"
            and isinstance(kw.value, ast.Call)
            and kw.value.args
            and getattr(kw.value.args[0], "value", None) == "co"
            for kw in n.keywords
        )
    ]
    assert len(co_calls) == 1
    assert [ast.unparse(a) for a in co_calls[0].args[1:3]] == ["model_defs.CO_MIN", "model_defs.CO_MAX_PHOTO_OFF"]


def test_photo_off_carbon_rich_fd_dlnco_row_is_accepted():
    """The slice the photolysis axis opens: photolysis off, C/O above the
    network's photolysis-ON ceiling, with the dlnCO Jacobian row free.

    The stencil SHAPE keys on CO_MAX[network] (photolysis-ON) even photo-off,
    so the row is one-sided where a central one would also solve. That is
    conservative, not wrong -- the one-sided reach is never worse against
    either bound -- and keeping it photolysis-independent leaves every cached
    Jacobian bit-identical.
    """
    cp = forward.canonical_params(
        _p(network="sncho", use_photo=False, co_ratio=5.0, jac_method="fd", fisher_params=["dlnCO"])
    )
    assert cp["co_ratio"] == 5.0 and cp["use_photo"] is False
    offs, h = model_defs.fd_stencil("dlnCO", 5.0, model_defs.CO_MAX["sncho"])
    assert offs == (-1, -2, -4) and h == model_defs.FD_STEPS["dlnCO"] / 2
    # every stencil point stays inside the photolysis-aware envelope
    lo, hi = model_defs.co_bounds("sncho", False)
    assert all(lo <= 5.0 * math.exp(s * h) <= hi for s in offs)
    # ... and the same request with photolysis ON is refused before any solve
    with pytest.raises(ValueError, match="too high for this setup"):
        forward.canonical_params(
            _p(network="sncho", use_photo=True, co_ratio=5.0, jac_method="fd", fisher_params=["dlnCO"])
        )


@pytest.mark.filterwarnings("ignore:genfromtxt:UserWarning")
def test_a_t_p_table_that_cannot_be_read_is_a_value_error_and_a_defect_is_not(tmp_path, monkeypatch):
    comments_only = tmp_path / "comments.txt"
    comments_only.write_text("#\n" * 7)  # genfromtxt raises IndexError here
    with pytest.raises(ValueError, match="not parseable"):
        tp_tables._parse_table(comments_only)
    with pytest.raises(ValueError, match="not parseable"):
        tp_tables._parse_table(tmp_path / "absent.txt")

    def defect(*_a, **_k):
        raise TypeError("a bug")

    table = tmp_path / "table.txt"
    table.write_text("# units\nPressure Temp\n" + "1 2\n" * 5)
    monkeypatch.setattr(tp_tables.np, "genfromtxt", defect)
    with pytest.raises(TypeError, match="a bug"):
        tp_tables._parse_table(table)


def test_the_molecule_set_needs_the_network_named():
    with pytest.raises(KeyError, match="network"):
        model_defs.active_molecules({"extra_mols": []})
