"""Pure-Python validation of forward.canonical_params (no chemistry stack).

The parameter-scope contract: structure is a Guillot profile, an explicit
tabulated table (content-hash keyed, no T-P Fisher rows), or a PICASO climate
solve -- no profile is ever silently substituted. The reference target
defaults to the shipped W39b evening-terminator table; other planets default
to Guillot + constant Kzz. Condensation is detection-only: refused with
fisher_params (any jac_method), with photo off, and with moldiff off. Also
pins kzz_mode const/Pfunc/JM16/file, boundary conditions, emission-mode
hygiene, and the cloud/Mie Fisher parameters.
"""
import math

import numpy as np
import pytest

from jwst_tool import forward, planets


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


def test_condensation_is_detection_only():
    # v14: use_condense is a canonical parameter again (default False), and
    # a detection-only condensing run (photo + moldiff on, no Fisher) is
    # ACCEPTED -- in both T-P modes
    cp = forward.canonical_params(_p())
    assert cp["use_condense"] is False
    cp = forward.canonical_params(_p(use_condense=True))
    assert cp["use_condense"] is True
    cp = forward.canonical_params(_p(use_condense=True, tp_mode="guillot",
                                     Tirr=1560.0))
    assert cp["use_condense"] is True


def test_condensation_refuses_every_derivative_combination():
    # the method-science compatibility matrix: the pinned reservoir is not a
    # reproducible function of the parameters, so condensation + Fisher is
    # refused under EVERY jac_method (FD included, not just AD)
    for jm in ("fd", "ad"):
        with pytest.raises(ValueError, match="ANY Jacobian method"):
            forward.canonical_params(_p(use_condense=True, jac_method=jm,
                                        fisher_params=["lnZ"]))
    # a cold no-photo condensing column has no certifiable steady state
    with pytest.raises(ValueError, match="requires photochemistry ON"):
        forward.canonical_params(_p(use_condense=True, use_photo=False))
    # the condensation growth term IS the molecular-diffusion coefficient
    with pytest.raises(ValueError, match="requires molecular diffusion"):
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


def test_gcm_baseline_and_scale_are_removed():
    # no GCM profile may ever be silently substituted -- both modes raise,
    # for WASP-39b just like for every other planet
    with pytest.raises(ValueError, match="baseline"):
        forward.canonical_params(dict(planet="wasp39b", tp_mode="baseline"))
    with pytest.raises(ValueError, match="scale"):
        forward.canonical_params(_p(kzz_mode="scale", kzz_x=1.0))
    assert all("has_gcm_baseline" not in pd for pd in planets.PLANETS.values())


def test_default_structure_is_the_verified_table_where_one_ships():
    # 2026-08-11 maintainer decision (reversing the structure half of the
    # 2026-08-09 speed-first flip): a planet whose bundled measured T-P/Kzz
    # table is VERIFIED end-to-end defaults to it; every other planet
    # defaults to the analytic Guillot profile with constant Kzz. The
    # default W39b run must reproduce the literature-validated SO2 state.
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
    # today that verified set is exactly WASP-39 b
    assert forward.canonical_params(
        dict(planet="wasp39b"))["tp_mode"] == "file"


def test_having_a_table_does_not_by_itself_make_it_verified():
    # shipped_tp_table_is_default gates the default (again, since
    # 2026-08-11) AND is the verification record: HD 189733 b ships a good
    # profile that the solver does NOT certify at default settings, so it
    # must stay selectable-but-not-default, and any planet in that state
    # must carry a written reason.
    assert forward.shipped_tp_table_name("hd189733b")
    assert not forward.shipped_tp_table_is_default("hd189733b")
    assert forward.canonical_params(dict(planet="hd189733b"))["tp_mode"] == "guillot"
    # ... and choosing it explicitly still resolves to that planet's own table
    cp = forward.canonical_params(dict(planet="hd189733b", tp_mode="file"))
    assert cp["tp_mode"] == "file" and cp["kzz_mode"] == "file"
    for key in planets.PLANETS:
        if not forward.shipped_tp_table_is_default(key):
            assert planets.PLANETS[key]["tp_table_note"], key
    # only WASP-39 b's table is verified end-to-end today
    assert [k for k in planets.PLANETS
            if forward.shipped_tp_table_is_default(k)] == ["wasp39b"]


def test_shipped_table_is_per_planet_never_a_substitute():
    # Each planet resolves to ITS OWN table; a planet without one refuses
    # loudly (with the reason) rather than borrowing another's atmosphere.
    seen = {}
    for key in planets.PLANETS:
        name = forward.shipped_tp_table_name(key)
        if name:
            seen[key] = name
        else:
            with pytest.raises(ValueError, match="not available for planet"):
                forward.canonical_params(dict(planet=key, tp_mode="file"))
    assert len(set(seen.values())) == len(seen)       # no shared table
    # the verified table is the default under the vulcan provider only; the
    # picaso provider keeps the analytic default (checked on the pure
    # resolver -- a full canonical_params call under chem_provider="picaso"
    # would demand the PICASO refdata tree)
    assert forward._default_tp_mode(
        dict(planet="wasp39b", chem_provider="picaso")) == "guillot"
    assert forward._default_tp_mode(dict(planet="wasp39b")) == "file"


def test_guillot_default_tirr_follows_the_selected_planet():
    # T_irr default = sqrt(2) * T_eq of the SELECTED planet on the GUI's 20 K
    # grid; a bare constant here makes API and GUI defaults diverge per planet
    for key, p in planets.PLANETS.items():
        expect = min(max(round(p["teq_k"] * math.sqrt(2.0) / 10.0) * 10.0,
                         800.0), 2500.0)
        cp = forward.canonical_params(dict(planet=key, tp_mode="guillot"))
        assert cp["Tirr"] == expect, key
    # the values that used to disagree, pinned explicitly
    assert forward.default_tirr("wasp39b") == 1580.0
    assert forward.default_tirr("hd209458b") == 2050.0   # was 1580 via the API


def test_custom_planet_tirr_derives_from_the_entered_system():
    """The custom planet's T_irr default follows the entered star and orbit
    (T_eq = Teff sqrt(Rstar/2a)), never WASP-39 b's literature T_eq."""
    sys_cool = dict(star_teff=3300.0, rstar_rsun=0.30, orbit_au=0.05)
    teq = planets.system_teq(**sys_cool)                    # ~350 K
    expect = min(max(round(teq * math.sqrt(2.0) / 10.0) * 10.0, 800.0), 2500.0)
    cp = forward.canonical_params(dict(planet="custom", tp_mode="guillot",
                                       **sys_cool))
    assert cp["Tirr"] == expect
    # a different entered system must move the default
    cp_hot = forward.canonical_params(dict(
        planet="custom", tp_mode="guillot",
        star_teff=6100.0, rstar_rsun=1.2, orbit_au=0.03))
    assert cp_hot["Tirr"] != cp["Tirr"]
    # registry planets keep their literature-teq_k defaults (behavior pinned
    # above); the system fields must NOT override a registry entry
    cp_reg = forward.canonical_params(dict(planet="hd209458b",
                                           tp_mode="guillot"))
    assert cp_reg["Tirr"] == 2050.0
    # an explicit Tirr always wins over the derived default
    cp_exp = forward.canonical_params(dict(planet="custom", tp_mode="guillot",
                                           Tirr=1234.0, **sys_cool))
    assert cp_exp["Tirr"] == 1234.0


def test_tp_table_window_check_is_chemistry_grid_scoped():
    # The T-window gate judges the profile the ENGINE evaluates (re-gridded
    # onto CHEM_P_SPAN_DYN), not every raw row: a table extending past the
    # grid (e.g. a hot thermosphere) is fine if the in-grid part is modelable.
    def _write(tmp, P, T):
        tmp.write_text("#(dyne/cm2) (K)\nPressure Temp\n"
                       + "\n".join(f"{p:.6e} {t:.2f}" for p, t in zip(P, T)))
        return tmp

    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    lo, hi = forward.CHEM_P_SPAN_DYN
    # extends both ways past the grid, in-grid profile is a modelable 900 K
    P = np.array([lo / 100, lo, hi, hi * 100])
    T = np.array([5000.0, 900.0, 900.0, 5000.0])       # out of window only outside
    assert forward._read_tp_table(_write(d / "ok.txt", P, T))["T"].size == 4
    # in-grid profile itself breaches the ceiling -> refused
    T_bad = np.array([5000.0, 2990.0, 900.0, 5000.0])
    with pytest.raises(ValueError, match="chemistry grid"):
        forward._read_tp_table(_write(d / "bad.txt", P, T_bad))


def test_isothermal_is_removed():
    with pytest.raises(ValueError, match="isothermal profile"):
        forward.canonical_params(_p(tp_mode="isothermal"))
    # its old T_iso companion key is unknown now, and unknown keys REFUSE
    with pytest.raises(ValueError, match="unknown parameter"):
        forward.canonical_params(_p(T_iso=1100.0))


def test_guillot_const_are_accepted():
    for tp, extra in (("guillot", dict(Tirr=1560.0, Tint=100.0,
                                       log_kappa=-2.3, log_gamma=-1.0)),):
        cp = forward.canonical_params(_p(tp_mode=tp, **extra))
        assert cp["tp_mode"] == tp
        assert cp["kzz_mode"] == "const"


def test_resolution_defaults_replace_quality():
    # the fidelity "quality" tier is gone; explicit nz/nu_pts/yconv default to
    # the old "fast" tier, and the RT layer count is derived (not a cache field)
    cp = forward.canonical_params(_p())
    assert cp["nz"] == forward.NZ_DEFAULT == 100
    assert cp["nu_pts"] == forward.NU_PTS_DEFAULT == 4000
    assert cp["yconv_cri"] == forward.YCONV_DEFAULT == 1.0e-2
    assert "quality" not in cp        # retired
    assert "art_nlayer" not in cp     # locked to nz in run_model, not cache-keyed


def test_resolution_ceiling_accepted():
    cp = forward.canonical_params(_p(nz=150, nu_pts=8000, yconv_cri=1.0e-3))
    assert (cp["nz"], cp["nu_pts"], cp["yconv_cri"]) == (150, 8000, 1.0e-3)


def test_resolution_out_of_range_raises():
    # nu_pts cap raised 8000 -> 32000 at v27: the old cap sat BELOW the model
    # resolving power that NIRSpec G395H's line-spread function needs, so
    # opacity convergence could not be demonstrated inside the allowed range
    for bad in (dict(nz=40), dict(nz=200), dict(nu_pts=1000),
                dict(nu_pts=40000), dict(yconv_cri=1.0), dict(yconv_cri=1.0e-6)):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(**bad))


def test_v25_extras_present_and_resolve_in_engine_table():
    # v25 (Shami Tsai request): CS2 photochemical sulfur + the CH4-photolysis
    # hydrocarbons. Import-light: vulcan_forward.constants is pure constants.
    from vulcan_forward import constants as _vfc
    for mol in ("CS2", "C2H4", "C2H6"):
        assert mol in forward.EXTRA_MOLECULES
    # EVERY extra must resolve in the shared engine's molecule table -- a
    # token listed here but absent there would only fail at run time.
    for mol in forward.EXTRA_MOLECULES:
        spec = _vfc.MOLECULES[mol]
        assert spec["molmass"] > 0
        assert spec["vulcan"]


def test_unknown_rt_molecule_points_to_engine():
    # An out-of-set RT molecule is refused loudly, with how to add it. Since the
    # engine became a shared distribution its molecule table is INJECTABLE, so
    # the remedy names that route rather than telling the user to go edit a
    # constant inside another package.
    with pytest.raises(ValueError, match="molecule_table"):
        forward.canonical_params(_p(extra_mols=["PH3"]))


def test_vm_mol_is_pinned_explicitly_default_off():
    # the tool must PIN the vm_mol scheme in the canonical params (cache-
    # keyed), never inherit the upstream YAML default; False = the validated
    # baseline chemistry
    cp = forward.canonical_params(_p())
    assert cp["use_vm_mol"] is False
    assert forward.canonical_params(_p(use_vm_mol=True))["use_vm_mol"] is True


def test_yconv_range_widened_floor():
    # the tolerance ladder reaches 1e-4 (strict, slow); below it still raises
    assert forward.YCONV_RANGE == (1.0e-4, 1.0e-2)
    assert forward.canonical_params(_p(yconv_cri=1.0e-4))["yconv_cri"] == 1.0e-4
    with pytest.raises(ValueError):
        forward.canonical_params(_p(yconv_cri=5.0e-5))


def test_co_baseline_is_the_cfg_elemental_basis():
    # CO_BASELINE must be the network cfg's C_H/O_H (~0.549), never the
    # FastChem EQ-init ratio (0.458), which only seeds the initial guess;
    # run_model additionally cross-checks the live cfg
    assert abs(forward.CO_BASELINE - 0.00295 / 0.00537) < 1e-12
    assert abs(forward.CO_BASELINE - 0.549) < 1e-3


def test_composition_is_structural_one_path():
    # v13: composition is ONE structural path -- co_ratio (absolute N_C/N_O)
    # and met_x_solar go straight into the cfg elemental abundances; the
    # legacy differential knobs are gone from the canonical params entirely
    cp = forward.canonical_params(_p())
    assert abs(cp["co_ratio"] - forward.CO_BASELINE) < 1e-5
    assert "dco" not in cp and "co_baseline" not in cp
    # C-rich is the same path, with NO detection-only restriction: FD Fisher
    # rows are certified re-solves, valid at any baseline
    cp = forward.canonical_params(_p(co_ratio=1.5, met_x_solar=30.0,
                                     fisher_params=["lnZ", "dlnCO"]))
    assert cp["co_ratio"] == 1.5 and cp["met_x_solar"] == 30.0


def test_composition_ranges():
    for bad in (dict(co_ratio=0.05), dict(co_ratio=2.5),
                dict(met_x_solar=0.05), dict(met_x_solar=150.0)):
        with pytest.raises(ValueError):
            forward.canonical_params(_p(**bad))


def test_fisher_works_photo_off_and_validates_names():
    # v13: FD Jacobians are certified re-solves -- no photo-on tangent regime,
    # so the old photo-on Fisher gate is gone
    cp = forward.canonical_params(_p(use_photo=False, fisher_params=["lnZ"]))
    assert cp["fisher_params"] == ["lnZ"] and cp["use_photo"] is False
    # unknown Fisher parameters are refused loudly (FD needs a defined step)
    with pytest.raises(ValueError, match="unknown Fisher parameter"):
        forward.canonical_params(_p(fisher_params=["lnFoo"]))
    with pytest.raises(ValueError, match="unknown Fisher parameter"):
        forward.canonical_params(_p(fisher_params=["Tint_cl"]))  # climate-only


def test_jac_method_default_fd_and_validated():
    # v14: jac_method joins the canonical params -- certified FD by default,
    # unknown values refused loudly
    cp = forward.canonical_params(_p(fisher_params=["lnKzz"]))
    assert cp["jac_method"] == "fd"
    with pytest.raises(ValueError, match="jac_method"):
        forward.canonical_params(_p(fisher_params=["lnKzz"],
                                    jac_method="magic"))


def test_jac_method_ad_requires_photo_on():
    # the warm-jvp AD rows are validated only in the photo-on regime; FD
    # keeps working photo-off (previous test), AD does not
    cp = forward.canonical_params(_p(fisher_params=["lnKzz", "Tirr"],
                                     jac_method="ad"))
    assert cp["jac_method"] == "ad"
    with pytest.raises(ValueError, match="photo-on"):
        forward.canonical_params(_p(fisher_params=["lnKzz"], jac_method="ad",
                                    use_photo=False))


def test_jac_method_ad_covers_every_row_but_neutralizes_when_inert():
    # v14: 'ad' applies to EVERY requested row, composition directions
    # included (the cross-validated differential map; the C-rich b_z corner
    # refuses at run time) -- so comp-only selections KEEP 'ad'
    cp = forward.canonical_params(_p(fisher_params=["lnZ", "dlnCO"],
                                     jac_method="ad"))
    assert cp["jac_method"] == "ad"
    # with no Jacobian requested at all the knob is inert -- normalized to
    # 'fd' so it cannot fragment the cache key, and photo-off is then fine
    cp = forward.canonical_params(_p(jac_method="ad"))
    assert cp["jac_method"] == "fd"
    cp = forward.canonical_params(_p(jac_method="ad", use_photo=False))
    assert cp["jac_method"] == "fd"


def test_condensation_error_states_91_percent_wrong():
    # the refusal message must be misread-proof: the ~0.91 jvp-vs-FD number
    # is a RELATIVE ERROR (the tangent is ~91% wrong), never presentable as
    # a 0.91 agreement ratio / 9% mismatch
    with pytest.raises(ValueError) as ei:
        forward.canonical_params(_p(use_condense=True, fisher_params=["lnZ"]))
    msg = str(ei.value)
    assert "91% wrong" in msg and "not a 9% mismatch" in msg


def test_rt_knobs_v15_defaults_and_validation():
    # v15: three ExoJAX RT knobs are canonical (cache-keyed). Defaults are
    # the pre-v15 hard-coded values, so a default run reproduces v14 physics.
    cp = forward.canonical_params(_p())
    assert cp["rt_ptop_bar"] == 1.0e-8
    assert cp["rt_integration"] == "simpson"
    assert cp["rt_dit_res"] == 1.0
    # the exercised ranges / choices are validated loudly
    cp = forward.canonical_params(_p(rt_ptop_bar=1.0e-6,
                                     rt_integration="trapezoid",
                                     rt_dit_res=0.2))
    assert (cp["rt_ptop_bar"], cp["rt_integration"], cp["rt_dit_res"]) == \
        (1.0e-6, "trapezoid", 0.2)
    with pytest.raises(ValueError, match="rt_ptop_bar"):
        forward.canonical_params(_p(rt_ptop_bar=1.0e-5))
    with pytest.raises(ValueError, match="rt_ptop_bar"):
        forward.canonical_params(_p(rt_ptop_bar=1.0e-10))
    with pytest.raises(ValueError, match="rt_integration"):
        forward.canonical_params(_p(rt_integration="euler"))
    with pytest.raises(ValueError, match="rt_dit_res"):
        forward.canonical_params(_p(rt_dit_res=0.01))
    with pytest.raises(ValueError, match="rt_dit_res"):
        forward.canonical_params(_p(rt_dit_res=2.0))


def test_rt_knobs_fragment_the_cache_key():
    # changing any RT knob must change the cache key (different physics)
    k0 = forward.params_key(_p())
    assert forward.params_key(_p(rt_ptop_bar=1.0e-7)) != k0
    assert forward.params_key(_p(rt_integration="trapezoid")) != k0
    assert forward.params_key(_p(rt_dit_res=0.5)) != k0


# --- WASP-39 b reference state: DO NOT let this drift ------------------------
# The REFERENCE W39b configuration (tp_mode="file", the shipped evening-
# terminator table) is the one measured against the published JWST detection
# (G395H SO2 sigma_detect 2.89 at v27, vs 4.5-4.8 published; the 4.16 this
# line used to carry predates the v26 radius anchoring -- see the cache-key
# test below for the re-measurement). Since 2026-08-11 it is the
# DEFAULT again (it was demoted to selectable 2026-08-09/-11 under the
# Guillot-everywhere flip). The guard stays anchored to the EXPLICIT
# file-mode config so it keeps protecting the reference atmosphere even if
# the default ever moves again; a separate assertion below pins default ==
# reference. Re-measure against the literature before updating expected
# values.
W39B_REFERENCE = {
    "tp_mode": "file",                      # measured evening-terminator table
    "tp_file": "shipped",
    "tp_file_sha1": "1a4ce744e65205d8",     # exact profile bytes (T AND Kzz)
    "kzz_mode": "file",                     # mixing from the table, not a stand-in
    "kzz_const": 0.0,                       # inert once tabulated
    "met_x_solar": 10.0,                    # Tsai+2023 10x solar
    "co_ratio": 0.549348,                   # cfg C_H/O_H
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


def test_wasp39b_reference_cache_key_is_stable():
    # RE-PINNED at v31 (was acba771e80c772a1 at v30, 9b30d6d526e2a78a at v29,
    # ead8394cf913ff67 at v28, ba52447cad5b40c5 at v27). v31 removes the
    # interim "ckd" opacity mode with forward 0.8.0 -- a previously-canonical
    # value is now rejected, so the buster moves; the reference spectrum is
    # bit-identical to v30's. v30 changed the OPACITY DATA: the default
    # opacity_mode is "exomolop", the published ExoMol/HITEMP k-tables with
    # H2/He broadening, replacing tables built here from HITRAN's 296 K
    # air-broadened lines.
    #
    # Measured on WASP-39 b with the published Tsai et al. 2023 chemistry, T-P,
    # geometry and pressure grid held fixed, R=100 over 1.02-5.26 um. Spectral
    # amplitude (std of the binned depth), against the JWST PRISM data:
    #     R = 1,477 HITRAN            1256 ppm   (2.80x the data)
    #     R = 700,000 HITRAN           870 ppm   (1.94x)
    #     ExoMolOP, 5 species          695 ppm   (1.55x)
    #     ExoMolOP, 6 species (+H2S)   663 ppm   (1.48x)
    #     published gCMCRT             622 ppm   (1.39x)
    #     the data itself              448 ppm
    #
    # The old "~2.2x amplitude gap vs the published spectra" was RESOLVED
    # 2026-08-15: it was a decode error in OUR comparison harness, not a
    # model discrepancy. The published gCMCRT files decode as
    # depth = (H(1)^2 + 2*col2)/Rstar^2 with H(1) = each file's OWN header
    # value (proved from gCMCRT's source); the earlier R_ref^2 + col2 rule
    # halved their amplitude (311 ppm) and manufactured the gap. Correctly
    # decoded, the residual vs the published model is ~8% in amplitude, and
    # this engine's RT is separately verified against petitRADTRANS to
    # 0.0013-0.096% rms (vulcan-forward README + its committed e2e fixtures).
    # Full decode walkthrough: vulcan-forward README, gCMCRT section.
    #
    # NOT RE-MEASURED at v30, and required before this key is quoted as a
    # science result: the tool's own default-geometry median depth (it was
    # 19,712 ppm at v28-v29 under p_ref_bar = 1e-3, and ExoMolOP adds window
    # opacity so it will be deeper) and the G395H SO2 significance (2.89 at
    # v27). Both need a full run; the SO2 number additionally needs the pandeia
    # backend, which was not configured where this key was re-pinned.
    #
    # Earlier history, still accurate for the runs it describes:
    # RE-PINNED at v29 (was ead8394cf913ff67 at v28, ba52447cad5b40c5 at v27).
    # v29 moved EMISSION to correlated-k as well. This reference run is
    # TRANSMISSION, so its spectrum is bit-identical to v28's -- the key moves
    # only because forward._VERSION is itself a canonical parameter, which is
    # the point of the buster. Every measured number below still stands.
    #
    # Emission needed the move far more than transmission did. Flux carries
    # exp(-tau) where a transit depth carries ln(tau), so the same "too opaque"
    # sampling bias that cost transmission a contrast error suppressed most of
    # the emergent flux: measured over 3-5 um against an R = 700,000 emission
    # reference on the identical atmosphere, R = 1,477 returned 45% of the
    # correct band-integrated flux (59% rms per R=100 bin, spectral contrast
    # 70% where the truth is 47%). Correlated-k gives 1.9% rms, +0.6% bias.
    #
    # v28 notes, still current:
    # Transmission moved to correlated-k, and "opacity_mode" joined the
    # canonical set.
    #
    # WHY: exojax evaluates a cross section on the grid it is handed, and its
    # own wavenumber_grid warns below R = 700,000. This tool ran at R = 1,477.
    # Measured on WASP-39 b with the published Tsai et al. 2023 chemistry, T-P,
    # geometry and pressure grid held fixed, binned to R = 100 over 1.02-5.3 um
    # against the NIRSpec PRISM data: 2.80x the observed spectral contrast at
    # R = 1,477 versus 1.93x at R = 700,000. Correlated-k reproduces the
    # R = 700,000 line-by-line spectrum to 32 ppm rms (median data uncertainty
    # 105 ppm) in 2.8 s instead of ~20 minutes.
    #
    # RE-MEASURED on the run this key names: median 3.0-5.5 um depth 19,712 ppm
    # against the JWST ERS FIREFLy 21,381 (-7.8%), where v27 sat at 21,126
    # (-1.2%). That is NOT a regression: roughly 1500 ppm of the old agreement
    # was the grid bias (which made the column too opaque) cancelling against
    # the p_ref_bar = 1e-3 default. With the bias removed, p_ref_bar is exposed
    # for what it is -- a free normalization degenerate with Rp, which every
    # published model fits as a vertical offset. It has deliberately NOT been
    # re-tuned to make the default land on the data; doing that would be
    # fitting a nuisance parameter to the answer.
    #
    # NOT RE-MEASURED, and required before this key is quoted as a science
    # result: the G395H SO2 significance. It was 2.89 at v27. Re-running it
    # needs the pandeia backend (JWST_TOOL_PANDEIA_PYTHON), which was not
    # configured where this key was re-pinned.
    # The key hashes every canonical parameter: if ANY default feeding the
    # reference run changes, this trips even when the pins above still pass.
    # RE-PINNED at v27 (was de55467c4a459b4e at v26, f14f4d10512552ea before
    # that). Two changes: p_btm_bar joined the canonical set, and the radius
    # anchor moved to the bottom layer's lower BOUNDARY, where exojax actually
    # defines radius_btm, instead of its centre. Measured on this profile the
    # boundary fix is dR/R = -0.176%, i.e. -0.35% in depth, and it also makes
    # the depth independent of art_nlayer (it was +0.20% at 100 layers and
    # +1.02% at 20).
    #
    # RE-MEASURED as this test's comment requires, on the run this key names:
    # median 3.0-5.5 um depth 21,126 ppm against the JWST ERS FIREFLy
    # spectrum's 21,381 (-1.2%), and G395H SO2 sigma_detect 2.89 at R=100 with
    # no noise floor. That significance is NOT the 4.16 the docs carried: 4.16
    # belongs to the pre-v26 geometry, whose 1.42x excess spectral contrast
    # inflated every feature, and 4.16/1.42 = 2.93 recovers what is measured
    # here. The tool therefore forecasts BELOW the published 4.5-4.8 sigma
    # detection, and that gap is real and open, not a regression from v26.
    assert forward.params_key(forward.canonical_params(
        dict(planet="wasp39b", tp_mode="file"))) == "c20582509e215eb0"
    # ... and since 2026-08-11 the bare DEFAULT run is that same atmosphere:
    # a default W39b forecast is the literature-validated configuration.
    assert forward.params_key(forward.canonical_params(
        dict(planet="wasp39b"))) == "c20582509e215eb0"


def test_wasp39b_shipped_table_bytes_are_unchanged():
    # The sha1 above is only meaningful if it is re-derived from the file the
    # run actually reads -- this catches the table itself being swapped.
    path = forward._shipped_tp_file("wasp39b")
    assert path.name == "atm_W39b_evening_TP_Kzz.txt"
    tab = forward._read_tp_table(path)
    assert tab["Kzz"] is not None, "the reference table must carry its Kzz column"
    import hashlib
    assert hashlib.sha1(path.read_bytes()).hexdigest()[:16] == \
        W39B_REFERENCE["tp_file_sha1"]
# --- tp_mode="file" ---------------------------------------------------------

def test_file_mode_is_content_addressed(tmp_path):
    p1 = _table(tmp_path, name="a.txt")
    p2 = _table(tmp_path, name="b.txt")               # same content, other path
    p3 = _table(tmp_path, name="c.txt", tmax=1500.0)  # different content
    cp1 = forward.canonical_params(_pf(p1))
    assert cp1["tp_file_sha1"] and len(cp1["tp_file_sha1"]) == 16
    assert forward.params_key(_pf(p1)) == forward.params_key(_pf(p2))
    assert forward.params_key(_pf(p1)) != forward.params_key(_pf(p3))


def test_file_mode_has_no_tp_fisher_rows(tmp_path):
    assert forward.TP_PARAM_NAMES["file"] == []
    p = _table(tmp_path)
    cp = forward.canonical_params(_pf(p, fisher_params=["lnZ", "lnKzz"]))
    assert cp["fisher_params"] == ["lnKzz", "lnZ"]
    with pytest.raises(ValueError, match="NO T-P Fisher rows"):
        forward.canonical_params(_pf(p, fisher_params=["Tirr"]))


def test_file_mode_zeroes_parametric_tp_knobs(tmp_path):
    p = _table(tmp_path)
    cp = forward.canonical_params(_pf(p, Tirr=1560.0))
    assert cp["Tirr"] == cp["Tint"] == cp["log_kappa"] == cp["log_gamma"] == 0.0
    # outside file mode the file identity is empty (cache hygiene)
    cp_iso = forward.canonical_params(_p())
    assert cp_iso["tp_file"] == "" and cp_iso["tp_file_sha1"] == ""


def test_bad_tables_rejected(tmp_path):
    with pytest.raises(ValueError, match="monotonic"):
        forward.canonical_params(_pf(_table(tmp_path, scramble=True)))
    with pytest.raises(ValueError, match="modelable window"):
        forward.canonical_params(_pf(_table(tmp_path, tmax=3300.0)))
    with pytest.raises(ValueError, match="rows"):
        forward.canonical_params(_pf(_table(tmp_path, rows=3)))
    bad = tmp_path / "nocol.txt"
    bad.write_text("#(dyne/cm2) (K)\nPress\tT\n1e6\t1000\n1e5\t900\n"
                   "1e4\t800\n1e3\t700\n")
    with pytest.raises(ValueError, match="Pressure"):
        forward.canonical_params(_pf(bad))
    with pytest.raises(ValueError, match="not found"):
        forward.canonical_params(_pf(tmp_path / "missing.txt"))


def test_canonical_params_round_trip_in_file_mode(tmp_path):
    # The GUI hands the SUBPROCESS canonical_params(params) as its params
    # file, so canonicalization must be idempotent (given a resolvable path).
    p = _table(tmp_path)
    cp = forward.canonical_params(_pf(p))
    cp2 = forward.canonical_params({**cp, "tp_file_path": str(p)})
    assert cp2 == cp


# --- kzz_mode ---------------------------------------------------------------

def test_kzz_file_requires_file_tp_and_kzz_column(tmp_path):
    with pytest.raises(ValueError, match="requires tp_mode='file'"):
        forward.canonical_params(_p(kzz_mode="file"))
    no_kzz = _table(tmp_path, kzz=False, name="nokzz.txt")
    with pytest.raises(ValueError, match="Kzz.*column"):
        forward.canonical_params(_pf(no_kzz, kzz_mode="file"))
    cp = forward.canonical_params(_pf(_table(tmp_path), kzz_mode="file"))
    assert cp["kzz_const"] == cp["kzz_kmax"] == cp["kzz_plev"] == 0.0
    assert cp["kzz_kdeep"] == 0.0


def test_kzz_parametric_modes_validate_and_zero_inert_knobs():
    cp = forward.canonical_params(_p(kzz_mode="Pfunc", kzz_kmax=1.0e5,
                                     kzz_plev=0.1))
    assert cp["kzz_kmax"] == 1.0e5 and cp["kzz_plev"] == 0.1
    assert cp["kzz_const"] == 0.0 and cp["kzz_kdeep"] == 0.0
    cp = forward.canonical_params(_p(kzz_mode="JM16", kzz_kdeep=1.0e6))
    assert cp["kzz_kdeep"] == 1.0e6
    assert cp["kzz_const"] == cp["kzz_kmax"] == cp["kzz_plev"] == 0.0
    with pytest.raises(ValueError, match="kzz_kmax"):
        forward.canonical_params(_p(kzz_mode="Pfunc", kzz_kmax=1.0e15,
                                    kzz_plev=0.1))
    with pytest.raises(ValueError, match="kzz_plev"):
        forward.canonical_params(_p(kzz_mode="Pfunc", kzz_kmax=1.0e5,
                                    kzz_plev=1.0e5))
    with pytest.raises(ValueError, match="kzz_kdeep"):
        forward.canonical_params(_p(kzz_mode="JM16", kzz_kdeep=1.0))
    with pytest.raises(ValueError, match="unknown kzz_mode"):
        forward.canonical_params(_p(kzz_mode="scale"))


# --- boundary conditions ----------------------------------------------------

def test_settling_gating():
    cp = forward.canonical_params(_p(use_settling=True))
    assert cp["use_settling"] is True
    with pytest.raises(ValueError, match="requires use_moldiff"):
        forward.canonical_params(_p(use_settling=True, use_moldiff=False))
    with pytest.raises(ValueError, match="pins settling OFF"):
        forward.canonical_params(_p(use_settling=True, use_condense=True))


def test_diff_esc_curated_choices():
    cp = forward.canonical_params(_p(diff_esc=["H2", "H"]))
    assert cp["diff_esc"] == ["H", "H2"]          # sorted, deduped
    with pytest.raises(ValueError, match="diff_esc"):
        forward.canonical_params(_p(diff_esc=["SO2"]))


def test_diff_esc_requires_moldiff():
    # escape flux ~ TOA molecular-diffusion coefficient: with moldiff off it
    # would silently be zero, so it is refused (not silently kept).
    with pytest.raises(ValueError, match="requires use_moldiff"):
        forward.canonical_params(_p(diff_esc=["H"], use_moldiff=False))
    # empty escape list is fine with moldiff off (nothing to escape)
    cp = forward.canonical_params(_p(use_moldiff=False))
    assert cp["diff_esc"] == []


def test_vm_mol_zeroed_without_moldiff():
    # use_vm_mol is inert when moldiff is off (engine gates use_vm on both);
    # it is zeroed so it cannot fragment the cache into two identical setups.
    on = forward.canonical_params(_p(use_moldiff=False, use_vm_mol=True))
    off = forward.canonical_params(_p(use_moldiff=False, use_vm_mol=False))
    assert on["use_vm_mol"] is False
    assert forward.params_key(on) == forward.params_key(off)
    # with moldiff on the flag is honored
    assert forward.canonical_params(_p(use_vm_mol=True))["use_vm_mol"] is True


def test_bc_entries_canonicalized():
    cp = forward.canonical_params(_p(
        top_flux=[["H2O", 1.0e8], ["CH4", 0.0]],          # zero row dropped
        bot_flux=[["SO2", 1.0e9, 0.1]]))
    assert cp["top_flux"] == [["H2O", 1.0e8]]
    assert cp["bot_flux"] == [["SO2", 1.0e9, 0.1]]
    # defaults stay empty (cache hygiene: v15 keys unchanged by absence)
    cp0 = forward.canonical_params(_p())
    assert cp0["top_flux"] == [] and cp0["bot_flux"] == []
    assert cp0["use_settling"] is False and cp0["diff_esc"] == []
    with pytest.raises(ValueError, match="duplicate"):
        forward.canonical_params(_p(top_flux=[["H2O", 1e8], ["H2O", 2e8]]))
    with pytest.raises(ValueError, match="bad species token"):
        forward.canonical_params(_p(top_flux=[["H2 O", 1e8]]))
    with pytest.raises(ValueError, match="expected 3 fields"):
        forward.canonical_params(_p(bot_flux=[["SO2", 1e9]]))
    with pytest.raises(ValueError, match="vdep"):
        forward.canonical_params(_p(bot_flux=[["SO2", 1e9, -1.0]]))
    with pytest.raises(ValueError, match="beyond"):
        forward.canonical_params(_p(top_flux=[["H2O", 1e30]]))


# --- cloud Fisher marginalization ------------------------------------------

def test_cloud_fisher_params_require_cloud_on():
    cp = forward.canonical_params(_p(cloud_on=True,
                                     fisher_params=["lnZ",
                                                    "log_kappa_cloud",
                                                    "alpha_cloud"]))
    assert set(cp["fisher_params"]) == {"alpha_cloud", "lnZ",
                                        "log_kappa_cloud"}
    with pytest.raises(ValueError, match="cloud"):
        forward.canonical_params(_p(fisher_params=["log_kappa_cloud"]))
    # cloud rows are labeled + stepped like every other parameter
    for n in forward.CLOUD_FISHER_PARAMS:
        assert n in forward.FD_STEPS
        assert n in forward.PARAM_LABELS
        assert n in forward.PARAM_SYMBOLS


def test_cloud_fisher_params_work_in_file_mode(tmp_path):
    p = _table(tmp_path)
    cp = forward.canonical_params(_pf(p, cloud_on=True,
                                      fisher_params=["lnZ",
                                                     "log_kappa_cloud"]))
    assert "log_kappa_cloud" in cp["fisher_params"]


# --- Mie condensate deck ----------------------------------------------------

def test_mie_deck_off_by_default_and_zeroed():
    cp = forward.canonical_params(_p())
    assert cp["mie_condensate"] == ""
    # the three continuous knobs are zeroed when the deck is off (cache hygiene)
    assert cp["mie_log_rg"] == cp["mie_sigmag"] == cp["mie_log_mmr"] == 0.0
    # a set deck keys the cache; two condensates never collide
    on = forward.canonical_params(_p(mie_condensate="MgSiO3"))
    assert forward.params_key(on) != forward.params_key(_p())
    fe = forward.canonical_params(_p(mie_condensate="Fe"))
    assert forward.params_key(on) != forward.params_key(fe)


def test_mie_condensate_and_ranges_validated():
    with pytest.raises(ValueError, match="mie_condensate"):
        forward.canonical_params(_p(mie_condensate="Diamond"))
    with pytest.raises(ValueError, match="mie_log_rg"):
        forward.canonical_params(_p(mie_condensate="MgSiO3", mie_log_rg=-1.0))
    with pytest.raises(ValueError, match="mie_sigmag"):
        forward.canonical_params(_p(mie_condensate="MgSiO3", mie_sigmag=5.0))
    with pytest.raises(ValueError, match="mie_log_mmr"):
        forward.canonical_params(_p(mie_condensate="MgSiO3", mie_log_mmr=0.0))


def test_mie_fisher_params_require_a_condensate():
    cp = forward.canonical_params(_p(
        mie_condensate="MgSiO3",
        fisher_params=["lnZ", "mie_log_rg", "mie_sigmag", "mie_log_mmr"]))
    assert set(cp["fisher_params"]) == {"lnZ", "mie_log_rg", "mie_sigmag",
                                        "mie_log_mmr"}
    with pytest.raises(ValueError, match="require a mie_condensate"):
        forward.canonical_params(_p(fisher_params=["mie_log_rg"]))
    # every Mie row is labeled + stepped like any other parameter
    for n in forward.MIE_FISHER_PARAMS:
        assert n in forward.FD_STEPS
        assert n in forward.PARAM_LABELS
        assert n in forward.PARAM_SYMBOLS
    # Mie and the power-law deck are independent (both freeable together)
    cp2 = forward.canonical_params(_p(
        cloud_on=True, mie_condensate="Fe",
        fisher_params=["log_kappa_cloud", "mie_log_mmr"]))
    assert {"log_kappa_cloud", "mie_log_mmr"} <= set(cp2["fisher_params"])


def test_mie_fisher_params_work_in_file_mode(tmp_path):
    p = _table(tmp_path)
    cp = forward.canonical_params(_pf(p, mie_condensate="MgSiO3",
                                      fisher_params=["lnZ", "mie_log_rg"]))
    assert "mie_log_rg" in cp["fisher_params"]


def test_every_freeable_param_has_display_metadata():
    # Any parameter that can be freed in the Fisher forecast is also a valid
    # constraint goal and a Jacobian row, so it must carry a display label,
    # unit, symbol, and an FD step -- a missing entry KeyErrors the GUI/run.
    freeable = (set(forward.CHEM_PARAM_NAMES) | set(forward.CLOUD_FISHER_PARAMS)
                | set(forward.MIE_FISHER_PARAMS)
                | {p for ns in forward.TP_PARAM_NAMES.values() for p in ns})
    for m in (forward.PARAM_LABELS, forward.PARAM_UNITS, forward.PARAM_SYMBOLS,
              forward.FD_STEPS):
        assert not (freeable - set(m)), sorted(freeable - set(m))


# --- emission mode (science_mode) ------------------------------------------

def test_science_mode_gating():
    with pytest.raises(ValueError, match="science_mode"):
        forward.canonical_params(_p(science_mode="reflection"))
    # isothermal was removed, so the old emission+isothermal "featureless
    # blackbody" refusal is gone; guillot emission is the normal path
    cp = forward.canonical_params(_p(science_mode="emission",
                                     tp_mode="guillot", Tirr=1560.0))
    assert cp["science_mode"] == "emission"


def test_emission_star_params_and_hygiene(tmp_path):
    cp = forward.canonical_params(_p(science_mode="emission",
                                     tp_mode="guillot", Tirr=1560.0))
    # defaults come from the planet registry star
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
    with pytest.raises(ValueError, match="star_teff"):
        forward.canonical_params(_p(science_mode="emission",
                                    tp_mode="guillot", Tirr=1560.0,
                                    star_teff=10000.0))
    # emission works with a tabulated profile too
    cp_f = forward.canonical_params(_pf(_table(tmp_path),
                                        science_mode="emission"))
    assert cp_f["science_mode"] == "emission"
    # BOTH geometries default to the shipped column bottom (v27). The emission
    # default was briefly 100 bar, on the theory that its refusals meant the
    # column was too shallow. Measured, that was wrong: at 7.6 bar the bottom
    # optical depth is 30 to 370 across every instrument window, and the only
    # thin part is a 60 nm notch at the blue edge where no continuum opacity is
    # modeled. The gate is flux-weighted now; the depth did not need to move.
    assert cp_f["p_btm_bar"] == forward.P_BTM_TRANSMISSION_BAR
    assert cp_t["p_btm_bar"] == forward.P_BTM_TRANSMISSION_BAR
    # a deeper column is still reachable, and still gated on the table covering
    with pytest.raises(ValueError, match="chemistry-grid bottom"):
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


# --- v17 (2026-07-19 audit response) ----------------------------------------

def test_tp_table_must_reach_chemistry_bottom(tmp_path):
    """A table stopping above the chemistry-grid bottom is REFUSED: the engine
    would clamp-extend the last tabulated T isothermally over the quench
    region (v17). The standard fixture reaches 10^6.9 = 7.9e6 dyn/cm^2 and
    passes; a table ending at 1 bar must raise."""
    P = np.logspace(6.0, -1.0, 8)              # 1 bar bottom: too shallow
    lines = ["#(dyne/cm2) (K)", "Pressure\tTemp"]
    for i in range(8):
        lines.append(f"{P[i]:.6e}\t{1200.0 - 40.0 * i:.1f}")
    p = tmp_path / "shallow.txt"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="chemistry-grid bottom"):
        forward.canonical_params(_pf(p))
    # the standard fixture (spans past P_b) still validates
    forward.canonical_params(_pf(_table(tmp_path)))


def test_composition_fd_stencil_envelope():
    """FD Fisher rows for lnZ/dlnCO refuse a baseline within one 2h stencil of
    the validated range edge (v17): the stencil would otherwise silently solve
    the chemistry outside the exercised envelope (met=100 -> 122x solar,
    co=2.0 -> C/O 2.44). AD rows take no stencil and are exempt here (the
    dlnCO AD row has its own run-time b_z margin)."""
    forward.canonical_params(_p(met_x_solar=80.0, fisher_params=["lnZ"]))
    with pytest.raises(ValueError, match="stencil"):
        forward.canonical_params(_p(met_x_solar=100.0, fisher_params=["lnZ"]))
    with pytest.raises(ValueError, match="stencil"):
        forward.canonical_params(_p(met_x_solar=0.1, fisher_params=["lnZ"]))
    forward.canonical_params(_p(co_ratio=1.6, fisher_params=["dlnCO"]))
    with pytest.raises(ValueError, match="stencil"):
        forward.canonical_params(_p(co_ratio=2.0, fisher_params=["dlnCO"]))
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


# --- unknown-key rejection (v31) ---------------------------------------------

def test_unknown_keys_are_refused_with_a_hint():
    # The bug this pins: a validation driver passed {"mode": "emission"}, the
    # key was silently ignored, and a TRANSMISSION spectrum was scored against
    # eclipse data (chi2/N ~ 5e5). Unknown keys must refuse, never drop.
    with pytest.raises(ValueError, match="science_mode"):
        forward.canonical_params(_p(mode="emission"))
    with pytest.raises(ValueError, match="opacity_mode"):
        forward.canonical_params(_p(opacity="exomolop"))
    with pytest.raises(ValueError, match="unknown parameter"):
        forward.canonical_params(_p(totally_made_up=1))


def test_canonical_output_round_trips_through_validation():
    # share_config validates a SAVED canonical payload by feeding it back in;
    # every output key (echo fields included) must therefore be accepted.
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
             "default_opacity_mode", "_default_tp_mode"}
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


# --- ExoMolOP no-table gate (v31) --------------------------------------------

def test_species_without_an_exomolop_table_refuse_early_under_the_default():
    # ExoMolOP publishes no CS2/C2H6 k-table: under the default opacity_mode
    # the run must refuse at the API, not minutes later inside the RT build.
    for mol in sorted(forward._NO_EXOMOLOP_TABLE):
        with pytest.raises(ValueError, match="no published ExoMolOP k-table"):
            forward.canonical_params(_p(extra_mols=[mol]))
        cp = forward.canonical_params(_p(extra_mols=[mol],
                                         opacity_mode="lbl"))
        assert mol in cp["extra_mols"]


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
