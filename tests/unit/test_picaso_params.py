"""Pure-Python validation of the v18 PICASO provider / climate matrix
(no chemistry stack, no picaso, no reference data -- the fingerprint seams
are monkeypatched exactly so these tests never touch the tree)."""
import sys

import pytest

from jwst_tool import forward


def _p(**kw):
    base = dict(planet="wasp39b", tp_mode="guillot",
                kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return base


def _pp(**kw):
    """A minimal valid picaso-provider param set."""
    base = dict(planet="wasp39b", chem_provider="picaso",
                tp_mode="guillot", co_ratio=0.50)
    base.update(kw)
    return base


def _pc(**kw):
    """A minimal valid climate-mode param set (exact CK node)."""
    base = dict(planet="wasp39b", tp_mode="picaso_climate",
                met_x_solar=10.0, co_ratio=0.55)
    base.update(kw)
    return base


@pytest.fixture(autouse=True)
def _fp(monkeypatch):
    monkeypatch.setattr(forward, "_picaso_fingerprint",
                        lambda: {"picaso_version": "4.0.1",
                                 "chemgrid_sha1": "deadbeefdeadbeef"})
    monkeypatch.setattr(forward, "_picaso_climate_fingerprint",
                        lambda node, tio_vo: "cafecafecafecafe")


# --- v18 key surface --------------------------------------------------------

def test_vulcan_defaults_carry_inert_picaso_keys():
    cp = forward.canonical_params(_p())
    assert cp["version"] == 24  # v24 = inverse-square g(r) in the shared RT dtau
    assert cp["chem_provider"] == "vulcan"
    assert cp["picaso_version"] == ""
    assert cp["picaso_chemgrid_sha1"] == ""
    assert cp["picaso_climate_sha1"] == ""
    assert cp["picaso_ck_node"] == ""
    assert cp["tint_cl"] == 0.0 and cp["rfacv"] == 0.0
    assert cp["tio_vo"] is False and cp["climate_rcb"] == 0


def test_v17_to_v18_key_regression():
    # the v18 vulcan canonical dict is exactly the v17 dict plus the new keys
    new_keys = {"chem_provider", "picaso_version", "picaso_chemgrid_sha1",
                "picaso_climate_sha1", "tint_cl", "rfacv", "tio_vo",
                "climate_rcb", "picaso_ck_node"}
    cp = forward.canonical_params(_p())
    v17_keys = set(cp) - new_keys
    # frozen v17 canonical key list (from the v17 release)
    assert v17_keys == {
        "planet", "science_mode", "star_teff", "star_logg", "star_feh",
        "nz", "nu_pts", "yconv_cri", "rp_rjup", "gs_cgs", "rstar_rsun",
        "orbit_au", "sflux", "met_x_solar", "co_ratio", "kzz_mode", "kzz_x",
        "kzz_const", "kzz_kmax", "kzz_plev", "kzz_kdeep", "tp_mode",
        "tp_file", "tp_file_sha1", "Tirr", "Tint", "log_kappa",
        "log_gamma", "use_photo", "sl_angle_deg", "f_diurnal", "use_moldiff",
        "use_vm_mol", "use_rayleigh", "broadening", "rt_ptop_bar",
        "rt_integration", "rt_dit_res", "cloud_on", "log_kappa_cloud",
        "alpha_cloud", "mie_condensate", "mie_log_rg", "mie_sigmag",
        "mie_log_mmr", "use_condense", "use_settling", "diff_esc",
        "top_flux", "bot_flux", "extra_mols", "fisher_params", "jac_method",
        "version"}


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="chem_provider"):
        forward.canonical_params(_p(chem_provider="chimera"))


# --- picaso provider matrix -------------------------------------------------

def test_picaso_accepts_and_normalizes():
    cp = forward.canonical_params(_pp())
    assert cp["chem_provider"] == "picaso"
    assert cp["picaso_version"] == "4.0.1"
    assert cp["picaso_chemgrid_sha1"] == "deadbeefdeadbeef"
    # kinetics knobs normalized (equilibrium has none)
    assert cp["use_photo"] is False and cp["use_moldiff"] is False
    assert cp["use_vm_mol"] is False
    assert cp["kzz_mode"] == "const" and cp["kzz_const"] == 0.0
    assert cp["kzz_x"] == 1.0
    assert cp["yconv_cri"] == forward.YCONV_DEFAULT
    # photo-off normalization cascades (sflux/orbit pinned to system values)
    assert cp["sl_angle_deg"] == 0.0


@pytest.mark.parametrize("bad,match", [
    (dict(jac_method="ad"), "not differentiable"),
    (dict(tp_mode="file"), "tabulated profiles"),
    (dict(use_photo=True), "photochemistry"),
    (dict(use_moldiff=True), "molecular diffusion"),
    (dict(use_vm_mol=True), "advection"),
    (dict(use_condense=True), "kinetics feature"),
    (dict(use_settling=True), "boundary-condition"),
    (dict(diff_esc=["H"]), "boundary-condition"),
    (dict(kzz_mode="Pfunc"), "mixing profile"),
    (dict(fisher_params=["lnKzz"]), "no effect in equilibrium"),
    (dict(co_ratio=1.5), "Visscher"),
    (dict(extra_mols=["SO2"]), "SO2"),
])
def test_picaso_refusals(bad, match):
    with pytest.raises(ValueError, match=match):
        forward.canonical_params(_pp(**bad))


def test_picaso_fisher_menu_and_envelope():
    cp = forward.canonical_params(
        _pp(fisher_params=["lnZ", "dlnCO", "Tirr"]))
    assert sorted(cp["fisher_params"]) == ["Tirr", "dlnCO", "lnZ"]
    # the composition stencil must stay ON the tables: co near the 1.10 cap
    with pytest.raises(ValueError, match="stencil"):
        forward.canonical_params(
            _pp(co_ratio=1.05, fisher_params=["dlnCO"]))
    with pytest.raises(ValueError, match="stencil"):
        forward.canonical_params(
            _pp(met_x_solar=95.0, fisher_params=["lnZ"]))


def test_picaso_extras_accepted():
    cp = forward.canonical_params(_pp(extra_mols=["HCN", "NH3"]))
    assert forward.active_molecules(cp) == ["H2O", "CO2", "CO", "CH4", "H2S",
                                            "HCN", "NH3"]
    # v25: the hydrocarbon extras ride the Visscher gas columns under picaso
    cp = forward.canonical_params(_pp(extra_mols=["C2H4", "C2H6"]))
    assert forward.active_molecules(cp) == ["H2O", "CO2", "CO", "CH4", "H2S",
                                            "C2H4", "C2H6"]


def test_picaso_refuses_cs2():
    # CS2 is photochemical sulfur, VULCAN-only -- same refusal as SO2/S2/S8
    with pytest.raises(ValueError, match="SO2/S2/S8/CS2"):
        forward.canonical_params(_pp(extra_mols=["CS2"]))


# --- climate T-P mode matrix ------------------------------------------------

def test_climate_accepts_exact_node_both_providers():
    for prov in ("vulcan", "picaso"):
        cp = forward.canonical_params(_pc(chem_provider=prov))
        assert cp["picaso_ck_node"] == "feh1.0_co0.55"
        assert cp["picaso_climate_sha1"] == "cafecafecafecafe"
        assert cp["tint_cl"] == forward.TINT_CL_DEFAULT
        assert cp["rfacv"] == 0.5
        # transmission + climate KEEPS the star identity (climate consumes it)
        assert cp["star_teff"] > 0.0


@pytest.mark.parametrize("bad,match", [
    (dict(met_x_solar=7.0), "exactly ON"),          # off-node metallicity
    (dict(co_ratio=0.50), "exactly ON"),            # off-node C/O
    (dict(met_x_solar=100.0, co_ratio=1.10), "exactly ON"),  # unshipped node
    (dict(rfacv=0.3), "rfacv"),
    (dict(tint_cl=1000.0), "tint_cl"),
    (dict(climate_rcb=2), "climate_rcb"),
    (dict(jac_method="ad"), "not certified"),
])
def test_climate_refusals(bad, match):
    with pytest.raises(ValueError, match=match):
        forward.canonical_params(_pc(**bad))


def test_climate_tint_fisher_row_and_stencil():
    cp = forward.canonical_params(_pc(fisher_params=["Tint_cl"]))
    assert cp["fisher_params"] == ["Tint_cl"]
    with pytest.raises(ValueError, match="stencil"):
        forward.canonical_params(_pc(tint_cl=60.0,
                                     fisher_params=["Tint_cl"]))


def test_climate_keys_inert_outside_mode():
    cp = forward.canonical_params(_p(tint_cl=300.0, rfacv=1.0, tio_vo=True,
                                     climate_rcb=40))
    assert cp["tint_cl"] == 0.0 and cp["rfacv"] == 0.0
    assert cp["tio_vo"] is False and cp["climate_rcb"] == 0


def test_kzz_file_mode_needs_tp_file_still_holds_under_climate():
    with pytest.raises(ValueError, match="kzz_mode='file'"):
        forward.canonical_params(_pc(kzz_mode="file"))


# --- cache-key hygiene ------------------------------------------------------

def test_cache_fragmentation():
    base = forward.params_key(_pc())
    assert forward.params_key(_pc(chem_provider="picaso")) != base
    assert forward.params_key(_pc(tio_vo=True)) != base
    assert forward.params_key(_pc(tint_cl=250.0)) != base
    assert forward.params_key(_pc(rfacv=1.0)) != base
    assert forward.params_key(_pc(met_x_solar=1.0)) != base
    assert forward.params_key(_pc(climate_rcb=65)) != base


def test_picaso_keys_never_fragment_vulcan_cache():
    assert forward.params_key(_p()) == forward.params_key(
        _p(chem_provider="vulcan"))


# --- import hygiene ---------------------------------------------------------

def test_fast_path_never_imports_picaso():
    import jwst_tool.datacheck   # noqa: F401
    import jwst_tool.picaso_chem   # noqa: F401
    import jwst_tool.picaso_env   # noqa: F401
    forward.canonical_params(_p())
    assert "picaso" not in sys.modules
    assert "jax" not in sys.modules


# --- v18.1 GUI-review behavior fixes ----------------------------------------

def test_orbit_kept_under_climate_mode_despite_photo_off():
    # review finding 26: the climate solve consumes orbit_au (stellar
    # irradiation), so the photo-off normalization must not overwrite it
    cp = forward.canonical_params(_pc(chem_provider="picaso",
                                      orbit_au=0.1))
    assert cp["use_photo"] is False
    assert cp["orbit_au"] == 0.1               # kept: climate physics
    # outside climate mode the photolysis-only normalization still applies
    cp2 = forward.canonical_params(_pp(orbit_au=0.1))
    assert cp2["orbit_au"] == 0.04828          # normalized to the system value
    # and two climate requests differing only in orbit fragment the key
    assert (forward.params_key(_pc(chem_provider="picaso", orbit_au=0.1))
            != forward.params_key(_pc(chem_provider="picaso")))


def test_co_default_is_mode_aware():
    # review finding 21: a bare climate request must not refuse its own
    # default; the picaso provider defaults mid-cell
    cp = forward.canonical_params(dict(planet="wasp39b",
                                       tp_mode="picaso_climate"))
    assert cp["co_ratio"] == 0.55              # the 10x-solar node
    cp2 = forward.canonical_params(dict(planet="wasp39b",
                                        chem_provider="picaso",
                                        tp_mode="guillot"))
    assert cp2["co_ratio"] == 0.50             # mid-cell
    cp3 = forward.canonical_params(_p())
    assert cp3["co_ratio"] == round(forward.CO_BASELINE, 6)


def test_dlnco_refused_under_picaso_climate():
    # review finding 20: exact-node climate composition means the picaso
    # C/O stencil always straddles a table kink -- refuse at the API,
    # never mid-run
    with pytest.raises(ValueError, match="unavailable under the PICASO"):
        forward.canonical_params(_pc(chem_provider="picaso",
                                     fisher_params=["dlnCO"]))
    # fine under the VULCAN engine (its own chemistry differentiates)
    cp = forward.canonical_params(_pc(fisher_params=["dlnCO"]))
    assert cp["fisher_params"] == ["dlnCO"]
    # and fine under picaso OUTSIDE climate mode at the mid-cell default
    cp2 = forward.canonical_params(_pp(fisher_params=["dlnCO"]))
    assert cp2["fisher_params"] == ["dlnCO"]
