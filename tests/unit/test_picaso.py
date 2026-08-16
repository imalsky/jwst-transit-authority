"""PICASO provider tests, numpy-only (no picaso, no jax, no reference data):
canonical-param surface (fingerprint seams monkeypatched exactly), climate
cache-key / certificate / atm-table machinery, and the picaso_chem blend /
normalization / kink machinery on synthetic tables. Real-table behavior is
pinned by the env-gated live tests (node-exact vs native picaso at 4e-15 dex).
"""
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jwst_tool import forward
from jwst_tool import picaso_chem as pc
from jwst_tool import picaso_climate as pcl


@pytest.fixture(autouse=True)
def _fp(monkeypatch):
    monkeypatch.setenv(forward.PICASO_EXPERIMENTAL_ENV, "1")
    monkeypatch.setattr(forward, "_picaso_fingerprint",
                        lambda: {"picaso_version": "4.0.1",
                                 "chemgrid_sha1": "deadbeefdeadbeef"})
    monkeypatch.setattr(forward, "_picaso_climate_fingerprint",
                        lambda node, tio_vo: "cafecafecafecafe")


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


# --- provider key surface ---------------------------------------------------

def test_picaso_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(forward.PICASO_EXPERIMENTAL_ENV, raising=False)
    with pytest.raises(RuntimeError, match="uncertified"):
        forward.canonical_params(_pp())


def test_vulcan_defaults_carry_inert_picaso_keys():
    cp = forward.canonical_params(_p())
    assert cp["version"] == forward._VERSION
    assert cp["chem_provider"] == "vulcan"
    assert cp["picaso_version"] == ""
    assert cp["picaso_chemgrid_sha1"] == ""
    assert cp["picaso_climate_sha1"] == ""
    assert cp["picaso_ck_node"] == ""
    assert cp["tint_cl"] == 0.0 and cp["rfacv"] == 0.0
    assert cp["tio_vo"] is False and cp["climate_rcb"] == 0
    # explicit climate knobs outside climate mode are normalized back to
    # inert, so they can never fragment the vulcan cache
    cp2 = forward.canonical_params(_p(tint_cl=300.0, rfacv=1.0, tio_vo=True,
                                      climate_rcb=40))
    assert cp2["tint_cl"] == 0.0 and cp2["rfacv"] == 0.0
    assert cp2["tio_vo"] is False and cp2["climate_rcb"] == 0
    assert forward.params_key(_p()) == forward.params_key(
        _p(chem_provider="vulcan"))


def test_v17_to_v18_key_regression():
    # the vulcan canonical dict is exactly the frozen v17 key list plus the
    # provider keys
    new_keys = {"chem_provider", "picaso_version", "picaso_chemgrid_sha1",
                "picaso_climate_sha1", "tint_cl", "rfacv", "tio_vo",
                "climate_rcb", "picaso_ck_node", "network"}
    cp = forward.canonical_params(_p())
    v17_keys = set(cp) - new_keys
    assert v17_keys == {
        "planet", "science_mode", "star_teff", "star_logg", "star_feh",
        "nz", "nu_pts", "yconv_cri", "rp_rjup", "gs_cgs", "rstar_rsun",
        "orbit_au", "sflux", "met_x_solar", "co_ratio", "kzz_mode", "kzz_x",
        "kzz_const", "kzz_kmax", "kzz_plev", "kzz_kdeep", "tp_mode",
        "tp_file", "tp_file_sha1", "Tirr", "Tint", "log_kappa",
        "log_gamma", "use_photo", "sl_angle_deg", "f_diurnal", "use_moldiff",
        "use_vm_mol", "use_rayleigh", "broadening", "rt_ptop_bar",
        "rt_integration", "rt_dit_res",
        "opacity_mode",   # v28: correlated-k vs the biased sampled-lbl grid
        "cloud_on", "log_kappa_cloud",
        "alpha_cloud", "mie_condensate", "mie_log_rg", "mie_sigmag",
        "mie_log_mmr", "use_condense", "use_settling", "diff_esc",
        "top_flux", "bot_flux", "extra_mols", "fisher_params", "jac_method",
        "p_ref_bar",      # v26: the pressure where rp_rjup/gs_cgs are defined
        "p_btm_bar",      # v27: chemistry + RT column bottom
        "wo_mols",        # v32: the leave-one-out spectrum set
        "version"}


# --- picaso provider matrix -------------------------------------------------

def test_picaso_accepts_normalizes_and_takes_extras():
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
    cp = forward.canonical_params(_pp(extra_mols=["HCN", "NH3"]))
    assert forward.active_molecules(cp) == ["H2O", "CO2", "CO", "CH4", "H2S",
                                            "HCN", "NH3"]
    # hydrocarbon extras ride the Visscher gas columns under picaso.
    # C2H6 has no published ExoMolOP k-table, so it needs an explicit
    # opacity_mode="lbl"; under the default mode it is refused EARLY (v31)
    with pytest.raises(ValueError, match="no published ExoMolOP k-table"):
        forward.canonical_params(_pp(extra_mols=["C2H4", "C2H6"]))
    cp = forward.canonical_params(_pp(extra_mols=["C2H4", "C2H6"],
                                      opacity_mode="lbl"))
    assert forward.active_molecules(cp) == ["H2O", "CO2", "CO", "CH4", "H2S",
                                            "C2H4", "C2H6"]


def test_picaso_provider_refusal_matrix():
    # explicit kinetics/BC knobs, AD, tabulated profiles, out-of-envelope
    # composition and photochemical species all refuse loudly rather than
    # silently normalizing (refuse-if-explicit / normalize-if-default)
    with pytest.raises(ValueError, match="chem_provider"):
        forward.canonical_params(_p(chem_provider="chimera"))
    for bad, match in (
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
            # CS2 is photochemical sulfur, VULCAN-only (same class as SO2/S2/S8)
            (dict(extra_mols=["CS2"]), "SO2/S2/S8/CS2")):
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


# --- climate T-P mode matrix ------------------------------------------------

def test_climate_accepts_exact_node_and_tint_fisher_row():
    for prov in ("vulcan", "picaso"):
        cp = forward.canonical_params(_pc(chem_provider=prov))
        assert cp["picaso_ck_node"] == "feh1.0_co0.55"
        assert cp["picaso_climate_sha1"] == "cafecafecafecafe"
        assert cp["tint_cl"] == forward.TINT_CL_DEFAULT
        assert cp["rfacv"] == 0.5
        # transmission + climate KEEPS the star identity (climate consumes it)
        assert cp["star_teff"] > 0.0
    cp = forward.canonical_params(_pc(fisher_params=["Tint_cl"]))
    assert cp["fisher_params"] == ["Tint_cl"]
    with pytest.raises(ValueError, match="stencil"):
        forward.canonical_params(_pc(tint_cl=60.0,
                                     fisher_params=["Tint_cl"]))


def test_climate_refusal_matrix():
    # climate composition is EXACT-CK-NODE only; off-node values and
    # uncertified methods refuse at the API, never mid-solve
    for bad, match in (
            (dict(met_x_solar=7.0), "exactly ON"),          # off-node metallicity
            (dict(co_ratio=0.50), "exactly ON"),            # off-node C/O
            (dict(met_x_solar=100.0, co_ratio=1.10), "exactly ON"),  # unshipped
            (dict(rfacv=0.3), "rfacv"),
            (dict(tint_cl=1000.0), "tint_cl"),
            (dict(climate_rcb=2), "climate_rcb"),
            (dict(jac_method="ad"), "not certified"),
            (dict(kzz_mode="file"), "kzz_mode='file'")):    # still needs tp file
        with pytest.raises(ValueError, match=match):
            forward.canonical_params(_pc(**bad))


def test_climate_cache_fragmentation():
    base = forward.params_key(_pc())
    for kw in (dict(chem_provider="picaso"), dict(tio_vo=True),
               dict(tint_cl=250.0), dict(rfacv=1.0),
               dict(met_x_solar=1.0), dict(climate_rcb=65)):
        assert forward.params_key(_pc(**kw)) != base, kw


# --- import hygiene ---------------------------------------------------------

def test_fast_path_never_imports_picaso():
    """The light path (canonical_params, datacheck, the picaso shims) must not
    drag in picaso or jax. Run in a SUBPROCESS: sys.modules is global to the
    pytest session, so in-process the assertion is vacuous whenever an earlier
    test touched the engine -- a fresh interpreter is the only place the claim
    means anything."""
    import subprocess
    src = (
        "import sys\n"
        "import jwst_tool.datacheck, jwst_tool.picaso_chem, jwst_tool.picaso_env\n"
        "from jwst_tool import forward\n"
        "forward.canonical_params(dict(planet='wasp39b', tp_mode='guillot',\n"
        "                              kzz_mode='const', kzz_const=1.0e9))\n"
        "assert 'picaso' not in sys.modules, 'picaso imported on the light path'\n"
        "assert 'jax' not in sys.modules, 'jax imported on the light path'\n"
    )
    r = subprocess.run([sys.executable, "-c", src], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[-2000:]


# --- GUI-review behavior fixes ----------------------------------------------

def test_gui_review_fixes_orbit_co_default_and_dlnco():
    # the climate solve consumes orbit_au (stellar irradiation), so the
    # photo-off normalization must not overwrite it under climate mode
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
    # C/O default is mode-aware: a bare climate request must not refuse its
    # own default (exact node); the picaso provider defaults mid-cell
    cp = forward.canonical_params(dict(planet="wasp39b",
                                       tp_mode="picaso_climate"))
    assert cp["co_ratio"] == 0.55              # the 10x-solar node
    cp2 = forward.canonical_params(dict(planet="wasp39b",
                                        chem_provider="picaso",
                                        tp_mode="guillot"))
    assert cp2["co_ratio"] == 0.50             # mid-cell
    assert forward.canonical_params(
        _p())["co_ratio"] == forward.CO_DEFAULT   # 0.55 (v32 default)
    # exact-node climate composition means the picaso C/O stencil always
    # straddles a table kink: dlnCO refuses at the API, never mid-run
    with pytest.raises(ValueError, match="unavailable under the PICASO"):
        forward.canonical_params(_pc(chem_provider="picaso",
                                     fisher_params=["dlnCO"]))
    # fine under the VULCAN engine (its own chemistry differentiates), and
    # fine under picaso OUTSIDE climate mode at the mid-cell default
    assert forward.canonical_params(
        _pc(fisher_params=["dlnCO"]))["fisher_params"] == ["dlnCO"]
    assert forward.canonical_params(
        _pp(fisher_params=["dlnCO"]))["fisher_params"] == ["dlnCO"]


# --- climate cache key + subset ---------------------------------------------

def _cp(**kw):
    base = dict(tp_mode="picaso_climate", picaso_version="4.0.1",
                picaso_climate_sha1="cafecafecafecafe",
                picaso_ck_node="feh1.0_co0.55", tio_vo=False,
                tint_cl=200.0, rfacv=0.5, climate_rcb=60,
                rp_rjup=1.279, gs_cgs=422.0, rstar_rsun=0.932,
                orbit_au=0.04828, star_teff=5485.0, star_logg=4.5,
                star_feh=0.0,
                # keys that must NOT enter the climate cache
                chem_provider="vulcan", nz=100, nu_pts=4000,
                rt_ptop_bar=1e-8, broadening="air")
    base.update(kw)
    return base


def test_climate_subset_and_key_scope():
    sub = pcl.climate_subset(_cp())
    # provider + RT/chem-resolution knobs are EXCLUDED (the converged climate
    # is provider-independent and knows nothing about the RT)
    for k in ("chem_provider", "nz", "nu_pts", "rt_ptop_bar", "broadening"):
        assert k not in sub, k
    # everything the solve consumes is INCLUDED
    for k in ("picaso_version", "picaso_climate_sha1", "picaso_ck_node",
              "tio_vo", "tint_cl", "rfacv", "climate_rcb", "rp_rjup",
              "gs_cgs", "rstar_rsun", "orbit_au", "star_teff", "star_logg",
              "star_feh"):
        assert k in sub, k
    assert sub["_climate_version"] == pcl._CLIMATE_VERSION
    base = pcl.climate_key(_cp())
    assert pcl.climate_key(_cp(tio_vo=True)) != base
    assert pcl.climate_key(_cp(picaso_climate_sha1="0" * 16)) != base
    assert pcl.climate_key(_cp(), tint_override=215.0) != base
    # provider + RT knobs shared: same key
    assert pcl.climate_key(_cp(chem_provider="picaso", nz=150)) == base
    with pytest.raises(ValueError):
        pcl.climate_subset(_cp(tp_mode="guillot"))


# --- certificate revalidation on load ---------------------------------------

def _good_cert(key="k1"):
    return {"climate_key": key, "_climate_version": pcl._CLIMATE_VERSION,
            "converged": True, "flux_toa_over_tidal": 1.5e-6,
            "cvz_locs": [0, 60, 89, 0, 0, 0]}


def _write_cache(tmp_path, key, cert, T=None):
    p = tmp_path / f"{key}.npz"
    P = np.logspace(-6, 2, 10)
    if T is None:
        T = 900.0 * (P / P[0]) ** 0.06     # gentle, in-envelope gradient
    np.savez_compressed(p, pressure_bar=P, temperature_K=T,
                        dtdp=np.zeros(10), cert_json=json.dumps(cert),
                        provenance_json=json.dumps({}))
    return p


def test_load_accepts_only_matching_certified_entries_and_regates(tmp_path):
    p = _write_cache(tmp_path, "k1", _good_cert("k1"))
    assert pcl._load(p, "k1") is not None
    assert pcl._load(p, "OTHER") is None                  # key mismatch
    stale = dict(_good_cert("k2"), _climate_version=pcl._CLIMATE_VERSION - 1)
    assert pcl._load(_write_cache(tmp_path, "k2", stale), "k2") is None
    uncert = dict(_good_cert("k3"), converged=False)
    assert pcl._load(_write_cache(tmp_path, "k3", uncert), "k3") is None
    assert pcl._load(tmp_path / "absent.npz", "k4") is None
    # unreadable/foreign file: recompute, never trust it
    bad = tmp_path / "k5.npz"
    bad.write_bytes(b"not an npz")
    assert pcl._load(bad, "k5") is None
    # loading re-runs every gate evaluable from the stored data; a cert that
    # SAYS converged but fails a metric or structural gate is treated absent
    no_flux = {k: v for k, v in _good_cert("k6").items()
               if k != "flux_toa_over_tidal"}
    assert pcl._load(_write_cache(tmp_path, "k6", no_flux), "k6") is None
    bad_flux = dict(_good_cert("k7"), flux_toa_over_tidal=0.5)
    assert pcl._load(_write_cache(tmp_path, "k7", bad_flux), "k7") is None
    conv_top = dict(_good_cert("k8"), cvz_locs=[0, 2, 89, 0, 0, 0])
    assert pcl._load(_write_cache(tmp_path, "k8", conv_top), "k8") is None
    P = np.logspace(-6, 2, 10)
    runaway = 500.0 * (P / P[0]) ** 0.7                  # gradient breach
    assert pcl._load(_write_cache(tmp_path, "k9", _good_cert("k9"),
                                  T=runaway), "k9") is None
    nan_T = np.full(10, 1200.0)
    nan_T[4] = np.nan
    assert pcl._load(_write_cache(tmp_path, "kA", _good_cert("kA"),
                                  T=nan_T), "kA") is None


def test_lock_file_is_never_unlinked(tmp_path, monkeypatch):
    # unlinking a flock'd path creates the two-inode double-lock race, so
    # the lock file must persist across runs
    monkeypatch.setattr(pcl, "CLIMATE_CACHE", tmp_path)
    P = np.logspace(np.log10(pcl.CLIMATE_P_SPAN_BAR[0]),
                    np.log10(pcl.CLIMATE_P_SPAN_BAR[1]),
                    pcl.CLIMATE_N_LEVELS)
    T = 900.0 + 1900.0 * (np.log10(P) + 6.0) / 8.5
    fake = dict(pressure=P, temperature=T, converged=True,
                cvz_locs=[0, 60, 89, 0, 0, 0], _wall_s=0.1,
                flux_balance={"flux_net": np.array([1e-4]),
                              "tidal": np.array([9e4])})
    calls = {"n": 0}

    def fake_solve(cp, tint, log):
        calls["n"] += 1
        return dict(fake)

    monkeypatch.setattr(pcl, "_solve", fake_solve)
    cp = _cp()
    clim = pcl.get_or_run(cp, lambda *a: None)
    key = pcl.climate_key(cp)
    lock = tmp_path / f"{key}.lock"
    assert lock.is_file(), "lock file must persist after the solve"
    meta = json.loads(lock.read_text())
    assert meta["pid"] > 0
    # second call: cache hit (no new solve), lock file still present
    clim2 = pcl.get_or_run(cp, lambda *a: None)
    assert calls["n"] == 1
    assert np.array_equal(clim.T, clim2.T)
    assert lock.is_file()


# --- certification gates (pure logic; no solver) ----------------------------

def _out(**kw):
    P = np.logspace(-6, np.log10(300.0), 91)
    T = 900.0 + 2000.0 * (np.log10(P) + 6.0) / 8.5
    base = dict(pressure=P, temperature=T, converged=True,
                cvz_locs=[0, 60, 89, 0, 0, 0],
                flux_balance={"flux_net": np.array([1e-4]),
                              "tidal": np.array([90699.2])})
    base.update(kw)
    return base


def test_certify_passes_sane_and_refuses_each_gate():
    cert = pcl._certify(_out(), "k")
    assert cert["converged"] and cert["flux_toa_over_tidal"] < 1e-6
    with pytest.raises(RuntimeError, match="did NOT converge"):
        pcl._certify(_out(converged=False), "k")
    with pytest.raises(RuntimeError, match="flux balance"):
        pcl._certify(_out(flux_balance={"flux_net": np.array([1e4]),
                                        "tidal": np.array([9e4])}), "k")
    hot = _out()
    hot["temperature"] = 500.0 * (hot["pressure"] / 1e-6) ** 0.7
    with pytest.raises(RuntimeError, match="gradient"):
        pcl._certify(hot, "k")
    with pytest.raises(RuntimeError, match="model top"):
        pcl._certify(_out(cvz_locs=[0, 2, 89, 0, 0, 0]), "k")


# --- atm-table writer -------------------------------------------------------

def test_write_atm_table_and_deterministic_guillot_guess(tmp_path):
    P = np.logspace(-6, np.log10(300.0), 91)
    T = 850.0 + 1900.0 * (np.log10(P) + 6.0) / 8.5      # ~2750 K at 300 bar
    path = tmp_path / "atm.txt"
    pcl._write_atm_table(P, T, path, forward.CHEM_P_SPAN_DYN[1])
    tab = forward._read_tp_table(path)                   # the engine's parser
    assert tab["P_dyn"].max() == pytest.approx(forward.CHEM_P_SPAN_DYN[1])
    assert np.all(np.diff(tab["P_dyn"]) < 0)             # descending (bottom first)
    # the bottom-row T is the log-P interpolant at exactly the chem bottom
    want = float(np.interp(np.log(forward.CHEM_P_SPAN_DYN[1]),
                           np.log(P * 1e6), T))
    assert tab["T"][0] == pytest.approx(want, abs=0.01)
    # an out-of-window profile is refused, never written
    T_hot = np.full(91, 1500.0)
    T_hot[P > 1.0] = 3400.0                              # too hot at depth
    with pytest.raises(RuntimeError, match="modelable window"):
        pcl._write_atm_table(P, T_hot, tmp_path / "bad.txt",
                             forward.CHEM_P_SPAN_DYN[1])
    # the guillot initial guess is bit-deterministic and monotone with depth
    p = np.logspace(-6, 2, 50)
    a = pcl.guillot_guess(p, 422.0, 200.0, 1643.4)
    assert np.array_equal(a, pcl.guillot_guess(p.copy(), 422.0, 200.0, 1643.4))
    assert np.all(np.diff(a) >= 0)


# --- exact composition transforms -------------------------------------------

def test_comp_step_exact_exponential_and_kink_metric():
    met, co = pc.comp_step(10.0, 0.55, "lnZ", 0.1)
    assert met == pytest.approx(10.0 * np.exp(0.1), rel=1e-15)
    assert co == 0.55
    met, co = pc.comp_step(10.0, 0.55, "dlnCO", -0.04)
    assert met == 10.0
    assert co == pytest.approx(0.55 * np.exp(-0.04), rel=1e-15)
    with pytest.raises(ValueError):
        pc.comp_step(10.0, 0.55, "lnKzz", 0.1)
    j = np.array([1.0, 2.0, -1.0])
    assert pc.kink_metric(j, j, j) == 0.0
    assert pc.kink_metric(j, 2.0 * j, j) == pytest.approx(1.0)
    z = np.zeros(3)
    assert pc.kink_metric(z, z, z) == 0.0
    assert np.isinf(pc.kink_metric(z, z + 1.0, z))


# --- node geometry ----------------------------------------------------------

def test_ck_node_geometry_and_bracketing():
    assert len(pc.CK_NODES_AVAILABLE) == 70               # the shipped pairs
    assert "feh1.0_co0.55" in pc.CK_NODES_AVAILABLE
    assert "feh2.0_co0.14" not in pc.CK_NODES_AVAILABLE   # extreme-met gap
    assert "feh-2.0_co1.10" not in pc.CK_NODES_AVAILABLE
    lo, hi, w = pc._bracket(pc.CO_NODES, 0.50, "C/O")
    assert (pc.CO_NODES[lo], pc.CO_NODES[hi]) == (0.46, 0.55)
    assert w == pytest.approx((0.50 - 0.46) / (0.55 - 0.46))
    # exact top edge: weight 1 in the last cell
    lo, hi, w = pc._bracket(pc.CO_NODES, 1.10, "C/O")
    assert (pc.CO_NODES[lo], pc.CO_NODES[hi]) == (0.82, 1.10)
    assert w == pytest.approx(1.0)
    with pytest.raises(ValueError, match="outside the grid"):
        pc._bracket(pc.CO_NODES, 1.2, "C/O")
    cell = pc.bracketing_cells(10.0 ** 0.6, 0.50)
    assert cell["nodes"] == [["feh0.5_co0.46", "feh0.5_co0.55"],
                             ["feh0.7_co0.46", "feh0.7_co0.55"]]


# --- masses + stoichiometry -------------------------------------------------

def test_species_masses_and_atom_counts():
    assert pc.species_mass("H2O") == pytest.approx(18.015, abs=0.01)
    assert pc.species_mass("e-") == pytest.approx(5.486e-4, rel=1e-3)
    assert pc.species_mass("C-gr_l_s") == pytest.approx(12.011, abs=0.001)
    # ion mass = parent neutral (electron mass below the tabulated precision)
    assert pc.species_mass("Fe+") == pc.species_mass("Fe")
    sp = ["CO2", "CH4", "H2O", "C-gr_l_s", "e-"]
    assert list(pc._atom_counts(sp, "C")) == [1, 1, 0, 1, 0]
    assert list(pc._atom_counts(sp, "O")) == [2, 0, 1, 0, 0]


# --- synthetic-table blend + evaluation -------------------------------------

_SP = ["e-", "H2", "He", "H2O", "CH4", "CO", pc.GRAPHITE]


def _tab(node, const=None, lin=None):
    """Synthetic node table: log10 abundance either constant per species or
    linear in (1/T, log10 P)."""
    T = np.linspace(75.0, 6000.0, 25)
    Pl = np.linspace(-6.0, 4.0, 9)
    cube = np.zeros((25, 9, len(_SP)))
    for j in range(len(_SP)):
        if lin is not None:
            a, b, c = lin[j]
            cube[:, :, j] = (a + b * (1.0 / T)[:, None]
                             + c * Pl[None, :])
        else:
            cube[:, :, j] = const[j]
    return SimpleNamespace(node=node, T=T, Plog=Pl, cube=cube, species=_SP,
                           suspect_cells=[], corrections_applied=[])


def test_blend_cubes_bilinear_recovery_and_species_guard():
    # a field linear in (feh, co) is recovered exactly by the blend
    def cube_at(feh, co):
        return _tab("x", const=[feh + 2.0 * co] * len(_SP)).cube
    tabs = [[SimpleNamespace(node="a", species=_SP, cube=cube_at(0.5, 0.46)),
             SimpleNamespace(node="b", species=_SP, cube=cube_at(0.5, 0.55))],
            [SimpleNamespace(node="c", species=_SP, cube=cube_at(0.7, 0.46)),
             SimpleNamespace(node="d", species=_SP, cube=cube_at(0.7, 0.55))]]
    wf, wc = 0.25, 0.5
    out = pc.blend_cubes(tabs, wf, wc)
    feh = 0.5 + wf * 0.2
    co = 0.46 + wc * 0.09
    assert np.allclose(out, feh + 2.0 * co, atol=1e-14)
    # mismatched species columns refuse, never blend positionally
    t1 = _tab("a", const=[0.0] * len(_SP))
    t2 = _tab("b", const=[0.0] * len(_SP))
    t2.species = list(reversed(_SP))
    with pytest.raises(RuntimeError, match="species columns"):
        pc.blend_cubes([[t1, t2], [t1, t1]], 0.5, 0.5)


def test_eval_cube_tp_exact_on_linear_field_and_refuses_outside():
    lin = [(0.0, 500.0, 0.25)] * len(_SP)
    tab = _tab("x", lin=lin)
    T_prof = np.array([400.0, 1234.5, 2999.0])
    p_bar = np.array([1e-4, 1.0, 100.0])
    out = pc._eval_cube_tp(tab.cube, tab.T, tab.Plog, T_prof, p_bar)
    want = 500.0 / T_prof[:, None] + 0.25 * np.log10(p_bar)[:, None]
    assert np.allclose(out, want, rtol=1e-12)
    with pytest.raises(ValueError, match="refusing to extrapolate"):
        pc._eval_cube_tp(tab.cube, tab.T, tab.Plog,
                         np.array([50.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="refusing to extrapolate"):
        pc._eval_cube_tp(tab.cube, tab.T, tab.Plog,
                         np.array([500.0]), np.array([1e5]))


def _patch_tables(monkeypatch, const):
    monkeypatch.setattr(pc, "load_node_table",
                        lambda node: _tab(node, const=const))


def test_evaluate_contract_floor_and_normalization_gate(monkeypatch):
    # gas sums ~ 0.5 (H2) + 0.1 (He) = 0.6 < GAS_SUM_MIN -> refuse
    _patch_tables(monkeypatch, [np.log10(v) for v in
                                (1e-9, 0.5, 0.1, 1e-4, 1e-6, 1e-4, 1e-9)])
    with pytest.raises(RuntimeError, match="GAS_SUM_MIN"):
        pc.evaluate(10.0, 0.55, np.array([800.0, 1000.0]),
                    np.array([1e-3, 1.0]))
    # output contract on a healthy table
    _patch_tables(monkeypatch, [np.log10(v) for v in
                                (1e-9, 0.70, 0.14, 5e-3, 1e-3, 4e-3, 1e-30)])
    st = pc.evaluate(10.0, 0.55, np.array([800.0, 2500.0]),
                     np.array([1e-3, 1.0]))
    assert st.species[-1] == pc.GRAPHITE_OUT       # renamed for the RT mask
    assert st.y.shape == (2, len(_SP))
    assert st.cert["gas_sum_min"] == pytest.approx(0.85, abs=0.01)
    # the graphite column is excluded from the gas sum
    assert st.pre_norm_sum[0] == pytest.approx(
        st.y[0].sum() - st.y[0][-1], rel=1e-12)
    # realized gas C/O: CH4 + CO carbons over H2O + CO oxygens
    y = st.y[1]
    want = (y[4] + y[5]) / (y[3] + y[5])
    assert st.cert["realized_gas_co_hotT"] == pytest.approx(want, rel=1e-6)
    assert st.cert["n_floored_entries"] == 0
    assert st.species_masses[1] == pytest.approx(2.016, abs=0.01)
    # floor masking: a 1e-50 entry becomes an EXACT zero and is counted
    _patch_tables(monkeypatch, [np.log10(v) for v in
                                (1e-9, 0.84, 0.15, 5e-3, 1e-50, 4e-3, 1e-30)])
    st = pc.evaluate(10.0, 0.55, np.array([800.0]), np.array([1.0]))
    assert st.y[0, _SP.index("CH4")] == 0.0
    assert st.cert["n_floored_entries"] >= 1


def test_suspect_cell_bookkeeping_and_isolated_refusal(monkeypatch):
    const = [np.log10(v) for v in (1e-9, 0.84, 0.15, 5e-3, 1e-3, 4e-3, 1e-30)]
    # NON-isolated (systematic) suspects: flagged, never refused
    tab = _tab("x", const=const)
    tab.suspect_cells = [(900.0, -3.0, 0.75, False), (5000.0, 3.0, 0.8, False)]
    monkeypatch.setattr(pc, "load_node_table", lambda node: tab)
    st = pc.evaluate(10.0, 0.55, np.array([850.0, 950.0]),
                     np.array([5e-4, 2e-3]))
    hits = st.cert["suspect_cells_in_span"]
    assert [900.0, -3.0, 0.75, False] in hits      # inside the profile box
    assert [5000.0, 3.0, 0.8, False] not in hits   # far outside it
    # an ISOLATED suspect (clean T-neighbors = point corruption) inside the
    # span is refused, never renormalized through
    tab2 = _tab("x", const=const)
    tab2.suspect_cells = [(900.0, -3.0, 0.746, True)]
    monkeypatch.setattr(pc, "load_node_table", lambda node: tab2)
    with pytest.raises(RuntimeError, match="ISOLATED anomalous gas sum"):
        pc.evaluate(10.0, 0.55, np.array([850.0, 950.0]),
                    np.array([5e-4, 2e-3]))
    # the SAME cell outside the span: no refusal
    st = pc.evaluate(10.0, 0.55, np.array([2400.0, 2600.0]),
                     np.array([5e-4, 2e-3]))
    assert st.cert["suspect_cells_in_span"] == []


def _write_node_file(path, corrupt=False):
    """A shape-valid synthetic Visscher node file (101 T x 21 P, the real
    header) with an optionally-corrupted row at (900 K, logP=-5.5)."""
    real_header = ("T(K)  P(bar)  " + "  ".join(
        s if s != pc.GRAPHITE_OUT else pc.GRAPHITE
        for s in pc.SPECIES_ELEMENTS))
    n_sp = len(pc.SPECIES_ELEMENTS)
    T = np.linspace(pc.TABLE_T_K[0], pc.TABLE_T_K[1], pc.N_T)
    T[np.argmin(np.abs(T - 900.0))] = 900.0
    Pl = np.linspace(*pc.TABLE_P_LOGBAR, pc.N_P)
    Pl[np.argmin(np.abs(Pl - -5.5))] = -5.5
    names = list(pc.SPECIES_ELEMENTS)
    base = np.full(n_sp, 1e-50)
    base[names.index("H2")] = 0.83
    base[names.index("He")] = 0.165
    base[names.index("H2O")] = 5e-3
    rows = []
    for t in T:
        for p in Pl:
            r = base.copy()
            if corrupt and t == 900.0 and p == -5.5:
                r = r * 0.747
            rows.append([t, p] + r.tolist())
    with open(path, "w") as fh:
        fh.write(real_header + "\n")
        for r in rows:
            fh.write("  ".join("%.6e" % v for v in r) + "\n")
    raw = np.loadtxt(path, skiprows=1)
    i = int(np.where((raw[:, 0] == 900.0) & (raw[:, 1] == -5.5))[0][0])
    return hashlib.sha1(raw[i].tobytes()).hexdigest()[:16]


def test_registered_correction_applied_and_content_guarded(tmp_path,
                                                           monkeypatch):
    good = tmp_path / "sonora_2121grid_feh0.0_co0.55.txt"
    sha = _write_node_file(good, corrupt=True)
    monkeypatch.setattr("jwst_tool.picaso_env.chem_node_path",
                        lambda node, root=None: good)
    pc.load_node_table.cache_clear()
    # (a) matching registered hash -> corrected, no suspects
    monkeypatch.setitem(pc.KNOWN_TABLE_CORRECTIONS, "synthetic",
                        ({"T": 900.0, "logP": -5.5, "corrupt_row_sha1": sha,
                          "method": "T-neighbor log-mean", "why": "test"},))
    tab = pc.load_node_table("synthetic")
    assert len(tab.corrections_applied) == 1
    assert tab.corrections_applied[0]["row_sha1"] == sha
    assert tab.suspect_cells == []
    iT = int(np.argmin(np.abs(tab.T - 900.0)))
    iP = int(np.argmin(np.abs(tab.Plog - -5.5)))
    j = tab.species.index("H2")
    assert tab.cube[iT, iP, j] == pytest.approx(
        0.5 * (tab.cube[iT - 1, iP, j] + tab.cube[iT + 1, iP, j]))
    # (b) hash mismatch (upstream "fixed" the row differently) -> NO-OP
    # correction; the still-corrupt row surfaces as an ISOLATED suspect
    pc.load_node_table.cache_clear()
    monkeypatch.setitem(pc.KNOWN_TABLE_CORRECTIONS, "synthetic",
                        ({"T": 900.0, "logP": -5.5,
                          "corrupt_row_sha1": "0" * 16,
                          "method": "T-neighbor log-mean", "why": "test"},))
    tab2 = pc.load_node_table("synthetic")
    assert tab2.corrections_applied == []
    assert len(tab2.suspect_cells) == 1
    assert tab2.suspect_cells[0][3] is True        # isolated
    pc.load_node_table.cache_clear()
