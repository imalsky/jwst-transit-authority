"""Inputs the tool cannot model are refused, never clipped or dropped, and what a run leaves out rides its result."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
from param_helpers import _custom, _table

from jwst_tool import forward, jacobian, picaso_engine as pe, share_config

# Guillot T_irr defaults (K) of every registry planet, transmission and emission: removing the clip moved none.
_REGISTRY_TIRR = {
    "wasp39b": (1640.0, 1960.0),
    "hd189733b": (1680.0, 2000.0),
    "hd209458b": (2080.0, 2480.0),
    "wasp107b": (1040.0, 1240.0),
}


def test_the_tirr_default_is_refused_outside_its_range_never_clipped(tmp_path):
    for planet, defaults in _REGISTRY_TIRR.items():
        for mode, want in zip(("transmission", "emission"), defaults, strict=True):
            cp = forward.canonical_params(dict(planet=planet, science_mode=mode, tp_mode="guillot"))
            assert cp["Tirr"] == want, (planet, mode)
    cool = _custom(star_teff=3300.0, rstar_rsun=0.30, orbit_au=0.05)  # sqrt(2) T_eq = 560 K
    with pytest.raises(ValueError, match=r"default Tirr .* is 560 K .* \[800, 2500\] K\. Set Tirr explicitly"):
        forward.canonical_params(cool)
    with pytest.raises(ValueError, match="Tirr=2600 K is outside"):
        forward.canonical_params(dict(planet="wasp39b", tp_mode="guillot", Tirr=2600.0))
    # T_irr is inert outside Guillot mode, so the cool system runs on a table
    table = dict(tp_mode="file", tp_file="upload", tp_file_path=str(_table(tmp_path)))
    assert forward.canonical_params(dict(cool, **table))["Tirr"] == 0.0


_GUILLOT = dict(planet="wasp39b", tp_mode="guillot")


@pytest.mark.parametrize(
    ("key", "bad", "extra"),
    [
        ("gs_cgs", -422.0, {}),
        ("gs_cgs", float("nan"), {}),
        ("rp_rjup", -1.0, {}),
        ("rp_rjup", float("inf"), {}),
        ("rstar_rsun", 0.0, {}),
        ("orbit_au", -0.05, {}),
        ("orbit_au", float("nan"), {}),
        ("Tirr", float("nan"), {}),
        ("Tint", -100.0, {}),
        ("Tint", float("nan"), {}),
        ("log_kappa_cloud", float("nan"), {"cloud_on": True}),
        ("alpha_cloud", -1.0, {"cloud_on": True}),
        ("co_ratio", float("nan"), {}),
    ],
)
def test_a_negative_or_non_finite_input_is_refused_by_name(key, bad, extra):
    with pytest.raises(ValueError, match=key):
        forward.canonical_params({**_GUILLOT, **extra, key: bad})


# Each sidebar widget and the canonical key it feeds: (widget, key, widget value -> key units, settings making it live)
_WIDGETS = [
    ("teff", "star_teff", float, {"science_mode": "emission"}),
    ("logg", "star_logg", float, {"science_mode": "emission"}),
    ("feh", "star_feh", float, {"science_mode": "emission"}),
    ("rp", "rp_rjup", float, {}),
    ("g", "gs_cgs", lambda v: 100.0 * v, {}),
    ("rstar", "rstar_rsun", float, {}),
    ("a", "orbit_au", float, {}),
    ("tirr", "Tirr", float, {}),
    ("tint", "Tint", float, {}),
    ("lk", "log_kappa", float, {}),
    ("lg", "log_gamma", float, {}),
    ("kzz", "kzz_const", lambda v: 10.0**v, {}),
    ("kzkmax", "kzz_kmax", lambda v: 10.0**v, {"kzz_mode": "Pfunc"}),
    ("kzplev", "kzz_plev", lambda v: 10.0**v, {"kzz_mode": "Pfunc"}),
    ("kzkdeep", "kzz_kdeep", lambda v: 10.0**v, {"kzz_mode": "JM16"}),
    ("kzzx", "kzz_x", lambda v: 10.0**v, {}),
    ("sza", "sl_angle_deg", float, {}),
    ("fdiur", "f_diurnal", float, {}),
    ("ck", "log_kappa_cloud", float, {"cloud_on": True}),
    ("ca", "alpha_cloud", float, {"cloud_on": True}),
    ("rtptop", "rt_ptop_bar", float, {}),
    ("pref", "p_ref_bar", float, {"rt_ptop_bar": 1.0e-9, "p_btm_bar": 300.0}),  # the widest RT grid
]


@pytest.mark.parametrize(("widget", "key", "to_key", "extra"), _WIDGETS)
def test_the_api_accepts_exactly_the_widget_range(widget, key, to_key, extra):
    """The GUI widget and canonical_params read one range: both ends pass; a value a millionth outside, which the
    canonical rounding would carry back inside, and one percent outside are both refused."""
    lo, hi = share_config.WIDGET_RANGES[widget][:2]
    pad = 0.01 * (hi - lo)
    for v in (lo, hi):
        forward.canonical_params({**_GUILLOT, **extra, key: to_key(v)})
    k_lo, k_hi = to_key(lo), to_key(hi)
    just_outside = (k_lo - 1.0e-6 * max(abs(k_lo), 1.0), k_hi + 1.0e-6 * max(abs(k_hi), 1.0))
    for value in (*just_outside, to_key(lo - pad), to_key(hi + pad)):
        with pytest.raises(ValueError, match=key):
            forward.canonical_params({**_GUILLOT, **extra, key: value})


def test_the_rendered_widgets_offer_exactly_the_shared_ranges():
    """Each sidebar number input, as Streamlit builds it, carries the WIDGET_RANGES bounds the API test above reads."""
    pytest.importorskip("streamlit")
    from app_helpers import run_app

    at = run_app()
    at.selectbox(key="n0_planet").set_value("custom").run()
    at.selectbox(key="n0_custom_tp").set_value("guillot").run()
    found = {}

    def collect():
        for w in at.number_input:
            found.setdefault(w.key.rsplit("_", 1)[-1], (w.min, w.max))

    collect()
    for mode in ("Pfunc", "JM16"):
        at.selectbox(key="n0_custom_kzzmode").set_value(mode).run()
        collect()
    at.checkbox(key="n0_cloud").check().run()
    collect()
    for widget, *_ in _WIDGETS:
        assert widget in found, f"no rendered widget for {widget}"
        assert found[widget] == pytest.approx(share_config.WIDGET_RANGES[widget][:2]), widget


@pytest.mark.parametrize(
    ("params", "match"),
    [
        (dict(nz=150.9), r"nz=150.9 is not a whole number"),
        (dict(nz=99.5), r"nz=99.5 is not a whole number"),
        (dict(rp_rjup=0.09999), "rp_rjup"),
        (dict(gs_cgs=99.99), "gs_cgs"),
        (dict(Tirr=799.999), "Tirr"),
        (dict(log_gamma=0.3001), "log_gamma"),
        (dict(met_x_solar=100.00001), "met_x_solar"),
        (dict(cloud_on=True, alpha_cloud=-0.001), "alpha_cloud"),
    ],
)
def test_the_request_value_is_checked_before_rounding(params, match):
    with pytest.raises(ValueError, match=match):
        forward.canonical_params({**_GUILLOT, **params})


@pytest.mark.parametrize(
    "setting",
    [
        dict(yconv_cri=1.0e-3),
        dict(yconv_min=0.05),
        dict(sflux="Gueymard_solar.txt"),
        dict(sl_angle_deg=60.0),
        dict(f_diurnal=0.5),
        dict(kzz_x=2.0),
        dict(kzz_kmax=1.0e6),
        dict(kzz_plev=0.1),
        dict(kzz_kdeep=1.0e6),
    ],
)
def test_picaso_refuses_a_vulcan_only_setting_by_name(setting):
    (key,) = setting
    with pytest.raises(ValueError, match=rf"{key}=.* are VULCAN-JAX settings"):
        forward.canonical_params(dict(planet="wasp39b", chem_provider="picaso", **setting))


def test_left_out_absorbers_an_unstated_tint_and_picaso_retries_ride_the_result():
    pytest.importorskip("pandas")  # the result page's own import; the unit CI lane installs no GUI stack
    from jwst_tool import gui_results

    stated = forward.canonical_params(dict(planet="wasp39b", tp_mode="guillot", Tint=300.0))
    base = {
        "unmodeled": np.array([], dtype="U32"),
        "photo_escalated": np.array([], dtype="U48"),
        "rate_rows_outside": np.array([0]),
        "rate_rows_total": np.array([0]),
        "params_json": np.array(json.dumps(stated)),
    }
    vulcan = dict(base, chem_provider=np.array("vulcan"), unlisted_core=np.array(["HCN", "H2S"], dtype="U8"))
    caveats = gui_results._model_caveats(vulcan)
    assert any(c.startswith("HCN, H2S absorb in this band and are not in the spectrum") for c in caveats)
    assert not any("Interior temperature" in c for c in caveats)
    unstated = forward.canonical_params(dict(planet="wasp39b", tp_mode="guillot"))
    caveats = gui_results._model_caveats(dict(vulcan, params_json=np.array(json.dumps(unstated))))
    assert "Interior temperature not stated: the run used the default T_int = 300 K." in caveats
    no_core = np.array([], dtype="U8")
    picaso = dict(base, chem_provider=np.array("picaso"), unlisted_core=no_core, picaso_retried_layers=np.array([3]))
    caveats = gui_results._model_caveats(picaso)
    assert any("converged 3 layer solves only at a shifted temperature" in c for c in caveats)


def test_a_run_that_does_not_state_tint_says_so_in_its_log(monkeypatch):
    """run_model logs the result's T_int caveat before the solve; a stated T_int logs none."""

    class _Stop(Exception):
        pass

    def stop(*_a):
        raise _Stop

    monkeypatch.setattr(forward, "assemble_chem", stop)
    warning = "[fwd] WARNING: Interior temperature not stated: the run used the default T_int = 300 K."
    for extra, logged in (({}, True), ({"Tint": 300.0}, False)):
        lines = []
        with pytest.raises(_Stop):
            forward.run_model(
                dict(planet="wasp39b", tp_mode="guillot", **extra), log=lambda *a: lines.append(" ".join(a))
            )
        assert (warning in lines) is logged, lines


def test_the_picaso_worker_counts_the_layers_it_re_solves_at_a_shifted_temperature(monkeypatch):
    """picaso's ladder retries a layer at T(1 + eps); every layer that needed it is counted and certified with it."""
    from jwst_tool import picaso_worker as pw

    class _Equilibrium:  # converges at any T except exactly 2000 K, so that layer needs the ladder
        def __init__(self, thermofile, method):
            self.atoms_names, self.gas_names, self.gas_mass = ["H", "C"], ["H2"], [2.0]

        def solve(self, P, T, molfracs_atoms):
            self.molfracs_species_gas, self.molfracs_atoms = np.ones(1), np.array([0.9, 0.1])
            return T != 2000.0

    photochem = types.ModuleType("picaso.photochem")
    photochem.EquilibriumChemistry = _Equilibrium
    monkeypatch.setitem(sys.modules, "picaso", types.ModuleType("picaso"))
    monkeypatch.setitem(sys.modules, "picaso.photochem", photochem)
    req = {"xh": {"C": 0.1}, "metallicity": 1.0, "p_bar": [0.01, 0.1, 1.0], "T": [1000.0, 2000.0, 1500.0]}
    out = pw._run(None, None, {"refdata": "/unused"}, dict(req, climate=None, quench_kzz=None))
    assert int(out["chem_retry"]) == 1 and out["converged"].all()
    # the certificate records the count
    res = {
        "converged": np.ones(1, bool),
        "atoms_names": np.array(["H", "C", "N", "O", "S"]),
        "atoms": np.ones((1, 5)),
        "target": np.ones(5),
        "chem_fail": np.array(0),
        "chem_retry": np.array(4),
        "vmr": np.ones((1, 1)),
        "T": np.ones(1),
    }
    assert pe.certify({"p_bar": [1.0], "climate": None}, res, "t")["retried_layers"] == 4


def test_the_fd_refusal_names_the_numbers_and_the_tools_own_remedies():
    with pytest.raises(RuntimeError) as exc:
        jacobian.fd_row("lnZ", np.array([1.0, 2.0]), np.array([1.0, 3.0]), 0.1)
    msg = str(exc.value)
    assert "= 0.500 > 0.25 (h = 0.1)" in msg and "Drop lnZ from the Fisher parameters" in msg
    assert "model_defs" not in msg and "FD_STEPS" not in msg


def test_the_result_sums_every_retried_layer(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(forward, "cache_path", lambda _params: tmp_path / "entry.npz")
    chem = SimpleNamespace(
        p_bar=np.array([1.0, 0.1]), species_masses=np.ones(2), sidx={"H2": 0, "H2O": 1}, cert={}, quench={}
    )
    y = np.full((2, 2), 0.5)
    col = forward._Column(y, y, chem, None, ["H2", "H2O"], [], None, False)
    rt = SimpleNamespace(wl_um=np.array([1.0, 2.0]), unlisted_core_molecules=[])
    setup = SimpleNamespace(rt=rt, emis=None)
    # the assembly's own columns plus the ones a T-P stencil assembly appended
    retried = [("geometry probe", 2), ("baseline", 0), ("FD Tirr +1h baseline", 3)]
    assembly = SimpleNamespace(
        theta=np.zeros(4), theta_names=["a"], profile={"molecules": ["H2O"]}, g_anchor=400.0, retried=retried
    )
    res = forward._RunResult(
        np.ones(2), np.ones(2), np.ones((0, 2)), None, None, [], {"any_outside": 0, "thermal_rows": 0}, 0.0, []
    )
    cp = {"science_mode": "transmission", "chem_provider": "picaso"}
    out = np.load(forward._write_result({}, cp, assembly, setup, col, res), allow_pickle=False)
    assert int(out["picaso_retried_layers"][0]) == 5


def test_the_sidebar_shows_the_tirr_refusal_in_an_empty_widget_never_a_clipped_number():
    pytest.importorskip("streamlit")
    from app_helpers import run_app

    at = run_app()
    at.selectbox(key="n0_planet").set_value("custom").run()
    at.selectbox(key="n0_custom_tp").set_value("guillot").run()
    for name, cool in (("teff", 3300.0), ("rstar", 0.30), ("a", 0.05)):
        at.number_input(key=f"n0_custom_{name}").set_value(cool).run()
    assert at.number_input(key="n0_custom_tirr").value is None
    # the API's own sentence, beside the widget
    with pytest.raises(ValueError, match="is 560 K") as refusal:
        forward.canonical_params(_custom(star_teff=3300.0, rstar_rsun=0.30, orbit_au=0.05))
    assert [e.value for e in at.sidebar.error] == [str(refusal.value)]
    assert all(b.disabled for b in at.button if "Run" in str(b.label))
    # back in range, the empty widget takes the default again
    for name, back in (("teff", 5485.0), ("rstar", 0.932), ("a", 0.04828)):
        at.number_input(key=f"n0_custom_{name}").set_value(back).run()
    assert at.number_input(key="n0_custom_tirr").value == 1640.0 and not at.sidebar.error


def test_the_model_top_widget_follows_the_table_top_and_shows_the_api_refusal():
    """The sidebar seeds the model top from run_params.default_rt_ptop_bar and follows the profile until edited; an
    edited top above the table is refused beside Run with the API's sentence."""
    pytest.importorskip("streamlit")
    from app_helpers import run_app

    at = run_app()
    assert at.number_input(key="n0_rtptop").value == 5.349e-6
    at.selectbox(key="n0_wasp39b_tp").set_value("guillot").run()
    assert at.number_input(key="n0_rtptop").value == 1.0e-7
    at.number_input(key="n0_rtptop").set_value(1.0e-8).run()
    at.selectbox(key="n0_wasp39b_tp").set_value("file").run()
    assert at.number_input(key="n0_rtptop").value == 1.0e-8
    assert any("top pressure 5.349e-06 bar does not reach the model top 1e-08 bar" in e.value for e in at.error)


def test_a_custom_star_is_checked_wherever_it_is_read():
    """A custom planet's T_irr default reads its star in transit too, so its T_eff is range-checked there; with T_irr
    stated a transit reads no star and the payload zeroes it, so a saved custom transit loads again."""
    with pytest.raises(ValueError, match="star_teff=-100"):
        forward.canonical_params(_custom(star_teff=-100.0))
    cp = forward.canonical_params(_custom(star_teff=5000.0))
    assert cp["star_teff"] == 0.0 and forward.canonical_params(cp) == cp


def test_the_sidebar_sends_a_custom_planets_whole_system():
    """The custom form sends every key a custom request must state, so the sidebar's run is refused for nothing; it
    starts on the Guillot profile, since no table of another planet is offered."""
    pytest.importorskip("streamlit")
    from app_helpers import run_app

    at = run_app()
    at.selectbox(key="n0_planet").set_value("custom").run()
    assert at.selectbox(key="n0_custom_tp").value == "guillot"
    assert not at.error, [e.value for e in at.error]
    at.selectbox(key="n0_provider").set_value("picaso").run()
    assert not at.error, [e.value for e in at.error]


def test_the_sidebar_shows_the_refusal_of_advection_without_diffusion():
    """The sidebar sends what the user ticked; canonical_params' sentence stops the Run, never a silent False."""
    pytest.importorskip("streamlit")
    from app_helpers import run_app

    at = run_app()
    at.checkbox(key="n0_moldiff").uncheck().run()
    at.checkbox(key="n0_vmmol").check().run()
    assert any("use_vm_mol=True needs use_moldiff=True" in e.value for e in at.error)
