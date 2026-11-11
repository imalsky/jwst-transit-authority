"""The tool's calls into the shared engine must exist on the INSTALLED engine.

Why, cost and tolerances: docs/validation.md, section `test_engine_api_contract`.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

from jwst_tool import chem_build

SRC = Path(__file__).resolve().parents[2] / "src" / "jwst_tool"

# module -> the attributes forward.py / engine_config.py reach for. Kept
# explicit rather than discovered, so ADDING a call to the engine is a
# deliberate edit here too.
EXPECTED = {
    "vulcan_forward.exojax_rt": ["build_rt_model", "build_emis_model"],
    "vulcan_forward.vulcan_chem": ["build_chem_model", "ChemParams", "select_network"],
    "vulcan_forward.interp_map": ["make_to_art"],
    "vulcan_forward.paths": ["opacity_cache_dir"],
    "vulcan_forward.exomolop": ["table_path", "provenance", "table_info"],
    "vulcan_forward.constants": [
        "MOLECULES",
        "ATOM_COLS",
        "ATOMIC_MASSES",
        "ART_PTOP_BAR",
        "T_OPA_MIN_K",
        "T_OPA_MAX_K",
        "P_REF_EMISSION_BAR",
        "K_FLOOR",
    ],
}

# The RT namespaces build_rt_model / build_emis_model return. A missing entry
# means an engine too old for this tool, which is exactly the pin bug: the
# tool would call it and fail at run time on a user's machine instead of here.
# Exactly the attributes forward.py reads (the RT echo keys included).
RT_NAMESPACE = [
    "transmission_depth_r",
    "wl_um",
    "nu_grid",
    "p_art_bar",
    "molecules",
    "art_ptop_bar",
    "art_pbtm_bar",
    "p_ref_bar",
    "rt_integration",
    "omit_core_molecules",
    "unlisted_core_molecules",
    "nu_edges",
]
EMIS_NAMESPACE = ["eclipse_flux_tau", "emission_radius", "art_pbtm_bar"]


def _skip_without_engine():
    if importlib.util.find_spec("vulcan_forward") is None:  # pragma: no cover
        pytest.skip("vulcan-forward not installed (light CI job)")


@pytest.mark.parametrize("mod_name", sorted(EXPECTED))
def test_engine_module_exposes_what_the_tool_calls(mod_name):
    _skip_without_engine()
    if mod_name in ("vulcan_forward.exojax_rt", "vulcan_forward.interp_map"):
        # these import exojax; honour the load-bearing import order
        if importlib.util.find_spec("exojax") is None:  # pragma: no cover
            pytest.skip("RT stack (exojax) not installed")
        importlib.import_module("vulcan_forward.vulcan_chem")
    mod = importlib.import_module(mod_name)
    missing = [a for a in EXPECTED[mod_name] if not hasattr(mod, a)]
    assert not missing, (
        f"{mod_name} is missing {missing}. The installed vulcan-forward is "
        "older than this tool needs; check deploy/pins.env."
    )


def _engine_tree(mod_name):
    return ast.parse(Path(importlib.import_module(mod_name).__file__).read_text())


def _returned_fields(tree, function):
    """Keyword names of the record ``function`` returns, or of the helper it returns the result of."""
    defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    fields, todo, seen = set(), [function], set()
    while todo:
        name = todo.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        for node in ast.walk(defs[name]):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                fields |= {k.arg for k in node.value.keywords if k.arg}
                if isinstance(node.value.func, ast.Name):
                    todo.append(node.value.func.id)
    assert seen, f"{function} is not in the engine module: the scan is not looking where it thinks it is"
    return fields


def _profile_reads(tree):
    """Keys read off a ``profile``: subscripts, .get, membership, and the literal keys of a loop that subscripts it."""
    reads = set()

    def on_profile(node):
        return isinstance(node, ast.Name) and node.id == "profile"

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and on_profile(node.value) and isinstance(node.slice, ast.Constant):
            reads.add(node.slice.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and on_profile(node.func.value)
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            reads.add(node.args[0].value)
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Constant)
            and any(on_profile(c) for c in node.comparators)
        ):
            reads.add(node.left.value)
        if isinstance(node, ast.For) and any(
            isinstance(n, ast.Subscript) and on_profile(n.value) and isinstance(n.slice, ast.Name)
            for n in ast.walk(node)
        ):
            reads |= {c.value for c in ast.walk(node.iter) if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    return reads


def test_the_rt_namespaces_carry_what_the_tool_reads_off_them():
    """build_rt_model / build_emis_model return SimpleNamespaces, so a missing
    attribute is an AttributeError deep inside a multi-minute run rather than
    an import error. Checked against the record the engine's source returns, because building the
    real namespaces needs the k-tables and an opacity build.
    """
    _skip_without_engine()
    if importlib.util.find_spec("exojax") is None:  # pragma: no cover
        pytest.skip("RT stack (exojax) not installed")
    importlib.import_module("vulcan_forward.vulcan_chem")
    tree = _engine_tree("vulcan_forward.exojax_rt")
    assert set(RT_NAMESPACE) <= _returned_fields(tree, "build_rt_model")
    assert set(EMIS_NAMESPACE) <= _returned_fields(tree, "build_emis_model")


def test_the_profile_keys_the_tool_sets_are_ones_the_engine_reads():
    """Every RT knob the tool puts in its profile must be a key the engine
    actually consumes. An engine that does not know a key ignores it SILENTLY,
    which would cache a spectrum under a key describing physics that was never
    applied. run_model's echo check catches this at run time; this catches it
    in CI.
    """
    _skip_without_engine()
    if importlib.util.find_spec("exojax") is None:  # pragma: no cover
        pytest.skip("RT stack (exojax) not installed")
    importlib.import_module("vulcan_forward.vulcan_chem")
    reads = _profile_reads(_engine_tree("vulcan_forward.exojax_rt"))
    for key in ("art_ptop_bar", "art_pbtm_bar", "rt_integration", "p_ref_bar", "art_nlayer"):
        assert key in reads, f"the engine never reads profile[{key!r}], so the tool sets a knob that does nothing"
    # the chemistry side: the solver knobs the tool drives, plus the two
    # margins the AD dlnCO gate reads (their absence must fail here, not at
    # run time)
    chem_tree = _engine_tree("vulcan_forward.vulcan_chem")
    reads = _profile_reads(chem_tree)
    for key in ("yconv_cri", "yconv_min", "dt_max", "co_mode", "network", "atom_list", "vulcan_cfg_name"):
        assert key in reads, f"the engine never reads profile[{key!r}]"
    returned = _returned_fields(chem_tree, "build_chem_model")
    for attr in ("co_bz_bound", "co_bz_margin"):
        assert attr in returned, f"the engine no longer returns {attr!r}; certify.check_ad_co_margin reads it"


def test_the_tools_engine_calls_pass_arguments_it_accepts():
    """AST-scan the tool's own calls into the engine and check each against the
    live signature. This catches a sibling that drops a parameter while the
    caller keeps passing it.
    """
    _skip_without_engine()
    if importlib.util.find_spec("exojax") is None:  # pragma: no cover
        pytest.skip("RT stack (exojax) not installed")
    importlib.import_module("vulcan_forward.vulcan_chem")
    mods = {
        "exojax_rt": importlib.import_module("vulcan_forward.exojax_rt"),
        "vulcan_chem": importlib.import_module("vulcan_forward.vulcan_chem"),
        "interp_map": importlib.import_module("vulcan_forward.interp_map"),
    }
    trees = [ast.parse(path.read_text()) for path in sorted(SRC.glob("*.py"))]
    checked = 0
    for node in (node for tree in trees for node in ast.walk(tree)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in mods):
            continue
        fn = getattr(mods[f.value.id], f.attr, None)
        assert fn is not None, (
            f"the tool calls {f.value.id}.{f.attr}, which the installed vulcan-forward does not define"
        )
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        names = set(sig.parameters)
        var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
        for kw in node.keywords:
            if kw.arg is None or var_kw:
                continue
            assert kw.arg in names, (
                f"the tool calls {f.value.id}.{f.attr}({kw.arg}=...) but the "
                f"installed signature is {sig}. The pinned engine and this "
                "tool disagree."
            )
        checked += 1
    assert checked >= 3, (
        f"only {checked} engine calls found in the tool -- the scan is not looking where it thinks it is"
    )


def test_the_engine_internals_the_eclipse_photosphere_reads_still_exist():
    """spectrum.emission_photosphere_bar reads the engine's tau = 2/3 photosphere through private names (it returns only
    the flux): eclipse_flux_tau bound to _Emission, its _dtau and _g_layers, its lnp_em and opacities fields, and
    _rt_geometry._photosphere_lnp. A rename must fail here, not inside an eclipse run.
    """
    _skip_without_engine()
    if importlib.util.find_spec("exojax") is None:  # pragma: no cover
        pytest.skip("RT stack (exojax) not installed")
    importlib.import_module("vulcan_forward.vulcan_chem")
    from vulcan_forward import _rt_geometry, exojax_rt

    assert callable(_rt_geometry._photosphere_lnp)
    emission = exojax_rt._Emission
    assert callable(emission._dtau) and callable(emission._g_layers) and callable(emission.eclipse_flux_tau)
    assert {"lnp_em", "opacities"} <= set(emission.__dataclass_fields__)
    # the g-ordinate weights, read as the engine itself reads them
    assert "ckd_pack" in exojax_rt._Opacities._fields and "opacities.ckd_pack.gw" in inspect.getsource(exojax_rt)


@pytest.mark.parametrize(("dist", "mod_name"), [("vulcan-forward", "vulcan_forward"), ("vulcan3", "vulcan3")])
def test_pinned_engine_floor_is_recorded(dist, mod_name):
    """Pyproject must state a floor for BOTH engines, and each installed engine
    must meet it. A pin without a floor is a string, not a contract.
    """
    if importlib.util.find_spec(mod_name) is None:  # pragma: no cover
        pytest.skip(f"{dist} not installed (light CI job)")
    root = Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text()
    line = next((ln for ln in text.splitlines() if dist in ln and ">=" in ln), None)
    assert line, f"pyproject does not pin a {dist} floor"
    floor = line.split(">=")[1].split("#")[0].strip().strip('",').strip()
    got = getattr(importlib.import_module(mod_name), "__version__", None)
    assert got, f"{mod_name} exposes no __version__"
    to_t = lambda v: tuple(int(x) for x in v.split(".")[:3])
    assert to_t(got) >= to_t(floor), f"installed {dist} {got} is below this tool's floor {floor}"


def test_the_chemistry_radius_anchor_matches_the_engines_own_rule():
    """`chem_build.chem_radius_anchor_bar` RESTATES one engine rule -- which grid
    interface VULCAN anchors g = G*Mp/Rp**2 at -- because the engine offers no
    way to ask. Restating is only safe while it agrees, so check it against
    `compute_mu_dz_g` on both branches (gas giant reaching 1 bar; shallow or
    rocky column). If this fails the engine moved its anchor and the tool is
    handing chemistry a radius quoted at the wrong pressure.
    """
    if importlib.util.find_spec("vulcan3") is None:  # pragma: no cover
        pytest.skip("VULCAN-JAX not installed (light CI job)")
    from types import SimpleNamespace

    import numpy as np
    from vulcan3.atm_setup import compute_mu_dz_g, compute_pico

    for p_b_bar, p_t_bar, nz, rocky in (
        (7.6, 1e-7, 100, False),
        (300.0, 1e-7, 60, False),
        (7.6, 1e-7, 100, True),
        (0.5, 1e-7, 40, False),
    ):
        pco = np.logspace(np.log10(p_b_bar * 1e6), np.log10(p_t_bar * 1e6), nz)
        pico = np.asarray(compute_pico(pco))
        cfg = SimpleNamespace(Mp=5.4e29, Rp=9.1e9, rocky=rocky, P_b=p_b_bar * 1e6, use_moldiff=False)
        out = compute_mu_dz_g(cfg, np.full((nz, 1), 1.0), np.array([2.3]), pico, np.full(nz, 1200.0))
        engine = float(pico[int(out["pref_indx"])]) / 1e6
        assert chem_build.chem_radius_anchor_bar(pco / 1e6, p_b_bar * 1e6, rocky) == pytest.approx(engine, rel=0), (
            f"anchor rule drifted at P_b={p_b_bar} rocky={rocky}"
        )
