"""The tool's calls into the shared engine must exist on the INSTALLED engine.

`deploy/pins.env` pins the sibling repositories by commit, and
`test_deploy_pins.py` checks that the Dockerfile and CI agree on those strings.
What neither can check is whether the pinned pair is COMPATIBLE. Shipping
VULCAN_JAX at a commit that has removed a parameter alongside vulcan-forward
at a commit that still passes it makes every fresh forward model on the
deployed Space raise TypeError, while three green suites and a passing pin
test say nothing, because no test called the engine.

This closes that at the consumer boundary: CI installs the siblings at the
pinned commits, so an incompatible pair fails HERE. It is deliberately a
signature check, not a run -- it needs no data, no line lists and no solve.

The sibling repository carries the mirror of this test for its own calls into
VULCAN-JAX's private API (`vulcan-forward/tests/test_vulcan_jax_private_api.py`).
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "jwst_tool"

# module -> the attributes forward.py / engine_config.py reach for. Kept
# explicit rather than discovered, so ADDING a call to the engine is a
# deliberate edit here too.
EXPECTED = {
    "vulcan_forward.exojax_rt": ["build_rt_model", "build_emis_model"],
    "vulcan_forward.vulcan_chem": ["build_chem_model", "ChemParams"],
    "vulcan_forward.interp_map": ["make_to_art"],
    "vulcan_forward.paths": ["linelist_dir", "opacity_cache_dir"],
    "vulcan_forward.constants": [
        "MOLECULES", "ATOM_COLS", "ATOMIC_MASSES", "BULK_H2_VULCAN",
        "ART_PTOP_BAR", "ART_PBTM_BAR", "T_OPA_MIN_K", "T_OPA_MAX_K",
        "P_REF_BAR", "P_REF_EMISSION_BAR",
    ],
}

# The RT namespaces build_rt_model / build_emis_model return. A missing entry
# means an engine too old for this tool, which is exactly the pin bug: the
# tool would call it and fail at run time on a user's machine instead of here.
RT_NAMESPACE = ["transmission_depth", "transmission_depth_r", "wl_um",
                "art_ptop_bar", "art_pbtm_bar", "p_ref_bar", "molecules"]
EMIS_NAMESPACE = ["emission_flux", "emission_flux_tau", "emission_radius",
                  "tau_bottom", "art_pbtm_bar", "p_ref_emission_bar"]


def _skip_without_engine():
    if importlib.util.find_spec("vulcan_forward") is None:  # pragma: no cover
        pytest.skip("vulcan-forward not installed (light CI job)")


@pytest.mark.parametrize("mod_name", sorted(EXPECTED))
def test_engine_module_exposes_what_the_tool_calls(mod_name):
    _skip_without_engine()
    if mod_name in ("vulcan_forward.exojax_rt", "vulcan_forward.interp_map"):
        # these import exojax; honour the load-bearing import order
        if importlib.util.find_spec("exojax") is None:       # pragma: no cover
            pytest.skip("RT stack (exojax) not installed")
        importlib.import_module("vulcan_forward.vulcan_chem")
    mod = importlib.import_module(mod_name)
    missing = [a for a in EXPECTED[mod_name] if not hasattr(mod, a)]
    assert not missing, (
        f"{mod_name} is missing {missing}. The installed vulcan-forward is "
        "older than this tool needs; check deploy/pins.env.")


def test_the_rt_namespaces_carry_what_the_tool_reads_off_them():
    """build_rt_model / build_emis_model return SimpleNamespaces, so a missing
    attribute is an AttributeError deep inside a multi-minute run rather than
    an import error. Checked against the engine's source, because building the
    real namespaces needs line lists and an opacity build."""
    _skip_without_engine()
    if importlib.util.find_spec("exojax") is None:           # pragma: no cover
        pytest.skip("RT stack (exojax) not installed")
    importlib.import_module("vulcan_forward.vulcan_chem")
    src = (Path(importlib.import_module("vulcan_forward.exojax_rt").__file__)
           .read_text())
    for name in RT_NAMESPACE + EMIS_NAMESPACE:
        assert f"{name}=" in src, (
            f"the engine's RT namespace has no {name!r}; the tool reads it")


def test_the_profile_keys_the_tool_sets_are_ones_the_engine_reads():
    """Every RT knob the tool puts in its profile must be a key the engine
    actually consumes. An engine that does not know a key ignores it SILENTLY,
    which would cache a spectrum under a key describing physics that was never
    applied. run_model's echo check catches this at run time; this catches it
    in CI."""
    _skip_without_engine()
    if importlib.util.find_spec("exojax") is None:           # pragma: no cover
        pytest.skip("RT stack (exojax) not installed")
    importlib.import_module("vulcan_forward.vulcan_chem")
    src = (Path(importlib.import_module("vulcan_forward.exojax_rt").__file__)
           .read_text())
    for key in ("art_ptop_bar", "art_pbtm_bar", "rt_integration",
                "dit_grid_resolution", "p_ref_bar", "p_ref_emission_bar",
                "mie_condensate", "art_nlayer", "nu_pts", "broadening"):
        assert f'"{key}"' in src, (
            f"the engine never reads profile[{key!r}], so the tool sets a knob "
            "that does nothing")


def test_the_engine_calls_in_forward_py_pass_arguments_it_accepts():
    """AST-scan the tool's own calls into the engine and check each against the
    live signature. This is the shape of the break that shipped: a sibling
    dropped a parameter and the caller kept passing it."""
    _skip_without_engine()
    if importlib.util.find_spec("exojax") is None:           # pragma: no cover
        pytest.skip("RT stack (exojax) not installed")
    importlib.import_module("vulcan_forward.vulcan_chem")
    mods = {"exojax_rt": importlib.import_module("vulcan_forward.exojax_rt"),
            "vulcan_chem": importlib.import_module("vulcan_forward.vulcan_chem"),
            "interp_map": importlib.import_module("vulcan_forward.interp_map")}
    tree = ast.parse((SRC / "forward.py").read_text())
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id in mods):
            continue
        fn = getattr(mods[f.value.id], f.attr, None)
        assert fn is not None, (
            f"forward.py calls {f.value.id}.{f.attr}, which the installed "
            "vulcan-forward does not define")
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        names = set(sig.parameters)
        var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
        for kw in node.keywords:
            if kw.arg is None or var_kw:
                continue
            assert kw.arg in names, (
                f"forward.py calls {f.value.id}.{f.attr}({kw.arg}=...) but the "
                f"installed signature is {sig}. The pinned engine and this "
                "tool disagree.")
        checked += 1
    assert checked >= 3, (
        f"only {checked} engine calls found in forward.py -- the scan is not "
        "looking where it thinks it is")


def test_pinned_engine_floor_is_recorded():
    """pyproject must state a floor for the engine, and the installed engine
    must meet it. A pin without a floor is a string, not a contract."""
    _skip_without_engine()
    root = Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text()
    line = next((ln for ln in text.splitlines()
                 if "vulcan-forward" in ln and ">=" in ln), None)
    assert line, "pyproject does not pin a vulcan-forward floor"
    floor = line.split(">=")[1].split("#")[0].strip().strip('",').strip()
    import vulcan_forward
    got = getattr(vulcan_forward, "__version__", None)
    assert got, "vulcan_forward exposes no __version__"
    to_t = lambda v: tuple(int(x) for x in v.split(".")[:3])   # noqa: E731
    assert to_t(got) >= to_t(floor), (
        f"installed vulcan-forward {got} is below this tool's floor {floor}")
