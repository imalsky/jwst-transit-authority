"""The gravitational constant is duplicated across repos; pin it.

`forward.py:1640` converts this tool's surface-gravity input into the planet
mass VULCAN-JAX wants:

    Mp = gs_cgs * rp_cm**2 / planets.G_CGS

and `vulcan_jax.atm_setup.surface_gravity` inverts it:

    gs = phy_const.G_grav * Mp / Rp**2

so `gs` round-trips ONLY because two independently-declared literals happen to
carry the same digits. A one-digit edit
on either side would silently rescale every atmosphere this tool submits, with
no error anywhere: the value stays physical, just wrong.

The 2026-08-14 audit's other option was to move the constants into
`vulcan_forward.constants` and re-export. That was declined: VULCAN-JAX sits
BELOW vulcan-forward in the dependency DAG and must keep its own `phy_const`
regardless, so a move would leave the same two-copy coupling plus an extra
import edge. Pinning is the honest fix, in the style of vulcan-forward's
`_assert_composition_tables`.

VULCAN-JAX is read from SOURCE with `ast`, never imported: this suite is
deliberately numpy-only and fast, and importing `vulcan_jax` pulls in JAX.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from jwst_tool import planets


def _vulcan_jax_constant(name: str) -> float:
    """Read a module-level float from vulcan_jax/phy_const.py without importing."""
    spec = importlib.util.find_spec("vulcan_jax")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip(
            "vulcan_jax is not installed, so the cross-repo constant pin "
            "cannot run. Install the sibling (deploy/pins.env names the "
            "commit) to exercise it.")
    path = Path(list(spec.submodule_search_locations)[0]) / "phy_const.py"
    assert path.is_file(), f"expected {path} to exist"
    tree = ast.parse(path.read_text(), str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return float(ast.literal_eval(node.value))
    raise AssertionError(
        f"vulcan_jax.phy_const no longer defines {name!r}. The gs <-> Mp "
        f"round-trip in forward.py depends on it; find where it moved and "
        f"re-point this pin rather than deleting the check.")


@pytest.mark.parametrize("name,ours,exact", [
    # G must be bit-identical: forward.py divides by it and VULCAN-JAX
    # multiplies by it, so any difference silently rescales every atmosphere.
    ("G_grav", planets.G_CGS, True),
    # R_sun converts rstar_rsun on both sides; same argument as G.
    ("r_sun", planets.R_SUN_CM, True),
    # AU legitimately differs by rounding (1.496e13 vs 1.49597871e13): it
    # only converts orbit_au into local irradiation geometry and never
    # crosses into the chemistry. The tolerance keeps that true.
    ("au", planets.AU_CM, False),
])
def test_constants_match_vulcan_jax(name, ours, exact):
    theirs = _vulcan_jax_constant(name)
    if exact:
        assert ours == theirs, (
            f"{name}: {ours!r} here vs {theirs!r} in vulcan_jax.phy_const. "
            "Change both together or not at all.")
    else:
        rel = abs(ours - theirs) / theirs
        assert rel < 2e-5, (
            f"{name}: {ours!r} vs {theirs!r}, relative difference {rel:.2e} "
            "exceeds the documented 2e-5 rounding tolerance.")
