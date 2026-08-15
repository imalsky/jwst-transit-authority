"""The gravitational constant is duplicated across repos; pin it.

`forward.py:1561` converts this tool's surface-gravity input into the planet
mass VULCAN-JAX wants:

    Mp = gs_cgs * rp_cm**2 / planets.G_CGS

and `vulcan_jax.atm_setup.surface_gravity` inverts it:

    gs = phy_const.G_grav * Mp / Rp**2

so `gs` round-trips ONLY because two independently-declared literals happen to
carry the same digits. Nothing checked that until 2026-08-14. A one-digit edit
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


def test_gravitational_constant_matches_vulcan_jax_exactly():
    """G must be bit-identical on both sides of the gs <-> Mp round-trip."""
    theirs = _vulcan_jax_constant("G_grav")
    assert planets.G_CGS == theirs, (
        f"planets.G_CGS={planets.G_CGS!r} but vulcan_jax.phy_const.G_grav="
        f"{theirs!r}. forward.py divides by the first and VULCAN-JAX "
        f"multiplies by the second, so any difference rescales every "
        f"submitted atmosphere silently. Both are CODATA 2018; change them "
        f"together or not at all."
    )


def test_solar_radius_matches_vulcan_jax_exactly():
    """R_sun converts rstar_rsun on both sides; same argument as G."""
    theirs = _vulcan_jax_constant("r_sun")
    assert planets.R_SUN_CM == theirs, (
        f"planets.R_SUN_CM={planets.R_SUN_CM!r} but vulcan_jax r_sun="
        f"{theirs!r} (IAU 2015 Resolution B3 nominal value on both sides)."
    )


def test_astronomical_unit_agrees_to_the_documented_tolerance():
    """AU is the one constant that legitimately differs, and by how much.

    This tool rounds to 1.496e13 while VULCAN-JAX carries 1.49597871e13. The
    two are used on OPPOSITE sides of a boundary that no value crosses -- AU
    here only converts `orbit_au` into the local irradiation geometry, and is
    never handed to the chemistry -- so this is a rounding choice, not a
    coupling. The test exists so that stays true: a future edit that widens
    the gap, or that starts passing an AU-derived quantity across the
    boundary, has to come here and say so.
    """
    theirs = _vulcan_jax_constant("au")
    for name, ours in (("planets.AU_CM", planets.AU_CM),):
        rel = abs(ours - theirs) / theirs
        assert rel < 2e-5, (
            f"{name}={ours!r} vs vulcan_jax au={theirs!r}: relative "
            f"difference {rel:.2e} exceeds the documented 2e-5 rounding "
            f"tolerance.")


def test_picaso_climate_reuses_the_registry_constants():
    """picaso_climate must not carry a third copy of R_sun / AU.

    It declared its own `_RSUN_CM` / `_AU_CM` until 2026-08-14. Two modules in
    ONE package disagreeing about a unit conversion is the failure this whole
    file exists to prevent.
    """
    from jwst_tool import picaso_climate
    assert picaso_climate._RSUN_CM == planets.R_SUN_CM
    assert picaso_climate._AU_CM == planets.AU_CM
