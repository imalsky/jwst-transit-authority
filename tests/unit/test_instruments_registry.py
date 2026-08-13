"""Mode-registry invariants, looped over EVERY ``instruments.MODES`` key so a
future mode cannot land half-wired.

Pins: the fixed display encodings (MODE_COLOR hex + unique, MODE_MARKER
unique -- series must never rely on color alone), a LITERATURE_NOISE_FACTORS
reference entry per mode, the ngroup ordering + PandExo instrument caps, the
explicit TSO pinning rule (readout_pattern, background AND background_level,
extraction strategy, sane wavelength span -- never leave these implicit on a
new mode), the G395M entry's refdata-verified tokens and appended-last
palette slot, and that the parity harness's MODE_KEYS stays the frozen
7-mode experiment (G395M is deliberately NOT parity-covered; the README's
open-gaps list says so).
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

from jwst_tool import instruments as ins

SCRIPTS = Path(__file__).resolve().parents[1] / "parity" / "scripts"


def test_every_mode_has_color_marker_and_literature_factor():
    """A mode key missing any per-mode table entry is a half-registered mode."""
    for key in ins.MODES:
        assert key in ins.MODE_COLOR, f"{key}: no MODE_COLOR"
        assert re.fullmatch(r"#[0-9a-f]{6}", ins.MODE_COLOR[key]), (
            f"{key}: MODE_COLOR {ins.MODE_COLOR[key]!r} is not lowercase "
            "#rrggbb hex")
        assert key in ins.MODE_MARKER, f"{key}: no MODE_MARKER"
        assert ins.MODE_MARKER[key], f"{key}: empty marker"
        assert key in ins.LITERATURE_NOISE_FACTORS, (
            f"{key}: no LITERATURE_NOISE_FACTORS reference entry")
        f = ins.LITERATURE_NOISE_FACTORS[key]
        assert 1.0 <= f <= 2.0, f"{key}: implausible noise factor {f!r}"


def test_colors_and_markers_are_unique_across_modes():
    """The palettes cycle at their length; while len(MODES) fits, every mode
    must get a DISTINCT color and marker (grayscale print, CVD)."""
    assert len(ins.MODES) <= 8, (
        "MODES outgrew the 8-slot color/marker palettes -- extend _COLORS/"
        "_MARKERS (palette-checker validated) before adding the mode")
    colors = list(ins.MODE_COLOR.values())
    markers = list(ins.MODE_MARKER.values())
    assert len(set(colors)) == len(colors), "duplicate MODE_COLOR"
    assert len(set(markers)) == len(markers), "duplicate MODE_MARKER"


def test_ngroup_ordering_and_pandexo_caps_hold_for_every_mode():
    """1 <= ngroup_min <= ngroup_warn_below <= ngroup_max, and the PandExo
    per-instrument hard caps -- the same conditions the import guard raises
    on, re-asserted here so the guard itself cannot be quietly deleted."""
    for key, m in ins.MODES.items():
        assert 1 <= m["ngroup_min"] <= m["ngroup_warn_below"] \
            <= m["ngroup_max"], (
            f"{key}: ngroup ordering broken "
            f"({m['ngroup_min']}/{m['ngroup_warn_below']}/{m['ngroup_max']})")
        cap = ins.PANDEXO_NGROUP_MAX.get(m["instrument"])
        if cap is not None:
            assert m["ngroup_max"] <= cap, (
                f"{key}: ngroup_max {m['ngroup_max']} above the PandExo "
                f"{m['instrument']} cap {cap}")


def test_every_mode_pins_the_full_tso_configuration():
    """The 2026-07-12 parity lesson: engine defaults are the WRONG non-TSO
    choices, so every mode must pin readout_pattern, the extraction strategy,
    and BOTH background keys explicitly."""
    for key, m in ins.MODES.items():
        assert m["config"]["detector"].get("readout_pattern"), (
            f"{key}: readout_pattern not pinned (engine defaults are "
            "non-TSO patterns and drift between releases)")
        assert m["config"]["detector"].get("subarray"), (
            f"{key}: subarray not pinned")
        assert m.get("strategy"), f"{key}: extraction strategy not pinned"
        assert m.get("background") and m.get("background_level"), (
            f"{key}: background AND background_level must both be pinned")
        assert 0.5 < m["wl_min"] < m["wl_max"] <= 15.0, (
            f"{key}: wavelength span {m['wl_min']}-{m['wl_max']} outside the "
            "forward model's 1-15 um coverage convention")
        assert m.get("label"), f"{key}: no display label"
        assert 0.0 < m["floor_ppm_suggested"] <= 200.0, key
        assert m["noise_infl"] == 1.0, (
            f"{key}: noise_infl default must be 1.0 (the Pandeia prediction "
            "as-is; literature ratios are reference points, never defaults)")


def test_g395m_registry_entry_matches_the_verified_refdata_tokens():
    """The 2026-08-12 addition: tokens verified against pandeia_data-2026.7
    (and 2026.2) nirspec config.json; wl span is the jwst-docs usable G395M
    science bandpass; the palette slot is the appended-last 8th."""
    m = ins.MODES["nirspec_g395m"]
    assert list(ins.MODES)[-1] == "nirspec_g395m", (
        "nirspec_g395m must stay LAST in MODES: colors/markers are assigned "
        "by enumeration order, so reordering silently recolors every mode")
    assert m["instrument"] == "nirspec" and m["mode"] == "bots"
    assert m["config"]["instrument"] == dict(disperser="g395m",
                                             filter="f290lp")
    assert m["config"]["detector"] == dict(subarray="sub2048",
                                           readout_pattern="nrsrapid")
    assert (m["wl_min"], m["wl_max"]) == (2.87, 5.10)
    assert m["ngroup_max"] == ins.PANDEXO_UNBOUNDED_NGROUP, (
        "NIRSpec ramps are saturation-limited, not registry-capped")
    assert ins.MODE_COLOR["nirspec_g395m"] == "#006c8e"
    assert ins.MODE_MARKER["nirspec_g395m"] == "*"
    assert ins.LITERATURE_NOISE_FACTORS["nirspec_g395m"] == 1.10, (
        "extrapolated from G395H (no published number) -- changing it needs "
        "a measurement and a decision record")


def test_parity_harness_mode_set_is_frozen_without_g395m():
    """tests/parity is a FROZEN 7-mode experiment (its committed artifact is
    the release gate); G395M is deliberately not in it, and that gap is
    disclosed in the README's open-gaps list, not silently closed here."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        pg = importlib.import_module("parity_gate")
    finally:
        sys.path.remove(str(SCRIPTS))
    frozen = {"nirspec_prism", "nirspec_g395h", "nirspec_g235h",
              "niriss_soss", "nircam_f322w2", "nircam_f444w", "miri_lrs"}
    assert set(pg.MODE_KEYS) == frozen, (
        "parity MODE_KEYS changed -- the harness is a frozen experiment; "
        "extending it invalidates the committed artifact and needs a "
        "regenerated parity run plus a decision record")
    assert "nirspec_g395m" not in pg.MODE_KEYS
