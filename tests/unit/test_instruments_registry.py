"""Mode-registry invariants, looped over EVERY ``instruments.MODES`` key so a
future mode cannot land half-wired.

Pins: the fixed display encodings (MODE_COLOR hex + unique, MODE_MARKER
unique -- series must never rely on color alone), a LITERATURE_NOISE_FACTORS
reference entry per mode, the ngroup ordering + PandExo instrument caps, the
explicit TSO pinning rule (readout_pattern, background AND background_level,
extraction strategy, sane wavelength span -- never leave these implicit on a
new mode), the G395M entry's refdata-verified tokens and fixed palette slot,
and that the parity harness's MODE_KEYS covers every registered mode.
"""
from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from jwst_tool import instruments as ins

SCRIPTS = Path(__file__).resolve().parents[1] / "parity" / "scripts"


def test_display_encodings_are_complete_and_unique():
    """A mode key missing any per-mode table entry is a half-registered mode,
    and while len(MODES) fits the 12-slot palettes every mode must get a
    DISTINCT color and marker (grayscale print, CVD)."""
    assert len(ins.MODES) <= 12, (
        "MODES outgrew the 12-slot color/marker palettes -- extend _COLORS/"
        "_MARKERS (palette-checker validated) before adding the mode")
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
    """Parity lesson: engine defaults are the WRONG non-TSO
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
    """Tokens verified against pandeia_data-2026.7
    (and 2026.2) nirspec config.json; the palette slot is the 8th
    (enumeration position 7 -- colors/markers are assigned by enumeration
    order, so a mode may only ever be APPENDED after it, never inserted
    before). The band is pinned with every other band below."""
    m = ins.MODES["nirspec_g395m"]
    assert list(ins.MODES)[7] == "nirspec_g395m", (
        "nirspec_g395m must stay in MODES slot 7: colors/markers are "
        "assigned by enumeration order, so reordering silently recolors "
        "every mode")
    assert m["instrument"] == "nirspec" and m["mode"] == "bots"
    assert m["config"]["instrument"] == dict(disperser="g395m",
                                             filter="f290lp")
    assert m["config"]["detector"] == dict(subarray="sub2048",
                                           readout_pattern="nrsrapid")
    assert m["ngroup_max"] == ins.PANDEXO_UNBOUNDED_NGROUP, (
        "NIRSpec ramps are saturation-limited, not registry-capped")
    assert ins.MODE_COLOR["nirspec_g395m"] == "#006c8e"
    assert ins.MODE_MARKER["nirspec_g395m"] == "*"
    assert ins.LITERATURE_NOISE_FACTORS["nirspec_g395m"] == 1.10, (
        "extrapolated from G395H (no published number) -- changing it needs "
        "a measurement and a decision record")


# Every mode's band, pinned to its PUBLISHED source so a hand-narrowed edge
# cannot pass again (a G395M red edge of 5.10 shipped for weeks, cited to a
# jwst-docs number that does not exist in any jwst-docs table).
#   NIRSpec: Birkmann et al. 2022 (A&A 661, A83) Table 2, "Available filter
#     and disperser combinations for time-series observations with the
#     S1600A1 aperture", SUB2048 column -- identical to jwst-docs "NIRSpec
#     BOTS Wavelength Ranges and Gaps" Table 1. wl_min 1.0 on PRISM/G140H is
#     the forward model's short edge, not the instrument's.
#   NIRISS: Pandeia gr700xd order ranges (0.83-2.81 and 0.63-1.26), rounded
#     inward; wl_min 1.0 is again the model edge.
#   NIRCam: filter half-power points measured from the shipped transmission
#     curves, rounded inward to 0.05 um.
#   MIRI: 5-12 um; the 5.0 edge is the jwst-docs slitless-dispersion caution.
_BAND_SOURCES = {
    "nirspec_prism": (1.0, 5.30),      # Birkmann T2 PRISM/CLEAR 0.60-5.30
    "nirspec_g140h": (1.0, 1.83),      # Birkmann T2 0.97-1.31, 1.35-1.83
    "nirspec_g235h": (1.66, 3.07),     # Birkmann T2 1.66-2.20, 2.27-3.07
    "nirspec_g235m": (1.66, 3.12),     # Birkmann T2 1.66-3.12
    "nirspec_g395h": (2.87, 5.18),     # Birkmann T2 2.87-3.72, 3.82-5.18
    "nirspec_g395m": (2.87, 5.18),     # Birkmann T2 2.87-5.18 (same as G395H)
    "niriss_soss": (1.0, 2.8),         # pandeia gr700xd_1 clear 0.83-2.81
    "niriss_soss_ord2": (1.0, 1.26),   # pandeia gr700xd_2 0.63-1.26
    "nircam_f277w": (2.45, 3.10),      # F277W HWHM 2.419-3.130
    "nircam_f322w2": (2.45, 4.00),     # F322W2 HWHM 2.425-4.012
    "nircam_f444w": (3.9, 5.00),       # F444W HWHM 3.881-5.009
    "miri_lrs": (5.0, 12.0),           # LRS 5-12; slitless caution below 5
}


def test_every_mode_band_matches_its_published_source():
    assert set(_BAND_SOURCES) == set(ins.MODES), (
        "a mode was added or removed without pinning its band to a published "
        "source -- see _BAND_SOURCES")
    for key, (lo, hi) in _BAND_SOURCES.items():
        m = ins.MODES[key]
        assert (m["wl_min"], m["wl_max"]) == (lo, hi), (
            f"{key}: band {m['wl_min']}-{m['wl_max']} does not match its "
            f"pinned published source {lo}-{hi}; narrowing an edge silently "
            "discards real coverage")
    # G395M and G395H are the SAME disperser band; only the detector gap
    # differs. They drifted apart once and must not again.
    assert (ins.MODES["nirspec_g395m"]["wl_min"],
            ins.MODES["nirspec_g395m"]["wl_max"]) == \
           (ins.MODES["nirspec_g395h"]["wl_min"],
            ins.MODES["nirspec_g395h"]["wl_max"]), (
        "G395M and G395H share the F290LP BOTS band (Birkmann Table 2)")


def test_parity_harness_mode_set_covers_every_registered_mode():
    """The release parity experiment covers every registered mode."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        pg = importlib.import_module("parity_gate")
    finally:
        sys.path.remove(str(SCRIPTS))
    assert set(pg.MODE_KEYS) == set(ins.MODES), (
        "parity MODE_KEYS must match the registered instrument set; regenerate "
        "the parity artifact whenever this experiment changes"
    )


def test_r_native_med_still_matches_the_refdata_over_the_registry_band():
    """The displayed native R is a MEASUREMENT over the registry band, so a
    band edit silently invalidates it.

    Skipped wherever the Pandeia reference tree or astropy is absent (the
    light CI job has neither). Where it runs, it re-measures the median of
    R(lambda) from the same dispersion file the worker reads, over the band
    the registry now declares, and requires the shipped figure to be within
    3% -- loose enough for the deliberate rounding onto jwst-docs' nominal
    R ~ 2,700 / 1,000 for the NIRSpec gratings, tight enough to catch a band
    change that moved the median."""
    fits = pytest.importorskip("astropy.io.fits")

    root = ins.DATA_DIR / f"pandeia_data-{ins.BACKEND_RELEASE}-jwst" / "jwst"
    if not root.is_dir():
        pytest.skip(f"no Pandeia reference tree at {root}")

    for key, m in ins.MODES.items():
        # same resolution the worker's _native_r does, minus the glob
        disp = (m["config"].get("instrument", {}) or {}).get("disperser")
        if m["instrument"] == "miri":
            disp = "p750l"
        if m["instrument"] == "niriss":
            disp = f"gr700xd-ord{int(m['strategy'].get('order', 1))}"
        if m["instrument"] == "nircam":
            pat = f"{m['instrument']}/dispersion/jwst_nircam_disp_*.fits"
        else:
            pat = f"{m['instrument']}/dispersion/*{disp}*disp*.fits"
        hits = sorted(root.glob(pat))
        if not hits:
            pytest.skip(f"{key}: no dispersion file matching {pat}")
        with fits.open(hits[0]) as h:
            cols = {c.upper(): c for c in h[1].columns.names}
            w = np.asarray(h[1].data[cols["WAVELENGTH"]], float)
            r = np.asarray(h[1].data[cols["R"]], float)
        o = np.argsort(w)
        w, r = w[o], r[o]
        sel = (w >= m["wl_min"]) & (w <= m["wl_max"])
        assert sel.any(), f"{key}: band {m['wl_min']}-{m['wl_max']} is off the file"
        measured = float(np.median(r[sel]))
        shown = float(m["r_native_med"])
        assert abs(shown - measured) / measured <= 0.03, (
            f"{key}: r_native_med {shown:g} is {100 * abs(shown - measured) / measured:.1f}% "
            f"from the median {measured:.1f} measured over the current band "
            f"{m['wl_min']}-{m['wl_max']} -- re-measure after any band change")


def test_lsf_width_table_is_well_formed():
    """Every measured response width names a registered mode, sits inside
    that mode's band in ascending wavelength order, and is a finite width
    >= 1 (the PSF can only broaden the dispersion response); lsf_r divides
    R by it and leaves a mode without a measurement untouched."""
    assert set(ins.LSF_WIDTH) <= set(ins.MODES)
    for key, pts in ins.LSF_WIDTH.items():
        lam, width = np.transpose(pts)
        m = ins.MODES[key]
        assert np.all(np.diff(lam) > 0), key
        assert m["wl_min"] <= lam.min() and lam.max() <= m["wl_max"], key
        assert np.all(np.isfinite(width)) and np.all(width >= 1.0), key
        r = np.full(lam.size, 1000.0)
        assert np.allclose(ins.lsf_r(key, lam, r), r / width)
    assert np.array_equal(ins.lsf_r("nirspec_g395h", [3.0, 4.0], [2000.0, 2500.0]),
                          [2000.0, 2500.0])
