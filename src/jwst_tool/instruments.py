"""JWST time-series instrument-mode registry + paths for the noise backend.

Each mode entry carries the Pandeia configuration used by ``pandeia_worker.py``
(running inside the selected backend's conda env -- the DEFAULT ``current``
backend is pandeia.engine 2026.2 + pandeia_data-2026.2-jwst; ``legacy`` is the
pinned 3.0 pair, see the BACKEND SELECTION block below) plus display metadata
and a default systematic noise floor.

Mode tokens in ``MODES`` are the canonical (pinned-3.0) Pandeia names; some were
renamed in 2026-era engines. ``engine_mode()`` resolves each to the token the
ACTIVE backend accepts, and both the production path (``noise.noise_job``) and
the parity harness go through it -- so a mode is never submitted under a name the
running engine rejects.

Noise floors (``floor_ppm``): default CONSTANT minimum-uncertainty values per
mode, applied with PandExo semantics (sigma_final = max(sigma_random, floor)
on the final bins -- noise.resolve_floor; never quadrature, never rescaled by
the binning R). The values are the pre-flight planning convention (Greene et
al. 2016 assumed 20/30/50 ppm for NIRISS/NIRCam/MIRI); in-flight results are
often better (e.g. Schlawin et al. 2021 find ~<10 ppm for NIRCam grism), so
the defaults sit between the two. Every floor is editable in the GUI,
including "none" and a wavelength-dependent table.

Noise sensitivity factor (``noise_infl``): OPTIONAL multiplicative factor on
the Pandeia random sigma. DEFAULT 1.0 FOR EVERY MODE -- the baseline forecast
is the Pandeia prediction as-is. Published achieved-vs-predicted ratios
(LITERATURE_NOISE_FACTORS below) depend on target brightness, groups,
wavelength, detector, extraction, and pipeline; they are reference points for
sensitivity studies, NOT a transferable calibration, and are never applied by
default (2026-07-12 external audit). Proportional noise: averages down with
transits, unlike the floor. Editable in the GUI.
"""
from __future__ import annotations

import os
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent        # pandeia_worker.py lives here
# this file lives at <repo>/src/jwst_tool/instruments.py, so parents[2] is the
# repo root in an editable checkout (the src/jwst_tool marker check below tells
# a checkout apart from a site-packages install).
_REPO_DIR = Path(__file__).resolve().parents[2]
_IN_CHECKOUT = (_REPO_DIR / "src" / "jwst_tool").is_dir()

# INPUT data root (the minimal synphot CDBS). JWST_TOOL_DATA_DIR overrides;
# default is the checkout's data/. A site-packages install must set the env
# var -- fail loudly, no silent fallbacks.
_env_data = os.environ.get("JWST_TOOL_DATA_DIR")
if _env_data:
    DATA_DIR = Path(_env_data).expanduser()
elif _IN_CHECKOUT and (_REPO_DIR / "data").is_dir():
    DATA_DIR = _REPO_DIR / "data"
else:
    raise RuntimeError(
        "jwst_tool data root not found: set JWST_TOOL_DATA_DIR to a directory holding "
        "the tool's cdbs/ tree (a site-packages install cannot infer it), or run from "
        "an editable checkout of vulcan-jwst-tool.")

# GENERATED caches (model spectra + pandeia results) live in the repo output/;
# JWST_TOOL_OUTPUT_DIR overrides. Created on demand by the writers.
_env_out = os.environ.get("JWST_TOOL_OUTPUT_DIR")
if _env_out:
    OUTPUT_DIR = Path(_env_out).expanduser()
elif _IN_CHECKOUT:
    OUTPUT_DIR = _REPO_DIR / "output"
else:
    raise RuntimeError(
        "jwst_tool output root not found: set JWST_TOOL_OUTPUT_DIR (a site-packages "
        "install cannot infer it), or run from an editable checkout of vulcan-jwst-tool.")
MODEL_CACHE = OUTPUT_DIR / "model_cache"
NOISE_CACHE = OUTPUT_DIR / "noise_cache"

# Pandeia backend environment (the real STScI ETC engine, same as PandExo's core).
# The worker runs in its own conda env (pandeia has heavy deps); noise.run_pandeia
# refuses loudly if the python is missing.
#
# BACKEND SELECTION (JWST_TOOL_BACKEND, 2026-07-13): DEFAULT is "current" --
# pandeia.engine 2026.2 + pandeia_data-2026.2-jwst, the STScI JWST 5.1 release,
# the pair validated mode-by-mode against PandExo (tests/parity/; PandExo
# itself pins engine 2026.2). "current" is the backend TOKEN, not a currency
# claim: STScI's supported Cycle 6 release moved to 2026.7 on 2026-07-16, so
# 2026.2 output is one calibration release behind the live ETC until the full
# engine/refdata/PSF tuple is upgraded and parity is regenerated
# (docs/audit_decisions_2026-07-21.md). "legacy" selects the pinned pandeia
# 3.0 + pandeia_data-3.0rc3 pair,
# retained ONLY as an explicit reproducibility backend. Every result records the
# exact engine/refdata versions in "__provenance__", and the versions are in
# every cache key (switching backends self-invalidates caches). "current"'s
# refdata/psf default under DATA_DIR (data/pandeia_data-2026.2-jwst,
# data/pandeia_psfs-2026.2-jwst -- one-time download, see README), so it is
# portable across checkouts; only the conda python path and "legacy"'s refdata
# (an external picaso tree) are machine-specific. The explicit
# JWST_TOOL_PANDEIA_{PYTHON,REFDATA,PSF_DIR} env vars override any of them
# per-path on another machine.
_BACKENDS = {
    "current": dict(
        python="/opt/homebrew/Caskroom/miniforge/base/envs/pandeia_2026/bin/python",
        refdata=str(DATA_DIR / "pandeia_data-2026.2-jwst"),
        psf=str(DATA_DIR / "pandeia_psfs-2026.2-jwst"),
        status="Pandeia 2026.2 / pandeia_data-2026.2-jwst (STScI JWST 5.1 "
               "release, validated vs PandExo in tests/parity/; STScI's "
               "supported Cycle 6 release is now 2026.7 -- forecasts here "
               "are one calibration release behind the live ETC)"),
    "legacy": dict(
        python="/opt/homebrew/Caskroom/miniforge/base/envs/picaso_base/bin/python",
        refdata="/Users/imalsky/Documents/Important_Docs/JWST_CYCLE5/picaso_ian/data/pandeia_data-3.0rc3",
        psf="",
        status="LEGACY Pandeia 3.0 / pandeia_data-3.0rc3 (pinned reproducibility "
               "backend; several releases behind the STScI ETC -- set "
               "JWST_TOOL_BACKEND=current for the 2026.2 backend)"),
}
JWST_TOOL_BACKEND = os.environ.get("JWST_TOOL_BACKEND", "current").lower()
if JWST_TOOL_BACKEND not in _BACKENDS:
    raise RuntimeError(
        f"JWST_TOOL_BACKEND={JWST_TOOL_BACKEND!r} unknown; choose 'current' "
        f"(Pandeia 2026.2, default) or 'legacy' (pinned 3.0).")
_BE = _BACKENDS[JWST_TOOL_BACKEND]
BACKEND_STATUS = _BE["status"]

PICASO_PYTHON = os.environ.get("JWST_TOOL_PANDEIA_PYTHON", _BE["python"])
PANDEIA_REFDATA = os.environ.get("JWST_TOOL_PANDEIA_REFDATA", _BE["refdata"])
# pandeia_data >= 2026 splits the PSF library out of the refdata tree and the
# engine reads it from $PSF_DIR (the "current" backend sets it; the 3.0-era
# "legacy" tree carries its own PSFs, so it is empty there). When set it is
# passed through to the worker, preflighted, and joins the cache key.
PANDEIA_PSF_DIR = os.environ.get("JWST_TOOL_PANDEIA_PSF_DIR", _BE["psf"])
# Minimal synphot CDBS assembled for this tool: phoenix grid (symlink here;
# fetch the STScI reference-atlases synphot5 tarball on other machines), the
# 2MASS Ks bandpass (tracked in git), and the CALSPEC Vega spectrum
# (gitignored; ssb.stsci.edu/trds). The Vega copy feeds the LEGACY 3.x
# engine's vegamag path only -- the 2026+ engine normalizes against its OWN
# refdata Vega (sed/hst_calspec; the two agree to 0.08 mmag in Ks), so the
# worker requires the local file just for JWST_TOOL_BACKEND=legacy.
# `jwst-tool data` reports each piece.
PYSYN_CDBS = str(DATA_DIR / "cdbs")

# Engine-generation mode-name renames. MODES stores the canonical (pinned-3.0)
# Pandeia token; 2026-era engines renamed some modes, and the running engine
# hard-rejects an unknown token (ValueError: Invalid mode). The ACTIVE backend
# decides which name is valid, so this is keyed by JWST_TOOL_BACKEND and both the
# production noise path and the parity harness resolve through engine_mode() --
# one source of truth, no path can send a name the selected engine refuses.
#   current (Pandeia 2026.x): NIRCam grism time series "ssgrism" -> "lw_tsgrism".
#   legacy  (Pandeia 3.0):    identity (3.0 still calls it "ssgrism").
_MODE_RENAMES = {
    "current": {"nircam": {"ssgrism": "lw_tsgrism"}},
    "legacy": {},
}
ENGINE_MODE_RENAMES = _MODE_RENAMES[JWST_TOOL_BACKEND]


def engine_mode(instrument: str, mode: str) -> str:
    """Resolve a registry mode token to the name the ACTIVE backend accepts.

    MODES carries the pinned-3.0 token; 2026-era engines renamed some modes
    (e.g. NIRCam ``ssgrism`` -> ``lw_tsgrism``). Returns ``mode`` unchanged when
    the active backend needs no rename. Both ``noise.noise_job`` (production) and
    ``tests/parity`` go through here so a mode is never submitted under a name the
    running engine rejects.
    """
    return ENGINE_MODE_RENAMES.get(instrument, {}).get(mode, mode)


# Star normalization is band-integrated 2MASS Ks (vegamag, synphot "2mass,ks"
# bandpass) inside the worker -- the web-ETC convention. The retired shortcut
# was a monochromatic at_lambda flux from the Cohen et al. 2003 zero point
# (666.7 Jy at 2.159 um), which mis-scaled cool/warm stars by ~1-4% and fed
# that error into saturation/ngroup selection. The minimal CDBS now carries
# comp/nonhst/2mass_ks_001_syn.fits + calspec/alpha_lyr_stis_011.fits for it.

# Fixed categorical color order -- one color per mode, never re-assigned when
# the user's selection changes. Every color holds >= 3:1 contrast against the
# white figure surface (WCAG 2.2 non-text target; the 2026-07-29 UX review
# measured the previous aqua/yellow/pink at 2.2-2.8:1), and the set passes the
# dataviz palette validator in the spectrum's wavelength-adjacency order
# (soss, prism, g235h, f322w2, g395h, f444w, miri) with marker shapes
# (MODE_MARKER below) as the secondary, color-independent encoding.
_COLORS = ["#2a78d6", "#199e70", "#a35a00", "#007a00",
           "#4a3aa7", "#d43f3e", "#a83a9e", "#c2571f"]

# Fixed marker shape per mode (same assignment rule as the colors): data
# series must never rely on color alone (grayscale print, color-vision
# deficiency), so every plotted point carries a mode-specific marker too.
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

# PandExo-compatible hard maximum group counts per instrument. Current PandExo
# caps NIRCam grism at 100 groups; a request above that falls outside the
# PandExo-supported range and can trip a backend/config failure on faint
# targets (2026-07-12 external audit, item 5). Every mode's ngroup_max below
# must respect its instrument's cap (asserted at import); the worker clamps its
# selected ramp to [ngroup_min, ngroup_max], so this bounds the optimizer.
# (A dynamic query of the installed Pandeia/PandExo config is the preferred
# long-term source; this static table is the pinned-backend equivalent.)
PANDEXO_NGROUP_MAX = {"nircam": 100, "niriss": 30}

# NIRISS SOSS: 30 groups is the APT hard limit for NISRAPID/SUBSTRIP256 TSO
# (NIS with more groups requires SUBARRAY=FULL, not a TSO config), enforced
# upstream by PandExo master 877c0f4 (2026-07-13, max_ngroup['niriss']
# 65535 -> 30). Since this tool pins readout_pattern="nisrapid", the cap is
# a schedulability limit, not just PandExo parity. Note the 2026-07-12
# parity artifacts predate that upstream change by one day.
#
# For nirspec/miri PandExo's optimizer is effectively unbounded (its
# max_ngroup table reads 65535 there): SATURATION picks the ramp, not a
# registry cap. The old self-imposed BOTS-90 cap bound before the saturation
# optimum on moderate stars (G395H on a Ks=10.7 star wants ngroup~125 with
# NRSRAPID) and made the tool's ramps -- and therefore its sigmas and
# efficiencies -- silently diverge from PandExo/ETC output (2026-07-12
# parity harness finding). Policy anchor is PandExo MASTER: the tagged v3.0
# release has a scalar max_ngroup=65536 with no per-instrument caps at all.
PANDEXO_UNBOUNDED_NGROUP = 65535

# Extraction strategy + sky background: pinned to PandExo's TSO conventions
# (its reference templates: NIRSpec specapphot 0.7" aperture / [0.75, 1.5]"
# annulus, NIRCam 0.4"/[0.5, 1.5]", MIRI 0.6"/[1.0, 2.8]", background
# "ecliptic" at "background_level" "medium" -- BOTH background keys are
# required together, the engine resolves <background>_<level>.fits), NOT
# pandeia's generic point-source defaults (0.3" apertures, "minzodi").
# The 2026-07-12 parity harness measured the default-strategy
# mismatch at 8-20% in extracted flux (mode-dependent), which propagated
# straight into sigma. The tool's stated benchmark is PandExo-style planning
# noise, so the extraction must match that benchmark's convention.
# wl_min/wl_max: the usable science bandpass we bin over (intersected with the
# forward model's 1-15 um coverage; NIRISS SOSS order 1 nominally reaches 0.85 um
# but the model band starts at 1.0 um -- the H2-H2 CIA table's short edge).
#
# readout_pattern is pinned EXPLICITLY on every mode (2026-07-12 PandExo
# parity work): the TSO patterns real time-series programs use -- NRSRAPID
# (BOTS), NISRAPID (SOSS), RAPID (NIRCam grism), FASTR1 (MIRI LRS) -- which
# are also PandExo's choices. Inheriting build_default_calc's default was a
# latent config drift: the engine default for BOTS is the frame-averaged
# "nrs" pattern (~3.6 s groups vs NRSRAPID's 0.902 s), and defaults change
# between engine releases. Never leave readout_pattern implicit on a new mode.
MODES = {
    "nirspec_prism": dict(
        label="NIRSpec PRISM",
        instrument="nirspec", mode="bots",
        config=dict(instrument=dict(disperser="prism", filter="clear"),
                    detector=dict(subarray="sub512",
                                  readout_pattern="nrsrapid")),
        strategy=dict(aperture_size=0.7, sky_annulus=[0.75, 1.5]),
        background="ecliptic", background_level="medium",
        wl_min=0.6, wl_max=5.25,
        floor_ppm=20.0, noise_infl=1.0, ngroup_min=2,
        ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
    ),
    "nirspec_g395h": dict(
        label="NIRSpec G395H",
        instrument="nirspec", mode="bots",
        config=dict(instrument=dict(disperser="g395h", filter="f290lp"),
                    detector=dict(subarray="sub2048",
                                  readout_pattern="nrsrapid")),
        strategy=dict(aperture_size=0.7, sky_annulus=[0.75, 1.5]),
        background="ecliptic", background_level="medium",
        wl_min=2.87, wl_max=5.18,
        floor_ppm=15.0, noise_infl=1.0, ngroup_min=2,
        ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
    ),
    "nirspec_g235h": dict(
        label="NIRSpec G235H",
        instrument="nirspec", mode="bots",
        config=dict(instrument=dict(disperser="g235h", filter="f170lp"),
                    detector=dict(subarray="sub2048",
                                  readout_pattern="nrsrapid")),
        strategy=dict(aperture_size=0.7, sky_annulus=[0.75, 1.5]),
        background="ecliptic", background_level="medium",
        wl_min=1.66, wl_max=3.07,
        floor_ppm=15.0, noise_infl=1.0, ngroup_min=2,
        ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
    ),
    "niriss_soss": dict(
        label="NIRISS SOSS (ord 1)",
        instrument="niriss", mode="soss",
        config=dict(instrument=dict(filter="clear", disperser="gr700xd"),
                    detector=dict(subarray="substrip256",
                                  readout_pattern="nisrapid")),
        strategy=dict(order=1),
        background="ecliptic", background_level="medium",
        wl_min=0.85, wl_max=2.8,
        floor_ppm=20.0, noise_infl=1.0, ngroup_min=2, ngroup_max=30,
    ),
    "nircam_f322w2": dict(
        label="NIRCam F322W2",
        instrument="nircam", mode="ssgrism",
        config=dict(instrument=dict(filter="f322w2", disperser="grismr"),
                    detector=dict(subarray="subgrism64", readout_pattern="rapid")),
        strategy=dict(aperture_size=0.4, sky_annulus=[0.5, 1.5]),
        background="ecliptic", background_level="medium",
        wl_min=2.45, wl_max=3.95,
        floor_ppm=25.0, noise_infl=1.0, ngroup_min=2, ngroup_max=100,
    ),
    "nircam_f444w": dict(
        label="NIRCam F444W",
        instrument="nircam", mode="ssgrism",
        config=dict(instrument=dict(filter="f444w", disperser="grismr"),
                    detector=dict(subarray="subgrism64", readout_pattern="rapid")),
        strategy=dict(aperture_size=0.4, sky_annulus=[0.5, 1.5]),
        background="ecliptic", background_level="medium",
        wl_min=3.9, wl_max=4.95,
        floor_ppm=25.0, noise_infl=1.0, ngroup_min=2, ngroup_max=100,
    ),
    "miri_lrs": dict(
        label="MIRI LRS (slitless)",
        instrument="miri", mode="lrsslitless",
        config=dict(detector=dict(subarray="slitlessprism",
                                  readout_pattern="fastr1")),
        strategy=dict(aperture_size=0.6, sky_annulus=[1.0, 2.8]),
        background="ecliptic", background_level="medium",
        wl_min=5.0, wl_max=12.0,
        floor_ppm=40.0, noise_infl=1.0, ngroup_min=5,
        ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
    ),
}

# Literature-anchored achieved-vs-predicted noise ratios: REFERENCE POINTS
# for user-driven sensitivity studies, never applied by default (see module
# docstring). Provenance per entry -- three are published MEASUREMENTS, two
# are editorial: COMPASS G395H reanalysis (Gordon et al. 2025): measured
# errors average 1.05x PandExo on NRS1, 1.12x on NRS2; the G235H value is
# EXTRAPOLATED from that G395H measurement (same NRS1/NRS2 detectors and
# SUB2048 BOTS readout, different bandpass -- no published G235H
# achieved-vs-PandExo ratio exists as of 2026-07); NIRISS SOSS transit fits
# conventionally inflate errors 1.2x over photon noise (Radica et al. 2023,
# MNRAS 524, 835); the NIRCam 1.05 is an EDITORIAL PLACEHOLDER -- Ahrer et
# al. 2023 report "minimal systematics" with no published
# achieved-vs-predicted number, so a small nominal margin stands in (no
# measurement behind the digit); MIRI LRS measured ~15-20% above
# random-noise simulations (Bouwman et al. 2023); PRISM was photon-limited
# on the quiet ERS target (Rustamkulov et al. 2023).
LITERATURE_NOISE_FACTORS = {
    "nirspec_prism": 1.0,
    "nirspec_g395h": 1.10,
    "nirspec_g235h": 1.10,
    "niriss_soss": 1.20,
    "nircam_f322w2": 1.05,
    "nircam_f444w": 1.05,
    "miri_lrs": 1.15,
}

# enforce the PandExo group caps at import (loud, no silent out-of-range mode)
for _key, _m in MODES.items():
    _cap = PANDEXO_NGROUP_MAX.get(_m["instrument"])
    if _cap is not None and _m["ngroup_max"] > _cap:
        raise RuntimeError(
            f"mode {_key!r} sets ngroup_max={_m['ngroup_max']}, above the "
            f"PandExo-compatible maximum {_cap} for {_m['instrument']}; the "
            "optimizer would select an unsupported group count on faint "
            "targets (2026-07-12 audit item 5).")

MODE_COLOR = {key: _COLORS[i % len(_COLORS)] for i, key in enumerate(MODES)}
MODE_MARKER = {key: _MARKERS[i % len(_MARKERS)] for i, key in enumerate(MODES)}

# GUI default selection: the modes WASP-39b was actually observed in, which is
# also blue-to-red coverage. NIRISS SOSS (Feinstein et al. 2023), NIRSpec PRISM
# (Rustamkulov et al. 2023), NIRSpec G395H (Alderson et al. 2023) and NIRCam
# F322W2 (Ahrer et al. 2023) are the four JWST/ERS 1366 transit observations;
# MIRI LRS is the later mid-IR SO2 transit (Powell et al. 2024). G235H and
# NIRCam F444W stay available but were never pointed at this target.
# (The ETC always computes ALL modes per star, so changing the selection is free.)
DEFAULT_MODES = ["niriss_soss", "nirspec_prism", "nirspec_g395h",
                 "nircam_f322w2", "miri_lrs"]

# Per-planet system defaults (star, geometry, T14, UV spectrum) live in planets.py.
