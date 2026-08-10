"""JWST time-series instrument-mode registry + paths for the noise backend.

Each ``MODES`` entry carries the Pandeia configuration used by
``pandeia_worker.py`` (running in the selected backend's conda env), display
metadata, and an illustrative systematic noise floor.

Mode tokens are the engine mode names of the supported (2026-era) releases;
``engine_mode()`` applies any per-backend renames (none at present -- the
3.0-era ``ssgrism`` token retired with the legacy backend), and both the
production path (``noise.noise_job``) and the parity harness go through it.

Noise floors (``floor_ppm_suggested``) are ILLUSTRATIVE planning values, never
defaults: ``detect.evaluate_mode`` and ``noise.depth_error_bins`` require
``floor_spec`` explicitly and the GUI preselects no floor, so a floor enters a
result only through a recorded explicit choice. The floor uses PandExo
semantics (sigma_final = max(sigma_random, floor) on the final bins), so a
15-40 ppm floor DOMINATES any well-observed target, while zero claims a
precision nobody has demonstrated -- neither is a neutral default, so the tool
refuses to pick one. The values are per-mode planning suggestions INFORMED BY
the Greene et al. 2016 convention (20/30/50 ppm for NIRISS/NIRCam/MIRI), not
that convention verbatim (here: NIRSpec 15-20, NIRISS 20, NIRCam 25, MIRI 40);
no value here is a measured end-to-end floor. Any caption describing the
prefills must describe THESE values, not Greene's.

Noise sensitivity factor (``noise_infl``): optional multiplier on the Pandeia
random sigma, DEFAULT 1.0 for every mode. Published achieved-vs-predicted
ratios live in ``LITERATURE_NOISE_FACTORS`` as reference points only, never
applied by default. Unlike the floor, it averages down with transits.
"""
from __future__ import annotations

import os
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent        # pandeia_worker.py lives here
# parents[2] is the repo root in an editable checkout; the src/jwst_tool
# marker tells a checkout apart from a site-packages install.
_REPO_DIR = Path(__file__).resolve().parents[2]
_IN_CHECKOUT = (_REPO_DIR / "src" / "jwst_tool").is_dir()

# INPUT data root (the minimal synphot CDBS). JWST_TOOL_DATA_DIR overrides;
# a site-packages install must set it -- fail loudly, no silent fallbacks.
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

# Pandeia backend environment (the real STScI ETC engine, PandExo's core).
# The worker runs in its own conda env; noise.run_pandeia refuses loudly if
# the python is missing.
#
# BACKEND SELECTION (JWST_TOOL_BACKEND): DEFAULT "current" = the supported
# STScI release as a MATCHED TRIPLE (pandeia.engine == 2026.7 +
# pandeia_data-2026.7-jwst + pandeia_psfs-2026.7-jwst), enforced by
# `pandeia_worker._check_backend_match` and recorded in "__provenance__" and
# the cache fingerprint. "archival_2026_2" (the old 2026.2 tuple, kept under
# its honest name -- never silently repoint "current", or old caches would
# look like current-release output) is reproducibility-only. Switching
# backends self-invalidates caches.
#
# PORTABILITY: refdata/psf default under DATA_DIR. There is deliberately NO
# baked-in interpreter path: the backend env is machine-specific and must come
# from JWST_TOOL_PANDEIA_PYTHON. JWST_TOOL_PANDEIA_{PYTHON,REFDATA,PSF_DIR}
# override any path per-machine.
_SUPPORTED_PANDEIA_RELEASE = "2026.7"

_BACKENDS = {
    "current": dict(
        python=None,          # no baked-in path: see JWST_TOOL_PANDEIA_PYTHON
        refdata=str(DATA_DIR / "pandeia_data-2026.7-jwst"),
        psf=str(DATA_DIR / "pandeia_psfs-2026.7-jwst"),
        release="2026.7",
        supported=True,
        status="Pandeia 2026.7 / pandeia_data-2026.7-jwst / "
               "pandeia_psfs-2026.7-jwst (the STScI-supported release, "
               "enforced as a matched triple)"),
    "archival_2026_2": dict(
        python=None,
        refdata=str(DATA_DIR / "pandeia_data-2026.2-jwst"),
        psf=str(DATA_DIR / "pandeia_psfs-2026.2-jwst"),
        release="2026.2",
        supported=False,
        status="ARCHIVAL Pandeia 2026.2 / pandeia_data-2026.2-jwst "
               "(reproducibility only; STScI supports 2026.7 and labels this "
               "release archival -- NOT suitable for planning new proposals)"),
}
JWST_TOOL_BACKEND = os.environ.get("JWST_TOOL_BACKEND", "current").lower()
if JWST_TOOL_BACKEND not in _BACKENDS:
    raise RuntimeError(
        f"JWST_TOOL_BACKEND={JWST_TOOL_BACKEND!r} unknown; choose "
        f"{sorted(_BACKENDS)} -- 'current' is the supported "
        f"{_SUPPORTED_PANDEIA_RELEASE} triple, 'archival_2026_2' is the "
        "reproducibility-only backend. (The Pandeia 3.0 'legacy' backend "
        "was removed.)")
_BE = _BACKENDS[JWST_TOOL_BACKEND]
BACKEND_STATUS = _BE["status"]
BACKEND_RELEASE = _BE["release"]
BACKEND_IS_SUPPORTED = _BE["supported"]

# The PANDEIA backend interpreter (machine-specific, no portable default;
# require_pandeia_python() turns a missing setting into one actionable
# message). Named PICASO_PYTHON until 2026-08-05, from before the repo had a
# real PICASO provider; the PICASO env is JWST_TOOL_PICASO_REFDATA.
PANDEIA_PYTHON = os.environ.get("JWST_TOOL_PANDEIA_PYTHON", _BE["python"])


def require_pandeia_python() -> str:
    """Return the backend interpreter path, or raise one actionable error."""
    if PANDEIA_PYTHON:
        return PANDEIA_PYTHON
    raise RuntimeError(
        "No Pandeia backend interpreter configured. The engine runs in its own "
        "environment (heavy dependencies), and its path is machine-specific, so "
        "there is no default to fall back on.\n"
        f"  Set JWST_TOOL_PANDEIA_PYTHON to the python of an environment with "
        f"pandeia.engine=={_BE['release']} installed, e.g.\n"
        "    export JWST_TOOL_PANDEIA_PYTHON=/path/to/envs/pandeia/bin/python\n"
        f"  Backend '{JWST_TOOL_BACKEND}' also expects\n"
        f"    refdata: {_BE['refdata']}\n"
        f"    PSFs:    {_BE['psf'] or '(none: this backend carries its own)'}\n"
        "  See README 'Pandeia backend' and `jwst-tool data` for the setup "
        "steps.")


PANDEIA_REFDATA = os.environ.get("JWST_TOOL_PANDEIA_REFDATA", _BE["refdata"])
# pandeia_data >= 2026 splits the PSF library out of the refdata tree; the
# engine reads it from $PSF_DIR. Passed to the worker, preflighted, and joins
# the cache key.
PANDEIA_PSF_DIR = os.environ.get("JWST_TOOL_PANDEIA_PSF_DIR", _BE["psf"])
# Minimal synphot CDBS assembled for this tool: phoenix grid, 2MASS Ks
# bandpass, CALSPEC Vega. The Vega copy is only an OFFLINE pin for the
# tool-side synphot/stsynphot (stellar.py + the worker); the engine
# normalizes against its OWN refdata Vega (the two agree to 0.08 mmag in
# Ks). `jwst-tool data` reports each piece.
PYSYN_CDBS = str(DATA_DIR / "cdbs")

# Engine-generation mode-name renames, keyed by backend; every path resolves
# through engine_mode() -- one source of truth, never a parity-only rename.
# Both supported backends speak the same 2026-era tokens, so the maps are
# empty since the 3.0 "legacy" backend was removed (MODES stored the 3.0
# "ssgrism" token until then; it is now "lw_tsgrism" directly -- cache-safe,
# because job dicts always carried the RESOLVED token). The machinery stays:
# the next engine generation that renames a mode gets one entry here, and the
# assert below forces a decision for every backend.
_MODE_RENAMES = {
    "current": {},
    "archival_2026_2": {},
}
# Every backend token MUST appear above: a missing entry would submit an
# unresolved token to an engine that may hard-reject it mid-run.
assert set(_MODE_RENAMES) == set(_BACKENDS), (
    f"mode-rename map {sorted(_MODE_RENAMES)} does not cover the backend "
    f"tokens {sorted(_BACKENDS)}")
ENGINE_MODE_RENAMES = _MODE_RENAMES[JWST_TOOL_BACKEND]


def engine_mode(instrument: str, mode: str) -> str:
    """Resolve a registry mode token to the name the ACTIVE backend accepts.

    Returns ``mode`` unchanged when the active backend needs no rename. Both
    ``noise.noise_job`` and ``tests/parity`` go through here.
    """
    return ENGINE_MODE_RENAMES.get(instrument, {}).get(mode, mode)


# Star normalization is band-integrated 2MASS Ks vegamag inside the worker
# (the web-ETC convention) -- never the retired monochromatic at_lambda
# shortcut, which mis-scaled cool/warm stars by ~1-4% and fed that error into
# saturation/ngroup selection.

# Fixed categorical color per mode, never re-assigned when the selection
# changes. Every color holds >= 3:1 contrast on white (WCAG 2.2 non-text) and
# the set passes the dataviz palette validator in wavelength-adjacency order;
# MODE_MARKER is the secondary, color-independent encoding.
_COLORS = ["#2a78d6", "#199e70", "#a35a00", "#007a00",
           "#4a3aa7", "#d43f3e", "#a83a9e", "#c2571f"]

# Fixed marker shape per mode: series must never rely on color alone
# (grayscale print, color-vision deficiency).
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

# PandExo-compatible hard maximum group counts per instrument (NIRCam grism
# capped at 100). Every mode's ngroup_max must respect its instrument's cap
# (asserted at import); the worker clamps its ramp to [ngroup_min, ngroup_max].
PANDEXO_NGROUP_MAX = {"nircam": 100, "niriss": 30}

# NIRISS SOSS: 30 groups is the APT hard limit for NISRAPID/SUBSTRIP256 TSO,
# adopted by PandExo master -- a schedulability limit, not just parity.
# For nirspec/miri PandExo's optimizer is effectively unbounded: SATURATION
# picks the ramp, not a registry cap. A self-imposed cap made the tool's
# ramps/sigmas silently diverge from PandExo/ETC output (history: notes.md).
PANDEXO_UNBOUNDED_NGROUP = 65535

# Extraction strategy + sky background are pinned to PandExo's TSO conventions
# (per-instrument apertures/annuli; background "ecliptic" + background_level
# "medium" -- BOTH keys required together), NOT pandeia's generic point-source
# defaults: the default-strategy mismatch measured 8-20% in extracted flux.
# wl_min/wl_max: the usable science bandpass, intersected with the forward
# model's 1-15 um coverage (short edge = the H2-H2 CIA table).
# readout_pattern is pinned EXPLICITLY on every mode (NRSRAPID/NISRAPID/RAPID/
# FASTR1, PandExo's TSO choices): engine defaults are non-TSO patterns and
# drift between releases. Never leave readout_pattern implicit on a new mode.
#
# SCOPE (deliberate, reviewers keep re-finding it): each entry is ONE fixed
# detector configuration (subarray + readout pattern), not the whole
# instrument mode. The tool ranks these fixed configurations; it does NOT
# search alternate subarrays (PRISM multistripe, other SOSS substrips) or
# optimize the readout pattern. The GUI says so and shows each mode's
# configuration in the details table.
#
# ngroup_min equals pandeia 2026.7 `mingroups` for each mode's detector
# (pandeia_data-2026.7-jwst/jwst/<instrument>/config.json detector_config:
# nirspec/niriss/nircam 1, miri 2). PandExo reads the same field
# (timing_det_pars['mingroups'] at the pinned parity commit 34e42d81, no
# instrument branching), so both tools search the same ramp space. Verified
# against jwst-docs 2026-08-09: NIRSpec BOTS permits 1-group NRSRAPID for
# very bright targets (2 recommended); NIRISS SOSS permits 1-group NISRAPID
# (APT warns at 1); MIRI FASTR1 permits 2 groups with 5+ recommended for
# calibration accuracy.
# ngroup_warn_below is a DISCLOSURE threshold, not a bound: a selected ramp
# below it still ranks, with an instrument-specific warning from detect
# (reasons in NGROUP_WARN_REASON; thresholds verified on jwst-docs
# 2026-08-09: NIRSpec/NIRISS warn at 1 group; NIRCam TSO guidance says avoid
# data saturating in fewer than 4 groups, to limit reliance on the linearity
# correction; MIRI guidance calls 2-5 group ramps very difficult to
# calibrate accurately, 5+ recommended). History: through 0.24.0 the tool
# floored NIR at 2 / MIRI at 5 and reported bright targets "saturated at the
# shortest ramp" where PandExo passed at 1 group (closed 2026-08-09; see
# docs/decision_records.md).
# Instrument-specific reason a short ramp is cautioned (composed into the
# detect warning). Sources: jwst-docs, verified 2026-08-09 -- NIRSpec
# Detector Recommended Strategies; NIRISS SOSS Recommended Strategies;
# NIRCam TSO Recommended Strategies; MIRI TSO/LRS Recommended Strategies.
NGROUP_WARN_REASON = {
    "nirspec": ("1-group NRSRAPID ramps are permitted for very bright "
                "targets but are new since Cycle 4 and lightly tested"),
    "niriss": ("1-group NISRAPID ramps carry an APT calibration warning"),
    "nircam": ("STScI advises avoiding data that saturate in fewer than 4 "
               "groups, to limit reliance on the linearity correction"),
    "miri": ("STScI reports 2-5 group MIRI ramps are very difficult to "
             "calibrate accurately; 5+ groups recommended"),
}

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
        floor_ppm_suggested=20.0, noise_infl=1.0, ngroup_min=1,
        ngroup_warn_below=2, ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
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
        floor_ppm_suggested=15.0, noise_infl=1.0, ngroup_min=1,
        ngroup_warn_below=2, ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
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
        floor_ppm_suggested=15.0, noise_infl=1.0, ngroup_min=1,
        ngroup_warn_below=2, ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
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
        floor_ppm_suggested=20.0, noise_infl=1.0, ngroup_min=1,
        ngroup_warn_below=2, ngroup_max=30,
    ),
    "nircam_f322w2": dict(
        label="NIRCam F322W2",
        instrument="nircam", mode="lw_tsgrism",
        config=dict(instrument=dict(filter="f322w2", disperser="grismr"),
                    detector=dict(subarray="subgrism64", readout_pattern="rapid")),
        strategy=dict(aperture_size=0.4, sky_annulus=[0.5, 1.5]),
        background="ecliptic", background_level="medium",
        wl_min=2.45, wl_max=3.95,
        floor_ppm_suggested=25.0, noise_infl=1.0, ngroup_min=1,
        ngroup_warn_below=4, ngroup_max=100,
    ),
    "nircam_f444w": dict(
        label="NIRCam F444W",
        instrument="nircam", mode="lw_tsgrism",
        config=dict(instrument=dict(filter="f444w", disperser="grismr"),
                    detector=dict(subarray="subgrism64", readout_pattern="rapid")),
        strategy=dict(aperture_size=0.4, sky_annulus=[0.5, 1.5]),
        background="ecliptic", background_level="medium",
        wl_min=3.9, wl_max=4.95,
        floor_ppm_suggested=25.0, noise_infl=1.0, ngroup_min=1,
        ngroup_warn_below=4, ngroup_max=100,
    ),
    "miri_lrs": dict(
        label="MIRI LRS (slitless)",
        instrument="miri", mode="lrsslitless",
        config=dict(detector=dict(subarray="slitlessprism",
                                  readout_pattern="fastr1")),
        strategy=dict(aperture_size=0.6, sky_annulus=[1.0, 2.8]),
        background="ecliptic", background_level="medium",
        wl_min=5.0, wl_max=12.0,
        floor_ppm_suggested=40.0, noise_infl=1.0, ngroup_min=2,
        ngroup_warn_below=6, ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
    ),
}

# Literature achieved-vs-predicted noise ratios: reference points only, never
# applied by default (see module docstring). G395H measured 1.05-1.12x PandExo
# (Gordon et al. 2025); G235H is extrapolated from that (no published number);
# SOSS 1.2x (Radica et al. 2023); NIRCam 1.05 is an editorial placeholder (no
# measurement behind the digit); MIRI LRS ~15-20% (Bouwman et al. 2023);
# PRISM photon-limited (Rustamkulov et al. 2023).
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
    if not (1 <= _m["ngroup_min"] <= _m["ngroup_warn_below"]
            <= _m["ngroup_max"]):
        raise RuntimeError(
            f"mode {_key!r} breaks 1 <= ngroup_min={_m['ngroup_min']} <= "
            f"ngroup_warn_below={_m['ngroup_warn_below']} <= "
            f"ngroup_max={_m['ngroup_max']}; the ramp search and the "
            "below-recommended-ramp warning both assume this ordering.")

MODE_COLOR = {key: _COLORS[i % len(_COLORS)] for i, key in enumerate(MODES)}
MODE_MARKER = {key: _MARKERS[i % len(_MARKERS)] for i, key in enumerate(MODES)}

# GUI default selection (speed-first trio, 2026-08-10 maintainer decision):
# PRISM + G395H cover 0.6-5.25 um including the 4.05 um SO2 band (G395H is
# the default detect-SO2 goal's workhorse), MIRI LRS keeps the mid-IR
# 7-8.5 um SO2 band. All seven modes stay selectable. Since 0.27.0 the ETC
# computes ONLY the selected modes and caches each mode separately, so the
# default run costs three modes and adding a mode later costs exactly that
# mode (the old design computed all seven every first run; the five-mode
# observed-planet default made that ~2.5x slower than needed).
DEFAULT_MODES = ["nirspec_prism", "nirspec_g395h", "miri_lrs"]

# Per-planet system defaults (star, geometry, T14, UV spectrum) live in planets.py.
