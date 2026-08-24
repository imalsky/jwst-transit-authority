"""JWST time-series instrument-mode registry + paths for the noise backend.

Each ``MODES`` entry carries the Pandeia configuration used by
``pandeia_worker.py`` (running in the selected backend's conda env), display
metadata, and an illustrative systematic noise floor.

Mode tokens are the engine mode names of the supported (2026-era) releases;
``engine_mode()`` applies any per-backend renames (none at present), and both
the production path (``noise.noise_job``) and the parity harness go through it.

Noise floors (``floor_ppm_suggested``) are ILLUSTRATIVE planning values, not
measured calibrations: ``detect.evaluate_mode`` and ``noise.depth_error_bins``
require ``floor_spec`` explicitly. The GUI preselects a constant floor using
these values, displays the selection, and records it in provenance; users can
choose no floor or upload a wavelength-dependent table. The floor uses PandExo
semantics (sigma_final = max(sigma_random, floor) on the final bins), so a
15-40 ppm floor DOMINATES any well-observed target, while zero assumes a
precision no program has demonstrated. The values are per-mode planning suggestions INFORMED BY
the Greene et al. 2016 convention (20/30/50 ppm for NIRISS/NIRCam/MIRI), not
that convention verbatim (here: NIRSpec 15-20, NIRISS 20, NIRCam 25, MIRI 40);
no value here is a measured end-to-end floor. Any caption describing the
prefills must describe THESE values, not Greene's.

Noise sensitivity factor (``noise_infl``): optional multiplier on the Pandeia
random sigma, DEFAULT 1.0 for every mode. Published achieved-vs-reference
ratios live in ``LITERATURE_NOISE_FACTORS`` as reference points only, never
applied by default; read the per-entry comments there, because the published
REFERENCE differs by paper (PandExo's prediction vs the bare photon-noise
limit). Unlike the floor, it averages down with transits.

Band edges (``wl_min``/``wl_max``): the mode's usable science bandpass from
a published source, intersected with the forward model's 1-15 um coverage
(short edge = the H2-H2 CIA table). Per instrument:
  * NIRSpec -- the BOTS/S1600A1 table, SUB2048 column: jwst-docs "NIRSpec
    BOTS Wavelength Ranges and Gaps" Table 1, identical to Birkmann et al.
    2022 (A&A 661, A83) Table 2. NOT the nominal disperser table, which is
    wider: the BOTS values already carry the red-end detector cutoffs
    (G235x 3.07/3.12 there against a nominal 3.17).
  * NIRISS -- the Pandeia order ranges for GR700XD (0.83-2.81 order 1,
    0.63-1.26 order 2), rounded inward.
  * NIRCam -- the filter half-power points measured from the shipped
    transmission curves, rounded INWARD to 0.05 um.
  * MIRI -- LRS 5-12 um; the 5.0 short edge is the jwst-docs caution about
    the slitless dispersion turnover below 4.5 um, not a rounding.
Every edge is checked against the Pandeia extracted grid the worker returns,
and pinned to its source by tests/unit/test_instruments_registry.py. Never
narrow an edge without a sourced reason on the entry: a G395M red edge of
5.10 shipped for weeks against a jwst-docs number that does not exist.

Saturation vocabulary: Pandeia's ``fraction_saturation`` is the fraction of
the engine's per-mode ``saturation_fullwell``, which is NOT the physical full
well and is not the same fraction of it on every instrument -- NIRCam
lw_tsgrism 58,100 e- (pandeia_data deliberately holds time-series modes at
70% of the ~83,000 e- well, and jwst-docs states the NIRCam grism-TSO
saturation limits "as 70% of the pixel well capacity"), NIRISS 72,000 e- (a
soft-nonlinearity value below the ~87,300 e- physical well), NIRSpec 65,000
e-, MIRI 193,655 e-. ``sat_limit`` multiplies THAT number, so an 80% limit is
~56% of the NIRCam well but 80% of NIRSpec's adopted value. Never call it a
physical full-well fraction.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

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


def atomic_write(path: Path, writer) -> None:
    """Write ``writer(fh)`` to ``path`` atomically (temp file in the same
    directory, fsync, ``os.replace``).

    The caches under OUTPUT_DIR are shared by concurrent sessions and
    subprocesses; a direct write to the final path lets a reader (or a
    second same-key writer) see a partial file, and a killed writer leaves
    one behind permanently. With the rename, a reader sees the old content
    or the complete new content, never a torn file. ``fh`` is binary.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=path.name + ".tmp.")
    try:
        with os.fdopen(fd, "wb") as fh:
            writer(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

# Pandeia backend environment (the real STScI ETC engine, PandExo's core).
# The worker runs in its own conda env; noise.run_pandeia refuses loudly if
# the python is missing.
#
# BACKEND SELECTION (JWST_TOOL_BACKEND): DEFAULT "current" = the supported
# STScI release as a MATCHED TRIPLE (pandeia.engine == 2026.7 +
# pandeia_data-2026.7-jwst + pandeia_psfs-2026.7-jwst), enforced by
# `pandeia_worker._check_backend_match` and recorded in "__provenance__" and
# the cache fingerprint. Older backends are deliberately not selectable: this
# release neither ships nor validates their complete matched data triples.
# Switching a future validated backend would self-invalidate caches.
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
}
JWST_TOOL_BACKEND = os.environ.get("JWST_TOOL_BACKEND", "current").lower()
if JWST_TOOL_BACKEND not in _BACKENDS:
    raise RuntimeError(
        f"JWST_TOOL_BACKEND={JWST_TOOL_BACKEND!r} unknown; choose "
        f"{sorted(_BACKENDS)}. This release validates only the supported "
        f"{_SUPPORTED_PANDEIA_RELEASE} matched triple; archival backends are "
        "not selectable.")
_BE = _BACKENDS[JWST_TOOL_BACKEND]
BACKEND_STATUS = _BE["status"]
BACKEND_RELEASE = _BE["release"]
# (no BACKEND_IS_SUPPORTED export. The "supported" flag stays in _BACKENDS so
# a future entry can declare itself unsupported.)

# The PANDEIA backend interpreter (machine-specific, no portable default;
# require_pandeia_python() turns a missing setting into one actionable
# message).
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
        "  See the README's Install section and `jwst-tool data` for the setup "
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
# Every supported backend speaks the same 2026-era tokens, so the maps are
# empty. The machinery stays: the next engine generation that renames a mode
# gets one entry here, and the assert below forces a decision for every
# backend.
_MODE_RENAMES = {
    "current": {},
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
           "#4a3aa7", "#d43f3e", "#a83a9e", "#006c8e",
           "#c2185b", "#8b46c8", "#00929e", "#d81b8c"]
# The 8th slot (nirspec_g395m) was re-chosen while still unused:
# the original "#c2571f" sat at deltaE(Lab) ~16 from the G235H orange and ~24
# from wavelength-neighbor F444W; "#006c8e" holds 5.94:1 on white and
# deltaE >= 38 to every existing color (the palette's own internal minimum
# is 32.6), >= 54 to its wavelength-adjacent neighbors (G395H, F444W).
# Slots 9-12 (G140H rose, G235M violet, SOSS-ord2 cyan, F277W pink) fill the
# only hue niches the first 8 left open. The 12-slot set passes the dataviz
# palette validator in wavelength-adjacency order (the two flags it reports
# are properties of the frozen first 8: g395m's chroma 0.098 vs the 0.10
# floor, and the g395h/f444w deutan pair, both carried by the markers).
# Contrast on white 3.75-5.87:1 for the new four; a 12-slot palette cannot
# keep the old Lab deltaE >= 32 minimum -- the new colors hold >= 20 to
# their nearest neighbor and the fixed per-mode MARKERS stay the
# color-independent encoding.

# Fixed marker shape per mode: series must never rely on color alone
# (grayscale print, color-vision deficiency).
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*",
            "<", ">", "p", "h"]

# Hard maximum group counts per instrument, matching BOTH APT's template
# ranges and PandExo master: NIRCam grism time series 1-100, NIRISS SOSS
# NISRAPID 1-30. Every mode's ngroup_max must respect its instrument's cap
# (asserted at import); the worker clamps its ramp to [ngroup_min, ngroup_max].
PANDEXO_NGROUP_MAX = {"nircam": 100, "niriss": 30}

# NIRISS SOSS: 30 is the APT range limit for the NISRAPID readout pattern
# (jwst-docs SOSS template parameters: "for NISRAPID the range is 1-30";
# NIS allows 1-200), not a subarray property, and PandExo master carries the
# same 30. NIRCam grism time series is 1-100 in APT, matching PandExo.
#
# NIRSpec and MIRI get PANDEXO_UNBOUNDED_NGROUP because SATURATION, not a
# registry cap, picks the ramp there; a self-imposed cap made the tool's
# ramps/sigmas silently diverge from PandExo/ETC output (history: notes.md).
# 65535 is PandExo's own value for both, and for NIRSpec it is exactly APT's
# stated maximum NUMBER OF GROUPS/INTEGRATION. For MIRI it is NOT the real
# ceiling: APT limits a MIRI integration to 2000 s, which is ~12,575 groups
# on SLITLESSPRISM (tframe 0.15904 s). That limit is far above anything a
# saturation-limited search reaches (MIRI's ramp is background-limited: the
# faintest parity star lands at 1021 groups / 162 s), so it is DISCLOSED by
# detect via INT_LENGTH_LIMIT_S rather than enforced here.
PANDEXO_UNBOUNDED_NGROUP = 65535

# Integration-length limits per instrument, checked against the SELECTED
# ramp's cycle time and reported by detect as a DISCLOSURE, never a bound
# (the tool ranks configurations; APT adjudicates them). Sources, jwst-docs:
#   nirspec -- Detector Recommended Strategies: "The maximum recommended
#     integration length is 1,500 seconds ... Longer integrations can be
#     taken, but the benefits over taking 2 shorter integrations will be
#     limited due to cosmic ray effects."
#   miri    -- MIRI LRS TSOs: "MIRI integration duration may not be greater
#     than 2000 seconds." (a hard APT limit, not advice)
# NIRISS and NIRCam are bounded by their group caps (30 / 100) long before
# any integration-length rule bites, so they have no entry.
# The test uses t_cycle, which is the ramp plus the between-integration
# reset, so it triggers marginally EARLY -- the conservative direction.
INT_LENGTH_LIMIT_S = {
    "nirspec": (1500.0, "STScI-recommended maximum integration length"),
    "miri": (2000.0, "APT maximum MIRI integration duration"),
}

# Extraction strategy + sky background are pinned to PandExo's TSO conventions
# (per-instrument apertures/annuli; background "ecliptic" + background_level
# "medium" -- BOTH keys required together), NOT pandeia's generic point-source
# defaults: the default-strategy mismatch measured 8-20% in extracted flux.
# wl_min/wl_max: see the module docstring for the per-instrument source of
# every band edge; each entry below carries its own citation.
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
# against jwst-docs 2026-08-18: NIRSpec BOTS permits 1-group NRSRAPID for
# very bright targets (2 recommended); NIRISS SOSS permits 1-group NISRAPID
# (APT warns at 1); MIRI FASTR1 permits 2 groups with 5 recommended for
# calibration accuracy.
# ngroup_warn_below is a DISCLOSURE threshold, not a bound: a selected ramp
# below it still ranks, with an instrument-specific warning from detect
# (reasons in NGROUP_WARN_REASON). Do not restore floors above
# pandeia's mingroups: that wrongly reported bright targets "saturated at
# the shortest ramp" where PandExo passed (notes.md, Decision records).
#
# Instrument-specific reason a short ramp is cautioned (composed into the
# detect warning). Each string paraphrases the quoted jwst-docs sentence
# below it; re-verified 2026-08-18 against those pages.
#   nirspec -- NIRSpec Detector Recommended Strategies: "The minimum
#     recommended number of groups is 2, although 1 can be used if really
#     necessary (e.g., very bright targets in BOTS observations)."
#   niriss  -- NIRISS SOSS template parameters: "1 is likely to result in
#     lower calibration accuracy. APT provides a warning if NUMBER OF
#     GROUPS/INTEGRATION=1."
#   nircam  -- NIRCam Time-Series Observation Recommended Strategies:
#     "Observers should avoid saturation of any part of their data that
#     occurs in less than 4 groups of their integrations. This will provide
#     3 groups for determining the flux while avoiding undue reliance on the
#     linearity correction." NOTE the rule is about the groups-to-saturation
#     count, while the test below uses the SELECTED ramp; since the selected
#     ramp is ~sat_limit x groups-to-saturation, this warns one step early.
#   miri    -- MIRI LRS TSOs: "The minimum Ngroups required is 2. However,
#     experience from early cycles shows that the quality of the calibration
#     is sub-optimal for small numbers of groups. Integrations with 2-5
#     groups are very difficult to calibrate accurately, due to the presence
#     of persistence effects on the detector. Therefore, integrations should
#     ideally use >5 or even >10 groups."
NGROUP_WARN_REASON = {
    "nirspec": ("STScI's minimum recommended NIRSpec ramp is 2 groups; 1 is "
                "allowed only if really necessary, e.g. a very bright BOTS "
                "target"),
    "niriss": ("1-group NISRAPID ramps are likely to give lower calibration "
               "accuracy, and APT warns on them"),
    "nircam": ("STScI advises avoiding data that saturate in fewer than 4 "
               "groups, to limit reliance on the linearity correction"),
    "miri": ("STScI reports 2-5 group MIRI ramps are very difficult to "
             "calibrate accurately because of detector persistence; more "
             "than 5 groups is recommended, more than 10 preferred"),
}

# r_native_med: the mode's typical native resolving power, shown in the GUI
# mode picker. Median of R(lambda) from the mode's 2026.7 refdata dispersion
# file over the registry band, rounded to a readable figure (NIRCam from
# jwst_nircam_disp_*.fits, the LW grism file, which carries no disperser
# token in its name). For the NIRSpec gratings the rounding deliberately
# lands on the published nominal figures -- jwst-docs quotes R ~ 2,700 for
# the high-resolution and R ~ 1,000 for the medium-resolution gratings, and
# the measured medians (2757/2722/2733 and 1019/1018) agree to ~2% -- so
# G395H reads 2700, not the 2800 a blind round would give.
# RE-MEASURE ON ANY BAND OR REFDATA CHANGE: the median is taken over the
# registry band, so moving an edge moves this number.
# Display metadata only -- the LSF operator reads the full R(lambda) curve
# from the worker, never this number.
MODES = {
    "nirspec_prism": dict(
        label="NIRSpec PRISM",
        instrument="nirspec", mode="bots",
        config=dict(instrument=dict(disperser="prism", filter="clear"),
                    detector=dict(subarray="sub512",
                                  readout_pattern="nrsrapid")),
        strategy=dict(aperture_size=0.7, sky_annulus=[0.75, 1.5]),
        background="ecliptic", background_level="medium",
        # BOTS Table 1, PRISM/CLEAR on SUB512: 0.60-5.30. 1.0 is the model's
        # short edge (the band contract in the module docstring); 5.30 is the
        # instrument edge and the Pandeia grid runs to 5.298, with no
        # throughput cliff before it (the reddest pixels still carry ~1% of
        # the peak extracted rate).
        wl_min=1.0, wl_max=5.30,
        r_native_med=110,
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
        # BOTS Table 1: 2.87-3.72, 3.82-5.18 (the NRS1/NRS2 gap is measured
        # from the Pandeia grid, never hard-coded here)
        wl_min=2.87, wl_max=5.18,
        r_native_med=2700,
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
        # BOTS Table 1: 1.66-2.20, 2.27-3.07. The red edge is the BOTS
        # detector cutoff, BELOW the nominal disperser table's 3.17.
        wl_min=1.66, wl_max=3.07,
        r_native_med=2700,
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
        # Pandeia's order-1 CLEAR range is 0.83-2.81 um (jwst-docs quotes the
        # three orders together as 0.6-2.8); 1.0 is the model's short edge
        # (same intersection contract as PRISM), 2.8 is 2.81 rounded inward.
        wl_min=1.0, wl_max=2.8,
        r_native_med=970,
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
        # F322W2 half-power points 2.425-4.012 (measured from the shipped
        # transmission curve), rounded inward to 0.05. jwst-docs quotes the
        # F322W2 grism band as 2.4-4.0, and the published ERS analysis of
        # WASP-39 b used 2.420-4.025; the Pandeia grid runs past it to 4.22.
        wl_min=2.45, wl_max=4.00,
        r_native_med=1400,
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
        # F444W half-power points 3.881-5.009, rounded inward to 0.05. The
        # red edge also sits inside the detector cutoff: jwst-docs notes the
        # F444W footprint above 5.03 um runs off the array. The Pandeia grid
        # ends at 4.998, so this edge is not what binds.
        wl_min=3.9, wl_max=5.00,
        r_native_med=1700,
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
        # LRS covers 5-12 um. The 5.0 short edge is NOT cosmetic: jwst-docs
        # warns that the slitless dispersion profile turns over below 4.5 um,
        # mapping several wavelengths onto the same pixels, so "caution is
        # warranted when analyzing spectra obtained in slitless mode at
        # wavelengths below 5 um". The Pandeia grid itself runs 5.02-13.86.
        wl_min=5.0, wl_max=12.0,
        # Native R RISES with wavelength here (42 at 5 um, 150 median,
        # 209 at 12 um) -- the opposite of the gratings.
        r_native_med=150,
        floor_ppm_suggested=40.0, noise_infl=1.0, ngroup_min=2,
        ngroup_warn_below=6, ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
    ),
    # Appended LAST on purpose: MODE_COLOR/MODE_MARKER key by
    # enumeration order, and per-mode colors are never re-assigned, so a new
    # mode may only be appended, never inserted mid-registry. Same band as
    # G395H at ~4x lower R; the medium-resolution grating trades resolving
    # power for a brighter saturation limit than PRISM while keeping the full
    # 3-5 um band on one detector pair (the G395M-vs-G395H duty-cycle /
    # saturation trade). Tokens verified against
    # pandeia_data-2026.7-jwst/jwst/nirspec/config.json (and the 2026.2
    # tree): bots dispersers include "g395m"; config_constraints allow
    # f290lp + sub2048 for it; readout_patterns include "nrsrapid";
    # range.s1600a1.f290lp spans 2.87-5.27 um.
    "nirspec_g395m": dict(
        label="NIRSpec G395M",
        instrument="nirspec", mode="bots",
        config=dict(instrument=dict(disperser="g395m", filter="f290lp"),
                    detector=dict(subarray="sub2048",
                                  readout_pattern="nrsrapid")),
        strategy=dict(aperture_size=0.7, sky_annulus=[0.75, 1.5]),
        background="ecliptic", background_level="medium",
        # BOTS Table 1: G395M/F290LP on SUB2048 covers 2.87-5.18, the SAME
        # band as G395H (which only adds the NRS1/NRS2 gap). The Pandeia
        # grid agrees: usable pixels run 2.871-5.177 with none dropped.
        wl_min=2.87, wl_max=5.18,
        r_native_med=1000,
        floor_ppm_suggested=15.0, noise_infl=1.0, ngroup_min=1,
        ngroup_warn_below=2, ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
    ),
    # Slots 9-12, appended in this order on purpose -- see the palette note
    # above. Tokens verified against pandeia_data-2026.7-jwst config.json
    # files and one live Pandeia 2026.7 calculation per mode; r_native_med is
    # the median R(lambda) of the refdata dispersion file over the registry
    # band.
    #
    # G140H: the only high-R coverage below G235H's 1.66 um. Instrument band
    # 0.97-1.83 (f100lp; measured good-bin grid 0.970-1.831, matching
    # Birkmann et al. 2022 Table 2); wl_min = 1.0 is the model's short edge
    # (same intersection contract as PRISM/SOSS). NRS1/NRS2 detector gap
    # measured at 1.314-1.351 um by the same largest-grid-step method that
    # reproduces the shipped G235H/G395H display edges.
    "nirspec_g140h": dict(
        label="NIRSpec G140H",
        instrument="nirspec", mode="bots",
        config=dict(instrument=dict(disperser="g140h", filter="f100lp"),
                    detector=dict(subarray="sub2048",
                                  readout_pattern="nrsrapid")),
        strategy=dict(aperture_size=0.7, sky_annulus=[0.75, 1.5]),
        background="ecliptic", background_level="medium",
        wl_min=1.0, wl_max=1.83,
        r_native_med=2700,   # measured median 2734 over 1.0-1.83 um
        floor_ppm_suggested=15.0, noise_infl=1.0, ngroup_min=1,
        ngroup_warn_below=2, ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
    ),
    # G235M: the medium-R companion to G235H (the same trade G395M offers
    # against G395H). Band 1.66-3.12: the measured good-bin grid
    # (1.661-3.120), matching Birkmann et al. 2022 Table 2; the medium
    # gratings sit entirely on NRS1, no detector gap.
    "nirspec_g235m": dict(
        label="NIRSpec G235M",
        instrument="nirspec", mode="bots",
        config=dict(instrument=dict(disperser="g235m", filter="f170lp"),
                    detector=dict(subarray="sub2048",
                                  readout_pattern="nrsrapid")),
        strategy=dict(aperture_size=0.7, sky_annulus=[0.75, 1.5]),
        background="ecliptic", background_level="medium",
        wl_min=1.66, wl_max=3.12,
        r_native_med=1000,   # measured median 1018 over 1.66-3.12 um
        floor_ppm_suggested=15.0, noise_infl=1.0, ngroup_min=1,
        ngroup_warn_below=2, ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
    ),
    # SOSS order 2: same optics and subarray as order 1, extracted at
    # strategy order=2. Instrument order-2 band is 0.63-1.26 um
    # (pandeia range gr700xd_2), so the model's 1.0 um short edge leaves a
    # NARROW usable band, 1.0-1.26 um -- deliberate: it is the only
    # higher-R-than-PRISM coverage at the short end besides G140H.
    #
    # ORDER 1 AND ORDER 2 ARE ONE EXPOSURE, not two. SUBSTRIP256 records both
    # traces in a single readout, so selecting both registry modes costs one
    # observation, not two, while the tool's transit count is per mode.
    # Saturation follows from the same fact: Pandeia builds one CombinedSignal
    # over all SOSS orders, so `fraction_saturation` (and therefore the
    # `saturated` verdict) is measured over the WHOLE detector image and is
    # driven by the brighter order-1 trace -- the two modes always report the
    # identical ramp, sat_frac and cadence. jwst-docs puts the SUBSTRIP256
    # bright limit at J ~ 8.5 in order 1 but J ~ 6.3 in order 2, so between
    # those magnitudes the order-2 trace is still clean while this mode is
    # reported saturated. That is conservative, not optimistic, and detect
    # discloses it per mode; changing the verdict would mean giving the ramp
    # search a per-order saturation measure, which also drops PandExo parity.
    "niriss_soss_ord2": dict(
        label="NIRISS SOSS (ord 2)",
        instrument="niriss", mode="soss",
        config=dict(instrument=dict(filter="clear", disperser="gr700xd"),
                    detector=dict(subarray="substrip256",
                                  readout_pattern="nisrapid")),
        strategy=dict(order=2),
        background="ecliptic", background_level="medium",
        wl_min=1.0, wl_max=1.26,
        r_native_med=1140,   # measured median 1137 over 1.0-1.26 um
        floor_ppm_suggested=20.0, noise_infl=1.0, ngroup_min=1,
        ngroup_warn_below=2, ngroup_max=30,
    ),
    # F277W: the fourth NIRCam LW grism TSO filter this registry covers.
    # Band 2.45-3.1 = the filter's half-power points (2.419-3.130, measured
    # from the shipped transmission curve) rounded inward to 0.05, the same
    # convention the F322W2 (2.425-4.012 -> 2.45-4.00) and F444W
    # (3.881-5.009 -> 3.9-5.00) entries follow. F356W is the one grism-TSO
    # filter this registry leaves out; F322W2 already spans its band.
    "nircam_f277w": dict(
        label="NIRCam F277W",
        instrument="nircam", mode="lw_tsgrism",
        config=dict(instrument=dict(filter="f277w", disperser="grismr"),
                    detector=dict(subarray="subgrism64", readout_pattern="rapid")),
        strategy=dict(aperture_size=0.4, sky_annulus=[0.5, 1.5]),
        background="ecliptic", background_level="medium",
        wl_min=2.45, wl_max=3.1,
        r_native_med=1300,   # measured median 1276 over 2.45-3.1 um
        floor_ppm_suggested=25.0, noise_infl=1.0, ngroup_min=1,
        ngroup_warn_below=4, ngroup_max=100,
    ),
}

# Literature achieved-vs-reference noise ratios: reference points only, never
# applied by default (see module docstring).
#
# THE REFERENCE IS NOT THE SAME IN EVERY PAPER, so these numbers are not
# strictly comparable with each other and none of them is a calibration:
#   * vs PANDEXO'S PREDICTION -- Gordon et al. 2025 (JWST COMPASS, uniform
#     reanalysis of seven G395H transmission spectra): real error bars
#     average 5% larger on NRS1 and 12% larger on NRS2 than PandExo predicts.
#     This is the only entry measured against the quantity this tool computes.
#   * vs the BARE PHOTON-NOISE LIMIT -- Radica et al. 2023 (WASP-96 b,
#     NIRISS/SOSS): "an average precision of 1.2 and 1.4x the photon noise
#     for orders 1 and 2, respectively". Bouwman et al. 2023 (MIRI/LRS):
#     measured noise 15-20% above random-noise simulations over
#     6.8 < lambda < 11 um. Pandeia's prediction includes read noise and
#     background, so these overstate the gap against it, more so where the
#     target is not photon-dominated.
#   * NOT MEASURED -- Rustamkulov et al. 2023 report the PRISM ERS light
#     curves as very close to the photon-plus-read-noise expectation, hence
#     1.0. The NIRCam 1.05 is an editorial placeholder with no measurement
#     behind the digit. The G235H/G395M/G140H/G235M entries are extrapolated
#     from G395H; no published per-mode number was found for them.
LITERATURE_NOISE_FACTORS = {
    "nirspec_prism": 1.0,
    "nirspec_g395h": 1.10,
    "nirspec_g235h": 1.10,
    "nirspec_g395m": 1.10,   # extrapolated from G395H (no published number)
    "nirspec_g140h": 1.10,   # extrapolated from G395H (no published number)
    "nirspec_g235m": 1.10,   # extrapolated from G395H (no published number)
    "niriss_soss": 1.20,       # Radica et al. 2023, order 1
    "niriss_soss_ord2": 1.40,  # Radica et al. 2023, order 2 -- the SAME paper
                               # measures order 2 worse than order 1; do not
                               # copy the order-1 value here again
    "nircam_f322w2": 1.05,
    "nircam_f444w": 1.05,
    "nircam_f277w": 1.05,    # same editorial placeholder as the other NIRCam
    "miri_lrs": 1.15,
}

# Width of the extracted line response relative to the Gaussian
# lambda/R_refdata kernel, per mode as (wavelength_um, width) points: the
# single-Gaussian FWHM scale fitted to a narrow line pushed through Pandeia
# (tests/parity/scripts/run_parity.py --impulse, parity_summary.json
# ["lsf_impulse"][mode][line]["width_fit"]). The refdata dispersion R is the
# pixel dispersion; on the slitless modes the PSF along the dispersion axis
# sets the response, which comes out this much broader. Interpolated in
# wavelength, held flat outside the measured range. The NIRSpec modes fit
# 0.94-1.02 and are left at 1. RE-MEASURE ON ANY REFDATA OR PSF CHANGE.
LSF_WIDTH = {
    "niriss_soss": ((1.1, 1.39), (1.5, 1.41), (2.0, 1.48), (2.6, 1.58)),
    "niriss_soss_ord2": ((1.1, 1.33),),
    "nircam_f277w": ((2.6, 1.37),),
    "nircam_f322w2": ((2.6, 1.37), (3.1, 1.43), (3.6, 1.45)),
    "nircam_f444w": ((4.1, 1.46), (4.6, 1.44)),
    "miri_lrs": ((7.5, 1.59), (10.5, 1.49)),
}


def lsf_r(key: str, wl, r_native):
    """Effective resolving power of the extracted response, R_refdata /
    width, on the caller's wavelength grid; ``r_native`` unchanged for a
    mode with no measured width."""
    pts = LSF_WIDTH.get(key)
    r = np.asarray(r_native, float)
    if pts is None:
        return r
    lam, width = np.transpose(pts)
    return r / np.interp(np.asarray(wl, float), lam, width)


# enforce the PandExo group caps at import (loud, no silent out-of-range mode)
for _key, _m in MODES.items():
    _cap = PANDEXO_NGROUP_MAX.get(_m["instrument"])
    if _cap is not None and _m["ngroup_max"] > _cap:
        raise RuntimeError(
            f"mode {_key!r} sets ngroup_max={_m['ngroup_max']}, above the "
            f"PandExo-compatible maximum {_cap} for {_m['instrument']}; the "
            "optimizer would select an unsupported group count on faint "
            "targets.")
    if not (1 <= _m["ngroup_min"] <= _m["ngroup_warn_below"]
            <= _m["ngroup_max"]):
        raise RuntimeError(
            f"mode {_key!r} breaks 1 <= ngroup_min={_m['ngroup_min']} <= "
            f"ngroup_warn_below={_m['ngroup_warn_below']} <= "
            f"ngroup_max={_m['ngroup_max']}; the ramp search and the "
            "below-recommended-ramp warning both assume this ordering.")

MODE_COLOR = {key: _COLORS[i % len(_COLORS)] for i, key in enumerate(MODES)}
MODE_MARKER = {key: _MARKERS[i % len(_MARKERS)] for i, key in enumerate(MODES)}

# GUI default selection (speed-first trio):
# PRISM + G395H cover 1.0-5.30 um including the 4.05 um SO2 band (G395H is
# the default detect-SO2 goal's workhorse), MIRI LRS keeps the mid-IR
# 7-8.5 um SO2 band. All registry modes stay selectable. The ETC computes
# ONLY the selected modes and caches each mode separately, so the default run
# costs three modes and adding a mode later costs exactly that mode.
DEFAULT_MODES = ["nirspec_prism", "nirspec_g395h", "miri_lrs"]

# Per-planet system defaults (star, geometry, T14, UV spectrum) live in planets.py.
