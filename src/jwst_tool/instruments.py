"""JWST time-series instrument-mode registry + paths for the noise backend.

Each ``MODES`` entry carries the Pandeia configuration used by
``pandeia_worker.py`` (running in the selected backend's conda env), display
metadata, and an illustrative systematic noise floor.

Mode tokens are the engine mode names of the supported (2026-era) release
and are submitted verbatim by ``noise.noise_job`` and the parity harness alike.

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

Band edges (``wl_min``/``wl_max``): the mode's usable science bandpass from
a published source, intersected with the forward model's 1-15 um coverage
(short edge = the H2-H2 CIA table). Per instrument:
  * NIRSpec -- the BOTS/S1600A1 table, SUB2048 column: jwst-docs "NIRSpec
    BOTS Wavelength Ranges and Gaps" Table 1, identical to Birkmann et al.
    2022 (A&A 661, A83) Table 2. NOT the nominal disperser table, which is
    wider: the BOTS values already carry the red-end detector cutoffs
    (G235x 3.07/3.12 there against a nominal 3.17).
  * NIRISS -- the Pandeia order-1 range for GR700XD (0.83-2.81), rounded
    inward.
  * NIRCam -- the filter half-power points measured from the shipped
    transmission curves, rounded INWARD to 0.05 um.
  * MIRI -- LRS 5-12 um; the 5.0 short edge is the jwst-docs caution about
    the slitless dispersion turnover below 4.5 um, not a rounding.
Every edge is checked against the Pandeia extracted grid the worker returns,
and pinned to its source by tests/unit/test_instruments_registry.py. Never
narrow an edge without a sourced reason on the entry.

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
        "an editable checkout of jwst-transit-authority.")

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
        "install cannot infer it), or run from an editable checkout of jwst-transit-authority.")
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
# the python is missing. There is exactly ONE backend, "current" = the
# supported STScI release as a MATCHED TRIPLE (pandeia.engine == 2026.7 +
# pandeia_data-2026.7-jwst + pandeia_psfs-2026.7-jwst), enforced by
# `pandeia_worker._check_backend_match` and recorded in "__provenance__" and
# the cache fingerprint. Older releases are not selectable: this release
# neither ships nor validates their matched data triples.
#
# PORTABILITY: refdata/psf default under DATA_DIR. There is deliberately NO
# baked-in interpreter path: the backend env is machine-specific and must come
# from JWST_TOOL_PANDEIA_PYTHON. JWST_TOOL_PANDEIA_{PYTHON,REFDATA,PSF_DIR}
# override any path per-machine.
_SUPPORTED_PANDEIA_RELEASE = "2026.7"
JWST_TOOL_BACKEND = "current"
BACKEND_RELEASE = _SUPPORTED_PANDEIA_RELEASE
BACKEND_STATUS = ("Pandeia 2026.7 / pandeia_data-2026.7-jwst / "
                  "pandeia_psfs-2026.7-jwst (the STScI-supported release, "
                  "enforced as a matched triple)")

# The PANDEIA backend interpreter (machine-specific, no portable default;
# require_pandeia_python() turns a missing setting into one actionable
# message).
PANDEIA_PYTHON = os.environ.get("JWST_TOOL_PANDEIA_PYTHON")


def require_pandeia_python() -> str:
    """Return the backend interpreter path, or raise one actionable error."""
    if PANDEIA_PYTHON:
        return PANDEIA_PYTHON
    raise RuntimeError(
        "No Pandeia backend interpreter configured. The engine runs in its own "
        "environment (heavy dependencies), and its path is machine-specific, so "
        "there is no default to fall back on.\n"
        f"  Set JWST_TOOL_PANDEIA_PYTHON to the python of an environment with "
        f"pandeia.engine=={BACKEND_RELEASE} installed, e.g.\n"
        "    export JWST_TOOL_PANDEIA_PYTHON=/path/to/envs/pandeia/bin/python\n"
        f"  Backend '{JWST_TOOL_BACKEND}' also expects\n"
        f"    refdata: {PANDEIA_REFDATA}\n"
        f"    PSFs:    {PANDEIA_PSF_DIR}\n"
        "  See the README's Install section and `jwst-tool data` for the setup "
        "steps.")


PANDEIA_REFDATA = os.environ.get("JWST_TOOL_PANDEIA_REFDATA",
                                 str(DATA_DIR / "pandeia_data-2026.7-jwst"))
# pandeia_data >= 2026 splits the PSF library out of the refdata tree; the
# engine reads it from $PSF_DIR. Passed to the worker, preflighted, and joins
# the cache key.
PANDEIA_PSF_DIR = os.environ.get("JWST_TOOL_PANDEIA_PSF_DIR",
                                 str(DATA_DIR / "pandeia_psfs-2026.7-jwst"))
# Minimal synphot CDBS assembled for this tool: phoenix grid, 2MASS Ks
# bandpass, CALSPEC Vega. The Vega copy is only an OFFLINE pin for the
# tool-side synphot/stsynphot (stellar.py + the worker); the engine
# normalizes against its OWN refdata Vega (the two agree to 0.08 mmag in
# Ks). `jwst-tool data` reports each piece.
PYSYN_CDBS = str(DATA_DIR / "cdbs")

# Star normalization is band-integrated 2MASS Ks vegamag inside the worker
# (the web-ETC convention) -- never the monochromatic at_lambda shortcut,
# which mis-scales flux by ~1-4% with spectral type and carries that error
# into saturation/ngroup selection.

# Fixed categorical color per mode, never re-assigned when the selection
# changes. Every color holds >= 3:1 contrast on white (WCAG 2.2 non-text) and
# the set passes the dataviz palette validator in wavelength-adjacency order,
# with two residual flags (one chroma, one deutan pair) carried by
# MODE_MARKER, the secondary color-independent encoding.
_COLORS = ["#2a78d6", "#199e70", "#a35a00", "#007a00",
           "#4a3aa7", "#d43f3e", "#a83a9e", "#006c8e",
           "#c2185b", "#8b46c8", "#00929e", "#d81b8c"]

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
# (jwst-docs SOSS template parameters), not a subarray property, and PandExo
# master carries the same 30. NIRCam grism time series is 1-100 in APT,
# matching PandExo.
#
# NIRSpec and MIRI get PANDEXO_UNBOUNDED_NGROUP because SATURATION, not a
# registry cap, picks the ramp there -- never impose one. 65535 is PandExo's
# own value for both, and APT's stated maximum groups/integration for
# NIRSpec. For MIRI it is NOT the real ceiling (APT's 2000 s integration is
# ~12,575 groups on SLITLESSPRISM), but that sits far above anything a
# saturation-limited search reaches, so it is not enforced here.
PANDEXO_UNBOUNDED_NGROUP = 65535

# Extraction strategy + sky background are pinned to PandExo's TSO conventions
# (per-instrument apertures/annuli; background "ecliptic" + background_level
# "medium" -- BOTH keys required together), NOT pandeia's generic
# point-source defaults, which extract a different flux.
# wl_min/wl_max: see the module docstring for the per-instrument source of
# every band edge; each entry below carries its own citation.
# readout_pattern is pinned EXPLICITLY on every mode (NRSRAPID/NISRAPID/RAPID/
# FASTR1, PandExo's TSO choices): engine defaults are non-TSO patterns and
# drift between releases. Never leave readout_pattern implicit on a new mode.
#
# SCOPE (deliberate): each entry is ONE fixed detector configuration
# (subarray + readout pattern), not the whole instrument mode. The tool ranks
# these fixed configurations; it does NOT search alternate subarrays (PRISM
# multistripe, other SOSS substrips) or optimize the readout pattern. Anywhere
# a mode is reported unusable the GUI must NAME that configuration
# (config_label): "NIRISS SOSS: saturated" reads as the instrument being
# incapable of the target when it is one pinned subarray that is.
#
# ngroup_min equals pandeia 2026.7 `mingroups` for each mode's detector
# (pandeia_data-2026.7-jwst/jwst/<instrument>/config.json detector_config:
# nirspec/niriss/nircam 1, miri 2). PandExo reads the same field
# (timing_det_pars['mingroups'], no instrument branching), so both tools
# search the same ramp space. Do not restore floors above pandeia's
# mingroups: that reports bright targets "saturated at the shortest ramp"
# where PandExo passes.

# r_native_med: the mode's typical native resolving power, shown in the GUI
# mode picker. Median of R(lambda) from the mode's 2026.7 refdata dispersion
# file over the registry band, rounded to a readable figure (NIRCam from
# jwst_nircam_disp_*.fits, the LW grism file, which carries no disperser
# token in its name). For the NIRSpec gratings the rounding lands on the
# published nominal figures (jwst-docs R ~ 2,700 high-res, R ~ 1,000
# medium-res, which the measured medians match to ~2%), so G395H reads 2700,
# not the 2800 a blind round would give.
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
        floor_ppm_suggested=20.0, ngroup_min=1,
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
        # BOTS Table 1: 2.87-3.72, 3.82-5.18 (the NRS1/NRS2 gap is measured
        # from the Pandeia grid, never hard-coded here)
        wl_min=2.87, wl_max=5.18,
        r_native_med=2700,
        floor_ppm_suggested=15.0, ngroup_min=1,
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
        # BOTS Table 1: 1.66-2.20, 2.27-3.07. The red edge is the BOTS
        # detector cutoff, BELOW the nominal disperser table's 3.17.
        wl_min=1.66, wl_max=3.07,
        r_native_med=2700,
        floor_ppm_suggested=15.0, ngroup_min=1,
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
        # Pandeia's order-1 CLEAR range is 0.83-2.81 um (jwst-docs quotes the
        # three orders together as 0.6-2.8); 1.0 is the model's short edge
        # (same intersection contract as PRISM), 2.8 is 2.81 rounded inward.
        wl_min=1.0, wl_max=2.8,
        r_native_med=970,
        floor_ppm_suggested=20.0, ngroup_min=1,
        ngroup_max=30,
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
        floor_ppm_suggested=25.0, ngroup_min=1,
        ngroup_max=100,
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
        floor_ppm_suggested=25.0, ngroup_min=1,
        ngroup_max=100,
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
        floor_ppm_suggested=40.0, ngroup_min=2,
        ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
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
        floor_ppm_suggested=15.0, ngroup_min=1,
        ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
    ),
    # Slots 9-11, appended in this order on purpose -- see the palette note
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
        floor_ppm_suggested=15.0, ngroup_min=1,
        ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
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
        floor_ppm_suggested=15.0, ngroup_min=1,
        ngroup_max=PANDEXO_UNBOUNDED_NGROUP,
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
        floor_ppm_suggested=25.0, ngroup_min=1,
        ngroup_max=100,
    ),
}

# Width of the extracted line response relative to the Gaussian
# lambda/R_refdata kernel, per mode as (wavelength_um, width) points: the
# single-Gaussian FWHM scale fitted to a narrow line pushed through Pandeia
# (validation/parity/scripts/run_parity.py --impulse, parity_summary.json
# ["lsf_impulse"][mode][line]["width_fit"]). The refdata dispersion R is the
# pixel dispersion; on the slitless modes the PSF along the dispersion axis
# sets the response, which comes out this much broader. Interpolated in
# wavelength, held flat outside the measured range. The NIRSpec modes fit
# 0.94-1.02 and are left at 1. RE-MEASURE ON ANY REFDATA OR PSF CHANGE.
LSF_WIDTH = {
    "niriss_soss": ((1.1, 1.39), (1.5, 1.41), (2.0, 1.48), (2.6, 1.58)),
    "nircam_f277w": ((2.6, 1.37),),
    "nircam_f322w2": ((2.6, 1.37), (3.1, 1.43), (3.6, 1.45)),
    "nircam_f444w": ((4.1, 1.46), (4.6, 1.44)),
    "miri_lrs": ((7.5, 1.59), (10.5, 1.49)),
}


# Per-mode advised ceiling on the saturation fraction. STScI's SOSS Known
# Issues, from commissioning PID 1541 on HAT-P-14 b: keep counts below
# 56,000 e-, above which the light curves "exhibited degraded precision".
# Pandeia's NIRISS saturation_fullwell is 72,000 e- (jwst/niriss/config.json,
# detector_config default), so the advised fraction is 56000/72000 = 0.778 --
# below the 0.80 default, which allows 57,600 e-. It is a PRECISION ceiling,
# not physical saturation.
#
# REPORTED, NEVER APPLIED (no silent substitution, and applying it would move
# measured worker output without moving a parity gate key). The user lowers
# the slider; the tool says so.
SAT_LIMIT_ADVISED = {
    "niriss_soss": 56000.0 / 72000.0,
}

# Pandeia warning keys worth showing. The rest of its dict is housekeeping the
# tool never consumes -- `bad_waveref` is about 2D diagnostic planes (only
# rpt["1d"] is read), `wavelength_truncated_blue_0` / `scene_range_truncated`
# are a 0.51-vs-0.50 um SED blue-limit mismatch -- and forwarding it verbatim
# put a permanent yellow box of internal tokens on nearly every NIRCam run.
REPORTED_WARNINGS = frozenset({"full_saturated", "partial_saturated"})


def config_label(mode_key: str) -> str:
    """The pinned detector configuration, for any message that reports a mode
    as unusable -- the SCOPE note above is why this is never omitted."""
    det = MODES[mode_key].get("config", {}).get("detector", {})
    return "/".join(str(det[k]).upper() for k in ("subarray", "readout_pattern")
                    if det.get(k))


# MIRI's advice concerns the selected ramp; NIRCam and general TSO advice
# concern the estimated group at saturation. These notes never change the
# worker's ramp search or its minimum group count.
def ngroup_advisory(mode_key: str, ngroup, sat_ngroups=None) -> str:
    """Selected-ramp and early-saturation risks, when supported by the result."""
    instrument = MODES[mode_key]["instrument"]
    notes = []
    if instrument == "miri" and int(ngroup) < 6:
        notes.append(f"{int(ngroup)}-group ramp; STScI describes MIRI ramps with "
                     "2-5 groups as very difficult to calibrate accurately")
    # A missing saturation estimate is not evidence of early saturation.
    if sat_ngroups is not None and np.isfinite(float(sat_ngroups)):
        threshold = 4 if instrument == "nircam" else 3
        if 0 <= float(sat_ngroups) < threshold:
            notes.append(
                f"estimated saturation at {float(sat_ngroups):.2g} groups; "
                + ("STScI advises avoiding NIRCam saturation before 4 groups"
                   if instrument == "nircam" else
                   "STScI identifies saturation before 3 groups as a risk "
                   "requiring instrument-specific guidance"))
    return "; ".join(notes)


def sat_limit_advisory(mode_key: str, sat_limit: float) -> str:
    """Note for a run whose saturation limit exceeds this mode's published
    advice, or "" when it does not. Advice only -- nothing is changed."""
    advised = SAT_LIMIT_ADVISED.get(mode_key)
    if advised is None or float(sat_limit) <= advised:
        return ""
    return (f"saturation limit {float(sat_limit):.2f} is above the "
            f"{advised:.3f} (56,000 e-) STScI advises for SOSS precision")


# NIRCam grism TSO data excess: PandExo's estimate of what APT computes
# (pandexo/engine/jwst.py `estimate_nircam_data_excess`; reproduces the
# parity artifact's 27.82 / 27.54 GB rows to 2e-4 GB). Raw data-generation
# rate of the standard grism template (two SW + one LW detector) on a
# 4-output subgrism subarray, the sustainable downlink rate, and the no-TA
# overhead APT allocates nominal volume over (694 s scheduling + 2100 s
# initial slew; PandExo's extra half frame is < 1e-3 GB and dropped).
# Reported per visit, never ranked on: APT decides.
NIRCAM_RAW_GROUP_RATE_GB_PER_HOUR = 8.60103
NIRCAM_SUSTAINABLE_GB_PER_HOUR = 3.132
NIRCAM_ALLOCATION_OVERHEAD_S = 694.0 + 2100.0
NIRCAM_DATA_EXCESS_GB = (5.0, 15.0)         # PandExo's flag threshold, APT limit
# readout pattern -> (frame cadence per group, frames averaged per group)
NIRCAM_READOUT = {"rapid": (1, 1), "bright1": (2, 1), "bright2": (2, 2),
                  "shallow2": (5, 2), "shallow4": (5, 4), "medium2": (10, 2),
                  "medium8": (10, 8), "mediumdeep2": (15, 2),
                  "mediumdeep8": (15, 8), "deep2": (20, 2), "deep8": (20, 8)}


def nircam_data_excess_gb(mode_key: str, ngroup, exposure_hours: float) -> float:
    """Estimated data excess (GB) of one visit of ``exposure_hours`` on a
    NIRCam grism mode; 0.0 for every other instrument."""
    m = MODES[mode_key]
    if m["instrument"] != "nircam":
        return 0.0
    det = m["config"]["detector"]
    if not str(det["subarray"]).startswith("subgrism"):
        raise ValueError(f"{mode_key}: data-excess rate is known for the "
                         "4-output subgrism subarrays only")
    cadence, nframe = NIRCAM_READOUT[str(det["readout_pattern"]).lower()]
    ngroup = int(ngroup)
    clock_frames = 1 + nframe + (ngroup - 1) * cadence
    saved_groups = ngroup + int(nframe > 1)
    rate = (NIRCAM_RAW_GROUP_RATE_GB_PER_HOUR * saved_groups / clock_frames
            - NIRCAM_SUSTAINABLE_GB_PER_HOUR)
    allocation = (NIRCAM_SUSTAINABLE_GB_PER_HOUR
                  * NIRCAM_ALLOCATION_OVERHEAD_S / 3600.0)
    return max(0.0, rate * float(exposure_hours) - allocation)


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
    if not (1 <= _m["ngroup_min"] <= _m["ngroup_max"]):
        raise RuntimeError(
            f"mode {_key!r} breaks 1 <= ngroup_min={_m['ngroup_min']} <= "
            f"ngroup_max={_m['ngroup_max']}; the ramp search assumes this "
            "ordering.")

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
