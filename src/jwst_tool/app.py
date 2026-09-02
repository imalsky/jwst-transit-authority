"""JWST exoplanet observation planner -- Streamlit GUI.

Launch via the console script ``jwst-tool`` (installed with jwst-transit-authority), or
directly:  streamlit run src/jwst_tool/app.py  (from the repo root).

Pipeline per run: VULCAN-JAX photochemistry -> ExoJAX
transmission/emission spectrum (local subprocess, disk-cached) -> Pandeia ETC
noise per instrument mode (subprocess in its own conda env, disk-cached) ->
science-goal scoring per mode.
Two goal types: DETECT a molecule (conditional template S/N) or CONSTRAIN a
parameter (Fisher forecast from consistency-checked Jacobians, vs a target
uncertainty). Planets beyond WASP-39b come from the registry in planets.py (or
a fully custom system).

Layout: four numbered sidebar steps (Target, Atmosphere, Science goal,
Observation) plus one "More settings" group; the result page leads with a
shortfall warning when the target is missed, then physical structure, the
constraint forecast, and the summary figure. Widget KEYS must never change
(tests and cached session state rely on them); only placement, labels, and
help text may move. A downloaded configuration JSON is a complete, shareable
run setup; "Load a configuration" (share_config.py) restores it into the
widget state before any widget instantiates.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import select
import uuid
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

TOOL_DIR = Path(__file__).resolve().parent   # forward.py subprocess lives here

from jwst_tool import binning, datacheck, detect, \
    fisher as fisher_mod, forward
from jwst_tool import archive
from jwst_tool import noise as noise_mod
from jwst_tool import posteriors
from jwst_tool import proc as proc_mod
from jwst_tool import plotting
from jwst_tool import provenance
from jwst_tool import share_config
from jwst_tool import summary_figure
from jwst_tool import instruments as ins
from jwst_tool import planets
from jwst_tool import runlimit

# House figure style: the vendored science.mplstyle plus summary_figure's
# overrides (serif + STIX math; white faces so a downloaded figure stays white
# on any Streamlit theme). ONE definition of the overrides, applied globally
# here and per-figure there, so an in-app figure and a headless render match.
# Data colors/markers stay the fixed per-mode palette in
# instruments.MODE_COLOR / MODE_MARKER (no series relies on color alone).
plt.style.use(str(TOOL_DIR / "science.mplstyle"))
plt.rcParams.update(summary_figure._STYLE_OVERRIDES)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()


# On-screen width of every rendered figure, in CSS pixels: FIXED, not
# "stretch", so figures stop rescaling as the window changes (see _show_fig).
# Streamlit clamps it to the container width on a narrower screen.
_FIG_DISPLAY_PX = 1100

# Display label for the every-usable-mode combination. A saturated mode is
# unusable data and is excluded from every combination -- that exclusion is
# disclosed per combo, not encoded in this label.
# Display only: never a stored config key.
_ALL_USABLE = "All usable modes"

# How many forecast-posterior panels the summary figure will draw. A LAYOUT
# limit, not a science one: every free parameter's width is reported in the
# Fisher table regardless. The figure solves its height so each panel is square
# at a fixed total width, so each extra panel shrinks all of them -- 3 is where
# a panel stops being readable.
_MAX_POST_PANELS = 3

# Custom-combination palette, deliberately disjoint from instruments.MODE_COLOR
# so a combo line never collides with a member mode's color. Module level
# because _series_color() (results section) reads it from further UP the
# script than the results block that defines the combinations.
_COMBO_COLORS = ("#6a3d9a", "#117733", "#882255", "#88ccee")


def _fig_bytes(fig, fmt: str, tight: bool = True) -> bytes:
    """Export a figure for download: raster PNG at dpi 200 (house convention)
    or vector PDF (proposal-ready; text and lines stay vector).

    ``tight=True`` crops to the figure's OWN ink, so two figures on the same
    canvas come out different sizes when one carries a legend the other lacks;
    the paired T-P / mixing-ratio panels therefore export with ``tight=False``.
    bbox_inches=None would NOT mean "no crop" -- matplotlib reads it as "use
    rcParams['savefig.bbox']", which science.mplstyle sets to `tight` -- so the
    figure's own bbox is the only way to force the full canvas.

    Under plotting.render_lock: savefig measures text through the
    process-global mathtext parser (see plotting.py).
    """
    buf = io.BytesIO()
    with plotting.render_lock:
        fig.savefig(buf, format=fmt, facecolor="white",
                    bbox_inches=("tight" if tight else fig.bbox_inches),
                    **({"dpi": 200} if fmt == "png" else {}))
    return buf.getvalue()


def _show_fig(fig, tight: bool = True) -> None:
    """Render a figure into the page under the render lock, then close it.

    st.pyplot rasterizes, so it enters the same shared mathtext parser as
    layout does -- it belongs inside the lock like every other
    materialization (plotting.py has the full argument). ``tight=False``
    shows the full canvas at a fixed size: st.pyplot accepts no savefig
    keywords any more, so that branch rasterizes through ``_fig_bytes`` (the
    same dpi 200 as st.pyplot's default) and displays the PNG with st.image.

    FIXED DISPLAY WIDTH: st.pyplot's width defaults to "stretch", which
    re-scales the figure on every window resize while its text stays at
    fixed points -- everything wiggles. An explicit pixel width pins it;
    Streamlit still caps the element at the container width on a narrower
    screen.
    """
    if tight:
        with plotting.render_lock:
            st.pyplot(fig, width=_FIG_DISPLAY_PX)
    else:
        st.image(_fig_bytes(fig, "png", tight=False), width=_FIG_DISPLAY_PX)
    plt.close(fig)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV with a one-line identity header (`pd.read_csv(..., comment="#")`)."""
    p = provenance.snapshot()
    repos = " ".join(f"{k}={p['repositories'][k]['commit'][:12]}"
                     for k in ("jwst-transit-authority", "vulcan-forward", "vulcan-jax"))
    ps, cs = p["pandeia_stack"], p["cache_schema"]
    head = (f"# provenance: jwst-transit-authority {p['software']['jwst-transit-authority']} | {repos} | "
            f"pandeia engine {ps['engine']} refdata {ps['refdata']['version']} "
            f"psf {ps['psf']['version']} | cache model v{cs['model']} "
            f"worker v{cs['pandeia_worker']}\n")
    return (head + df.to_csv(index=False)).encode()


def _has_floor(r: dict) -> bool:
    """Does this evaluated mode carry a noise floor anywhere? (floor_spec=None
    gives an all-zero array, and then no N-to-infinity limit exists.)"""
    return bool(np.any(np.asarray(r["floor"]) > 0.0))


# Which statistic a template S/N actually is. They are not interchangeable:
# the projected one additionally profiles the T-P / cloud / lnR0 directions, so
# it answers "after those are fitted away", and a bare number that could be
# either is not a reportable quantity.
_METRIC_LABEL = {True: "T-P + cloud projected", False: "calibration profiled"}


def _detection_metric(r: dict) -> tuple[float, bool]:
    """Collaborator-facing score and whether physical nuisances were used."""
    projected = float(r.get("sigma_detect_proj", float("nan")))
    if np.isfinite(projected):
        return projected, True
    return float(r["sigma_detect"]), False

# initial_sidebar_state: the default ("auto") starts the sidebar COLLAPSED
# on a narrow viewport -- the huggingface.co iframe is one -- hiding every
# input behind a small toggle, which reads as "half the tool is missing".
st.set_page_config(page_title="JWST Transit Authority",
                   layout="wide", initial_sidebar_state="expanded")

# Header: short orientation, no acknowledgment gate
st.title("How to use this tool")
st.subheader("Warning: This tool is in beta mode. That means if you find "
             "a bug, you beta email isaacmalsky@gmail.com")
st.markdown(
    "0. **Configuration**: load a shared configuration file, or start "
    "fresh.\n"
    "1. **Target**: select the system and the observation type.\n"
    "2. **Atmosphere**: set up the atmosphere model. By default the model "
    "spectrum is computed at R = 1000 (correlated-k) and scored on the "
    "analysis bins (default R = 100).\n"
    "3. **Science goal**: detect a molecule, or constrain a parameter.\n"
    "4. **Observation**: select instrument modes and noise assumptions. "
    "The model is smoothed to the instrument resolution where that is "
    "coarser than the model, then binned to the analysis res. For an "
    "observation that uses more than one mode, add the modes as a set in "
    "'Parameter constraint forecast' (in the results), then select that set "
    "in 'Spectrum & forecast series' under the figure.\n\n"
    "The tool computes a forward "
    "spectrum and a Pandeia noise forecast, ranks the selected modes, and "
    "reports how many transits or eclipses reach your target. "
    "Detection values are conditional template "
    "S/N estimates, and parameter constraints are local Fisher "
    "estimates.")

# The Run row renders HERE (above the explainers). Its widgets depend on
# sidebar state that is read further down, so the slot is reserved now and
# filled once those values exist.
_run_slot = st.container()

# Mirrors the README "Validation" section; nothing here runs.
with st.expander("Validation"):
    st.markdown(
        "This tool includes test suites, as well as other validation checks. "
        "The suites run in CI for each repository: "
        "[jax-vulcan](https://github.com/imalsky/jax-vulcan), "
        "[vulcan-forward](https://github.com/imalsky/vulcan-forward), "
        "[jwst-transit-authority](https://github.com/imalsky/jwst-transit-authority), and "
        "[vulcan-retrieval](https://github.com/imalsky/vulcan-retrieval). "
        "For end-to-end tests, see the set of validation figures that I've "
        "created [here](https://github.com/imalsky/jwst-transit-authority/tree/main/"
        "validation/figures). This includes trying to recreate the results "
        "of [Tsai et al. 2023](https://doi.org/10.5281/zenodo.7542781), the "
        "[JWST ERS carbon dioxide paper](https://doi.org/10.5281/zenodo."
        "6959427), and VULCAN 2.0 and petitRADTRANS on identical inputs. "
        "The correlated-k binning validation is shown "
        "[here](https://github.com/imalsky/jwst-transit-authority/blob/main/"
        "validation/figures/ckd_verification_vs_exojax_exok.png). "
        "The power-law cloud deck is checked against petitRADTRANS on "
        "identical inputs, clear and cloudy, "
        "[here](https://github.com/imalsky/jwst-transit-authority/blob/main/"
        "validation/figures/cloud_verification_vs_petitradtrans.png); its "
        "autodiff gradient is checked against the exact analytic derivative "
        "in the test suite.")

# Data availability -- detected live. The GUI reports only what BLOCKS a run;
# the full install inventory is `jwst-tool data`.
def _data_label(it) -> str:
    """Public name without installation details or parenthetical clauses."""
    if it.key.startswith("pkg:"):
        return f"{it.key.removeprefix('pkg:')} Python package"
    labels = {
        "engine:config": "Opacity data folder",
        "cia:H2-H2": "H2-H2 absorption table",
        "cia:H2-He": "H2-He absorption table",
        "ktable:provenance": "ExoMolOP source record",
        "fastchem:binary": "FastChem equilibrium solver",
        "uv:package": "Stellar UV data folder",
        "pandeia:python": "Pandeia engine",
        "pandeia:refdata": "Pandeia JWST reference data",
        "pandeia:psf": "Pandeia PSF data",
        "cdbs:phoenix": "PHOENIX stellar spectra",
    }
    if it.key in labels:
        return labels[it.key]
    if it.key.startswith("ktable:"):
        return f"{it.key.removeprefix('ktable:')} ExoMolOP opacity table"
    return it.label.split(" (")[0]


@st.cache_data(ttl=3600, show_spinner="Checking installed data ...")
def _cached_full_report(backend: str):
    # Disk-persisted (the Space entrypoint warms it at boot) on top of the
    # in-process st.cache. The manifest check is sampled, not exhaustive
    # (full pass: `jwst-tool data --deep`). Invalidation: the two 1-hour TTLs
    # (this one and datacheck.REPORT_CACHE_MAX_AGE_S) plus `backend` in the
    # cache key. NO leading underscore on the param: Streamlit skips
    # underscore-prefixed args when hashing, which silently made this a
    # single shared entry across backends.
    cached = datacheck.load_cached_report()
    if cached is not None:
        return cached
    return datacheck.warm_report_cache(base_mols=forward.MOLECULES,
                                       extra_mols=forward.EXTRA_MOLECULES)


_data_report = _cached_full_report(ins.JWST_TOOL_BACKEND)
_missing_req = datacheck.missing_required(_data_report)
if _missing_req:
    st.error(
        f"**Required data missing ({len(_missing_req)} item"
        f"{'s' if len(_missing_req) > 1 else ''}).** A run stops if it needs "
        "one of these files:\n\n"
        + "\n".join(f"- **{_data_label(it)}**  \n  Location: {it.detail}  \n"
                    f"  Install: {it.remedy}" for it in _missing_req)
        # on the hosted Space the visitor cannot install anything; the CLI
        # remedies are for local installs (SPACE_ID is set by Hugging Face)
        + ("\n\nThis hosted deployment is missing data it should ship "
           "with; please report it to isaacmalsky@gmail.com."
           if os.environ.get("SPACE_ID") else
           "\n\nUse `jwst-tool data` to check files and `jwst-tool fetch` "
           "to download supported data."))
# Opacity sources: the fetcher's provenance record + each k-table's header
@st.cache_data(show_spinner=False)
def _ktable_sources(stamp: int) -> dict:
    # stamp = provenance.json mtime_ns, so a re-fetch invalidates the cache;
    # exceptions are not cached, so an absent file re-raises on every rerun.
    # NO leading underscore: Streamlit skips underscore-prefixed args when
    # hashing, which made this cache permanent for the process lifetime.
    return provenance.ktable_sources()


_prov_path = datacheck.exomolop_provenance_path()
try:
    _ksrc = _ktable_sources(_prov_path.stat().st_mtime_ns
                            if _prov_path and _prov_path.is_file() else 0)
    _ksrc_err = None
except (ImportError, RuntimeError, FileNotFoundError) as exc:
    _ksrc, _ksrc_err = {}, str(exc)   # the engine's text carries the remedy
# fixed prefix: the smoke test recognises the warning by it, whatever the
# engine's wording (no data root, no provenance.json, no table)
_KSRC_WARN = "Opacity sources unavailable: "


# Every non-k-table source the tool reads, with the paper that defines it.
# DOI + page are both REQUIRED here: a row with either one blank is a citation
# the user cannot follow, and test_app_smoke pins that none is. Every DOI was
# resolved against doi.org. The k-table rows get theirs from the engine.
_CIA_SRC = ("HITRAN CIA 2011", "10.1016/j.jqsrt.2011.11.004",
            "https://hitran.org/cia/")
_EXOJAX_SRC = ("ExoJAX", "10.3847/1538-4365/ac3b4d",
               "https://github.com/HajimeKawahara/exojax")

def _source_rows(base_set, extra_set, extra_on, rayleigh_on, cloud_on,
                 photo_on, sflux_sel):
    """One row per data source the tool reads: the offered k-tables (base
    first), the continuum terms, and the non-opacity data files. Every row
    carries a resolvable DOI and page; `used in this setup` marks the ones
    this configuration actually reads. The k-table band grid is one line in
    the Opacity panel rather than a column repeating 25 identical values."""
    rows = []
    for m in list(base_set) + list(extra_set):
        r = _ksrc.get(m, {})
        iso = ("natural abundance" if r.get("natural_abundance")
               else r.get("iso", ""))
        rows.append({
            "component": m,
            "used in this setup": m in base_set or m in extra_on,
            "data set": f"ExoMol {r['dataset']} ({iso})" if r else "",
            "source DOI": r.get("doi") or "",
            "source page": r.get("url", "")})
    rows += [
        {"component": "H2-H2 CIA", "used in this setup": True,
         "data set": _CIA_SRC[0],
         "source DOI": _CIA_SRC[1], "source page": _CIA_SRC[2]},
        {"component": "H2-He CIA", "used in this setup": True,
         "data set": _CIA_SRC[0],
         "source DOI": _CIA_SRC[1], "source page": _CIA_SRC[2]},
        {"component": "Rayleigh scattering (H2, He)",
         "used in this setup": bool(rayleigh_on),
         "data set": "ExoJAX H2/He polarizabilities",
         "source DOI": _EXOJAX_SRC[1], "source page": _EXOJAX_SRC[2]},
        # Cites the code that runs: exojax.atm.simple_clouds.powerlaw_clouds.
        # Its alpha is -gamma of the petitRADTRANS definition
        # (Molliere et al. 2019, 10.1051/0004-6361/201935470).
        {"component": "Power-law cloud/haze",
         "used in this setup": bool(cloud_on),
         "data set": "ExoJAX powerlaw_clouds (kappa_cloud, alpha)",
         "source DOI": _EXOJAX_SRC[1], "source page": _EXOJAX_SRC[2]},
        {"component": "Chemical equilibrium initializer",
         "used in this setup": True,
         "data set": "FastChem 2.0 (VULCAN build)",
         "source DOI": "10.1093/mnras/sty1531",
         "source page": "https://github.com/exoclime/FastChem"},
    ]
    for fname, label in planets.SFLUX_CHOICES.items():
        ds, doi, url = planets.SFLUX_SOURCES[fname]
        rows.append({
            "component": f"Stellar UV — {label}",
            "used in this setup": bool(photo_on) and fname == sflux_sel,
            "data set": ds, "source DOI": doi, "source page": url})
    rows += [
        {"component": "Stellar spectrum grid",
         "used in this setup": True,
         "data set": "PHOENIX (Allard et al., synphot CDBS)",
         "source DOI": "10.1098/rsta.2011.0269",
         "source page": "https://www.stsci.edu/hst/instrumentation/"
                        "reference-data-for-calibration-and-tools/"
                        "astronomical-catalogs/"
                        "phoenix-models-available-in-synphot"},
        {"component": "Ks bandpass (flux normalization)",
         "used in this setup": True, "data set": "2MASS",
         "source DOI": "10.1086/498708",
         "source page": "https://irsa.ipac.caltech.edu/Missions/2mass.html"},
        {"component": "Flux standard (Vega)",
         "used in this setup": True,
         "data set": "CALSPEC alpha_lyr_stis_011",
         "source DOI": "10.1086/677655",
         "source page": "https://www.stsci.edu/hst/instrumentation/"
                        "reference-data-for-calibration-and-tools/"
                        "astronomical-catalogs/calspec"},
        {"component": "JWST exposure & noise model",
         "used in this setup": True,
         "data set": f"Pandeia {ins.BACKEND_RELEASE}",
         "source DOI": "10.1117/12.2231768",
         "source page": "https://jwst-docs.stsci.edu/"
                        "jwst-exposure-time-calculator-overview/"
                        "jwst-etc-pandeia-engine-tutorial"},
    ]
    return rows


with st.expander("Data" +
                 (f" ({len(_missing_req)} required item(s) missing)"
                  if _missing_req else "")):
    _opacity_slot = st.container()

# Measured AD Fisher-row wall time (WASP-39b defaults), threaded through the
# GUI mention below so a re-measurement updates one place. (FD costs come
# from the run-time estimator's own model -- "Jacobian-row cost model".)
_AD_ROW_MIN, _AD_ROW_MAX = 1, 2        # minutes per AD row

# All wall-time constants below were measured on the maintainer's laptop; the
# HF Space (SPACE_ID set there and only there) is slower, so every DISPLAYED
# estimate and progress-bar prior is scaled by this factor. The forward.py
# PROG weights are relative fractions and need no scaling.
_RUNTIME_SCALE = 2.5 if os.environ.get("SPACE_ID") else 1.0

_PROG_RE = re.compile(r"\[fwd\] PROG ([0-9.]+) (.*)")


def _fmt_clock(s: float) -> str:
    """Fixed-width duration, always 9 chars: ' HH:MM:SS'.

    The ONLY duration formatter here: a variable-width format changes string
    length at unit boundaries, which makes the progress row -- and every
    widget below it -- jitter on each tick.
    """
    s = max(0, int(round(s)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{min(h, 99):3d}:{m:02d}:{sec:02d}"


# Fixed pixel height for the streaming solver/ETC log boxes: the content
# grows line by line, and an auto-height box moves everything under it.
_LOG_BOX_PX = 220


class _TimedBar:
    """st.progress wrapper that appends a live elapsed / time-remaining
    readout to every label.

    The remaining estimate blends the pre-run prior (when given) with the
    measured pace, weighted by the completed fraction; with no prior it is
    purely measured. Blend the two REMAINING-time estimates, never the two
    totals -- blending totals cancels the measured pace and freezes the
    countdown for the whole stage.

    LAYOUT STABILITY: the label is built at a CONSTANT width so
    the bar and everything under it stop shifting on every tick. Two rules
    make that hold: the clock is fixed-width ``HH:MM:SS`` (``_fmt_clock``, so
    59s -> 1m 00s never changes the string length), and the stage label is
    padded/truncated to ``_STAGE_W`` characters. The text is also wrapped in a
    monospace span, because character-count padding alone does not give a
    constant PIXEL width in a proportional UI font -- and because a label that
    reflows to a second line moves every widget below it."""

    _STAGE_W = 46          # stage-label field; long labels are truncated

    def __init__(self, prior_total_s: float | None = None,
                 text: str = "starting ..."):
        self._t0 = time.monotonic()
        self._prior = prior_total_s
        self._frac = 0.0
        self._label = text
        self._bar = st.progress(0.0, text=self._compose())

    def _compose(self) -> str:
        e = time.monotonic() - self._t0
        remaining = None
        if self._frac > 0.0:
            # measured pace: grows with e when a stage overruns
            measured_left = e * (1.0 - self._frac) / self._frac
            if self._prior:
                prior_left = max(self._prior * (1.0 - self._frac), 0.0)
                remaining = (self._frac * measured_left
                             + (1.0 - self._frac) * prior_left)
            else:
                remaining = measured_left
        elif self._prior:
            remaining = max(self._prior - e, 0.0)
        stage = str(self._label)
        if len(stage) > self._STAGE_W:
            stage = stage[:self._STAGE_W - 1] + "…"
        left = (_fmt_clock(remaining) if remaining is not None
                else "  --:--:--")
        # monospace + non-breaking spaces: constant pixel width, no reflow
        body = (f"{stage:<{self._STAGE_W}}  elapsed {_fmt_clock(e)}"
                f"  left {left}")
        return "`" + body.replace(" ", "\u00a0") + "`"

    def _render(self) -> None:
        self._bar.progress(min(1.0, self._frac), text=self._compose())

    def update(self, frac: float, label: str) -> None:
        self._frac, self._label = float(frac), label
        self._render()

    def tick(self) -> None:
        """Refresh the clock without new progress information."""
        self._render()

    def done(self, label: str = "done") -> None:
        # same fixed-width shape as the live ticks, so the final render does
        # not resize the row one last time
        self._frac, self._label = 1.0, label
        elapsed = time.monotonic() - self._t0
        stage = label[:self._STAGE_W]
        body = (f"{stage:<{self._STAGE_W}}  elapsed {_fmt_clock(elapsed)}"
                f"  left {_fmt_clock(0.0)}")
        self._bar.progress(1.0, text="`" + body.replace(" ", "\u00a0") + "`")


def _watch_proc(proc, on_line, on_tick, tick_s: float = 1.0) -> None:
    """Dispatch each stdout line of ``proc`` to ``on_line``, calling
    ``on_tick`` at least every ``tick_s`` seconds of silence. Selects on the
    raw pipe fd (os.read chunking) so the clock ticks through silent solver
    stages; never revert to blocking readline loops (they freeze the
    readout). On Windows (no select on pipes) it degrades to blocking
    reads."""
    fd = proc.stdout.fileno()
    tail = b""
    can_select = sys.platform != "win32"
    while True:
        if can_select:
            ready, _, _ = select.select([fd], [], [], tick_s)
            if not ready:
                on_tick()
                continue
        chunk = os.read(fd, 65536)
        if not chunk:
            if tail:
                on_line(tail.decode(errors="replace").rstrip())
            return
        tail += chunk
        *full, tail = tail.split(b"\n")
        for raw in full:
            on_line(raw.decode(errors="replace").rstrip())


def _managed_proc(cmd):
    """``proc.terminating`` over a stdout-piped child: the worker must not
    outlive the script run that started it (see jwst_tool.proc)."""
    return proc_mod.terminating(subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))


# EVERY download button is rendered with on_click=ignore. Without it a click
# queues a rerun, and a queued rerun CANCELS the script run in flight -- on the
# Run pass that is the forward model itself.

# default target precision per parameter (DISPLAY units: dex / K / absolute C/O)
_TARGET_DEFAULT = {"lnZ": 0.10, "dlnCO": 0.10, "lnKzz": 0.30,
                   "Tirr": 50.0, "Tint": 50.0,
                   "Tint_cl": 50.0,
                   "log_kappa": 0.30, "log_gamma": 0.30,
                   "log_kappa_cloud": 0.30, "alpha_cloud": 0.50}
# Every freeable Fisher parameter can be the constraint goal and looks up
# _TARGET_DEFAULT[goal_param]; guard for missing entries at import, not at
# click time.
_FREEABLE = (set(forward.CHEM_PARAM_NAMES) | set(forward.CLOUD_FISHER_PARAMS)
             | {p for ns in forward.TP_PARAM_NAMES.values() for p in ns})
_missing_target = _FREEABLE - set(_TARGET_DEFAULT)
if _missing_target:
    raise RuntimeError(f"_TARGET_DEFAULT is missing {sorted(_missing_target)}: "
                       "every freeable Fisher parameter needs a target default.")


# Sidebar controls
# Reset = bump a nonce that namespaces EVERY widget key: session_state.clear()
# alone does not reset keyless widgets.
_NONCE = st.session_state.setdefault("reset_nonce", 0)


def _reset_all():
    n = st.session_state.get("reset_nonce", 0) + 1
    st.session_state.clear()
    st.session_state["reset_nonce"] = n


def _arm_reset():
    st.session_state["confirm_reset"] = True


def _disarm_reset():
    st.session_state["confirm_reset"] = False


def K(name: str) -> str:
    return f"n{_NONCE}_{name}"


def _axis_range(container, label: str, key: str, warn, *, unit: str = "",
                positive: bool = False,
                positive_reason: str = "it is drawn on a log axis",
                fmt: str = "%.6g",
                step: float | None = None, help: str | None = None):
    """A plain min/max pair of number boxes for one plot axis.

    Every axis control in this app is this widget:
    two typed numbers, no slider. Both boxes start BLANK, meaning "fit the
    data", rather than prefilled with the current automatic values -- the
    automatic fit tracks the run and the mode selection, so a prefilled number
    would silently pin a stale window the moment either changed.

    Returns ``(lo, hi)`` or None. Both boxes or neither: a one-sided window has
    no second edge to fall back on, and the automatic fit lives inside the
    figure builders, not here, so a half-specified range is refused VISIBLY
    through ``warn`` rather than half-applied.

    ``positive``: a bound at or below zero is refused here, with
    ``positive_reason`` naming WHY for this axis (log-scale axes like
    wavelength; positive-definite linear axes like C/O). The figure builders
    raise on it -- correctly, they are the API backstop -- but that exception
    reaches Streamlit uncaught and kills the whole results page. A typed
    number is a user choice, not a defect: warn and fall back to the
    automatic fit, the same way a one-sided window does.
    """
    _u = f" ({unit})" if unit else ""
    c_lo, c_hi = container.columns(2)
    lo = c_lo.number_input(f"{label} min{_u}", value=None, step=step,
                           format=fmt, key=f"{key}_min", help=help)
    hi = c_hi.number_input(f"{label} max{_u}", value=None, step=step,
                           format=fmt, key=f"{key}_max")
    if (lo is None) != (hi is None):
        warn(f"{label} range needs both boxes filled. Fitting to the data.")
        return None
    if lo is None:
        return None
    if float(lo) >= float(hi):
        warn(f"{label} range needs min below max. Fitting to the data.")
        return None
    if positive and float(lo) <= 0.0:
        warn(f"{label} min must be above 0 ({positive_reason}). "
             "Fitting to the data.")
        return None
    return (float(lo), float(hi))


def _apply_pending_config() -> None:
    """Apply a loaded configuration file to the widget state.

    Must run BEFORE any widget instantiates (Streamlit forbids writing a
    widget's key afterwards). The content hash makes it apply once, not on
    every rerun; either the whole mapping applies or nothing does."""
    up = st.session_state.get(K("cfg_upload"))
    if up is None:
        return
    raw = up.getvalue()
    sha = hashlib.sha1(raw).hexdigest()
    if st.session_state.get("_cfg_applied_sha") == sha:
        return
    st.session_state["_cfg_applied_sha"] = sha
    st.session_state.pop("_cfg_load_error", None)
    st.session_state.pop("_cfg_load_notes", None)
    try:
        cfg = json.loads(raw.decode())
        state, notes = share_config.widget_state(cfg, K)
    except (ValueError, RuntimeError, TypeError, UnicodeDecodeError) as e:
        st.session_state["_cfg_load_error"] = str(e)
        return
    st.session_state.pop("restored_tp_path", None)
    st.session_state.pop("restored_floor_table", None)
    # A loaded configuration supersedes any queued archive fill AND its
    # banner: the fill notes describe values the restore may have just
    # replaced, so leaving them would misattribute the form to the archive.
    st.session_state.pop("_archive_fill_pending", None)
    st.session_state.pop("_archive_fill_notes", None)
    for k, v in state.items():
        st.session_state[k] = v
    st.session_state["_cfg_load_notes"] = notes


_apply_pending_config()


def _apply_pending_archive_fill() -> None:
    """Fill the custom planet's widgets from a queued archive lookup.

    Same ordering contract as _apply_pending_config: widget keys can only be
    written BEFORE the widgets instantiate, so the Fill button's callback
    just stashes the planet name (un-namespaced on purpose -- a reset's
    session_state.clear() must kill a queued fill) and this applies it at
    the top of the next run. Out-of-range/missing fields are never written
    (Streamlit silently discards an out-of-range value, substituting the
    default); archive.custom_fill reports them by name instead."""
    name = st.session_state.pop("_archive_fill_pending", None)
    if not name:
        return
    try:
        values, notes = archive.custom_fill(archive.lookup(name))
    except (KeyError, archive.SnapshotError) as e:
        st.session_state["_archive_fill_notes"] = [("error", str(e))]
        return
    # The archive duration is the PRIMARY-TRANSIT duration; in emission mode
    # the event is the secondary eclipse, whose duration can differ (eccentric
    # systems), so the field is left for the user to set.
    if (st.session_state.get(K("scimode")) == "emission"
            and "t14" in values):
        values.pop("t14")
        notes = list(notes) + [
            "the archive transit duration was not applied: the observation "
            "type is emission, and a secondary-eclipse duration can differ "
            "from the transit duration; set the event duration yourself."]
    for suffix, v in values.items():
        st.session_state[K(f"custom_{suffix}")] = v
    st.session_state["_archive_fill_notes"] = (
        [("success", f"Initial values filled from the archive snapshot row "
                     f"for {name}. Later edits and loaded configurations "
                     "are not from the archive; review every field.")]
        + [("warning", n) for n in notes])


_apply_pending_archive_fill()


def _queue_archive_fill() -> None:
    st.session_state["_archive_fill_pending"] = \
        st.session_state.get(K("custom_arch_name"))


def _combo_add() -> None:
    """Add a named mode combination (results-area builder). Callbacks run
    before widgets instantiate, so writing the name widget's key back to ""
    here is allowed. The note key is un-namespaced on purpose: a reset's
    session_state.clear() must kill it (same pattern as the archive fill)."""
    name = str(st.session_state.get(K("cb_name")) or "").strip()
    modes = list(st.session_state.get(K("cb_modes")) or [])
    if not name:
        st.session_state["_combo_note"] = (
            "error", "The combination needs a name.")
        return
    combos = st.session_state.setdefault(K("combos"), [])
    if any(c["name"] == name for c in combos):
        st.session_state["_combo_note"] = (
            "error", f"A combination named {name!r} already exists; choose "
                     "another name.")
        return
    if not modes:
        st.session_state["_combo_note"] = (
            "error", "Select at least one instrument mode for the "
                     "combination.")
        return
    combos.append(dict(name=name, modes=modes))
    st.session_state[K("cb_name")] = ""
    st.session_state["_combo_note"] = (
        "success", f"Added the combination {name!r}.")


def _combo_remove(i: int) -> None:
    combos = st.session_state.get(K("combos")) or []
    if 0 <= i < len(combos):
        removed = combos.pop(i)
        st.session_state["_combo_note"] = (
            "info", f"Removed the combination {removed['name']!r}.")


with st.sidebar:
    # Step 0: Load a configuration. The file was already APPLIED by
    # _apply_pending_config(); this is the widget plus the outcome messages.
    st.markdown("### 0 · Configuration")
    # Every step's controls sit behind an expander,
    # so the sidebar reads as a short list of titled sections. The
    # load-outcome messages stay OUTSIDE the expander: a failed restore must
    # be loud without opening anything.
    with st.expander("Load a configuration (JSON)"):
        _cfg_up = st.file_uploader(
            "Configuration file", type=["json"], key=K("cfg_upload"))
    if st.session_state.get("_cfg_load_error"):
        st.error("The configuration file could not be applied: "
                 + st.session_state["_cfg_load_error"])
    elif _cfg_up is not None:
        st.success("Configuration loaded. Review the steps below and "
                   "press Run.")
        for _n in st.session_state.get("_cfg_load_notes") or []:
            st.warning(f"Not restored: {_n}")
    st.divider()

    # Step 1: Target
    st.markdown("### 1 · Target")
    with st.expander("Planet & observation type", expanded=True):
        planet_key = st.selectbox(
            "Planet", list(planets.PLANETS) + ["custom"], key=K("planet"),
            format_func=lambda k: planets.PLANETS[k]["label"]
            if k in planets.PLANETS else "Custom planet …")
        pdef = planets.PLANETS.get(planet_key, planets.CUSTOM_DEFAULTS)
        science_mode = st.radio(
            "Observation type", ["transmission", "emission"], horizontal=True,
            key=K("scimode"),
            format_func={"transmission": "Transmission",
                         "emission": "Emission"}.get)
    # Event word for observation-facing labels: only the vocabulary changes
    # with the observation type, not the calculation.
    _evw = "eclipse" if science_mode == "emission" else "transit"

    def _k(name: str) -> str:            # per-planet widget state
        return K(f"{planet_key}_{name}")

    with st.expander("System parameters", expanded=(planet_key == "custom")):
        if planet_key == "custom":
            # Optional archive fill: values come from the SHIPPED PSCompPars
            # snapshot (date disclosed), never a live query. The button's
            # callback only queues the name; _apply_pending_archive_fill
            # wrote the widget keys at the top of this run.
            try:
                _snap = archive.load_snapshot()
            except archive.SnapshotError as e:
                _snap = None
                st.error(f"Archive lookup unavailable: {e}")
            if _snap is not None:
                _arch_sel = st.selectbox(
                    "Fill from the NASA Exoplanet Archive (optional)",
                    _snap.names, index=None,
                    placeholder="Search by planet name ...",
                    key=K("custom_arch_name"))
                st.button("Fill the system parameters",
                          disabled=(_arch_sel is None),
                          on_click=_queue_archive_fill,
                          key=K("custom_arch_fill"))
            for _kind, _msg in st.session_state.get(
                    "_archive_fill_notes") or []:
                getattr(st, _kind)(_msg)
        _R = planets.CUSTOM_FIELD_RANGES   # single source with archive fill
        teff = st.number_input(
            "Stellar effective temperature, Teff (K)", *_R["teff"],
            pdef["star"]["teff"], 50.0, key=_k("teff"))
        logg = st.number_input(
            "Stellar surface gravity, log10(g) (g in cm s^-2)", *_R["logg"],
            pdef["star"]["log_g"], 0.1, key=_k("logg"))
        feh = st.number_input("Stellar metallicity, [Fe/H] (dex)", *_R["feh"],
                              pdef["star"]["metallicity"], 0.1, key=_k("feh"))
        ks_mag = st.number_input(
            "2MASS Ks magnitude", *_R["ks"],
            pdef["star"]["ks_mag"], 0.1, key=_k("ks"))
        rstar = st.number_input(
            "Stellar radius (solar radii)", *_R["rstar"], pdef["rstar_rsun"],
            0.01, key=_k("rstar"), format="%.3f")
        rp = st.number_input("Planet radius (Jupiter radii)", *_R["rp"],
                             pdef["rp_rjup"], 0.01,
                             key=_k("rp"), format="%.3f")
        g_ms2 = st.number_input(
            "Planet surface gravity (m s^-2)", *_R["g"],
            pdef["gs_cgs"] / 100.0, 0.5, key=_k("g"))
        orbit_au = st.number_input(
            "Semi-major axis (AU)", *_R["a"],
            pdef["orbit_au"], 0.001, key=_k("a"), format="%.4f")
        t14 = st.number_input(
            ("Transit duration, T14 (hours)" if _evw == "transit"
             else "Eclipse duration (hours)"), *_R["t14"],
            pdef["t14_hr"], 0.1, key=_k("t14"))
        _uv_ok = datacheck.uv_spectra_status()
        sflux = st.selectbox(
            "Stellar UV spectrum (photochemistry)",
            list(planets.SFLUX_CHOICES),
            index=list(planets.SFLUX_CHOICES).index(pdef["sflux"]),
            format_func=lambda f: (
                planets.SFLUX_CHOICES[f]
                + ("" if _uv_ok.get(f) else "  [FILE MISSING]")),
            key=_k("sflux"))

    # T_eq is derived from the star and orbit for every planet, registry or
    # custom: a stored literature value is a second source of truth that goes
    # stale when the star parameters are refreshed.
    teq = planets.system_teq(teff, rstar, orbit_au)

    # Step 2: Atmosphere
    st.divider()
    st.markdown("### 2 · Atmosphere")

    with st.expander("Temperature-pressure profile"):
        _tp_opts = ["guillot", "file"]
        # Mirror canonical_params' default so GUI and API defaults agree --
        # INCLUDING the science mode (emission defaults to Guillot on every
        # planet; the bundled tables are terminator profiles). On a science-
        # mode switch, a widget still holding the other mode's default moves
        # to this mode's default; an explicit user choice survives.
        _tp_default = forward._default_tp_mode(
            {"planet": planet_key, "science_mode": science_mode})
        _tp_other = forward._default_tp_mode(
            {"planet": planet_key,
             "science_mode": ("transmission" if science_mode == "emission"
                              else "emission")})
        if st.session_state.get(K("tp_scimode_seen")) != science_mode:
            st.session_state[K("tp_scimode_seen")] = science_mode
            if st.session_state.get(_k("tp")) == _tp_other != _tp_default:
                st.session_state[_k("tp")] = _tp_default
        if st.session_state.get(_k("tp")) not in _tp_opts:
            st.session_state[_k("tp")] = _tp_default
        # the key is always set above, so no index= default: passing one as
        # well makes Streamlit log a "created with a default value but also
        # set via the Session State API" stack trace on every rerun
        tp_mode = st.selectbox(
            "Temperature-pressure profile", _tp_opts,
            key=_k("tp"),
            format_func={"guillot": "Guillot (2010)",
                         "file": "Tabulated table (T-P, optional Kzz)"}.get)
        tp_kwargs = {}
        tp_file, tp_file_path, tp_file_ok = "", None, True
        if tp_mode == "guillot":
            # one definition shared with canonical_params, so the widget and
            # API defaults cannot drift apart
            tirr0 = planets.default_tirr(
                pdef, system=(dict(star_teff=teff, rstar_rsun=rstar,
                                   orbit_au=orbit_au)
                              if planet_key == "custom" else None),
                science_mode=science_mode)
            if planet_key == "custom":
                # follow-until-overridden: while T_irr still equals the last
                # derived default, keep it tracking the star/orbit fields;
                # the first manual edit breaks the link (otherwise T_irr
                # freezes at the first-render seed).
                _prev_auto = st.session_state.get(_k("tirr_auto"))
                if (_prev_auto is not None
                        and st.session_state.get(_k("tirr")) == _prev_auto):
                    st.session_state[_k("tirr")] = tirr0
                st.session_state[_k("tirr_auto")] = tirr0
            # key always seeded, so no value= default
            if _k("tirr") not in st.session_state:
                st.session_state[_k("tirr")] = tirr0
            tp_kwargs["Tirr"] = st.number_input(
                "T_irr (K)", 800.0, 2500.0, step=20.0, key=_k("tirr"))
            tp_kwargs["Tint"] = st.number_input(
                "T_int (K)", 50.0, 500.0, 100.0, 25.0, key=_k("tint"))
            # default 0.01 cm^2/g, Guillot (2010)'s canonical thermal opacity
            tp_kwargs["log_kappa"] = st.number_input(
                "log10 kappa_IR (cm^2/g)", -4.0, 0.0, -2.0, 0.1, key=_k("lk"))
            tp_kwargs["log_gamma"] = st.number_input(
                "log10 gamma (kappa_vis/kappa_IR)", -2.0, 0.3, -1.0, 0.05,
                key=_k("lg"))
        elif tp_mode == "file":
            # the shipped table is PER-PLANET; a planet without one may only
            # upload
            _ship_name = forward.shipped_tp_table_name(planet_key)
            _src_opts = ([forward.TP_FILE_SHIPPED, forward.TP_FILE_UPLOAD]
                         if _ship_name else [forward.TP_FILE_UPLOAD])
            if not _ship_name and st.session_state.get(
                    _k("tpsrc")) == forward.TP_FILE_SHIPPED:
                st.session_state[_k("tpsrc")] = forward.TP_FILE_UPLOAD
            tp_file = st.radio(
                "Profile source", _src_opts,
                horizontal=True, key=_k("tpsrc"),
                format_func={forward.TP_FILE_SHIPPED:
                             f"Shipped measured profile ({_ship_name})",
                             forward.TP_FILE_UPLOAD: "Upload an array"}.get)
            if not _ship_name:
                st.warning(
                    "No shipped profile for "
                    f"{pdef['label'] if planet_key in planets.PLANETS else 'this planet'}: "
                    + (pdef.get("tp_table_note") or "none is bundled.")
                    + " Upload a table, or switch to the Guillot profile.")
            elif (tp_file == forward.TP_FILE_SHIPPED
                  and pdef.get("tp_table_note")):
                # the note carries the reason in both directions (planets.py):
                # show it when the shipped table is CHOSEN, not only missing
                st.warning(pdef["tp_table_note"])
            if tp_file == forward.TP_FILE_UPLOAD:
                _tp_example = (
                    "#(dyne/cm2) (K) (cm2/s)\n"
                    "Pressure   Temp   Kzz\n"
                    "1.000e+09  2255.  1.0e+07\n"
                    "1.000e+08  2100.  1.0e+07\n"
                    "1.000e+07  1800.  3.0e+07\n"
                    "1.000e+06  1400.  1.0e+08\n"
                    "1.000e+05  1150.  3.0e+08\n"
                    "1.000e+04   980.  1.0e+09\n"
                    "1.000e+03   920.  3.0e+09\n"
                    "1.000e+02   890.  1.0e+10\n"
                    "1.000e+01   875.  3.0e+10\n"
                    "1.000e+00   870.  1.0e+11\n")
                with st.expander("Example array (what the file must look like)"):
                    st.code(_tp_example)
                    st.download_button(
                        "Download this example (edit and re-upload)",
                        _tp_example, "example_atm.txt",
                        key=_k("tpex"), on_click="ignore")
                up_tp = st.file_uploader(
                    "Upload an array: T-P (+ optional Kzz) as text",
                    type=["txt", "dat"],
                    key=_k("tpup"))
                # a loaded configuration can carry the table; the uploader
                # (whose state cannot be set) wins when a new file is picked
                if up_tp is None and st.session_state.get("restored_tp_path"):
                    try:
                        _rp_tp = Path(st.session_state["restored_tp_path"])
                        _tab_tp = forward._read_tp_table(_rp_tp)
                        tp_file_path = str(_rp_tp)
                    except (OSError, ValueError) as e:
                        st.error("The restored T-P table is not usable: "
                                 f"{e} Upload the table again.")
                        tp_file_ok = False
                elif up_tp is not None:
                    _raw_tp = up_tp.getvalue()
                    _sha_tp = hashlib.sha1(_raw_tp).hexdigest()[:16]
                    _dst_tp = forward._uploads_dir() / f"{_sha_tp}.txt"
                    _dst_tp.parent.mkdir(parents=True, exist_ok=True)
                    if not _dst_tp.exists():
                        # atomic: the exists() guard makes a torn copy
                        # permanent (the sha1 check would refuse it forever)
                        ins.atomic_write(_dst_tp,
                                         lambda fh: fh.write(_raw_tp))
                    try:                       # loud validation, immediate
                        _tab_tp = forward._read_tp_table(_dst_tp)
                        tp_file_path = str(_dst_tp)
                    except ValueError as e:
                        st.error(
                            "The temperature-pressure table is not valid: "
                            f"{e} Edit the file and upload it again.")
                        tp_file_ok = False
                else:
                    st.warning("Upload a table to run in file mode.")
                    tp_file_ok = False

    # The column's pressure boundaries and the radius anchor: physics-defining
    # inputs, so they live here in step 2. Widget keys stay as they are
    # (shipped key contract).
    with st.expander("Pressure limits & reference radius"):
        rt_ptop_bar = st.number_input(
            "Model top pressure (bar)",
            1.0e-9, 1.0e-6, 1.0e-9, 1.0e-9,
            format="%.1e", key=K("rtptop"))
        p_ref_bar = st.number_input(
            "Reference pressure for the planet radius (bar)",
            1.0e-6, 7.0, 1.0e-3, format="%.1e", key=K("pref"),
            help="The pressure level the planet radius Rp refers to; it "
                 "sets the absolute depth level of the transmission "
                 "spectrum.")
        # keyed PER GEOMETRY (shipped key contract). The DEFAULT follows the
        # structure mode -- a measured T-P table caps the honest column at its
        # own bottom (7.6 bar for the shipped tables), parametric profiles get
        # a round 10 bar -- and tracks that mode until the user edits the box
        # (same seed pattern as the Kzz mode below; an untouched default must
        # never survive a structure switch and refuse the run).
        _pbtm_key = K(f"pbtm_{science_mode}")
        _pbtm_default = forward.default_p_btm_bar(dict(tp_mode=tp_mode))
        _pbtm_now = st.session_state.get(_pbtm_key)
        if (_pbtm_now is None
                or (_pbtm_now in (forward.P_BTM_FILE_BAR,
                                  forward.P_BTM_PARAMETRIC_BAR)
                    and _pbtm_now != _pbtm_default)):
            st.session_state[_pbtm_key] = _pbtm_default
        # key always seeded above, so no value= default
        p_btm_bar = st.number_input(
            "Column bottom pressure (bar)",
            *forward.P_BTM_RANGE,
            step=1.0, format="%.4g", key=_pbtm_key)

    with st.expander("Composition"):
        # Composition is STRUCTURAL, one path for every value:
        # metallicity scales O/C/N/S together, C/O sets C_H = co * O_H,
        # FastChem re-initializes at exactly that composition. No
        # perturbative knob; C-rich (> 1) is the same code path; an
        # uncertified corner errors loudly (longdy gate).
        met = st.number_input(
            "Metallicity (× solar)", 0.1, 100.0, 10.0, 0.5,
            format="%.2f", key=K("met"))
        co_ratio = st.number_input(
            "C/O (carbon/oxygen number ratio)",
            0.10, 2.00, float(forward.CO_DEFAULT), 0.05,
            format="%.3f", key=K("co"))

    # Kzz and photochemistry render BEFORE the science-goal step, so the AD
    # photo-lock reads the EFFECTIVE differentiation method from session
    # state: the widget value counts only when a Jacobian is actually
    # requested, matching canonical_params' normalization to "fd" otherwise.
    _goal_ss = st.session_state.get(K("goal"), "detect")
    _dofish_ss = bool(st.session_state.get(K("dofish"), False))
    # fallback "ad" mirrors the method widget's default (index=1), so
    # the photo-lock is right on the FIRST constrain render, before the
    # selectbox has seeded session state.
    # (No condensation term: the GUI offers no condensation widget and
    # share_config REFUSES a condensing config rather than restoring one.)
    _jac_hint = (st.session_state.get(K("jacm"), "ad")
                 if (_goal_ss == "constrain" or _dofish_ss)
                 else "fd")

    with st.expander("Vertical mixing (Kzz)"):
        _kzz_opts = ["const", "Pfunc", "JM16"]
        # tabulated Kzz needs the tabulated T-P table (its Kzz column)
        _kzz_file_ok = tp_mode == "file"
        if _kzz_file_ok:
            _kzz_opts.append("file")
            # Same rule as canonical_params: a table that carries Kzz
            # supplies the mixing profile, so "file" is the default.
            if _k("kzzmode") not in st.session_state:
                st.session_state[_k("kzzmode")] = "file"
        elif st.session_state.get(_k("kzzmode")) in ("file", None):
            st.session_state[_k("kzzmode")] = "const"
        # key always seeded above, so no index= default
        kzz_mode = st.selectbox(
            "Vertical-mixing profile, Kzz", _kzz_opts,
            key=_k("kzzmode"),
            format_func={"const": "Constant",
                         "Pfunc": "Power law in P (Pfunc)",
                         "JM16": "Moses-type P^-0.5 (JM16)",
                         "file": "Tabulated (Kzz column of the T-P table)"}.get)
        kzz_const = kzz_kmax = kzz_plev = kzz_kdeep = 0.0
        kzz_x = 1.0
        if kzz_mode == "const":
            log_kzz = st.number_input(
                "log10 Kzz (cm^2 s^-1)", 6.0, 12.0, 9.0, 0.25,
                key=_k("kzz"))
            kzz_const = 10.0 ** log_kzz
        elif kzz_mode == "Pfunc":
            kzz_kmax = 10.0 ** st.number_input(
                "log10 deep Kzz (cm^2 s^-1)", 4.0, 11.0, 5.0, 0.25,
                key=_k("kzkmax"))
            kzz_plev = 10.0 ** st.number_input(
                "log10 transition level (bar)", -5.0, 2.0, -1.0, 0.25,
                key=_k("kzplev"))
        elif kzz_mode == "JM16":
            kzz_kdeep = 10.0 ** st.number_input(
                "log10 deep-floor Kzz (cm^2 s^-1)", 4.0, 11.0, 5.0, 0.25,
                key=_k("kzkdeep"))
        if kzz_mode != "const":
            kzz_x = 10.0 ** st.number_input(
                "Kzz profile multiplier, log10(f)", -1.0, 1.0, 0.0, 0.05,
                key=_k("kzzx"))

    with st.expander("Photochemistry & transport"):
        network = st.selectbox(
            "Chemical network", list(forward.NETWORKS),
            key=K("network"),
            format_func={"sncho": "S-N-C-H-O (full, default)",
                         "ncho": "N-C-H-O (no sulfur, faster)"}.get)
        if K("photo") not in st.session_state or _jac_hint == "ad":
            st.session_state[K("photo")] = True   # AD needs photolysis ON
        # key always seeded above, so no value= default
        use_photo = st.checkbox(
            "Photochemistry (UV photolysis)", key=K("photo"),
            disabled=(_jac_hint == "ad"))
        # default 83 deg = upstream VULCAN's dayside-average zenith angle
        sl_angle_deg = st.number_input(
            "Photolysis zenith angle (degrees)", 0.0, 89.0, 83.0, 1.0,
            key=K("sza"), disabled=not use_photo)
        f_diurnal = st.number_input(
            "Diurnal photolysis factor", 0.1, 1.0, 1.0, 0.05,
            key=K("fdiur"), disabled=not use_photo)
        use_moldiff = st.checkbox(
            "Molecular diffusion", value=True, key=K("moldiff"))
        use_vm_mol = st.checkbox(
            "Upwind molecular-diffusion advection (vm_mol)", value=False,
            key=K("vmmol"), disabled=not use_moldiff)

    # Opacity settings live in the Atmosphere step so extra_mols is a
    # live variable by step 3; all extras default ON.
    with st.expander("Opacity (ExoJAX)"):
        _base_set, _extra_set = forward.MOLECULES, forward.EXTRA_MOLECULES
        # The sulfur-free network removes the S species from both sets; its
        # widgets get their own keys (same pattern as the provider switch) so
        # a selection from one network never strands in the other's options.
        _net_sfx = "" if network == "sncho" else f"_{network}"
        if network == "ncho":
            _base_set = [m for m in _base_set
                         if m not in forward._S_MOLECULES]
            _extra_set = [m for m in _extra_set
                          if m not in forward._S_MOLECULES]
        st.caption("Correlated-k on ExoMolOP k-tables (R = 1000, "
                   "16 g-points, 1-15 µm), scored on the analysis bins "
                   "(default R = 100). Sources per species are in the Data "
                   "panel.")
        st.caption(
            f"The base set **{' · '.join(_base_set)}** is always on.")
        # Species with no published k-table are not in EXTRA_MOLECULES;
        # this filter only guards a future widening of that list.
        _extra_set = [m for m in _extra_set
                      if m not in forward._NO_EXOMOLOP_TABLE]
        _unattributed = [m for m in list(_base_set) + list(_extra_set)
                         if m not in _ksrc]
        if _ksrc_err or _unattributed:
            st.warning(_KSRC_WARN + (_ksrc_err or (
                "no provenance record for " + ", ".join(_unattributed)
                + "; run python -m vulcan_forward.fetch_exomolop "
                "--molecules " + ",".join(_unattributed))))
        # Preselect the MEASURED-relevant subset, not everything with a
        # table: most species contribute under 1 ppm, and each default-on
        # molecule pays a leave-one-out spectrum. The ppm measurements are
        # in forward.py (EXTRA_MOLECULES_DEFAULT).
        extra_mols = st.multiselect(
            "Extra opacity molecules", list(_extra_set),
            default=[m for m in _extra_set
                     if m in forward.EXTRA_MOLECULES_DEFAULT],
            key=K(f"xmols_vulcan{_net_sfx}"))

    with st.expander("Clouds & scattering (ExoJAX)"):
        if K("rayl") not in st.session_state:
            st.session_state[K("rayl")] = True
        if science_mode == "emission":
            # canonical_params forces Rayleigh OFF in emission (no scattering
            # channel); show the forced state, not a checked-but-ignored box
            st.session_state[K("rayl")] = False
        # key always seeded above, so no value= default
        use_rayleigh = st.checkbox(
            "H2/He Rayleigh scattering", key=K("rayl"),
            disabled=(science_mode == "emission"))
        cloud_on = st.checkbox(
            "Power-law cloud/haze opacity", value=False, key=K("cloud"))
        if cloud_on:
            log_kappa_cloud = st.number_input(
                "log10 kappa_cloud (cm^2/g at 3.5 µm)", -4.0, 2.0, -1.0, 0.1,
                key=K("ck"))
            alpha_cloud = st.number_input(
                "Cloud spectral slope alpha (kappa ∝ nu^alpha)",
                0.0, 4.0, 0.0, 0.25, key=K("ca"))
        else:
            log_kappa_cloud, alpha_cloud = -1.0, 0.0

    # Step 3: Science goal (only the controls the selected goal needs)
    st.divider()
    st.markdown("### 3 · Science goal")
    avail_free = list(forward.CHEM_PARAM_NAMES) + forward.TP_PARAM_NAMES[tp_mode]
    if cloud_on:
        avail_free = avail_free + list(forward.CLOUD_FISHER_PARAMS)
    mol_options = forward.active_molecules(
        {"network": network, "extra_mols": extra_mols})

    goal_param, target_prec, marginalize = None, None, True
    do_fisher = False
    with st.expander("Goal & target"):
        goal = st.radio(
            "Goal", ["detect", "constrain"], horizontal=True, key=K("goal"),
            format_func={"detect": "Detect a molecule",
                         "constrain": "Constrain a parameter"}.get)
        if goal == "detect":
            # SO2 is the flagship W39b science under VULCAN; the equilibrium
            # provider has no SO2, so its default detection target is H2O
            _mol_default = "SO2" if "SO2" in mol_options else "H2O"
            target_mol = st.selectbox(
                "Molecule to detect", mol_options,
                index=mol_options.index(_mol_default),
                help="The comparison spectrum zeroes this species in the "
                     "OPACITY only. The chemistry is not re-solved, so the "
                     "T-P profile, mean molecular weight, gravity, continuum "
                     "and every other species are unchanged -- it is not a "
                     "model of an atmosphere that lacks the species.",
                key=K(f"mol_vulcan{_net_sfx}_"
                      + "_".join(sorted(extra_mols))))
            target_sig = st.number_input(
                "Target significance (σ)", 1.0, 10.0, 3.0, 0.5, key=K("tsig"))
            do_fisher = st.checkbox(
                "Also calculate parameter constraints", value=True,
                key=K("dofish"))
        else:
            target_mol = None
            goal_param = st.selectbox(
                "Parameter to constrain", avail_free,
                key=K(f"gp_vulcan_{tp_mode}_{int(cloud_on)}"),
                format_func=lambda n: forward.PARAM_LABELS[n])
            marginalize = st.checkbox(
                "Marginalize over the other parameters", value=True,
                key=K("marg"))
            if not marginalize:
                st.warning(
                    "Marginalization is off: every other parameter is held "
                    "fixed, so the bound is a best-case sensitivity.")
            unit = forward.PARAM_UNITS[goal_param]
            # label uses the unit when there is one (dex / K), else the bare
            # symbol -- C/O is a dimensionless number ratio
            _tgt_lbl = (f"Target uncertainty (±{unit})" if unit else
                        f"Target uncertainty "
                        f"(±{forward.PARAM_SYMBOLS[goal_param]})")
            if unit == "K":
                target_prec = st.number_input(_tgt_lbl, 5.0, 500.0,
                                              _TARGET_DEFAULT[goal_param], 5.0,
                                              key=K(f"tgt_{goal_param}"))
            else:
                target_prec = st.number_input(_tgt_lbl, 0.01, 3.0,
                                              _TARGET_DEFAULT[goal_param], 0.01,
                                              key=K(f"tgt_{goal_param}"))
            target_sig = st.number_input(
                "Report Fisher half-width at Nσ", 1.0, 10.0, 3.0, 0.5,
                key=K("tsig"))

    # Free-parameter settings render only when the run computes derivatives.
    fisher_params: list = []
    jac_method = "fd"
    if goal == "constrain" or do_fisher:
        with st.expander("Free parameters", expanded=(goal == "constrain")):
            if goal == "constrain" and marginalize:
                # Defaults FILTERED by the live menu, key carries the
                # provider: Streamlit hard-raises on a default outside the
                # options.
                fisher_extra = st.multiselect(
                    "Jointly free parameters", avail_free,
                    # lnKzz is not in the defaults (speed); still
                    # selectable, and dropping it tightens the remaining
                    # sigmas toward the conditional bound.
                    default=[p for p in ("lnZ", "dlnCO")
                             if p in avail_free],
                    key=K(f"fx_vulcan_{tp_mode}_{int(cloud_on)}"),
                    format_func=lambda n: forward.PARAM_LABELS[n])
                fisher_params = sorted(set(fisher_extra) | {goal_param})
            elif goal == "constrain":
                fisher_params = [goal_param]
            else:
                fisher_params = st.multiselect(
                    "Free parameters", avail_free,
                    key=K(f"fp_vulcan_{tp_mode}_{int(cloud_on)}"),
                    default=[p for p in ("lnZ", "dlnCO")
                             if p in avail_free],
                    format_func=lambda n: forward.PARAM_LABELS[n])
            jac_method = st.selectbox(
                "Differentiation method", ["fd", "ad"], index=1,
                key=K("jacm"),
                format_func={"fd": "Finite differences",
                             "ad": "Automatic differentiation "
                                   "(forward-mode, default)"}.get)
            # Loud slow-path flag: FD re-solves the chemistry per row, so
            # point the user at AD before a multi-hour run.
            if fisher_params and jac_method == "fd":
                _n_comp = sum(p in ("lnZ", "dlnCO") for p in fisher_params)
                _n_theta = sum(p in ("lnKzz", "Tirr", "Tint", "log_kappa",
                                     "log_gamma") for p in fisher_params)
                _est_min = int((_n_comp * 7 + _n_theta * 4) * _RUNTIME_SCALE)
                if _est_min >= 20:
                    st.warning(
                        f"Finite differences with {len(fisher_params)} free "
                        f"parameters is slow: roughly {_est_min}-"
                        f"{int(_est_min * 1.4)} min. Switch the "
                        "differentiation method to AD "
                        f"(~{int(_AD_ROW_MIN * _RUNTIME_SCALE)}-"
                        f"{int(_AD_ROW_MAX * _RUNTIME_SCALE)} min per row, "
                        "photochemistry locked on) or free fewer "
                        "parameters.")

    # Step 4: Observation
    st.divider()
    st.markdown("### 4 · Observation")
    # Display-only band strings: the modelled band (registry envelope
    # intersected with the forward-model span), with the NRS detector gap
    # shown for the H gratings. Gap edges measured from this tool's own
    # Pandeia 2026.7 wavelength grids (largest grid step) and equal to
    # Birkmann et al. 2022 Table 2 / jwst-docs BOTS Table 1; scoring always
    # uses the worker's actual pixels, so the gap never enters the math.
    # The OUTER endpoints must equal the registry's wl_min/wl_max
    # (test_app_smoke pins that): this string restates a band the registry
    # already owns.
    _MODE_BAND_DISPLAY = {
        "nirspec_g140h": "1.00-1.31 + 1.35-1.83 µm",
        "nirspec_g235h": "1.66-2.20 + 2.27-3.07 µm",
        "nirspec_g395h": "2.87-3.72 + 3.82-5.18 µm",
    }
    with st.expander(f"Instrument modes & {_evw}s", expanded=True):
        mode_keys = st.multiselect(
            "Instrument modes",
            options=list(ins.MODES),
            default=ins.DEFAULT_MODES, key=K("modes"),
            help="One fixed detector configuration per mode; noise is "
                 "computed once per star, so adding "
                 "modes later is instant. Ranges are the modelled bands, not "
                 "the full instrument coverage; R is the median native "
                 "resolving power from the Pandeia reference data.",
            format_func=lambda k: (
                f"{ins.MODES[k]['label']}  "
                "(" + _MODE_BAND_DISPLAY.get(
                    k, f"{ins.MODES[k]['wl_min']:g}-"
                       f"{ins.MODES[k]['wl_max']:g} µm")
                + f", R ~ {ins.MODES[k]['r_native_med']})"))
        # "per mode" is load-bearing: every selected mode is evaluated at this
        # count, and a combined forecast sums their information at the same
        # count, so K modes cost K x this many events -- except SOSS orders 1
        # and 2, which come from one readout (their mode rows say so).
        n_transits = st.number_input(f"Number of {_evw}s per mode", 1, 10, 1, 1,
                                     key=K("ntr"))

    # Mock-observation layer: generated AFTER the forward model and the
    # noise model (posteriors.mock_realization), never inside them. The
    # draw IS fitted -- mock_recovery overlays the parameters recovered
    # from it -- so this is not a cosmetic layer. What it may never touch:
    # the FORECAST (detection/Fisher scores, caches, result CSVs), which
    # stays realization-independent by construction.
    with st.expander("Mock noise draw & noise multiplier"):
        show_noise = st.checkbox(
            "Mock noise draw", value=True, key=K("shownoise"),
            help="One seeded Gaussian draw per point at its own error bar, "
                 "fitted in the forecast panels; the forecast numbers do "
                 "not depend on the draw.")
        seed = st.number_input(
            "Seed", 0, 9999, 0, key=K("seed"), disabled=not show_noise)
        # ONE knob for "more jitter". It scales the
        # NOISE MODEL, not the draw: posteriors.mock_realization deliberately
        # refuses a draw-only scale factor, because a 2x draw beside 1x error
        # bars is not a realization of the plotted model, and the S/N and
        # Fisher numbers would still assume the unscaled sigma. Scaling the
        # noise model instead moves the error bars, the scores, the forecast
        # widths and the draw together, so the figure stays internally
        # consistent and the value is recorded with the run. Composes with
        # (multiplies) the per-mode multipliers in the Noise model expander,
        # which stay for mode-specific tuning.
        noise_scale = st.number_input(
            "Global noise multiplier", 0.5, 3.0, 1.0, 0.05,
            key=K("noisescale"),
            help="Scales every mode's noise (1.0 = the Pandeia prediction): "
                 "error bars, S/N, forecast widths and the mock draw "
                 "together. Composes with the per-mode multipliers.")

    with st.expander("Timing, saturation & binning (Pandeia)"):
        # follow-until-overridden (same pattern as _k("tirr")): while the
        # baseline still equals the T14 it last tracked, it keeps following the
        # event duration (PandExo convention baseline = T14); the first manual
        # edit breaks the link. Without this it froze at the first-render seed
        # and silently kept a former planet's duration.
        _tb_auto = float(t14)
        _tb_prev = st.session_state.get(_k("tbase_auto"))
        if (_tb_prev is not None
                and st.session_state.get(_k("tbase")) == _tb_prev):
            st.session_state[_k("tbase")] = _tb_auto
        st.session_state[_k("tbase_auto")] = _tb_auto
        # key always seeded above, so no value= default
        if _k("tbase") not in st.session_state:
            st.session_state[_k("tbase")] = _tb_auto
        t_base = st.number_input(
            f"Out-of-{_evw} baseline (hours)", 0.5, 10.0,
            step=0.1, key=_k("tbase"),
            help="Out-of-event time anchoring the stellar flux; the PandExo "
                 "convention is baseline = T14.")
        sat_limit = st.number_input(
            "Saturation limit (fraction of Pandeia's saturation level)",
            0.5, 0.95, 0.80, 0.05, key=K("sat"),
            help="The longest ramp kept is the one whose brightest pixel "
                 "stays under this fraction of Pandeia's saturation level "
                 "for the mode. That level is not the physical full well and "
                 "is not the same fraction of it on every instrument: "
                 "Pandeia already holds NIRCam time-series modes at 70% of "
                 "the well, so 0.80 there is about 56% of it, against 0.80 "
                 "of NIRSpec's adopted value. 0.80 is PandExo's default and "
                 "is what the parity comparison uses. STScI's time-series "
                 "guidance is different: sample up the ramp to about half "
                 "the saturation level, which is 0.50 here.")
        r_bin = st.number_input(
            "Analysis resolving power, R", 25, 500, 100, 25, key=K("rbin"))

    with st.expander("Noise model (Pandeia)"):
        st.markdown("**Minimum noise floor** (PandExo convention)")
        # DEFAULTS TO CONSTANT. A default is acceptable in ONE direction
        # only: preselecting "No floor" would claim undemonstrated precision
        # on the user's behalf, while a constant floor at the suggested
        # values claims LESS precision -- the conservative side. The floor
        # is recorded with the run and shown per mode below.
        floor_mode = st.radio(
            "Floor type", ["constant", "none", "file"], horizontal=True,
            index=0, key=K("floormode"),
            format_func={"constant": "Constant (ppm)", "none": "No floor",
                         "file": "Wavelength table"}.get)
        floor_table = None
        floors = {k: None for k in mode_keys}
        if floor_mode == "constant":
            floors = {k: st.number_input(
                f"{ins.MODES[k]['label']} minimum floor (ppm)", 0.0, 200.0,
                ins.MODES[k]["floor_ppm_suggested"], 5.0, key=K(f"floor_{k}"))
                for k in mode_keys}
        elif floor_mode == "file":
            up = st.file_uploader(
                "Two columns: wavelength (µm), floor (ppm)",
                type=["txt", "csv", "dat"], key=K("floorfile"),
                help="Whitespace- or comma-separated; interpolated to the "
                     "final bins with constant edge extension; applied to "
                     "every selected mode.")
            if up is not None:
                try:
                    raw = up.getvalue().decode()
                    delim = "," if "," in raw.splitlines()[0] else None
                    floor_table = np.loadtxt(raw.splitlines(), delimiter=delim,
                                             ndmin=2)
                    noise_mod.resolve_floor(np.array([1.0]),
                                            floor_table)  # validate loudly now
                except Exception as e:
                    st.error(
                        "The floor table is not valid: "
                        f"{e} Edit the file and upload it again.")
                    floor_table = None
            elif st.session_state.get("restored_floor_table") is not None:
                # carried by a loaded configuration; the uploader wins when
                # the user picks a new file
                try:
                    floor_table = np.asarray(
                        st.session_state["restored_floor_table"], float)
                    noise_mod.resolve_floor(np.array([1.0]), floor_table)
                except Exception as e:
                    st.error("The restored floor table is not usable: "
                             f"{e} Upload the table again.")
                    floor_table = None
            if floor_table is None:
                st.warning("No valid floor table is loaded. Upload one, or "
                           "select 'No floor'. The run is blocked until "
                           "the choice is explicit.")
        if floor_mode == "file" and floor_table is not None:
            floors = {k: floor_table for k in mode_keys}
        # An unmade floor choice blocks the run: both candidate defaults bias
        # the headline number, so it must not be implicit.
        floor_choice_made = (floor_mode == "none"
                             or floor_mode == "constant"
                             or (floor_mode == "file" and floor_table is not None))

        # per-mode multipliers; the global "Noise multiplier" above scales all
        # of them (composed below, so neither control is a hidden override)
        _infl_mode = {k: st.number_input(
            f"{ins.MODES[k]['label']} random-noise multiplier", 1.0, 3.0,
            1.0, 0.05, key=K(f"infl_{k}"))
            for k in mode_keys}
        infl = {k: float(noise_scale) * float(_infl_mode[k])
                for k in mode_keys}

    # More settings: solver grid and advanced RT, behind one entry point.
    # No help tooltips in here: the labels stand on their own and the
    # reference material lives in README.md.
    st.divider()
    st.markdown("### More settings")

    with st.expander("Solver & vertical grid"):
        # One control sets both layer COUNTS; the two pressure grids stay
        # distinct (interp_map regrids chemistry onto the RT grid).
        nz = st.number_input(
            "Layers per grid (chemistry and RT)", *forward.NZ_RANGE,
            forward.NZ_DEFAULT, 10, key=K("nz"))
        yconv_cri = st.number_input(
            "Solver convergence tolerance", 1.0e-4, 1.0e-2,
            forward.YCONV_DEFAULT, 1.0e-4,
            format="%.1e", key=K("yconv"))

    # Condensation is API-only: the GUI pins it off, and share_config REFUSES
    # a loaded configuration that enables it rather than silently computing a
    # different atmosphere than the file describes. Settling, escape and the
    # boundary-condition fluxes were REMOVED -- forward.canonical_params
    # refuses any enabling value -- but their keys stay in the canonical
    # payload, pinned off, so no cached spectrum's key changes.
    use_condense = use_settling = False
    diff_esc, top_flux, bot_flux = [], [], []

    with st.expander("Advanced radiative transfer (ExoJAX)"):
        # the pressure boundaries and the radius anchor live in step 2
        # ("Pressure limits & reference radius"); only the chord
        # integration choice stays advanced
        if (K("rtint") not in st.session_state
                or science_mode == "emission"):
            # canonical_params pins simpson in emission (no transit chord);
            # show the pinned state, not a silently ignored choice
            st.session_state[K("rtint")] = "simpson"
        # key always seeded above, so no index= default
        rt_integration = st.selectbox(
            "Transit chord integration", ["simpson", "trapezoid"],
            key=K("rtint"), disabled=(science_mode == "emission"),
            format_func={"simpson": "Simpson (ExoJAX default)",
                         "trapezoid": "Trapezoid"}.get)

    # Reset sits behind a confirmation step: one click must not clear a long
    # configuration and the current results.
    st.divider()
    if st.session_state.get("confirm_reset"):
        st.warning("Reset all settings to their defaults and clear the "
                   "current results?")
        _rc1, _rc2 = st.columns(2)
        _rc1.button("Confirm reset", on_click=_reset_all, type="primary")
        _rc2.button("Keep settings", on_click=_disarm_reset)
    else:
        st.button("Reset all settings", on_click=_arm_reset)

params = dict(planet=planet_key, science_mode=science_mode,
              network=network,
              star_teff=teff, star_logg=logg, star_feh=feh,
              nz=nz, yconv_cri=yconv_cri,
              rp_rjup=rp, gs_cgs=g_ms2 * 100.0, rstar_rsun=rstar,
              orbit_au=orbit_au, sflux=sflux,
              met_x_solar=met, co_ratio=float(co_ratio),
              kzz_mode=kzz_mode, kzz_x=kzz_x, kzz_const=kzz_const,
              kzz_kmax=kzz_kmax, kzz_plev=kzz_plev, kzz_kdeep=kzz_kdeep,
              tp_mode=tp_mode, tp_file=tp_file, tp_file_path=tp_file_path,
              fisher_params=fisher_params,
              jac_method=jac_method,
              use_photo=use_photo, sl_angle_deg=sl_angle_deg,
              f_diurnal=f_diurnal, use_moldiff=use_moldiff,
              use_vm_mol=use_vm_mol and use_moldiff,
              use_condense=use_condense,
              use_settling=use_settling, diff_esc=diff_esc,
              top_flux=top_flux, bot_flux=bot_flux,
              use_rayleigh=use_rayleigh,
              rt_ptop_bar=float(rt_ptop_bar), rt_integration=rt_integration,
              p_ref_bar=float(p_ref_bar),
              p_btm_bar=float(p_btm_bar),
              cloud_on=cloud_on,
              log_kappa_cloud=log_kappa_cloud, alpha_cloud=alpha_cloud,
              # Detect computes the removed-molecule spectrum for the TARGET
              # only; the score reads exactly one row. wo_mols is cache-keyed,
              # so switching the detection target is a real re-run; API callers
              # can pass wo_mols=None for the full block. Constrain reads none
              # of them and skips the block entirely.
              wo_mols=([target_mol] if goal == "detect" else []),
              extra_mols=extra_mols, **tp_kwargs)
star = dict(teff=teff, log_g=logg, metallicity=feh, ks_mag=ks_mag)
planet_label = (planets.PLANETS[planet_key]["label"]
                if planet_key in planets.PLANETS else "custom planet")

_canon = None
try:
    _canon = forward.canonical_params(params)
    # load, not exists(): a file that fails to load is not a cache hit
    cached = forward.load_result(params) is not None
    params_error = None
except (ValueError, RuntimeError) as e:  # stale widget combo mid-rerun, or a
    cached, params_error = False, str(e)  # missing/invalid T-P table
if tp_mode == "file" and not tp_file_ok and params_error is None:
    params_error = "file-mode T-P selected but no valid table is loaded"

with _opacity_slot:
    st.markdown("#### Data sources")
    if _ksrc_err:
        st.warning(_KSRC_WARN + _ksrc_err)
    else:
        st.dataframe(
            _source_rows(_base_set, _extra_set, extra_mols, use_rayleigh
                         and science_mode == "transmission", cloud_on,
                         use_photo, sflux),
            column_config={
                "used in this setup": st.column_config.CheckboxColumn(
                    "used in this setup"),
                # No display_text: it is a regex over the URL, and these
                # pages span ExoMol, HITRAN, STScI and others.
                "source page": st.column_config.LinkColumn("source page")},
            width="stretch", hide_index=True)
        # A shared range is valid only when the full numerical grid signatures
        # agree. Never turn incompatible table domains into a plausible union.
        if _ksrc:
            _signatures = {r["grid_sha256"] for r in _ksrc.values()}
            if len(_signatures) != 1:
                st.error(
                    "Installed k-tables use incompatible numerical grids. "
                    "They cannot be mixed; re-fetch all species from one release.")
            else:
                _one = next(iter(_ksrc.values()))
                _tr, _pr, _wr = (_one[k] for k in
                                 ("t_range_K", "p_range_bar", "wl_range_um"))
                st.caption(
                    f"Common k-table grid: T {_tr[0]:g}-{_tr[1]:g} K, "
                    f"P {_pr[0]:g}-{_pr[1]:g} bar, "
                    f"λ {_wr[0]:.2g}-{_wr[1]:.3g} µm.")

# rough runtime hint keyed off the resolution settings
base_min = 0.1 + 0.010 * nz
if yconv_cri <= 1.5e-3:              # strict convergence costs extra iterations
    base_min += 0.5
base_min += 0.25 * len(extra_mols)   # k-table load + one more overlap fold
# cool columns (<~900 K) converge much more slowly
t_char = {"guillot": tp_kwargs.get("Tirr", 1560.0) / np.sqrt(2.0),
          "file": float(teq)}.get(tp_mode, 1100.0)
if t_char < 900.0:
    base_min += 2.5

# Jacobian-row cost model: fd = 4 solves per row; cloud rows are RT-only
# (~seconds); ad = ONE shared warm primal plus a batched tangent per row
_solve_min = max(1.0, base_min * 0.5)
_rt_only = set(forward.CLOUD_FISHER_PARAMS)
n_cloud_rows = sum(1 for n in fisher_params if n in _rt_only)
_solve_rows = [n for n in fisher_params if n not in _rt_only]
if jac_method == "ad":
    fd_min = (((0.9 + 0.35 * len(_solve_rows)) * _solve_min
               if _solve_rows else 0.0) + 0.2 * n_cloud_rows)
else:
    n_fd_comp = sum(1 for n in _solve_rows if n in forward.FD_COMP_PARAMS)
    n_fd_theta = len(_solve_rows) - n_fd_comp
    fd_min = (n_fd_comp * 4 * (_solve_min + 0.8)
              + n_fd_theta * 4 * _solve_min + 0.2 * n_cloud_rows)

# The removed-molecule spectrum (TARGET ONLY on detect; constrain computes
# none). This term sits OUTSIDE base_min on purpose: a Jacobian row computes
# ONE spectrum, not the wo chain, so _solve_min must not inherit it.
_n_mols_est = len(_base_set) + len(extra_mols)
_wo_min = ((0.05 * _n_mols_est * (nz / forward.NZ_DEFAULT))
           if goal == "detect" else 0.0)

# The resolving power the run uses: the k-tables' own band grid, R=1000.
grid_lbl = f"{nz}-layer, correlated-k R=1000"
# One scaled total feeds the estimate string AND the progress-bar prior,
# so the number the user reads is the number the bar starts from.
_est_total_min = (base_min + _wo_min + fd_min) * _RUNTIME_SCALE
est = "instant (cached)" if cached else (
    f"~{_est_total_min:.0f} min ({grid_lbl} run"
    + (f" + {len(fisher_params)} Jacobian rows" if fisher_params else "")
    + ")")

# --- Run row: validation messages, run button, review summary --------------
with _run_slot:
    if params_error:
        st.error(f"Cannot run with the current settings: {params_error}")
    _still_needs = []
    if not mode_keys:
        _still_needs.append("an instrument mode (step 4 · Observation)")
    if not floor_choice_made:
        _still_needs.append("a minimum noise floor type "
                            "(step 4 · Observation, Noise model)")
    if _still_needs:
        st.error("This run still needs: " + "; ".join(_still_needs) + ".")
    # The config download sits BESIDE Run, in a column filled further down
    # (it needs _canon and the goal/observation blocks, which are built
    # below). Streamlit columns accept deferred writes, so the slot is
    # reserved here and the button lands in the right place on the page.
    col_btn, _cfg_col, _ = st.columns([1, 1.4, 1.6])
    run_clicked = col_btn.button("Run", type="primary", width="stretch",
                                 disabled=(bool(params_error) or not mode_keys
                                           or not floor_choice_made))
    st.caption(f"Estimated runtime: {est}")

# ONE description of everything a run consumes OUTSIDE the canonical model
# parameters: the science goal and the observation setup. Built once and used
# three times -- the shareable config, the stored run meta, and the staleness
# guard -- so a new setting joins all three at once. The guard compares the
# whole block minus _DISPLAY_ONLY below, never a hand-picked subset: a
# computed field it misses reaches the page silently as stale results.
_goal_meta = dict(goal=goal, target_mol=target_mol,
                  target_sig=float(target_sig),
                  goal_param=goal_param,
                  target_prec=(None if target_prec is None
                               else float(target_prec)),
                  marginalize=bool(marginalize),
                  do_fisher=bool(do_fisher),
                  fisher_params=list(fisher_params),
                  jac_method=jac_method)
_obs_meta = dict(
    ks_mag=float(ks_mag), t14=float(t14), t_base=float(t_base),
    # the Pandeia star, recorded HERE for both science modes: the canonical
    # block zeroes star_teff/logg/feh in transmission (cache hygiene), so a
    # share file without these could not reproduce the noise simulation
    star_teff=float(teff), star_logg=float(logg), star_feh=float(feh),
    sat_limit=float(sat_limit), modes=list(mode_keys),
    n_transits=int(n_transits), r_bin=int(r_bin),
    floor_mode=floor_mode,
    floors={k: (None if floors[k] is None
                else (float(floors[k]) if np.isscalar(floors[k])
                      else "wavelength table"))
            for k in mode_keys},
    # the PER-MODE widget values, not the composed product: the global scale is
    # recorded separately below, and share_config restores noise_infl into the
    # per-mode widgets -- writing the product here would re-multiply by the
    # global scale on every restore.
    noise_infl={k: float(_infl_mode[k]) for k in mode_keys},
    noise_scale=float(noise_scale),
    show_noise=bool(show_noise),
    seed=int(seed),
    combos=[dict(name=str(c["name"]), modes=[str(m) for m in c["modes"]])
            for c in (st.session_state.get(K("combos")) or [])])
# Observation fields that change only what is DRAWN from the cached run (the
# mock draw, its seed, the named mode sets). They stay in _obs_meta -- the
# shareable config and floor_selected read it -- but the staleness guard
# ignores them: they recompute nothing, so they can never make a result stale.
_DISPLAY_ONLY = ("show_noise", "seed", "combos")
_run_sig = dict(goal=_goal_meta,
                observation={k: v for k, v in _obs_meta.items()
                             if k not in _DISPLAY_ONLY})

# The shareable configuration: built here (it needs _canon and the goal /
# observation blocks above) and rendered into the slot reserved beside the
# Run button.
if _canon is not None:
    _share = share_config.build_share(
        canon=_canon,
        goal=_goal_meta,
        observation=_obs_meta,
        tp_table_text=(Path(tp_file_path).read_text()
                       if tp_file_path else None),
        floor_table=(np.asarray(floor_table).tolist()
                     if floor_table is not None else None))
    with _cfg_col:
        st.download_button(
            "Download configuration (JSON)",
            json.dumps(_share, indent=2, default=str).encode(),
            f"jwst_tool_{_slug(planet_label)}_config.json",
            "application/json", key=K("dl_config"), on_click="ignore")


# Compute on click
def compute():
    if params_error:
        st.error(f"Cannot run with the current settings: {params_error}")
        return None
    if not mode_keys:
        st.error("Select at least one instrument mode (step 4).")
        return None
    # Heavy subprocesses (forward + ETC) hold ONE concurrency slot for their
    # whole duration; when every slot is busy the launch is declined. Cached
    # results never need a slot.
    _slot = runlimit.acquire("forward+etc")
    if _slot is None:
        st.error(
            f"This instance is already running {runlimit.MAX_CONCURRENT} "
            "heavy calculations (it is shared, public hardware). Please try "
            "again in a few minutes -- previously computed results stay "
            "instant.")
        return None
    try:
        return _compute_locked()
    finally:
        _slot.release()


def _compute_locked():

    model = forward.load_result(params)
    if model is None:
        with st.status("Running VULCAN-JAX + ExoJAX forward model …",
                       expanded=True) as status:
            # prior = the same rough pre-run estimate shown next to the Run
            # button; the bar's remaining time converges to the measured pace
            bar = _TimedBar(prior_total_s=_est_total_min * 60.0,
                            text="starting …")
            # unique per launch: a key-only name is SHARED between two
            # same-key runs, and one visitor's rewrite can truncate the file
            # under the other's subprocess mid-read
            pfile = forward.MODEL_CACHE / (
                f"{forward.params_key(params)}.params."
                f"{uuid.uuid4().hex[:8]}.json")
            forward.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            pfile.write_text(json.dumps(forward.canonical_params(params)))
            box = st.empty()
            lines = []

            def _fwd_line(line):
                m = _PROG_RE.match(line)
                if m:
                    bar.update(min(1.0, float(m.group(1))), m.group(2))
                else:
                    lines.append(line)
                    # FIXED height: a growing code block shoves every widget
                    # below it down each time a line arrives
                    box.code("\n".join(lines[-10:]), height=_LOG_BOX_PX)
                    bar.tick()

            try:
                with _managed_proc(
                        [sys.executable, str(TOOL_DIR / "forward.py"),
                         str(pfile)]) as proc:
                    _watch_proc(proc, _fwd_line, bar.tick)
                    proc.wait()
            finally:
                pfile.unlink(missing_ok=True)
            if proc.returncode != 0:
                status.update(label="Forward model failed", state="error")
                st.error("The forward model failed. Last output:\n\n```\n"
                         + "\n".join(lines[-25:]) + "\n```")
                return None
            bar.done()
            status.update(label="Forward model done", state="complete")
        model = forward.load_result(params)
        if model is None:
            st.error("The forward model finished but produced no cache "
                     "file. Re-run; if this repeats, report it as a bug.")
            return None

    # ETC: ONLY the selected modes, cached per mode -- a later selection
    # change computes exactly the newly added modes.
    etc_missing = noise_mod.missing_modes(star, list(mode_keys),
                                          sat_limit=sat_limit)
    if not etc_missing:
        etc = noise_mod.run_modes(star, list(mode_keys), sat_limit=sat_limit)
    else:
        with st.status(f"Running Pandeia ETC ({ins.BACKEND_STATUS.split(' /')[0]}) …",
                       expanded=True) as status:
            # no reliable prior for the ETC; the remaining-time readout is
            # purely measured, appearing once the first mode completes
            bar = _TimedBar(text="starting the ETC …")
            box = st.empty()
            lines = []
            n_started = [0]

            def _cb(s):
                if s.startswith("[pandeia] ") and s.endswith("..."):
                    bar.update(n_started[0] / len(etc_missing),
                               s.removeprefix("[pandeia] ")
                               .removesuffix("...")
                               + f" ({n_started[0] + 1}/{len(etc_missing)})")
                    n_started[0] += 1
                else:
                    lines.append(s)
                    box.code("\n".join(lines[-8:]), height=_LOG_BOX_PX)
                    bar.tick()

            etc = noise_mod.run_modes(star, list(mode_keys),
                                      sat_limit=sat_limit, progress=_cb)
            bar.done()
            status.update(label="Pandeia ETC done", state="complete")

    t_in_s, t_out_s = t14 * 3600.0, t_base * 3600.0
    results, failed, unusable = [], [], []
    for k in mode_keys:
        if "error" in etc[k]:
            failed.append((k, etc[k]["error"]))
        elif etc[k].get("unusable") or not etc[k].get("wl"):
            unusable.append((k, etc[k].get("reason", "no usable pixels")))
        else:
            try:
                results.append(detect.evaluate_mode(
                    k, etc[k], model, target_mol, r_bin, t_in_s, t_out_s,
                    n_transits, floors[k], noise_inflation=infl[k]))
            except Exception as e:
                # one bad mode must not kill the whole run -- report it with
                # its label + the actual reason, keep evaluating the rest
                failed.append((k, f"{type(e).__name__}: {e}\n\n"
                                  f"(binning/noise evaluation for {k}; the other "
                                  "modes are unaffected)"))
    return dict(model=model, results=results, failed=failed, unusable=unusable,
                fisher_names=list(fisher_params),
                provenance=etc.get("__provenance__"))


if run_clicked:
    out = compute()
    if out is not None:
        st.session_state["out"] = out
        st.session_state["out_meta"] = dict(
            goal=goal, target=target_mol, goal_param=goal_param,
            target_prec=target_prec, target_sig=target_sig,
            n_transits=n_transits, show_noise=show_noise, seed=seed,
            r_bin=r_bin, planet=planet_label,
            floor_mode=floor_mode,
            # the SELECTED floor, not the registry suggestion: a result must
            # carry the number that produced it
            floor_selected=_obs_meta["floors"],
            # the COMPLETE non-canonical input set, for the staleness guard
            run_sig=_run_sig)

# Render order: staleness/failure notices, the VERDICT, then physical
# structure, the constraint forecast, and the summary figure.
if "out" not in st.session_state:
    st.stop()

out = st.session_state["out"]
meta = st.session_state["out_meta"]
model, results = out["model"], out["results"]
goal_r = meta.get("goal", "detect")
# the atmosphere's absolute C/O, for the dlnCO -> absolute-C/O display
# conversion (sigma_CO = C/O * sigma_lnCO)
_cpj = json.loads(str(model["params_json"]))
# Event word for the CACHED run's results (the sidebar radio may have moved
# since the run; the stored canonical params are the truth for this output).
_ev = ("eclipse" if str(_cpj.get("science_mode", "transmission")) == "emission"
       else "transit")
co_eval = float(_cpj.get("co_ratio", forward.CO_BASELINE))

# Staleness guard: results persist in session_state across sidebar edits, so
# the spectrum shown can be from DIFFERENT settings than the sidebar now
# reads -- most visibly the transmission/emission geometry. Say so loudly and
# LIST the changed fields.
_shown_stale, _changed = False, []
try:
    _cur_cp = json.loads(json.dumps(forward.canonical_params(params),
                                    default=str))
    _shown_stale = forward.params_key(params) != forward.params_key(_cpj)
    if _shown_stale:
        _changed = sorted(k for k in set(_cur_cp) | set(_cpj)
                          if _cur_cp.get(k) != _cpj.get(k))
    # RUN-META, not just canonical model params -- the WHOLE non-canonical
    # input set (see _run_sig, built next to the shareable config). The
    # canonical set deliberately excludes the detection target and the
    # observing setup because they do not change the SPECTRUM, but they do
    # change what is DISPLAYED and what the scores are computed from. Compared
    # through json so nested floors/combos/mode lists compare by value.
    # A stored result with no run_sig predates this guard (it is written by the
    # same block that reads it, so that is the only way to get here). Compare
    # nothing rather than report every field as changed; the canonical-params
    # half of the guard above still applies.
    _sig_was = meta.get("run_sig")
    _meta_chg = []
    for _sec, _now_d in (_run_sig if _sig_was else {}).items():
        _was_d = _sig_was.get(_sec) or {}
        _meta_chg += [
            _k for _k in sorted(set(_now_d) | set(_was_d))
            if json.dumps(_now_d.get(_k), sort_keys=True, default=str)
            != json.dumps(_was_d.get(_k), sort_keys=True, default=str)]
    if _meta_chg:
        _shown_stale = True
        _changed = sorted(set(_changed) | set(_meta_chg))
except (ValueError, RuntimeError):
    _shown_stale = True   # the current sidebar settings do not even validate
if _shown_stale:
    _chg = ((" Changed: " + ", ".join(f"`{c}`" for c in _changed[:8])
             + (", …" if len(_changed) > 8 else "") + ".")
            if _changed else "")
    st.warning("Showing your previous run." + _chg
               + " Press **Run** to recompute.")

for k, err in out["failed"]:
    first = str(err).strip().splitlines()[-1] if "Traceback" in str(err) else \
        str(err).strip().splitlines()[0]
    st.error(f"{ins.MODES[k]['label']}: the calculation failed. {first}")
    with st.expander(f"{ins.MODES[k]['label']}: technical details"):
        st.code(str(err)[-2500:])
for k, reason in out["unusable"]:
    # ngroup_min equals pandeia's permitted minimum ramp for the mode (the
    # same floor PandExo searches to), so "saturated at the shortest ramp"
    # is a real brightness limit, not a tool policy bound.
    st.warning(f"**{ins.MODES[k]['label']}**: unusable, {reason}.")
for r in results:
    if r["saturated"]:
        st.warning(f"**{r['label']}**: saturated at the shortest ramp, "
                   "excluded from the ranking.")

if not results:
    st.stop()

fisher_names = ([str(x) for x in model["jac_names"][:-1]]
                if "jac_names" in model else [])
ok = [r for r in results if not r["saturated"]]

# --- named mode combinations (results-side builder) -------------------------
# Evaluated through posteriors.combo_forecast (the SAME combination math as
# the all-usable-modes row -- never reimplemented here). A combo that
# cannot be evaluated (mode not run, all modes saturated) is reported as an
# error next to the builder; the others still render.
_results_by_mode = {r["mode_key"]: r for r in results
                    if r.get("jac_bins") is not None}
_combos_cfg = st.session_state.get(K("combos")) or []
combo_recs, combo_errs = [], []
if _combos_cfg and fisher_names and _results_by_mode:
    for _c in _combos_cfg:
        try:
            combo_recs.append(posteriors.combo_forecast(
                str(_c["name"]), list(_c["modes"]), _results_by_mode,
                fisher_names, co_eval=co_eval))
        except ValueError as _e:
            combo_errs.append((str(_c["name"]), str(_e)))
elif _combos_cfg:
    combo_errs = [(str(_c["name"]),
                   "no parameter constraints in this run: combinations "
                   "need a run with free parameters (a Fisher forecast)")
                  for _c in _combos_cfg]

# --- shortfall warning (first; nothing when the target is met, warning =
# valid result that misses it, error = only for failed calculations) --------
if goal_r == "detect":
    tsig = float(meta.get("target_sig") or 3.0)
    ntr = meta["n_transits"]
    if not ok:
        # every selected mode saturates: never rank saturated results
        st.warning(f"**No usable mode**: all selected modes saturate "
                   f"(Ks = {star['ks_mag']:g}).")
    else:
        ranked = sorted(ok, key=lambda r: -_detection_metric(r)[0])
        best = ranked[0]
        bsig, _best_projected = _detection_metric(best)
        # "template S/N" stays: a bare sigma reads as a retrieval claim. And
        # SAY WHICH ONE: the projected score profiles the T-P/cloud/lnR0
        # directions as well as the per-segment calibration offsets, so the two
        # are different statistics and must never share one bare label.
        verdict = (f"**{best['label']}**: template S/N {bsig:.1f}σ "
                   f"({_METRIC_LABEL[_best_projected]}) in {ntr} "
                   f"{_ev}{'s' if ntr > 1 else ''} (target {tsig:g}σ).")
        if bsig >= tsig:
            pass        # target met: the figure and table carry the number
        elif bsig > 0:
            # floor-aware transit solver: the photon term averages down with
            # N, the systematic floor does not -- a plain 1/sqrt(N) law is
            # optimistic exactly where it matters
            tt = detect.transits_to_target(
                best, tsig, projected=_best_projected)
            if tt["reachable"]:
                st.warning(verdict + f"  {tt['n']} {_ev}s reach it.")
            elif _has_floor(best):
                st.warning(verdict
                           + f"  Floor caps it at {tt['sig_inf']:.1f}σ.")
            else:
                st.warning(verdict + f"  >{detect.N_TRANSITS_CAP} {_ev}s "
                           "(scan limit).")
        else:
            st.warning(verdict + "  No signal.")
else:
    gp = meta["goal_param"]
    unit = forward.PARAM_UNITS[gp]
    usp = (" " + unit) if unit else ""       # " dex"/" K", or "" for C/O (ratio)
    glabel = forward.PARAM_LABELS[gp]
    target = float(meta["target_prec"])
    tsig = float(meta.get("target_sig") or 3.0)
    with_jac = [r for r in results if r.get("jac_bins") is not None]
    # one saturation policy everywhere: a saturated mode is unusable data,
    # excluded from BOTH the per-mode ranking and the combined forecast
    usable_jac = [r for r in with_jac if not r["saturated"]]
    per_mode = {}          # tsig-sigma half-widths, display units
    for r in usable_jac:
        s = fisher_mod.display_sigma(gp, fisher_mod.mode_forecast(r, fisher_names)[gp],
                                     co_eval=co_eval)
        if np.isfinite(s):
            per_mode[r["mode_key"]] = tsig * s
    comb = (tsig * fisher_mod.display_sigma(
        gp, fisher_mod.combined_forecast(usable_jac, fisher_names)[gp], co_eval=co_eval)
        if len(usable_jac) >= 2 else np.inf)
    # distinguish "all modes saturated" from "no spectral response"
    if with_jac and not usable_jac:
        st.warning(f"**No usable mode**: all selected modes saturate "
                   f"(Ks = {star['ks_mag']:g}).")
        st.stop()
    if not per_mode:
        st.warning(f"No selected mode constrains {glabel}.")
        st.stop()
    bk = min(per_mode, key=per_mode.get)
    bs = per_mode[bk]
    ntr = meta["n_transits"]
    verdict = (f"**{ins.MODES[bk]['label']}**: ±{bs:.3g}{usp} at "
               f"{tsig:g}σ in {ntr} {_ev}{'s' if ntr > 1 else ''} "
               f"(target ±{target:g}{usp}).")
    if bs <= target:
        pass                      # target met: nothing to add
    elif np.isfinite(comb) and comb <= target:
        st.warning(verdict + f"  Combined modes: ±{comb:.3g}{usp}.")
    else:
        best_r = next(r for r in usable_jac if r["mode_key"] == bk)
        tt = fisher_mod.transits_to_target(best_r, fisher_names, gp,
                                           target / tsig, detect.sigma_at_transits,
                                           co_eval=co_eval)
        if tt["reachable"]:
            st.warning(verdict + f"  {tt['n']} {_ev}s reach it.")
        elif _has_floor(best_r):
            st.warning(verdict + "  Floor caps it at "
                       f"±{tsig * tt['sig_inf']:.3g}{usp}.")
        else:
            st.warning(verdict + f"  >{detect.N_TRANSITS_CAP} {_ev}s "
                       "(scan limit).")

# --- spectrum data (rendered ONCE, on the summary figure below) -------------
wl = model["wl_um"]
order = np.argsort(wl)
wl_s, d_s = wl[order], model["depth"][order] * 1e6
_fname_base = f"jwst_tool_{_slug(meta.get('planet', 'planet'))}"

# DISPLAY smoothing: at the model's own resolving power (R = 1000 on the
# correlated-k band grid) the unresolved line
# forest renders as one-sample spikes, so the PLOT is convolved to a
# constant display R (>= 3x the analysis R, floor 300) with the SAME tested
# LSF operator the science path uses (flat weight). That operator no-ops
# when the model grid cannot resolve the kernel, so past an analysis R of
# about 140 this is already the native curve. No score touches it; the
# native model stays in the "Native model (CSV)" download.
_disp_R = float(max(300, 3 * int(meta["r_bin"])))
_disp_wl_r = np.array([float(wl_s[0]), float(wl_s[-1])])
_disp_curve = np.array([_disp_R, _disp_R])


def _display_smooth(y_ppm):
    return binning.smooth_to_native_r(wl_s, y_ppm, _disp_wl_r, _disp_curve,
                                      float(wl_s[0]), float(wl_s[-1]))


d_plot = _display_smooth(d_s)
d_wo_s = None
if goal_r == "detect":
    # depth_wo rows align with the model's wo_mols set, not mols
    wo_mols_r = [str(x) for x in model["wo_mols"]]
    d_wo_s = model["depth_wo"][wo_mols_r.index(meta["target"])][order] * 1e6
# Mock-observation layer: one seeded N(0, sigma_i) draw per
# bin ON TOP of the binned noiseless model, generated AFTER the forward
# model and the noise model (posteriors.mock_realization). The LIVE widget
# state drives it (not the stored run meta), so editing the seed redraws
# without recomputing anything; the seed is displayed and reproducible.
# HARD RULE: nothing from this layer enters detection/Fisher scores, caches,
# or the result CSVs -- only the clearly-named mock download below, and the
# mock_recovery overlay on the posterior panels, which IS fitted to it.
_mock = (posteriors.mock_realization(results, int(seed))
         if show_noise else None)
_depth_lbl = ("eclipse depth (ppm)"
              if str(model.get("science_mode", "transmission")) == "emission"
              else "transit depth (ppm)")


def _binned_series(r, mock=None):
    """One evaluated mode as (wl_eff um, depth ppm, sigma ppm) -- the plotted
    x coordinate (it differs from wl_um near a detector gap), depths in ppm.
    ``mock`` substitutes that mode's seeded draw for the noiseless depth."""
    depth = (mock["modes"][r["mode_key"]]["depth_mock"] if mock is not None
             else r["depth"])
    return (np.asarray(r.get("wl_eff", r["wl"]), dtype=float),
            np.asarray(depth, dtype=float) * 1e6,
            np.asarray(r["sigma"], dtype=float) * 1e6)


# plotted/binned numbers for the downloads under the summary figure
_bin_rows = []
for r in results:
    _bwl, _bd, _bs = _binned_series(r)
    _bin_rows.append(pd.DataFrame({
        "mode": r["mode_key"], "label": r["label"],
        "wl_um": np.asarray(r["wl"], dtype=float),
        "wl_eff_um": _bwl, "depth_ppm": _bd, "sigma_ppm": _bs,
        "saturated": bool(r["saturated"])}))
_bin_df = pd.concat(_bin_rows, ignore_index=True)
_native = {"wl_um": wl_s, "depth_ppm": d_s}
if d_wo_s is not None:
    _native[f"depth_{meta['target']}_opacity_removed_ppm"] = d_wo_s

with st.expander("Physical structure (T-P profile, mixing ratios)"):
    # ONE two-panel figure (plotting.build_structure_figure, pure and
    # importable without streamlit, so the threaded regression test
    # exercises this exact code). No figure-settings block: the panels use
    # the module's default windows, and the shared pressure axis is the
    # figure's own (sharey), not something a widget can desynchronize.
    _p_arr = np.asarray(model["p_bar"], dtype=float)
    _T_arr = np.asarray(model["T"], dtype=float)
    _ymix = model.get("ymix")
    _ysp = model.get("ymix_species")
    if _ymix is None or _ysp is None:
        st.info("This run predates the labelled mixing-ratio profiles "
                "(re-run to populate them).")
    else:
        _ymix = np.asarray(_ymix, dtype=float)
        _ysp = [str(s) for s in np.asarray(_ysp)]
        if _ymix.ndim != 2 or _ymix.shape[0] != _p_arr.size:
            raise ValueError(
                f"ymix shape {_ymix.shape} does not match the pressure grid "
                f"({_p_arr.size} layers): the stored model is inconsistent")
        if len(_ysp) != _ymix.shape[1]:
            raise ValueError(
                f"ymix has {_ymix.shape[1]} columns but "
                f"{len(_ysp)} species names: the stored model is "
                "inconsistent")
        # Select the RT molecules BY NAME: ymix is the full network state
        # while model["mols"] is the RT subset, so zipping them
        # positionally reads the wrong species entirely. Ordered by peak
        # abundance so the legend reads top-down.
        # engine_config.MOLECULES maps the RT token to the VULCAN species
        # name; they differ (the tool's OCS is VULCAN's COS)
        from jwst_tool import engine_config as _ec
        _want = {_ec.MOLECULES[str(m)]["vulcan"]: str(m)
                 for m in np.asarray(model["mols"]).tolist()
                 if str(m) in _ec.MOLECULES}
        _cols = [(_want[s], _ymix[:, _i]) for _i, s in enumerate(_ysp)
                 if s in _want]
        _cols.sort(key=lambda kv: -float(np.nanmax(kv[1])))
        fig3 = plotting.build_structure_figure(_p_arr, _T_arr, _cols)
        _struct_png = _fig_bytes(fig3, "png", tight=False)
        _show_fig(fig3, tight=False)

        if _cpj.get("science_mode") == "emission":
            if float(_T_arr.max()) > 2000.0:
                st.warning(
                    f"Layers exceed 2000 K (max {_T_arr.max():.0f} K); "
                    "ultra-hot opacities not modeled.")
        else:
            # transmission probes p <~ 0.1 bar; a hot deep adiabat below that is
            # invisible to the chord geometry and must not trip the warning
            _probe = _p_arr <= 0.1
            if _probe.any() and float(_T_arr[_probe].max()) > 2000.0:
                st.warning("Photosphere >2000 K; ultra-hot opacities not "
                           "modeled.")

        _tp_df = pd.DataFrame({"p_bar": _p_arr, "T_K": _T_arr})
        _vmr_df = pd.DataFrame({"p_bar": _p_arr} | {m: y for m, y in _cols})
        _s1, _s2, _s3 = st.columns(3)
        _s1.download_button("Figure (PNG)", _struct_png,
                            f"{_fname_base}_structure.png", "image/png",
                            key=K("dl_struct_png"), on_click="ignore")
        _s2.download_button("T-P values (CSV)", _csv_bytes(_tp_df),
                            f"{_fname_base}_tp_profile.csv", "text/csv",
                            key=K("dl_tp_csv"), on_click="ignore")
        _s3.download_button("Mixing ratios (CSV)", _csv_bytes(_vmr_df),
                            f"{_fname_base}_mixing_ratios.csv", "text/csv",
                            key=K("dl_vmr_csv"), on_click="ignore")


with st.expander("Parameter constraint forecast (local Fisher)"):
    # --- parameter constraint forecast (Fisher) --------------------------------
    # authoritative parameter order = the Jacobian rows as cached (canonical/sorted),
    # NOT the multiselect order
    if fisher_names and "jac" in model:
        # --- custom mode sets (builder) ------------------------------------
        # Named combinations of the modes that were run, evaluated through
        # posteriors.combo_forecast (the same combination math as the
        # all-usable-modes row). They add rows to the table below and bars
        # to the comparison chart above.
        _cb_opts = [r["mode_key"] for r in results
                    if r.get("jac_bins") is not None]
        # a stored selection can reference a mode absent from this run's results
        # (Streamlit crashes at widget instantiation on off-menu session state)
        _cb_stale = st.session_state.get(K("cb_modes"))
        if _cb_stale is not None:
            _cb_kept = [m for m in _cb_stale if m in _cb_opts]
            if _cb_kept != list(_cb_stale):
                st.session_state[K("cb_modes")] = _cb_kept
        _note = st.session_state.pop("_combo_note", None)
        if _note is not None:
            getattr(st, _note[0])(_note[1])
        _usable_keys = [r["mode_key"] for r in results
                        if r.get("jac_bins") is not None and not r["saturated"]]

        def _combo_add_all_usable() -> None:
            combos = st.session_state.setdefault(K("combos"), [])
            if any(c["name"] == "All usable" for c in combos):
                st.session_state["_combo_note"] = (
                    "info", "The 'All usable' combination already exists.")
                return
            combos.append(dict(name="All usable", modes=list(_usable_keys)))
            st.session_state["_combo_note"] = (
                "success", "Added the 'All usable' combination.")

        _cbc1, _cbc2 = st.columns([1.6, 2.2])
        _cbc1.text_input("Combination name", key=K("cb_name"),
                         placeholder="e.g. SOSS + G395H")
        _cbc2.multiselect("Modes in the combination", _cb_opts,
                          key=K("cb_modes"),
                          format_func=lambda k: ins.MODES[k]["label"],
                          help="Pick the modes the observation would use "
                               "together. The set gets one combined "
                               "forecast row and can be drawn as one "
                               "series in the figure.")
        # Buttons on their OWN row, not in a third column beside the inputs:
        # the inputs carry labels, so a column-mounted button floats to the
        # top of the row and lines up with nothing.
        _cba1, _cba2, _ = st.columns([1.2, 1.6, 2.0])
        _cba1.button("Add combination", key=K("cb_add"), on_click=_combo_add,
                     width="stretch")
        if len(_usable_keys) >= 2:
            _cba2.button("Add preset: all usable modes", key=K("cb_add_all"),
                         on_click=_combo_add_all_usable, width="stretch")
        for _i, _c in enumerate(st.session_state.get(K("combos")) or []):
            _cc1, _cc2 = st.columns([4.0, 1.0])
            _cc1.markdown(
                f"- **{_c['name']}**: "
                + ", ".join(ins.MODES[m]["label"] if m in ins.MODES else m
                            for m in _c["modes"]))
            _cc2.button("Remove", key=K(f"cb_rm_{_i}"), on_click=_combo_remove,
                        args=(_i,))
        for _cname, _cerr in combo_errs:
            st.error(f"Combination {_cname!r} could not be forecast: {_cerr}")
        for _rec in combo_recs:
            if _rec["excluded"]:
                st.warning(
                    f"Combination {_rec['name']!r}: "
                    + ", ".join(ins.MODES[e["mode_key"]]["label"]
                                if e["mode_key"] in ins.MODES else e["mode_key"]
                                for e in _rec["excluded"])
                    + (" saturate; uses " if len(_rec["excluded"]) > 1
                       else " saturates; uses ")
                    + " + ".join(ins.MODES[m]["label"]
                                 for m in _rec["usable_modes"]) + ".")
        st.divider()

        tsig_f = float(meta.get("target_sig") or 3.0)
        with_jac = [r for r in results if r.get("jac_bins") is not None]

        def _cell(n, s):
            v = tsig_f * fisher_mod.display_sigma(n, s, co_eval=co_eval)
            return "unconstrained" if not np.isfinite(v) or v > 1e4 else f"{v:.3g}"

        # long format, one row per mode x parameter, marginalized and conditional
        # side by side -- both read off the SAME nuisance-augmented Fisher matrix
        _marg_col = f"marginalized ± at {tsig_f:g}σ"
        _cond_col = "conditional ± (others fixed)"

        def _param_rows(mode_label, sig, cond):
            return [{"mode": mode_label,
                     "parameter": forward.PARAM_LABELS[n],
                     _marg_col: _cell(n, sig[n]),
                     _cond_col: _cell(n, cond[n])}
                    for n in fisher_names]

        frows = []
        usable_f = [r for r in with_jac if not r["saturated"]]
        for r in with_jac:
            if r["saturated"]:
                # shown for completeness; a saturated mode contributes no usable
                # data (same exclusion policy as the verdict + combined)
                frows.append({"mode": r["label"],
                              "parameter": "(saturated, excluded)",
                              _marg_col: "", _cond_col: ""})
                continue
            cond = {}
            sig = fisher_mod.mode_forecast(r, fisher_names, conditional=cond)
            frows.extend(_param_rows(r["label"], sig, cond))
        fdiag = {}
        if len(usable_f) >= 2:
            cond = {}
            sig = fisher_mod.combined_forecast(usable_f, fisher_names, diag=fdiag,
                                               conditional=cond)
            frows.extend(_param_rows(_ALL_USABLE, sig, cond))
        # named combinations: long-format rows from the SAME records feeding the
        # comparison chart (posteriors.combo_forecast; display units already
        # applied, so the rows re-scale to tsig directly)
        for _rec in combo_recs:
            for n in fisher_names:
                _sm = tsig_f * float(_rec["sigma_marginalized_display"][n])
                _sc = tsig_f * float(_rec["sigma_conditional_display"][n])
                frows.append({
                    # the user's own name for the set, unprefixed
                    "mode": str(_rec["name"]),
                    "parameter": forward.PARAM_LABELS[n],
                    _marg_col: ("unconstrained"
                                if not np.isfinite(_sm) or _sm > 1e4
                                else f"{_sm:.3g}"),
                    _cond_col: ("unconstrained"
                                if not np.isfinite(_sc) or _sc > 1e4
                                else f"{_sc:.3g}")})
        # Custom combinations FIRST: they are what the user built, so they
        # lead the table. Order within each group is preserved.
        _combo_names = {str(_rec["name"]) for _rec in combo_recs}
        frows = ([_r for _r in frows if _r["mode"] in _combo_names]
                 + [_r for _r in frows if _r["mode"] not in _combo_names])

        # Repeated mode names blanked for READING only; the CSV below is
        # built from the UNBLANKED rows -- a machine-readable file must carry
        # the mode on every row. _is_combo rides along for the highlight and
        # is dropped before display.
        _disp, _prev = [], None
        for _r in frows:
            _r2 = dict(_r)
            _r2["_is_combo"] = _r2["mode"] in _combo_names
            if _r2["mode"] == _prev:
                _r2["mode"] = ""
            else:
                _prev = _r2["mode"]
            _disp.append(_r2)
        _disp_df = pd.DataFrame(_disp)
        _combo_mask = _disp_df.pop("_is_combo")
        # st.table, NOT st.dataframe: st.dataframe is interactive (a column-
        # header click re-sorts), and this table cannot survive that -- the
        # mode column is blanked on repeat rows, so a re-sort detaches every
        # blank row from its mode and silently attributes numbers to the
        # wrong instrument. st.table is static by definition.
        if _combo_mask.any():
            # tint the custom-set rows so they read as a group distinct from
            # the per-mode rows. TEXT color, not a background tint (a filled
            # row reads as a highlight/alert); #5b3a8e is the combo
            # palette's purple, dark enough for body text on white.
            st.table(
                _disp_df.style.apply(
                    lambda row: ["color: #5b3a8e; font-weight: 600"
                                 if _combo_mask.loc[row.name] else ""
                                 for _ in row], axis=1))
        else:
            st.table(_disp_df)
        st.download_button("Constraint forecast (CSV)",
                           _csv_bytes(pd.DataFrame(frows)),
                           f"{_fname_base}_fisher_forecast.csv", "text/csv",
                           key=K("dl_fisher_csv"), on_click="ignore")
        # No full-rank numerical-health prose; a rank DEFICIENCY is still
        # disclosed -- degenerate directions would otherwise read as
        # silently missing rows
        if fdiag and fdiag["fisher_rank"] < fdiag["fisher_dimension"]:
            st.caption(
                f"**Fisher matrix is rank-deficient ({fdiag['fisher_rank']} "
                f"of {fdiag['fisher_dimension']}): degenerate directions "
                "read as unconstrained.**")
        # Reference material on reading this table belongs in README.md, not
        # in the GUI.
        st.divider()
        # Filled by the marginalized-forecast block below, which needs the
        # posterior records built after this panel renders.
        _post_slot = st.container()
    elif out.get("fisher_names"):
        st.info("A constraint forecast was requested but the cached model has "
                "no Jacobian. Press Run to compute it.")



# --- marginalized forecast posteriors + proposal summary figure ------------
def _param_center(name: str, cpj: dict):
    """Input-model value of ``name`` in DISPLAY units (the Gaussian's
    center), or None when the run's stored parameters define no single value
    (e.g. lnKzz under a non-constant mixing profile, or a field the cached
    run predates). A None center draws no curve -- the panel says so
    explicitly instead of guessing a center."""
    if name == "lnZ":
        v = cpj.get("met_x_solar")
        return None if v in (None, "") else float(np.log10(float(v)))
    if name == "dlnCO":
        return float(cpj.get("co_ratio", forward.CO_BASELINE))
    if name == "lnKzz":
        v = cpj.get("kzz_const")
        if str(cpj.get("kzz_mode", "const")) == "const" and v not in (None, ""):
            return float(np.log10(float(v)))
        return None
    direct = {"Tirr": "Tirr", "Tint": "Tint", "log_kappa": "log_kappa",
              "log_gamma": "log_gamma", "Tint_cl": "tint_cl",
              "log_kappa_cloud": "log_kappa_cloud",
              "alpha_cloud": "alpha_cloud"}
    k = direct.get(name)
    if k is not None and cpj.get(k) is not None:
        return float(cpj[k])
    return None


_post_panels: list[dict] = []       # summary_figure-compatible panel dicts
_post_sel: list[str] = []
_have_fisher = bool(fisher_names) and "jac" in model
if _have_fisher:
    # These controls live INSIDE the constraint-forecast panel (the slot
    # reserved there), not in an expander of their own: on their own they
    # were one multiselect in an otherwise empty section.
    # "Fisher forecasts", never "posteriors": these curves are Gaussian
    # slices of the local Fisher ellipse, not retrieval posteriors
    _post_box = _post_slot

    # ONE color per series, shared by the spectrum and the forecast panels.
    # A mode keeps instruments.MODE_COLOR so its points and its posterior
    # match; a custom set draws from _COMBO_COLORS, which never collides
    # with a member mode's color; the all-usable row gets a neutral gray.
    _ALL_USABLE_COLOR = "#444444"

    def _series_color(label: str) -> str:
        if label == _ALL_USABLE:
            return _ALL_USABLE_COLOR
        for _r in results:
            if _r["label"] == label:
                return ins.MODE_COLOR.get(_r["mode_key"], "#333333")
        _names = [str(c["name"]) for c in
                  (st.session_state.get(K("combos")) or [])]
        if label in _names:
            return _COMBO_COLORS[_names.index(label) % len(_COMBO_COLORS)]
        return "#333333"

    _usable_post = [r for r in results
                    if r.get("jac_bins") is not None and not r["saturated"]]
    # forecast sources: each usable mode, the all-usable combination, and
    # every named combination from the builder above
    _sources: dict[str, list] = {}
    for r in _usable_post:
        _sources[r["label"]] = [r]
    if len(_usable_post) >= 2:
        _sources[_ALL_USABLE] = list(_usable_post)
    for _rec in combo_recs:
        _sources[str(_rec["name"])] = [
            _results_by_mode[k] for k in _rec["usable_modes"]]

    if not _sources:
        with _post_box:
            st.info("No usable mode carries a Jacobian, so there is no "
                    "forecast to draw.")
    else:
        _centers_all = {n: _param_center(n, _cpj) for n in fisher_names}
        _pp_key = K("post_params_" + "_".join(fisher_names))
        # a stored selection can go stale across runs (different free set)
        if any(p not in fisher_names
               for p in st.session_state.get(_pp_key, [])):
            st.session_state.pop(_pp_key, None)
        with _post_box:
            _post_sel = st.multiselect(
                "Marginalized forecast curves to draw", fisher_names,
                default=fisher_names[:2], key=_pp_key,
                max_selections=_MAX_POST_PANELS,
                format_func=lambda n: forward.PARAM_LABELS[n])
        # record per source (sigmas always; curves for centered params)
        _curve_params = [p for p in _post_sel
                         if _centers_all.get(p) is not None]
        _centers = {p: _centers_all[p] for p in _curve_params}
        _recs_by_src = {}
        for _lbl, _rl in _sources.items():
            _recs_by_src[_lbl] = posteriors.marginalized_posteriors(
                _rl, fisher_names, _centers,
                params=_curve_params, co_eval=co_eval)

        # SOURCES FOLLOW THE SPECTRUM SERIES: there is no separate "Forecast
        # source" control -- the figure's series multiselect drives both
        # halves of the figure, so the panels can never show a mode the
        # spectrum is not displaying.
        #
        # That multiselect renders LATER in the script (it belongs beside the
        # figure), so read its committed state here -- the same cross-section
        # read the sidebar already uses. On the very first render the key is
        # absent and every source is drawn, which matches the multiselect's own
        # default of all modes.
        _sel_ids = st.session_state.get(K("sum_series"))
        if _sel_ids:
            _by_key = {r["mode_key"]: r["label"] for r in _usable_post}
            _want = []
            for _i in _sel_ids:
                _kind, _key = _i.split(":", 1)
                if _kind == "allusable":
                    _want.append(_ALL_USABLE)
                elif _kind == "mode" and _key in _by_key:
                    _want.append(_by_key[_key])       # saturated/Jacobian-less
                elif _kind == "combo":                # modes drop out here
                    _want.append(_key)
            _sources = {k: v for k, v in _sources.items() if k in _want}
        if not _sources:
            with _post_box:
                st.info("No selected series carries a Jacobian, so there is "
                        "no forecast curve to draw. Widen 'Spectrum & "
                        "forecast series' below, or read the widths from "
                        "the table above.")
        _src_labels_drawn = list(_sources)

        _mock_rec = {}
        if _mock is not None and _post_sel:
            # linearized single-realization recovery on the SAME stacked
            # system as the forecast (posteriors.mock_recovery), per source
            for _lbl in _src_labels_drawn:
                _mock_rec[_lbl] = posteriors.mock_recovery(
                    _sources[_lbl], fisher_names, _mock, co_eval=co_eval)

        for _p in _post_sel:
            _curves, _notes = [], []
            for _lbl in _src_labels_drawn:
                # one curve per SELECTED SERIES, colored to match its series
                # on the spectrum so the two halves of the figure read
                # together (see _series_color below for the shared mapping)
                _pr = _recs_by_src[_lbl]["params"].get(_p)
                _col = _series_color(_lbl)
                if _pr is None:
                    _sig = _recs_by_src[_lbl]["sigma_marginalized"][_p]
                    if np.isfinite(_sig):
                        _notes.append(f"{_lbl}: no input-model center is "
                                      "defined for this parameter under "
                                      "the run's settings, so no curve is "
                                      "drawn (its forecast width is in the "
                                      "table above)")
                    else:
                        _notes.append(f"{_lbl}: unconstrained (no curve)")
                elif _pr["constrained"]:
                    _mr = _mock_rec.get(_lbl)
                    _mc = None
                    if _mr is not None and _mr["recovered"].get(_p):
                        # THE JITTERED CURVE is the one drawn: same width as
                        # the forecast (it is the forecast, recentered on
                        # what this realization recovers), so no width
                        # information is lost by dropping the unshifted twin.
                        # The dotted center line marks the input value.
                        if _p == "dlnCO":
                            # C/O lives on (0, inf): shift the center by the
                            # INTERNAL ln-space draw (multiplicative, stays
                            # positive; center + delta_display can go
                            # negative for an unconstrained C/O) and draw
                            # the same clipped-at-zero Gaussian family the
                            # no-draw forecast uses. A weakly constrained C/O
                            # sends exp(delta) off the forecast scale, where
                            # the input-point width no longer describes the
                            # shifted center: that draw has no curve
                            # (mock_center_co returns None) and the unshifted
                            # forecast is drawn instead.
                            _mu_d = posteriors.mock_center_co(
                                _pr["center"], _pr["sigma_display"],
                                _mr["delta"][_p])
                            if _mu_d is None:
                                _notes.append(
                                    f"{_lbl}: C/O is effectively "
                                    "unconstrained -- the jitter shifts "
                                    f"ln(C/O) by {_mr['delta'][_p]:+.3g}, "
                                    "off the panel scale, so the unshifted "
                                    "forecast is drawn")
                            else:
                                _mc = posteriors.truncated_gaussian_curve(
                                    _mu_d, _pr["sigma_display"])
                        else:
                            _mu_d = _pr["center"] + float(
                                _mr["delta_display"][_p])
                            _mc = posteriors.gaussian_curve(
                                _mu_d, _pr["sigma_display"])
                    if _mc is not None:
                        _curves.append(dict(
                            # The series keeps its plain name: the panel's
                            # y axis ("relative density", not "relative
                            # forecast density") already says this curve is
                            # a fit whose center moved off the input value.
                            label=str(_lbl),
                            theta=_mc["theta"], pdf=_mc["pdf"],
                            # mu/sigma make the figure REPORT the width: the
                            # curve's outline cannot, since each axis
                            # auto-scales to its own +/-5 sigma
                            mu=_mu_d,
                            sigma=_pr["sigma_display"],
                            curve_family=_pr["curve_family"],
                            color=_col, ls="-", lw=1.8,
                            kind=posteriors.MOCK_RECOVERY_KIND))
                    else:
                        _curves.append(dict(
                            label=str(_lbl), theta=_pr["theta"],
                            pdf=_pr["pdf"], mu=_pr["center"],
                            sigma=_pr["sigma_display"],
                            curve_family=_pr["curve_family"],
                            color=_col, ls="-", lw=1.8))
                else:
                    _notes.append(f"{_lbl}: unconstrained -- this "
                                  "direction carries no information in "
                                  "the fitted band (no curve, by design)")
            # A panel whose curves are FITS to the jitter draw is not a
            # forecast: its centers move with the realization, while a
            # forecast's center is the input value by construction. The y-axis
            # name says which of the two the reader is looking at.
            _fitted = any(c.get("kind") == posteriors.MOCK_RECOVERY_KIND
                          for c in _curves)
            _post_panels.append(dict(
                # the parameter NAME, so the axis-range widget keys below stay
                # stable when the selection reorders (compose_summary_figure
                # rebuilds each panel and drops keys it does not use)
                param=str(_p),
                axis_label=forward.param_axis(_p),
                axis_unit=forward.PARAM_UNITS.get(_p, ""),
                density_label=("relative density" if _fitted
                               else "relative forecast density"),
                curves=_curves, notes=_notes,
                center=_centers_all.get(_p)))

# --- the results figure (spectrum + forecast posteriors, rendered once) -----
# one target significance for every number on this page
_target_sig = float(meta.get("target_sig") or 3.0)
# An EXPANDER like every other results section. Expanded by DEFAULT: it
# holds the figure the whole page builds toward, so it should be visible on
# arrival, unlike the supporting sections above it.
_fig_box = st.expander(
    "Simulated eclipse emission spectrum & forecast summary"
    if str(_cpj.get("science_mode", "transmission")) == "emission"
    else "Simulated transmission spectrum & forecast summary",
    expanded=True)

# Per-mode expected performance, rendered IN the legend label of each
# point series: the conditional template S/N for a detection goal, the
# expected ± on a chosen parameter for a constraint/Fisher run.
_leg_num: dict = {}
if goal_r == "detect":
    # saturated modes carry no usable data anywhere else (rankings,
    # combinations, forecasts); they get no score in the legend either
    _leg_projected = set()
    for r in results:
        _score, _proj = _detection_metric(r)
        if not r["saturated"] and np.isfinite(_score):
            _leg_num[r["mode_key"]] = f"S/N {_score:.1f}σ"
            _leg_projected.add(_proj)
elif _have_fisher:
    _rk_key = K("sum_rank_param_" + "_".join(fisher_names))
    if st.session_state.get(_rk_key) not in fisher_names:
        st.session_state.pop(_rk_key, None)
    _gp_default = meta.get("goal_param")
    with _fig_box:
        _rk_param = st.selectbox(
            "Legend parameter (per-mode Fisher ±)", fisher_names,
            index=(fisher_names.index(_gp_default)
                   if _gp_default in fisher_names else 0),
            key=_rk_key, format_func=lambda n: forward.PARAM_LABELS[n])
    for r in [x for x in results if x.get("jac_bins") is not None
              and not x["saturated"]]:
        _v = _target_sig * fisher_mod.display_sigma(
            _rk_param, fisher_mod.mode_forecast(r, fisher_names)[_rk_param],
            co_eval=co_eval)
        if np.isfinite(_v):
            _leg_num[r["mode_key"]] = f"±{_v:.3g}"

# --- figure controls ---------------------------------------------------------
# The figure re-renders from the CACHED run: which series are drawn and what
# wavelength window is shown are display choices, so none of this recomputes
# a spectrum or an ETC job. forward.params_key excludes the mode selection
# and the Pandeia cache is per-mode, so re-rendering is free.
_usable = [r for r in results if not r["saturated"]]
_series_opts = [("mode", r["mode_key"], r["label"]) for r in _usable]
# custom sets from the combination builder: scored through Fisher, with no
# spectrum of their own, so a combo's series is its members' points merged
# onto one line (disclosed in the label).
_combo_members = {str(c["name"]): [m for m in c["modes"]
                                   if any(r["mode_key"] == m for r in _usable)]
                  for c in (st.session_state.get(K("combos")) or [])}
_series_opts += [("combo", n, f"{n} (combined)")
                 for n, mem in _combo_members.items() if mem]
# The all-usable row is a FORECAST source, not a distinct spectrum: selecting
# it draws every usable mode's points (which is what those modes already
# draw), and its value is that its combined posterior can appear in the
# panels. Labelled so that redundancy is not a surprise.
if len(_usable) >= 2:
    _series_opts.append(("allusable", "all",
                         f"{_ALL_USABLE} (forecast only)"))
_series_ids = [f"{kind}:{key}" for kind, key, _ in _series_opts]
_series_lbl = {f"{kind}:{key}": lbl for kind, key, lbl in _series_opts}

_fig_ctx = _fig_box.container()
with _fig_ctx:
    _sel_series = st.multiselect(
        "Spectrum & forecast series", _series_ids,
        default=[i for i in _series_ids if i.startswith("mode:")],
        format_func=lambda i: _series_lbl[i], key=K("sum_series"))


# Default window = the span the SELECTED modes actually cover, so the
# figure is not mostly empty spectrum. Recomputed from the selection.
def _series_modes(i: str) -> list:
    """Mode keys a series id covers. The all-usable entry spans every
    usable mode; without this it fell through to an empty combo lookup and
    contributed nothing to the window."""
    _kind, _key = i.split(":", 1)
    if _kind == "mode":
        return [_key]
    if _kind == "allusable":
        return [r["mode_key"] for r in _usable]
    return _combo_members.get(_key, [])


_cov = [(ins.MODES[k]["wl_min"], ins.MODES[k]["wl_max"])
        for i in _sel_series for k in _series_modes(i)
        if k in ins.MODES]
_grid_lo, _grid_hi = float(wl_s[0]), float(wl_s[-1])
if _cov:
    _fit = (max(_grid_lo, min(c[0] for c in _cov) * 0.97),
            min(_grid_hi, max(c[1] for c in _cov) * 1.03))
else:
    _fit = (_grid_lo, _grid_hi)

# Figure settings. These move the WINDOW and the SCALES the figure is drawn
# with and never the data, so
# nothing downstream reads them -- the scores, the CSVs and the forecast are
# unchanged by anything in here. One row per axis: typed min and max, blank
# for automatic (see _axis_range), with that axis's log toggle on the same
# row. x defaults to log (the wavelength convention here); y to linear,
# since transit depth spans a narrow range where log adds nothing.
with _fig_ctx.expander("Figure settings"):
    _wx, _wlg = st.columns([2.6, 0.9], vertical_alignment="bottom")
    _wl_range = _axis_range(
        _wx, "Wavelength", K("sum_x"), _fig_box.warning, unit="um", step=0.1,
        positive=True)
    _x_log = _wlg.checkbox("Log x", value=True, key=K("sum_xlog"))
    _dx, _dlg = st.columns([2.6, 0.9], vertical_alignment="bottom")
    _depth_range = _axis_range(
        _dx, "Depth", K("sum_y"), _fig_box.warning, unit="ppm", step=1.0,
)
    _y_log = _dlg.checkbox("Log y", value=False, key=K("sum_ylog"))
    # no unit= here: forward.param_axis already carries it inside the label
    # ("[M/H] [dex]"), so passing one again reads "[M/H] [dex] min (dex)"
    _post_xlims = [
        _axis_range(st, p["axis_label"], K("sum_post_" + p["param"]),
                    _fig_box.warning,
                    # C/O is the one panel on a positive-only axis: the
                    # axis is LINEAR, but a ratio has no values at or
                    # below zero (the wavelength axis is the log case)
                    positive=(p["param"] == "dlnCO"),
                    positive_reason="C/O is a ratio, so only positive "
                                    "values exist",
            )
        for p in _post_panels]
# Blank wavelength boxes fall back to the span the SELECTED modes cover, so
# choosing modes remains a wavelength control on its own.
if _wl_range is None:
    _wl_range = _fit

_sum_points = []
for r in results:
    # saturated modes are excluded from every ranking, combination and
    # forecast; plotting simulated points for one would give unusable data
    # scientific weight. The saturation itself is disclosed in the verdict.
    if r["saturated"]:
        continue
    if (f"mode:{r['mode_key']}" not in _sel_series
            and "allusable:all" not in _sel_series):
        continue                      # deselected in the controls above
    _pwl, _pdep, _psig = _binned_series(r, _mock)
    _lbl = r["label"]
    if r["mode_key"] in _leg_num:
        _lbl += f": {_leg_num[r['mode_key']]}"
    _sum_points.append(dict(
        label=_lbl,
        color=ins.MODE_COLOR[r["mode_key"]],
        # Uniform marker for every mode: distinct shapes are
        # indistinguishable at the ~3.6 pt size these points render at.
        # Modes are distinguished by color plus the legend entry.
        marker=ins.MODE_MARKER.get(r["mode_key"], "o"),
        wl_um=_pwl, depth_ppm=_pdep, sigma_ppm=_psig))
_leg_note = None
if _leg_num:
    # ONE short line, rendered as the legend's TITLE (not folded into the
    # model label, which made that entry multi-line and broke the legend's
    # row spacing). Says what the per-mode numbers are, nothing more.
    _leg_note = (
        f"{meta['target']} template S/N per mode "
        f"({'/'.join(_METRIC_LABEL[p] for p in sorted(_leg_projected))}), "
        f"{meta['n_transits']} {_ev}"
        f"{'s' if meta['n_transits'] > 1 else ''}"
        if goal_r == "detect" else
        f"Fisher ±{forward.param_axis(_rk_param)} per mode "
        f"at {_target_sig:g}σ")
for _ci, (_cname, _members) in enumerate(_combo_members.items()):
    if f"combo:{_cname}" not in _sel_series:
        continue
    _cw, _cd, _cs = [], [], []
    for _mk in _members:
        _rr = next((x for x in _usable if x["mode_key"] == _mk), None)
        if _rr is None:
            continue
        for _dst, _arr in zip((_cw, _cd, _cs), _binned_series(_rr, _mock)):
            _dst.append(_arr)
    if not _cw:
        continue
    _cw, _cd, _cs = (np.concatenate(_cw), np.concatenate(_cd),
                     np.concatenate(_cs))
    _o = np.argsort(_cw)
    _sum_points.append(dict(
        label=f"{_cname} (combined)", marker="d",
        color=_COMBO_COLORS[_ci % len(_COMBO_COLORS)],
        wl_um=_cw[_o], depth_ppm=_cd[_o], sigma_ppm=_cs[_o]))

_sum_spectrum = dict(wl_um=wl_s, depth_ppm=d_plot,
                     depth_label=_depth_lbl,
                     model_label="model",
                     legend_title=_leg_note,
                     wl_range=_wl_range,
                     depth_range=_depth_range,
                     x_log=_x_log, y_log=_y_log,
                     points=_sum_points)
if d_wo_s is not None:
    # detect goal: the same without-target comparison curve the old
    # standalone spectrum carried (smoothed identically for display)
    _sum_spectrum["depth2_ppm"] = _display_smooth(d_wo_s)
    _sum_spectrum["depth2_label"] = f"No {meta['target']}"

_sum_foot = None


def _compose(spec):
    # No in-figure title: the section header above already names the figure;
    # the exported PNG/PDF carry the planet name in their FILENAME.
    return summary_figure.compose_summary_figure(
        spec, posterior_panels=_post_panels or None,
        footnote=_sum_foot, panel_xlims=_post_xlims)


try:
    fig_sum = _compose(_sum_spectrum)
except ValueError as _e:
    # A log DEPTH axis is refused when the visible range reaches zero or below
    # (eclipse depths and low-S/N jitter draws can). That is a widget choice,
    # not a broken run: say why, fall back to linear, and keep the page alive.
    # Any other ValueError is a real defect and must not be swallowed.
    if "y_log=True" not in str(_e):
        raise
    _fig_box.warning("Log axis unavailable; linear shown.")
    _sum_spectrum["y_log"] = False
    fig_sum = _compose(_sum_spectrum)
_sum_png = _fig_bytes(fig_sum, "png")
_sum_pdf = _fig_bytes(fig_sum, "pdf")
with _fig_box:
    _show_fig(fig_sum)
_s1, _s2, _s3, _s4, _s5 = _fig_box.columns([1.5, 1.2, 1.5, 1.5, 1.9])
_s1.download_button("Figure (PDF, vector)", _sum_pdf,
                    f"{_fname_base}_proposal_summary.pdf",
                    "application/pdf", key=K("dl_summary_pdf"),
                    on_click="ignore")
_s2.download_button("Figure (PNG)", _sum_png,
                    f"{_fname_base}_proposal_summary.png", "image/png",
                    key=K("dl_summary_png"), on_click="ignore")
_s3.download_button("Binned points (CSV)", _csv_bytes(_bin_df),
                    f"{_fname_base}_binned_points.csv", "text/csv",
                    key=K("dl_spec_bins"), on_click="ignore")
_s4.download_button("Native model (CSV)", _csv_bytes(pd.DataFrame(_native)),
                    f"{_fname_base}_model_spectrum.csv", "text/csv",
                    key=K("dl_spec_native"), on_click="ignore")
if _mock is not None:
    # the mock data is downloadable, but ONLY under a name that says what it
    # is (a seeded mock realization) -- the result CSVs above stay noiseless
    _mock_rows = []
    for r in results:
        # wl_eff is the PLOTTED x coordinate: the figure must be reproducible
        # from this file alone
        _mwl, _md, _ms = _binned_series(r, _mock)
        _mock_rows.append(pd.DataFrame({
            "mode": r["mode_key"], "label": r["label"],
            "wl_um": np.asarray(r["wl"], dtype=float),
            "wl_eff_um": _mwl, "depth_mock_ppm": _md, "sigma_ppm": _ms,
            "seed": int(seed),
            "disclosure": _mock["label"],
            "seed_scheme": _mock["seed_scheme"],
            "numpy_version": _mock["numpy_version"]}))
    _mock_df = pd.concat(_mock_rows, ignore_index=True)
    _s5.download_button(
        "Mock observation (CSV)", _csv_bytes(_mock_df),
        f"{_fname_base}_mock_realization_seed{int(seed)}.csv", "text/csv",
        key=K("dl_spec_mock"), on_click="ignore",
        help=posteriors.MOCK_SHORT_LABEL)
