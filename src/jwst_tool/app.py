"""JWST exoplanet observation planner -- Streamlit GUI.

Launch via the console script ``jwst-tool`` (installed with vulcan-jwst-tool), or
directly:  streamlit run src/jwst_tool/app.py  (from the repo root).

Pipeline per run: VULCAN-JAX photochemistry (or PICASO equilibrium) -> ExoJax
transmission/emission spectrum (local subprocess, disk-cached; ~1.5-2 min at the
default 100-layer resolution) -> Pandeia ETC noise per instrument mode
(subprocess in its own conda env, disk-cached) -> science-goal scoring per mode.
Two goal types: DETECT a molecule (conditional template S/N) or CONSTRAIN a
parameter (Fisher forecast from consistency-checked Jacobians, vs a target
uncertainty). Planets beyond WASP-39b come from the registry in planets.py (or
a fully custom system).

Layout: four numbered sidebar steps (Target, Atmosphere, Science goal,
Observation) plus one "More settings" group; the result page leads with the
verdict and collapses certificates/provenance. Widget KEYS must never change
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
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import streamlit as st

TOOL_DIR = Path(__file__).resolve().parent   # forward.py subprocess lives here

from jwst_tool import adjoint_diag, binning, datacheck, detect, \
    fisher as fisher_mod, forward
from jwst_tool import noise as noise_mod
from jwst_tool import proc as proc_mod
from jwst_tool import share_config
from jwst_tool import instruments as ins
from jwst_tool import planets
from jwst_tool import runlimit
from jwst_tool import picaso_chem

# House figure style: the vendored science.mplstyle, flipped to serif + STIX
# math. White faces are pinned so downloaded figures stay white on any
# Streamlit theme. Data colors/markers stay the fixed per-mode palette in
# instruments.MODE_COLOR / MODE_MARKER (no series relies on color alone).
plt.style.use(str(TOOL_DIR / "science.mplstyle"))
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_").lower()


def _fig_png(fig, dpi: int = 200) -> bytes:
    """Rasterize a figure for download (PNG, dpi 200 -- house convention)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    return buf.getvalue()


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _not_reached_reason(scenario: str, val: str, floored: bool,
                        evw: str) -> str:
    """Honest phrasing for a target the transit scan did not reach.

    With a floor, sig_inf is an exact ceiling under the random scenario and
    only the N-to-infinity limit under a correlated one. With NO floor
    nothing caps the score: say the scan reached its limit, never "never"."""
    if not floored:
        return (f"not reached within the scan limit of "
                f"{detect.N_TRANSITS_CAP} {evw}s; with no noise floor set, a "
                "larger count could still reach the target")
    if scenario == "random":
        return f"the noise floor caps the score at {val}"
    return (f"no count up to {detect.N_TRANSITS_CAP} reaches it; "
            f"the N-to-infinity limit is {val}")


def _has_floor(r: dict) -> bool:
    """Does this evaluated mode carry a noise floor anywhere? (floor_spec=None
    gives an all-zero array, and then no N-to-infinity limit exists.)"""
    return bool(np.any(np.asarray(r["floor"]) > 0.0))


def _mode_cfg(mode_key: str) -> str:
    """The exact fixed detector configuration (subarray/readout) a registry
    mode evaluates -- shown next to headline labels so a "mode" is never
    mistaken for the whole instrument mode (2026-08-05 review)."""
    det = ins.MODES[mode_key]["config"].get("detector", {})
    return f"{det.get('subarray', '?')}/{det.get('readout_pattern', '?')}"


def _op_status(r: dict) -> str:
    """Honest operational-status cell for the mode table: this tool checks
    saturation and relays Pandeia warnings, and checks nothing else -- every
    usable row still needs APT verification."""
    if r["saturated"]:
        return "saturated at the shortest ramp tried"
    if r["warnings"]:
        return "warnings (see notes); verify in APT"
    return "verify in APT"


def _transits_cell(tt: dict, scenario: str, val_never: str, floored: bool,
                   evw: str) -> str:
    """'events to target' table cell, window-aware for correlated scenarios.

    "unreachable" is reserved for a floor-proven result; a scan that only ran
    out of events reads ">N (scan limit)"."""
    if not tt["reachable"]:
        if not floored:
            return f">{detect.N_TRANSITS_CAP} (scan limit; no floor set)"
        return (f"unreachable "
                f"({_not_reached_reason(scenario, val_never, floored, evw)})")
    cell = str(tt["n"])
    if (tt.get("n_last") is not None
            and tt["n_last"] < detect.N_TRANSITS_CAP):
        cell += f" (lost again past {tt['n_last']})"
    return cell

st.set_page_config(page_title="JWST Exoplanet Observation Planner",
                   layout="wide")

# ---------------------------------------------------------------------------
# Header: short orientation, no acknowledgment gate
# ---------------------------------------------------------------------------
st.title("JWST exoplanet observation planner")
st.markdown(
    "0. **Configuration**: load a shared configuration file, or start "
    "fresh.\n"
    "1. **Target**: select the system and the observation type.\n"
    "2. **Atmosphere**: select the chemistry engine and the atmosphere "
    "model.\n"
    "3. **Science goal**: detect a molecule, or constrain a parameter.\n"
    "4. **Observation**: select instrument modes and noise assumptions.\n\n"
    "The tool computes a forward "
    "spectrum and a Pandeia noise forecast, ranks the selected modes, and "
    "reports how many transits or eclipses reach your target.")

with st.expander("How the model works, and its limits"):
    st.markdown(
        f"""
Each run computes a spectrum live, for exactly the atmosphere you configure:

1. **Chemistry**: the engine you select computes the abundances.
   **VULCAN** solves steady-state photochemical kinetics (run through its
   JAX port, VULCAN-JAX); **PICASO** reads chemical equilibrium from
   precomputed tables.
2. **Temperature profile**: VULCAN supports a Guillot profile, an uploaded
   table, or a PICASO radiative-convective climate profile. PICASO
   equilibrium chemistry supports a Guillot profile or a PICASO climate
   profile.
3. **Spectrum**: ExoJAX radiative transfer computes the transit depth or
   the dayside eclipse depth against a PHOENIX stellar model.
4. **Noise**: the STScI Pandeia ETC engine
   ({ins.BACKEND_STATUS.split(' /')[0]} on this instance) predicts the
   uncertainty per instrument mode, with a PandExo-style minimum noise
   floor.
5. **Scores**: a conditional template S/N per molecule, Fisher (Cramer-Rao)
   parameter forecasts, and a floor-aware count of the transits or eclipses
   needed to reach your target.

If a calculation does not pass its numerical quality checks, the tool stops
and shows an error instead of returning an uncertified spectrum.

**Read the results as optimistic.** The detection score is conditional on
the selected atmosphere: if a retrieval frees more atmospheric parameters
under the same model and noise assumptions, the detection significance
will usually decrease. The Fisher values are local Cramer-Rao lower bounds
with no priors, not posterior widths: nonlinear responses and degeneracies
can make a posterior wider, and informative priors can make it narrower.
The noise model omits time-correlated systematics (visit-long trends, 1/f
residuals, detrending covariance, stellar heterogeneity), so achieved
precision is usually poorer. Treat mode rankings as more robust than
absolute ppm numbers.
        """)

# ---------------------------------------------------------------------------
# Data availability -- detected live; the display adapts to what is installed
# (missing data still fails loudly at run time; this is the up-front view)
# ---------------------------------------------------------------------------
_STATUS_LABEL = {datacheck.OK: "installed", datacheck.MISSING: "MISSING",
                 datacheck.AUTO: "downloads on first use"}


@st.cache_data(ttl=3600, show_spinner="Checking installed data ...")
def _cached_full_report(_nonce: int, _backend: str, _picaso_root: str):
    # Disk-persisted (the Space entrypoint warms it at boot) on top of the
    # in-process st.cache. The manifest check is sampled, not exhaustive
    # (full pass: `jwst-tool data --deep`); the refresh button deletes the
    # disk cache and rebuilds.
    cached = datacheck.load_cached_report()
    if cached is not None:
        return cached
    return datacheck.warm_report_cache(base_mols=forward.MOLECULES,
                                       extra_mols=forward.EXTRA_MOLECULES)


def _bump_data_nonce():
    try:
        datacheck.REPORT_CACHE_FILE.unlink()
    except OSError:
        pass
    st.session_state["data_report_nonce"] = (
        st.session_state.get("data_report_nonce", 0) + 1)


_data_report = _cached_full_report(
    st.session_state.get("data_report_nonce", 0),
    ins.JWST_TOOL_BACKEND,
    os.environ.get("JWST_TOOL_PICASO_REFDATA", ""))
_missing_req = datacheck.missing_required(_data_report)
if _missing_req:
    st.error(
        f"**Missing required data ({len(_missing_req)} item"
        f"{'s' if len(_missing_req) > 1 else ''}).** The affected steps will "
        "stop with an error until it is installed (nothing degrades "
        "silently):\n\n"
        + "\n".join(f"- **{it.label}** -- {it.detail}. How to get it: "
                    f"{it.remedy}" for it in _missing_req)
        + "\n\nInstall commands are in the README's *Install* section. "
          "Console commands: `jwst-tool data` (status report) and "
          "`jwst-tool fetch` (download).")
with st.expander("Data status: what this machine has installed"
                 + (f"  ({len(_missing_req)} required item(s) missing)"
                    if _missing_req else "")):
    st.dataframe(
        [{"component": it.label,
          "status": (_STATUS_LABEL[it.status]
                     + ("" if it.required else " (optional)")),
          "detail": it.detail,
          "how to get it": it.remedy if it.status != datacheck.OK else ""}
         for it in datacheck.all_items(_data_report)],
        width="stretch", hide_index=True)
    _c = _data_report["caches"]
    st.caption(
        "\"Downloads on first use\" items are fetched automatically the "
        "first time a run needs them (network required at that moment). "
        f"Generated caches: {_c['model_cache']['n']} model spectra "
        f"({_c['model_cache']['mb']} MB), {_c['noise_cache']['n']} noise "
        f"results ({_c['noise_cache']['mb']} MB). Safe to delete, rebuilt "
        "on demand. Console equivalent: `jwst-tool data` (add `--deep` to "
        "probe the Pandeia env's engine version).")
    st.button("Refresh data status", on_click=_bump_data_nonce,
              key="data_refresh_btn",
              help="The status above is cached for an hour. Refresh after "
                   "installing data.")

# Measured FD/AD Fisher-row wall times (WASP-39b defaults; see forward.py's
# FD_STEPS). Threaded through every GUI mention below so a re-measurement
# updates one place (the slow-run estimator uses the midpoints, 7 and 4).
_FD_COMP_MIN, _FD_COMP_MAX = 6, 8      # minutes per FD composition row
_FD_THETA_MIN, _FD_THETA_MAX = 3, 5    # minutes per FD Kzz/T row
_AD_ROW_MIN, _AD_ROW_MAX = 1, 2        # minutes per AD row

_PROG_RE = re.compile(r"\[fwd\] PROG ([0-9.]+) (.*)")
_ADJ_PROG_RE = re.compile(r"\[adj\] PROG ([0-9.]+) (.*)")


def _fmt_dur(s: float) -> str:
    """Compact duration: '42 s', '3 m 05 s', '1 h 12 m'."""
    s = max(0, int(round(s)))
    if s < 60:
        return f"{s} s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m} m {sec:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} m"


class _TimedBar:
    """st.progress wrapper that appends a live elapsed / time-remaining
    readout to every label.

    The remaining estimate blends the pre-run prior (when given) with the
    measured pace, weighted by the completed fraction; with no prior it is
    purely measured. Blend the two REMAINING-time estimates, never the two
    totals -- blending totals cancels the measured pace and freezes the
    countdown for the whole stage."""

    def __init__(self, prior_total_s: float | None = None,
                 text: str = "starting ..."):
        self._bar = st.progress(0.0, text=text)
        self._t0 = time.monotonic()
        self._prior = prior_total_s
        self._frac = 0.0
        self._label = text

    def _render(self) -> None:
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
        txt = f"{self._label}  (elapsed {_fmt_dur(e)}"
        txt += (f", about {_fmt_dur(remaining)} left)"
                if remaining is not None else ")")
        self._bar.progress(min(1.0, self._frac), text=txt)

    def update(self, frac: float, label: str) -> None:
        self._frac, self._label = float(frac), label
        self._render()

    def tick(self) -> None:
        """Refresh the clock without new progress information."""
        self._render()

    def done(self, label: str = "done") -> None:
        self._bar.progress(
            1.0, text=f"{label}  ({_fmt_dur(time.monotonic() - self._t0)})")


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


# --- downloads that must not cancel a run ----------------------------------
# Clicking a download button queues a rerun, and a queued rerun CANCELS the
# script run in flight -- on the Run pass that is the forward model itself.
# So every download rendered before `run_clicked` is known reserves its slot
# with `_deferred_download` and is filled by `_render_deferred_downloads`:
# dead for the duration of a run, live again the moment compute() returns.
_DEFERRED_DOWNLOADS: list = []


def _deferred_download(label: str, data, file_name: str, mime=None, *,
                       key: str, help: str | None = None) -> None:
    """Reserve this position for a download button. Call it where the button
    belongs; `_render_deferred_downloads` decides live or dead."""
    _DEFERRED_DOWNLOADS.append(
        (st.empty(), label, data, file_name, mime, key, help))


def _render_deferred_downloads(busy: bool = False) -> None:
    """Fill every reserved slot. ``busy`` renders dead stand-ins that say why,
    for the duration of a run."""
    for slot, label, data, file_name, mime, key, help_ in _DEFERRED_DOWNLOADS:
        if busy:
            slot.button(
                label, disabled=True, key=f"{key}_busy",
                help="Available again as soon as the run finishes. "
                     "Downloading is a page interaction, and a page "
                     "interaction during a run cancels the run.")
        else:
            slot.download_button(label, data, file_name, mime, key=key,
                                 help=help_)


# default target precision per parameter (DISPLAY units: dex / K / absolute C/O)
_TARGET_DEFAULT = {"lnZ": 0.10, "dlnCO": 0.05, "lnKzz": 0.30,
                   "Tirr": 50.0, "Tint": 50.0,
                   "Tint_cl": 50.0,
                   "log_kappa": 0.30, "log_gamma": 0.30,
                   "log_kappa_cloud": 0.30, "alpha_cloud": 0.50,
                   "mie_log_rg": 0.30, "mie_sigmag": 0.20,
                   "mie_log_mmr": 0.50}
# Every freeable Fisher parameter can be the constraint goal and looks up
# _TARGET_DEFAULT[goal_param]; guard for missing entries at import, not at
# click time.
_FREEABLE = (set(forward.CHEM_PARAM_NAMES) | set(forward.CLOUD_FISHER_PARAMS)
             | set(forward.MIE_FISHER_PARAMS)
             | {p for ns in forward.TP_PARAM_NAMES.values() for p in ns})
_missing_target = _FREEABLE - set(_TARGET_DEFAULT)
if _missing_target:
    raise RuntimeError(f"_TARGET_DEFAULT is missing {sorted(_missing_target)}: "
                       "every freeable Fisher parameter needs a target default.")


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
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
    for k, v in state.items():
        st.session_state[k] = v
    st.session_state["_cfg_load_notes"] = notes


_apply_pending_config()

# Chemistry-engine choice, read EARLY from session state so widgets that
# render before the engine selectbox can adapt; on the very first render it
# is the default (VULCAN).
_pic_hint = st.session_state.get(K("provider"), "vulcan") == "picaso"

with st.sidebar:
    # -----------------------------------------------------------------------
    # Step 0: Load a configuration. The file was already APPLIED by
    # _apply_pending_config(); this is the widget plus the outcome messages.
    # -----------------------------------------------------------------------
    st.markdown("### 0 · Configuration")
    _cfg_up = st.file_uploader(
        "Load a configuration (JSON)", type=["json"], key=K("cfg_upload"))
    if st.session_state.get("_cfg_load_error"):
        st.error("The configuration file could not be applied: "
                 + st.session_state["_cfg_load_error"])
    elif _cfg_up is not None:
        st.success("Configuration loaded. Review the steps below and "
                   "press Run.")
        for _n in st.session_state.get("_cfg_load_notes") or []:
            st.warning(f"Not restored: {_n}")
    else:
        st.caption("Optional: start from a configuration file downloaded "
                   "from this tool (Run summary & configuration), or set "
                   "everything below yourself.")
    st.divider()

    # -----------------------------------------------------------------------
    # Step 1: Target
    # -----------------------------------------------------------------------
    st.markdown("### 1 · Target")
    planet_key = st.selectbox(
        "Planet", list(planets.PLANETS) + ["custom"], key=K("planet"),
        format_func=lambda k: planets.PLANETS[k]["label"] if k in planets.PLANETS
        else "Custom planet …")
    pdef = planets.PLANETS.get(planet_key, planets.CUSTOM_DEFAULTS)
    st.caption(pdef["note"] if planet_key in planets.PLANETS
               else "Starts from WASP-39 b values. Edit everything below.")
    science_mode = st.radio(
        "Observation type", ["transmission", "emission"], horizontal=True,
        key=K("scimode"),
        format_func={"transmission": "Transmission (transit)",
                     "emission": "Emission (secondary eclipse)"}.get)
    # Event word for observation-facing labels: only the vocabulary changes
    # with the observation type, not the calculation.
    _evw = "eclipse" if science_mode == "emission" else "transit"

    def _k(name: str) -> str:            # per-planet widget state
        return K(f"{planet_key}_{name}")

    with st.expander("System parameters", expanded=(planet_key == "custom")):
        teff = st.number_input(
            "Stellar effective temperature, Teff (K)", 3000.0, 7000.0,
            pdef["star"]["teff"], 50.0, key=_k("teff"))
        logg = st.number_input(
            "Stellar surface gravity, log10(g) (g in cm s^-2)", 3.5, 5.5,
            pdef["star"]["log_g"], 0.1, key=_k("logg"))
        feh = st.number_input("Stellar metallicity, [Fe/H] (dex)", -2.0, 0.5,
                              pdef["star"]["metallicity"], 0.1, key=_k("feh"))
        ks_mag = st.number_input(
            "2MASS Ks magnitude", 4.0, 16.0,
            pdef["star"]["ks_mag"], 0.1, key=_k("ks"))
        rstar = st.number_input(
            "Stellar radius (solar radii)", 0.2, 3.0, pdef["rstar_rsun"],
            0.01, key=_k("rstar"), format="%.3f")
        rp = st.number_input("Planet radius (Jupiter radii)", 0.1, 2.5,
                             pdef["rp_rjup"], 0.01,
                             key=_k("rp"), format="%.3f")
        g_ms2 = st.number_input(
            "Planet surface gravity (m s^-2)", 1.0, 100.0,
            pdef["gs_cgs"] / 100.0, 0.5, key=_k("g"))
        orbit_au = st.number_input(
            "Semi-major axis (AU)", 0.005, 1.0,
            pdef["orbit_au"], 0.001, key=_k("a"), format="%.4f")
        t14 = st.number_input(
            "Event duration, T14 (hours)", 0.5, 10.0,
            pdef["t14_hr"], 0.1, key=_k("t14"))
        _uv_ok = datacheck.uv_spectra_status()
        sflux = st.selectbox(
            "Stellar UV spectrum (photochemistry)",
            list(planets.SFLUX_CHOICES),
            index=list(planets.SFLUX_CHOICES).index(pdef["sflux"]),
            format_func=lambda f: (
                planets.SFLUX_CHOICES[f]
                + ("" if _uv_ok.get(f) else "  [FILE MISSING]")),
            key=_k("sflux"), disabled=_pic_hint)
        if _pic_hint:
            st.caption("UV spectrum unused: the PICASO engine has no "
                       "photolysis, so this selection has no effect there.")

    # Registry planets carry a literature T_eq; the CUSTOM planet derives it
    # from the entered star and orbit so it does not inherit WASP-39 b's
    # temperature scale.
    if planet_key == "custom":
        teq = planets.system_teq(teff, rstar, orbit_au)
        st.caption(f"T_eq derived from the star and orbit above: "
                   f"{teq:.0f} K (zero albedo, full redistribution). It "
                   "sets the default Guillot T_irr in step 2.")
    else:
        teq = float(pdef["teq_k"])

    # -----------------------------------------------------------------------
    # Step 2: Atmosphere
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("### 2 · Atmosphere")
    chem_provider = st.selectbox(
        "Chemistry engine", ["vulcan", "picaso"], index=0, key=K("provider"),
        format_func={"vulcan": "VULCAN (photochemical kinetics)",
                     "picaso": "PICASO (chemical equilibrium)"}.get)
    _pic = chem_provider == "picaso"
    if _pic:
        st.session_state[K("jacm")] = "fd"   # tables are not differentiable

    with st.expander("Temperature-pressure profile"):
        # canonical_params refuses tp_mode='file' under PICASO; hide it here
        # too (GUI gating is convenience, canonical_params is the hard guard).
        _tp_opts = (["guillot", "picaso_climate"] if _pic
                    else ["guillot", "file", "picaso_climate"])
        # Mirror canonical_params' default so GUI and API defaults agree.
        _tp_default = forward._default_tp_mode(
            {"planet": planet_key, "chem_provider": chem_provider})
        if st.session_state.get(_k("tp")) not in _tp_opts:
            st.session_state[_k("tp")] = _tp_default
        tp_mode = st.selectbox(
            "Temperature-pressure profile", _tp_opts,
            index=_tp_opts.index(_tp_default),
            key=_k("tp"),
            format_func={"guillot": "Guillot (2010)",
                         "file": "Tabulated table (T-P, optional Kzz)",
                         "picaso_climate":
                             "PICASO radiative-convective (climate solve)"}.get)
        tp_kwargs = {}
        tp_file, tp_file_path, tp_file_ok = "", None, True
        if tp_mode == "guillot":
            # one definition shared with canonical_params, so the widget and
            # API defaults cannot drift apart
            tirr0 = planets.default_tirr(
                pdef, system=(dict(star_teff=teff, rstar_rsun=rstar,
                                   orbit_au=orbit_au)
                              if planet_key == "custom" else None))
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
            tp_kwargs["Tirr"] = st.number_input(
                "T_irr (K)", 800.0, 2500.0, tirr0, 20.0, key=_k("tirr"))
            tp_kwargs["Tint"] = st.number_input(
                "T_int (K)", 50.0, 500.0, 100.0, 25.0, key=_k("tint"))
            tp_kwargs["log_kappa"] = st.number_input(
                "log10 kappa_IR (cm^2/g)", -4.0, 0.0, -2.3, 0.1, key=_k("lk"))
            tp_kwargs["log_gamma"] = st.number_input(
                "log10 gamma (kappa_vis/kappa_IR)", -2.0, 0.3, -1.0, 0.05,
                key=_k("lg"))
        elif tp_mode == "picaso_climate":
            tp_kwargs["tint_cl"] = st.number_input(
                "Internal temperature T_int (K)", *forward.TINT_CL_RANGE,
                forward.TINT_CL_DEFAULT, 10.0, key=_k("tintcl"))
            tp_kwargs["rfacv"] = st.selectbox(
                "Day-night heat redistribution",
                list(forward.RFACV_CHOICES), index=1, key=_k("rfacv"),
                format_func={0.0: "0: no irradiation (isolated interior)",
                             0.5: "0.5: full redistribution (default)",
                             1.0: "1: dayside-only"}.get)
            tp_kwargs["tio_vo"] = st.checkbox(
                "Include TiO/VO in climate opacity only", value=False,
                key=_k("tiovo"))
            tp_kwargs["climate_rcb"] = st.number_input(
                "Convective-zone seed (deep layer index)",
                *forward.CLIMATE_RCB_RANGE, forward.CLIMATE_RCB_DEFAULT, 1,
                key=_k("rcb"))
            if science_mode == "emission":
                st.warning(
                    "Emission with the climate solve: some wavelengths "
                    "probe the deep layers set by the convective-zone "
                    "seed. Read deep emission features as conditional on "
                    "that setting.")
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
                    st.caption(
                        "Column 1: pressure in dyne/cm^2 (1 bar = 10^6), "
                        "bottom of the atmosphere first or top first. "
                        "Column 2: temperature in K. Column 3 (optional): "
                        "Kzz in cm^2 s^-1, used when the vertical-mixing "
                        "profile is set to 'Tabulated'. Any number of "
                        "rows. The two header lines are required.")
                    # deferred: clicking a download cancels a run in flight
                    _deferred_download(
                        "Download this example (edit and re-upload)",
                        _tp_example, "example_atm.txt",
                        key=_k("tpex"))
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
                        st.caption(
                            "T-P table restored from the loaded "
                            f"configuration: {_tab_tp['P_dyn'].size} rows, "
                            f"T {_tab_tp['T'].min():.0f}-"
                            f"{_tab_tp['T'].max():.0f} K"
                            + (", Kzz column present"
                               if _tab_tp["Kzz"] is not None else
                               ", no Kzz column")
                            + ". Upload a file to replace it.")
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
                        _dst_tp.write_bytes(_raw_tp)
                    try:                       # loud validation, immediate
                        _tab_tp = forward._read_tp_table(_dst_tp)
                        tp_file_path = str(_dst_tp)
                        st.caption(
                            f"Loaded {_tab_tp['P_dyn'].size} rows, "
                            f"P {_tab_tp['P_dyn'].min()/1e6:.2g}-"
                            f"{_tab_tp['P_dyn'].max()/1e6:.2g} bar, "
                            f"T {_tab_tp['T'].min():.0f}-"
                            f"{_tab_tp['T'].max():.0f} K"
                            + (", Kzz column present"
                               if _tab_tp["Kzz"] is not None else
                               ", no Kzz column"))
                    except ValueError as e:
                        st.error(
                            "The temperature-pressure table is not valid: "
                            f"{e} Edit the file and upload it again.")
                        tp_file_ok = False
                else:
                    st.warning("Upload a table to run in file mode.")
                    tp_file_ok = False
            st.caption(
                "Note: file mode has no temperature constraint row (a "
                "fixed table has no temperature parameter). Constraint "
                "forecasts assume this profile is exactly right, so the "
                "bounds are optimistic.")

    with st.expander("Composition"):
        if tp_mode == "picaso_climate":
            # EXACT-CK-NODE selectors: the climate correlated-k tables have
            # no composition interpolation, so these menus only offer exact
            # nodes (canonical_params is the hard guard).
            from jwst_tool import picaso_chem as pchem
            _feh_opts = [x for x in pchem.FEH_NODES if -1.0 <= x <= 2.0]
            _feh = st.selectbox(
                "Metallicity", _feh_opts,
                index=_feh_opts.index(1.0), key=K("metnode"),
                format_func=lambda x: f"{10.0 ** x:.3g} × solar "
                                      f"([M/H] = {x:+.1f})")
            met = float(10.0 ** _feh)
            _co_opts = [c for c in pchem.CO_NODES
                        if (f"feh{_feh:.1f}_co{c:.2f}"
                            in pchem.CK_NODES_AVAILABLE)]
            co_ratio = st.selectbox(
                "C/O", _co_opts,
                index=_co_opts.index(0.55) if 0.55 in _co_opts else 0,
                key=K(f"conode_{_feh:.1f}"),
                format_func=lambda c: f"{c:.2f}")
        elif _pic:
            met = st.number_input(
                "Metallicity (× solar)", *forward.PICASO_MET_RANGE, 10.0, 0.5,
                format="%.2f", key=K("met_pic"))
            co_ratio = st.number_input(
                "C/O (carbon/oxygen number ratio)",
                *forward.PICASO_CO_RANGE, 0.50, 0.01,
                format="%.3f", key=K("co_pic"))
        else:
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
                0.10, 2.00, float(forward.CO_BASELINE), 0.05,
                format="%.3f", key=K("co"))

    # Kzz and photochemistry render BEFORE the science-goal step, so the AD
    # photo-lock reads the EFFECTIVE differentiation method from session
    # state: the widget value counts only when a Jacobian is actually
    # requested, matching canonical_params' normalization to "fd" otherwise.
    if _pic:
        kzz_mode, kzz_x = "const", 1.0
        kzz_const, kzz_kmax, kzz_plev, kzz_kdeep = 1.0e9, 0.0, 0.0, 0.0
        use_photo, sl_angle_deg, f_diurnal = False, 83.0, 1.0
        use_moldiff = use_vm_mol = False
        st.caption("Vertical mixing and photochemistry apply to the VULCAN "
                   "kinetics engine only, so they are not shown for PICASO "
                   "equilibrium.")
    else:
        _goal_ss = st.session_state.get(K("goal"), "detect")
        _dofish_ss = bool(st.session_state.get(K("dofish"), False))
        _jac_hint = (st.session_state.get(K("jacm"), "fd")
                     if (_goal_ss == "constrain" or _dofish_ss)
                     and not bool(st.session_state.get(K("conden"), False))
                     else "fd")

        with st.expander("Vertical mixing (Kzz)"):
            _kzz_opts = ["const", "Pfunc", "JM16"]
            # tabulated Kzz needs the tabulated T-P table (its Kzz column)
            _kzz_file_ok = tp_mode == "file"
            if _kzz_file_ok:
                _kzz_opts.append("file")
                # Same rule as canonical_params: a table that carries Kzz
                # supplies the mixing profile, so "file" is the default.
                # Seed session_state on first render (Streamlit ignores
                # index= once the key exists).
                if _k("kzzmode") not in st.session_state:
                    st.session_state[_k("kzzmode")] = "file"
            elif st.session_state.get(_k("kzzmode")) == "file":
                st.session_state[_k("kzzmode")] = "const"
            kzz_mode = st.selectbox(
                "Vertical-mixing profile, Kzz", _kzz_opts,
                index=_kzz_opts.index("file") if _kzz_file_ok else 0,
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
            else:
                st.caption("Kzz comes from the Kzz column of the T-P "
                           "table. A table without that column stops with "
                           "an error.")
            if kzz_mode != "const":
                kzz_x = 10.0 ** st.number_input(
                    "Kzz profile multiplier, log10(f)", -1.0, 1.0, 0.0, 0.05,
                    key=_k("kzzx"))

        with st.expander("Photochemistry & transport"):
            if _jac_hint == "ad":
                st.session_state[K("photo")] = True   # AD needs photolysis ON
            use_photo = st.checkbox(
                "Photochemistry (UV photolysis)", value=True, key=K("photo"),
                disabled=(_jac_hint == "ad"))
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

    # Opacity & line lists live in the Atmosphere step so extra_mols is a
    # live variable by step 3; all extras default ON.
    with st.expander("Opacity & line lists (ExoJAX)"):
        _base_set, _extra_set = ((forward.MOLECULES, forward.EXTRA_MOLECULES)
                                 if not _pic else
                                 (picaso_chem.PICASO_MOLECULES,
                                  picaso_chem.PICASO_EXTRA_MOLECULES))
        st.caption(
            "The opacity always includes the base set "
            f"**{' · '.join(_base_set)}**. The extras are "
            f"**{' · '.join(_extra_set)}**, all on by default. Deselect "
            "any to save about 7 s each on a new run."
            + (" No SO2 here: in equilibrium, sulfur sits in H2S and "
               "OCS. Making SO2 needs the VULCAN engine." if _pic else ""))
        # live line-list availability for the CURRENT broadening choice
        # (widget below; via session_state, default "air")
        _mol_status = datacheck.molecule_linelist_status(
            list(_extra_set),
            broadening=st.session_state.get(K("broad"), "air"))
        _MOL_NOTE = {datacheck.OK: "opacity cached",
                     datacheck.AUTO: "downloads on first use",
                     datacheck.MISSING: "engine data missing"}
        # All extras default ON: ~7 s each on a new run; a complete detection
        # report is worth more than a faster first run.
        extra_mols = st.multiselect(
            "Extra opacity molecules", list(_extra_set),
            default=list(_extra_set),
            key=K(f"xmols_{chem_provider}"),
            format_func=lambda m: f"{m}  ({_MOL_NOTE[_mol_status[m]]})")
        # h2he has per-molecule caches; CO is cached ExoMol and ignores the
        # broadening choice
        _h2he_mols = [m for m in forward.MOLECULES + extra_mols if m != "CO"]
        _h2he_cached = sum(
            1 for v in datacheck.molecule_linelist_status(
                _h2he_mols, broadening="h2he").values() if v == datacheck.OK)
        broadening = st.selectbox(
            "Pressure-broadening gas", ["air", "h2he"], index=0,
            key=K("broad"),
            format_func=lambda b: (
                "air (HITRAN terrestrial widths, default)"
                if b == "air" else
                f"H2/He blend (planetary, {_h2he_cached}/{len(_h2he_mols)} "
                "line-list caches present)"))
        nu_pts = st.number_input(
            "Native spectral grid points", *forward.NU_PTS_RANGE,
            forward.NU_PTS_DEFAULT, 500, key=K("nupts"),
            help="Wavelength sampling of the model spectrum across "
                 "1-15 µm, shared by all instrument modes; not an "
                 "instrument setting. Model R is about nu_pts/2.7 "
                 "(4000 = R 1500), well above the analysis binning. Weak "
                 "mid-infrared features still gain at the 8000 cap, so "
                 "quote weak-band MIRI results as lower bounds.")

    with st.expander("Clouds & scattering (ExoJAX)"):
        if science_mode == "emission":
            # canonical_params forces Rayleigh OFF in emission (no scattering
            # channel); show the forced state, not a checked-but-ignored box
            st.session_state[K("rayl")] = False
        use_rayleigh = st.checkbox(
            "H2/He Rayleigh scattering", value=True, key=K("rayl"),
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
        # Mie condensate deck: condensate optics from an exojax miegrid.
        _mie_opts = [""] + list(forward.MIE_CONDENSATES)
        mie_condensate = st.selectbox(
            "Mie condensate cloud", _mie_opts, index=0, key=K("miec"),
            format_func=lambda c: "off" if not c else c)
        mie_log_rg, mie_sigmag, mie_log_mmr = -5.0, 2.0, -6.0
        if mie_condensate:
            if datacheck.miegrid_status([mie_condensate])[mie_condensate] \
                    != datacheck.OK:
                st.warning(
                    f"No Mie grid for {mie_condensate} yet. Generate it once "
                    f"with 'python tools/generate_miegrid.py {mie_condensate}' "
                    "(~1 h), or the run will stop with an error.")
            mie_log_rg = st.number_input(
                "log10 mean radius r_g (cm)", float(forward.MIE_LOG_RG_RANGE[0]),
                float(forward.MIE_LOG_RG_RANGE[1]), -5.0, 0.1, key=K("mierg"))
            mie_sigmag = st.number_input(
                "Size dispersion sigma_g", float(forward.MIE_SIGMAG_RANGE[0]),
                float(forward.MIE_SIGMAG_RANGE[1]), 2.0, 0.05, key=K("miesg"))
            mie_log_mmr = st.number_input(
                "log10 condensate mass mixing ratio",
                float(forward.MIE_LOG_MMR_RANGE[0]),
                float(forward.MIE_LOG_MMR_RANGE[1]), -6.0, 0.25, key=K("miemmr"))

    # -----------------------------------------------------------------------
    # Step 3: Science goal (only the controls the selected goal needs)
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("### 3 · Science goal")
    # Condensation renders LATER (More settings), so read it from
    # session_state; matches the widget except on the very first render.
    _conden_ss = bool(st.session_state.get(K("conden"), False))
    # Under PICASO + climate, dlnCO is NOT offered: climate composition sits
    # exactly on a chemistry-table node with no trustworthy C/O derivative
    # (canonical_params refuses it too).
    if _pic:
        _chem_free = (["lnZ"] if tp_mode == "picaso_climate"
                      else ["lnZ", "dlnCO"])
    else:
        _chem_free = list(forward.CHEM_PARAM_NAMES)
    avail_free = _chem_free + forward.TP_PARAM_NAMES[tp_mode]
    if cloud_on:
        avail_free = avail_free + list(forward.CLOUD_FISHER_PARAMS)
    if mie_condensate:
        avail_free = avail_free + list(forward.MIE_FISHER_PARAMS)
    mol_options = forward.active_molecules(
        {"chem_provider": chem_provider, "extra_mols": extra_mols})

    if _conden_ss:
        st.session_state[K("goal")] = "detect"
    goal = st.radio(
        "Goal", ["detect", "constrain"], horizontal=True, key=K("goal"),
        disabled=_conden_ss,
        format_func={"detect": "Detect a molecule",
                     "constrain": "Constrain a parameter"}.get,
        help="Detect: compare the full spectrum with one that omits the "
             "selected molecule. The score is a conditional template "
             "signal-to-noise ratio, not a retrieval detection. "
             "Constrain: a local Fisher (Cramer-Rao) bound from the "
             "spectrum derivatives, not a posterior uncertainty. "
             "Constraints cost minutes per free parameter.")
    if _conden_ss:
        goal = "detect"
        st.caption(
            "Condensation is on (More settings). It locks the goal to "
            "detection: no derivative through the condensation pin is "
            "valid, so constraints are unavailable.")
    goal_param, target_prec, marginalize = None, None, True
    do_fisher = False
    if goal == "detect":
        # SO2 is the flagship W39b science under VULCAN; the equilibrium
        # provider has no SO2, so its default detection target is H2O
        _mol_default = "SO2" if "SO2" in mol_options else "H2O"
        target_mol = st.selectbox(
            "Molecule to detect", mol_options,
            index=mol_options.index(_mol_default),
            key=K(f"mol_{chem_provider}_" + "_".join(sorted(extra_mols))))
        target_sig = st.number_input(
            "Target significance (σ)", 1.0, 10.0, 3.0, 0.5, key=K("tsig"))
    else:
        target_mol = None
        goal_param = st.selectbox(
            "Parameter to constrain", avail_free,
            key=K(f"gp_{chem_provider}_{tp_mode}_{int(cloud_on)}_"
                  f"{int(bool(mie_condensate))}"),
            format_func=lambda n: forward.PARAM_LABELS[n],
            help="The forecast is a local Fisher (Cramer-Rao) bound from "
                 "the spectrum derivatives. It is not a posterior "
                 "uncertainty.")
        marginalize = st.checkbox(
            "Marginalize over the other parameters", value=True,
            key=K("marg"),
            help="On (default): the other selected parameters and "
                 "calibration terms are free. Off: they are held fixed, "
                 "which gives a narrower, more optimistic bound. Neither "
                 "option is a posterior uncertainty.")
        if not marginalize:
            st.warning(
                "Marginalization is off: every other parameter is held "
                "fixed. Read the bound as a best-case sensitivity, not a "
                "joint-fit forecast.")
        unit = forward.PARAM_UNITS[goal_param]
        # label uses the unit when there is one (dex / K), else the bare
        # symbol -- C/O is a dimensionless number ratio
        _tgt_lbl = (f"Target uncertainty (±{unit})" if unit else
                    f"Target uncertainty (±{forward.PARAM_SYMBOLS[goal_param]})")
        if unit == "K":
            target_prec = st.number_input(_tgt_lbl, 5.0, 500.0,
                                          _TARGET_DEFAULT[goal_param], 5.0,
                                          key=K(f"tgt_{goal_param}"))
        else:
            target_prec = st.number_input(_tgt_lbl, 0.01, 3.0,
                                          _TARGET_DEFAULT[goal_param], 0.01,
                                          key=K(f"tgt_{goal_param}"))
        target_sig = st.number_input(
            "Report bounds at significance (σ)", 1.0, 10.0, 3.0, 0.5,
            key=K("tsig"),
            help="Scales the reported bounds: 3 means the tables and the "
                 "verdict quote 3σ half-widths.")

    # Constraint settings render only when the run will compute derivatives.
    fisher_params: list = []
    jac_method = "fd"
    if _conden_ss:
        pass                       # constraints unavailable; caption above
    elif goal == "detect":
        do_fisher = st.checkbox(
            "Also calculate parameter constraints", value=False,
            key=K("dofish"),
            help="Adds a Fisher parameter forecast to the detection run. "
                 "Each free parameter costs extra solves (see the cost "
                 "note in Constraint settings).")
    if goal == "constrain" or do_fisher:
        with st.expander("Constraint settings", expanded=(goal == "constrain")):
            if goal == "constrain" and marginalize:
                # Defaults FILTERED by the live menu, key carries the
                # provider: Streamlit hard-raises on a default outside the
                # options (e.g. lnKzz under picaso).
                fisher_extra = st.multiselect(
                    "Jointly free parameters", avail_free,
                    default=[p for p in ("lnZ", "dlnCO", "lnKzz")
                             if p in avail_free],
                    key=K(f"fx_{chem_provider}_{tp_mode}_{int(cloud_on)}_"
                          f"{int(bool(mie_condensate))}"),
                    format_func=lambda n: forward.PARAM_LABELS[n],
                    help="The goal parameter is always included. More free "
                         "parameters give a wider, more honest forecast. "
                         "Each adds minutes.")
                fisher_params = sorted(set(fisher_extra) | {goal_param})
            elif goal == "constrain":
                st.caption(
                    "Marginalization is off (above): only the goal "
                    "parameter is free. This is the optimistic "
                    "conditional bound.")
                fisher_params = [goal_param]
            else:
                fisher_params = st.multiselect(
                    "Free parameters", avail_free,
                    key=K(f"fp_{chem_provider}_{tp_mode}_{int(cloud_on)}_"
                          f"{int(bool(mie_condensate))}"),
                    default=[p for p in ("lnZ", "dlnCO", "lnKzz")
                             if p in avail_free],
                    format_func=lambda n: forward.PARAM_LABELS[n])
            jac_method = st.selectbox(
                "Differentiation method", ["fd", "ad"], index=0,
                key=K("jacm"), disabled=_pic,
                format_func={"fd": "Finite differences (default)",
                             "ad": "Automatic differentiation (forward-mode)"}.get,
                help="Finite differences re-run the model at shifted "
                     "parameter values and check two step sizes against "
                     "each other. They work everywhere. AD differentiates "
                     "the numerical model directly. It is about 1.7-4x "
                     "faster per row, needs photochemistry on, and stops "
                     "with an error on the C/O row for carbon-rich "
                     "compositions. Every result row is labeled with the "
                     "method that produced it.")
            if _pic:
                st.caption("Locked to finite differences: PICASO's "
                           "chemistry is table interpolation, with no "
                           "solver for AD to differentiate.")
            st.caption(
                ("Cost per row under PICASO: well under a minute. A "
                 "climate T_int row re-runs the climate and takes "
                 "several minutes.") if _pic else
                (f"Cost per row: finite differences take about "
                 f"{_FD_COMP_MIN}-{_FD_COMP_MAX} min for a composition "
                 f"parameter and {_FD_THETA_MIN}-{_FD_THETA_MAX} min for "
                 f"Kzz or temperature. AD takes about "
                 f"{_AD_ROW_MIN}-{_AD_ROW_MAX} min per row. Cloud and "
                 "Mie rows take seconds."))
            # Loud slow-path flag: FD re-solves the chemistry per row, so
            # point the user at AD before a multi-hour run (VULCAN only).
            if fisher_params and jac_method == "fd" and not _pic:
                _n_comp = sum(p in ("lnZ", "dlnCO") for p in fisher_params)
                _n_theta = sum(p in ("lnKzz", "Tirr", "Tint", "log_kappa",
                                     "log_gamma") for p in fisher_params)
                _est_min = _n_comp * 7 + _n_theta * 4
                if _est_min >= 20:
                    st.warning(
                        f"Finite differences with {len(fisher_params)} free "
                        f"parameters is slow: roughly {_est_min}-"
                        f"{int(_est_min * 1.4)} min. Switch the "
                        "differentiation method to AD "
                        f"(~{_AD_ROW_MIN}-{_AD_ROW_MAX} min per row, "
                        "photochemistry locked on) or free fewer "
                        "parameters.")

    # -----------------------------------------------------------------------
    # Step 4: Observation
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("### 4 · Observation")
    mode_keys = st.multiselect(
        "Instrument modes",
        options=list(ins.MODES),
        default=ins.DEFAULT_MODES, key=K("modes"),
        help="The noise engine computes every mode once per star, so adding "
             "modes later is instant. Each mode is one fixed detector "
             "configuration (subarray and readout pattern, listed in the "
             "mode details table); the tool does not search alternative "
             "subarrays.",
        format_func=lambda k: (f"{ins.MODES[k]['label']}  "
                               f"({ins.MODES[k]['wl_min']:g}-"
                               f"{ins.MODES[k]['wl_max']:g} µm)"))
    n_transits = st.number_input(f"Number of {_evw}s", 1, 10, 1, 1,
                                 key=K("ntr"))

    with st.expander("Timing, saturation & binning (Pandeia)"):
        t_base = st.number_input(
            f"Out-of-{_evw} baseline (hours)", 0.5, 10.0,
            float(t14), 0.1, key=_k("tbase"),
            help="Sets how well the stellar flux is anchored. The PandExo "
                 "convention is a baseline equal to T14.")
        sat_limit = st.number_input(
            "Saturation limit (full-well fraction)",
            0.5, 0.95, 0.80, 0.05, key=K("sat"),
            help="Group selection keeps the brightest pixel below this "
                 "full-well fraction.")
        r_bin = st.number_input(
            "Analysis resolving power, R", 25, 500, 100, 25, key=K("rbin"),
            help="Width of the final analysis bins, shared by all modes "
                 "(R = 100 is the standard planning convention). This is "
                 "not the instrument's resolving power: the model is "
                 "separately blurred with a Gaussian approximation of each "
                 "mode's tabulated native resolution R(λ). "
                 "Not display-only: one binning operator at this R computes "
                 "the noise, the scores, the forecasts, and the plotted "
                 "spectrum. 50-200 is the usual range.")

    with st.expander("Noise model (Pandeia)"):
        st.markdown("**Minimum noise floor** (PandExo convention)")
        st.caption("A hard minimum: σ_final = max(σ_random, floor) on the "
                   "final bins. It is never added in quadrature and never "
                   f"averages down with more {_evw}s.")
        # NO PRESELECTION (index=None): a 15-40 ppm floor dominates the
        # reported precision and a zero floor claims undemonstrated precision
        # -- neither is neutral, so the user must own the choice.
        floor_mode = st.radio(
            "Floor type", ["constant", "none", "file"], horizontal=True,
            index=None, key=K("floormode"),
            format_func={"constant": "Constant (ppm)", "none": "No floor",
                         "file": "Wavelength table"}.get)
        floor_table = None
        if floor_mode is None:
            st.info(
                "Choose a floor type to continue. The choice changes the "
                "reported precision and is saved with the run.")
        floors = {k: None for k in mode_keys}
        if floor_mode == "constant":
            floors = {k: st.number_input(
                f"{ins.MODES[k]['label']} minimum floor (ppm)", 0.0, 200.0,
                ins.MODES[k]["floor_ppm_suggested"], 5.0, key=K(f"floor_{k}"))
                for k in mode_keys}
            st.caption(
                "Prefilled with per-mode planning values (15-40 ppm), "
                "informed by the Greene+2016 convention but not identical "
                "to it. These are illustrative, not in-flight calibrations. "
                "Edit them for your program.")
        elif floor_mode == "file":
            up = st.file_uploader(
                "Two columns: wavelength (µm), floor (ppm)",
                type=["txt", "csv", "dat"], key=K("floorfile"),
                help="Whitespace- or comma-separated. Linearly "
                     "interpolated to the final bin wavelengths. Endpoint "
                     "values extend beyond the supplied range. Applied to "
                     "every selected mode.")
            if up is not None:
                try:
                    raw = up.getvalue().decode()
                    delim = "," if "," in raw.splitlines()[0] else None
                    floor_table = np.loadtxt(raw.splitlines(), delimiter=delim,
                                             ndmin=2)
                    noise_mod.resolve_floor(np.array([1.0]),
                                            floor_table)  # validate loudly now
                    st.caption(f"Loaded {floor_table.shape[0]} rows, "
                               f"{floor_table[:, 0].min():g}-"
                               f"{floor_table[:, 0].max():g} µm, "
                               f"{floor_table[:, 1].min():g}-"
                               f"{floor_table[:, 1].max():g} ppm.")
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
                    st.caption(
                        "Floor table restored from the loaded configuration "
                        f"({floor_table.shape[0]} rows). Upload a file to "
                        "replace it.")
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

        st.markdown("**Random-noise multiplier** (× Pandeia σ, optional)")
        st.caption("Default **1.0** uses the Pandeia prediction as-is. "
                   "Published achieved-vs-predicted ratios run about "
                   "1.05-1.2x and are reference points, not a calibration. "
                   "Unlike the floor, this noise averages down with more "
                   f"{_evw}s.")
        infl = {k: st.number_input(
            f"{ins.MODES[k]['label']} random-noise multiplier", 1.0, 3.0,
            float(ins.MODES[k].get("noise_infl", 1.0)),
            0.05, key=K(f"infl_{k}"))
            for k in mode_keys}

        st.markdown("**Experimental: correlated-floor scenarios**")
        scenario = st.radio(
            "Systematics scenario", list(noise_mod.SCENARIOS),
            format_func=lambda s: noise_mod.SCENARIOS[s]["label"],
            key=K("scenario"), label_visibility="collapsed")
        st.caption("The default (**random**) is the exact diagonal noise "
                   "model. The correlated presets move the floor excess "
                   "into a spectrally smooth kernel with the same per-bin "
                   "totals. They are stated assumptions, **not calibrated "
                   "JWST systematics models**. Use them to stress-test "
                   "how rankings move.")

    # -----------------------------------------------------------------------
    # More settings: solver grid, condensation, boundary conditions,
    # advanced RT, and display controls, behind one entry point.
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("### More settings")

    with st.expander("Solver & vertical grid"):
        # The chemistry AND the ExoJAX RT share this one grid (art_nlayer is
        # LOCKED to nz); the PICASO climate solve has its own internal grid.
        nz = st.number_input(
            "Vertical layers (chemistry + RT)", *forward.NZ_RANGE,
            forward.NZ_DEFAULT, 10, key=K("nz_pic" if _pic else "nz"),
            help="Pressure levels shared by the chemistry and the "
                 "radiative transfer. More layers resolve steeper "
                 "gradients but run slower. The PICASO climate solve uses "
                 "its own internal grid and is re-gridded onto these "
                 "layers.")
        if _pic:
            yconv_cri = forward.YCONV_DEFAULT   # equilibrium: no iterative solver
            st.caption("The solver convergence tolerance applies to the "
                       "VULCAN engine only (PICASO equilibrium has no "
                       "iterative solver).")
        else:
            yconv_cri = st.number_input(
                "Solver convergence tolerance", 1.0e-4, 1.0e-2,
                forward.YCONV_DEFAULT, 1.0e-4,
                format="%.1e", key=K("yconv"),
                help="Steady-state criterion (yconv). 1e-2 is the VULCAN "
                     "default. Tightening it does not improve weak mid-IR "
                     "results: raise the native spectral grid points "
                     "instead. The results page reports the actual "
                     "residual, and an uncertified run stops with an "
                     "error.")

    if _pic:
        # Equilibrium provider: condensation and boundary conditions do not
        # exist; canonical_params refuses explicit requests, the GUI never
        # offers them.
        use_condense = use_settling = False
        diff_esc, top_flux, bot_flux = [], [], []
        st.caption("Condensation and boundary conditions apply to the "
                   "VULCAN kinetics engine only, so they are not shown for "
                   "PICASO equilibrium.")
    else:
        with st.expander("Condensation (detection goals only)"):
            _conden_allowed = use_photo and use_moldiff and jac_method == "fd"
            if not _conden_allowed:
                st.session_state[K("conden")] = False
            use_condense = st.checkbox(
                "S8 condensation (sulfur rainout)", value=False,
                key=K("conden"), disabled=not _conden_allowed,
                help="Sulfur rainout with the standard VULCAN treatment. "
                     "Detection goals only: the pinned sulfur reservoir "
                     "depends on the solver's step history, so no "
                     "derivative through it is trustworthy. Warning: on a "
                     "column too hot to condense, the pin still freezes "
                     "sulfur at an arbitrary early value. Use it only for "
                     "planets cool enough for sulfur to condense.")
            if not _conden_allowed:
                st.caption(
                    "Condensation needs photochemistry ON, molecular "
                    "diffusion ON, and the finite-difference method in "
                    "the science-goal step.")
            use_condense = bool(use_condense and _conden_allowed)
            st.caption(
                "For aerosol opacity in a constraint forecast, use the "
                "cloud decks in step 2 instead.")

        with st.expander("Boundary conditions & escape"):
            st.caption(
                "Upstream VULCAN boundary-condition options, all off by "
                "default. Negligible for a typical hot Jupiter.")
            _settle_ok = use_moldiff and not use_condense
            if not _settle_ok:
                st.session_state[K("settle")] = False
            use_settling = st.checkbox(
                "Gravitational settling", value=False, key=K("settle"),
                disabled=not _settle_ok,
                help="Adds the particle settling velocity to the "
                     "transport operator. Needs molecular diffusion ON. "
                     "Not available with condensation.")
            if not _settle_ok:
                st.caption("Settling needs molecular diffusion ON and "
                           "condensation OFF.")
            if not use_moldiff:            # escape flux ~ TOA Dzz; zero without moldiff
                st.session_state[K("descape")] = []
            diff_esc = st.multiselect(
                "Diffusion-limited escape at the top of atmosphere",
                list(forward.DIFF_ESC_CHOICES), default=[], key=K("descape"),
                disabled=not use_moldiff,
                help="Applies the diffusion-limited escape flux at the "
                     "top of the atmosphere for the selected species. "
                     "Needs molecular diffusion ON.")
            if not use_moldiff:
                st.caption("Escape needs molecular diffusion ON.")
            top_lines = st.text_area(
                "Top-boundary fluxes", value="", key=K("topflux"),
                placeholder="H2O 1.0e8",
                help="One species per line: 'SPECIES FLUX', flux in "
                     "molecules cm^-2 s^-1 (negative = outflux to space). "
                     "An unknown species stops the run with an error.")
            bot_lines = st.text_area(
                "Bottom-boundary fluxes + deposition", value="",
                key=K("botflux"), placeholder="SO2 1.0e9 0.1",
                help="One species per line: 'SPECIES FLUX VDEP' (VDEP "
                     "optional, default 0), flux in molecules cm^-2 s^-1 "
                     "(positive = outgassing), deposition velocity in "
                     "cm s^-1 (surface sink).")

            def _parse_bc_lines(text: str, kind: str) -> list:
                rows = []
                for ln in (text or "").splitlines():
                    tok = ln.split()
                    if not tok or tok[0].startswith("#"):
                        continue
                    if kind == "bot" and len(tok) == 2:
                        tok = tok + ["0.0"]
                    rows.append(tok)
                return rows

            top_flux = _parse_bc_lines(top_lines, "top")
            bot_flux = _parse_bc_lines(bot_lines, "bot")
            try:                              # immediate loud feedback on typos
                forward._canon_bc_entries(top_flux, kind="top")
                forward._canon_bc_entries(bot_flux, kind="bot")
            except ValueError as e:
                st.error(
                    "The boundary-condition entry is not valid: "
                    f"{e} Edit the line and try again.")

    with st.expander("Advanced radiative transfer (ExoJAX)"):
        st.caption("ExoJAX modeling choices that can move the spectrum. "
                   "The defaults are the validated baseline. Changing one "
                   "re-runs the model.")
        rt_ptop_bar = st.number_input(
            "RT top pressure (bar)",
            1.0e-9, 1.0e-6, 1.0e-8, 1.0e-9,
            format="%.1e", key=K("rtptop"),
            help="Where the RT column ends. Above the chemistry grid, the "
                 "topmost abundances and temperature are held constant, a "
                 "standard convention. Too low a top saturates strong "
                 "bands (CO2 4.3, CO 4.7 µm).")
        if science_mode == "emission":
            # canonical_params pins simpson in emission (no transit chord);
            # show the pinned state, not a silently ignored choice
            st.session_state[K("rtint")] = "simpson"
        rt_integration = st.selectbox(
            "Transit chord integration", ["simpson", "trapezoid"], index=0,
            key=K("rtint"), disabled=(science_mode == "emission"),
            format_func={"simpson": "Simpson (ExoJAX default)",
                         "trapezoid": "Trapezoid"}.get,
            help="Numerical scheme for the transit chord integral. If "
                 "switching it moves your answer, raise the layer count. "
                 "Locked in emission (no transit chord there).")
        rt_dit_res = st.number_input(
            "Line-wing (broadening) grid resolution",
            0.1, 1.0, 1.0, 0.1, format="%.1f", key=K("rtdit"),
            help="PreMODIT broadening-grid spacing. Smaller resolves line "
                 "wings finer but builds opacity slower. 1.0 is the "
                 "validated value here. 0.2 is ExoJAX's own default.")

    with st.expander("Display & reproducibility"):
        show_noise = st.checkbox("Show simulated noise realization",
                                 value=False, key=K("shownoise"))
        seed = st.number_input("Random-noise seed", 0, 9999, 0, key=K("seed"))

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
        st.button("Reset all settings", on_click=_arm_reset,
                  help="Returns every setting to its default and clears "
                       "the current results (asks for confirmation first).")

params = dict(planet=planet_key, science_mode=science_mode,
              chem_provider=chem_provider,
              star_teff=teff, star_logg=logg, star_feh=feh,
              nz=nz, nu_pts=nu_pts, yconv_cri=yconv_cri,
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
              use_rayleigh=use_rayleigh, broadening=broadening,
              rt_ptop_bar=float(rt_ptop_bar), rt_integration=rt_integration,
              rt_dit_res=float(rt_dit_res),
              cloud_on=cloud_on,
              log_kappa_cloud=log_kappa_cloud, alpha_cloud=alpha_cloud,
              mie_condensate=mie_condensate, mie_log_rg=mie_log_rg,
              mie_sigmag=mie_sigmag, mie_log_mmr=mie_log_mmr,
              extra_mols=extra_mols, **tp_kwargs)
star = dict(teff=teff, log_g=logg, metallicity=feh, ks_mag=ks_mag)
planet_label = (planets.PLANETS[planet_key]["label"]
                if planet_key in planets.PLANETS else "custom planet")

_canon = None
try:
    _canon = forward.canonical_params(params)
    cached = forward.load_result(params) is not None
    params_error = None
except (ValueError, RuntimeError) as e:  # stale widget combo mid-rerun, or a
    cached, params_error = False, str(e)  # missing/invalid T-P table
if tp_mode == "file" and not tp_file_ok and params_error is None:
    params_error = "file-mode T-P selected but no valid table is loaded"

# rough runtime hint keyed off the resolution settings
if _pic:
    # equilibrium states are seconds; the RT/opacity build dominates
    base_min = 0.6 + 0.25 * len(extra_mols)
else:
    base_min = 0.8 + 0.010 * nz + 0.00005 * (nu_pts - forward.NU_PTS_DEFAULT)
    if yconv_cri <= 1.5e-3:          # strict convergence costs extra iterations
        base_min += 0.5
    base_min += 0.25 * len(extra_mols)   # opa build + removed spectrum per extra
if rt_dit_res < 1.0:                 # finer broadening grid = slower opa builds
    base_min += 0.3 * (5 + len(extra_mols))
# cool columns (<~900 K) converge much more slowly (a W107b run took ~5 min)
t_char = {"guillot": tp_kwargs.get("Tirr", 1560.0) / np.sqrt(2.0),
          "file": float(teq),
          "picaso_climate": float(teq)}.get(tp_mode, 1100.0)
if t_char < 900.0 and not _pic:
    base_min += 2.5
if tp_mode == "picaso_climate":      # climate solve (cached after the first)
    base_min += 1.5
# condensing solves carry the window + pin + stricter gate overhead
if use_condense:
    base_min += 1.5
# Jacobian-row cost model: fd = 4 solves per row; Tint_cl = 4 full climate
# re-solves; cloud and Mie rows are RT-only (~seconds); ad = ~1 warm jvp
_solve_min = (0.15 if _pic else max(1.0, base_min * 0.5))
_rt_only = set(forward.CLOUD_FISHER_PARAMS) | set(forward.MIE_FISHER_PARAMS)
n_cloud_rows = sum(1 for n in fisher_params if n in _rt_only)
_solve_rows = [n for n in fisher_params
               if n not in _rt_only and n != "Tint_cl"]
_tint_min = 0.0
if "Tint_cl" in fisher_params:
    _tint_min = 4 * (1.3 + (_solve_min if _pic else _solve_min + 0.8))
if jac_method == "ad":
    fd_min = len(_solve_rows) * 1.7 * _solve_min + 0.2 * n_cloud_rows
else:
    n_fd_comp = sum(1 for n in _solve_rows if n in forward.FD_COMP_PARAMS)
    n_fd_theta = len(_solve_rows) - n_fd_comp
    fd_min = (n_fd_comp * 4 * (_solve_min + (0.0 if _pic else 0.8))
              + n_fd_theta * 4 * _solve_min + 0.2 * n_cloud_rows)
fd_min += _tint_min
native_r = int(round(nu_pts * 2950 / 8000 / 50) * 50)
grid_lbl = f"{nz}-layer, native R≈{native_r}"
est = "instant (cached)" if cached else (
    f"~{base_min + fd_min:.0f} min (local {grid_lbl} run"
    + (f" + {len(fisher_params)} Jacobian rows" if fisher_params else "")
    + ")")

# --- Run row: validation messages, run button, review summary --------------
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
col_btn, _ = st.columns([1, 3])
run_clicked = col_btn.button("Run", type="primary", width="stretch",
                             disabled=(bool(params_error) or not mode_keys
                                       or not floor_choice_made))

with st.expander("Run summary & configuration"):
    _goal_txt = (f"detect {target_mol} at {target_sig:g}σ" if goal == "detect"
                 else f"constrain {forward.PARAM_LABELS[goal_param]} to "
                      f"±{target_prec:g} at {target_sig:g}σ")
    _fish_txt = (", ".join(forward.PARAM_LABELS[p] for p in fisher_params)
                 if fisher_params else "none")
    st.markdown(
        f"- **Target**: {planet_label}, {science_mode}\n"
        f"- **Atmosphere**: {chem_provider.upper()} chemistry, "
        f"{tp_mode} T-P profile, {met:g}× solar, C/O {float(co_ratio):g}\n"
        f"- **Science goal**: {_goal_txt}\n"
        f"- **Free parameters**: {_fish_txt}"
        + (f" ({'AD' if jac_method == 'ad' else 'finite differences'})"
           if fisher_params else "") + "\n"
        f"- **Observation**: {len(mode_keys)} mode(s), {n_transits} {_evw}"
        f"{'s' if n_transits > 1 else ''}, R={r_bin}, floor: {floor_mode}\n"
        f"- **Estimated runtime**: {est}")
    if _canon is not None:
        _share = share_config.build_share(
            canon=_canon,
            goal=dict(goal=goal, target_mol=target_mol,
                      target_sig=float(target_sig),
                      goal_param=goal_param,
                      target_prec=(None if target_prec is None
                                   else float(target_prec)),
                      marginalize=bool(marginalize),
                      do_fisher=bool(do_fisher),
                      fisher_params=list(fisher_params),
                      jac_method=jac_method),
            observation=dict(
                ks_mag=float(ks_mag), t14=float(t14), t_base=float(t_base),
                sat_limit=float(sat_limit), modes=list(mode_keys),
                n_transits=int(n_transits), r_bin=int(r_bin),
                floor_mode=floor_mode,
                floors={k: (None if floors[k] is None
                            else (float(floors[k]) if np.isscalar(floors[k])
                                  else "wavelength table"))
                        for k in mode_keys},
                noise_infl={k: float(infl[k]) for k in mode_keys},
                scenario=scenario, show_noise=bool(show_noise),
                seed=int(seed)),
            tp_table_text=(Path(tp_file_path).read_text()
                           if tp_file_path else None),
            floor_table=(np.asarray(floor_table).tolist()
                         if floor_table is not None else None))
        # deferred: sits right above the progress box, the natural thing to
        # click during a multi-minute wait
        _deferred_download(
            "Download configuration (JSON)",
            json.dumps(_share, indent=2, default=str).encode(),
            f"jwst_tool_{_slug(planet_label)}_config.json",
            "application/json", key=K("dl_config"),
            help="The full setup of this run: the canonical model "
                 "parameters (the same dictionary the result stores as "
                 "provenance) plus the science goal and observation "
                 "settings, with any uploaded tables embedded. Load it "
                 "below -- here or on another machine -- to reproduce "
                 "the run.")
    else:
        st.caption("Configuration download is unavailable while the "
                   "settings do not validate (see the message above).")
    st.caption("To load a downloaded configuration, use step 0 at the top "
               "of the sidebar.")


# run_clicked is finally known: every reserved download slot can be filled
_render_deferred_downloads(busy=run_clicked)


# ---------------------------------------------------------------------------
# Compute on click
# ---------------------------------------------------------------------------
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
        _engine_lbl = "PICASO" if chem_provider == "picaso" else "VULCAN-JAX"
        with st.status(f"Running {_engine_lbl} + ExoJAX forward model "
                       "locally …",
                       expanded=True) as status:
            # prior = the same rough pre-run estimate shown next to the Run
            # button; the bar's remaining time converges to the measured pace
            bar = _TimedBar(prior_total_s=(base_min + fd_min) * 60.0,
                            text="starting …")
            pfile = forward.MODEL_CACHE / f"{forward.params_key(params)}.params.json"
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
                    box.code("\n".join(lines[-10:]))
                    bar.tick()

            with _managed_proc(
                    [sys.executable, str(TOOL_DIR / "forward.py"),
                     str(pfile)]) as proc:
                _watch_proc(proc, _fwd_line, bar.tick)
                proc.wait()
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

    # ETC: always ALL modes (one cache per star; selection changes stay instant)
    all_modes = list(ins.MODES)
    job = noise_mod.noise_job(star, all_modes, sat_limit=sat_limit)
    have_cache = (ins.NOISE_CACHE / f"{noise_mod.job_key(job)}.json").exists()
    if have_cache:
        etc = noise_mod.run_pandeia(job)
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
                    bar.update(n_started[0] / len(all_modes),
                               s.removeprefix("[pandeia] ")
                               .removesuffix("...")
                               + f" ({n_started[0] + 1}/{len(all_modes)})")
                    n_started[0] += 1
                else:
                    lines.append(s)
                    box.code("\n".join(lines[-8:]))
                    bar.tick()

            etc = noise_mod.run_pandeia(job, progress=_cb)
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
                    n_transits, floors[k], noise_inflation=infl[k],
                    scenario=scenario))
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
    # the run is over (finished, failed, or declined): give the buttons back
    # without waiting for the next rerun
    _render_deferred_downloads()
    if out is not None:
        st.session_state["out"] = out
        st.session_state["out_meta"] = dict(
            goal=goal, target=target_mol, goal_param=goal_param,
            target_prec=target_prec, target_sig=target_sig,
            n_transits=n_transits, show_noise=show_noise, seed=seed,
            r_bin=r_bin, planet=planet_label, scenario=scenario,
            floor_mode=floor_mode,
            # the SELECTED floor, not the registry suggestion: a result must
            # carry the number that produced it
            floor_selected={
                k: (None if floors[k] is None
                    else (float(floors[k]) if np.isscalar(floors[k])
                          else "wavelength table"))
                for k in mode_keys})

# ---------------------------------------------------------------------------
# Render order: staleness/failure notices, the VERDICT, then spectrum /
# ranking / mode details / forecast, then the collapsed certificates, then
# the adjoint diagnostics.
# ---------------------------------------------------------------------------
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
_tt_col = f"{_ev}s to target"
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
except (ValueError, RuntimeError):
    _shown_stale = True   # the current sidebar settings do not even validate
if _shown_stale:
    _shown_sci = str(_cpj.get("science_mode", "transmission"))
    _geom = (f" The sidebar observation type is now **{science_mode}**, but "
             f"this spectrum is **{_shown_sci}**."
             if science_mode != _shown_sci else "")
    _chg = ((" Changed settings: " + ", ".join(f"`{c}`" for c in _changed[:8])
             + (", …" if len(_changed) > 8 else "") + ".")
            if _changed else "")
    st.warning(
        "These results are from your previous run; the sidebar has changed "
        "since (or that run failed)." + _chg + _geom
        + " Press **Run** at the top to recompute with the current settings.")

for k, err in out["failed"]:
    first = str(err).strip().splitlines()[-1] if "Traceback" in str(err) else \
        str(err).strip().splitlines()[0]
    st.error(f"{ins.MODES[k]['label']}: the calculation failed. {first}")
    with st.expander(f"{ins.MODES[k]['label']}: technical details"):
        st.code(str(err)[-2500:])
for k, reason in out["unusable"]:
    # "saturated" here means saturated at the tool's OWN shortest ramp
    # (instruments.py ngroup_min), which is a policy bound above the physical
    # minimum for some modes -- say so instead of implying a physical limit
    st.warning(
        f"**{ins.MODES[k]['label']}: unusable on this star**, {reason}. "
        f"Note: this tool never tries ramps shorter than "
        f"{ins.MODES[k]['ngroup_min']} groups; STScI permits shorter ramps "
        "for some modes, and those are not searched.")

if not results:
    st.stop()

fisher_names = ([str(x) for x in model["jac_names"][:-1]]
                if "jac_names" in model else [])
ok = [r for r in results if not r["saturated"]]

# --- verdict (first; success = target met, warning = valid result that
# misses the target, error = only for failed calculations) ------------------
if goal_r == "detect":
    tsig = float(meta.get("target_sig") or 3.0)
    ntr = meta["n_transits"]
    if not ok:
        # every selected mode saturates: never rank saturated results
        st.warning(
            f"**No selected mode is usable: all selected modes saturate "
            f"for this star** (Ks = {star['ks_mag']:g}). Select other "
            "modes or a different target; only raise the saturation limit "
            "(step 4) if that is scientifically justified.")
    else:
        ranked = sorted(ok, key=lambda r: -r["sigma_detect"])
        best = ranked[0]
        bsig = best["sigma_detect"]
        verdict = (f"**Best mode for detecting {meta['target']} on "
                   f"{meta.get('planet', '?')}: {best['label']} "
                   f"({_mode_cfg(best['mode_key'])})**, "
                   f"{bsig:.1f}σ in {ntr} {_ev}{'s' if ntr > 1 else ''} "
                   f"(target {tsig:g}σ; Δχ² = {bsig * bsig:.0f}; median precision "
                   f"{best['median_sigma_ppm']:.0f} ppm per R={meta['r_bin']} bin).")
        if best["warnings"]:
            verdict += (" This configuration has warnings (see the "
                        "mode details table).")
        if bsig >= tsig:
            st.success(verdict + "  Meets the target.")
        elif bsig > 0:
            # floor-aware transit solver: the photon term averages down with
            # N, the systematic floor does not -- a plain 1/sqrt(N) law is
            # optimistic exactly where it matters
            tt = detect.transits_to_target(best, tsig)
            _scen = meta.get("scenario", "random")
            if tt["reachable"]:
                _win = ("" if tt.get("n_last") is None
                        or tt["n_last"] >= detect.N_TRANSITS_CAP else
                        f" Correlated-scenario window: past {tt['n_last']} "
                        f"{_ev}s the detection is lost again.")
                st.warning(verdict + f"  Below the target; {tt['n']} {_ev}s "
                           f"of {best['label']} would reach it (floor-aware "
                           "estimate)." + _win)
            else:
                _fl = _has_floor(best)
                _lim = f"{tt['sig_inf']:.1f}σ" if _fl else ""
                st.warning(
                    verdict + "  The target was "
                    + _not_reached_reason(_scen, _lim, _fl, _ev) + ". "
                    + ("Lower the floor, choose other modes, or relax the "
                       "target." if _fl else
                       "Choose other modes or relax the target."))
        else:
            st.warning(verdict + "  No signal in the selected bands; try "
                       "other modes or a different goal.")
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
        st.warning(
            f"**No selected mode is usable: all selected modes saturate "
            f"for this star** (Ks = {star['ks_mag']:g}), so no constraint "
            f"on {glabel} can be forecast. Select other modes or a "
            "different target; only raise the saturation limit (step 4) "
            "if that is scientifically justified.")
        st.stop()
    if not per_mode:
        st.warning(f"No usable selected mode constrains {glabel}: its "
                   "Jacobian has no signal in these bands. Try other modes "
                   "or a different goal.")
        st.stop()
    bk = min(per_mode, key=per_mode.get)
    bs = per_mode[bk]
    ntr = meta["n_transits"]
    verdict = (f"**Best mode for constraining {glabel} on "
               f"{meta.get('planet', '?')}: {ins.MODES[bk]['label']} "
               f"({_mode_cfg(bk)})**, "
               f"±{bs:.3g}{usp} at {tsig:g}σ in {ntr} {_ev}"
               f"{'s' if ntr > 1 else ''} (target ±{target:g}{usp} "
               f"at {tsig:g}σ).")
    _bw = next(r for r in usable_jac if r["mode_key"] == bk)["warnings"]
    if _bw:
        verdict += (" This configuration has warnings (see the "
                    "mode details table).")
    if bs <= target:
        st.success(verdict + "  Meets the target.")
    elif np.isfinite(comb) and comb <= target:
        st.warning(verdict + f"  No single mode reaches the target, but the "
                   f"combination of all usable selected modes does "
                   f"(±{comb:.3g}{usp} at {tsig:g}σ).")
    else:
        best_r = next(r for r in usable_jac if r["mode_key"] == bk)
        tt = fisher_mod.transits_to_target(best_r, fisher_names, gp,
                                           target / tsig, detect.sigma_at_transits,
                                           co_eval=co_eval)
        _scen = meta.get("scenario", "random")
        if tt["reachable"]:
            _win = ("" if tt.get("n_last") is None
                    or tt["n_last"] >= detect.N_TRANSITS_CAP else
                    f" Correlated-scenario window: past {tt['n_last']} "
                    f"{_ev}s the target is missed again.")
            st.warning(verdict + f"  Below the target; {tt['n']} {_ev}s of "
                       f"{ins.MODES[bk]['label']} would reach it (floor-aware "
                       "estimate)." + _win)
        else:
            _fl = _has_floor(best_r)
            _lim = (f"±{tsig * tt['sig_inf']:.3g}{usp} at {tsig:g}σ" if _fl
                    else "")
            st.warning(
                verdict + "  The target was "
                + _not_reached_reason(_scen, _lim, _fl, _ev) + ". "
                + ("Lower the floor, combine modes, or relax the target."
                   if _fl else "Combine modes or relax the target."))

# The ranking compares science information only. It does not check APT
# feasibility (data volume, schedulability, calibration warnings), so the
# verdict must carry that caveat -- an operationally unsupportable
# configuration can otherwise be presented as "Best" (2026-08-05 review).
if ok:
    st.caption(
        "Best-mode rankings compare expected science information under this "
        "tool's fixed detector configurations; scores are conditional "
        "statistics for the selected atmosphere, not retrieval results. "
        "Operational feasibility (data volume, scheduling, calibration "
        "warnings) is not checked; verify the chosen configuration in APT "
        "before proposing.")

# --- spectrum figure -------------------------------------------------------
st.subheader("Simulated eclipse emission spectrum"
             if str(_cpj.get("science_mode", "transmission")) == "emission"
             else "Simulated transmission spectrum")
wl = model["wl_um"]
order = np.argsort(wl)
wl_s, d_s = wl[order], model["depth"][order] * 1e6
_fname_base = f"jwst_tool_{_slug(meta.get('planet', 'planet'))}"

# DISPLAY smoothing: at native R ~ 1500 the unresolved line forest renders
# as one-sample spikes, so the PLOT is convolved to a constant display R
# (>= 3x the analysis R, floor 300) with the SAME tested LSF operator the
# science path uses (flat weight). No score touches this curve; the native
# model stays in the "Native model (CSV)" download.
_disp_R = float(max(300, 3 * int(meta["r_bin"])))
_disp_wl_r = np.array([float(wl_s[0]), float(wl_s[-1])])
_disp_curve = np.array([_disp_R, _disp_R])


def _display_smooth(y_ppm):
    return binning.smooth_to_native_r(wl_s, y_ppm, _disp_wl_r, _disp_curve,
                                      float(wl_s[0]), float(wl_s[-1]))


d_plot = _display_smooth(d_s)
fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=200)
ax.plot(wl_s, d_plot, color="#444444", lw=1.2, alpha=0.9, zorder=2,
        label="model (smoothed for display)")
d_wo_s = None
if goal_r == "detect":
    mols = [str(x) for x in model["mols"]]
    d_wo_s = model["depth_wo"][mols.index(meta["target"])][order] * 1e6
    ax.plot(wl_s, _display_smooth(d_wo_s), color="#888888", lw=1.1, ls="--",
            zorder=1, label=f"model without {meta['target']}")
rng = np.random.default_rng(int(meta["seed"]))
pt_lo, pt_hi = [], []            # plotted point extents (keep error bars in view)
for r in results:
    c = ins.MODE_COLOR[r["mode_key"]]
    mk = ins.MODE_MARKER.get(r["mode_key"], "o")
    y = r["depth"] * 1e6
    if meta["show_noise"]:
        y = y + rng.normal(0.0, r["sigma"] * 1e6)
    label = r["label"] + (" (saturated!)" if r["saturated"] else "")
    # plot at the response-weighted effective wavelength (matters near
    # detector gaps); marker + color together identify the mode
    x = r.get("wl_eff", r["wl"])
    ax.errorbar(x, y, yerr=r["sigma"] * 1e6, fmt=mk, ms=4.2, lw=1.0,
                color=c, ecolor=c, elinewidth=0.8, capsize=0, zorder=3,
                label=label)
    pt_lo.append(float(np.min(y - r["sigma"] * 1e6)))
    pt_hi.append(float(np.max(y + r["sigma"] * 1e6)))
ax.set_xscale("log")
lo = min(min(r["wl"].min() for r in results), 1.0)
hi = max(r["wl"].max() for r in results)
ticks = [t for t in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0)
         if lo * 0.97 <= t <= hi * 1.03]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.set_xlim(lo * 0.97, hi * 1.03)
# y-limits: keep the model in-window AND every plotted error bar in view
sel = (wl_s >= lo * 0.97) & (wl_s <= hi * 1.03)
y_lo = min(float(d_plot[sel].min()), min(pt_lo))
y_hi = max(float(d_plot[sel].max()), max(pt_hi))
pad = 0.06 * (y_hi - y_lo)
ax.set_ylim(y_lo - pad, y_hi + pad)
ax.set_xlabel("wavelength (µm)")
_depth_lbl = ("eclipse depth (ppm)"
              if str(model.get("science_mode", "transmission")) == "emission"
              else "transit depth (ppm)")
ax.set_ylabel(_depth_lbl)
ax.grid(alpha=0.25)
# legend below the axes; marker + color pairs identify each mode
_handles, _labels = ax.get_legend_handles_labels()
_lg_rows = int(np.ceil(len(_labels) / 3))
fig.subplots_adjust(bottom=0.11 + 0.058 * _lg_rows)
fig.legend(_handles, _labels, loc="lower center", ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, 0.01))
st.pyplot(fig, width="stretch")
_spec_png = _fig_png(fig)
plt.close(fig)
st.caption(
    f"Model {_cpj.get('science_mode', 'transmission')} spectrum with the "
    "depths and predicted uncertainties of each selected mode binned to "
    f"the shared analysis resolution R = λ/Δλ = {meta['r_bin']}, for "
    f"{meta['n_transits']} {_ev}"
    f"{'s' if meta['n_transits'] > 1 else ''}. Each mode is first "
    "simulated at its instrument's native resolution, then binned.")

# downloads: the figure + the plotted numbers (binned points, native model)
_bin_df = pd.concat([
    pd.DataFrame({
        "mode": r["mode_key"], "label": r["label"],
        "wl_um": np.asarray(r["wl"], dtype=float),
        "wl_eff_um": np.asarray(r.get("wl_eff", r["wl"]), dtype=float),
        "depth_ppm": np.asarray(r["depth"], dtype=float) * 1e6,
        "sigma_ppm": np.asarray(r["sigma"], dtype=float) * 1e6,
        "saturated": bool(r["saturated"]),
    }) for r in results], ignore_index=True)
_native = {"wl_um": wl_s, "depth_ppm": d_s}
if d_wo_s is not None:
    _native[f"depth_without_{meta['target']}_ppm"] = d_wo_s
_d1, _d2, _d3, _ = st.columns([1.2, 1.5, 1.5, 2.8])
_d1.download_button("Figure (PNG)", _spec_png,
                    f"{_fname_base}_spectrum.png", "image/png",
                    key=K("dl_spec_png"))
_d2.download_button("Binned points (CSV)", _csv_bytes(_bin_df),
                    f"{_fname_base}_binned_points.csv", "text/csv",
                    key=K("dl_spec_bins"))
_d3.download_button("Native model (CSV)", _csv_bytes(pd.DataFrame(_native)),
                    f"{_fname_base}_model_spectrum.csv", "text/csv",
                    key=K("dl_spec_native"))

# --- goal chart + T-P profile ----------------------------------------------
col1, col2 = st.columns([2.6, 1.4])

with col1:
    if goal_r == "detect":
        st.subheader(f"{meta['target']} conditional template S/N (σ_detect)")
        rs = sorted(results, key=lambda r: r["sigma_detect"])
        names = [r["label"] + (" (saturated)" if r["saturated"] else "")
                 for r in rs]
        vals = [r["sigma_detect"] for r in rs]
        cols = [ins.MODE_COLOR[r["mode_key"]] for r in rs]
        xrefs, xlabel = (3.0, 5.0), (f"conditional template S/N "
                                     f"({meta['n_transits']} {_ev}"
                                     f"{'s' if meta['n_transits'] > 1 else ''})")
        fmt_v = lambda v: f"{v:.1f}σ"
        vline_target = float(meta.get("target_sig") or 3.0)
    else:
        st.subheader(f"Expected uncertainty on {glabel}")
        items = sorted(per_mode.items(), key=lambda kv: -kv[1])   # best at top
        names = [ins.MODES[k]["label"] for k, _ in items]
        vals = [v for _, v in items]
        cols = [ins.MODE_COLOR[k] for k, _ in items]
        if np.isfinite(comb):
            names.append("ALL USABLE (combined)")
            vals.append(comb)
            cols.append("#555555")
        # no parenthetical: it pushed the label past the axes width, and the
        # count and direction are already on the page
        xrefs, xlabel = (), f"expected ±{forward.param_axis(gp)} at {tsig:g}σ"
        fmt_v = lambda v: f"{v:.3g}"
        vline_target = target
    fig2, ax2 = plt.subplots(figsize=(7.4, 0.52 * len(names) + 1.5), dpi=200)
    bars = ax2.barh(names, vals, color=cols, height=0.62)
    for b, v in zip(bars, vals):
        ax2.text(b.get_width() + max(vals) * 0.02,
                 b.get_y() + b.get_height() / 2, fmt_v(v),
                 va="center", fontsize=10, color="#333333")
    # reference/target labels sit just ABOVE the axes box so they never
    # collide with the bars; a reference equal to the target is skipped
    # (coincident labels overprint)
    _blend2 = mtransforms.blended_transform_factory(ax2.transData,
                                                    ax2.transAxes)
    for ref in xrefs:
        if ref < max(vals) * 1.15 and ref != vline_target:
            ax2.axvline(ref, color="#bbbbbb", lw=0.8, ls=":")
            ax2.text(ref, 1.02, f"{ref:.0f}σ", transform=_blend2,
                     fontsize=9, color="#888888", ha="center", va="bottom")
    if vline_target is not None:
        ax2.axvline(vline_target, color="#333333", lw=1.0, ls="--")
        ax2.text(vline_target, 1.02, "target", transform=_blend2,
                 fontsize=10, color="#333333", ha="center", va="bottom")
    ax2.set_xlim(0, max(max(vals), vline_target or 0) * 1.18 + 1e-12)
    ax2.set_xlabel(xlabel)
    ax2.grid(axis="x", alpha=0.25)
    fig2.tight_layout()
    st.pyplot(fig2, width="stretch")
    _rank_png = _fig_png(fig2)
    plt.close(fig2)
    _metric = (f"sigma_detect_{meta['target']}" if goal_r == "detect"
               else f"uncertainty_{gp}_at_{tsig:g}sigma")
    _rank_df = pd.DataFrame({"mode": names, _metric: vals})
    _r1, _r2, _ = st.columns([1.2, 1.2, 2.6])
    _r1.download_button("Figure (PNG)", _rank_png,
                        f"{_fname_base}_{_slug(_metric)}_ranking.png",
                        "image/png", key=K("dl_rank_png"))
    _r2.download_button("Values (CSV)", _csv_bytes(_rank_df),
                        f"{_fname_base}_{_slug(_metric)}_ranking.csv",
                        "text/csv", key=K("dl_rank_csv"))

with col2:
    st.subheader("T-P profile")
    cpj = _cpj
    fig3, ax3 = plt.subplots(figsize=(3.8, 4.2), dpi=200)
    ax3.plot(model["T"], model["p_bar"], color="#2a78d6", lw=1.6)
    for tlim in (320.0, 2980.0):
        ax3.axvline(tlim, color="#cccccc", lw=0.8, ls=":")
    ax3.set_yscale("log")
    ax3.invert_yaxis()
    ax3.set_xlabel("temperature (K)")
    ax3.set_ylabel("pressure (bar)")
    ax3.grid(alpha=0.25)
    fig3.tight_layout()
    st.pyplot(fig3, width="stretch")
    _tp_png = _fig_png(fig3)
    plt.close(fig3)
    st.caption(f"As modeled ({cpj.get('tp_mode', '?')} mode). Dotted lines: "
               "the [320, 2980] K opacity window; profiles outside it are "
               "rejected, never clipped.")
    _p_arr = np.asarray(model["p_bar"], dtype=float)
    _T_arr = np.asarray(model["T"], dtype=float)
    if cpj.get("science_mode") == "emission":
        if float(_T_arr.max()) > 2000.0:
            st.warning(
                f"Layers hotter than 2000 K are present (deepest "
                f"{_T_arr.max():.0f} K). Eclipse spectra can probe them "
                "through opacity windows, and the ultra-hot opacity "
                "sources (H- continuum, Na/K/Fe atomic lines, TiO/VO/FeH) "
                "are not modeled, so fluxes and forecasts in those "
                "windows are uncertain.")
    else:
        # transmission probes p <~ 0.1 bar; a hot deep adiabat below that is
        # invisible to the chord geometry and must not trip the warning
        _probe = _p_arr <= 0.1
        if _probe.any() and float(_T_arr[_probe].max()) > 2000.0:
            st.warning(
                "The transmission photosphere (p <= 0.1 bar) exceeds "
                "2000 K. Ultra-hot opacity sources are not modeled (no "
                "H- continuum, no Na/K/Fe atomic lines, no TiO/VO/FeH), "
                "so spectra and forecasts up here overstate molecular "
                "detectability.")
    _tp_df = pd.DataFrame({"p_bar": np.asarray(model["p_bar"], dtype=float),
                           "T_K": np.asarray(model["T"], dtype=float)})
    _t1, _t2 = st.columns(2)
    _t1.download_button("Figure (PNG)", _tp_png,
                        f"{_fname_base}_tp_profile.png", "image/png",
                        key=K("dl_tp_png"))
    _t2.download_button("Values (CSV)", _csv_bytes(_tp_df),
                        f"{_fname_base}_tp_profile.csv", "text/csv",
                        key=K("dl_tp_csv"))

# --- mode details table ------------------------------------------------------
st.subheader("Mode details")
rows = []
key_order = (lambda r: -r["sigma_detect"]) if goal_r == "detect" else (
    lambda r: per_mode.get(r["mode_key"], np.inf))
for r in sorted(results, key=key_order):
    notes = []
    if r["saturated"]:
        notes.append(f"saturates (full-well {r['sat_frac']:.2f} at min groups)")
    n_part = int(np.sum(np.asarray(r.get("n_pix_partial_sat", 0)) > 0))
    # Quote the NATIVE-grid count: a fully saturated channel has non-finite
    # extracted noise, so the worker's own filter drops it and the
    # post-filter count under-reports. None = the payload predates the
    # census; say "unmeasured", never a number that reads "none saturated".
    _n_full_native = r.get("n_pix_full_sat_native")
    if _n_full_native is None:
        if r.get("n_pix_full_sat_dropped"):
            notes.append(f"at least {r['n_pix_full_sat_dropped']} fully "
                         "saturated pixels excluded (native count unmeasured; "
                         "re-run for a worker v7 census)")
    elif _n_full_native:
        notes.append(f"{_n_full_native} fully saturated pixels excluded")
    if r.get("n_pix_degenerate_dropped"):
        notes.append(f"{r['n_pix_degenerate_dropped']} degenerate-wavelength "
                     "pixels excluded")
    if n_part:
        notes.append(f"partial saturation in {n_part} bins")
    if r["warnings"]:
        notes.append("; ".join(list(r["warnings"])[:2]))
    row = {"mode": r["label"],
           "band (µm)": f"{r['wl'].min():.2f}-{r['wl'].max():.2f}"}
    # NOTE: this column must stay all-string -- mixing int and str values makes
    # streamlit's Arrow serialization fail (loud pyarrow tracebacks per render)
    if r.get("lsf_applied"):
        notes.append("model blurred to native R (Gaussian LSF approximation)")
    if r.get("n_segments", 1) > 1:
        notes.append(f"{r['n_segments']} detector segments (offset per segment)")
    if goal_r == "detect":
        row["σ_detect"] = round(r["sigma_detect"], 1)
        _proj = r.get("sigma_detect_proj", float("nan"))
        if np.isfinite(_proj):
            row["σ_detect (proj)"] = round(_proj, 1)
        # experimental correlated scenarios stay OUT of the table unless the
        # user explicitly selected one
        if meta.get("scenario", "random") != "random":
            for _sc, _v in r.get("sigma_detect_by_scenario", {}).items():
                if _sc != r.get("scenario"):
                    row[f"σ ({_sc}, exp.)"] = round(_v, 1)
        _t = float(meta.get("target_sig") or 3.0)
        if r["sigma_detect"] > 0:
            _tt = detect.transits_to_target(r, _t)
            _fl = _has_floor(r)
            row[_tt_col] = _transits_cell(
                _tt, meta.get("scenario", "random"),
                f"{_tt['sig_inf']:.1f}σ" if _fl else "", _fl, _ev)
        else:
            row[_tt_col] = ""
    else:
        s = per_mode.get(r["mode_key"], np.inf)
        row[f"±{forward.param_axis(gp)} at {tsig:g}σ"] = (
            f"{s:.3g}" if np.isfinite(s)
            else ("saturated" if r["saturated"] else "unconstrained"))
        if np.isfinite(s) and not r["saturated"]:
            _tt = fisher_mod.transits_to_target(r, fisher_names, gp,
                                                target / tsig,
                                                detect.sigma_at_transits,
                                                co_eval=co_eval)
            _fl = _has_floor(r)
            row[_tt_col] = _transits_cell(
                _tt, meta.get("scenario", "random"),
                f"±{tsig * _tt['sig_inf']:.3g}" if _fl else "", _fl, _ev)
        else:
            row[_tt_col] = ""
    # the exact fixed detector configuration this row evaluated (each MODES
    # entry is one subarray + readout pattern, never the whole instrument
    # mode) plus its honest operational status
    row.update({"configuration": _mode_cfg(r["mode_key"]),
                "operational status": _op_status(r),
                "median σ (ppm)": round(r["median_sigma_ppm"]),
                "bins": r["n_bins"], "groups/integration": r["ngroup"],
                "cadence (s)": round(r["t_cycle_s"], 1),
                "notes": "; ".join(notes)})
    rows.append(row)
st.dataframe(rows, width="stretch", hide_index=True)
st.download_button("Mode details (CSV)", _csv_bytes(pd.DataFrame(rows)),
                   f"{_fname_base}_mode_details.csv", "text/csv",
                   key=K("dl_modes_csv"))
if goal_r == "detect":
    st.caption(
        "**σ_detect is a conditional matched-template S/N** for the selected "
        "atmosphere, not a retrieval detection. A retrieval that frees more "
        "atmospheric parameters under the same model and noise assumptions "
        "will usually report a lower significance. σ_detect (proj) also "
        "profiles out the available temperature-profile, reference-radius, "
        "and cloud directions; prefer it for narrow margins."
        + (f" σ_detect is scored under the **{meta.get('scenario')}** "
           "correlated-floor scenario (EXPERIMENTAL, a stated assumption, "
           "not a calibrated systematics model); the σ (…, exp.) columns "
           "re-score it under the other scenarios at identical per-bin "
           "totals." if meta.get("scenario", "random") != "random" else "")
    )
else:
    st.caption(
        f"± per mode is the marginalized Fisher forecast scaled to {tsig:g}σ "
        f"(see the table below); '{_tt_col}' re-solves the Fisher forecast "
        f"at each {_ev} count with the random term scaled 1/N and the minimum "
        "floor as a hard lower bound. A floor-limited target reads "
        "'unreachable' with its floor limit, and a no-floor scan that runs "
        f"out at {detect.N_TRANSITS_CAP} {_ev}s reads '>{detect.N_TRANSITS_CAP} "
        "(scan limit)', never an optimistic 1/√N extrapolation. Saturated "
        "modes are excluded from all forecasts."
    )

# --- parameter constraint forecast (Fisher) --------------------------------
# authoritative parameter order = the Jacobian rows as cached (canonical/sorted),
# NOT the multiselect order
if fisher_names and "jac" in model:
    tsig_f = float(meta.get("target_sig") or 3.0)
    st.subheader("Parameter constraint forecast (Fisher)")
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
                 _cond_col: _cell(n, cond[n]),
                 "unit": forward.PARAM_UNITS[n] or (
                     "C/O ratio" if n == "dlnCO" else "dimensionless")}
                for n in fisher_names]

    frows = []
    usable_f = [r for r in with_jac if not r["saturated"]]
    for r in with_jac:
        if r["saturated"]:
            # shown for completeness; a saturated mode contributes no usable
            # data (same exclusion policy as the verdict + combined)
            frows.append({"mode": r["label"],
                          "parameter": "(saturated, excluded)",
                          _marg_col: "", _cond_col: "", "unit": ""})
            continue
        cond = {}
        sig = fisher_mod.mode_forecast(r, fisher_names, conditional=cond)
        frows.extend(_param_rows(r["label"], sig, cond))
    fdiag = {}
    if len(usable_f) >= 2:
        cond = {}
        sig = fisher_mod.combined_forecast(usable_f, fisher_names, diag=fdiag,
                                           conditional=cond)
        frows.extend(_param_rows("ALL USABLE (combined)", sig, cond))
    st.dataframe(frows, width="stretch", hide_index=True)
    st.caption(
        "One row per mode and parameter. **Marginalized**: joint fit; all "
        "other parameters plus the lnR0 / per-segment offset / slope "
        "nuisances are free and marginalized out. **Conditional**: "
        "everything else held fixed at the input values, a narrower and "
        "more optimistic bound. Both are local Cramer-Rao lower bounds "
        "from the same Fisher matrix, and neither is a posterior "
        "uncertainty; a large gap between them means the parameter is "
        "degenerate with the others in that band, not that the spectral "
        "response is missing.")
    st.download_button("Constraint forecast (CSV)",
                       _csv_bytes(pd.DataFrame(frows)),
                       f"{_fname_base}_fisher_forecast.csv", "text/csv",
                       key=K("dl_fisher_csv"))
    if fdiag:
        rank, dim = fdiag["fisher_rank"], fdiag["fisher_dimension"]
        st.caption(
            f"Numerical health (combined): Fisher rank {rank}/{dim}, condition "
            f"number {fdiag['condition_number']:.2g}."
            + (" **Rank-deficient: degenerate directions are reported as "
               "unconstrained, not as fake finite numbers.**" if rank < dim else ""))
    with st.expander("How to read this table"):
        st.markdown(
            f"- Each bound is the **expected ±uncertainty at {tsig_f:g}σ** "
            f"(= {tsig_f:g} × the Fisher 1σ) on that parameter if you fitted "
            "all listed parameters *simultaneously* to that mode's simulated "
            "data. It is a linearized, local best case (Cramer-Rao bound) "
            "with no priors: nonlinear responses and degeneracies can make "
            "a real posterior wider, and informative priors can make it "
            "narrower.\n"
            "- The sensitivities d(spectrum)/d(parameter) use the "
            "differentiation method selected in the science-goal step, "
            "shown per row in Model quality and provenance below: "
            "**central finite differences** of independently re-converged "
            "solves (default; composition rows re-initialize the chemistry "
            "at the perturbed elemental abundances, the standard VULCAN "
            "workflow), or **warm-started forward-mode automatic "
            "differentiation** (the AD metallicity row holds the "
            "structural hydrostatic grid fixed, a stated 1.6%-level "
            "difference from FD in the earlier benchmark).\n"
            "- Each per-mode row also fits (and marginalizes over) a reference-"
            "radius nuisance **lnR0** plus one absolute-depth **offset per "
            "detector segment**, so the two-detector NIRSpec gratings (G395H, "
            "G235H) float independent **NRS1 and NRS2** steps, as every real "
            "G395H fit does (Moran+2023, Madhusudhan+2023). The combined row "
            "shares lnR0 across modes and keeps one offset per segment across "
            "all of them; that is what keeps multi-instrument combinations "
            "honest.\n"
            "- **No priors** are applied: a parameter with no spectral response "
            "in a mode's band reads *unconstrained* rather than a fake number.\n"
            "- Metallicity **[M/H]** and vertical mixing **log Kzz** are reported "
            "in **dex** (factors of 10); **C/O** is the absolute carbon/oxygen "
            "number ratio N_C/N_O (default ≈ 0.55, the network's WASP-39b "
            "elemental set from Tsai et al. 2023).\n"
            f"- σ is evaluated at the {_ev} count you set. Only the "
            "photon/detector term averages down with more of them; the "
            f"systematic floor does not. Use the '{_tt_col}' column, "
            "not a 1/√N extrapolation."
        )
elif out.get("fisher_names"):
    st.info("A constraint forecast was requested but the cached model has "
            "no Jacobian. Press Run to compute it.")

# --- model quality and provenance (collapsed: supports the result) ---------
with st.expander("Model quality and provenance"):
    if out.get("provenance"):
        _pv = out["provenance"]
        st.caption(
            f"**{ins.BACKEND_STATUS}.** Backend: pandeia.engine "
            f"{_pv['engine_version']} + {_pv['refdata_name']} "
            f"(refdata {_pv['refdata_version']}), worker v{_pv['worker_version']} "
            ",  recorded in every noise cache.")
    # chemistry certificate, provider-aware: everything shown passed its
    # gate, or run_model would have raised
    if "conv_longdy" in model and np.asarray(model["conv_longdy"]).size:
        _gate = float(np.asarray(model["conv_gate"], float)[0])
        st.caption(
            "Chemistry convergence check (every stage passed, or the run "
            "would have stopped with an error): " + "; ".join(
                f"{s}: residual {l:.3g} below the {_gate:g} threshold "
                f"({int(a)} solver steps)"
                for s, a, l in zip(model["conv_stages"], model["conv_accept"],
                                   np.asarray(model["conv_longdy"], float)))
            + ".")
    if "picaso_cert_json" in model:
        _pcert = json.loads(str(model["picaso_cert_json"]))
        st.caption(
            "PICASO equilibrium quality check (every item passed, or the run "
            "would have stopped with an error): built from the table grid "
            f"points {_pcert['nodes'][0]} | {_pcert['nodes'][1]} "
            f"(weights {_pcert['wf']:.3f} / {_pcert['wc']:.3f}). Before "
            "renormalizing, the listed gas species summed to "
            f"{_pcert['gas_sum_min']:.4f} at worst / "
            f"{_pcert['gas_sum_median']:.4f} typically "
            "(the tables omit some minor species, so the sum is renormalized "
            "per layer -- standard practice"
            + (f"; {_pcert['n_layers_below_warn']} layer(s) fell below the "
               f"{_pcert['gas_sum_warn']:g} attention flag" if
               _pcert.get("n_layers_below_warn") else "")
            + ")"
            + (f". Hot-layer carbon-to-oxygen came out at "
               f"{_pcert['realized_gas_co_hotT']:.3f}"
               if _pcert.get("realized_gas_co_hotT") is not None else "")
            + (f". {len(_pcert['suspect_cells_in_span'])} known imperfect table "
               "cell(s) sit in this profile's range (details: "
               "docs/physics_and_conventions.md, PICASO section)"
               if _pcert.get("suspect_cells_in_span") else "")
            + (f". {len(_pcert['corrections_applied'])} catalogued table "
               "fix(es) were applied (only ever to defects vetted and listed "
               "in docs/physics_and_conventions.md, PICASO section)"
               if _pcert.get("corrections_applied") else "")
            + ". Ions and electrons count toward the gas total; graphite is "
              "treated as a solid, not a gas.")
    if "climate_provenance_json" in model:
        _clj = json.loads(str(model["climate_provenance_json"]))
        _clc = _clj.get("cert", {})
        _cvz = [int(x) for x in (_clc.get("cvz_locs") or []) if int(x) > 0]
        _cvz_top = _cvz[0] if _cvz else "?"
        st.caption(
            "Climate quality check (passed): energy balanced at the top of "
            "the atmosphere to "
            f"{_clc.get('flux_toa_over_tidal', float('nan')):.1e} of the "
            "internal heat flux; temperature-gradient range "
            f"[{_clc.get('grad_min', float('nan')):.2f}, "
            f"{_clc.get('grad_max', float('nan')):.2f}] (dlnT/dlnP); "
            f"convective from layer {_cvz_top} down"
            ". The chemistry runs on this profile afterwards and never feeds "
            "back into the climate's heating. Reminder: the deep profile "
            "depends on the radiative-convective boundary setting (its widget "
            "help has the measured sensitivity).")
    if fisher_names and "jac" in model and "fd_err" in model:
        # per-row provenance: FD rows passed the h-vs-2h consistency gate at
        # run time (a failed row raises and no model is cached); AD rows are
        # warm-started jvp (no step to vary, so no consistency metric)
        _methods = ([str(m) for m in model["jac_row_method"]]
                    if "jac_row_method" in model
                    else ["fd-central"] * len(model["jac_names"]))
        _parts = []
        for n, m, e in zip(model["jac_names"], _methods,
                           np.asarray(model["fd_err"], float)):
            if str(n) == "lnR0":
                continue
            _sym = forward.PARAM_SYMBOLS.get(str(n), str(n))
            if m == "ad-jvp":
                _parts.append(f"{_sym} AD (warm jvp)")
            elif not np.isfinite(e):
                # ungated single-central-difference RT rows report their
                # h-vs-2h metric as NaN (unmeasured), never a false 0.0
                _parts.append(f"{_sym} FD-RT (smooth, ungated)")
            else:
                _parts.append(f"{_sym} FD {e:.3f}")
        st.caption(
            "How each Jacobian row was computed. "
            "**FD** rows are central finite differences. Every step "
            "re-solves the model to convergence, and two step sizes are "
            "combined (Richardson). The number after FD compares the "
            "derivative at step h with the one at step 2h. 0 means they "
            f"agree exactly; a row fails above {forward.FD_CONSISTENCY_TOL}. "
            "**AD** rows differentiate the solver itself, in forward mode, "
            "warm-started from the converged state. Photochemistry is on "
            "for these rows. "
            "The two methods were compared once on WASP-39 b. They agreed "
            "to within 0.07-1.6% per row. The 1.6% row is metallicity, "
            "where the gap is a known fixed-grid difference. That "
            "comparison checks the method. It is not an error bar for this "
            "run. "
            f"Rows: {', '.join(_parts)}.")

# --- advanced diagnostics: adjoint sensitivities (reverse-mode AD) -----------
# One reverse-mode adjoint solve gives dL/dlnk over EVERY reaction and dL/dT
# over EVERY layer. Analyzes the CACHED model's canonical params (_cpj),
# never the live sidebar. Expert feature, kept out of the default flow; when
# unavailable, one caption says so instead of a full panel.
# The full panel is hidden while the adjoint diagnostics are in development;
# flip _ADJOINT_PANEL_IN_DEV to False to restore it unchanged.
_ADJOINT_PANEL_IN_DEV = True
with st.expander("Advanced diagnostics (in development)"):
    if _ADJOINT_PANEL_IN_DEV:
        st.caption("In development.")
    elif _cpj.get("chem_provider") == "picaso":
        st.caption(
            "Unavailable for this run: adjoint diagnostics differentiate "
            "the VULCAN reaction network (reverse-mode AD through the "
            "steady-state solver), and the PICASO equilibrium engine has "
            "no reaction network. Re-run with the VULCAN engine to use "
            "this panel.")
    elif _cpj.get("use_condense"):
        st.caption(
            "Unavailable for this run: it used condensation, whose pinned "
            "reservoir is frozen at a step-history-dependent transient, so "
            "no derivative through it is trustworthy. Re-run with "
            "condensation off to use this panel.")
    else:
        st.caption(
            "One **reverse-mode adjoint solve** through the converged "
            "VULCAN-JAX state gives the sensitivity of a molecule's "
            "photosphere abundance to **every reaction rate** in the "
            "network and to **every layer's temperature** at once: the "
            "high-dimensional questions where automatic differentiation "
            "is the only practical tool (the constraint Jacobians above, "
            "over a handful of parameters, use finite differences "
            "instead). Validated upstream to 0.2-0.8% against finite "
            "differences on the WASP-39 b SO2 and HD 189733 b CH4 "
            "benchmarks. This is an expert research feature: the first "
            "run on a machine can take hours on CPU (compilation).")
        _adj_mols = [str(m) for m in model["mols"]]
        _adj_tgt = str(meta.get("target") or "")
        _adj_default = (_adj_tgt if _adj_tgt in _adj_mols
                        else "SO2" if "SO2" in _adj_mols else _adj_mols[0])
        adj_species = st.selectbox(
            "Target molecule", _adj_mols, index=_adj_mols.index(_adj_default),
            key=K("adjsp"),
            help="L = log10 VMR of this molecule at its peak-abundance layer "
                 "inside the transit photosphere (1e-5 to 0.1 bar).")
        adj = adjoint_diag.load_result(_cpj, adj_species)
        if adj is None:
            st.caption(
                "Not cached for this model + molecule. Cost: one chemistry "
                "re-solve with an extended budget, plus the adjoint "
                "ensemble. The FIRST adjoint run on a machine also compiles "
                "the solver's step-VJP, which can take hours on CPU; the "
                "result lands in the persistent JAX compile cache, so later "
                "runs skip straight to the solve.")
            # temporarily disabled in the GUI (in development); a disabled
            # button always returns False, so the run path below is unreachable
            st.caption("In development. Temporarily disabled.")
            if st.button("Run adjoint diagnostics", key=K("adjrun"),
                         disabled=True, help="In development"):
                _adj_slot = runlimit.acquire("adjoint")
                if _adj_slot is None:
                    st.error(
                        f"This instance is already running "
                        f"{runlimit.MAX_CONCURRENT} heavy calculations "
                        "(shared, public hardware). Try again in a few "
                        "minutes.")
                    st.stop()
                try:
                    with st.status("Running reverse-mode adjoint diagnostics …",
                                   expanded=True) as status:
                        # no prior: the first run on a machine compiles the
                        # step-VJP (hours on CPU), later runs skip it; the
                        # remaining-time readout is purely measured
                        bar = _TimedBar(text="starting …")
                        adjoint_diag.ADJOINT_CACHE.mkdir(parents=True,
                                                         exist_ok=True)
                        _apf = (adjoint_diag.ADJOINT_CACHE /
                                f"{adjoint_diag.adjoint_key(_cpj, adj_species)}"
                                ".params.json")
                        _apf.write_text(json.dumps(_cpj))
                        box = st.empty()
                        lines = []

                        def _adj_line(line):
                            m = _ADJ_PROG_RE.match(line)
                            if m:
                                bar.update(min(1.0, float(m.group(1))),
                                           m.group(2))
                            else:
                                lines.append(line)
                                box.code("\n".join(lines[-10:]))
                                bar.tick()

                        with _managed_proc(
                                [sys.executable, "-m",
                                 "jwst_tool.adjoint_diag", str(_apf),
                                 adj_species]) as proc:
                            _watch_proc(proc, _adj_line, bar.tick)
                            proc.wait()
                        if proc.returncode != 0:
                            status.update(label="Adjoint diagnostics failed",
                                          state="error")
                            st.error("Adjoint diagnostics failed. Last "
                                     "output:\n\n```\n"
                                     + "\n".join(lines[-25:]) + "\n```")
                        else:
                            bar.done()
                            status.update(label="Adjoint diagnostics done",
                                          state="complete")
                            st.rerun()
                finally:
                    # release even on failure: a leaked slot would stay held
                    # until a server restart
                    _adj_slot.release()
        else:
            _trust = bool(adj["magnitudes_trusted"])
            # pair antisymmetry: older cached npz predate it, so show only
            # when present. Diagnostic-only, never a trust gate.
            _pa_txt = (f", forward/reverse pair antisymmetry "
                       f"{float(adj['pair_antisym']):.3g} (diagnostic only, "
                       "not a trust gate)" if "pair_antisym" in adj else "")
            st.caption(
                f"Loss: log10 VMR({adj_species}) = "
                f"{float(adj['loss_log10_vmr']):.2f} at "
                f"P = {float(adj['loss_p_bar']):.1e} bar. Certification: "
                f"fixed-point error {float(adj['fp_err']):.1e}, adjoint "
                f"residual (median) {float(adj['resid_median']):.3g}, "
                f"twin-ensemble spread {float(adj['ensemble_spread']):.3g} "
                f"over {int(adj['n_solves'])} solves{_pa_txt}; scope-audit worst "
                f"defect {float(adj['audit_max_rel_defect']):.1e} "
                f"(loss footprint {float(adj['audit_loss_footprint_defect']):.1e})"
                f"; photolysis feedback "
                f"{'ON' if bool(adj['photo_feedback']) else 'off'}. "
                + ("**Magnitudes are trustworthy** (residual <= 0.2 and "
                   "spread <= 0.15, the upstream gates)." if _trust else
                   "**Treat as a RANKING**: residual or spread exceeds the "
                   "upstream trust gates, so magnitudes are indicative only."))
            _adj_df = pd.DataFrame({
                "reaction": [str(x) for x in adj["top_label"][:10]],
                "type": [str(x) for x in adj["top_kind"][:10]],
                "dL/dln k": [f"{v:+.3g}" for v in
                             np.asarray(adj["top_S"], float)[:10]],
            })
            st.dataframe(_adj_df, width="stretch", hide_index=True)
            st.caption(
                "Physical (detailed-balance pair-summed) sensitivities of the "
                "loss to each reaction's rate constant, top 10 of the full "
                "network. Sign: positive means a faster rate increases the "
                f"abundance. Under a uniform Agundez (2025) class-B rate "
                f"uncertainty ({float(adj['uq_class_dex']):g} dex per "
                "reaction), the implied abundance spread is sigma(log10 VMR) "
                f"= {float(adj['uq_sigma_log10']):.2f} dex. That is a stated "
                "assumption, not a per-reaction uncertainty assessment.")
            fig_adj, ax_adj = plt.subplots(figsize=(6.0, 3.4))
            _pb = np.asarray(adj["p_bar"], float)
            _dt = np.asarray(adj["dLdT"], float)
            ax_adj.plot(_dt * 1.0e3, _pb, lw=1.4)
            ax_adj.axhline(float(adj["loss_p_bar"]), ls=":", lw=0.8,
                           color="gray")
            ax_adj.set_yscale("log")
            ax_adj.invert_yaxis()
            ax_adj.set_xlabel(f"d log10 VMR({adj_species}) / dT  [1e-3 per K]")
            ax_adj.set_ylabel("Pressure [bar]")
            ax_adj.set_title("Per-layer temperature sensitivity (adjoint)")
            st.pyplot(fig_adj, width="stretch")
            st.caption(
                "Chemistry-path gradient: how the target's photosphere "
                "abundance responds to warming each layer. Photolysis cross "
                "sections and the diffusion/geometry rebuild are frozen by "
                "design, the upstream contract, with rebuild consistency "
                f"{float(adj['rebuild_consistency']):.1e}. The dotted line "
                "marks the loss layer.")
            _a1, _a2 = st.columns(2)
            _a1.download_button(
                "Reactions (CSV)", _csv_bytes(pd.DataFrame({
                    "reaction": [str(x) for x in adj["top_label"]],
                    "type": [str(x) for x in adj["top_kind"]],
                    "dL_dlnk": np.asarray(adj["top_S"], float),
                })),
                f"{_fname_base}_adjoint_{adj_species}_reactions.csv",
                "text/csv", key=K("dl_adj_csv"))
            _a2.download_button(
                "dL/dT profile (CSV)", _csv_bytes(pd.DataFrame({
                    "p_bar": _pb, "dlog10vmr_dT_perK": _dt})),
                f"{_fname_base}_adjoint_{adj_species}_dLdT.csv",
                "text/csv", key=K("dl_adj_dt_csv"))







