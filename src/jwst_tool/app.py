"""JWST instrument selector -- Streamlit GUI.

Launch via the console script ``jwst-tool`` (installed with vulcan-jwst-tool), or
directly:  streamlit run src/jwst_tool/app.py  (from the repo root).

Pipeline per run: VULCAN-JAX photochemistry -> ExoJax transmission spectrum
(local subprocess, disk-cached; ~1.5-2 min at the default 100-layer resolution) ->
Pandeia ETC noise per instrument mode (subprocess in its own conda env, disk-cached) ->
science-goal scoring per mode. Two goal types: DETECT a molecule (nested-model
delta-chi2 significance) or CONSTRAIN a parameter (Fisher forecast from
consistency-checked finite-difference Jacobians, vs a target precision). Planets beyond
WASP-39b come from the registry in planets.py (or a fully custom system).
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
import pandas as pd
import streamlit as st

TOOL_DIR = Path(__file__).resolve().parent   # forward.py subprocess lives here

from jwst_tool import adjoint_diag, datacheck, detect, fisher as fisher_mod, forward
from jwst_tool import noise as noise_mod
from jwst_tool import instruments as ins
from jwst_tool import planets
from jwst_tool import runlimit
from jwst_tool import picaso_chem

# House figure style: recessive axes/grid, consistent typography, white face
# (figures must download clean on any Streamlit theme). Data colors stay the
# fixed per-mode palette in instruments.MODE_COLOR.
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#9aa0a6", "axes.linewidth": 0.8,
    "axes.labelcolor": "#333333", "xtick.color": "#555555",
    "ytick.color": "#555555", "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.labelsize": 10, "legend.fontsize": 9,
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


def _never_reason(scenario: str, val: str) -> str:
    """Honest 'unreachable' phrasing. Under the default random scenario
    sig_inf is an exact ceiling (score monotone in N: 'floor caps at').
    Under a correlated scenario sig_inf is only the N-to-infinity limit --
    scores can peak at finite N -- so unreachability comes from the full
    1..N_TRANSITS_CAP scan and is phrased that way."""
    if scenario == "random":
        return f"floor caps at {val}"
    return (f"no count up to {detect.N_TRANSITS_CAP} works; "
            f"N→∞ limit {val}")


def _transits_cell(tt: dict, scenario: str, val_never: str) -> str:
    """'transits → target' table cell, window-aware for correlated scenarios."""
    if not tt["reachable"]:
        return f"never ({_never_reason(scenario, val_never)})"
    cell = str(tt["n"])
    if (tt.get("n_last") is not None
            and tt["n_last"] < detect.N_TRANSITS_CAP):
        cell += f" (window: lost again past {tt['n_last']})"
    return cell

st.set_page_config(page_title="JWST Instrument Selector",
                   layout="wide")


# ---------------------------------------------------------------------------
# One-time "how it works" gate (plain-language intro + acknowledge button)
# ---------------------------------------------------------------------------
def _intro_gate() -> None:
    """Show a short plain-English explainer the first time the app loads this
    session; the tool proper does not render until the user clicks through.
    The acknowledgment survives the sidebar reset (see _reset_all)."""
    if st.session_state.get("intro_ack"):
        return
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        st.title("JWST instrument selector")
        st.markdown(
            f"""
**vulcan-jwst-tool** ranks JWST time-series instrument modes by how well
each one can **detect a target molecule** or **constrain an atmospheric
parameter** (metallicity, C/O, vertical mixing, temperature, clouds) for a
given exoplanet, in transmission or in eclipse emission, and reports how
many transits or eclipses reach a chosen precision.

It follows the same principle as **PandExo** (a Pandeia
exposure-time-calculator forecast for JWST exoplanet spectra), but instead
of an assumed input spectrum it computes one live, for exactly the
atmosphere you configure:

1. **Chemistry**: two independent choices. First the **chemistry engine**
   -- who computes the abundances: steady-state photochemical kinetics
   with [VULCAN](https://github.com/exoclime/VULCAN) (run through its JAX
   port VULCAN-JAX), or PICASO chemical-equilibrium tables (fast, no
   photochemistry, so no SO2). Second the **temperature profile** that
   chemistry runs on: Guillot, your own table, or a PICASO
   radiative-convective climate solve -- all available under EITHER
   engine, so VULCAN photochemistry can run on a PICASO-computed climate
   profile. A solve that cannot pass its quality gate errors loudly
   instead of returning a wrong spectrum.
2. **Spectrum**: ExoJAX radiative transfer, either the transit depth or the
   dayside eclipse depth against a PHOENIX stellar model.
3. **Noise**: the real STScI Pandeia ETC engine
   ({ins.BACKEND_STATUS.split(' /')[0]} on this instance) per instrument
   mode, with a PandExo-style minimum noise floor.
4. **Scores**: a conditional template S/N per molecule, Fisher (Cramer-Rao)
   parameter forecasts, and a floor-aware count of the transits or eclipses
   needed to reach your target.

**How to use it**: work down the sidebar in order (planet and star,
atmosphere physics, science goal, instrument modes and noise), then press
Run. A fresh configuration solves the chemistry from scratch and takes
minutes; any configuration computed before loads instantly from the cache.
Constraint forecasts add several minutes per freed parameter with the
default finite differences (a faster forward-AD method is a sidebar
option); every number is labeled with the method that produced it, and
every result stores its convergence and version provenance.

The forecasts are **deliberately optimistic** in three ways:

- Detection scores are a conditional matched-template signal-to-noise
  ratio at one fixed atmosphere, so a real retrieval does worse.
- Fisher results are local Cramer-Rao lower bounds rather than posterior
  widths.
- The noise model omits time-correlated systematics (visit-long trends,
  1/f residuals, detrending covariance, stellar heterogeneity), so achieved
  precision is usually poorer. Treat mode rankings as more robust than
  absolute ppm numbers.

This is a **planning tool, not an atmospheric retrieval**. Condensation
(S8 rainout) is offered for detection goals only and never combines with a
derivative-based forecast.
            """
        )
        st.write("")
        if st.button("I understand, open the tool", type="primary"):
            st.session_state["intro_ack"] = True
            st.rerun()
    st.stop()


_intro_gate()

st.title("JWST instrument selector")
st.caption(
    "Chemistry (VULCAN-JAX photochemistry or PICASO equilibrium) → ExoJAX "
    "spectrum (transmission or eclipse emission) → Pandeia ETC noise. "
    "Pick a planet and a science goal, run the live forward model, and see "
    "which instrument mode achieves it best. Uncertainties are a **Pandeia "
    "instrument-model forecast** with an optional PandExo-style minimum noise "
    "floor, not a full time-series systematics forecast; treat mode rankings "
    "as more robust than absolute ppm."
)
st.caption(f"**{ins.BACKEND_STATUS}**. Every result records the exact "
           "engine + refdata versions in its provenance block.")

# ---------------------------------------------------------------------------
# Data availability -- detected live; the display adapts to what is installed
# (missing data still fails loudly at run time; this is the up-front view)
# ---------------------------------------------------------------------------
_STATUS_LABEL = {datacheck.OK: "installed", datacheck.MISSING: "MISSING",
                 datacheck.AUTO: "downloads on first use"}


@st.cache_data(ttl=3600, show_spinner="Checking installed data ...")
def _cached_full_report(_nonce: int, _backend: str, _picaso_root: str):
    # v18.1 latency fix, twice over: the report is DISK-persisted (the Space
    # entrypoint warms it in the background at boot, so even the first
    # visitor gets an instant panel) and the in-process st.cache keeps
    # reruns free. The manifest check inside is sampled, not exhaustive
    # (full pass: `jwst-tool data --deep`). The refresh button deletes the
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
        "refuse to run until it is installed (nothing degrades silently):\n\n"
        + "\n".join(f"- **{it.label}** -- {it.detail}. How to get it: "
                    f"{it.remedy}" for it in _missing_req)
        + "\n\nInstall commands are in the README's *Data setup* section; "
          "console report: `jwst-tool data`.")
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
        f"results ({_c['noise_cache']['mb']} MB); safe to delete, rebuilt "
        "on demand. Console equivalent: `jwst-tool data` (add `--deep` to "
        "probe the Pandeia env's engine version).")
    st.button("Refresh data status", on_click=_bump_data_nonce,
              key="data_refresh_btn",
              help="The status above is cached for an hour (st.cache_data "
                   "ttl=3600 here + datacheck's 3600 s disk cache; scanning "
                   "every dataset is slow on remote volumes); refresh after "
                   "installing data.")

# Measured FD/AD Fisher-row wall times (WASP-39b defaults; rationale in
# forward.py's FD_STEPS block): an FD composition row re-solves the chemistry
# 4x, an FD Kzz/T row is 4 cold solves, an AD row is one warm-started
# derivative. Threaded through every GUI mention below so a re-measurement
# updates one place (the slow-run estimator at the FD warning uses their
# midpoints, 7 and 4).
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
    measured pace, weighting the measurement by the completed fraction, so
    it starts at the prior and converges to the measured rate; with no
    prior it is purely measured and appears once the first progress
    fraction lands. It is labeled "about" -- stage weights are rough.

    The blend is over the two REMAINING-time estimates, not the two totals.
    Blending totals is the natural-looking thing to write and is what this
    class did until 2026-07-21:

        total = frac * (e / frac) + (1 - frac) * prior

    but ``frac * (e / frac)`` is identically ``e``, so that collapses to
    ``remaining = (1 - frac) * prior`` -- the measured pace cancels out and
    the countdown is FROZEN for the whole stage, no matter how long it
    actually runs. A photochemistry stage that overran its weight by 13x
    still read "about 58 s left" at 7m50s elapsed. Keep the blend on
    remaining times."""

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
            # measured: at the pace observed so far, (1 - frac) of the work
            # is still ahead. Grows with e when a stage overruns -- which is
            # the whole point of showing a live estimate.
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
    ``on_tick`` at least every ``tick_s`` seconds of silence so the
    elapsed/remaining readout keeps counting through long quiet solver
    stages. Reads the raw pipe fd (select + os.read), so no completed line
    ever sits hidden in a Python-level buffer. Where select() cannot watch
    pipes (Windows) it degrades to blocking reads: same lines, the clock
    just only advances when output arrives."""
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

# default target precision per parameter (DISPLAY units: dex / K / absolute C/O)
_TARGET_DEFAULT = {"lnZ": 0.10, "dlnCO": 0.05, "lnKzz": 0.30,
                   "Tirr": 50.0, "Tint": 50.0,
                   "Tint_cl": 50.0,
                   "log_kappa": 0.30, "log_gamma": 0.30,
                   "log_kappa_cloud": 0.30, "alpha_cloud": 0.50,
                   "mie_log_rg": 0.30, "mie_sigmag": 0.20,
                   "mie_log_mmr": 0.50}
# Every freeable Fisher parameter can be chosen as the constraint goal, which
# looks up _TARGET_DEFAULT[goal_param] -- so a missing entry would KeyError the
# UI. Guard it at import (caught by the smoke test) rather than at click time.
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
# Reset = bump a nonce that namespaces EVERY widget key: all widgets are
# re-created at their defaults (session_state.clear() alone does not reset
# keyless widgets, whose state lives outside the exposed dict).
_NONCE = st.session_state.setdefault("reset_nonce", 0)


def _reset_all():
    n = st.session_state.get("reset_nonce", 0) + 1
    ack = st.session_state.get("intro_ack", False)   # a settings reset is not
    st.session_state.clear()                          # a reason to re-show the
    st.session_state["reset_nonce"] = n               # how-it-works intro
    st.session_state["intro_ack"] = ack


def K(name: str) -> str:
    return f"n{_NONCE}_{name}"


# The chemistry-engine choice, read EARLY from session state so widgets that
# render before the engine selectbox (the stellar-UV selector in Planet &
# star) can adapt. Streamlit updates widget state before the rerun executes,
# so this matches the selectbox that renders later in the same run; on the
# very first render it is the default (VULCAN).
_pic_hint = st.session_state.get(K("provider"), "vulcan") == "picaso"


with st.sidebar:
    st.markdown("### Planet & star")
    planet_key = st.selectbox(
        "Planet", list(planets.PLANETS) + ["custom"], key=K("planet"),
        format_func=lambda k: planets.PLANETS[k]["label"] if k in planets.PLANETS
        else "Custom planet …",
        help="Every planet runs the same chemistry+RT machinery (validated on "
             "WASP-39 b; other planets share the code path, not per-planet "
             "validation); the system identity (gravity, radii, star, orbit, "
             "UV spectrum) is swapped in. The T-P profile is set below "
             "(Guillot, table, or climate).")
    pdef = planets.PLANETS.get(planet_key, planets.CUSTOM_DEFAULTS)
    st.caption(pdef["note"] if planet_key in planets.PLANETS
               else "Starts from WASP-39 b values, edit everything below.")

    def _k(name: str) -> str:            # per-planet widget state
        return K(f"{planet_key}_{name}")

    with st.expander("System parameters", expanded=(planet_key == "custom")):
        teff = st.number_input("Star T_eff (K)", 3000.0, 7000.0,
                               pdef["star"]["teff"], 50.0, key=_k("teff"),
                               help="PHOENIX SED for the ETC (with log g, [Fe/H]).")
        logg = st.number_input("Star log g (cgs, log10 cm/s^2)", 3.5, 5.5,
                               pdef["star"]["log_g"], 0.1, key=_k("logg"),
                               help="Stellar surface gravity as log10(g) in cgs "
                                    "(g in cm/s^2); feeds the PHOENIX SED for the ETC.")
        feh = st.number_input("Star [Fe/H]", -2.0, 0.5,
                              pdef["star"]["metallicity"], 0.1, key=_k("feh"))
        ks_mag = st.number_input("Ks mag (2MASS)", 4.0, 16.0,
                                 pdef["star"]["ks_mag"], 0.1, key=_k("ks"),
                                 help="Sets absolute count rates → saturation & "
                                      "photon noise. Brighter = more saturation.")
        rstar = st.number_input("R_star (R_sun)", 0.2, 3.0, pdef["rstar_rsun"],
                                0.01, key=_k("rstar"), format="%.3f",
                                help="Transit-depth normalization + UV flux at "
                                     "the planet.")
        rp = st.number_input("R_p (R_Jup)", 0.1, 2.5, pdef["rp_rjup"], 0.01,
                             key=_k("rp"), format="%.3f")
        g_ms2 = st.number_input("Surface gravity (m/s²)", 1.0, 100.0,
                                pdef["gs_cgs"] / 100.0, 0.5, key=_k("g"),
                                help="Sets the scale height: lower gravity = "
                                     "bigger spectral features.")
        orbit_au = st.number_input("Semi-major axis (au)", 0.005, 1.0,
                                   pdef["orbit_au"], 0.001, key=_k("a"),
                                   format="%.4f",
                                   help="Scales the stellar UV reaching the "
                                        "planet (photochemistry).")
        t14 = st.number_input("Transit / eclipse duration T14 (hr)", 0.5, 10.0,
                              pdef["t14_hr"], 0.1, key=_k("t14"),
                              help="Duration of the observed event: the "
                                   "transit in transmission, the secondary "
                                   "eclipse in emission (near-equal on a "
                                   "circular orbit). Sets the in-event "
                                   "integration time per visit.")
        _uv_ok = datacheck.uv_spectra_status()
        sflux = st.selectbox("Stellar UV spectrum (photochemistry)",
                             list(planets.SFLUX_CHOICES),
                             index=list(planets.SFLUX_CHOICES).index(pdef["sflux"]),
                             format_func=lambda f: (
                                 planets.SFLUX_CHOICES[f]
                                 + ("" if _uv_ok.get(f) else "  [FILE MISSING]")),
                             key=_k("sflux"), disabled=_pic_hint,
                             help="Shipped VULCAN spectra, pick the nearest "
                                  "spectral type. Drives photolysis (SO2, CH4 …). "
                                  "A [FILE MISSING] entry means a broken "
                                  "vulcan-jax install; the run would refuse it.")
        if _pic_hint:
            st.caption("UV spectrum unused: the PICASO engine has no "
                       "photolysis, so this selection has no effect there.")

    teq = float(pdef["teq_k"])
    st.divider()
    st.markdown("### Observation geometry")
    science_mode = st.radio(
        "Geometry", ["transmission", "emission"], horizontal=True,
        key=K("scimode"),
        format_func={"transmission": "Transmission (transit)",
                     "emission": "Emission (secondary eclipse)"}.get)
    # Event word for every observation-facing label: the noise model is the
    # same machinery either way (N events + out-of-event baseline), only the
    # vocabulary changes with the geometry.
    _evw = "eclipse" if science_mode == "emission" else "transit"
    st.divider()
    st.markdown("### Forward model")
    chem_provider = st.selectbox(
        "Chemistry engine", ["vulcan", "picaso"], index=0, key=K("provider"),
        format_func={"vulcan": "VULCAN", "picaso": "PICASO"}.get,
        help="Which model computes the atmosphere's chemical makeup.\n\n"
             "**VULCAN** simulates the chemistry in motion: starlight "
             "breaking molecules apart (photochemistry), winds mixing "
             "layers, and reactions racing each other. That is how "
             "out-of-equilibrium molecules like SO2 -- the WASP-39 b "
             "headline detection -- appear. A run takes a few minutes.\n\n"
             "**PICASO** assumes every reaction has fully settled "
             "(chemical equilibrium) and reads the answer from "
             "pre-computed tables -- the chemistry itself takes seconds, "
             "and a full run about a minute (the radiative transfer "
             "dominates). The trade: no "
             "photochemistry means no SO2, and its tables stop at "
             "C/O = 1.10. Parameter constraints use finite differences "
             "only.\n\n"
             "Both engines feed the exact same radiative transfer, noise "
             "model, and statistics, so switching engines shows you what "
             "the disequilibrium physics does -- that comparison is the "
             "point of having both.")
    _pic = chem_provider == "picaso"
    if _pic:
        st.session_state[K("jacm")] = "fd"   # tables are not differentiable
    st.markdown("### Differentiation method")
    jac_method = st.selectbox(
        "How derivatives are computed", ["fd", "ad"], index=0, key=K("jacm"),
        disabled=_pic,
        format_func={"fd": "Finite differences (default)",
                     "ad": "Automatic differentiation (forward-mode)"}.get,
        help="This choice applies wherever the tool needs the derivative "
             "of the spectrum with respect to a parameter, which means the "
             "Fisher constraint forecast and its projected detection "
             "score. Finite differences re-run the converged model at "
             "shifted parameter values and difference the results, "
             "checking two step sizes against each other. They are "
             "transparent and self-verifying, and they work everywhere, "
             "at any composition, with photochemistry on or off. They are "
             "also the only method compatible with condensation. "
             "Automatic differentiation carries one exact derivative "
             "through the solver per row and is about 1.7 to 4 times "
             "faster. It requires photochemistry, which is locked on "
             "below while AD is selected. It disables condensation, and "
             "it refuses the C/O row on carbon-rich compositions rather "
             "than reporting a bad number. The two methods agree to "
             "0.07-1.6% per row on the WASP-39b defaults, and every "
             "result row is labeled with the method that produced it. "
             "The post-run adjoint diagnostics panel uses reverse-mode "
             "AD regardless of this choice.")
    if _pic:
        st.caption("Locked to finite differences: PICASO's chemistry is "
                   "table lookups, and there is no way to carry an exact "
                   "derivative through a table the way AD does through "
                   "VULCAN's equations.")
    # --- Science goal + Fisher controls, placed by the FD/AD choice they
    # depend on. The constraint/Fisher MENUS depend on selections made further
    # down the sidebar (T-P mode, cloud/Mie decks, condensation, extra
    # molecules); those are read from session_state -- the value persisted from
    # the previous rerun. Streamlit reruns top-to-bottom on every interaction,
    # so the menus refresh one interaction after such a change; the model run
    # below always uses the live widget values.
    _tp_ss = st.session_state.get(_k("tp"), "guillot")
    if _tp_ss not in forward.TP_PARAM_NAMES:
        _tp_ss = "guillot"
    _cloud_ss = bool(st.session_state.get(K("cloud"), False))
    _mie_ss = str(st.session_state.get(K("miec"), "") or "")
    _conden_ss = bool(st.session_state.get(K("conden"), False))
    _extra_ss = list(st.session_state.get(K(f"xmols_{chem_provider}"), []) or [])
    # Freeable parameters per engine/mode. Under PICASO + climate, C/O
    # (dlnCO) is NOT offered: climate composition sits exactly on a
    # chemistry-table node where no trustworthy C/O derivative exists
    # (canonical_params refuses it too -- the menu just spares the user the
    # error).
    if _pic:
        _chem_free = (["lnZ"] if _tp_ss == "picaso_climate"
                      else ["lnZ", "dlnCO"])
    else:
        _chem_free = list(forward.CHEM_PARAM_NAMES)
    avail_free = _chem_free + forward.TP_PARAM_NAMES[_tp_ss]
    if _cloud_ss:                        # v16: cloud-deck marginalization
        avail_free = avail_free + list(forward.CLOUD_FISHER_PARAMS)
    if _mie_ss:                  # v16: Mie-deck marginalization
        avail_free = avail_free + list(forward.MIE_FISHER_PARAMS)
    mol_options = forward.active_molecules(
        {"chem_provider": chem_provider, "extra_mols": _extra_ss})

    st.divider()
    st.markdown("### Science goal")

    with st.expander("Goal", expanded=True):
        if _conden_ss:
            st.session_state[K("goal")] = "detect"
        goal = st.radio(
            "Goal", ["detect", "constrain"], horizontal=True, key=K("goal"),
            disabled=_conden_ss,
            format_func={"detect": "Detect a molecule",
                         "constrain": "Constrain a parameter"}.get,
            help="Detecting a molecule scores the significance of its "
                 "presence, the Δχ² between the spectrum with and without "
                 "it. Constraining a parameter forecasts how tightly "
                 "metallicity, C/O, Kzz, or a temperature could be "
                 "measured, via a Fisher forecast whose Jacobian uses the "
                 "differentiation method selected above. Constraints cost "
                 "minutes per freed parameter.")
        if _conden_ss:
            goal = "detect"
            st.caption(
                "Condensation is detection-only: parameter constraints need "
                "d(spectrum)/d(parameter), and no differentiation method is "
                "valid through the condensation pin (see the condensation "
                "note above).")
        goal_param, target_prec, marginalize = None, None, True
        if goal == "detect":
            # SO2 is the headline W39b science under VULCAN; the equilibrium
            # provider has no SO2, so its default detection target is H2O
            _mol_default = "SO2" if "SO2" in mol_options else "H2O"
            target_mol = st.selectbox(
                "Detect molecule", mol_options,
                index=mol_options.index(_mol_default),
                key=K(f"mol_{chem_provider}_" + "_".join(sorted(_extra_ss))))
        else:
            target_mol = None
            goal_param = st.selectbox(
                "Constrain parameter", avail_free,
                key=K(f"gp_{chem_provider}_{_tp_ss}_{int(_cloud_ss)}_"
                      f"{int(bool(_mie_ss))}"),
                format_func=lambda n: forward.PARAM_LABELS[n],
                help="By default the constraint is marginalized over the other "
                     "free parameters (Fisher forecast section) and a "
                     "reference-radius nuisance -- an honest joint-fit "
                     "uncertainty. Uncheck marginalization below to condition "
                     "on the others instead (optimistic).")
            marginalize = st.checkbox(
                "Marginalize over the other parameters", value=True,
                key=K("marg"),
                help="On (default, recommended): the joint-fit Cramér-Rao "
                     "bound, marginalized over the other free parameters plus a "
                     "reference-radius nuisance -- a realistic retrieval "
                     "uncertainty. Off: every other parameter is held FIXED and "
                     "the forecast conditions on them, an OPTIMISTIC lower bound "
                     "that is NOT a real retrieval uncertainty.")
            if not marginalize:
                st.warning(
                    "Marginalization OFF: the forecast holds every other "
                    "parameter fixed and reports a conditional (others-known) "
                    "bound. This is optimistic and NOT a realistic retrieval "
                    "uncertainty -- read it only as a best-case sensitivity. "
                    "Leave it on unless you specifically want that.")
            unit = forward.PARAM_UNITS[goal_param]
            # label uses the unit when there is one (dex / K), else the bare
            # symbol -- C/O is a dimensionless number ratio
            _tgt_lbl = (f"Target precision (±{unit})" if unit else
                        f"Target precision (±{forward.PARAM_SYMBOLS[goal_param]})")
            if unit == "K":
                target_prec = st.number_input(_tgt_lbl, 5.0, 500.0,
                                              _TARGET_DEFAULT[goal_param], 5.0,
                                              key=K(f"tgt_{goal_param}"))
            else:
                target_prec = st.number_input(_tgt_lbl, 0.01, 3.0,
                                              _TARGET_DEFAULT[goal_param], 0.01,
                                              key=K(f"tgt_{goal_param}"))
        target_sig = st.number_input(
            "Significance level (σ)", 1.0, 10.0, 3.0, 0.5, key=K("tsig"))

    with st.expander("Fisher forecast"):
        if _conden_ss:
            st.caption(
                "Unavailable with condensation on: the Fisher forecast "
                "needs d(spectrum)/d(parameter), and no differentiation "
                "method is valid through the condensation pin. Turn "
                "condensation off to free parameters.")
            fisher_params = []
        else:
            st.caption(
                "Expected 1σ constraints on atmosphere parameters from "
                "this observation, a linearized retrieval (Cramér-Rao "
                "bound) built from the spectrum's parameter derivatives"
                + (" via finite differences of the equilibrium tables. "
                   "Under PICASO each composition or temperature row is a "
                   "handful of fast table lookups plus the radiative "
                   "transfer -- well under a minute; a climate T_int row "
                   "re-runs the climate a few times and takes several "
                   "minutes. There is no MCMC and there are no priors."
                   if _pic else
                   " using the **differentiation method selected at the "
                   "top** of the sidebar. With finite differences, a "
                   "composition row re-solves the chemistry four times "
                   f"from scratch and takes about {_FD_COMP_MIN} to "
                   f"{_FD_COMP_MAX} minutes, and a "
                   "Kzz or temperature row adds four cold solves at about "
                   f"1 minute each ({_FD_THETA_MIN} to {_FD_THETA_MAX} "
                   "minutes per row). With AD, each "
                   "row is one warm-started derivative and takes about "
                   f"{_AD_ROW_MIN} to {_AD_ROW_MAX} minutes. There is no "
                   "MCMC and there are no priors."))
            if goal == "constrain" and marginalize:
                # defaults FILTERED by the live menu and the key carries the
                # provider (v18.1): under picaso lnKzz is not an option, and
                # Streamlit hard-raises on a default outside the options --
                # an unfiltered default crashed every constrain-goal render
                # after switching to the PICASO engine.
                fisher_extra = st.multiselect(
                    "Jointly free parameters", avail_free,
                    default=[p for p in ("lnZ", "dlnCO", "lnKzz")
                             if p in avail_free],
                    key=K(f"fx_{chem_provider}_{_tp_ss}_{int(_cloud_ss)}_"
                          f"{int(bool(_mie_ss))}"),
                    format_func=lambda n: forward.PARAM_LABELS[n],
                    help="The goal parameter is always included. More free "
                         "parameters = a more honest (wider) forecast; each "
                         "adds minutes of solves (see above).")
                fisher_params = sorted(set(fisher_extra) | {goal_param})
            elif goal == "constrain":
                # marginalization turned OFF in the Goal section: free only the
                # constraint parameter (condition on everything else).
                st.caption(
                    "Marginalization is off (Goal section): only the "
                    "constraint parameter is freed; every other parameter is "
                    "held fixed, so this is the optimistic conditional bound.")
                fisher_params = [goal_param]
            else:
                do_fisher = st.checkbox(
                    "Compute parameter constraints too", value=False,
                    key=K("dofish"),
                    help=("Adds constraint rows. Under the PICASO engine "
                          "each row is a handful of fast table lookups "
                          "plus the radiative transfer -- well under a "
                          "minute per parameter (a climate T_int row is "
                          "the exception: it re-runs the climate a few "
                          "times, several minutes)." if _pic else
                          "Adds Jacobian rows by the method selected at "
                          f"the top: FD ~{_FD_COMP_MIN}-{_FD_COMP_MAX} min "
                          "per composition parameter "
                          f"and ~{_FD_THETA_MIN}-{_FD_THETA_MAX} min per "
                          "Kzz/T parameter "
                          f"(photochemistry on or off); AD ~{_AD_ROW_MIN}-"
                          f"{_AD_ROW_MAX} min per "
                          "row (photochemistry on)."))
                fisher_params = st.multiselect(
                    "Free parameters", avail_free,
                    key=K(f"fp_{chem_provider}_{_tp_ss}_{int(_cloud_ss)}_"
                          f"{int(bool(_mie_ss))}"),
                    default=[p for p in ("lnZ", "dlnCO", "lnKzz")
                             if p in avail_free],
                    format_func=lambda n: forward.PARAM_LABELS[n],
                    ) if do_fisher else []
            # Loud slow-path flag: finite differences re-solve the chemistry per
            # row, so freeing many parameters can take a long time -- point the
            # user at AD before they launch a multi-hour run (AD is table-only
            # for picaso, so this only applies to the VULCAN engine).
            if fisher_params and jac_method == "fd" and not _pic:
                _n_comp = sum(p in ("lnZ", "dlnCO") for p in fisher_params)
                _n_theta = sum(p in ("lnKzz", "Tirr", "Tint", "log_kappa",
                                     "log_gamma") for p in fisher_params)
                _est_min = _n_comp * 7 + _n_theta * 4
                if _est_min >= 20:
                    st.warning(
                        f"Finite differences with {len(fisher_params)} free "
                        f"parameters is slow: roughly {_est_min}-"
                        f"{int(_est_min * 1.4)} min (each composition row "
                        f"re-solves the chemistry 4x, ~{_FD_COMP_MIN}-"
                        f"{_FD_COMP_MAX} min; each Kzz/T row "
                        f"~{_FD_THETA_MIN}-{_FD_THETA_MAX} min). Switch the "
                        "differentiation method to AD at "
                        f"the top (~{_AD_ROW_MIN}-{_AD_ROW_MAX} min/row, "
                        "photochemistry forced on) or "
                        "free fewer parameters.")

    st.markdown("### PICASO chemistry" if _pic else "### VULCAN chemistry")
    if not _pic:
        st.caption("Inputs to the VULCAN-JAX photochemical-kinetics forward "
                   "model (composition + transport + photochemistry → "
                   "steady-state abundances). The T-P profile is shared: it "
                   "also sets the ExoJAX radiative transfer below.")

    # Vertical layers: the chemistry AND the ExoJAX RT share this one grid
    # (art_nlayer is LOCKED to nz). Kept at the top of the chemistry controls,
    # always visible, so it is not mistaken for the PICASO climate solve's OWN
    # fixed 91-level internal grid (a separate thing -- see the RCB help).
    nz = st.number_input(
        "Vertical layers (chemistry + RT)", *forward.NZ_RANGE,
        forward.NZ_DEFAULT, 10, key=K("nz_pic" if _pic else "nz"),
        help="Number of pressure levels the chemistry AND the ExoJAX radiative "
             "transfer share -- the RT grid is LOCKED to this count. More layers "
             "resolve steep gradients at a slower run (~2 min at 100, ~2.5 at "
             "150). This is the ONLY vertical-layer setting. If you pick the "
             "PICASO radiative-convective climate T-P, that solver runs on its "
             "OWN fixed 91-level internal grid and is re-gridded onto these "
             "layers -- the 91 is unrelated to this number.")

    with st.expander("Atmosphere structure, T-P profile (shared with RT)"):
        # canonical_params refuses tp_mode='file' under the PICASO provider
        # (no default-temperature path in equilibrium); hide it here too --
        # GUI gating is convenience, canonical_params stays the hard guard.
        _tp_opts = (["guillot", "picaso_climate"] if _pic
                    else ["guillot", "file", "picaso_climate"])
        # Mirror canonical_params' default exactly, so "the defaults" mean the
        # same profile whether the run comes from the GUI or the API: the
        # measured WASP-39b table under the kinetics engine, Guillot otherwise.
        _tp_default = forward._default_tp_mode(
            {"planet": planet_key, "chem_provider": chem_provider})
        if st.session_state.get(_k("tp")) not in _tp_opts:
            st.session_state[_k("tp")] = _tp_default
        tp_mode = st.selectbox(
            "T-P profile", _tp_opts, index=_tp_opts.index(_tp_default),
            key=_k("tp"),
            format_func={"guillot": "Guillot (2010)",
                         "file": "Tabulated table (T-P, optional Kzz)",
                         "picaso_climate":
                             "PICASO radiative-convective (climate solve)"}.get,
            help="Sets the temperature the chemistry AND the radiative "
                 "transfer see. Three choices: a Guillot analytic profile, an "
                 "explicit tabulated table you provide, or a PICASO "
                 "radiative-convective climate solve that computes the "
                 "profile self-consistently from the stellar irradiation "
                 "(a GCM profile is never silently substituted; a tabulated "
                 "table also sets the hydrostatic grid and the equilibrium "
                 "initialization). A globally isothermal profile was removed: "
                 "it holds the deep CO/CH4/NH3 quench region at a single "
                 "temperature and biases disequilibrium abundances.")
        tp_kwargs = {}
        tp_file, tp_file_path, tp_file_ok = "", None, True
        if tp_mode == "guillot":
            # one definition shared with canonical_params, so the widget
            # default and the API default can never drift apart again
            tirr0 = planets.default_tirr(pdef)
            tp_kwargs["Tirr"] = st.number_input(
                "T_irr (K)", 800.0, 2500.0, tirr0, 20.0, key=_k("tirr"),
                help="≈ √2 × equilibrium temperature.")
            tp_kwargs["Tint"] = st.number_input(
                "T_int (K)", 50.0, 500.0, 100.0, 25.0, key=_k("tint"),
                help="Heat escaping the planet's own interior. (The PICASO "
                     "climate mode's equivalent knob defaults to 200 K; "
                     "the two are separate settings.)")
            tp_kwargs["log_kappa"] = st.number_input(
                "log₁₀ κ_IR (cm²/g)", -4.0, 0.0, -2.3, 0.1, key=_k("lk"))
            tp_kwargs["log_gamma"] = st.number_input(
                "log₁₀ γ (κ_vis/κ_IR)", -2.0, 0.3, -1.0, 0.05, key=_k("lg"))
        elif tp_mode == "picaso_climate":
            tp_kwargs["tint_cl"] = st.number_input(
                "Internal temperature T_int (K)", *forward.TINT_CL_RANGE,
                forward.TINT_CL_DEFAULT, 10.0, key=_k("tintcl"),
                help="How much heat leaks out of the planet's own interior, "
                     "expressed as a temperature. It is the one adjustable "
                     "structure parameter of the climate solve (and can be "
                     "a constraint target, computed by re-running the full "
                     "climate at nudged values). On strongly irradiated "
                     "planets the star dominates, so the upper atmosphere "
                     "barely notices T_int.")
            tp_kwargs["rfacv"] = st.selectbox(
                "Day-night heat redistribution (rfacv)",
                list(forward.RFACV_CHOICES), index=1, key=_k("rfacv"),
                format_func={0.0: "0 -- no irradiation (isolated interior)",
                             0.5: "0.5 -- full redistribution (default)",
                             1.0: "1 -- dayside-only"}.get,
                help="How the star's heat is shared around the planet: 0.5 "
                     "means winds spread it evenly over both hemispheres "
                     "(the usual choice), 1 means the day side keeps it "
                     "all, 0 means no starlight at all (an isolated, "
                     "self-heated object). The star settings above feed "
                     "this, so in climate mode they are part of the model "
                     "itself. Note: at the extreme settings the profile "
                     "can leave the modelable temperature range, and the "
                     "run then refuses loudly.")
            tp_kwargs["tio_vo"] = st.checkbox(
                "Include TiO/VO in climate opacity only", value=False,
                key=_k("tiovo"),
                help="Titanium- and vanadium-oxide absorb strongly in very "
                     "hot atmospheres; in cooler ones they rain out and "
                     "vanish. Off (default) = tables without them, right "
                     "for warm planets like WASP-39 b. This choice affects "
                     "ONLY how the climate solver computes heating -- the "
                     "final spectrum's molecule list never includes "
                     "TiO/VO either way.")
            tp_kwargs["climate_rcb"] = st.number_input(
                "Convective-zone seed (deep layer index)",
                *forward.CLIMATE_RCB_RANGE, forward.CLIMATE_RCB_DEFAULT, 1,
                key=_k("rcb"),
                help="A NUMERICAL SEED for PICASO's climate solver -- the "
                     "initial guess for the deepest layer that starts "
                     "convective -- as an index on the climate solve's OWN "
                     f"internal {forward.CLIMATE_N_LEVELS}-level grid. This is "
                     "NOT the 'Vertical layers' setting at the top of this "
                     "section (that is the chemistry+RT grid; the climate "
                     "profile is re-gridded onto it). The default "
                     f"({forward.CLIMATE_RCB_DEFAULT}) seeds the deepest few "
                     "layers: PICASO grows convective zones upward but cannot "
                     "shrink one seeded too shallow, so a shallow seed can "
                     "impose a spurious convective region (the ~1 bar kink of "
                     "the old shallow default). Deep seeds should converge to "
                     "the same profile. A shallow seed is refused only when it "
                     "drives the profile outside the "
                     f"[{forward.T_WINDOW[0]:.0f}, {forward.T_WINDOW[1]:.0f}] K "
                     "opacity window; an in-window shallow seed still runs and "
                     "carries that bias. Leave it at the default unless you are "
                     "probing seed sensitivity.")
            if science_mode == "emission":
                st.warning(
                    "Emission + climate mode: day-side light at some "
                    "wavelengths comes from the deep layers the convective-zone "
                    "seed sets. With the deep default the profile should be "
                    "seed-independent, but that is not yet certified per case, "
                    "so read deep-probing emission features as conditional on "
                    "the seed. Transmission is far less affected (mostly a "
                    "uniform shift the analysis absorbs into the reference "
                    "radius).")
        elif tp_mode == "file":
            # the shipped table is PER-PLANET; a planet without one may only
            # upload (and is told why its table is missing/refused)
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
                             forward.TP_FILE_UPLOAD: "Upload an array"}.get,
                help="The shipped table is the measured T-P + Kzz profile "
                     "VULCAN bundles for THIS planet, and is the default "
                     "structure wherever one exists. Uploads use the same "
                     "VULCAN atm format.")
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
                        "Column 1: pressure in dyne/cm² (1 bar = 10⁶), "
                        "monotonic (bottom of the atmosphere first, as "
                        "here, or top first). Column 2: temperature in K. "
                        "Column 3 (optional): Kzz in cm²/s, used when the "
                        "K_zz profile mode below is set to 'Tabulated'. "
                        "Any number of rows; the array is re-gridded onto "
                        "the layer count you choose. The two header lines "
                        "are required.")
                    st.download_button(
                        "Download this example (edit and re-upload)",
                        _tp_example, file_name="example_atm.txt",
                        key=_k("tpex"))
                up_tp = st.file_uploader(
                    "Upload an array: T-P (+ optional Kzz) as text",
                    type=["txt", "dat"],
                    key=_k("tpup"),
                    help="Same format as the example above (the VULCAN atm "
                         "format): units comment line, column-name line "
                         "'Pressure Temp' (+ optional 'Kzz'), then rows of "
                         "pressure in dyne/cm² (monotonic), T in K, Kzz in "
                         "cm²/s. Re-gridded onto the chosen layer count.")
                if up_tp is not None:
                    _raw_tp = up_tp.getvalue()
                    _sha_tp = hashlib.sha1(_raw_tp).hexdigest()[:16]
                    _dst_tp = forward._uploads_dir() / f"{_sha_tp}.txt"
                    _dst_tp.parent.mkdir(parents=True, exist_ok=True)
                    if not _dst_tp.exists():
                        _dst_tp.write_bytes(_raw_tp)
                    try:                       # loud validation, immediately
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
                        st.error(f"Table rejected: {e}")
                        tp_file_ok = False
                else:
                    st.warning("Upload a table to run in file mode.")
                    tp_file_ok = False
            st.caption(
                "Note: file mode has NO temperature Fisher row (a fixed "
                "table has no temperature parameter). Constraint forecasts "
                "are conditional on this profile being exactly right, so "
                "the reported sigmas are optimistic.")

    with st.expander("Composition"):
        if tp_mode == "picaso_climate":
            # EXACT-CK-NODE selectors: the climate correlated-k tables are
            # per-node files with no composition interpolation, so climate
            # mode only accepts compositions sitting exactly on a node
            # (canonical_params is the hard guard; these menus simply can't
            # produce anything else).
            from jwst_tool import picaso_chem as pchem
            _feh_opts = [x for x in pchem.FEH_NODES if -1.0 <= x <= 2.0]
            _feh = st.selectbox(
                "Metallicity", _feh_opts,
                index=_feh_opts.index(1.0), key=K("metnode"),
                format_func=lambda x: f"{10.0 ** x:.3g} × solar "
                                      f"([M/H] = {x:+.1f})",
                help="The climate solver's opacity tables exist only at "
                     "these fixed metallicity values and cannot be blended "
                     "between them, so climate mode offers exactly the "
                     "shipped choices.")
            met = float(10.0 ** _feh)
            _co_opts = [c for c in pchem.CO_NODES
                        if (f"feh{_feh:.1f}_co{c:.2f}"
                            in pchem.CK_NODES_AVAILABLE)]
            co_ratio = st.selectbox(
                "C/O", _co_opts,
                index=_co_opts.index(0.55) if 0.55 in _co_opts else 0,
                key=K(f"conode_{_feh:.1f}"),
                format_func=lambda c: f"{c:.2f}",
                help=("Same story as metallicity: only the shipped table "
                      "values are available (the most extreme "
                      "metallicities ship fewer C/O options)."
                      + (" With the PICASO engine, C/O is not offered as a "
                         "constraint parameter in climate mode -- the "
                         "chemistry tables kink exactly at these values, "
                         "so no trustworthy C/O derivative exists here; "
                         "metallicity constraints are fine, and the "
                         "VULCAN engine constrains C/O normally."
                         if _pic else
                         " The VULCAN engine computes its own chemistry, "
                         "so all constraint parameters work normally at "
                         "any of these values.")))
        elif _pic:
            met = st.number_input(
                "Metallicity (× solar)", *forward.PICASO_MET_RANGE, 10.0, 0.5,
                format="%.2f", key=K("met_pic"),
                help="How enriched the atmosphere is in elements heavier "
                     "than helium, from 0.1x to 100x the Sun's value. "
                     "PICASO's tables are computed at fixed metallicity "
                     "steps; values in between are smoothly interpolated.")
            co_ratio = st.number_input(
                "C/O (carbon/oxygen number ratio)",
                *forward.PICASO_CO_RANGE, 0.50, 0.01,
                format="%.3f", key=K("co_pic"),
                help="Carbon-to-oxygen ratio, capped at 1.10 by PICASO's "
                     "tables (the VULCAN engine goes to 2.0). One honest "
                     "quirk: the tables are computed at a handful of C/O "
                     "values, and the chemistry takes a sharp turn right "
                     "at the 0.55 table point -- so a C/O CONSTRAINT "
                     "forecast requested exactly at 0.55 will refuse (the "
                     "tool never reports a derivative it cannot trust). "
                     "The default 0.50 sits between table points, where "
                     "everything works cleanly.")
        else:
            # Composition is fully STRUCTURAL (v13, one path for every
            # value): metallicity scales the cfg's O/C/N/S abundances
            # together, C/O then sets C_H = co * O_H, and FastChem
            # re-initializes at exactly that composition -- the
            # upstream-VULCAN workflow. No perturbative knob, no fixed-O
            # validity ceiling: C-rich (> 1) is the same code path. A corner
            # with no certified steady state errors loudly (longdy gate); it
            # can never return a wrong spectrum.
            met = st.number_input(
                "Metallicity (× solar)", 0.1, 100.0, 10.0, 0.5,
                format="%.2f", key=K("met"),
                help="Any value in [0.1, 100] × solar. Scales the network's "
                     "O/C/N/S abundances together (He "
                     "fixed); every value is a full FastChem re-initialization. "
                     "Reported in dex ([M/H]) by the constraint forecast. Far "
                     "corners (0.1x, 100x on cold profiles) may fail the "
                     "convergence gate loudly.")
            co_ratio = st.number_input(
                "C/O (carbon/oxygen number ratio)",
                0.10, 2.00, float(forward.CO_BASELINE), 0.05,
                format="%.3f", key=K("co"),
                help="Any value in [0.1, 2.0]. Total carbon/oxygen number ratio "
                     "N_C/N_O: sets C_H = C/O × "
                     "O_H at the metallicity-scaled oxygen, then the network "
                     "re-initializes. The default 0.549 is the network's "
                     "WASP-39b elemental set (Tsai et al. 2023). Carbon-rich "
                     "values (> 1) work "
                     "too, but near C/O = 1 solves slow down and derivatives "
                     "are ill-conditioned: constrain C/O per side, not across "
                     "it.")

    if _pic:
        # Equilibrium provider: the kinetics sections (mixing,
        # photochemistry, condensation, boundary conditions) do not
        # exist -- canonical_params refuses explicit requests, the GUI
        # simply never offers them. Quench/lnKzz is a deferred feature
        # (docs/picaso_roadmap.md in the repo).
        kzz_mode, kzz_x = "const", 1.0
        kzz_const, kzz_kmax, kzz_plev, kzz_kdeep = 1.0e9, 0.0, 0.0, 0.0
        use_photo, sl_angle_deg, f_diurnal = False, 83.0, 1.0
        use_moldiff = use_vm_mol = use_condense = use_settling = False
        diff_esc, top_flux, bot_flux = [], [], []
        yconv_cri = forward.YCONV_DEFAULT   # equilibrium: no iterative solver
    else:
        with st.expander("Vertical mixing (K_zz)"):
            _kzz_opts = ["const", "Pfunc", "JM16"]
            # tabulated Kzz needs the tabulated T-P table (its Kzz column)
            _kzz_file_ok = tp_mode == "file"
            if _kzz_file_ok:
                _kzz_opts.append("file")
                # Same rule as canonical_params: a table that carries Kzz
                # supplies the mixing profile, so it is the default rather
                # than a flat stand-in. Seed session_state on first render
                # (Streamlit ignores index= once the key exists).
                if _k("kzzmode") not in st.session_state:
                    st.session_state[_k("kzzmode")] = "file"
            elif st.session_state.get(_k("kzzmode")) == "file":
                st.session_state[_k("kzzmode")] = "const"
            kzz_mode = st.selectbox(
                "K_zz profile", _kzz_opts,
                index=_kzz_opts.index("file") if _kzz_file_ok else 0,
                key=_k("kzzmode"),
                format_func={"const": "Constant",
                             "Pfunc": "Power law in P (Pfunc)",
                             "JM16": "Moses-type P^-0.5 (JM16)",
                             "file": "Tabulated (Kzz column of the T-P table)"}.get,
                help="Eddy-diffusion profile. Constant is the validated "
                     "baseline; Pfunc rises as P^-0.4 above a chosen level; "
                     "JM16 rises as P^-0.5 above 300 mbar with a deep floor; "
                     "'Tabulated' uses the Kzz column of the tp_mode='file' "
                     "table (only offered in file mode). The Fisher lnKzz row "
                     "is a multiplicative scale of the WHOLE profile in every "
                     "mode.")
            kzz_const = kzz_kmax = kzz_plev = kzz_kdeep = 0.0
            kzz_x = 1.0
            if kzz_mode == "const":
                log_kzz = st.number_input(
                    "log₁₀ K_zz (cm²/s)", 6.0, 12.0, 9.0, 0.25,
                    key=_k("kzz"),
                    help="Constant eddy-diffusion coefficient: "
                         "stronger mixing quenches photochemical "
                         "gradients.")
                kzz_const = 10.0 ** log_kzz
            elif kzz_mode == "Pfunc":
                kzz_kmax = 10.0 ** st.number_input(
                    "log₁₀ deep K_zz (cm²/s)", 4.0, 11.0, 5.0, 0.25,
                    key=_k("kzkmax"),
                    help="Deep-atmosphere Kzz; above the transition level the "
                         "profile rises as (P_lev/P)^0.4.")
                kzz_plev = 10.0 ** st.number_input(
                    "log₁₀ transition level (bar)", -5.0, 2.0, -1.0, 0.25,
                    key=_k("kzplev"),
                    help="Pressure above which Kzz starts rising (VULCAN "
                         "Pfunc K_p_lev).")
            elif kzz_mode == "JM16":
                kzz_kdeep = 10.0 ** st.number_input(
                    "log₁₀ deep-floor K_zz (cm²/s)", 4.0, 11.0, 5.0, 0.25,
                    key=_k("kzkdeep"),
                    help="Deep floor of the Moses-type profile "
                         "Kzz = max(K_deep, 1e5 (300 mbar/P)^0.5).")
            else:
                st.caption("Kzz is read from the Kzz column of the uploaded "
                           "array / selected table (rejected loudly if it has "
                           "no Kzz column).")
            if kzz_mode != "const":
                kzz_x = 10.0 ** st.number_input(
                    "log₁₀ K_zz scale factor", -1.0, 1.0, 0.0, 0.05,
                    key=_k("kzzx"),
                    help="Multiplies the whole profile (the same on-graph "
                         "direction the Fisher lnKzz row uses); 0 = the profile "
                         "as specified.")

        with st.expander("Photochemistry & transport"):
            if jac_method == "ad":
                st.session_state[K("photo")] = True   # AD needs photolysis ON
            use_photo = st.checkbox(
                "Photochemistry (UV photolysis)", value=True, key=K("photo"),
                disabled=(jac_method == "ad"),
                help="Off = thermochemistry + transport only (no photolysis "
                     "products such as SO2). Detection and the default "
                     "finite-difference Fisher forecast work either way; the AD "
                     "differentiation method requires photolysis ON (its "
                     "validated tangent regime), so this is locked while AD is "
                     "selected.")
            sl_angle_deg = st.number_input(
                "Photolysis zenith angle (°)", 0.0, 89.0, 83.0, 1.0, key=K("sza"),
                disabled=not use_photo,
                help="Slant path of the stellar UV. 83° = terminator slant "
                     "(Tsai et al. 2023 W39b); smaller angles = more direct "
                     "illumination.")
            f_diurnal = st.number_input(
                "Diurnal photolysis factor", 0.1, 1.0, 1.0, 0.05, key=K("fdiur"),
                disabled=not use_photo,
                help="Multiplies every photolysis rate. 1.0 = permanent dayside "
                     "(tidally locked); 0.5 mimics day-night averaging.")
            use_moldiff = st.checkbox(
                "Molecular diffusion", value=True, key=K("moldiff"),
                help="Species-dependent molecular diffusion competing with Kzz "
                     "(sets the homopause; matters high up).")
            use_vm_mol = st.checkbox(
                "Upwind molecular-diffusion advection (vm_mol)", value=False,
                key=K("vmmol"), disabled=not use_moldiff,
                help="Adds the advective settling flux with upwind differencing, "
                     "refreshed in-loop (the hybrid vm_mol scheme; VULCAN-JAX's "
                     "own default since 2026-07-14). OFF reproduces this tool's "
                     "validated baseline chemistry; ON is the newer scheme, not "
                     "yet re-baselined for these forecasts, and mainly moves "
                     "heavy species in the upper atmosphere. Requires molecular "
                     "diffusion.")

        with st.expander("Convergence tolerance"):
            st.caption("Steady-state solver tolerance (same physics, finer "
                       "convergence). Vertical layers are set at the top of "
                       "this section.")
            yconv_cri = st.number_input(
                "Convergence tolerance (yconv)",
                1.0e-4, 1.0e-2, forward.YCONV_DEFAULT, 1.0e-4,
                format="%.1e", key=K("yconv"),
                help="Strict-branch steady-state criterion, any value in "
                     "[1e-4, 1e-2]. 1e-2 is the VULCAN master default; 1e-3 "
                     "(with more layers) is the strict tier for final mid-IR "
                     "numbers. Note the solver also accepts via VULCAN's "
                     "loose branch (residual < 0.1 with a near-zero slope and "
                     "settled photolysis flux), so this value is NOT a "
                     "guaranteed bound on the certified residual -- the "
                     "results panel reports the actual residual and the gate "
                     "that certified it. A run failing certification errors "
                     "loudly instead of returning an unconverged spectrum.")

        with st.expander("Condensation (detection-only)"):
            _conden_allowed = use_photo and use_moldiff and jac_method == "fd"
            if not _conden_allowed:
                st.session_state[K("conden")] = False
            use_condense = st.checkbox(
                "S8 condensation (sulfur rainout)", value=False, key=K("conden"),
                disabled=not _conden_allowed,
                help="Sulfur rainout with the standard VULCAN treatment. The "
                     "solver runs a condensation window, pins S8 and its "
                     "condensate across the whole column, and then converges "
                     "the remaining chemistry under the usual certification "
                     "gate. This is a forward-model option for detection "
                     "goals only. It cannot support parameter constraints, "
                     "because the pinned reservoir depends on the solver's "
                     "step history rather than on the input parameters alone, "
                     "so no derivative through it is trustworthy. One warning "
                     "applies. If the column is too hot to condense, the pin "
                     "still freezes sulfur at an arbitrary early value instead "
                     "of reducing to the no-condensation result. Use this only "
                     "for planets cool enough aloft for sulfur to condense.")
            if not _conden_allowed:
                st.caption(
                    "Condensation needs photochemistry and molecular diffusion "
                    "switched on, and the finite-difference method selected at "
                    "the top. A cold column without photochemistry never "
                    "reaches a certifiable steady state. The condensation "
                    "growth rate comes from the molecular-diffusion "
                    "coefficient. Selecting AD means you want derivatives, "
                    "which condensation cannot provide.")
            use_condense = bool(use_condense and _conden_allowed)
            st.caption(
                "If you need aerosol opacity in a constraint forecast, use the "
                "ExoJAX cloud deck in the radiative-transfer section instead.")

        with st.expander("Boundary conditions & escape (advanced)"):
            st.caption(
                "Upstream VULCAN boundary-condition machinery, all OFF by "
                "default (the validated baseline: closed column, no escape, no "
                "settling). Negligible for a typical hot Jupiter; these exist "
                "for escape, surface-flux, and settling studies. Every entry is "
                "cache-keyed.")
            _settle_ok = use_moldiff and not use_condense
            if not _settle_ok:
                st.session_state[K("settle")] = False
            use_settling = st.checkbox(
                "Gravitational settling", value=False, key=K("settle"),
                disabled=not _settle_ok,
                help="Adds the particle settling velocity to the transport "
                     "operator. Needs molecular diffusion; refused together "
                     "with condensation (the certified S8 recipe pins settling "
                     "off).")
            if not _settle_ok:
                st.caption("Settling needs molecular diffusion ON and "
                           "condensation OFF.")
            if not use_moldiff:            # escape flux ~ TOA Dzz; zero without moldiff
                st.session_state[K("descape")] = []
            diff_esc = st.multiselect(
                "Diffusion-limited escape at the top of atmosphere",
                list(forward.DIFF_ESC_CHOICES), default=[], key=K("descape"),
                disabled=not use_moldiff,
                help="Applies the classic diffusion-limited escape flux at the "
                     "TOA for the selected light species (H, H2, He). Needs "
                     "molecular diffusion (the escape flux is proportional to the "
                     "top-of-atmosphere Kzz-diffusion coefficient).")
            if not use_moldiff:
                st.caption("Escape needs molecular diffusion ON.")
            top_lines = st.text_area(
                "Top-boundary fluxes", value="", key=K("topflux"),
                placeholder="H2O 1.0e8",
                help="One species per line: 'SPECIES FLUX', flux in molecules "
                     "cm^-2 s^-1 (negative = outflux to space). Species must "
                     "exist in the SNCHO network; unknown names are refused "
                     "loudly at run time, never silently ignored.")
            bot_lines = st.text_area(
                "Bottom-boundary fluxes + deposition", value="", key=K("botflux"),
                placeholder="SO2 1.0e9 0.1",
                help="One species per line: 'SPECIES FLUX VDEP' (VDEP optional, "
                     "default 0), flux in molecules cm^-2 s^-1 (positive = "
                     "outgassing), deposition velocity in cm/s (surface sink).")

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
                st.error(f"Boundary-condition entry rejected: {e}")

    st.divider()
    st.markdown("### ExoJAX radiative transfer")

    with st.expander("Opacity, scattering & clouds"):
        _base_set, _extra_set = ((forward.MOLECULES, forward.EXTRA_MOLECULES)
                                 if not _pic else
                                 (picaso_chem.PICASO_MOLECULES,
                                  picaso_chem.PICASO_EXTRA_MOLECULES))
        st.caption(
            "RT opacity always includes the base set "
            f"**{' · '.join(_base_set)}** (solved on every run). The "
            f"opt-in extras are **{' · '.join(_extra_set)}**. "
            + ("No SO2 here: in settled (equilibrium) chemistry sulfur "
               "hides in H2S and OCS instead -- SO2 only exists because "
               "starlight keeps making it, which is the VULCAN engine's "
               "territory. " if _pic else "")
            + "Adding more is currently in development.")
        # live line-list availability for the CURRENT broadening choice (the
        # widget below; previous-run value via session_state, default "air")
        _mol_status = datacheck.molecule_linelist_status(
            list(_extra_set),
            broadening=st.session_state.get(K("broad"), "air"))
        _MOL_NOTE = {datacheck.OK: "opacity cached",
                     datacheck.AUTO: "downloads on first use",
                     datacheck.MISSING: "engine data missing"}
        extra_mols = st.multiselect(
            "Extra RT molecules", list(_extra_set), default=[],
            key=K(f"xmols_{chem_provider}"),
            format_func=lambda m: f"{m}  ({_MOL_NOTE[_mol_status[m]]})",
            help="Added to the base opacity set (the chemistry always solves "
                 "them). C2H2/HCN matter at high C/O, H2S at 3.8-4.6 µm "
                 "(already in the base set under PICASO), NH3 on cool "
                 "(≲900 K) planets, and OCS is the second equilibrium "
                 "sulfur carrier (~4.85 µm). Each entry shows whether its "
                 "HITRAN line list is already cached locally or will be "
                 "downloaded on first use (~10-15 s each, network required).")
        if science_mode == "emission":
            # canonical_params forces Rayleigh OFF in emission (the pure-
            # absorption day-side solver has no scattering channel) -- show
            # the forced state instead of a checked-but-ignored box (v18.1
            # review)
            st.session_state[K("rayl")] = False
        use_rayleigh = st.checkbox(
            "H₂/He Rayleigh scattering", value=True, key=K("rayl"),
            disabled=(science_mode == "emission"),
            help="Zero-free-parameter known physics; matters shortward of "
                 "~1.5 µm (the SOSS blue end). Leave ON except for "
                 "comparisons. In EMISSION mode this is off and locked: the "
                 "day-side flux solver is pure absorption with no "
                 "scattering channel.")
        cloud_on = st.checkbox(
            "Power-law cloud/haze opacity", value=False, key=K("cloud"),
            help="ExoJAX power-law retrieval cloud, uniformly mixed: "
                 "κ(ν) = κ₀·(ν/ν₀)^α per gram of atmosphere (no cloud-top "
                 "pressure or particle microphysics). With the deck on, its "
                 "two parameters can be FREED in the Fisher forecast "
                 "(marginalized over) -- cheap RT-only Jacobian rows; leave "
                 "them out of the free list to condition on a known deck.")
        if cloud_on:
            log_kappa_cloud = st.number_input(
                "log₁₀ κ_cloud (cm²/g at 3.5 µm)", -4.0, 2.0, -1.0, 0.1,
                key=K("ck"),
                help="Gray amplitude. τ=1 pressure ≈ g/(κ·10⁶) bar: at WASP-39b "
                     "gravity, −1 → ~4 mbar deck, −3 → ~0.4 bar.")
            alpha_cloud = st.number_input(
                "Cloud spectral slope α (κ ∝ ν^α)", 0.0, 4.0, 0.0, 0.25,
                key=K("ca"),
                help="0 = gray deck; 4 ≈ Rayleigh-like small-particle haze.")
        else:
            log_kappa_cloud, alpha_cloud = -1.0, 0.0
        # Mie condensate deck (v16): physically-grounded condensate optics from
        # an exojax miegrid, an alternative (or addition) to the power-law deck.
        _mie_opts = [""] + list(forward.MIE_CONDENSATES)
        mie_condensate = st.selectbox(
            "Mie condensate cloud", _mie_opts, index=0, key=K("miec"),
            format_func=lambda c: "off" if not c else c,
            help="A physically-grounded condensate cloud (ExoJAX "
                 "PdbCloud/OpaMie): a column-uniform lognormal size "
                 "distribution with real refractive-index optics. Each "
                 "condensate needs a one-time Mie grid (python "
                 "tools/generate_miegrid.py <species>); a missing grid is "
                 "refused loudly at run time. Its three knobs can be freed in "
                 "the Fisher forecast (radius/dispersion ride the grid "
                 "interpolation, abundance is exact).")
        mie_log_rg, mie_sigmag, mie_log_mmr = -5.0, 2.0, -6.0
        if mie_condensate:
            if datacheck.miegrid_status([mie_condensate])[mie_condensate] \
                    != datacheck.OK:
                st.warning(
                    f"No Mie grid for {mie_condensate} yet; generate it once "
                    f"with 'python tools/generate_miegrid.py {mie_condensate}' "
                    "(~1 h) or the run will refuse.")
            mie_log_rg = st.number_input(
                "log₁₀ mean radius r_g (cm)", float(forward.MIE_LOG_RG_RANGE[0]),
                float(forward.MIE_LOG_RG_RANGE[1]), -5.0, 0.1, key=K("mierg"),
                help="Lognormal mean particle radius: −5 = 0.1 µm, −4 = 1 µm.")
            mie_sigmag = st.number_input(
                "size dispersion σ_g", float(forward.MIE_SIGMAG_RANGE[0]),
                float(forward.MIE_SIGMAG_RANGE[1]), 2.0, 0.05, key=K("miesg"),
                help="Geometric standard deviation of the lognormal size "
                     "distribution (≈1.05 near-monodisperse, 2 typical).")
            mie_log_mmr = st.number_input(
                "log₁₀ condensate MMR", float(forward.MIE_LOG_MMR_RANGE[0]),
                float(forward.MIE_LOG_MMR_RANGE[1]), -6.0, 0.25, key=K("miemmr"),
                help="Mass mixing ratio of the condensate (column-uniform).")
        # h2he uses separate per-molecule caches; count what is already local
        # for the molecule set in play (CO is cached ExoMol, ignores the knob)
        _h2he_mols = [m for m in forward.MOLECULES + extra_mols if m != "CO"]
        _h2he_cached = sum(
            1 for v in datacheck.molecule_linelist_status(
                _h2he_mols, broadening="h2he").values() if v == datacheck.OK)
        broadening = st.selectbox(
            "Line broadening perturber", ["air", "h2he"], index=0,
            key=K("broad"),
            format_func=lambda b: (
                "air (HITRAN terrestrial widths, default)"
                if b == "air" else
                f"H2/He blend (planetary; {_h2he_cached}/{len(_h2he_mols)} "
                "line-list caches present)"),
            help="'air' = HITRAN terrestrial widths (the default every "
                 "committed result used; a documented approximation for an "
                 "H2/He envelope); 'h2he' = planetary H₂/He blend, first use "
                 "downloads separate line-list caches, and molecules with no "
                 "H₂/He coverage raise loudly instead of silently falling "
                 "back.")
        nu_pts = st.number_input(
            "Native spectral sampling (nu_pts)", *forward.NU_PTS_RANGE,
            forward.NU_PTS_DEFAULT, 500, key=K("nupts"),
            help="Wavenumber grid points across 1-15 µm, any value in "
                 "[4000, 8000] (native R ≈ nu_pts/2.7: 4000 ≈ R 1500, "
                 "8000 ≈ R 3000), before binning to your "
                 "chosen display R below. Higher sampling sharpens narrow "
                 "features (the weak mid-IR bands) at more runtime.")

    with st.expander("Advanced RT (top pressure, integration, line wings)"):
        st.caption("ExoJAX modeling choices that can move the spectrum. The "
                   "defaults are the validated baseline; every choice is "
                   "cache-keyed, so changing one re-runs the model.")
        rt_ptop_bar = st.number_input(
            "RT top pressure (bar)",
            1.0e-9, 1.0e-6, 1.0e-8, 1.0e-9,
            format="%.1e", key=K("rtptop"),
            help="Where the RT column ends, any value in [1e-9, 1e-6] bar "
                 "(1e-8 is the default; the chemistry grid tops out at "
                 "1e-7 bar under the VULCAN engine and 1e-6 bar under "
                 "PICASO). Above the chemistry top the topmost abundances "
                 "and temperature are held constant (a standard "
                 "transmission-modeling convention, not chemistry). Too "
                 "low a top saturates "
                 "strong bands (CO2 4.3, CO 4.7 µm) into a flat wall: on "
                 "WASP-39b ~4.8% of the 4.2-5.2 µm band saturates at 1e-6 "
                 "bar vs 0.1% at the 1e-8 default.")
        if science_mode == "emission":
            # canonical_params pins simpson in emission (no transit chord
            # exists there) -- show the pinned state, not a live-looking
            # choice that is silently ignored (v18.1 review)
            st.session_state[K("rtint")] = "simpson"
        rt_integration = st.selectbox(
            "Transit chord integration", ["simpson", "trapezoid"], index=0,
            key=K("rtint"), disabled=(science_mode == "emission"),
            format_func={"simpson": "Simpson (ExoJAX default)",
                         "trapezoid": "Trapezoid"}.get,
            help="Numerical scheme for the transit chord integral in ExoJAX "
                 "ArtTransPure. Simpson is higher order; trapezoid is the "
                 "conservative comparison choice. The difference is a "
                 "grid-convergence diagnostic: if it moves your answer, "
                 "raise the layer count. Locked in EMISSION mode: the "
                 "day-side flux has no transit chord.")
        rt_dit_res = st.number_input(
            "Line-wing (broadening) grid resolution",
            0.1, 1.0, 1.0, 0.1, format="%.1f", key=K("rtdit"),
            help="Any value in [0.1, 1.0]. "
                 "PreMODIT broadening-parameter grid spacing "
                 "(dit_grid_resolution). Smaller resolves pressure-broadened "
                 "line wings finer at a slower opacity build; 1.0 is the "
                 "value every validated result here used, 0.2 is ExoJAX's "
                 "own default.")

    st.divider()
    st.markdown("### Instrument & noise")
    st.caption(f"The JWST measurement itself: which modes, how many {_evw}s, "
               "detector saturation, and the Pandeia/PandExo noise model. "
               "Independent of the atmosphere physics above.")

    with st.expander("Observation & instrument modes", expanded=True):
        mode_keys = st.multiselect(
            "Instrument modes",
            options=list(ins.MODES),
            default=ins.DEFAULT_MODES, key=K("modes"),
            help="The ETC computes every mode once per star, so adding modes "
                 "later is instant.",
            format_func=lambda k: (f"{ins.MODES[k]['label']}  "
                                   f"({ins.MODES[k]['wl_min']:g}-"
                                   f"{ins.MODES[k]['wl_max']:g} µm)"))
        r_bin = st.number_input(
            "Analysis binning R (λ/Δλ)", 25, 500, 100, 25, key=K("rbin"),
            help="NOT display-only: this sets the wavelength bins that "
                 "noise, the template S/N, and the Fisher forecasts are all "
                 "computed on (one binning operator for noise, model, and "
                 "Jacobians), as well as the plotted spectrum. Any value in "
                 "[25, 500]; 50-200 is the usual range.")
        n_transits = st.number_input(f"Number of {_evw}s", 1, 10, 1, 1,
                                     key=K("ntr"))
        t_base = st.number_input(f"Out-of-{_evw} baseline (hr)", 0.5, 10.0,
                                 float(t14), 0.1, key=_k("tbase"),
                                 help="Sets how well the stellar flux is "
                                      "anchored; PandExo convention is ≈ T14.")
        sat_limit = st.number_input(
            "Saturation limit (full-well fraction)",
            0.5, 0.95, 0.80, 0.05, key=K("sat"),
            help="Group selection keeps the brightest pixel "
                 "below this full-well fraction.")

    with st.expander("Noise model"):
        st.markdown("**Minimum noise floor** (PandExo convention)")
        st.caption("Applied as σ_final = max(σ_random, floor) on the final "
                   "binned uncertainties: a hard minimum, never added in "
                   "quadrature, never rescaled by the binning R, and never "
                   f"averaged below by adding {_evw}s.")
        floor_mode = st.radio(
            "Floor type", ["constant", "none", "file"], horizontal=True,
            key=K("floormode"),
            format_func={"constant": "Constant (ppm)", "none": "No floor",
                         "file": "Wavelength table"}.get)
        floor_table = None
        if floor_mode == "constant":
            floors = {k: st.number_input(ins.MODES[k]["label"], 0.0, 200.0,
                                         ins.MODES[k]["floor_ppm"], 5.0,
                                         key=K(f"floor_{k}"))
                      for k in mode_keys}
        elif floor_mode == "file":
            up = st.file_uploader(
                "Two columns: wavelength (µm), floor (ppm)",
                type=["txt", "csv", "dat"], key=K("floorfile"),
                help="Whitespace- or comma-separated. Linearly interpolated "
                     "to the final bin wavelengths; endpoint values extend "
                     "beyond the supplied range (PandExo behavior). Applied "
                     "to every selected mode.")
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
                    st.error(f"Floor table rejected: {e}")
                    floor_table = None
            if floor_table is None:
                st.warning("No valid floor table loaded, runs will use NO "
                           "floor until one is provided.")
        if floor_mode != "constant":
            floors = {k: None for k in mode_keys}
        if floor_mode == "file" and floor_table is not None:
            floors = {k: floor_table for k in mode_keys}

        st.markdown("**Empirical noise sensitivity factor** (× Pandeia σ, "
                    "optional)")
        st.caption("Default **1.0** = the Pandeia prediction as-is. Published "
                   "achieved-vs-predicted ratios (COMPASS/Gordon+2025 G395H "
                   "≈1.05-1.12×; Radica+2023 1.2× NIRISS SOSS; Bouwman+2023 "
                   "≈1.15× MIRI LRS) are program-specific reference points "
                   "for sensitivity studies, not a calibration. Proportional "
                   f"noise, averages down with {_evw}s, unlike the floor. "
                   "Recorded in the result metadata.")
        infl = {k: st.number_input(ins.MODES[k]["label"] + " ", 1.0, 3.0,
                                   float(ins.MODES[k].get("noise_infl", 1.0)),
                                   0.05, key=K(f"infl_{k}"))
                for k in mode_keys}

        st.markdown("**Experimental: correlated-floor scenarios**")
        scenario = st.radio(
            "Systematics scenario", list(noise_mod.SCENARIOS),
            format_func=lambda s: noise_mod.SCENARIOS[s]["label"],
            key=K("scenario"), label_visibility="collapsed")
        st.caption("The default (**random**) is the exact diagonal, "
                   "PandExo-style noise model and is what the headline "
                   "numbers use. The correlated presets re-allocate the part "
                   "of the variance the floor adds (the floor *excess*) into "
                   "a spectrally smooth kernel at identical per-bin totals. "
                   "They are stated assumptions, **not calibrated JWST "
                   "systematics models**, for stress-testing how rankings "
                   "move with correlation structure; conservative also "
                   "profiles per-segment slopes.")

    with st.expander("Display"):
        show_noise = st.checkbox("Show simulated noise realization", value=False,
                                 key=K("shownoise"))
        seed = st.number_input("Realization seed", 0, 9999, 0, key=K("seed"))

    st.button("Reset all settings", on_click=_reset_all,
              help="Back to the defaults (also clears the current results).")

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

try:
    cached = forward.load_result(params) is not None
    params_error = None
except (ValueError, RuntimeError) as e:  # stale widget combo mid-rerun, or a
    cached, params_error = False, str(e)  # missing/invalid T-P table
if tp_mode == "file" and not tp_file_ok and params_error is None:
    params_error = "file-mode T-P selected but no valid table is loaded"

# rough runtime hint keyed off the resolution knobs (old fast ~1.8, high ~2.8 min)
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
# Jacobian rows: fd = 4 re-init build+solve cycles per composition row and
# 4 cold solves per Kzz/T-P row (picaso: 4 fast table re-evaluations; the RT
# call dominates); Tint_cl = 4 full climate re-solves; the cloud AND Mie deck
# rows are RT-only (~seconds); ad = ~1 warm jvp per solve row
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
col_btn, col_note = st.columns([1, 3])
run_clicked = col_btn.button("Run", type="primary", width="stretch")
col_note.caption(f"**{planet_label}**, {grid_lbl}, model spectrum: "
                 f"**{est}**. ETC noise is cached per star.")


# ---------------------------------------------------------------------------
# Compute on click
# ---------------------------------------------------------------------------
def compute():
    if params_error:
        st.error(f"Invalid parameter combination: {params_error}")
        return None
    if not mode_keys:
        st.error("Select at least one instrument mode.")
        return None
    # v18.1 public-instance protection: heavy subprocesses (forward + ETC)
    # hold ONE concurrency slot for their whole duration; when every slot is
    # busy the launch is declined instead of piling more solvers onto the
    # shared hardware. Cached results never need a slot.
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
            proc = subprocess.Popen(
                [sys.executable, str(TOOL_DIR / "forward.py"), str(pfile)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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

            _watch_proc(proc, _fwd_line, bar.tick)
            proc.wait()
            if proc.returncode != 0:
                status.update(label="Forward model failed", state="error")
                st.error("Forward model failed:\n\n```\n"
                         + "\n".join(lines[-25:]) + "\n```")
                return None
            bar.done()
            status.update(label="Forward model done", state="complete")
        model = forward.load_result(params)
        if model is None:
            st.error("Forward model finished but produced no cache file.")
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
    if out is not None:
        st.session_state["out"] = out
        st.session_state["out_meta"] = dict(
            goal=goal, target=target_mol, goal_param=goal_param,
            target_prec=target_prec, target_sig=target_sig,
            n_transits=n_transits, show_noise=show_noise, seed=seed,
            r_bin=r_bin, planet=planet_label, scenario=scenario,
            floor_mode=floor_mode)

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
if "out" not in st.session_state:
    st.info("Pick a planet and a science goal in the sidebar, then press **Run**.")
    st.stop()

out = st.session_state["out"]
meta = st.session_state["out_meta"]
model, results = out["model"], out["results"]
goal_r = meta.get("goal", "detect")
# the atmosphere's absolute C/O, for the dlnCO -> absolute-C/O display
# conversion (sigma_CO = C/O * sigma_lnCO); since v13 params_json carries it
# directly as co_ratio (composition is structural)
_cpj = json.loads(str(model["params_json"]))
# Event word for the CACHED run's results (the sidebar radio may have moved
# since the run; the stored canonical params are the truth for this output).
_ev = ("eclipse" if str(_cpj.get("science_mode", "transmission")) == "emission"
       else "transit")
_tt_col = f"{_ev}s → target"
co_eval = float(_cpj.get("co_ratio", forward.CO_BASELINE))

# Staleness guard: results persist in session_state across sidebar edits, so the
# spectrum shown can be from DIFFERENT settings than the sidebar now reads --
# most visibly the transmission/emission GEOMETRY (and a failed run, e.g. an
# emission corner that refuses, leaves the previous result on screen). Say so
# loudly instead of silently showing a transmission spectrum under an "emission"
# sidebar.
try:
    _shown_stale = forward.params_key(params) != forward.params_key(_cpj)
except (ValueError, RuntimeError):
    _shown_stale = True   # the current sidebar settings do not even validate
if _shown_stale:
    _shown_sci = str(_cpj.get("science_mode", "transmission"))
    _geom = (f" The sidebar geometry is now **{science_mode}**, but this "
             f"spectrum is **{_shown_sci}**." if science_mode != _shown_sci
             else "")
    st.warning(
        "This spectrum is from your previous run; the sidebar has changed since "
        f"(or that run failed).{_geom} Press **Run** at the top to recompute "
        "with the current settings.")

for k, err in out["failed"]:
    first = str(err).strip().splitlines()[-1] if "Traceback" in str(err) else \
        str(err).strip().splitlines()[0]
    st.error(f"{ins.MODES[k]['label']}: failed, {first}")
    with st.expander(f"{ins.MODES[k]['label']} details"):
        st.code(str(err)[-2500:])
for k, reason in out["unusable"]:
    st.warning(f"**{ins.MODES[k]['label']}: unusable on this star**, {reason}.")

if not results:
    st.stop()

if out.get("provenance"):
    _pv = out["provenance"]
    st.caption(
        f"**{ins.BACKEND_STATUS}.** Backend: pandeia.engine "
        f"{_pv['engine_version']} + {_pv['refdata_name']} "
        f"(refdata {_pv['refdata_version']}), worker v{_pv['worker_version']} "
        ",  recorded in every noise cache.")

# chemistry certificate, provider-aware (v18): the VULCAN runner's own
# longdy per gated stage, or the PICASO provider's blend/normalization
# certificate -- everything shown passed its gate, or run_model would have
# raised
if "conv_longdy" in model and np.asarray(model["conv_longdy"]).size:
    _gate = float(np.asarray(model["conv_gate"], float)[0])
    st.caption(
        "Chemistry convergence check (every stage passed, or the run "
        "would have refused): " + "; ".join(
            f"{s}: residual {l:.3g} below the {_gate:g} threshold "
            f"({int(a)} solver steps)"
            for s, a, l in zip(model["conv_stages"], model["conv_accept"],
                               np.asarray(model["conv_longdy"], float)))
        + ".")
if "picaso_cert_json" in model:
    _pcert = json.loads(str(model["picaso_cert_json"]))
    st.caption(
        "PICASO equilibrium quality check (every item passed, or the run "
        "would have refused): built from the table grid points "
        f"{_pcert['nodes'][0]} | {_pcert['nodes'][1]} "
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
           "docs/picaso_roadmap.md)"
           if _pcert.get("suspect_cells_in_span") else "")
        + (f". {len(_pcert['corrections_applied'])} catalogued table "
           "fix(es) were applied (only ever to defects vetted and listed "
           "in docs/picaso_roadmap.md)"
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

fisher_names = ([str(x) for x in model["jac_names"][:-1]]
                if "jac_names" in model else [])
ok = [r for r in results if not r["saturated"]]

# --- verdict ---------------------------------------------------------------
if goal_r == "detect":
    tsig = float(meta.get("target_sig") or 3.0)
    ranked = sorted(ok or results, key=lambda r: -r["sigma_detect"])
    best = ranked[0]
    bsig = best["sigma_detect"]
    ntr = meta["n_transits"]
    verdict = (f"**Best mode for detecting {meta['target']} on "
               f"{meta.get('planet', '?')}: {best['label']}**, "
               f"{bsig:.1f}σ in {ntr} {_ev}{'s' if ntr > 1 else ''} "
               f"(target {tsig:g}σ; Δχ² = {bsig * bsig:.0f}; median precision "
               f"{best['median_sigma_ppm']:.0f} ppm per R={meta['r_bin']} bin).")
    if bsig >= tsig:
        st.success(verdict + "  Meets the target.")
    elif bsig > 0:
        # floor-aware transit solver: the photon term averages down with N, the
        # systematic floor does not -- a plain 1/sqrt(N) law was optimistic
        # exactly where it mattered (floor-dominated bright stars)
        tt = detect.transits_to_target(best, tsig)
        _scen = meta.get("scenario", "random")
        if tt["reachable"]:
            _win = ("" if tt.get("n_last") is None
                    or tt["n_last"] >= detect.N_TRANSITS_CAP else
                    f" Correlated-scenario window: past {tt['n_last']} "
                    f"{_ev}s the detection is lost again.")
            st.error(verdict + f"  Missing the target, {tt['n']} {_ev}s of "
                     f"{best['label']} would reach it (floor-aware estimate)."
                     + _win)
        else:
            _lim = f"{tt['sig_inf']:.1f}σ"
            st.error(verdict + "  Missing the target, and NO number of "
                     f"{_ev}s reaches it ({_never_reason(_scen, _lim)}). "
                     "Lower the floor, choose other modes, or relax the "
                     "target.")
    else:
        st.error(verdict + "  No signal in the selected bands, try other "
                 "modes or a different goal.")
else:
    gp = meta["goal_param"]
    unit = forward.PARAM_UNITS[gp]
    usp = (" " + unit) if unit else ""       # " dex"/" K", or "" for C/O (ratio)
    glabel = forward.PARAM_LABELS[gp]
    target = float(meta["target_prec"])
    tsig = float(meta.get("target_sig") or 3.0)
    with_jac = [r for r in results if r.get("jac_bins") is not None]
    # one saturation policy everywhere: a saturated mode is unusable data, so it
    # is excluded from BOTH the per-mode ranking and the combined forecast (the
    # combined row used to silently include modes the per-mode view dropped)
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
    if not per_mode:
        st.error(f"No selected mode constrains {glabel}, its Jacobian has no "
                 "signal in these bands. Try other modes or a different goal.")
        st.stop()
    bk = min(per_mode, key=per_mode.get)
    bs = per_mode[bk]
    ntr = meta["n_transits"]
    verdict = (f"**Best mode for constraining {glabel} on "
               f"{meta.get('planet', '?')}: {ins.MODES[bk]['label']}**, "
               f"±{bs:.3g}{usp} at {tsig:g}σ in {ntr} transit"
               f"{'s' if ntr > 1 else ''} (target ±{target:g}{usp} "
               f"at {tsig:g}σ).")
    if bs <= target:
        st.success(verdict + "  Meets the target.")
    elif np.isfinite(comb) and comb <= target:
        st.warning(verdict + f"  No single mode reaches the target, but the "
                   f"combination of all selected modes does "
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
            st.error(verdict + f"  Missing the target, {tt['n']} {_ev}s of "
                     f"{ins.MODES[bk]['label']} would reach it (floor-aware "
                     "estimate)." + _win)
        else:
            _lim = f"±{tsig * tt['sig_inf']:.3g}{usp} at {tsig:g}σ"
            st.error(verdict + "  Missing the target, and NO number of "
                     f"{_ev}s reaches it ({_never_reason(_scen, _lim)}). "
                     "Lower the floor, combine modes, or relax the target.")

# --- spectrum figure -------------------------------------------------------
st.subheader("Simulated eclipse emission spectrum"
             if str(_cpj.get("science_mode", "transmission")) == "emission"
             else "Simulated transmission spectrum")
wl = model["wl_um"]
order = np.argsort(wl)
wl_s, d_s = wl[order], model["depth"][order] * 1e6
_fname_base = f"jwst_tool_{_slug(meta.get('planet', 'planet'))}"

fig, ax = plt.subplots(figsize=(11, 4.4), dpi=200)
ax.plot(wl_s, d_s, color="#555555", lw=0.7, alpha=0.8, zorder=2,
        label="model (native)")
d_wo_s = None
if goal_r == "detect":
    mols = [str(x) for x in model["mols"]]
    d_wo_s = model["depth_wo"][mols.index(meta["target"])][order] * 1e6
    ax.plot(wl_s, d_wo_s, color="#999999", lw=0.9, ls="--", zorder=1,
            label=f"model without {meta['target']}")
rng = np.random.default_rng(int(meta["seed"]))
pt_lo, pt_hi = [], []            # plotted point extents (keep error bars in view)
for r in results:
    c = ins.MODE_COLOR[r["mode_key"]]
    y = r["depth"] * 1e6
    if meta["show_noise"]:
        y = y + rng.normal(0.0, r["sigma"] * 1e6)
    label = r["label"] + (" (saturated!)" if r["saturated"] else "")
    # plot at the response-weighted effective wavelength (matters near detector
    # gaps / steep throughput); falls back to the bin center if absent
    x = r.get("wl_eff", r["wl"])
    ax.errorbar(x, y, yerr=r["sigma"] * 1e6, fmt="o", ms=3.0, lw=1.0,
                color=c, ecolor=c, elinewidth=0.8, capsize=0, zorder=3, label=label)
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
# y-limits: the model in-window AND every plotted error bar (large-sigma
# points used to clip out of view)
sel = (wl_s >= lo * 0.97) & (wl_s <= hi * 1.03)
y_lo = min(float(d_s[sel].min()), min(pt_lo))
y_hi = max(float(d_s[sel].max()), max(pt_hi))
pad = 0.06 * (y_hi - y_lo)
ax.set_ylim(y_lo - pad, y_hi + 3 * pad)
ax.set_xlabel("wavelength (μm)")
_depth_lbl = ("eclipse depth (ppm)"
              if str(model.get("science_mode", "transmission")) == "emission"
              else "transit depth (ppm)")
ax.set_ylabel(_depth_lbl)
ax.grid(alpha=0.25, lw=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9,
          edgecolor="#dddddd")
st.pyplot(fig, width="stretch")
_spec_png = _fig_png(fig)
plt.close(fig)

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
        names = [r["label"] + (" (sat)" if r["saturated"] else "") for r in rs]
        vals = [r["sigma_detect"] for r in rs]
        cols = [ins.MODE_COLOR[r["mode_key"]] for r in rs]
        xrefs, xlabel = (3.0, 5.0), (f"conditional template S/N "
                                     f"({meta['n_transits']} transit"
                                     f"{'s' if meta['n_transits'] > 1 else ''})")
        fmt_v = lambda v: f"{v:.1f}σ"
        vline_target = float(meta.get("target_sig") or 3.0)
    else:
        st.subheader(f"Expected precision on {glabel}")
        items = sorted(per_mode.items(), key=lambda kv: -kv[1])   # best at top
        names = [ins.MODES[k]["label"] for k, _ in items]
        vals = [v for _, v in items]
        cols = [ins.MODE_COLOR[k] for k, _ in items]
        if np.isfinite(comb):
            names.append("ALL SELECTED (combined)")
            vals.append(comb)
            cols.append("#555555")
        xrefs, xlabel = (), (f"expected ±{forward.param_axis(gp)} at {tsig:g}σ "
                             f"({meta['n_transits']} transit"
                             f"{'s' if meta['n_transits'] > 1 else ''}; "
                             "lower is better)")
        fmt_v = lambda v: f"{v:.3g}"
        vline_target = target
    fig2, ax2 = plt.subplots(figsize=(6.4, 0.55 * len(names) + 1.2), dpi=200)
    bars = ax2.barh(names, vals, color=cols, height=0.62)
    for b, v in zip(bars, vals):
        ax2.text(b.get_width() + max(vals) * 0.02,
                 b.get_y() + b.get_height() / 2, fmt_v(v),
                 va="center", fontsize=9, color="#333333")
    for ref in xrefs:
        if ref < max(vals) * 1.15:
            ax2.axvline(ref, color="#bbbbbb", lw=0.8, ls=":")
            ax2.text(ref, len(names) - 0.3, f"{ref:.0f}σ", fontsize=7,
                     color="#888888", ha="center", va="bottom")
    if vline_target is not None:
        ax2.axvline(vline_target, color="#e34948", lw=1.0, ls="--")
        ax2.text(vline_target, len(names) - 0.28, " target", fontsize=7,
                 color="#e34948", ha="left", va="bottom")
    ax2.set_xlim(0, max(max(vals), vline_target or 0) * 1.18 + 1e-12)
    ax2.set_xlabel(xlabel)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.25, lw=0.5)
    fig2.tight_layout()
    st.pyplot(fig2, width="stretch")
    _rank_png = _fig_png(fig2)
    plt.close(fig2)
    _metric = (f"sigma_detect_{meta['target']}" if goal_r == "detect"
               else f"precision_{gp}_at_{tsig:g}sigma")
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
    fig3, ax3 = plt.subplots(figsize=(3.4, 3.6), dpi=200)
    ax3.plot(model["T"], model["p_bar"], color="#2a78d6", lw=1.6)
    for tlim in (320.0, 2980.0):
        ax3.axvline(tlim, color="#cccccc", lw=0.8, ls=":")
    ax3.set_yscale("log")
    ax3.invert_yaxis()
    ax3.set_xlabel("temperature (K)")
    ax3.set_ylabel("pressure (bar)")
    ax3.grid(alpha=0.25, lw=0.5)
    ax3.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3, width="stretch")
    _tp_png = _fig_png(fig3)
    plt.close(fig3)
    st.caption(f"As modeled ({cpj.get('tp_mode', '?')} mode). Dotted lines: "
               "the [320, 2980] K opacity window, profiles outside it are "
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
        # transmission probes p <~ 0.1 bar (the tool's photosphere
        # convention); a hot deep adiabat below that is invisible to the
        # chord geometry and must not trip the ultra-hot warning
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
    if r.get("n_pix_full_sat_dropped"):
        notes.append(f"{r['n_pix_full_sat_dropped']} fully saturated pixels excluded")
    if r.get("n_pix_degenerate_dropped"):
        notes.append(f"{r['n_pix_degenerate_dropped']} degenerate-wavelength "
                     "pixels excluded")
    if n_part:
        notes.append(f"partial saturation in {n_part} bins")
    if r["warnings"]:
        notes.append("; ".join(list(r["warnings"])[:2]))
    row = {"mode": r["label"],
           "band (μm)": f"{r['wl'].min():.2f}-{r['wl'].max():.2f}"}
    # NOTE: this column must stay all-string -- mixing int and str values makes
    # streamlit's Arrow serialization fail (loud pyarrow tracebacks per render)
    if r.get("lsf_applied"):
        notes.append("model blurred to native R (LSF)")
    if r.get("n_segments", 1) > 1:
        notes.append(f"{r['n_segments']} detector segments (offset per segment)")
    if goal_r == "detect":
        row["σ_detect"] = round(r["sigma_detect"], 1)
        _proj = r.get("sigma_detect_proj", float("nan"))
        if np.isfinite(_proj):
            row["σ_detect (proj)"] = round(_proj, 1)
        # experimental correlated scenarios stay OUT of the headline table
        # unless the user explicitly selected one
        if meta.get("scenario", "random") != "random":
            for _sc, _v in r.get("sigma_detect_by_scenario", {}).items():
                if _sc != r.get("scenario"):
                    row[f"σ ({_sc}, exp.)"] = round(_v, 1)
        _t = float(meta.get("target_sig") or 3.0)
        if r["sigma_detect"] > 0:
            _tt = detect.transits_to_target(r, _t)
            row[_tt_col] = _transits_cell(
                _tt, meta.get("scenario", "random"), f"{_tt['sig_inf']:.1f}σ")
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
            row[_tt_col] = _transits_cell(
                _tt, meta.get("scenario", "random"),
                f"±{tsig * _tt['sig_inf']:.3g}")
        else:
            row[_tt_col] = ""
    row.update({"median σ (ppm)": round(r["median_sigma_ppm"]),
                "bins": r["n_bins"], "ngroup": r["ngroup"],
                "cadence (s)": round(r["t_cycle_s"], 1),
                "notes": "; ".join(notes)})
    rows.append(row)
st.dataframe(rows, width="stretch", hide_index=True)
st.download_button("Mode details (CSV)", _csv_bytes(pd.DataFrame(rows)),
                   f"{_fname_base}_mode_details.csv", "text/csv",
                   key=K("dl_modes_csv"))
if goal_r == "detect":
    st.caption(
        "**σ_detect is a conditional matched-template S/N at the specified "
        "atmospheric state**, not a retrieval detection: √Δχ² of "
        "(full − without-molecule) over the mode's bins, with a constant depth "
        "offset, plus one step per detector segment (NRS1/NRS2), profiled out. "
        "**σ_detect (proj)** additionally projects out the temperature-structure "
        "and reference-radius (lnR0) Jacobian directions (chemistry and clouds "
        "stay fixed, still conditional); prefer it for narrow margins. σ_bin "
        "is the Pandeia photon+detector noise for in/out-of-transit "
        "integrations (× the optional sensitivity factor, default 1.0), with "
        "the minimum floor applied as a hard maximum on the final bins "
        f"(PandExo convention). '{_tt_col}' averages down the random "
        "term only, the floor is a lower bound at every N, so it can "
        "honestly read 'never'. Groups are chosen and verified against "
        "Pandeia's measured saturation. Because T, clouds, and the other "
        "abundances are not re-fit, a full retrieval detection can only be "
        "weaker."
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
        "floor as a hard lower bound, floor-limited targets read 'never' instead "
        "of an optimistic 1/√N estimate. Saturated modes are excluded from all "
        "forecasts."
    )

# --- Fisher forecast -------------------------------------------------------
# authoritative parameter order = the Jacobian rows as cached (canonical/sorted),
# NOT the multiselect order
if fisher_names and "jac" in model:
    tsig_f = float(meta.get("target_sig") or 3.0)
    st.subheader("Fisher parameter forecast")
    with_jac = [r for r in results if r.get("jac_bins") is not None]

    def _cell(n, s):
        v = tsig_f * fisher_mod.display_sigma(n, s, co_eval=co_eval)
        return "unconstrained" if not np.isfinite(v) or v > 1e4 else f"{v:.3g}"

    # two columns per parameter: marginalized (joint fit) + conditional
    # (others fixed) -- both read off the SAME nuisance-augmented Fisher matrix
    _fcols = [c for n in fisher_names
              for c in (f"±{forward.param_axis(n)} at {tsig_f:g}σ",
                        f"±{forward.param_axis(n)} (others fixed)")]

    def _row_cells(sig, cond):
        cells = {}
        for n in fisher_names:
            cells[f"±{forward.param_axis(n)} at {tsig_f:g}σ"] = _cell(n, sig[n])
            cells[f"±{forward.param_axis(n)} (others fixed)"] = _cell(n, cond[n])
        return cells

    frows = []
    usable_f = [r for r in with_jac if not r["saturated"]]
    for r in with_jac:
        if r["saturated"]:
            # shown for completeness, but a saturated mode contributes no usable
            # data -- same exclusion policy as the verdict + combined row
            frows.append({"mode": r["label"] + "  [saturated, excluded]",
                          **{c: "" for c in _fcols}})
            continue
        cond = {}
        sig = fisher_mod.mode_forecast(r, fisher_names, conditional=cond)
        frows.append({"mode": r["label"], **_row_cells(sig, cond)})
    fdiag = {}
    if len(usable_f) >= 2:
        cond = {}
        sig = fisher_mod.combined_forecast(usable_f, fisher_names, diag=fdiag,
                                           conditional=cond)
        frows.append({"mode": "ALL SELECTED (combined, non-saturated)",
                      **_row_cells(sig, cond)})
    st.dataframe(frows, width="stretch", hide_index=True)
    st.caption(
        "Two columns per parameter: the **marginalized** forecast (joint fit; "
        "all other parameters plus the lnR0 / per-segment offset / slope "
        "nuisances are free and marginalized out) and the **conditional** "
        "bound (everything else held fixed at truth). Both are Cramer-Rao "
        "lower bounds from the same Fisher matrix; a large gap between them "
        "means the parameter is degenerate with the others in that band, not "
        "that the spectral response is missing.")
    st.download_button("Fisher forecast (CSV)", _csv_bytes(pd.DataFrame(frows)),
                       f"{_fname_base}_fisher_forecast.csv", "text/csv",
                       key=K("dl_fisher_csv"))
    if fdiag:
        rank, dim = fdiag["fisher_rank"], fdiag["fisher_dimension"]
        st.caption(
            f"Numerical health (combined): Fisher rank {rank}/{dim}, condition "
            f"number {fdiag['condition_number']:.2g}."
            + (" **Rank-deficient, degenerate directions are reported as "
               "unconstrained, not as fake finite numbers.**" if rank < dim else ""))
    if "fd_err" in model:
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
                # v17: ungated single-central-difference RT rows report their
                # h-vs-2h metric as NaN (unmeasured), never a false 0.0
                _parts.append(f"{_sym} FD-RT (smooth, ungated)")
            else:
                _parts.append(f"{_sym} FD {e:.3f}")
        st.caption(
            "Jacobian provenance, per row: **FD** rows are central finite "
            "differences of independently converged solves, Richardson-"
            "combined, shown "
            "with their h-vs-2h consistency (0 = perfect, gate "
            f"{forward.FD_CONSISTENCY_TOL}); **AD** rows are warm-started "
            "forward-mode derivatives through the solver (photo-on regime, "
            "cross-validated against FD to 0.07-1.6% per row on W39b "
            "defaults, where the 1.6% is the metallicity row's stated "
            f"fixed-grid difference). Rows: {', '.join(_parts)}.")
    with st.expander("How to read this table"):
        st.markdown(
            f"- Each cell is the **expected ±uncertainty at {tsig_f:g}σ** "
            f"(= {tsig_f:g} × the Fisher 1σ) on that parameter if you fitted "
            "all listed parameters *simultaneously* to that mode's simulated "
            "data, a linearized best case (Cramér-Rao bound), so real "
            "retrieval posteriors can only be wider.\n"
            "- The sensitivities d(spectrum)/d(parameter) use the method "
            "you picked in the sidebar's **Differentiation method** menu, "
            "shown per row in the provenance line above: **central finite "
            "differences** of independently re-converged "
            "VULCAN-JAX solves (default; composition rows re-initialize "
            "the chemistry at the perturbed elemental abundances, the "
            "standard VULCAN workflow), or **warm-started forward-mode "
            "automatic differentiation** (the AD metallicity row holds the "
            "structural hydrostatic grid fixed, a stated 1.6%-level "
            "difference from FD on the defaults).\n"
            "- Each per-mode row also fits (and marginalizes over) a reference-"
            "radius nuisance **lnR0** plus one absolute-depth **offset per "
            "detector segment**, so the two-detector NIRSpec gratings (G395H, "
            "G235H) float independent **NRS1 and NRS2** steps, as every real "
            "G395H fit does (Moran+2023, Madhusudhan+2023). The combined row "
            "shares lnR0 across modes and keeps one offset per segment across "
            "all of them, that's what keeps multi-instrument combinations "
            "honest.\n"
            "- **No priors** are applied: a parameter with no spectral response "
            "in a mode's band reads *unconstrained* rather than a fake number.\n"
            "- Metallicity **[M/H]** and vertical mixing **log Kzz** are reported "
            "in **dex** (factors of 10); **C/O** is the absolute carbon/oxygen "
            "number ratio N_C/N_O (default ≈ 0.55, the network's WASP-39b "
            "elemental set from Tsai et al. 2023).\n"
            f"- σ is evaluated at the {_ev} count you set. Only the "
            "photon/detector term averages down with more transits; the "
            f"systematic floor does not, use the '{_tt_col}' column, "
            "not a 1/√N extrapolation."
        )
elif out.get("fisher_names"):
    st.info("Fisher forecast requested but the cached model has no Jacobian, "
            "press Run.")

# --- Adjoint diagnostics (reverse-mode AD) ----------------------------------
# High-dimensional sensitivities of the model just run: one reverse-mode
# adjoint solve gives dL/dlnk over EVERY reaction and dL/dT over EVERY layer
# -- questions finite differences cannot afford (thousands of re-runs).
# Analyzes the CACHED model's canonical params (_cpj), never the live sidebar.
st.subheader("Adjoint diagnostics (reverse-mode AD)")
with st.expander("Which reactions and temperatures control a molecule?"):
    st.caption(
        "One **reverse-mode adjoint solve** through the converged VULCAN-JAX "
        "state gives the sensitivity of a molecule's photosphere abundance "
        "to **every reaction rate** in the network and to **every layer's "
        "temperature** at once. These are the high-dimensional questions "
        "where automatic differentiation is the only practical tool. The "
        "Fisher Jacobians above, over a handful of parameters, use "
        "finite differences instead. The adjoint is validated "
        "upstream to 0.2-0.8% against finite differences on the WASP-39b "
        "SO2 and HD 189733 b CH4 benchmarks.")
    _adj_mols = [str(m) for m in model["mols"]]
    _adj_tgt = str(meta.get("target") or "")
    _adj_default = (_adj_tgt if _adj_tgt in _adj_mols
                    else "SO2" if "SO2" in _adj_mols else _adj_mols[0])
    adj_species = st.selectbox(
        "Target molecule", _adj_mols, index=_adj_mols.index(_adj_default),
        key=K("adjsp"),
        help="L = log10 VMR of this molecule at its peak-abundance layer "
             "inside the transit photosphere (1e-5 to 0.1 bar).")
    if _cpj.get("chem_provider") == "picaso":
        st.info(
            "Adjoint diagnostics are a VULCAN-JAX kinetics feature "
            "(reverse-mode AD through the steady-state solver: dL/d ln k "
            "over every reaction, dL/dT per layer). The PICASO equilibrium "
            "provider has no reaction network to differentiate. Re-run with "
            "the VULCAN engine to use this panel.")
        st.stop()
    if _cpj.get("use_condense"):
        st.info(
            "Adjoint diagnostics are unavailable for this model: it was run "
            "with condensation, whose pinned reservoir is frozen at a "
            "step-sequence-dependent transient, so no derivative through "
            "it is trustworthy. Re-run with condensation off to use this "
            "panel.")
        # st.stop() halts the whole render from here on: this panel MUST
        # remain the last section of the page (nothing below to lose)
        st.stop()
    adj = adjoint_diag.load_result(_cpj, adj_species)
    if adj is None:
        st.caption(
            "Not cached for this model + molecule. Cost: one chemistry "
            "re-solve with an extended budget, plus the adjoint ensemble. "
            "The FIRST adjoint run on a machine also compiles the solver's "
            "step-VJP, which can take hours on CPU. The result lands in "
            "the persistent JAX compile cache, so later runs skip straight "
            "to the solve.")
        if st.button("Run adjoint diagnostics", key=K("adjrun")):
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
                    # no prior: the first run on a machine compiles the step-VJP
                    # (hours on CPU) while later runs skip it, so any pre-run
                    # estimate would be wrong one way or the other. The
                    # remaining-time readout is purely measured.
                    bar = _TimedBar(text="starting …")
                    adjoint_diag.ADJOINT_CACHE.mkdir(parents=True, exist_ok=True)
                    _apf = (adjoint_diag.ADJOINT_CACHE /
                            f"{adjoint_diag.adjoint_key(_cpj, adj_species)}"
                            ".params.json")
                    _apf.write_text(json.dumps(_cpj))
                    proc = subprocess.Popen(
                        [sys.executable, "-m", "jwst_tool.adjoint_diag",
                         str(_apf), adj_species],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    box = st.empty()
                    lines = []

                    def _adj_line(line):
                        m = _ADJ_PROG_RE.match(line)
                        if m:
                            bar.update(min(1.0, float(m.group(1))), m.group(2))
                        else:
                            lines.append(line)
                            box.code("\n".join(lines[-10:]))
                            bar.tick()

                    _watch_proc(proc, _adj_line, bar.tick)
                    proc.wait()
                    if proc.returncode != 0:
                        status.update(label="Adjoint diagnostics failed",
                                      state="error")
                        st.error("Adjoint diagnostics failed:\n\n```\n"
                                 + "\n".join(lines[-25:]) + "\n```")
                    else:
                        bar.done()
                        status.update(label="Adjoint diagnostics done",
                                      state="complete")
                        st.rerun()
            finally:
                # release even on failure: this runs inside the
                # long-lived Streamlit server, so a leaked slot
                # would stay held until a server restart
                _adj_slot.release()
    else:
        _trust = bool(adj["magnitudes_trusted"])
        # pair antisymmetry is an ADDITIONAL diagnostic (added 2026-07-21);
        # older cached npz predate it, so show it only when present. It is
        # diagnostic-only, never a trust gate (see adjoint_diag docstring).
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
        ax_adj.axhline(float(adj["loss_p_bar"]), ls=":", lw=0.8, color="gray")
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
