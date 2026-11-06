"""The Run button's work: progress readout, worker subprocesses, the forward model and ETC launches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import select
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import streamlit as st

from jwst_tool import (
    datacheck,
    detect,
    instruments as ins,
    jacobian,
    model_cache,
    model_defs,
    noise as noise_mod,
    planets,
    proc as proc_mod,
    run_params,
    runlimit,
    share_config,
)
from jwst_tool.gui_common import K, slug
from jwst_tool.gui_sources import data_label

if TYPE_CHECKING:
    from collections.abc import Callable

    from jwst_tool.gui_inputs import Atmosphere, Goal, Observation, Solver, Target

# The VULCAN wall-time constants below were measured on the maintainer's
# laptop; the HF Space (SPACE_ID set there and only there) is slower, so the
# progress-bar prior is scaled by this factor. The forward model's PROG weights are
# relative fractions and need no scaling.
_RUNTIME_SCALE = 2.5 if os.environ.get("SPACE_ID") else 1.0


# PICASO Guillot/file run, measured on daw: 94 s with three FD rows
_PICASO_BASE_S, _PICASO_ROW_S = 40.0, 18.0


_PROG_RE = re.compile(r"\[fwd\] PROG ([0-9.]+) (.*)")


def _fmt_clock(s: float) -> str:
    """Fixed-width duration, always 9 chars: ' HH:MM:SS'.

    The ONLY duration formatter here: a variable-width format changes string
    length at unit boundaries, which makes the progress row -- and every
    widget below it -- jitter on each tick.
    """
    s = max(0, round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{min(h, 99):3d}:{m:02d}:{sec:02d}"


# Fixed pixel height for the streaming solver/ETC log boxes: the content
# grows line by line, and an auto-height box moves everything under it.
_LOG_BOX_PX = 220


class _TimedBar:
    """st.progress wrapper that appends a live elapsed / time-remaining readout to every label.

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
    reflows to a second line moves every widget below it.
    """

    _STAGE_W = 46  # stage-label field; long labels are truncated

    def __init__(self, prior_total_s: float | None = None, text: str = "starting ...") -> None:
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
                # Hand over to the measured pace by a THIRD of the way in, not
                # halfway. The prior is a generic pre-run guess and it cannot
                # see the two things that dominate a slow run -- an FD stencil
                # point that rides the step cap before its escalation fires,
                # and an AD row that falls to the cadence-1 retry -- so it
                # underestimates badly there. Weighting it by (1 - frac) would
                # keep a wrong prior in charge of the countdown for most of
                # the run.
                w = min(1.0, 3.0 * self._frac)
                remaining = w * measured_left + (1.0 - w) * prior_left
            else:
                remaining = measured_left
        elif self._prior:
            remaining = max(self._prior - e, 0.0)
        stage = str(self._label)
        if len(stage) > self._STAGE_W:
            stage = stage[: self._STAGE_W - 1] + "…"
        left = _fmt_clock(remaining) if remaining is not None else "  --:--:--"
        # monospace + non-breaking spaces: constant pixel width, no reflow
        body = f"{stage:<{self._STAGE_W}}  elapsed {_fmt_clock(e)}  left {left}"
        return "`" + body.replace(" ", "\u00a0") + "`"

    def _render(self) -> None:
        self._bar.progress(min(1.0, self._frac), text=self._compose())

    def update(self, frac: float, label: str) -> None:
        self._frac, self._label = float(frac), label
        self._render()

    def tick(self) -> None:
        """Refresh the clock without new progress information."""
        self._render()

    def done(self) -> None:
        self._frac, self._label = 1.0, "done"
        self._render()


def _watch_proc(proc: subprocess.Popen, on_line: Callable[[str], None], on_tick: Callable[[], None]) -> None:
    """Dispatch each stdout line of ``proc`` to ``on_line``, calling ``on_tick`` at least once a second of silence.

    Selects on the raw pipe fd (os.read chunking) so the clock ticks through silent solver stages; never use a
    blocking readline loop (it freezes the readout).
    """
    fd = proc.stdout.fileno()
    tail = b""
    while True:
        ready, _, _ = select.select([fd], [], [], 1.0)
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


def _managed_proc(cmd: Any) -> Any:
    """``proc.terminating`` over a stdout-piped child.

    The worker must not outlive the script run that started it (see jwst_tool.proc).
    """
    return proc_mod.terminating(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))


TOOL_DIR = Path(__file__).resolve().parent  # the worker script lives here


def _record_refusal(message: str) -> None:
    """Record a settings refusal.

    One line in ``refusals.log`` and the same alert mail a busy instance sends (one shared one-per-hour limit). Best
    effort -- the refusal the user sees never depends on it. Once per distinct message per session: the catch
    re-runs on every widget change while the settings stay illegal.
    """
    if st.session_state.get("_last_refusal") == message:
        return
    st.session_state["_last_refusal"] = message
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with open(Path(ins.OUTPUT_DIR) / "refusals.log", "a") as fh:
            fh.write(f"{stamp} {message}\n")
    except OSError:
        pass
    runlimit.notify_refused(f"parameter refused: {message}")


@dataclass(frozen=True)
class RunRequest:
    """One Run's inputs: the model parameters, the star, the goal and observation records, and the validation result."""

    params: dict
    star: dict
    planet_label: str
    canon: dict | None
    params_error: str | None
    est_total_min: float
    goal_meta: dict
    obs_meta: dict
    run_sig: dict


def _model_params(t: Target, a: Atmosphere, g: Goal, s: Solver) -> dict:
    params = dict(
        planet=t.planet_key,
        science_mode=t.science_mode,
        network=a.network,
        star_teff=t.teff,
        star_logg=t.logg,
        star_feh=t.feh,
        nz=s.nz,
        yconv_cri=s.yconv_cri,
        yconv_min=s.yconv_min,
        rp_rjup=t.rp,
        gs_cgs=t.g_ms2 * 100.0,
        rstar_rsun=t.rstar,
        orbit_au=t.orbit_au,
        sflux=t.sflux,
        met_x_solar=a.met,
        co_ratio=float(a.co_ratio),
        kzz_mode=a.kzz_mode,
        kzz_x=a.kzz_x,
        kzz_const=a.kzz_const,
        kzz_kmax=a.kzz_kmax,
        kzz_plev=a.kzz_plev,
        kzz_kdeep=a.kzz_kdeep,
        tp_mode=a.tp_mode,
        tp_file=a.tp_file,
        tp_file_path=a.tp_file_path,
        fisher_params=g.fisher_params,
        jac_method=g.jac_method,
        use_photo=a.use_photo,
        sl_angle_deg=a.sl_angle_deg,
        f_diurnal=a.f_diurnal,
        use_moldiff=a.use_moldiff,
        use_vm_mol=a.use_vm_mol and a.use_moldiff,
        use_rayleigh=a.use_rayleigh,
        rt_ptop_bar=float(a.rt_ptop_bar),
        rt_integration=s.rt_integration,
        p_ref_bar=float(a.p_ref_bar),
        p_btm_bar=float(a.p_btm_bar),
        cloud_on=a.cloud_on,
        log_kappa_cloud=a.log_kappa_cloud,
        alpha_cloud=a.alpha_cloud,
        # Detect computes the removed-molecule spectrum for the TARGET
        # only; the score reads exactly one row. wo_mols is cache-keyed,
        # so switching the detection target is a real re-run; API callers
        # can pass wo_mols=None for the full block. Constrain reads none
        # of them and skips the block entirely.
        wo_mols=([g.target_mol] if g.goal == "detect" else []),
        extra_mols=a.extra_mols,
        **a.tp_kwargs,
    )
    if a.pic:
        # VULCAN-JAX settings are omitted, never sent at a neutral value; rfacv
        # rides a.tp_kwargs (climate only)
        for vk in (
            "network",
            "yconv_cri",
            "yconv_min",
            "sflux",
            "kzz_x",
            "kzz_kmax",
            "kzz_plev",
            "kzz_kdeep",
            "use_photo",
            "sl_angle_deg",
            "f_diurnal",
            "use_moldiff",
            "use_vm_mol",
        ):
            del params[vk]
        params.update(chem_provider="picaso", picaso_quench=a.picaso_quench)
    return params


def _validate(params: dict, a: Atmosphere) -> tuple[dict | None, str | None]:
    """Canonicalize the parameters, or say in one sentence why the current settings cannot run."""
    canon = None
    try:
        canon = run_params.canonical_params(params)
        params_error = None
    except (ValueError, RuntimeError) as e:  # stale widget combo mid-rerun, or a
        params_error = str(e)  # missing/invalid T-P table
        _record_refusal(params_error)
    if a.tp_mode == "file" and not a.tp_file_ok and params_error is None:
        params_error = "file-mode T-P selected but no valid table is loaded"
    # A PICASO run needs its own environment and data (datacheck's optional
    # PICASO section).
    # fresh (about 40 stat calls), not the 1 h report cache: a PICASO install or
    # fetch must unblock the Run at once; VULCAN runs pay nothing
    pic_missing = datacheck.picaso_missing() if a.pic else []
    if a.pic and pic_missing and params_error is None:
        pic_labels = ", ".join(data_label(it) for it in pic_missing)
        params_error = (
            f"this hosted deployment is missing {pic_labels}; please report it to isaacmalsky@gmail.com."
            if os.environ.get("SPACE_ID")
            else f"PICASO is not set up here (missing: {pic_labels}); `jwst-tool data` lists what to install."
        )
    return canon, params_error


def _estimate_total_min(t: Target, a: Atmosphere, g: Goal, s: Solver) -> float:
    """Return the rough run time in minutes, the progress bar's prior; it is never shown as text."""
    # rough runtime prior for the progress bar, keyed off the resolution settings
    base_min = 0.1 + 0.010 * s.nz
    if s.yconv_cri <= 1.5e-3:  # strict convergence costs extra iterations
        base_min += 0.5
    base_min += 0.25 * len(a.extra_mols)  # k-table load + one more overlap fold
    # cool columns (<~900 K) converge much more slowly
    t_char = {"guillot": a.tp_kwargs.get("Tirr", 1560.0) / np.sqrt(2.0), "file": float(t.teq)}.get(a.tp_mode, 1100.0)
    if t_char < 900.0:
        base_min += 2.5

    # Jacobian-row cost model: fd = 4 solves per row; cloud rows are RT-only
    # (~seconds); ad = one warm re-converge + tangent per CHEMISTRY row on the
    # capped build, measured at 1.5 solve-equivalents per row (0.74 WASP-39 b
    # eclipse, 1.6 WASP-39 b transmission, 1.85 TOI-7169 b). T-P rows
    # take the FD branch under either method.
    solve_min = max(1.0, base_min * 0.5)
    rt_only = set(model_defs.CLOUD_FISHER_PARAMS)
    n_cloud_rows = sum(1 for n in g.fisher_params if n in rt_only)
    cloud_min = 0.2 * n_cloud_rows  # RT-only rows, either engine
    solve_rows = [n for n in g.fisher_params if n not in rt_only]
    n_fd_comp = sum(1 for n in solve_rows if n in jacobian.FD_COMP_PARAMS)
    n_fd_theta = len(solve_rows) - n_fd_comp
    if g.jac_method == "ad":
        fd_min = 1.5 * n_fd_comp * solve_min + n_fd_theta * 4 * solve_min + cloud_min
    else:
        fd_min = n_fd_comp * 4 * (solve_min + 0.8) + n_fd_theta * 4 * solve_min + cloud_min

    # The removed-molecule spectrum (TARGET ONLY on detect; constrain computes
    # none). This term sits OUTSIDE base_min on purpose: a Jacobian row computes
    # ONE spectrum, not the wo chain, so solve_min must not inherit it.
    n_mols_est = len(a.base_set) + len(a.extra_mols)
    wo_min = (0.05 * n_mols_est * (s.nz / model_defs.NZ_DEFAULT)) if g.goal == "detect" else 0.0

    # One scaled total is the progress-bar prior (no runtime text is shown).
    est_total_min = (base_min + wo_min + fd_min) * _RUNTIME_SCALE
    if a.pic:
        # The climate baseline is two solves (its gravity fixed point) and a
        # climate FD row four; the measured figures include the k-table loads.
        base_s, row_s = (
            (2 * model_defs.PICASO_CLIMATE_SOLVE_S, 4 * model_defs.PICASO_CLIMATE_SOLVE_S)
            if a.tp_mode == "picaso_climate"
            else (2 * model_defs.PICASO_QUENCH_COLUMN_S, 4 * model_defs.PICASO_QUENCH_COLUMN_S)
            if a.picaso_quench
            else (_PICASO_BASE_S, _PICASO_ROW_S)
        )
        est_total_min = ((base_s + row_s * len(solve_rows)) / 60.0 + cloud_min + wo_min) * _RUNTIME_SCALE
    return est_total_min


def _run_records(t: Target, g: Goal, o: Observation) -> tuple[dict, dict, dict]:
    """Return (goal record, observation record, staleness signature)."""
    # ONE description of everything a run consumes OUTSIDE the canonical model
    # parameters: the science g.goal and the observation setup. Built once and used
    # three times -- the shareable config, the stored run meta, and the staleness
    # guard -- so a new setting joins all three at once. The guard compares the
    # whole block minus display_only below, never a hand-picked subset: a
    # computed field it misses reaches the page silently as stale results.
    goal_meta = dict(
        goal=g.goal,
        target_mol=g.target_mol,
        target_sig=float(g.target_sig),
        goal_param=g.goal_param,
        target_prec=(None if g.target_prec is None else float(g.target_prec)),
        marginalize=bool(g.marginalize),
        do_fisher=bool(g.do_fisher),
        fisher_params=list(g.fisher_params),
        jac_method=g.jac_method,
    )
    obs_meta = dict(
        ks_mag=float(t.ks_mag),
        t14=float(t.t14),
        t_base=float(o.t_base),
        # the Pandeia star, recorded HERE for both science modes: the canonical
        # block zeroes star_teff/t.logg/t.feh in transmission (cache hygiene), so a
        # share file without these could not reproduce the noise simulation
        star_teff=float(t.teff),
        star_logg=float(t.logg),
        star_feh=float(t.feh),
        sat_limit=float(o.sat_limit),
        modes=list(o.mode_keys),
        n_transits=int(o.n_transits),
        r_bin=int(o.r_bin),
        floor_mode=o.floor_mode,
        # a 2-D table is recorded by CONTENT: the staleness guard reads this
        # block, and a fixed label would make two different uploaded tables
        # look like one run
        floors={
            k: (
                None
                if o.floors[k] is None
                else (
                    float(o.floors[k])
                    if np.isscalar(o.floors[k])
                    else "table:" + hashlib.sha1(np.ascontiguousarray(o.floors[k]).tobytes()).hexdigest()[:16]
                )
            )
            for k in o.mode_keys
        },
        # the PER-MODE widget values, not the composed product: the global scale is
        # recorded separately below, and share_config restores noise_infl into the
        # per-mode widgets -- writing the product here would re-multiply by the
        # global scale on every restore.
        noise_infl={k: float(o.infl_mode[k]) for k in o.mode_keys},
        noise_scale=float(o.noise_scale),
        show_noise=bool(o.show_noise),
        seed=int(o.seed),
        combos=[
            dict(name=str(c["name"]), modes=[str(m) for m in c["modes"]])
            for c in (st.session_state.get(K("combos")) or [])
        ],
    )
    # Observation fields that change only what is DRAWN from the cached run (the
    # mock draw, its o.seed, the named mode sets). They stay in obs_meta -- the
    # shareable config and floor_selected read it -- but the staleness guard
    # ignores them: they recompute nothing, so they can never make a result stale.
    display_only = ("show_noise", "seed", "combos")
    run_sig = dict(goal=goal_meta, observation={k: v for k, v in obs_meta.items() if k not in display_only})
    return goal_meta, obs_meta, run_sig


def build_request(t: Target, a: Atmosphere, g: Goal, o: Observation, s: Solver) -> RunRequest:
    """Assemble one Run's inputs from the sidebar."""
    params = _model_params(t, a, g, s)
    star = dict(teff=t.teff, log_g=t.logg, metallicity=t.feh, ks_mag=t.ks_mag)
    planet_label = planets.PLANETS[t.planet_key]["label"] if t.planet_key in planets.PLANETS else "custom planet"
    canon, params_error = _validate(params, a)
    goal_meta, obs_meta, run_sig = _run_records(t, g, o)
    return RunRequest(
        params=params,
        star=star,
        planet_label=planet_label,
        canon=canon,
        params_error=params_error,
        est_total_min=_estimate_total_min(t, a, g, s),
        goal_meta=goal_meta,
        obs_meta=obs_meta,
        run_sig=run_sig,
    )


def render_run_row(run_slot: Any, req: RunRequest, a: Atmosphere, o: Observation) -> bool:
    """Draw the validation messages, the Run button and the configuration download; return whether Run was clicked."""
    params_error = req.params_error
    mode_keys, floor_choice_made = o.mode_keys, o.floor_choice_made
    # Run row: validation messages, run button, review summary
    with run_slot:
        if params_error:
            st.error(f"Cannot run with the current settings: {params_error}")
        still_needs = []
        if not mode_keys:
            still_needs.append("an instrument mode (step 4 · Observation)")
        if not floor_choice_made:
            still_needs.append("a minimum noise floor type (step 4 · Observation, Noise model)")
        if still_needs:
            st.error("This run still needs: " + "; ".join(still_needs) + ".")
        # The config download sits BESIDE Run, in a column filled further down
        # (it needs _canon and the goal/observation blocks, which are built
        # below). Streamlit columns accept deferred writes, so the slot is
        # reserved here and the button lands in the right place on the page.
        col_btn, cfg_col, _ = st.columns([1, 1.4, 1.6])
        run_clicked = col_btn.button(
            "Run",
            type="primary",
            width="stretch",
            disabled=(bool(params_error) or not mode_keys or not floor_choice_made),
        )
    # The shareable configuration: built here (it needs req.canon and the g.goal /
    # observation blocks above) and rendered into the slot reserved beside the
    # Run button.
    if req.canon is not None:
        share = share_config.build_share(
            canon=req.canon,
            goal=req.goal_meta,
            observation=req.obs_meta,
            tp_table_text=(Path(a.tp_file_path).read_text() if a.tp_file_path else None),
            floor_table=(np.asarray(o.floor_table).tolist() if o.floor_table is not None else None),
            cloud_top_bar=a.cloud_top_bar,
        )
        with cfg_col:
            st.download_button(
                "Download configuration (JSON)",
                json.dumps(share, indent=2, default=str).encode(),
                f"jwst_tool_{slug(req.planet_label)}_config.json",
                "application/json",
                key=K("dl_config"),
                on_click="ignore",
            )
    return run_clicked


def compute(req: RunRequest, a: Atmosphere, t: Target, o: Observation) -> dict | None:
    # Run is disabled while params_error is set or no mode is selected.
    # Heavy subprocesses (forward + ETC) hold ONE concurrency slot from the
    # first heavy launch to the end of the run; when every slot is busy that
    # launch is declined. A fully cached run holds no slot -- it starts
    # nothing -- and every heavy call site goes through _take_slot, so a cache
    # MISS can still never start chemistry or Pandeia outside the limiter.
    """Run the request, holding at most one heavy slot, released on the way out."""
    slot = []
    try:
        return _compute_locked(slot, req, a, t, o)
    finally:
        for s in slot:
            s.release()


def _take_slot(slot: list, req: RunRequest, a: Atmosphere, o: Observation) -> bool:
    """Return true once this run holds a heavy-computation slot, acquiring one on the first heavy launch.

    Otherwise declines the launch and says so.
    """
    if not slot:
        s = runlimit.acquire("forward+etc")
        if s is None:
            runlimit.notify_refused(
                f"instance busy: {req.planet_label}, engine={a.chem_provider}, "
                f"tp_mode={a.tp_mode}, modes={list(o.mode_keys)}"
            )
            t0 = runlimit.oldest_start()
            since = (
                ""
                if t0 is None
                else f" The longest-running one started about {max(0.0, time.time() - t0) / 60.0:.0f} min ago."
            )
            st.error(
                f"This instance is already running {runlimit.MAX_CONCURRENT} "
                f"heavy calculations (it is shared, public hardware).{since} "
                "A default run takes 20-30 min here, so please try again "
                "later -- previously computed results stay instant."
            )
            return False
        slot.append(s)
    return True


def _run_forward_model(slot: list, req: RunRequest, a: Atmosphere, o: Observation) -> dict | None:
    """Return the cached model, solving it in a worker first on a miss; None when the run was declined or failed."""
    params, pic, est_total_min = req.params, a.pic, req.est_total_min
    model = model_cache.load_result(params)
    if model is None:
        if not _take_slot(slot, req, a, o):
            return None
        with st.status(
            f"Running {'PICASO' if pic else 'VULCAN-JAX'} + ExoJAX forward model …", expanded=True
        ) as status:
            # prior = the rough pre-run estimate (est_total_min, never
            # shown as text); the bar's remaining time converges to the
            # measured pace
            bar = _TimedBar(prior_total_s=est_total_min * 60.0, text="starting …")
            # unique per launch: a key-only name is SHARED between two
            # same-key runs, and one visitor's rewrite can truncate the file
            # under the other's subprocess mid-read
            pfile = model_cache.MODEL_CACHE / (f"{model_cache.params_key(params)}.params.{uuid.uuid4().hex[:8]}.json")
            model_cache.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            pfile.write_text(json.dumps(run_params.canonical_params(params)))
            box = st.empty()
            lines = []

            def _fwd_line(line: str) -> None:
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
                with _managed_proc([sys.executable, str(TOOL_DIR / "forward.py"), str(pfile)]) as proc:
                    _watch_proc(proc, _fwd_line, bar.tick)
                    proc.wait()
            finally:
                pfile.unlink(missing_ok=True)
            if proc.returncode != 0:
                status.update(label="Forward model failed", state="error")
                # the exception line of the subprocess traceback is the
                # tool's own user-facing sentence; the raw tail stays behind
                # a collapsed expander
                st.error(proc_mod.exception_sentence(lines, "the forward model stopped without a message."))
                with st.expander("Technical details"):
                    st.code("\n".join(lines[-25:]))
                return None
            bar.done()
            status.update(label="Forward model done", state="complete")
        model = model_cache.load_result(params)
        if model is None:
            st.error(
                "The forward model finished but produced no cache file. Re-run; if this repeats, report it as a bug."
            )
            return None
    return model


def _run_etc(slot: list, req: RunRequest, a: Atmosphere, o: Observation) -> dict | None:
    """Return the selected modes' ETC results, computing the missing ones; None when declined or failed."""
    star, mode_keys, sat_limit = req.star, o.mode_keys, o.sat_limit
    # ETC: ONLY the selected modes, cached per mode -- a later selection
    # change computes exactly the newly added modes.
    etc_missing = noise_mod.missing_modes(star, list(mode_keys), sat_limit=sat_limit)
    # launch=False: an entry that vanished between the two cache reads must
    # not start Pandeia without a slot; it falls into the slot path below
    etc = None if etc_missing else noise_mod.run_modes(star, list(mode_keys), sat_limit=sat_limit, launch=False)
    if etc is None:
        etc_missing = etc_missing or list(mode_keys)
        if not _take_slot(slot, req, a, o):
            return None
        with st.status(f"Running Pandeia ETC ({ins.BACKEND_STATUS.split(' /')[0]}) …", expanded=True) as status:
            # no reliable prior for the ETC; the remaining-time readout is
            # purely measured, appearing once the first mode completes
            bar = _TimedBar(text="starting the ETC …")
            box = st.empty()
            lines = []
            n_started = [0]

            def _cb(s: str) -> None:
                if s.startswith("[pandeia] ") and s.endswith("..."):
                    bar.update(
                        n_started[0] / len(etc_missing),
                        s.removeprefix("[pandeia] ").removesuffix("...") + f" ({n_started[0] + 1}/{len(etc_missing)})",
                    )
                    n_started[0] += 1
                else:
                    lines.append(s)
                    box.code("\n".join(lines[-8:]), height=_LOG_BOX_PX)
                    bar.tick()

            try:
                etc = noise_mod.run_modes(star, list(mode_keys), sat_limit=sat_limit, progress=_cb)
            except RuntimeError as _e:
                status.update(label="Pandeia ETC failed", state="error")
                # same handling as the forward model above: the first line is
                # the tool's own sentence, the worker's stderr tail stays
                # behind a collapsed expander
                st.error(str(_e).strip().splitlines()[0])
                with st.expander("Technical details"):
                    st.code(str(_e)[-2500:])
                return None
            bar.done()
            status.update(label="Pandeia ETC done", state="complete")
    return etc


def _evaluate_modes(model: dict, etc: Any, req: RunRequest, t: Target, o: Observation) -> dict:
    """Score every selected mode; one bad mode is reported and the rest still evaluate."""
    mode_keys, r_bin, n_transits, floors, infl = o.mode_keys, o.r_bin, o.n_transits, o.floors, o.infl
    target_mol, t14, t_base = req.goal_meta["target_mol"], t.t14, o.t_base
    t_in_s, t_out_s = t14 * 3600.0, t_base * 3600.0
    results, failed, unusable = [], [], []
    for k in mode_keys:
        if "error" in etc[k]:
            failed.append((k, etc[k]["error"]))
        elif etc[k].get("unusable") or not etc[k].get("wl"):
            unusable.append((k, etc[k].get("reason", "no usable pixels")))
        else:
            try:
                results.append(
                    detect.evaluate_mode(
                        k,
                        etc[k],
                        model,
                        target_mol,
                        r_bin,
                        t_in_s,
                        t_out_s,
                        n_transits,
                        floors[k],
                        noise_inflation=infl[k],
                    )
                )
            except detect.ModeUnusable as e:
                # not a failed calculation: this mode's data cannot support
                # the measurement, so it is excluded WITH its configuration
                unusable.append((k, str(e)))
            except Exception as e:
                # one bad mode must not kill the whole run -- report it with
                # its label + the actual reason, keep evaluating the rest
                failed.append(
                    (
                        k,
                        (
                            f"{type(e).__name__}: {e}\n\n"
                            f"(binning/noise evaluation for {k}; the other "
                            "modes are unaffected)"
                        ),
                    )
                )
    return dict(model=model, results=results, failed=failed, unusable=unusable)


def _compute_locked(slot: list, req: RunRequest, a: Atmosphere, t: Target, o: Observation) -> dict | None:
    model = _run_forward_model(slot, req, a, o)
    if model is None:
        return None
    etc = _run_etc(slot, req, a, o)
    if etc is None:
        return None
    return _evaluate_modes(model, etc, req, t, o)


def handle_run_click(req: RunRequest, t: Target, a: Atmosphere, g: Goal, o: Observation) -> None:
    """Run and store the result, or clear the previous one when the Run produced nothing."""
    goal, target_mol, goal_param = g.goal, g.target_mol, g.goal_param
    target_prec, target_sig = g.target_prec, g.target_sig
    n_transits, r_bin = o.n_transits, o.r_bin
    planet_label, obs_meta, run_sig = req.planet_label, req.obs_meta, req.run_sig
    out = compute(req, a, t, o)
    if out is None:
        # a Run that produced nothing leaves nothing stale on screen
        st.session_state.pop("out", None)
        st.session_state.pop("out_meta", None)
    else:
        st.session_state["out"] = out
        st.session_state["out_meta"] = dict(
            goal=goal,
            target=target_mol,
            goal_param=goal_param,
            target_prec=target_prec,
            target_sig=target_sig,
            n_transits=n_transits,
            r_bin=r_bin,
            planet=planet_label,
            # the limit the ramp search actually ran at, so the per-mode
            # saturation advisory is measured against THIS run
            sat_limit=obs_meta["sat_limit"],
            # the COMPLETE non-canonical input set, for the staleness guard
            run_sig=run_sig,
        )
