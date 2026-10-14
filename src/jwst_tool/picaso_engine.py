"""PICASO engine, host side: the alternative to VULCAN-JAX kinetics.

PICASO's radiative-convective climate solver supplies T(P) (or the run's
Guillot / tabulated profile does) and its on-the-fly equilibrium chemistry
(photochem, Gibbs minimization) supplies the abundances, optionally quenched
at one constant Kzz. The spectrum, binning, noise and Fisher are the tool's
own (ExoJAX + ExoMolOP correlated-k), so both engines share one opacity
dataset. The coupling is ONE-WAY: the climate solve uses PICASO's own
correlated-k tables, the spectrum the ExoMolOP tables; never call the pair
self-consistent.

The physics runs in ``picaso_worker.py`` inside the PICASO environment
(``JWST_TOOL_PICASO_PYTHON``; picaso 4.1 needs numpy >= 2). This module builds
the worker's requests, caches and certifies its results, and hands
``forward.run_model`` a chemistry object shaped like the VULCAN build
(``p_bar`` bottom to top, ``sidx``, ``species_masses``, ``T_base``) plus the
solved column ``y``.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from jwst_tool import datacheck, planets
from jwst_tool import instruments as ins
from jwst_tool import proc as proc_mod

PICASO_VERSIONS = {"picaso": datacheck.PICASO_RELEASE, "photochem": "0.9.2"}
WORKER_VERSION = 3      # bump when picaso_worker.py output changes
PICASO_CACHE = ins.OUTPUT_DIR / "picaso_cache"
# flock files stay container-local like runlimit's slots: the output dir may be
# a network volume without working advisory locks (the Space's bucket)
LOCK_DIR = Path(tempfile.gettempdir()) / "jwst_tool_picaso_locks"
# the climate correlated-k set picaso must load, every table of the pinned record
CK_MOLECULES = sorted(f.split("_1460")[0] for f in datacheck.PICASO_CK_MD5)
# One worker launch is one request; a climate solve takes ~9 min on one daw
# core (~17 min on the Space) and is not threaded. A waiter on the lock
# outlasts a holder at its cap.
WALL_CAP_S = 4.0 * 3600.0
LOCK_TIMEOUT_S = WALL_CAP_S + 600.0
LOCK_POLL_S = 5.0
LOCK_LOG_S = 60.0

# Lodders (2020) Table 8, present-day solar system, log eps (H = 12), for the
# atoms picaso's on-the-fly solver carries besides H, He, C, N, O, S. They
# scale with metallicity like the tool's C, N, O, S (forward.CFG_ABUNDANCES,
# the same table). Al, Ca and Ni are absent from picaso's species list.
LODDERS2020_PD = {"Li": 3.27, "F": 4.61, "Na": 6.27, "Mg": 7.52, "Si": 7.51,
                  "P": 5.43, "Cl": 5.23, "K": 5.07, "Ti": 4.90, "V": 3.95,
                  "Cr": 5.63, "Fe": 7.45, "Rb": 2.36, "Cs": 1.08}
# The helium the VULCAN seed uses (VULCAN-JAX thermo/solar_element_abundances.dat)
HE_H = 10.0 ** (10.9232 - 12.0)

# Climate grid: picaso's WASP-39 b notebook (91 levels, 1e-6 to 100 bar, the
# deep radiative-convective seed nlevel - 6). The CK tables start at 1e-6 bar.
CLIMATE_N_LEVELS = 91
CLIMATE_P_SPAN_BAR = (1.0e-6, 100.0)
CLIMATE_RCB = CLIMATE_N_LEVELS - 6

# Climate certificate. A converged W39b solve lands at TOA ~7e-6.
FLUX_BALANCE_MAX = 1.0e-3      # TOA |flux_net / tidal|
GRAD_RANGE = (-0.25, 0.50)     # dlnT/dlnP: inversions allowed, runaways refused
CVZ_TOP_MIN = 5                # a convective zone at the model top is refused
CK_T_MAX = 4000.0              # top of picaso's 1460-point CK grid
# Element budget per layer, all phases, against the request (cold solves
# close to <= 2e-4 on the WASP-39 b, cool and hot columns at 1-100x solar)
ELEMENT_TOL = 1.0e-3
ANCHOR_TOL = 5.0e-3            # baseline gravity fixed point (|dg/g|)
ANCHOR_PASSES = 3

# picaso species -> VULCAN names (ymix_species, the structure figure and the
# RT molecule map read VULCAN names); ions and electrons are not carried
SPECIES_ALIAS = {"HCHO": "H2CO", "SN": "NS"}


def eval_grid(p_btm_bar: float, rt_ptop_bar: float, nz: int, quench: bool) -> np.ndarray:
    """Ascending evaluation pressures (bar): the tool's chemistry grid, plus,
    for the quench search, whole tool steps below it that never pass the
    climate column's bottom."""
    p = np.logspace(np.log10(rt_ptop_bar), np.log10(p_btm_bar), nz)
    if not quench:
        return p
    step = np.log10(p[-1] / p[-2])
    n_ext = int(np.floor(np.log10(CLIMATE_P_SPAN_BAR[1] / p[-1]) / step + 1e-9))
    return np.r_[p, p[-1] * 10.0 ** (step * np.arange(1, n_ext + 1))]


def composition(met_x_solar: float, co_ratio: float) -> dict:
    """X/H for every non-H atom: C, N, O, S exactly the VULCAN path's
    (forward.elemental_abundances: C/O moves carbon at fixed oxygen), He the
    VULCAN seed's, the other metals Lodders 2020 present-day times Z."""
    from jwst_tool import forward as fw
    ab = fw.elemental_abundances(met_x_solar, co_ratio)
    xh = {k: met_x_solar * 10.0 ** (v - 12.0) for k, v in LODDERS2020_PD.items()}
    xh.update(He=HE_H, C=ab["C_H"], N=ab["N_H"], O=ab["O_H"], S=ab["S_H"])
    return xh


def require_picaso_python() -> str:
    if ins.PICASO_PYTHON:
        return ins.PICASO_PYTHON
    raise RuntimeError(
        "No PICASO interpreter configured. The PICASO engine runs in its own "
        "environment (picaso 4.1 needs numpy >= 2): set JWST_TOOL_PICASO_PYTHON "
        "to its python (`jwst-tool data` reports the setup).")


# --- certificates (pure functions of a stored result, re-run on cache load) ---

def certify(req: dict, res: dict, stage: str) -> dict:
    """Raise on the first failed gate; return the certificate record."""
    p = np.asarray(req["p_bar"], float)
    for k in ("vmr", "atoms", "T"):
        if not np.all(np.isfinite(np.asarray(res[k], float))):
            raise RuntimeError(f"PICASO {stage}: non-finite {k} in the worker result.")
    # the column's own layers first: chem_fail counts them too, so checked
    # second it names what is left (climate iterations, the quench recipe)
    bad = ~np.asarray(res["converged"], bool)
    if bad.any():
        raise RuntimeError(
            f"PICASO {stage}: equilibrium chemistry did not converge in "
            f"{int(bad.sum())} of {bad.size} layers ({p[bad].min():.3g}-"
            f"{p[bad].max():.3g} bar); refusing the column.")
    if int(res["chem_fail"]):
        raise RuntimeError(
            f"PICASO {stage}: {int(res['chem_fail'])} chemistry layer solves did "
            "not converge inside picaso (climate iterations or the quench "
            "recipe).")
    names = [str(a) for a in res["atoms_names"]]
    iH = names.index("H")
    ix = [names.index(a) for a in ("C", "N", "O", "S")]
    atoms = np.asarray(res["atoms"], float)
    tgt = np.asarray(res["target"], float)
    err = np.abs((atoms[:, ix] / atoms[:, [iH]]) / (tgt[ix] / tgt[iH]) - 1.0)
    cert = {"element_err": float(err.max())}
    if not err.max() <= ELEMENT_TOL:          # NaN refuses too
        k = np.unravel_index(int(np.argmax(err)), err.shape)
        raise RuntimeError(
            f"PICASO {stage}: element budget off by {err.max():.2e} "
            f"({'CNOS'[k[1]]} at {p[k[0]]:.3g} bar) > {ELEMENT_TOL:g}: the "
            "column does not carry the requested composition.")
    if req["climate"] is None:
        return cert
    P = np.asarray(res["clim_p"], float)
    T = np.asarray(res["clim_T"], float)
    if not int(res["conv_flag"]):
        raise RuntimeError(f"PICASO {stage}: the climate solve did not converge.")
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(T)) and np.all(T > 0)
            and np.all(np.diff(P) > 0)):
        raise RuntimeError(f"PICASO {stage}: climate profile is not finite and ordered.")
    fn = np.asarray(res["flux_net"], float)
    tid = np.asarray(res["tidal"], float)
    cert["flux_toa"] = float(abs(fn[0]) / max(abs(tid[0]), 1e-300))
    if not cert["flux_toa"] <= FLUX_BALANCE_MAX:
        raise RuntimeError(
            f"PICASO {stage}: climate TOA flux balance |F_net/F_int| = "
            f"{cert['flux_toa']:.2e} > {FLUX_BALANCE_MAX:g}.")
    grad = np.gradient(np.log(T), np.log(P))
    cert["grad"] = [float(grad.min()), float(grad.max())]
    if not GRAD_RANGE[0] <= grad.min() <= grad.max() <= GRAD_RANGE[1]:
        raise RuntimeError(
            f"PICASO {stage}: climate dlnT/dlnP spans [{grad.min():.3f}, "
            f"{grad.max():.3f}], outside {list(GRAD_RANGE)}.")
    cvz = [int(x) for x in np.asarray(res["cvz"])]
    cert["cvz"] = cvz
    if 0 < cvz[1] < CVZ_TOP_MIN:
        raise RuntimeError(
            f"PICASO {stage}: a convective zone reaches level {cvz[1]} at the "
            "model top.")
    if not T.max() <= CK_T_MAX:
        raise RuntimeError(
            f"PICASO {stage}: climate T reaches {T.max():.0f} K > {CK_T_MAX:g} K, "
            "the top of picaso's correlated-k grid (picaso extrapolates there).")
    return cert


# --- worker, cache, locks ---

def request_key(req: dict) -> str:
    blob = json.dumps({"req": req, "worker": WORKER_VERSION,
                       "versions": PICASO_VERSIONS, "ck": datacheck.PICASO_CK_MD5,
                       "data_revision": os.environ.get("JWST_TOOL_DATA_REVISION", "")},
                      sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _load(path: Path, req: dict, stage: str):
    """A cached result with its certificate re-run. One that no longer passes
    raises here: the solve is reproducible, so a re-solve would only refuse
    the same way after minutes."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            res = {k: z[k] for k in z.files}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error):
        return None      # unreadable: a miss; the solve rewrites it atomically
    res.pop("error", None)       # older cache files carry an always-empty one
    res["cert"] = certify(req, res, stage)
    return res


@contextlib.contextmanager
def _locked(key: str, log):
    """Per-key fcntl lock; the lock FILE is never unlinked (unlinking a path
    another process may still hold gives two "exclusive" locks)."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK_DIR / f"{key}.lock", "a+")   # noqa: SIM115 (held across the yield)
    t0 = next_log = time.time()
    try:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    raise RuntimeError(
                        f"PICASO cache lock failed on {fh.name} ({exc!r}): set "
                        "TMPDIR to a filesystem with working advisory locks.") from exc
                now = time.time()
                if now - t0 > LOCK_TIMEOUT_S:
                    raise RuntimeError(
                        f"timed out after {LOCK_TIMEOUT_S:.0f} s waiting for a "
                        f"concurrent PICASO solve of the same request ({key}).")
                if now >= next_log:
                    log(f"[fwd] PICASO: waiting for a concurrent identical solve "
                        f"({now - t0:.0f} s)")
                    next_log = now + LOCK_LOG_S
                time.sleep(LOCK_POLL_S)
        yield
    finally:
        fh.close()


def _run_worker(req: dict, stage: str, log) -> dict:
    py = Path(require_picaso_python())
    if not py.exists():
        raise RuntimeError(f"PICASO python not found at {py} (JWST_TOOL_PICASO_PYTHON).")
    PICASO_CACHE.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:8]
    job_path = PICASO_CACHE / f"job.{token}.json"
    out_path = PICASO_CACHE / f"out.{token}.npz"
    job_path.write_text(json.dumps({
        "refdata": str(ins.PICASO_REFDATA), "cdbs": ins.PYSYN_CDBS,
        "versions": PICASO_VERSIONS, "ck_molecules": CK_MOLECULES,
        "request": req}))
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               NUMBA_CACHE_DIR=str(ins.OUTPUT_DIR / "numba_cache" / "picaso"))
    lines = []
    try:
        proc = subprocess.Popen(
            [str(py), str(ins.TOOL_DIR / "picaso_worker.py"), str(job_path),
             str(out_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env)
        with proc_mod.terminating(proc):
            killer = threading.Timer(WALL_CAP_S, proc.kill)
            killer.start()
            try:
                for line in proc.stdout:
                    line = line.rstrip()
                    lines.append(line)
                    if line.startswith("[pic]"):
                        log(f"[fwd] {line}")
                proc.wait()
            finally:
                killer.cancel()
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"PICASO {stage}: " + proc_mod.exception_sentence(
                lines, f"the worker exited with code {proc.returncode} and no "
                       "traceback"))
        with np.load(out_path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    finally:
        job_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def solve(req: dict, stage: str, log) -> dict:
    """The certified result of one request: cache, else lock + worker.
    Raises (never degrades) on any failed gate."""
    key = request_key(req)
    path = PICASO_CACHE / f"{key}.npz"
    hit = _load(path, req, stage)
    if hit is not None:
        log(f"[fwd] PICASO {stage}: cached ({key})")
        return hit
    with _locked(key, log):
        hit = _load(path, req, stage)
        if hit is not None:
            return hit
        t0 = time.time()
        log(f"[fwd] PICASO {stage}: "
            + ("climate solve (minutes) ..." if req["climate"] else "chemistry ..."))
        res = _run_worker(req, stage, log)
        res["cert"] = certify(req, res, stage)
        arrays = {k: v for k, v in res.items() if k != "cert"}
        ins.atomic_write(path, lambda fh: np.savez_compressed(fh, **arrays))
        log(f"[fwd] PICASO {stage}: certified in {time.time() - t0:.0f} s "
            f"({json.dumps(res['cert'])})")
        return res


# --- the chemistry object run_model consumes ---

def column(res: dict, n_tool: int) -> SimpleNamespace:
    """VULCAN-shaped chemistry object from a certified result: the first
    ``n_tool`` evaluation levels (the tool grid; deeper levels exist only for
    the quench search), reversed to bottom-to-top like a VULCAN build."""
    names = [str(s) for s in res["gas_names"]]
    keep = [j for j, s in enumerate(names) if not s.endswith(("+", "-"))]
    sp = [SPECIES_ALIAS.get(names[j], names[j]) for j in keep]
    if len(set(sp)) != len(sp):
        raise RuntimeError(f"PICASO species map is not one-to-one: {sp}")
    y = np.asarray(res["vmr"], float)[:n_tool, keep][::-1]
    return SimpleNamespace(
        p_bar=np.ascontiguousarray(np.asarray(res["p_eval"], float)[:n_tool][::-1]),
        T_base=np.ascontiguousarray(np.asarray(res["T"], float)[:n_tool][::-1]),
        y=np.ascontiguousarray(y),
        sidx={s: i for i, s in enumerate(sp)},
        species_masses=np.asarray(res["gas_mass"], float)[keep],
        cert=res["cert"], quench=json.loads(str(res["quench_json"])),
        versions=json.loads(str(res["versions_json"])))


def _mu(col) -> np.ndarray:
    y = col.y / col.y.sum(axis=1, keepdims=True)
    return y @ col.species_masses


def assemble(cp: dict, log) -> SimpleNamespace:
    """run_model's assembly for chem_provider='picaso' (the counterpart of
    forward._assemble_chem): RT profile, theta, the T-P hook and a factory
    whose builds are already-certified columns.

    Gravity: ONE constant g per run for the climate solve, the quench scale
    height and the Guillot profile -- g at the tool's chemistry anchor (the
    grid interface nearest 1 bar), as on the VULCAN path. A climate baseline
    iterates g with its own T and mean molecular weight to ANCHOR_TOL; the
    composition stencil points of a run reuse that g (``build_chem``).
    """
    # import order is load-bearing: vulcan_chem (env + x64) before exojax
    from vulcan_forward import vulcan_chem  # noqa: F401, I001
    from jwst_tool import engine_config as config
    from jwst_tool import forward as fw

    profile = fw._rt_profile_common(cp, config)
    nz = int(cp["nz"])
    p_eval = eval_grid(cp["p_btm_bar"], cp["rt_ptop_bar"], nz, cp["picaso_quench"])
    p_tool = p_eval[:nz][::-1]                 # bottom to top, like a VULCAN build
    p_anchor = fw.chem_radius_anchor_bar(p_tool, cp["p_btm_bar"] * 1.0e6, False)
    rp_cm = cp["rp_rjup"] * planets.R_JUP_CM
    star = {"teff": cp["star_teff"], "logg": cp["star_logg"], "feh": cp["star_feh"],
            "rstar_rsun": cp["rstar_rsun"], "orbit_au": cp["orbit_au"]}
    mode = cp["tp_mode"]

    def g_at_anchor(T_tool, mu):
        # both aligned with p_tool (bottom to top)
        r = fw.radius_at_anchor(p_tool, T_tool, mu, rp_cm, cp["gs_cgs"],
                                cp["p_ref_bar"], p_anchor)
        return cp["gs_cgs"] * (rp_cm / r) ** 2

    def request(met, co, g, T_eval=None):
        clim = None
        if T_eval is None:
            teq = planets.system_teq(cp["star_teff"], cp["rstar_rsun"], cp["orbit_au"])
            clim = {"tint": float(cp["Tint"]),
                    "rfacv": float(cp["rfacv"]), "teq_guess": float(teq),
                    "nlevel": CLIMATE_N_LEVELS, "p_top_bar": CLIMATE_P_SPAN_BAR[0],
                    "p_bottom_bar": CLIMATE_P_SPAN_BAR[1], "rcb": CLIMATE_RCB,
                    "star": star}
        # T and g are rounded far below anything physical so the request key
        # does not move with the last bits of a JAX profile (CPU vs GPU)
        return {"xh": composition(met, co), "metallicity": float(met),
                "g_cgs": round(float(g), 4), "p_bar": p_eval.tolist(),
                "T": (None if T_eval is None
                      else np.round(np.asarray(T_eval, float), 6).tolist()),
                "quench_kzz": float(cp["kzz_const"]) if cp["picaso_quench"] else None,
                "climate": clim}

    def run(req, stage):
        res = solve(req, stage, log)
        res["p_eval"] = np.asarray(req["p_bar"])
        return column(res, nz)

    tp_eval, theta_tp, names_tp = None, [0.0], ["dT"]
    if mode == "guillot":
        # the VULCAN path's joint iteration: mu from one probe column, then
        # g and the Guillot T (tau = kappa P / g) to their fixed point
        tp_eval0 = fw._build_tp(cp, cp["gs_cgs"])[0]
        tp = np.array([cp["Tirr"], cp["Tint"], cp["log_kappa"], cp["log_gamma"]])
        mu = _mu(run(request(cp["met_x_solar"], cp["co_ratio"], cp["gs_cgs"],
                             np.asarray(tp_eval0(tp, p_eval))), "geometry probe"))
        g = cp["gs_cgs"]
        for _ in range(3):
            g = g_at_anchor(np.asarray(fw._build_tp(cp, g)[0](tp, p_tool)), mu)
        tp_eval = fw._build_tp(cp, g)[0]
        theta_tp, names_tp = list(tp), fw.TP_PARAM_NAMES["guillot"]
        T_fixed = np.asarray(tp_eval(tp, p_eval))
    elif mode == "file":
        # canonical_params has refused a table too shallow for the quench search
        tab = fw._read_tp_table(fw._tp_file_from_cp(cp), fw.chem_p_span_dyn(cp))
        P_tab, T_tab = tab["P_dyn"] / 1.0e6, tab["T"]
        o = np.argsort(P_tab)
        # linear in P with the end rows held, as VULCAN-JAX's file mode reads a table
        T_fixed = np.interp(p_eval, P_tab[o], T_tab[o])
        mu = _mu(run(request(cp["met_x_solar"], cp["co_ratio"], cp["gs_cgs"], T_fixed),
                     "geometry probe"))
        g = g_at_anchor(T_fixed[:nz][::-1], mu)
    elif mode == "picaso_climate":     # T_int is a setting, not a theta row
        # first gravity from a Guillot guess column, then the climate's own
        tirr = planets.system_teq(cp["star_teff"], cp["rstar_rsun"],
                                  cp["orbit_au"]) * np.sqrt(2.0)
        tp0 = np.array([tirr, cp["Tint"], -2.0, -1.0])
        T0 = np.asarray(fw._build_tp(dict(cp, tp_mode="guillot"),
                                     cp["gs_cgs"])[0](tp0, p_eval))
        g = g_at_anchor(T0[:nz][::-1], _mu(run(request(
            cp["met_x_solar"], cp["co_ratio"], cp["gs_cgs"], T0), "geometry probe")))
        for k in range(ANCHOR_PASSES):
            col = run(request(cp["met_x_solar"], cp["co_ratio"], g),
                      f"climate (gravity pass {k + 1})")
            g_new = g_at_anchor(col.T_base, _mu(col))
            log(f"[fwd] PICASO gravity anchor pass {k + 1}: g = {g:.2f} -> "
                f"{g_new:.2f} cm/s2 at {p_anchor:g} bar")
            if abs(g_new / g - 1.0) <= ANCHOR_TOL:
                break
            g = g_new
        else:
            raise RuntimeError(
                f"PICASO climate gravity anchor did not settle to {ANCHOR_TOL:g} "
                f"in {ANCHOR_PASSES} passes (last {g:.2f} -> {g_new:.2f} cm/s2).")
        T_fixed = None               # every build solves its own climate
    else:
        raise ValueError(f"tp_mode {mode!r} is not a PICASO profile")

    theta = np.array([0.0, 0.0, 0.0] + list(theta_tp), dtype=np.float64)

    def build_chem(extra_abun: dict | None = None, tag: str = "baseline",
                   photo_frq=None):
        met = (extra_abun or {}).get("met_x_solar", cp["met_x_solar"])
        co = (extra_abun or {}).get("co_ratio", cp["co_ratio"])
        return run(request(met, co, g, T_fixed), tag)

    def abundance_overrides(met_x_solar: float, co_ratio: float) -> dict:
        return {"met_x_solar": float(met_x_solar), "co_ratio": float(co_ratio)}

    log(f"[fwd] PICASO engine: {mode} profile, g = {g:.2f} cm/s2 at the "
        f"{p_anchor:g} bar anchor (catalogue {cp['gs_cgs']:.1f} at "
        f"{cp['p_ref_bar']:g} bar), {'quench at Kzz ' + format(cp['kzz_const'], 'g') if cp['picaso_quench'] else 'equilibrium'}")
    return SimpleNamespace(
        profile=profile, theta=theta,
        theta_names=fw.CHEM_PARAM_NAMES + list(names_tp),
        tp_eval=tp_eval, n_tp=len(theta_tp) if tp_eval is not None else 0,
        build_chem=build_chem, abundance_overrides=abundance_overrides,
        config=config, g_anchor=g)
