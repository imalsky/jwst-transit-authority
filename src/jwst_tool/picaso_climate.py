"""PICASO radiative-convective climate runner: certified, cached, locked.

Heavy-path module (imports picaso via picaso_env on use; never imported by
the GUI's light path or the numpy-only test suite).

Contracts:

* CERTIFICATION -- a solve is cached and served ONLY when every gate passes
  (fail loud, never a degraded profile): converged flag; TOA flux metric
  below ``FLUX_BALANCE_MAX``; finite ordered P and finite positive T;
  dlnT/dlnP within ``(GRAD_MIN, GRAD_MAX)``; convective zone below the model
  top. The certificate is stored WITH the profile and revalidated on every
  cache load.
* DETERMINISM -- the solve is bit-deterministic for identical inputs, so the
  cache is exact. ``climate_rcb`` is a NUMERICAL SEED (PICASO's
  ``rcb_guess``), NOT a physical parameter: PICASO cannot shrink a zone
  seeded too shallow, so the default is the deep seed (85). It stays
  cache-keyed and the Tint_cl FD row differentiates at FIXED seed until
  seed-independence is certified.
* CACHE KEY -- ``climate_subset(cp)``: the climate inputs + the FULL climate
  reference-data content fingerprint. ``chem_provider`` is EXCLUDED (the
  converged T-P is provider-independent; both providers share one solve).
* LOCKING -- atomic writes (tmp + os.replace) plus a per-key fcntl.flock
  whose file is NEVER unlinked: unlink-based "stale lock breaking" creates
  the two-inode double-lock race. A dead holder's flock releases on its own;
  a LIVE holder is waited on (up to ``LOCK_TIMEOUT_S``, then raise -- never
  break). The pid/start-time in the file is observability metadata only.
* GUESS -- a DETERMINISTIC analytic Guillot profile from the canonical
  inputs only (never warm-started). rfacv=0 runs star-free
  (``setup_nostar``).
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from jwst_tool import instruments as _ins
from jwst_tool import picaso_env as pe
from jwst_tool import planets
from jwst_tool.forward import (CHEM_P_SPAN_DYN, CLIMATE_N_LEVELS,
                               CLIMATE_P_SPAN_BAR, T_WINDOW)

_CLIMATE_VERSION = 2   # bump to invalidate cached solves (history: notes.md)
CLIMATE_CACHE = Path(_ins.OUTPUT_DIR) / "picaso_climate_cache"

FLUX_BALANCE_MAX = 1.0e-3   # TOA |flux_net[0]/tidal[0]|: converged W39b lands
                            # ~1e-6; catches gross imbalance behind the flag
GRAD_MIN, GRAD_MAX = -0.25, 0.50   # dlnT/dlnP envelope: inversions allowed,
                                   # runaways refused (steepest adiabat ~0.35)
CVZ_TOP_MIN = 5             # convective zone must not reach the model top
LOCK_TIMEOUT_S = 3600.0     # no healthy climate solve takes an hour
LOCK_POLL_S = 5.0

# Guillot 2010 eq. 29 guess constants -- fixed so the guess is a pure
# function of the canonical inputs (guess only; the converged profile does
# not depend on them beyond the solve's basin)
_GUESS_KAPPA_IR = 1.0e-2    # cm^2/g
_GUESS_GAMMA = 0.4
_GUESS_F = 0.25

# One copy per package: these are `planets`' values, re-exported under the
# local names this module already uses. Two modules in one package disagreeing
# about a unit conversion is exactly the drift test_shared_constants_pin.py
# guards against.
_RSUN_CM = planets.R_SUN_CM
_AU_CM = planets.AU_CM


def climate_subset(cp: dict) -> dict:
    """The exact inputs a climate solve depends on (and nothing else)."""
    if cp["tp_mode"] != "picaso_climate":
        raise ValueError("climate_subset called outside picaso_climate mode")
    return {
        "_climate_version": _CLIMATE_VERSION,
        "picaso_version": cp["picaso_version"],
        "picaso_climate_sha1": cp["picaso_climate_sha1"],
        "picaso_ck_node": cp["picaso_ck_node"],
        "tio_vo": cp["tio_vo"],
        "tint_cl": cp["tint_cl"],
        "rfacv": cp["rfacv"],
        "climate_rcb": cp["climate_rcb"],
        "rp_rjup": cp["rp_rjup"],
        "gs_cgs": cp["gs_cgs"],
        "rstar_rsun": cp["rstar_rsun"],
        "orbit_au": cp["orbit_au"],
        "star_teff": cp["star_teff"],
        "star_logg": cp["star_logg"],
        "star_feh": cp["star_feh"],
    }


def climate_key(cp: dict, tint_override: float | None = None) -> str:
    import hashlib
    sub = climate_subset(cp)
    if tint_override is not None:
        sub["tint_cl"] = round(float(tint_override), 2)
    s = json.dumps(sub, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _paths(key: str) -> tuple[Path, Path, Path]:
    return (CLIMATE_CACHE / f"{key}.npz",
            CLIMATE_CACHE / f"{key}_atm.txt",
            CLIMATE_CACHE / f"{key}.lock")


def guillot_guess(p_bar: np.ndarray, gs_cgs: float, tint: float,
                  tirr: float) -> np.ndarray:
    """Deterministic analytic guess (Guillot 2010 eq. 29, fixed constants)."""
    tau = _GUESS_KAPPA_IR * (np.asarray(p_bar, float) * 1.0e6) / gs_cgs
    g3 = _GUESS_GAMMA * np.sqrt(3.0)
    T4 = (0.75 * tint**4 * (2.0 / 3.0 + tau)
          + 0.75 * tirr**4 * _GUESS_F * (
              2.0 / 3.0 + 1.0 / g3
              + (_GUESS_GAMMA / np.sqrt(3.0) - 1.0 / g3) * np.exp(-g3 * tau)))
    return T4 ** 0.25


def _certify(out: dict, key: str) -> dict:
    """Evaluate every certification gate; raise on the first failure."""
    P = np.asarray(out["pressure"], float)
    T = np.asarray(out["temperature"], float)
    cert: dict = {"climate_key": key, "_climate_version": _CLIMATE_VERSION}
    if not bool(out.get("converged")):
        raise RuntimeError(
            "picaso climate solve did NOT converge (converged flag false). "
            "Refusing to cache or use the profile; adjust tint_cl / "
            "climate_rcb / rfacv or treat this configuration as outside the "
            "certified envelope.")
    cert["converged"] = True
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(T))
            and np.all(T > 0.0) and np.all(np.diff(P) > 0.0)):
        raise RuntimeError(
            "climate profile failed structural sanity (finite, ordered P; "
            "finite positive T). Refusing.")
    fn = np.atleast_1d(np.asarray(out["flux_balance"]["flux_net"], float))
    tidal = np.atleast_1d(np.asarray(out["flux_balance"]["tidal"], float))
    flux_metric = float(abs(fn[0]) / max(abs(tidal[0]), 1e-300))
    cert["flux_toa_over_tidal"] = flux_metric
    if flux_metric > FLUX_BALANCE_MAX:
        raise RuntimeError(
            f"climate TOA flux balance |flux_net[0]/tidal[0]| = "
            f"{flux_metric:.3e} exceeds {FLUX_BALANCE_MAX:g}: the solver's "
            "converged flag is not backed by its own flux metric. Refusing.")
    grad = np.gradient(np.log(T), np.log(P))
    cert["grad_min"] = float(grad.min())
    cert["grad_max"] = float(grad.max())
    if grad.min() < GRAD_MIN or grad.max() > GRAD_MAX:
        raise RuntimeError(
            f"climate profile gradient dlnT/dlnP spans [{grad.min():.3f}, "
            f"{grad.max():.3f}], outside the physical envelope "
            f"[{GRAD_MIN}, {GRAD_MAX}] (pathological RCB / runaway profile "
            "-- the failure mode picaso's docs warn can hide behind the "
            "converged flag). Refusing.")
    cvz = [int(x) for x in np.atleast_1d(out.get("cvz_locs", []))]
    cert["cvz_locs"] = cvz
    if len(cvz) >= 2 and 0 < cvz[1] < CVZ_TOP_MIN:
        raise RuntimeError(
            f"climate convective zone reaches layer {cvz[1]} (< "
            f"{CVZ_TOP_MIN}): a convective zone at the model top is outside "
            "the certified envelope. Refusing.")
    return cert


def _write_atm_table(P_bar: np.ndarray, T: np.ndarray, path: Path) -> None:
    """VULCAN atm table (descending P, bottom row EXACTLY the chemistry-grid
    bottom) for the vulcan-provider structural path.

    The bottom row is interpolated to CHEM_P_SPAN_DYN[1]: a raw climate level
    just below it can sit above the RT temperature window while T at the
    chemistry bottom is in-window, so writing raw levels would trip the
    T-window refusal. Above the table top the engine holds the topmost T
    constant (standard file-mode convention)."""
    P_dyn = np.asarray(P_bar, float) * 1.0e6
    T = np.asarray(T, float)
    bottom = CHEM_P_SPAN_DYN[1]
    if P_dyn.max() < bottom:
        raise RuntimeError(
            f"climate grid bottom {P_dyn.max():.3g} dyn/cm^2 does not cover "
            f"the chemistry bottom {bottom:.3g}. (The climate solve grid "
            f"spans {CLIMATE_P_SPAN_BAR} bar -- this cannot happen unless "
            "the constants drifted.)")
    lo = np.log(P_dyn)
    T_bottom = float(np.interp(np.log(bottom), lo, T))
    keep = P_dyn < bottom
    rows = [(bottom, T_bottom)] + [
        (float(p), float(t)) for p, t in zip(P_dyn[keep][::-1], T[keep][::-1])]
    _t_lo = min(t for _, t in rows)
    _t_hi = max(t for _, t in rows)
    if _t_hi > T_WINDOW[1] or _t_lo < T_WINDOW[0]:
        raise RuntimeError(
            f"climate T-P leaves the modelable window {T_WINDOW} K over the "
            f"chemistry span (profile spans [{_t_lo:.0f}, {_t_hi:.0f}] K; "
            f"T at the {bottom / 1e6:.1f} bar bottom = {T_bottom:.0f} K): "
            "this planet/irradiation/Tint configuration is outside the "
            "certified envelope (out-of-window profiles are rejected, never "
            "clipped). Too-cold tops come from weak irradiation (low rfacv "
            "on a cool star), too-hot bottoms from strong irradiation or a "
            "shallow radiative-convective boundary.")
    tmp = path.with_suffix(".txt.tmp.%d" % os.getpid())
    with open(tmp, "w") as fh:
        fh.write("#(dyne/cm2) (K) picaso_climate v%d\n" % _CLIMATE_VERSION)
        fh.write("Pressure\tTemp\n")
        for p, t in rows:
            fh.write("%.6e\t%.2f\n" % (p, t))
    os.replace(tmp, path)


def interp_T(clim, p_bar: np.ndarray) -> np.ndarray:
    """Climate T on an arbitrary pressure grid (log-P linear interp).

    Above the climate top the topmost T is held CONSTANT (stated policy);
    below the climate bottom is refused."""
    p = np.asarray(p_bar, float)
    Pc = np.asarray(clim.pressure_bar, float)
    if np.any(p > Pc.max() * (1 + 1e-9)):
        raise RuntimeError(
            f"requested pressure {p.max():.3g} bar below the climate grid "
            f"bottom {Pc.max():.3g} bar -- refusing to extrapolate.")
    return np.interp(np.log(np.clip(p, Pc.min(), None)),
                     np.log(Pc), np.asarray(clim.T, float))


def _revalidate(ns) -> bool:
    """Re-run every gate evaluable from the STORED data: a cache entry is
    served only if it still passes -- loading is never weaker than solving."""
    P = np.asarray(ns.pressure_bar, float)
    T = np.asarray(ns.T, float)
    c = ns.cert
    try:
        if not (P.size >= 4 and np.all(np.isfinite(P))
                and np.all(np.isfinite(T)) and np.all(T > 0.0)
                and np.all(np.diff(P) > 0.0)):
            return False
        grad = np.gradient(np.log(T), np.log(P))
        if grad.min() < GRAD_MIN or grad.max() > GRAD_MAX:
            return False
        if not (float(c.get("flux_toa_over_tidal", np.inf))
                <= FLUX_BALANCE_MAX):
            return False
        cvz = [int(x) for x in (c.get("cvz_locs") or [])]
        if len(cvz) >= 2 and 0 < cvz[1] < CVZ_TOP_MIN:
            return False
    except (TypeError, ValueError):
        return False
    return True


def _load(npz_path: Path, key: str):
    """Cached result IF present and its certificate REVALIDATES (identity +
    every re-runnable gate on the stored arrays/metrics), else None."""
    if not npz_path.is_file():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            cert = json.loads(str(z["cert_json"]))
            prov = json.loads(str(z["provenance_json"]))
            if (cert.get("climate_key") != key
                    or cert.get("_climate_version") != _CLIMATE_VERSION
                    or not cert.get("converged")):
                return None
            ns = SimpleNamespace(
                pressure_bar=np.asarray(z["pressure_bar"], float),
                T=np.asarray(z["temperature_K"], float),
                dtdp=np.asarray(z["dtdp"], float),
                cert=cert, provenance=prov, key=key, npz_path=npz_path)
            if not _revalidate(ns):
                return None
            return ns
    except (OSError, ValueError, KeyError):
        return None      # unreadable/foreign file: recompute (never trust it)


def _solve(cp: dict, tint: float, log) -> dict:
    root = pe.bootstrap()
    jdi = pe.import_picaso()
    import astropy.units as u

    ck = pe.ck_path(cp["picaso_ck_node"], cp["tio_vo"], root)
    log(f"[fwd] climate: CK table {ck.name}, Tint={tint:g} K, "
        f"rfacv={cp['rfacv']:g}, rcb_guess={cp['climate_rcb']}")
    t0 = time.time()
    opa = jdi.opannection(ck_db=str(ck), method="preweighted")
    cl = jdi.inputs(calculation="planet", climate=True)
    cl.effective_temp(float(tint))
    cl.gravity(gravity=float(cp["gs_cgs"]), gravity_unit=u.Unit("cm/(s**2)"))
    tirr = 0.0
    if cp["rfacv"] > 0.0:
        pe.bootstrap()      # picaso's stellar module re-pins PYSYN_CDBS per
        #                     call; re-pin OURS immediately before star()
        cl.star(opa, temp=float(cp["star_teff"]), metal=float(cp["star_feh"]),
                logg=float(cp["star_logg"]), radius=float(cp["rstar_rsun"]),
                radius_unit=u.R_sun, semi_major=float(cp["orbit_au"]),
                semi_major_unit=u.AU, database="ck04models")
        tirr = float(cp["star_teff"]) * np.sqrt(
            cp["rstar_rsun"] * _RSUN_CM / (cp["orbit_au"] * _AU_CM))
    else:
        cl.setup_nostar()
    p_levels = np.logspace(np.log10(CLIMATE_P_SPAN_BAR[0]),
                           np.log10(CLIMATE_P_SPAN_BAR[1]), CLIMATE_N_LEVELS)
    guess = guillot_guess(p_levels, float(cp["gs_cgs"]), float(tint), tirr)
    cl.inputs_climate(temp_guess=guess, pressure=p_levels, rfaci=1,
                      rcb_guess=int(cp["climate_rcb"]),
                      rfacv=float(cp["rfacv"]))
    log("[fwd] climate: solving (iteration lines stream below; the first "
        "solve on a machine also compiles, which can add minutes) ...")
    # verbose=True streams one line per iteration to the GUI console --
    # without it the solve is silent for minutes and reads as a hang
    out = cl.climate(opa, save_all_profiles=False, with_spec=False,
                     verbose=True)
    out["_wall_s"] = time.time() - t0
    return out


def get_or_run(cp: dict, log, tint_override: float | None = None):
    """The certified climate profile for ``cp`` (cache -> lock -> solve).

    Returns SimpleNamespace(pressure_bar, T, dtdp, cert, provenance, key,
    npz_path, atm_table). Raises (never degrades) on any certification
    failure, a stale-lock timeout, or missing reference data.
    """
    tint = float(cp["tint_cl"] if tint_override is None else tint_override)
    key = climate_key(cp, None if tint_override is None else tint)
    npz_path, atm_path, lock_path = _paths(key)
    hit = _load(npz_path, key)
    if hit is not None and atm_path.is_file():
        log(f"[fwd] climate: cache hit {key} (Tint={tint:g} K)")
        hit.atm_table = atm_path
        return hit

    CLIMATE_CACHE.mkdir(parents=True, exist_ok=True)
    # ONE fd for the whole acquisition; the lock FILE is never unlinked. A
    # dead holder's flock releases on its own; a live holder is waited on,
    # never broken.
    lf = open(lock_path, "a+")
    t_wait0 = time.time()
    t_last_note = 0.0
    got_lock = False
    try:
        while True:
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                got_lock = True
                break
            except OSError as exc:
                # ONLY contention errnos mean "wait" -- anything else (e.g. a
                # no-flock filesystem) must raise loudly, never poll for an
                # hour looking like a hang
                import errno as _errno
                if exc.errno not in (_errno.EAGAIN, _errno.EACCES,
                                     _errno.EWOULDBLOCK):
                    raise RuntimeError(
                        f"climate cache lock failed on {lock_path} with "
                        f"{exc!r} -- this filesystem does not support "
                        "flock (common on some network volumes). The cache "
                        "directory must live on a filesystem with working "
                        "advisory locks; set JWST_TOOL_OUTPUT_DIR "
                        "accordingly.") from exc
                if time.time() - t_last_note > 30.0:
                    t_last_note = time.time()
                    log(f"[fwd] climate: waiting for a concurrent solve of "
                        f"this exact configuration "
                        f"({time.time() - t_wait0:.0f}s; it will be reused "
                        "when it finishes)")
                if time.time() - t_wait0 > LOCK_TIMEOUT_S:
                    try:
                        meta = json.loads(lock_path.read_text() or "{}")
                    except (OSError, ValueError):
                        meta = {}
                    raise RuntimeError(
                        f"timed out waiting {LOCK_TIMEOUT_S:.0f}s for the "
                        f"climate lock {lock_path} (holder metadata: "
                        f"{meta}). A solve of this exact configuration is "
                        "still running or stuck; investigate before "
                        "retrying -- the lock is never broken while its "
                        "holder lives.")
                time.sleep(LOCK_POLL_S)
                hit = _load(npz_path, key)           # holder may finish
                if hit is not None and atm_path.is_file():
                    hit.atm_table = atm_path
                    return hit
        # lock held: record holder metadata (observability only)
        lf.truncate(0)
        lf.write(json.dumps({"pid": os.getpid(), "t0": time.time()}))
        lf.flush()
        hit = _load(npz_path, key)                   # double-checked
        if hit is not None and atm_path.is_file():
            hit.atm_table = atm_path
            return hit
        out = _solve(cp, tint, log)
        cert = _certify(out, key)
        P = np.asarray(out["pressure"], float)
        T = np.asarray(out["temperature"], float)
        _write_atm_table(P, T, atm_path)
        prov = {"climate_subset": climate_subset(cp) | {
                    "tint_cl": round(tint, 2)},
                "wall_s": round(float(out["_wall_s"]), 1),
                "flux_balance": {k: (np.asarray(v).tolist()
                                     if np.ndim(v) else float(v))
                                 for k, v in out["flux_balance"].items()}}
        # tmp name must END in .npz: savez_compressed appends the suffix to
        # any other name, orphaning the tmp and breaking os.replace
        tmp = npz_path.with_name(f"{npz_path.stem}.tmp{os.getpid()}.npz")
        np.savez_compressed(
            tmp, pressure_bar=P, temperature_K=T,
            dtdp=np.asarray(out.get("dtdp", []), float),
            cert_json=json.dumps(cert), provenance_json=json.dumps(prov))
        os.replace(tmp, npz_path)
        log(f"[fwd] climate: converged in {out['_wall_s']:.0f}s, certified "
            f"(flux metric {cert['flux_toa_over_tidal']:.2e}, grad "
            f"[{cert['grad_min']:.2f}, {cert['grad_max']:.2f}], cvz "
            f"{cert['cvz_locs']}), cached as {key}")
        return SimpleNamespace(pressure_bar=P, T=T,
                               dtdp=np.asarray(out.get("dtdp", []), float),
                               cert=cert, provenance=prov, key=key,
                               npz_path=npz_path, atm_table=atm_path)
    finally:
        if got_lock:
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        lf.close()
        # the lock FILE stays: unlinking a path another process may still
        # flock creates two simultaneous "exclusive" locks


def warm_default() -> None:
    """Pre-solve the GUI's default climate configuration (Space boot warmer).

    Idempotent (a cache hit returns in milliseconds). Takes a run slot so
    warming never starves visitors; skips (and says so) when busy. Also
    compiles picaso's numba kernels, persisted via NUMBA_CACHE_DIR.

    No-ops with a printed reason when the PICASO path is gated off. That is
    the shipped state: canonical_params refuses tp_mode="picaso_climate"
    unless JWST_TOOL_ENABLE_UNCERTIFIED_PICASO=1, so before this check every
    boot launched a background solve that died on its first statement.
    """
    from jwst_tool import forward, runlimit
    if not forward.picaso_experimental_enabled():
        print(f"[warm] PICASO is disabled ({forward.PICASO_EXPERIMENTAL_ENV} "
              "is not 1); no climate pre-solve to do", flush=True)
        return
    cp = forward.canonical_params({"tp_mode": "picaso_climate"})
    slot = runlimit.acquire("climate-warm")
    if slot is None:
        print("[warm] all run slots busy; skipping the climate pre-solve",
              flush=True)
        return
    try:
        clim = get_or_run(cp, lambda m: print(m, flush=True))
        print(f"[warm] default climate solve cached ({clim.key}, "
              f"{float(clim.cert['flux_toa_over_tidal']):.1e} TOA metric)",
              flush=True)
    finally:
        slot.release()
