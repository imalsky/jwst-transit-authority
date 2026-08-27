"""Adjoint diagnostics: reverse-mode AD sensitivities the FD Fisher cannot do.

One adjoint solve replaces thousands of re-runs for the high-dimensional
questions: dL/d(ln k_r) over every reaction
(``steady_state_reaction_sensitivity``) and dL/dT per layer
(``steady_state_input_sensitivity``). L is the log10 VMR of the target
molecule at its peak-VMR layer inside the transit photosphere
(``PHOTOSPHERE_P_BAR``).

Contract (all recorded in the npz and shown in the GUI):

* same build path as the forecasts (``forward._assemble_chem``); the
  geometry/operator splice is ``make_body_terms``, never a manual one.
* the scope audit runs FIRST and an audit error refuses the run. It scans
  the upstream ``BODY_MAP_DT_CANDIDATES`` (near-zero trace cells oscillate
  under small probe steps); the gradient uses the first passing ``body_dt``
  and the scan trail is cached.
* magnitudes are trusted only when resid_median <= 0.2 AND ensemble_spread
  <= 0.15 (upstream thresholds); otherwise the result is a RANKING.
  ``pair_antisym`` is a diagnostic only, never a trust gate.
* the rate-uncertainty spread assumes a UNIFORM Agundez (2025) class-B
  0.65 dex per reaction -- a stated assumption, not per-reaction.
* dL/dT is the chemistry-path gradient (photolysis cross sections and the
  diffusion/geometry rebuild frozen, upstream contract); the
  rebuild-consistency metric is stored.

Two faces like forward.py: the light cache API (no JAX imports) and the
heavy script mode (``python -m jwst_tool.adjoint_diag params.json SO2``).
Condensation is refused up front: the pinned reservoir is a
step-sequence-dependent transient (tangents through it are ~91% wrong, see
forward.py).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from jwst_tool import forward
from jwst_tool import instruments as _ins

ADJOINT_CACHE = _ins.OUTPUT_DIR / "adjoint_cache"
_ADJ_VERSION = 3          # bump to invalidate cached adjoint diagnostics
#                           (history: notes.md)

# Loss-layer window (bar): transmission probes ~mbar-0.1 bar; picking the
# peak-VMR layer inside it keeps a deep quenched maximum from hijacking the loss.
PHOTOSPHERE_P_BAR = (1.0e-5, 1.0e-1)

# Upstream certification thresholds (steady_state_grad module constants):
# above these the gradient is reported as a RANKING, not trusted magnitudes.
RESID_MEDIAN_TRUST = 0.2
SPREAD_TRUST = 0.15

# Delta-method rate uncertainty: Agundez (2025) class B = 0.65 dex per rate
# constant, applied uniformly -- a stated assumption, not per-reaction.
UQ_CLASS_DEX = 0.65


def adjoint_key(params: dict, species: str) -> str:
    cp = forward.canonical_params(params)
    payload = {k: v for k, v in cp.items()
               if k not in ("fisher_params", "jac_method",
                            "use_rayleigh", "cloud_on",
                            "log_kappa_cloud", "alpha_cloud", "extra_mols",
                            "wo_mols",
                            "rt_integration",
                            "science_mode", "star_teff",
                            "star_logg", "star_feh")}
    # RT/observable-only knobs are dropped: the adjoint runs on the chemistry
    # state alone, and leaving them in re-triggered the multi-hour adjoint on
    # RT-only changes. rt_ptop_bar stays: the chemistry top follows it.
    payload["adjoint_species"] = str(species)
    payload["adjoint_version"] = _ADJ_VERSION
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def cache_path(params: dict, species: str) -> Path:
    return ADJOINT_CACHE / f"{adjoint_key(params, species)}.npz"


def load_result(params: dict, species: str):
    """Cached adjoint diagnostics dict or None (unreadable entries are
    quarantined and recompute, same policy as forward.load_result)."""
    return forward._load_cached_npz(cache_path(params, species))


# ---------------------------------------------------------------------------
# Heavy path (script mode only below this line)
# ---------------------------------------------------------------------------

def _pair_physical(g: np.ndarray, network) -> list[dict]:
    """Collapse the directional dL/dlnk rows into physical reaction
    sensitivities: forwards live on odd slots; a non-photo/ion forward below
    stop_rev_indx has its detailed-balance reverse on the next (even) slot
    and the physical sensitivity is the SIGNED SUM g[fwd] + g[rev]; anything
    else is a single directional row."""
    g = np.asarray(g, dtype=float)
    n = len(g)
    is_photo = np.asarray(network.is_photo, dtype=bool)
    is_ion = np.asarray(network.is_ion, dtype=bool)
    stop_rev = int(network.stop_rev_indx)
    conden = int(network.conden_indx)
    rows = []
    for fwd in range(1, n, 2):
        photo = fwd < len(is_photo) and bool(is_photo[fwd])
        ion = fwd < len(is_ion) and bool(is_ion[fwd])
        formula = str(network.Rf.get(fwd, f"r{fwd}"))
        if fwd < stop_rev and not photo and not ion:
            gr = float(g[fwd + 1]) if fwd + 1 < n else 0.0
            rows.append(dict(fwd=fwd, S=float(g[fwd]) + gr, kind="reversible",
                             label=formula.replace("->", "<->")))
        else:
            kind = ("photolysis" if photo else
                    "photoionization" if ion else
                    "condensation" if fwd >= conden else "one-way")
            rows.append(dict(fwd=fwd, S=float(g[fwd]), kind=kind,
                             label=formula))
    rows.sort(key=lambda r: -abs(r["S"]))
    return rows


def run_adjoint(params: dict, species: str, log=print) -> Path:
    """One reverse-mode adjoint analysis of the CURRENT forward model state.

    Builds the identical chemistry (forward._assemble_chem), re-converges it
    cold, gates convergence on longdy exactly like run_model, runs the scope
    audit (refusing on audit errors), then computes dL/dlnk (all reactions)
    and dL/dT (all layers) with full certification info, and caches the lot.
    """
    cp = forward.canonical_params(params)
    if cp["use_condense"]:
        raise RuntimeError(
            "adjoint diagnostics are not available for a condensing state: "
            "the fix-species pin freezes the S8 reservoir at a step-"
            "sequence-dependent transient, so the state is not a "
            "reproducible function of the parameters -- temperature "
            "sensitivities through it are refused upstream, and reaction "
            "sensitivities would be conditional on the frozen reservoir "
            "(not validated for this tool). Turn condensation off to run "
            "the adjoint diagnostics.")
    A = forward._assemble_chem(cp, log)   # also arms the XLA compile cache
    # Solve to the TIGHTEST reachable state: extended step budget, stall exit
    # disabled. longdy itself floors at ~0.09 from relative creep of near-zero
    # trace cells (physically steady), so the gate stays the runner's canonical
    # one and per-cell tightness is judged by the scope audit below.
    # (cfg_overrides is the same dict A.build_chem closes over -- update in
    # place.)
    A.profile["cfg_overrides"].update(
        {"count_max": 8000, "conv_stall_window": 10 ** 9})
    import jax.numpy as jnp

    mol_map = A.config.MOLECULES
    if species not in mol_map:
        raise ValueError(
            f"unknown adjoint target {species!r}: choose an RT molecule "
            f"({sorted(mol_map)}) -- the chemistry solves many more species, "
            "but the tool's science goals are phrased on these.")
    vulcan_sp = mol_map[species]["vulcan"]

    t0 = time.time()
    log("[adj] PROG 0.02 building chemistry model")
    chem = A.build_chem(tag="adjoint baseline")
    forward._check_t_window(A.tp_eval, A.theta, chem.p_bar, log,
                            T_base=getattr(chem, "T_base", None))
    if chem.sidx.get(vulcan_sp) is None:
        raise RuntimeError(f"species {vulcan_sp} not in the solved network")
    sp = int(chem.sidx[vulcan_sp])

    log("[adj] PROG 0.15 solving photochemistry (cold, certified)")
    final, _init, atm_T = chem.run_diag(
        jnp.asarray(A.theta, dtype=jnp.float64), return_atm=True)
    longdy = float(final.longdy)
    # Canonical certification: longdy alone accepted budget-exhausted exits
    # with still-drifting photolysis flux; conv_normal is the runner's own
    # gate recomputed at the exit state. A check that cannot run must refuse.
    _cn = getattr(chem, "conv_normal_at_exit", None)
    if _cn is None:
        raise RuntimeError(
            "the sibling forward engine does not export conv_normal_at_exit: "
            "the adjoint fixed point cannot be canonically certified. "
            "Upgrade vulcan-forward.")
    if not (bool(_cn(final)) and longdy < chem.yconv_min):
        raise RuntimeError(
            f"chemistry did NOT converge (longdy={longdy:.3g}, gate "
            f"yconv_min={chem.yconv_min:g}, conv_normal={bool(_cn(final))}): "
            "the adjoint requires a canonically certified fixed point. "
            "Tighten yconv_cri or move the parameters.")
    y_star = final.y
    k_arr = final.k_arr
    y_np = np.asarray(y_star)
    log(f"[adj] converged in {time.time()-t0:.0f} s (longdy {longdy:.3g})")

    # --- loss: log10 VMR of the target at its peak-VMR photosphere layer ----
    p_bar = np.asarray(chem.p_bar)
    vmr = y_np[:, sp] / y_np.sum(axis=1)
    win = (p_bar >= PHOTOSPHERE_P_BAR[0]) & (p_bar <= PHOTOSPHERE_P_BAR[1])
    if not win.any():
        raise RuntimeError("photosphere window empty on this pressure grid")
    if not (vmr[win] > 0.0).any():
        raise RuntimeError(
            f"{species} has zero mixing ratio everywhere in the transit "
            "photosphere -- there is no signal for the adjoint to explain.")
    Lz = int(np.flatnonzero(win)[np.argmax(vmr[win])])
    loss_value = float(np.log10(vmr[Lz]))
    log(f"[adj] loss: log10 VMR({species}) at layer {Lz} "
        f"(P = {p_bar[Lz]:.2e} bar), value {loss_value:.3f}")

    def loss_fn(y):
        return jnp.log10(y[Lz, sp] / jnp.sum(y[Lz]))

    # --- adjoint machinery (VULCAN-JAX) --------------------------------------
    from vulcan_jax import chem_funs
    from vulcan_jax import steady_state_grad as ssg

    net = chem_funs._NET_JAX
    network = chem_funs._NETWORK
    integ = chem._integ
    compo_j = jnp.asarray(np.asarray(chem.compo_array))
    dz_j = jnp.asarray(np.asarray(chem.dz))

    # make_body_terms carries the geometry/operator splice (incl. the hybrid
    # vm_mol choice and boundary pins) -- never the manual 5-field splice.
    atm_step, body_terms = ssg.make_body_terms(integ, final, atm_T)
    recompute_k = (ssg.make_photo_recompute_k(integ._photo_static, final)
                   if cp["use_photo"] else None)

    # --- scope audit FIRST: refuse on errors ---------------------------------
    # Near-zero trace cells OSCILLATE under small probe steps, so scan the
    # upstream-sanctioned candidate steps and use the first body_dt whose
    # audit passes -- for BOTH the audit and the gradient solves; if none
    # passes, refuse. The scan trail is cached.
    audit, body_dt, audit_trail = None, None, []
    for dt in sorted(ssg.BODY_MAP_DT_CANDIDATES):
        log(f"[adj] PROG 0.35 adjoint scope audit (body_dt {dt:.0e})")
        a = ssg.audit_adjoint_scope(
            y_star, k_arr, atm_step, net, cfg=integ._cfg, final_state=final,
            loss_fn=loss_fn, photo_recompute_k=recompute_k,
            body_terms=body_terms, body_dt=dt, print_report=False)
        audit_trail.append(dict(
            body_dt=dt, ok=bool(a.get("ok", False)),
            max_rel_defect=float(a["max_rel_defect"]),
            loss_footprint_defect=float(a["loss_footprint_defect"])))
        log(f"[adj] audit at body_dt {dt:.0e}: ok={a.get('ok')}, "
            f"max_rel_defect {a['max_rel_defect']:.3g}, "
            f"loss_footprint {a['loss_footprint_defect']:.3g}")
        if a.get("ok", False):
            audit, body_dt = a, float(dt)
            break
    findings = ([dict(f) if isinstance(f, dict) else {"finding": str(f)}
                 for f in a.get("findings", [])])
    if audit is None:
        errors = [f for f in findings
                  if str(f.get("severity", "")).lower() == "error"]
        # Say whether the blocking cells are CREEPING or OSCILLATING: a
        # genuine fixed point has G(y) -> y as body_dt -> 0, so a defect that
        # does not fall with the probe step is oscillation -- never relax the
        # gate to get past it.
        worst = [f"{w['species']} layer {w['layer']} "
                 f"(defect {w['rel_defect']:.2f}, ymix {w['ymix']:.1e})"
                 for w in (a.get("worst_cells") or [])[:3]]
        _by_dt = sorted(audit_trail, key=lambda r: r["body_dt"])
        _falls = (len(_by_dt) > 1
                  and _by_dt[0]["max_rel_defect"]
                  < 0.5 * _by_dt[-1]["max_rel_defect"])
        _diag = (
            "the defect FALLS with the probe step, so these cells are still "
            "creeping at the forward tolerance -- converging y_star tighter "
            "(lower yconv_cri) may certify them"
            if _falls else
            "the defect does NOT fall with the probe step, so no probe step "
            "certifies this state. Tightening yconv_cri is unlikely to help "
            "either: on the W39b reference case 1e-2 / 1e-3 / 1e-4 were "
            "MEASURED to give a bit-identical converged state (longdy "
            "plateaus at 0.0997 in all three), so the tolerance knob is inert "
            "there. Do not relax this gate to get past it")
        _dead = a.get("n_clip_dead_excluded")
        _dead_txt = ("" if not _dead else
                     f" ({_dead} zero-clip-dead cell(s) were already excluded "
                     "as solver noise, so these are real abundances.)")
        raise RuntimeError(
            "adjoint scope audit REFUSED this state at every sanctioned "
            "probe step -- the gradient would drop a live process or "
            "differentiate a defective fixed point. Worst cells: "
            + ("; ".join(worst) if worst else "(none reported)")
            + f". Diagnosis: {_diag}.{_dead_txt} "
            "The forward model and the Fisher path are unaffected -- this "
            "gate is specific to the reverse-mode adjoint. Scan: "
            + json.dumps(audit_trail) + "; findings at the last step: "
            + json.dumps(errors or findings))
    log(f"[adj] audit ok at body_dt {body_dt:.0e}: max_rel_defect "
        f"{audit['max_rel_defect']:.3g}, loss_footprint_defect "
        f"{audit['loss_footprint_defect']:.3g}")

    # --- dL/dlnk over every reaction (one adjoint solve ensemble) ------------
    log("[adj] PROG 0.45 reaction sensitivities dL/dlnk "
        "(first call compiles the step-VJP; ~10-20 min cold)")
    t1 = time.time()
    dLdlnk, info = ssg.steady_state_reaction_sensitivity(
        loss_fn, y_star, k_arr, atm_step, net,
        compo_array=compo_j, dz=dz_j, body_dt=body_dt,
        photo_recompute_k=recompute_k, body_terms=body_terms,
        return_info=True)
    g = np.asarray(dLdlnk, dtype=float)
    log(f"[adj] dL/dlnk in {time.time()-t1:.0f} s: fp_err "
        f"{info['fp_err']:.2e}, resid_median {info['resid_median']:.3g}, "
        f"spread {info['ensemble_spread']:.3g}, "
        f"pair_antisym {info['pair_antisym']:.3g}")

    phys = _pair_physical(g, network)
    trust = (float(info["resid_median"]) <= RESID_MEDIAN_TRUST
             and float(info["ensemble_spread"]) <= SPREAD_TRUST)

    # delta-method rate-uncertainty spread (uniform class-B, stated above)
    sigma_lnk = np.log(10.0) * UQ_CLASS_DEX
    contrib = np.array([(r["S"] * sigma_lnk) ** 2 for r in phys])
    sigma_uq = float(np.sqrt(contrib.sum()))

    # --- dL/dT per layer (input sensitivity, chemistry path) -----------------
    log("[adj] PROG 0.80 per-layer temperature sensitivity dL/dT")
    t1 = time.time()
    from vulcan_jax.gibbs import load_nasa9
    from vulcan_jax import rates_jax
    from vulcan_jax._paths import resolve_data_path
    from vulcan_jax.phy_const import kb

    cfg = integ._cfg
    thermo_dir = resolve_data_path(cfg.network).parent
    if not (thermo_dir / "NASA9").exists():
        import vulcan_jax
        thermo_dir = Path(vulcan_jax.__file__).resolve().parent / "thermo"
    nasa9, _present = load_nasa9(network.species, thermo_dir)
    nasa9_j = jnp.asarray(nasa9)
    remove_list = getattr(cfg, "remove_list", None)
    use_caps = bool(getattr(cfg, "use_lowT_limit_rates", False))
    # frozen hydrostatic pressures: rebuild(T) varies T at fixed P (the
    # upstream-validated d/dT recipe; photolysis rows spliced in FROZEN)
    pco_j = jnp.asarray(np.asarray(atm_step.M) * kb * np.asarray(atm_step.Tco))
    photo_rows = jnp.asarray(np.asarray(network.is_photo, dtype=bool))
    k_arr_j = jnp.asarray(k_arr)

    def rebuild(T):
        M = pco_j / (kb * T)
        k = rates_jax.build_rate_array(net, T, M, nasa9_j, remove_list,
                                       use_lowT_caps=use_caps)
        if cp["use_photo"]:
            k = jnp.where(photo_rows[:, None], k_arr_j, k)
        return k, atm_step._replace(Tco=T, Ti=0.5 * (T[:-1] + T[1:]), M=M)

    dLdT, info_T = ssg.steady_state_input_sensitivity(
        loss_fn, y_star, k_arr, atm_step, net,
        jnp.asarray(np.asarray(atm_step.Tco)), rebuild,
        compo_array=compo_j, dz=dz_j, body_dt=body_dt,
        photo_recompute_k=recompute_k, body_terms=body_terms,
        return_info=True)
    dLdT_np = np.asarray(dLdT, dtype=float)
    rc = info_T.get("rebuild_consistency", {})
    rc_worst = float(max(rc.values())) if rc else 0.0
    log(f"[adj] dL/dT in {time.time()-t1:.0f} s: rebuild consistency "
        f"{rc_worst:.2e}, resid_median {info_T['resid_median']:.3g}")

    # --- cache ---------------------------------------------------------------
    ADJOINT_CACHE.mkdir(parents=True, exist_ok=True)
    out = cache_path(params, species)
    n_top = min(25, len(phys))
    # atomic (via the temp+rename helper): a kill mid-write must not leave a
    # torn npz that poisons this key for every later load_result
    _ins.atomic_write(out, lambda fh: np.savez_compressed(
        fh,
        species=np.array(species, dtype="U8"),
        vulcan_species=np.array(vulcan_sp, dtype="U16"),
        loss_layer=np.int64(Lz),
        loss_p_bar=np.float64(p_bar[Lz]),
        loss_log10_vmr=np.float64(loss_value),
        dLdlnk=g,
        top_fwd=np.array([r["fwd"] for r in phys[:n_top]], dtype=np.int64),
        top_S=np.array([r["S"] for r in phys[:n_top]], dtype=np.float64),
        top_kind=np.array([r["kind"] for r in phys[:n_top]], dtype="U16"),
        top_label=np.array([r["label"] for r in phys[:n_top]], dtype="U64"),
        uq_sigma_log10=np.float64(sigma_uq),
        uq_class_dex=np.float64(UQ_CLASS_DEX),
        uq_top_frac=np.float64(contrib[:n_top].sum() / contrib.sum()
                               if contrib.sum() > 0 else 0.0),
        fp_err=np.float64(info["fp_err"]),
        resid_median=np.float64(info["resid_median"]),
        ensemble_spread=np.float64(info["ensemble_spread"]),
        # Diagnostic only, NOT a trust gate (reads ~1 for the accurate renorm
        # default). Never enters `trust`.
        pair_antisym=np.float64(info["pair_antisym"]),
        n_solves=np.int64(info["n_solves"]),
        magnitudes_trusted=np.bool_(trust),
        photo_feedback=np.bool_(bool(info["photo_feedback"])),
        solver_map=np.array(str(info["solver_map"]), dtype="U16"),
        audit_max_rel_defect=np.float64(audit["max_rel_defect"]),
        audit_loss_footprint_defect=np.float64(audit["loss_footprint_defect"]),
        # JSON payloads take the auto-sized U dtype; a fixed width truncates
        # silently
        audit_findings_json=np.array(json.dumps(findings)),
        body_dt=np.float64(body_dt),
        audit_trail_json=np.array(json.dumps(audit_trail)),
        dLdT=dLdT_np,
        p_bar=p_bar,
        rebuild_consistency=np.float64(rc_worst),
        dLdT_resid_median=np.float64(info_T["resid_median"]),
        conv_longdy=np.float64(longdy),
        conv_gate=np.float64(chem.yconv_min),
        params_json=np.array(json.dumps(cp)),
        adjoint_version=np.int64(_ADJ_VERSION),
    ))
    log("[adj] PROG 1.000 done")
    log(f"[adj] cached -> {out.name}")
    return out


def main():
    params = json.load(open(sys.argv[1]))
    species = sys.argv[2]
    # line-buffer stdout: the GUI pipes this process, and block-buffered
    # library prints would sit invisible while the GUI shows nothing
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    # vulcan_jax's legacy IO creates RELATIVE output/ + plot/ dirs in the
    # process CWD; run this subprocess from a scratch cwd instead (library
    # callers are unaffected).
    import os
    _cwd = __import__('pathlib').Path(_ins.OUTPUT_DIR) / "cwd"
    _cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(_cwd)
    run_adjoint(params, species, log=lambda *a: print(*a, flush=True))
    print("[adj] DONE", flush=True)


if __name__ == "__main__":
    main()
