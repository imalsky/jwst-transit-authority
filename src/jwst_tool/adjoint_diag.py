"""Adjoint diagnostics: reverse-mode AD sensitivities the FD Fisher cannot do.

The Fisher forecast (forward.py) differentiates the spectrum with respect to a
HANDFUL of parameters -- certified finite differences (or the opt-in warm jvp)
are the right tool there. This module answers the HIGH-DIMENSIONAL questions,
where one reverse-mode adjoint solve replaces thousands of re-runs:

* which REACTIONS control the target molecule?  dL/d(ln k_r) over every rate
  in the network (~800 directional rows), via VULCAN-JAX
  ``steady_state_reaction_sensitivity`` (validated 0.2-0.8% against FD on the
  W39b SO2 / HD189 CH4 benchmarks with the renorm map + photolysis feedback);
* which LAYERS' temperature does it respond to?  dL/dT(P) per layer, via
  ``steady_state_input_sensitivity`` (chemistry-path gradient; d/dT validated
  against forward-mode on HD189).

L is the log10 volume mixing ratio of the target molecule at its peak-VMR
layer inside the transit photosphere (``PHOTOSPHERE_P_BAR``) -- the same loss
form as the validated jax_paper reference caller (adj_w39b_so2.py), which
this module replaces: that script predates the YAML-only config migration
(it imported the deleted ``cfg_examples``) and used the manual 5-field
geometry splice; here the state, cfg, and composition come from the SAME
``forward._assemble_chem`` path the forecasts run, the splice is the
current-correct ``make_body_terms``, and every run is preceded by the
adjoint SCOPE AUDIT (``audit_adjoint_scope``) -- an audit ERROR refuses the
run rather than reporting an untrustworthy gradient.

Honesty contract (all recorded in the npz and shown in the GUI):

* the scope audit's findings + the loss-footprint defect (a defect inside
  the cells the loss reads disqualifies the gradient -- RuntimeError). The
  audit runs over the upstream-sanctioned probe steps
  (``BODY_MAP_DT_CANDIDATES``) because near-zero trace cells oscillate
  under small probe steps (measured: W39b max defect 0.65 at dt 1e6 vs
  0.023 at 3e7); the gradient then uses the first passing ``body_dt`` and
  the whole scan trail is cached;
* the ensemble certification: ``fp_err`` (fixed-point tightness),
  ``resid_median`` (adjoint solve residual), ``ensemble_spread`` (the honest
  magnitude error bar over ulp-perturbed twin solves). Magnitudes are
  labeled trustworthy only when resid_median <= 0.2 AND spread <= 0.15
  (the upstream thresholds); otherwise the result is presented as a RANKING.
  ``pair_antisym`` (worst forward/reverse row asymmetry near partial
  equilibrium) is also recorded and shown, but it is a DIAGNOSTIC ONLY, never a
  trust gate: it is bare-map calibrated and reads ~1 for the accurate renorm
  default (upstream contract), so it does not enter the magnitudes-trusted
  decision;
* the rate-uncertainty spread: a delta-method sigma(log10 VMR) under a
  UNIFORM Agundez (2025) class-B rate uncertainty (0.65 dex per reaction) --
  a stated assumption, not a per-reaction assessment;
* dL/dT is the chemistry-path gradient (photolysis cross sections and the
  diffusion/geometry rebuild are frozen by design, upstream contract), and
  the rebuild-consistency metric is stored (the call itself refuses above
  1e-3 relative).

Like forward.py this module has two faces: the light cache API (no JAX
imports) and the heavy script mode
(``python -m jwst_tool.adjoint_diag params.json SO2``).

Condensation never appears here: it is detection-only in this tool, and
``run_adjoint`` refuses a condensing state up front -- the pinned reservoir
is frozen at a step-sequence-dependent transient (tangents through it are
~91% wrong, see forward.py), and conditional-on-frozen-reservoir reaction
sensitivities are not validated for this tool.
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
#                           v2 (2026-07-19): key strips ALL RT/observable-only
#                           knobs (the v15/v16 additions had been fragmenting
#                           the cache) + canonical conv_normal certification
#                           v3 (2026-07-20): JSON npz fields auto-size their
#                           U dtype (fixed widths could truncate silently)

# Transit-photosphere pressure window for picking the loss layer (bar):
# transmission spectra probe roughly mbar-to-0.1-bar; the peak-VMR layer is
# taken inside this window so a deep quenched maximum cannot hijack the loss.
PHOTOSPHERE_P_BAR = (1.0e-5, 1.0e-1)

# Upstream certification thresholds (steady_state_grad module constants):
# above these the gradient is reported as a RANKING, not trusted magnitudes.
RESID_MEDIAN_TRUST = 0.2
SPREAD_TRUST = 0.15

# Uniform rate-uncertainty class for the delta-method spread: Agundez (2025)
# class B = 0.65 dex per rate constant (class A 0.30, C 1.00). A stated
# assumption applied to every reaction, NOT a per-reaction assessment.
UQ_CLASS_DEX = 0.65


def adjoint_key(params: dict, species: str) -> str:
    cp = forward.canonical_params(params)
    payload = {k: v for k, v in cp.items()
               if k not in ("fisher_params", "jac_method", "nu_pts",
                            "use_rayleigh", "broadening", "cloud_on",
                            "log_kappa_cloud", "alpha_cloud", "extra_mols",
                            # v15/v16 RT/observable-only additions (_ADJ_VERSION
                            # 2): none of these touch the chemistry state, and
                            # leaving them in re-triggered the multi-hour
                            # adjoint on e.g. an RT top-pressure change
                            "rt_ptop_bar", "rt_integration", "rt_dit_res",
                            "mie_condensate", "mie_log_rg", "mie_sigmag",
                            "mie_log_mmr", "science_mode", "star_teff",
                            "star_logg", "star_feh")}
    # RT-only knobs are dropped from the key: the adjoint runs on the
    # chemistry state alone, so spectra-only settings must not fragment it.
    payload["adjoint_species"] = str(species)
    payload["adjoint_version"] = _ADJ_VERSION
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def cache_path(params: dict, species: str) -> Path:
    return ADJOINT_CACHE / f"{adjoint_key(params, species)}.npz"


def load_result(params: dict, species: str):
    """Cached adjoint diagnostics dict or None."""
    p = cache_path(params, species)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Heavy path (script mode only below this line)
# ---------------------------------------------------------------------------

def _pair_physical(g: np.ndarray, network) -> list[dict]:
    """Collapse the directional dL/dlnk rows into physical reaction
    sensitivities (the validated jax_paper pairing rules): forwards live on
    odd slots; a forward below stop_rev_indx that is not photo/ion has its
    detailed-balance reverse on the next (even) slot and the physical
    sensitivity is the SIGNED SUM g[fwd] + g[rev]; everything else is a
    single directional row (photolysis / one-way)."""
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
    A = forward._assemble_chem(cp, log)   # also arms the persistent XLA
    #                                       compile cache (see _assemble_chem)
    # The adjoint linearizes around the fixed point, so solve to the TIGHTEST
    # REACHABLE state: extended step budget, stall early-exit disabled. The
    # longdy metric itself cannot be pushed to ~1e-3 here -- it is floored by
    # RELATIVE creep of near-zero trace cells (W39b photo-on plateaus at
    # longdy ~0.09 with |dy/dt| ~ 1e-11, i.e. physically steady; measured
    # 2026-07-15, 8000 steps) -- so the convergence gate stays the runner's
    # canonical one and PER-CELL tightness is judged where it matters, by
    # the scope audit below. (cfg_overrides is the same dict object
    # A.build_chem closes over -- update in place.)
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
    # Canonical certification (v2): longdy alone accepted a budget-exhausted
    # exit whose photolysis flux was still changing. conv_normal is the
    # runner's own two-branch gate recomputed at the exit state -- the
    # documented adjoint plateau (longdy ~0.09 with |dy/dt| ~1e-11,
    # physically steady) passes its slope branch, so this tightening keeps
    # the validated W39b case. A check that cannot run must refuse.
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

    # current-correct geometry/operator splice (NOT the manual 5-field splice
    # of the retired reference script -- make_body_terms also carries the
    # hybrid vm_mol operator choice and any boundary pins)
    atm_step, body_terms = ssg.make_body_terms(integ, final, atm_T)
    recompute_k = (ssg.make_photo_recompute_k(integ._photo_static, final)
                   if cp["use_photo"] else None)

    # --- scope audit FIRST: refuse on errors ---------------------------------
    # The audit's per-cell fixed-point defect is measured under ONE probe
    # step of length body_dt, and near-zero trace cells OSCILLATE under
    # small probe steps (measured 2026-07-15 on the then-default W39b
    # configuration -- isothermal 1100 K, constant Kzz 1e9; the structure
    # default is now the tabulated table, so read these as illustrative
    # magnitudes: max defect 0.65 at dt 1e6 falling to 0.023 at dt 3e7, on
    # H2S/C cells at ymix ~ 1e-13 while the loss footprint stayed <= 9e-3
    # throughout). Nothing below depends on the digits -- the scan refuses
    # when no candidate passes, whatever the configuration.
    # So scan the upstream-sanctioned candidate probe steps and use the
    # first dt whose audit passes -- for BOTH the audit and the gradient
    # solves; if none passes, refuse. The scan trail is cached.
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
        # Name the cells that actually blocked it, and say whether they are
        # CREEPING or OSCILLATING -- the two need opposite responses and the
        # bare scan table cannot tell them apart. A genuine fixed point has
        # G(y) -> y as body_dt -> 0, so a defect that does not fall with the
        # probe step is oscillation: converging tighter or probing smaller
        # will not fix it, and the gate must not be relaxed to get past it.
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
    np.savez_compressed(
        out,
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
        # Worst forward/reverse pair antisymmetry over the top rows. Reported
        # for context, NOT a trust gate (bare-map calibrated; reads ~1 for the
        # accurate renorm default -- upstream contract). Never enters `trust`.
        pair_antisym=np.float64(info["pair_antisym"]),
        n_solves=np.int64(info["n_solves"]),
        magnitudes_trusted=np.bool_(trust),
        photo_feedback=np.bool_(bool(info["photo_feedback"])),
        solver_map=np.array(str(info["solver_map"]), dtype="U16"),
        audit_max_rel_defect=np.float64(audit["max_rel_defect"]),
        audit_loss_footprint_defect=np.float64(audit["loss_footprint_defect"]),
        # JSON payloads take the auto-sized U dtype; a fixed width truncates
        # silently once the serialized dict outgrows it
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
    )
    log("[adj] PROG 1.000 done")
    log(f"[adj] cached -> {out.name}")
    return out


def main():
    params = json.load(open(sys.argv[1]))
    species = sys.argv[2]
    # line-buffer stdout: the GUI pipes this process, which makes Python
    # BLOCK-buffer prints from libraries (picaso's climate iteration lines
    # would sit invisible in the buffer while the GUI shows nothing)
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    # vulcan_jax's legacy IO creates RELATIVE output/ + plot/ directories in
    # the process CWD (legacy_io.py) -- junk wherever the app was launched
    # from. Run the subprocess from a dedicated scratch cwd instead (the
    # same fix the Space entrypoint uses). Library callers of run_model are
    # unaffected: only this subprocess entrypoint changes directory.
    import os
    _cwd = __import__('pathlib').Path(_ins.OUTPUT_DIR) / "cwd"
    _cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(_cwd)
    run_adjoint(params, species, log=lambda *a: print(*a, flush=True))
    print("[adj] DONE", flush=True)


if __name__ == "__main__":
    main()
