"""Adjoint diagnostics: reverse-mode AD sensitivities the FD Fisher cannot do.

Method and contract: docs/forward_model.md, section `adjoint_diag`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from jwst_tool import certify, chem_build, instruments as _ins, model_cache, run_params

if TYPE_CHECKING:
    from collections.abc import Callable

    from jwst_tool.tool_types import AnyArray, Log

ADJOINT_CACHE = _ins.OUTPUT_DIR / "adjoint_cache"
_ADJ_VERSION = 4  # bump to invalidate cached adjoint diagnostics

# Loss-layer window (bar): transmission probes ~mbar-0.1 bar; picking the
# peak-VMR layer inside it keeps a deep quenched maximum from hijacking the loss.
PHOTOSPHERE_P_BAR = (1.0e-5, 1.0e-1)

# Upstream certification thresholds (vulcan3.steady_state_grad
# _ADJOINT_RESID_WARN and _SPREAD_WARN): above these the gradient is reported
# as a RANKING, not trusted magnitudes.
RESID_MEDIAN_TRUST = 0.2
SPREAD_TRUST = 0.15

# Delta-method rate uncertainty: Agundez (2025) class B = 0.65 dex per rate
# constant, applied uniformly -- a stated assumption, not per-reaction.
UQ_CLASS_DEX = 0.65


def adjoint_key(params: dict, species: str) -> str:
    """Return the cache key of a chemistry state and target species."""
    cp = run_params.canonical_params(params)
    payload = {
        k: v
        for k, v in cp.items()
        if k not in model_cache.CHEM_IRRELEVANT_PARAMS  # the adjoint runs on the chemistry state alone
    }
    payload["adjoint_species"] = str(species)
    payload["adjoint_version"] = _ADJ_VERSION
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def cache_path(params: dict, species: str) -> Path:
    """Return the adjoint cache entry of a chemistry state and target species."""
    return ADJOINT_CACHE / f"{adjoint_key(params, species)}.npz"


def load_result(params: dict, species: str) -> dict | None:
    """Return cached adjoint diagnostics or None; an unreadable entry is quarantined, as in model_cache.load_result."""
    return model_cache.load_cached_npz(cache_path(params, species))


def _pair_physical(g: np.ndarray, network: Any) -> list[dict]:
    """Collapse the directional dL/dlnk rows into physical reaction sensitivities.

    Forwards live on odd slots; a non-photo/ion forward below stop_rev_indx has its detailed-balance reverse on the
    next (even) slot and the physical sensitivity is the SIGNED SUM g[fwd] + g[rev]; anything else is a single
    directional row.
    """
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
            rows.append(dict(fwd=fwd, S=float(g[fwd]) + gr, kind="reversible", label=formula.replace("->", "<->")))
        else:
            kind = (
                "photolysis" if photo else "photoionization" if ion else "condensation" if fwd >= conden else "one-way"
            )
            rows.append(dict(fwd=fwd, S=float(g[fwd]), kind=kind, label=formula))
    rows.sort(key=lambda r: -abs(r["S"]))  # type: ignore[arg-type]  # S is a float
    return rows


@dataclass(frozen=True)
class _Converged:
    """The converged state both sensitivity stages read."""

    ssg: Any
    y_star: AnyArray
    k_arr: AnyArray
    atm_step: Any
    net: Any
    integ: Any
    final: Any
    loss_fn: Callable[[AnyArray], AnyArray]
    recompute_k: Any
    body_terms: Any
    log: Log


def _audit_refusal(a: dict, audit_trail: list, findings: list) -> str:
    """Say which cells block the audit and whether they are creeping or oscillating.

    A genuine fixed point has G(y) -> y as body_dt -> 0, so a defect that does not fall with the probe step is
    oscillation: never relax the gate to get past it. Measured basis: notes "Known and deliberately untracked".
    """
    errors = [f for f in findings if str(f.get("severity", "")).lower() == "error"]
    worst = [
        f"{w['species']} layer {w['layer']} (defect {w['rel_defect']:.2f}, ymix {w['ymix']:.1e})"
        for w in (a.get("worst_cells") or [])[:3]
    ]
    by_dt = sorted(audit_trail, key=lambda r: r["body_dt"])
    falls = len(by_dt) > 1 and by_dt[0]["max_rel_defect"] < 0.5 * by_dt[-1]["max_rel_defect"]
    diag = (
        "the defect FALLS with the probe step, so these cells are still "
        "creeping at the forward tolerance -- converging y_star tighter "
        "(lower yconv_cri) may certify them"
        if falls
        else "the defect does NOT fall with the probe step, so no probe step "
        "certifies this state, and tightening yconv_cri is unlikely to help "
        "either. Do not relax this gate to get past it"
    )
    dead = a.get("n_clip_dead_excluded")
    dead_txt = (
        ""
        if not dead
        else f" ({dead} zero-clip-dead cell(s) were already excluded as solver noise, so these are real abundances.)"
    )
    return (
        "adjoint scope audit REFUSED this state at every sanctioned "
        "probe step -- the gradient would drop a live process or "
        "differentiate a defective fixed point. Worst cells: "
        + ("; ".join(worst) if worst else "(none reported)")
        + f". Diagnosis: {diag}.{dead_txt} "
        "The forward model and the Fisher path are unaffected -- this "
        "gate is specific to the reverse-mode adjoint. Scan: "
        + json.dumps(audit_trail)
        + "; findings at the last step: "
        + json.dumps(errors or findings)
    )


def _audit_scope(c: _Converged) -> tuple:
    """Audit the scope at the sanctioned probe steps, refusing if none passes; return (audit, step, scan, findings)."""
    # Scope audit FIRST: refuse on errors
    # Near-zero trace cells OSCILLATE under small probe steps, so scan the
    # upstream-sanctioned candidate steps and use the first body_dt whose
    # audit passes -- for BOTH the audit and the gradient solves; if none
    # passes, refuse. The scan trail is cached.
    audit, body_dt, audit_trail = None, None, []
    for dt in sorted(c.ssg.BODY_MAP_DT_CANDIDATES):
        c.log(f"[adj] PROG 0.35 adjoint scope audit (body_dt {dt:.0e})")
        a = c.ssg.audit_adjoint_scope(
            c.y_star,
            c.k_arr,
            c.atm_step,
            c.net,
            cfg=c.integ.config,
            final_state=c.final,
            loss_fn=c.loss_fn,
            photo_recompute_k=c.recompute_k,
            body_terms=c.body_terms,
            body_dt=dt,
            print_report=False,
        )
        audit_trail.append(
            dict(
                body_dt=dt,
                ok=bool(a["ok"]),
                max_rel_defect=float(a["max_rel_defect"]),
                loss_footprint_defect=float(a["loss_footprint_defect"]),
            )
        )
        c.log(
            f"[adj] audit at body_dt {dt:.0e}: ok={a['ok']}, "
            f"max_rel_defect {a['max_rel_defect']:.3g}, "
            f"loss_footprint {a['loss_footprint_defect']:.3g}"
        )
        if a["ok"]:
            audit, body_dt = a, float(dt)
            break
    findings = a["findings"]
    if audit is None:
        raise RuntimeError(_audit_refusal(a, audit_trail, findings))
    c.log(
        f"[adj] audit ok at body_dt {body_dt:.0e}: max_rel_defect "
        f"{audit['max_rel_defect']:.3g}, loss_footprint_defect "
        f"{audit['loss_footprint_defect']:.3g}"
    )
    return audit, body_dt, audit_trail, findings


def _temperature_sensitivity(c: _Converged, cp: dict, network: Any, sens_kwargs: dict) -> tuple:
    """Return dL/dT per layer (the chemistry path) and the rebuild-consistency metric."""
    import jax.numpy as jnp

    # dL/dT per layer (input sensitivity, chemistry path)
    c.log("[adj] PROG 0.80 per-layer temperature sensitivity dL/dT")
    t1 = time.time()
    from vulcan3 import rates_jax
    from vulcan3.gibbs import load_nasa9
    from vulcan3.phy_const import kb

    cfg = c.integ.config
    nasa9, _present = load_nasa9(network.species, rates_jax.thermo_dir(cfg.network))
    nasa9_j = jnp.asarray(nasa9)
    remove_list = cfg.remove_list
    use_caps = bool(cfg.use_lowT_limit_rates)
    # frozen hydrostatic pressures: rebuild(T) varies T at fixed P (the
    # upstream-validated d/dT recipe; photolysis rows spliced in FROZEN)
    pco_j = jnp.asarray(np.asarray(c.atm_step.M) * kb * np.asarray(c.atm_step.Tco))
    photo_rows = jnp.asarray(np.asarray(network.is_photo, dtype=bool))
    k_arr_j = jnp.asarray(c.k_arr)

    def rebuild(T: AnyArray) -> tuple:
        M = pco_j / (kb * T)
        k = rates_jax.build_rate_array(c.net, T, M, nasa9_j, remove_list, use_lowT_caps=use_caps)
        if cp["use_photo"]:
            k = jnp.where(photo_rows[:, None], k_arr_j, k)
        return k, c.atm_step._replace(Tco=T, Ti=0.5 * (T[:-1] + T[1:]), M=M)

    dLdT, info_T = c.ssg.steady_state_input_sensitivity(
        c.loss_fn,
        c.y_star,
        c.k_arr,
        c.atm_step,
        c.net,
        jnp.asarray(np.asarray(c.atm_step.Tco)),
        rebuild,
        **sens_kwargs,
        return_info=True,
    )
    dLdT_np = np.asarray(dLdT, dtype=float)
    rc_worst = float(max(info_T["rebuild_consistency"].values()))
    c.log(
        f"[adj] dL/dT in {time.time() - t1:.0f} s: rebuild consistency "
        f"{rc_worst:.2e}, resid_median {info_T['resid_median']:.3g}"
    )
    return dLdT_np, info_T, rc_worst


def _refuse_unsupported(cp: dict) -> None:
    if cp["chem_provider"] != "vulcan":
        raise RuntimeError(
            "adjoint diagnostics differentiate VULCAN-JAX's reaction network; "
            "the PICASO engine has no reactions to differentiate."
        )
    if cp["use_condense"]:
        raise RuntimeError(
            "adjoint diagnostics are not available for a condensing state: "
            "the fix-species pin freezes the S8 reservoir at a step-"
            "sequence-dependent transient, so the state is not a "
            "reproducible function of the parameters -- temperature "
            "sensitivities through it are refused upstream, and reaction "
            "sensitivities would be conditional on the frozen reservoir "
            "(not validated for this tool). Turn condensation off to run "
            "the adjoint diagnostics."
        )


def _target_species(A: Any, species: str) -> str:
    """Return the network's name for an RT molecule, refusing anything else."""
    mol_map = A.config.MOLECULES
    if species not in mol_map:
        raise ValueError(
            f"unknown adjoint target {species!r}: choose an RT molecule "
            f"({sorted(mol_map)}) -- the chemistry solves many more species, "
            "but the tool's science goals are phrased on these."
        )
    return mol_map[species]["vulcan"]


def _certified_final(chem: Any, theta: AnyArray) -> tuple[Any, Any, float]:
    """Solve cold and require the runner's canonical certificate: (final state, its T, longdy).

    longdy alone would accept a budget-exhausted exit whose photolysis flux still drifts; conv_normal is the runner's
    own gate recomputed at the exit state.
    """
    import jax.numpy as jnp

    final, _init, atm_T = chem.run_diag(jnp.asarray(theta, dtype=jnp.float64), return_atm=True)
    longdy = float(final.longdy)
    converged = bool(chem.conv_normal_at_exit(final))
    if not (converged and longdy < chem.yconv_min):
        raise RuntimeError(
            f"chemistry did NOT converge (longdy={longdy:.3g}, gate "
            f"yconv_min={chem.yconv_min:g}, conv_normal={converged}): "
            "the adjoint requires a canonically certified fixed point. "
            "Tighten yconv_cri or move the parameters."
        )
    return final, atm_T, longdy


def _reaction_sensitivity(c: _Converged, sens_kwargs: dict) -> tuple[np.ndarray, dict]:
    """Return dL/dlnk over every reaction (one adjoint solve ensemble) and its diagnostics."""
    c.log("[adj] PROG 0.45 reaction sensitivities dL/dlnk (first call compiles the step-VJP; ~10-20 min cold)")
    t1 = time.time()
    dLdlnk, info = c.ssg.steady_state_reaction_sensitivity(
        c.loss_fn, c.y_star, c.k_arr, c.atm_step, c.net, **sens_kwargs, return_info=True
    )
    c.log(
        f"[adj] dL/dlnk in {time.time() - t1:.0f} s: fp_err "
        f"{info['fp_err']:.2e}, resid_median {info['resid_median']:.3g}, "
        f"spread {info['ensemble_spread']:.3g}, "
        f"pair_antisym {info['pair_antisym']:.3g}"
    )
    return np.asarray(dLdlnk, dtype=float), info


def _loss_layer(y_np: np.ndarray, sp: int, p_bar: np.ndarray, species: str) -> tuple[int, float]:
    """Return the loss layer and value: log10 VMR of the target at its peak-VMR photosphere layer."""
    vmr = y_np[:, sp] / y_np.sum(axis=1)
    win = (p_bar >= PHOTOSPHERE_P_BAR[0]) & (p_bar <= PHOTOSPHERE_P_BAR[1])
    if not win.any():
        raise RuntimeError("photosphere window empty on this pressure grid")
    if not (vmr[win] > 0.0).any():
        raise RuntimeError(
            f"{species} has zero mixing ratio everywhere in the transit "
            "photosphere -- there is no signal for the adjoint to explain."
        )
    Lz = int(np.flatnonzero(win)[np.argmax(vmr[win])])
    return Lz, float(np.log10(vmr[Lz]))


def _write_adjoint(out: Path, arrays: dict[str, Any], phys: list, contrib: np.ndarray) -> None:
    """Write the cache entry, with the top reactions; atomic, since a torn npz poisons this key for every load."""
    n_top = min(25, len(phys))
    top: dict[str, Any] = dict(
        top_fwd=np.array([r["fwd"] for r in phys[:n_top]], dtype=np.int64),
        top_S=np.array([r["S"] for r in phys[:n_top]], dtype=np.float64),
        top_kind=np.array([r["kind"] for r in phys[:n_top]], dtype="U16"),
        top_label=np.array([r["label"] for r in phys[:n_top]], dtype="U64"),
        uq_top_frac=np.float64(contrib[:n_top].sum() / contrib.sum() if contrib.sum() > 0 else 0.0),
    )
    _ins.atomic_write(out, lambda fh: np.savez_compressed(fh, **arrays, **top))


def run_adjoint(params: dict, species: str, log: Log = print) -> Path:
    """One reverse-mode adjoint analysis of the CURRENT forward model state.

    Builds the identical chemistry (chem_build.assemble_chem), re-converges it
    cold on an extended step budget, and certifies the fixed point on the
    runner's own conv_normal at the exit state plus longdy < yconv_min -- NOT
    run_model's check_converged/check_elements gate. Then runs the scope audit
    (refusing on audit errors), computes dL/dlnk (all reactions) and dL/dT
    (all layers) with full certification info, and caches the lot.
    """
    cp = run_params.canonical_params(params)
    _refuse_unsupported(cp)
    A = chem_build.assemble_chem(cp, log)  # also arms the XLA compile cache
    # Solve to the TIGHTEST reachable state: extended step budget. longdy
    # itself floors at ~0.09 from relative creep of near-zero trace cells
    # (physically steady), so the gate stays the runner's canonical one and
    # per-cell tightness is judged by the scope audit below.
    # (cfg_overrides is the same dict A.build_chem closes over -- update in
    # place.)
    A.profile["cfg_overrides"].update({"count_max": 8000})
    import jax.numpy as jnp

    vulcan_sp = _target_species(A, species)

    t0 = time.time()
    log("[adj] PROG 0.02 building chemistry model")
    chem = A.build_chem(tag="adjoint baseline")
    certify.check_t_window(A.tp_eval, A.theta, chem.p_bar, log, T_base=chem.T_base)
    if chem.sidx.get(vulcan_sp) is None:
        raise RuntimeError(f"species {vulcan_sp} not in the solved network")
    sp = int(chem.sidx[vulcan_sp])

    log("[adj] PROG 0.15 solving photochemistry (cold, certified)")
    final, atm_T, longdy = _certified_final(chem, A.theta)
    y_star = final.y
    k_arr = final.k_arr
    y_np = np.asarray(y_star)
    log(f"[adj] converged in {time.time() - t0:.0f} s (longdy {longdy:.3g})")

    p_bar = np.asarray(chem.p_bar)
    Lz, loss_value = _loss_layer(y_np, sp, p_bar, species)
    log(f"[adj] loss: log10 VMR({species}) at layer {Lz} (P = {p_bar[Lz]:.2e} bar), value {loss_value:.3f}")

    def loss_fn(y: AnyArray) -> AnyArray:
        return jnp.log10(y[Lz, sp] / jnp.sum(y[Lz]))

    # Adjoint machinery (VULCAN-JAX)
    from vulcan3 import chem_funs, steady_state_grad as ssg

    net = chem_funs.NET_JAX
    network = chem_funs.NETWORK
    integ = chem._integ
    compo_j = jnp.asarray(np.asarray(chem.compo_array))
    dz_j = jnp.asarray(np.asarray(chem.dz))

    # make_body_terms carries the geometry/operator splice (incl. the hybrid
    # vm_mol choice and boundary pins) -- never the manual 5-field splice.
    atm_step, body_terms = ssg.make_body_terms(integ, final, atm_T)
    recompute_k = ssg.make_photo_recompute_k(integ.photo_static, final) if cp["use_photo"] else None
    converged = _Converged(ssg, y_star, k_arr, atm_step, net, integ, final, loss_fn, recompute_k, body_terms, log)
    audit, body_dt, audit_trail, findings = _audit_scope(converged)

    sens_kwargs = dict(
        compo_array=compo_j, dz=dz_j, body_dt=body_dt, photo_recompute_k=recompute_k, body_terms=body_terms
    )
    g, info = _reaction_sensitivity(converged, sens_kwargs)

    phys = _pair_physical(g, network)
    trust = float(info["resid_median"]) <= RESID_MEDIAN_TRUST and float(info["ensemble_spread"]) <= SPREAD_TRUST

    # delta-method rate-uncertainty spread (uniform class-B, stated above)
    sigma_lnk = np.log(10.0) * UQ_CLASS_DEX
    contrib = np.array([(r["S"] * sigma_lnk) ** 2 for r in phys])
    sigma_uq = float(np.sqrt(contrib.sum()))
    dLdT_np, info_T, rc_worst = _temperature_sensitivity(converged, cp, network, sens_kwargs)

    ADJOINT_CACHE.mkdir(parents=True, exist_ok=True)
    out = cache_path(params, species)
    _write_adjoint(
        out,
        dict(
            species=np.array(species, dtype="U8"),
            vulcan_species=np.array(vulcan_sp, dtype="U16"),
            loss_layer=np.int64(Lz),
            loss_p_bar=np.float64(p_bar[Lz]),
            loss_log10_vmr=np.float64(loss_value),
            dLdlnk=g,
            uq_sigma_log10=np.float64(sigma_uq),
            uq_class_dex=np.float64(UQ_CLASS_DEX),
            fp_err=np.float64(info["fp_err"]),
            resid_median=np.float64(info["resid_median"]),
            ensemble_spread=np.float64(info["ensemble_spread"]),
            # Diagnostic only, NOT a trust gate (reads ~1 for the accurate renorm default). Never enters `trust`.
            pair_antisym=np.float64(info["pair_antisym"]),
            n_solves=np.int64(info["n_solves"]),
            magnitudes_trusted=np.bool_(trust),
            photo_feedback=np.bool_(bool(info["photo_feedback"])),
            solver_map=np.array(str(info["solver_map"]), dtype="U16"),
            audit_max_rel_defect=np.float64(audit["max_rel_defect"]),
            audit_loss_footprint_defect=np.float64(audit["loss_footprint_defect"]),
            # JSON payloads take the auto-sized U dtype; a fixed width truncates silently
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
        ),
        phys,
        contrib,
    )
    log("[adj] PROG 1.000 done")
    log(f"[adj] cached -> {out.name}")
    return out


def main() -> None:
    """Run one adjoint analysis from a params file and a species name."""
    from jwst_tool import proc

    params = json.loads(Path(sys.argv[1]).read_text())
    species = sys.argv[2]
    proc.worker_prologue()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    run_adjoint(params, species, log=lambda *a: print(*a, flush=True))
    print("[adj] DONE", flush=True)


if __name__ == "__main__":
    main()
