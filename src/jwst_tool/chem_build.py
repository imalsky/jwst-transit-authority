"""Build the VULCAN chemistry model of a canonical run: RT profile, cfg overrides, gravity anchor, T-P hook."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from jwst_tool import planets
from jwst_tool.model_defs import (
    ART_PBTM_FRACTION,
    CFG_ABUNDANCES,
    CHEM_P_SPAN_DYN,
    CHEM_PARAM_NAMES,
    DT_MAX_S,
    NETWORKS,
    P_BTM_FILE_BAR,
    TP_PARAM_NAMES,
    active_molecules,
    chem_p_span_dyn,
    elemental_abundances,
)
from jwst_tool.tp_tables import read_tp_table, tp_file_from_cp

if TYPE_CHECKING:
    from collections.abc import Callable


def chem_radius_anchor_bar(p_bar: np.ndarray, p_btm_dyn: float, rocky: bool) -> float:
    """Pressure (bar) VULCAN anchors ``g = G*Mp/Rp**2`` at, for a built grid.

    Mirrors ``vulcan3.atm_setup.compute_mu_dz_g``: the grid INTERFACE
    nearest 1 bar for a non-rocky planet whose column reaches 1 bar, else the
    column bottom. Test-pinned against the engine, since it is the one rule
    this module restates rather than imports.
    """
    from vulcan3.atm_setup import compute_pico

    pico = np.asarray(compute_pico(np.asarray(p_bar, dtype=np.float64) * 1.0e6))
    if rocky or float(p_btm_dyn) < 1.0e6:
        return float(pico[0]) / 1.0e6
    return float(pico[int(np.argmin(np.abs(np.log10(pico) - 6.0)))]) / 1.0e6


def radius_at_anchor(
    p_bar: np.ndarray, T: np.ndarray, mu: np.ndarray, rp_cm: float, gs_cgs: float, p_ref_bar: float, p_anchor_bar: float
) -> float:
    """Radius (cm) at ``p_anchor_bar`` given ``rp_cm`` at ``p_ref_bar``.

    Hydrostatic equilibrium at fixed GM, ``g(r) = G*M/r**2``, linearises
    exactly in ``u = 1/r``::

        du/dlnP = k_B T / (mu m_u G M)

    so the anchor needs one cumulative integral and no ODE solve. The same
    identity ``vulcan_forward.exojax_rt._radius_at`` integrates to place the RT
    grid bottom; deeper means smaller radius and stronger gravity. Evaluated at
    the exact anchor pressure, so the partial layer between it and the nearest
    grid point is accounted for rather than snapped away.
    """
    from vulcan_forward import constants as _vc

    p = np.asarray(p_bar, dtype=np.float64)
    order = np.argsort(p)
    lnp, T, mu = np.log(p[order]), np.asarray(T)[order], np.asarray(mu)[order]
    # The engine can anchor at the exterior bottom interface (not a cell
    # centre), e.g. when p_btm_bar is exactly 1 bar. Continue the bottom
    # layer's T/mu across that half cell, as the RT radius integral does.
    ln_bottom = 1.5 * lnp[-1] - 0.5 * lnp[-2]
    ln_ref, ln_anchor = np.log(p_ref_bar), np.log(p_anchor_bar)
    if not (lnp[0] <= ln_ref <= ln_bottom + 1e-12 and lnp[0] <= ln_anchor <= ln_bottom + 1e-12):
        raise ValueError(
            f"radius reference {p_ref_bar:g} bar and chemistry anchor "
            f"{p_anchor_bar:g} bar must both lie inside the column "
            f"[{p.min():g}, {np.exp(ln_bottom):g}] bar including its bottom interface"
        )
    gm = float(gs_cgs) * float(rp_cm) ** 2  # the true GM, level-free
    c = _vc.K_B_CGS * T / (mu * _vc.M_U_CGS * gm)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (c[1:] + c[:-1]) * np.diff(lnp))])
    integral = np.interp(np.log(p_anchor_bar), lnp, cum) - np.interp(np.log(p_ref_bar), lnp, cum)
    integral += c[-1] * (max(0.0, ln_anchor - lnp[-1]) - max(0.0, ln_ref - lnp[-1]))
    return 1.0 / (1.0 / float(rp_cm) + float(integral))


# VULCAN condensation channel on the SNCHO network. DETECTION-ONLY: the
# whole-column fix_species pin freezes the reservoir at a step-sequence-
# dependent transient, so no derivative through it is trustworthy (the
# canonical_params raises carry the user-facing reason). One reaction,
# S8 -> S8_l_s, on rainout-sized 50 um orthorhombic sulfur (smaller radii make
# the growth term stiffer than Ros2 resolves). The cold-trap argmin degenerates
# on isothermal columns, so the pin is whole-column; without it every solve
# exhausts count_max.
CONDEN_CFG = {
    "use_condense": True,
    "condense_sp": ["S8"],
    "non_gas_sp": ["S8_l_s"],
    "r_p": {"S8_l_s": 5.0e-3},
    "rho_p": {"S8_l_s": 2.07},
    "use_relax": [],
    "use_settling": False,
    "fix_species": ["S8", "S8_l_s"],
    "fix_species_from_coldtrap_lev": False,
    "start_conden_time": 0.0,
    "stop_conden_time": 1.0e6,
    # At the 1e-20 default, glacial trace species gate longdy forever on cold
    # columns; 1e-15 is still far below RT relevance.
    "mtol_conv": 1.0e-15,
    # Heavy hydrocarbons + trace sulfur allotropes re-equilibrate on
    # unreachable timescales far below RT relevance against a pinned S8. The
    # observable sulfur species (SO2, H2S, SO) STAY in the gate.
    "conver_ignore": [
        "C6H6",
        "C2H2",
        "C6H5",
        "C2H",
        "C2H4",
        "C2H5",
        "C2H6",
        "C3H2",
        "C3H3",
        "C4H5",
        "CH2NH",
        "CH3NH2",
        "H2CCO",
        "S",
        "S2",
        "S3",
        "S4",
    ],
    # bounded from below so the conden window + pin complete before the gate
    "trun_min": 1.0e6,
}


def build_tp(cp: dict, gs_cgs: float) -> tuple:
    """(tp_eval, n_tp, tp_values, theta_names) for the chosen T-P mode.

    tp_eval(tp_params, p_bar) is pure JAX (differentiable) for the parametric
    modes. In file mode tp_eval is None: the engine's temperature path is
    T = T_base + theta[3] over the pre-loop's re-grid of the tabulated profile
    (atm_type="file"), and theta[3] is PINNED to 0, so there is no dT
    parameter and no T-P Fisher row.
    """
    import jax.numpy as jnp

    mode = cp["tp_mode"]
    if mode == "guillot":
        from exojax.atm.atmprof import atmprof_Guillot

        def tp_eval(tp: Any, p_bar: np.ndarray) -> Any:
            p = jnp.asarray(p_bar)
            Tirr, Tint = tp[0], tp[1]
            kappa, gamma = 10.0 ** tp[2], 10.0 ** tp[3]
            return atmprof_Guillot(p, gs_cgs, kappa, gamma, Tint, Tirr, 0.25)

        vals = [cp["Tirr"], cp["Tint"], cp["log_kappa"], cp["log_gamma"]]
        return tp_eval, 4, vals, CHEM_PARAM_NAMES + TP_PARAM_NAMES["guillot"]
    if mode == "file":
        # theta keeps its 4th slot for the engine's uniform-shift path, pinned
        # to 0.0 and named "dT"
        return None, 0, [0.0], [*CHEM_PARAM_NAMES, "dT"]
    raise ValueError(f"unknown tp_mode {mode!r}")


def rt_profile_common(cp: dict, config: Any) -> dict:
    """Return the RT-facing profile keys (exojax_rt / build_emis_model read exactly these).

    Pinned by tests/unit/test_rt_profile_golden.py.
    """
    profile = dict(config.WIDE)
    # The COUNTS are locked equal for one resolution control; ExoJAX builds its
    # own pressure coordinates and interp_map maps chemistry onto them.
    profile["nz"] = cp["nz"]
    profile["art_nlayer"] = cp["nz"]
    # The engine ECHOES these on the built rt namespace and run_model verifies
    # the echo, so an engine ignoring a profile key cannot return a spectrum
    # differing from what the cache key claims.
    profile["art_ptop_bar"] = cp["rt_ptop_bar"]
    # RT bottom, just inside the chemistry bottom (interp_map refuses an ART
    # grid reaching below it)
    profile["art_pbtm_bar"] = cp["p_btm_bar"] * ART_PBTM_FRACTION
    profile["rt_integration"] = cp["rt_integration"]
    profile["p_ref_bar"] = cp["p_ref_bar"]
    # Correlated-k over the published ExoMolOP k-tables is the engine's only
    # opacity path; set explicitly so the echo check can verify it.
    profile["opacity_mode"] = "exomolop"
    profile["molecules"] = active_molecules(cp)
    profile["use_photo"] = cp["use_photo"]  # build_chem_model reads this key
    profile["use_rayleigh"] = cp["use_rayleigh"]  # exojax_rt reads this flag
    # Planet identity
    rp_cm = cp["rp_rjup"] * planets.R_JUP_CM
    rstar_cm = cp["rstar_rsun"] * planets.R_SUN_CM
    profile["rp_cm"] = rp_cm  # RT geometry (exojax_rt reads these)
    profile["gs_cgs"] = cp["gs_cgs"]
    profile["rstar_cm"] = rstar_cm
    return profile


def _cfg_overrides(cp: dict, rp_cm: float, net_path: str, net_atoms: str, log: Callable[..., None]) -> tuple:
    """Return the chemistry side's config overrides and the T-P table path (None for a parametric profile)."""
    _net_path, _net_atoms = net_path, net_atoms
    tp_path = None
    ovr = {  # chemistry side (applied pre-pre-loop)
        # VULCAN derives g = G*Mp/Rp^2, so convert gs_cgs to a planet mass
        "Mp": cp["gs_cgs"] * rp_cm**2 / planets.G_CGS,
        # the cfg network must agree with the import-frozen one (the engine
        # raises when it does not); S_H stays in the cfg under ncho, harmless
        # with an S-free atom_list
        "network": _net_path,
        "atom_list": _net_atoms.split(","),
        "Rp": rp_cm,
        "r_star": cp["rstar_rsun"],
        "orbit_radius": cp["orbit_au"],
        "sflux_file": f"atm/stellar_flux/{cp['sflux']}",
        "use_moldiff": cp["use_moldiff"],
        # pin the vm_mol scheme EXPLICITLY, never inherit the upstream YAML
        # default; the two flags travel together (hybrid is how vm_mol runs)
        "use_vm_mol": cp["use_vm_mol"],
        "use_hybrid_vm_mol": cp["use_vm_mol"],
    }
    if cp["use_photo"]:  # photolysis geometry/averaging knobs
        ovr["sl_angle"] = float(np.deg2rad(cp["sl_angle_deg"]))
        ovr["f_diurnal"] = cp["f_diurnal"]
    if cp["use_condense"]:
        # canonical_params already confirmed detection-only, photo on and
        # moldiff on; the engine rebuilds the arrays on-graph per solve
        ovr.update(CONDEN_CFG)
    # Structural baseline. Guillot mode: isothermal -- the on-graph tp_eval
    # supplies the actual T(P), the structural profile only sets the hydrostatic
    # grid + EQ init. File mode: the table IS the structure, hash-verified.
    if cp["tp_mode"] == "file":
        tp_path = tp_file_from_cp(cp)  # sha-verified re-resolution
        ovr.update({"atm_type": "file", "atm_file": str(tp_path)})
        log(
            f"[fwd] planet {cp['planet']}: tabulated T-P structure from "
            f"{tp_path.name} (sha1 {cp['tp_file_sha1']}), UV = {cp['sflux']}"
        )
    else:
        T_struct = cp["Tirr"] / np.sqrt(2.0)  # guillot: ~equilibrium T at f=0.25
        ovr.update({"atm_type": "isothermal", "Tiso": float(T_struct)})
        log(f"[fwd] planet {cp['planet']}: isothermal structural baseline {T_struct:.0f} K, UV = {cp['sflux']}")
    # Kzz profile (lnKzz = theta[2] scales ANY of them on-graph).
    if cp["kzz_mode"] == "const":
        ovr.update({"Kzz_prof": "const", "const_Kzz": cp["kzz_const"]})
        log(f"[fwd] Kzz: const {cp['kzz_const']:.1e} cm2/s")
    elif cp["kzz_mode"] == "Pfunc":
        ovr.update({"Kzz_prof": "Pfunc", "K_max": cp["kzz_kmax"], "K_p_lev": cp["kzz_plev"]})
        log(f"[fwd] Kzz: Pfunc deep {cp['kzz_kmax']:.1e} cm2/s rising as P^-0.4 above {cp['kzz_plev']:g} bar")
    elif cp["kzz_mode"] == "JM16":
        ovr.update({"Kzz_prof": "JM16", "K_deep": cp["kzz_kdeep"]})
        log(f"[fwd] Kzz: JM16 (deep floor {cp['kzz_kdeep']:.1e} cm2/s, 1e5 (300 mbar/P)^0.5 above)")
    else:  # "file": gated to tp_mode="file"
        ovr.update({"Kzz_prof": "file"})
        log("[fwd] Kzz: tabulated column of the T-P table")
    # Chemistry-grid bottom: a run needing a deeper column overrides P_b here
    # rather than editing the YAML the siblings were validated on.
    if abs(cp["p_btm_bar"] - P_BTM_FILE_BAR) > 1e-9:
        ovr["P_b"] = cp["p_btm_bar"] * 1.0e6
        log(
            f"[fwd] chemistry grid bottom {cp['p_btm_bar']:g} bar "
            f"(cfg default {P_BTM_FILE_BAR:g} bar), RT bottom "
            f"{cp['p_btm_bar'] * ART_PBTM_FRACTION:g} bar; "
            f"{cp['nz'] / np.log10(cp['p_btm_bar'] / cp['rt_ptop_bar']):.0f} "
            "layers per decade"
        )
    return ovr, tp_path


def _check_cfg_constants(cp: dict, config: Any, tp_path: Path | None, log: Callable[..., None]) -> Any:
    """Refuse drift between this tool's constants and the loaded cfg; return the cfg."""
    # CFG_ABUNDANCES must equal the loaded cfg's O_H/C_H/N_H/S_H: a wrong set
    # here mislabels every C/O display and every exported abundance.
    import vulcan3 as _vj

    cfg_chk = _vj.load_config(config.W39B_CFG_NAME)
    for k, v in CFG_ABUNDANCES.items():
        cfg_v = float(getattr(cfg_chk, k))
        if abs(cfg_v / v - 1.0) > 1e-9:
            raise RuntimeError(
                f"model_defs.CFG_ABUNDANCES[{k!r}]={v:g} no longer matches the "
                f"network cfg's {k}={cfg_v:g}: the C/O display baseline and "
                "the exported abundances would be mislabeled. Update "
                "CFG_ABUNDANCES to the cfg values (and bump _VERSION)."
            )
    # CHEM_P_SPAN_DYN must match the live cfg too: the light path's
    # bottom-coverage refusal keys on it.
    span_cfg = (float(cfg_chk.P_t), float(cfg_chk.P_b))
    if any(abs(a / b - 1.0) > 1e-9 for a, b in zip(span_cfg, CHEM_P_SPAN_DYN, strict=True)):
        raise RuntimeError(
            f"model_defs.CHEM_P_SPAN_DYN={CHEM_P_SPAN_DYN} no longer matches "
            f"the network cfg's (P_t, P_b)={span_cfg}: the T-P table "
            "span validation would gate against the wrong grid. Update the "
            "constant (and bump _VERSION)."
        )
    if cp["tp_mode"] == "file":
        # Log the conventional TOP clamp (the bottom is hard-gated at the API)
        P_tab = read_tp_table(tp_path, span=chem_p_span_dyn(cp))["P_dyn"]
        p_top_run = chem_p_span_dyn(cp)[0]
        dec = float(np.log10(P_tab.min() / p_top_run))
        if dec > 0.0:
            log(
                f"[fwd] NOTE: T-P table top ({P_tab.min():.3g} dyn/cm^2) "
                f"sits {dec:.1f} decades below the chemistry-grid top "
                f"({p_top_run:g}): the topmost tabulated T is held constant "
                "over that range (the upstream file-mode convention)."
            )
    return cfg_chk


def _radius_anchor(
    cp: dict, theta: Any, probe: Any, rp_cm: float, tp_eval: Any, rocky: bool, log: Callable[..., None]
) -> tuple:
    """Return (radius, gravity) at the grid level VULCAN anchors g on, given the catalogue's radius at p_ref_bar.

    The catalogue radius and gravity are quoted at p_ref_bar (a transit radius, ~1 mbar) and the RT honors that. VULCAN
    anchors g = G*Mp/Rp^2 at the grid interface nearest 1 bar. Mp is already right, since G*Mp is the true GM whichever
    level it is quoted at, so only the level Rp attaches to moves. Guillot's tau = kappa*P/g takes the same gravity,
    hence the joint iteration, which costs no solve. The probe build supplies the initial-equilibrium mean molecular
    weight the integral needs.
    """
    import jax.numpy as jnp

    y0 = np.asarray(probe.y0, dtype=np.float64)
    mu_col = (y0 / y0.sum(axis=1, keepdims=True)) @ np.asarray(probe.species_masses, dtype=np.float64)
    p_anchor = chem_radius_anchor_bar(probe.p_bar, chem_p_span_dyn(cp)[1], rocky)
    g_anchor, r_anchor = cp["gs_cgs"], rp_cm
    for _ in range(3):
        T_col = (
            np.asarray(probe.T_base, dtype=np.float64)
            if tp_eval is None
            else np.asarray(build_tp(cp, g_anchor)[0](jnp.asarray(theta[3:]), jnp.asarray(probe.p_bar)))
        )
        r_anchor = radius_at_anchor(probe.p_bar, T_col, mu_col, rp_cm, cp["gs_cgs"], cp["p_ref_bar"], p_anchor)
        g_anchor = cp["gs_cgs"] * (rp_cm / r_anchor) ** 2
    log(
        f"[fwd] radius anchor: chemistry runs R_p = {r_anchor:.4e} cm, "
        f"g = {g_anchor:.1f} cm/s2 at its {p_anchor:g} bar grid anchor, from "
        f"the catalogue {rp_cm:.4e} cm / {cp['gs_cgs']:.1f} cm/s2 at "
        f"p_ref_bar = {cp['p_ref_bar']:g} bar "
        f"({100.0 * (1.0 - r_anchor / rp_cm):.2f}% smaller radius, "
        f"{100.0 * (g_anchor / cp['gs_cgs'] - 1.0):.1f}% stronger gravity); "
        "the Guillot T-P uses the same gravity"
    )
    return r_anchor, g_anchor


# size-limit-ok: the factory closes over the override set and the T-P hook this function rebinds
def assemble_chem(cp: dict, log: Callable[..., None]) -> Any:
    """Shared heavy-path assembly (run_model AND adjoint_diag).

    The resolved run profile with the structural composition pinned into cfg_overrides, the on-graph T-P hook,
    theta, and a chemistry-build factory. One code path, so the adjoint diagnostics analyze exactly the model the
    forecasts ran.
    """
    # import order is load-bearing: vulcan_chem (env + x64) before jax/exojax
    from types import SimpleNamespace

    from jwst_tool import engine_config as config

    # Kinetics network: the engine freezes network/atom_list at ITS first
    # import, so the selection must land before vulcan_chem arrives; a
    # conflicting in-process import raises. Set for EVERY network, the default
    # included: skipping it for "sncho" would let a sncho request that follows
    # a sncho2025 run in the same process be served by sncho2025 and cached
    # under the sncho key.
    net_path, net_atoms = NETWORKS[cp["network"]]
    os.environ["VULCAN3_NETWORK"] = net_path
    os.environ["VULCAN3_ATOM_LIST"] = net_atoms
    log(f"[fwd] kinetics network: {cp['network']} ({net_path}, atoms {net_atoms})")
    from vulcan_forward import vulcan_chem

    # isort: split
    import jax

    # Persistent XLA compile cache, for adjoint_diag's step-VJP: a long cold
    # compile on CPU.
    jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "vulcan3"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    tp_eval, n_tp, tp_vals, theta_names = build_tp(cp, cp["gs_cgs"])
    # Composition is set STRUCTURALLY in the cfg abundances (below), never as a
    # parameter perturbation, so lnZ and c_o are exactly 0; only lnKzz and the
    # T-P parameters are live theta directions.
    theta = np.asarray(
        vulcan_chem.ChemParams(lnZ=0.0, c_o=0.0, lnKzz=np.log(cp["kzz_x"]), tp=tuple(tp_vals)).to_vector(),
        dtype=np.float64,
    )
    log(f"[fwd] params {cp}")
    log(f"[fwd] theta {dict(zip(theta_names, np.round(theta, 4), strict=True))}")

    profile = rt_profile_common(cp, config)
    profile["yconv_cri"] = cp["yconv_cri"]
    profile["yconv_min"] = cp["yconv_min"]
    profile["co_mode"] = "fixed_O"
    profile["dt_max"] = DT_MAX_S
    rp_cm = profile["rp_cm"]
    ovr, tp_path = _cfg_overrides(cp, rp_cm, net_path, net_atoms, log)
    profile["cfg_overrides"] = ovr
    cfg_chk = _check_cfg_constants(cp, config, tp_path, log)

    ovr.update(elemental_abundances(cp["met_x_solar"], cp["co_ratio"]))
    log(
        f"[fwd] structural composition: {cp['met_x_solar']:g}x solar metals, "
        f"C/O = {cp['co_ratio']:.3f} (C_H {ovr['C_H']:.3e}, O_H {ovr['O_H']:.3e})"
    )

    def _build_chem(extra_abun: dict | None = None, tag: str = "baseline", photo_frq: int | None = None) -> Any:
        prof = dict(profile)
        prof["cfg_overrides"] = {**ovr, **extra_abun} if extra_abun else ovr
        if photo_frq is not None:  # escalation rebuild; see certified_solve
            prof["cfg_overrides"] = {
                **prof["cfg_overrides"],
                "ini_update_photo_frq": photo_frq,
                "final_update_photo_frq": photo_frq,
            }
        t_b = time.time()
        chem_b = vulcan_chem.build_chem_model(prof, tp_eval=tp_eval, n_tp_params=n_tp)
        log(f"[fwd] chemistry model ({tag}) ready in {time.time() - t_b:.0f} s")
        return chem_b

    # One gravity per run: the chemistry's own anchor, the same in the Guillot T-P.
    probe = _build_chem(tag="geometry probe")
    r_anchor, g_anchor = _radius_anchor(cp, theta, probe, rp_cm, tp_eval, bool(cfg_chk.rocky), log)
    ovr["Rp"] = r_anchor  # Mp untouched; VULCAN derives gs from the pair
    tp_eval = build_tp(cp, g_anchor)[0]

    return SimpleNamespace(
        profile=profile,
        theta=theta,
        theta_names=theta_names,
        tp_eval=tp_eval,
        n_tp=n_tp,
        build_chem=_build_chem,
        abundance_overrides=elemental_abundances,
        config=config,
    )
