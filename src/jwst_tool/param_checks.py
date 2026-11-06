"""Refusals of a filled canonical payload: settings each run's modes cannot model.

Each check raises one ValueError naming the value and the remedy; ``run_params.canonical_params`` runs them in order, so
the same refusal wins for any request.
"""

from __future__ import annotations

import numpy as np

from jwst_tool.jacobian import fd_stencil
from jwst_tool.model_defs import (
    ART_PBTM_FRACTION,
    CHEM_PARAM_NAMES,
    CLOUD_FISHER_PARAMS,
    CO_MAX,
    CO_MAX_PHOTO_OFF,
    EXTRA_MOLECULES,
    JAC_METHODS,
    KZZ_MODES,
    MET_RANGE,
    MOLECULES,
    NO_EXOMOLOP_TABLE,
    PICASO_MET_RANGE,
    RT_PTOP_DEFAULT,
    S_MOLECULES,
    TP_PARAM_NAMES,
    active_molecules,
    co_bounds,
)


def check_rt_settings(cp: dict) -> None:
    """Refuse photolysis geometry, RT top, radius reference and chord scheme outside their ranges."""
    if not 0.0 <= cp["sl_angle_deg"] <= 89.0:
        raise ValueError(f"sl_angle_deg={cp['sl_angle_deg']} outside [0, 89] deg")
    if not 0.0 < cp["f_diurnal"] <= 1.0:
        raise ValueError(f"f_diurnal={cp['f_diurnal']} outside (0, 1]")
    if not 1.0e-9 <= cp["rt_ptop_bar"] <= 1.0e-6:
        raise ValueError(
            f"rt_ptop_bar={cp['rt_ptop_bar']:g} outside [1e-9, 1e-6] bar (the "
            f"exercised RT-top range; {RT_PTOP_DEFAULT:g} is the default)"
        )
    art_pbtm = cp["p_btm_bar"] * ART_PBTM_FRACTION
    if not cp["rt_ptop_bar"] <= cp["p_ref_bar"] <= art_pbtm:
        raise ValueError(
            f"p_ref_bar={cp['p_ref_bar']:g} must lie inside the RT grid "
            f"[{cp['rt_ptop_bar']:g}, {art_pbtm:g}] bar. It is the pressure "
            "at which rp_rjup and gs_cgs are defined, so the grid has to "
            "cover it. 1e-3 bar is the validated default for a "
            "transit-derived radius."
        )
    if cp["rt_integration"] not in ("simpson", "trapezoid"):
        raise ValueError(
            f"rt_integration={cp['rt_integration']!r}: exojax ArtTransPure supports 'simpson' (default) or 'trapezoid'"
        )


def check_molecules(cp: dict, network: str) -> None:
    """Refuse molecules the run cannot model, and put wo_mols into fold order."""
    no_table = sorted(set(cp["extra_mols"]) & NO_EXOMOLOP_TABLE)
    if no_table:  # before the universe check: the pointed reason wins
        raise ValueError(
            f"extra_mols {no_table} have no published ExoMolOP k-table and "
            "cannot be modelled (correlated-k over the published tables is "
            "the only opacity path). Drop them."
        )
    bad_mols = set(cp["extra_mols"]) - set(EXTRA_MOLECULES)
    if bad_mols:
        raise ValueError(
            f"unknown RT molecule(s) {sorted(bad_mols)}: this tool ships "
            f"opacity for {MOLECULES} plus the opt-in extras "
            f"{EXTRA_MOLECULES}. Adding one means extending the shared "
            "vulcan-forward engine (inject profile['molecule_table'] or "
            "extend vulcan_forward.constants.MOLECULES with molmass + VULCAN "
            "species name), fetching its ExoMolOP k-table, and listing it in "
            "model_defs.EXTRA_MOLECULES."
        )
    if network == "ncho":
        s_req = sorted(set(cp["extra_mols"]) & S_MOLECULES)
        if s_req:
            raise ValueError(
                f"extra_mols {s_req} are sulfur species and do not exist in "
                "the ncho network. Drop them, or use network='sncho2025'. "
                "(Sulfur species are refused rather than dropped, so the "
                "model computed is always the model asked for.)"
            )
    # Leave-one-out spectrum set
    # After the molecule-universe checks, so the RT set is final. Canonical form
    # is a deduped subset of active_molecules(cp) in fold order, which keeps the
    # cache key and the stored depth_wo rows in one order.
    active = active_molecules(cp)
    if cp["wo_mols"] is None:
        cp["wo_mols"] = list(active)
    else:
        bad_wo = sorted(set(cp["wo_mols"]) - set(active))
        if bad_wo:
            raise ValueError(
                f"wo_mols {bad_wo} are not in this run's RT molecule set "
                f"{active}. A removed-molecule spectrum exists only for "
                "molecules the RT actually includes; fix wo_mols or add the "
                "molecule via extra_mols."
            )
        req = set(cp["wo_mols"])
        cp["wo_mols"] = [m for m in active if m in req]


def check_composition(cp: dict, network: str, pic: bool) -> tuple:
    """Refuse a composition outside its range; return (C/O bounds, metallicity range)."""
    co_lo, co_hi = co_bounds(network, cp["use_photo"])  # photo off under PICASO
    if cp["co_ratio"] < co_lo:
        raise ValueError(f"C/O {cp['co_ratio']:g} is below the modelled minimum {co_lo:g}.")
    if cp["co_ratio"] > co_hi:
        # Only name an escape that would actually raise the ceiling: at
        # CO_MAX_PHOTO_OFF every one of them is already in force.
        fix = (
            ""
            if pic or co_hi >= CO_MAX_PHOTO_OFF
            else " Try the sncho2025 network, the sulfur-free ncho network, or photolysis off."
        )
        raise ValueError(
            f"C/O {cp['co_ratio']:g} is too high for this setup "
            f"(network {network!r}, photolysis "
            f"{'on' if cp['use_photo'] else 'off'}): the limit is "
            f"{co_hi:g}.{fix}"
        )
    met_rng = PICASO_MET_RANGE if pic else MET_RANGE
    if not met_rng[0] <= cp["met_x_solar"] <= met_rng[1]:
        raise ValueError(
            f"met_x_solar={cp['met_x_solar']} outside "
            f"[{met_rng[0]:g}, {met_rng[1]:g}] x solar" + (" (the PICASO engine's range)" if pic else "")
        )
    return (co_lo, co_hi), met_rng


def check_fisher_rows(cp: dict, tp_mode: str, pic: bool, network: str, co_bound: tuple, met_rng: tuple) -> None:
    """Refuse rows the run's modes do not offer, and FD stencils that would solve outside a validated range."""
    climate = tp_mode == "picaso_climate"
    _co_lo, _co_hi = co_bound
    # Fisher menu: chemistry + the tp_mode's T-P parameters (none in file mode)
    # + the cloud-deck parameters when the deck is in the model.
    if climate and "Tint" in cp["fisher_params"]:
        raise ValueError(
            "Tint is not a PICASO climate Fisher row: at the default T_int the "
            "photosphere moves < 0.02 K per K, under the climate solve's own "
            "noise, and the row failed its step-size check at 7, 10 and 25 K. "
            "T_int stays a climate setting; Guillot rows remain available."
        )
    if pic and "lnKzz" in cp["fisher_params"]:
        raise ValueError(
            "lnKzz is not a PICASO Fisher row: PICASO quenches each family at "
            "a whole model level, so the quenched abundances are a staircase "
            "in Kzz (on WASP-39 b the CO-CH4-H2O level steps once per ~0.25 "
            "dex of Kzz) and no step size gives a certified derivative."
        )
    allowed_fp = {"lnZ", "dlnCO", "lnKzz"} | set(TP_PARAM_NAMES[tp_mode])
    if cp["cloud_on"]:
        allowed_fp |= set(CLOUD_FISHER_PARAMS)
    bad_fp = set(cp["fisher_params"]) - allowed_fp
    if bad_fp:
        raise ValueError(
            f"unknown Fisher parameter(s) {sorted(bad_fp)} for tp_mode="
            f"{tp_mode!r}: choose from {CHEM_PARAM_NAMES} + "
            f"{TP_PARAM_NAMES[tp_mode]}"
            + (f" + {list(CLOUD_FISHER_PARAMS)}" if cp["cloud_on"] else "")
            + (
                ". (tp_mode='file' has NO T-P Fisher rows by design; the "
                f"cloud parameters {list(CLOUD_FISHER_PARAMS)} require "
                "cloud_on.)"
            )
        )
    if cp["jac_method"] not in JAC_METHODS:
        raise ValueError(
            f"jac_method={cp['jac_method']!r}: choose 'fd' (certified central "
            "finite differences, the default) or 'ad' (one warm-started jvp "
            "per Jacobian row)"
        )
    if cp["jac_method"] == "ad" and not cp["fisher_params"]:
        cp["jac_method"] = "fd"  # no Jacobian requested: inert knob --
        #                           normalize so it can't fragment the cache
    if (
        cp["jac_method"] == "ad"
        and not cp["use_photo"]
        and any(n not in CLOUD_FISHER_PARAMS for n in cp["fisher_params"])
    ):
        # Only CHEMISTRY rows are gated: a cloud-deck row (and the appended
        # lnR0) is an RT-only jvp taken AT the converged column, so it carries
        # no through-solver tangent -- which is the thing that is unvalidated
        # photo-off. The refusal names the asymmetry (FD self-checks and can
        # refuse, AD cannot), not a blanket noise claim.
        raise ValueError(
            "jac_method='ad' chemistry rows "
            f"({[n for n in cp['fisher_params'] if n not in CLOUD_FISHER_PARAMS]}) "
            "are not validated with photolysis off, and an AD row carries no "
            "self-check. Enable photochemistry, or use jac_method='fd', whose "
            "h-vs-2h consistency check may also refuse the row. Cloud-deck "
            "rows are RT-only and stay available photo-off."
        )
    _check_stencil_reach(cp, network, pic, co_bound, met_rng)


def _check_stencil_reach(cp: dict, network: str, pic: bool, co_bound: tuple, met_rng: tuple) -> None:
    """Refuse an FD row whose stencil would solve outside a validated range."""
    co_lo, co_hi = co_bound
    # Composition FD rows solve the chemistry at every stencil point, so each
    # point must stay inside the validated envelope (a row requested AT a
    # range edge would otherwise silently solve outside it). T-P rows
    # window-check every stencil point for the same reason.
    if cp["jac_method"] == "fd":
        for name, key, rng in (("lnZ", "met_x_solar", met_rng), ("dlnCO", "co_ratio", (co_lo, co_hi))):
            if name not in cp["fisher_params"]:
                continue
            # Stencil SHAPE keys on the photolysis-on ceiling even photo-off:
            # a one-sided row where a central one would also solve is merely
            # conservative, and a photo-independent threshold keeps every
            # cached Jacobian identical.
            offs, h = fd_stencil(name, cp[key], co_hi if pic else CO_MAX[network], PICASO_MET_RANGE[1] if pic else None)
            pts = [cp[key] * float(np.exp(s * h)) for s in offs]
            if not all(rng[0] <= p <= rng[1] for p in pts):
                raise ValueError(
                    f"{key}={cp[key]:g}: the {name} FD stencil would solve "
                    f"the chemistry at {[round(p, 3) for p in pts]}, outside "
                    f"the validated range {list(rng)}. Move {key} inward or "
                    f"drop {name} from fisher_params."
                )


def check_condensation(cp: dict, network: str) -> None:
    """Condensation is detection-only: refuse every derivative and every setting its recipe excludes."""
    # Condensation: detection-only -- refuse every derivative combo
    # (the "91% wrong" and "not a 9% mismatch" wording is test-pinned)
    if cp["use_condense"]:
        if network != "sncho":
            raise ValueError(
                "condensation (use_condense) requires network='sncho': the "
                "certified recipe condenses S8, which does not exist in the "
                "sulfur-free ncho network, and its conver_ignore list was "
                "tuned on the 89-species file, not on sncho2025's added C3 "
                "and H2CS species. Select network='sncho' or turn "
                "condensation off."
            )
        if cp["fisher_params"]:
            raise ValueError(
                "condensation (use_condense) cannot be combined with a "
                "Fisher forecast under ANY Jacobian method: the pinned "
                "reservoir is frozen at a step-sequence-dependent transient "
                "and the condensing-layer set switches discretely in "
                "temperature, so the AD tangent through it is about 91% "
                "wrong (jvp-vs-FD relative error ~0.91, not a 9% mismatch) "
                "and finite differences of it are no better. Clear "
                "fisher_params (detection still works), or use the "
                "differentiable ExoJAX cloud deck (cloud_on) for aerosol "
                "opacity."
            )
        if not cp["use_photo"]:
            raise ValueError(
                "condensation (use_condense) requires photochemistry ON: "
                "a cold no-photo condensing column has no certifiable longdy "
                "steady state, and a runtime-capped solve is never presented "
                "as converged. Enable photochemistry, or turn condensation "
                "off."
            )
        if not cp["use_moldiff"]:
            raise ValueError(
                "condensation (use_condense) requires molecular diffusion "
                "(use_moldiff): the growth term IS the species' "
                "molecular-diffusion coefficient, so every condensation rate "
                "would silently be zero. Enable it, or turn condensation "
                "off."
            )


def _zero_inactive_kzz(cp: dict) -> None:
    """Only the active mode's settings key the cache."""
    if cp["kzz_mode"] != "const":
        cp["kzz_const"] = 0.0
    if cp["kzz_mode"] != "Pfunc":
        cp["kzz_kmax"] = cp["kzz_plev"] = 0.0
    if cp["kzz_mode"] != "JM16":
        cp["kzz_kdeep"] = 0.0


def check_kzz(cp: dict, tp_mode: str, tp_table: dict | None, pic: bool) -> None:
    """Refuse a Kzz profile outside its range, and zero the settings of the modes not in use."""
    # Kzz profile mode (const / Pfunc / JM16 / file)
    if cp["kzz_mode"] not in KZZ_MODES:
        raise ValueError(
            f"unknown kzz_mode {cp['kzz_mode']!r}: choose from "
            f"{list(KZZ_MODES)}. Kzz profiles are explicit -- the GCM-scaled "
            "'scale' mode was removed."
        )
    if not 0.01 <= cp["kzz_x"] <= 100.0:
        raise ValueError(
            f"kzz_x={cp['kzz_x']} outside [0.01, 100] (multiplicative scale applied on-graph to the whole Kzz profile)"
        )
    if pic and not cp["picaso_quench"]:  # equilibrium: no transport anywhere
        cp["kzz_const"] = 0.0
    elif cp["kzz_mode"] == "const":
        if not 1.0e3 <= cp["kzz_const"] <= 1.0e13:
            raise ValueError(f"kzz_const={cp['kzz_const']:g} outside [1e3, 1e13] cm^2/s")
    elif cp["kzz_mode"] == "Pfunc":
        if not 1.0e3 <= cp["kzz_kmax"] <= 1.0e12:
            raise ValueError(f"kzz_kmax={cp['kzz_kmax']:g} outside [1e3, 1e12] cm^2/s (Pfunc deep Kzz)")
        if not 1.0e-6 <= cp["kzz_plev"] <= 1.0e3:
            raise ValueError(f"kzz_plev={cp['kzz_plev']:g} outside [1e-6, 1e3] bar (Pfunc transition pressure)")
    elif cp["kzz_mode"] == "JM16":
        if not 1.0e3 <= cp["kzz_kdeep"] <= 1.0e12:
            raise ValueError(f"kzz_kdeep={cp['kzz_kdeep']:g} outside [1e3, 1e12] cm^2/s (JM16 deep floor)")
    elif cp["kzz_mode"] == "file":
        if tp_mode != "file":
            raise ValueError(
                "kzz_mode='file' requires tp_mode='file': the tabulated Kzz "
                "lives in the Kzz column of the atm table (the upstream "
                "constraint -- Kzz_prof='file' needs atm_type='file')."
            )
        if tp_table is None or tp_table["Kzz"] is None:
            raise ValueError(
                "kzz_mode='file' requires a 'Kzz' column in the T-P table; "
                "the selected table has none. Add the column (cm^2/s) or "
                "pick a parametric kzz_mode."
            )
    _zero_inactive_kzz(cp)
