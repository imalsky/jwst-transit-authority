"""Refusals of a filled canonical payload: settings each run's modes cannot model.

Each check raises one ValueError naming the value and the remedy; ``run_params.canonical_params`` runs them in order, so
the same refusal wins for any request.
"""

from __future__ import annotations

import numpy as np

from jwst_tool import planets
from jwst_tool.model_defs import (
    ALPHA_CLOUD_RANGE,
    ART_PBTM_FRACTION,
    CHEM_PARAM_NAMES,
    CLOUD_FISHER_PARAMS,
    CO_MAX,
    CO_MAX_PHOTO_OFF,
    EXTRA_MOLECULES,
    F_DIURNAL_RANGE,
    JAC_METHODS,
    KZZ_CONST_RANGE,
    KZZ_KDEEP_RANGE,
    KZZ_KMAX_RANGE,
    KZZ_MODES,
    KZZ_PLEV_RANGE,
    KZZ_X_RANGE,
    LOG_GAMMA_RANGE,
    LOG_KAPPA_CLOUD_RANGE,
    LOG_KAPPA_RANGE,
    MET_RANGE,
    MOLECULES,
    NO_EXOMOLOP_TABLE,
    P_REF_DEFAULT_BAR,
    P_REF_RANGE_BAR,
    PICASO_MET_RANGE,
    RT_PTOP_DEFAULT,
    RT_PTOP_RANGE,
    S_MOLECULES,
    SL_ANGLE_RANGE,
    TINT_RANGE,
    TP_PARAM_NAMES,
    active_molecules,
    co_bounds,
    fd_stencil,
)


def _check_range(key: str, value: float, rng: tuple, unit: str = "") -> None:
    """Refuse ``value`` outside the inclusive ``rng``; NaN refuses too."""
    if not rng[0] <= value <= rng[1]:
        raise ValueError(f"{key}={value:g} outside [{rng[0]:g}, {rng[1]:g}]{' ' + unit if unit else ''}")


def check_system(cp: dict) -> None:
    """Refuse planet, star, orbit, temperature-profile and cloud values outside the ranges the sidebar offers.

    Each is checked only where the run's modes use it, since an inert value is zeroed in the canonical payload.
    """
    rng = planets.CUSTOM_FIELD_RANGES
    _check_range("rp_rjup", cp["rp_rjup"], rng["rp"], "R_jup")
    _check_range("gs_cgs", cp["gs_cgs"], (planets.CM_PER_M * rng["g"][0], planets.CM_PER_M * rng["g"][1]), "cm s^-2")
    _check_range("rstar_rsun", cp["rstar_rsun"], rng["rstar"], "R_sun")
    _check_range("orbit_au", cp["orbit_au"], rng["a"], "au")
    climate = cp["tp_mode"] == "picaso_climate"
    # The star is read by Fp/Fs and the climate solve (all three values), and by a custom planet's T_irr and T_int
    # defaults (its T_eff), in either mode.
    if cp["science_mode"] == "emission" or climate or cp["planet"] == "custom":
        _check_range("star_teff", cp["star_teff"], rng["teff"], "K (the PHOENIX grid range exercised)")
    if cp["science_mode"] == "emission" or climate:
        _check_range("star_logg", cp["star_logg"], rng["logg"], "(log10 cm s^-2)")
        _check_range("star_feh", cp["star_feh"], rng["feh"], "dex")
    if cp["tp_mode"] == "guillot":
        _check_range("log_kappa", cp["log_kappa"], LOG_KAPPA_RANGE, "(log10 cm^2/g)")
        _check_range("log_gamma", cp["log_gamma"], LOG_GAMMA_RANGE)
    if cp["cloud_on"]:
        _check_range("log_kappa_cloud", cp["log_kappa_cloud"], LOG_KAPPA_CLOUD_RANGE, "(log10 cm^2/g)")
        _check_range("alpha_cloud", cp["alpha_cloud"], ALPHA_CLOUD_RANGE)


def tint_refusal(tint_k: float, derived: bool) -> str | None:
    """Return the refusal sentence of a T_int outside TINT_RANGE, None inside; the API and the sidebar seeder share it.

    Parameters
    ----------
    tint_k : float
        Interior temperature, K.
    derived : bool
        True when ``tint_k`` is the default planets.default_tint computed, not a value the user set.

    Returns
    -------
    str or None
        The sentence, or None when the value is accepted.
    """
    lo, hi = TINT_RANGE
    if lo <= tint_k <= hi:
        return None
    if derived:
        return (
            f"the default Tint of this star and orbit is {tint_k:g} K (planets.default_tint), outside the supported "
            f"range [{lo:g}, {hi:g}] K. Set Tint explicitly inside that range."
        )
    return f"Tint={tint_k:g} outside [{lo:g}, {hi:g}] K"


def check_tint(cp: dict, derived: bool) -> None:
    """Refuse a Guillot or climate T_int outside TINT_RANGE; run after the default is filled from the checked system."""
    if cp["tp_mode"] in ("guillot", "picaso_climate"):
        refusal = tint_refusal(cp["Tint"], derived)
        if refusal:
            raise ValueError(refusal)


def check_rt_settings(cp: dict) -> None:
    """Refuse photolysis geometry, RT top, radius reference and chord scheme outside their ranges."""
    _check_range("sl_angle_deg", cp["sl_angle_deg"], SL_ANGLE_RANGE, "deg")
    _check_range("f_diurnal", cp["f_diurnal"], F_DIURNAL_RANGE)
    _check_range("rt_ptop_bar", cp["rt_ptop_bar"], RT_PTOP_RANGE, f"bar (default {RT_PTOP_DEFAULT:g})")
    _check_range("p_ref_bar", cp["p_ref_bar"], P_REF_RANGE_BAR, "bar (the widest RT grid)")
    art_pbtm = cp["p_btm_bar"] * ART_PBTM_FRACTION
    if not cp["rt_ptop_bar"] <= cp["p_ref_bar"] <= art_pbtm:
        raise ValueError(
            f"p_ref_bar={cp['p_ref_bar']:g} must lie inside the RT grid "
            f"[{cp['rt_ptop_bar']:g}, {art_pbtm:g}] bar. It is the pressure "
            "at which rp_rjup and gs_cgs are defined, so the grid has to "
            f"cover it. {P_REF_DEFAULT_BAR:g} bar is the default for a "
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
    if not np.isfinite(cp["co_ratio"]):
        raise ValueError(f"co_ratio={cp['co_ratio']!r} is not finite: give a C/O inside [{co_lo:g}, {co_hi:g}].")
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


def check_diffusion(cp: dict) -> None:
    """Refuse upwind advection with molecular diffusion off: it moves the diffusion flux, so it would do nothing.

    vulcan3 refuses the pair too, after the build has started; this refusal costs no solve.
    """
    if cp["use_vm_mol"] and not cp["use_moldiff"]:
        raise ValueError(
            "use_vm_mol=True needs use_moldiff=True: the upwind advection acts on the molecular-diffusion flux, "
            "which is off. Set use_moldiff=True, or use_vm_mol=False."
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
    _check_range("kzz_x", cp["kzz_x"], KZZ_X_RANGE, "(multiplier on the whole Kzz profile)")
    if pic and not cp["picaso_quench"]:  # equilibrium: no transport anywhere
        cp["kzz_const"] = 0.0
    elif cp["kzz_mode"] == "const":
        _check_range("kzz_const", cp["kzz_const"], KZZ_CONST_RANGE, "cm^2/s")
    elif cp["kzz_mode"] == "Pfunc":
        _check_range("kzz_kmax", cp["kzz_kmax"], KZZ_KMAX_RANGE, "cm^2/s (Pfunc deep Kzz)")
        _check_range("kzz_plev", cp["kzz_plev"], KZZ_PLEV_RANGE, "bar (Pfunc transition pressure)")
    elif cp["kzz_mode"] == "JM16":
        _check_range("kzz_kdeep", cp["kzz_kdeep"], KZZ_KDEEP_RANGE, "cm^2/s (JM16 deep floor)")
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
