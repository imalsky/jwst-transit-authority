"""Validate a run request and return its canonical parameter dict."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jwst_tool import picaso_engine as _pe, planets
from jwst_tool.model_defs import (
    _VERSION,
    ALPHA_CLOUD_DEFAULT,
    CHEM_PROVIDERS,
    CLOUD_ON_DEFAULT,
    CO_DEFAULT,
    DEFAULT_NETWORK,
    F_DIURNAL_DEFAULT,
    FORMER_EXTRA_MOLECULES,
    JAC_METHOD_DEFAULT,
    KZZ_CONST_DEFAULT,
    KZZ_KDEEP_DEFAULT,
    KZZ_KMAX_DEFAULT,
    KZZ_PLEV_DEFAULT_BAR,
    KZZ_X_DEFAULT,
    LOG_GAMMA_DEFAULT,
    LOG_KAPPA_CLOUD_DEFAULT,
    LOG_KAPPA_DEFAULT,
    MET_DEFAULT,
    NETWORKS,
    NZ_DEFAULT,
    NZ_RANGE,
    P_BTM_FILE_BAR,
    P_BTM_PARAMETRIC_BAR,
    P_BTM_RANGE,
    P_REF_DEFAULT_BAR,
    PICASO_NO_SPECIES,
    PICASO_P_BTM_MAX,
    PICASO_QUENCH_DEFAULT,
    RFACV_DEFAULT,
    RFACV_RANGE,
    RT_INTEGRATION_DEFAULT,
    RT_PTOP_DEFAULT,
    T_WINDOW,
    TP_FILE_SHIPPED,
    TP_PARAM_NAMES,
    USE_CONDENSE_DEFAULT,
    USE_MOLDIFF_DEFAULT,
    USE_PHOTO_DEFAULT,
    USE_RAYLEIGH_DEFAULT,
    USE_VM_MOL_DEFAULT,
    YCONV_DEFAULT,
    YCONV_MIN_DEFAULT,
    YCONV_MIN_RANGE,
    YCONV_RANGE,
    default_tirr,
)
from jwst_tool.param_checks import (
    check_composition,
    check_condensation,
    check_diffusion,
    check_fisher_rows,
    check_kzz,
    check_molecules,
    check_rt_settings,
    check_system,
    check_tint,
)
from jwst_tool.tp_tables import read_tp_table, resolve_tp_file, shipped_tp_table_is_default

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


def default_p_btm_bar(params: dict) -> float:
    """Chemistry-grid bottom (bar) for this run.

    The shipped W39b cfg's 7.6 bar in file mode, a round 10 bar for the parametric profiles.
    """
    mode = str(params.get("tp_mode", default_tp_mode(params)))
    return P_BTM_FILE_BAR if mode == "file" else P_BTM_PARAMETRIC_BAR


# Structure default: a MEASURED T-P/Kzz table VERIFIED end-to-end for a planet
# is that planet's default structure, Guillot everywhere else. File mode has no
# T-P Fisher row, so switch to Guillot when a temperature row is needed.
def default_tp_mode(params: dict) -> str:
    """tp_mode default.

    The planet's measured table where a default run on it is VERIFIED, else the analytic Guillot profile (having a
    table is not enough -- see shipped_tp_table_is_default).

    EMISSION always defaults to Guillot: the bundled tables are TERMINATOR
    profiles, the wrong structure for a dayside eclipse. Selecting one for
    emission stays possible as the user's explicit choice.

    The PICASO engine defaults to its climate solver in both geometries.
    """
    if str(params.get("chem_provider", "vulcan")) == "picaso":
        return "picaso_climate"
    if str(params.get("science_mode", "transmission")) == "emission":
        return "guillot"
    planet = str(params.get("planet", "wasp39b"))
    if shipped_tp_table_is_default(planet):
        return "file"
    return "guillot"


# Every input key canonical_params (and its helpers) actually reads. An
# AST-walking test re-derives this set from the source, so it cannot rot.
_PARAM_KEYS_READ = frozenset(
    {
        "Tint",
        "Tirr",
        "alpha_cloud",
        "chem_provider",
        "cloud_on",
        "co_ratio",
        "extra_mols",
        "f_diurnal",
        "fisher_params",
        "gs_cgs",
        "jac_method",
        "kzz_const",
        "kzz_kdeep",
        "kzz_kmax",
        "kzz_mode",
        "kzz_plev",
        "kzz_x",
        "log_gamma",
        "log_kappa",
        "log_kappa_cloud",
        "met_x_solar",
        "network",
        "nz",
        "orbit_au",
        "p_btm_bar",
        "p_ref_bar",
        "picaso_quench",
        "planet",
        "rfacv",
        "rp_rjup",
        "rstar_rsun",
        "rt_integration",
        "rt_ptop_bar",
        "science_mode",
        "sflux",
        "sl_angle_deg",
        "star_feh",
        "star_logg",
        "star_teff",
        "tp_file",
        "tp_file_path",
        "tp_file_sha1",
        "tp_mode",
        "tint_from_default",
        "use_condense",
        "use_moldiff",
        "use_photo",
        "use_rayleigh",
        "use_vm_mol",
        "wo_mols",
        "yconv_cri",
        "yconv_min",
    }
)


# Output-only keys of the canonical dict itself: share_config validates a SAVED
# payload by feeding it back in, so echo fields must not read as unknown.
# "data_revision" is appended from the environment below and re-derived on
# every call (never inherited), so canonical_params must accept its own output.
_PARAM_KEYS_ECHOED = frozenset({"version", "data_revision", "picaso_stack"})


_KNOWN_PARAM_KEYS = _PARAM_KEYS_READ | _PARAM_KEYS_ECHOED


# Misspellings worth a pointed hint; unknown keys are refused, never dropped
# quietly (standing fail-loud rule).
_PARAM_KEY_HINTS = {"mode": "science_mode", "metallicity": "met_x_solar", "co": "co_ratio"}


# Keys of the removed line-by-line opacity mode and Mie condensate deck,
# refused BY NAME so an old config says what was dropped.
_REMOVED_PARAM_KEYS = frozenset(
    {"opacity_mode", "nu_pts", "rt_dit_res", "broadening", "mie_condensate", "mie_log_rg", "mie_sigmag", "mie_log_mmr"}
)

# Keys of the removed settling, escape and boundary-flux settings, refused by name.
_REMOVED_TRANSPORT_KEYS = frozenset({"use_settling", "diff_esc", "top_flux", "bot_flux"})


# What a custom request states itself: its system, UV spectrum, T-P mode and interior temperature. Nothing is taken from
# another planet. A PICASO request reads no UV spectrum, so it may leave sflux out.
CUSTOM_REQUIRED_KEYS = (
    "rp_rjup",
    "gs_cgs",
    "rstar_rsun",
    "orbit_au",
    "star_teff",
    "star_logg",
    "star_feh",
    "sflux",
    "tp_mode",
    "Tint",
)


def _check_request_keys(params: dict) -> None:
    """Refuse unknown and removed keys by name, and the removed settling, escape and boundary-flux settings."""
    transport = sorted(set(params) & _REMOVED_TRANSPORT_KEYS)
    if transport:
        raise ValueError(
            f"parameter key(s) {transport} were removed: settling, escape and boundary fluxes are not modelled, and "
            "from model version 63 a payload no longer carries them. Drop the keys."
        )
    removed = sorted(set(params) & _REMOVED_PARAM_KEYS)
    if removed:
        raise ValueError(
            f"parameter key(s) {removed} were removed in 0.48.0 with the "
            "sampled line-by-line opacity mode and the Mie condensate deck; "
            "correlated-k is the only opacity path now. Drop the keys (the "
            "power-law cloud deck, cloud_on, stays)."
        )
    unknown = sorted(set(params) - _KNOWN_PARAM_KEYS)
    if unknown:
        hints = "".join(
            f" (did you mean {_PARAM_KEY_HINTS[k]!r} instead of {k!r}?)" for k in unknown if k in _PARAM_KEY_HINTS
        )
        raise ValueError(
            f"unknown parameter key(s) {unknown}.{hints} An unknown key is "
            "refused, never dropped: the canonical key set is "
            "run_params._KNOWN_PARAM_KEYS."
        )


# The values the VULCAN-only settings take in a PICASO payload: the solver gates are pinned to their defaults, the
# photolysis settings are zeroed with photolysis off, and the unused Kzz profiles are zeroed (sflux comes per planet).
PICASO_INERT = {
    "yconv_cri": YCONV_DEFAULT,
    "yconv_min": YCONV_MIN_DEFAULT,
    "sl_angle_deg": 0.0,
    "f_diurnal": F_DIURNAL_DEFAULT,
    "kzz_x": KZZ_X_DEFAULT,
    "kzz_kmax": 0.0,
    "kzz_plev": 0.0,
    "kzz_kdeep": 0.0,
}


def _engine_settings(params: dict, sysd: dict) -> tuple[str, str, bool, str]:
    """Return (tp_mode, chem_provider, is PICASO, network), refusing settings of the other engine."""
    tp_mode = str(params.get("tp_mode", default_tp_mode(params)))
    if tp_mode not in TP_PARAM_NAMES:
        raise ValueError(
            f"unknown tp_mode {tp_mode!r}: choose from "
            f"{list(TP_PARAM_NAMES)} -- a Guillot profile, 'file' with an "
            "explicit table, or the PICASO climate solver."
        )
    provider = str(params.get("chem_provider", "vulcan"))
    if provider not in CHEM_PROVIDERS:
        raise ValueError(
            f"unknown chem_provider {provider!r}: choose 'vulcan' (VULCAN-JAX "
            "photochemical kinetics, the default) or 'picaso' (PICASO climate "
            "+ equilibrium chemistry)."
        )
    pic = provider == "picaso"
    if tp_mode == "picaso_climate" and not pic:
        raise ValueError("tp_mode='picaso_climate' is PICASO's climate solver: it needs chem_provider='picaso'.")
    if pic:
        # VULCAN-JAX physics has no PICASO counterpart: refused, never
        # silently dropped. A value equal to the one a PICASO payload carries
        # passes, so canonical_params still maps its own output to itself.
        vulcan_only = [k for k in ("use_photo", "use_moldiff", "use_vm_mol", "use_condense") if params.get(k)]
        if params.get("network") not in (None, ""):
            vulcan_only.append("network")
        if str(params.get("kzz_mode", "const")) != "const":
            vulcan_only.append(f"kzz_mode={params.get('kzz_mode')!r}")
        inert = dict(PICASO_INERT, sflux=sysd["sflux"])
        vulcan_only += [f"{k}={params[k]!r}" for k, v in inert.items() if k in params and params[k] != v]
        if vulcan_only:
            raise ValueError(
                f"{vulcan_only} are VULCAN-JAX settings. The PICASO engine is "
                "equilibrium chemistry, optionally quenched at one constant "
                "Kzz (picaso_quench + kzz_const); drop them for "
                "chem_provider='picaso'."
            )
        if params.get("jac_method", JAC_METHOD_DEFAULT) == "ad" and params.get("fisher_params"):
            raise ValueError(
                "jac_method='ad' is not available with chem_provider='picaso': "
                "PICASO runs outside JAX, so its Jacobian rows are certified "
                "finite differences (jac_method='fd')."
            )
    elif "rfacv" in params or "picaso_quench" in params:
        raise ValueError("rfacv and picaso_quench are PICASO settings; set chem_provider='picaso' or drop them.")
    network = "" if pic else str(params.get("network", DEFAULT_NETWORK))
    if not pic and network not in NETWORKS:
        raise ValueError(
            f"unknown network {network!r}: choose from {list(NETWORKS)} "
            "('sncho2025' = current upstream's S-N-C-H-O network, the "
            "default; 'sncho' = the 2021 S-N-C-H-O file, needed for "
            "condensation; 'ncho' = the sulfur-free N-C-H-O network, a "
            "cheaper solve with no SO2/H2S/CS2/OCS)."
        )
    return tp_mode, provider, pic, network


@dataclass(frozen=True)
class _ColumnSettings:
    """The science mode, the column bottom and the T-P table behind it."""

    science_mode: str
    p_btm_bar: float
    rt_ptop_bar: float
    tp_file: str
    tp_file_sha1: str
    tp_table: Any


@dataclass(frozen=True)
class _SolverSettings:
    """The layer count, the convergence gates and the UV spectrum."""

    nz: int
    yconv_cri: float
    yconv_min: float
    sflux: str


def _column_settings(params: dict, tp_mode: str, pic: bool) -> _ColumnSettings:
    """Return the science mode, the column bottom and the T-P table, validating the table on this path."""
    # tp_mode="file": resolve + validate the table NOW (numpy parse + hash, no
    # engine imports) so a bad upload fails at the API and the cache key is
    # content-addressed, never path-addressed.
    science_mode = str(params.get("science_mode", "transmission"))
    # Resolved BEFORE the table is validated: the coverage gate runs against
    # THIS run's span, not the shipped cfg's.
    p_btm_bar = float(params.get("p_btm_bar", default_p_btm_bar(params)))
    if not P_BTM_RANGE[0] <= p_btm_bar <= P_BTM_RANGE[1]:
        raise ValueError(
            f"p_btm_bar={p_btm_bar:g} outside {P_BTM_RANGE} bar (the "
            "chemistry and RT column bottom): below it the deep quench region "
            "is cut off, above it no shipped profile stays inside the "
            f"modelable window {T_WINDOW} K."
        )
    if pic and p_btm_bar > PICASO_P_BTM_MAX:
        raise ValueError(
            f"p_btm_bar={p_btm_bar:g} bar is below the PICASO climate and "
            f"quench column, which ends at {PICASO_P_BTM_MAX:g} bar."
        )
    rt_ptop_bar = float(params.get("rt_ptop_bar", RT_PTOP_DEFAULT))
    tp_file, tp_file_sha1, tp_table = "", "", None
    if tp_mode == "file":
        tp_path, tp_file_sha1 = resolve_tp_file(params)
        tp_table = read_tp_table(tp_path, span=(rt_ptop_bar * 1.0e6, p_btm_bar * 1.0e6))
        tp_file = str(params.get("tp_file", TP_FILE_SHIPPED))
    if science_mode not in ("transmission", "emission"):
        raise ValueError(
            f"unknown science_mode {science_mode!r}: choose 'transmission' "
            "(transit depth) or 'emission' (secondary-eclipse depth)"
        )
    return _ColumnSettings(science_mode, p_btm_bar, rt_ptop_bar, tp_file, tp_file_sha1, tp_table)


def _planet(params: dict) -> tuple[str, dict]:
    """Return the planet and its system: the registry entry, or a custom planet's own, in the registry's shape.

    A custom request missing any of CUSTOM_REQUIRED_KEYS is refused in one sentence naming each. Its sflux under PICASO,
    inert, is the stated value or "".
    """
    planet = str(params.get("planet", "wasp39b"))
    if planet in planets.PLANETS:
        return planet, planets.PLANETS[planet]
    if planet != "custom":
        raise ValueError(f"unknown planet {planet!r}")
    pic = str(params.get("chem_provider", "vulcan")) == "picaso"
    missing = [k for k in CUSTOM_REQUIRED_KEYS if k not in params and not (pic and k == "sflux")]
    if missing:
        raise ValueError(
            f"a custom planet states its own system, and {missing} are missing: give the planet's radius and gravity, "
            "the star's radius, T_eff, log g and [Fe/H], the orbit, the UV spectrum (VULCAN), the T-P profile (Guillot "
            "or an uploaded table) and the interior temperature Tint. Nothing is taken from another planet."
        )
    return planet, {
        "rp_rjup": params["rp_rjup"],
        "gs_cgs": params["gs_cgs"],
        "rstar_rsun": params["rstar_rsun"],
        "orbit_au": params["orbit_au"],
        "sflux": params.get("sflux", ""),
        "star": {"teff": params["star_teff"], "log_g": params["star_logg"], "metallicity": params["star_feh"]},
    }


def _solver_settings(params: dict, sysd: dict, pic: bool) -> _SolverSettings:
    """Return the grid, the convergence gates and the UV spectrum, refusing out-of-range values."""
    nz_raw = params.get("nz", NZ_DEFAULT)
    if isinstance(nz_raw, bool) or not float(nz_raw).is_integer():
        raise ValueError(f"nz={nz_raw!r} is not a whole number of layers: give an integer in {list(NZ_RANGE)}.")
    nz = int(float(nz_raw))
    if not NZ_RANGE[0] <= nz <= NZ_RANGE[1]:
        raise ValueError(
            f"nz={nz} outside the validated layer-count range {NZ_RANGE} "
            "(applied to the distinct chemistry and RT grids)"
        )
    yconv_cri = float(params.get("yconv_cri", YCONV_DEFAULT))
    if not YCONV_RANGE[0] <= yconv_cri <= YCONV_RANGE[1]:
        raise ValueError(
            f"yconv_cri={yconv_cri:g} outside the validated range {YCONV_RANGE} (steady-state convergence tolerance)"
        )
    yconv_min = float(params.get("yconv_min", YCONV_MIN_DEFAULT))
    if not YCONV_MIN_RANGE[0] <= yconv_min <= YCONV_MIN_RANGE[1]:
        raise ValueError(
            f"yconv_min={yconv_min:g} outside the validated range {YCONV_MIN_RANGE} (the loose certification branch)"
        )
    if yconv_min < yconv_cri:
        raise ValueError(
            f"yconv_min={yconv_min:g} is below yconv_cri={yconv_cri:g}: the "
            "tight branch could then never certify. Raise yconv_min or lower "
            "yconv_cri."
        )
    sflux = str(params.get("sflux", sysd["sflux"]))
    if sflux not in planets.SFLUX_CHOICES and not (pic and sflux == ""):  # "": a custom PICASO run, no UV read
        raise ValueError(f"unknown stellar UV spectrum {sflux!r} (choose from {list(planets.SFLUX_CHOICES)})")
    return _SolverSettings(nz, yconv_cri, yconv_min, sflux)


@dataclass(frozen=True)
class _Resolved:
    """What the checks settled before the payload is filled."""

    tp_mode: str
    provider: str
    pic: bool
    network: str
    planet: str
    sysd: dict
    column: _ColumnSettings
    solver: _SolverSettings


def _filled_payload(params: dict, r: _Resolved) -> dict:
    """Return the canonical key set with every default filled, at the request's own values (_rounded rounds last)."""
    star_ref = r.sysd["star"]
    extra_mols = {str(m) for m in (params.get("extra_mols") or [])}
    if extra_mols & FORMER_EXTRA_MOLECULES:
        logger.info("extra_mols %s dropped: always in the RT set now", sorted(extra_mols & FORMER_EXTRA_MOLECULES))
    return {
        "planet": r.planet,
        "science_mode": r.column.science_mode,
        # Star identity for the eclipse normalization Fp/Fs: part of the MODEL
        # only in emission (zeroed in transmission, where it is noise-side).
        "star_teff": float(params.get("star_teff", star_ref["teff"])),
        "star_logg": float(params.get("star_logg", star_ref["log_g"])),
        "star_feh": float(params.get("star_feh", star_ref["metallicity"])),
        "nz": r.solver.nz,
        "yconv_cri": r.solver.yconv_cri,
        "yconv_min": r.solver.yconv_min,
        "rp_rjup": float(params.get("rp_rjup", r.sysd["rp_rjup"])),
        "gs_cgs": float(params.get("gs_cgs", r.sysd["gs_cgs"])),
        "rstar_rsun": float(params.get("rstar_rsun", r.sysd["rstar_rsun"])),
        "orbit_au": float(params.get("orbit_au", r.sysd["orbit_au"])),
        "sflux": r.solver.sflux,
        "met_x_solar": float(params.get("met_x_solar", MET_DEFAULT)),
        "co_ratio": float(params.get("co_ratio", CO_DEFAULT)),
        # Kzz default follows the structure: a tabulated T-P with a Kzz column
        # supplies the mixing profile too, otherwise constant.
        "kzz_mode": str(
            params.get(
                "kzz_mode",
                "file"
                if (
                    not r.pic
                    and r.tp_mode == "file"
                    and r.column.tp_table is not None
                    and r.column.tp_table["Kzz"] is not None
                )
                else "const",
            )
        ),
        "kzz_x": float(params.get("kzz_x", KZZ_X_DEFAULT)),
        "kzz_const": float(params.get("kzz_const", KZZ_CONST_DEFAULT)),
        # parametric Kzz profiles: Pfunc (kzz_kmax cm^2/s, kzz_plev bar) and
        # JM16 (kzz_kdeep); param_checks zeroes the unused ones for cache hygiene
        "kzz_kmax": float(params.get("kzz_kmax", KZZ_KMAX_DEFAULT)),
        "kzz_plev": float(params.get("kzz_plev", KZZ_PLEV_DEFAULT_BAR)),
        "kzz_kdeep": float(params.get("kzz_kdeep", KZZ_KDEEP_DEFAULT)),
        "tp_mode": r.tp_mode,
        # file-mode identity: source label + content hash, so two tables can
        # never share a cache entry ("" outside file mode)
        "tp_file": r.column.tp_file,
        "tp_file_sha1": r.column.tp_file_sha1,
        # _default_tint and _guillot_tirr fill the defaults
        "Tirr": float(params.get("Tirr", 0.0)),
        "Tint": float(params.get("Tint", 0.0)),
        "log_kappa": float(params.get("log_kappa", LOG_KAPPA_DEFAULT)),
        "log_gamma": float(params.get("log_gamma", LOG_GAMMA_DEFAULT)),
        # physical VULCAN knobs (all flow through cfg_overrides; the defaults
        # are the W39b cfg values)
        "use_photo": bool(params.get("use_photo", USE_PHOTO_DEFAULT and not r.pic)),
        "sl_angle_deg": float(params.get("sl_angle_deg", planets.SL_ANGLE_DEG[r.column.science_mode])),
        "f_diurnal": float(params.get("f_diurnal", F_DIURNAL_DEFAULT)),
        "use_moldiff": bool(params.get("use_moldiff", USE_MOLDIFF_DEFAULT and not r.pic)),
        # Upwind molecular-diffusion advection: PINNED explicitly, never
        # inherited from the engine.
        "use_vm_mol": bool(params.get("use_vm_mol", USE_VM_MOL_DEFAULT)),
        "use_rayleigh": bool(params.get("use_rayleigh", USE_RAYLEIGH_DEFAULT)),
        # ExoJAX RT knobs. rt_ptop_bar is the MODEL top and the chemistry grid
        # follows it (engine rule), so no RT layer is clamped above the
        # chemistry. rt_integration picks the ArtTransPure chord scheme.
        "rt_ptop_bar": r.column.rt_ptop_bar,
        "rt_integration": str(params.get("rt_integration", RT_INTEGRATION_DEFAULT)),
        # The pressure rp_rjup and gs_cgs apply at. A catalogue radius is the
        # transit radius near the terminator photosphere, NOT the RT-grid
        # bottom exojax defaults to; anchoring it there inflates every depth.
        "p_ref_bar": float(params.get("p_ref_bar", P_REF_DEFAULT_BAR)),
        # chemistry + RT column bottom (model_defs.P_BTM_RANGE)
        "p_btm_bar": r.column.p_btm_bar,
        "cloud_on": bool(params.get("cloud_on", CLOUD_ON_DEFAULT)),
        "log_kappa_cloud": float(params.get("log_kappa_cloud", LOG_KAPPA_CLOUD_DEFAULT)),
        "alpha_cloud": float(params.get("alpha_cloud", ALPHA_CLOUD_DEFAULT)),
        # Detection-only condensation: param_checks refuses ANY derivative
        "use_condense": bool(params.get("use_condense", USE_CONDENSE_DEFAULT)),
        "extra_mols": sorted(extra_mols - FORMER_EXTRA_MOLECULES),
        # Leave-one-out spectrum set: None = every RT molecule (the detect
        # default), [] skips the block. param_checks puts it in fold order.
        "wo_mols": None if params.get("wo_mols") is None else [str(m) for m in params["wo_mols"]],
        "fisher_params": sorted({str(p) for p in (params.get("fisher_params") or [])}),
        # Jacobian method: "fd" (certified FD, the default) or "ad" (one
        # warm-started jvp per chemistry row, photo-on and VULCAN only; docs/forward_model.md).
        "jac_method": str(params.get("jac_method", JAC_METHOD_DEFAULT)),
        "chem_provider": r.provider,
        "network": r.network,
        "version": _VERSION,
    }


def _picaso_fields(cp: dict, params: dict, tp_table: Any, science_mode: str, tp_mode: str, nz: int) -> None:
    """Add the PICASO-only keys, which exist only in PICASO payloads, so no VULCAN cache key moves."""
    climate = tp_mode == "picaso_climate"
    # PICASO-only keys exist only in PICASO payloads, so no VULCAN cache
    # key moves. rfacv is a climate input (inert, 0, for the other
    # profiles); kzz_const keys only a quenched column.
    cp["rfacv"] = float(params.get("rfacv", RFACV_DEFAULT[science_mode])) if climate else 0.0
    cp["picaso_quench"] = bool(params.get("picaso_quench", PICASO_QUENCH_DEFAULT))
    cp["picaso_stack"] = _pe.PICASO_STACK
    if climate and not RFACV_RANGE[0] <= cp["rfacv"] <= RFACV_RANGE[1]:
        raise ValueError(
            f"rfacv={cp['rfacv']:g} outside {list(RFACV_RANGE)} (climate "
            "heat redistribution: 0.5 planet average, 1.0 dayside)"
        )
    # VULCAN solver and photolysis knobs are inert under PICASO
    cp["yconv_cri"], cp["yconv_min"] = YCONV_DEFAULT, YCONV_MIN_DEFAULT
    if tp_mode == "file" and cp["picaso_quench"]:
        need = _pe.eval_grid(cp["p_btm_bar"], cp["rt_ptop_bar"], nz, True)[-1]
        if tp_table["P_dyn"].max() / 1.0e6 < need * (1 - 1e-9):
            raise ValueError(
                f"the T-P table ends at {tp_table['P_dyn'].max() / 1e6:.3g} "
                f"bar but the PICASO quench search needs it to {need:.3g} "
                "bar; extend the table or switch quench off."
            )
    no_sp = sorted(set(cp["extra_mols"]) & PICASO_NO_SPECIES)
    if no_sp:
        raise ValueError(
            f"extra_mols {no_sp} are not in PICASO's equilibrium species list; drop them for chem_provider='picaso'."
        )


def _system(cp: dict) -> dict:
    return dict(star_teff=cp["star_teff"], rstar_rsun=cp["rstar_rsun"], orbit_au=cp["orbit_au"])


def _default_tint(params: dict, cp: dict, r: _Resolved) -> None:
    """Fill T_int from planets.default_tint, the GUI's definition too, when the request does not state it.

    ``tint_from_default`` records that with the result, which then carries a caveat. A saved payload carries the flag,
    so it maps to itself; a flag that names a T_int other than the default is refused. Both are zeroed where inert.
    """
    from_default = "Tint" not in params or bool(params.get("tint_from_default", False))
    if from_default:
        if r.planet not in planets.PLANETS:
            raise ValueError("tint_from_default=True: a custom planet has no default T_int; state Tint.")
        default = planets.default_tint(r.sysd)
        if "Tint" in params and float(params["Tint"]) != default:
            raise ValueError(
                f"tint_from_default=True names the default T_int, {default:g} K, but Tint={params['Tint']}: drop "
                "tint_from_default to run at the stated Tint."
            )
        cp["Tint"] = default
    cp["tint_from_default"] = from_default


def tint_caveat(cp: dict) -> str | None:
    """Return the caveat of a run that read T_int without the request stating it; the run log and the result share it.

    Parameters
    ----------
    cp : dict
        Canonical parameters.

    Returns
    -------
    str or None
        The sentence, or None when T_int was stated or is not read.
    """
    if cp["tint_from_default"]:
        return f"Interior temperature not stated: the run used the default T_int = {cp['Tint']:g} K."
    return None


def _guillot_tirr(params: dict, cp: dict, r: _Resolved) -> None:
    """Fill and check the Guillot T_irr: the request's value, else sqrt(2) * T_eq of the selected system.

    The default is the DAYSIDE T_eq in emission and comes from planets.default_tirr, so the API and the GUI cannot
    diverge. Outside Guillot mode T_irr is inert and zeroed later.
    """
    if r.tp_mode != "guillot":
        return
    if "Tirr" in params:
        planets.check_tirr(cp["Tirr"], derived=False)
        return
    cp["Tirr"] = default_tirr(r.planet, system=_system(cp), science_mode=r.column.science_mode)
    planets.check_tirr(cp["Tirr"], derived=True)


def _zero_inert_settings(cp: dict, sysd: dict, science_mode: str, tp_mode: str) -> None:
    """Normalize settings inert for the chosen modes, so one physics maps to one key."""
    climate = tp_mode == "picaso_climate"
    if not cp["use_photo"]:  # photolysis settings are inert without photo
        cp["sl_angle_deg"] = 0.0
        cp["f_diurnal"] = F_DIURNAL_DEFAULT
        # The UV spectrum feeds only photolysis: normalize it photo-off so
        # identical physics never caches under two keys.
        cp["sflux"] = str(sysd["sflux"])
        if not climate:  # the semi-major axis heats the climate
            cp["orbit_au"] = float(sysd["orbit_au"])
    if not cp["cloud_on"]:  # cloud knobs are inert when the deck is off
        cp["log_kappa_cloud"] = 0.0
        cp["alpha_cloud"] = 0.0
    # Science-mode hygiene + gating
    if science_mode == "emission":
        # Rayleigh is transmission-only physics (the pure-absorption emission
        # solver must not read scattering as thermal absorption) and the chord
        # scheme exists only in transmission: normalize both for the cache.
        cp["use_rayleigh"] = False
        cp["rt_integration"] = RT_INTEGRATION_DEFAULT
    elif not climate:
        # transmission: the star lives only on the noise side.
        cp["star_teff"] = cp["star_logg"] = cp["star_feh"] = 0.0
    # drop fields inert for the chosen modes so they don't fragment the cache
    if tp_mode != "guillot":
        cp["Tirr"] = cp["log_kappa"] = cp["log_gamma"] = 0.0
        if not climate:
            cp["Tint"] = 0.0
            cp["tint_from_default"] = False


# Decimals a canonical value keeps, enough for every registry value (rp 5, g 2), and those kept to six significant
# figures. Rounding is the last step: every check reads the request's own value, so a value outside its range is
# refused even when rounding would carry it inside.
_DECIMALS = {
    "star_teff": 1,
    "star_logg": 2,
    "star_feh": 2,
    "yconv_cri": 6,
    "yconv_min": 6,
    "rp_rjup": 5,
    "gs_cgs": 2,
    "rstar_rsun": 4,
    "orbit_au": 5,
    "met_x_solar": 4,
    "co_ratio": 6,
    "kzz_x": 4,
    "kzz_const": 1,
    "kzz_kmax": 1,
    "kzz_kdeep": 1,
    "Tirr": 2,
    "Tint": 2,
    "log_kappa": 3,
    "log_gamma": 3,
    "sl_angle_deg": 1,
    "f_diurnal": 3,
    "p_btm_bar": 4,
    "log_kappa_cloud": 3,
    "alpha_cloud": 2,
    "rfacv": 3,
}
_SIGNIFICANT_6 = ("kzz_plev", "rt_ptop_bar", "p_ref_bar")


def _rounded(cp: dict) -> None:
    """Round the checked payload to its canonical precision, so one physics maps to one key."""
    for key, decimals in _DECIMALS.items():
        if key in cp:
            cp[key] = round(cp[key], decimals)
    for key in _SIGNIFICANT_6:
        cp[key] = float(f"{cp[key]:.6e}")


def canonical_params(params: dict) -> dict:
    """Validate a run request and return its canonical parameter dict.

    Every default is filled in, values are rounded, settings that are inert
    for the chosen modes are zeroed so one physics maps to one key, and
    ``version`` (plus ``picaso_stack`` under PICASO and ``data_revision`` when
    set) is appended. params_key hashes this dict; chem_key hashes it without
    CHEM_IRRELEVANT_PARAMS. Unknown or removed keys, out-of-range values and
    conflicting settings raise; nothing is clipped and no unknown key is
    dropped. The output maps to itself, so a saved payload can be validated
    by feeding it back in (share_config).
    """
    _check_request_keys(params)
    planet, sysd = _planet(params)
    tp_mode, provider, pic, network = _engine_settings(params, sysd)
    column = _column_settings(params, tp_mode, pic)
    solver = _solver_settings(params, sysd, pic)
    r = _Resolved(
        tp_mode=tp_mode,
        provider=provider,
        pic=pic,
        network=network,
        planet=planet,
        sysd=sysd,
        column=column,
        solver=solver,
    )
    cp = _filled_payload(params, r)
    check_system(cp, tirr_derived=tp_mode == "guillot" and "Tirr" not in params)
    _default_tint(params, cp, r)
    check_tint(cp)
    _guillot_tirr(params, cp, r)
    if pic:
        _picaso_fields(cp, params, column.tp_table, column.science_mode, tp_mode, solver.nz)
    check_rt_settings(cp)
    check_molecules(cp, network)
    co_bound, met_range = check_composition(cp, network, pic)
    check_fisher_rows(cp, tp_mode, pic, network, co_bound, met_range)
    check_condensation(cp, network)
    check_diffusion(cp)
    _zero_inert_settings(cp, sysd, column.science_mode, tp_mode)
    check_kzz(cp, tp_mode, column.tp_table, pic)
    # Science-data identity. Every key above describes the MODEL; none of them
    # moves when a k-table or CIA file is corrected under the same filename.
    # It stays out of CHEM_IRRELEVANT_PARAMS on purpose: an extra chemistry
    # miss is cheap and a false hit is not. Unset means omitted, so existing
    # keys are unchanged.
    _rounded(cp)
    rev = os.environ.get("JWST_TOOL_DATA_REVISION", "").strip()
    if rev:
        cp["data_revision"] = rev
    return cp
