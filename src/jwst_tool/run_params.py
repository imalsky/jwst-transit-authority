"""Validate a run request and return its canonical parameter dict."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jwst_tool import picaso_engine as _pe, planets
from jwst_tool.model_defs import (
    _VERSION,
    CHEM_PROVIDERS,
    CO_DEFAULT,
    DEFAULT_NETWORK,
    NETWORKS,
    NZ_DEFAULT,
    NZ_RANGE,
    P_BTM_FILE_BAR,
    P_BTM_PARAMETRIC_BAR,
    P_BTM_RANGE,
    PICASO_NO_SPECIES,
    PICASO_P_BTM_MAX,
    RFACV_DEFAULT,
    RFACV_RANGE,
    RT_PTOP_DEFAULT,
    T_WINDOW,
    TINT_RANGE,
    TP_FILE_SHIPPED,
    TP_PARAM_NAMES,
    YCONV_DEFAULT,
    YCONV_MIN_DEFAULT,
    YCONV_MIN_RANGE,
    YCONV_RANGE,
    default_tirr,
)
from jwst_tool.param_checks import (
    check_composition,
    check_condensation,
    check_fisher_rows,
    check_kzz,
    check_molecules,
    check_rt_settings,
)
from jwst_tool.tp_tables import read_tp_table, resolve_tp_file, shipped_tp_table_is_default

if TYPE_CHECKING:
    from typing import Any


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
        "bot_flux",
        "chem_provider",
        "cloud_on",
        "co_ratio",
        "diff_esc",
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
        "top_flux",
        "tp_file",
        "tp_file_path",
        "tp_file_sha1",
        "tp_mode",
        "use_condense",
        "use_moldiff",
        "use_photo",
        "use_rayleigh",
        "use_settling",
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


def _check_request_keys(params: dict) -> None:
    """Refuse unknown and removed keys by name, and the removed settling, escape and boundary-flux settings."""
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
    # The keys stay in the canonical payload, pinned off, so no cached
    # spectrum's key changes.
    if params.get("use_settling", False) or params.get("diff_esc") or params.get("top_flux") or params.get("bot_flux"):
        raise ValueError(
            "boundary fluxes, escape and settling were removed from this "
            "tool: leave use_settling False and diff_esc/top_flux/bot_flux "
            "empty."
        )


def _engine_settings(params: dict) -> tuple[str, str, bool, str]:
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
        # silently dropped
        vulcan_only = [k for k in ("use_photo", "use_moldiff", "use_vm_mol", "use_condense") if params.get(k)]
        if params.get("network") not in (None, ""):
            vulcan_only.append("network")
        if str(params.get("kzz_mode", "const")) != "const":
            vulcan_only.append(f"kzz_mode={params.get('kzz_mode')!r}")
        if float(params.get("kzz_x", 1.0)) != 1.0:
            vulcan_only.append("kzz_x")
        if vulcan_only:
            raise ValueError(
                f"{vulcan_only} are VULCAN-JAX settings. The PICASO engine is "
                "equilibrium chemistry, optionally quenched at one constant "
                "Kzz (picaso_quench + kzz_const); drop them for "
                "chem_provider='picaso'."
            )
        if params.get("jac_method", "fd") == "ad" and params.get("fisher_params"):
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
    span = (float(params.get("rt_ptop_bar", RT_PTOP_DEFAULT)) * 1.0e6, p_btm_bar * 1.0e6)
    tp_file, tp_file_sha1, tp_table = "", "", None
    if tp_mode == "file":
        tp_path, tp_file_sha1 = resolve_tp_file(params)
        tp_table = read_tp_table(tp_path, span=span)
        tp_file = str(params.get("tp_file", TP_FILE_SHIPPED))
    if science_mode not in ("transmission", "emission"):
        raise ValueError(
            f"unknown science_mode {science_mode!r}: choose 'transmission' "
            "(transit depth) or 'emission' (secondary-eclipse depth)"
        )
    return _ColumnSettings(science_mode, p_btm_bar, tp_file, tp_file_sha1, tp_table)


def _planet(params: dict) -> tuple[str, dict]:
    """Return the planet and its registry entry, refusing an unknown name."""
    planet = str(params.get("planet", "wasp39b"))
    if planet not in planets.PLANETS and planet != "custom":
        raise ValueError(f"unknown planet {planet!r}")
    return planet, planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS)


def _solver_settings(params: dict, sysd: dict) -> _SolverSettings:
    """Return the grid, the convergence gates and the UV spectrum, refusing out-of-range values."""
    nz = int(params.get("nz", NZ_DEFAULT))
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
    if sflux not in planets.SFLUX_CHOICES:
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
    """Return the canonical key set with every default filled and every value rounded."""
    star_ref = planets.PLANETS.get(r.planet, planets.CUSTOM_DEFAULTS)["star"]
    # before the literal: the custom planet's Tirr default derives from these
    # three (planets.system_teq)
    teff = round(float(params.get("star_teff", star_ref["teff"])), 1)
    rstar = round(float(params.get("rstar_rsun", r.sysd["rstar_rsun"])), 4)
    orbit = round(float(params.get("orbit_au", r.sysd["orbit_au"])), 5)
    return {
        "planet": r.planet,
        "science_mode": r.column.science_mode,
        # Star identity for the eclipse normalization Fp/Fs: part of the MODEL
        # only in emission (zeroed in transmission, where it is noise-side).
        "star_teff": teff,
        "star_logg": round(float(params.get("star_logg", star_ref["log_g"])), 2),
        "star_feh": round(float(params.get("star_feh", star_ref["metallicity"])), 2),
        "nz": r.solver.nz,
        "yconv_cri": round(r.solver.yconv_cri, 6),
        "yconv_min": round(r.solver.yconv_min, 6),
        "rp_rjup": round(float(params.get("rp_rjup", r.sysd["rp_rjup"])), 4),
        "gs_cgs": round(float(params.get("gs_cgs", r.sysd["gs_cgs"])), 1),
        "rstar_rsun": rstar,
        "orbit_au": orbit,
        "sflux": r.solver.sflux,
        "met_x_solar": round(float(params.get("met_x_solar", 10.0)), 4),
        "co_ratio": round(float(params.get("co_ratio", CO_DEFAULT)), 6),
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
        "kzz_x": round(float(params.get("kzz_x", 1.0)), 4),
        "kzz_const": round(float(params.get("kzz_const", 1.0e9)), 1),
        # parametric Kzz profiles: Pfunc (kzz_kmax cm^2/s, kzz_plev bar) and
        # JM16 (kzz_kdeep); param_checks zeroes the unused ones for cache hygiene
        "kzz_kmax": round(float(params.get("kzz_kmax", 1.0e5)), 1),
        "kzz_plev": float(f"{float(params.get('kzz_plev', 0.1)):.6e}"),
        "kzz_kdeep": round(float(params.get("kzz_kdeep", 1.0e5)), 1),
        "tp_mode": r.tp_mode,
        # file-mode identity: source label + content hash, so two tables can
        # never share a cache entry ("" outside file mode)
        "tp_file": r.column.tp_file,
        "tp_file_sha1": r.column.tp_file_sha1,
        # sqrt(2) * T_eq of the SELECTED planet (the DAYSIDE T_eq in
        # emission), via planets.default_tirr so API and GUI cannot diverge
        "Tirr": round(
            float(
                params.get(
                    "Tirr",
                    default_tirr(
                        r.planet,
                        system=dict(star_teff=teff, rstar_rsun=rstar, orbit_au=orbit),
                        science_mode=r.column.science_mode,
                    ),
                )
            ),
            2,
        ),
        # the planet's published interior temperature where the registry
        # carries one (WASP-107 b), via planets.default_tint like Tirr
        "Tint": round(
            float(params.get("Tint", planets.default_tint(planets.PLANETS.get(r.planet, planets.CUSTOM_DEFAULTS)))), 2
        ),
        # 0.01 cm^2/g, Guillot (2010); the sidebar's default must agree
        "log_kappa": round(float(params.get("log_kappa", -2.0)), 3),
        "log_gamma": round(float(params.get("log_gamma", -1.0)), 3),
        # physical VULCAN knobs (all flow through cfg_overrides; the defaults
        # are the W39b cfg values)
        "use_photo": bool(params.get("use_photo", not r.pic)),
        "sl_angle_deg": round(float(params.get("sl_angle_deg", 83.0)), 1),
        "f_diurnal": round(float(params.get("f_diurnal", 1.0)), 3),
        "use_moldiff": bool(params.get("use_moldiff", not r.pic)),
        # Upwind molecular-diffusion advection: PINNED explicitly, never
        # inherited from the engine. False is the validated baseline here.
        "use_vm_mol": bool(params.get("use_vm_mol", False)),
        # Rayleigh is zero-parameter physics, ON by default (off it biases the
        # <1.5 um slope); the power-law cloud deck is OFF by default.
        "use_rayleigh": bool(params.get("use_rayleigh", True)),
        # ExoJAX RT knobs. rt_ptop_bar is the MODEL top and the chemistry grid
        # follows it (engine rule), so no RT layer is clamped above the
        # chemistry. rt_integration picks the ArtTransPure chord scheme.
        "rt_ptop_bar": float(f"{float(params.get('rt_ptop_bar', RT_PTOP_DEFAULT)):.6e}"),
        "rt_integration": str(params.get("rt_integration", "simpson")),
        # The pressure rp_rjup and gs_cgs apply at. A catalogue radius is the
        # transit radius near the terminator photosphere, NOT the RT-grid
        # bottom exojax defaults to; anchoring it there inflates every depth.
        "p_ref_bar": float(f"{float(params.get('p_ref_bar', 1.0e-3)):.6e}"),
        # chemistry + RT column bottom (model_defs.P_BTM_RANGE)
        "p_btm_bar": round(r.column.p_btm_bar, 4),
        "cloud_on": bool(params.get("cloud_on", False)),
        "log_kappa_cloud": round(float(params.get("log_kappa_cloud", -1.0)), 3),
        "alpha_cloud": round(float(params.get("alpha_cloud", 0.0)), 2),
        # Detection-only condensation: param_checks refuses ANY derivative
        "use_condense": bool(params.get("use_condense", False)),
        # removed and refused above; pinned off so the cache key is unchanged
        "use_settling": False,
        "diff_esc": [],
        "top_flux": [],
        "bot_flux": [],
        "extra_mols": sorted({str(m) for m in (params.get("extra_mols") or [])}),
        # Leave-one-out spectrum set: None = every RT molecule (the detect
        # default), [] skips the block. param_checks puts it in fold order.
        "wo_mols": None if params.get("wo_mols") is None else [str(m) for m in params["wo_mols"]],
        "fisher_params": sorted({str(p) for p in (params.get("fisher_params") or [])}),
        # Jacobian method: "fd" (certified FD, the default) or "ad" (one
        # warm-started jvp per chemistry row, photo-on and VULCAN only; docs/forward_model.md).
        "jac_method": str(params.get("jac_method", "fd")),
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
    cp["rfacv"] = round(float(params.get("rfacv", RFACV_DEFAULT[science_mode])), 3) if climate else 0.0
    cp["picaso_quench"] = bool(params.get("picaso_quench", False))
    cp["picaso_stack"] = _pe.PICASO_STACK
    if climate and not RFACV_RANGE[0] <= cp["rfacv"] <= RFACV_RANGE[1]:
        raise ValueError(
            f"rfacv={cp['rfacv']:g} outside {list(RFACV_RANGE)} (climate "
            "heat redistribution: 0.5 planet average, 1.0 dayside)"
        )
    if climate and not TINT_RANGE[0] <= cp["Tint"] <= TINT_RANGE[1]:
        raise ValueError(
            f"Tint={cp['Tint']:g} K outside {list(TINT_RANGE)} K (the climate solve's interior temperature)"
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


def _zero_inert_settings(cp: dict, sysd: dict, science_mode: str, tp_mode: str) -> None:
    """Normalize settings inert for the chosen modes, so one physics maps to one key."""
    climate = tp_mode == "picaso_climate"
    if not cp["use_photo"]:  # photolysis knobs are inert without photo
        cp["sl_angle_deg"] = 0.0
        cp["f_diurnal"] = 1.0
        # The UV spectrum feeds only photolysis: normalize it photo-off so
        # identical physics never caches under two keys.
        cp["sflux"] = str(sysd["sflux"])
        if not climate:  # the semi-major axis heats the climate
            cp["orbit_au"] = round(float(sysd["orbit_au"]), 5)
    if not cp["use_moldiff"]:  # upwind vm_mol is inert without moldiff
        cp["use_vm_mol"] = False  # (engine gates use_vm on both); keep the
        # key from fragmenting the cache
    if not cp["cloud_on"]:  # cloud knobs are inert when the deck is off
        cp["log_kappa_cloud"] = 0.0
        cp["alpha_cloud"] = 0.0
    # Science-mode hygiene + gating
    if science_mode == "emission" or climate:  # the climate solve heats with the star
        if not 3000.0 <= cp["star_teff"] <= 7000.0:
            raise ValueError(
                f"star_teff={cp['star_teff']:g} outside [3000, 7000] K (the "
                "range exercised against the PHOENIX grid for Fp/Fs)"
            )
        if not 3.0 <= cp["star_logg"] <= 5.5:
            raise ValueError(f"star_logg={cp['star_logg']:g} outside [3.0, 5.5]")
        if not -2.5 <= cp["star_feh"] <= 0.5:
            raise ValueError(f"star_feh={cp['star_feh']:g} outside [-2.5, 0.5]")
    if science_mode == "emission":
        # Rayleigh is transmission-only physics (the pure-absorption emission
        # solver must not read scattering as thermal absorption) and the chord
        # scheme exists only in transmission: normalize both for the cache.
        cp["use_rayleigh"] = False
        cp["rt_integration"] = "simpson"
    elif not climate:
        # transmission: the star lives only on the noise side.
        cp["star_teff"] = cp["star_logg"] = cp["star_feh"] = 0.0
    # drop fields inert for the chosen modes so they don't fragment the cache
    if tp_mode != "guillot":
        cp["Tirr"] = cp["log_kappa"] = cp["log_gamma"] = 0.0
        if not climate:
            cp["Tint"] = 0.0


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
    tp_mode, provider, pic, network = _engine_settings(params)
    column = _column_settings(params, tp_mode, pic)
    planet, sysd = _planet(params)
    solver = _solver_settings(params, sysd)
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
    if pic:
        _picaso_fields(cp, params, column.tp_table, column.science_mode, tp_mode, solver.nz)
    check_rt_settings(cp)
    check_molecules(cp, network)
    co_bound, met_range = check_composition(cp, network, pic)
    check_fisher_rows(cp, tp_mode, pic, network, co_bound, met_range)
    check_condensation(cp, network)
    _zero_inert_settings(cp, sysd, column.science_mode, tp_mode)
    check_kzz(cp, tp_mode, column.tp_table, pic)
    # Science-data identity. Every key above describes the MODEL; none of them
    # moves when a k-table or CIA file is corrected under the same filename.
    # It stays out of CHEM_IRRELEVANT_PARAMS on purpose: an extra chemistry
    # miss is cheap and a false hit is not. Unset means omitted, so existing
    # keys are unchanged.
    rev = os.environ.get("JWST_TOOL_DATA_REVISION", "").strip()
    if rev:
        cp["data_revision"] = rev
    return cp
