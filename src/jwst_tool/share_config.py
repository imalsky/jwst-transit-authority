"""Shareable run configurations: build the download, restore the upload.

Method and contract: docs/gui_and_sharing.md, section `share_config`.
"""

from __future__ import annotations

import math
import os
import re
from typing import TYPE_CHECKING, Any

from jwst_tool import fisher, instruments as ins, model_defs, picaso_engine, planets, provenance, run_params, tp_tables

if TYPE_CHECKING:
    from collections.abc import Collection

    from jwst_tool.tool_types import KeyFn

SHARE_FORMAT = 1

# The display name of the every-usable-mode forecast row. A mode combination
# may take neither it nor a mode's label: either would replace that series.
ALL_USABLE = "All usable modes"


def reserved_combo_names() -> set:
    """Return the series names a combination may not take."""
    return {ALL_USABLE} | {m["label"] for m in ins.MODES.values()}


def _log10_range(rng: tuple[float, float]) -> tuple[float, float]:
    return math.log10(rng[0]), math.log10(rng[1])


# Every number-input widget's range, keyed by its widget-key suffix: the
# sidebar widgets and the restore check below both read it, so they cannot
# drift. A range canonical_params also enforces points at its one definition
# (log10 of it for a log-scale widget). An entry is (min, max, default) where the widget is given a fixed
# default, else (min, max): its default is computed, or its key is always
# pre-set in session state and takes none. Per-planet widgets (Target.k) and
# session-global ones (gui_common.K) share it; a name is only ever written under one
# of the two. Values canonical_params range-checks itself (nz, p_btm_bar,
# rfacv, C/O, metallicity) are refused there, before this check runs, and are
# not listed.
WIDGET_RANGES: dict[str, tuple[Any, ...]] = {
    **planets.CUSTOM_FIELD_RANGES,
    "tbase": (0.5, 20.0),
    "tirr": planets.TIRR_RANGE,
    "tint": model_defs.TINT_RANGE,
    "lk": model_defs.LOG_KAPPA_RANGE,
    "lg": model_defs.LOG_GAMMA_RANGE,
    "kzz": _log10_range(model_defs.KZZ_CONST_RANGE),
    "kzkmax": (*_log10_range(model_defs.KZZ_KMAX_RANGE), math.log10(model_defs.KZZ_KMAX_DEFAULT)),
    "kzplev": (*_log10_range(model_defs.KZZ_PLEV_RANGE), math.log10(model_defs.KZZ_PLEV_DEFAULT_BAR)),
    "kzkdeep": (*_log10_range(model_defs.KZZ_KDEEP_RANGE), math.log10(model_defs.KZZ_KDEEP_DEFAULT)),
    "kzzx": (*_log10_range(model_defs.KZZ_X_RANGE), math.log10(model_defs.KZZ_X_DEFAULT)),
    # the default follows the science mode (planets.SL_ANGLE_DEG), so the widget is seeded, not given one
    "sza": model_defs.SL_ANGLE_RANGE,
    "fdiur": (*model_defs.F_DIURNAL_RANGE, model_defs.F_DIURNAL_DEFAULT),
    "yconv": (*model_defs.YCONV_RANGE, model_defs.YCONV_DEFAULT),
    "yconvmin": (*model_defs.YCONV_MIN_RANGE, model_defs.YCONV_MIN_DEFAULT),
    "ck": (*model_defs.LOG_KAPPA_CLOUD_RANGE, model_defs.LOG_KAPPA_CLOUD_DEFAULT),
    "ca": (*model_defs.ALPHA_CLOUD_RANGE, model_defs.ALPHA_CLOUD_DEFAULT),
    "cpt": (-6.0, 1.0, -3.0),
    # the default follows the T-P table (run_params.default_rt_ptop_bar), so the widget is seeded
    "rtptop": model_defs.RT_PTOP_RANGE,
    "pref": (*model_defs.P_REF_RANGE_BAR, model_defs.P_REF_DEFAULT_BAR),
    "tsig": (1.0, 10.0),
    "ntr": (1, 10, 1),
    "sat": (0.5, 0.95, model_defs.SAT_LIMIT_DEFAULT),
    "rbin": (25, 500, 100),
    "seed": (0, 9999, 0),
    "noisescale": (0.5, 3.0, 1.0),
    # per-mode and per-parameter families; the target-uncertainty range
    # depends on the parameter's unit
    "floor": (0.0, 200.0),
    "infl": (1.0, 3.0, 1.0),
    "tgt_K": (5.0, 500.0),
    "tgt": (0.01, 3.0),
}


def _check_widget_ranges(state: dict, key: KeyFn, pk: KeyFn) -> None:
    """Refuse any restored numeric outside its widget's range.

    Streamlit does not reject an out-of-range session-state value: it
    silently discards it and the widget falls back to its default, which
    would run a different model than the file describes.
    """
    checks = {}
    for w, b in WIDGET_RANGES.items():
        checks[key(w)] = checks[pk(w)] = (w, b)
    for m in ins.MODES:
        checks[key(f"floor_{m}")] = (f"floor for {m}", WIDGET_RANGES["floor"])
        checks[key(f"infl_{m}")] = (f"noise multiplier for {m}", WIDGET_RANGES["infl"])
    for p, unit in fisher.PARAM_UNITS.items():
        checks[key(f"tgt_{p}")] = (f"target uncertainty for {p}", WIDGET_RANGES["tgt_K" if unit == "K" else "tgt"])
    bad = []
    for k, (name, (lo, hi, *_)) in checks.items():
        if k in state and not lo <= float(state[k]) <= hi:
            bad.append(f"{name}={float(state[k]):g} outside [{lo:g}, {hi:g}]")
    if bad:
        raise ValueError(
            "configuration value(s) outside the interface's supported "
            "range: " + "; ".join(bad) + ". Out-of-range values are refused "
            "rather than replaced, so the run is always the run the file "
            "describes."
        )


def build_share(
    canon: dict,
    goal: dict,
    observation: dict,
    tp_table_text: str | None = None,
    floor_table: list | None = None,
    cloud_top_bar: float | None = None,
) -> dict:
    """Assemble the downloadable configuration dict."""
    share = {
        "jwst_tool_config": SHARE_FORMAT,
        "canonical_params": dict(canon),
        "goal": dict(goal),
        "observation": dict(observation),
        # The elemental set REQUESTED of the equilibrium seed (number ratios
        # to H): informational, derived from met_x_solar + co_ratio, never
        # read back. Not necessarily what the solved column carries. PICASO's
        # solver is given every atom it carries, He included.
        "elemental_abundances": (
            {f"{a}_H": v for a, v in picaso_engine.composition(canon["met_x_solar"], canon["co_ratio"]).items()}
            if _is_picaso(canon)
            else model_defs.elemental_abundances(canon["met_x_solar"], canon["co_ratio"])
        ),
        # Which software wrote this file is recorded ONCE, inside the
        # provenance block ("software"), never as a second top-level copy.
        # Informational only -- a configuration must load on any tool
        # version, and widget_state never reads it.
        "provenance": provenance.snapshot(observation.get("seed")),
    }
    # The pressure a GRAY cloud was entered as; its presence IS the mode.
    # Not a canonical parameter -- the run is fully determined by cloud_on /
    # log_kappa_cloud / alpha_cloud / fisher_params, which are. Recorded rather
    # than re-derived on load because the canonical block zeroes star_teff in
    # transmission, so the mapping's T_eq cannot be rebuilt from it.
    if cloud_top_bar and canon["cloud_on"]:
        share["cloud_top_bar"] = float(cloud_top_bar)
    if tp_table_text:
        share["tp_table_text"] = str(tp_table_text)
    if floor_table:
        share["floor_table"] = [[float(a), float(b)] for a, b in floor_table]
    return share


def _whole(value: Any, name: str) -> int:
    """Return an integer setting, refusing a fraction rather than truncating it (10.9 transits is not 10)."""
    if isinstance(value, bool) or not float(value).is_integer():
        raise ValueError(f"{name}={value!r} in the configuration is not a whole number: give an integer.")
    return int(float(value))


def _log10(value: float, name: str) -> float:
    v = float(value)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0 in the configuration (got {v})")
    return math.log10(v)


def widget_state(cfg: dict, key: KeyFn) -> dict:
    """Map a shared configuration onto widget session-state keys.

    ``key`` is the app's widget-key namespacer (``gui_common.K``). Returns the
    complete {session_key: value} mapping. Raises ValueError on an invalid
    file or on an entry this interface does not offer (a mode, molecule,
    parameter, floor type or combination); nothing is partially applied
    because nothing is applied here at all.
    """
    if not isinstance(cfg, dict):
        raise ValueError("the configuration file must contain a JSON object")
    # Format gate: the marker this tool writes must be one it can read. A
    # bare canonical dict (no marker) stays a supported path.
    if "jwst_tool_config" in cfg and cfg["jwst_tool_config"] != SHARE_FORMAT:
        raise ValueError(
            f"configuration format {cfg['jwst_tool_config']!r} is not "
            f"supported by this tool version (it reads format "
            f"{SHARE_FORMAT}); re-download the configuration from a "
            "matching tool version"
        )
    if "canonical_params" in cfg:
        cp = cfg["canonical_params"]
        goal = cfg.get("goal") or {}
        obs = cfg.get("observation") or {}
    else:  # bare canonical dict
        cp, goal, obs = cfg, {}, {}
    if not isinstance(cp, dict) or "planet" not in cp:
        raise ValueError(
            "this is not a jwst-tool configuration file (no canonical parameter set with a 'planet' entry)"
        )

    try:
        return _widget_state(cp, goal, obs, cfg, key)
    except KeyError as e:
        raise ValueError(
            f"the configuration file is missing entry {e}; re-download it from a current tool version"
        ) from e


def _resolve_embedded_tp(cp: dict, cfg: dict) -> tuple:
    """Validate the canonical payload and STAGE its T-P upload.

    Returns ``(canonical, path, pending)``: the restore reads the canonical
    dict, never the raw file entries. The staged file is not committed here:
    the caller finishes validating the rest of the restore first, then
    commits or discards, so a later failure leaves nothing behind
    (``_widget_state`` promises that nothing is partially applied).
    """
    restored = None
    pending = None
    validate = dict(cp)
    if cp.get("tp_mode") == "file" and cp.get("tp_file") == model_defs.TP_FILE_UPLOAD:
        text = cfg.get("tp_table_text")
        if text:
            raw = str(text).encode()
            existing = tp_tables.uploads_dir() / f"{tp_tables.content_key(raw)}.txt"
            if existing.exists():
                restored = str(existing)
            else:
                pending = tp_tables.stage_upload(raw)
                restored = str(pending[0])
        else:
            sha = str(cp.get("tp_file_sha1", ""))
            # an archive KEY (sha1[:16]), never a path: the file is untrusted
            candidate = tp_tables.uploads_dir() / f"{sha}.txt" if re.fullmatch(r"[0-9a-f]{16}", sha) else None
            if candidate is not None and candidate.exists():
                restored = str(candidate)
            else:
                raise ValueError(
                    "the configuration uses an uploaded T-P table whose "
                    "content is not embedded in the file and is not in the "
                    "local uploads archive; upload the table again after loading"
                )
        validate["tp_file_path"] = restored
    try:
        canonical = run_params.canonical_params(validate)
    except Exception:
        if pending is not None:
            pending[0].unlink(missing_ok=True)
        raise
    return canonical, restored, pending


def _restore_goal(state: dict, cp: dict, goal: dict, key: KeyFn, tp_mode: str, cloud_mode: str) -> None:
    pic = _is_picaso(cp)
    network = str(cp["network"])
    suffix = goal_suffix(pic, tp_mode, cp["cloud_on"], cloud_mode)
    avail = free_parameters(pic, tp_mode, cp["cloud_on"], cloud_mode)
    fisher = [str(p) for p in (goal.get("fisher_params") or cp["fisher_params"])]
    selected_goal = str(goal.get("goal") or "")
    if not selected_goal:
        return
    if selected_goal not in ("detect", "constrain"):
        raise ValueError(f"unknown goal {selected_goal!r}: the interface offers 'detect' and 'constrain'")
    _offered("free parameter", fisher, avail)
    state[key("goal")] = selected_goal
    if goal.get("target_sig") is not None:
        state[key("tsig")] = float(goal["target_sig"])
    method = str(goal.get("jac_method", cp["jac_method"]))
    # PICASO is FD-only on its own widget: writing "fd" here would overwrite
    # the stored VULCAN choice
    if method in ("fd", "ad") and not pic:
        state[key("jacm")] = method
    if selected_goal == "detect":
        extras = state[key(xmols_key(pic, network))]
        target = goal.get("target_mol")
        if target is not None:
            _offered(
                "detection target", [target], model_defs.active_molecules({"network": network, "extra_mols": extras})
            )
            state[key(target_mol_key(pic, network, extras))] = target
        state[key("dofish")] = bool(goal.get("do_fisher", bool(fisher)))
        if fisher:
            state[key(f"fp_{suffix}")] = fisher
        return
    goal_param = goal.get("goal_param")
    if goal_param is not None:
        _offered("goal parameter", [goal_param], avail)
        state[key(f"gp_{suffix}")] = goal_param
        if goal.get("target_prec") is not None:
            state[key(f"tgt_{goal_param}")] = float(goal["target_prec"])
    marginalize = bool(goal.get("marginalize", True))
    state[key("marg")] = marginalize
    if marginalize and fisher:
        state[key(f"fx_{suffix}")] = [p for p in fisher if p != goal_param]


def _offered(what: str, values: list, options: Collection[str]) -> None:
    """Refuse a restored selection the interface does not offer.

    Dropping it would run a different setup than the file describes.
    """
    bad = [v for v in values if v not in options]
    if bad:
        raise ValueError(f"{what} {bad} not offered by this interface (choose from {list(options)})")


def _numeric(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in the configuration (got {value!r})")
    return float(value)


def _restore_combos(obs: dict) -> list[dict]:
    """Return the saved mode combinations; a malformed one is refused, never dropped."""
    combos: list[dict] = []
    for candidate in obs.get("combos") or []:
        name = str(candidate.get("name", "")).strip() if isinstance(candidate, dict) else ""
        modes = list(candidate.get("modes") or []) if name else []
        if not name or not modes:
            raise ValueError(f"mode combination {candidate!r} needs a name and at least one mode")
        if name in reserved_combo_names() or any(item["name"] == name for item in combos):
            raise ValueError(f"mode combination name {name!r} is already taken by a combination or a mode")
        _offered(f"mode combination {name!r}: mode", modes, ins.MODES)
        combos.append({"name": name, "modes": modes})
    return combos


def _restore_observation(state: dict, obs: dict, cfg: dict, key: KeyFn, pk: KeyFn) -> None:
    if not obs:
        return
    for source, widget, cast in (
        ("ks_mag", "ks", float),
        ("t14", "t14", float),
        ("t_base", "tbase", float),
        ("star_teff", "teff", float),
        ("star_logg", "logg", float),
        ("star_feh", "feh", float),
    ):
        if obs.get(source) is not None:
            state[pk(widget)] = cast(obs[source])
    modes = list(obs.get("modes") or [])
    _offered("instrument mode", modes, ins.MODES)
    if modes:
        state[key("modes")] = modes
    for source, widget, cast in (  # type: ignore[assignment]  # float then int: each is a converter
        ("n_transits", "ntr", lambda v: _whole(v, "n_transits")),
        ("sat_limit", "sat", float),
        ("r_bin", "rbin", lambda v: _whole(v, "r_bin")),
        ("seed", "seed", lambda v: _whole(v, "seed")),
        ("show_noise", "shownoise", bool),
    ):
        if obs.get(source) is not None:
            state[key(widget)] = cast(obs[source])
    floor_mode = obs.get("floor_mode")
    if floor_mode is not None:
        _offered("floor type", [floor_mode], ("constant", "none", "file"))
        state[key("floormode")] = floor_mode
    if floor_mode == "constant":
        floors = obs.get("floors") or {}
        _offered("floor mode key", list(floors), ins.MODES)
        for mode, value in floors.items():
            state[key(f"floor_{mode}")] = _numeric(f"floor for {mode}", value)
    elif floor_mode == "file":
        table = cfg.get("floor_table")
        if table:
            state["restored_floor_table"] = [[float(a), float(b)] for a, b in table]
    infl = obs.get("noise_infl") or {}
    _offered("noise multiplier mode key", list(infl), ins.MODES)
    for mode, value in infl.items():
        state[key(f"infl_{mode}")] = _numeric(f"noise multiplier for {mode}", value)
    if obs.get("noise_scale") is not None:
        state[key("noisescale")] = _numeric("noise_scale", obs["noise_scale"])
    combos = _restore_combos(obs)
    if combos:
        state[key("combos")] = combos


def _restore_system_profile(
    cp: dict, key: KeyFn, pk: KeyFn, planet: str, tp_mode: str, restored_tp: str | None
) -> dict:
    state = {
        key("planet"): planet,
        key("scimode"): str(cp["science_mode"]),
        pk("rp"): float(cp["rp_rjup"]),
        pk("g"): float(cp["gs_cgs"]) / planets.CM_PER_M,
        pk("rstar"): float(cp["rstar_rsun"]),
        pk("a"): float(cp["orbit_au"]),
        # always written: a VULCAN file loaded under PICASO switches back
        key("provider"): "picaso" if _is_picaso(cp) else "vulcan",
        pk("tp_picaso" if _is_picaso(cp) else "tp"): tp_mode,
    }
    # The Pandeia star. canonical_params zeroes star_teff/logg/feh in
    # transmission mode (the star lives only on the noise side there), so
    # the canonical block carries the real star only in emission mode. The
    # observation block records it for both modes; _restore_observation
    # overwrites these keys from it when present. The zero sentinel must
    # never reach the widgets: Teff=0 is outside every widget range.
    if float(cp["star_teff"]) != 0.0:
        state[pk("teff")] = float(cp["star_teff"])
        state[pk("logg")] = float(cp["star_logg"])
        state[pk("feh")] = float(cp["star_feh"])
    # the UV spectrum feeds only VULCAN photolysis; a PICASO payload carries
    # the registry default, which must not replace the VULCAN choice
    if not _is_picaso(cp) and str(cp["sflux"]) in planets.SFLUX_CHOICES:
        state[pk("sflux")] = str(cp["sflux"])
    if tp_mode == "picaso_climate":
        state[pk("tint")] = float(cp["Tint"])
        state[key(f"rfacv_picaso_{cp['science_mode']}")] = float(cp["rfacv"])
    elif tp_mode == "guillot":
        state.update(
            {
                pk("tirr"): float(cp["Tirr"]),
                pk("tirr_auto"): None,
                pk("tint"): float(cp["Tint"]),
                pk("lk"): float(cp["log_kappa"]),
                pk("lg"): float(cp["log_gamma"]),
            }
        )
    elif tp_mode == "file":
        state[pk("tpsrc")] = str(cp["tp_file"])
        if restored_tp:
            state["restored_tp_path"] = restored_tp
    state[key("met")] = float(cp["met_x_solar"])
    state[key("co")] = float(cp["co_ratio"])
    return state


def _reject_removed_physics(cp: dict) -> None:
    if cp["use_condense"]:
        raise ValueError(
            "this configuration enables condensation (use_condense), "
            "atmospheric physics the interface no longer offers. It remains "
            "available through the programmatic interface "
            "(jwst_tool.run_params.canonical_params); loading it in the GUI "
            "would change the atmosphere."
        )


def _is_picaso(cp: dict) -> bool:
    return cp["chem_provider"] == "picaso"


# Widget-key and menu helpers shared with the sidebar, so the keys the restore
# writes are the keys the widgets read.
def network_suffix(network: str) -> str:
    """Widget-key suffix per kinetics network.

    The sncho keys carry no suffix (shipped key contract, from when sncho was the default), ncho and sncho2025
    widgets get their own keys (same pattern as the provider-suffixed molecule keys).
    """
    return "" if network == "sncho" else f"_{network}"


def xmols_key(pic: bool, network: str) -> str:
    """Return the extra-molecule widget key: PICASO's own, else per network."""
    return "xmols_picaso" if pic else f"xmols_vulcan{network_suffix(network)}"


def target_mol_key(pic: bool, network: str, extras: list) -> str:
    """Return the detection-target widget key; its options follow the extra molecules, so the key does too."""
    return ("mol_picaso_" if pic else f"mol_vulcan{network_suffix(network)}_") + "_".join(sorted(extras))


def goal_suffix(pic: bool, tp_mode: str, cloud_on: bool, cloud_mode: str) -> str:
    """Key suffix of the free-parameter widgets.

    Each engine, T-P mode and cloud setting (off 0, haze 1, gray 2) keeps its own selections. ONE key axis with
    three cloud values, not a second boolean gate; haze is the default so stored keys keep meaning.
    """
    cloud = 0 if not cloud_on else (2 if cloud_mode == "gray" else 1)
    return f"{'picaso' if pic else 'vulcan'}_{tp_mode}_{cloud}"


def free_parameters(pic: bool, tp_mode: str, cloud_on: bool, cloud_mode: str) -> list:
    """Return the free-parameter menu.

    PICASO offers no lnKzz row (its quench levels make the abundances a staircase in Kzz; canonical_params refuses
    the row); a gray cloud has no spectral-slope row.
    """
    avail = [p for p in model_defs.CHEM_PARAM_NAMES if not (pic and p == "lnKzz")] + list(
        model_defs.TP_PARAM_NAMES[tp_mode]
    )
    if cloud_on:
        avail += [p for p in model_defs.CLOUD_FISHER_PARAMS if cloud_mode != "gray" or p != "alpha_cloud"]
    return avail


def _restore_picaso_physics(state: dict, cp: dict, key: KeyFn, pk: KeyFn) -> None:
    quench = bool(cp["picaso_quench"])
    state[key("quench_picaso")] = quench
    if quench:
        state[pk("kzz")] = _log10(cp["kzz_const"], "kzz_const")


def _restore_vulcan_physics(state: dict, cp: dict, key: KeyFn, pk: KeyFn) -> None:
    state[key("network")] = str(cp["network"])
    mode = str(cp["kzz_mode"])
    state[pk("kzzmode")] = mode
    if mode == "const":
        state[pk("kzz")] = _log10(cp["kzz_const"], "kzz_const")
    elif mode == "Pfunc":
        state[pk("kzkmax")] = _log10(cp["kzz_kmax"], "kzz_kmax")
        state[pk("kzplev")] = _log10(cp["kzz_plev"], "kzz_plev")
    elif mode == "JM16":
        state[pk("kzkdeep")] = _log10(cp["kzz_kdeep"], "kzz_kdeep")
    state[pk("kzzx")] = _log10(cp["kzz_x"], "kzz_x")
    state.update(
        {
            key("photo"): bool(cp["use_photo"]),
            key("sza"): float(cp["sl_angle_deg"]),
            key("fdiur"): float(cp["f_diurnal"]),
            key("moldiff"): bool(cp["use_moldiff"]),
            key("vmmol"): bool(cp["use_vm_mol"]),
            key("yconv"): float(cp["yconv_cri"]),
            key("yconvmin"): float(cp["yconv_min"]),
        }
    )
    _reject_removed_physics(cp)


def _restore_rt_state(state: dict, cp: dict, key: KeyFn, cloud_top_bar: float | None) -> None:
    state[key("rayl")] = bool(cp["use_rayleigh"])
    state[key("cloud")] = bool(cp["cloud_on"])
    if cp["cloud_on"]:
        gray = cloud_top_bar is not None
        state[key("cmode")] = "gray" if gray else "haze"
        if gray:
            state[key("cpt")] = _log10(cloud_top_bar, "cloud_top_bar")  # type: ignore[arg-type]  # gray means it is set
        else:
            state[key("ck")] = float(cp["log_kappa_cloud"])
            state[key("ca")] = float(cp["alpha_cloud"])
    state[key(xmols_key(_is_picaso(cp), cp["network"]))] = list(cp["extra_mols"])
    state.update(
        {
            key("nz"): int(cp["nz"]),
            # a loaded top is explicit, never re-seeded from the table
            key("rtptop"): float(cp["rt_ptop_bar"]),
            key("rtptop_auto"): None,
            key("rtint"): str(cp["rt_integration"]),
            key("pref"): float(cp["p_ref_bar"]),
            # per-geometry widget key (the sidebar's key contract); a loaded bottom is explicit, never re-seeded
            key(f"pbtm_{cp['science_mode']}"): float(cp["p_btm_bar"]),
            key(f"pbtm_{cp['science_mode']}_auto"): None,
        }
    )


def _widget_state(cp: dict, goal: dict, obs: dict, cfg: dict, key: KeyFn) -> dict:
    # canonical_params refuses the removed opacity keys (mie_condensate,
    # opacity_mode) with their removal message; everything below reads its
    # output, so the restore is the run canonical_params would compute.
    cp, restored_tp, pending_tp = _resolve_embedded_tp(cp, cfg)
    planet = str(cp["planet"])
    tp_mode = str(cp["tp_mode"])

    def pk(name: str) -> str:  # per-planet widget keys
        return key(f"{planet}_{name}")

    # A gray cloud records the pressure it was entered as; its absence is the
    # power-law haze.
    cloud_top_bar = cfg.get("cloud_top_bar")
    cloud_mode = "haze" if cloud_top_bar is None else "gray"
    try:
        if cloud_mode == "gray" and cp["science_mode"] == "emission":
            raise ValueError(
                "this configuration pairs a gray cloud deck with an emission "
                "run. The cloud-top pressure is mapped to an opacity through "
                "the transmission chord, which the vertical emission "
                "integration does not share, so the interface offers the "
                "gray deck in transmission only. Restoring it as a power-law "
                "haze would run a different cloud than the file describes; "
                "re-create the run with the haze parameterization."
            )
        state = _restore_system_profile(cp, key, pk, planet, tp_mode, restored_tp)

        if _is_picaso(cp):
            _restore_picaso_physics(state, cp, key, pk)
        else:
            _restore_vulcan_physics(state, cp, key, pk)

        _restore_rt_state(state, cp, key, cloud_top_bar)
        _restore_goal(state, cp, goal, key, tp_mode, cloud_mode)
        _restore_observation(state, obs, cfg, key, pk)
        _check_widget_ranges(state, key, pk)
    except Exception:
        if pending_tp is not None:
            pending_tp[0].unlink(missing_ok=True)
        raise
    if pending_tp is not None:
        os.replace(pending_tp[0], pending_tp[1])
        state["restored_tp_path"] = str(pending_tp[1])
    return state
