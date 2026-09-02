"""Shareable run configurations: build the download, restore the upload.

The GUI's "Download configuration (JSON)" button writes the dict `build_share`
assembles: the canonical model parameters (the same dictionary every result
stores as provenance) plus the science-goal and observation settings, and --
when the run uses uploaded tables -- the table content itself. That is every
input the GUI takes -- condensation, settling, escape and the boundary-condition
fluxes are API-only, pinned off in the GUI and REFUSED on load -- so the same
tool version reproduces a GUI run from the file alone. Exact code revisions,
science-data checksums, cache schemas,
Pandeia/PandExo identity, and the random seed are recorded in ``provenance``.
They remain informational on load so old configurations stay portable, but a
collaborator can compare them before claiming an exact reproduction.

`widget_state` is the inverse: it maps such a file (or a bare canonical-params
dict from an older download) onto Streamlit session-state widget keys. It
validates through `forward.canonical_params` first, then checks every numeric
value against its widget's range, and raises ValueError on anything invalid --
the caller applies either the whole mapping or nothing. The range check is
load-bearing: Streamlit does not reject an out-of-range session-state value,
it silently discards it and the widget falls back to its default, so an
unchecked restore would run a different model than the file describes.

This lives outside app.py so the mapping is importable and unit-testable
without running Streamlit.
"""
from __future__ import annotations

import hashlib
import math
import os

from jwst_tool import forward, planets, provenance

SHARE_FORMAT = 1

# Widget ranges mirrored from app.py, cross-checked by
# tests/unit/test_share_config.py so they cannot drift. Session-global keys;
# forward.* constants are shared with the widgets themselves.
_GLOBAL_BOUNDS = {
    "met": (0.1, 100.0), "co": (0.10, 0.95),
    "sza": (0.0, 89.0), "fdiur": (0.1, 1.0),
    "yconv": (1.0e-4, 1.0e-2),
    "ck": (-4.0, 2.0), "ca": (0.0, 4.0),
    "nz": forward.NZ_RANGE,
    "rtptop": (1.0e-9, 1.0e-6),
    "pref": (1.0e-6, 7.0),
    "tsig": (1.0, 10.0),
    "ntr": (1, 10), "sat": (0.5, 0.95), "rbin": (25, 500),
    "seed": (0, 9999), "noisescale": (0.5, 3.0),
}
# Per-planet keys (app.py's _k). The system/star fields share
# planets.CUSTOM_FIELD_RANGES with the widgets; the rest are widget literals.
_PLANET_BOUNDS = {
    **planets.CUSTOM_FIELD_RANGES,
    "tbase": (0.5, 10.0),
    "tirr": (800.0, 2500.0), "tint": (50.0, 500.0),
    "lk": (-4.0, 0.0), "lg": (-2.0, 0.3),
    "kzz": (6.0, 12.0), "kzkmax": (4.0, 11.0), "kzplev": (-5.0, 2.0),
    "kzkdeep": (4.0, 11.0), "kzzx": (-1.0, 1.0),
}
_FLOOR_BOUNDS = (0.0, 200.0)
_INFL_BOUNDS = (1.0, 3.0)
_TGT_BOUNDS_K = (5.0, 500.0)      # target-uncertainty widget, Kelvin params
_TGT_BOUNDS = (0.01, 3.0)         # target-uncertainty widget, dex/ratio params


def _check_widget_ranges(state: dict, key, pk, science_mode: str) -> None:
    """Refuse any restored numeric outside its widget's range.

    Streamlit does not reject an out-of-range session-state value: it
    silently discards it and the widget falls back to its default, which
    would run a different model than the file describes."""
    from jwst_tool import instruments as ins
    checks = {key(w): (w, b) for w, b in _GLOBAL_BOUNDS.items()}
    checks.update({pk(w): (w, b) for w, b in _PLANET_BOUNDS.items()})
    for m in ins.MODES:
        checks[key(f"floor_{m}")] = (f"floor for {m}", _FLOOR_BOUNDS)
        checks[key(f"infl_{m}")] = (f"noise multiplier for {m}", _INFL_BOUNDS)
    checks[key(f"pbtm_{science_mode}")] = ("p_btm_bar", forward.P_BTM_RANGE)
    for p, unit in forward.PARAM_UNITS.items():
        checks[key(f"tgt_{p}")] = (
            f"target uncertainty for {p}",
            _TGT_BOUNDS_K if unit == "K" else _TGT_BOUNDS)
    bad = []
    for k, (name, (lo, hi)) in checks.items():
        if k in state and not lo <= float(state[k]) <= hi:
            bad.append(f"{name}={float(state[k]):g} outside [{lo:g}, {hi:g}]")
    if bad:
        raise ValueError(
            "configuration value(s) outside the interface's supported "
            "range: " + "; ".join(bad) + ". Out-of-range values are refused "
            "rather than replaced, so the run is always the run the file "
            "describes.")


def build_share(canon: dict, goal: dict, observation: dict,
                tp_table_text: str | None = None,
                floor_table: list | None = None) -> dict:
    """Assemble the downloadable configuration dict."""
    share = {
        "jwst_tool_config": SHARE_FORMAT,
        "canonical_params": dict(canon),
        "goal": dict(goal),
        "observation": dict(observation),
        # The elemental set the run actually solved (number ratios to H):
        # informational, derived from met_x_solar + co_ratio, never read back.
        "elemental_abundances": forward.elemental_abundances(
            canon["met_x_solar"], canon["co_ratio"]),
        # Which software wrote this file is recorded ONCE, inside the
        # provenance block ("software"), never as a second top-level copy.
        # Informational only -- a configuration must load on any tool
        # version, and widget_state never reads it.
        "provenance": provenance.snapshot(observation.get("seed")),
    }
    if tp_table_text:
        share["tp_table_text"] = str(tp_table_text)
    if floor_table:
        share["floor_table"] = [[float(a), float(b)] for a, b in floor_table]
    return share


def _log10(value, name: str) -> float:
    v = float(value)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0 in the configuration (got {v})")
    return math.log10(v)


def widget_state(cfg: dict, key) -> tuple[dict, list[str]]:
    """Map a shared configuration onto widget session-state keys.

    ``key`` is the app's widget-key namespacer (``app.K``). Returns
    ``(state, notes)``: the complete {session_key: value} mapping and
    human-readable notes about anything that could not be restored.
    Raises ValueError on an invalid file; nothing is partially applied
    because nothing is applied here at all.
    """
    if not isinstance(cfg, dict):
        raise ValueError("the configuration file must contain a JSON object")
    # Format gate: the marker this tool writes must be one it can read. A
    # bare canonical dict (older download, no marker) stays a supported path.
    if "jwst_tool_config" in cfg and cfg["jwst_tool_config"] != SHARE_FORMAT:
        raise ValueError(
            f"configuration format {cfg['jwst_tool_config']!r} is not "
            f"supported by this tool version (it reads format "
            f"{SHARE_FORMAT}); re-download the configuration from a "
            "matching tool version")
    if "canonical_params" in cfg:
        cp = cfg["canonical_params"]
        goal = cfg.get("goal") or {}
        obs = cfg.get("observation") or {}
    else:                       # bare canonical dict from an older download
        cp, goal, obs = cfg, {}, {}
    if not isinstance(cp, dict) or "planet" not in cp:
        raise ValueError(
            "this is not a jwst-tool configuration file (no canonical "
            "parameter set with a 'planet' entry)")

    notes: list[str] = []
    try:
        return _widget_state(cp, goal, obs, cfg, key, notes)
    except KeyError as e:
        raise ValueError(
            f"the configuration file is missing entry {e}; re-download it "
            "from a current tool version") from e


def _resolve_embedded_tp(cp: dict, cfg: dict, tp_mode: str):
    """Validate the canonical payload and STAGE its T-P upload.

    Returns ``(path, pending)``. The staged file is not committed here: the
    caller finishes validating the rest of the restore first, then commits or
    discards, so a later failure leaves nothing behind (``_widget_state``
    promises that nothing is partially applied).
    """
    restored = None
    pending = None
    validate = dict(cp)
    if tp_mode == "file" and str(cp.get("tp_file", "")) == forward.TP_FILE_UPLOAD:
        text = cfg.get("tp_table_text")
        if text:
            raw = str(text).encode()
            sha = hashlib.sha1(raw).hexdigest()[:16]
            destination = forward._uploads_dir() / f"{sha}.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                restored = str(destination)
            else:
                temporary = destination.with_name(
                    f".{sha}.{os.getpid()}.tmp")
                temporary.write_bytes(raw)
                pending = (temporary, destination)
                restored = str(temporary)
        else:
            sha = str(cp.get("tp_file_sha1", ""))
            candidate = forward._uploads_dir() / f"{sha}.txt" if sha else None
            if candidate is not None and candidate.exists():
                restored = str(candidate)
            else:
                raise ValueError(
                    "the configuration uses an uploaded T-P table whose "
                    "content is not embedded in the file and is not in the "
                    "local uploads archive; upload the table again after loading")
        validate["tp_file_path"] = restored
    try:
        forward.canonical_params(validate)
    except Exception:
        if pending is not None:
            pending[0].unlink(missing_ok=True)
        raise
    return restored, pending


def _restore_goal(state: dict, cp: dict, goal: dict, key,
                  tp_mode: str, notes: list[str]) -> None:
    cloud_i = int(bool(cp["cloud_on"]))
    suffix = f"vulcan_{tp_mode}_{cloud_i}"
    avail = list(forward.CHEM_PARAM_NAMES) + list(forward.TP_PARAM_NAMES[tp_mode])
    if cp["cloud_on"]:
        avail += list(forward.CLOUD_FISHER_PARAMS)
    requested = [str(p) for p in (goal.get("fisher_params")
                                  or cp.get("fisher_params") or [])]
    fisher = [p for p in requested if p in avail]
    dropped_fp = [p for p in requested if p not in avail]
    if dropped_fp:
        notes.append(f"free parameter(s) {dropped_fp} are not available "
                     "under the restored settings and were dropped")
    selected_goal = str(goal.get("goal", "") or "")
    if selected_goal not in ("detect", "constrain"):
        if goal:
            notes.append(f"unknown science goal {selected_goal!r}; the goal "
                         "settings keep their current values")
        return
    state[key("goal")] = selected_goal
    if goal.get("target_sig") is not None:
        state[key("tsig")] = float(goal["target_sig"])
    method = str(goal.get("jac_method", cp.get("jac_method", "fd")))
    if method in ("fd", "ad"):
        state[key("jacm")] = method
    else:
        notes.append(f"unknown Jacobian method {method!r} was not restored")
    if selected_goal == "detect":
        net_sfx = _network_suffix(cp)
        extras = state[key(f"xmols_vulcan{net_sfx}")]
        mol_opts = forward.active_molecules(
            {"network": str(cp.get("network", "sncho")),
             "extra_mols": extras})
        target = goal.get("target_mol")
        if target in mol_opts:
            state[key(f"mol_vulcan{net_sfx}_"
                      + "_".join(sorted(extras)))] = target
        elif target:
            notes.append(f"detection target {target} is not in the restored "
                         "molecule set and was not restored")
        state[key("dofish")] = bool(goal.get("do_fisher", bool(fisher)))
        if fisher:
            state[key(f"fp_{suffix}")] = fisher
        return
    goal_param = goal.get("goal_param")
    if goal_param in avail:
        state[key(f"gp_{suffix}")] = goal_param
        if goal.get("target_prec") is not None:
            state[key(f"tgt_{goal_param}")] = float(goal["target_prec"])
    elif goal_param:
        notes.append(f"constraint parameter {goal_param} is not available "
                     "under the restored settings")
    marginalize = bool(goal.get("marginalize", True))
    state[key("marg")] = marginalize
    if marginalize and fisher:
        state[key(f"fx_{suffix}")] = [p for p in fisher if p != goal_param]


def _restore_combos(obs: dict, valid_modes: dict,
                    notes: list[str]) -> list[dict]:
    combos = []
    for candidate in obs.get("combos") or []:
        if not isinstance(candidate, dict) \
                or not str(candidate.get("name", "")).strip():
            notes.append("a saved mode combination without a name was not restored")
            continue
        name = str(candidate["name"]).strip()
        modes = [m for m in (candidate.get("modes") or []) if m in valid_modes]
        dropped = [m for m in (candidate.get("modes") or []) if m not in valid_modes]
        if dropped:
            notes.append(f"combination {name!r}: unknown instrument mode(s) "
                         f"{dropped} were dropped")
        if not modes:
            notes.append(f"combination {name!r} has no valid instrument "
                         "modes and was not restored")
        elif any(item["name"] == name for item in combos):
            notes.append(f"duplicate combination name {name!r}: only the "
                         "first was restored")
        else:
            combos.append({"name": name, "modes": modes})
    return combos


def _restore_observation(state: dict, obs: dict, cfg: dict, key, pk,
                         notes: list[str]) -> None:
    if not obs:
        return
    from jwst_tool import instruments as ins
    for source, widget, cast in (
            ("ks_mag", "ks", float), ("t14", "t14", float),
            ("t_base", "tbase", float), ("star_teff", "teff", float),
            ("star_logg", "logg", float), ("star_feh", "feh", float)):
        if obs.get(source) is not None:
            state[pk(widget)] = cast(obs[source])
    modes = [m for m in (obs.get("modes") or []) if m in ins.MODES]
    dropped = [m for m in (obs.get("modes") or []) if m not in ins.MODES]
    if dropped:
        notes.append(f"unknown instrument mode(s) {dropped} were not restored")
    if modes:
        state[key("modes")] = modes
    for source, widget, cast in (
            ("n_transits", "ntr", int), ("sat_limit", "sat", float),
            ("r_bin", "rbin", int), ("seed", "seed", int),
            ("show_noise", "shownoise", bool)):
        if obs.get(source) is not None:
            state[key(widget)] = cast(obs[source])
    floor_mode = obs.get("floor_mode")
    if floor_mode not in ("constant", "none", "file"):
        if floor_mode is not None:
            notes.append(f"unknown noise-floor type {floor_mode!r}; the "
                         "floor settings keep their current values")
    else:
        state[key("floormode")] = floor_mode
        if floor_mode == "constant":
            for mode, value in (obs.get("floors") or {}).items():
                if mode not in ins.MODES:
                    continue
                if isinstance(value, (int, float)):
                    state[key(f"floor_{mode}")] = float(value)
                else:
                    notes.append(f"the constant noise floor for {mode} is "
                                 "not numeric and was not restored")
        elif floor_mode == "file":
            table = cfg.get("floor_table")
            if table:
                state["restored_floor_table"] = [
                    [float(a), float(b)] for a, b in table]
            else:
                notes.append("the wavelength-table noise floor is not embedded "
                             "in this file; upload it again")
    for mode, value in (obs.get("noise_infl") or {}).items():
        if mode not in ins.MODES:
            continue
        if isinstance(value, (int, float)):
            state[key(f"infl_{mode}")] = float(value)
        else:
            notes.append(f"the noise multiplier for {mode} is not numeric "
                         "and was not restored")
    scale = obs.get("noise_scale")
    if isinstance(scale, (int, float)):
        # range-checked by _check_widget_ranges, never silently dropped
        state[key("noisescale")] = float(scale)
    elif scale is not None:
        notes.append("the global noise multiplier is not numeric and was "
                     "not restored")
    if obs.get("scenario") not in (None, "random"):
        notes.append(
            f"this configuration selected the removed experimental noise "
            f"scenario {obs['scenario']!r}; the standard (diagonal) noise "
            "model is used instead")
    combos = _restore_combos(obs, ins.MODES, notes)
    if combos:
        state[key("combos")] = combos


def _restore_system_profile(cp: dict, key, pk, planet: str,
                            tp_mode: str, restored_tp: str | None,
                            notes: list[str]) -> dict:
    state = {
        key("planet"): planet,
        key("scimode"): str(cp.get("science_mode", "transmission")),
        pk("rp"): float(cp["rp_rjup"]),
        pk("g"): float(cp["gs_cgs"]) / 100.0,
        pk("rstar"): float(cp["rstar_rsun"]),
        pk("a"): float(cp["orbit_au"]),
        pk("tp"): tp_mode,
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
    if str(cp.get("sflux", "")) in planets.SFLUX_CHOICES:
        state[pk("sflux")] = str(cp["sflux"])
    else:
        notes.append("the configuration names no stellar UV spectrum; the "
                     "menu keeps its current selection")
    if tp_mode == "guillot":
        state.update({
            pk("tirr"): float(cp["Tirr"]), pk("tirr_auto"): None,
            pk("tint"): float(cp["Tint"]), pk("lk"): float(cp["log_kappa"]),
            pk("lg"): float(cp["log_gamma"]),
        })
    elif tp_mode == "file":
        state[pk("tpsrc")] = str(cp.get("tp_file", forward.TP_FILE_SHIPPED))
        if restored_tp:
            state["restored_tp_path"] = restored_tp
    state[key("met")] = float(cp["met_x_solar"])
    state[key("co")] = float(cp["co_ratio"])
    return state


def _reject_removed_physics(cp: dict) -> None:
    if bool(cp.get("use_condense", False)):
        raise ValueError(
            "this configuration enables condensation (use_condense), "
            "atmospheric physics the interface no longer offers. It remains "
            "available through the programmatic interface "
            "(jwst_tool.forward.canonical_params); loading it in the GUI "
            "would change the atmosphere.")


def _network_suffix(cp: dict) -> str:
    """Widget-key suffix for the non-default kinetics network: the sncho keys
    carry no suffix (shipped key contract), ncho widgets get their own keys
    (same pattern as the provider-suffixed molecule keys)."""
    net = str(cp.get("network", "sncho"))
    return "" if net == "sncho" else f"_{net}"


def _restore_vulcan_physics(state: dict, cp: dict, key, pk) -> None:
    state[key("network")] = str(cp.get("network", "sncho"))
    mode = str(cp.get("kzz_mode", "const"))
    state[pk("kzzmode")] = mode
    if mode == "const":
        state[pk("kzz")] = _log10(cp["kzz_const"], "kzz_const")
    elif mode == "Pfunc":
        state[pk("kzkmax")] = _log10(cp["kzz_kmax"], "kzz_kmax")
        state[pk("kzplev")] = _log10(cp["kzz_plev"], "kzz_plev")
    elif mode == "JM16":
        state[pk("kzkdeep")] = _log10(cp["kzz_kdeep"], "kzz_kdeep")
    if mode != "const":
        state[pk("kzzx")] = _log10(cp.get("kzz_x", 1.0), "kzz_x")
    state.update({
        key("photo"): bool(cp["use_photo"]),
        key("sza"): float(cp["sl_angle_deg"]),
        key("fdiur"): float(cp["f_diurnal"]),
        key("moldiff"): bool(cp["use_moldiff"]),
        key("vmmol"): bool(cp["use_vm_mol"]),
        key("yconv"): float(cp["yconv_cri"]),
    })
    _reject_removed_physics(cp)


def _restore_rt_state(state: dict, cp: dict, key) -> None:
    state[key("rayl")] = bool(cp["use_rayleigh"])
    state[key("cloud")] = bool(cp["cloud_on"])
    if cp["cloud_on"]:
        state[key("ck")] = float(cp["log_kappa_cloud"])
        state[key("ca")] = float(cp["alpha_cloud"])
    extras_all = list(forward.EXTRA_MOLECULES)
    state[key(f"xmols_vulcan{_network_suffix(cp)}")] = [
        molecule for molecule in (cp.get("extra_mols") or [])
        if molecule in extras_all]
    state.update({
        key("nz"): int(cp["nz"]),
        key("rtptop"): float(cp["rt_ptop_bar"]),
        key("rtint"): str(cp["rt_integration"]),
        # .get with the default so a configuration saved before the key
        # existed still loads: restoring it onto the current default is the
        # correct physics, and the stale-key banner already says the run must
        # be redone.
        key("pref"): float(cp.get("p_ref_bar", 1.0e-3)),
        # same .get treatment; per-geometry widget key (app.py's key contract)
        key(f"pbtm_{cp.get('science_mode', 'transmission')}"):
            float(cp.get("p_btm_bar", forward.P_BTM_FILE_BAR)),
    })


def _widget_state(cp: dict, goal: dict, obs: dict, cfg: dict, key,
                  notes: list[str]) -> tuple[dict, list[str]]:
    planet = str(cp["planet"])
    provider = str(cp.get("chem_provider", "vulcan"))
    tp_mode = str(cp.get("tp_mode", "guillot"))
    if provider == "picaso" or tp_mode == "picaso_climate":
        raise ValueError(
            "this configuration uses the removed PICASO engine "
            "(chem_provider='picaso' or tp_mode='picaso_climate'); the "
            "subsystem was removed in 0.43.0 because it was disabled and "
            "uncertified. Re-create the run with the VULCAN engine.")
    if str(cp.get("mie_condensate", "") or "") or \
            str(cp.get("opacity_mode", "exomolop")) == "lbl":
        raise ValueError(
            "this configuration uses the Mie condensate deck or the sampled "
            "line-by-line opacity mode (mie_condensate / opacity_mode='lbl'), "
            "both removed in 0.48.0: correlated-k over the published ExoMolOP "
            "k-tables is the only opacity path. Re-create the run with the "
            "correlated-k default (the power-law cloud deck stays).")

    def pk(name: str) -> str:            # per-planet widget keys
        return key(f"{planet}_{name}")

    restored_tp, pending_tp = _resolve_embedded_tp(cp, cfg, tp_mode)
    try:
        state = _restore_system_profile(
            cp, key, pk, planet, tp_mode, restored_tp, notes)

        _restore_vulcan_physics(state, cp, key, pk)

        _restore_rt_state(state, cp, key)
        _restore_goal(state, cp, goal, key, tp_mode, notes)
        _restore_observation(state, obs, cfg, key, pk, notes)

        # A transmission-mode file that predates the observation block's star
        # record: the Pandeia noise simulation still needs Teff/log g/[Fe/H],
        # so the fields keeping their current values must be said, not silent.
        if obs and pk("teff") not in state:
            notes.append(
                "this file does not record the stellar parameters used for the "
                "noise simulation (Teff, log g, [Fe/H]); those fields keep "
                "their current values")

        _check_widget_ranges(state, key, pk,
                             str(cp.get("science_mode", "transmission")))
    except Exception:
        if pending_tp is not None:
            pending_tp[0].unlink(missing_ok=True)
        raise
    if pending_tp is not None:
        os.replace(pending_tp[0], pending_tp[1])
        state["restored_tp_path"] = str(pending_tp[1])
    return state, notes
