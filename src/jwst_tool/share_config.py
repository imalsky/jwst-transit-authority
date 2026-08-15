"""Shareable run configurations: build the download, restore the upload.

The GUI's "Download configuration (JSON)" button writes the dict `build_share`
assembles: the canonical model parameters (the same dictionary every result
stores as provenance) plus the science-goal and observation settings, and --
when the run uses uploaded tables -- the table content itself. That is every
INPUT the tool takes, so the same tool version reproduces the run from the
file alone. Exact code revisions, science-data checksums, cache schemas,
Pandeia/PandExo identity, and the random seed are recorded in ``provenance``.
They remain informational on load so old configurations stay portable, but a
collaborator can compare them before claiming an exact reproduction.

`widget_state` is the inverse: it maps such a file (or a bare canonical-params
dict from an older download) onto Streamlit session-state widget keys. It
validates through `forward.canonical_params` first and raises ValueError on
anything invalid -- the caller applies either the whole mapping or nothing.

This lives outside app.py so the mapping is importable and unit-testable
without running Streamlit.
"""
from __future__ import annotations

import hashlib
import math
import os

from jwst_tool import forward, planets, provenance

SHARE_FORMAT = 1


def build_share(canon: dict, goal: dict, observation: dict,
                tp_table_text: str | None = None,
                floor_table: list | None = None) -> dict:
    """Assemble the downloadable configuration dict."""
    share = {
        "jwst_tool_config": SHARE_FORMAT,
        "canonical_params": dict(canon),
        "goal": dict(goal),
        "observation": dict(observation),
        # Which software wrote this file is recorded ONCE, inside the
        # provenance block ("software"): provenance._versions() is a superset
        # of the top-level copy this key used to carry (it adds jax, numpy and
        # picaso). Informational only -- a configuration must load on any tool
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


def _resolve_embedded_tp(cp: dict, cfg: dict, tp_mode: str) -> str | None:
    """Validate the full canonical payload and atomically archive its T-P upload."""
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
    if pending is not None:
        os.replace(pending[0], pending[1])
        restored = str(pending[1])
    return restored


def _restore_goal(state: dict, cp: dict, goal: dict, key, provider: str,
                  tp_mode: str, notes: list[str]) -> None:
    cloud_i = int(bool(cp["cloud_on"]))
    mie_i = int(bool(state[key("miec")]))
    suffix = f"{provider}_{tp_mode}_{cloud_i}_{mie_i}"
    if provider == "picaso":
        chem_free = (["lnZ"] if tp_mode == "picaso_climate"
                     else ["lnZ", "dlnCO"])
    else:
        chem_free = list(forward.CHEM_PARAM_NAMES)
    avail = chem_free + list(forward.TP_PARAM_NAMES[tp_mode])
    if cp["cloud_on"]:
        avail += list(forward.CLOUD_FISHER_PARAMS)
    if mie_i:
        avail += list(forward.MIE_FISHER_PARAMS)
    fisher = [p for p in (goal.get("fisher_params")
                          or cp.get("fisher_params") or []) if p in avail]
    selected_goal = str(goal.get("goal", "") or "")
    if selected_goal not in ("detect", "constrain"):
        return
    state[key("goal")] = selected_goal
    if goal.get("target_sig") is not None:
        state[key("tsig")] = float(goal["target_sig"])
    method = str(goal.get("jac_method", cp.get("jac_method", "fd")))
    if method in ("fd", "ad"):
        state[key("jacm")] = method
    if selected_goal == "detect":
        extras = state[key(f"xmols_{provider}")]
        mol_opts = forward.active_molecules(
            {"chem_provider": provider, "extra_mols": extras})
        target = goal.get("target_mol")
        if target in mol_opts:
            state[key(f"mol_{provider}_" + "_".join(sorted(extras)))] = target
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
            ("t_base", "tbase", float)):
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
    if floor_mode in ("constant", "none", "file"):
        state[key("floormode")] = floor_mode
        if floor_mode == "constant":
            for mode, value in (obs.get("floors") or {}).items():
                if mode in ins.MODES and isinstance(value, (int, float)):
                    state[key(f"floor_{mode}")] = float(value)
        elif floor_mode == "file":
            table = cfg.get("floor_table")
            if table:
                state["restored_floor_table"] = [
                    [float(a), float(b)] for a, b in table]
            else:
                notes.append("the wavelength-table noise floor is not embedded "
                             "in this file; upload it again")
    for mode, value in (obs.get("noise_infl") or {}).items():
        if mode in ins.MODES and isinstance(value, (int, float)):
            state[key(f"infl_{mode}")] = float(value)
    scale = obs.get("noise_scale")
    if isinstance(scale, (int, float)) and float(scale) > 0.0:
        state[key("noisescale")] = float(scale)
    if obs.get("scenario") not in (None, "random"):
        notes.append(
            f"this configuration selected the removed experimental noise "
            f"scenario {obs['scenario']!r}; the standard (diagonal) noise "
            "model is used instead")
    combos = _restore_combos(obs, ins.MODES, notes)
    if combos:
        state[key("combos")] = combos


def _restore_system_profile(cp: dict, key, pk, planet: str, provider: str,
                            tp_mode: str, restored_tp: str | None,
                            notes: list[str]) -> dict:
    state = {
        key("planet"): planet,
        key("scimode"): str(cp.get("science_mode", "transmission")),
        key("provider"): provider,
        pk("teff"): float(cp["star_teff"]),
        pk("logg"): float(cp["star_logg"]),
        pk("feh"): float(cp["star_feh"]),
        pk("rp"): float(cp["rp_rjup"]),
        pk("g"): float(cp["gs_cgs"]) / 100.0,
        pk("rstar"): float(cp["rstar_rsun"]),
        pk("a"): float(cp["orbit_au"]),
        pk("tp"): tp_mode,
    }
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
    elif tp_mode == "picaso_climate":
        state.update({
            pk("tintcl"): float(cp["tint_cl"]),
            pk("rfacv"): float(cp["rfacv"]),
            pk("tiovo"): bool(cp["tio_vo"]),
            pk("rcb"): int(cp["climate_rcb"]),
        })
    elif tp_mode == "file":
        state[pk("tpsrc")] = str(cp.get("tp_file", forward.TP_FILE_SHIPPED))
        if restored_tp:
            state["restored_tp_path"] = restored_tp
    metallicity = float(cp["met_x_solar"])
    co_ratio = float(cp["co_ratio"])
    if tp_mode == "picaso_climate":
        from jwst_tool import picaso_chem as pchem
        feh = round(math.log10(metallicity), 1)
        if feh in [x for x in pchem.FEH_NODES if -1.0 <= x <= 2.0]:
            state[key("metnode")] = feh
            if co_ratio in pchem.CO_NODES:
                state[key(f"conode_{feh:.1f}")] = co_ratio
        else:
            notes.append(f"climate metallicity {metallicity:g}x solar is not "
                         "a selectable node and was not restored")
    elif provider == "picaso":
        state[key("met_pic")] = metallicity
        state[key("co_pic")] = co_ratio
    else:
        state[key("met")] = metallicity
        state[key("co")] = co_ratio
    return state


def _reject_removed_physics(cp: dict) -> None:
    removed = []
    for active, label in (
            (bool(cp.get("use_condense", False)), "condensation (use_condense)"),
            (bool(cp.get("use_settling", False)),
             "gravitational settling (use_settling)"),
            (bool(cp.get("diff_esc") or []), "diffusion-limited escape (diff_esc)"),
            (bool(cp.get("top_flux") or []), "top-boundary fluxes (top_flux)"),
            (bool(cp.get("bot_flux") or []), "bottom-boundary fluxes (bot_flux)")):
        if active:
            removed.append(label)
    if removed:
        raise ValueError(
            "this configuration enables atmospheric physics the interface no "
            "longer offers -- " + ", ".join(removed) + ". These settings "
            "remain available through the programmatic interface "
            "(jwst_tool.forward.canonical_params); loading them in the GUI "
            "would change the atmosphere.")


def _restore_vulcan_physics(state: dict, cp: dict, key, pk) -> None:
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


def _restore_rt_state(state: dict, cp: dict, key, provider: str) -> None:
    state[key("rayl")] = bool(cp["use_rayleigh"])
    state[key("cloud")] = bool(cp["cloud_on"])
    if cp["cloud_on"]:
        state[key("ck")] = float(cp["log_kappa_cloud"])
        state[key("ca")] = float(cp["alpha_cloud"])
    mie = str(cp.get("mie_condensate", "") or "")
    state[key("miec")] = mie if mie in forward.MIE_CONDENSATES else ""
    if state[key("miec")]:
        state[key("mierg")] = float(cp["mie_log_rg"])
        state[key("miesg")] = float(cp["mie_sigmag"])
        state[key("miemmr")] = float(cp["mie_log_mmr"])
    if provider == "picaso":
        from jwst_tool import picaso_chem as pchem
        extras_all = list(pchem.PICASO_EXTRA_MOLECULES)
    else:
        extras_all = list(forward.EXTRA_MOLECULES)
    state[key(f"xmols_{provider}")] = [
        molecule for molecule in (cp.get("extra_mols") or [])
        if molecule in extras_all]
    state.update({
        key("broad"): str(cp.get("broadening", "air")),
        key("nupts"): int(cp["nu_pts"]),
        key("nz_pic" if provider == "picaso" else "nz"): int(cp["nz"]),
        key("rtptop"): float(cp["rt_ptop_bar"]),
        key("rtint"): str(cp["rt_integration"]),
        key("rtdit"): float(cp["rt_dit_res"]),
    })


def _widget_state(cp: dict, goal: dict, obs: dict, cfg: dict, key,
                  notes: list[str]) -> tuple[dict, list[str]]:
    planet = str(cp["planet"])
    provider = str(cp.get("chem_provider", "vulcan"))
    tp_mode = str(cp.get("tp_mode", "guillot"))

    def pk(name: str) -> str:            # per-planet widget keys
        return key(f"{planet}_{name}")

    restored_tp = _resolve_embedded_tp(cp, cfg, tp_mode)

    state = _restore_system_profile(
        cp, key, pk, planet, provider, tp_mode, restored_tp, notes)

    if provider == "vulcan":
        _restore_vulcan_physics(state, cp, key, pk)

    _restore_rt_state(state, cp, key, provider)
    _restore_goal(state, cp, goal, key, provider, tp_mode, notes)
    _restore_observation(state, obs, cfg, key, pk, notes)

    return state, notes
