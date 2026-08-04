"""Shareable run configurations: build the download, restore the upload.

The GUI's "Download configuration (JSON)" button writes the dict `build_share`
assembles: the canonical model parameters (the same dictionary every result
stores as provenance) plus the science-goal and observation settings, and --
when the run uses uploaded tables -- the table content itself, so the file
fully configures the run on any machine.

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
from pathlib import Path

from jwst_tool import forward, planets

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


def _widget_state(cp: dict, goal: dict, obs: dict, cfg: dict, key,
                  notes: list[str]) -> tuple[dict, list[str]]:
    planet = str(cp["planet"])
    provider = str(cp.get("chem_provider", "vulcan"))
    tp_mode = str(cp.get("tp_mode", "guillot"))

    def pk(name: str) -> str:            # per-planet widget keys
        return key(f"{planet}_{name}")

    # -- restore an embedded uploaded T-P table BEFORE validation, so the
    #    canonical re-resolution by content sha can find it
    restored_tp = None
    validate = dict(cp)
    if tp_mode == "file" and str(cp.get("tp_file", "")) == forward.TP_FILE_UPLOAD:
        text = cfg.get("tp_table_text")
        if text:
            raw = str(text).encode()
            sha = hashlib.sha1(raw).hexdigest()[:16]
            dst = forward._uploads_dir() / f"{sha}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(raw)
            restored_tp = str(dst)
        else:
            sha = str(cp.get("tp_file_sha1", ""))
            cand = forward._uploads_dir() / f"{sha}.txt" if sha else None
            if cand is not None and cand.exists():
                restored_tp = str(cand)
            else:
                raise ValueError(
                    "the configuration uses an uploaded T-P table whose "
                    "content is not embedded in the file and is not in the "
                    "local uploads archive; upload the table again after "
                    "loading")
        validate["tp_file_path"] = restored_tp
    # the loud gate: an invalid file must fail HERE, before anything applies
    forward.canonical_params(validate)

    state: dict = {}
    state[key("planet")] = planet
    state[key("scimode")] = str(cp.get("science_mode", "transmission"))
    state[key("provider")] = provider

    # -- system parameters (step 1; per-planet keys)
    state[pk("teff")] = float(cp["star_teff"])
    state[pk("logg")] = float(cp["star_logg"])
    state[pk("feh")] = float(cp["star_feh"])
    state[pk("rp")] = float(cp["rp_rjup"])
    state[pk("g")] = float(cp["gs_cgs"]) / 100.0
    state[pk("rstar")] = float(cp["rstar_rsun"])
    state[pk("a")] = float(cp["orbit_au"])
    if str(cp.get("sflux", "")) in planets.SFLUX_CHOICES:
        state[pk("sflux")] = str(cp["sflux"])

    # -- temperature-pressure profile (step 2)
    state[pk("tp")] = tp_mode
    if tp_mode == "guillot":
        state[pk("tirr")] = float(cp["Tirr"])
        state[pk("tint")] = float(cp["Tint"])
        state[pk("lk")] = float(cp["log_kappa"])
        state[pk("lg")] = float(cp["log_gamma"])
    elif tp_mode == "picaso_climate":
        state[pk("tintcl")] = float(cp["tint_cl"])
        rf = float(cp["rfacv"])
        state[pk("rfacv")] = min(forward.RFACV_CHOICES,
                                 key=lambda c: abs(c - rf))
        state[pk("tiovo")] = bool(cp["tio_vo"])
        state[pk("rcb")] = int(cp["climate_rcb"])
    elif tp_mode == "file":
        state[pk("tpsrc")] = str(cp.get("tp_file", forward.TP_FILE_SHIPPED))
        if restored_tp:
            state["restored_tp_path"] = restored_tp

    # -- composition (step 2)
    met = float(cp["met_x_solar"])
    co = float(cp["co_ratio"])
    if tp_mode == "picaso_climate":
        from jwst_tool import picaso_chem as pchem
        feh = round(math.log10(met), 1)
        feh_opts = [x for x in pchem.FEH_NODES if -1.0 <= x <= 2.0]
        if feh in feh_opts:
            state[key("metnode")] = feh
            if co in pchem.CO_NODES:
                state[key(f"conode_{feh:.1f}")] = co
        else:
            notes.append(f"climate metallicity {met:g}x solar is not a "
                         "selectable node and was not restored")
    elif provider == "picaso":
        state[key("met_pic")] = met
        state[key("co_pic")] = co
    else:
        state[key("met")] = met
        state[key("co")] = co

    # -- kinetics-engine physics (step 2; VULCAN only)
    if provider == "vulcan":
        kzz_mode = str(cp.get("kzz_mode", "const"))
        state[pk("kzzmode")] = kzz_mode
        if kzz_mode == "const":
            state[pk("kzz")] = _log10(cp["kzz_const"], "kzz_const")
        elif kzz_mode == "Pfunc":
            state[pk("kzkmax")] = _log10(cp["kzz_kmax"], "kzz_kmax")
            state[pk("kzplev")] = _log10(cp["kzz_plev"], "kzz_plev")
        elif kzz_mode == "JM16":
            state[pk("kzkdeep")] = _log10(cp["kzz_kdeep"], "kzz_kdeep")
        if kzz_mode != "const":
            state[pk("kzzx")] = _log10(cp.get("kzz_x", 1.0), "kzz_x")
        state[key("photo")] = bool(cp["use_photo"])
        state[key("sza")] = float(cp["sl_angle_deg"])
        state[key("fdiur")] = float(cp["f_diurnal"])
        state[key("moldiff")] = bool(cp["use_moldiff"])
        state[key("vmmol")] = bool(cp["use_vm_mol"])
        state[key("conden")] = bool(cp["use_condense"])
        state[key("settle")] = bool(cp["use_settling"])
        state[key("descape")] = [s for s in (cp.get("diff_esc") or [])
                                 if s in forward.DIFF_ESC_CHOICES]
        state[key("topflux")] = "\n".join(
            " ".join(str(x) for x in row) for row in (cp.get("top_flux") or []))
        state[key("botflux")] = "\n".join(
            " ".join(str(x) for x in row) for row in (cp.get("bot_flux") or []))
        state[key("yconv")] = float(cp["yconv_cri"])

    # -- clouds & scattering (step 2)
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

    # -- opacity & line lists (step 2)
    if provider == "picaso":
        from jwst_tool import picaso_chem as pchem
        extras_all = list(pchem.PICASO_EXTRA_MOLECULES)
    else:
        extras_all = list(forward.EXTRA_MOLECULES)
    extras = [m for m in (cp.get("extra_mols") or []) if m in extras_all]
    state[key(f"xmols_{provider}")] = extras
    state[key("broad")] = str(cp.get("broadening", "air"))
    state[key("nupts")] = int(cp["nu_pts"])

    # -- More settings: solver grid + advanced RT
    state[key("nz_pic" if provider == "picaso" else "nz")] = int(cp["nz"])
    state[key("rtptop")] = float(cp["rt_ptop_bar"])
    state[key("rtint")] = str(cp["rt_integration"])
    state[key("rtdit")] = float(cp["rt_dit_res"])

    # -- science goal (step 3)
    cloud_i, mie_i = int(bool(cp["cloud_on"])), int(bool(state[key("miec")]))
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
    g = str(goal.get("goal", "") or "")
    if g in ("detect", "constrain"):
        state[key("goal")] = g
        if goal.get("target_sig") is not None:
            state[key("tsig")] = float(goal["target_sig"])
        jm = str(goal.get("jac_method", cp.get("jac_method", "fd")))
        if jm in ("fd", "ad"):
            state[key("jacm")] = jm
        if g == "detect":
            mol_opts = forward.active_molecules(
                {"chem_provider": provider, "extra_mols": extras})
            tm = goal.get("target_mol")
            if tm in mol_opts:
                state[key(f"mol_{provider}_"
                          + "_".join(sorted(extras)))] = tm
            elif tm:
                notes.append(f"detection target {tm} is not in the restored "
                             "molecule set and was not restored")
            state[key("dofish")] = bool(goal.get("do_fisher", bool(fisher)))
            if fisher:
                state[key(f"fp_{suffix}")] = fisher
        else:
            gp = goal.get("goal_param")
            if gp in avail:
                state[key(f"gp_{suffix}")] = gp
                if goal.get("target_prec") is not None:
                    state[key(f"tgt_{gp}")] = float(goal["target_prec"])
            elif gp:
                notes.append(f"constraint parameter {gp} is not available "
                             "under the restored settings")
            marg = bool(goal.get("marginalize", True))
            state[key("marg")] = marg
            if marg and fisher:
                state[key(f"fx_{suffix}")] = [p for p in fisher if p != gp]

    # -- observation (step 4)
    if obs:
        from jwst_tool import instruments as ins
        from jwst_tool import noise as noise_mod
        if obs.get("ks_mag") is not None:
            state[pk("ks")] = float(obs["ks_mag"])
        if obs.get("t14") is not None:
            state[pk("t14")] = float(obs["t14"])
        if obs.get("t_base") is not None:
            state[pk("tbase")] = float(obs["t_base"])
        modes = [m for m in (obs.get("modes") or []) if m in ins.MODES]
        dropped = [m for m in (obs.get("modes") or []) if m not in ins.MODES]
        if dropped:
            notes.append(f"unknown instrument mode(s) {dropped} were not "
                         "restored")
        if modes:
            state[key("modes")] = modes
        if obs.get("n_transits") is not None:
            state[key("ntr")] = int(obs["n_transits"])
        if obs.get("sat_limit") is not None:
            state[key("sat")] = float(obs["sat_limit"])
        if obs.get("r_bin") is not None:
            state[key("rbin")] = int(obs["r_bin"])
        fm = obs.get("floor_mode")
        if fm in ("constant", "none", "file"):
            state[key("floormode")] = fm
            if fm == "constant":
                for m, v in (obs.get("floors") or {}).items():
                    if m in ins.MODES and isinstance(v, (int, float)):
                        state[key(f"floor_{m}")] = float(v)
            if fm == "file":
                tab = cfg.get("floor_table")
                if tab:
                    state["restored_floor_table"] = [
                        [float(a), float(b)] for a, b in tab]
                else:
                    notes.append("the wavelength-table noise floor is not "
                                 "embedded in this file; upload it again")
        for m, v in (obs.get("noise_infl") or {}).items():
            if m in ins.MODES and isinstance(v, (int, float)):
                state[key(f"infl_{m}")] = float(v)
        sc = obs.get("scenario")
        if sc in noise_mod.SCENARIOS:
            state[key("scenario")] = sc
        if obs.get("seed") is not None:
            state[key("seed")] = int(obs["seed"])
        if obs.get("show_noise") is not None:
            state[key("shownoise")] = bool(obs["show_noise"])

    return state, notes
