"""The sidebar: configuration load and reset, target, science goal, observation and solver settings."""

from __future__ import annotations

import json

import numpy as np
import streamlit as st

from jwst_tool import (
    archive,
    datacheck,
    fisher as fisher_mod,
    instruments as ins,
    model_defs,
    noise as noise_mod,
    planets,
    share_config,
)
from jwst_tool.gui_atmosphere import atmosphere_step
from jwst_tool.gui_common import K
from jwst_tool.gui_inputs import Atmosphere, Goal, NoiseModel, Observation, Solver, Target

# default target precision per parameter (DISPLAY units: dex / K / absolute C/O)
_TARGET_DEFAULT = {
    "lnZ": 0.10,
    "dlnCO": 0.10,
    "lnKzz": 0.30,
    "Tirr": 50.0,
    "Tint": 50.0,
    "log_kappa": 0.30,
    "log_gamma": 0.30,
    "log_kappa_cloud": 0.30,
    "alpha_cloud": 0.50,
}


# Every freeable Fisher parameter can be the constraint goal and looks up
# _TARGET_DEFAULT[goal_param]; guard for missing entries at import, not at
# click time.
_FREEABLE = (
    set(model_defs.CHEM_PARAM_NAMES)
    | set(model_defs.CLOUD_FISHER_PARAMS)
    | {p for ns in model_defs.TP_PARAM_NAMES.values() for p in ns}
)


_missing_target = _FREEABLE - set(_TARGET_DEFAULT)


if _missing_target:
    raise RuntimeError(
        f"_TARGET_DEFAULT is missing {sorted(_missing_target)}: every freeable Fisher parameter needs a target default."
    )


def _reset_all() -> None:
    n = st.session_state.get("reset_nonce", 0) + 1
    st.session_state.clear()
    st.session_state["reset_nonce"] = n


def _arm_reset() -> None:
    st.session_state["confirm_reset"] = True


def _disarm_reset() -> None:
    st.session_state["confirm_reset"] = False


_W = share_config.WIDGET_RANGES  # widget ranges (and fixed defaults), shared


def _populate_config() -> None:
    """Populate-button callback: write the uploaded configuration file into the widget state.

    A callback runs to completion before the script, so no widget has instantiated yet and a rerun cannot interrupt
    it half-way; either the whole mapping applies or nothing does.
    """
    for k in ("_cfg_load_error", "_cfg_populated"):
        st.session_state.pop(k, None)
    up = st.session_state.get(K("cfg_upload"))
    if up is None:
        return
    try:
        cfg = json.loads(up.getvalue().decode())
        state = share_config.widget_state(cfg, K)
    except Exception as e:
        # Anything the file can provoke is a bad file, not a tool defect: an
        # out-of-float literal reaches the int() casts as OverflowError, and a
        # raw traceback on a public page is never the right answer.
        st.session_state["_cfg_load_error"] = str(e)
        return
    st.session_state.pop("restored_tp_path", None)
    st.session_state.pop("restored_floor_table", None)
    # A loaded configuration supersedes any queued archive fill AND its
    # banner: the fill notes describe values the restore may have just
    # replaced, so leaving them would misattribute the form to the archive.
    st.session_state.pop("_archive_fill_pending", None)
    st.session_state.pop("_archive_fill_error", None)
    for k, v in state.items():
        st.session_state[k] = v
    st.session_state["_cfg_populated"] = True


def apply_pending_archive_fill() -> None:
    """Fill the custom planet's widgets from a queued archive lookup.

    Widget keys can only be written BEFORE the widgets instantiate, so the
    Fill button's callback just stashes the planet name (un-namespaced on
    purpose -- a reset's session_state.clear() must kill a queued fill) and
    this applies it at the top of the next run, before any widget renders.
    Out-of-range/missing fields are never written
    (Streamlit silently discards an out-of-range value, substituting the
    default); archive.custom_fill leaves them out of its result, so those
    fields keep their current values.
    """
    name = st.session_state.pop("_archive_fill_pending", None)
    if not name:
        return
    try:
        values = archive.custom_fill(archive.lookup(name))
    except (KeyError, archive.SnapshotError) as e:
        st.session_state["_archive_fill_error"] = str(e)
        return
    # The archive duration is the PRIMARY-TRANSIT duration; in emission mode
    # the event is the secondary eclipse, whose duration can differ (eccentric
    # systems), so the field is left for the user to set.
    if st.session_state.get(K("scimode")) == "emission" and "t14" in values:
        values.pop("t14")
    for suffix, v in values.items():
        st.session_state[K(f"custom_{suffix}")] = v
    st.session_state.pop("_archive_fill_error", None)


def _queue_archive_fill() -> None:
    st.session_state["_archive_fill_pending"] = st.session_state.get(K("custom_arch_name"))


def keep_hidden_engine_state() -> None:
    """Write the hidden engine's settings back to themselves, so they survive an engine switch."""
    # The chemistry engine. Its widget renders in step 2, but step 1's UV menu
    # already depends on it, so it is read from the session state here.
    pic = st.session_state.get(K("provider"), "vulcan") == "picaso"
    # Streamlit deletes the state of a widget a run does not render, so the hidden
    # engine's settings are written back to THEMSELVES (Streamlit's idiom; never
    # a different value) and survive an engine switch. Only hidden keys (or
    # seeded ones without a value= default): a rendered widget with a value=
    # default logs a trace when its key is set.
    pl = st.session_state.get(K("planet"))
    if pic:
        hidden = ["network", "photo", "sza", "fdiur", "moldiff", "vmmol", "jacm", "yconv", "yconvmin"]
        hidden += [f"{pl}_{n}" for n in ("tp", "kzzmode", "kzkmax", "kzplev", "kzkdeep", "kzzx")]
        if not st.session_state.get(K("quench_picaso")):
            hidden.append(f"{pl}_kzz")  # the constant-Kzz box both engines use
        # the Guillot boxes both engines use, hidden unless PICASO's T-P is Guillot
        tp_pic = st.session_state.get(K(f"{pl}_tp_picaso"))
        if tp_pic != "guillot":
            hidden += [f"{pl}_tirr", f"{pl}_lk", f"{pl}_lg"]
        if tp_pic == "file":
            hidden.append(f"{pl}_tint")
    else:
        # the T-P and Kzz boxes both engines share, hidden under a VULCAN table
        # profile or Kzz mode; all are seeded without a value= default
        hidden = [f"{pl}_{n}" for n in ("tint", "tirr", "lk", "lg", "kzz")]
    # Engine-tagged keys of the hidden engine: its molecules and goal selections
    # (and PICASO's T-P, quench and FD boxes).
    tag = "_vulcan" if pic else "picaso"
    for hk in [K(h) for h in [*hidden, f"{pl}_tpsrc", "rfacv_picaso_transmission", "rfacv_picaso_emission"]] + [
        k for k in list(st.session_state) if tag in str(k)
    ]:
        if hk in st.session_state:
            st.session_state[hk] = st.session_state[hk]


def _configuration_step() -> None:
    # Step 0: Load a configuration. Selecting the file does nothing by
    # itself; the Populate button's callback (_populate_config) writes it
    # into the widgets; the line below confirms it.
    st.markdown("### 0 · Configuration")
    # Every step's controls sit behind an expander,
    # so the sidebar reads as a short list of titled sections. The
    # load-outcome messages stay OUTSIDE the expander: a failed restore must
    # be loud without opening anything.
    with st.expander("Load a configuration (JSON)"):
        cfg_up = st.file_uploader("Configuration file", type=["json"], key=K("cfg_upload"))
        st.button("Populate", disabled=(cfg_up is None), on_click=_populate_config, key=K("cfg_populate"))
    if st.session_state.get("_cfg_load_error"):
        st.error("The configuration file could not be applied: " + st.session_state["_cfg_load_error"])
    elif st.session_state.get("_cfg_populated"):
        st.success("Populated")
    st.divider()


def _target_step() -> Target:
    """Draw step 1 and return its settings."""
    pic = st.session_state.get(K("provider"), "vulcan") == "picaso"

    def _k(name: str) -> str:  # per-planet widget state
        return K(f"{planet_key}_{name}")

    # Step 1: Target
    st.markdown("### 1 · Target")
    with st.expander("Planet & observation type", expanded=True):
        planet_key = st.selectbox(
            "Planet",
            [*list(planets.PLANETS), "custom"],
            key=K("planet"),
            format_func=lambda k: planets.PLANETS[k]["label"] if k in planets.PLANETS else "Custom planet …",
        )
        pdef = planets.PLANETS.get(planet_key, planets.CUSTOM_DEFAULTS)
        science_mode = st.radio(
            "Observation type",
            ["transmission", "emission"],
            horizontal=True,
            key=K("scimode"),
            format_func={"transmission": "Transmission", "emission": "Emission"}.__getitem__,
        )
    # Event word for observation-facing labels: only the vocabulary changes
    # with the observation type, not the calculation.
    evw = "eclipse" if science_mode == "emission" else "transit"

    with st.expander("System parameters", expanded=(planet_key == "custom")):
        if planet_key == "custom":
            # Optional archive fill: values come from the SHIPPED PSCompPars
            # snapshot (date disclosed), never a live query. The button's
            # callback only queues the name; apply_pending_archive_fill
            # wrote the widget keys at the top of this run.
            try:
                snap = archive.load_snapshot()
            except archive.SnapshotError as e:
                snap = None
                st.error(f"Archive lookup unavailable: {e}")
            if snap is not None:
                arch_sel = st.selectbox(
                    "Fill from the NASA Exoplanet Archive (optional)",
                    snap.names,
                    index=None,
                    placeholder="Search by planet name ...",
                    key=K("custom_arch_name"),
                )
                st.button(
                    "Fill the system parameters",
                    disabled=(arch_sel is None),
                    on_click=_queue_archive_fill,
                    key=K("custom_arch_fill"),
                )
            if st.session_state.get("_archive_fill_error"):
                st.error(st.session_state["_archive_fill_error"])
        ranges = planets.CUSTOM_FIELD_RANGES  # single source with archive fill
        teff = st.number_input(
            "Stellar effective temperature, Teff (K)", *ranges["teff"], pdef["star"]["teff"], 50.0, key=_k("teff")
        )
        logg = st.number_input(
            "Stellar surface gravity, log10(g) (g in cm s^-2)",
            *ranges["logg"],
            pdef["star"]["log_g"],
            0.1,
            key=_k("logg"),
        )
        feh = st.number_input(
            "Stellar metallicity, [Fe/H] (dex)", *ranges["feh"], pdef["star"]["metallicity"], 0.1, key=_k("feh")
        )
        ks_mag = st.number_input("2MASS Ks magnitude", *ranges["ks"], pdef["star"]["ks_mag"], 0.1, key=_k("ks"))
        rstar = st.number_input(
            "Stellar radius (solar radii)", *ranges["rstar"], pdef["rstar_rsun"], 0.01, key=_k("rstar"), format="%.3f"
        )
        rp = st.number_input(
            "Planet radius (Jupiter radii)", *ranges["rp"], pdef["rp_rjup"], 0.01, key=_k("rp"), format="%.3f"
        )
        g_ms2 = st.number_input(
            "Planet surface gravity (m s^-2)", *ranges["g"], pdef["gs_cgs"] / planets.CM_PER_M, 0.5, key=_k("g")
        )
        orbit_au = st.number_input(
            "Semi-major axis (AU)", *ranges["a"], pdef["orbit_au"], 0.001, key=_k("a"), format="%.4f"
        )
        t14 = st.number_input(
            ("Transit duration, T14 (hours)" if evw == "transit" else "Eclipse duration (hours)"),
            *ranges["t14"],
            pdef["t14_hr"],
            0.1,
            key=_k("t14"),
        )
        uv_ok = datacheck.uv_spectra_status()
        sflux = st.selectbox(
            "Stellar UV spectrum (photochemistry)",
            list(planets.SFLUX_CHOICES),
            index=list(planets.SFLUX_CHOICES).index(pdef["sflux"]),
            format_func=lambda f: planets.SFLUX_CHOICES[f] + ("" if uv_ok.get(f) else "  [FILE MISSING]"),
            key=_k("sflux"),
            # Photo-off canonical_params normalizes this to the registry
            # default, so gate the widget like every other photolysis knob --
            # otherwise the pick is silently replaced, and shared that way.
            # Session state, not use_photo: the checkbox renders further down.
            # PICASO has no photolysis.
            disabled=pic or not st.session_state.get(K("photo"), model_defs.USE_PHOTO_DEFAULT),
        )

    # T_eq is derived from the star and orbit for every planet, registry or
    # custom: a stored literature value is a second source of truth that goes
    # stale when the star parameters are refreshed.
    teq = planets.system_teq(teff, rstar, orbit_au)
    return Target(
        planet_key=planet_key,
        pdef=pdef,
        science_mode=science_mode,
        evw=evw,
        teff=teff,
        logg=logg,
        feh=feh,
        ks_mag=ks_mag,
        rstar=rstar,
        rp=rp,
        g_ms2=g_ms2,
        orbit_au=orbit_au,
        t14=t14,
        sflux=sflux,
        teq=teq,
    )


def _free_parameter_section(
    goal: str, do_fisher: bool, goal_param: str | None, marginalize: bool, avail_free: list, goal_sfx: str, pic: bool
) -> tuple:
    """Draw the differentiation method and free parameters, when the run computes derivatives."""
    # Free-parameter settings render only when the run computes derivatives.
    fisher_params: list = []
    jac_method = "fd"
    if goal == "constrain" or do_fisher:
        with st.expander("Differentiation method & free parameters", expanded=(goal == "constrain")):
            if goal == "constrain" and marginalize:
                # Defaults FILTERED by the live menu, key carries the
                # provider: Streamlit hard-raises on a default outside the
                # options.
                fisher_extra = st.multiselect(
                    "Jointly free parameters",
                    avail_free,
                    # lnKzz is not in the defaults (speed); still
                    # selectable, and dropping it tightens the remaining
                    # sigmas toward the conditional bound.
                    default=[p for p in ("lnZ", "dlnCO") if p in avail_free],
                    key=K(f"fx_{goal_sfx}"),
                    format_func=lambda n: fisher_mod.PARAM_LABELS[n],
                )
                fisher_params = sorted(set(fisher_extra) | {goal_param})
            elif goal == "constrain":
                fisher_params = [goal_param]
            else:
                fisher_params = st.multiselect(
                    "Free parameters",
                    avail_free,
                    key=K(f"fp_{goal_sfx}"),
                    default=[p for p in ("lnZ", "dlnCO") if p in avail_free],
                    format_func=lambda n: fisher_mod.PARAM_LABELS[n],
                )
            if pic:
                # PICASO runs outside JAX: finite differences only. Its own
                # key, so the VULCAN choice (K("jacm")) is never written here.
                jac_method = st.selectbox(
                    "Differentiation method",
                    ["fd"],
                    key=K("jacm_picaso"),
                    disabled=True,
                    format_func={"fd": "Finite differences"}.__getitem__,
                )
            else:
                # Seeded once (share_config writes it on Populate), so no
                # index= default.
                if K("jacm") not in st.session_state:
                    st.session_state[K("jacm")] = "ad"
                jac_method = st.selectbox(
                    "Differentiation method",
                    ["fd", "ad"],
                    key=K("jacm"),
                    format_func={
                        "fd": "Finite differences",
                        "ad": "Automatic differentiation (forward-mode, default)",
                    }.__getitem__,
                )
                # Loud slow-path flag: FD re-solves the chemistry per row.
                if fisher_params and jac_method == "fd":
                    st.warning(
                        "Finite differences re-solve the chemistry "
                        "per free parameter and are much slower than "
                        "automatic differentiation."
                    )
    return fisher_params, jac_method


def _goal_step(atm: Atmosphere) -> Goal:
    """Draw step 3, only the controls the selected goal needs, and return its settings."""
    target_mol = None
    st.divider()
    st.markdown("### 3 · Science goal")
    avail_free = share_config.free_parameters(atm.pic, atm.profile.tp_mode, atm.clouds.cloud_on, atm.clouds.cloud_mode)
    goal_sfx = share_config.goal_suffix(atm.pic, atm.profile.tp_mode, atm.clouds.cloud_on, atm.clouds.cloud_mode)
    mol_options = model_defs.active_molecules({"network": atm.photo.network, "extra_mols": atm.extra_mols})

    goal_param, target_prec, marginalize = None, None, True
    do_fisher = False
    # Keys share_config writes on Populate: seeded once here, so neither
    # widget below takes a value=/index= default as well (Streamlit logs a
    # stack trace per rerun when a key is both pre-set and given one).
    if K("tsig") not in st.session_state:
        st.session_state[K("tsig")] = 3.0
    if K("dofish") not in st.session_state:
        st.session_state[K("dofish")] = True
    with st.expander("Goal & target"):
        goal = st.radio(
            "Goal",
            ["detect", "constrain"],
            horizontal=True,
            key=K("goal"),
            format_func={"detect": "Detect a molecule", "constrain": "Constrain a parameter"}.__getitem__,
        )
        if goal == "detect":
            # SO2 is the flagship W39b science under VULCAN (photochemical);
            # in equilibrium (PICASO) it is negligible, so H2O is the default
            mol_default = "SO2" if "SO2" in mol_options and not atm.pic else "H2O"
            target_mol = st.selectbox(
                "Molecule to detect",
                mol_options,
                index=mol_options.index(mol_default),
                key=K(share_config.target_mol_key(atm.pic, atm.photo.network, atm.extra_mols)),
            )
            target_sig = st.number_input("Target significance (σ)", *_W["tsig"], step=0.5, key=K("tsig"))
            do_fisher = st.checkbox("Also calculate parameter constraints", key=K("dofish"))
        else:
            goal_param = st.selectbox(
                "Parameter to constrain",
                avail_free,
                key=K(f"gp_{goal_sfx}"),
                format_func=lambda n: fisher_mod.PARAM_LABELS[n],
            )
            marginalize = st.checkbox("Marginalize over the other parameters", value=True, key=K("marg"))
            unit = fisher_mod.PARAM_UNITS[goal_param]
            # label uses the unit when there is one (dex / K), else the bare
            # symbol -- C/O is a dimensionless number ratio
            tgt_lbl = (
                f"Target uncertainty (±{unit})"
                if unit
                else f"Target uncertainty (±{fisher_mod.PARAM_SYMBOLS[goal_param]})"
            )
            if unit == "K":
                target_prec = st.number_input(
                    tgt_lbl, *_W["tgt_K"], _TARGET_DEFAULT[goal_param], 5.0, key=K(f"tgt_{goal_param}")
                )
            else:
                target_prec = st.number_input(
                    tgt_lbl, *_W["tgt"], _TARGET_DEFAULT[goal_param], 0.01, key=K(f"tgt_{goal_param}")
                )
            target_sig = st.number_input("Report Fisher half-width at Nσ", *_W["tsig"], step=0.5, key=K("tsig"))
    fisher_params, jac_method = _free_parameter_section(
        goal, do_fisher, goal_param, marginalize, avail_free, goal_sfx, atm.pic
    )
    return Goal(
        goal=goal,
        target_mol=target_mol,
        target_sig=target_sig,
        goal_param=goal_param,
        target_prec=target_prec,
        marginalize=marginalize,
        do_fisher=do_fisher,
        fisher_params=fisher_params,
        jac_method=jac_method,
    )


def _noise_section(mode_keys: list, noise_scale: float) -> NoiseModel:
    """Draw the noise floor and multipliers; return (floor type, table, per-mode floors, choice made, multipliers)."""
    with st.expander("Noise model (Pandeia)"):
        st.markdown("**Minimum noise floor** (PandExo convention)")
        # DEFAULTS TO CONSTANT. A default is acceptable in ONE direction
        # only: preselecting "No floor" would claim undemonstrated precision
        # on the user's behalf, while a constant floor at the suggested
        # values claims LESS precision -- the conservative side. The floor
        # is recorded with the run and shown per mode below.
        floor_mode = st.radio(
            "Floor type",
            ["constant", "none", "file"],
            horizontal=True,
            index=0,
            key=K("floormode"),
            format_func={"constant": "Constant (ppm)", "none": "No floor", "file": "Wavelength table"}.__getitem__,
        )
        floor_table = None
        floors = dict.fromkeys(mode_keys)
        if floor_mode == "constant":
            floors = {
                k: st.number_input(
                    f"{ins.MODES[k]['label']} minimum floor (ppm)",
                    *_W["floor"],
                    ins.MODES[k]["floor_ppm_suggested"],
                    5.0,
                    key=K(f"floor_{k}"),
                )
                for k in mode_keys
            }
        elif floor_mode == "file":
            up = st.file_uploader(
                "Two columns: wavelength (µm), floor (ppm)", type=["txt", "csv", "dat"], key=K("floorfile")
            )
            if up is not None:
                try:
                    raw = up.getvalue().decode()
                    delim = "," if "," in raw.splitlines()[0] else None
                    floor_table = np.loadtxt(raw.splitlines(), delimiter=delim, ndmin=2)
                    noise_mod.resolve_floor(np.array([1.0]), floor_table)  # validate loudly now
                except Exception as e:
                    st.error(f"The floor table is not valid: {e} Edit the file and upload it again.")
                    floor_table = None
            elif st.session_state.get("restored_floor_table") is not None:
                # carried by a loaded configuration; the uploader wins when
                # the user picks a new file
                try:
                    floor_table = np.asarray(st.session_state["restored_floor_table"], float)
                    noise_mod.resolve_floor(np.array([1.0]), floor_table)
                except Exception as e:
                    st.error(f"The restored floor table is not usable: {e} Upload the table again.")
                    floor_table = None
            if floor_table is None:
                st.warning(
                    "No valid floor table is loaded. Upload one, or "
                    "select 'No floor'. The run is blocked until "
                    "the choice is explicit."
                )
        if floor_mode == "file" and floor_table is not None:
            floors = dict.fromkeys(mode_keys, floor_table)
        # An unmade floor choice blocks the run: both candidate defaults bias
        # the headline number, so it must not be implicit.
        floor_choice_made = floor_mode in {"none", "constant"} or (floor_mode == "file" and floor_table is not None)

        # per-mode multipliers; the "Global noise multiplier" above scales all
        # of them (composed below, so neither control is a hidden override)
        infl_mode = {
            k: st.number_input(
                f"{ins.MODES[k]['label']} random-noise multiplier", *_W["infl"], 0.05, key=K(f"infl_{k}")
            )
            for k in mode_keys
        }
        infl = {k: float(noise_scale) * float(infl_mode[k]) for k in mode_keys}
    return NoiseModel(floor_mode, floor_table, floors, floor_choice_made, infl_mode, infl)


def _observation_step(target: Target) -> Observation:
    """Draw step 4 and return its settings."""
    evw, _k, t14 = target.evw, target.k, target.t14
    st.divider()
    st.markdown("### 4 · Observation")
    with st.expander(f"Instrument modes & {evw}s", expanded=True):
        mode_keys = st.multiselect(
            "Instrument modes",
            options=list(ins.MODES),
            default=ins.DEFAULT_MODES,
            key=K("modes"),
            format_func=lambda k: ins.MODES[k]["label"],
        )
        # "per mode" is load-bearing: every selected mode is evaluated at this
        # count, and a combined forecast sums their information at the same
        # count, so K modes cost K x this many events.
        n_transits = st.number_input(f"Number of {evw}s per mode", *_W["ntr"], 1, key=K("ntr"))

    with st.expander("Mock noise draw & noise multiplier"):
        show_noise = st.checkbox("Mock noise draw", value=True, key=K("shownoise"))
        seed = st.number_input("Seed", *_W["seed"], key=K("seed"), disabled=not show_noise)
        # ONE knob for "more jitter". It scales the
        # NOISE MODEL, not the draw: posteriors.mock_realization deliberately
        # refuses a draw-only scale factor, because a 2x draw beside 1x error
        # bars is not a realization of the plotted model, and the S/N and
        # Fisher numbers would still assume the unscaled sigma. Scaling the
        # noise model instead moves the error bars, the scores, the forecast
        # widths and the draw together, so the figure stays internally
        # consistent and the value is recorded with the run. Composes with
        # (multiplies) the per-mode multipliers in the Noise model expander,
        # which stay for mode-specific tuning.
        noise_scale = st.number_input("Global noise multiplier", *_W["noisescale"], 0.05, key=K("noisescale"))

    with st.expander("Timing, saturation & binning (Pandeia)"):
        # follow-until-overridden (same pattern as the T_irr widget): while the
        # baseline still equals the T14 it last tracked, it keeps following the
        # event duration (PandExo convention baseline = T14); the first manual
        # edit breaks the link, so a former planet's duration never sticks.
        tb_auto = float(t14)
        tb_prev = st.session_state.get(_k("tbase_auto"))
        if tb_prev is not None and st.session_state.get(_k("tbase")) == tb_prev:
            st.session_state[_k("tbase")] = tb_auto
        st.session_state[_k("tbase_auto")] = tb_auto
        # key always seeded above, so no value= default
        if _k("tbase") not in st.session_state:
            st.session_state[_k("tbase")] = tb_auto
        t_base = st.number_input(
            # max 20 h, not 10: t14 itself spans 0.5-10 h, so a 10 h cap
            # would make a 2 x T14 baseline unreachable above T14 = 5 h.
            f"Out-of-{evw} baseline (hours)",
            *_W["tbase"],
            step=0.1,
            key=_k("tbase"),
            help="Total out-of-event time per visit. Default T14 (PandExo).",
        )
        sat_limit = st.number_input(
            "Saturation limit (fraction of Pandeia's saturation level)", *_W["sat"], 0.05, key=K("sat")
        )
        r_bin = st.number_input("Analysis resolving power, R", *_W["rbin"], 25, key=K("rbin"))
    return Observation(
        mode_keys=mode_keys,
        n_transits=n_transits,
        show_noise=show_noise,
        seed=seed,
        noise_scale=noise_scale,
        t_base=t_base,
        sat_limit=sat_limit,
        r_bin=r_bin,
        noise=_noise_section(mode_keys, noise_scale),
    )


def _more_settings_step(target: Target, atm: Atmosphere) -> Solver:
    """Draw the solver grid and advanced RT settings and return them."""
    science_mode = target.science_mode
    # More settings: solver grid and advanced RT, behind one entry point.
    # No help tooltips in here: the labels stand on their own and the
    # reference material lives in README.md.
    st.divider()
    st.markdown("### More settings")

    with st.expander("Solver & vertical grid"):
        # One control sets both layer COUNTS; the two pressure grids stay
        # distinct (interp_map regrids chemistry onto the RT grid).
        nz = st.number_input(
            "Layers per grid (chemistry and RT)", *model_defs.NZ_RANGE, model_defs.NZ_DEFAULT, 10, key=K("nz")
        )
        # the VULCAN-JAX kinetics solver's gates; not sent under PICASO
        yconv_cri, yconv_min = model_defs.YCONV_DEFAULT, model_defs.YCONV_MIN_DEFAULT
        if not atm.pic:
            yconv_cri = st.number_input(
                "Solver convergence tolerance", *_W["yconv"], 1.0e-4, format="%.1e", key=K("yconv")
            )
            yconv_min = st.number_input(
                "Loose-branch convergence gate", *_W["yconvmin"], 1.0e-4, format="%.1e", key=K("yconvmin")
            )

    # Condensation is API-only: the GUI never sends the key, and
    # canonical_params pins it (and the removed settling/escape/boundary-flux
    # keys) off on its own.

    with st.expander("Advanced radiative transfer (ExoJAX)"):
        # the pressure boundaries and the radius anchor live in step 2
        # ("Pressure limits & reference radius"); only the chord
        # integration choice stays advanced
        if K("rtint") not in st.session_state or science_mode == "emission":
            # canonical_params pins simpson in emission (no transit chord);
            # show the pinned state, not a silently ignored choice
            st.session_state[K("rtint")] = model_defs.RT_INTEGRATION_DEFAULT
        # key always seeded above, so no index= default
        rt_integration = st.selectbox(
            "Transit chord integration",
            ["simpson", "trapezoid"],
            key=K("rtint"),
            disabled=(science_mode == "emission"),
            format_func={"simpson": "Simpson (ExoJAX default)", "trapezoid": "Trapezoid"}.__getitem__,
        )
    return Solver(
        nz=nz,
        yconv_cri=yconv_cri,
        yconv_min=yconv_min,
        rt_integration=rt_integration,
    )


def _reset_control() -> None:
    # Reset sits behind a confirmation step: one click must not clear a long
    # configuration and the current results.
    st.divider()
    if st.session_state.get("confirm_reset"):
        st.warning("Reset all settings to their defaults and clear the current results?")
        rc1, rc2 = st.columns(2)
        rc1.button("Confirm reset", on_click=_reset_all, type="primary")
        rc2.button("Keep settings", on_click=_disarm_reset)
    else:
        st.button("Reset all settings", on_click=_arm_reset)


def render_sidebar(ksrc: dict, ksrc_err: str | None) -> tuple[Target, Atmosphere, Goal, Observation, Solver]:
    """Draw the whole sidebar and return each step's settings."""
    with st.sidebar:
        _configuration_step()
        target = _target_step()
        atm = atmosphere_step(target, ksrc, ksrc_err)
        goal = _goal_step(atm)
        obs = _observation_step(target)
        solver = _more_settings_step(target, atm)
        _reset_control()
    return target, atm, goal, obs, solver
