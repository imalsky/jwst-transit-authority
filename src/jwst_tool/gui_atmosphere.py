"""Sidebar step 2: the chemistry engine, structure, composition, mixing, opacity and clouds."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from jwst_tool import model_defs, planets, run_params, share_config, tp_tables
from jwst_tool.gui_common import K
from jwst_tool.gui_inputs import Atmosphere, Clouds, Mixing, Photochemistry, Profile, Target
from jwst_tool.gui_sources import KSRC_WARN

if TYPE_CHECKING:
    from jwst_tool.tool_types import KeyFn

_W = share_config.WIDGET_RANGES


def _engine_section() -> tuple[bool, str]:
    st.divider()
    st.markdown("### 2 · Atmosphere")

    with st.expander("Chemistry engine", expanded=True):
        # no index=: VULCAN (the first option) is the default
        pic = (
            st.selectbox(
                "Chemistry engine",
                ["vulcan", "picaso"],
                key=K("provider"),
                format_func={
                    "vulcan": "VULCAN-JAX (photochemical kinetics)",
                    "picaso": "PICASO (climate + equilibrium chemistry)",
                }.__getitem__,
            )
            == "picaso"
        )
    chem_provider = "picaso" if pic else "vulcan"
    return pic, chem_provider


def _guillot_boxes(target: Target) -> dict:
    """Draw the Guillot (2010) profile's settings."""
    planet_key, pdef, science_mode, k = target.planet_key, target.pdef, target.science_mode, target.k
    teff, rstar, orbit_au = target.teff, target.rstar, target.orbit_au
    tp_kwargs = {}
    # one definition shared with canonical_params, so the widget and
    # API defaults cannot drift apart
    tirr0 = planets.default_tirr(
        pdef,
        system=(dict(star_teff=teff, rstar_rsun=rstar, orbit_au=orbit_au) if planet_key == "custom" else None),
        science_mode=science_mode,
    )
    # A default outside the supported range leaves the widget empty with the
    # API's refusal beside it, never a clipped number.
    seed: float | None = tirr0
    tirr_error = None
    try:
        planets.check_tirr(tirr0, derived=True)
    except ValueError as e:
        seed, tirr_error = None, str(e)
    if planet_key == "custom":
        # follow-until-overridden: while T_irr still equals the last
        # derived default, keep it tracking the star/orbit fields;
        # the first manual edit breaks the link (otherwise T_irr
        # freezes at the first-render seed).
        prev_auto = st.session_state.get(k("tirr_auto"))
        if prev_auto is not None and st.session_state.get(k("tirr")) == prev_auto:
            st.session_state[k("tirr")] = seed
        st.session_state[k("tirr_auto")] = seed
    # key always seeded (an empty widget refills once the default is valid),
    # so no value= default
    if st.session_state.get(k("tirr")) is None:
        st.session_state[k("tirr")] = seed
    tirr = st.number_input("T_irr (K)", *_W["tirr"], step=20.0, key=k("tirr"))
    if tirr is None:
        # no Tirr in the request: canonical_params refuses with the same sentence
        st.error(tirr_error)
    else:
        tp_kwargs["Tirr"] = tirr
    tp_kwargs["Tint"] = st.number_input("T_int (K)", *_W["tint"], step=25.0, key=k("tint"))
    # default 0.01 cm^2/g, Guillot (2010)'s canonical thermal opacity
    tp_kwargs["log_kappa"] = st.number_input("log10 kappa_IR (cm^2/g)", *_W["lk"], step=0.1, key=k("lk"))
    tp_kwargs["log_gamma"] = st.number_input("log10 gamma (kappa_vis/kappa_IR)", *_W["lg"], step=0.05, key=k("lg"))
    return tp_kwargs


def _table_source(target: Target) -> tuple:
    """Draw the table source and upload; return (source, table path, whether a table is usable)."""
    planet_key, pdef, k = target.planet_key, target.pdef, target.k
    tp_file_path, tp_file_ok = None, True
    # the shipped table is PER-PLANET; a planet without one may only
    # upload
    ship_name = tp_tables.shipped_tp_table_name(planet_key)
    src_opts = [model_defs.TP_FILE_SHIPPED, model_defs.TP_FILE_UPLOAD] if ship_name else [model_defs.TP_FILE_UPLOAD]
    if not ship_name and st.session_state.get(k("tpsrc")) == model_defs.TP_FILE_SHIPPED:
        st.session_state[k("tpsrc")] = model_defs.TP_FILE_UPLOAD
    tp_file = st.radio(
        "Profile source",
        src_opts,
        horizontal=True,
        key=k("tpsrc"),
        format_func={
            model_defs.TP_FILE_SHIPPED: f"Shipped measured profile ({ship_name})",
            model_defs.TP_FILE_UPLOAD: "Upload an array",
        }.__getitem__,
    )
    if not ship_name:
        st.warning(
            "No shipped profile for "
            f"{pdef['label'] if planet_key in planets.PLANETS else 'this planet'}: "
            + (pdef.get("tp_table_note") or "none is bundled.")
            + " Upload a table, or switch to the Guillot profile."
        )
    elif tp_file == model_defs.TP_FILE_SHIPPED and pdef.get("tp_table_note"):
        # the note carries the reason in both directions (planets.py):
        # show it when the shipped table is CHOSEN, not only missing
        st.warning(pdef["tp_table_note"])
    if tp_file == model_defs.TP_FILE_UPLOAD:
        tp_example = (
            "#(dyne/cm2) (K) (cm2/s)\n"
            "Pressure   Temp   Kzz\n"
            "1.000e+09  2255.  1.0e+07\n"
            "1.000e+08  2100.  1.0e+07\n"
            "1.000e+07  1800.  3.0e+07\n"
            "1.000e+06  1400.  1.0e+08\n"
            "1.000e+05  1150.  3.0e+08\n"
            "1.000e+04   980.  1.0e+09\n"
            "1.000e+03   920.  3.0e+09\n"
            "1.000e+02   890.  1.0e+10\n"
            "1.000e+01   875.  3.0e+10\n"
            "1.000e+00   870.  1.0e+11\n"
        )
        with st.expander("Example array"):
            st.code(tp_example)
            st.download_button("Download this example", tp_example, "example_atm.txt", key=k("tpex"), on_click="ignore")
        up_tp = st.file_uploader("Upload an array: T-P (+ optional Kzz) as text", type=["txt", "dat"], key=k("tpup"))
        # a loaded configuration can carry the table; the uploader
        # (whose state cannot be set) wins when a new file is picked
        if up_tp is None and st.session_state.get("restored_tp_path"):
            try:
                rp_tp = Path(st.session_state["restored_tp_path"])
                tp_tables.read_tp_table(rp_tp)
                tp_file_path = str(rp_tp)
            except (OSError, ValueError) as e:
                st.error(f"The restored T-P table is not usable: {e} Upload the table again.")
                tp_file_ok = False
        elif up_tp is not None:
            # validated BEFORE it enters the archive, so a rejected upload leaves nothing behind
            tmp_tp, dst_tp = tp_tables.stage_upload(up_tp.getvalue())
            try:  # loud validation, immediate
                tp_tables.read_tp_table(tmp_tp)
                os.replace(tmp_tp, dst_tp)  # atomic commit
                tp_file_path = str(dst_tp)
            except ValueError as e:
                tmp_tp.unlink(missing_ok=True)
                st.error(f"The temperature-pressure table is not valid: {e} Edit the file and upload it again.")
                tp_file_ok = False
        else:
            st.warning("Upload a table to run in file mode.")
            tp_file_ok = False

    return tp_file, tp_file_path, tp_file_ok


def _tp_section(target: Target, pic: bool, chem_provider: str) -> Profile:
    """Draw the T-P profile; return (tp_mode, its settings, table source, table path, whether a table is usable)."""
    planet_key, pdef, science_mode, k = target.planet_key, target.pdef, target.science_mode, target.k
    with st.expander("Temperature-pressure profile"):
        # each engine keeps its own T-P choice (PICASO adds its climate solver)
        tp_key = k("tp_picaso") if pic else k("tp")
        tp_opts = ["picaso_climate", "guillot", "file"] if pic else ["guillot", "file"]
        # Mirror canonical_params' default so GUI and API defaults agree --
        # INCLUDING the science mode (emission defaults to Guillot on every
        # planet; the bundled tables are terminator profiles). On a science-
        # mode switch, a widget still holding the other mode's default moves
        # to this mode's default; an explicit user choice survives. Under
        # PICASO both defaults are its climate solver, so nothing moves.
        tp_default = run_params.default_tp_mode(
            {"planet": planet_key, "science_mode": science_mode, "chem_provider": chem_provider}
        )
        tp_other = run_params.default_tp_mode(
            {"planet": planet_key, "science_mode": ("transmission" if science_mode == "emission" else "emission")}
        )
        if not pic and st.session_state.get(K("tp_scimode_seen")) != science_mode:
            st.session_state[K("tp_scimode_seen")] = science_mode
            if st.session_state.get(k("tp")) == tp_other != tp_default:
                st.session_state[k("tp")] = tp_default
        if st.session_state.get(tp_key) not in tp_opts:
            st.session_state[tp_key] = tp_default
        # the key is always set above, so no index= default: passing one as
        # well makes Streamlit log a "created with a default value but also
        # set via the Session State API" stack trace on every rerun
        tp_mode = st.selectbox(
            "Temperature-pressure profile",
            tp_opts,
            key=tp_key,
            format_func={
                "picaso_climate": "Radiative-convective (PICASO climate)",
                "guillot": "Guillot (2010)",
                "file": ("Tabulated table (T-P)" if pic else "Tabulated table (T-P, optional Kzz)"),
            }.__getitem__,
        )
        tp_kwargs = {}
        tp_file, tp_file_path, tp_file_ok = "", None, True
        # seeded once (no value= default), so the engine switch can write
        # them back while hidden (gui_sidebar.keep_hidden_engine_state)
        for sk, sv in (("tint", planets.default_tint(pdef)), ("lk", -2.0), ("lg", -1.0), ("kzz", 9.0)):
            if k(sk) not in st.session_state:
                st.session_state[k(sk)] = sv
        if tp_mode == "picaso_climate":
            # T_int shares its key with Guillot: the same physics and default
            tp_kwargs["Tint"] = st.number_input("T_int (K)", *_W["tint"], step=25.0, key=k("tint"))
            rf_key = K(f"rfacv_picaso_{science_mode}")
            if rf_key not in st.session_state:
                st.session_state[rf_key] = model_defs.RFACV_DEFAULT[science_mode]
            tp_kwargs["rfacv"] = st.number_input(
                "Heat redistribution",
                *model_defs.RFACV_RANGE,
                step=0.05,
                key=rf_key,
                help="0.5 = planet average, 1.0 = dayside.",
            )
        elif tp_mode == "guillot":
            tp_kwargs = _guillot_boxes(target)
        elif tp_mode == "file":
            tp_file, tp_file_path, tp_file_ok = _table_source(target)
    return Profile(tp_mode, tp_kwargs, tp_file, tp_file_path, tp_file_ok)


def _pressure_section(science_mode: str, tp_mode: str) -> tuple:
    with st.expander("Pressure limits & reference radius"):
        rt_ptop_bar = st.number_input("Model top pressure (bar)", *_W["rtptop"], 1.0e-9, format="%.1e", key=K("rtptop"))
        p_ref_bar = st.number_input(
            "Reference pressure for the planet radius (bar)", *_W["pref"], format="%.1e", key=K("pref")
        )
        # keyed PER GEOMETRY (shipped key contract). The DEFAULT follows the
        # structure mode -- a measured T-P table caps the honest column at its
        # own bottom (7.6 bar for the shipped tables), parametric profiles get
        # a round 10 bar -- and tracks that mode until the user edits the box
        # (same seed pattern as the Kzz mode below; an untouched default must
        # never survive a structure switch and refuse the run).
        pbtm_key = K(f"pbtm_{science_mode}")
        pbtm_default = run_params.default_p_btm_bar(dict(tp_mode=tp_mode))
        pbtm_now = st.session_state.get(pbtm_key)
        if pbtm_now is None or (
            pbtm_now in (model_defs.P_BTM_FILE_BAR, model_defs.P_BTM_PARAMETRIC_BAR) and pbtm_now != pbtm_default
        ):
            st.session_state[pbtm_key] = pbtm_default
        # key always seeded above, so no value= default
        p_btm_bar = st.number_input(
            "Column bottom pressure (bar)", *model_defs.P_BTM_RANGE, step=1.0, format="%.4g", key=pbtm_key
        )
    return rt_ptop_bar, p_ref_bar, p_btm_bar


def _composition_section() -> tuple:
    with st.expander("Composition"):
        # Composition is STRUCTURAL, one path for every value:
        # metallicity scales O/N/S, C/O sets C_H = co * O_H, the equilibrium
        # seed re-initializes at exactly that composition. No perturbative
        # knob. The whole range under both engines, like C/O below: above
        # PICASO's cap (model_defs.PICASO_MET_RANGE) canonical_params' sentence
        # stops the Run.
        met = st.number_input("Metallicity (× solar)", *model_defs.MET_RANGE, 10.0, 0.5, format="%.2f", key=K("met"))
        # The widget spans the WHOLE admissible range over every (network,
        # photolysis) pair; which part of it is legal for the current pair is
        # canonical_params' call, and an illegal value stops the Run with its
        # one-sentence reason. A widget ceiling that moved with the other two
        # settings would have to clamp a stored value silently to stay in
        # range.
        if K("co") not in st.session_state:
            st.session_state[K("co")] = float(model_defs.CO_DEFAULT)
        # key always seeded above, so no value= default
        co_ratio = st.number_input(
            "C/O", model_defs.CO_MIN, model_defs.CO_MAX_PHOTO_OFF, step=0.05, format="%.3f", key=K("co")
        )
    return met, co_ratio


def _kzz_section(k: KeyFn, pic: bool, tp_mode: str) -> Mixing:
    with st.expander("Vertical mixing (Kzz)"):
        picaso_quench = False
        if pic:
            # PICASO is equilibrium chemistry, optionally quenched at one
            # constant Kzz: no profile menu and no multiplier
            kzz_mode, kzz_x = "const", 1.0
            kzz_const = kzz_kmax = kzz_plev = kzz_kdeep = 0.0
            picaso_quench = st.checkbox("Quench (constant Kzz)", value=False, key=K("quench_picaso"))
            if picaso_quench:
                kzz_const = 10.0 ** st.number_input("log10 Kzz (cm^2 s^-1)", *_W["kzz"], step=0.25, key=k("kzz"))
        else:
            kzz_opts = ["const", "Pfunc", "JM16"]
            # tabulated Kzz needs the tabulated T-P table (its Kzz column)
            kzz_file_ok = tp_mode == "file"
            if kzz_file_ok:
                kzz_opts.append("file")
                # Same rule as canonical_params: a table that carries Kzz
                # supplies the mixing profile, so "file" is the default.
                if k("kzzmode") not in st.session_state:
                    st.session_state[k("kzzmode")] = "file"
            elif st.session_state.get(k("kzzmode")) in ("file", None):
                st.session_state[k("kzzmode")] = "const"
            # key always seeded above, so no index= default
            kzz_mode = st.selectbox(
                "Vertical-mixing profile, Kzz",
                kzz_opts,
                key=k("kzzmode"),
                format_func={
                    "const": "Constant",
                    "Pfunc": "Power law in P",
                    "JM16": "Moses-type P^-0.5 (JM16)",
                    "file": "Tabulated (Kzz column of the T-P table)",
                }.__getitem__,
            )
            kzz_const = kzz_kmax = kzz_plev = kzz_kdeep = 0.0
            kzz_x = 1.0
            if kzz_mode == "const":
                log_kzz = st.number_input("log10 Kzz (cm^2 s^-1)", *_W["kzz"], step=0.25, key=k("kzz"))
                kzz_const = 10.0**log_kzz
            elif kzz_mode == "Pfunc":
                kzz_kmax = 10.0 ** st.number_input("log10 deep Kzz (cm^2 s^-1)", *_W["kzkmax"], 0.25, key=k("kzkmax"))
                kzz_plev = 10.0 ** st.number_input("log10 transition level (bar)", *_W["kzplev"], 0.25, key=k("kzplev"))
            elif kzz_mode == "JM16":
                kzz_kdeep = 10.0 ** st.number_input(
                    "log10 deep-floor Kzz (cm^2 s^-1)", *_W["kzkdeep"], 0.25, key=k("kzkdeep")
                )
            if kzz_mode != "const":
                kzz_x = 10.0 ** st.number_input("Kzz profile multiplier, log10(f)", *_W["kzzx"], 0.05, key=k("kzzx"))
    return Mixing(picaso_quench, kzz_mode, kzz_x, kzz_const, kzz_kmax, kzz_plev, kzz_kdeep)


def _photochemistry_section(pic: bool, science_mode: str) -> Photochemistry:
    if pic:  # VULCAN-JAX settings; none of them is sent under PICASO
        network, use_photo, use_moldiff, use_vm_mol = "", False, False, False
        sl_angle_deg, f_diurnal = 0.0, 1.0
    else:
        with st.expander("Photochemistry & transport"):
            network = st.selectbox(
                "Chemical network",
                list(model_defs.NETWORKS),
                key=K("network"),
                format_func={
                    "sncho2025": "S-N-C-H-O 2025 (default)",
                    "sncho": "S-N-C-H-O 2021",
                    "ncho": "N-C-H-O (no sulfur, faster)",
                }.__getitem__,
            )
            if K("photo") not in st.session_state:
                st.session_state[K("photo")] = True
            # key always seeded above, so no value= default. AD needs
            # photolysis on; canonical_params refuses that combination and
            # the Run block shows its sentence, rather than the widget
            # silently reverting.
            use_photo = st.checkbox("Photochemistry (UV photolysis)", key=K("photo"))
            # The default follows the science mode (planets.SL_ANGLE_DEG, shared with canonical_params): the
            # terminator's slant for a transit, the dayside average for an eclipse. On a mode switch a widget still
            # holding the other mode's default moves to this mode's; an explicit choice survives.
            sza_default = planets.SL_ANGLE_DEG[science_mode]
            sza_other = set(planets.SL_ANGLE_DEG.values()) - {sza_default}
            if K("sza") not in st.session_state or (
                st.session_state.get(K("sza_scimode_seen")) != science_mode and st.session_state[K("sza")] in sza_other
            ):
                st.session_state[K("sza")] = sza_default
            st.session_state[K("sza_scimode_seen")] = science_mode
            sl_angle_deg = st.number_input(
                "Photolysis zenith angle (degrees)", *_W["sza"][:2], step=1.0, key=K("sza"), disabled=not use_photo
            )
            f_diurnal = st.number_input(
                "Diurnal photolysis factor", *_W["fdiur"], 0.05, key=K("fdiur"), disabled=not use_photo
            )
            use_moldiff = st.checkbox("Molecular diffusion", value=True, key=K("moldiff"))
            use_vm_mol = st.checkbox(
                "Upwind molecular-diffusion advection", value=False, key=K("vmmol"), disabled=not use_moldiff
            )

    return Photochemistry(network, use_photo, use_moldiff, use_vm_mol, sl_angle_deg, f_diurnal)


def _opacity_section(pic: bool, network: str, ksrc: dict, ksrc_err: str | None) -> tuple:
    with st.expander("Opacity (ExoJAX)"):
        base_set, extra_set = model_defs.MOLECULES, model_defs.EXTRA_MOLECULES
        # The sulfur-free network removes the S species from both sets. Each
        # non-default network's widgets get their own keys (same pattern as
        # the provider switch) so a selection from one network never strands
        # in another's options.
        if network == "ncho":
            base_set = [m for m in base_set if m not in model_defs.S_MOLECULES]
            extra_set = [m for m in extra_set if m not in model_defs.S_MOLECULES]
        if pic:  # PICASO's equilibrium species list lacks some extras
            extra_set = [m for m in extra_set if m not in model_defs.PICASO_NO_SPECIES]
        st.caption(
            "Correlated-k on ExoMolOP k-tables (R = 1000, "
            "16 g-points, 1-15 µm), scored on the analysis bins "
            "(default R = 100). Sources per species are in the Data "
            "panel."
        )
        st.caption(f"The base set **{' · '.join(base_set)}** is always on.")
        unattributed = [m for m in list(base_set) + list(extra_set) if m not in ksrc]
        if ksrc_err or unattributed:
            st.warning(
                KSRC_WARN
                + (
                    ksrc_err
                    or (
                        "no provenance record for "
                        + ", ".join(unattributed)
                        + "; run python -m vulcan_forward.fetch_exomolop "
                        "--molecules " + ",".join(unattributed)
                    )
                )
            )
        # Preselect the MEASURED-relevant subset, not everything with a
        # table: most species contribute under 1 ppm, and each default-on
        # molecule pays a leave-one-out spectrum.
        extra_mols = st.multiselect(
            "Extra opacity molecules",
            list(extra_set),
            default=[m for m in extra_set if m in model_defs.EXTRA_MOLECULES_DEFAULT],
            key=K(share_config.xmols_key(pic, network)),
        )
    return base_set, extra_set, extra_mols


def _cloud_section(target: Target) -> Clouds:
    science_mode, g_ms2, rp, teq = target.science_mode, target.g_ms2, target.rp, target.teq
    with st.expander("Clouds & scattering (ExoJAX)"):
        if K("rayl") not in st.session_state:
            st.session_state[K("rayl")] = True
        if science_mode == "emission":
            # canonical_params forces Rayleigh OFF in emission (no scattering
            # channel); show the forced state, not a checked-but-ignored box
            st.session_state[K("rayl")] = False
        # key always seeded above, so no value= default
        use_rayleigh = st.checkbox("H2/He Rayleigh scattering", key=K("rayl"), disabled=(science_mode == "emission"))
        cloud_on = st.checkbox("Cloud / haze opacity", value=False, key=K("cloud"))
        cloud_mode = "haze"
        if cloud_on:
            # The gray deck is entered as a cloud-top PRESSURE and mapped to an
            # opacity through the SLANT chord (planets.gray_cloud_log_kappa);
            # the emission RT integrates vertically, so that pressure label is
            # wrong by the chord/scale-height factor there. Emission is offered
            # the haze parameterization only, and a session value carried in
            # from the other geometry is moved back into range before the
            # widget reads it (Streamlit raises on an off-menu value).
            cmode_opts = ["haze"] if science_mode == "emission" else ["haze", "gray"]
            if st.session_state.get(K("cmode")) not in cmode_opts:
                st.session_state[K("cmode")] = "haze"
            cloud_mode = st.radio(
                "Cloud model",
                cmode_opts,
                horizontal=True,
                key=K("cmode"),
                format_func={"haze": "Power-law haze", "gray": "Gray cloud"}.__getitem__,
            )
        cloud_top_bar = None
        if cloud_on and cloud_mode == "gray":
            # Entered where the cloud blocks rather than as an opacity per
            # gram of atmosphere; the two differ by a constant in log space
            # (planets.gray_cloud_log_kappa), so the Fisher width in dex is
            # the same number for either coordinate.
            log_p_top = st.number_input("log10 cloud top pressure (bar)", *_W["cpt"], 0.25, key=K("cpt"))
            cloud_top_bar = 10.0**log_p_top
            log_kappa_cloud = planets.gray_cloud_log_kappa(cloud_top_bar, g_ms2 * 100.0, rp, teq)
            alpha_cloud = 0.0
        elif cloud_on:
            log_kappa_cloud = st.number_input("log10 kappa_cloud (cm^2/g at 3.5 µm)", *_W["ck"], 0.1, key=K("ck"))
            alpha_cloud = st.number_input("Cloud spectral slope alpha (kappa ∝ nu^alpha)", *_W["ca"], 0.25, key=K("ca"))
        else:
            log_kappa_cloud, alpha_cloud = -1.0, 0.0
    return Clouds(use_rayleigh, cloud_on, cloud_mode, cloud_top_bar, log_kappa_cloud, alpha_cloud)


def atmosphere_step(target: Target, ksrc: dict, ksrc_err: str | None) -> Atmosphere:
    """Draw step 2 and return its settings."""
    pic, chem_provider = _engine_section()
    profile = _tp_section(target, pic, chem_provider)
    rt_ptop_bar, p_ref_bar, p_btm_bar = _pressure_section(target.science_mode, profile.tp_mode)
    met, co_ratio = _composition_section()
    mixing = _kzz_section(target.k, pic, profile.tp_mode)
    photo = _photochemistry_section(pic, target.science_mode)
    base_set, extra_set, extra_mols = _opacity_section(pic, photo.network, ksrc, ksrc_err)
    clouds = _cloud_section(target)
    return Atmosphere(
        pic,
        chem_provider,
        profile,
        rt_ptop_bar,
        p_ref_bar,
        p_btm_bar,
        met,
        co_ratio,
        mixing,
        photo,
        base_set,
        extra_set,
        extra_mols,
        clouds,
    )
