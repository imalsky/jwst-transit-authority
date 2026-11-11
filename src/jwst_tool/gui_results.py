"""The result page above the forecast: the stored run, its notices, the shortfall verdict and the structure panel."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

from jwst_tool import detect, fisher as fisher_mod, instruments as ins, model_cache, plotting, run_params
from jwst_tool.gui_common import K, csv_bytes, fig_bytes, show_fig, slug

if TYPE_CHECKING:
    from jwst_tool.tool_types import Slot


@dataclass(frozen=True)
class RunView:
    """The stored run the page renders; it can come from older settings than the sidebar now reads."""

    out: dict
    meta: dict
    model: dict
    results: list
    producer: list | None
    goal: str
    canon: dict
    emission: bool
    event: str
    co_eval: float
    tsig: float
    fisher_names: list
    ok: list
    with_jac: list
    usable_jac: list
    have_fisher: bool
    fname_base: str
    caveats: list

    def csv(self, df: pd.DataFrame) -> bytes:
        """Return a table as CSV bytes carrying this run's identity and caveats."""
        return csv_bytes(df, self.caveats, self.producer)


def _model_caveats(model: dict) -> list:
    """Return the caveats a downloaded table or figure must carry."""
    # Opacity honesty: a species the network solved but the RT cannot see is
    # missing absorption, and the sign of the resulting error is NOT known -- see
    # the matching note in forward._opacity_advisory. Carbon-rich columns are where this bites
    # (C6H6 on sncho2025 at C/O 10). Read from the cache, so a cached run says it
    # too -- run_model logs it, but a cache hit never solves.
    caveats = []
    unmodeled = [str(u).split("|") for u in np.atleast_1d(model["unmodeled"])]
    if unmodeled:
        caveats.append(
            "No opacity table for "
            + ", ".join(f"{sp} (VMR {float(v):.1e})" for sp, v in unmodeled)
            + ": the modelled feature contrast carries an unquantified error, in "
            "either direction."
        )

    # Core absorbers of the band the RT leaves out (HCN, H2S, SO2): their
    # features are missing, which a cache hit must say too.
    unlisted = [str(m) for m in np.atleast_1d(model["unlisted_core"])]
    if unlisted:
        caveats.append(
            f"{', '.join(unlisted)} absorb in this band and are not in the spectrum: their features are missing. "
            "Add them under 'Extra opacity molecules'."
        )
    # A run that read T_int without the request stating it ran at planets.default_tint: say so and which value.
    tint = run_params.tint_caveat(json.loads(str(model["params_json"])))
    if tint:
        caveats.append(tint)
    # Retry honesty: picaso re-solves a layer that does not converge at a
    # temperature shifted by up to 1e-4 of T; the run says how often.
    if str(model["chem_provider"]) == "picaso" and int(model["picaso_retried_layers"][0]):
        caveats.append(
            f"PICASO's equilibrium chemistry converged {int(model['picaso_retried_layers'][0])} layer solves only at "
            "a shifted temperature: at most 1e-4 of T, its own retry ladder."
        )

    # Derivative honesty: a stage that needed the photolysis-cadence escalation
    # (certify.certified_solve) solved on a different photolysis cadence than the
    # configuration asked for, so the run says which stages did. Read from the
    # cache for the same reason as unmodeled above: run_model logs it, but a cache
    # hit never solves.
    escalated = [str(e) for e in np.atleast_1d(model["photo_escalated"])]
    if escalated:
        caveats.append(
            "Photolysis had to be refreshed every accepted step to converge "
            + ", ".join(escalated)
            + ". Those stages were re-solved and certified at that cadence, where "
            "the tool's derivative closures have been checked (transmission "
            "finite differences and eclipse automatic differentiation, both "
            "against independent finite differences); the spectrum itself is "
            "certified as usual."
        )
    # Rate-law honesty: the network's documented temperature ranges are advisory
    # in both VULCAN implementations, so a hot column extrapolates many fitted
    # rates; forward stores the counts.
    rate_out = int(model["rate_rows_outside"][0])
    if rate_out > 0:
        caveats.append(
            f"{rate_out} of {int(model['rate_rows_total'][0])} reaction rates are "
            "evaluated outside their measured temperature range in at least one "
            "layer (advisory: no rate is altered)."
        )
    return caveats


def result_view() -> RunView:
    """Read the stored run from the session."""
    out = st.session_state["out"]
    meta = st.session_state["out_meta"]
    model, results = out["model"], out["results"]
    # what SOLVED this column (forward stamps it into the npz); every CSV export
    # below reads it instead of the installed commits
    producer = [str(x) for x in model["producer"]] if "producer" in model else None
    goal_r = meta["goal"]
    cpj = json.loads(str(model["params_json"]))
    # Event word for the CACHED run's results (the sidebar radio may have moved
    # since the run; the stored canonical params are the truth for this output).
    emis = cpj["science_mode"] == "emission"
    ev = "eclipse" if emis else "transit"
    # the atmosphere's absolute C/O, for the dlnCO -> absolute-C/O display
    # conversion (sigma_CO = C/O * sigma_lnCO)
    co_eval = float(cpj["co_ratio"])
    # one target significance for every number on this page
    tsig = float(meta["target_sig"])
    fisher_names = [str(x) for x in model["jac_names"][:-1]] if "jac_names" in model else []
    ok = [r for r in results if not r["saturated"]]
    # modes with a Jacobian, and those of them with usable (unsaturated) data
    with_jac = [r for r in results if r.get("jac_bins") is not None]
    usable_jac = [r for r in with_jac if not r["saturated"]]
    return RunView(
        out=out,
        meta=meta,
        model=model,
        results=results,
        producer=producer,
        goal=goal_r,
        canon=cpj,
        emission=emis,
        event=ev,
        co_eval=co_eval,
        tsig=tsig,
        fisher_names=fisher_names,
        ok=ok,
        with_jac=with_jac,
        usable_jac=usable_jac,
        have_fisher=bool(fisher_names) and "jac" in model,
        fname_base=f"jwst_tool_{slug(meta['planet'])}",
        caveats=_model_caveats(model),
    )


def render_stale_notice(view: RunView, stale_slot: Slot, params: dict, run_sig: dict) -> None:
    """Say so when the sidebar no longer matches the run on screen, listing what changed."""
    # Staleness guard: results persist in session_state across sidebar edits, so
    # the spectrum shown can be from DIFFERENT settings than the sidebar now
    # reads -- most visibly the transmission/emission geometry. Say so loudly and
    # LIST the changed fields.
    shown_stale, changed = False, []
    try:
        cur_cp = json.loads(json.dumps(run_params.canonical_params(params), default=str))
        shown_stale = model_cache.params_key(params) != model_cache.params_key(view.canon)
        if shown_stale:
            changed = sorted(k for k in set(cur_cp) | set(view.canon) if cur_cp.get(k) != view.canon.get(k))
        # RUN-META, not just canonical model params -- the WHOLE non-canonical
        # input set (see run_sig, built next to the shareable config). The
        # canonical set deliberately excludes the detection target and the
        # observing setup because they do not change the SPECTRUM, but they do
        # change what is DISPLAYED and what the scores are computed from. Compared
        # through json so nested floors/combos/mode lists compare by value.
        meta_chg = []
        for sec, now_d in run_sig.items():
            was_d = view.meta["run_sig"][sec]
            meta_chg += [
                _k
                for _k in sorted(set(now_d) | set(was_d))
                if json.dumps(now_d.get(_k), sort_keys=True, default=str)
                != json.dumps(was_d.get(_k), sort_keys=True, default=str)
            ]
        if meta_chg:
            shown_stale = True
            changed = sorted(set(changed) | set(meta_chg))
    except (ValueError, RuntimeError):
        shown_stale = True  # the current sidebar settings do not even validate
    if shown_stale:
        chg = (
            (" Changed: " + ", ".join(f"`{c}`" for c in changed[:8]) + (", …" if len(changed) > 8 else "") + ".")
            if changed
            else ""
        )
        stale_slot.warning("Showing your previous run." + chg + " Press **Run** to recompute.")


def render_exclusions(view: RunView) -> list:
    """Report failed and unusable modes; return the excluded (mode, reason) pairs."""
    for k, err in view.out["failed"]:
        first = str(err).strip().splitlines()[-1] if "Traceback" in str(err) else str(err).strip().splitlines()[0]
        st.error(f"{ins.MODES[k]['label']}: the calculation failed. {first}")
        with st.expander(f"{ins.MODES[k]['label']}: technical details"):
            st.code(str(err)[-2500:])
    # A mode with no usable data is dropped from every ranking, combination and
    # forecast below, so SAY which and say which CONFIGURATION failed: only the
    # pinned one was evaluated (instruments.py SCOPE note).
    excluded = [(r["mode_key"], "saturated") for r in view.results if r["saturated"]] + [
        (k, str(why)) for k, why in view.out["unusable"]
    ]
    if excluded:
        st.warning(
            "No usable data, excluded from every ranking and forecast: "
            + "; ".join(f"{ins.MODES[k]['label']} ({ins.config_label(k)}) \u2014 {why}" for k, why in excluded)
            + ". Alternative subarrays and readout patterns are not evaluated."
        )
    return excluded


def render_mode_notes(view: RunView, excluded: list) -> dict:
    """Report per-mode risk notes; return the notes the constraint table carries."""
    # Per-mode risk notes: Pandeia's curated warnings, the short-ramp advisory, and
    # the published saturation advice for the run's own limit. Reported, never
    # acted on -- the ramp search already returned the largest measured-safe count,
    # and the saturation limit stays exactly what the user set. Modes already named
    # in excluded are skipped, so a saturated mode is not reported twice.
    ex_keys = {k for k, _ in excluded}
    # A mode with a Jacobian carries its notes in the constraint-forecast table
    # (a "notes" column on its first row); the banner keeps only the modes that
    # table cannot show, so nothing is reported twice or lost.
    in_table = {r["mode_key"] for r in view.with_jac}
    mode_notes, warn_rows = {}, []
    for r in view.results:
        if r["mode_key"] in ex_keys:
            continue
        notes = [str(v) for v in (r.get("warnings") or {}).values()]
        adv = ins.sat_limit_advisory(r["mode_key"], view.meta["sat_limit"])
        if adv:
            notes.append(adv)
        if r.get("ramp_search_complete") is False:
            notes.append("ngroup search hit its budget; the ramp shown is not the maximal safe one")
        if not notes:
            continue
        mode_notes[r["mode_key"]] = "; ".join(notes)
        if r["mode_key"] not in in_table:
            warn_rows.append(f"{r['label']} ({ins.config_label(r['mode_key'])}): " + "; ".join(notes))
    if warn_rows:
        st.warning("\n\n".join(warn_rows))
    return mode_notes


def _detect_verdict(view: RunView, verdict_slot: Slot) -> None:
    ntr = view.meta["n_transits"]
    # every selected mode saturates: never rank saturated results
    if view.ok:
        ranked = sorted(view.ok, key=lambda r: -detect.detection_metric(r)[0])
        best = ranked[0]
        bsig, best_projected = detect.detection_metric(best)
        # "template S/N" stays: a bare sigma reads as a retrieval claim. And
        # SAY WHICH ONE: the projected score profiles the T-P/cloud/lnR0
        # directions as well as the per-segment calibration offsets, so the two
        # are different statistics and must never share one bare label.
        verdict = (
            f"**{best['label']}**: template S/N {bsig:.1f}σ "
            f"({detect.metric_label(best)}) in {ntr} "
            f"{view.event}{'s' if ntr > 1 else ''} (target {view.tsig:g}σ)."
        )
        if bsig >= view.tsig:
            pass  # target met: the figure and table carry the number
        elif bsig > 0:
            # floor-aware transit solver: the photon term averages down with
            # N, the systematic floor does not -- a plain 1/sqrt(N) law is
            # optimistic exactly where it matters
            tt = detect.transits_to_target(best, view.tsig, projected=best_projected)
            if tt["reachable"]:
                verdict_slot.warning(verdict + f"  {tt['n']} {view.event}s reach it.")
            elif detect.has_floor(best) and not np.isnan(tt["sig_inf"]):
                verdict_slot.warning(verdict + f"  Floor caps it at {tt['sig_inf']:.1f}σ.")
            else:
                verdict_slot.warning(verdict + f"  >{detect.N_TRANSITS_CAP} {view.event}s (scan limit).")
        else:
            verdict_slot.warning(verdict + "  No signal.")


def _constrain_verdict(view: RunView, verdict_slot: Slot) -> bool:
    """Fill the constraint warning; False when there is nothing further to draw."""
    gp = view.meta["goal_param"]
    unit = fisher_mod.PARAM_UNITS[gp]
    usp = (" " + unit) if unit else ""  # " dex"/" K", or "" for C/O (ratio)
    glabel = fisher_mod.PARAM_LABELS[gp]
    target = float(view.meta["target_prec"])
    # one saturation policy everywhere: a saturated mode is unusable data,
    # excluded from BOTH the per-mode ranking and the combined forecast
    per_mode = {}  # tsig-sigma half-widths, display units
    for r in view.usable_jac:
        s = fisher_mod.display_sigma(gp, fisher_mod.mode_forecast(r, view.fisher_names)[gp], co_eval=view.co_eval)
        if np.isfinite(s):
            per_mode[r["mode_key"]] = view.tsig * s
    comb = (
        view.tsig
        * fisher_mod.display_sigma(
            gp, fisher_mod.combined_forecast(view.usable_jac, view.fisher_names)[gp], co_eval=view.co_eval
        )
        if len(view.usable_jac) >= 2
        else np.inf
    )
    # distinguish "all modes saturated" from "no spectral response"
    if view.with_jac and not view.usable_jac:
        return False
    ntr = view.meta["n_transits"]
    if not per_mode:
        # No mode constrains gp ALONE, but complementary degeneracies can
        # still leave the joint Fisher full rank. That joint number is the
        # answer, so report it and carry on: stopping here would also discard
        # the structure panel, the constraint forecast and the summary figure.
        if not np.isfinite(comb):
            verdict_slot.warning(f"{glabel} is unconstrained in this run, by any mode and by the modes combined.")
            return False
        if comb > target:
            # the joint Fisher is the answer here; warn only when it misses
            # the target, like both sibling branches
            verdict_slot.warning(
                f"No single mode constrains {glabel}. Combined modes: "
                f"±{comb:.3g}{usp} at {view.tsig:g}σ in {ntr} "
                f"{view.event}{'s' if ntr > 1 else ''} (target ±{target:g}{usp})."
            )
    else:
        bk = min(per_mode, key=per_mode.__getitem__)
        bs = per_mode[bk]
        verdict = (
            f"**{ins.MODES[bk]['label']}**: ±{bs:.3g}{usp} at "
            f"{view.tsig:g}σ in {ntr} {view.event}{'s' if ntr > 1 else ''} "
            f"(target ±{target:g}{usp})."
        )
        if bs <= target:
            pass  # target met: nothing to add
        elif np.isfinite(comb) and comb <= target:
            verdict_slot.warning(verdict + f"  Combined modes: ±{comb:.3g}{usp}.")
        else:
            best_r = next(r for r in view.usable_jac if r["mode_key"] == bk)
            tt = fisher_mod.transits_to_target(best_r, view.fisher_names, gp, target / view.tsig, co_eval=view.co_eval)
            if tt["reachable"]:
                verdict_slot.warning(verdict + f"  {tt['n']} {view.event}s reach it.")
            elif np.isinf(tt["sig_inf"]):
                verdict_slot.warning(verdict + f"  {glabel} is unconstrained at any {view.event} count.")
            elif detect.has_floor(best_r) and not np.isnan(tt["sig_inf"]):
                verdict_slot.warning(verdict + f"  Floor caps it at ±{view.tsig * tt['sig_inf']:.3g}{usp}.")
            else:
                verdict_slot.warning(verdict + f"  >{detect.N_TRANSITS_CAP} {view.event}s (scan limit).")
    return True


def render_verdict(view: RunView, verdict_slot: Slot) -> bool:
    """Fill the shortfall warning: nothing when the target is met, a warning when a valid result misses it.

    Returns False when the run left nothing further to draw: the page ends there.
    """
    if view.goal == "detect":
        _detect_verdict(view, verdict_slot)
        return True
    return _constrain_verdict(view, verdict_slot)


def render_structure_panel(view: RunView) -> None:
    """Draw the T-P and mixing-ratio figure with its downloads."""
    with st.expander("Physical structure (T-P profile, mixing ratios)"):
        # ONE two-panel figure (plotting.build_structure_figure, pure and
        # importable without streamlit, so the threaded regression test
        # exercises this exact code). No figure-settings block: the panels use
        # the module's default windows, and the shared pressure axis is the
        # figure's own (sharey), not something a widget can desynchronize.
        p_arr = np.asarray(view.model["p_bar"], dtype=float)
        T_arr = np.asarray(view.model["T"], dtype=float)
        ymix = np.asarray(view.model["ymix"], dtype=float)
        ysp = [str(s) for s in np.asarray(view.model["ymix_species"])]
        if ymix.ndim != 2 or ymix.shape[0] != p_arr.size:
            raise ValueError(
                f"ymix shape {ymix.shape} does not match the pressure grid "
                f"({p_arr.size} layers): the stored model is inconsistent"
            )
        if len(ysp) != ymix.shape[1]:
            raise ValueError(
                f"ymix has {ymix.shape[1]} columns but {len(ysp)} species names: the stored model is inconsistent"
            )
        # Select the RT molecules BY NAME: ymix is the full network state
        # while model["mols"] is the RT subset, so zipping them
        # positionally reads the wrong species entirely. Ordered by peak
        # abundance so the legend reads top-down.
        # engine_config.MOLECULES maps the RT token to the VULCAN species
        # name; they differ (the tool's OCS is VULCAN's COS)
        from jwst_tool import engine_config as _ec

        want = {
            _ec.MOLECULES[str(m)]["vulcan"]: str(m)
            for m in np.asarray(view.model["mols"]).tolist()
            if str(m) in _ec.MOLECULES
        }
        cols = [(want[s], ymix[:, i]) for i, s in enumerate(ysp) if s in want]
        cols.sort(key=lambda kv: -float(np.nanmax(kv[1])))
        # the page caveats ride the figure too: a downloaded PNG carries
        # neither the st.warning above nor the CSV header
        fig3 = plotting.build_structure_figure(p_arr, T_arr, cols, caveats=view.caveats)
        struct_png = fig_bytes(fig3, "png", tight=False)
        show_fig(fig3, struct_png)

        if view.emission:
            if float(T_arr.max()) > 2000.0:
                st.warning(f"Layers exceed 2000 K (max {T_arr.max():.0f} K); ultra-hot opacities not modeled.")
        else:
            # transmission probes p <~ 0.1 bar; a hot deep adiabat below that is
            # invisible to the chord geometry and must not trip the warning
            probe = p_arr <= 0.1
            if probe.any() and float(T_arr[probe].max()) > 2000.0:
                st.warning("Photosphere >2000 K; ultra-hot opacities not modeled.")

        tp_df = pd.DataFrame({"p_bar": p_arr, "T_K": T_arr})
        vmr_df = pd.DataFrame({"p_bar": p_arr} | dict(cols))
        s1, s2, s3 = st.columns(3)
        s1.download_button(
            "Figure (PNG)",
            struct_png,
            f"{view.fname_base}_structure.png",
            "image/png",
            key=K("dl_struct_png"),
            on_click="ignore",
        )
        s2.download_button(
            "T-P values (CSV)",
            view.csv(tp_df),
            f"{view.fname_base}_tp_profile.csv",
            "text/csv",
            key=K("dl_tp_csv"),
            on_click="ignore",
        )
        s3.download_button(
            "Mixing ratios (CSV)",
            view.csv(vmr_df),
            f"{view.fname_base}_mixing_ratios.csv",
            "text/csv",
            key=K("dl_vmr_csv"),
            on_click="ignore",
        )
