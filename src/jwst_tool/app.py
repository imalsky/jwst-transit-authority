"""JWST exoplanet observation planner -- Streamlit GUI.

Method and contract: docs/gui_and_sharing.md, section `app`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from jwst_tool import (
    gui_forecast,
    gui_results,
    gui_run,
    gui_sidebar,
    gui_sources,
    gui_summary,
    posteriors,
    summary_figure,
)

TOOL_DIR = Path(__file__).resolve().parent

# House figure style: science.mplstyle (the validation figures use the same
# file) plus summary_figure's white-face overrides, so a downloaded figure
# stays white on any Streamlit theme. ONE definition of the overrides, applied globally
# here and per-figure there, so an in-app figure and a headless render match.
# Data colors/markers stay the fixed per-mode palette in
# instruments.MODE_COLOR / MODE_MARKER (no series relies on color alone).
plt.style.use(str(TOOL_DIR / "science.mplstyle"))
plt.rcParams.update(summary_figure._STYLE_OVERRIDES)  # type: ignore[arg-type]  # the stub lists only literal keys


# initial_sidebar_state: the default ("auto") starts the sidebar COLLAPSED
# on a narrow viewport -- the huggingface.co iframe is one -- hiding every
# input behind a small toggle, which reads as "half the tool is missing".
st.set_page_config(page_title="JWST Transit Authority", layout="wide", initial_sidebar_state="expanded")

# Header: short orientation, no acknowledgment gate
st.title("How to use this tool")
st.subheader("Warning: This tool is in beta mode. That means if you find a bug, you beta email isaacmalsky@gmail.com")
st.markdown(
    "- Set up the star, planet, and atmosphere conditions\n"
    r"- Pick a science goal (e.g. detect SO$_2$ at $3\sigma$)"
    "\n"
    "- Select the instrument modes, noise assumptions, analysis $R$, etc. "
    'You can combine multiple modes in "Parameter constraint forecast" '
    'after the run, then pick that set under "Modes" above the figure'
    "\n\n"
    "The tool computes a forward spectrum and a Pandeia noise forecast and "
    "reports how many transits/eclipses are needed for the science target. "
    "The detection values are template S/N estimates and the parameter "
    "constraints are local Fisher estimates."
)

# The Run row renders HERE (above the explainers). Its widgets depend on
# sidebar state that is read further down, so the slot is reserved now and
# filled once those values exist. The staleness warning is filled the same
# way: it belongs with the Run button it tells the user to press, not
# further down beside the results it qualifies.
_run_slot = st.container()
_stale_slot = st.container()
# The shortfall verdict is the ANSWER -- which mode, what significance,
# against the target -- so it sits above the explainers too, not below them
# with the figure it summarizes. Filled only when the target is missed.
_verdict_slot = st.container()

# Mirrors the README "Validation" section; nothing here runs.
with st.expander("Validation"):
    st.markdown(
        "This tool includes test suites, as well as other validation checks. "
        "The suites run in CI for each repository: "
        "[VULCAN3](https://github.com/imalsky/VULCAN3), "
        "[vulcan-forward](https://github.com/imalsky/vulcan-forward), and "
        "[jwst-transit-authority](https://github.com/imalsky/jwst-transit-authority). "
        "For end-to-end tests, see the set of validation figures that I've "
        "created [here](https://github.com/imalsky/jwst-transit-authority/tree/main/"
        "validation/figures). This includes trying to recreate the results "
        "of [Tsai et al. 2023](https://doi.org/10.5281/zenodo.7542781), the "
        "[JWST ERS carbon dioxide paper](https://doi.org/10.5281/zenodo."
        "6959427), and VULCAN 2.0 and petitRADTRANS on identical inputs."
    )

_missing_req, _ksrc, _ksrc_err = gui_sources.data_status()
with st.expander("Data" + (f" ({len(_missing_req)} required item(s) missing)" if _missing_req else "")):
    _opacity_slot = st.container()


# Sidebar controls
# Reset = bump a nonce that namespaces EVERY widget key: session_state.clear()
# alone does not reset keyless widgets.
st.session_state.setdefault("reset_nonce", 0)

gui_sidebar.apply_pending_archive_fill()
gui_sidebar.keep_hidden_engine_state()
target, atm, goal, obs, solver = gui_sidebar.render_sidebar(_ksrc, _ksrc_err)

request = gui_run.build_request(target, atm, goal, obs, solver)
gui_sources.render_source_table(_opacity_slot, _ksrc, _ksrc_err, target, atm)
if gui_run.render_run_row(_run_slot, request, atm, obs):
    gui_run.handle_run_click(request, target, atm, goal, obs)

# Render order: staleness/failure notices, the VERDICT, then physical
# structure, the constraint forecast, and the summary figure.
if "out" not in st.session_state:
    st.stop()

view = gui_results.result_view()
gui_results.render_stale_notice(view, _stale_slot, request.params, request.run_sig)
excluded = gui_results.render_exclusions(view)
if not view.results:
    st.stop()
mode_notes = gui_results.render_mode_notes(view, excluded)
combos = gui_forecast.combo_records(view)
if not gui_results.render_verdict(view, _verdict_slot):
    st.stop()
# Mock-observation layer: one seeded N(0, sigma_i) draw per
# bin ON TOP of the binned noiseless model, generated AFTER the forward
# model and the noise model (posteriors.mock_realization). The LIVE widget
# state drives it (not the stored run meta), so editing the seed redraws
# without recomputing anything; the seed is displayed and reproducible.
# HARD RULE: nothing from this layer enters detection/Fisher scores, caches,
# or the result CSVs -- only the clearly-named mock download below, and the
# mock_recovery overlay on the posterior panels, which IS fitted to it.
mock = posteriors.mock_realization(view.results, int(obs.seed)) if obs.show_noise else None
for caveat in view.caveats:
    st.warning(caveat)
gui_results.render_structure_panel(view)
post_slot = gui_forecast.render_forecast_panel(view, combos, mode_notes)
post_panels = gui_forecast.forecast_panels(view, combos, mock, post_slot)
gui_summary.render_summary(view, combos, post_panels, mock, int(obs.seed))
