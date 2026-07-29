"""JWST instrument-selection tool: VULCAN-JAX chemistry -> ExoJax RT -> Pandeia noise.

A live-forward-model instrument selector (PandExo-style GUI): pick a science
goal (detect molecule X on a WASP-39b-like planet), and the tool runs the
live VULCAN-JAX + ExoJax forward model locally, builds parameter Jacobians
as central finite differences of independently converged solves
by default (each row ships its own step-size consistency bound -- see
forward.py's FD block) with an opt-in warm-jvp AD path for every row,
simulates each JWST instrument mode's transit-depth precision with the real
STScI Pandeia ETC engine, and ranks the modes by conditional
matched-template S/N (not a retrieval detection significance -- see
detect.py) plus rank-aware Fisher forecasts. adjoint_diag.py adds
reverse-mode adjoint diagnostics (per-reaction dL/dlnk, per-layer dL/dT)
with a scope audit and full numerical certification.

Lives in src/jwst_tool/ (dist: vulcan-jwst-tool); the shared
forward-model modules come from the shared vulcan-forward package
(vulcan_forward.*).

Entry point: the console script ``jwst-tool``, or
``streamlit run src/jwst_tool/app.py`` from the repo root.
"""
from jwst_tool._version import __version__  # noqa: F401
