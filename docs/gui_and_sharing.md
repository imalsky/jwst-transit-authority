# The GUI, its figures and shared configurations

## `app`

JWST exoplanet observation planner -- Streamlit GUI.

Launch via the console script ``jwst-tool`` (installed with jwst-transit-authority), or
directly:  streamlit run src/jwst_tool/app.py  (from the repo root).

Pipeline per run: VULCAN-JAX photochemistry (or PICASO climate + equilibrium
chemistry) -> ExoJAX
transmission/emission spectrum (local subprocess, disk-cached) -> Pandeia ETC
noise per instrument mode (subprocess in its own environment, disk-cached) ->
science-goal scoring per mode.
Two goal types: DETECT a molecule (conditional template S/N) or CONSTRAIN a
parameter (Fisher forecast from consistency-checked Jacobians, vs a target
uncertainty). Planets beyond WASP-39b come from the registry in planets.py (or
a fully custom system).

Layout: five numbered sidebar steps (0 Configuration, 1 Target, 2 Atmosphere,
3 Science goal, 4 Observation) plus one "More settings" group; the result
page leads with a shortfall warning when the target is missed, then physical
structure, the constraint forecast, and the summary figure. Widget KEYS must
never change (tests and cached session state rely on them); only placement,
labels, and help text may move. A downloaded configuration JSON is a
complete, shareable run setup; "Load a configuration" (share_config.py)
restores it into the widget state before any widget instantiates.

## `plotting`

Pure figure builders + the process-wide render lock.

Importable WITHOUT streamlit: every figure the GUI shows is built here from
plain arrays, so the same code the app renders is the code the tests render.

WHY THE LOCK (the ParseException crash on the Space)
----------------------------------------------------
Matplotlib's mathtext parser is process-global state. ``Figure.draw()``
holds matplotlib's own ``Figure._render_lock``, but ``Figure.tight_layout()``
runs the layout engine OUTSIDE it, and layout measures every tick label --
mathtext on a log axis. Two Streamlit sessions (each its own script-runner
thread) laying out figures at once can therefore be inside the shared parser
concurrently; the loser raises ``ValueError: ParseException`` out of
``tight_layout``. Bare ``Figure`` objects do NOT help -- the race is in the
shared parser, not in pyplot's figure registry.

RULE: every figure lifecycle -- construct, lay out, draw, export, close --
happens inside ``render_lock``. Do not lay out, save, or draw a figure
outside it, and do not narrow the span to just the export: the layout call is
the one that crashed. ``build_*`` acquire it themselves (it is reentrant), so
a caller that wraps a build + export in one ``with render_lock:`` block is
correct and is the intended pattern. The lock is process-wide and
in-process; it says nothing about other processes (the ETC/forward
subprocesses render nothing).

Pinned by tests/unit/test_plotting.py (a threaded check that every builder
blocks while another thread holds the lock + a structural check that no
unlocked layout/export call survives in app.py).

## `summary_figure`

One-figure proposal summary: spectra + marginalized forecast densities.

Pure matplotlib + numpy, importable and renderable without Streamlit (the GUI
builds the input dicts; tests render headless). The composition takes plain
dicts/arrays so it is unit-testable without any engine or noise backend.

One row of panels:

* LEFT (golden-ratio wide) -- can the hypothesis be tested, and which mode
  is best at what precision? The model spectrum with each mode's simulated
  data points (per-mode color + marker; optionally one seeded mock noise
  realization). Each legend entry carries that mode's expected performance
  number (the caller appends it to the point-series label: a conditional
  template S/N for a detection goal, an expected +/- for a constraint goal).
* RIGHT (up to three square panels) -- what would the measurement look
  like? One 1D marginalized Fisher-Gaussian forecast curve per parameter
  (linearized Cramer-Rao forecasts, never sampled retrieval posteriors). An
  unconstrained direction renders as an EMPTY panel, never a fake finite
  curve and never a caption on the axes: no figure this module draws
  carries prose.

House style: science.mplstyle (also the validation figures' style) plus
the white-face overrides this module owns (_STYLE_OVERRIDES, which the GUI
applies globally), so a standalone render matches the in-app figures.

## `share_config`

Shareable run configurations: build the download, restore the upload.

The GUI's "Download configuration (JSON)" button writes the dict `build_share`
assembles: the canonical model parameters (the same dictionary every result
stores as provenance) plus the science-goal and observation settings, and --
when the run uses uploaded tables -- the table content itself. That is every
input the GUI takes -- condensation is API-only, pinned off in the GUI and
REFUSED on load; settling, escape and the boundary-condition fluxes are
refused by ``run_params.canonical_params`` everywhere -- so the same tool version
reproduces a GUI run from the file alone. Exact code revisions, science-data
checksums, cache schemas, Pandeia/PandExo identity, and the random seed are
recorded in ``provenance``.
They remain informational on load so old configurations stay portable, but a
collaborator can compare them before claiming an exact reproduction.

`widget_state` is the inverse: it maps such a file (or a bare canonical-params
dict) onto Streamlit session-state widget keys. It validates through
`run_params.canonical_params` first and restores from its output, then checks
every widget-only numeric against its widget's range, and raises ValueError on
anything invalid or unrestorable -- the caller applies either the whole
mapping or nothing. The range check is load-bearing: Streamlit does not
reject an out-of-range session-state value, it silently discards it and the
widget falls back to its default, so an unchecked restore would run a
different model than the file describes.

Both chemistry engines (VULCAN-JAX kinetics and PICASO climate + equilibrium
chemistry) round-trip, each onto its own widget keys.

This lives outside app.py so the mapping is importable and unit-testable
without running Streamlit.

## `cli`

Console entry point: ``jwst-tool`` launches the Streamlit GUI.

``jwst-tool``              launch the GUI (preflight checks first, loud)
``jwst-tool fetch``        download every missing dataset with a public URL,
                           then print the short manual list (two STScI Box
                           downloads + the Pandeia conda environment)
``jwst-tool fetch --picaso``  also the optional PICASO engine's reference
                           data and climate tables (about 2.4 GB)
``jwst-tool data``         print the full data-availability report + remedies
``jwst-tool data --deep``  also probe the Pandeia and PICASO envs for versions
``jwst-tool archive-refresh``  refresh the shipped NASA Exoplanet Archive
                           snapshot (maintainer/release step; network)

The GUI launch is equivalent to ``streamlit run src/jwst_tool/app.py``, but
works from anywhere once the package is installed. Preflight checks catch the
two external requirements with actionable messages instead of a mid-run stack
trace: an importable ``vulcan3`` package and the Pandeia backend env
(``$JWST_TOOL_PANDEIA_PYTHON``). A missing PICASO env
(``$JWST_TOOL_PICASO_PYTHON``) only prints a note.
