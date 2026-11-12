# The GUI, its figures and shared configurations

## `app`

`app.py` is the Streamlit script and reads as the page: five numbered sidebar steps (0 Configuration, 1 Target, 2
Atmosphere, 3 Science goal, 4 Observation) plus "More settings", then the shortfall warning when a target is missed,
physical structure, the constraint forecast and the summary figure. Widget keys never change (tests and cached session
state rely on them); only placement, labels and help text may move.

`gui_sidebar` and `gui_atmosphere` draw the steps and return `gui_inputs` records; `gui_run` validates a request and
launches the forward model and Pandeia runs; `gui_results`, `gui_forecast`, `gui_summary` and `gui_sources` draw the page.

## `plotting`

Pure figure builders and the process-wide render lock. Importable without Streamlit: every figure the GUI shows is built
here from plain arrays, so the code the app renders is the code the tests render.

### Why the lock

Matplotlib's mathtext parser is process-global state. `Figure.draw()` holds matplotlib's own `Figure._render_lock`, but
`Figure.tight_layout()` runs the layout engine outside it, and layout measures every tick label, which is mathtext on a
log axis. Two Streamlit sessions, each its own script-runner thread, can then be inside the shared parser at once, and
the loser raises `ValueError: ParseException` out of `tight_layout`. Bare `Figure` objects do not help: the race is in
the parser, not in pyplot's figure registry.

The rule: every figure lifecycle (construct, lay out, draw, export, close) happens inside `render_lock`. Do not narrow
the span to the export: the layout call is the one that crashed. `build_*` acquire the lock themselves and it is
reentrant, so a caller wrapping a build and an export in one `with render_lock:` is the intended pattern. The lock is
in-process; the worker subprocesses render nothing.

`tests/unit/test_plotting.py` pins this with a threaded check that every builder blocks while another thread holds the
lock, and a structural check that no unlocked layout or export call survives in the GUI modules.

## `summary_figure`

One-figure proposal summary, taking plain dicts and arrays so it renders without Streamlit, an engine or a noise backend.

- Left, golden-ratio wide: the model spectrum with each mode's simulated points, optionally one seeded mock noise
  realization. Each legend entry carries the mode's expected performance: conditional template S/N for a detection goal,
  an expected ± for a constraint goal.
- Right, up to three square panels: one 1D marginalized Fisher-Gaussian curve per parameter, a linearized Cramér-Rao
  forecast and never a sampled posterior. An unconstrained direction renders as an empty panel, never a finite curve or
  a caption: no figure this module draws carries prose.

`compose_summary_figure` takes:

- `spectrum`: `wl_um` (um), `depth_ppm` (ppm), `depth_label`, `model_label`, `points=[dict(label, color, marker, wl_um,
  depth_ppm, sigma_ppm), ...]`, optional `wl_range` and `depth_range` axis windows (None to auto-fit), `x_log` and
  `y_log` scales, and an optional comparison curve `depth2_ppm` with `depth2_label`. Per-mode performance belongs in
  each point's label.
- `posterior_panels`: up to three dicts, one per parameter: `axis_label`, `axis_unit`, `density_label`, `center` and
  `curves=[dict(label, theta, pdf, color, ls, lw), ...]`. A curve may add `mu` and `sigma` (display units: the legend
  quotes the width and the automatic window is sized from them), `width_text` (a caller-built width string, as for
  C/O), and `curve_family="ln_gaussian"` with `sigma_ln` for a multiplicative width on a log axis.
- `panel_xlims`: one `(lo, hi)` in the panel's own units or None per panel, a window on the existing curves, never a
  resampling. Shorter than the panel list is allowed.
- `caveats`: the run caveats the page shows.

Style: `science.mplstyle` plus the white-face overrides this module owns (`_STYLE_OVERRIDES`, which the GUI applies
globally).

## `share_config`

"Download configuration (JSON)" writes what `build_share` assembles: the canonical model parameters plus the goal and
observation settings and, for uploaded tables, the table content. That is every input the GUI takes, so the same tool
version reproduces a GUI run from the file. Code revisions, data checksums, cache schemas, Pandeia and PandExo identity
and the seed are recorded in `provenance`, informational on load so old configurations stay portable.

`widget_state` is the inverse. It validates through `run_params.canonical_params`, restores from the output, then checks
every widget-only number against its widget's range, and raises on anything invalid or unrestorable, so the caller
applies the whole mapping or nothing. The range check is load-bearing: Streamlit discards an out-of-range session-state
value silently and the widget falls back to its default, so an unchecked restore would run a different model than the
file describes. Condensation is API-only and refused on load. Both engines round-trip, each onto its own widget keys.
A file saved before v64 whose model top lies above an undeclared T-P table, or whose custom planet runs on a shipped
table, is refused with the sentence `canonical_params` gives.
