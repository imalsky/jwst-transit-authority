# CLAUDE.md — vulcan-jwst-tool (jwst-transit-authority)

JWST Transit Authority forecasts how well a JWST transit or eclipse observation detects
a molecule or constrains metallicity and C/O, from a VULCAN (default) or PICASO chemistry
column, ExoJAX radiative transfer and Pandeia noise. Proposers use it through the public
Hugging Face Space or a local install (Streamlit GUI, Python API, `jwst-tool` CLI). In
the chain vulcan3 <- vulcan-forward <- {vulcan-retrieval, jwst-transit-authority} it
depends on vulcan-forward (and through it vulcan3), never on vulcan-retrieval.

## Docs

Four root docs, one job each: README.md (hand-written by the maintainer; never edit it
unasked), notes.md (what is true and why, by topic section: read the section a task
needs; put a number under the topic it justifies, never as a dated entry), todo.md (open
work: read it first; open or close items in the same pass) and CLAUDE.md (directions).
The last three are tracked and public: no personal names except cited authors, no
emails, IDs, account or billing details, keys or chat quotes. Refer to a notes section by
its heading name, never a number. No other .md file; the two that exist are generated or
required elsewhere (`validation/parity/outputs/REPORT.md`, the Space card
`deploy/hf-space/README.md`). Policy: the project-root CLAUDE.md; format and where each
fact goes: the `doc-upkeep` skill.

## Setup and commands

- Names: dist `jwst-transit-authority` (GitHub repo and Space
  `imalsky/jwst-transit-authority`), import `jwst_tool`, console script `jwst-tool`, env
  vars `JWST_TOOL_*`. The data repo `imalsky/vulcan-jwst-tool-data` and the bucket
  `imalsky/jwst-tool-cache` keep their old names.
- Install: editable member of the root uv workspace; standalone,
  `pip install --no-deps -e .`. Rebuild the editable metadata after a version bump
  (`provenance._versions` reads it).
- Import order is load-bearing: `engine_config` -> `vulcan_forward.vulcan_chem` ->
  exojax (`vulcan_chem` raises if exojax came first; `tests/conftest.py` takes the order).
- Paths resolve in `instruments.py`, loud on failure: `data/` holds inputs
  (`JWST_TOOL_DATA_DIR`), `output/` generated caches (`JWST_TOOL_OUTPUT_DIR`). Engine data
  comes from `$VULCAN_FORWARD_DATA`.
- Pandeia runs in a subprocess at `JWST_TOOL_PANDEIA_PYTHON` (+
  `JWST_TOOL_PANDEIA_REFDATA`); on daw the root `.venv`, exported in the root `.env`.
  `noise.run_pandeia` refuses if it is unset. One backend: the matched engine + refdata +
  PSF triple `instruments.py` names; mixed sets are refused.
- PICASO runs `picaso_worker.py` in its own environment at `JWST_TOOL_PICASO_PYTHON`,
  built from `deploy/picaso-conda-lock.txt` + `deploy/requirements-picaso-lock.txt`; its
  reference data is `data/picaso_refdata-4.1` (`JWST_TOOL_PICASO_REFDATA`).
- Commands: `jwst-tool` launches the GUI after preflight checks; `jwst-tool fetch
  [--picaso]` downloads missing data, including the gitignored machine-local PHOENIX grid
  `data/cdbs/grid/phoenix`; `jwst-tool data [--deep]` prints the data report.
- Normal suite: `python -m pytest tests -q -n 6` (give `-n` explicitly; on daw `unset
  PYTHONSAFEPATH` first). `tests/conftest.py` sets `JAX_PLATFORMS=cpu` and points an unset
  `JWST_TOOL_OUTPUT_DIR` at a throwaway directory, so it never writes into `output/`.
- Slow tier (`JWST_TOOL_RUN_SLOW=1`, real `output/` caches): the transmission FD/FD and
  emission AD/FD closures at the config cadence and at cadence 1, the e2e of both engines
  against their fixtures in `tests/live/data/`, the TOI-7169 b dlnCO escalation, and the
  PICASO live tests (skipped without the PICASO env). It runs for hours (notes "Current
  state"): ask before starting it.
- Lint: `ruff check .` must pass (rules in pyproject). After a push, check CI as the root
  CLAUDE.md says (`gh run list --repo imalsky/jwst-transit-authority --limit 1`).

## Rules

### Fail loud and parameters

- No behavior-changing fallbacks, no silent substitution; unknown values raise.
  `canonical_params` rejects unknown keys; `_PARAM_KEYS_READ` is pinned by an
  AST-walking test. Public noise APIs validate their inputs and raise.
- Never auto-select a proxy value ("nearest X"); the archive fill never writes the
  UV-spectrum menu.

### Composition, C/O and networks

- Composition is structural: `met_x_solar` + `co_ratio` re-initialize the equilibrium
  seed. Pin `use_vm_mol` explicitly (default False); never inherit it.
- `co_ratio` is bounded by `forward.co_bounds(network, use_photo)`, enforced in
  `canonical_params` and so shared by the API and `share_config`. Two gate axes, network
  and photolysis: never add temperature or Kzz, never hard-code an excluded C/O band,
  never call the permissive bounds a ceiling, and never raise the sncho photolysis-on
  0.99 without a science decision (notes "Composition, C/O bounds and networks"). A
  refusal must cost no solve.
- `sncho2025` is EXPERIMENTAL and not the default.
- Condensation requires `sncho`. It is detection-only and API-only (the GUI pins it off;
  `share_config` refuses files that enable it), refused with any Jacobian, photolysis off
  or moldiff off. Settling, escape and boundary fluxes are removed: their four keys stay
  in the payload pinned off and enabling values are refused.

### Certificates

- Every certified stage records branch, longdydt, aflux_change and the controlling
  `species@z` cell, and a failure names them (`forward.check_converged`).
- Every certified column passes `forward.check_elements` (layer-median X/H within
  `ELEMENT_TOL` of the build column) and `forward.check_unmodeled` (an absorber with no
  k-table above `UNMODELED_VMR_REFUSE` is refused before any spectrum; above
  `UNMODELED_VMR_WARN` the run warns). Values and basis: notes "Composition, C/O bounds
  and networks", "Convergence certificate and element gate".
- `yconv_min` is canonical and may only tighten below 0.1: cold baseline solves exit on
  the loose branch, so `yconv_cri` alone cannot tighten them.

### Jacobians, Fisher and statistics

- Fisher Jacobians are certified FD (central, or one-sided toward lower C/O via
  `fd_stencil`; h-vs-2h gate, Richardson) or warm-jvp AD. Never report an uncertified
  derivative.
- AD chemistry rows are photolysis-on only. Cloud-deck rows and `lnR0` are RT-only jvps
  and stay available photolysis-off. `lnR0` is the engine's radius change at fixed planet
  mass in both modes: pass it to the engine, never multiply a depth by e^(2 lnR0).
  Refuse AD dlnCO unless the oxygen-reservoir margin exceeds `CO_BZ_MIN_AD` on both the
  build and the converged column.
- AD rows: one warm jvp per row through the engine's `converged_y_jvp`, direction
  `FD_STEPS[name] * e_i`, row divided by the step; never a vmap over tangent directions,
  never a plain `jax.jvp` read at the column's certificate, never a step floor. The first
  attempt runs on an `AD_BUILD_OVERRIDES` build (derivative builds only; cold solves and
  FD stencil points keep `DT_MAX_S`) at the config cadence; a row that does not certify
  is re-solved at cadence 1 with `AD_RETRY_COUNT_MAX` and flagged in `photo_escalated`.
- Photolysis-cadence escalation is per FD row, never per stencil point, and only a
  `NotCertified` triggers it (notes "Jacobians and Fisher").
- Composition FD stencil points keep the baseline's geometry anchor by design: do not
  "fix" it from the code shape; settle it by measurement (notes "Jacobians and Fisher").
- Fisher inversion stays rank-aware (`fisher._marg_sigmas`, whitened
  eigendecomposition), with per-segment depth offsets including segment 0.
- The C/O forecast curve is the density per d ln(C/O) on a log C/O axis
  (`posteriors.ln_gaussian_curve`; no 1/(C/O) Jacobian). Quote every finite C/O
  uncertainty in log10(C/O) dex (`fisher.format_co_width`, the "log10 C/O" table row);
  no mapped physical C/O range anywhere; never show `fisher.display_sigma`'s dlnCO value.
  Share files carry the run's `elemental_abundances`.
- Sigma labels say "conditional template S/N": never a retrieval claim, never
  "PandExo-identical".

### Measurement operator and noise

- One measurement operator (`binning.py`) bins noise, model and Jacobians; cell averages
  use the exact `_pl_antideriv`.
- The LSF is the count-density-weighted (per d ln lambda) Gaussian, applied to depth,
  removed-molecule and Jacobian rows, of width R_refdata / `instruments.LSF_WIDTH`
  fitted per mode to a Pandeia impulse (`run_parity.py --impulse`). Re-measure it on any
  refdata or PSF change, with an impulse SED dense enough to resolve the broadened line
  (notes "Measurement operator, LSF, noise, ramps, registry"). A missing dispersion file
  is an error, never a no-blur.
- Noise floor: `sigma = max(sigma_random, floor)` on final bins; never quadrature, never
  rescaled or averaged down by transits.
- Pin every TSO instrument config explicitly (readout pattern, extraction strategy,
  background); never add a mode without all of them. ngroup search: floors are Pandeia
  `mingroups`, "saturated" only on measurement, the largest measured-safe count,
  maximality proven by bracket.
- Per-mode advisories (`result["warnings"]`) are reported, never ranked on or applied.

### Opacity, emission and data sources

- Correlated-k over the published ExoMolOP k-tables is the only opacity path; the
  removed line-by-line and Mie keys are refused by name. `run_model` verifies the
  engine's echo of every RT knob: never drop an echo check.
- Species with no k-table are not offered (`_NO_EXOMOLOP_TABLE`). Never widen
  `EXTRA_MOLECULES_DEFAULT` without a measured ppm impact; widening either molecule list
  is a deploy change (re-run `upload_data.sh`).
- Every "Data sources" row carries a resolvable DOI and page (`test_app_smoke` fails on a
  blank or placeholder). Curate k-table DOIs in vulcan-forward's
  `fetch_exomolop._DATASET_DOI`, never in the GUI; `ktable_sources` keeps the header value
  as `header_doi`. A new `SFLUX_CHOICES` entry needs a `planets.SFLUX_SOURCES` entry.
- Emission: Guillot default profile, flux-weighted tau-bottom gate, and the engine's
  `ibased_linsap` solver, never `ibased` (notes "Emission"). Eclipse depths come from the
  engine's `eclipse_flux_tau`, never `emission_flux_tau`; `emission_radius` stays the
  single (R_em/R_star)^2 anchor. Guillot `Tint` and `Tirr` defaults have one definition
  each (`planets.default_tint`, the registry) for `canonical_params` and the widget.

### PICASO engine

- PICASO is the second engine (`chem_provider='picaso'`); VULCAN payloads, keys and
  fixture never move for it. PICASO-only keys (`rfacv`, `picaso_quench`, `picaso_stack`)
  exist only in PICASO payloads.
- The climate is solved in equilibrium and the quench applied after: never put the
  quench in picaso's climate loop, never re-normalize the TOA flux gate to admit it. The
  coupling is one-way: never call it self-consistent.
- The composition is imposed through the injected `chemeq_solver`, never mapped onto
  picaso's (mh, C/O); every layer is solved cold.
- FD only; the lnKzz and climate T_int rows are refused; metallicity is capped at
  `PICASO_MET_RANGE`; VULCAN-only settings are refused, never dropped.
  `picaso_engine.certify` gates every result and re-runs on every cache load.
- picaso, photochem and virga are imported only by `picaso_worker.py` (which imports
  nothing from the tool) and pinned only in the picaso locks (test-pinned).
  `datacheck.PICASO_REFDATA_MD5` + `PICASO_CK_MD5` is the one manifest that fetch,
  datacheck and the deploy scripts read.

### Caches and versions

- `forward._VERSION` busts `model_cache`: bump it whenever the physics, the canonical key
  set or `forward.DT_MAX_S` changes.
- Every engine pin bump (vulcan3 or vulcan-forward in `deploy/pins.env`, the Dockerfile
  and CI) bumps `forward._VERSION` too: the key carries no engine version, so the Space
  would serve columns the old engine computed.
- A `_VERSION` bump is incomplete until both e2e fixtures are regenerated
  (`test_e2e_fixture_sanity` fails until they agree; never skip it). After a bump that
  moves a spectrum, regenerate the six `validation/data/atmos_*.npz`
  (`validation/scripts/inputs/atmos_cases.py`), the pRT references
  (`inputs/prt_reference.py`) and the figures (`make_figures.py`); no test catches a
  stale one (notes "Opacity and radiative transfer").
- `noise.WORKER_VERSION` (+ the backend fingerprint) busts `noise_cache`: bump it
  whenever `pandeia_worker.py` output changes. `adjoint_diag._ADJ_VERSION` keys its own
  cache. `picaso_engine.WORKER_VERSION` keys `output/picaso_cache`; PICASO payloads carry
  `forward.PICASO_STACK`, re-derived like `version`.
- `$JWST_TOOL_DATA_REVISION` keys `model_cache` to the science data, which no parameter
  and no `_VERSION` tracks. Unset omits the key, so local keys never move. Data updated
  under a fixed filename must change the revision. `exomolop`'s `grid_sha256` is not a
  substitute (notes "Opacity and radiative transfer").

### GUI and copy

- No explanatory prose in the GUI: a short label, and a `help=` tooltip only when the
  effect is not obvious. Caveats belong in README or notes; precise statistical terms
  stay. The GUI shows no install inventory; `jwst-tool data` is the only place for that.
- All figure work goes through `plotting.py` under `render_lock`; figures are pure
  builders, never re-implemented in `app.py` (AST-checked). Legends never cover data; no
  legend title; axis ranges are typed min/max pairs, never sliders; only the summary
  figure has axis controls.
- The structure figure (`build_structure_figure`) shows pressure tick labels on both
  panels but the "pressure (bar)" label once, at the far left (test-pinned;
  `STRUCT_AXES_RECT` leaves room for it). T-P is black; mixing ratios use `VMR_COLORS`
  (the same colour per species as the note's figures) and `VMR_EXTRA_COLORS`.
- The shortfall warning renders in `_verdict_slot` beside the Run button. Results below
  Validation, in order: physical structure, parameter constraint forecast, summary
  figure. Do not re-add "Mode details", "Run summary & configuration", or separate
  custom-mode-set / marginalized-forecast expanders.
- A curve fitted to the jitter realization is never called a forecast: its y axis is
  "relative density", the forecast's "relative forecast density" (test-pinned); no "(one
  draw)" wording. The C/O panel carries no " per d ln(C/O)" suffix. The gray cloud is
  transmission-only.
- The "Chemistry engine" selector opens Step 2. PICASO widgets carry a `picaso` key infix
  and every VULCAN key stays as it was; under PICASO the method is a disabled FD-only
  widget on its own key, so `K("jacm")` is never written; the hidden engine's widgets are
  written back to themselves across a switch. A cap is a refusal sentence, never a moved
  widget ceiling (the C/O widget spans `CO_MIN` to `CO_MAX_PHOTO_OFF`).
- Widget keys are a shipped contract: labels may move, keys may not. A widget whose key
  is always pre-set in session_state takes no `index=`/`value=` default. Keep
  `st.set_page_config(..., initial_sidebar_state="expanded")`.
- A loaded configuration applies only through the Populate button's callback
  (`_populate_config`); the success line says only "Populated". The staleness guard
  compares the full `_run_sig` (goal + observation minus the display-only `show_noise`,
  `seed`, `combos`).
- Live readouts are fixed-width; progress bars keep the "[fwd]/[adj] PROG <frac>
  <label>" stdout protocol. A Run that returns nothing pops the previous result. A failed
  forward model shows only its exception sentence; the log tail sits behind a collapsed
  "Technical details".
- Copy: ">N (scan limit)", never "never"; "stops and shows an error"; "setting", not
  "knob"; eclipse vocabulary in emission; internal tokens stay out of labels.

### Figures, reports and deploy

- Committed figures live in `validation/figures/` and `validation/parity/figs/` and are
  made only by `validation/scripts` through `validation/figstyle.py`; each PNG embeds its
  generator (`test_validation_figures` compares the parsed code). The app and figstyle
  both load `src/jwst_tool/science.mplstyle`.
- `REPORT.md` stays the short summary `make_report.py` generates; never hand-edit it.
- Never deploy or touch the Space unasked. The runbook (pins, upload, verification) is
  notes "Hugging Face Space". `test_deploy_pins.py` pins `deploy/pins.env` against the
  Dockerfile.

## Where things live

- Current versions, cache keys, suite counts and daw environments: notes "Current state";
  version history: notes "Cache keys and version history".
- The Space: `deploy/hf-space/`; sibling pins: `deploy/pins.env` (read by CI and the
  Dockerfile). PandExo parity: `validation/parity/` and notes "Parity testing".
- Before proposing a fix or a new approach: notes "Accepted limitations", "Known and
  deliberately untracked", "Do not retry".
