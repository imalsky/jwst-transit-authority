# CLAUDE.md — JWST Transit Authority (jwst-transit-authority)

## Docs

Four root docs: README.md (hand-written by the maintainer; never edit it unasked),
notes.md, todo.md and CLAUDE.md. The last three are tracked and public: write them for
public reading (no personal names except cited authors, no account or billing details).
No other .md files; the two that exist are generated or required elsewhere
(`validation/parity/outputs/REPORT.md`, the Space card `deploy/hf-space/README.md`).
Follow the project-root CLAUDE.md for policy. Read todo.md first and open or close items
in the same pass. notes.md holds the evidence behind each rule here, by topic section;
read the section a task needs, and add a number under the topic it justifies, never as
a dated entry.

## Paths and environment

- Dist `jwst-transit-authority` (GitHub and Space `imalsky/jwst-transit-authority`),
  import `jwst_tool`, console script `jwst-tool`, env vars `JWST_TOOL_*`. The data repo
  `imalsky/vulcan-jwst-tool-data` and the bucket `imalsky/jwst-tool-cache` keep their
  names. Depends on `vulcan-forward`, NOT vulcan-retrieval. Local install: editable
  member of the root uv workspace; standalone, `pip install --no-deps -e .`
- Import order is load-bearing: engine_config -> vulcan_chem -> exojax (guard-enforced
  in `vulcan_forward.vulcan_chem`).
- Path roots resolve in `instruments.py`, loud on failure: `data/` = inputs
  (`JWST_TOOL_DATA_DIR`), `output/` = generated caches (`JWST_TOOL_OUTPUT_DIR`). Engine
  data via `$VULCAN_FORWARD_DATA`.
- Pandeia runs in a subprocess at `JWST_TOOL_PANDEIA_PYTHON` (+
  `JWST_TOOL_PANDEIA_REFDATA`); on daw that is the root `.venv` (export in the root
  `.env`). `noise.run_pandeia` refuses if unset. ONE backend, Pandeia 2026.7, a matched
  triple (engine + refdata + PSFs); mixed sets are refused.
- `data/cdbs/grid/phoenix` is a large machine-local grid (gitignored); obtain it with
  `jwst-tool fetch`.
- Normal suite: `python -m pytest tests -q -n 6` (give `-n` explicitly). `conftest.py`
  defaults `JAX_PLATFORMS=cpu` and points an unset `JWST_TOOL_OUTPUT_DIR` at a throwaway
  directory, so the normal suite never writes into `output/`. Slow tier
  (`JWST_TOOL_RUN_SLOW=1`): the transmission FD/FD and emission AD/FD closures at the
  config cadence and at cadence 1, the e2e of both engines vs
  `tests/live/data/w39b_v60_reference.npz` and `w39b_picaso_reference.npz`, the TOI-7169
  b dlnCO escalation, and the PICASO live tests (skipped without the PICASO env).
- The PICASO engine runs `picaso_worker.py` in its own environment at
  `JWST_TOOL_PICASO_PYTHON` (picaso 4.1 + photochem 0.9.2, numpy >= 2; built from
  `deploy/picaso-conda-lock.txt` + `deploy/requirements-picaso-lock.txt`). Reference data
  `data/picaso_refdata-4.1` (`JWST_TOOL_PICASO_REFDATA`, `jwst-tool fetch --picaso`); its
  manifest `datacheck.PICASO_REFDATA_MD5` + `PICASO_CK_MD5` is the one list fetch,
  datacheck and the deploy scripts read.
- Ruff: `ruff check .` must pass (rules pinned in pyproject).

## Version keys (cache busters)

- `forward._VERSION` busts model_cache: bump whenever the physics or the canonical key
  set changes. A bump is incomplete until both e2e fixtures are regenerated
  (`test_e2e_fixture_sanity` fails until they agree; never skip it). Changing
  `forward.DT_MAX_S` needs a bump.
- `noise.WORKER_VERSION` (+ the backend fingerprint) busts noise_cache: bump whenever
  `pandeia_worker.py` output changes. `adjoint_diag._ADJ_VERSION` keys its own cache.
  `picaso_engine.WORKER_VERSION` keys `output/picaso_cache`; PICASO payloads carry
  `forward.PICASO_STACK`, re-derived like `version`.
- `$JWST_TOOL_DATA_REVISION` keys model_cache to the science data, which no parameter
  and no `_VERSION` tracks. Unset omits the key, so local keys never move. Data updated
  under a fixed filename must change the revision. `exomolop`'s `grid_sha256` is NOT a
  substitute (notes: Opacity and radiative transfer).
- After a `_VERSION` bump that moves a spectrum, regenerate the validation npz, the pRT
  references and the figures (notes: Opacity and radiative transfer).
- Version history: notes, Cache keys and version history.

## Invariants (do not regress)

- Fail fast and loud: no behavior-changing fallbacks, no silent substitution, unknown
  values raise. `canonical_params` REJECTS unknown keys; `_PARAM_KEYS_READ` is pinned by
  an AST-walking test. Never auto-select a proxy value ("nearest X"); the archive fill
  never writes the UV-spectrum menu.
- One measurement operator (`binning.py`) bins noise, model and Jacobians. Cell averages
  use the exact `_pl_antideriv`. The LSF is the count-density-weighted (per d ln lambda)
  Gaussian, applied to depth, removed-molecule and Jacobian rows; width R_refdata /
  `instruments.LSF_WIDTH`, fitted per mode to a Pandeia impulse (`run_parity.py
  --impulse`): re-measure on any refdata/PSF change, with an impulse SED dense enough to
  resolve the BROADENED line. A missing dispersion file is an error, never a no-blur.
- Noise floor: `sigma = max(sigma_random, floor)` on final bins; never quadrature,
  rescaled or averaged down by transits. Public noise APIs validate inputs and raise.
- Fisher inversion stays rank-aware (`fisher._marg_sigmas`, whitened
  eigendecomposition); per-segment depth offsets INCLUDING segment 0.
- Composition is structural: `met_x_solar` + `co_ratio` re-initialize the equilibrium
  seed. `use_vm_mol` is pinned explicitly (default False), never inherited.
- `co_ratio` is bounded by `forward.co_bounds(network, use_photo)`, read by the API and
  `share_config`: two gate axes, network and photolysis. Never call the permissive
  bounds a ceiling; never raise sncho's photolysis-on 0.99 without a science decision
  (notes: Composition, C/O bounds and networks). Never add temperature or Kzz as gate
  axes and never hard-code an excluded C/O band. A refusal must cost no solve.
- Every certified stage records branch, longdydt, aflux_change and the controlling
  `species@z` cell, and a failure names them (`forward.check_converged`). Every certified
  column passes `forward.check_elements` (layer-median X/H within `ELEMENT_TOL` 1.5% of
  the build column) and `forward.check_unmodeled` (an absorber with no k-table above
  `UNMODELED_VMR_REFUSE` 1e-3 is refused before any spectrum; above
  `UNMODELED_VMR_WARN` 1e-4 the run warns).
- `sncho2025` is EXPERIMENTAL and not the default. Condensation requires `sncho`.
- `yconv_min` is canonical and may only tighten below 0.1: cold baseline solves exit on
  the LOOSE branch, so `yconv_cri` alone cannot tighten them.
- C/O widths: the forecast curve is the density per d ln(C/O) on a log C/O axis
  (`posteriors.ln_gaussian_curve`; no 1/(C/O) Jacobian). Every finite C/O uncertainty is
  quoted in log10(C/O) dex (`fisher.format_co_width`, the "log10 C/O" table row); no
  mapped physical C/O range anywhere. `fisher.display_sigma`'s dlnCO value is never
  shown. Share files carry the run's `elemental_abundances`.
- Fisher Jacobians: certified FD (central, or one-sided toward lower C/O via
  `fd_stencil`; h-vs-2h gate, Richardson) or warm-jvp AD. AD CHEMISTRY rows are
  photolysis-on only. Cloud-deck rows and `lnR0` are RT-only jvps and stay available
  photolysis-off. lnR0 is the engine's radius change at fixed planet mass in both modes:
  pass it to the engine, never multiply a depth by e^(2 lnR0). AD dlnCO is refused unless
  the oxygen-reservoir margin exceeds `CO_BZ_MIN_AD` on both the build and the converged
  column.
- AD rows: one warm jvp per row through the engine's `converged_y_jvp`, NEVER a vmap over
  tangent directions, never a plain `jax.jvp` read at the column's certificate, never a
  step floor. Direction `FD_STEPS[name] * e_i`, row divided by the step. First on an
  `AD_BUILD_OVERRIDES` build (derivative builds only; cold solves and FD stencil points
  keep `DT_MAX_S`) at the config cadence; a row that does not certify is re-solved at
  cadence 1 with `AD_RETRY_COUNT_MAX` and flagged in `photo_escalated`. An uncertified
  derivative is never reported.
- Photolysis-cadence escalation is per FD ROW, never per stencil point, and only a
  `NotCertified` triggers it (notes: Jacobians and Fisher).
- Composition FD stencil points keep the baseline's geometry anchor by design; do not
  "fix" it from the code shape; settle it by measurement (notes: Jacobians and Fisher).
- Condensation is detection-only, API-only (the GUI pins it off; `share_config` refuses
  files that enable it), refused with any Jacobian, photolysis off or moldiff off.
  Settling, escape and boundary fluxes are removed; their four keys stay in the payload
  pinned off and enabling values are refused.
- Opacity: correlated-k over the published ExoMolOP k-tables is the ONLY path; the
  removed line-by-line and Mie keys are refused by name. `run_model` verifies the
  engine's echo of every RT knob: never drop an echo check. Species with no k-table are
  not offered (`_NO_EXOMOLOP_TABLE`). Never widen `EXTRA_MOLECULES_DEFAULT` without a
  measured ppm impact; widening either molecule list is a DEPLOY change (re-run
  `upload_data.sh`).
- Every "Data sources" row carries a resolvable DOI AND page (`test_app_smoke` fails on a
  blank or placeholder). Curate k-table DOIs in `fetch_exomolop._DATASET_DOI`, never in
  the GUI; `ktable_sources` keeps the header value as `header_doi`. A new
  `SFLUX_CHOICES` entry needs a `planets.SFLUX_SOURCES` entry.
- The GUI shows NO install inventory; `jwst-tool data` is the only place for that.
- Emission: `ibased_linsap` solver, Guillot default profile, flux-weighted tau-bottom
  gate. Eclipse depths come from the engine's `eclipse_flux_tau`, never
  `emission_flux_tau`; `emission_radius` stays the single (R_em/R_star)^2 anchor.
  Guillot `Tint` and `Tirr` defaults have one definition each (`planets.default_tint`,
  the registry) for `canonical_params` and the widget.
- Per-mode advisories (`result["warnings"]`) are REPORTED, never ranked on or applied.
- TSO instrument configs are pinned explicitly (readout pattern, extraction strategy,
  background); never add a mode without all of them. ngroup search: floors = pandeia
  mingroups, "saturated" only on measurement, the largest measured-safe count,
  maximality proven by bracket.
- Sigma labels: "conditional template S/N"; never a retrieval claim, never
  "PandExo-identical". `REPORT.md` stays a 32-line generated summary.
- The staleness guard compares the full `_run_sig` (goal + observation minus the
  display-only `show_noise`, `seed`, `combos`). Widget KEYS are a shipped contract:
  labels may move, keys may not.
- PICASO is the second engine (`chem_provider='picaso'`); VULCAN payloads, keys and
  fixture never move for it. The climate is solved in equilibrium and a quench applied
  after; never put the quench in picaso's climate loop and never re-normalize the TOA
  flux gate to admit it. ONE-WAY coupled: never call it self-consistent. The
  composition is imposed through the injected `chemeq_solver`, never mapped onto
  picaso's (mh, C/O); every layer solved cold. PICASO-only keys (`rfacv`,
  `picaso_quench`, `picaso_stack`) exist only in PICASO payloads. FD only; lnKzz and the
  climate T_int row refused; metallicity capped at `PICASO_MET_RANGE`; VULCAN-only
  settings refused, never dropped. `picaso_engine.certify` gates every result and
  re-runs on every cache load. picaso, photochem and virga are imported ONLY by
  `picaso_worker.py` (which imports nothing from the tool) and pinned ONLY in the picaso
  locks (test-pinned).
- Committed figures live in validation/ (figures/, parity/figs/) and are made only by
  validation/scripts through `validation/figstyle.py`; each PNG embeds its generator
  (`test_validation_figures` compares the parsed code). Both the app and figstyle load
  `src/jwst_tool/science.mplstyle`.

## GUI policy

- No explanatory prose in the GUI: a short label, and a `help=` tooltip only when the
  effect is not obvious. Caveats belong in README/notes. Precise statistical terms stay.
- ALL figure work goes through `plotting.py` under `render_lock`; figures are pure
  builders, never re-implemented in app.py (AST-checked). Legends never cover data; no
  legend title; axes are typed min/max pairs, no sliders. ONLY the summary figure has
  axis controls. The structure figure (`build_structure_figure`) shows pressure tick
  labels on both panels but the "pressure (bar)" label once, at the far left
  (test-pinned; `STRUCT_AXES_RECT` leaves room for it). T-P is black; mixing ratios use
  `VMR_COLORS` (the same colour per species as the note's figures) and
  `VMR_EXTRA_COLORS`.
- The shortfall warning renders in `_verdict_slot` beside the Run button. Results below
  Validation, in order: physical structure, parameter constraint forecast, summary
  figure. Do not re-add "Mode details", "Run summary & configuration", or separate
  custom-mode-set / marginalized-forecast expanders.
- A curve fitted to the jitter realization is never called a forecast: its y axis is
  "relative density", the forecast's "relative forecast density" (test-pinned); no
  "(one draw)" wording. The C/O panel carries no " per d ln(C/O)" suffix. The gray cloud
  is transmission-only.
- The "Chemistry engine" selector opens Step 2. PICASO widgets carry a `picaso` key
  infix and every VULCAN key stays as it was; under PICASO the method is a disabled
  FD-only widget on its own key, so `K("jacm")` is never written; the hidden engine's
  widgets are written back to themselves across a switch. A cap is a refusal sentence,
  never a moved widget ceiling (the C/O widget spans `CO_MIN` to `CO_MAX_PHOTO_OFF`).
- `st.set_page_config(..., initial_sidebar_state="expanded")` stays. A widget whose key
  is always pre-set in session_state takes NO `index=`/`value=` default.
- A loaded configuration applies ONLY through the Populate button's callback
  (`_populate_config`); the success line says only "Populated".
- Live readouts are fixed-width; progress bars keep the "[fwd]/[adj] PROG <frac>
  <label>" stdout protocol.
- A Run that returns nothing pops the previous result. A failed forward model shows only
  its exception sentence; the log tail sits behind a collapsed "Technical details".
- Copy rules: ">N (scan limit)" never "never"; "stops and shows an error"; "setting" not
  "knob"; eclipse vocabulary in emission; internal tokens stay out of labels.

## Deploy

- Never deploy or touch the Space unasked. The runbook (pins, the Space, verification)
  is in notes, Hugging Face Space. `test_deploy_pins.py` pins `deploy/pins.env` against
  the Dockerfile.
