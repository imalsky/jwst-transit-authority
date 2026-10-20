# todo

Open, owed or waiting on a decision, in priority order. Evidence is in notes.md, by
section name.

1. [blocker] Move the vulcan-forward floor and pins to the release that drops the `skip_warmup` profile key before the next tool release or Space pin bump: on the pinned 0.29.0 every chemistry build now runs the engine's warm-up solve.
2. [bug?] Cache keys carry no installed engine version, and the Space keys the data revision to the remote dataset HEAD rather than to the mounted bytes, so a cached result can be served for an engine or data state it was not computed on.
3. [science] Composition Fisher rows keep the baseline's radius/gravity anchor and Guillot T-P while T-P rows and a fresh run re-assemble them; measured negligible on one Guillot case and the TOI-7169 b transmission references, never on an eclipse case since lnR0 moves gravity.
4. [decision] The overflow Space is paused, so a visitor refused on a full primary gets a link to a Space that does not wake: restart it or unset the primary's `JWST_TOOL_OVERFLOW_URL`.
5. [low] The README forecast figure carries a "Model caveats" footnote inside the image; the figure rules put explanations outside a figure.

## 1. Engine floor after the profile-key removal

- Evidence: the tool no longer sets `profile["skip_warmup"]` (`forward._assemble_chem`,
  `_build_chem`) and reads the emission anchor from `vulcan_forward.constants.
  P_REF_EMISSION_BAR` instead of the engine's echo, because vulcan-forward is removing
  both profile keys and its build-time warm-up solve. On the pinned 0.29.0 the key still
  exists with default False, so each
  `_build_chem` call (geometry probe, baseline, every FD stencil point, AD builds) runs
  one extra full solve: roughly double the chemistry time of a run [E, from the call
  count; not measured]. The unit suite does not see it (it builds no chemistry); the
  slow tier would.
- Files: `pyproject.toml` (`vulcan-forward>=`), `deploy/pins.env`, the Dockerfile ARGs,
  `.github/workflows/tests.yml` (installs at the pins).
- Done: the floor and both pins name a vulcan-forward commit without the warm-up solve;
  `test_engine_api_contract` and `test_deploy_pins` pass; one cold default run shows no
  "warming up runner" line in its log.

## 2. Cache identity: engine version and data revision

- Code (verified): `forward.canonical_params` keys `version` (`_VERSION`) and, when set,
  `data_revision`; no vulcan3 or vulcan-forward version or commit enters `params_key` or
  `chem_key`. PICASO payloads do carry `picaso_stack` (picaso, photochem, CK record,
  worker version), so the asymmetry is VULCAN-only. `run_model` writes a `producer` stamp
  into each npz, but nothing compares it on load. `deploy/hf-space/entrypoint.sh`
  exports `JWST_TOOL_DATA_REVISION` from `HfApi().dataset_info(repo).sha`, the remote
  HEAD, once at boot.
- Why it matters: the Space's `model_cache` persists across restarts and pin bumps. An
  engine move that changes a column without a `_VERSION` bump serves the old column
  (notes, Cache keys and version history: the exemption does not generalize). For the
  data revision, the in-code comment says "a partial sync then costs a miss, never a
  stale hit". That holds only for later requests: a run made while the mount still holds
  the old bytes is stored under the NEW revision and is served as a hit after the mount
  catches up. The mount's live-update behaviour is inferred, not measured (notes, Data,
  provenance and the engine split).
- Choices for the maintainer: add the engine versions (or pinned commits) to the
  canonical payload, which moves every key and needs a `_VERSION` bump and fixture
  regeneration; check the `producer` stamp on load; read the revision from the mount
  rather than the Hub; or accept and record.
- Done: a decision recorded in notes, and the chosen check landed with a test.

## 3. Composition FD rows and the frozen geometry anchor

- Code (verified): `_assemble_chem` returns the `_build_chem` closure, which holds the
  baseline's `ovr["Rp"]` (the chemistry radius anchor) and the Guillot `tp_eval` built at
  the baseline gravity. The lnZ/dlnCO stencil points in `run_model` call that closure
  with new abundances only, so they keep the baseline anchor and T-P; T-P stencil points
  call `_assemble_chem(cp_s)` afresh, as a full `run_model` at the perturbed composition
  does. So a composition row is not exactly the derivative of `run_model`'s own output.
- Record: notes, Jacobians and Fisher, keeps this as a deliberate design with a
  measurement: on W39b Guillot at 10x the own-anchor and frozen-anchor lnZ rows agree to
  corr 0.999997, scale 1.0007 (~0.5% at the top of the metallicity range), and the
  marginalized `lnR0` absorbs an anchor shift to leading order. That measurement
  predates v60 (lnR0 now moves gravity at fixed mass) and does not record its science
  mode; the independent TOI-7169 b references are transmission. No eclipse case exists.
- Owed run: repeat the own-vs-frozen comparison on the default WASP-39 b eclipse case
  (Guillot) for the lnZ and dlnCO rows, through the production Fisher with lnR0
  marginalized.
- Done: the eclipse numbers are in notes beside the existing ones, and the maintainer
  keeps the design or orders the rows re-anchored.

## 4. Overflow Space paused

- Evidence: notes, Current state (Hub read 2026-10-14: overflow PAUSED). On a slot
  refusal `app._take_slot` shows "Open the overflow instance" whenever
  `JWST_TOOL_OVERFLOW_URL` is set; it was set on the primary (not re-read since).
- A paused Space stays down until restarted on its page.
- Choices: restart the overflow, or unset the primary's variable so a full primary
  refuses plainly.
- Done: the maintainer picks one; the choice goes to notes, Hugging Face Space.

## 5. Caveat footnote inside the README figure

`validation/scripts/fig_readme_example.py` exports the app's summary figure as the PNG
download does, and that export draws the model caveats as text under the panels.
Moving them out is an app change (`plotting.py` / `summary_figure`, under the GUI
policy), then a re-run of the script. Done when `assets/w39b_so2_forecast.png` carries
panels, labels and legends only.
