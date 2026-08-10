# TODO: open gaps and shortcomings

The one live list of everything known to be missing, approximate, or
deferred in this tool. Updated 2026-08-09 (ramp-floor fix + archive fill,
tool 0.25.0). Keep this file current: close items here when they
land, add new ones as they are found. The reasoning behind every decision
lives in `docs/decision_records.md`; scope and conventions live in
`docs/physics_and_conventions.md`.

## Correctness-affecting: fix next, in this order

1. **CLOSED 2026-08-09 (worker v9, 0.25.1): ngroup_min ramp search.**
   Floors equal pandeia 2026.7 per-detector `mingroups` (NIR 1, MIRI 2 --
   the field PandExo reads), with instrument-specific short-ramp warnings
   (`ngroup_warn_below`: 2 NIRSpec/NIRISS, 4 NIRCam, 6 MIRI,
   jwst-docs-verified). The first cut (0.25.0/worker v8) fixed the floors
   but exposed an optimizer defect -- min-of-predictions with a down-only
   verifier under-selected (1-group SOSS where 2 was measured safe, 7x
   sigma) -- caught by external review the same day and fixed in v9: the
   search now returns the largest MEASURED-safe count and the parity gate
   requires exact group agreement on short ramps plus a sigma anomaly
   band. Records: docs/decision_records.md, "ramp floors" + "2026-08-09
   external review" sections.

2. **NIRCam data-volume estimate.** A mode can rank "Best" while its full
   visit implies ~28 GB of data excess (seen as PandExo warnings in the
   parity artifact). Pandeia's own warning does not fire at the worker's
   nint=1, so the tool must estimate it: visit integrations x ngroup x a
   per-subarray frame-size table, surfaced in the operational-status
   column and the verdict qualifier, labeled an estimate of what APT would
   compute. Flag, do not demote the ranking. Half a day to a day.

3. **Gaussian LSF impulse-response validation.** The model is blurred with
   a flux-weighted Gaussian built from tabulated R(lambda), not Pandeia's
   real wavelength response. Unquantified error exactly on the modes where
   it bites (PRISM, MIRI LRS, blue SOSS). Fix: inject narrow features into
   the SED at several wavelengths per mode, run pandeia, measure the
   effective kernel against the Gaussian prediction, commit a report. One
   to two days; open-ended if it disagrees (then: empirical per-mode
   kernels, a separate decision).

## Needs a decision, not code

- **Upstream PICASO report** (`docs/decision_records.md`, part 3): four
  findings drafted for filing as GitHub issues on natashabatalha/picaso.
  Never posted; needs explicit approval to file.
- **Adjoint diagnostics panel**: hidden as "in development" since 0.23.2
  (`app._ADJOINT_PANEL_IN_DEV`; the full panel code is intact behind the
  flag). Decide: finish and re-enable, or remove.
- **Subarray/readout search as a feature**: each mode is one fixed
  configuration by design (disclosed everywhere since 0.23.3). Ranking
  alternate PRISM/SOSS subarrays and readout patterns would be a new
  scope, not a fix.

## Small refinements (known, low stakes, do opportunistically)

- LSF column-width division: the per-column count rate is used as a
  continuous LSF weight without dividing by column wavelength width;
  measured worst case ~6-7 ppm on PRISM/MIRI LRS (S2-09a).
- datacheck does not inventory dispersion files, so a missing native-R
  file is only caught at run time as a warning (S2-09b).
- Exact separate in/out transit-depth error propagation: the symmetric
  approximation is conservative by ~3d/4 of the depth (documented in
  noise.pixel_depth_variance); a refinement, not a bug.
- The eps Eri UV file's GUI "MUSCLES" attribution is loose (the raw input
  is an HST UV-sum product); cosmetic (S2-01).

## Validation gaps (absence of evidence, not defects)

- CI runs the numpy-only suite; the slow forward model, the PandExo
  parity harness, and the deployed full stack are not exercised per
  commit. The scheduled full-stack smoke mostly covers chemistry.
- Per-pixel saturation-mask parity against PandExo has never been
  compared (only configuration-level saturation agreement).
- The PICASO-native RT cross-model report
  (tests/parity_picaso/outputs/REPORT.md) is a FAIL and its numbers are
  STALE (they predate the inverse-square-gravity change). Rerun pending;
  never cite it as validation.
- The PICASO climate mode is certified around WASP-39 b only; other
  planets/nodes/rfacv values are gate-checked dynamically but have no run
  history (`tests/live/test_picaso_live.py` smoke matrix).

## Deferred features (recorded with re-entry sketches in the PICASO section of docs/physics_and_conventions.md)

- PICASO quench machinery, restoring a physical lnKzz row under the
  equilibrium engine.
- Cloudy climate solves (virga refdata is empty).
- Off-node climate composition (needs correlated-k table blending).
- Per-side one-sided composition derivatives at table nodes (the kink
  gate currently refuses).
- `jwst-tool fetch` for the PICASO reference tree (user-supplied Zenodo
  data; datacheck reports it).
- AD through climate mode (uncertified combination).
- Live TAP lookup as an opt-in alternative to the shipped archive snapshot
  (0.25.0 ships snapshot-only by decision; `archive.py` is shaped for a
  `lookup_live` provider returning the same row schema, failures shown,
  never a silent fallback between sources).
- Archive-fill depth (2026-08-09 review): per-field references and
  uncertainties in the snapshot, field-level provenance tracking in the
  GUI (which widgets still hold archive values vs edits), and uncertainty
  propagation for the derived surface gravity. The limit-flag refusals and
  composite-value disclosure shipped in 0.25.1; these are the next layer.

## Accepted limitations (deliberate; reasoning in docs/decision_records.md)

- Cache/share identity is canonical params + a hand-bumped version, not
  content pins of the engine stack (S2-05). Share files record installed
  versions as information only.
- The worker ramp is transit-independent; short events warn about <3
  in-event integrations and are never re-run with a restructured ramp
  (S2-10).
- Emission is absorption-only; Mie clouds in emission are refused, not
  approximated (S2-02). No scattering-aware emission solver is planned.
- Room-temperature HITRAN lists and the hot-band caveat above ~2000 K;
  swap line lists for publication-grade absolute work (S2-06).
- Stellar contamination (spots/faculae) is not modeled (README limit).
- ExoJAX capabilities not wired: reflected light, scattering emission,
  correlated-k, H-minus, atomic/FeH lists, rotational broadening, GP
  noise kernels (README).
- UV data inherited from upstream VULCAN as-is: eps Eri 115-283 nm
  coverage, GJ 1214 zero-flux FUV runs (S2-01 addendum).
- σ_detect is a conditional matched-template S/N and the Fisher numbers
  are local Cramer-Rao bounds; neither is a retrieval product. This is a
  statement of what the tool is, permanently disclosed, not a gap to fix.
