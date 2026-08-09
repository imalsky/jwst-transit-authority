# Decision records

Every audit and review disposition plus the reportable upstream findings, in
one file. Consolidated 2026-08-05 from `audit_decisions_2026-07-21.md`,
`review_decisions_2026-08-05.md`, and `upstream_report_picaso.md` (those
filenames are retired); content is verbatim apart from retargeted
cross-references. Three parts, oldest first:

1. the 2026-07-21 second-pass science audit (findings S2-01 to S2-11),
2. the 2026-08-05 adversarial review (findings 1 to 11),
3. the draft upstream PICASO report (never posted without explicit approval).

Also retired the same day: `picaso_forward_model_plan.md` (2026-07-20,
plan-only) was deleted as superseded -- the plan shipped as v18, its record
lives in the PICASO section of `physics_and_conventions.md`, and its open
decisions were all resolved by that implementation.

This file records WHY each call was made and never changes retroactively.
The LIVE list of open gaps and deferred work is `TODO.md` at the repo root;
when an item below says "open" or "deferred", TODO.md is its current
status.

---

# Decisions on the 2026-07-21 second-pass science audit

Status: written 2026-07-21 against tool 0.14.2 / retrieval 0.11.1+ / vulcan-jax
0.3.0. This is the decision record for `JWST_TOOL_SCIENCE_AUDIT_SECOND_PASS`
(audited snapshot: tool `c5c8b1a9`, retrieval `fa39bbb`, vulcan-jax `f626563`).
Every claim was re-verified against the code before a decision was made; where
the verification contradicts the audit, that is recorded too. Most findings are
deliberate, previously documented design decisions; this file says which, and
why the rest were fixed or accepted.

Two context facts the audit lacked:

1. **The audited snapshot predates same-day fixes.** Retrieval `5eed184`
   (refuse Mie clouds in emission; g(r)-consistent transmission tau) and tool
   `3937835`/`d65d7e9` (upfront Mie+emission refusal, missing-LSF warning
   channel, RCB-as-seed reinterpretation, per-molecule emission tau gate)
   landed 2026-07-21 afternoon, after the audit's clone.
2. **The two data-level findings are inherited from upstream VULCAN.** Both the
   eps Eri UV file (S2-01) and every photochemistry-table anomaly (S2-08) are
   byte-identical to exoclime/VULCAN (Tsai); the buggy builder line is verbatim
   upstream. They are upstream data-quality issues this stack vendored under
   its match-master parity policy, not defects introduced here.

## Per-finding decisions

### S2-01 -- eps Eri UV surface flux low by 3.4265x  [FIXED]

TRUE and exact: upstream `atm/make_spectra_in_nm.py` multiplied by
R_star = 0.735 R_sun where the surface-flux conversion divides
(`F_surface = F_earth (d/R_star)^2`), so the shipped spectrum is low by
R_star^4 = 0.735^4. Inherited verbatim from exoclime/VULCAN; only the
WASP-107 b registry default consumes the file.

**Decision: fix the normalization only.** The vendored
`vulcan_jax/atm/stellar_flux/sflux-epseri.txt` is rebuilt from the raw HST
file with R_star in the denominator; construction otherwise identical
(positive-only filter, DQ ignored, 115-283 nm span, duplicates retained).
Full record: VULCAN-JAX `docs/validation.md` entry C4;
parity-audit allowlist `KNOWN_SFLUX_RESCALES`. `forward._VERSION` bumped to
22 because the UV file is cache-keyed by NAME, not content. WASP-107 b
chemistry/spectra/scores regenerate on next run.

**Accepted, not fixed:** (a) the 115-283 nm coverage -- the photolysis grid
clamps to the file span (master-identical behavior), so EUV and >283 nm bands
are omitted; a spliced 2-700 nm product needs a sourcing/splice policy and is
not worth inventing for a planning proxy; (b) the positive-only/DQ-blind
construction -- measured photolysis-integral sensitivity is only 2-6% for
H2O/CH4/H2S/SO2/HCN (HO2 ~2x under a signed variant). The audit's claimed
1.4-2.2x sensitivity for six molecules did not reproduce (5 of 6 measure
1.02-1.06x). The GUI label's "MUSCLES" attribution is loose (the raw input is
an HST UV-sum product); cosmetic, fix opportunistically.

### S2-02 -- Mie scattering counted as thermal absorption in emission  [ALREADY FIXED]

TRUE for v16-v18.1 (the branch was opt-in but reachable end-to-end).
Fixed before this document existed: retrieval `emission_flux` raises on a Mie
deck (conservative-scattering zero-emission limit named in the error), and
`canonical_params` refuses `mie_condensate` + emission upfront. Transmission
keeps extinction-only Mie (correct chord attenuation; forward-scattering
caveat documented). The analytic power-law cloud stays allowed in emission as
a deliberately absorbing phenomenology. A scattering-aware emission solver is
NOT planned; the refusal is the design.

### S2-03 -- convergence gate is yconv_min, not the UI's yconv_cri  [DELIBERATE; WORDING FIXED]

Mechanically TRUE, and by design: certification is the runner's canonical gate
(`conv_normal` AND `longdy < yconv_min = 0.1`), upstream-faithful, and the
loose branch also requires a near-zero slope (1e-8..1e-10) and settled
photolysis flux -- accepted states are demonstrably steady, just not bounded
by the user's strict-branch value. W39b photo-on physically plateaus at
longdy ~0.06-0.09 with |dy/dt| ~ 1e-11, so a 1e-3 "requirement" would refuse
a genuinely steady column; that is why the tool does not force
`yconv_min = yconv_cri`. The results panel already reports the actual
residual against the actual gate.

**Fixed:** the GUI help sentence that implied the selected value is the
enforced bound now states the loose branch and points at the results panel.

**Accepted:** the recorded MIRI LRS SO2 sensitivity (0.9 -> 1.9 sigma between
the retired fast and high tiers, which conflate nz/nu_pts/yconv) stands as a
two-point record with no committed third refinement. Decision: no plateau
campaign. The committed guidance (raise nz / tighten yconv for final mid-IR
numbers) is the intended mitigation; weak-MIRI numbers are quoted from the
high tier, not the default.

### S2-04 -- Pandeia 2026.2 labeled "current" while STScI ships 2026.7  [WORDING FIXED; UPGRADE NOT PLANNED -- SUPERSEDED 2026-08-04: upgraded to the 2026.7 triple, parity regenerated and gate-PASSED; "current" now IS 2026.7 and 2026.2 is the archival_2026_2 backend]

TRUE. The label was written 2026-07-13 when 2026.2 was the supported release;
STScI's news page moved to 2026.7 (Cycle 6) on ~2026-07-16 and now lists
2026.2 as an old release. The status string and comments no longer claim
currency: "current" is documented as the backend TOKEN, and the user-facing
status now says forecasts are one calibration release behind the live ETC.

**Accepted:** staying on 2026.2 for now. Rationale: PandExo itself pins engine
2026.2, so the measured parity anchor (tests/parity/, the committed noise-model
envelope) is only valid on this pair; upgrading means re-downloading the full
engine/refdata/PSF tuple and regenerating every mode's parity. That is a real
campaign with no current science driver (relative mode rankings are the tool's
product, not absolute ETC currency). Revisit before using the tool for an
actual proposal submission.

### S2-05 -- builds/caches not content-addressed  [DELIBERATE, NOW DOCUMENTED]

Mostly TRUE and deliberate; partly wrong. The HF-space Dockerfile shallow-clones
branch heads with a manual SRC_STAMP cache-bust and post-hoc BUILD_INFO SHA
receipt; /data caches persist across rebuilds; the model key is canonical
params + hand-bumped `_VERSION`; the noise key is the job + engine/refdata
VERSION-marker hashes + PHOENIX catalog stat. This is the documented operating
model: content-hashing multi-GB payloads is off the table, and correctness is
a maintainer discipline ("bump the version when physics changes") rather than
a hash guarantee. The audit's "no instrument configuration in the key" is
FALSE (the full tool-side mode config, star, and strategy are keyed); the
PICASO subsystem DOES content-fingerprint its refdata.

**Accepted risks, explicitly:** (a) an in-place edit to a same-named UV/data
file without a version bump serves stale results -- mitigated for the one real
instance (S2-01) by the v22 bump, and the discipline is now written down here;
(b) refdata payload edits under an unchanged VERSION marker go undetected;
(c) a rebuild without a SRC_STAMP bump can reuse a stale clone layer (the
SETUP.md procedure covers it). Full pinning/manifesting is deliberately NOT
adopted for a single-maintainer research tool; this file is the record that
the trade was chosen, not overlooked.

### S2-06 -- room-T HITRAN + air broadening over a 320-2980 K domain  [DELIBERATE, PRE-DOCUMENTED]

TRUE as a limitation, and already documented before the audit: the retrieval
config carries an explicit KNOWN LIMITS block (HITRAN 296 K lists
under-represent hot bands vs HITEMP/ExoMol; swap sources one line per
molecule), README repeats it, the GUI warns on layers hotter than 2000 K
about missing ultra-hot opacity (H-, Na/K/Fe, TiO/VO/FeH), and H2/He
broadening ships as a real opt-in (per-molecule coverage enforced loudly).
CO is the one hot ExoMol list. Decision: no change. The [320, 2980] K window
is the PreMODIT table range (reject-never-clip), not a fitness-for-purpose
claim; treat absolute hot-band amplitudes as approximate, use the tool for
mode ranking and relative comparisons, swap line lists for publication-grade
absolute work.

### S2-07 -- "every planet ... validated machinery" wording  [WORDING FIXED]

PARTLY: the phrasing always named W39b as the validated anchor, but it was
overclaim-prone. GUI help, `planets.py`, and `forward.py` now say explicitly:
shared code path validated on WASP-39 b, NOT per-planet validation; committed
parity/live evidence is W39b-centered. Cross-planet end-to-end validation is
not planned; registry values remain editable planning defaults audited against
the NASA Exoplanet Archive.

### S2-08 -- photochemistry-table anomalies  [UPSTREAM; ACCEPTED]

All TRUE, all inherited byte-identical from exoclime/VULCAN (74/74 active
cross/branch files match master), and the loader's silent-sort/accept behavior
is parity-faithful to master. The conspicuous items were already documented
in-repo (CH3SH 354/254 reversal: `photo_setup.py` docstring + VULCAN-JAX docs;
duplicate policy: corrections doc "deliberately not logged" scope note; C4H2
0.06 and C6H6 1.14 branch sums: upstream data-file headers -- the former is
the intentionally un-modeled C4H2* channel, the latter a two-photon
accounting). Newly recorded here for completeness: CH4 and NO2 carry
dissociation-over-absorption excursions of at most 0.5% (1 row / 32 rows).

**Decision: no local data repairs and no strict load-time validator.** Match-
master parity is the standing policy; silently "fixing" upstream science data
would fork the oracle. A results-affecting anomaly gets the C1/C4 treatment
(documented divergence + parity-audit allowlist) case by case; none of these
rise to that (trace channels, sub-percent excursions).

### S2-09 -- LSF column-width weighting; missing-dispersion skip  [ACCEPTED; PARTIALLY FIXED]

(a) TRUE and distinct from the documented sub-pixel-stellar-line limit: the
per-column extracted count rate is interpolated as a continuous LSF weight
without dividing by the per-column wavelength width, adding a spurious
dlam_pix(lambda) factor where the dispersion is chirped. Measured maxima are
5.87 ppm (PRISM) / 6.96 ppm (MIRI LRS) / 0.011 ppm (SOSS) at final R=100,
sub-ppm median (independently reproduced at the same order). **Accepted:**
few-ppm worst-case on the two low-R modes is far below the noise floors in
play; dividing by column width is a clean future refinement, not a correctness
gate. Now documented (here) rather than silent.

(b) PARTLY: a missing native-R/dispersion file skips the blur. Since `3937835`
this records into the result warnings channel (shown in the GUI notes), no
longer fully silent. **Accepted** that it warns rather than raises: high-R
gratings legitimately no-op the LSF, and refusing a run for a display-level
few-ppm blur on a mode the kernel barely resolves would be disproportionate.
`datacheck.py` does not inventory dispersion files; add opportunistically.

### S2-10 -- ramp allows < 3 in-transit integrations  [DELIBERATE, PRE-DOCUMENTED]

TRUE mechanics, deliberate design: the worker ramp is transit-independent so
one noise cache per star serves any transit; `detect.py` emits a loud warning
when fewer than 3 integration cycles fit in transit, naming PandExo's
restructuring rule, and `pixel_depth_variance` hard-refuses below one cycle.
Decision: no change. The box-depth variance stays valid at 1-2 cycles; time
resolution and PandExo ramp comparability degrade, which is exactly what the
warning says.

### S2-11 -- native PICASO RT vs ExoJAX residuals  [PRE-DOCUMENTED]

TRUE and already recorded with the exact numbers (-2207 ppm offset; 688 ppm
median / 1540 ppm p95 after offset removal; targets missed) in
`tests/parity_picaso/outputs/REPORT.md`, `docs/physics_and_conventions.md` (PICASO section), and
notes.md, with the correct framing: offline cross-model envelope, never a
production path, residuals attributed to opacity sources + gravity
conventions. Both chemistry providers run production RT through ExoJAX.
Decision: no change; PICASO-native RT is not and was never the validation
reference.

## Errors in the audit itself (for the record)

- The 1.4-2.2x positive-only/DQ photolysis sensitivity (S2-01 addendum) did
  not reproduce: 5 of the 6 named molecules measure 1.02-1.06x on the
  measured-band TOA integrals; only HO2 reaches ~2.1x under a signed
  construction.
- "Noise identity omits instrument configuration" (S2-05) is false; the full
  tool-side mode configuration is in the key.
- The audit missed that S2-01 and S2-08 are inherited verbatim from upstream
  exoclime/VULCAN, and that S2-02 clouds/S2-06 opacity/S2-10 ramps carried
  prior documented caveats (retrieval notes.md, config KNOWN LIMITS,
  detect.py warning).
- Its snapshot predates the 2026-07-21 fix commits, so its "release blockers"
  1 (partially), 2, and parts of S2-09 were already closed by the time of
  this record.

## Addendum (same day): third-pass conclusions and two new hazards

The auditor's follow-up conclusions restate S2-01..S2-05 (all addressed above;
the eps Eri normalization is FIXED at the source as of this record, so
"default WASP-107 results remain invalid" no longer holds for freshly
generated results -- forward v22 busts every stale cache) and add two new
enabled-branch hazards, both verified:

- **GJ 1214 selectable UV file, zero flux across 133.75-181.55 nm: TRUE,
  inherited, explained.** `sflux-GJ1214.txt` carries 464 zero-flux rows in
  seven runs between 133.75 and 181.55 nm, byte-identical to upstream
  exoclime/VULCAN. This is the MUSCLES-style treatment of a faint M dwarf's
  FUV: bins consistent with zero are floored at zero rather than filled with
  a model, so FUV photolysis (H2O, CO2 bands in that window) is undercounted
  under this proxy. ACCEPTED: it reflects the measurement floor of the source
  data, the file is not any registry planet's default, and inventing flux to
  fill it would be worse. Recorded here as the explanation the audit said was
  missing.
- **GUI zenith angles can reach the two-stream pole: TRUE with a safe
  default.** The documented upstream two-stream particular-solution pole
  (VULCAN-JAX corrections guide, "Two-stream particular-solution pole")
  requires `1/mu^2 = (1-w0)/edd^2`; with edd = 0.5 it is reachable only for
  zenith angles below 60 deg (mu > edd). The GUI range is [0, 89] deg, so
  pole-reachable angles are selectable; the default 83 deg (Tsai 2023
  terminator slant, mu = 0.12) is far outside the reachable regime. ACCEPTED
  without a range clamp: the pole is an inherited upstream defect shared
  identically with master (parity unbiased), quantified upstream as touching
  only the diffuse actinic-flux correction in ~0.1-0.5% of layer/wavelength
  cells; clamping the GUI to > 60 deg would remove legitimate dayside-average
  configurations to guard a thin band. Revisit only if a forecast is run at
  a low zenith angle with strongly scattering layers.

## Open items (accepted, no committed plan)

- ~~Pandeia 2026.7 tuple upgrade + parity regeneration (S2-04)~~ DONE
  2026-08-04: upgraded, parity regenerated, gate PASS.
- MIRI LRS SO2 three-point convergence plateau (S2-03): the high-tier
  guidance stands in for it.
- LSF column-width division and datacheck dispersion inventory (S2-09):
  refinements, few-ppm stakes.
- Scattering-aware cloudy emission (S2-02): refusal is the design; no solver
  planned.

---

# Decisions on the 2026-08-05 adversarial review

Status: written 2026-08-05 against tool 0.23.2 (reviewed snapshot `1435920`,
functional parent `20246fb`). The review confirmed no defect in the
count-space, floor, matched-template, or Fisher mathematics. Its findings
concern configuration scope, operational feasibility, provenance, and
user-facing wording. This file records what changed in 0.23.3 and what did
not, finding by finding. Companion record: the 2026-07-21
audit decisions above (two findings restate decisions made there).

## High severity

### 1. ngroup_min excludes permitted bright-target ramps  [DISCLOSED; SEARCH CHANGE OPEN]

TRUE as a limitation and previously documented as a policy delta
(`tests/parity/outputs/REPORT.md`, "Residual policy differences": PandExo
drops to 1 group on PRISM where this tool floors at 2). It was not disclosed
in the registry or the GUI. Fixed in 0.23.3: `instruments.py` documents that
`ngroup_min` is a tool policy bound, not the physical minimum ramp, and the
GUI's "unusable on this star" warning now says shorter permitted ramps exist
and are not searched.

**Open, not fixed:** searching the full permitted group range (1-group
NRSRAPID/NISRAPID, warned or limited-access MIRI ramps) and classifying
physical saturation vs calibration warnings vs limited access. That is a
worker + registry + verdict rework, invalidates the noise caches
(worker_version bump), and requires a parity re-run. Deliberately deferred
to a decision by the maintainer, not silently skipped.

### 2. Modes presented generically while one fixed configuration is evaluated  [FIXED (disclosure path)]

TRUE. The review offered two remedies; the labeling remedy was taken:
the mode multiselect help now states each mode is one fixed subarray +
readout configuration with no subarray search, the mode details table (and
its CSV) gained a `configuration` column (e.g. `sub512/nrsrapid`), and the
README lists the fixed-configuration scope as a headline limit. Ranking
across alternate subarrays remains out of scope, now stated rather than
implied.

### 3. Cache/share identities do not pin forward-engine state  [PARTLY S2-05 (DELIBERATE); WORDING + INFO PROVENANCE FIXED]

The cache-identity half restates the accepted S2-05 trade
(above): identity is canonical params + a
hand-bumped `_VERSION`, by maintainer discipline, not content addressing;
the PICASO subsystem is the exception. A pointer comment now sits on
`forward._VERSION` so this stops being re-found.

The share-file half was a real overclaim: `share_config.py` said the file
"fully configures the run on any machine". Fixed: the docstring now says the
file captures every tool INPUT and does not pin software or science data,
and `build_share` records the installed vulcan-jwst-tool / vulcan-forward /
vulcan-jax / exojax versions as an informational `software` block (ignored
on load; older files stay loadable). Full commit/line-list pinning in keys
and share files remains NOT adopted (S2-05 stands).

### 4. "Best mode" without operational feasibility  [DISCLOSED; MODELING OPEN]

TRUE that the ranking is pure science information. Fixed in 0.23.3 with the
review's labeling remedy: a caption under every verdict states that
operational feasibility (data volume, scheduling, calibration warnings) is
not checked and the configuration must be verified in APT; the README limit
bullet says the same. Modeling the full integration sequence and data
volume (the ~27.8 GB NIRCam excess in the parity artifact) is real work on
the worker and is deferred to a maintainer decision.

## Medium severity

### 5. Short events scored on a ramp PandExo would restructure  [DELIBERATE, S2-10]

Restates S2-10, decided 2026-07-21: the ramp is transit-independent by
design (one noise cache per star), `detect` warns loudly below 3 in-event
cycles, and `pixel_depth_variance` refuses below one cycle. No re-run with
a shortened ramp is performed. A pointer comment now sits at the warning
site. No change.

### 6. LSF wording overstates the Gaussian approximation; README contradicts it  [FIXED]

TRUE on both counts. `binning.py` was already honest. Fixed the other
surfaces: the R help text now says the model is blurred with a Gaussian
approximation of the tabulated native R(lambda); the per-mode note says
"Gaussian LSF approximation"; the README describes the treatment and its
pending impulse-response validation, and no longer lists instrumental
broadening as absent (ExoJAX's own operator is what is not wired).

### 7. Projected-detection caption says clouds are fixed  [FIXED]

TRUE at the reviewed commit. The offending caption was already deleted in
the 0.23.2 copy pass; 0.23.3 adds the honest replacement (the (proj) column
profiles the available temperature-profile, reference-radius, and cloud
directions) and fixes the stale `detect.py` docstring, which still said
"T-P and lnR0" while `_NUISANCE_JAC` includes the cloud and Mie rows.

### 8. README universal-bound language  [FIXED]

TRUE. "Upper-bounds any real retrieval result" and "real posteriors can
only be wider" replaced with the conditional statements (usually lower
under the same model and noise assumptions; Fisher values are local
likelihood-based bounds, and informative priors can narrow a posterior).
Same fix in the `detect.py` docstring.

## Low severity

### 9. Parity benchmark generalized  [FIXED]

TRUE. The README noise bullet now names the scope: a three-star,
fixed-configuration, no-floor benchmark, ranges are benchmark results and
not a guarantee.

### 10. Floor-provenance caption mismatch  [FIXED]

TRUE: the caption claimed Greene 20/30/50 while the prefills are 15-40 ppm
per mode. The caption and the `instruments.py` docstring now describe the
actual values as informed by, not identical to, the Greene et al. 2016
convention.

### 11. Stale pyproject metadata  [FIXED]

TRUE. Header comment now states the 2026.7 matched triple, the
archival_2026_2 backend, and the removed legacy backend; the description
says "transmission and emission spectra".

## Style opinions (the review's best-practice list)

Adopted in 0.23.3, at small scale:

1. **Caveats next to the headline verdict**: the caption under every Best
   verdict names all three (fixed configuration, conditional statistic,
   APT feasibility unchecked), and a verdict whose winning mode carries
   warnings says so inline.
2. **Exact configuration labels**: the verdict shows the winning mode's
   subarray/readout (`app._mode_cfg`) and the mode table carries a
   `configuration` column. NOT adopted in the spectrum-plot legend: the
   short labels stay there to avoid clutter, and color + marker key each
   series to the table row that holds its exact configuration.
3. **Operational-status field**: the mode table gains an honest
   three-value `operational status` column (`app._op_status`): saturated
   at the shortest ramp tried / warnings, verify in APT / verify in APT.
   The tool checks saturation and relays Pandeia warnings and checks
   nothing else, so no row can ever read "recommended".
4. **Separate science / feasibility / schedulability rankings**: not
   adopted structurally (only the science ranking exists to sort by); the
   caption and status column keep the three concerns verbally separate.

## Review statements accepted without change

The confirmed-clean components (count-space binning, floor semantics,
saturation masking given the configuration, detection/Fisher algebra,
chemistry/RT gating) match this repo's own validation records. The
validation-gap notes (CI excludes the slow stack; the parity artifact
validates the estimator, not configuration policy; the PICASO-native RT
report is a FAIL and says so) were already documented where the review
found them.

---

# Decision 2026-08-09: speed-first GUI defaults (AD method, Guillot structure, no lnKzz row)

Maintainer decision (Isaac), motivated by default constraint runs taking
~30+ minutes, almost all of it FD Jacobian rows. Four changes:

1. **Differentiation method defaults to AD in the GUI** (`jac_method="ad"`,
   the warm-jvp path; ~2 min per row vs ~4-7 min FD). The API default stays
   `"fd"`: it works everywhere, while an `"ad"` API default would turn
   photo-off or PICASO Fisher calls into hard errors. The GUI already forces
   `fd` under PICASO and locks photochemistry ON while AD is effective; the
   photo-lock's session-state fallback now mirrors the new widget default.
2. **Default structure is the analytic Guillot profile for every planet**
   (`_default_tp_mode` returns `"guillot"` unconditionally). WASP-39 b
   previously defaulted to its verified measured table
   (`atm_W39b_evening_TP_Kzz.txt`). The table stays selectable and
   `shipped_tp_table_is_default` remains as the verification record.
   Stated trade-off (measured 2026-07-21, unchanged): Guillot + constant
   Kzz runs ~100 K hot through the SO2 formation zone with Kzz 4-33x low,
   so the published-detection agreement (G395H SO2 4.16 sigma) belongs to
   the shipped table. The W39B_REFERENCE test guard was re-anchored from
   "the default" to the explicit `tp_mode="file"` configuration; its cache
   key (`f14f4d10512552ea`) is unchanged, proving the validated atmosphere
   is bit-identical to the old default.
3. **lnKzz is out of the default free-parameter set** (now lnZ + dlnCO).
   Still selectable; dropping it tightens the remaining sigmas toward the
   conditional bound, which the results table discloses (marginalized and
   conditional shown side by side).
4. **VULCAN as default engine**: already the case; no change needed.

No `forward._VERSION` bump: a given canonical parameter set means the same
physics as before; only which set the defaults resolve to changed.

---

# Draft upstream report: PICASO 4.0.1 findings from the vulcan-jwst-tool integration

Status: DRAFT for Isaac's review. Nothing here has been posted anywhere;
posting (e.g. as GitHub issues on natashabatalha/picaso) requires explicit
approval. Each item is self-contained so it can be filed separately. All
measurements 2026-07-20/21 against picaso 4.0.1 and the v4.0 reference data
release.

## 1. Corrupted row in the Visscher 2121 chemistry grid (data)

File: `chemistry/visscher_grid_2121/sonora_2121grid_feh1.0_co0.55.txt`,
row at T = 900.0 K, log10 P = -5.523 (file row 925, counting data rows from
0 after the two header lines).

Anatomy: every reported species in the row is uniformly deflated by a
factor ~0.747 relative to the interpolation of its T-neighbors (H2 0.7471,
He 0.7471, H2O 0.7477, CO 0.7474, N2 0.7471, Na 0.7467, K 0.7466 ...), so
the gas-phase sum is 0.746 where every neighboring row sums to >= 0.9987.
Two species additionally carry junk residues: VO is ~9.9e6x too high
(5.2e-12 vs ~5e-19 expected from neighbors) and CrH ~4.8e4x (9.2e-13).
The same (T, P) cell is clean in the four neighboring composition files
(feh0.7_co0.55, feh1.5_co0.55, feh1.0_co0.46, feh1.0_co0.82).

The pattern suggests a spurious ~25% phantom abundance entered this row's
normalization during generation (deflating every reported species), with
corrupted VO/CrH values as residues of the same event.

## 2. chemeq_visscher_2121 docstring says 20 pressures; the files carry 21

The docstring block ("2020 data points: 20 pressures ... 101 temperatures")
disagrees with the shipped files, which are 21 pressures x 101 temperatures
= 2121 rows (log10 P from -6.0 to +4.0). Cosmetic, but the stated grid
shape is load-bearing for anyone validating a re-implementation.

## 3. Feature request: denser C/O sampling near the low-pressure CH4/H2O transition

At low pressure the equilibrium CH4/H2O transition is sharp and sits inside
the [0.55, 0.82] C/O cell: at 1 mbar / 1100 K the per-cell table slopes
d log10 X / d ln(C/O) are CH4 +1.24 (cell 0.46-0.55) vs +9.56 (cell
0.55-0.82), H2O -0.94 vs -9.31, CO2 -0.64 vs -9.07. Any interpolation
across the existing nodes therefore cannot produce a trustworthy local
composition derivative near C/O ~ 0.55-0.82 at low pressure (we evaluated
monotone-cubic interpolation as an alternative to linear; its node
derivatives are interpolant convention rather than data, and its
leave-one-node-out error is worse near the transition). One or two extra
nodes in (0.55, 0.82) would resolve this for derivative-based applications.

## 4. Native transmission silently returns all-NaN when gravity() gets bare gravity (code)

`case.gravity(gravity=..., gravity_unit=..., radius=...)` followed by
`case.spectrum(opa, calculation='transmission')` returns transit_depth =
all NaN: `atmsetup.get_altitude` computes g = G * planet.mass / z^2, and
`planet.mass` is NaN when only gravity+radius were provided
(constant_gravity is only forced when the RADIUS is NaN). Passing
mass + radius works. Suggestion: raise loudly in the transmission branch
when planet.mass is NaN instead of propagating NaN depths.

## 5. Observation (no action needed): find_strat keeps the guessed radiative-convective boundary

On a strongly irradiated planet (WASP-39b-like inputs, Tint 200 K,
rfacv 0.5), rcb guesses of 60/65/70/75 (91-level grid, 1e-6..300 bar) all
converge with the final convective zone starting exactly at the guess, all
Schwarzschild-consistent against the shipped adiabat table, with deep
temperatures differing by up to ~1000 K at 7.6 bar across the family
(shallower guesses fail flux balance and are correctly reported
unconverged). This appears to be the physical deep-adiabat degeneracy of
static irradiated RCE rather than a bug; we note it because "converged"
output can differ substantially at depth depending on the rcb guess, which
users of the climate mode may not expect.
