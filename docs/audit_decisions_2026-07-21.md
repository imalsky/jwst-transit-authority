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
`tests/parity_picaso/outputs/REPORT.md`, `docs/picaso_roadmap.md`, and
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
