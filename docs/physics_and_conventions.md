# Physics choices and conventions

The modeling conventions this tool adopts, the defaults it ships, and why. Moved
out of the README in 2026-07.

Since the 2026-08-05 doc consolidation this file also holds the PICASO
engine's scope/limits/roadmap (formerly `docs/picaso_roadmap.md`) and the
config-vs-upstream-master deviation table (formerly
`docs/jwst_tool_upstream_deviations.md`), appended as their own parts below.
Decision records live in the one other doc, `docs/decision_records.md`.

## Composition scaling

Any metallicity or C/O knob must pick a convention, because C/O and metallicity
alone do not determine the composition: the same (Z, C/O) with different C/H and
O/H gives a different equilibrium composition and a different spectrum
(Drummond et al. 2019, MNRAS 486, 1123). So this tool states which direction it
moves.

**Metallicity scales the network's O, C, N, and S abundances together, with He/H
held fixed.** That is the common choice for kinetics codes (e.g. Moses et al.
2011); codes that anchor metallicity on carbon instead hold oxygen out of the
scaling (Goyal et al. 2018; Drummond et al. 2019 does it both ways). **C/O moves
carbon at fixed, metallicity-scaled oxygen** — the fixed-oxygen C/O direction of
Tsai et al. 2017 (ApJS 228, 20: "we keep f_O fixed and vary the value of f_C"),
applied at the metallicity-scaled oxygen abundance, since VULCAN's own paper has
no metallicity axis to scale it with.

This matters when comparing across codes. petitRADTRANS (see its
`pre_calculated_chemistry` documentation, and Mollière et al. 2017 for easyCHEM;
the 2019 code paper does not state the convention) and the ATMO and Goyal grids
anchor metallicity on carbon and vary oxygen — for the Goyal grid, use the 2019
erratum (MNRAS 486, 783), which corrected an oxygen-abundance rainout bug. The
Sonora and PICASO family preserves C+O (Marley et al. 2021). GGchem takes
complete elemental abundance sets as input — it has no metallicity or C/O
parameter, so the convention is the user's, and Woitke et al. 2018's own C/O
sweep varies carbon at fixed oxygen, the same direction as this tool.

So at non-solar C/O, **the same physical atmosphere maps to different (Z, C/O)
coordinates in different codes**, and at C/O below unity different molecules
carry the C/O signature: C-bearing species here (CO tracks C/O directly), H2O and
CO2 in the oxygen-varied codes (where CO carries no signal at all). The
conventions agree exactly at solar C/O and diverge first-order in ln(C/O) — by
C/O = 0.7, only 27% above solar, the two differ by ~28% in both C and O, a
~0.1 dex metallicity offset.

## Temperature-pressure profiles

Profiles are explicit only. Three options:

- **Guillot**, analytic, with a temperature parameter.
- **A tabulated table.** Either the shipped WASP-39b evening-terminator profile or
  an upload. The cache key carries the table's content hash.
- **A PICASO radiative-convective climate solve.**

A globally isothermal profile was removed in July 2026. It held the deep CO, CH4,
and NH3 quench region at one temperature and biased the disequilibrium abundances.

**The trade-off is stated rather than hidden.** A tabulated profile has no
temperature parameter, so file-mode Fisher forecasts carry **no temperature row**.
They are conditional on the profile being exactly right, and the reported
uncertainties are optimistic by however much temperature uncertainty would add.
Switch to Guillot when you need a temperature row.

## Eddy diffusion profiles

Four options: constant, two parametric forms, or the table's own `Kzz` column. In
every mode the Fisher `lnKzz` row is a multiplicative scale of the whole profile.

The two parametric names are upstream VULCAN's, kept for config compatibility
rather than coined here — `build_atm.py` documents `JM16` as the profile form
assumed in Moses et al. (2016) and `Pfunc` as the one in Tsai (2020). `JM16` is
`Kzz = max(K_deep, 1e5 (300 mbar/P)^0.5)`; `Pfunc` is
`max(K_max, K_max (K_p_lev/P)^0.4)`.

## Default structure is the analytic Guillot profile

Every planet, under both engines, defaults to a Guillot (2010) `T(P)` with
constant `Kzz` (2026-08-09 decision; before that WASP-39 b defaulted to its
bundled measured table). The analytic profile is the differentiable choice: it
carries T-P Fisher rows and works with the AD-default differentiation method,
while a tabulated `T(P)` has no temperature parameter at all. Bundled measured
tables stay **selectable** (`tp_mode="file"` resolves to that planet's own
table, and its `Kzz` column then supplies the mixing profile too).

| Planet | Bundled table (selectable) |
|---|---|
| WASP-39 b | `atm_W39b_evening_TP_Kzz.txt` (Tsai et al. 2023 evening terminator), verified end to end |
| HD 189733 b | `atm_HD189_Kzz.txt` — selectable but not verified: the solver does not certify a steady state on it at default settings, while the analytic profile converges in about 36 s |
| HD 209458 b | Refused. It is a full thermosphere model reaching 2997 K inside the chemistry grid, above the 2980 K opacity ceiling, and it is never clipped |
| WASP-107 b | None bundled |

Two facts are kept separate on purpose: whether a table **exists** for a planet,
and whether a run on it has been **verified end to end**. Tables are per-planet
and never substituted; selecting a planet without a usable one tells you why.

**Stated trade-off: the analytic defaults are biased in a systematic
direction.** A constant `Kzz` cannot follow a profile that climbs orders of
magnitude with altitude, and it is the photochemically active upper atmosphere
that pays. Measured against the bundled tables over the chemistry grid at
p < 1 mbar, the constant 1e9 cm²/s default runs 3.8-48x low for WASP-39 b
(`atm_W39b_evening_TP_Kzz.txt`) and 10-17x low for HD 189733 b
(`atm_HD189_Kzz.txt`), always suppressing photochemical products. The factor is
pressure-cut dependent — deeper than ~10 mbar the table falls *below* 1e9 for
WASP-39 b — so quote it with the cut. On WASP-39 b the Guillot default also ran
about 100 K hot through the SO2 formation zone when this was measured
(2026-07-21); the published-detection agreement (G395H SO2 4.16 sigma) belongs
to the **shipped table**, so select `tp_mode="file"` for W39b SO2 work.

## Boundary conditions

All off by default: gravitational settling, diffusion-limited escape for H, H2, and
He, and constant top and bottom per-species fluxes with deposition velocities.

## Clouds

Two independent decks, which can be combined:

- **An analytic power-law deck**, a gray-to-sloped opacity per gram.
- **A Mie condensate deck**, using real refractive-index optics from the **virga**
  condensate database (Batalha, Lodge & Moran, Zenodo; accessed through ExoJAX's
  `PdbCloud`), with a column-uniform lognormal size distribution. virga asks that
  the upstream literature source for each condensate's optical constants be cited
  individually.

Either deck's parameters can be freed and marginalized in the Fisher forecast when
that deck is on: the power-law amplitude and slope, and the Mie particle radius,
size dispersion, and abundance.

The Mie radius and dispersion ride a piecewise-linear lookup grid, so their
finite-difference steps are sized to stay **inside one grid cell** (`FD_STEPS`:
0.03 dex in log r_g against a ~0.1 dex cell, 0.05 in sigma_g against ~0.33) —
the row is then the local piecewise-linear slope rather than a knot average. The
allowed parameter ranges are inset from the grid edges by more than the full
±2h stencil, so a legal value can never produce an edge-clamped (silently zero)
derivative; an out-of-range value is refused. The Mie abundance is exactly
linear in the optical depth and rides no grid. Each Mie condensate needs a
one-time lookup grid built with `tools/generate_miegrid.py`.

The **node-kink gate** (`picaso_chem.FD_KINK_TOL`) is a separate mechanism and
applies to the PICASO equilibrium provider's composition rows, whose tables are
per-node with no interpolation: there the one-sided secants are compared and a
row whose left and right derivatives disagree materially is refused outright
rather than reported.

## Height-dependent gravity in the transmission optical depth

The forward engine converts pressure to column mass with an inverse-square
gravity profile, `g(r) = g_btm (R_btm/r)^2`, evaluated at layer midpoints, so the
chord heights and the opacity columns share one geometry. Emission is
plane-parallel and keeps the constant bottom gravity. The profile and its
measured accuracy live in the sibling engine, `vulcan-retrieval`
(`docs/forward_model.md`, "Height-dependent gravity"); this tool only selects it
by importing that engine.

It is recorded here because it is cache-visible. The change moves every
transmission spectrum at the tens-of-ppm level, so `forward._VERSION` was bumped
to **24** and all spectra cached under earlier versions are stale.

The pairing is guarded by the dependency floor plus the cache label. The
`vulcan-retrieval` floor in `pyproject.toml` is `>=0.12.1`, the release that
carries `_gravity_profile_invsq`; 0.12.0 is deliberately excluded, because it
used ExoJax's own `gravity_profile`, which is linear in `1/r` and removes only
about half the constant-g bias. Note what the floor does not cover: nothing in
this tool version-gates the profile at run time, so an editable or
manually-pinned install below that floor would still produce spectra that do not
match this version's cache labels.

## Backend configuration

The Pandeia engine runs in its own conda environment and is deliberately not a
package dependency.

**The default backend (`current`) is the Pandeia 2026.7 matched triple**
(`pandeia.engine` 2026.7 + `pandeia_data-2026.7-jwst` +
`pandeia_psfs-2026.7-jwst`), the STScI-supported release; the worker refuses a
mismatched triple. **The committed PandExo parity artifact is a gate-evaluated
PASS on 2026.7** (`tests/parity/outputs/REPORT.md`, worker v9, both sides on
the same triple, PandExo master at the pinned commit): a fixed-configuration
estimator comparison in which configuration, timing, wavelength grids, and
extracted flux matched, and the remaining sigma difference is the noise model,
with this tool conservative. Since 2026-08-09 the gate also requires exact
group agreement on short ramps (either side at 3 groups or fewer), bounds the
per-integration-time gap, and fails any matched sigma ratio outside an
anomaly band -- a same-day review showed the earlier +-1-group tolerance
passing a wrong 1-vs-2-group SOSS selection with a 7x sigma discrepancy.
The public Space runs this backend.

`JWST_TOOL_BACKEND=archival_2026_2` selects the previous 2026.2 tuple under its
honest archival name, for reproducing older results only. Its own parity
artifact predates the fail-closed gate and was never gate-evaluated. STScI
labels 2026.2 archival and unsuitable for planning new proposals. See
[`decision_records.md`](decision_records.md) (its S2-04
"stay on 2026.2" decision is superseded by the 2026.7 migration).

The pinned pandeia 3.0 / `pandeia_data-3.0rc3` `legacy` backend was removed.
Reproducing a pre-2026.2 (3.0-era) run now requires checking out a commit
that still carries it.

Three guarantees hold across backends. The worker refuses to run a mismatched
engine and reference-data pair. Every result and cache file records the exact
engine, reference-data, and worker versions in a provenance block. That block is
hashed into the cache key, so switching backends invalidates caches automatically.

Per-machine overrides: `JWST_TOOL_PANDEIA_PYTHON`, `JWST_TOOL_PANDEIA_REFDATA`, and
`JWST_TOOL_PANDEIA_PSF_DIR`. They are resolved in `src/jwst_tool/instruments.py`
with loud failures. The PICASO reference tree is selected by
`JWST_TOOL_PICASO_REFDATA`.

---

# PICASO provider + climate T-P mode: scope, limits, and roadmap

Status: v18 (tool 0.12.0, 2026-07-20). This is the versioned record of what
the PICASO integration ships, the science limits it states, the measured
findings behind its design decisions, and the features deliberately deferred
(with re-entry sketches). The GUI links here wherever a limit bites.

## What shipped (v18)

- **`chem_provider="picaso"`**: PICASO 4.0.1 thermochemical-equilibrium
  chemistry (Visscher 2121 grid, 101 T x 21 P x 50 species per node file,
  13 [M/H] x 6 C/O nodes) as a second forward-model engine. Everything
  downstream is SHARED with the VULCAN-JAX engine: ExoJax RT, the one
  binning operator, Pandeia noise, detect/fisher. Equilibrium-vs-kinetics
  on identical machinery is the science axis.
- **Composition blending**: the stock `chemeq_visscher_2121` picks the
  NEAREST node file (no composition interpolation -- FD rows through it
  would be exactly zero). The provider blends the 2x2 bracketing node files
  bilinearly in ([M/H] dex, C/O) of the log10 abundances, then interpolates
  (T, P) with exactly picaso's own `chem_interp` convention (bilinear in 1/T
  and log10 P; verified to 4e-15 dex against native picaso). Outside the
  tables the provider REFUSES where picaso would silently extrapolate.
- **`tp_mode="picaso_climate"`**: the PICASO radiative-convective climate
  solver (preweighted correlated-k tables, 196 bins x 8 g-points) as a T-P
  mode for BOTH providers -- including full VULCAN kinetics (photochemistry,
  SO2) running on the PICASO RCE profile. Certified, cached
  (`output/picaso_climate_cache/`, atomic writes + process-safe locking with
  stale-lock recovery), shared between providers.
- **Fisher rows** (`jac_method="fd"` only): lnZ / dlnCO as symmetric
  two-cell interpolant secants with a one-sided-secant kink gate
  (`picaso_chem.FD_KINK_TOL = 0.5`, hard error); T-P rows by table
  re-equilibration; `Tint_cl` (climate mode) as a full-climate-re-solve row
  ("fd-climate", 4 certified solves, h = 15 K). Provenance per row:
  `jac_row_method`, `fd_h`, `fd_err`, `fd_kink`, `fd_grid_cell`.
- **Certificates**: the provider writes `picaso_cert_json` (blend nodes +
  weights, per-layer pre-normalization gas sums, realized gas C/O, floored
  entries, suspect-cell hits); climate mode adds `climate_provenance_json`
  (convergence + flux metric + gradient envelope + convective-zone
  structure). `jwst-tool data` gains a "PICASO provider data" section;
  reference data is selected ONLY by `JWST_TOOL_PICASO_REFDATA` (no baked-in
  path) and fingerprinted by CONTENT into every cache key.
- **Native-RT cross-model harness**: `tests/parity_picaso/` compares picaso's
  own `get_transit_1d` against the tool's ExoJax RT on one identical state --
  offline only, never a production path. It is NOT a parity result: all three
  declared targets are missed, so it records a cross-model DISCREPANCY and does
  not validate absolute spectral agreement. MEASURED (2026-07-20, W39b
  isothermal 1100 K, shared absorbers H2O/CO2/CO/CH4, R = 100 bins): broadband
  offset -2207 ppm (reference-radius conventions; removed), then median
  |residual| 688 ppm, p95 1540 ppm -- OUTSIDE the up-front targets (150/400
  ppm), dominated by the opacity sources (the native DB is the resampled
  R=15,000 'default_3.3' product; the tool uses HITRAN through exojax
  PreMODIT). **Correction 2026-08-03:** the old attribution here also blamed
  "g(z)-vs-constant-gravity conventions". That is wrong for current code -- the
  tool's RT moved to an inverse-square profile in the 2026-07-28 audit, so BOTH
  sides now integrate altitude on g(r) ~ 1/r^2 and gravity is not a difference
  between them. The archived numbers predate that change and are STALE pending a
  rerun. Reported in tests/parity_picaso/outputs/REPORT.md; the size of the
  envelope is exactly why the production path never mixes the two RTs.

## Stated science limits (intrinsic, not bugs)

- **No SO2 / S2 / S8 / CS2** under the picaso provider: equilibrium sulfur
  sits in H2S / OCS. The WASP-39b photochemical-sulfur headline science
  stays VULCAN-only, and so does CS2 (a v25 extra: it is a photochemical
  sulfur carrier with no Visscher gas column). The GUI removes them from
  the menus; `canonical_params` refuses them loudly. Since v20 both
  equilibrium sulfur carriers ARE modeled: H2S is in the picaso BASE RT
  set (it is the dominant equilibrium S reservoir at 700-1500 K and part
  of picaso's own default species set; measured max removed-molecule
  signal 1259 ppm on the W39b 10x-solar default) and OCS is an extra
  under BOTH engines (nu3 ~4.85 um inside G395H/PRISM; 69 ppm max on
  the same default; the GUI selects every extra by default). The SNCHO
  network names the species COS; the registry token maps to the table's
  OCS column via `picaso_chem.VULCAN_TO_TABLE`. The v25 hydrocarbon
  extras C2H4 / C2H6 ARE available under both engines: the Visscher
  tables carry both gas columns.
- **Species tabulated but spectrally invisible**: the Visscher tables
  carry Na, K, TiO, VO, Fe, FeH, CrH, PH3, N2 (and more) -- all counted
  in the mean molecular weight, none in the opacity. Na/K are doubly out
  of scope (the RT native grid starts at 1 um, above both resonance
  doublets, and exojax atomic lines + Allard wing profiles are a
  project); PH3 has no line list on disk (HITRAN room-T entry would be
  the unblock) and would need a provider-split extras menu; TiO/VO/FeH
  and H- matter only above ~1600-2000 K photospheres, which the
  ultra-hot warning already covers. N2 is spectrally inactive and
  correctly mmw-only.
- **C/O hard-capped at 1.10** by the Visscher grid (VULCAN handles 2.0
  structurally). Metallicity spans 0.1-100x solar ([M/H] in [-1, 2]).
- **Composition derivatives are TWO-CELL INTERPOLANT SECANTS**, not local
  derivatives: the grid nodes are kinks of the interpolant. MEASURED
  (2026-07-20/21, W39b defaults): at the C/O = 0.55 NODE the one-sided
  dlnCO secants disagree by 152% of the symmetric row, so the kink gate
  HARD-ERRORS there -- by design. The kink is the TABLE's own physics: at
  1 bar the per-cell abundance slopes are nearly symmetric (ratios
  1.0-1.7), but at 1 mbar -- where transmission forms -- the right cell
  [0.55, 0.82] carries 7-14x the left cell's slope (d log10 X / d ln C/O:
  CH4 +1.2 vs +9.6, H2O -0.9 vs -9.3, CO2 -0.6 vs -9.1) because the sharp
  upper-atmosphere CH4/H2O equilibrium transition sits INSIDE that cell.
  A smoother interpolant does NOT fix this: PCHIP along the C/O axis was
  evaluated and rejected (its node derivative is the interpolant's slope
  convention, not data, and its leave-one-node-out p95 error is WORSE than
  linear for CH4/HCN, 0.093 vs 0.019 dex). The only real fix is denser
  upstream C/O sampling (see the upstream report). At the mid-cell
  C/O = 0.50 (the GUI default for the provider) the whole stencil stays
  inside one cell: kink 0.089, h-vs-2h 0.003. lnZ at the [M/H] = +1.0 node
  passes (kink 0.17; the metallicity cells are symmetric). Cross-node
  blend accuracy (leave-one-node-out at feh0.5/co0.46): median ~0.01-0.03
  dex, p95 <~ 0.05 dex for major species; worst ~0.2 dex (CO2); ~1 dex
  locally at the K condensation edge.
- **Climate composition is EXACT-CK-NODE only**: the correlated-k tables
  carry no composition interpolation, so climate mode accepts only shipped
  nodes (extreme metallicities +-1.5/+-2.0 ship only C/O 0.27-0.82).
  Consequence: under the PICASO engine, climate-mode C/O (dlnCO)
  constraint rows are REFUSED at the API (v18.1 GUI review): exact-node
  composition means the stencil always straddles a table kink, so the
  mid-run kink-gate failure was guaranteed. The GUI does not offer the row
  there; the VULCAN engine constrains C/O in climate mode normally.
- **One-way coupling**: climate T-P is solved with PICASO equilibrium CK
  opacities, then post-processed with either engine's chemistry and ExoJax
  RT. The chemistry NEVER feeds back into the climate opacity. This is not
  radiative-chemical self-consistency and is never labeled as such.
- **`climate_rcb` is a model assumption, bounded by certification on the
  shallow side** (measured in full 2026-07-21, W39b default node): rcb 45
  and 50 FAIL the TOA flux-balance gate (metrics 4.2e-2, 2.8e-3 > 1e-3)
  and rcb 55 drives T(7.6 bar) to 3074 K, refused by the T-window gate --
  the shipped certification already rejects the shallow branch. Every
  certified deep guess (60/65/70/75) is Schwarzschild-CONSISTENT against
  the solver's own adiabat table (radiative margins +0.17..+0.23 to the
  zone top; no unstable radiative layers), so the deep-adiabat attachment
  is genuinely degenerate in a static RCE solve (the classic irradiated-
  planet non-uniqueness, set by interior entropy/evolution; T at 1 bar:
  1820 K at rcb 60 vs ~1595 K at 65-75; T at 7.6 bar: 2832/2514/2075/
  1790 K). OBSERVABLE consequence: T(0.1 bar) is identical across
  certified choices, but the deep scale height lifts the whole transit
  spectrum broadband by ~360-630 ppm median (largely absorbed by the lnR0
  reference-radius nuisance the forecast machinery profiles out), while
  EMISSION is genuinely sensitive in deep-probing windows (up to ~86% of
  Fp at specific wavelengths, median ~0.05%) -- the GUI warns on the
  emission + climate combination. rcb is cache-keyed and the Tint_cl row
  differentiates at FIXED rcb. The climate solve itself is
  bit-deterministic (repeat and fresh-opacity reruns: exactly 0 K).
- **Pressure policy**: the equilibrium tables and the climate grid start at
  1e-6 bar; above it the topmost layer is held constant (the sibling
  interp_map's documented edge clamp -- it logs the clamped layer count on
  every run). The provider chemistry grid spans exactly 1e-6 bar to the
  chemistry bottom (7.6 bar); the VULCAN+climate path goes through the
  file-mode top-clamp logging. Nothing extrapolates silently.
- **Certified domain**: v1 climate mode is certified around the WASP-39b
  configuration. Other planets / nodes / rfacv values are dynamically
  convergence-gated (the certification refuses anything unconverged,
  flux-imbalanced, gradient-pathological, or top-convective) and should be
  treated as experimental until `tests/live/test_picaso_live.py`'s smoke
  matrix has been run for them.
- **T-window interaction**: climate profiles are truncated/interpolated to
  end exactly at the 7.6-bar chemistry bottom (W39b default: 2832 K there,
  inside the 320-2980 K opacity window with ~150 K margin). Hotter
  planets/Tint may legitimately REFUSE at the window -- a stated envelope
  limit, never clipped.

## Measured data-quality findings (upstream-reportable)

- **One corrupted cell** in `sonora_2121grid_feh1.0_co0.55.txt` at
  (T = 900 K, logP = -5.523). Full anatomy (2026-07-21, superseding the
  first characterization): EVERY species in the row is uniformly deflated
  by ~x0.747 (H2 0.7471, He 0.7471, H2O 0.7477, CO 0.7474, Na 0.7467 ...
  vs T-neighbor interpolation) -- a spurious ~25% phantom abundance entered
  the row's normalization at generation -- plus two junk residues: VO
  ~9.9e6x too high (5.2e-12 vs ~5e-19) and CrH ~4.8e4x (both
  spectroscopically inert and not RT species). The same cell is clean in
  all four neighboring node files. Handling (v18.1): the CONTENT-GUARDED
  `picaso_chem.KNOWN_TABLE_CORRECTIONS` registry replaces the row by its
  T-neighbor log-mean while the file still hashes to the registered
  corrupt bytes (an upstream fix makes the entry a no-op); every
  application is recorded in the certificate/npz and shown in the GUI.
  Measured bound: the correction differs from the previous renormalize-
  through treatment by <= 2.2 ppm worst-case (900 K profile), 0.0 ppm on
  the 1100 K default. Any OTHER isolated anomaly (clean T-neighbors)
  inside an evaluated span now REFUSES loudly -- unvetted corruption is
  never renormalized through; the systematic extreme-metallicity cold-T
  deficits (equally-low neighbors) remain renormalize + certificate.
- **Extreme-metallicity gas sums**: the |feh| >= 1.5 files sum to 0.86-0.98
  at T <~ 500 K (documented missing-species behavior). The provider
  renormalizes per layer (the upstream-recommended treatment), records
  pre-normalization sums, refuses below `GAS_SUM_MIN = 0.70`, and flags
  below `GAS_SUM_WARN = 0.98`.
- **Gas accounting**: ions and electrons are COUNTED in the gas total and
  the mean molecular weight (they are gas-phase number density; e- mass
  5.49e-4 amu); only graphite is excluded as a condensate (renamed
  `C-gr_l_s` so the shared RT condensate mask handles it exactly like
  VULCAN's reservoir columns).
- **Realized gas C/O != the file label below ~1700 K** (silicate
  condensation sequesters O: label 0.46 -> gas-phase 0.55 at 800-1200 K;
  matches at 2000 K). Real physics, recorded in the certificate; label
  comparisons only above `CO_CHECK_T_K = 2000 K`.
- **`chemeq_visscher_2121`'s docstring says 20 pressures; the files carry
  21** (2121 = 21 x 101). The loader validates the real shape.
- **Native transmission returns all-NaN when `gravity()` is given bare
  gravity**: the altitude integration needs planet.mass (g = GM/z^2). The
  parity script documents this trap; always pass mass + radius.

## Deferred features (why, and how to re-enter)

1. **Quench / lnKzz row** (the reason there is NO lnKzz under picaso: it
   has no effect in equilibrium, so the row would be identically zero).
   Re-entry: PICASO 4's `atmosphere(quench=...)` / `find_kzz` /
   `adjust_quench_chemistry` machinery restores a physical lnKzz direction
   (quench approximation vs VULCAN's full kinetics -- a scientifically
   interesting comparison axis). Needs its own FD smoothness study (quench
   levels move discretely with Kzz) and compatibility rules before any
   Fisher row is certified.
2. **Cloudy climate (virga)**: the reference tree's `virga/` directory is
   EMPTY; cloudy climate solves would download condensate files and need
   their own certification. Re-entry: populate virga refdata, extend
   `climate_refdata_fingerprint`, add a virga toggle with its own
   compatibility matrix.
3. **Sonora guess profiles**: `sonora_grids/` is empty, so the climate
   guess is a deterministic analytic Guillot profile (measured: converges
   in ~1 min on W39b). Re-entry only if some configuration cannot converge
   from the analytic guess (then: a guess ladder, never warm-starting from
   previous solves -- determinism is a certification property).
4. **PICASO-native RT as a GUI backend**: rejected for production (decision
   2026-07-20): the local opacities.db is R=15,000 with only 10 line
   species (no NH3/HCN), and a second RT path would break the
   one-measurement-operator rule. The parity harness
   (`tests/parity_picaso/`) is the supported use.
5. **Off-node climate composition**: would require blending correlated-k
   TABLES (not log abundances) or on-the-fly k-table mixing; out of scope
   for v1. Climate mode stays exact-node.
6. **Per-side (left/right) composition derivatives at nodes**: the kink
   gate currently refuses; reporting both one-sided secants as an interval
   is a possible v2 presentation.
7. **`jwst-tool fetch` for PICASO refdata**: the reference tree is
   user-supplied science data (Zenodo: chemistry/CK 10.5281/zenodo.13733116,
   opacities 10.5281/zenodo.14861730); datacheck reports it, fetch does not
   download it.
8. **AD through climate mode**: refused in v1 (`jac_method="ad"` +
   picaso_climate); the VULCAN warm-jvp rows on a fixed climate T-P would
   be well-defined, but the combination is uncertified.

## Live validation

`JWST_TOOL_RUN_PICASO_LIVE=1 python -m pytest tests/live -q` runs the
measured-2026-07-20 battery: within-node native parity, leave-one-node-out
blend accuracy, lnZ FD closure, picaso-vs-vulcan spectrum sanity, and the
climate smoke matrix (W39b x rfacv {0, 0.5, 1}, solar node, HD 189733 b,
WASP-107 b). The native-RT CROSS-MODEL report (outside target, not a parity
result) lives in `tests/parity_picaso/outputs/REPORT.md`.

## v18.1 (tool 0.12.1, model-cache v19): review-response hardening

Release-gate fixes from the 2026-07-21 external code review (all verified by
reproduction or inspection before fixing):

- **GUI Fisher defaults**: both Fisher multiselects crashed with
  StreamlitAPIException under the PICASO engine (default lnKzz not in the
  provider's menu). Defaults are now filtered by the live menu and the
  widget keys carry the provider; the exact crash paths are pinned in
  test_app_smoke.
- **Climate lock lifecycle**: the lock file is now NEVER unlinked. The
  previous stale-lock breaking (unlink + retry) created the classic
  two-inode double-lock race (two processes each holding an "exclusive"
  flock on different inodes of the same path, reproduced in review), and
  could fire on a LIVE holder past the age threshold. flock's own
  release-on-death is the recovery mechanism; a live slow holder is waited
  on (bounded, loud timeout), never broken. Verified: two concurrent
  uncached solves share one computation (80.1 s solver / 79.4 s waiter,
  bit-identical results); a kill -9'd holder releases to the survivor.
- **Cache-load revalidation**: loading a cached climate profile now re-runs
  every gate evaluable from the stored data (structure, gradient envelope,
  stored flux metric, convective-zone sanity) -- loading is never weaker
  than solving.
- **Corrupt-cell policy** (above): catalogued content-guarded correction +
  isolated-anomaly refusal replaced the renormalize-and-warn treatment.
- **Data-status scan off the rerun path**: the full datacheck report
  (including the ~2.5k-entry PICASO manifest stat pass, slow on remote
  Space volumes) is cached for 5 minutes with a manual refresh button --
  it no longer runs on every Streamlit interaction.
- **Public-instance protection**: a cross-process flock semaphore caps
  concurrent heavy subprocesses (forward + ETC, adjoint) at 2 per
  instance; further launches are declined with a message instead of piling
  onto shared hardware. Slot files follow the same never-unlink lifecycle.
- **Deployment reproducibility**: upload_data.sh stages picaso-reference
  (from JWST_TOOL_PICASO_REFDATA, APFS clonefile when possible) and
  generates its manifest.json; the bootstrap fallback exports
  JWST_TOOL_PICASO_REFDATA when the snapshot brought the tree.
- **Provenance made exact**: the certificate stores the per-layer
  pre-normalization gas-sum ARRAY; fd_grid_cell records BOTH cells a
  node-centered stencil traverses; the npz ymix uses the same gas
  normalization as the RT (graphite/_l_s excluded); the stellar-grid part
  of the climate fingerprint is documented as a name+size MANIFEST
  (content hashes cover the CK table, continuum DBs, climate_INPUTS, and
  config/version files); the kink-gate refusal prints both one-sided
  secant scales.

The upstream-reportable items are drafted in `docs/decision_records.md`
(upstream PICASO report part; not posted anywhere without explicit approval).

### v18.1 follow-up (GUI review, 2026-07-21)

A 4-dimension multi-agent review of the rendered GUI (layout, text,
defaults, promised-vs-actual behavior; 31 confirmed findings) produced two
behavior fixes on top of the text pass:

- **orbit_au is no longer normalized away in climate mode**: the photo-off
  normalization treated the semi-major axis as photolysis-only (true
  pre-v18), silently running every picaso-engine climate at the planet's
  default orbit. It now gets the same climate-mode carve-out as the star
  identity; sflux stays normalized (genuinely photolysis-only).
- **Mode-aware C/O defaults**: bare API calls now default C/O to 0.55 in
  climate mode (the node -- a bare climate request no longer refuses its
  own default) and 0.50 under the picaso engine (mid-cell), with
  CO_BASELINE unchanged for VULCAN.
- dlnCO under picaso + climate is refused at the API (above), and the GUI
  menus exclude it there.
- The remaining findings were GUI truthfulness fixes: engine-aware and
  observable-aware captions everywhere (run status, RT section, main
  header, intro), inert widgets now disabled with reasons (stellar UV
  under PICASO; Rayleigh + transit-chord in emission), engine-aware
  Fisher timing text, plain-language certificate captions, clean node
  labels, and event-word (transit/eclipse) consistency in the results.

---

# jwst-tool effective chemistry config vs upstream VULCAN master (W39b)

Date: 2026-07-15. Method: field-by-field diff of `VULCAN-master/vulcan_cfg.py`
(W39b-configured, the parity oracle) against the tool's EFFECTIVE VULCAN-JAX
cfg at GUI defaults (`W39b.yaml` + `vulcan_chem` profile application + the
tool's `cfg_overrides`, forward v12). Script: session scratchpad `cfg_diff.py`.

## Fields identical in both (the reassuring bulk)

Elemental abundances (C_H 2.95e-3, O_H 5.37e-3, N_H, S_H, He_H), solver
tolerances (rtol 0.2, atol 0.1, mtol, mtol_conv), step counters (count_max
3e4, count_min 120, trun_min), convergence thresholds (yconv_cri 1e-2,
yconv_min 0.1, slope_cri, flux_cri), photolysis geometry (sl_angle 83 deg,
f_diurnal 1.0), photo update cadence (ini/final 5/5), EQ initialization +
FastChem file, stellar UV file, boundary-condition files, use_photo on,
use_moldiff on, use_vm_mol off (tool PINS it since v11; upstream YAML default
flipped on 2026-07-14), condensation/settling off in both.

## Real deviations (tool vs master), with status

| Field | Master (W39b) | Tool effective | Status |
|---|---|---|---|
| `atm_type` | `file` (GCM evening-terminator T-P) | `isothermal` structural + live `tp_eval` (iso/Guillot) | DELIBERATE, documented: GCM baseline removed 2026-07-13; every planet gets an explicit T-P. Tool answers differ from Tsai-2023-style runs by construction. |
| `Kzz_prof` | `file` (GCM Kzz(z)) | `const`, GUI default 1e9 | DELIBERATE (same removal). Master's `const_Kzz=1e10` is inert there. Note the GUI default 1e9 is a choice, not Tsai's profile; slider spans 1e6-1e12. |
| `dt_max` | 1e17 (`runtime*1e-5`) | 1e11 | DELIBERATE, documented (validated state-preserving; prevents the adaptive-dt balloon; master's own uncapped value implicated in the photo-off blow-up, see VULCAN-JAX/docs/vulcan_jax_notes.md (2026-07-15 photo-off entry)). |
| `nz` | 150 | 100 (GUI default; 150 available) | DELIBERATE fast-tier default; use 150 + yconv 1e-3 for final numbers (documented deltas: MIRI LRS halves at 100/1e-2). |
| `conver_ignore` | 13 heavy hydrocarbons (C2H2, C6H6, ...) | `['HC3N']` only | SETTLED 2026-07-30 (was FLAGGED "provenance unclear, review"): the 13-species list exists in NO upstream repository (the local VULCAN-master copy was contaminated with it); fetched exoclime master ships `[]`, shami vm_branch ships `['HC3N']`, and `[]` vs `['HC3N']` measured behaviorally IDENTICAL on HD189/HD209/W39b. The stricter JAX-side gate stands; nothing to adopt. |
| `network` | `NCHO_photo_network.txt` (current oracle parking) | `SNCHO_photo_network.txt` | Not a tool deviation: SNCHO is the Tsai 2023 W39b science network (sulfur/SO2). The workspace master cfg is parked on NCHO for the oracle tests. |
| `wall_clock_max` | 1800 s | 3600 s | Runner backstop doubled on the JAX side. Benign (a backstop, not physics). |
| `Tiso` | 1000 (inert; atm_type=file) | 1100 (GUI default near W39b Teq) | Inert in master; not a comparison. |
| `gs` vs `Mp` | gs directly | `Mp = gs*Rp^2/G` | Exactly equivalent by construction (checked 2026-07-15). |

## VULCAN-JAX-only feature toggles (confirmed inert at tool defaults)

`use_chunked_runner`, `use_fix_H2He`, `use_fix_all_bot`, `use_ini_cold_trap`,
`use_pi_controller`, `use_sat_surfaceH2O`, `use_adapt_rtol` all False;
`use_hybrid_vm_mol` pinned False by the tool (v11). No silent feature is
active that master lacks.

## Tool-level constructions with no master counterpart

- Two-stage warm-started composition continuation (lnZ/dco steps from the
  relaxed column) instead of cold re-solves; validated around the O-rich
  baseline; refused elsewhere (v12 gates).
- The differential fixed-O `dco` knob (exact delta-ln C/O through the solve,
  AD-capable). Master's own convention IS fixed-O (its cfg anchors `O_H` and
  edits `C_H`), so the tool's differential direction matches the upstream
  convention; the perturbative form adds the C/O < 1 ceiling (b_z bound).
- The structural `co_baseline` mode (v12) reproduces master's workflow
  exactly: pin `C_H = co * O_H`, FastChem re-initializes at the target C/O
  (any value incl. C-rich); detection-only.
- The longdy convergence gate refuses non-certified states loudly (master
  reports but does not refuse).

## Engine-level (VULCAN-JAX vs master) differences

Documented separately: `VULCAN-JAX/docs/validation.md` (VULCAN-JAX
parity and bug guide) and `jax_paper/paper/notes_gaps.md` (ranked gaps;
headline: reservoir projection default-on in JAX vs none in master). Those
apply to every consumer of VULCAN-JAX, not just this tool.
