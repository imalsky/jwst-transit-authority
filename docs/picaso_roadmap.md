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
- **Native-RT parity harness**: `tests/parity_picaso/` compares picaso's own
  `get_transit_1d` against the tool's ExoJax RT on one identical state --
  offline validation only, never a production path. MEASURED (2026-07-20,
  W39b isothermal 1100 K, shared absorbers H2O/CO2/CO/CH4, R = 100 bins):
  broadband offset -2207 ppm (reference-radius conventions; removed), then
  median |residual| 688 ppm, p95 1540 ppm -- OUTSIDE the up-front targets
  (150/400 ppm), dominated by the opacity sources (the native DB is the
  resampled R=15,000 'default_3.3' product; the tool uses HITRAN through
  exojax PreMODIT) plus the g(z)-vs-constant-gravity conventions. This is
  the honest cross-model envelope, reported in
  tests/parity_picaso/outputs/REPORT.md; it is exactly why the production
  path never mixes the two RTs.

## Stated science limits (intrinsic, not bugs)

- **No SO2 / S2 / S8 / CS2** under the picaso provider: equilibrium sulfur
  sits in H2S / OCS. The WASP-39b photochemical-sulfur headline science
  stays VULCAN-only, and so does CS2 (a v25 extra: it is a photochemical
  sulfur carrier with no Visscher gas column). The GUI removes them from
  the menus; `canonical_params` refuses them loudly. Since v20 both
  equilibrium sulfur carriers ARE modeled: H2S is in the picaso BASE RT
  set (it is the dominant equilibrium S reservoir at 700-1500 K and part
  of picaso's own default species set; measured max removed-molecule
  signal 1259 ppm on the W39b 10x-solar default) and OCS is an opt-in
  extra under BOTH engines (nu3 ~4.85 um inside G395H/PRISM; 69 ppm max
  on the same default -- real but marginal, hence opt-in). The SNCHO
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
WASP-107 b). The native-RT parity report lives in
`tests/parity_picaso/outputs/REPORT.md`.

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

The upstream-reportable items are drafted in `docs/upstream_report_picaso.md`
(not posted anywhere without explicit approval).

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
