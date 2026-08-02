# PICASO forward-model provider in vulcan-jwst-tool: integration plan

Status: PLAN ONLY (2026-07-20). Nothing implemented. Explored live against
picaso 4.0.1 in the tool's own env, the two local PICASO reference trees, and
both repos' forward-model seams.

## Goal

Add PICASO as a second forward-model provider in vulcan-jwst-tool, alongside
the VULCAN-JAX kinetics engine:

- PICASO supplies the atmospheric state (chemical-equilibrium mixing ratios on
  the tool's canonical T-P), and the existing ExoJax machinery does everything
  downstream: PreMODIT opacities, CIA, clouds, ArtTransPure/ArtEmisPure,
  native-R LSF, one binning operator, Pandeia noise, detect/fisher.
- Jacobians via the existing certified central-FD machinery only
  (`jac_method="fd"`); `"ad"` is refused because PICASO is numpy/numba and not
  differentiable.

## What was verified locally (2026-07-20)

- picaso 4.0.1 imports in the SAME base env as jwst_tool + exojax 2.2.3
  (`/opt/homebrew/Caskroom/miniforge/base`); no subprocess worker needed.
  A separate `picaso_base` env holds an old picaso 3.3 egg (pandeia legacy
  backend only, ignore it).
- The shell `picaso_refdata` env var is STALE (`~/Documents/picaso/reference`
  does not exist). Two real v4.0 reference trees exist:
  - `/Users/imalsky/Desktop/Emulators/RT-Project/picaso/reference` (chemistry +
    climate_INPUTS + sonora_grids + stellar_grids + virga; the tool's
    `data/cdbs/grid/phoenix` symlink already points into its stellar_grids).
  - `/Users/imalsky/Documents/Astronomy-Software/picaso/reference` (also has
    `opacities/opacities/opacities.db` + continuum.db, needed only for
    PICASO-native RT, i.e. phase 4).
- Live chemistry run works: `inputs()` + `gravity()` + `guillot_pt()` +
  `chemeq_visscher(...)` returns a per-level DataFrame with 34 gas columns:
  H2, He, H2O, CO, CO2, CH4, NH3, H2S, HCN, C2H2, N2, PH3, OCS, SiO, C2H4,
  C2H6, Na, K, TiO, VO, Fe, FeH, MgH, CrH, Cs, Rb, H, ions/e-, graphite.
  NO SO2, no S2/S8 (equilibrium; sulfur sits in H2S/OCS).
- `chemeq_visscher_2121(cto_absolute, log_mh)` is the current-generation call:
  ABSOLUTE C/O basis (matches the tool's v11+ structural `co_ratio`
  convention). The plain `chemeq_visscher` is deprecated and points at the
  1060 grid with RELATIVE C/O; never use it.
- Grid extent (visscher_grid_2121 files): feh nodes
  {-2.0,-1.5,-1.0,-0.7,-0.5,-0.3,0.0,+0.3,+0.5,+0.7,+1.0,+1.5,+2.0};
  C/O nodes {0.14, 0.27, 0.46, 0.55, 0.82, 1.10}. Within each file:
  20 pressures 1e-6..3e3 bar, 101 temperatures 75..6000 K.
- CRITICAL numeric fact: `chemeq_visscher_2121` (and `_1060`) selects the
  NEAREST grid file in (feh, C/O). There is NO interpolation across the
  composition axes; only (T, P) within the chosen file is interpolated
  smoothly (`chem_interp`: bilinear in log10 P and 1/T of log abundances).
  Naive central-FD lnZ/dlnCO rows through the stock API are therefore
  EXACTLY ZERO inside a grid cell and a step function across cell borders.
  The provider must do its own composition blending (design below).
- PICASO 4 also has quench machinery (`atmosphere(quench=...)`,
  `find_kzz`, `adjust_quench_chemistry`), rainout/cold-trap options, and a
  full RCE `climate` mode. Phases 2-3 material.

## Where PICASO plugs in (the seam)

The 2026-07-20 exploration of both repos confirmed the RT stage is already
chemistry-agnostic and the seam is narrow:

- `vulcan_forward.exojax_rt.build_rt_model(profile)` (the engine moved to the
  `vulcan-forward` distribution on 2026-07-29; it was
  `retrieval_framework.forward.exojax_rt` when this plan was written) +
  `rt.transmission_depth_r(vmr, vmr_h2, T_art, mmw_art, lnR0, vmr_he=...,
  cloud=..., mie=...)` (exojax_rt.py:269/442) consume PLAIN per-layer arrays:
  a VMR dict keyed by `config.MOLECULES` names, H2, He, T, mmw. No VULCAN
  state object crosses the seam. Emission (`build_emis_model`) is the same.
- `interp_map.make_to_art(p_bar_source, rt.p_art_bar)` bridges ANY source
  pressure grid (log-P interp; loud on bottom-coverage failure).
- The VULCAN-specific glue is confined to three thin adapters
  (`jwst_tool/forward.py:_art_profiles`, retrieval's `aux_from_y`,
  `sensitivity.forward`) plus `forward._assemble_chem` (forward.py:1233-1459).
- Downstream is provider-agnostic already: `binning.py`, `noise.py`,
  `detect.evaluate_mode`, `fisher.py` consume pure arrays from the npz
  (`depth`, `depth_wo`, `jac`, `jac_names`, `mols`, `wl_um`). No change.
- RT-only Jacobian rows (lnR0, cloud, Mie) are already chemistry-independent.

So the work is: a new chemistry provider module + a provider branch in
`forward.py` + GUI/datacheck/provenance. `exojax_rt.py`, `interp_map.py`,
`binning`, `noise`, `detect`, `fisher` are untouched.

## Architecture

### New module `src/jwst_tool/picaso_chem.py` (tool-local provider)

PICASO lives in the TOOL, not in retrieval_framework: it is non-differentiable
so it is useless to the SMC/MALA retrieval, and keeping it out of the sibling
avoids a heavy dependency and a version-floor bump on the HPC side.

Responsibilities:

1. Refdata resolution, loudly: `JWST_TOOL_PICASO_REFDATA` env var (pattern:
   `instruments.py` path roots), default pointing at the RT-Project tree.
   MUST set `os.environ["picaso_refdata"]` (and `PYSYN_CDBS`) BEFORE importing
   `picaso.justdoit`, because picaso reads them at import time and the user's
   shell values are stale. Lazy import inside the provider only; the numpy-only
   test suite never imports picaso.
2. Composition-blended equilibrium (the fix for nearest-neighbor):
   - Load the 2x2 surrounding `sonora_2121grid_feh*_co*.txt` node files for
     the requested (log_mh, cto_absolute), evaluate each on the tool's T-P
     profile via picaso's own (T,P) interpolation (`chem_interp` per node
     table), then blend the per-species LOG abundances bilinearly in
     (log_mh, cto). Roughly 100-150 lines; no patching of picaso.
   - Result: chemistry is continuous piecewise-linear in composition, so
     central-FD lnZ/dlnCO rows are well-defined table secants and the
     existing h-vs-2h gate + Richardson machinery works as designed.
   - Honesty note for provenance: the composition derivative is the table
     secant over the local grid cell (0.3-0.5 dex in feh, 0.09-0.28 in C/O),
     not a local analytic derivative. Record the bracketing nodes per
     composition row in the npz (new `fd_grid_cell` field or similar).
     The default baseline (10x solar = feh +1.0, C/O 0.55) sits exactly ON
     nodes; blended interp is continuous there with a kinked derivative, and
     the central difference cleanly returns the average slope (h-consistent).
   - The tables floor uncalculated low-T abundances at 1e-50; log-blending
     with a floored node can produce huge ln-slopes on species that are
     physically absent. Harmless for the spectrum (VMR ~ 0) but the provider
     should mask floored cells (treat <= 1e-45 as absent) so FD rows are not
     polluted by floor artifacts.
3. Output contract (mirrors the narrow RT seam): a namespace with
   `p_bar` (provider grid, must cover the ART bottom; build it from the tool's
   T-P span, e.g. 1e-8..2e1 bar, ~90 levels), `T`, per-RT-molecule VMR
   profiles keyed by `config.MOLECULES` names, `vmr_h2`, `vmr_he`, `mmw`
   (computed gas-only from all 30+ gas columns with a species-mass table;
   exclude graphite and ions, mirroring the v17 gas-mask hygiene), plus the
   certification block (below).
4. Certification (replaces longdy/conv_normal; fail loud):
   - (log_mh, cto) strictly inside the grid extent; cto > 1.10 or outside
     feh [-2, 2] refused (with the FD stencil envelope check: baseline must
     be >= 2h from an edge, same rule as the existing composition envelope).
   - T-P inside the table (75..6000 K, 1e-6..3e3 bar) AND inside the RT
     window (300..3000 K) - reuse `_check_t_window`.
   - Gas VMR closure: sum over gas columns within tolerance of 1 per level.
   - Emit provider-tagged npz certificate fields (see npz section).

### `forward.py` changes

- `canonical_params`: new canonical key `chem_provider` ("vulcan" | "picaso",
  default "vulcan"). Bump `_VERSION` to 18. Under "picaso":
  - `jac_method="ad"` refused (message: PICASO provider is FD-only).
  - `use_photo`, `use_condense`, `use_vm_mol`, `use_moldiff`, Kzz knobs,
    boundary-condition knobs, `nz`, `yconv_cri`, sflux/orbit: refused if
    explicitly set non-default, else zeroed/normalized for cache hygiene
    (existing inert-knob pattern, forward.py:1012-1120).
  - `fisher_params` menu: {lnZ, dlnCO} | TP_PARAM_NAMES[tp_mode] | cloud/Mie
    params. NO lnKzz (no mixing in equilibrium; returns in phase 2 as quench).
  - `co_ratio` range under picaso: [0.14+2h, 1.10-2h]; met range
    [0.1, 100] x solar already inside feh [-2, 2] but keep the 2h inset.
  - SO2 (and any RT molecule the provider cannot supply) refused in the
    active molecule set with an explicit message (equilibrium chemistry has
    no SO2; it is a photochemical product). Default molecule set under
    picaso: [H2O, CO2, CO, CH4]; extras menu {HCN, C2H2, H2S, NH3}.
  - Cache key gains picaso version + a refdata fingerprint (version.md hash +
    grid-file manifest hash), same spirit as the pandeia backend fingerprint.
- `_assemble_chem_picaso` (parallel to `_assemble_chem`): builds the tool's
  canonical T-P evaluator (reuse `retrieval_framework.tp_profile.build_tp_model`
  via the existing profile plumbing so isothermal/Guillot/file are one code
  path), the provider pressure grid, and returns a namespace with a
  `state(met, co, tp_params)` closure that re-equilibrates (seconds, vs
  minutes for VULCAN).
- `run_model` provider branch: chem build/solve/certify blocks swap; RT build,
  echo checks, `make_depth_fn`, spectrum + removed-molecule loop, RT-only
  Jacobian rows, and npz write stay SHARED. Import-order footgun: under
  picaso still import `vulcan_chem` before `exojax_rt` (it sets jax x64 at
  import; skipping it would silently run the RT in float32). Cost is import
  time only. (Cleaner long-term: hoist the x64/env bootstrap in the sibling;
  not required for phase 1.)
- FD rows under picaso: lnZ/dlnCO = 4 cheap re-equilibrations + RT calls
  (seconds each; the RT call dominates); T-P rows = re-equilibrate at
  perturbed T (chemistry is T-dependent through the table); lnR0/cloud/Mie
  rows unchanged. All through the same `_fd_row` gate + Richardson. Provenance
  `jac_row_method` stays "fd-central"/"fd-rt"; add the grid-cell record for
  composition rows.
- `_make_progress`: picaso stage list (chemistry is seconds; RT/opacity build
  dominates). Keep the `"[fwd] PROG <frac> <label>"` protocol identical.

### npz / certificate

Keep the array contract identical (detect/fisher untouched). Certificate
fields become provider-tagged: write `chem_provider`, and under picaso write
`cert_gate` ("picaso-eq-grid"), grid-cell bounds, VMR-closure residual;
`conv_stages/accept/longdy/gate` are simply absent (app.py already guards with
`in model` checks; reword the caption provider-aware).

### GUI (`app.py`)

- New top-level "Forward model" selectbox ABOVE "Differentiation method"
  (same gating pattern): "VULCAN-JAX kinetics" | "PICASO equilibrium".
- Under picaso: Differentiation menu locked to fd; the VULCAN chemistry
  section's photochem/transport, Kzz mode, numerical grid, condensation, and
  boundary-condition expanders hidden; Composition + T-P + the whole ExoJAX RT
  section + Science goal + Instrument sections shared as-is. C/O slider range
  swaps to the picaso envelope; SO2 removed from the molecule menu; the
  adjoint panel hidden (VULCAN-only). Convergence-certificate caption becomes
  provider-aware.
- `params` dict gains `chem_provider`.

### datacheck / cli

- `check_picaso_data`: picaso importability + refdata root + chemistry grid
  manifest (13x6 files) + version.md; registered as a new `full_report`
  section so the GUI data panel and `jwst-tool data` pick it up automatically.
  Phase 3 adds climate ck tables; phase 4 adds opacities.db.
- `cli.py` preflight: picaso import check next to the vulcan_jax preflight,
  only when the provider is requested.

## Tests and validation

- Unit (numpy-only suite, no picaso import): provider compatibility matrix in
  `test_forward_params` (refusals: ad, photo, conden, Kzz, SO2, C/O > 1.10);
  composition-blend math on tiny synthetic fixture tables (bilinear blend,
  node-exactness, floor masking, FD-secant correctness against hand values);
  mmw gas-mask; certificate emission.
- Env-gated live tests (JWST_TOOL_RUN_SLOW pattern, Isaac schedules):
  - FD closure on a picaso lnZ row (3 forward runs, mirrors test_closure.py).
  - PICASO-eq vs FastChem-eq cross-check at identical (met, C/O): overlapping
    species (H2O/CO/CO2/CH4/NH3/H2S) should agree in the deep atmosphere to
    interpolation error. This is the natural correctness oracle since
    FastChem is already in the stack.
  - Spectrum-level sanity: picaso-provider vs vulcan-provider W39b-like
    transmission; differences must be attributable to disequilibrium
    (SO2 absent, CH4/NH3 quench above the quench level).

## Phases

1. Core provider (this plan): blended chemeq + provider branch + FD rows +
   GUI + datacheck + cache/provenance + tests. Rough size: provider module
   ~300 lines, forward.py ~200, app.py ~120, datacheck ~60, tests ~250.
2. Quench option: `atmosphere(quench=True)` / `find_kzz` /
   `adjust_quench_chemistry` restores a physical lnKzz Fisher row under
   picaso (quench approximation vs VULCAN's full kinetics: a scientifically
   interesting comparison axis). Also opt-in rainout/cold-trap (distinct
   semantics from VULCAN conden; needs its own compatibility rules).
3. PICASO climate mode: self-consistent RCE T-P as a new tp_mode feeding the
   same ExoJax RT (ck tables exist locally). FD through climate is expensive
   but well-defined; per-row cost ~minutes.
4. Optional cross-model parity harness: full PICASO-native RT spectrum
   (opacities.db) vs picaso-chem + ExoJax on the same state; a validation
   script under tests/parity-style tooling, never a production path.

## Known science limitations (state loudly, do not hide)

- No SO2/S2/S8: the W39b photochemical-sulfur headline case is VULCAN-only.
  PICASO provider is for equilibrium-species science and cross-model
  comparison.
- C/O capped at 1.10 by the Visscher grid (VULCAN handles up to 2.0
  structurally).
- Composition Jacobians are grid-cell secants (table resolution limit).
- Equilibrium only in phase 1: no quenching, no photochemistry, no transport.

## Open decisions for Isaac

1. Accept the SO2/C-rich limitations for the picaso provider (they are
   intrinsic to the Visscher equilibrium grid)?
2. Composition blending lives tool-side (recommended, self-contained) vs
   contributing an interpolating chemeq upstream to picaso?
3. Are phases 2 (quench + Kzz row) and 3 (climate T-P) wanted, and in what
   order? Phase 2 is the cheaper, more retrieval-relevant one.
4. Default refdata root: pin the RT-Project tree via JWST_TOOL_PICASO_REFDATA
   (recommended; the phoenix symlink already depends on it) or consolidate the
   two trees first?
