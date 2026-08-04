# Physics choices and conventions

The modeling conventions this tool adopts, the defaults it ships, and why. Moved
out of the README in 2026-07.

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

## Defaults are the measured structure where one exists

Under the VULCAN kinetics engine, a planet defaults to the temperature-pressure and
`Kzz` table that VULCAN bundles for **that** planet, used for both `T(P)` and
`Kzz(P)`.

| Planet | Default structure | Bundled table |
|---|---|---|
| WASP-39 b | `atm_W39b_evening_TP_Kzz.txt` (Tsai et al. 2023 evening terminator) | Default |
| HD 189733 b | Guillot plus constant `Kzz` | `atm_HD189_Kzz.txt` is selectable but not the default: the solver does not certify a steady state on it at default settings, while the analytic default converges in about 36 s |
| HD 209458 b | Guillot plus constant `Kzz` | Refused. It is a full thermosphere model reaching 2997 K inside the chemistry grid, above the 2980 K opacity ceiling, and it is never clipped |
| WASP-107 b | Guillot plus constant `Kzz` | None bundled |

Two facts are kept separate on purpose: whether a table **exists** for a planet,
and whether a default run on it has been **verified end to end**. A table becomes
the default only once it has been verified, so enabling one can never turn a
working planet into one that errors on arrival. Tables are per-planet and never
substituted; selecting a planet without a usable one tells you why. The PICASO
equilibrium provider keeps the analytic default in every case.

**This matters because the analytic defaults are biased in a systematic
direction.** A constant `Kzz` cannot follow a profile that climbs orders of
magnitude with altitude, and it is the photochemically active upper atmosphere
that pays. Measured against the bundled tables over the chemistry grid at
p < 1 mbar, the constant 1e9 cm²/s default runs 3.8-48x low for WASP-39 b
(`atm_W39b_evening_TP_Kzz.txt`) and 10-17x low for HD 189733 b
(`atm_HD189_Kzz.txt`), always suppressing photochemical products. The factor is
pressure-cut dependent — deeper than ~10 mbar the table falls *below* 1e9 for
WASP-39 b — so quote it with the cut.

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
mismatched triple. **No PandExo parity artifact exists yet for 2026.7**: the
fail-closed gate in `tests/parity/` marks the committed report NOT EVALUATED,
so treat 2026.7 output as unvalidated against PandExo until the suite is rerun
and its report passes the gate.

`JWST_TOOL_BACKEND=archival_2026_2` selects the previous 2026.2 tuple under its
honest archival name. That is the backend the per-mode PandExo parity was
measured on (configuration, timing, wavelength grids, and extracted flux
matched; the sigma difference is the noise model, with this tool conservative),
and it is what the public Space pins for reproducibility. STScI labels 2026.2
archival and unsuitable for planning new proposals. See
[`audit_decisions_2026-07-21.md`](audit_decisions_2026-07-21.md).

Set `JWST_TOOL_BACKEND=legacy` to select the pinned pandeia 3.0 and
`pandeia_data-3.0rc3` pair. It is retained only as an explicit reproducibility
backend, and a legacy run always labels itself LEGACY. It never presents as
2026-backend output.

Three guarantees hold across backends. The worker refuses to run a mismatched
engine and reference-data pair. Every result and cache file records the exact
engine, reference-data, and worker versions in a provenance block. That block is
hashed into the cache key, so switching backends invalidates caches automatically.

Per-machine overrides: `JWST_TOOL_PANDEIA_PYTHON`, `JWST_TOOL_PANDEIA_REFDATA`, and
`JWST_TOOL_PANDEIA_PSF_DIR`. They are resolved in `src/jwst_tool/instruments.py`
with loud failures. The PICASO reference tree is selected by
`JWST_TOOL_PICASO_REFDATA`.
