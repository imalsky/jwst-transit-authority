# Physics choices and conventions

The modeling conventions this tool adopts, the defaults it ships, and why. Moved
out of the README in 2026-07.

## Composition scaling

Any metallicity or C/O knob must pick a convention. Papers that leave it implicit
are the literature's main complaint on this point (Drummond et al. 2019, MNRAS 486,
1123), so this tool states it.

**Metallicity scales the network's O, C, N, and S abundances together, with He/H
held fixed.** That is the universal practice. **C/O moves carbon at fixed,
metallicity-scaled oxygen.** This is VULCAN's own published convention (Tsai et al.
2017, ApJS 228, 20).

This matters when comparing across codes. petitRADTRANS, the ATMO and Goyal grids,
and GGchem instead anchor metallicity on carbon and vary oxygen. The Sonora and
PICASO family preserves C+O. So at non-solar C/O, **the same physical atmosphere
maps to different (Z, C/O) coordinates in different codes**, and different molecules
carry the C/O signature: C-bearing species here, H2O and CO2 in oxygen-varied
codes. Near solar C/O the conventions agree.

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

Four options: constant, two parametric forms (Pfunc and JM16), or the table's own
`Kzz` column. In every mode the Fisher `lnKzz` row is a multiplicative scale of the
whole profile.

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
that pays. The constant 1e9 cm²/s default runs 4-33x low for WASP-39 b and 15-17x
low for HD 189733 b, always suppressing photochemical products.

## Boundary conditions

All off by default: gravitational settling, diffusion-limited escape for H, H2, and
He, and constant top and bottom per-species fluxes with deposition velocities.

## Clouds

Two independent decks, which can be combined:

- **An analytic power-law deck**, a gray-to-sloped opacity per gram.
- **A Mie condensate deck**, using real refractive-index optics from the ExoJAX
  virga database with a column-uniform lognormal size distribution.

Either deck's parameters can be freed and marginalized in the Fisher forecast when
that deck is on: the power-law amplitude and slope, and the Mie particle radius,
size dispersion, and abundance.

The Mie radius and dispersion ride a piecewise-linear lookup grid, so their
finite-difference rows carry a step-size consistency check that **refuses** a step
straddling a grid node. The Mie abundance is exactly linear. Each Mie condensate
needs a one-time lookup grid built with `tools/generate_miegrid.py`.

## Backend configuration

The Pandeia engine runs in its own conda environment and is deliberately not a
package dependency.

**The default backend is Pandeia 2026.2 with `pandeia_data-2026.2-jwst`**, the
STScI JWST 5.1 release, validated mode by mode against PandExo in `tests/parity/`.

The backend token `current` is a token, not a currency claim. STScI's supported
Cycle 6 release moved to 2026.7 on 2026-07-16, so forecasts here are one
calibration release behind the live ETC until the full engine, reference-data, and
PSF tuple is upgraded and the parity results are regenerated. See
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
