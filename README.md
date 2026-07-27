# vulcan-jwst-tool

`vulcan-jwst-tool` plans JWST exoplanet spectroscopy observations with a
live forward model: VULCAN-JAX photochemistry feeds ExoJAX radiative
transfer, with instrument noise from the STScI Pandeia engine. Given a
planet and a science goal, it ranks JWST time-series modes and estimates
the transits required. Two geometries: transmission (transit depth) and
thermal emission (secondary-eclipse depth Fp/Fs x (Rp/Rs)^2, with a
PHOENIX stellar SED for Fs). Import name `jwst_tool`, console script
`jwst-tool`.

Since v18 a second forward-model engine is available: PICASO 4
thermochemical-equilibrium chemistry (Visscher grid), plus a PICASO
radiative-convective climate T-P mode usable under EITHER engine (VULCAN
kinetics can run on the PICASO climate profile). Both engines feed the
identical RT, binning, noise, and Fisher machinery, so
equilibrium-vs-kinetics is directly comparable. The PICASO engine has no
photochemistry and therefore no SO2 (equilibrium sulfur is H2S -- in the
base opacity set -- plus opt-in OCS), is
capped at C/O 1.10 by its tables, and is finite-difference only; its
reference data is selected by `JWST_TOOL_PICASO_REFDATA`. Scope, measured
limits, and deferred features: `docs/picaso_roadmap.md`.

## Installation

1. Install from TestPyPI. The sibling packages resolve automatically:

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]>=0.10.3'
```

2. Create the Pandeia engine environment (once):

```
conda create -n pandeia_2026 python=3.11
conda run -n pandeia_2026 pip install pandeia.engine==2026.2
```

3. Tell the tool where to keep data and caches. Add these lines to your
shell profile (`~/.zshrc` on macOS, `~/.bashrc` on Linux) so they persist
across sessions, then open a new terminal or `source` the file:

```
export VULCAN_PROJECT_ROOT="$HOME/vulcan"
export JWST_TOOL_DATA_DIR="$HOME/vulcan/jwst_data"
export JWST_TOOL_OUTPUT_DIR="$HOME/vulcan/jwst_output"
export JWST_TOOL_PANDEIA_PYTHON="$(conda run -n pandeia_2026 which python)"
```

4. Fetch the reference data:

```
jwst-tool fetch
```

This downloads every dataset with a public URL (CIA tables, the PHOENIX
grid, CALSPEC pieces) and prints the two STScI Box downloads it cannot
script, with the exact paths to extract them to. Everything else fetches
itself on first use. `jwst-tool data` shows a live status report with a
remedy per item at any time.

## Running

`jwst-tool` preflights the stack and launches the Streamlit GUI. A new
parameter set takes about 2 minutes; the app shows a runtime estimate
before each run. Results are disk-cached and downloadable as PNG or CSV.

## Backend configuration

The Pandeia engine runs in its own conda environment and is deliberately
not a package dependency.

**Backend status: Pandeia 2026.2 / pandeia_data-2026.2-jwst (default) --
the STScI JWST 5.1 release, validated mode by mode against PandExo in
`tests/parity/`.** "current" is the backend *token*, not a currency claim:
STScI's supported Cycle 6 release moved to 2026.7 on 2026-07-16, so
forecasts here are one calibration release behind the live ETC until the
full engine/refdata/PSF tuple is upgraded and parity is regenerated
(`docs/audit_decisions_2026-07-21.md`). Set `JWST_TOOL_BACKEND=legacy` to
select the pinned pandeia 3.0 + pandeia_data-3.0rc3 pair, retained only as
an explicit reproducibility backend -- a legacy run always labels itself
LEGACY and never presents as 2026-backend output. The worker refuses to
run a mismatched engine/refdata pair, and every result and cache file
records the exact engine, refdata, and worker versions in a
`__provenance__` block hashed into the cache key, so switching backends
invalidates caches automatically. The env vars
`JWST_TOOL_PANDEIA_{PYTHON,REFDATA,PSF_DIR}` override the backend-default
paths per machine (resolved in `src/jwst_tool/instruments.py` with loud
failures).

## Physics choices and conventions

Composition scaling (stated because any metallicity / C/O knob must pick a
convention, and papers that leave it implicit are the literature's main
complaint, Drummond et al. 2019, MNRAS 486, 1123): metallicity scales the
network's O/C/N/S abundances together with He/H held fixed (the universal
practice), and C/O moves CARBON at fixed, metallicity-scaled OXYGEN. This
is VULCAN's own published convention (Tsai et al. 2017, ApJS 228, 20).
Note when comparing across codes: petitRADTRANS, the ATMO/Goyal grids, and
GGchem instead anchor metallicity on carbon and vary oxygen, and the
Sonora/PICASO family preserves C+O, so at non-solar C/O the same physical
atmosphere maps to different (Z, C/O) coordinates in different codes, and
different molecules carry the C/O signature (C-bearing species here;
H2O/CO2 in O-varied codes). Near-solar C/O the conventions agree.

T-P profiles are explicit only: Guillot, a tabulated table (the shipped
WASP-39b evening-terminator profile or an upload; the cache key carries the
table's content hash), or a PICASO radiative-convective climate solve. A
globally isothermal profile was removed in July 2026 -- it held the deep
CO/CH4/NH3 quench region at one temperature and biased disequilibrium
abundances. Kzz profiles: constant, two parametric forms (Pfunc, JM16), or
the table's Kzz column; the Fisher lnKzz row is a multiplicative scale of
the whole profile in every mode.

**Defaults are the measured structure wherever one exists, not an analytic
stand-in.** Under the VULCAN kinetics engine a planet defaults to the T-P +
Kzz table VULCAN bundles for *that* planet, used for BOTH T(P) and Kzz(P):

| Planet | Default structure | Bundled table |
|---|---|---|
| WASP-39 b | `atm_W39b_evening_TP_Kzz.txt` (Tsai et al. 2023 evening terminator) | default |
| HD 189733 b | Guillot + constant Kzz | `atm_HD189_Kzz.txt`, selectable but not default: the solver does not certify a steady state on it at default settings, while the analytic default converges in ~36 s |
| HD 209458 b | Guillot + constant Kzz | refused -- a full thermosphere model reaching 2997 K inside the chemistry grid, above the 2980 K opacity ceiling, and never clipped |
| WASP-107 b | Guillot + constant Kzz | none bundled |

Two facts are kept separate on purpose: whether a table *exists* for a planet,
and whether a default run on it has been *verified end to end*. A table only
becomes the default once it has, so enabling one can never turn a working
planet into one that errors on arrival. Tables are per-planet and never
substituted -- selecting a planet without a usable one tells you why. The
PICASO equilibrium provider keeps the analytic default in every case.

This matters because the analytic defaults are biased in a systematic
direction: a constant Kzz cannot follow a profile that climbs orders of
magnitude with altitude, and it is the photochemically active upper atmosphere
that pays. The constant 1e9 cm²/s default runs 4-33x low for WASP-39 b and
15-17x low for HD 189733 b, always suppressing photochemical products.

The trade-off is stated rather than hidden: a tabulated profile has no
temperature parameter, so file-mode Fisher forecasts carry NO temperature row.
They are conditional on the profile being exactly right, and the reported
sigmas are optimistic by the amount temperature uncertainty would add --
switch to Guillot when you need a temperature row.

Boundary conditions (all off by default): gravitational settling,
diffusion-limited escape (H/H2/He), and constant top/bottom per-species
fluxes with deposition velocities.

Clouds come in two forms. The analytic power-law deck (a gray-to-sloped
opacity per gram) and a physically-grounded Mie condensate deck (real
refractive-index optics from the ExoJAX virga database, a column-uniform
lognormal size distribution) are independent and can be combined. Either
deck's parameters can be freed in the Fisher forecast (marginalized) when
the deck is on: the power-law amplitude and slope, and the Mie particle
radius, size dispersion, and abundance. The Mie radius and dispersion ride
the piecewise-linear Mie lookup grid, so their finite-difference rows carry
a step-size consistency check that refuses a step straddling a grid node;
the Mie abundance is exactly linear. Each Mie condensate needs a one-time
lookup grid built with `tools/generate_miegrid.py`.

## Scope and limits

This is a planner, not a retrieval. Detection scores assume one fixed
atmosphere. The Fisher forecast is linear and local, so real posteriors
can only be wider. The noise model omits time-correlated systematics and
is conservative against PandExo by roughly 2 to 24 percent in the near
infrared and 33 to 56 percent for MIRI LRS. Stellar contamination (the
transit light source effect) is NOT modeled: unocculted spots and faculae
can dominate transit-depth systematics for active hosts, strongest
shortward of ~3 um (Rackham et al. 2018, ApJ 853, 122; Lim et al. 2023,
ApJL 955, L22), so treat short-wavelength depths around active stars with
care. Emission mode is pure-absorption thermal emission (ExoJAX
ArtEmisPure): no scattering in the emergent flux, no reflected light, and
the run refuses atmospheres whose RT column is not optically thick at its
bottom (there is no interior flux term).

ExoJAX capabilities present upstream but NOT wired here: reflected-light
spectra (ArtReflectPure/ArtReflectEmis), scattering emission (ArtEmisScat;
the Mie deck here enters as extinction in the pure-absorption solvers,
without a forward-scattering source term), correlated-k opacities (OpaCKD),
H-minus continuum, atomic/FeH line lists, rotational and instrumental
broadening operators, and GP noise kernels.

## Layout

```
src/jwst_tool/
├── app.py             Streamlit GUI
├── forward.py         forward-model driver: chemistry, RT, Jacobians
├── adjoint_diag.py    reverse-mode adjoint diagnostics
├── fisher.py          Fisher forecasts
├── detect.py          detection statistics
├── noise.py           ETC noise model and floors
├── binning.py         the single measurement operator
├── pandeia_worker.py  Pandeia subprocess
├── instruments.py     mode registry and path roots
├── datacheck.py       data-availability detection
├── planets.py         planet registry
├── picaso_env.py      PICASO refdata bootstrap + content fingerprints
├── picaso_chem.py     PICASO equilibrium provider (blended Visscher grid)
├── picaso_climate.py  PICASO radiative-convective climate runner + cache
└── cli.py             console entry point
tests/unit/            numpy-only suite: python -m pytest tests -q
tests/live/            env-gated live validation (JWST_TOOL_RUN_PICASO_LIVE)
tests/parity/          PandExo parity harness and report
tests/parity_picaso/   PICASO-native RT vs ExoJax parity (offline)
docs/picaso_roadmap.md PICASO scope, measured limits, deferred features
```

## Science goals

Detection scores how strongly one molecule imprints on the spectrum with a
**conditional matched-template signal-to-noise ratio** (`sigma_detect`):
the chi-square distance between the model and the same model without that
molecule's opacity, with calibration nuisances profiled out. It is
conditional on the assumed atmosphere being exactly right -- never a
retrieval detection -- and upper-bounds any real retrieval result.
Constraint builds a Fisher-information forecast from the spectrum's
parameter derivatives and reports local Cramer-Rao lower bounds, not
posterior widths.
