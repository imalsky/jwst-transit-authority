# vulcan-jwst-tool

vulcan-jwst-tool plans JWST exoplanet spectroscopy observations with a live
forward model. VULCAN-JAX photochemistry feeds ExoJAX radiative transfer through
the shared [vulcan-forward](https://github.com/imalsky/vulcan-forward) engine,
and instrument noise comes from the STScI Pandeia engine. Given a planet and a science
goal, it ranks JWST time-series modes and estimates how many transits are needed.

Two geometries are supported: transmission, using transit depth, and thermal
emission, using secondary-eclipse depth with a PHOENIX stellar spectrum. The
package imports as `jwst_tool` and installs the `jwst-tool` console script.

A second chemistry engine is available: PICASO 4 thermochemical equilibrium, plus a
PICASO radiative-convective climate profile that either engine can use. Both
engines feed the same radiative transfer, binning, noise, and Fisher machinery, so
equilibrium and kinetics are directly comparable. See the
[PICASO engine](#picaso-engine) section below for its scope and measured
limits.

## Install

1. Install the package. The engine (`vulcan-forward`) and the chemistry core
   (`vulcan-jax`) resolve automatically:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]'
```

2. Create the Pandeia engine environment. It runs in its own environment and is
   deliberately not a package dependency:

```bash
conda create -n pandeia_2026_7 python=3.12
conda run -n pandeia_2026_7 pip install pandeia.engine==2026.7
```

The backend is a **matched triple**: the engine, the reference data, and the PSF
library must all be the same release. The default `current` backend means

| component | required release |
|---|---|
| `pandeia.engine` | 2026.7 |
| `pandeia_data-2026.7-jwst` | 2026.7 |
| `pandeia_psfs-2026.7-jwst` | 2026.7 |

The worker checks all three before any calculation and refuses a mixed set, so a
2026.2 PSF tree can no longer serve a 2026.7 engine and silently change the
extracted flux and noise. All three versions are recorded in each result's
`__provenance__` and in the cache key.

2026.7 is the release STScI supports; it labels older releases archival and
unsuitable for planning new proposals
([installation page](https://outerspace.stsci.edu/spaces/PEN/pages/77530136/Pandeia%2BEngine%2BInstallation)).
The previous 2026.2 tuple is still selectable for reproducibility as
`JWST_TOOL_BACKEND=archival_2026_2` (renamed, not repointed, so older caches and
artifacts recorded as `current` cannot pass for current-release output). The
pinned Pandeia 3.0 `legacy` backend was removed. The full policy is in
[Backend configuration](#backend-configuration) below.

3. Tell the tool where to keep data and caches. Add these to your shell profile so
   they persist, then open a new terminal:

```bash
export VULCAN_PROJECT_ROOT="$HOME/vulcan"
export JWST_TOOL_DATA_DIR="$HOME/vulcan/jwst_data"
export JWST_TOOL_OUTPUT_DIR="$HOME/vulcan/jwst_output"
export JWST_TOOL_PANDEIA_PYTHON="$(conda run -n pandeia_2026_7 which python)"
export VULCAN_FORWARD_DATA="$HOME/vulcan/forward_data"
```

`VULCAN_FORWARD_DATA` is where the forward engine keeps its line lists and
opacity caches (`exojax_linelists/` and `opacity_cache/` beneath it). They are
tens of gigabytes, so they are never bundled; `jwst-tool fetch` downloads what
it can and `jwst-tool data` reports the rest.

4. Fetch the reference data:

```bash
jwst-tool fetch
```

This downloads every dataset with a public URL, then prints the two STScI Box
downloads it cannot script, with the exact paths to extract them to. Everything
else fetches itself on first use.

Run `jwst-tool data` at any time for a live status report with a remedy per item.

## Run

```bash
jwst-tool
```

This preflights the stack and launches the Streamlit GUI. A new parameter set
takes a few minutes with the default molecule set, and the app shows a runtime
estimate before each run. Results are cached on disk and downloadable as PNG
or CSV. Deselecting extra molecules makes a new run faster.

### First run, step by step

1. Keep the defaults in step 1 (WASP-39 b, transmission).
2. Keep the default VULCAN chemistry in step 2.
3. In step 3, keep the goal "Detect a molecule" with SO2 at 3 sigma.
4. In step 4, keep the default instrument modes (PRISM, G395H, MIRI LRS;
   all eight are selectable) and one transit.
5. Press Run. The first run solves the chemistry and takes a few minutes;
   the noise forecast is cached per star and per mode, so later runs on the
   same star are much faster and adding a mode only computes that mode.

The result page leads with a verdict of the form "Best mode for detecting SO2
on WASP-39 b: <mode>, <score> in 1 transit", followed by the spectrum, the
mode ranking, and per-mode details.

### Any transiting planet

Step 1's "Custom planet" mode plans any target. Its form can auto-fill by
planet name from a NASA Exoplanet Archive PSCompPars snapshot shipped with
each release (~4400 transiting planets; the fetch date is shown in the GUI,
and it is never a live query). Values outside the tool's supported ranges or
missing from the archive are reported by name and left unchanged, never
clamped. The stellar UV spectrum for photochemistry is never selected for
you: the archive carries no UV spectra, so the fill leaves the menu alone
and the GUI only shows which shipped template is nearest in Teff to the
entered star, as a suggestion. Maintainers refresh the snapshot with
`jwst-tool archive-refresh`.

## Science goals

**Detection** scores how strongly one molecule imprints on the spectrum. The
statistic is a conditional matched-template signal-to-noise ratio: the chi-square
distance between the model and the same model without that molecule's opacity,
with calibration nuisances profiled out. It is conditional on the assumed
atmosphere being exactly right: an optimistic fixed-atmosphere sensitivity
metric after profiling the listed nuisance directions. A retrieval that frees
more parameters under the same model and noise assumptions often reports weaker
evidence, but retrieval significances come in many forms (Bayes factors,
posterior exclusions, profile likelihoods with boundaries) and this statistic
is not a universal upper bound on any of them. It is never a retrieval
detection.

**Constraint** builds a Fisher-information forecast from the spectrum's parameter
derivatives and reports local Cramer-Rao lower bounds. Those are not posterior
widths: they are local, likelihood-based approximations, and informative priors
or external data can make a real posterior narrower.

**Treat the mode rankings as more robust than the absolute ppm numbers.** The
systematic effects the tool leaves out -- time-correlated noise, the
conditional-atmosphere assumption behind the detection score, the linearization
behind the Fisher bounds -- push every mode in the same direction and largely
divide out of a comparison BETWEEN modes, while they move the absolute numbers
by more than their quoted precision suggests. "G395H beats PRISM for this
molecule" is the durable result; "4.2 sigma" is the fragile one. (This
paragraph used to sit in the GUI's intro; it moved here in the 2026-08-13 UI
cleanup, with the interface keeping only labels, numbers, and loud failures.)

## Forecast products (beta)

Four result-page features are new in 0.29.0 and carry a beta statement in the
GUI: the mode-combination builder, the marginalized forecast posteriors, the
simulated mock observation, and the proposal summary figure. Sanity-check any
forecast from this tool against a full retrieval before submitting a proposal.

### Mode combinations

The results page can compare named combinations of instrument modes -- for
example "SOSS + G395H" against "SOSS + G395H + MIRI" -- to answer the
proposal question "what does adding a mode buy?". A combination's forecast
is the joint Fisher forecast over its modes, with the same shared and
per-segment calibration nuisances as the single-mode rows, so a
one-mode combination reports exactly that mode's numbers. Saturated modes
are excluded from a combination (the same policy as the all-usable combined
row) and the exclusion is stated on the result; a combination with no usable
mode stops and shows an error. Combination rows appear in the Fisher table
and as bars in the comparison chart, and saved configurations restore them.

### Marginalized forecast posteriors

For each free parameter the results page can draw a one-dimensional
marginalized forecast curve: the Gaussian centered on the input model whose
width is the marginalized Cramer-Rao bound, in the same display units as the
Fisher table. These are linearized Fisher forecasts, not sampled posteriors:
Gaussian by construction, local to the input model, and best-case under the
quoted noise model. A retrieval freeing more parameters under the same model
and noise assumptions usually reports lower significance. A parameter the
data cannot constrain (a numerically null Fisher direction) is annotated as
unconstrained; the tool never draws a curve for it.

### Simulated mock observation

The spectrum plot can overlay one simulated noise realization: the binned
noiseless model plus one seeded Gaussian draw per bin, at each bin's final
forecast uncertainty (floor included). It is one realization of the adopted
effective diagonal noise model: when a noise floor stands in for correlated
instrumental systematics, drawing it as independent bin noise is internally
consistent with the tool's diagonal likelihood but is not a physical model
of correlated systematics. The draw is generated after the
forward model and the noise model, never inside them, and it is fitted: the
posterior panels overlay the parameters recovered from that one realization
(`posteriors.mock_recovery`, delta = F^+ J^T Sigma^-1 n on the same
nuisance-profiled system as the forecast), which move with the seed. The
forward model itself stays noiseless, and the FORECAST -- every quoted
precision and the conditional template S/N -- is computed from the noiseless
model and is independent of the plotted draw by construction. A single realization can be lucky or unlucky;
the seed is shown with the results, reproduces the identical realization,
and is saved in shared configurations. The mock data can be downloaded, but
only under a filename that names its seed; the result CSVs stay noiseless.

Forecasting on noiseless simulated data is a standard convention: it avoids
a single random noise draw biasing the result, at the documented cost that
simulated-retrieval posteriors then sit optimistically centered on the true
values. Their widths approximate ensemble-average widths in regular,
well-constrained regimes, though not near parameter boundaries or strong
nonlinearities (Feng et al. 2018, AJ, 155, 200, who also compare retrievals
on multiple noise instances against the noiseless convention in their
Section 5.2).

With the mock layer on and Jacobians available, the posterior panels also
show what the linearized fit would recover from that one plotted
realization: a curve of the same width whose center is shifted by the draw.
Over many realizations the shift has zero mean and the marginalized Fisher
covariance -- it is a visualization of realization scatter, not a retrieval.

### The results figure

The results page renders one composed, proposal-ready graphic (no figure
appears twice): the model spectrum with each mode's simulated data points
on the left, and up to two marginalized forecast posterior panels (best
mode or combination, with an optional dashed comparison) on the right.
Each mode's expected performance -- its conditional template S/N for a
detection goal, its expected ± for a constraint goal -- rides in that
mode's legend entry as a value (the legend is not sorted by performance;
the Fisher table carries the ordered comparison). Saturated modes carry no
value, matching every other ranking in the tool.
Downloadable as vector PDF and PNG alongside the binned-points, native
model, and (when the mock layer is on) seeded mock-observation CSVs; the
footnote carries the same linearized-forecast wording as the posterior
section. The T-P profile remains its own small figure, and the Fisher
table carries the full per-mode and per-combination numbers.

**Figure conventions (2026-08-13).** Every figure is square (equal panel box
aspect), every legend sits OUTSIDE its axes, and log axes label at most seven
decades with no minor ticks. The legends moved out of the axes because keeping
them inside required inflating the y limits to park them clear of the data,
which distorted the visible range for the legend's benefit; per-mode numbers
stay in their own legend entries and the note explaining what those numbers
mean is the legend's title, never folded into an entry label. The builders
live in `plotting.py` (`build_tp_figure`, `build_vmr_figure`) and in
`summary_figure.py`, all importable without Streamlit, so the tests render
exactly what the app renders.

**Rendering is serialized** behind `plotting.render_lock`, a process-wide
in-process reentrant lock held across each figure's whole lifecycle:
construct, lay out, draw, export, close. Matplotlib guards its process-global
mathtext parser with `Figure._render_lock` only for the duration of
`Figure.draw`; `tight_layout` runs the layout engine outside that lock and
measures every tick label, and on a log axis those labels are mathtext. With
one Streamlit session per thread, two users rendering at once could enter the
shared parser concurrently and the loser raised
`ValueError: ParseException: exception raised in parse action`. Measured on
the deployed pin (matplotlib 3.10.0): 7 of 8 concurrent threads failed before
the fix, 0 of 8 after. `tests/unit/test_plotting.py` pins both the behavior
and the structure -- one test fails if a builder stops taking the lock, another
if an unlocked `tight_layout`/`savefig`/`st.pyplot` call reappears in `app.py`.

### NIRSpec G395M

The mode list now includes NIRSpec G395M (F290LP, SUB2048, NRSRAPID,
2.87-5.10 um) alongside the seven original modes. Trade-offs in that band,
scoped to this tool's fixed configurations: PRISM (as configured here, the
SUB512 subarray -- statements about PRISM saturation apply to that
configuration, not to NIRSpec PRISM generally; STScI offers other PRISM
subarrays for brighter targets that this registry does not carry) gives the
broadest wavelength coverage at R ~ 100 and is often attractive when it
stays acceptably unsaturated. G395M gives continuous 2.87-5.18 um coverage
at R ~ 1000 on one detector; G395H gives R ~ 2700, a higher bright limit in
many configurations, and an NRS1/NRS2 detector gap near 3.72-3.82 um.
Neither grating is universally preferred: longer G395H ramps improve
efficiency when G395M is genuinely limited to very few groups, but at
moderate group counts the reset-overhead difference is small, and the full
noise calculation (read noise, saturation, extraction, throughput, final
binning) -- not a duty-cycle ratio -- decides which configuration carries
more information for a stated goal. That is what this tool's Pandeia-based
comparison is for: run the candidate modes at the intended final binning
and compare. For scale, on WASP-39 b (Ks = 10.2) the tool's live noise runs
select unsaturated ramps of 1 group for PRISM (at the ramp minimum, 65%
full well), 32 groups for G395M, and 82 for G395H -- numbers specific to
that target and these fixed configurations. G395M is not part of the frozen
PandExo parity benchmark, and its literature noise factor is an
extrapolation from G395H -- both are listed under
[Open gaps](#open-gaps-and-accepted-limitations).

## Scope and limits

This is a planner, not a retrieval. Detection scores assume one fixed atmosphere,
and the Fisher forecast is linear and local.

Five limits to keep in view:

- **Each instrument mode is one fixed detector configuration** (subarray and
  readout pattern, shown in the mode details table). The tool does not search
  alternative subarrays or readout patterns, and does not check APT feasibility
  (data volume, scheduling). The ramp search reaches the instrument's shortest
  permitted ramp (pandeia's per-detector minimum: 1 group in the near infrared,
  2 for MIRI); ramps below the STScI-recommended minimum are flagged with a
  warning. Verify the chosen configuration in APT before proposing.
- **The noise model omits time-correlated systematics.** In a three-star,
  fixed-configuration, no-floor parity benchmark (`tests/parity/`) it is
  conservative against PandExo by roughly 2-24% in the near infrared and 33-56%
  for MIRI LRS. Those ranges are benchmark results, not a general guarantee.
- **Stellar contamination is not modeled.** Unocculted spots and faculae can
  dominate transit-depth systematics for active hosts, most strongly below about
  3 um (Rackham et al. 2018; Lim et al. 2023). Treat short-wavelength depths
  around active stars with care.
- **Emission is pure-absorption thermal emission.** There is no scattering in the
  emergent flux and no reflected light. The run refuses atmospheres whose column
  is not optically thick at its bottom, because there is no interior flux term.
- **The PICASO engine has no photochemistry**, and therefore no SO2 and no CS2.
  It is capped at C/O of 1.10 by its tables and is finite-difference only.

Line-spread treatment: for modes whose analysis bins approach the native
resolving power (PRISM, MIRI LRS, blue SOSS), the model is blurred with a
flux-weighted Gaussian built from the tabulated R(λ), not with Pandeia's full
wavelength-response matrix. This approximation has not yet been validated
against mode-specific Pandeia impulse responses.

ExoJAX capabilities that exist upstream but are not wired here: reflected-light
spectra, scattering emission, correlated-k opacities, H-minus continuum, atomic
and FeH line lists, rotational broadening, ExoJAX's own instrumental-broadening
operator (the tool applies its Gaussian R(λ) approximation instead), and GP
noise kernels.

Physics conventions, default structures, and the backend policy are in
[Physics and conventions](#physics-and-conventions) below. Read
[Composition scaling](#composition-scaling) before comparing metallicity or C/O
against another code. The same physical atmosphere maps to different coordinates in different
codes.

## Layout

```
src/jwst_tool/
├── app.py             Streamlit GUI
├── forward.py         forward-model driver: chemistry, RT, Jacobians
├── engine_config.py   this tool's view of the shared engine's config
├── adjoint_diag.py    reverse-mode adjoint diagnostics
├── fisher.py          Fisher forecasts
├── detect.py          detection statistics
├── noise.py           ETC noise model and floors
├── binning.py         the single measurement operator
├── plotting.py        pure figure builders + the render lock
├── pandeia_worker.py  Pandeia subprocess
├── instruments.py     mode registry and path roots
├── datacheck.py       data-availability detection
├── planets.py         planet registry
├── picaso_env.py      PICASO refdata bootstrap and fingerprints
├── picaso_chem.py     PICASO equilibrium provider
├── picaso_climate.py  PICASO climate runner and cache
└── cli.py             console entry point

tests/unit/            numpy-only suite: python -m pytest tests -q
tests/live/            env-gated live validation
tests/parity/          PandExo parity harness and report
tests/parity_picaso/   PICASO-native RT vs ExoJAX cross-model check, offline
```

## Documentation

The full reference set lives in this file, in the sections below:

| Section | Contents |
|---|---|
| [Physics and conventions](#physics-and-conventions) | Composition scaling, T-P and Kzz options, default structures, clouds, boundary conditions, backend policy |
| [PICASO engine](#picaso-engine) | PICASO provider + climate mode: scope, measured limits, deferred features |
| [Chemistry config vs upstream VULCAN](#chemistry-config-vs-upstream-vulcan) | Field-by-field deviation table against the upstream W39b oracle config |
| Decision records (`notes.md`, local) | Disposition of every audit and review finding plus the draft upstream PICASO report |
| [Open gaps and accepted limitations](#open-gaps-and-accepted-limitations) | The live list of every known gap, shortcoming, deferred feature, and accepted limitation |
| [Data provenance](#data-provenance) | What ships in `data/`, what is fetched per machine, and the source of each piece |
| [Deployment](#deployment) | The AWS VM and Hugging Face Space runbooks |
| [Parity testing](#parity-testing) | The PandExo numerical parity harness and its gate |

Dev logs, investigations, and the historical version log live in the
gitignored `notes.md` at the repo root.

## Physics and conventions

The modeling conventions this tool adopts, the defaults it ships, and why.
(History: moved out of the README in 2026-07 as `docs/physics_and_conventions.md`,
which absorbed `docs/picaso_roadmap.md` and `docs/jwst_tool_upstream_deviations.md`
in the 2026-08-05 consolidation, and was merged back into this README in the
2026-08 three-doc consolidation. The PICASO and config-deviation parts are the
next two sections; the decision records follow them.)

The end-to-end guided tour of the pipeline, including the noise model
described carefully (formerly the project-level autodiff guide PDF, Part 6,
"From gradients to data: the JWST tool", deleted 2026-08-11), is summarized
in the untracked `VULCAN-JAX/notes.md` (autodiff-guide section); the
reference-depth material is in `VULCAN-JAX/README.md` (Differentiability
section).

### Composition scaling

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

### Temperature-pressure profiles

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

### Eddy diffusion profiles

Four options: constant, two parametric forms, or the table's own `Kzz` column. In
every mode the Fisher `lnKzz` row is a multiplicative scale of the whole profile.

The two parametric names are upstream VULCAN's, kept for config compatibility
rather than coined here — `build_atm.py` documents `JM16` as the profile form
assumed in Moses et al. (2016) and `Pfunc` as the one in Tsai (2020). `JM16` is
`Kzz = max(K_deep, 1e5 (300 mbar/P)^0.5)`; `Pfunc` is
`max(K_max, K_max (K_p_lev/P)^0.4)`.

### Default structure is the verified measured table where one ships

A planet whose bundled measured `T(P)`/`Kzz` table is verified end to end
defaults to that table under the VULCAN engine; every other planet, and the
PICASO engine, defaults to a Guillot (2010) `T(P)` with constant `Kzz`
(2026-08-11 decision, reversing the structure half of the 2026-08-09
speed-first defaults: the default WASP-39 b run must reproduce the
literature-validated SO2 state, and the table also converges faster). Today
the verified set is exactly WASP-39 b. The trade-off runs the other way now:
a tabulated `T(P)` has no temperature parameter, so default file-mode Fisher
forecasts carry no temperature row and are conditional on the profile; switch
to `tp_mode="guillot"` when you need one.

| Planet | Bundled table |
|---|---|
| WASP-39 b | `atm_W39b_evening_TP_Kzz.txt` (Tsai et al. 2023 evening terminator), verified end to end and the default |
| HD 189733 b | `atm_HD189_Kzz.txt` — selectable but not verified: the solver does not certify a steady state on it at default settings, while the analytic profile converges in about 36 s |
| HD 209458 b | Refused. It is a full thermosphere model reaching 2997 K inside the chemistry grid, above the 2980 K opacity ceiling, and it is never clipped |
| WASP-107 b | None bundled |

Two facts are kept separate on purpose: whether a table **exists** for a planet,
and whether a run on it has been **verified end to end**. Tables are per-planet
and never substituted; selecting a planet without a usable one tells you why.

**Stated trade-off: the analytic stand-in is biased in a systematic
direction.** A constant `Kzz` cannot follow a profile that climbs orders of
magnitude with altitude, and it is the photochemically active upper atmosphere
that pays. Measured against the bundled tables over the chemistry grid at
p < 1 mbar, the constant 1e9 cm²/s stand-in runs 3.8-48x low for WASP-39 b
(`atm_W39b_evening_TP_Kzz.txt`) and 10-17x low for HD 189733 b
(`atm_HD189_Kzz.txt`), always suppressing photochemical products. The factor is
pressure-cut dependent — deeper than ~10 mbar the table falls *below* 1e9 for
WASP-39 b — so quote it with the cut. On WASP-39 b the Guillot profile also ran
about 100 K hot through the SO2 formation zone when this was measured
(2026-07-21); the published-detection agreement (G395H SO2 4.16 sigma) belongs
to the **shipped table**, which is why the table is the W39b default again.
Never quote Guillot-mode W39b SO2 numbers against Alderson/Tsai 2023.

### Boundary conditions and condensation (programmatic interface only)

All off by default: gravitational settling, diffusion-limited escape for H, H2, and
He, and constant top and bottom per-species fluxes with deposition velocities.
S8 condensation (sulfur rainout) is likewise off by default and remains
detection-only, refused with any Jacobian method, with photochemistry off, or
with molecular diffusion off (see the condensation rules below).

**These five settings left the GUI on 2026-08-13** (`use_condense`,
`use_settling`, `diff_esc`, `top_flux`, `bot_flux`). They are unchanged
canonical parameters with their full compatibility matrix and are reachable
through `jwst_tool.forward.canonical_params` and the CLI; only the interface
stopped offering them. A shared configuration that leaves them at their
defaults loads normally, but one that ENABLES any of them is **refused** with an
error naming the setting: pinning them off silently would present a successful
restore while Run computed a different atmosphere than the file describes.

### Clouds

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

### Height-dependent gravity in the transmission optical depth

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

### Backend configuration

The Pandeia engine runs in its own conda environment and is deliberately not a
package dependency.

**The default backend (`current`) is the Pandeia 2026.7 matched triple**
(`pandeia.engine` 2026.7 + `pandeia_data-2026.7-jwst` +
`pandeia_psfs-2026.7-jwst`), the STScI-supported release; the worker refuses a
mismatched triple. **The committed PandExo parity artifact is a gate-evaluated
PASS on 2026.7** (`tests/parity/outputs/REPORT.md`, worker v10, both sides on
the same triple, PandExo master at the pinned commit): a fixed-configuration
estimator comparison in which configuration, timing, wavelength grids, and
extracted flux matched, and the remaining sigma difference is the noise model,
with this tool conservative. Since 2026-08-09 the gate also requires exact
group agreement on short ramps (either side at 3 groups or fewer), bounds the
per-integration-time gap, fails any matched sigma ratio outside an anomaly
band, treats missing timing/sigma fields on validation rows as failures, and
gates the saturation claim itself on saturated rows (measured fraction +
PandExo's full-well verdict) -- a same-day review showed the earlier
+-1-group tolerance passing a wrong 1-vs-2-group SOSS selection with a 7x
sigma discrepancy. The worker's group search proves maximality (complete
only when the next integer measured unsafe or the cap is reached); a
budget-exhausted search is disclosed in the results, never presented as
optimal. The public Space runs this backend.

`JWST_TOOL_BACKEND=archival_2026_2` selects the previous 2026.2 tuple under its
honest archival name, for reproducing older results only. Its own parity
artifact predates the fail-closed gate and was never gate-evaluated. STScI
labels 2026.2 archival and unsuitable for planning new proposals. See
the decision records in `notes.md` (its S2-04
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

## PICASO engine

The PICASO provider + climate T-P mode: scope, limits, and roadmap.
Status: v18 (tool 0.12.0, 2026-07-20). This is the versioned record of what
the PICASO integration ships, the science limits it states, the measured
findings behind its design decisions, and the features deliberately deferred
(with re-entry sketches). The GUI links here wherever a limit bites.

### What shipped (v18)

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

### Stated science limits (intrinsic, not bugs)

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

### Measured data-quality findings (upstream-reportable)

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

### Deferred features (why, and how to re-enter)

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

### Live validation

`JWST_TOOL_RUN_PICASO_LIVE=1 python -m pytest tests/live -q` runs the
measured-2026-07-20 battery: within-node native parity, leave-one-node-out
blend accuracy, lnZ FD closure, picaso-vs-vulcan spectrum sanity, and the
climate smoke matrix (W39b x rfacv {0, 0.5, 1}, solar node, HD 189733 b,
WASP-107 b). The native-RT CROSS-MODEL report (outside target, not a parity
result) lives in `tests/parity_picaso/outputs/REPORT.md`.

### v18.1 (tool 0.12.1, model-cache v19): review-response hardening
## Chemistry config vs upstream VULCAN

The tool's effective chemistry config vs upstream VULCAN master (W39b).
Date: 2026-07-15. Method: field-by-field diff of `VULCAN-master/vulcan_cfg.py`
(W39b-configured, the parity oracle) against the tool's EFFECTIVE VULCAN-JAX
cfg at GUI defaults (`W39b.yaml` + `vulcan_chem` profile application + the
tool's `cfg_overrides`, forward v12). Script: session scratchpad `cfg_diff.py`.

### Fields identical in both (the reassuring bulk)

Elemental abundances (C_H 2.95e-3, O_H 5.37e-3, N_H, S_H, He_H), solver
tolerances (rtol 0.2, atol 0.1, mtol, mtol_conv), step counters (count_max
3e4, count_min 120, trun_min), convergence thresholds (yconv_cri 1e-2,
yconv_min 0.1, slope_cri, flux_cri), photolysis geometry (sl_angle 83 deg,
f_diurnal 1.0), photo update cadence (ini/final 5/5), EQ initialization +
FastChem file, stellar UV file, boundary-condition files, use_photo on,
use_moldiff on, use_vm_mol off (tool PINS it since v11; upstream YAML default
flipped on 2026-07-14), condensation/settling off in both.

### Real deviations (tool vs master), with status

| Field | Master (W39b) | Tool effective | Status |
|---|---|---|---|
| `atm_type` | `file` (GCM evening-terminator T-P) | `isothermal` structural + live `tp_eval` (iso/Guillot) | DELIBERATE, documented: GCM baseline removed 2026-07-13; every planet gets an explicit T-P. Tool answers differ from Tsai-2023-style runs by construction. |
| `Kzz_prof` | `file` (GCM Kzz(z)) | `const`, GUI default 1e9 | DELIBERATE (same removal). Master's `const_Kzz=1e10` is inert there. Note the GUI default 1e9 is a choice, not Tsai's profile; slider spans 1e6-1e12. |
| `dt_max` | 1e17 (`runtime*1e-5`) | 1e11 | DELIBERATE, documented (validated state-preserving; prevents the adaptive-dt balloon; master's own uncapped value implicated in the photo-off blow-up, see VULCAN-JAX/notes.md (2026-07-15 photo-off entry)). |
| `nz` | 150 | 100 (GUI default; 150 available) | DELIBERATE fast-tier default; use 150 + yconv 1e-3 for final numbers (documented deltas: MIRI LRS halves at 100/1e-2). |
| `conver_ignore` | 13 heavy hydrocarbons (C2H2, C6H6, ...) | `['HC3N']` only | SETTLED 2026-07-30 (was FLAGGED "provenance unclear, review"): the 13-species list exists in NO upstream repository (the local VULCAN-master copy was contaminated with it); fetched exoclime master ships `[]`, shami vm_branch ships `['HC3N']`, and `[]` vs `['HC3N']` measured behaviorally IDENTICAL on HD189/HD209/W39b. The stricter JAX-side gate stands; nothing to adopt. |
| `network` | `NCHO_photo_network.txt` (current oracle parking) | `SNCHO_photo_network.txt` | Not a tool deviation: SNCHO is the Tsai 2023 W39b science network (sulfur/SO2). The workspace master cfg is parked on NCHO for the oracle tests. |
| `wall_clock_max` | 1800 s | 3600 s | Runner backstop doubled on the JAX side. Benign (a backstop, not physics). |
| `Tiso` | 1000 (inert; atm_type=file) | 1100 (GUI default near W39b Teq) | Inert in master; not a comparison. |
| `gs` vs `Mp` | gs directly | `Mp = gs*Rp^2/G` | Exactly equivalent by construction (checked 2026-07-15). |

### VULCAN-JAX-only feature toggles (confirmed inert at tool defaults)

`use_chunked_runner`, `use_fix_H2He`, `use_fix_all_bot`, `use_ini_cold_trap`,
`use_pi_controller`, `use_sat_surfaceH2O`, `use_adapt_rtol` all False;
`use_hybrid_vm_mol` pinned False by the tool (v11). No silent feature is
active that master lacks.

### Tool-level constructions with no master counterpart

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

### Engine-level (VULCAN-JAX vs master) differences

Documented separately: `VULCAN-JAX/README.md`, Parity & bug guide section (VULCAN-JAX
parity and bug guide) and `jax_paper/paper/notes_gaps.md` (ranked gaps;
headline: reservoir projection default-on in JAX vs none in master). Those
apply to every consumer of VULCAN-JAX, not just this tool.

## Decision records

The full audit and review dispositions (2026-07-21 second-pass audit, 2026-08-05
adversarial review, the 2026-08-09/10/11 decision series, and the draft PICASO
upstream report) live in `notes.md` (local, gitignored), "Decision records"
section. Summary: every finding is dispositioned FIXED / DELIBERATE / ACCEPTED
with reasoning; accepted limitations are listed below under
[Open gaps](#open-gaps-and-accepted-limitations).
## Open gaps and accepted limitations

The one live list of everything known to be missing, approximate, or
deferred in this tool. Updated 2026-08-13 (0.29.3: MIRI native-R ordering
fix, post-push audit response).

* `app.py`'s post-run section is still one long top-level block sharing
  implicit variables; extracting pure result builders (mode performance,
  posterior panels, summary spectrum, export frames) is the next
  maintainability step. Deferred deliberately: it is a large refactor of the
  most-frequently-changed file and does not belong in a UI release.
* `science.mplstyle` is a vendored copy of an older matplotlib defaults file
  rather than a minimal set of intentional overrides, so it will keep
  surfacing upstream deprecations one at a time. Keep this section current: close items here when they
land, add new ones as they are found. The reasoning behind every decision
lives in notes.md, Decision records; scope and conventions live in
[Physics and conventions](#physics-and-conventions).

### Deferred from the 2026-08-13 external review

- No deterministic proposal-export seed policy yet: the mock-observation
  seed is user-selectable and displayed, which is reproducible but allows
  cycling seeds until a realization looks favorable. A proposal-export mode
  would derive a read-only seed from immutable identifiers (target, mode
  configuration, tool commit, seed-scheme version) and report the whitened
  residual statistic and its percentile so an extreme draw is disclosed,
  not replaced.
- The combined-mode forecast fixes one systematic model (shared lnR0 plus
  one free absolute offset per detector segment). Free absolute offsets
  remove most broadband radius information, so the reported lnR0 precision
  can ride a small nonconstant component of the radius derivative; the
  rank-aware solver prevents false finite numbers but a sensitivity
  display (no offsets / per-instrument / per-segment) is not offered yet.
- Rank detection runs on the Jacobi-whitened Fisher matrix (correct and
  scale-invariant); an SVD of the whitened design matrix would avoid
  squaring the condition number and is a deferred numerical refinement.
- The absolute-C/O posterior display uses the first-order (delta-method)
  transform of the ln-space Gaussian; for broad uncertainties the exact
  transformed density is asymmetric and a symmetric Gaussian can extend
  below zero.
- Marginalized forecast curves assume a locally flat prior in the DISPLAY
  parameterization (flat in lnZ is not flat in Z).

### Validation gaps (absence of evidence, not defects)

- The PandExo parity harness does not cover NIRSpec G395M: the committed
  parity artifact is a frozen seven-mode experiment, and extending it means
  regenerating the whole matrix. G395M's registry tokens are verified
  against the pandeia reference data and a live WASP-39 b noise run
  selected a sane unsaturated ramp, but no cross-tool comparison exists
  for the mode.
- The G395M literature noise factor (achieved-vs-predicted 1.10) is an
  extrapolation from G395H; no published G395M measurement backs the
  digit. Like every entry in that table it is a reference point, never
  applied by default.

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

### Accepted limitations (deliberate; reasoning in notes.md, Decision records)

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
- The marginalized forecast posterior curves are linearized Cramer-Rao
  forecasts: Gaussian by construction, local to the input model, and
  best-case under the quoted noise model. A full retrieval can return
  posteriors that are non-Gaussian, multimodal, or wider; the tool's
  curves cannot show that. The labeling on every curve, panel, and
  export states this permanently.

## Data provenance

Inputs for this repo live in `data/` (env `JWST_TOOL_DATA_DIR` overrides; an
editable checkout infers the root, a site-packages install must set it). The
model and noise caches are GENERATED and live in `output/`
(`JWST_TOOL_OUTPUT_DIR`): `model_cache/` spectra and `noise_cache/` Pandeia
results, regenerated per run and cache-busted by `forward._VERSION` /
the backend fingerprint in the code.

Run `jwst-tool data` for a live status report of every item below (and the
sibling-repo data the forward model needs), with per-item download remedies.

### Tracked in git (ships with a clone)

- `data/cdbs/comp/nonhst/2mass_ks_001_syn.fits` -- the 2MASS Ks bandpass used for
  the photsys vegamag normalization (required; worker preflight checks it).
  Source: https://ssb.stsci.edu/trds/comp/nonhst/2mass_ks_001_syn.fits
- `data/cdbs/comp/nonhst/johnson_j_003_syn.fits` -- UNUSED leftover of the retired
  J-band normalization (tracked but referenced by no code path).
- `data/cdbs/grid/phoenix` -- a SYMLINK into an external local stellar-grid tree
  (`RT-Project/picaso/reference/stellar_grids/...`); it dangles on other
  machines and the pandeia worker preflight fails loudly there. Do not
  replace it with a copy on THIS machine; on another machine, fetch the
  STScI reference-atlases PHOENIX tarball (~1.9 GB) and place its
  `grp/redcat/trds/grid/phoenix` tree at this path (a real directory works):
  https://archive.stsci.edu/hlsps/reference-atlases/hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_synphot5.tar
  (the 'pheonix' spelling in the filename is STScI's own).

### Gitignored (fetch once per machine)

- `data/cdbs/calspec/alpha_lyr_stis_011.fits` -- CALSPEC Vega for the vegamag
  normalization (288 KB; required, preflighted). Source:
  https://ssb.stsci.edu/trds/calspec/alpha_lyr_stis_011.fits
- `data/pandeia_data-2026.2-jwst/` -- Pandeia JWST reference data for the default
  "current" backend (~15 MiB download / 30 MB extracted; must carry
  VERSION_DATA matching the engine release).
  Source: https://stsci.box.com/v/pandeia-data-v2026p2-jwst
- `data/pandeia_psfs-2026.2-jwst/` -- the split PSF library (pandeia_data >= 2026;
  ~4 GiB; must contain VERSION_PSF).
  Source: https://stsci.box.com/v/pandeia-psfs-v2026p2-jwst

The forward model's own data (HITRAN line lists, CIA tables, CO ExoMol cache,
stellar UV spectra) lives in the SIBLING repos (vulcan-retrieval `data/`,
vulcan_jax package data) -- see the [Install](#install) section and
`jwst-tool data`.

## Deployment

The tool runs anywhere `pip install vulcan-jwst-tool` works. The two production
deployments (always-on AWS VM behind Caddy, and the Hugging Face Space) have
step-by-step runbooks in `notes.md`, "Deployment runbooks" section.

Two deployment runbooks (merged here from `deploy/DEPLOY.md` and
`deploy/hf-space/SETUP.md` in the 2026-08 doc consolidation). The scripts and
configs they reference live in `deploy/` and `deploy/hf-space/`. Caches
self-invalidate by version keys in both deployments.

## Parity testing

The 2026-07-12 external audit made mode-by-mode numerical parity against
current PandExo a release gate for any "PandExo-style" precision claim.
`tests/parity/` is that gate: `scripts/` holds `run_parity.py`,
`pandexo_worker.py`, the shared gate `parity_gate.py` (experiment declaration
+ validate), `make_report.py`, and `make_parity_plots.py`; `outputs/` holds
the committed `parity_summary.json` + `REPORT.md` (raw run JSON git-ignored);
`figs/` holds the committed timing and extracted-flux figures.

`scripts/run_parity.py` runs the SAME star and instrument configurations
through (1) this package's Pandeia worker plus its box-transit depth-error
propagation (`noise.pixel_depth_variance`) and (2) current PandExo master
(`scripts/pandexo_worker.py`, a standalone script run inside the
current-Pandeia conda env), both on the SAME engine/refdata/PSF release (the
gate requires the supported 2026.7 triple on both sides and records all
three). Differences are therefore estimator and policy differences, never
engine calibration differences. It is a FIXED-CONFIGURATION comparison: the
harness overrides PandExo's templates to this tool's registry hardware, so it
does not test PandExo's own configuration-selection policy. Per mode it
compares: detector configuration (subarray, readout pattern, filter,
disperser), the extracted wavelength grid, selected group count, integration
time, integration-counting policy, extracted stellar count rates, and the
per-pixel transit-depth sigma with no noise floor. The sigma comparison is
reported twice: with PandExo's integration counts substituted into the tool's
formula (noise-model parity in isolation) and with the tool's own
floor(T/t_int) counts (the shipped policy).

Known, intended differences:

* The tool floors partial integrations (`int(T/t_cycle)`); PandExo rounds.
  Worth at most one integration per window.
* The tool uses the out-of-transit flux and noise for both in- and
  out-of-transit terms (symmetric approximation, documented in
  `noise.pixel_depth_variance`); PandExo propagates separate in-transit
  counts with the (1-depth) factor. At depth 0.01 the tool is expected to
  sit ~0.5% ABOVE PandExo (conservative), growing with depth.
* Group-count caps: the registry's `ngroup_max` can bind before PandExo's
  optimizer does; where it binds the tool uses a shorter ramp (slightly
  higher sigma, never lower).

Running it requires a conda env with `pandeia.engine==2026.7` and PandExo at
the commit pinned by `run_parity.py`, the extracted
`pandeia_data-2026.7-jwst` and `pandeia_psfs-2026.7-jwst` trees, and a
synphot CDBS with the phoenix grid, CALSPEC Vega, and the Bessell J/H/K +
2MASS Ks bandpasses (fetch missing ones from
`https://ssb.stsci.edu/trds/comp/nonhst/`). Then:

```
JWST_TOOL_PANDEIA_PYTHON=<env python> JWST_TOOL_PANDEIA_REFDATA=<data tree> JWST_TOOL_PANDEIA_PSF_DIR=<psf tree> JWST_TOOL_DATA_DIR=<dir containing cdbs/> JWST_TOOL_OUTPUT_DIR=<the tool's noise cache> python tests/parity/scripts/run_parity.py
python tests/parity/scripts/make_report.py
python tests/parity/scripts/make_parity_plots.py
```

All five environment variables are required and fail loudly; no machine
paths are baked into the repository. `JWST_TOOL_OUTPUT_DIR` is only the
tool's Pandeia noise cache; the parity artifacts (`outputs/`, `figs/`) and
raw run JSON stay in `tests/parity/`. Both renderers re-run the shared gate
(`scripts/parity_gate.py`) on the summary instead of trusting its persisted
`gate.passed` boolean, so an archival, failed, or hand-edited run cannot be
rendered with current-release labels. The same module backs
`tests/unit/test_parity_gate.py`, which also requires the COMMITTED artifact
to re-validate as a pass.

## How to cite

Cite the software itself using the metadata in [`CITATION.cff`](CITATION.cff)
(GitHub's "Cite this repository" button renders it as BibTeX or APA).

The tool runs other groups' codes; published results should also cite the
components a run actually used:

- VULCAN (the chemistry this tool's solver reimplements):
  Tsai, S.-M., et al. 2017, ApJS, 228, 20 and Tsai, S.-M., et al. 2021,
  ApJ, 923, 264
- FastChem (equilibrium initialization): Stock, J. W., et al. 2018,
  MNRAS, 479, 865
- ExoJAX (radiative transfer and opacities; this tool runs the ExoJAX 2
  generation): Kawahara, H., et al. 2022, ApJS, 258, 31 and
  Kawahara, H., et al. 2025, ApJ, 985, 263
- Pandeia (instrument noise): Pontoppidan, K. M., et al. 2016,
  Proc. SPIE, 9910, 991016
- PICASO, only when the PICASO engine or climate mode is used:
  Batalha, N. E., et al. 2019, ApJ, 878, 70, and for the climate mode
  Mukherjee, S., et al. 2023, ApJ, 942, 71
- Cloud runs use the virga condensate database via ExoJAX: cite virga
  (Batalha, N., Rooney, C. M., and Mukherjee, S. 2020, Zenodo,
  doi:10.5281/zenodo.3759888), the refractive-index dataset actually
  supplying the condensate optics ("Refractive Indices For Virga Exoplanet
  Cloud Model", Zenodo, doi:10.5281/zenodo.15886530), and Ackerman, A. S.,
  and Marley, M. S. 2001, ApJ, 556, 872; that dataset also asks that each
  condensate's optical-constants source be cited individually (see
  the [Clouds](#clouds) section)
- The custom-planet fill ships a NASA Exoplanet Archive PSCompPars
  snapshot. Work using it should cite Christiansen, J. L., et al. 2025,
  PSJ, 6, 186 and carry the Archive's acknowledgment:
  "This research has made use of the NASA Exoplanet Archive, which is
  operated by the California Institute of Technology, under contract with
  the National Aeronautics and Space Administration under the Exoplanet
  Exploration Program." (table DOI: 10.26133/NEA13)
- The noise-model validation is benchmarked against PandExo
  (`tests/parity/`): Batalha, N. E., et al. 2017, PASP, 129, 064501

## License

GPLv3, inherited from VULCAN.
