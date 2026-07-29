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
equilibrium and kinetics are directly comparable. See
[`docs/picaso_roadmap.md`](docs/picaso_roadmap.md) for its scope and measured
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
conda create -n pandeia_2026 python=3.11
conda run -n pandeia_2026 pip install pandeia.engine==2026.2
```

This release uses Pandeia 2026.2, the release PandExo pins and the one the
parity suite validates against. STScI's current JWST Cycle 6 release is 2026.7,
so this tool is one calibration release behind the live ETC. The full policy is
in [`docs/physics_and_conventions.md`](docs/physics_and_conventions.md).

3. Tell the tool where to keep data and caches. Add these to your shell profile so
   they persist, then open a new terminal:

```bash
export VULCAN_PROJECT_ROOT="$HOME/vulcan"
export JWST_TOOL_DATA_DIR="$HOME/vulcan/jwst_data"
export JWST_TOOL_OUTPUT_DIR="$HOME/vulcan/jwst_output"
export JWST_TOOL_PANDEIA_PYTHON="$(conda run -n pandeia_2026 which python)"
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
takes about two minutes, and the app shows a runtime estimate before each run.
Results are cached on disk and downloadable as PNG or CSV.

### First run, step by step

1. Keep the defaults in step 1 (WASP-39 b, transmission).
2. Keep the default VULCAN chemistry in step 2.
3. In step 3, keep the goal "Detect a molecule" with SO2 at 3 sigma.
4. In step 4, keep the default instrument modes and one transit.
5. Press Run. The first run solves the chemistry and takes a few minutes;
   the noise forecast per star is cached, so later runs on the same star are
   much faster.

The result page leads with a verdict of the form "Best mode for detecting SO2
on WASP-39 b: <mode>, <score> in 1 transit", followed by the spectrum, the
mode ranking, and per-mode details. Quality certificates and backend
provenance are in the collapsed "Model quality and provenance" section.

## Science goals

**Detection** scores how strongly one molecule imprints on the spectrum. The
statistic is a conditional matched-template signal-to-noise ratio: the chi-square
distance between the model and the same model without that molecule's opacity,
with calibration nuisances profiled out. It is conditional on the assumed
atmosphere being exactly right, so it upper-bounds any real retrieval result. It is
never a retrieval detection.

**Constraint** builds a Fisher-information forecast from the spectrum's parameter
derivatives and reports local Cramer-Rao lower bounds. Those are not posterior
widths.

## Scope and limits

This is a planner, not a retrieval. Detection scores assume one fixed atmosphere,
and the Fisher forecast is linear and local, so real posteriors can only be wider.

Four limits to keep in view:

- **The noise model omits time-correlated systematics.** It is conservative
  against PandExo by roughly 2-24% in the near infrared and 33-56% for MIRI LRS.
- **Stellar contamination is not modeled.** Unocculted spots and faculae can
  dominate transit-depth systematics for active hosts, most strongly below about
  3 um (Rackham et al. 2018; Lim et al. 2023). Treat short-wavelength depths
  around active stars with care.
- **Emission is pure-absorption thermal emission.** There is no scattering in the
  emergent flux and no reflected light. The run refuses atmospheres whose column
  is not optically thick at its bottom, because there is no interior flux term.
- **The PICASO engine has no photochemistry**, and therefore no SO2. It is capped
  at C/O of 1.10 by its tables and is finite-difference only.

ExoJAX capabilities that exist upstream but are not wired here: reflected-light
spectra, scattering emission, correlated-k opacities, H-minus continuum, atomic and
FeH line lists, rotational and instrumental broadening, and GP noise kernels.

Physics conventions, default structures, and the backend policy are in
[`docs/physics_and_conventions.md`](docs/physics_and_conventions.md). Read the
composition-scaling section before comparing metallicity or C/O against another
code. The same physical atmosphere maps to different coordinates in different
codes.

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
├── picaso_env.py      PICASO refdata bootstrap and fingerprints
├── picaso_chem.py     PICASO equilibrium provider
├── picaso_climate.py  PICASO climate runner and cache
└── cli.py             console entry point

tests/unit/            numpy-only suite: python -m pytest tests -q
tests/live/            env-gated live validation
tests/parity/          PandExo parity harness and report
tests/parity_picaso/   PICASO-native RT vs ExoJAX parity, offline
```

## Documentation

| File | Contents |
|---|---|
| [`docs/physics_and_conventions.md`](docs/physics_and_conventions.md) | Composition scaling, T-P and Kzz options, default structures, clouds, boundary conditions, backend policy |
| [`docs/picaso_roadmap.md`](docs/picaso_roadmap.md) | PICASO engine scope, measured limits, deferred features |
| [`docs/audit_decisions_2026-07-21.md`](docs/audit_decisions_2026-07-21.md) | Disposition of every science-audit finding |

## License

GPLv3, inherited from VULCAN.
