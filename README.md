# VULCAN JWST tool

A local research application for comparing JWST time-series modes using a
shared atmospheric calculation. The certified path is:

`VULCAN-JAX chemistry -> vulcan-forward/ExoJAX spectrum -> Pandeia noise -> detection and local Fisher forecasts`

The application supports transmission and emission calculations, all eight
registered JWST modes, explicit noise-floor choices, conditional molecular
template scores, marginalized Fisher forecasts, seeded mock observations, and
machine-readable exports.

This is a planning and methods tool, not an observing-time calculator of
record. Confirm final configurations in the supported STScI tools and APT.

## Install

Use Python 3.10-3.12 and install the core stack from frozen revisions or a
released lock file:

```bash
python -m venv .venv
.venv/bin/pip install -e ../jax-vulcan -e ../vulcan-forward -e '.[gui]'
```

Pandeia runs in a separate environment. This release requires a matched
2026.7 triple: `pandeia.engine==2026.7`,
`pandeia_data-2026.7-jwst`, and `pandeia_psfs-2026.7-jwst`.

```bash
export JWST_TOOL_PANDEIA_PYTHON=/path/to/pandeia-2026.7/bin/python
export JWST_TOOL_PANDEIA_REFDATA=/path/to/pandeia_data-2026.7-jwst
export JWST_TOOL_PANDEIA_PSF_DIR=/path/to/pandeia_psfs-2026.7-jwst
export JWST_TOOL_DATA_DIR=/path/to/jwst-tool-data
export VULCAN_FORWARD_DATA=/path/to/vulcan-forward-data
```

`JWST_TOOL_DATA_DIR/cdbs` must contain the PHOENIX grid, the local Vega
spectrum, and the required photometric bandpasses. The forward-data root must
contain the opacity cache and line lists reported by the data checker.

```bash
jwst-tool data --deep
jwst-tool
```

Do not continue to a scientific run unless the data command reports every
required item present.

## Reproducible use

1. Choose the planet, chemistry/profile settings, science goal, modes, event
   timing, saturation threshold, resolving power, and an explicit floor model.
2. Review the run summary and execute the forward and Pandeia calculations.
3. Download the configuration and result tables. Configuration provenance
   records repository commits, dirty state, package versions, data checksums,
   cache schemas, the Pandeia/PandExo identity, and the random seed.
4. Treat a dirty tree, unknown commit, missing checksum, warning, failed
   convergence certificate, or rank-deficient requested parameter as a failed
   release input.

The random mock layer is display-only: it does not change Fisher widths or
detection scores. Noise floors are an effective diagonal approximation; they
do not physically model correlated systematics.

## Scientific interpretation

### Analysis resolving power (the R control)

The R control sets the resolving power of the final analysis bins,
`R = wavelength / bin width`. It is not the instrument's resolving power and
does not change the detector sampling or the native Pandeia calculation.
Counts and variances are accumulated on the native pixel grid before a final
bin is reported; partially saturated native pixels are excluded rather than
hidden by the coarser display grid.

- Detection scores compare a molecule-present spectrum with an RT-only
  molecule-removed template. The primary score projects detector-segment and
  requested nuisance directions; the raw conditional score is diagnostic.
- Parameter curves are local Fisher/Cramér-Rao forecasts, not sampled
  posteriors. Boundaries, nonlinearity, prior choices, multimodality, and model
  inadequacy are outside that approximation.
- Absolute C/O curves transform the local Gaussian in log C/O and therefore
  remain positive; broad curves are still only local approximations.
- Pandeia supplies extracted count-rate/noise products. This tool propagates a
  box-transit ratio with explicit in/out baselines and count-space binning.
- A selected floor is applied as a minimum final-bin uncertainty and never
  averages down with additional transits.

See [methods](docs/methods.md), [validation](docs/validation.md), and
[limitations](docs/limitations.md) before sharing numerical results.

## PICASO status

PICASO chemistry and climate code is retained for maintainer investigation but
is disabled by default and excluded from collaborator claims. PICASO 4.0.1
requires NumPy 2 while the validated ExoJAX 2.2.3 stack requires NumPy below 2,
and the native-RT cross-model artifact is failing and stale. There is no PICASO
install extra in this release. Setting
`JWST_TOOL_ENABLE_UNCERTIFIED_PICASO=1` exposes an explicitly labelled,
uncertified path; its output must not enter release results.

## Development checks

```bash
python -m pytest tests/unit -q
ruff check src tests
python -m build
```

These checks are necessary but are not scientific validation. The release gate
also requires live Pandeia/PandExo, VULCAN parity, forward-model closure,
Jacobian/Fisher, package-install, and artifact checks documented in
[validation](docs/validation.md).

License: GPL-3.0-only. Please cite the software and the underlying VULCAN,
ExoJAX, Pandeia, and instrument references appropriate to the calculation.
