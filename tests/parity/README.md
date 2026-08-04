# PandExo numerical parity harness

The 2026-07-12 external audit made mode-by-mode numerical parity against
current PandExo a release gate for any "PandExo-style" precision claim. This
directory is that gate, organized as:

```
tests/parity/
  scripts/   run_parity.py, pandexo_worker.py, make_report.py, make_parity_plots.py
  outputs/   parity_summary.json + REPORT.md (committed); raw run JSON (git-ignored)
  figs/      parity_config_timing.png, parity_extracted_flux.png (committed)
```

## What it does

`scripts/run_parity.py` runs the SAME star and instrument configurations
through

1. this package's Pandeia worker plus its box-transit depth-error
   propagation (`noise.pixel_depth_variance`), and
2. current PandExo master (`scripts/pandexo_worker.py`, a standalone script
   that runs inside the current-Pandeia conda env),

both on the SAME engine/refdata/PSF release (the gate requires the supported
2026.7 triple on both sides and records all three). Differences are
therefore estimator and policy differences, never engine calibration
differences. The tool's default pinned 3.0 backend is untouched.

Per mode it compares: detector configuration (subarray, readout pattern,
filter, disperser), the extracted wavelength grid, selected group count,
integration time, integration-counting policy, extracted stellar count
rates, and the per-pixel transit-depth sigma with no noise floor. The sigma
comparison is reported twice: with PandExo's integration counts substituted
into the tool's formula (noise-model parity in isolation) and with the
tool's own floor(T/t_int) counts (the shipped policy).

## Known, intended differences

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

## Running it

Requires a conda env with `pandeia.engine==2026.7` and PandExo at the commit
pinned by `run_parity.py`, the extracted `pandeia_data-2026.7-jwst` and
`pandeia_psfs-2026.7-jwst` trees,
and a synphot CDBS with the phoenix grid, CALSPEC Vega, and the Bessell
J/H/K + 2MASS Ks bandpasses (fetch missing ones from
`https://ssb.stsci.edu/trds/comp/nonhst/`). Then:

```
JWST_TOOL_PANDEIA_PYTHON=<env python> JWST_TOOL_PANDEIA_REFDATA=<data tree> JWST_TOOL_PANDEIA_PSF_DIR=<psf tree> JWST_TOOL_DATA_DIR=<dir containing cdbs/> JWST_TOOL_OUTPUT_DIR=<the tool's noise cache> python tests/parity/scripts/run_parity.py
python tests/parity/scripts/make_report.py
python tests/parity/scripts/make_parity_plots.py
```

The plotting command refuses a summary without a passing gate, so an archival
or failed run cannot be regenerated with current-release labels.

All five environment variables are required and fail loudly; no machine
paths are baked into the repository. `JWST_TOOL_OUTPUT_DIR` is only the
tool's Pandeia noise cache; the parity artifacts (`outputs/`, `figs/`) and
raw run JSON stay in this directory.
