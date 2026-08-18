# vulcan-jwst-tool

vulcan-jwst-tool plans JWST exoplanet spectroscopy. A live forward model
(VULCAN-JAX photochemistry, then ExoJAX radiative transfer through the shared
[vulcan-forward](https://github.com/imalsky/vulcan-forward) package) is scored
against STScI Pandeia instrument noise. Given a planet and a science goal, it
ranks JWST time-series modes and estimates how many transits or eclipses
are needed.
Transmission and thermal emission are both supported. The package imports as
`jwst_tool` and installs the `jwst-tool` console script.

## Install

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple 'vulcan-jwst-tool[gui]'
conda create -n pandeia_2026_7 python=3.12
conda run -n pandeia_2026_7 pip install pandeia.engine==2026.7
```

Set the data locations in your shell profile:

```bash
export JWST_TOOL_DATA_DIR="$HOME/vulcan/jwst_data"
export JWST_TOOL_OUTPUT_DIR="$HOME/vulcan/jwst_output"
export JWST_TOOL_PANDEIA_PYTHON="$(conda run -n pandeia_2026_7 which python)"
export VULCAN_FORWARD_DATA="$HOME/vulcan/forward_data"
```

Then fetch the reference data. `jwst-tool fetch` downloads everything with a
public URL and prints the two STScI downloads it cannot script. The ExoMolOP
k-tables (the default opacity source) are fetched separately:

```bash
jwst-tool fetch
python -m vulcan_forward.fetch_exomolop --molecules H2O,CO2,CO,CH4,SO2,C2H2,C2H4,H2S,HCN,NH3,OCS,SO,SH
```

`jwst-tool data` reports what is installed, with a remedy per missing item.

## Run

```bash
jwst-tool
```

This launches the Streamlit GUI. Keep the defaults (WASP-39 b, detect SO2 at
3 sigma, PRISM + G395H + MIRI LRS) and press Run for a first result. A fresh
parameter set takes a few minutes; results are cached. The "Custom planet"
mode plans transiting planets within the tool's supported parameter ranges,
with an optional auto-fill from a shipped NASA Exoplanet Archive snapshot.

## Scope and limits

The noise model is diagonal: per-bin Pandeia noise, an optional floor, and
per-detector-segment depth offsets. It omits time-correlated residuals,
visit-long trends, and stellar heterogeneity. Detection scores are
conditional matched-template S/N values, not retrieval posteriors; a
retrieval freeing more parameters usually reports lower significance. Noise
forecasts are pandeia-extracted, benchmarked against PandExo, and
conservative relative to PandExo on nearly every tested mode; the current
measured numbers are in `tests/parity/outputs/REPORT.md`. Fisher
constraints use certified derivatives (central finite differences that must
pass a step-halving consistency gate, or forward-mode AD; an uncertified
derivative is never reported). Fisher values are local, linearized
half-widths under the assumed atmosphere and noise model, marginalized over
the other free parameters unless labeled conditional; rank-deficient
directions are reported as unconstrained.

## Tests and validation

This tool includes test suites, as well as other validation checks. The suites
run in CI for each repository:
[jax-vulcan](https://github.com/imalsky/jax-vulcan),
[vulcan-forward](https://github.com/imalsky/vulcan-forward),
[vulcan-jwst-tool](https://github.com/imalsky/vulcan-jwst-tool), and
[vulcan-retrieval](https://github.com/imalsky/vulcan-retrieval). For
end-to-end tests, see the set of validation figures that I've created
[here](https://github.com/imalsky/vulcan-forward/tree/main/validation/figures).
This includes trying to recreate the results of
[Tsai et al. 2023](https://doi.org/10.5281/zenodo.7542781), the
[JWST ERS carbon dioxide paper](https://doi.org/10.5281/zenodo.6959427), and
VULCAN 2.0 and petitRADTRANS on identical inputs.

## Deployment

The public instance runs as a Hugging Face Space; the recipe is in `deploy/`.

## How to cite

Cite the software via [`CITATION.cff`](CITATION.cff). Published results should
also cite the components a run used: VULCAN (Tsai et al. 2017, 2021), FastChem
(Stock et al. 2018), ExoJAX (Kawahara et al. 2022, 2025), Pandeia
(Pontoppidan et al. 2016), PandExo for the noise benchmark (Batalha et al.
2017), virga and its refractive-index dataset for cloud runs, and the NASA
Exoplanet Archive for the custom-planet fill (Christiansen et al. 2025).

## License

GPLv3, inherited from VULCAN.
