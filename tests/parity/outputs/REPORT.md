# PandExo numerical parity report

Generated 2026-08-17 by `run_parity.py` + `make_report.py` in this directory. This file is a SUMMARY. The method, scope, and known intended differences live in the harness docstrings (`../scripts/run_parity.py`). The full per-mode rows, the noise-model attribution, and the raw PandExo warnings live in `parity_summary.json` beside this file. Figures live in `../figs/`.

**GATE: PASS** (re-validated by `make_report.py`, not read from the artifact). Worker v12, Pandeia 2026.7 on both sides, the full 3-star x 12-mode matrix present, every declared threshold met.

## Provenance

| star | side | engine | refdata | PSFs | worker / PandExo |
|---|---|---|---|---|---|
| w39_like | this tool | 2026.7 | 2026.7 | 2026.7 | worker v12 |
| w39_like | PandExo | 2026.7 | 2026.7 | 2026.7 | 2026.7 @ 34e42d81f782 |
| bright_hot | this tool | 2026.7 | 2026.7 | 2026.7 | worker v12 |
| bright_hot | PandExo | 2026.7 | 2026.7 | 2026.7 | 2026.7 @ 34e42d81f782 |
| faint_k | this tool | 2026.7 | 2026.7 | 2026.7 | worker v12 |
| faint_k | PandExo | 2026.7 | 2026.7 | 2026.7 | 2026.7 @ 34e42d81f782 |

Configuration: constant transit depth 0.01, transit duration 2.8036 h, equal out-of-transit baseline, saturation limit 80%, no noise floor, native (R=None) grids. Both sides run the SAME Pandeia release named above, so every difference below is an estimator or policy difference, never an engine calibration difference.

## Measured (35 unsaturated rows)

| quantity | measured | gate |
|---|---|---|
| wavelength-grid pixels matched | 100.0% (worst row) | >= 99% at rtol 1e-09 |
| extracted flux ratio, per-mode medians | 1.0000-1.0000 | median within 3% of unity |
| group agreement, moderate/bright | within 0 group(s) | exact when either side <= 3 groups |
| group agreement, faint Ks=13 | within 0 group(s) | 5 groups OR 1%, whichever is looser |
| largest total t_int gap | 4.5% (w39_like/nirspec_prism, ~1-group ramp) | 15% relative |
| sigma ratio, NIR matched | -8% to +31% (up to +34% policy) | median inside [0.8, 2.0] |
| sigma ratio, MIRI LRS | +35% to +53% | same band |

This tool is conservative relative to PandExo on every row but 2, which sit marginally below unity. The residual sigma difference is the noise model itself, not the configuration (mechanism: README.md). Saturation masks are wavelength-aligned and gated for complete coverage and exact agreement; rows above the saturation limit are diagnostic rows, not validation rows.
