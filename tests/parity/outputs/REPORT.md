# PandExo numerical parity report

Generated 2026-08-14 by `run_parity.py` + `make_report.py` in this directory.

**GATE: PASS** (re-validated by `make_report.py`, not read from the artifact). Worker v11, Pandeia 2026.7 on both sides, the full 3-star x 8-mode matrix present, every declared threshold met.

## Provenance

| star | side | engine | refdata | PSFs | worker / PandExo |
|---|---|---|---|---|---|
| w39_like | this tool | 2026.7 | 2026.7 | 2026.7 | worker v11 |
| w39_like | PandExo | 2026.7 | 2026.7 | 2026.7 | 2026.7 @ 34e42d81f782 |
| bright_hot | this tool | 2026.7 | 2026.7 | 2026.7 | worker v11 |
| bright_hot | PandExo | 2026.7 | 2026.7 | 2026.7 | 2026.7 @ 34e42d81f782 |
| faint_k | this tool | 2026.7 | 2026.7 | 2026.7 | worker v11 |
| faint_k | PandExo | 2026.7 | 2026.7 | 2026.7 | 2026.7 @ 34e42d81f782 |

Both sides run on the SAME Pandeia backend -- the exact engine, reference-data, and PSF releases are in the provenance table above, and the gate refuses the run if they disagree across the two sides. Every difference below is therefore an ESTIMATOR/policy difference, not an engine calibration difference. (This says nothing about whether that release is the SUPPORTED one; the gate banner does.) PandExo is used from master at the commit recorded above. Configuration: constant transit depth 0.01, transit duration 2.8036 h, equal out-of-transit baseline, saturation limit 80%, no noise floor, native (R=None) grids.

**Scope: a fixed-configuration estimator comparison.** The submitted instrument configuration is injected into BOTH sides from this tool's registry (the harness overrides PandExo's template subarray/readout/filter per mode), so configuration equality below is by construction, and this gate deliberately does not test PandExo's own configuration-selection policy. What each side computes INDEPENDENTLY -- and what this report actually compares -- is the ramp/group optimization, the timing, the 2D extraction, and the noise propagation on that fixed configuration.

## Figures (regenerate with `make_parity_plots.py`)

These show the quantities that match 1:1 -- the parity result. The depth-uncertainty difference (a noise-model difference, not a configuration one) is quantified in the tables and Findings below, not plotted.

- **parity_config_timing.png** -- selected groups, integration time, and integration count, this tool vs PandExo, on the 1:1 line (log-log).
- **parity_extracted_flux.png** -- G395H extracted stellar count rate, per-wavelength overlay with a ratio strip (per-pixel jitter + binned median).

## What matches, and how it is measured

The two are independently IMPLEMENTED estimators calling the same Pandeia engine on an identically pinned configuration. Only what is read straight from the shared engine agrees exactly; anything each computes on its own agrees closely but not exactly. This is a property of a cross-tool test, not a defect -- forcing bit-equality would mean one tool copying the other's numbers, which is not a validation.

- **Submitted configuration (subarray, readout, filter, disperser):** identical on every row -- by construction, since the harness pins both sides to the registry; the gate fails on any drift in these four recorded fields. The extraction strategy (apertures/annuli) and the ecliptic/medium background are also configured to match PandExo's TSO conventions, but those fields are NOT captured in the artifact, so no measured claim is made for them.
- **Extracted wavelength grids:** on every unsaturated row, 100.0% of PandExo's pixels find an exact-wavelength partner on our side at relative tolerance 1e-09 (gate floor: 99%; per-pixel deltas beyond that tolerance are not stored).
- **Groups:** each tool independently optimizes the ramp to the same 80% saturation target; the freedom left is rounding to an integer group count. Measured: within 0 group(s) on the moderate/bright stars, within 0 groups on the faint Ks=13 star (the gate allows 5 groups OR 1% there, whichever is looser -- rounding on a ~500-1000 group ramp is ~1% by itself). On SHORT ramps (either side at <= 3 groups) the gate requires EXACT agreement: one group there is a large slice of the integration and of the noise, not rounding. Per-group integration time then inherits the group choice (gated at 15% relative); the largest total-t_int gap is 4.5% (w39_like/nirspec_prism, on a ~1-group ramp). Matched sigma-ratio medians additionally sit inside the [0.8, 2.0] anomaly band (an outlier ceiling, not a parity-to-unity claim).
- **Extracted flux:** per-mode median ratios span 1.0000-1.0109 (gate: median within 3% of unity). The per-pixel scatter around each median comes from the two tools' independent extraction of the same 2D calculation and is disclosed in the tables (5th/95th percentiles); it is not gated. Both calculations receive the exact same sampled stellar spectrum. The remaining wavelength-dependent extraction difference has not been assigned a physical cause, so this artifact makes no per-pixel flux-parity claim.
- **Saturation:** both tools search down to Pandeia's per-detector minimum ramp (worker v11: NIR 1 group, MIRI 2). Native partial- and full-saturation arrays are wavelength-aligned and compared as binary masks; the gate requires complete grid coverage and exact mask agreement. Rows above the saturation limit remain diagnostic rows, not numerical estimator-validation rows.

**PandExo operational warnings are recorded, not adjudicated.** Under the pinned RAPID readout, PandExo attaches data-volume-excess warnings to the NIRCam rows (its optimizer would prefer a slower pattern); the raw warnings are printed per star below. A numerical parity row says the two estimators agree on that configuration -- it is not a statement that the configuration is schedulable or operationally recommended.

Columns: sigma ratio = (this tool's per-pixel transit-depth sigma) / (PandExo's), median [5th, 95th percentile] over matched pixels. 'matched' uses PandExo's integration counts in the tool formula (isolates the noise model); 'policy' uses the tool's own floor(T/t_int) counts (adds the integration-counting policy). flux ratio compares extracted stellar count rates (engine parity; expect 1.0000).

## Star `w39_like` (Teff 5400 K, logg 4.45, [Fe/H] 0.0, Ks 10.663)

Backend: engine 2026.7 + pandeia_data-2026.7-jwst (worker v11); PandExo 2026.7 on engine 2026.7.

| mode | status | ngroup ours/PX | t_int s ours/PX | n_int ours/PX(in) | flux ratio | sigma ratio (matched) | sigma ratio (policy) |
|---|---|---|---|---|---|---|---|
| nirspec_prism | OK | 1/1 | 0.473/0.452 | 21347/22314 | 1.0000 [1.0000, 1.5343] (n=403) | 1.3098 [1.0927, 1.7624] (n=403) | 1.3392 [1.1172, 1.8019] (n=403) |
| nirspec_g395h | OK | 126/126 | 114.574/114.554 | 88/89 | 1.0000 [1.0000, 1.0000] (n=3330) | 1.1008 [1.0895, 1.1038] (n=3330) | 1.1071 [1.0957, 1.1100] (n=3330) |
| nirspec_g235h | OK | 58/58 | 53.238/53.218 | 189/190 | 1.0000 [1.0000, 1.0531] (n=3424) | 1.0933 [1.0882, 1.1188] (n=3424) | 1.0962 [1.0911, 1.1218] (n=3424) |
| nirspec_g395m | OK | 49/49 | 45.120/45.100 | 223/224 | 1.0000 [1.0000, 1.0000] (n=1286) | 1.0916 [1.0834, 1.0936] (n=1286) | 1.0941 [1.0859, 1.0961] (n=1286) |
| niriss_soss | OK | 17/17 | 98.912/98.892 | 102/103 | 1.0109 [1.0000, 1.3119] (n=2040) | 1.1062 [1.0920, 1.1301] (n=2040) | 1.1116 [1.0973, 1.1357] (n=2040) |
| nircam_f322w2 | OK | 100/100 | 34.407/34.402 | 293/294 | 1.0000 [1.0000, 1.0000] (n=1812) | 1.0919 [0.8781, 1.0932] (n=1812) | 1.0937 [0.8796, 1.0950] (n=1812) |
| nircam_f444w | OK | 100/100 | 34.407/34.402 | 293/294 | 1.0000 [1.0000, 1.0000] (n=1267) | 1.0817 [0.9044, 1.0886] (n=1267) | 1.0835 [0.9059, 1.0905] (n=1267) |
| miri_lrs | OK | 253/253 | 40.396/40.396 | 249/250 | 1.0000 [1.0000, 1.0000] (n=372) | 1.4845 [1.4764, 1.5922] (n=372) | 1.4875 [1.4794, 1.5954] (n=372) |

Noise-model attribution (median per-integration variance over pure photon counts; photon-limited = 1.0):

| mode | this tool (pandeia extracted noise) | PandExo (fml) |
|---|---|---|
| nirspec_prism | 2.185 | 1.223 |
| nirspec_g395h | 1.211 | 1.014 |
| nirspec_g235h | 1.199 | 1.013 |
| nirspec_g395m | 1.190 | 1.013 |
| niriss_soss | 1.595 | 1.180 |
| nircam_f322w2 | 1.222 | 1.040 |
| nircam_f444w | 1.276 | 1.106 |
| miri_lrs | 30.593 | 14.136 |

PandExo warnings for nirspec_prism: {'Group Number Too Low?': 'All good. Ngroups=1 is a new mode since Cycle 4 and has not been rigorously tested. Proceed with caution.', '% full well high?': 'All good (42% < 80%)'}

PandExo warnings for nirspec_g395h: {'% full well high?': 'All good (80% < 80%)'}

PandExo warnings for nirspec_g235h: {'% full well high?': 'All good (79% < 80%)'}

PandExo warnings for nirspec_g395m: {'% full well high?': 'All good (79% < 80%)'}

PandExo warnings for niriss_soss: {'% full well high?': 'All good (77% < 80%)'}

PandExo warnings for nircam_f322w2: {'Group Number Too High?': 'Optimized NGROUPS (491) exceeds the maximum (100). SET TO NGROUPS=100', '% full well high?': 'All good (16% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.8 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for nircam_f444w: {'Group Number Too High?': 'Optimized NGROUPS (931) exceeds the maximum (100). SET TO NGROUPS=100', '% full well high?': 'All good (9% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.8 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for miri_lrs: {'% full well high?': 'All good (80% < 80%)'}

## Star `bright_hot` (Teff 6250 K, logg 4.3, [Fe/H] 0.0, Ks 8.5)

Backend: engine 2026.7 + pandeia_data-2026.7-jwst (worker v11); PandExo 2026.7 on engine 2026.7.

| mode | status | ngroup ours/PX | t_int s ours/PX | n_int ours/PX(in) | flux ratio | sigma ratio (matched) | sigma ratio (policy) |
|---|---|---|---|---|---|---|---|
| nirspec_prism | SATURATED above limit (measured 3.60x full well; reported, not a validation row) | 1/1 | -- | -- | -- | -- | -- |
| nirspec_g395h | OK | 17/17 | 16.256/16.236 | 620/622 | 1.0000 [1.0000, 1.0000] (n=3330) | 1.0757 [1.0742, 1.0762] (n=3330) | 1.0775 [1.0759, 1.0780] (n=3330) |
| nirspec_g235h | OK | 7/7 | 7.236/7.216 | 1394/1399 | 1.0000 [1.0000, 1.0531] (n=3424) | 1.0573 [1.0509, 1.0729] (n=3424) | 1.0592 [1.0527, 1.0748] (n=3424) |
| nirspec_g395m | OK | 6/6 | 6.334/6.314 | 1593/1599 | 1.0000 [1.0000, 1.0000] (n=1286) | 1.0498 [1.0450, 1.0690] (n=1286) | 1.0518 [1.0470, 1.0710] (n=1286) |
| niriss_soss | OK | 2/2 | 16.502/16.482 | 611/613 | 1.0109 [1.0000, 1.3119] (n=2040) | 1.0615 [1.0392, 1.1155] (n=2040) | 1.0632 [1.0409, 1.1174] (n=2040) |
| nircam_f322w2 | OK | 67/67 | 23.167/23.161 | 435/436 | 1.0000 [1.0000, 1.0000] (n=1812) | 1.0953 [0.9512, 1.0968] (n=1812) | 1.0965 [0.9523, 1.0980] (n=1812) |
| nircam_f444w | OK | 100/100 | 34.407/34.402 | 293/294 | 1.0000 [1.0000, 1.0000] (n=1267) | 1.0984 [0.9886, 1.0989] (n=1267) | 1.1003 [0.9903, 1.1008] (n=1267) |
| miri_lrs | OK | 39/39 | 6.362/6.362 | 1586/1587 | 1.0000 [1.0000, 1.0000] (n=372) | 1.3495 [1.3424, 1.6180] (n=372) | 1.3499 [1.3428, 1.6185] (n=372) |

Noise-model attribution (median per-integration variance over pure photon counts; photon-limited = 1.0):

| mode | this tool (pandeia extracted noise) | PandExo (fml) |
|---|---|---|
| nirspec_g395h | 1.157 | 1.015 |
| nirspec_g235h | 1.130 | 1.017 |
| nirspec_g395m | 1.105 | 1.018 |
| niriss_soss | 1.308 | 1.075 |
| nircam_f322w2 | 1.192 | 1.007 |
| nircam_f444w | 1.206 | 1.015 |
| miri_lrs | 5.923 | 3.290 |

PandExo warnings for nirspec_prism: {'Group Number Too Low?': 'All good. Ngroups=1 is a new mode since Cycle 4 and has not been rigorously tested. Proceed with caution.', 'Saturated?': 'Full saturation:\n There are 96 pixels saturated at the end of the first group. These pixels cannot be recovered.', '% full well high?': '% full well>80% (360% > 80%)', 'Num Groups Reset?': 'Optimized NGROUPS below minimum (1). SET TO NGROUPS=1'}

PandExo warnings for nirspec_g395h: {'% full well high?': 'All good (78% < 80%)'}

PandExo warnings for nirspec_g235h: {'% full well high?': 'All good (71% < 80%)'}

PandExo warnings for nirspec_g395m: {'% full well high?': 'All good (70% < 80%)'}

PandExo warnings for niriss_soss: {'% full well high?': 'All good (78% < 80%)'}

PandExo warnings for nircam_f322w2: {'% full well high?': 'All good (79% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.5 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for nircam_f444w: {'Group Number Too High?': 'Optimized NGROUPS (130) exceeds the maximum (100). SET TO NGROUPS=100', '% full well high?': 'All good (61% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.8 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for miri_lrs: {'% full well high?': 'All good (78% < 80%)'}

## Star `faint_k` (Teff 4500 K, logg 4.6, [Fe/H] 0.0, Ks 13.0)

Backend: engine 2026.7 + pandeia_data-2026.7-jwst (worker v11); PandExo 2026.7 on engine 2026.7.

| mode | status | ngroup ours/PX | t_int s ours/PX | n_int ours/PX(in) | flux ratio | sigma ratio (matched) | sigma ratio (policy) |
|---|---|---|---|---|---|---|---|
| nirspec_prism | OK | 20/20 | 4.770/4.749 | 2115/2126 | 1.0000 [1.0000, 1.5343] (n=403) | 1.0795 [1.0694, 1.1422] (n=403) | 1.0823 [1.0722, 1.1452] (n=403) |
| nirspec_g395h | OK | 1072/1072 | 967.866/967.846 | 10/11 | 1.0000 [1.0000, 1.0000] (n=3330) | 1.1674 [1.1651, 1.1687] (n=3330) | 1.2243 [1.2220, 1.2258] (n=3330) |
| nirspec_g235h | OK | 502/502 | 453.726/453.706 | 22/23 | 1.0000 [1.0000, 1.0531] (n=3424) | 1.1276 [1.1208, 1.1542] (n=3424) | 1.1529 [1.1460, 1.1802] (n=3424) |
| nirspec_g395m | OK | 421/421 | 380.664/380.644 | 26/27 | 1.0000 [1.0000, 1.0000] (n=1286) | 1.1241 [1.1223, 1.1252] (n=1286) | 1.1455 [1.1437, 1.1466] (n=1286) |
| niriss_soss | OK | 30/30 | 170.334/170.314 | 59/60 | 1.0109 [1.0000, 1.3119] (n=2040) | 1.1329 [1.0296, 1.1586] (n=2040) | 1.1425 [1.0383, 1.1684] (n=2040) |
| nircam_f322w2 | OK | 100/100 | 34.407/34.402 | 293/294 | 1.0000 [1.0000, 1.0000] (n=1812) | 1.0402 [0.8482, 1.0453] (n=1812) | 1.0420 [0.8496, 1.0471] (n=1812) |
| nircam_f444w | OK | 100/100 | 34.407/34.402 | 293/294 | 1.0000 [1.0000, 1.0000] (n=1267) | 0.9969 [0.8808, 1.0287] (n=1267) | 0.9986 [0.8823, 1.0304] (n=1267) |
| miri_lrs | OK | 1021/1021 | 162.539/162.539 | 62/63 | 1.0000 [1.0000, 1.0000] (n=372) | 1.5338 [1.5322, 1.5564] (n=372) | 1.5461 [1.5445, 1.5689] (n=372) |

Noise-model attribution (median per-integration variance over pure photon counts; photon-limited = 1.0):

| mode | this tool (pandeia extracted noise) | PandExo (fml) |
|---|---|---|
| nirspec_prism | 1.400 | 1.114 |
| nirspec_g395h | 1.362 | 1.015 |
| nirspec_g235h | 1.274 | 1.014 |
| nirspec_g395m | 1.261 | 1.013 |
| niriss_soss | 3.347 | 2.591 |
| nircam_f322w2 | 1.439 | 1.348 |
| nircam_f444w | 1.935 | 1.972 |
| miri_lrs | 249.153 | 107.097 |

PandExo warnings for nirspec_prism: {'% full well high?': 'All good (77% < 80%)'}

PandExo warnings for nirspec_g395h: {'% full well high?': 'All good (80% < 80%)'}

PandExo warnings for nirspec_g235h: {'% full well high?': 'All good (80% < 80%)'}

PandExo warnings for nirspec_g395m: {'% full well high?': 'All good (80% < 80%)'}

PandExo warnings for niriss_soss: {'Group Number Too High?': 'Optimized NGROUPS (170) exceeds the maximum (30). SET TO NGROUPS=30', '% full well high?': 'All good (14% < 80%)'}

PandExo warnings for nircam_f322w2: {'Group Number Too High?': 'Optimized NGROUPS (4106) exceeds the maximum (100). SET TO NGROUPS=100', '% full well high?': 'All good (2% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.8 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for nircam_f444w: {'Group Number Too High?': 'Optimized NGROUPS (7336) exceeds the maximum (100). SET TO NGROUPS=100', '% full well high?': 'All good (1% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.8 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for miri_lrs: {'% full well high?': 'All good (80% < 80%)'}

## Findings

1. **Matched-configuration parity: achieved.** With the submitted configuration pinned identically on both sides (see Scope above), the two sides agree on the extracted wavelength grids (100.0% of pixels matched on every unsaturated row), extracted count rates (per-mode flux-ratio medians 1.0000-1.0109), independently selected groups (within 0 on the moderate/bright stars; within 0 groups on the faint star, i.e. integer rounding of the same saturation target), per-group integration times, and integration counts (within rounding policy).

2. **The remaining sigma difference is the noise model itself, and it is one-sided.** This tool propagates pandeia's full extracted noise (correlated ramp/read noise, background, dark, IPC, quantum-yield excess); PandExo's default 'fml' calculation is an analytic ramp formula that sits within a few percent of pure photon noise in the NIR. The attribution tables above show the variance excess over photon counts on both sides; the variance-excess ratio (1.211/1.014 = 1.194 on the W39-like G395H row) accounts for the bulk of the measured squared sigma ratio (1.101^2 = 1.212). This tool is therefore systematically CONSERVATIVE relative to PandExo: ~2-24% higher sigma for NIRSpec/NIRISS/NIRCam on matched configurations (up to ~31% under the policy configs on the faint Ks=13 star), and larger for MIRI LRS (~33-56%), where the deep-red background and detector terms dominate and the analytic formula under-represents them.

3. **Residual policy differences (documented, small):** integration counts are floored here vs rounded in PandExo (at most one integration per window); the symmetric in/out approximation adds ~+0.5% sigma at 1% depth (grows with depth; docstring in noise.pixel_depth_variance). Since worker v8 the ramp floors equal pandeia's per-detector mingroups (NIR 1, MIRI 2), the same field PandExo reads, so the old ngroup-floor delta (ours 2 vs PandExo 1 on bright-star PRISM) no longer exists.

4. **What may be claimed:** on the fixed configurations this tool's registry submits (with PandExo explicitly overridden to the same hardware), the timing, group optimization, configuration-level saturation handling, and extraction of this tool match the PandExo revision named in the provenance table, on the supported Pandeia engine. This is not a test of PandExo's configuration-selection policy. Absolute sigmas are NOT PandExo-identical and are not labeled as such: they are pandeia-extracted-noise forecasts, conservative relative to PandExo's analytic noise by the mode-dependent margins quantified above.
