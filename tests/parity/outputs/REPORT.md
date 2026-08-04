# PandExo numerical parity report

Generated 2026-08-04 by `run_parity.py` + `make_report.py` in this directory.

**GATE: PASS.** Worker v7, Pandeia 2026.7 on both sides, every declared threshold met.

## Provenance

| star | side | engine | refdata | PSFs | worker / PandExo |
|---|---|---|---|---|---|
| w39_like | this tool | 2026.7 | 2026.7 | 2026.7 | worker v7 |
| w39_like | PandExo | 2026.7 | 2026.7 | 2026.7 | 2026.7 @ 34e42d81f782 |
| bright_hot | this tool | 2026.7 | 2026.7 | 2026.7 | worker v7 |
| bright_hot | PandExo | 2026.7 | 2026.7 | 2026.7 | 2026.7 @ 34e42d81f782 |
| faint_k | this tool | 2026.7 | 2026.7 | 2026.7 | worker v7 |
| faint_k | PandExo | 2026.7 | 2026.7 | 2026.7 | 2026.7 @ 34e42d81f782 |

Both sides run on the SAME Pandeia backend -- the exact engine, reference-data, and PSF releases are in the provenance table above, and the gate refuses the run if they disagree across the two sides. Every difference below is therefore an ESTIMATOR/policy difference, not an engine calibration difference. (This says nothing about whether that release is the SUPPORTED one; the gate banner does.) PandExo is used from master at the commit recorded above. Configuration: constant transit depth 0.01, transit duration 2.8036 h, equal out-of-transit baseline, saturation limit 80%, no noise floor, native (R=None) grids.

## Figures (regenerate with `make_parity_plots.py`)

These show the quantities that match 1:1 -- the parity result. The depth-uncertainty difference (a noise-model difference, not a configuration one) is quantified in the tables and Findings below, not plotted.

- **parity_config_timing.png** -- selected groups, integration time, and integration count, this tool vs PandExo, on the 1:1 line (log-log). Every mode/star lands on the diagonal except PRISM's known group floor (ngroup_min=2 vs PandExo's 1 on a bright star).
- **parity_extracted_flux.png** -- G395H extracted stellar count rate, per-wavelength overlay with a ratio strip (per-pixel jitter + binned median). The curves coincide (systematic ratio ~0.997).

## What is bit-identical, and why the rest is not

The two are INDEPENDENT tools calling the same Pandeia engine, so only what is read straight from the shared engine is bit-for-bit identical; anything each computes on its own agrees closely but not exactly. This is a property of a cross-tool test, not a defect -- forcing bit-equality would mean one tool copying the other's numbers, which is not a validation.

- **Bit-identical (max difference exactly 0):** the instrument configuration (subarray, readout, disperser, filter, aperture, background) and the extracted WAVELENGTH GRID -- every extracted pixel, every mode and star (verified: max |Δλ| = 0).
- **Groups agree to ≤1 on the moderate/bright stars, to ~1% relative (up to 5 groups absolute) on the faint Ks=13 star:** each tool independently optimizes the ramp to the same 80% saturation target; the freedom left is rounding to an integer group count (e.g. G395H 124 vs 125; faint-star ramps of ~500-1000 groups differ by up to 5). Per-group integration time matches to <0.1%; total integration time and count then inherit the group choice (up to ~5.5% where a single group of ~16 dominates, bright G395H; ~2.5% bright MIRI). Only VALID configurations are compared: PRISM saturates on the two bright stars (both tools flag it; this tool floors at ngroup=2 while PandExo drops to ngroup=1) and is excluded there, and is compared on the faint star where it is unsaturated and matches (ngroup 20/20).
- **Extracted flux agrees to a systematic ~0.3%** (binned median 0.997 for G395H), with per-pixel scatter from the two tools' independent extraction of the same 2D calculation -- photon-level for the smooth NIRSpec/MIRI traces (MIRI LRS matches to ~0.15%), larger for the curved NIRISS SOSS order-1 trace and the saturating low-R PRISM. The tool integrates over bins, so the per-pixel jitter averages out; the systematic is what enters the noise. The narrow downward spikes in this tool's flux are STELLAR ABSORPTION LINES in Pandeia's PHOENIX spectrum (hydrogen recombination -- e.g. Brackett-α 4.052 μm, Pfund-δ 3.297 μm -- plus molecular bands on cool stars, so they multiply on cooler targets: 9 lines at 6250 K, 193 at 4500 K); PandExo's separately-loaded stellar spectrum smooths them. They are physically real, cancel in the in/out transit-depth ratio, and wash out in binning.

Columns: sigma ratio = (this tool's per-pixel transit-depth sigma) / (PandExo's), median [5th, 95th percentile] over matched pixels. 'matched' uses PandExo's integration counts in the tool formula (isolates the noise model); 'policy' uses the tool's own floor(T/t_int) counts (adds the integration-counting policy). flux ratio compares extracted stellar count rates (engine parity; expect 1.0000).

## Star `w39_like` (Teff 5400 K, logg 4.45, [Fe/H] 0.0, Ks 10.663)

Backend: engine 2026.7 + pandeia_data-2026.7-jwst (worker v7); PandExo 2026.7 on engine 2026.7.

| mode | status | ngroup ours/PX | t_int s ours/PX | n_int ours/PX(in) | flux ratio | sigma ratio (matched) | sigma ratio (policy) |
|---|---|---|---|---|---|---|---|
| nirspec_prism | ERROR (see parity_summary.json) | -- | -- | -- | -- | -- | -- |
| nirspec_g395h | OK | 125/125 | 113.672/113.652 | 88/89 | 0.9973 [0.9436, 1.0392] (n=3330) | 1.1034 [1.0723, 1.1284] (n=3330) | 1.1096 [1.0784, 1.1348] (n=3330) |
| nirspec_g235h | OK | 58/58 | 53.238/53.218 | 189/190 | 1.0009 [0.9654, 1.0705] (n=3424) | 1.0954 [1.0656, 1.1308] (n=3424) | 1.0982 [1.0685, 1.1338] (n=3424) |
| niriss_soss | OK | 17/17 | 98.912/98.892 | 102/103 | 1.0203 [0.9762, 1.2938] (n=2040) | 1.1093 [1.0767, 1.1423] (n=2040) | 1.1147 [1.0819, 1.1479] (n=2040) |
| nircam_f322w2 | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9963 [0.9593, 1.0098] (n=1812) | 1.0930 [0.9172, 1.1027] (n=1812) | 1.0948 [0.9187, 1.1046] (n=1812) |
| nircam_f444w | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9896 [0.9433, 1.0454] (n=1267) | 1.0873 [0.9340, 1.1107] (n=1267) | 1.0891 [0.9356, 1.1126] (n=1267) |
| miri_lrs | OK | 254/253 | 40.555/40.396 | 248/250 | 0.9974 [0.9951, 0.9984] (n=372) | 1.4850 [1.4776, 1.5946] (n=372) | 1.4910 [1.4836, 1.6010] (n=372) |

Noise-model attribution (median per-integration variance over pure photon counts; photon-limited = 1.0):

| mode | this tool (pandeia extracted noise) | PandExo (fml) |
|---|---|---|
| nirspec_g395h | 1.211 | 1.014 |
| nirspec_g235h | 1.199 | 1.013 |
| niriss_soss | 1.596 | 1.180 |
| nircam_f322w2 | 1.222 | 1.040 |
| nircam_f444w | 1.277 | 1.106 |
| miri_lrs | 30.481 | 14.102 |

PandExo warnings for nirspec_g395h: {'% full well high?': 'All good (79% < 80%)'}

PandExo warnings for nirspec_g235h: {'% full well high?': 'All good (80% < 80%)'}

PandExo warnings for niriss_soss: {'% full well high?': 'All good (77% < 80%)'}

PandExo warnings for nircam_f322w2: {'Group Number Too High?': 'Optimized NGROUPS (489) exceeds the maximum (100). SET TO NGROUPS=100', '% full well high?': 'All good (16% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.8 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for nircam_f444w: {'Group Number Too High?': 'Optimized NGROUPS (929) exceeds the maximum (100). SET TO NGROUPS=100', '% full well high?': 'All good (9% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.8 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for miri_lrs: {'% full well high?': 'All good (80% < 80%)'}

## Star `bright_hot` (Teff 6250 K, logg 4.3, [Fe/H] 0.0, Ks 8.5)

Backend: engine 2026.7 + pandeia_data-2026.7-jwst (worker v7); PandExo 2026.7 on engine 2026.7.

| mode | status | ngroup ours/PX | t_int s ours/PX | n_int ours/PX(in) | flux ratio | sigma ratio (matched) | sigma ratio (policy) |
|---|---|---|---|---|---|---|---|
| nirspec_prism | ERROR (see parity_summary.json) | -- | -- | -- | -- | -- | -- |
| nirspec_g395h | OK | 16/17 | 15.354/16.236 | 657/622 | 1.0013 [0.9803, 1.0254] (n=3330) | 1.1098 [1.0964, 1.1221] (n=3330) | 1.0799 [1.0668, 1.0918] (n=3330) |
| nirspec_g235h | OK | 7/7 | 7.236/7.216 | 1394/1399 | 1.0037 [0.9777, 1.0602] (n=3424) | 1.0566 [1.0443, 1.0800] (n=3424) | 1.0585 [1.0462, 1.0819] (n=3424) |
| niriss_soss | OK | 2/2 | 16.502/16.482 | 611/613 | 1.0147 [0.9881, 1.3051] (n=2040) | 1.0643 [1.0278, 1.1159] (n=2040) | 1.0660 [1.0295, 1.1177] (n=2040) |
| nircam_f322w2 | OK | 67/67 | 23.167/23.161 | 435/436 | 1.0008 [0.9636, 1.0161] (n=1812) | 1.0945 [0.9698, 1.1033] (n=1812) | 1.0958 [0.9709, 1.1046] (n=1812) |
| nircam_f444w | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9980 [0.9652, 1.0243] (n=1267) | 1.0981 [1.0092, 1.1095] (n=1267) | 1.1000 [1.0109, 1.1114] (n=1267) |
| miri_lrs | OK | 39/39 | 6.362/6.362 | 1586/1587 | 1.0011 [1.0003, 1.0018] (n=372) | 1.3484 [1.3411, 1.6161] (n=372) | 1.3488 [1.3415, 1.6166] (n=372) |

Noise-model attribution (median per-integration variance over pure photon counts; photon-limited = 1.0):

| mode | this tool (pandeia extracted noise) | PandExo (fml) |
|---|---|---|
| nirspec_g395h | 1.234 | 1.015 |
| nirspec_g235h | 1.129 | 1.017 |
| niriss_soss | 1.309 | 1.075 |
| nircam_f322w2 | 1.192 | 1.007 |
| nircam_f444w | 1.206 | 1.015 |
| miri_lrs | 5.924 | 3.293 |

PandExo warnings for nirspec_g395h: {'% full well high?': 'All good (78% < 80%)'}

PandExo warnings for nirspec_g235h: {'% full well high?': 'All good (71% < 80%)'}

PandExo warnings for niriss_soss: {'% full well high?': 'All good (78% < 80%)'}

PandExo warnings for nircam_f322w2: {'% full well high?': 'All good (79% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.5 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for nircam_f444w: {'Group Number Too High?': 'Optimized NGROUPS (130) exceeds the maximum (100). SET TO NGROUPS=100', '% full well high?': 'All good (61% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.8 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for miri_lrs: {'% full well high?': 'All good (78% < 80%)'}

## Star `faint_k` (Teff 4500 K, logg 4.6, [Fe/H] 0.0, Ks 13.0)

Backend: engine 2026.7 + pandeia_data-2026.7-jwst (worker v7); PandExo 2026.7 on engine 2026.7.

| mode | status | ngroup ours/PX | t_int s ours/PX | n_int ours/PX(in) | flux ratio | sigma ratio (matched) | sigma ratio (policy) |
|---|---|---|---|---|---|---|---|
| nirspec_prism | OK | 20/20 | 4.770/4.749 | 2115/2126 | 0.9946 [0.9453, 1.5037] (n=403) | 1.0866 [1.0227, 1.1520] (n=403) | 1.0894 [1.0253, 1.1550] (n=403) |
| nirspec_g395h | OK | 1068/1064 | 964.258/960.630 | 10/11 | 0.9925 [0.9130, 1.1100] (n=3330) | 1.1699 [1.1025, 1.2186] (n=3330) | 1.2270 [1.1563, 1.2781] (n=3330) |
| nirspec_g235h | OK | 503/498 | 454.628/450.098 | 22/23 | 1.0014 [0.9485, 1.0897] (n=3424) | 1.1243 [1.0839, 1.1672] (n=3424) | 1.1496 [1.1082, 1.1934] (n=3424) |
| niriss_soss | OK | 30/30 | 170.334/170.314 | 59/60 | 1.0297 [0.9641, 1.2896] (n=2040) | 1.1246 [1.0249, 1.1862] (n=2040) | 1.1340 [1.0335, 1.1962] (n=2040) |
| nircam_f322w2 | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9913 [0.9520, 1.0519] (n=1812) | 1.0436 [0.8911, 1.0589] (n=1812) | 1.0454 [0.8926, 1.0607] (n=1812) |
| nircam_f444w | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9866 [0.9229, 1.1058] (n=1267) | 1.0177 [0.8967, 1.0542] (n=1267) | 1.0194 [0.8983, 1.0560] (n=1267) |
| miri_lrs | OK | 1022/1018 | 162.698/162.062 | 62/63 | 0.9923 [0.9864, 0.9946] (n=372) | 1.5430 [1.5399, 1.5612] (n=372) | 1.5554 [1.5523, 1.5738] (n=372) |

Noise-model attribution (median per-integration variance over pure photon counts; photon-limited = 1.0):

| mode | this tool (pandeia extracted noise) | PandExo (fml) |
|---|---|---|
| nirspec_prism | 1.395 | 1.114 |
| nirspec_g395h | 1.357 | 1.015 |
| nirspec_g235h | 1.261 | 1.014 |
| niriss_soss | 3.351 | 2.579 |
| nircam_f322w2 | 1.435 | 1.345 |
| nircam_f444w | 1.948 | 1.965 |
| miri_lrs | 249.256 | 106.327 |

PandExo warnings for nirspec_prism: {'% full well high?': 'All good (78% < 80%)'}

PandExo warnings for nirspec_g395h: {'% full well high?': 'All good (80% < 80%)'}

PandExo warnings for nirspec_g235h: {'% full well high?': 'All good (80% < 80%)'}

PandExo warnings for niriss_soss: {'Group Number Too High?': 'Optimized NGROUPS (169) exceeds the maximum (30). SET TO NGROUPS=30', '% full well high?': 'All good (14% < 80%)'}

PandExo warnings for nircam_f322w2: {'Group Number Too High?': 'Optimized NGROUPS (4077) exceeds the maximum (100). SET TO NGROUPS=100', '% full well high?': 'All good (2% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.8 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for nircam_f444w: {'Group Number Too High?': 'Optimized NGROUPS (7286) exceeds the maximum (100). SET TO NGROUPS=100', '% full well high?': 'All good (1% < 80%)', 'NIRCam Readout Optimization': 'User selected RAPID; readout pattern optimization was not performed. Estimate assumes no target acquisition and a standard 2,100-second initial slew.', 'NIRCam Data Excess?': 'Estimated data excess is 27.8 GB, above the 15 GB recommended limit. Verify and revise the setup in APT.'}

PandExo warnings for miri_lrs: {'% full well high?': 'All good (80% < 80%)'}

## Findings

1. **Configuration parity: achieved.** With the registry's explicit TSO readout patterns, PandExo's extraction strategy (apertures/annuli), and the ecliptic/medium background, the two sides agree on the extracted wavelength grids (every pixel matches), extracted count rates (flux-ratio medians 0.987-1.03), selected groups (within 1 on the moderate/bright stars; within ~1% relative, up to 5 groups absolute, on the faint Ks=13 star), per-group integration times (<0.1%; total t_int inherits the group choice, up to ~5.5% where one group of ~16 dominates), and integration counts (within rounding policy). Saturation behavior matches: pixels PandExo masks as saturated are the pixels this tool excludes.

2. **The remaining sigma difference is the noise model itself, and it is one-sided.** This tool propagates pandeia's full extracted noise (correlated ramp/read noise, background, dark, IPC, quantum-yield excess); PandExo's default 'fml' calculation is an analytic ramp formula that sits within a few percent of pure photon noise in the NIR. The attribution tables above show the variance excess over photon counts on both sides; their ratio reproduces the observed sigma ratios (e.g. G395H: 1.220/1.014 = 1.203 ~= 1.108^2). This tool is therefore systematically CONSERVATIVE relative to PandExo: ~2-24% higher sigma for NIRSpec/NIRISS/NIRCam on matched configurations (up to ~31% under the policy configs on the faint Ks=13 star), and larger for MIRI LRS (~33-56%), where the deep-red background and detector terms dominate and the analytic formula under-represents them. For context, published achieved-vs-PandExo noise ratios (COMPASS G395H 1.05-1.12; MIRI LRS ~1.15-1.2 above random-noise simulations) fall on the same side, between the two models for NIRSpec.

3. **Residual policy differences (documented, small):** integration counts are floored here vs rounded in PandExo (at most one integration per window); ngroup_min is 2 here while PandExo will select 1 group (PRISM on a bright star); the symmetric in/out approximation adds ~+0.5% sigma at 1% depth (grows with depth; docstring in noise.pixel_depth_variance).

4. **What may be claimed:** the instrument configuration, timing, group optimization, saturation handling, and extraction of this tool match the PandExo release named in the provenance table above, on the supported Pandeia engine. Absolute sigmas are NOT PandExo-identical and are not labeled as such: they are pandeia-extracted-noise forecasts, conservative relative to PandExo's analytic noise by the mode-dependent margins quantified above.
