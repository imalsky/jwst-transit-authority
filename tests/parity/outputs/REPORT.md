# PandExo numerical parity report

Generated 2026-08-03 by `run_parity.py` + `make_report.py` in this directory.

> **GATE: NOT EVALUATED.** This summary predates the fail-closed release gate, so nothing here has been checked for worker version, engine/refdata/PSF release agreement, PandExo identity, saturated rows, or the numerical thresholds. It is a forensic artifact, NOT a release certificate. Regenerate with `run_parity.py`.

## Provenance

| star | side | engine | refdata | PSFs | worker / PandExo |
|---|---|---|---|---|---|
| w39_like | this tool | 2026.2 | 2026.2 | not recorded | worker v5 |
| w39_like | PandExo | 2026.2 | not recorded | not recorded | unknown @ commit not r |
| bright_hot | this tool | 2026.2 | 2026.2 | not recorded | worker v5 |
| bright_hot | PandExo | 2026.2 | not recorded | not recorded | unknown @ commit not r |
| faint_k | this tool | 2026.2 | 2026.2 | not recorded | worker v5 |
| faint_k | PandExo | 2026.2 | not recorded | not recorded | 2026.2 @ commit not r |

Both sides in this historical artifact ran on the SAME archival Pandeia 2026.2 backend, so its internal differences isolate estimator/policy effects. It is not evidence for the supported 2026.7 backend or the reviewed PandExo commit and therefore cannot gate a release. Configuration: constant transit depth 0.01, transit duration 2.8036 h, equal out-of-transit baseline, saturation limit 80%, no noise floor, native (R=None) grids.

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

Backend: engine 2026.2 + pandeia_data-2026.2-jwst (worker v5); PandExo unknown on engine 2026.2.

| mode | status | ngroup ours/PX | t_int s ours/PX | n_int ours/PX(in) | flux ratio | sigma ratio (matched) | sigma ratio (policy) |
|---|---|---|---|---|---|---|---|
| nirspec_prism | OK | 2/1 | 0.699/0.452 | 14439/22314 | 0.9976 [0.9663, 1.5317] (n=403) | 1.2066 [1.0587, 1.5140] (n=403) | 1.5000 [1.3162, 1.8821] (n=403) |
| nirspec_g395h | OK | 124/125 | 112.770/113.652 | 89/89 | 0.9973 [0.9436, 1.0392] (n=3330) | 1.1078 [1.0767, 1.1330] (n=3330) | 1.1078 [1.0767, 1.1330] (n=3330) |
| nirspec_g235h | OK | 57/57 | 52.336/52.316 | 192/193 | 1.0009 [0.9654, 1.0705] (n=3424) | 1.0952 [1.0655, 1.1307] (n=3424) | 1.0981 [1.0683, 1.1336] (n=3424) |
| niriss_soss | OK | 19/19 | 109.900/109.880 | 91/92 | 1.0203 [0.9763, 1.2934] (n=2040) | 1.1158 [1.0837, 1.1509] (n=2040) | 1.1219 [1.0896, 1.1572] (n=2040) |
| nircam_f322w2 | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9963 [0.9592, 1.0098] (n=1812) | 1.0928 [0.9172, 1.1026] (n=1812) | 1.0947 [0.9188, 1.1044] (n=1812) |
| nircam_f444w | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9895 [0.9433, 1.0454] (n=1267) | 1.0872 [0.9341, 1.1107] (n=1267) | 1.0890 [0.9357, 1.1126] (n=1267) |
| miri_lrs | OK | 253/252 | 40.237/40.237 | 250/251 | 0.9974 [0.9951, 0.9984] (n=372) | 1.4848 [1.4775, 1.5945] (n=372) | 1.4878 [1.4804, 1.5977] (n=372) |

Noise-model attribution (median per-integration variance over pure photon counts; photon-limited = 1.0):

| mode | this tool (pandeia extracted noise) | PandExo (fml) |
|---|---|---|
| nirspec_prism | 1.914 | 1.221 |
| nirspec_g395h | 1.220 | 1.014 |
| nirspec_g235h | 1.199 | 1.013 |
| niriss_soss | 1.527 | 1.119 |
| nircam_f322w2 | 1.222 | 1.040 |
| nircam_f444w | 1.277 | 1.106 |
| miri_lrs | 30.476 | 14.103 |

PandExo warnings for nirspec_prism: {'Group Number Too Low?': 'All good. Ngroups=1 is a new mode since Cycle 4 and has not been rigorously tested. Proceed with caution.'}

PandExo warnings for nircam_f322w2: {'Group Number Too High?': 'Optimized NGROUPS above maximum (100). SET TO NGROUPS=100'}

PandExo warnings for nircam_f444w: {'Group Number Too High?': 'Optimized NGROUPS above maximum (100). SET TO NGROUPS=100'}

## Star `bright_hot` (Teff 6250 K, logg 4.3, [Fe/H] 0.0, Ks 8.5)

Backend: engine 2026.2 + pandeia_data-2026.2-jwst (worker v5); PandExo unknown on engine 2026.2.

| mode | status | ngroup ours/PX | t_int s ours/PX | n_int ours/PX(in) | flux ratio | sigma ratio (matched) | sigma ratio (policy) |
|---|---|---|---|---|---|---|---|
| nirspec_prism | OK | 2/1 | 0.699/0.452 | 14439/22314 | 1.0002 [0.9858, 1.0198] (n=265) | 1.0751 [1.0335, 1.1655] (n=265) | 1.3365 [1.2848, 1.4489] (n=265) |
| nirspec_g395h | OK | 16/17 | 15.354/16.236 | 657/622 | 1.0013 [0.9803, 1.0254] (n=3330) | 1.1099 [1.0964, 1.1222] (n=3330) | 1.0799 [1.0668, 1.0919] (n=3330) |
| nirspec_g235h | OK | 7/7 | 7.236/7.216 | 1394/1399 | 1.0037 [0.9777, 1.0602] (n=3424) | 1.0567 [1.0445, 1.0802] (n=3424) | 1.0586 [1.0463, 1.0821] (n=3424) |
| niriss_soss | OK | 2/2 | 16.502/16.482 | 611/613 | 1.0148 [0.9881, 1.3051] (n=2040) | 1.0648 [1.0281, 1.1186] (n=2040) | 1.0666 [1.0298, 1.1205] (n=2040) |
| nircam_f322w2 | OK | 67/67 | 23.167/23.161 | 435/436 | 1.0008 [0.9636, 1.0161] (n=1812) | 1.0944 [0.9699, 1.1031] (n=1812) | 1.0956 [0.9710, 1.1044] (n=1812) |
| nircam_f444w | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9980 [0.9652, 1.0242] (n=1267) | 1.0980 [1.0092, 1.1094] (n=1267) | 1.0999 [1.0109, 1.1113] (n=1267) |
| miri_lrs | OK | 39/39 | 6.203/6.362 | 1627/1587 | 1.0011 [1.0003, 1.0018] (n=372) | 1.3484 [1.3411, 1.6160] (n=372) | 1.3317 [1.3245, 1.5961] (n=372) |

Noise-model attribution (median per-integration variance over pure photon counts; photon-limited = 1.0):

| mode | this tool (pandeia extracted noise) | PandExo (fml) |
|---|---|---|
| nirspec_prism | 1.221 | 1.068 |
| nirspec_g395h | 1.234 | 1.015 |
| nirspec_g235h | 1.130 | 1.017 |
| niriss_soss | 1.299 | 1.067 |
| nircam_f322w2 | 1.191 | 1.007 |
| nircam_f444w | 1.206 | 1.015 |
| miri_lrs | 5.923 | 3.293 |

PandExo warnings for nirspec_prism: {'Group Number Too Low?': 'All good. Ngroups=1 is a new mode since Cycle 4 and has not been rigorously tested. Proceed with caution.', 'Saturated?': 'Full saturation:\n There are 98 pixels saturated at the end of the first group. These pixels cannot be recovered.', 'Num Groups Reset?': 'Optimized NGROUPS below minimum (1). SET TO NGROUPS=1'}

PandExo warnings for nircam_f444w: {'Group Number Too High?': 'Optimized NGROUPS above maximum (100). SET TO NGROUPS=100'}

## Star `faint_k` (Teff 4500 K, logg 4.6, [Fe/H] 0.0, Ks 13.0)

Backend: engine 2026.2 + pandeia_data-2026.2-jwst (worker v5); PandExo 2026.2 on engine 2026.2.

| mode | status | ngroup ours/PX | t_int s ours/PX | n_int ours/PX(in) | flux ratio | sigma ratio (matched) | sigma ratio (policy) |
|---|---|---|---|---|---|---|---|
| nirspec_prism | OK | 20/20 | 4.770/4.749 | 2115/2126 | 0.9946 [0.9453, 1.5037] (n=403) | 1.0874 [1.0233, 1.1524] (n=403) | 1.0902 [1.0259, 1.1554] (n=403) |
| nirspec_g395h | OK | 1063/1059 | 959.748/956.120 | 10/11 | 0.9925 [0.9130, 1.1100] (n=3330) | 1.1696 [1.1022, 1.2183] (n=3330) | 1.2267 [1.1560, 1.2777] (n=3330) |
| nirspec_g235h | OK | 497/492 | 449.216/444.686 | 22/23 | 1.0014 [0.9485, 1.0897] (n=3424) | 1.1239 [1.0834, 1.1668] (n=3424) | 1.1491 [1.1078, 1.1930] (n=3424) |
| niriss_soss | OK | 195/193 | 1076.844/1065.836 | 9/10 | 1.0296 [0.9640, 1.2897] (n=2040) | 1.2412 [1.1497, 1.3197] (n=2040) | 1.3083 [1.2119, 1.3911] (n=2040) |
| nircam_f322w2 | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9913 [0.9520, 1.0520] (n=1812) | 1.0437 [0.8911, 1.0589] (n=1812) | 1.0454 [0.8926, 1.0607] (n=1812) |
| nircam_f444w | OK | 100/100 | 34.407/34.402 | 293/294 | 0.9867 [0.9229, 1.1058] (n=1267) | 1.0177 [0.8967, 1.0543] (n=1267) | 1.0195 [0.8983, 1.0561] (n=1267) |
| miri_lrs | OK | 1021/1017 | 162.380/161.903 | 62/63 | 0.9923 [0.9864, 0.9946] (n=372) | 1.5430 [1.5398, 1.5612] (n=372) | 1.5554 [1.5522, 1.5737] (n=372) |

Noise-model attribution (median per-integration variance over pure photon counts; photon-limited = 1.0):

| mode | this tool (pandeia extracted noise) | PandExo (fml) |
|---|---|---|
| nirspec_prism | 1.396 | 1.113 |
| nirspec_g395h | 1.356 | 1.015 |
| nirspec_g235h | 1.261 | 1.014 |
| niriss_soss | 3.083 | 1.937 |
| nircam_f322w2 | 1.434 | 1.344 |
| nircam_f444w | 1.947 | 1.963 |
| miri_lrs | 249.215 | 106.315 |

PandExo warnings for nircam_f322w2: {'Group Number Too High?': 'Optimized NGROUPS above maximum (100). SET TO NGROUPS=100'}

PandExo warnings for nircam_f444w: {'Group Number Too High?': 'Optimized NGROUPS above maximum (100). SET TO NGROUPS=100'}

## Findings

1. **Configuration parity: achieved.** With the registry's explicit TSO readout patterns, PandExo's extraction strategy (apertures/annuli), and the ecliptic/medium background, the two sides agree on the extracted wavelength grids (every pixel matches), extracted count rates (flux-ratio medians 0.987-1.03), selected groups (within 1 on the moderate/bright stars; within ~1% relative, up to 5 groups absolute, on the faint Ks=13 star), per-group integration times (<0.1%; total t_int inherits the group choice, up to ~5.5% where one group of ~16 dominates), and integration counts (within rounding policy). Saturation behavior matches: pixels PandExo masks as saturated are the pixels this tool excludes.

2. **The remaining sigma difference is the noise model itself, and it is one-sided.** This tool propagates pandeia's full extracted noise (correlated ramp/read noise, background, dark, IPC, quantum-yield excess); PandExo's default 'fml' calculation is an analytic ramp formula that sits within a few percent of pure photon noise in the NIR. The attribution tables above show the variance excess over photon counts on both sides; their ratio reproduces the observed sigma ratios (e.g. G395H: 1.220/1.014 = 1.203 ~= 1.108^2). This tool is therefore systematically CONSERVATIVE relative to PandExo: ~2-24% higher sigma for NIRSpec/NIRISS/NIRCam on matched configurations (up to ~31% under the policy configs on the faint Ks=13 star), and larger for MIRI LRS (~33-56%), where the deep-red background and detector terms dominate and the analytic formula under-represents them. For context, published achieved-vs-PandExo noise ratios (COMPASS G395H 1.05-1.12; MIRI LRS ~1.15-1.2 above random-noise simulations) fall on the same side, between the two models for NIRSpec.

3. **Residual policy differences (documented, small):** integration counts are floored here vs rounded in PandExo (at most one integration per window); ngroup_min is 2 here while PandExo will select 1 group (PRISM on a bright star); the symmetric in/out approximation adds ~+0.5% sigma at 1% depth (grows with depth; docstring in noise.pixel_depth_variance).

4. **What may be claimed:** NOTHING, until this artifact passes its gate. The rows above are forensic measurements of whatever engine, refdata, PSF tree, worker version, and PandExo revision the provenance table names -- which is not necessarily the shipping configuration. Regenerate on the supported release before citing any parity claim.
