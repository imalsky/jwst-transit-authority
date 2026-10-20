# JWST Transit Authority notes

The evidence behind each rule in CLAUDE.md, the design records that still govern the
code, the accepted limitations and the do-not-retry register. Open work is in todo.md.
Numbers are measured [M] unless tagged [E] (estimated) or [I] (inferred). A version or
date next to a number names the code state it was measured on.

## Current state

- Tool 0.72.0 on main after the 2026-10-20 cleanup. VULCAN is the default chemistry engine,
  PICASO the second.
- Floors `vulcan-forward>=0.29.0`, `vulcan3>=0.18.0` (pyproject). `deploy/pins.env` and
  the Dockerfile pin vulcan3 92c3b1e and vulcan-forward 32adc85 (the 2026-10-20
  cleanup); the Dockerfile pins the tool at cd67563. The running Space was last
  deployed from the earlier pins (tool b23df79) and is unchanged until a redeploy.
- Cache versions: `forward._VERSION` 60, `noise.WORKER_VERSION` 12,
  `adjoint_diag._ADJ_VERSION` 4, `picaso_engine.WORKER_VERSION` 3. Default keys:
  transmission `8c37a8320c19c75a` (test-pinned), emission `f4b94ee63633f386`.
- Fixtures `tests/live/data/w39b_v60_reference.npz` and `w39b_picaso_reference.npz`
  (cold WASP-39 b, both modes; jax 0.6.2, vulcan3 0.18.0, vulcan-forward 0.29.0; 12.6 and
  42.8 min on daw).
- Tests. Normal suite `python -m pytest tests -q -n 6`; `tests/conftest.py` defaults
  `JAX_PLATFORMS=cpu` and points an unset `JWST_TOOL_OUTPUT_DIR` at a throwaway
  directory. `tests/unit` on daw (6 pinned cores, 2026-10-19): 405 passed, 4 skipped in
  35-38 s. CI runs the unit suite on Python 3.10-3.12 plus a GUI-smoke job;
  `full-stack-smoke.yml` runs by hand. Slow tier (`JWST_TOOL_RUN_SLOW=1`, serial, 8
  cores, tool 0.71.0): 345 passed in 4 h 07 min; longest: emission AD/FD closure 4310 s
  (config cadence) / 2551 s (cadence 1), TOI-7169 b escalation 3498 s, transmission
  closure 1933 / 1381 s, e2e 512 s per mode.
- Space, last read from the Hub API on 2026-10-14: `imalsky/jwst-transit-authority`
  RUNNING on cpu-basic (2 vCPU / 16 GB), sleep 48 h. The four live checks
  (cold transmission run, cold emission run, configuration export and reload,
  out-of-range metallicity refusal with its alert mail) last ran on the 0.67.0 deploy.
- daw environment: the root uv env `VULCAN_Project/.venv` also carries pandeia.engine
  2026.7, so `JWST_TOOL_PANDEIA_PYTHON` is that interpreter. Rebuild the editable
  metadata after a version bump (`uv pip install --no-deps -e vulcan-jwst-tool`):
  `provenance._versions` reads metadata. PICASO env `VULCAN_Project/.picaso-env`
  (micromamba), pRT env `VULCAN_Project/.prt-env`. PandExo is installed in no env, so a
  parity re-run reuses the committed PandExo side (`PARITY_REUSE_PANDEXO=1`). No env
  carries `huggingface_hub` and no `hf` CLI is on PATH. `unset PYTHONSAFEPATH` first.

## Composition, C/O bounds and networks

- Convention: metallicity scales O, C, N, S together with He/H fixed (Moses et al.
  2011); C/O moves carbon at fixed metallicity-scaled oxygen (Tsai et al. 2017). Codes
  that anchor on carbon (petitRADTRANS chemistry, ATMO, Goyal et al. 2018) differ first
  order in ln(C/O): at C/O 0.7 both C and O differ by ~28%. Abundances are Lodders 2020
  Table 8 present-day (log eps C 8.47, N 7.85, O 8.73, S 7.15, He 10.924), what upstream
  calls "Lodders 2019". The cfg's C_H/O_H is 0.549; `CO_DEFAULT` 0.55.
- `co_bounds(network, use_photo)`: photolysis on, `CO_MAX` = sncho 0.99, sncho2025 10,
  ncho 10; photolysis off, 10 for all three. `CO_MIN` 0.1 was measured on the three
  photolysis-on rows; the photolysis-off floor is carried over. The permissive bounds are
  the highest value SAMPLED, not a measured ceiling.
- The sncho photolysis-on ceiling is set by the DERIVATIVE. Default W39b, FD `dlnCO`,
  `CO_MAX` patched: the column certifies at 0.99-1.02 (loose, element budget within
  0.01%) and at 1.03 only after escalation. The one-sided row (h 0.05) against its 0.25
  h-vs-2h gate: 0.032 at 0.55, 0.210 at 0.95, 0.269 at 0.97 (first FAIL), 0.414 at 1.03.
  The AD margin (gate 0.1) is 0.0476 at 0.95: AD is refused from ~0.90. The band-median
  depth turns over near C/O 1.00. Decision: 0.99 stays; a raise unlocks spectra with no
  certifiable C/O sensitivity and moves the default network into the carbon-rich regime.
- CLI survey (92 rows; raw outputs deleted 2026-10-20; nz 100, cadence 5, cap 5000):
  every ceiling is a LOWER bound. sncho converges <= 1.02 and fails 1.03-4 (also at 30k
  steps). sncho2025 fails at 1.03 only, a notch that moves with temperature and nz. ncho
  converges 1.03-10. Photolysis off, every network reaches 10 in 121-229 steps. 54 of 58
  converged rows exit loose (median longdy 0.0988). No published VULCAN WASP-39 b run
  reaches C/O >= 1 photolysis-on; upstream exoclime@80f75b9 fails C/O 1.087 at its cap.
- sncho2025 converges where sncho does not because of its H2CS/CS2 sulfur block (alone it
  converges at C/O 1.087 in 2448 steps); the C3 species hold 2.5e-7 of the carbon:
  network-sensitive numerics. At C/O 1.03 its failure is a localized sulfur-cell
  instability (layers 37-39), neither a limit cycle nor a slow approach.
- Temperature and Kzz are NOT gate axes: +400 K, +800 K and -200 K all rescue C/O 1.087,
  and Kzz 1e10 and 5e10 do where 1e9 does not. vulcan3 forces `atm_type="file"` whenever
  Kzz is tabulated, so the survey's synthetic T-P tables kept the shipped Kzz.
- `MET_RANGE` (0.1, 100), a maintainer decision. Cold `certified_solve`, sncho,
  photolysis on, C/O 0.55, cap 30000 (vulcan3 0.11.0): W39b default certifies every point
  40x-300x (100x 7808 steps; steps climb because the loose slope gate scales with dt).
  W39b Guillot at Tirr 1640: 30x FAILS (a top-of-column photolysis stall), 50x
  certifies, 100x fails. HD 209458 b: 50x and 150x fail at both cadences (photolysis off
  certifies 50x in 452 steps). Cool planets show no ceiling to 1000x. The lnZ stencil
  solves at met x exp(+-0.2), so the range check caps a lnZ FD centre at 81.87x; 100x
  keeps every stencil point under the certified 150x. The run-time certificate, not the
  range, refuses an uncertified column. Cost accepted: ~3.5x the default's steps at 100x.
- Between 81.87x and 100x the lnZ row is AD-only. On an external user's case (LP 714-47
  b custom: Guillot 1020 K, const Kzz 1e9, 100x) the AD lnZ tangent certifies at 20900
  steps at both cadences; the 6000-step cap reads 0.105. Central FD with the range lifted
  certifies its four columns and fails h-vs-2h at 0.463. Basis of `AD_RETRY_COUNT_MAX`.
- The solved column runs C/O short of the request (10 -> 9.877): upstream's 4-decimal
  rounding and one-way species map. Every surface reports the REQUESTED value. Measured
  on the FastChem seed (`_VERSION` <= 58), not re-measured on the ExoGibbs seed.
- Carbon-rich ladder, photolysis off (default W39b transmission, tool 0.67.1). The drift
  is what `ELEMENT_TOL` 1.5% is set against:

  | network | C/O | steps | element drift | C6H6 photosphere | amplitude |
  |---|---|---|---|---|---|
  | sncho2025 | 2 | 349 | 0.06% | 1.84e-08 | 2947.5 ppm |
  | sncho2025 | 4 | 292 | 0.32% | 1.68e-05 | 2499.0 ppm |
  | sncho2025 | 7 | 490 | 0.81% | 7.75e-04 | 2075.4 ppm |
  | sncho2025 | 10 | 350 | 1.30% | 3.31e-03 | 1808.1 ppm |
  | sncho | 10 | 219 | 1.21% | 3.31e-03 | 1807.2 ppm |
  | ncho | 10 | 243 | 1.22% | 3.38e-03 | 1812.4 ppm |

  Benzene, the fifth carrier at C/O 10, has no k-table anywhere (ExoMol publishes no
  C6H6 line list) and is not photochemical (3.3e-3 off, 3.6e-3 on).
- Photolysis on, mean photosphere C6H6 on sncho2025: 3.0e-22 at C/O 1.0, 6.8e-5 at 1.2,
  1.40e-4 at 1.4, 4.58e-4 at 4, 1.15e-3 at 7, 3.62e-3 at 10. The 1e-4 warning floor is
  crossed between 1.2 and 1.4, the 1e-3 refusal between 4 and 7.
- Leave-one-out at C/O 10: CH4 1449 ppm, C2H4 1087, C2H2 504, CO 64. C2H2 at 1.6e-4 is
  worth 504 ppm, so a table-less absorber in the 1e-4..1e-3 band can be worth hundreds of
  ppm: a warning, not a licence.
- Decision (0.68.0): the advertised C/O top stays 10; `ELEMENT_TOL` 1.5% so the C/O 10
  column passes; an absorber with no table above `UNMODELED_VMR_REFUSE` 1e-3 is refused
  after the baseline solve, and above `UNMODELED_VMR_WARN` 1e-4 it warns. A 1e-4 refusal
  cut the path off near C/O 1.3, judged too restrictive; 1e-3 keeps sncho2025 to about
  C/O 5. The shipped default warns about nothing (HSO at 3.0e-5). The mode stays
  EXPERIMENTAL: no benzene line list exists.
- C2H4: the table spans 100-3400 K and 0.3-50 um, but ExoMol's `.def.json` gives 700 K
  and 1.408 um and its kcoeff is zero shortward of 1.427 um: interpolation, not
  completeness.
- Network-swap parity against pinned upstream vulcan2_ncho 80f75b9: matched-step ymix
  mismatch 1e-15 at 5 steps to 9e-9 at 50; free photolysis-off runs agree CH4 1.6%,
  worst 19.8% (C4H2), inside the ~10% per-step slack of a loose exit.
- Rate-law T-range exposure depends on the T profile, not C/O: on the W39b grid SNCHO has
  194 of 513 thermal rows outside their range in >= 1 layer, SNCHO_2025 200 of 532, NCHO
  130 of 390. `run_model` stores the counts and the app states them.
- `_assemble_chem` sets `VULCAN3_NETWORK` and the cfg `network`/`atom_list` for EVERY
  network: skipping the default let a sncho request after a sncho2025 run in one process
  be served by sncho2025.
- Networks: sncho (default) is the Tsai et al. 2023 89-species file with
  post-publication rate updates (never "exactly the published network"); ncho 69 species
  (sulfur refused, never dropped); sncho2025 93 (experimental, hash-pinned in VULCAN-JAX
  `tests/science_sources.yaml`). The network says COS where the opacity side says OCS.
- `CO_BZ_MIN_AD` 0.1 (TOI-7169 b-like case): AD dlnCO vs one-sided FD at margin 0.12
  corr 0.9904, scale 1.022; at margin 0.05 scale 1.116, a uniform 11-12% AD excess. Two
  points, one planet.
- Condensation: `CONDEN_CFG` (S8, whole-column pin, `mtol_conv` 1e-15, sulfur-allotrope
  `conver_ignore`, `trun_min = stop_conden_time`) was tuned on the 89-species network,
  each knob measured necessary. The pinned reservoir's jvp-vs-FD error is 91%, hence
  detection-only. Default W39b with `use_condense` (vulcan3 0.10.1): 2344 steps, longdy
  0.0998 (0.2% inside the gate), +72% steps; nothing condenses (S8 at 1.4e-47 of
  saturation).

## Convergence certificate and element gate

- Certification is `ConvDiag.conv_normal` AND `longdy < yconv_min` (upstream op.py's
  predicate). The default W39b solve exits loose at longdy 0.0994, controlled by N2O at
  3.5 mbar with VMR 6e-18: one ultratrace cell sets the exit. `yconv_min` is canonical
  (range [1e-4, 0.1], may only tighten) because a loose exit ignores `yconv_cri`.
  Tightening costs: at 0.02 a TOI-like stencil point ran to the cap.
- Presets (steps / longdy, cadence 5, vulcan3 0.10.1): WASP-39 b transmission 1363 /
  0.0978; HD 189733 b 1506 / 0.0118; WASP-107 b 5332 / 0.0539; HD 209458 b 21368 /
  0.0985; WASP-39 b emission 1602 / 0.0759; HD 189733 b emission 5913 / 0.0899. Cadence 1
  moves the spectra at most 3.8 ppm.
- `dt == dt_max` is not a convergence diagnostic: every member of a photolysis-off cap
  ladder exits at its cap; photolysis-on count_max exits sit at dt 92-95 s.
- The photolysis-off `dt_max` leak behind `DT_MAX_S` 1e13 and `check_elements` (default
  W39b photolysis off, requested O/H 5.37e-3):

  | cap (s) | steps | max abs d vs 1e13 (ppm) | O/H | S/H |
  |---|---|---|---|---|
  | 1e11 (through v45) | 5040 | 847 | 2.915e-3 | 3.12e-6 |
  | 1e12 | 124 | 12 | 5.252e-3 | 1.367e-4 |
  | 1e13 | 121 | 0 | 5.301e-3 | 1.364e-4 |
  | 1e17 (upstream) | 121 | 36 | 5.844e-3 | 1.380e-4 |

  The 1e11 column certified having lost 46% of its oxygen and 98% of its sulfur. The
  engine fix is VULCAN-JAX's Ros2 stage repair; with it the column drifts 0.00%.
- `check_elements`: layer-median X/H against the build column, `ELEMENT_TOL` 1.5% (the C/O
  10 top drifts 1.30%); cached presets sit at 0.006-0.008%. It runs on every certified
  column: baseline, chem-cache hit, each FD stencil point, each AD warm re-converge.

## Structure: T-P, Kzz, gravity anchor, model top

- Default structure is the verified measured table where one ships (exactly WASP-39 b,
  `atm_W39b_evening_TP_Kzz.txt`), Guillot + constant Kzz elsewhere; emission defaults to
  Guillot (the tables are terminator profiles). On WASP-39 b, Guillot + const Kzz runs
  ~100 K hot through the SO2 zone with Kzz 3.8-48x low above 1 mbar and reads SO2 at 2.1
  sigma against the published 4.5-4.8. HD 189733 b's table does not certify at defaults;
  HD 209458 b's is refused (2997 K, above the 2980 K opacity ceiling).
- Guillot `kappa_IR` -2.0; `T_irr` defaults per planet from the registry's star and
  orbit; emission uses the DAYSIDE T_irr (x 2^0.25): planet-average T_irr put the HD
  189733 b eclipse 28-49% low (chi2/N 934 -> 31 against Inglis et al. 2024). `Tint`:
  WASP-107 b 460 K (Sing et al. 2024), others `TINT_DEFAULT_K` 100.
- One gravity per run: the catalogue (Rp, g) pair is converted to the chemistry's anchor
  pressure before it reaches VULCAN (`_assemble_chem`; vulcan3 anchors g near 1 bar). On
  W39b (anchor 0.923 bar) R -6.7%, g +14.9%. mu comes from one probe build (2.30 is 0.62%
  off); the Guillot/gravity iteration converges in two passes. Handing VULCAN the
  catalogue pair put the radius ~7% high and every gravity ~13% low. `p_ref_bar` is 1
  mbar, deliberately not tuned to the data (7.8% low against the white-light depth).
- Transmission dtau uses inverse-square g(r) (exojax's own profile: -51 ppm at W39b
  against +1.5 ppm); emission takes the same g(r) from v59.
- Model top `rt_ptop_bar` 1e-7 bar (chemistry and RT; range [1e-9, 1e-6]; upstream 1e-8):
  1e-7 vs 1e-9 moves the W39b depth 105 ppm at R = 100 (+377 ppm at 14 um), accepted for
  the published convention. HD 189733 b file mode needs `rt_ptop_bar >= 1e-7` (a 6000 K
  thermosphere). Above 1e-5 bar the molecular-diffusion advection puts a 2dz oscillation
  on atomic H that the positivity clip zeroes; numerical, and the default top avoids it.
- `use_vm_mol` is pinned False (the engine flipped its default; <= 0.01 dex).

## Opacity and radiative transfer

- Correlated-k over the published ExoMolOP k-tables is the only path: R = 1000, 5115
  bands, 16 g-points, T 100-3400 K, P 1e-5-100 bar, 25 principal-isotopologue tables on
  one grid. RT knobs are echoed by the engine and refused on mismatch (they bite at
  6.8e-3 max relative on the smoke CO band).
- Preselection (max |depth - depth_wo| over 1-15 um, all 20 extras on). WASP-39 b, ppm:
  H2O 2332, CO2 1878, SO2 1030, HCN 742, CO 487, H2S 306, NH3 137, CH4 134, OCS 58.5, SO
  47.7, SH 8.2, CN 6.8. HD 189733 b: H2O 613, CO2 457, SO2 201, CO 144, CH4 121, H2S 85,
  NH3 81, SO 23.5, OCS 18.4. SO and SH default on; CN off. Each default-on molecule costs
  one leave-one-out fold (12.9 s). Not measured for carbon-rich columns (C/O 2: C2H2
  1434, C2H4 949, CH3 88, CS 25 ppm) or WASP-107 b (NO 14 ppm). Decision:
  `EXTRA_MOLECULES_DEFAULT` stays; a user adds CH3/CS per run.
- Against petitRADTRANS 3.4.0 on the same k-tables: engine fixtures rms 0.0013% /
  0.096% / 0.021% (isothermal H2O radius, W39b depth, H2O emission flux). The six
  `validation/data/atmos_*.npz` (tool 0.72.0, cadence 5), fitted offset at R = 100:
  transmission residual std WASP-39 b 17.46 ppm, HD 189733 b 4.31, WASP-107 b 48.39, HD
  209458 b 4.72; emission relative RMS 0.307% / 0.256%. An isothermal column emits
  pi*B(T) to 6.7e-16. The figure script refuses mixed cadences and a stale pRT reference.
  After a `_VERSION` bump that moves a spectrum, regenerate the npz (`atmos_cases.py` x 6,
  ~15 min), `prt_reference.py` and the figures (`make_figures.py`); nothing tests that.
- pRT on daw: `VULCAN_Project/.prt-env` (plus the `mpich` wheel);
  `VULCAN_Project/.prt-data/input_data` holds ONLY symlinks to
  `$VULCAN_FORWARD_DATA/exomolop/<m>.ktable.h5` under pRT's names. pRT downloads a
  different file when a name misses, so `find VULCAN_Project/.prt-data -type f` must
  stay empty.
- Against data: the cloud-free W39b model exceeds the ERS amplitude 1.18-1.23x where the
  published 3D models exceed it 1.04-1.11x; against those models amplitude ratio 1.080,
  shape correlation 0.9949. HD 189733 b transmission 0.70x (no haze model). gCMCRT files
  read `depth = (H(1)^2 + 2 col2)/Rstar^2`; the other reading makes a 2.2x discrepancy.
- Chemistry anchors: WASP-39 b vs Tsai et al. 2023's released output median 0.042 dex;
  HD 189733 b vs Tsai et al. 2021 median 0.020 dex; upstream VULCAN vs VULCAN 3 on W39b
  1202 / 1202 steps, max |d ymix| 3e-10. Spectral discrepancies are RT, opacity or
  aerosol.
- Clouds: one power-law deck, as haze (amplitude + slope) or gray (slope 0, entered as
  the pressure it blocks to; Heng & Kitzmann 2017). `cloud_top_bar` in a share file IS
  gray mode. Gray is transmission-only: it maps through the SLANT optical depth, while
  emission integrates vertically (~1.4 dex deeper on WASP-39 b).
- H2/He Rayleigh is on. CS2 and C2H6 have no ExoMolOP file; HSO (1.2e-4) and S2
  (9.7e-5) exceed SO2 in the W39b photosphere and have no usable line list.
- `grid_sha256` is not a content hash: `exomolop.table_info()` digests the grids and
  never `kcoeff`. Never key a cache or a provenance claim on it.

## Emission

- Solver `ibased_linsap`, never `ibased` (it drops the bottom boundary, 12-52% of the
  flux). `p_btm_bar` is canonical: 7.6 bar in file mode, 10 bar for parametric profiles.
- The thin-bottom gate is FLUX-WEIGHTED (`thin_flux_fraction`, `EMIS_THIN_FLUX_FRAC` 1%):
  a min() over the band let a 60 nm notch veto HD 189733 b, and an unweighted sum
  under-reported the short-edge share 2.9x. It gates every FD stencil point and AD primal,
  and runs per mode (`detect._certify_emission_band`). On the e2e emission reference the
  minimum bottom tau is 73.6 against tau = 3, and stencil points move tau <= 26%.
  Removed-H2O on W39b sits at 1.73% band-global and is refused. Re-measure before
  shipping a shallower `p_btm_bar`.
- Eclipse depths use the tau = 2/3 photospheric radius per k-ordinate
  (`eclipse_flux_tau`): W39b depth ratio photospheric/single median 1.070; the CO2
  feature contrast had been 9% low. `emission_radius` stays the (R_em/R_star)^2 factor.
- HD 189733 b MIRI LRS vs Inglis et al. 2024 (v60): chi2/N 67.6, 26.85 with a fitted
  -132 ppm offset, under the dayside f = 1/2 Guillot default.
- lnR0 is a radius change at fixed planet mass in both modes (maintainer decision with
  vulcan-forward 0.29.0). Before, emission multiplied the depth by e^(2 lnR0) at fixed g.
  Default W39b eclipse: AD and FD lnR0 rows agree to corr 0.99999997; the new row over
  the old is 0.992 at the median, 0.78-1.04 per bin. Emission lnR0 FD points are
  tau-gated.

## Measurement operator, LSF, noise, ramps, registry

- ONE operator (`binning.py`) for noise, model and every Jacobian row; exact sub-cell
  integration (`_pl_antideriv`; the cumulative trapezoid was 1-2 ppm off);
  degenerate-wavelength pixels masked (a G395H pileup made one bin's sigma 2.7x too
  small); per-segment offsets including segment 0.
- Segment label of a bin: the count-majority segment. No WASP-39 b G395H bin straddles
  the gap at R >= 50; at R = 25 one can (minority weight up to 43%). Decision: majority
  kept, no fractional row.
- LSF: a Gaussian of width R_refdata / LSF_WIDTH, weighted per d ln lambda (per d lambda
  was 4.3 ppm off). Widths fitted to a Pandeia impulse (`run_parity.py --impulse`, ~70
  min): NIRSpec 0.94-1.02 (left at 1), SOSS 1.39-1.58, NIRCam 1.37-1.46, MIRI LRS
  1.49-1.59. A retracted first fit had two traps: an impulse SED not dense far beyond
  +-8 sigma (Pandeia interpolates back onto the input grid: 25-150% spurious flux) and a
  half-maximum estimator on 2-3-pixel profiles. The blur is a no-op on the gratings and
  NIRCam grisms. MIRI LRS pixels arrive DESCENDING and `np.interp` does not raise on
  unsorted xp (binned depth off up to 328 ppm); `smooth_to_native_r` validates.
- Noise: the star is normalized by band-integrated 2MASS Ks (the monochromatic shortcut
  was up to 3.1% off). Pandeia's extracted noise includes the quantum-yield excess.
  Fisher inversion whitens to correlation form (a pure rescaling flipped rank 1 -> 2).
- Floors: exact PandExo semantics, `sigma = max(sigma_random, floor)` on final bins. The
  transit counters evaluate the floor-only limit only when every floor is positive; a
  `max(floor, 1e-30)` clip made a case met in one transit read unreachable.
  `detection_significance` snaps a profiled chi^2 below 1e-12 of the raw to 0 (unsnapped,
  an in-span null scored 0 only 52% of the time).
- Box transit, `t_in = t14`: ~5% optimistic against the information-equivalent
  trapezoid, partly offset by the symmetric in/out variance; both match PandExo.
- Ramps: floors are pandeia `mingroups` (NIR 1, MIRI 2); saturation is the MEASURED probe
  against `sat_limit` x the per-mode `saturation_fullwell`; the bracket search returns
  the largest measured-safe count and proves maximality; NIRCam capped at 100 groups,
  SOSS at 30. A predicted search under-selected (1 group at a 7.08x sigma ratio). MIRI
  `t_cycle_s` includes the FASTR1 reset. Saturation advice is REPORTED, never applied.
- Registry: band edges cite a source (`_BAND_SOURCES`; NIRSpec from Birkmann et al.
  2022); native R from the 2026.7 dispersion files (PRISM ~110, G395H 2700, SOSS 970,
  LRS 150 by decision); gaps G235H 2.20-2.27, G395H 3.72-3.82 um. The NIRCam data excess
  is PandExo's estimate ported (to 2e-4 GB), advisory only.

## Jacobians and Fisher

The closure gate is corr > 0.99 and scale within 15%; the FD self-check is the h-vs-2h
gate 0.25.

- A non-finite Fisher matrix or nuisance row is refused: a NaN radius-nuisance row once
  gave a finite sigma (the positive-diagonal mask dropped it).
- FD rows: central (Richardson) or one-sided (-h, -2h, -4h at h/2, third order) when the
  +2h dlnCO point would leave `CO_MAX[network]`. At C/O 0.89 on W39b the one-sided gate
  is 0.060; central h = 0.05 FAILS at 0.258.
- Composition FD holds the geometry anchor FIXED by design: lnZ/dlnCO stencil points
  reuse the baseline's `mu_col`, chemistry radius/gravity anchor (`ovr["Rp"]`) and
  Guillot `tp_eval`, while T-P stencil points and a fresh `run_model` re-assemble them.
  The h-vs-2h gate cannot see this; measurement settles it: the Fisher marginalizes
  `lnR0` and segment offsets, absorbing an anchor shift to leading order, and
  re-anchored independent references agree to corr 0.9991 / 0.9996. W39b Guillot Tirr
  1560 at 10x (own vs frozen anchor): corr 0.999997, scale 1.000693, median 0.10% over
  the top-decile bins; ~0.5% at the top of the range. Measured before v60, science mode
  not recorded (the TOI-7169 b reference is transmission); not measured on an eclipse
  case since lnR0 moves gravity ("Known and deliberately untracked").
- T-P rows are FD only (an AD Jacobian cache with a T-P row held a frozen-anchor
  derivative) and rebuild chemistry and T-P at each stencil point. The transmission Tirr
  FD/FD closure passes escalated: corr 0.996898 / scale 1.00571 against an independent 2
  K central difference, the noisiest usable step (per-solve error grows as 1/h^2).
- Eclipse AD rows need the step cap. Uncapped, the warm map never certifies within 6000
  steps (the tangent reaches 1e9-1e31); every certified column there is a snapshot of a
  photolysis sawtooth. A plain `jax.jvp` read at the column's certificate closed at corr
  0.3319. The timestep stop-gradient closes (0.997647) but breaks the independent Kzz
  derivative. Capping `dt_max` on the AD build alone closes the rows (lnZ corr 0.997635,
  dlnCO 0.998119), the same over a 300x range of caps.
- Shipped (v57): `AD_BUILD_OVERRIDES = {"count_max": 6000, "dt_max": 1e5}` on DERIVATIVE
  builds only; cold solves and FD stencil points keep `DT_MAX_S` 1e13, so no shipped
  spectrum moves (a global cap moves it ~6 ppm). W39b transmission AD rows close at corr
  0.99989 / scale 0.9977.
- `AD_RETRY_COUNT_MAX` 30000: the first attempt keeps the 6000 cap (a genuine stall is
  detected there at that price; on TOI-7169 b the warm re-converge stops improving by
  step ~900); only the cadence-1 retry gets the cold budget. On the LP 714-47 b case the
  retry certifies lnZ at 20936 steps, matching a both-attempts-30000 run (corr 0.999981).
- AD rows: one warm `jax.jvp` per row, never a vmap over tangent directions (NaN through
  VULCAN-JAX <= 0.3.4 on columns with exact zeros; 3% per bin off after the fix).
- Photolysis-cadence escalation (`certified_solve`): a column that does not certify is
  re-solved once at cadence 1. The unit is the FD ROW, never the stencil point: the
  cadence moves a spectrum 0.01-2.4 ppm, and in a mixed row that offset divides by h
  (~0.25% of the dlnCO row, invisible to the gate). Only `NotCertified` triggers it; a
  backend fault, the element gate, the emission gate and the T window propagate. A
  chem-cache hit of an escalated column rebuilds at cadence 1.
- Escalated FD rows end to end (TOI-7169 b CO case,
  `tests/live/data/toi7169_co_case.json`: C/O 0.55, 10x, FD dlnCO + lnZ; 72 min cold,
  ~52 of them in the two failing cadence-5 solves): h-vs-2h dlnCO 0.084, lnZ 0.034;
  against independent re-anchored references (Richardson, cadence 1) dlnCO corr 0.999063
  / scale 1.0065, lnZ 0.999574 / 1.0002. Forecast (Ks 10.81, one transit, R = 100):
  log10(C/O) -0.260 +/- 0.039.
- Three routes to the same TOI-7169 b rows: sigma(dlnCO) FD escalated 0.089252, AD
  escalated 0.083839, AD capped at config cadence 0.085336. The method spread on the C/O
  width is about 6%: quote 0.037 +- 0.002 across methods or FD's 0.039 as the
  conservative number; never present an AD figure as agreeing with FD better than that.
- Tighter references were attempted (yconv 1e-3, rtol 0.02, 30000 steps) and none
  certified, so none bounds the cadence-1 reference error.
- Photolysis off, every FD chemistry row on the shipped W39b column misses its gate by
  23-400x (dlnCO 6.9, lnZ 5.7, lnKzz 99.9) because neighbouring compositions stop at
  non-comparable states (145 vs 17074 steps). An FD row self-checks and an AD row cannot,
  so AD chemistry rows are refused photolysis-off. RT-only rows (deck, lnR0) stay; their
  values still ride the photolysis-off column.
- No width cut on C/O: every finite width is reported in log10(C/O); inf comes only from
  the `fisher.NULL_LOAD_TOL` null-direction test; "unconstrained" prints only above
  `app._UNCONSTRAINED_ABOVE` (1e4). `ln_gaussian_curve` clips its grid to the solvable
  C/O range (exp(+-5 sigma) overflows at sigma_ln 142). Mock recovery is the linearized
  identity delta = F^+ J^T Sigma^-1 n, labelled "not a retrieval result".
- README figure (`fig_readme_example.py`, v46, GUI defaults): PRISM S/N 2.6, G395H 4.2,
  MIRI LRS 3.1 sigma; log10(C/O) all modes -0.278 +/- 0.028. The published G395H
  detections are 4.5-4.8 sigma.
- The slow closure tests pass cold: 5 passed in 57 min on 4 cores. No test exercises the
  cadence-1 AD retry: every measured column certifies at the config cadence under the
  cap; the retry is measured end to end on TOI-7169 b only.

## Performance and run limits

- Default W39b transmission cold run (nz 100): RT build 7 s, chemistry ~40-80 s, one
  spectrum ~3 s per `ckd.overlap` fold; the GUI passes `wo_mols = [target]` on detect and
  `[]` on constrain. Runtime prior: `_wo_min = 0.05 n_mols (nz/100)` min on detect; AD
  rows 1.5 solve-equivalents per chemistry row (measured 0.74-1.85). The bar hands over
  to the measured pace a third of the way in.
- The Space is ~4.5x slower than a 10-core laptop on RT and ~1.2x on chemistry; never
  read Space wall times as engine cost.
- The ETC computes only selected modes, one worker for all misses. Per-mode CPU on
  WASP-39 b: prism 11.4 s, miri_lrs 11.1, nircam_f322w2 21.3, g395h 55.0, niriss_soss 597.
  A spawn pool would gain 1.3x on the default trio (nothing with SOSS) for more CPU, ~0.8
  GB more RSS and children that survive a cancelled run: not adopted.
- `runlimit.MAX_CONCURRENT` 3: one cold default worker averages 1.75 cores and peaks at
  6.34 GiB; on cpu-upgrade (29.8 GiB, idle 4.58) three slots want 23.6 GiB and four
  reach the ceiling, where an OOM takes every session. `noise.PANDEIA_WALL_CAP_S` 3600 s
  frees a slot from a silent hang.
- Unchecking "Also calculate parameter constraints" saves ~6.5 min and downgrades the
  headline score to the calibration-only fallback.

## Cache keys and version history

- `forward._VERSION` busts `model_cache`; every bump is in git (`git log
  -S"_VERSION = " -- src/jwst_tool/forward.py`). Default transmission / emission keys:
  v58 `fec633bfcd1095f7` / `de561305d715a10b`; v59 (ExoGibbs seed, emission g(r), exojax
  2.6.0) `21996e68a45060c0` / `a9ea64d452f0dee1`, transmission max |d| 0.128 ppm,
  emission -1.92% to +0.20%; v60 (lnR0 at fixed mass) `8c37a8320c19c75a` /
  `f4b94ee63633f386`, transmission 2.2e-6 ppm, emission 0.0049 ppm, PICASO fixture bitwise
  unchanged.
- No-bump exemptions: `data_revision` (omitted while the variable is unset, so local
  payloads are byte-identical, test-pinned); engine moves that reproduced the fixture to
  <= 0.0065 ppm (vulcan3 0.10.1 at v57, 0.11.0 at v58). An engine exemption does NOT
  generalize: the engine can move a column (HD 209458 b 0.90 ppm on 0.10.0), and a column
  cached on the Space's persistent `model_cache` is not covered. The key carries no
  engine version: accepted by the maintainer (2026-10-20), with every engine pin bump
  bumping `_VERSION` (CLAUDE.md). The Space reads `data_revision` from the dataset's
  remote HEAD at boot, so a run made while the mount still holds the old bytes is stored
  under the new revision; accepted, the mount's update timing not measured.
- `noise.WORKER_VERSION` (+ backend fingerprint) busts `noise_cache`; held at 12 through
  wording passes and the order-2 removal (no measured field moved).
- `_PARAM_KEYS_ECHOED` holds the keys `canonical_params` emits but never reads (`version`,
  `data_revision`, `picaso_stack`); without `data_revision` in it the function refused its
  own output wherever the variable was set (0.66.1, on the Space).
- Cached results carry a `producer` stamp ("repo=sha12", `+dirty` when dirty, empty on a
  non-checkout install); not a key. A chem-cache column has none; exports then show the
  installed commits marked "installed now".

## PICASO engine

A second, selectable engine (maintainer decision); VULCAN stays the default and its keys
and fixture never move for it. PICASO's climate solver (or Guillot / a table) gives T(P),
its on-the-fly equilibrium chemistry the abundances (optional quench at one constant Kzz,
default off); spectrum, binning, noise and Fisher are the tool's. FD only; lnKzz refused;
metallicity capped at 10x; picaso 4.1 pinned; climate redistribution 0.5 / 1.0.

- Process split: picaso 4.1 needs numpy >= 2, the stack numpy < 2, so `picaso_worker.py`
  runs in its own environment. `picaso_engine.py` caches certified results under
  `output/picaso_cache` (key = sha1 of request + worker version + stack + data revision; T
  and g rounded so CPU/GPU last bits cannot fragment it), locks per key in a
  container-local temp directory, and hands `run_model` a VULCAN-shaped column. A cached
  result that no longer certifies raises at load (the solve is bit-reproducible).
- Composition is IMPOSED: a subclass of picaso's EquilibriumChemistry solves from the
  requested atom fractions and is injected through `inputs['climate']['chemeq_solver']`;
  tripwires assert C/O_rel 1 and mh = Z. picaso's own path scales a Lodders 2020
  PROTOSOLAR table (+0.089 dex in C, N, O, S) and moves C and O at fixed C+O. With
  C/N/O/S/He only, picaso and the ExoGibbs seed agree to < 0.01 dex over 1e-5-1 bar at
  1x-100x (slow test).
- Metals kept (maintainer decision): MgO/SiO2 condense locally and take oxygen: H2O
  -0.09 to -0.22 dex at C/O 0.3-0.55, about -2 dex near 0.9; -138 ppm mean in
  transmission (shape ~25 ppm). Against FastChem 4.0.3 the sink is the same 18.2% of the
  oxygen (analytic ceiling 0.182); pinned by
  `test_picaso_metal_oxygen_sink_matches_fastchem` (0.05 dex; worst 0.029).
- Cold layers: picaso's warm start reports stale all-phase atom totals (budgets 10% off
  at 100x); cold solves close to <= 2e-4, hence `picaso_engine.ELEMENT_TOL` 1e-3.
- Identity with unmodified picaso 4.1 at its own composition: chemistry to 1e-11 dex, a
  climate solve to 5.4e-8 K; pinned by
  `test_picaso_worker_reproduces_stock_picaso_at_its_own_composition`.
- Climate time (WASP-39 b, one daw core): 497-632 s per solve, bit-reproducible; Newton
  iterations ~450 s (finite-difference Jacobian over 661 resort-rebin windows); numba
  compiles `picaso.climate.t_start` 88-90 s in EVERY new process (it never loads from
  cache). Four threads give the same time. TOA |F_net/F_int| 1e-6 to 2e-5 (gate 1e-3). 61
  levels halve the time and move the photosphere +-1.5 K (91 kept).
- The radiative-convective seed: rcb 75 vs 85 leaves 1e-5-1 bar unchanged but moves 1-10
  bar by up to 290 K; 85 is picaso's notebook default.
- Gravity: one constant g per run at the 1-bar anchor; a climate baseline iterates it
  (`ANCHOR_TOL` 0.5%: two solves per baseline); composition stencil points keep the
  baseline g; a Guillot T-P stencil point re-derives it. +25% g moves the photosphere -4
  to -21 K.
- Climate + quench: the climate is solved in equilibrium and the quench applied after.
  picaso's quench-in-the-loop climate reports converged at TOA 1.25e-2; through the same
  post-hoc quench its T moves the transit depth 11.3 ppm rms. The certificate covers the
  equilibrium climate, not the quenched column. The quench level steps once per ~0.25
  dex of Kzz (the h = 0.1 stencil reads levels [61, 61, 62, 62]).
- Certificates (`picaso_engine.certify`, re-run on every cache load): every layer
  converged, C/N/O/S budget <= 1e-3, climate converged + TOA flux <= 1e-3 + dlnT/dlnP in
  (-0.25, 0.5) + no convective zone in the top 5 levels + T <= 4000 K (the CK grid) + no
  failed chemistry layer. The worker refuses a correlated-k set other than the pinned 36.
- FD rows through the climate: lnZ (one-sided at the cap) h-vs-2h 0.088, dlnCO 0.116;
  closures corr 0.9976 / 0.9997. T_int at 100 K reads 0.541 at h = 10 K and 1.190 at 7 K
  (the photosphere moves <= 0.013 K per K), so the climate T_int row is refused.
- Quench vs VULCAN photolysis-off kinetics (same Guillot T-P, Kzz 1e7-1e11): CO 0.00
  dex, CH4 +0.12 to +0.22, H2O and CO2 -0.22 to -0.33; spectra 9 ppm rms.
- picaso issue #413 (climate stellar binning): R = 100 spectra -7.4 ppm mean, shape <= 14
  ppm; under the 50 ppm override line, so 4.1 stays stock.
- One-way coupling: picaso's own dayside thermal spectrum carries 5.0% less 1-12 um flux
  than the tool's eclipse flux.
- Published PICASO WASP-39 b PRISM best fit (Rustamkulov et al. 2023, Zenodo
  10.5281/zenodo.7388032) vs this engine: 39.0 ppm rms at R = 100 over 1.02-5.30 um; the
  four published ERS grids differ among themselves by 68-122 ppm. Pinned by
  `tests/live/test_picaso_published.py` (60 ppm).
- Runs on daw: default climate transmission 1198 s, emission 1029 s; one climate worker
  peaks at 3.27 GiB. On the Space (cpu-basic) one climate solve takes 914-1020 s; a
  default climate run with rows is estimated at 2.5-3 h [E] and did not finish inside a
  110 min watch. Kept as is (maintainer decision): no runtime limit, three slots.
- GUI: the "Chemistry engine" selector opens Step 2; PICASO widgets carry a `picaso` key
  infix; under PICASO the method is a disabled FD-only widget on its own key.

## Statistics: what is forecast and what is refused

- Posterior curves are Fisher-Gaussian (Laplace) from the same nuisance-augmented
  matrices as the table, never sampled. Single-realization retrieval as a precision
  estimator was rejected (realization bias, Feng et al. 2018); the mock is one seeded
  draw per bin and never enters scores or caches.
- `sigma_detect` is a conditional matched-template S/N; the label names the rows
  actually projected. Fisher numbers are local Cramer-Rao bounds, disclosed everywhere.
  The C/O row within one stencil of C/O = 1 is an oxygen-rich-side local derivative.
- Binned scores on the high-R gratings are sound: every analysis bin (R <= 500) is wider
  than the R = 1000 bands; a caption fires at r_bin >= 250 when a mode outresolves the
  model.
- Saturated modes are excluded from the ranking and the combined row and named.
- Complementary rank-deficient modes can leave the joint Fisher full rank: the page
  reports the combined width; only a non-finite combined width stops it.
- Analysis R is neither the instrument's resolving power nor a display setting; the
  plotted model is display-smoothed to max(300, 3 x R), untouched by any score.

## GUI, figures and copy

- The audience is an exoplanet proposer: precise terms stay, explaining prose goes (a
  lay-paraphrase sweep was reverted in full by the maintainer). Verdict first, caveats
  beside it; the Data expander is one fully-cited sources table.
- Figures are pure builders under `plotting.render_lock`: matplotlib 3.10's
  `tight_layout`/`savefig` measure mathtext through a process-global parser (without the
  lock 7 of 8 threads crashed).
- Streamlit silently DISCARDS an out-of-range session-state value, so `share_config`
  range-checks every restored numeric (`_check_widget_ranges`, AST-pinned to app.py).
  The Pandeia star is recorded in the observation block because `canonical_params`
  zeroes it in transmission.
- Restore reads `canonical_params`' OUTPUT and refuses by name, never drops. Files that
  enable condensation or boundary terms, or gray + emission, are refused: pinning them
  off would show a successful restore of a different atmosphere.
- CSVs start with one `# provenance:` line and carry the page's caveats as `# caveat:`
  comments; a rendered-app regression checks the actual download bytes.
- Constraint-table rows use `fisher.param_axis` labels except "log10 C/O": a bare
  "Metallicity" row read as the x-solar input, a decade off. The C/O panel ordinate has
  no " per d ln(C/O)" qualifier (removed in 312af43).
- `plotting.draw_caveats` draws ONE footnote line in a strip added below the panels,
  re-seating every axes at `get_position(original=True)`; panel boxes do not change.
- `initial_sidebar_state="expanded"`: "auto" starts the sidebar collapsed in the narrow
  huggingface.co iframe, hiding every input.

## Data, provenance and the engine split

- The forward engine lives in vulcan-forward, not VULCAN-JAX (exojax's `numpy<2` cap and
  an external data-path contract do not belong in the validation core). Engine-surgery
  gate: a forced fresh default solve reproduces the cached spectrum with max|diff| = 0.
- A downloaded configuration carries a `provenance` block (commits and dirty state,
  versions, Pandeia identity, checksums, cache versions, seed), informational on load.
- `jwst-tool fetch` is stdlib-only and idempotent; STScI Box links answer scripted
  requests (resolve the 302, range the signed URL, verify size and `gzip -t`).
- Tracked data: four FITS under `data/cdbs/comp/nonhst/` (2MASS Ks required; Bessell
  H/J/K for parity). Gitignored: `data/cdbs/grid/phoenix` (a plain directory), CALSPEC
  Vega, the Pandeia 2026.7 refdata and PSF trees (the PSF tarball unpacks as
  `pandeia_psfs-2026.7rc1-jwst` and must be renamed), `data/picaso_refdata-4.1` (picaso
  v4.1 `reference/`, GPL-3.0, plus 36 correlated-k tables, Zenodo
  10.5281/zenodo.18644980, CC-BY-4.0, 2.2 GB). Engine data lives behind
  `$VULCAN_FORWARD_DATA`.
- `data_revision`: nothing else tracks the science data. The k-tables are symlinked off a
  dataset mount that updates as commits land [I], while `$STATE` keeps `model_cache`
  across restarts. The entrypoint stamps the remote dataset HEAD
  (`HfApi().dataset_info().sha`), which can differ from the mounted bytes (accepted,
  "Cache keys and version history"). The
  dataset repo is private, so the revision needs a read token as the Space's `HF_TOKEN`;
  without one the boot log says "dataset revision unavailable". Read the container log in
  the browser: the `/logs/container` API path is 404.
- `noise.backend_fingerprint` keeps the "unavailable" placeholder only for a MISSING
  backend; one that exists but cannot be probed raises.

## Hugging Face Space

- Layout: the dataset repo mounts read-only at `/srv/hub-data`, a bucket at `/data`; the
  entrypoint symlinks `exomolop` (never copies 4.3 GB), sets `HDF5_USE_FILE_LOCKING=FALSE`
  (the mount does not lock), and symlinks the PICASO CK tables. The base image is pinned
  by digest. BUILD_INFO confirms a deploy; the served HTML title does not.
- Capacity is one Space and its slot limit. Spaces do not autoscale, and a replica is
  awake whenever the primary is. An in-app self-scaler and a queue were rejected (a new
  instance never serves the visitor who triggered it; a queue holds a session open for
  tens of minutes); a second overflow Space was removed (Superseded designs).
- The run slot is taken at the first heavy launch, so a fully cached run renders on a
  booked instance. A refused visitor is told how long the oldest job has run, never a
  promised wait.
- Refusal alerts: `runlimit.notify_refused` mails `ALERT_TO` from `ALERT_SMTP_USER`
  (secret `ALERT_SMTP_PASS`) when every slot is busy or a parameter set is illegal; one
  mail per hour per instance, never raising; off unless all three are set.
- Default-run warm: after the foreground warm-up, `entrypoint.sh` drives one AppTest per
  engine that clicks Run, so both default runs land in the shared cache under a
  visitor's exact key. On daw: cold vulcan 942 s, picaso 6192 s. On the primary vulcan
  solved in 1211 s with memory peaking at 14.2 of 16 GB.
- Public access: XSRF/CORS are off for the table uploads, mitigated by
  `--server.maxUploadSize=4` (a 200 MB table cost ~4.5 GB RSS), validation BEFORE
  archiving, `tp_file_sha1` accepted only as a 16-hex key (otherwise an open() could be
  steered under `uploads/`), content-addressed storage and the run limiter. Nothing
  prunes `uploads/` or the caches. If a mitigation goes, make the Space private.
- Updating code: push the content commits; move `deploy/pins.env` and the three
  Dockerfile `ARG *_SHA` (`JWST_TOOL_SHA` = the green-CI content commit;
  `test_deploy_pins.py` checks the siblings and that `JWST_TOOL_SHA` is an ancestor of
  HEAD); push the bump; then upload the Space files:

      uvx --from huggingface_hub python -c "
      from huggingface_hub import HfApi
      HfApi().upload_folder(repo_id='imalsky/jwst-transit-authority',
          repo_type='space', folder_path='deploy/hf-space',
          ignore_patterns=['__pycache__*'], commit_message='Pin to <sha>')"

  Verify: runtime stage exactly `RUNNING` (not `RUNNING_BUILDING`, while the old
  instance still answers health checks), the three SHAs in the build log, and
  `/_stcore/health` = ok. Widening either molecule list is a deploy change: re-run
  `upload_data.sh`.
- Data upload: `upload_data.sh` stages what the installed tool can select; to check a
  store, list it (`HfApi.list_repo_files`), never trust a provenance file. The dataset
  needs a `.gitattributes` routing `*.fits`/`*.pdf` through LFS before any upload. The
  legacy root folder `picaso-reference/` is read by nothing.
- Operations: hardware, sleep and restart are set on the Space page (the locally stored
  Hub token can upload but not change them). "Initialization step 'hf-mount' failed":
  check `dataset_info` and `bucket_info`; both present means transient, restart. A
  paused Space stays down until restarted on its page. `finish_hf_setup.py` is the
  one-time setup (both repos, Space files, `HF_TOKEN` from `$HF_DATASET_READ_TOKEN`, the
  resumable data upload).
- Old-name redirect: Space `imalsky/jwst-tool` serves one index.html that forwards to the
  new Space (a renamed Space's `.hf.space` subdomain returns 404).

## Parity testing

`validation/parity/` runs the same stars and registry configurations through this
package's Pandeia worker and through PandExo master (`pandexo_worker.py`), both on the
2026.7 triple, PandExo overridden to the registry hardware. Configuration, grids, ngroup
+-1 and t_int within 1% match; the residual sigma ratio ours/PandExo is 1.06-1.12 NIR and
1.35-1.48 MIRI LRS, entirely the noise model. The gate requires exact group agreement on
short ramps, a 15% t_int gap, a matched-sigma band [0.8, 2.0], and fails on missing
fields. `test_parity_gate.py` re-validates the COMMITTED artifact. `run_parity.py main()`
drops the `lsf_impulse` block, so a matrix re-run must be followed by `--impulse`.
`REPORT.md` is a 32-line generated summary. A re-run needs PandExo at the pinned commit
(not installed here: reuse with `PARITY_REUSE_PANDEXO=1` after rewriting the stored
`*_pandexo_job.json` paths), the 2026.7 trees and a synphot CDBS.

## Accepted limitations

None changes a shipped number.

- Box transit; one fixed configuration per mode; the LSF no-op on the high-R gratings and
  NIRCam; emission absorption-only (exojax's `ArtEmisScat` needs per-layer albedos the
  deck cannot supply); HSO, S2, CS2 without line lists; the C/O shortfall of the solved
  column; the WASP-39 b amplitude excess and SO2 S/N below the published detections
  (`p_ref_bar` not re-tuned); HD 189733 b without haze and its eclipse ~130 ppm high;
  photolysis-off chemistry AD refused; the near-unity C/O row read one-sided; the G395M
  noise factor extrapolated from G395H.
- CI skips the slow tier; the parity matrix shares PandExo's stellar spectrum;
  third-party data is not redistributed; cache identity is canonical params +
  hand-bumped versions; the worker ramp is transit-independent; stellar contamination,
  reflected light, scattering emission, H-minus and atomic lists are not modeled.
- The count-majority segment label at R = 25; UV data inherited from upstream as-is
  (eps Eri 115-283 nm coverage; GJ 1214's 464 zero-flux FUV rows); the upstream
  two-stream pole below 60 deg zenith (default 83, no clamp).
- PICASO engine: one-way coupled (Na, K, TiO, VO, FeH heat the climate and are absent
  from the spectrum RT); local condensation with no rainout; the quench approximation
  (picaso's kinetic CO2 included); a quenched run's T is the equilibrium climate's; the
  deep column fixed by the seed; constant-g climate; the climate top at 1e-6 bar; issue
  #413 binning; no Na/K silicate, FeS, Al, Ca or Ni condensates; metal atoms exempt from
  the unmodeled-absorber gate; no climate T_int row; an estimated 2.5-3 h default climate
  run on the Space.

## Known and deliberately untracked

Real, small, and out of todo.md by decision; nothing here changes a shipped number.

- Saturated channels the worker drops can open a gap `segment_ids` reads as a detector
  boundary (one extra offset nuisance); that mode is saturated and already excluded.
- The archive snapshot carries two unread columns (`st_spectype`, `pl_bmassprov`); drop
  them at the next networked `archive-refresh`.
- `adjoint_diag.run_adjoint` on the default WASP-39 b state is refused by its own scope
  audit at every probe step; the module stays (maintainer decision).
- `pandexo_worker._pandexo_commit` has no unit test; check it by hand at the next parity
  run.
- A PICASO Guillot/file configuration for the custom planet restores the registry
  semi-major axis; only a later switch to the climate profile would see it.
- The benzene error below the refusal threshold is unbounded (several hundred ppm, sign
  unknown); the C2H4 table is incomplete where used.
- The sncho photolysis-off C/O 10 column changed from C6H6 6.9e-12 (v42) to 3.31e-3
  (v58); never bisected.
- A stalling FD stencil point pays its whole step budget before the cadence-1 retry (~52
  of 72 min on TOI-7169 b); `ConvDiag.count_since_new_min` does not separate the stalling
  column, so no early predicate exists.
- On sncho photolysis-on a C/O row requested between 0.97 and 0.99 raises after its four
  solves (FD) or at build (AD).
- The `check_unmodeled` refusal returns as a forward-subprocess failure: no
  `refusals.log` line and no alert mail, unlike `canonical_params` refusals.
- NIRISS SOSS costs 597 s of ETC CPU and sets the wall floor of any selection with it [I:
  suspected cause, the ngroup bracket search re-running a full calculation per
  candidate].
- No test solves with condensation enabled; the one measured run sits 0.2% inside its
  gate.
- Code hygiene, no behavior change: `instruments.JWST_TOOL_BACKEND = "current"` rides
  every noise-cache key; `app.py` validates a restored T-P table against the shipped span;
  `forward._read_tp_table` re-wraps its own ValueErrors as "not parseable"; `stellar.py`
  keeps a numpy 1/2 `trapz` shim; `planets.py`/`stellar.py` restate constants
  `vulcan_forward.constants` owns; `app._LN10` duplicates `fisher._LN10`;
  `upload_data.sh` stages k-tables from the `vulcan-retrieval` checkout.
- pyproject's `vulcan-forward` floor still allows 0.29.0, which runs one extra warm-up
  solve per chemistry build (about 2x chemistry time [E]); CI, the Dockerfile and
  `deploy/pins.env` pin `32adc85`. Raise the floor at the next vulcan-forward release.
- Composition FD rows keep the baseline's radius anchor and T-P ("Jacobians and
  Fisher"); the own-vs-frozen check was never run on an eclipse case.
- The README forecast figure (`assets/w39b_so2_forecast.png`) carries the model caveats
  as text under the panels; moving them out is an app change (`summary_figure`).
- The raw logs and scripts behind several records here (`validation/co_stall/`,
  `validation/co_limits/`, untracked) were deleted 2026-10-20; the numbers in these
  notes are the record.

## Ideas not adopted

- A live TAP lookup beside the archive snapshot; archive-fill depth (per-field
  references, uncertainties, gravity propagation).
- A proposal-export mode with a read-only seed; a sensitivity display for the
  combined-mode systematic model; an SVD of the whitened design; marginalized forecast
  curves in a non-flat display parameterization.

## Superseded designs

- Sampled line-by-line opacity and the Mie condensate deck (`broadening`, `nu_pts`,
  `rt_dit_res`): line cores need ~1e6 points and ~127 GB; doubling `nu_pts` still moved R
  = 100 depths 90-262 ppm. Removed in 0.48.0 (d8e224f); the keys are refused by name.
- Correlated-floor noise scenarios: an assumption, not a calibrated systematics model,
  unused in any decision. Removed in 0.28.0 (81ad019); old files still load.
- Globally isothermal T-P (v21: one temperature biases the deep quench region), the GCM
  baseline modes (v16), the fast/high fidelity tiers (v17).
- The 0.42 PICASO provider (Visscher-grid blending, preweighted correlated-k at 70
  composition nodes; a node kink refused dlnCO at C/O 0.55; NumPy-1 conflict). Removed in
  0.43.0 (2606263).
- NIRISS SOSS order 2: usable band 1.0-1.26 um from the same exposure as order 1, and its
  0.832 response factor was an AREA ratio from Pandeia's extraction dilution, not a kernel
  effect. Removed in 0.58.0 (d950b7c).
- The `_BACKENDS` registry and `JWST_TOOL_BACKEND` override (one backend, the Pandeia
  2026.7 triple; 0209692, 871776b); the "legacy" Pandeia 3.0 backend (0.22.0, e9c1854) and
  "archival_2026_2" (never gate-evaluated).
- The AWS VM deployment, never buildable after the engine split (3ff18c7).
- `fisher.UNINFORMATIVE_SIGMA` (sigma_lnCO >= 1 read "unconstrained"): set for a linear
  C/O display. Removed in 0.56.0 (8a850b2).
- The 2000-step AD floor (0.63.0, corr 0.99746 / scale 1.0657, a lucky stop). Removed in
  0.64.0 (a1f3c44).
- The per-mode operational-warning subsystem, "How the model works" and "Notes on N
  modes" (removed by the maintainer); `_leg_note` / `legend_title`, never drawn
  (eaaa50f).
- The overflow Space (`imalsky/jwst-transit-authority-overflow`: the same shim, pins,
  secrets and mounts on a 30 min sleep timer, linked from the primary on a slot refusal
  through `JWST_TOOL_OVERFLOW_URL`). Removed 2026-10-20: it sat paused (Hub read
  2026-10-14), so a refused visitor was sent to an instance that did not wake. A full
  instance now refuses plainly; one Space plus the slot limit is the capacity model.

## Do not retry

- Guillot + constant Kzz as the W39b default: ~100 K hot in the SO2 zone, Kzz low, SO2 2.1 sigma against the published 4.8.
- An analytic photon-only noise model to close the SO2 S/N gap: pandeia extracted noise is deliberately conservative by 2-24%; never "PandExo-identical".
- Re-pairing the WASP-39 b radii to the white-light ratio (G395H SO2 4.44): a science decision, not a fix.
- `jit(ckd.overlap)` (2-7%) or a sort-axis-last layout (~10%): below the 10% bar; the win was calling overlap 45% less.
- Plotly: kaleido dependency, two renderers, panel geometry does not transfer.
- Content-hashing multi-GB payloads for cache identity: identity is a hand-bumped version plus discipline.
- `dt == dt_max` as a convergence veto: every cap-ladder member exits at its cap.
- Central FD near C/O = 1 (h = 0.05 at 0.89): gate 0.258; the one-sided h/2 row certifies at 0.060.
- `jax.vmap` over AD tangent directions: NaN through VULCAN-JAX <= 0.3.4; 3% per bin off after the fix.
- A hard-coded C/O excluded band (1.03-1.04): the notch is one point wide and moves with temperature and nz.
- Temperature or Kzz as C/O gate axes: +400 K, +800 K and -200 K all rescue 1.087; Kzz 1e10 does where 1e9 does not.
- A runtime-estimator term for the carbon-rich photolysis-off corner: the two measured corners disagree 8x.
- Tightening `yconv_min` to certify photolysis-off FD rows: 17074 steps at +1h, 1e-3 never completed.
- The 13-species `conver_ignore` list: exists in no upstream repository; `[]` and `['HC3N']` measured identical.
- Threading `p_ref_bar` through VULCAN-JAX for the radius anchor (7 files, 3 repos): buys < 0.07% of Rp.
- Per-stencil band-local emission evidence arrays: ~4x the emission cache for a gate that is a no-op on every shipped case.
- Applying saturation advice by min()-ing it into the user's limit: silent substitution that changed worker output under an unchanged version.
- A min() over the band for the emission bottom-tau gate, or deepening `p_btm_bar` to buy opacity: a 1.6% notch vetoed HD 189733 b.
- Copying the k-tables into the bucket, or a single H2O marker for seeded data: pays 4.3 GB twice; an empty directory counted as seeded.
- Trusting `provenance.json` for a store's contents: it claimed 17 tables when 11 existed.
- `curl -L -C -` against STScI Box: appends a restart to an oversized file that exits 0.
- PandExo's default templates as a substitute for the registry pins: diverge on NIRISS, NIRCam and MIRI.
- Extending `MODE_KEYS`/`PANDEXO_MODES` without a PandExo run: PandExo is installed nowhere here.
- CS2 from pRT Keeper: HITRAN 296 K and different T/P nodes, the data the ExoMolOP switch removed.
- The lay-paraphrase GUI prose sweep: reverted in full by the maintainer.
- The jitter scale knob and sliders: a plot that contradicts its own error bars; Streamlit clamps seeded values.
- A C/O widget ceiling that tracks network and photolysis: it clamps a stored value silently.
- A GUI warning when sigma_lnCO exceeds the distance to C/O = 1: caveats live in the notes, by policy.
- The AD photo-lock that force-wrote the photolysis checkbox: destroyed the stored value.
- Deleting `tests/unit/test_transits_window.py`: it pins the floored semantics `test_detect` delegates to.
- Reading the SOSS order-2 response factor as a kernel effect, or rescaling a whole Fisher row for a per-mode response.
- A noise-weighted FD gate (the noise vector is unknown at FD time), or a seed-map redesign before any carbon-rich case certifies.
- Explaining a step-count gap by exit `dt`: dt is not monotone and collapses on rejects.
- `max(floor, 1e-30)` as the floor-only transit limit under a mixed floor: a target met in one transit read unreachable.
- Re-raising dead audit leads: TestPyPI install remedies, "cached NPZ carry no producer", "fisher's `diag` has no caller".
- Re-enabling Streamlit XSRF/CORS on the Space: untestable behind the Spaces proxy; the 4 MB upload cap is the control.
- Rewording the beta banner: the maintainer's voice.
- Widening the photolysis-off `jac_method='ad'` refusal wording: conservative and test-pinned.
- Escalating the photolysis cadence per STENCIL POINT: the cadence offset divides by h and passes the gate unnoticed.
- Mapping the tool's composition onto picaso's (log mh, C/O): leaves He 0.074 dex and N, S, metals 0.08-0.29 dex off.
- picaso's layer-to-layer warm start in the composition-locked solver: stale atom totals blind the element gate.
- An lnKzz FD row under the PICASO quench: whole-level quench points make the row a staircase.
- Threads or a 16-gas correlated-k preload for the climate solve: same time; 16 gases land at TOA 9e-4.
- picaso's native RT as the PICASO spectrum: 688 ppm median against the tool's RT (gate 150) and two opacity datasets.
- Normalizing the PICASO TOA flux gate by the stellar flux so the quench-in-the-loop climate passes: the gate would pass any climate.
- The timestep stop-gradient as the eclipse AD repair: it breaks the independent full-integration Kzz derivative.
- A step floor on the AD warm map: a lucky stop, withdrawn as a magic number.
- Treating the composition FD frozen anchor as a bug from the code shape alone: settle it by measurement (Jacobians and Fisher).
