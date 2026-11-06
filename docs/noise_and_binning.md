# Noise, instruments and the measurement operator

## `instruments`

JWST time-series instrument-mode registry + paths for the noise backend.

Each ``MODES`` entry carries the Pandeia configuration used by
``pandeia_worker.py`` (running in the Pandeia backend's own environment),
display metadata, and an illustrative systematic noise floor.

Mode tokens are the engine mode names of the supported Pandeia release
and are submitted verbatim by ``noise.noise_job`` and the parity harness alike.

Noise floors (``floor_ppm_suggested``) are ILLUSTRATIVE planning values, not
measured calibrations: ``detect.evaluate_mode`` and ``noise.depth_error_bins``
require ``floor_spec`` explicitly. The GUI preselects a constant floor using
these values, displays the selection, and records it in provenance; users can
choose no floor or upload a wavelength-dependent table. The floor uses PandExo
semantics (sigma_final = max(sigma_random, floor) on the final bins), so a
15-40 ppm floor DOMINATES any well-observed target, while zero assumes a
precision no program has demonstrated. The values are per-mode planning suggestions INFORMED BY
the Greene et al. 2016 convention (20/30/50 ppm for NIRISS/NIRCam/MIRI), not
that convention verbatim (here: NIRSpec 15-20, NIRISS 20, NIRCam 25, MIRI 40);
no value here is a measured end-to-end floor. Any caption describing the
prefills must describe THESE values, not Greene's.

Band edges (``wl_min``/``wl_max``): the mode's usable science bandpass from
a published source, intersected with the forward model's 1-15 um coverage
(short edge = the H2-H2 CIA table). Per instrument:
  * NIRSpec -- the BOTS/S1600A1 table, SUB2048 column: jwst-docs "NIRSpec
    BOTS Wavelength Ranges and Gaps" Table 1, identical to Birkmann et al.
    2022 (A&A 661, A83) Table 2. NOT the nominal disperser table, which is
    wider: the BOTS values already carry the red-end detector cutoffs
    (G235x 3.07/3.12 there against a nominal 3.17).
  * NIRISS -- the Pandeia order-1 range for GR700XD (0.83-2.81), rounded
    inward.
  * NIRCam -- the filter half-power points measured from the shipped
    transmission curves, rounded INWARD to 0.05 um.
  * MIRI -- LRS 5-12 um; the 5.0 short edge is the jwst-docs caution about
    the slitless dispersion turnover below 4.5 um, not a rounding.
Every edge is checked against the Pandeia extracted grid the worker returns,
and pinned to its source by tests/unit/test_instruments_registry.py. Never
narrow an edge without a sourced reason on the entry.

Saturation vocabulary: Pandeia's ``fraction_saturation`` is the fraction of
the engine's per-mode ``saturation_fullwell``, which is NOT the physical full
well and is not the same fraction of it on every instrument -- NIRCam
lw_tsgrism 58,100 e- (pandeia_data deliberately holds time-series modes at
70% of the ~83,000 e- well, and jwst-docs states the NIRCam grism-TSO
saturation limits "as 70% of the pixel well capacity"), NIRISS 72,000 e- (a
soft-nonlinearity value below the ~87,300 e- physical well), NIRSpec 65,000
e-, MIRI 193,655 e-. ``sat_limit`` multiplies THAT number, so an 80% limit is
~56% of the NIRCam well but 80% of NIRSpec's adopted value. Never call it a
physical full-well fraction.

## `noise`

Host-side noise interface: subprocess to the Pandeia worker + transit-depth error math.

The worker gives, per instrument mode, the per-native-pixel extracted stellar
flux and its 1-integration sigma. This module turns that into a transit-depth
uncertainty per spectral bin:

    depth = 1 - F_in/F_out
    var(depth)_pixel = (sigma_1int/flux)^2 * (1/n_int_in + 1/n_int_out) / n_transits
    var(depth)_bin   = sum(w^2 var_pixel) / (sum w)^2,  w = flux   (count-space)
    sigma_bin        = max(sqrt(var_bin), floor)         (PandExo floor convention)

The count-space combination is the variance of the SAME estimator detect.py
uses to bin the model (binning.build_operator) -- never use the
inverse-variance combination, which quotes a different (optimal-weights)
estimator than the one the model is binned with.

The minimum floor follows PandExo semantics exactly (resolve_floor): a HARD
MINIMUM on the final binned uncertainty -- never added in quadrature, never
scaled with the requested resolving power, never averaged down by transits.

`floor_spec` has NO DEFAULT anywhere and must be passed explicitly: a default
floor would silently set the headline precision, and a default of zero would
claim precision no program has demonstrated. The caller chooses, and the
choice is recorded in the run provenance. The per-mode values in
`instruments.MODES` are named `floor_ppm_suggested` and only prefill a GUI
widget.

Scope: a Pandeia-extracted-noise BOX-TRANSIT planning forecast under the
selected extraction/detector configuration. NEVER label sigmas
PandExo-identical; parity status lives in
validation/parity/outputs/REPORT.md, enforced by the fail-closed gate
(validation/parity/scripts/parity_gate.py, run by run_parity.py). Pandeia's
extracted noise is not a time-series systematics model (no 1/f residuals,
visit-long trends, detrending covariance, or stellar heterogeneity).

Results are cached per mode (_mode_key) by a hash of the star, the mode's
submitted configuration, sat_limit, the refdata, CDBS and PSF paths, the
backend fingerprint (engine version, refdata and PSF identity, CDBS/PHOENIX
identity) and WORKER_VERSION, so the ETC runs once per star and mode and a
changed backend misses the cache. run_pandeia, the parity harness's path,
caches the whole job under its job_key.

## `binning`

One measurement operator per instrument mode.

Count-space binning shared by the noise, the model, and every Jacobian row.

Guardrails:
* ONE operator bins noise, model, and Jacobians -- never add a separate
  binning path; the model point must be the expectation of the statistic
  whose variance is quoted.
* Count-space weights (w_i = extracted stellar count rate F_i) are the
  estimator real reductions implement: d_b = sum(w d)/sum(w),
  Var_b = sum(w^2 Var)/(sum w)^2. In the photon limit this equals
  inverse-variance weighting; with background/read noise it is slightly
  wider -- the variance of the estimator actually used.
* The model is averaged over each pixel's wavelength cell (exact integral of
  the piecewise-linear model), never point-sampled, so a finer model grid
  cannot alias.
* Where final bins approach the NATIVE resolving power (MIRI LRS, PRISM,
  blue SOSS), detect.evaluate_mode blurs depth, removed-molecule depth, and
  every Jacobian row to R_native via smooth_to_native_r BEFORE the cell
  average; on high-R modes the blur is a no-op. The Gaussian kernel
  approximates the full Pandeia response matrix; its width is
  R_refdata / instruments.LSF_WIDTH, the per-mode width fitted to an impulse
  through the engine (validation/parity/scripts/run_parity.py --impulse,
  parity_summary.json["lsf_impulse"]).

## `pandeia_worker`

Pandeia ETC worker -- runs INSIDE the Pandeia environment (JWST_TOOL_PANDEIA_PYTHON).

Standalone on purpose: no imports from the rest of the tool; stdlib, numpy and the pandeia stack (pandeia.engine,
synphot, astropy) only.

    python pandeia_worker.py job.json result.json

job.json:
    {"refdata": <pandeia_refdata path>, "psf_dir": <pandeia_psfs path>,
     "cdbs": <PYSYN_CDBS path>,
     "star": {"teff":.., "log_g":.., "metallicity":.., "ks_mag":..},
     "star_spectrum": {"wave_um": [...], "flux_mjy": [...]},  # optional; the
                                    parity harness's input SED, used as-is
                                    instead of the PHOENIX + Ks normalization
     "sat_limit": 0.80,
     "modes": [{"key":.., "instrument":.., "mode":.., "config": {...},
                "strategy": {...}, "background":.., "background_level":..,
                "ngroup_min":.., "ngroup_max":..}, ...],
     "backend": {...}, "worker_version": ..}   # host cache-key fields; the
                                    worker echoes worker_version into the
                                    provenance

result.json: one entry per mode key, plus a reserved "__provenance__" entry
(the exact backend identity, written before any mode runs). Per mode key:
    {"wl": [...um], "flux": [...e-/s], "noise_1int": [...e-/s, sigma for 1 integ],
     "n_part_sat": [...], "n_full_sat": [...],
                                    NOT group counts: for each extracted
                                    wavelength channel, the NUMBER OF DETECTOR
                                    PIXELS inside that channel's extraction
                                    box that are partially (some groups lost)
                                    or fully (fewer than mingroups usable)
                                    saturated. Pandeia builds them as
                                    sum(saturation_mask == 1) and == 2. Only
                                    "> 0" is ever used downstream, but the
                                    counts are pixels, not groups.
     "r_native": [...],             native resolving power R(lambda) on the wl
                                    grid (a missing dispersion file raises),
     "r_native_source": "..",
     "t_cycle_s": ..,               per-integration CYCLE time in a TSO series
                                    (nint=2 minus nint=1 exposure difference,
                                    which includes the between-integration
                                    reset),
     "ngroup": ..,
     "sat_frac": ..,                pandeia's fraction_saturation: the peak
                                    accumulated charge over the WHOLE scene,
                                    divided by the engine's per-mode
                                    saturation_fullwell. That divisor is not
                                    the physical full well and differs by
                                    mode (instruments.py docstring), and for
                                    a multi-order mode (SOSS) it is the
                                    brightest order, not the extracted one.
     "sat_ngroups": ..,             min over the detector of pandeia's
                                    groups-before-saturation map, i.e. the
                                    ramp length at which the brightest pixel
                                    reaches saturation_fullwell.
     "saturated": bool, "engine_version": "..",
     "ramp_search_complete": bool,  False when the group search ran out of
                                    budget before proving maximality,
     "n_pix_native": ..,             NATIVE-GRID channel counts, taken BEFORE
     "n_pix_unusable_dropped": ..,   the finite/positive `good` filter (fully
     "n_pix_part_sat_native": ..,    saturated channels have non-finite
     "n_pix_full_sat_native": ..,    extracted noise, so `good` drops them).
                                     The two *_sat_native counts are numbers
                                     of CHANNELS holding at least one such
                                     pixel. Diagnostic only: no host code
                                     reads these four.
     "wl_native": [...], "usable_native": [...],
     "n_part_sat_native_curve": [...], "n_full_sat_native_curve": [...],
                                    the same native grid, unfiltered (the
                                    parity harness reads the grid and the
                                    two curves),
     "warnings": {...}}
A mode with no usable pixel returns {"unusable": True, "reason": "..",
"ngroup", "sat_frac", "saturated", "ramp_search_complete", the native counts
and curves, "warnings"} instead; a mode that raised returns
{"error": <traceback>}.

Star normalization: band-integrated synphot photsys normalization to the
2MASS Ks magnitude in vegamag (web-ETC convention). Never use the
at_lambda monochromatic shortcut -- it mis-scales the flux by ~1-4% with
spectral type. Vega reference: the engine uses its OWN refdata Vega; the
local synphot vega_file pin is only an offline guard (the two agree to
0.08 mmag in Ks).

Group selection: probe at ngroup_min; "saturated" means the MEASURED probe
exceeds sat_limit, never a prediction. Otherwise a bracket search returns
the largest measured-safe group count with maximality PROVEN (complete only
when the count equals ngroup_max or the next integer measured unsafe); the
analytic formulas pick the first candidate only. A budget-exhausted search
returns the measured-safe best with ramp_search_complete=False -- reported,
never presented as maximal. A mode saturated at ngroup_min is kept,
flagged, with its degraded numbers so the GUI can say why. Channel-level
saturation comes from the report's 1d curves so the host can exclude or
flag per pixel.

Both the search and the "saturated" verdict read the SCENE-WIDE
fraction_saturation, matching PandExo (on SOSS that is the whole detector
image, every order).

The seed formulas assume saturation_time grows linearly with ngroup. That is
EXACT for every shipped mode: pandeia's saturation_time is
tframe * (ndrop1 + (ngroup - 1) * (nframe + ndrop2) + nframe), and all four
readout patterns used here (NRSRAPID/NISRAPID/RAPID/FASTR1) have nframe = 1
and ndrop2 = 0. A future frame-averaging pattern would break the linearity,
not the search -- the bracket still proves maximality by measurement.
