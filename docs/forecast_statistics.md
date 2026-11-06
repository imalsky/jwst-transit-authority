# Detection scores and Fisher forecasts

## `detect`

Detection-significance math.

Bin the model per instrument THROUGH THE SAME count-space measurement operator as the noise, combine with the per-bin
depth uncertainty, and score the science goal.

Model, removed-molecule model, Jacobians, and noise all go through one
operator (binning.build_operator, flux-weighted). Fully saturated pixels are
excluded from the operator; partially saturated ones are kept (the worker's
native-grid census reports both). For modes whose final bins approach the
native resolving power (MIRI LRS, NIRSpec PRISM, SOSS) the model is first
blurred to the instrument's measured response, R(lambda) / instruments.LSF_WIDTH
(binning.smooth_to_native_r); a no-op for high-R gratings.

sigma_detect is a CONDITIONAL MATCHED-TEMPLATE S/N at the specified
atmospheric state: the nested-model chi-square distance between the full
spectrum and the spectrum with one molecule's opacity removed, with the
calibration nuisances profiled out --

    chi2 = s^T W s - b^T A^{-1} b,   W = diag(1/sigma^2),
    b = U W s,  A = U W U^T,         s_b = d_full - d_without_X

The rows of U are one constant depth offset and one step per extra detector
segment (NRS1|NRS2 -- real G395H fits float such offsets). It is NOT a
retrieval detection significance: the atmosphere
is held at the specified state, and a retrieval that frees more parameters
under the same model and noise assumptions will usually report a lower
significance (a best-case comparison under those conditions, not a universal
bound). When a Fisher Jacobian is available, ``sigma_detect_proj``
additionally projects out whichever of the T-P, lnR0 and cloud derivative
directions the run actually built (_NUISANCE_JAC below; still conditional) and
is the number to prefer for narrow margins -- any caption describing it must
name the rows that were present, which metric_label does.

Multi-transit extrapolation: the random variance scales as 1/N; the minimum
floor is a hard lower bound at every N, so with a floor in every bin the
score reaches a ceiling (``sig_inf`` in transits_to_target) instead of
growing as sqrt(N) without bound.

## `fisher`

Fisher-information parameter forecast from the spectrum Jacobian.

Given each mode's binned Jacobian J (n_par, n_bins) and per-bin depth sigma,

    F_ij = sum_b J_ib J_jb / sigma_b^2

and the marginalized 1-sigma forecast on parameter i is sqrt((F^-1)_ii).

Nuisances, jointly fit and marginalized out of the report:
  * per mode: lnR0 (reference radius) + one constant depth OFFSET per detector
    SEGMENT (NRS1/NRS2 separately for the two-detector gratings -- a
    detector-to-detector step of tens of ppm is common in real fits and can
    otherwise masquerade as atmospheric structure).
  * combined: one SHARED lnR0 + one offset per SEGMENT across all modes.

A parameter with no spectral response or loaded on a numerically null
Fisher direction comes back inf ("unconstrained" in the GUI). A finite width
is always reported, however wide. The inversion is
ALWAYS rank-aware and unit-invariant: rank detection happens on the
Jacobi-whitened (unit-diagonal) Fisher matrix (see _whitened_eig).
np.linalg.inv returns misleading finite numbers on ill-conditioned matrices,
so it is never used. Forecast sigmas are local Cramer-Rao lower bounds under
the quoted noise model -- best cases, not posterior widths (Vallisneri 2008).

## `posteriors`

Marginalized Fisher-Gaussian forecast densities + named mode combos.

Pure numpy, GUI-independent. Everything here is a LINEARIZED forecast read off
the same nuisance-augmented Fisher matrices as the constraint table
(``fisher.mode_forecast`` / ``fisher.combined_forecast``): each 1D curve is
the Gaussian N(theta_input, sigma_marg) with sigma_marg the marginalized
Cramer-Rao bound in the fitted coordinate. Curves are rendered in DISPLAY
units (the fisher-table convention:
dex for lnZ/lnKzz, absolute C/O number ratio for dlnCO via
``fisher.display_sigma``, K for temperatures; a parameter unknown to
``fisher.PARAM_UNITS`` is already in display units and reports unit "").
The dlnCO curve is the forecast density in ln(C/O) on a log C/O axis
(ln_gaussian_curve); its width is quoted in dex via fisher.format_co_width.

Labeling (every returned record carries ``kind`` and ``label``): these
are linearized Fisher/Cramer-Rao forecasts around the input model under the
quoted noise model -- Gaussian by construction, local, best-case. They are
NOT sampled posteriors, and a retrieval freeing more parameters under the
same model and noise assumptions usually reports lower significance. Never
present a curve from this module as a retrieval result.

Rank-deficient directions: a parameter on a numerically null Fisher direction
comes back from ``fisher._marg_sigmas`` as inf; here it is an explicit
``constrained=False`` record with NO curve (theta/pdf None) -- never a fake
finite Gaussian. Non-finite inputs raise (house style: fail fast and loud).

Named combinations (``combo_forecast``): a combo is a user-facing name plus
a list of registry mode keys, evaluated against the per-mode result dicts of
a run through ``fisher.combined_forecast`` (``mode_forecast`` is its
single-mode case). Saturated modes are excluded with the same policy as the
"All usable modes" row (a saturated mode is unusable data) and the
exclusion is disclosed per combo; a combo with zero usable modes raises.
