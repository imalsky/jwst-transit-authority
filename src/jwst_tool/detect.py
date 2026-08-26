"""Detection-significance math: bin the model per instrument THROUGH THE SAME
count-space measurement operator as the noise, combine with the per-bin depth
uncertainty, and score the science goal.

Model, removed-molecule model, Jacobians, and noise all go through one
operator (binning.build_operator, flux-weighted). Fully saturated pixels are
excluded from the operator; partially saturated pixels are kept but counted
per bin. For modes whose final bins approach the native resolving power
(MIRI LRS, NIRSpec PRISM, SOSS) the model is first blurred to the
instrument's measured response, R(lambda) / instruments.LSF_WIDTH
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
additionally projects out the available T-P, lnR0, AND cloud derivative
directions (_NUISANCE_JAC below; still conditional) and is the number to
prefer for narrow margins -- any caption describing it must include the
cloud directions.

Multi-transit extrapolation: the random term scales as 1/N; the minimum
floor is a hard lower bound at every N, so "transits to target" saturates
honestly instead of promising 1/sqrt(N) forever.
"""
from __future__ import annotations

import numpy as np

from . import binning
from . import instruments as ins
from . import forward
from . import noise as noise_mod

# transits-to-target scan cap: beyond this, effectively unreachable anyway
N_TRANSITS_CAP = 500

# Nuisance directions for sigma_detect_proj: T-P parameters, the reference
# radius, and the cloud deck (an uncertain deck absorbs broadband signal
# like an offset). Never add the chemistry rows (lnZ, dlnCO, lnKzz) -- they
# ARE the signal being scored. Must track forward.TP_PARAM_NAMES +
# forward.CLOUD_FISHER_PARAMS.
_NUISANCE_JAC = frozenset(
    {"Tirr", "Tint", "Tint_cl", "log_kappa", "log_gamma", "lnR0",
     "log_kappa_cloud", "alpha_cloud"})


def _segment_rows(seg: np.ndarray) -> list[np.ndarray]:
    """Indicator rows (one per segment beyond the first); with the constant
    offset they span the per-segment offset space."""
    seg = np.asarray(seg, int)
    return [(seg == s).astype(float) for s in range(1, int(seg.max()) + 1)]


def detection_significance(signal: np.ndarray, sigma: np.ndarray,
                           nuisance: list[np.ndarray] | None = None,
                           marginalize_offset: bool = True) -> float:
    """sqrt(Delta chi^2) of a binned signal against noise, with linear
    nuisance directions profiled out (rank-aware).

    ``marginalize_offset=True`` (default) includes a constant depth offset;
    ``nuisance`` adds arbitrary extra rows. The result depends only on the
    SPAN of the nuisance rows, never on their amplitudes: the normal matrix
    is Jacobi-normalized (correlation form) before the rank-revealing
    eigen-threshold -- never threshold raw eigenvalues of a mixed-unit
    matrix. Numerically null directions are dropped rather than inverted;
    zero-norm rows are excluded outright.

    The metric is the exact diagonal W = diag(1/sigma^2).

    Inputs are validated loudly: ``signal`` 1-D and finite; ``sigma``
    matching, finite, > 0; nuisance rows matching.
    """
    signal = np.asarray(signal, float)
    if signal.ndim != 1 or signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("detection_significance: signal must be a non-empty "
                         "1-D finite array")
    for i, r in enumerate(nuisance or []):
        if np.asarray(r).shape != signal.shape:
            raise ValueError(f"detection_significance: nuisance row {i} has "
                             f"shape {np.asarray(r).shape}, expected "
                             f"{signal.shape}")
    sig = np.asarray(sigma, float)
    if sig.shape != signal.shape or not np.all(np.isfinite(sig)) \
            or np.any(sig <= 0.0):
        raise ValueError("detection_significance: sigma must match signal's "
                         "shape and be finite and > 0")
    # keep the constant row even for a single bin: one bin + free offset =
    # zero shape information, so the honest score is 0 (never |s|/sigma)
    rows = [np.ones_like(signal)] if marginalize_offset else []
    rows += [np.asarray(r, float) for r in (nuisance or [])]
    w = 1.0 / np.asarray(sigma, float) ** 2
    chi2 = float(np.sum(w * signal ** 2))
    if rows:
        U = np.stack(rows)
        A = (U * w) @ U.T
        b = (U * w) @ signal
    if rows:
        # normalize to correlation form so the rank decision depends on the
        # nuisance SPAN, not on row amplitudes/units
        d = np.sqrt(np.clip(np.diag(A), 0.0, None))
        keep = d > 0.0
        if keep.any():
            An = A[np.ix_(keep, keep)] / np.outer(d[keep], d[keep])
            bn = b[keep] / d[keep]
            ew, ev = np.linalg.eigh(0.5 * (An + An.T))
            good = ew > 1e-12 * max(float(ew[-1]), 1e-300)
            if good.any():
                proj = ev[:, good].T @ bn
                chi2 -= float(np.sum(proj ** 2 / ew[good]))
    return float(np.sqrt(max(chi2, 0.0)))


# ONE transit-count validator for the whole stack; never max(1, int(n))
# (history: notes.md)
_n_transits = noise_mod.n_transits_int


def sigma_at_transits(result: dict, n_transits: int) -> np.ndarray:
    """Per-bin depth sigma of an evaluated mode re-scaled to ``n_transits``.

    Photon/detector variance (inflation included) scales as 1/N from the
    evaluated count; the minimum floor is a hard lower bound at every N
    (PandExo semantics): sigma_N = max(sigma_random_N, floor).
    """
    n0 = int(result["n_transits_eval"])
    scale = n0 / float(_n_transits(n_transits))
    return np.maximum(np.sqrt(np.asarray(result["var_phot"]) * scale),
                      np.asarray(result["floor"]))


def _result_nuisance(result: dict) -> list[np.ndarray]:
    """The evaluated mode's profiled calibration rows: per-segment offset
    steps."""
    return _segment_rows(result["seg"]) if "seg" in result else []


def _detection_nuisance(result: dict, projected: bool) -> list[np.ndarray]:
    """Calibration rows, plus local physical nuisance rows when requested."""
    rows = _result_nuisance(result)
    if not projected:
        return rows
    names = [str(x) for x in result.get("jac_names", [])]
    jac = result.get("jac_bins")
    if jac is None or not names:
        raise ValueError(
            "projected detection score requires jac_bins and jac_names")
    jac = np.asarray(jac, float)
    if jac.ndim != 2 or jac.shape[0] != len(names):
        raise ValueError(
            "projected detection score: jac_bins rows must match jac_names")
    return rows + [jac[i] for i, name in enumerate(names)
                   if name in _NUISANCE_JAC]


def detection_score(result: dict, sigma: np.ndarray | None = None,
                    *, projected: bool = False) -> float:
    """Recompute a result's matched-template score with one explicit metric.

    ``projected=False`` preserves the historical calibration-profiled score.
    ``projected=True`` additionally profiles the available local T-P,
    reference-radius, and cloud Jacobian directions. The latter is the
    collaborator-facing score whenever those Jacobians are available.
    """
    if result.get("depth_wo") is None:
        return float("nan")
    signal = np.asarray(result["depth"]) - np.asarray(result["depth_wo"])
    use_sigma = np.asarray(result["sigma"] if sigma is None else sigma)
    return detection_significance(
        signal, use_sigma,
        nuisance=_detection_nuisance(result, projected=projected))


def transits_to_target(result: dict, target_sig: float, *,
                       projected: bool = False) -> dict:
    """Smallest transit count reaching ``target_sig`` for the detect goal.

    Returns dict(n=int|None, reachable=bool, sig_inf=float). ``sig_inf`` is
    the infinite-transit (floor-only) limit; the score is monotone in N
    (diagonal noise; sigma_N = max(sigma_random_N, floor)), so sig_inf is an
    exact ceiling and a target above it is unreachable. With no floor set
    anywhere, ``sig_inf`` is inf and unreachable means "needs more than the
    N_TRANSITS_CAP scan limit", not a systematic ceiling.
    """
    if result.get("depth_wo") is None:
        return dict(n=None, reachable=False, sig_inf=float("nan"))
    floor = np.asarray(result["floor"])
    if not np.any(floor > 0.0):
        # no floor: the limit is genuinely INFINITE (report inf, not the
        # ~1e26 the 1e-30 clip would give)
        sig_inf = float("inf")
    else:
        sig_inf = detection_score(result, np.maximum(floor, 1e-30),
                                  projected=projected)
    if target_sig > sig_inf:
        return dict(n=None, reachable=False, sig_inf=sig_inf)
    for n in range(1, N_TRANSITS_CAP + 1):
        if detection_score(result, sigma_at_transits(result, n),
                           projected=projected) >= target_sig:
            return dict(n=n, reachable=True, sig_inf=sig_inf)
    return dict(n=None, reachable=False, sig_inf=sig_inf)


# Native-grid pixel-census keys, counted BEFORE the worker's finite/positive
# `good` filter: fully saturated channels have non-finite extracted noise,
# so a post-filter count reads low (zero for a mode saturated everywhere).
_NATIVE_PIXEL_KEYS = (
    "n_pix_native",
    "n_pix_unusable_dropped",
    "n_pix_part_sat_native",
    "n_pix_full_sat_native",
)


def _native_pixel_counts(mode_result: dict) -> dict:
    """The worker's native-grid pixel census, or None per key.

    None means UNMEASURED (an older payload) -- never substitute zero or the
    post-filter count, which would report a saturated column as clean.
    """
    return {k: (int(mode_result[k]) if mode_result.get(k) is not None else None)
            for k in _NATIVE_PIXEL_KEYS}


def _removed_spectrum(model: dict, mols: list[str], target_mol,
                      order: np.ndarray) -> np.ndarray | None:
    if target_mol is None:
        return None
    if target_mol not in mols:
        raise ValueError(
            f"target molecule {target_mol!r} is not in the cached model's RT "
            f"set {mols} -- re-run the forward model with it enabled "
            "(extra_mols)")
    # v32: depth_wo rows (and the emis_*_wo certificates) align with the
    # model's wo_mols set, not with mols. REQUIRED key: every v32 cache
    # stores it, and pre-v32 caches are unreachable (the version is in the
    # cache key), so a missing key means a malformed payload, not an old one.
    wo = [str(m) for m in np.asarray(model["wo_mols"])]
    if target_mol not in wo:
        raise ValueError(
            f"this cached model computed removed-molecule spectra only for "
            f"{wo or 'no molecules'} (wo_mols), not for {target_mol!r} -- it "
            "was a constrain-goal run. Re-run the forward model with the "
            "detect goal (or wo_mols including the target).")
    index = wo.index(target_mol)
    if str(model.get("science_mode", "transmission")) == "emission":
        # v27: FLUX-WEIGHTED. The min-tau form this replaces refused a target
        # whenever the column saw through ANYWHERE, and on every planet tried
        # that was a 60 nm notch at the extreme blue edge carrying under a
        # thousandth of the planet's emission -- so it refused everything.
        # run_model writes this key with every emission model, and _VERSION
        # rides in the cache key, so a readable model always carries it.
        if "emis_thin_flux_frac_wo" in model:
            frac = float(np.asarray(model["emis_thin_flux_frac_wo"])[index])
            if np.isfinite(frac) and frac > forward.EMIS_THIN_FLUX_FRAC:
                raise ValueError(
                    f"{target_mol} emission detection is not supported for "
                    f"this atmosphere: with {target_mol} removed, "
                    f"{100.0 * frac:.1f}% of the emitted flux comes from "
                    "wavelengths where the RT column bottom is optically thin, "
                    "so its eclipse detection contrast would be overstated. "
                    "Detect a molecule with deeper opacity, or use "
                    "transmission.")
    return np.asarray(model["depth_wo"])[index][order]


def _mode_warnings(mode: dict, mode_result: dict, t_in_s: float,
                   lsf_skip_note: str | None) -> dict:
    warnings = dict(mode_result.get("warnings", {}))
    t_cycle = float(mode_result["t_cycle_s"])
    n_cycles = t_in_s / t_cycle
    if n_cycles < 3.0:
        warnings[f"only {n_cycles:.1f} integration cycles fit in transit "
                 "(PandExo enforces >= 3 by shortening the ramp)"] = True
    # Long-ramp disclosure. The saturation search maximizes the ramp, which on
    # a faint target can run far past what STScI advises or APT allows; the
    # tool ranks configurations and never silently shortens one, so say so.
    # t_cycle includes the between-integration reset, so this fires marginally
    # early -- the conservative direction.
    _limit = ins.INT_LENGTH_LIMIT_S.get(mode["instrument"])
    if _limit is not None and t_cycle > _limit[0]:
        warnings[f"integration is {t_cycle:.0f} s, above the "
                 f"{_limit[0]:.0f} s {_limit[1]}; split the ramp or verify "
                 "it in APT"] = True
    if lsf_skip_note:
        warnings[lsf_skip_note] = True
    ngroup = int(mode_result["ngroup"])
    if ngroup < int(mode["ngroup_warn_below"]):
        reason = ins.NGROUP_WARN_REASON[mode["instrument"]]
        warnings[f"ramp uses {ngroup} group(s) per integration, below this "
                 f"mode's STScI-recommended ramp ({reason}); verify in APT"] = True
    if mode_result.get("ramp_search_complete") is False:
        warnings["the group search hit its calculation budget; the reported "
                 "ramp is measured-safe but may not be the longest possible "
                 "(costs sensitivity, never validity)"] = True
    if (mode["instrument"] == "miri"
            and ngroup == int(mode["ngroup_min"])
            and not bool(mode_result.get("saturated", False))):
        warnings["MIRI floor ramp: 2 groups/integration is MIRI's shortest "
                 "permitted ramp and is very difficult to calibrate "
                 "accurately; confirm in APT whether this configuration "
                 "needs special approval before proposing"] = True
    # A non-primary diffraction order is not a separate observation, and its
    # saturation verdict is not its own. Both facts are easy to get wrong from
    # the mode table alone, so state them on the row that carries the risk.
    if int((mode.get("strategy") or {}).get("order", 1)) > 1:
        warnings["this order shares one detector readout with order 1: "
                 "selecting both orders costs one observation, not two, and "
                 "both report the same ramp and cadence"] = True
        if bool(mode_result.get("saturated", False)):
            warnings["the saturation verdict here is set by the brighter "
                     "order-1 trace (Pandeia measures saturation over the "
                     "whole readout); STScI puts the SUBSTRIP256 bright limit "
                     "about 2 magnitudes fainter in order 1 than in order 2 "
                     "(J ~ 8.5 vs J ~ 6.3), so this order's own trace may "
                     "still be usable -- check it in the ETC"] = True
    return warnings


def _usable_pixels(mode_key: str, mode_result: dict) -> tuple:
    wl = binning._validate_wl(mode_result["wl"], f"{mode_key}: wl")
    flux = np.asarray(mode_result["flux"], dtype=float)
    full = np.asarray(mode_result.get("n_full_sat", np.zeros(wl.size)))
    partial = np.asarray(mode_result.get("n_part_sat", np.zeros(wl.size)))
    if wl.size < 2 or flux.shape != wl.shape:
        raise ValueError(f"{mode_key}: pixel wavelength and flux must be aligned 1-D arrays")
    if not np.all(np.isfinite(flux)) or np.any(flux < 0.0):
        raise ValueError(f"{mode_key}: pixel flux must be finite and non-negative")
    if full.shape != wl.shape or partial.shape != wl.shape:
        raise ValueError(f"{mode_key}: saturation arrays must match the pixel grid")
    degenerate = binning.degenerate_wl_mask(wl)
    usable = (full == 0) & ~degenerate
    if not usable.any():
        raise ValueError(
            f"{mode_key}: no usable pixels -- all {wl.size} are fully "
            f"saturated ({int((full > 0).sum())}) or wavelength-degenerate "
            f"({int(degenerate.sum())}). Choose a mode with a larger "
            "saturation margin or a different target, or check the "
            "worker's wavelength grid")
    return wl, flux, full, partial, degenerate, usable


def evaluate_mode(mode_key: str, mode_result: dict, model: dict, target_mol,
                  R_bin: float, t_in_s: float, t_out_s: float, n_transits: int,
                  floor_spec, noise_inflation: float = 1.0) -> dict:
    """One instrument mode -> binned model, sigmas, conditional template S/N.

    Bins cover the intersection of the mode's science band, the model's
    coverage, and the pixels pandeia returned; model, Jacobians, and noise
    share ONE count-space operator (module docstring). ``target_mol=None``
    (the parameter-constraint goal) skips the molecule-removed comparison:
    ``sigma_detect`` comes back NaN and ``depth_wo`` None.
    """
    m = ins.MODES[mode_key]
    wl_model = binning._validate_wl(model["wl_um"], f"{mode_key}: wl_model")
    depth_raw = np.asarray(model["depth"], dtype=float)
    if wl_model.size < 3 or depth_raw.shape != wl_model.shape:
        raise ValueError("model wavelength and depth must be aligned 1-D arrays with >=3 points")
    if not np.all(np.isfinite(depth_raw)):
        raise ValueError("model depth must be finite")
    order = np.argsort(wl_model)
    wl_model = wl_model[order]
    depth = depth_raw[order]
    # The model's own resolving power, measured from its grid (correlated-k
    # is fixed at the published k-tables' R = 1000 band grid).
    _lnw = np.log(wl_model)
    r_model = (1.0 / float(np.median(np.diff(_lnw)))
               if _lnw.size > 2 else float("inf"))
    if float(R_bin) > r_model:
        raise ValueError(
            f"{mode_key}: analysis resolving power R_bin={float(R_bin):g} is "
            f"finer than the model spectrum's own resolving power "
            f"(R = {r_model:.0f}). Bins narrower than the model grid would "
            "report interpolated structure the model does not contain. Lower "
            "R_bin to at most the model resolving power: the grid is the "
            "published k-tables' R = 1000 band grid and cannot be raised.")
    mols = [str(x) for x in model["mols"]]
    depth_wo = _removed_spectrum(model, mols, target_mol, order)

    wl_pix, flux_pix, n_full_sat, n_part_sat, degen, usable = _usable_pixels(
        mode_key, mode_result
    )

    lo = max(m["wl_min"], float(wl_model.min()), float(wl_pix[usable].min()))
    hi = min(m["wl_max"], float(wl_model.max()), float(wl_pix[usable].max()))
    if hi <= lo:
        raise ValueError(f"{mode_key}: no overlap between instrument band and model")

    # blur the model to the native R(lambda) when exported; a no-op for
    # high-R modes. Pixel cells extend at most one native pixel past the
    # bin span, hence the margin.
    r_native = mode_result.get("r_native")
    lsf_applied = False
    _lsf_skip_note = None
    jac_rows = None
    if "jac" in model:
        jac_rows = [np.asarray(row)[order] for row in model["jac"]]
    if r_native is not None:
        r_nat = np.asarray(r_native, float)
        b_lo = max(float(wl_model.min()), lo * 0.97)
        b_hi = min(float(wl_model.max()), hi * 1.03)
        # stellar flux weights the LSF so it forms the observed count ratio
        # L[F d]/L[F], never the flat depth blur L[d]; same weight for depth,
        # removed-molecule, and Jacobian rows (operator stays linear in depth).
        # Degenerate (duplicate-wavelength) pixels are masked out of the
        # operator; the R(lambda) table excludes them too, since a repeated
        # abscissa is not a strictly ascending curve.
        po = np.flatnonzero(~degen)[np.argsort(wl_pix[~degen])]
        flux_model = np.maximum(np.interp(wl_model, wl_pix[po], flux_pix[po]), 0.0)
        # R(lambda) must go in ASCENDING wavelength order: the pandeia pixel
        # grid is dispersion order, not wavelength order, and MIRI LRS ships
        # it DESCENDING (13.86 -> 5.02 um). Passing it raw made the operator's
        # np.interp read a reversed table and return the curve's LAST entry,
        # R = 42, everywhere. Note which end that is: MIRI LRS resolving power
        # RISES with wavelength (42 at 5 um, 150 median, 209 at 12 um), so 42
        # is the BLUE end, and pinning the whole 5-12 um band to it blurred it
        # with a ~3.5x-too-wide kernel. smooth_to_native_r now refuses an
        # out-of-order curve, so this sort is the contract, not a convenience.
        # The kernel width is the MEASURED response, R_refdata / the mode's
        # impulse-fitted width (instruments.LSF_WIDTH): on the slitless modes
        # the PSF, not the dispersion, sets it.
        wl_r = wl_pix[po]
        r_curve = ins.lsf_r(mode_key, wl_r, r_nat[po])
        depth_sm = binning.smooth_to_native_r(wl_model, depth, wl_r, r_curve,
                                              b_lo, b_hi, weight=flux_model)
        # metadata ONLY -- never gate the blur of OTHER vectors on this: a
        # flat baseline is a fixed point of the LSF while a narrow Jacobian
        # feature is not (gating left Jacobians unsmoothed by ~59 ppm)
        lsf_applied = bool(np.any(depth_sm != depth))
        depth = depth_sm
        if not lsf_applied:
            # The operator auto-no-ops when the model grid cannot resolve the
            # kernel. That is the more dangerous half and must be disclosed:
            # the model's own opacity smearing then stands in for the
            # instrument line-spread function, which is a different width set
            # by a numerical knob.
            # A Gaussian kernel needs R_model >= 2.3548 * R_native to be
            # resolved at all, and the binding R_native is the SMALLEST in
            # band (the widest kernel), matching smooth_to_native_r's test.
            # r_model is measured from the model grid itself: the band
            # grid is fixed by the published k-tables.
            _rn = float(np.nanmin(r_curve)) if np.isfinite(r_curve).any() else 0.0
            _lsf_skip_note = (
                f"native-R LSF is a NO-OP here: the model spectrum's own "
                f"resolving power (R = {r_model:.0f}) cannot resolve this "
                f"mode's line-spread function (response R down to {_rn:.0f}, "
                "which needs about 2.35x that in model resolving power). The "
                "model's own opacity sampling sets the effective width "
                "instead. The model grid is the published k-tables' band "
                "grid and cannot be raised.")
        if depth_wo is not None:
            depth_wo = binning.smooth_to_native_r(wl_model, depth_wo, wl_r,
                                                  r_curve, b_lo, b_hi,
                                                  weight=flux_model)
        if jac_rows is not None:
            jac_rows = [binning.smooth_to_native_r(wl_model, row, wl_r,
                                                   r_curve, b_lo, b_hi,
                                                   weight=flux_model)
                        for row in jac_rows]
    else:
        # a missing native-R leaves depth and Jacobians UNBLURRED -- safe only
        # for high-R modes (every shipped low-R mode has a dispersion file),
        # so surface it loudly via the result's warning channel below
        _reason = mode_result.get("r_native_source") or "no native-R exported"
        _lsf_skip_note = (f"native-R LSF NOT applied ({_reason}); depth and "
                          "Jacobians unblurred -- safe only for high-R modes, a "
                          "refdata/config error on a low-R mode")

    edges = noise_mod.make_bins(lo, hi, R_bin)
    op = binning.build_operator(wl_pix, flux_pix, edges,
                                wl_lo=float(wl_model.min()),
                                wl_hi=float(wl_model.max()), valid=usable)
    nz = noise_mod.depth_error_bins(mode_result, edges, t_in_s, t_out_s,
                                    n_transits, floor_spec, op=op,
                                    noise_inflation=noise_inflation)

    # detector segments (NRS1|NRS2 for the two-detector gratings) -> one
    # calibration-offset nuisance per segment in every score/forecast
    seg_full = np.zeros(wl_pix.size, int)
    seg_full[usable] = binning.segment_ids(wl_pix[usable])
    seg = binning.bin_segments(op, seg_full)
    steps = _segment_rows(seg)

    d_full_b = binning.bin_model(op, wl_model, depth)
    jac_bins = None
    jac_names = ([str(x) for x in model["jac_names"]]
                 if "jac_names" in model else [])
    if jac_rows is not None:
        jac_bins = np.stack([binning.bin_model(op, wl_model, row)
                             for row in jac_rows])
    if depth_wo is not None:
        d_wo_b = binning.bin_model(op, wl_model, depth_wo)
        s_b = d_full_b - d_wo_b
        sigma_detect = detection_significance(s_b, nz["sigma"], nuisance=steps)
        # also profile the T-P/cloud/lnR0 Jacobian directions (conditional)
        sigma_detect_proj = float("nan")
        if jac_bins is not None and jac_names:
            nuis = steps + [jac_bins[i] for i, n in enumerate(jac_names)
                            if n in _NUISANCE_JAC]
            sigma_detect_proj = detection_significance(s_b, nz["sigma"],
                                                       nuisance=nuis)
    else:
        d_wo_b, sigma_detect, sigma_detect_proj = None, float("nan"), float("nan")

    warnings = _mode_warnings(m, mode_result, t_in_s, _lsf_skip_note)

    keep = op["keep"]
    return dict(
        jac_bins=jac_bins, jac_names=jac_names,
        mode_key=mode_key, label=m["label"],
        wl=nz["wl_center"],
        wl_eff=binning.bin_values(op, wl_pix),
        bin_lo=edges[:-1][keep], bin_hi=edges[1:][keep],
        seg=seg, n_segments=int(seg.max()) + 1 if seg.size else 1,
        depth=d_full_b, depth_wo=d_wo_b, sigma=nz["sigma"],
        var_phot=nz["var_phot"], floor=nz["floor"],
        noise_infl=float(noise_inflation), lsf_applied=lsf_applied,
        n_transits_eval=int(nz["n_transits"]),
        sigma_detect=sigma_detect, sigma_detect_proj=sigma_detect_proj,
        median_sigma_ppm=float(np.median(nz["sigma"]) * 1e6),
        n_bins=int(nz["wl_center"].size),
        n_pix_partial_sat=binning.bin_counts(op, n_part_sat > 0).astype(int),
        # POST-FILTER count: blind to channels the worker already dropped
        # (exactly the fully saturated ones) -- use the *_native fields for
        # display/export.
        n_pix_full_sat_dropped=int(np.sum(n_full_sat > 0)),
        n_pix_degenerate_dropped=int(degen.sum()),
        # native-grid truth from the worker; None = not measured -- never
        # substitute the post-filter count
        **_native_pixel_counts(mode_result),
        ngroup=int(mode_result["ngroup"]),
        sat_frac=float(mode_result["sat_frac"]),
        sat_ngroups=mode_result.get("sat_ngroups"),
        saturated=bool(mode_result.get("saturated", False)),
        t_cycle_s=float(mode_result["t_cycle_s"]),
        warnings=warnings,
    )
