"""Detection-significance math: bin the model per instrument THROUGH THE SAME
count-space measurement operator as the noise, combine with the per-bin depth
uncertainty, and score the science goal.

Model, removed-molecule model, Jacobians, and noise all go through one
operator (binning.build_operator, flux-weighted). Fully saturated pixels are
excluded from the operator; partially saturated pixels are kept but counted
per bin. For modes whose final bins approach the native resolving power
(MIRI LRS, NIRSpec PRISM, blue SOSS) the model is first blurred to the
instrument's R(lambda) (binning.smooth_to_native_r); a no-op for high-R
gratings.

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
additionally projects out the available T-P, lnR0, AND cloud/Mie derivative
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
from . import noise as noise_mod

# transits-to-target scan cap: beyond this, effectively unreachable anyway
N_TRANSITS_CAP = 500

# Nuisance directions for sigma_detect_proj: T-P parameters, the reference
# radius, and both cloud decks (an uncertain deck absorbs broadband signal
# like an offset). Never add the chemistry rows (lnZ, dlnCO, lnKzz) -- they
# ARE the signal being scored. Must track forward.TP_PARAM_NAMES +
# forward.CLOUD_FISHER_PARAMS + forward.MIE_FISHER_PARAMS.
_NUISANCE_JAC = frozenset(
    {"Tirr", "Tint", "Tint_cl", "log_kappa", "log_gamma", "lnR0",
     "log_kappa_cloud", "alpha_cloud",
     "mie_log_rg", "mie_sigmag", "mie_log_mmr"})


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


def transits_to_target(result: dict, target_sig: float) -> dict:
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
    signal = np.asarray(result["depth"]) - np.asarray(result["depth_wo"])
    nuis = _result_nuisance(result)
    floor = np.asarray(result["floor"])
    if not np.any(floor > 0.0):
        # no floor: the limit is genuinely INFINITE (report inf, not the
        # ~1e26 the 1e-30 clip would give)
        sig_inf = float("inf")
    else:
        sig_inf = detection_significance(signal, np.maximum(floor, 1e-30),
                                         nuisance=nuis)
    if target_sig > sig_inf:
        return dict(n=None, reachable=False, sig_inf=sig_inf)
    for n in range(1, N_TRANSITS_CAP + 1):
        if detection_significance(signal, sigma_at_transits(result, n),
                                  nuisance=nuis) >= target_sig:
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
    wl_model = model["wl_um"]
    order = np.argsort(wl_model)
    wl_model = wl_model[order]
    depth = model["depth"][order]
    mols = [str(x) for x in model["mols"]]
    if target_mol is not None and target_mol not in mols:
        raise ValueError(
            f"target molecule {target_mol!r} is not in the cached model's RT set "
            f"{mols} -- re-run the forward model with it enabled (extra_mols)")
    depth_wo = (model["depth_wo"][mols.index(target_mol)][order]
                if target_mol is not None else None)
    # Emission only: if THIS target's removed-molecule spectrum went optically
    # thin at the RT column bottom (emis_tau_bottom_min_wo < 3), its eclipse
    # contrast is overstated -- refuse this target's detection only; the
    # spectrum, constraints, and other molecules stay usable.
    if (target_mol is not None
            and str(model.get("science_mode", "transmission")) == "emission"
            and "emis_tau_bottom_min_wo" in model):
        _tau_wo = float(np.asarray(model["emis_tau_bottom_min_wo"])[
            mols.index(target_mol)])
        if _tau_wo < 3.0:
            raise ValueError(
                f"{target_mol} emission detection is not supported for this "
                f"atmosphere: with {target_mol} removed, the emission RT column "
                f"bottom is optically thin (min tau {_tau_wo:.2f} < 3), so its "
                "eclipse detection contrast would be overstated. Detect a "
                "molecule with deeper opacity (e.g. SO2/CO2), or use "
                "transmission.")

    wl_pix = np.asarray(mode_result["wl"])
    flux_pix = np.asarray(mode_result["flux"])
    # fully saturated pixels are excluded from the estimator; partially
    # saturated ones kept but counted
    n_full_sat = np.asarray(mode_result.get("n_full_sat", np.zeros(wl_pix.size)))
    n_part_sat = np.asarray(mode_result.get("n_part_sat", np.zeros(wl_pix.size)))
    # degenerate-wavelength pixels (pandeia grid artifacts) claim spectral
    # information that does not exist -- drop them and report the count
    degen = binning.degenerate_wl_mask(wl_pix)
    usable = (n_full_sat == 0) & ~degen
    if not usable.any():
        raise ValueError(
            f"{mode_key}: no usable pixels -- all {wl_pix.size} are fully "
            f"saturated ({int((n_full_sat > 0).sum())}) or wavelength-"
            f"degenerate ({int(degen.sum())}). Reduce ngroup / pick a "
            "fainter-star configuration, or check the worker's wavelength "
            "grid")

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
        # removed-molecule, and Jacobian rows (operator stays linear in depth)
        po = np.argsort(wl_pix)
        flux_model = np.maximum(np.interp(wl_model, wl_pix[po], flux_pix[po]), 0.0)
        # R(lambda) must go in ASCENDING wavelength order: the pandeia pixel
        # grid is dispersion order, not wavelength order, and MIRI LRS ships
        # it DESCENDING (13.86 -> 5.02 um). Passing it raw made the operator's
        # np.interp read a reversed table and return R = R(red end) = 42
        # everywhere, blurring the whole 5-12 um band with a ~5x-too-wide
        # kernel. smooth_to_native_r now refuses an out-of-order curve, so
        # this sort is the contract, not a convenience.
        wl_r, r_curve = wl_pix[po], r_nat[po]
        depth_sm = binning.smooth_to_native_r(wl_model, depth, wl_r, r_curve,
                                              b_lo, b_hi, weight=flux_model)
        # metadata ONLY -- never gate the blur of OTHER vectors on this: a
        # flat baseline is a fixed point of the LSF while a narrow Jacobian
        # feature is not (gating left Jacobians unsmoothed by ~59 ppm)
        lsf_applied = bool(np.any(depth_sm != depth))
        depth = depth_sm
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

    # PandExo guarantees >= 3 in-transit integrations by restructuring the
    # ramp; this worker's ramp is deliberately transit-independent (the noise
    # cache is per star+mode, never per event), so warn loudly instead of
    # silently accepting 1-2 cycles. DELIBERATE, decision recorded as S2-10 in
    # notes.md, Decision records section: the box-depth variance stays valid
    # at 1-2 cycles; the result is NOT re-run with a shortened ramp, and
    # reviews that flag this are re-finding an accepted trade, not a bug.
    warnings = dict(mode_result.get("warnings", {}))
    n_cyc_in = t_in_s / float(mode_result["t_cycle_s"])
    if n_cyc_in < 3.0:
        warnings[f"only {n_cyc_in:.1f} integration cycles fit in transit "
                 "(PandExo enforces >= 3 by shortening the ramp)"] = True
    if _lsf_skip_note:
        warnings[_lsf_skip_note] = True
    # Disclosure, not a bound: since 0.25.0 the ramp search reaches pandeia's
    # permitted minimum (1 group NIR / 2 MIRI), so a very short selected ramp
    # ranks normally but is flagged against the mode's STScI-recommended ramp
    # with the instrument's own reason (thresholds and sources in
    # instruments.py: NGROUP_WARN_REASON + the ngroup_warn_below comment).
    if int(mode_result["ngroup"]) < int(m["ngroup_warn_below"]):
        _reason = ins.NGROUP_WARN_REASON[m["instrument"]]
        warnings[f"ramp uses {int(mode_result['ngroup'])} group(s) per "
                 "integration, below this mode's STScI-recommended ramp "
                 f"({_reason}); verify in APT"] = True
    # A budget-exhausted group search is disclosed, never presented as
    # optimal (worker v10 field; older payloads without it made no such
    # claim, so absence stays silent).
    if mode_result.get("ramp_search_complete") is False:
        warnings["the group search hit its calculation budget; the reported "
                 "ramp is measured-safe but may not be the longest possible "
                 "(costs sensitivity, never validity)"] = True
    # The MIRI floor (2 groups) gets a DISTINCT operational warning: STScI
    # calls 2-5 group MIRI ramps very difficult to calibrate, and a
    # 2026-08-09 review reported (unconfirmed on retrievable jwst-docs
    # pages; see notes.md, Decision records section) that APT treats 2-group FASTR1 as a
    # limited-access configuration -- so the user is told to confirm
    # approval requirements rather than assured either way.
    if (m["instrument"] == "miri"
            and int(mode_result["ngroup"]) == int(m["ngroup_min"])
            and not bool(mode_result.get("saturated", False))):
        warnings["MIRI floor ramp: 2 groups/integration is MIRI's shortest "
                 "permitted ramp and is very difficult to calibrate "
                 "accurately; confirm in APT whether this configuration "
                 "needs special approval before proposing"] = True

    keep = op["keep"]
    return dict(
        jac_bins=jac_bins,
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
        # display/export. Kept with unchanged meaning.
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
