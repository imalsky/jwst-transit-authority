"""Detection-significance math: bin the model per instrument THROUGH THE SAME
count-space measurement operator as the noise, combine with the per-bin depth
uncertainty, and score the science goal.

Model, removed-molecule model, Jacobians, and noise all go through one
operator (binning.build_operator, flux-weighted): the binned model is the
expectation value of the same estimator whose variance the noise module
reports. Pixels Pandeia flags as fully saturated are excluded from the
operator; partially saturated pixels are kept but counted per bin
(n_pix_partial_sat) so affected channels are visible, not silent.

For modes whose final bins approach the NATIVE resolving power (MIRI LRS
R~40-160, NIRSpec PRISM R~30-300, blue SOSS) the model is first blurred to
the instrument's R(lambda) exported by the pandeia worker
(binning.smooth_to_native_r); for high-R gratings this is automatically a
no-op, matching the sub-ppm edge-effect estimate at R_bin=100.

sigma_detect is a CONDITIONAL MATCHED-TEMPLATE S/N at the specified
atmospheric state: the nested-model chi-square distance between the full
spectrum and the spectrum with one molecule's opacity removed, with the
calibration nuisances profiled out --

    chi2 = s^T W s - b^T A^{-1} b,   W = diag(1/sigma^2)  or  C^{-1},
    b = U W s,  A = U W U^T,         s_b = d_full - d_without_X

where W is the diagonal metric under the "random" noise scenario (the
default; the correlated scenarios are experimental) or the inverse of the
full scenario covariance C (noise.build_cov: the floor EXCESS re-allocated
between white and ln-wavelength-smooth parts at identical per-bin totals),
and the rows of U are one constant depth offset PLUS one
step per extra detector segment (NRS1|NRS2 for the two-detector NIRSpec
gratings: real G395H fits universally float such offsets, at the
tens-of-ppm level -- Moran+2023, Madhusudhan+2023) PLUS, under a scenario
that says so, one centered slope per segment (real per-visit fits float
linear trends). The offset/step/slope profiling removes the part of the
molecule's signature a real fit would reabsorb into the continuum or
per-detector calibration. It is NOT a retrieval detection significance:
temperature, clouds, and the other abundances are not re-fit, so it
upper-bounds what a full retrieval would report. When a Fisher Jacobian is
available, ``sigma_detect_proj`` additionally projects the template against
the T-P and lnR0 derivative directions (still conditional -- chemistry and
clouds stay fixed) and is the number to prefer for narrow margins.

Multi-transit extrapolation uses the noise-model components (the random term
scales as 1/N; the minimum floor is a hard lower bound at every N), so
"transits to target" saturates honestly instead of promising 1/sqrt(N)
forever.
"""
from __future__ import annotations

import numpy as np

from . import binning
from . import instruments as ins
from . import noise as noise_mod

# hard cap for the transits-to-target search: beyond this the answer is
# "effectively unreachable" for any real proposal anyway
N_TRANSITS_CAP = 500

# Jacobian rows treated as NUISANCE directions for sigma_detect_proj:
# temperature-structure parameters, the reference radius, and (v16) the
# cloud-deck parameters -- both the analytic power-law deck AND the Mie
# condensate deck. An uncertain deck can absorb broadband signal exactly like
# an offset can, so a molecule score should not lean on it. Chemistry rows
# (lnZ, dlnCO, lnKzz) are the science axes -- projecting them out would eat the
# very signal being scored. Must track forward.TP_PARAM_NAMES +
# forward.CLOUD_FISHER_PARAMS + forward.MIE_FISHER_PARAMS.
_NUISANCE_JAC = frozenset(
    {"Tirr", "Tint", "Tint_cl", "log_kappa", "log_gamma", "lnR0",
     "log_kappa_cloud", "alpha_cloud",
     "mie_log_rg", "mie_sigmag", "mie_log_mmr"})


def _segment_rows(seg: np.ndarray) -> list[np.ndarray]:
    """Indicator rows (one per detector segment beyond the first) for the
    per-segment calibration-offset nuisances. Together with the constant
    offset they span exactly the per-segment offset space."""
    seg = np.asarray(seg, int)
    return [(seg == s).astype(float) for s in range(1, int(seg.max()) + 1)]


def _slope_rows(seg: np.ndarray, wl: np.ndarray) -> list[np.ndarray]:
    """Per-segment linear-in-ln(lambda) rows (unit RMS, centered within the
    segment so the offset rows keep spanning the constants): the slope
    freedom real per-visit fits float. EVERY segment gets one, including the
    first -- the constant offset spans segment means, not segment slopes."""
    seg = np.asarray(seg, int)
    lnl = np.log(np.asarray(wl, float))
    rows = []
    for s in range(int(seg.max()) + 1 if seg.size else 0):
        m = seg == s
        if m.sum() < 3:
            continue
        r = np.where(m, lnl - lnl[m].mean(), 0.0)
        rms = float(np.sqrt(np.mean(r[m] ** 2)))
        if rms > 0:
            rows.append(r / rms)
    return rows


def detection_significance(signal: np.ndarray, sigma: np.ndarray,
                           nuisance: list[np.ndarray] | None = None,
                           marginalize_offset: bool = True,
                           cov: np.ndarray | None = None) -> float:
    """sqrt(Delta chi^2) of a binned signal against noise, with linear
    nuisance directions profiled out (rank-aware).

    ``marginalize_offset=True`` (default) includes a constant depth offset;
    ``nuisance`` adds arbitrary extra rows (detector-segment steps/slopes,
    binned T-P/lnR0 Jacobian rows). The result depends only on the SPAN of
    the nuisance rows, never on their amplitudes: the normal matrix is
    Jacobi-normalized (unit diagonal, correlation form) before the
    rank-revealing eigen-threshold, so rescaling a row by any nonzero factor
    leaves the score unchanged (2026-07-12 external audit: the raw-eigenvalue
    threshold silently dropped down-scaled rows -- confirmed and fixed).
    Directions that are numerically null in the normalized matrix are
    dropped rather than inverted; rows with zero norm in the metric are
    excluded outright.

    ``cov`` (optional): full per-bin depth covariance (noise.build_cov, a
    correlated scenario); when given it REPLACES ``sigma`` in the metric
    (chi2 = s^T C^-1 s, A = U C^-1 U^T). With ``cov=None`` the metric is the
    exact diagonal W = diag(1/sigma^2) fast path -- identical numbers to a
    diagonal C.

    Inputs are validated (2026-07-13 recheck 5.2): a public scientific API
    must reject bad inputs loudly, not return inf (zero sigma) / NaN (NaN
    sigma) or silently broadcast a shape mismatch. ``signal`` must be 1-D and
    finite; ``sigma`` must match it and be finite and > 0 (unused when ``cov``
    is given); each nuisance row must match ``signal``'s length; ``cov`` must
    be a matching square, finite, symmetric, positive-definite matrix (a
    failed Cholesky raises).
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
    if cov is not None:
        C = np.asarray(cov, float)
        if C.shape != (signal.size, signal.size):
            raise ValueError(f"detection_significance: cov shape {C.shape} "
                             f"must be ({signal.size}, {signal.size})")
        if not np.all(np.isfinite(C)):
            raise ValueError("detection_significance: cov has non-finite values")
        if not np.allclose(C, C.T, rtol=1e-8, atol=1e-30):
            raise ValueError("detection_significance: cov is not symmetric")
        try:
            np.linalg.cholesky(C)
        except np.linalg.LinAlgError as e:
            raise ValueError("detection_significance: cov is not "
                             "positive-definite") from e
    else:
        sig = np.asarray(sigma, float)
        if sig.shape != signal.shape or not np.all(np.isfinite(sig)) \
                or np.any(sig <= 0.0):
            raise ValueError("detection_significance: sigma must match signal's "
                             "shape and be finite and > 0")
    # the constant row is included even for a single bin: with a free offset
    # one bin carries NO shape information, so the honest score is 0 -- the
    # old size>1 guard returned a false |s|/sigma "detection" there
    # (2026-07-12 recheck, P2-D)
    rows = [np.ones_like(signal)] if marginalize_offset else []
    rows += [np.asarray(r, float) for r in (nuisance or [])]
    if cov is not None:
        ci_s = np.linalg.solve(np.asarray(cov, float), signal)
        chi2 = float(signal @ ci_s)
        if rows:
            U = np.stack(rows)
            A = U @ np.linalg.solve(np.asarray(cov, float), U.T)
            b = U @ ci_s
    else:
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


# ONE transit-count validator for the stack, defined next to the variance it
# divides (2026-07-28: pixel_depth_variance floored fractional counts while
# these scalers refused them). Replaces the old max(1, int(n_transits)) that
# silently turned 0 / negative / fractional inputs into a 1-transit result
# (2026-07-13 recheck 5.1).
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


def cov_at_transits(result: dict, n_transits: int,
                    floor_only: bool = False) -> np.ndarray | None:
    """The evaluated mode's scenario covariance re-scaled to ``n_transits``
    (the random diagonal scales 1/N; build_cov re-derives the floor EXCESS at
    that diagonal, so diag(C) = max(var_N, floor^2) at every N); None under
    the diagonal random scenario. ``floor_only=True`` gives the
    infinite-transit limit (random term zero, floors clipped away from exact
    zero). NOTE: because the correlated budget is the floor EXCESS, the
    correlated part GROWS with N (absent where photon noise dominates, fully
    present at N -> infinity) -- scores are NOT monotone in N under a
    correlated scenario (see transits_to_target)."""
    scen = result.get("scenario", "random")
    floor = np.asarray(result["floor"])
    if floor_only:
        return noise_mod.build_cov(result["wl"], np.zeros_like(floor),
                                   np.maximum(floor, 1e-30), scen)
    n0 = int(result["n_transits_eval"])
    var = np.asarray(result["var_phot"]) * (n0 / float(_n_transits(n_transits)))
    return noise_mod.build_cov(result["wl"], var, floor, scen)


def _result_nuisance(result: dict) -> list[np.ndarray]:
    """The evaluated mode's profiled calibration rows: per-segment offset
    steps always, plus per-segment slopes when its scenario says so."""
    rows = _segment_rows(result["seg"]) if "seg" in result else []
    slope = result.get("slope_rows")
    if slope is not None and np.asarray(slope).size:
        rows += list(np.asarray(slope, float))
    return rows


def transits_to_target(result: dict, target_sig: float) -> dict:
    """Smallest transit count reaching ``target_sig`` for the detect goal.

    Returns dict(n=int|None, n_last=int|None, reachable=bool, sig_inf=float).
    ``sig_inf`` is the INFINITE-TRANSIT LIMIT of the mode's scenario noise
    model (floor-only). Under the default diagonal "random" scenario the
    score is monotone in N, sig_inf is an exact ceiling, and a target above
    it short-circuits to unreachable. Under a correlated scenario the
    systematic is the floor EXCESS (noise.build_cov), which GROWS as the
    photon term averages down: the score can PEAK at a finite N and then
    decline toward sig_inf, so sig_inf is a limit, NOT a bound, and
    reachability comes from the full n = 1..N_TRANSITS_CAP scan (2026-07-15
    audit: a smooth-bump signal beat its target only for n = 5..13 while
    sig_inf sat below it -- the old sig_inf gate returned a false "never").
    With NO floor set anywhere, sigma averages down without bound and
    ``sig_inf`` is ``inf`` (the scan is monotone -- no floor means no floor
    excess, so no correlated term exists at any finite N); unreachability then
    means the target needs more than N_TRANSITS_CAP transits, not that a
    systematic caps it.
    ``n`` is the smallest count meeting the target; ``n_last`` (correlated
    scenarios only, else None) is the largest scanned count still meeting it
    -- a finite window means over-observing past ``n_last`` loses the
    detection again. The mode's scenario (covariance + segment
    offsets/slopes) stays in force at every transit count.
    """
    if result.get("depth_wo") is None:
        return dict(n=None, n_last=None, reachable=False, sig_inf=float("nan"))
    signal = np.asarray(result["depth"]) - np.asarray(result["depth_wo"])
    nuis = _result_nuisance(result)
    floor = np.asarray(result["floor"])
    if not np.any(floor > 0.0):
        # No floor anywhere: sigma -> 0 as N -> infinity, so the limit is
        # genuinely INFINITE and there is nothing to cap the score. Report inf
        # rather than the ~1e26 that the 1e-30 clip below produces, and treat
        # the scan as monotone: with no floor there is no floor EXCESS, so
        # build_cov returns None at every finite N (verified) and the score
        # rises monotonically -- taking the correlated branch here reported a
        # meaningless "window: lost again past N_TRANSITS_CAP" (2026-07-28).
        sig_inf, cov_inf, diagonal = float("inf"), None, True
    else:
        cov_inf = cov_at_transits(result, 1, floor_only=True)
        sig_inf = detection_significance(signal, np.maximum(floor, 1e-30),
                                         nuisance=nuis, cov=cov_inf)
        diagonal = cov_inf is None    # random scenario: monotone in N
    if diagonal and target_sig > sig_inf:
        return dict(n=None, n_last=None, reachable=False, sig_inf=sig_inf)
    n_first = n_last = None
    for n in range(1, N_TRANSITS_CAP + 1):
        ok = detection_significance(signal, sigma_at_transits(result, n),
                                    nuisance=nuis,
                                    cov=cov_at_transits(result, n)) >= target_sig
        if ok:
            if n_first is None:
                n_first = n
                if diagonal:          # monotone: smallest n is the answer
                    return dict(n=n, n_last=None, reachable=True,
                                sig_inf=sig_inf)
            n_last = n
    return dict(n=n_first, n_last=n_last, reachable=n_first is not None,
                sig_inf=sig_inf)


# Native-grid pixel-census keys the worker exports from v7 on. They are counted
# BEFORE the worker's finite/positive `good` filter, which is the only place a
# fully saturated channel can still be seen: pandeia returns non-finite extracted
# noise exactly there, so a post-filter count of full saturation reads low, and
# reads ZERO for a mode that saturated everywhere.
_NATIVE_PIXEL_KEYS = (
    "n_pix_native",
    "n_pix_unusable_dropped",
    "n_pix_part_sat_native",
    "n_pix_full_sat_native",
)


def _native_pixel_counts(mode_result: dict) -> dict:
    """Pass the worker's native-grid pixel census through, or None per key.

    None means UNMEASURED (a pre-v7 payload), never zero and never the
    post-filter count -- substituting either would report a saturated column as
    clean. Same convention as `fd_err` being NaN rather than 0.0 for an
    unmeasured finite-difference error (2026-07-19 audit response).
    """
    return {k: (int(mode_result[k]) if mode_result.get(k) is not None else None)
            for k in _NATIVE_PIXEL_KEYS}


def evaluate_mode(mode_key: str, mode_result: dict, model: dict, target_mol,
                  R_bin: float, t_in_s: float, t_out_s: float, n_transits: int,
                  floor_spec, noise_inflation: float = 1.0,
                  scenario: str = "random") -> dict:
    """One instrument mode -> binned model, sigmas, conditional template S/N.

    Bins cover the intersection of the mode's science band, the model's coverage,
    and the pixels pandeia actually returned. Model, Jacobians, and noise are all
    binned through ONE count-space operator (module docstring). ``target_mol=None``
    (the parameter-constraint science goal) skips the molecule-removed comparison:
    ``sigma_detect`` comes back NaN and ``depth_wo`` None.

    ``scenario`` names a noise.SCENARIOS entry ("random" is the default and
    the headline configuration; the correlated presets are EXPERIMENTAL): it
    sets the floor excess's correlation structure (noise.build_cov) and
    whether per-segment slopes join the profiled nuisances, for
    sigma_detect/sigma_detect_proj and everything downstream (Fisher,
    transits-to-target read the stored ``cov``/``slope_rows``).
    ``sigma_detect_by_scenario`` reports the score under EVERY scenario so
    mode rankings can be compared across assumptions -- per-bin total
    variance is scenario-invariant by construction, so those differences are
    purely correlation structure.
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
    # PER-TARGET emission reliability (v20): if THIS target's removed-molecule
    # emission spectrum went optically thin at the RT column bottom (the forward
    # model records it as emis_tau_bottom_min_wo, no longer whole-run-refuses),
    # its full-minus-removed eclipse contrast is overstated (ArtEmisPure has no
    # surface/interior source at the grid bottom). Refuse THIS target's
    # detection; the spectrum, parameter constraints, and other molecules stay
    # usable. Only in emission mode; transmission has no such term.
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
    # channel-level saturation (worker_version >= 3): fully saturated pixels are
    # excluded from the estimator; partially saturated ones kept but counted.
    n_full_sat = np.asarray(mode_result.get("n_full_sat", np.zeros(wl_pix.size)))
    n_part_sat = np.asarray(mode_result.get("n_part_sat", np.zeros(wl_pix.size)))
    # degenerate-wavelength pixels (pandeia grid artifacts, e.g. the G395H
    # red-edge pileup) claim spectral information that does not exist -- drop
    # them and report the count (binning.DEGENERATE_WL_FRAC rationale).
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

    # blur the model to the instrument's native R(lambda) where the worker
    # exported it (worker_version >= 4); a no-op for high-R modes. Pixel cells
    # extend at most one native pixel past the bin span, hence the margin.
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
        # stellar flux on the model grid weights the LSF so it forms the
        # observed count ratio L[F d]/L[F], not the flat depth blur L[d]
        # (re-audit item 2); same weight for depth, removed-molecule, and each
        # Jacobian row, so the blurred operator stays linear in the depth.
        po = np.argsort(wl_pix)
        flux_model = np.maximum(np.interp(wl_model, wl_pix[po], flux_pix[po]), 0.0)
        depth_sm = binning.smooth_to_native_r(wl_model, depth, wl_pix, r_nat,
                                              b_lo, b_hi, weight=flux_model)
        # metadata ONLY: whether the blur changed the baseline. It must never
        # gate the operator on OTHER vectors -- a flat baseline is a fixed
        # point of any normalized LSF while a narrow Jacobian feature is not,
        # so the old "and lsf_applied" guard left Jacobians unsmoothed
        # exactly when the baseline happened to be LSF-invariant (2026-07-12
        # recheck, P1-C: 58.6 ppm on a flat-baseline reproducer). The
        # smoother is its own no-op when the kernel is unresolved.
        lsf_applied = bool(np.any(depth_sm != depth))
        depth = depth_sm
        if depth_wo is not None:
            depth_wo = binning.smooth_to_native_r(wl_model, depth_wo, wl_pix,
                                                  r_nat, b_lo, b_hi,
                                                  weight=flux_model)
        if jac_rows is not None:
            jac_rows = [binning.smooth_to_native_r(wl_model, row, wl_pix,
                                                   r_nat, b_lo, b_hi,
                                                   weight=flux_model)
                        for row in jac_rows]
    else:
        # Fail loud, never silently: a missing native-R means the depth,
        # removed-molecule depth, and Jacobians go UNBLURRED. That is only safe
        # for high-R modes -- every shipped low-R mode (PRISM, LRS, SOSS) ships
        # a dispersion file, so a None here on such a mode signals a refdata or
        # config problem, not a no-op (standing loud-error rule). Surfaced via
        # the result's own warning channel below (same as the <3-cycle notice).
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

    # the selected noise scenario: floor correlation structure + whether
    # per-segment slopes are profiled (unknown names raise via SCENARIOS)
    sc = noise_mod.SCENARIOS[scenario]
    slopes = _slope_rows(seg, nz["wl_center"]) if sc["slopes"] else []
    cov = noise_mod.build_cov(nz["wl_center"], nz["var_phot"], nz["floor"],
                              scenario)

    d_full_b = binning.bin_model(op, wl_model, depth)
    jac_bins = None
    jac_names = ([str(x) for x in model["jac_names"]]
                 if "jac_names" in model else [])
    if jac_rows is not None:
        jac_bins = np.stack([binning.bin_model(op, wl_model, row)
                             for row in jac_rows])
    sigma_detect_by_scenario = {}
    if depth_wo is not None:
        d_wo_b = binning.bin_model(op, wl_model, depth_wo)
        s_b = d_full_b - d_wo_b
        # the same template scored under EVERY scenario (cheap: <=few-hundred
        # bins) so the GUI can show how rankings move with the assumptions
        for name, sc_i in noise_mod.SCENARIOS.items():
            nuis_i = steps + (_slope_rows(seg, nz["wl_center"])
                              if sc_i["slopes"] else [])
            cov_i = (cov if name == scenario else
                     noise_mod.build_cov(nz["wl_center"], nz["var_phot"],
                                         nz["floor"], name))
            sigma_detect_by_scenario[name] = detection_significance(
                s_b, nz["sigma"], nuisance=nuis_i, cov=cov_i)
        sigma_detect = sigma_detect_by_scenario[scenario]
        # nuisance-projected variant: also profile the T-P + lnR0 Jacobian
        # directions (chemistry/clouds stay fixed -- still conditional)
        sigma_detect_proj = float("nan")
        if jac_bins is not None and jac_names:
            nuis = steps + slopes + [jac_bins[i] for i, n in enumerate(jac_names)
                                     if n in _NUISANCE_JAC]
            sigma_detect_proj = detection_significance(s_b, nz["sigma"],
                                                       nuisance=nuis, cov=cov)
    else:
        d_wo_b, sigma_detect, sigma_detect_proj = None, float("nan"), float("nan")

    # current PandExo (master jwst.py min_nint_trans=3) restructures the ramp
    # to guarantee >= 3 in-transit integrations; this tool's worker ramp is
    # deliberately transit-independent (one noise cache per star serves any
    # transit), so say so loudly instead of silently accepting 1-2 cycles:
    # the box-depth variance stays valid, but time resolution and PandExo
    # ramp comparability degrade on short transits with long saturated ramps.
    warnings = dict(mode_result.get("warnings", {}))
    n_cyc_in = t_in_s / float(mode_result["t_cycle_s"])
    if n_cyc_in < 3.0:
        warnings[f"only {n_cyc_in:.1f} integration cycles fit in transit "
                 "(PandExo enforces >= 3 by shortening the ramp)"] = True
    if _lsf_skip_note:
        warnings[_lsf_skip_note] = True

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
        scenario=scenario, cov=cov,
        slope_rows=(np.stack(slopes) if slopes
                    else np.zeros((0, nz["sigma"].size))),
        sigma_detect_by_scenario=sigma_detect_by_scenario,
        sigma_detect=sigma_detect, sigma_detect_proj=sigma_detect_proj,
        median_sigma_ppm=float(np.median(nz["sigma"]) * 1e6),
        n_bins=int(nz["wl_center"].size),
        n_pix_partial_sat=binning.bin_counts(op, n_part_sat > 0).astype(int),
        # POST-FILTER count: full-sat pixels among the ones the worker returned.
        # It cannot see channels the worker already dropped as non-finite, which
        # is precisely the fully saturated ones -- use the *_native fields below
        # for any display or export. Kept (unchanged meaning) so nothing that
        # reads it silently changes definition.
        n_pix_full_sat_dropped=int(np.sum(n_full_sat > 0)),
        n_pix_degenerate_dropped=int(degen.sum()),
        # NATIVE-GRID truth from the worker (v7+). None = "not measured" for a
        # payload predating v7; never substitute the post-filter count, which
        # would under-report saturation as if it had been counted.
        **_native_pixel_counts(mode_result),
        ngroup=int(mode_result["ngroup"]),
        sat_frac=float(mode_result["sat_frac"]),
        sat_ngroups=mode_result.get("sat_ngroups"),
        saturated=bool(mode_result.get("saturated", False)),
        t_cycle_s=float(mode_result["t_cycle_s"]),
        warnings=warnings,
    )
