"""One measurement operator per instrument mode.

Method and contract: docs/noise_and_binning.md, section `binning`.
"""

from __future__ import annotations

import numpy as np

# A pixel's cell half-width toward a neighbor is half that gap, capped at
# GAP_CAP x the smaller adjacent gap (matters at detector gaps and band ends,
# where a raw midpoint cell would smear the model).
GAP_CAP = 1.5

# A gap larger than this factor x the median pixel spacing marks a
# DETECTOR-SEGMENT boundary (NRS1|NRS2 gap ~150x median; real dispersion
# gradients vary smoothly by factors of a few).
SEGMENT_GAP_FACTOR = 20.0

# Local spacing below this fraction of the median = a DEGENERATE wavelength
# solution (e.g. the G395H red-edge pileup). Counting such pixels as
# independent samples overstates the information (~sqrt(n) too-small sigma),
# so detect.evaluate_mode excludes them from the operator.
DEGENERATE_WL_FRAC = 0.02

# The LSF blur's numerics (smooth_to_native_r). Gaussian FWHM per sigma, 2 sqrt(2 ln 2) to five figures: the per-mode
# instruments.LSF_WIDTH fits were made with this value.
FWHM_PER_SIGMA = 2.3548
# Smallest resolving power a kernel is built for: a native-R entry below it is raised to it. No basis recorded.
LSF_R_FLOOR = 5.0
# The working grid reaches this many widest-kernel sigmas past each band edge, so the edge blur sees model content.
LSF_PAD_SIGMAS = 5.0
# Working-grid samples per narrowest-kernel sigma.
LSF_SAMPLES_PER_SIGMA = 6.0
# Kernel half-width in widest-kernel sigmas: the weight beyond it is below exp(-8), 3e-4.
LSF_HALF_WIDTH_SIGMAS = 4.0
# Relative floor on the cell-averaged stellar-flux weight, so a window of zero flux does not divide by zero.
LSF_WEIGHT_FLOOR = 1.0e-12


def _validate_wl(wl: np.ndarray, name: str) -> np.ndarray:
    """Finite, positive, non-empty 1-D wavelength array or ValueError.

    Invalid samples are never silently dropped (a NaN would poison every gap statistic). Duplicates are NOT rejected
    here: degenerate_wl_mask masks them by design. Returns float array.
    """
    wl = np.asarray(wl, float)
    if wl.ndim != 1 or wl.size == 0:
        raise ValueError(f"{name}: wavelength must be a non-empty 1-D array")
    bad = ~np.isfinite(wl)
    if bad.any():
        raise ValueError(
            f"{name}: {int(bad.sum())}/{wl.size} non-finite wavelength(s) "
            f"(first at index {int(np.argmax(bad))}) -- fix the upstream "
            "grid; invalid samples are never silently dropped"
        )
    if np.any(wl <= 0.0):
        raise ValueError(f"{name}: non-positive wavelength(s)")
    return wl


def _positive_gap_median(gaps: np.ndarray) -> float:
    """Median pixel spacing over STRICTLY POSITIVE gaps.

    Zero-width duplicate gaps would collapse the median to 0 and silently disable both the degenerate mask and the
    segment splitter. Returns 0.0 when every gap is zero.
    """
    pos = gaps[gaps > 0.0]
    return float(np.median(pos)) if pos.size else 0.0


def degenerate_wl_mask(wl_pix: np.ndarray) -> np.ndarray:
    """Return true for pixels on a degenerate wavelength solution, in INPUT order.

    Exact duplicates are always degenerate (zero spectral support); an all-duplicate grid returns all-True.
    """
    wl_pix = _validate_wl(wl_pix, "degenerate_wl_mask: wl_pix")
    order = np.argsort(wl_pix)
    wl_s = wl_pix[order]
    if wl_s.size < 3:
        return np.zeros(wl_pix.size, bool)
    gaps = np.diff(wl_s)
    med = _positive_gap_median(gaps)
    if med == 0.0:  # every pixel shares one wavelength
        return np.ones(wl_pix.size, bool)
    local = np.minimum(np.concatenate([[gaps[0]], gaps]), np.concatenate([gaps, [gaps[-1]]]))
    bad_sorted = local < DEGENERATE_WL_FRAC * med
    bad = np.zeros(wl_pix.size, bool)
    bad[order] = bad_sorted
    return bad


def segment_ids(wl_pix: np.ndarray) -> np.ndarray:
    """Detector-segment id (0, 1, ...) per pixel, in the INPUT pixel order.

    Segments split at gaps larger than SEGMENT_GAP_FACTOR x the median pixel
    spacing (the NRS1|NRS2 split; one segment for every other mode). Each
    segment gets its own calibration-offset nuisance downstream.
    """
    wl_pix = _validate_wl(wl_pix, "segment_ids: wl_pix")
    order = np.argsort(wl_pix)
    if wl_pix.size < 2:
        return np.zeros(wl_pix.size, int)
    gaps = np.diff(wl_pix[order])
    med = _positive_gap_median(gaps)
    if med == 0.0:  # all-duplicate grid: one segment
        return np.zeros(wl_pix.size, int)
    split = gaps > SEGMENT_GAP_FACTOR * med
    seg_sorted = np.concatenate([[0], np.cumsum(split)])
    seg = np.empty(wl_pix.size, int)
    seg[order] = seg_sorted
    return seg


def bin_segments(op: dict, seg_pix: np.ndarray) -> np.ndarray:
    """Per-kept-bin segment id: the segment holding the bin's count weight.

    Bins never straddle a segment gap in practice (gap >> bin width); a bin that did would go to its count-majority
    segment.
    """
    seg = np.asarray(seg_pix)[op["pix_idx"]].astype(int)
    n_bins = op["wl_center"].size
    n_seg = int(seg.max()) + 1
    w = np.zeros((n_seg, n_bins))
    np.add.at(w, (seg, op["pix_bin"]), op["pix_w"])
    return np.argmax(w, axis=0)


def _validate_native_r(
    wl: np.ndarray, wl_r: np.ndarray, r_curve: np.ndarray, weight: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Refuse a mis-ordered or invalid native-R table and flux weight, before any early return of the blur."""
    # Validate BEFORE any early return, so a mis-ordered curve raises on
    # every mode, not only where the blur is active. Never silently repair
    # the caller's ordering: sorting here would hide a wl_r/r_curve pairing
    # bug.
    wl_r_v = _validate_wl(wl_r, "smooth_to_native_r: wl_r")
    r_v = np.asarray(r_curve, float)
    if r_v.shape != wl_r_v.shape:
        raise ValueError(
            f"smooth_to_native_r: r_curve shape {r_v.shape} != wl_r shape "
            f"{wl_r_v.shape} -- the native-R table must be one R per "
            "wavelength"
        )
    if not np.all(np.isfinite(r_v)) or np.any(r_v <= 0.0):
        raise ValueError(
            "smooth_to_native_r: r_curve (native resolving power) must be "
            f"finite and > 0 -- got {int((~np.isfinite(r_v)).sum())} "
            f"non-finite / {int((r_v <= 0.0).sum())} non-positive value(s)"
        )
    if wl_r_v.size > 1 and np.any(np.diff(wl_r_v) <= 0.0):
        _bad = int(np.argmax(np.diff(wl_r_v) <= 0.0))
        raise ValueError(
            "smooth_to_native_r: wl_r (the native-R wavelength table) must be "
            f"STRICTLY ASCENDING -- first violation at index {_bad} "
            f"({wl_r_v[_bad]:g} -> {wl_r_v[_bad + 1]:g}). The kernel width is "
            "read with np.interp, which requires increasing sample points and "
            "silently returns the end value everywhere on a descending table "
            "(MIRI LRS ships its pandeia grid in dispersion order, 13.86 -> "
            "5.02 um). Sort wl_r and r_curve together before calling."
        )
    # Validate the weight BEFORE any early return: the no-op paths must still
    # reject a NaN/negative stellar flux, or the same bad input raises or
    # passes depending on the mode.
    if weight is not None:
        wv = np.asarray(weight, float)
        if wv.shape != wl.shape or not np.all(np.isfinite(wv)) or np.any(wv < 0.0):
            raise ValueError(
                "smooth_to_native_r: weight (stellar flux on the model grid) "
                "must match wl_model's shape and be finite and >= 0 -- a NaN "
                "weight silently corrupts the count-ratio blur"
            )
    return wl_r_v, r_v


def _validate_operator_inputs(
    wl_pix: np.ndarray, w_pix: np.ndarray, edges: np.ndarray, valid: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the pixel grid, weights and bin edges as arrays, refusing invalid ones; nothing is ever sanitized."""
    wl_pix = _validate_wl(wl_pix, "build_operator: wl_pix")
    w_pix = np.asarray(w_pix, float)
    if w_pix.shape != wl_pix.shape:
        raise ValueError(f"build_operator: w_pix shape {w_pix.shape} != wl_pix shape {wl_pix.shape}")
    if not np.all(np.isfinite(w_pix)) or np.any(w_pix < 0.0):
        raise ValueError(
            "build_operator: pixel weights (extracted count rates) must be "
            "finite and >= 0 -- got "
            f"{int((~np.isfinite(w_pix)).sum())} non-finite / "
            f"{int((w_pix < 0.0).sum())} negative value(s); fix the worker "
            "output, invalid weights are never silently dropped"
        )
    edges = np.asarray(edges, float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError(
            "build_operator: bin edges must be a 1-D, finite, "
            f"strictly ascending array of >= 2 values, got "
            f"shape {edges.shape}"
        )
    if valid is not None and np.asarray(valid).shape != wl_pix.shape:
        raise ValueError(f"build_operator: valid mask shape {np.asarray(valid).shape} != wl_pix shape {wl_pix.shape}")
    return wl_pix, w_pix, edges


def smooth_to_native_r(
    wl_model: np.ndarray,
    y: np.ndarray,
    wl_r: np.ndarray,
    r_curve: np.ndarray,
    band_lo: float,
    band_hi: float,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Blur a depth model to the Gaussian LSF of R(lambda) over [band_lo, band_hi].

    Where the model grid does not resolve the kernel, the input is returned unchanged. The instrument measures
    LSF-averaged counts, so d_obs = L[F d] / L[F]; the operator is linear in d for fixed F, so a Jacobian row blurs
    with the same weight as the depth.

    Parameters
    ----------
    wl_model : ndarray [n_model]
        Model wavelengths, um, strictly ascending.
    y : ndarray [n_model]
        Depth or Jacobian row on ``wl_model``.
    wl_r : ndarray [n_r]
        Wavelengths of the native-R table, um, strictly ascending: the pandeia pixel grid is in dispersion order, and
        ``np.interp`` on a descending table silently returns its last entry, so sort it first.
    r_curve : ndarray [n_r]
        Resolving power on ``wl_r``.
    band_lo : float
        Short edge of the band the blur is evaluated over, um.
    band_hi : float
        Long edge of that band, um.
    weight : ndarray [n_model] or None
        Stellar flux on ``wl_model``; None or a constant is the flat blur L[d].

    Returns
    -------
    ndarray [n_model]
        A copy of ``y``, blurred inside the band.

    Raises
    ------
    ValueError
        A grid, table or weight is invalid, or no native-R sample lies inside the band.
    """
    wl, yv = _validate_model(wl_model, y, "smooth_to_native_r")
    wl_r_v, r_v = _validate_native_r(wl, wl_r, r_curve, weight)
    lnw = np.log(wl)
    x_lo, x_hi = np.log(band_lo), np.log(band_hi)
    in_band = (wl_r_v >= band_lo) & (wl_r_v <= band_hi)
    if not in_band.any():
        raise ValueError(f"smooth_to_native_r: no native-R sample inside the band [{band_lo:g}, {band_hi:g}] um")
    r_band = r_v[in_band]
    r_min = max(LSF_R_FLOOR, float(np.min(r_band)))
    r_max = max(r_min, float(np.max(r_band)))
    s_max = 1.0 / (FWHM_PER_SIGMA * r_min)  # widest kernel sigma, in ln-lambda
    s_min = 1.0 / (FWHM_PER_SIGMA * r_max)

    sel = (lnw >= x_lo) & (lnw <= x_hi)
    if not sel.any():
        return yv.copy()
    d_model = float(np.median(np.diff(lnw[sel]))) if sel.sum() > 2 else np.inf
    if s_max < d_model:  # kernel unresolved by the model grid
        return yv.copy()

    lo = max(float(lnw[0]), x_lo - LSF_PAD_SIGMAS * s_max)
    hi = min(float(lnw[-1]), x_hi + LSF_PAD_SIGMAS * s_max)
    dl = s_min / LSF_SAMPLES_PER_SIGMA
    n = int(np.ceil((hi - lo) / dl)) + 1
    grid = lo + dl * np.arange(n)

    # Exact piecewise-quadratic antideriv of the piecewise-linear model. Never
    # np.interp a cumulative-trapezoid integral to cell edges: only O(h^2),
    # ~1-2 ppm off.
    icum = _pl_cumint(lnw, yv)
    edges = np.concatenate([[grid[0] - 0.5 * dl], grid + 0.5 * dl])
    edges = np.clip(edges, lnw[0], lnw[-1])
    ic = _pl_antideriv(edges, lnw, yv, icum)
    w_raw = np.diff(edges)
    widths = np.maximum(w_raw, 1e-300)
    yg = np.diff(ic) / widths
    # Zero-width END cells (edge clip past the model span) hold NO model
    # content: constant-extend from the nearest cell WITH content, or the
    # spurious zeros fill the pad and drag the band-edge blur low.
    _valid = np.flatnonzero(w_raw > 0.0)
    if _valid.size == 0:  # cannot happen for a validated band, be loud
        raise ValueError(
            "smooth_to_native_r: every working-grid cell collapsed to zero "
            "width after clipping to the model span -- the model grid does "
            "not overlap the requested band"
        )
    if _valid[0] > 0:
        yg[: _valid[0]] = yg[_valid[0]]
    if _valid[-1] < yg.size - 1:
        yg[_valid[-1] + 1 :] = yg[_valid[-1]]

    # Stellar-flux weight cell-averaged the same way, so the convolution forms
    # L[F d]/L[F], not the flat L[d]. A tiny positive floor keeps an all-zero-
    # flux window from dividing by zero (falls back to flat weighting there).
    if weight is None:
        f_grid = np.ones_like(yg)
    else:
        wv = np.asarray(weight, float)  # already validated above
        icf = _pl_antideriv(edges, lnw, wv, _pl_cumint(lnw, wv))
        f_grid = np.maximum(np.diff(icf) / widths, 0.0)
        # zero-width end cells: constant-extend the flux weight exactly like yg
        if _valid[0] > 0:
            f_grid[: _valid[0]] = f_grid[_valid[0]]
        if _valid[-1] < f_grid.size - 1:
            f_grid[_valid[-1] + 1 :] = f_grid[_valid[-1]]
        fmax = float(f_grid.max()) if f_grid.size else 0.0
        f_grid = np.maximum(f_grid, LSF_WEIGHT_FLOOR * fmax) if fmax > 0.0 else np.ones_like(yg)

    k = int(np.ceil(LSF_HALF_WIDTH_SIGMAS * s_max / dl))
    pad = np.concatenate([np.full(k, yg[0]), yg, np.full(k, yg[-1])])
    pad_f = np.concatenate([np.full(k, f_grid[0]), f_grid, np.full(k, f_grid[-1])])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1)
    win_f = np.lib.stride_tricks.sliding_window_view(pad_f, 2 * k + 1)
    sig = 1.0 / (FWHM_PER_SIGMA * np.maximum(np.interp(np.exp(grid), wl_r_v, r_v), LSF_R_FLOOR))
    off = dl * (np.arange(2 * k + 1) - k)
    w = np.exp(-0.5 * (off[None, :] / sig[:, None]) ** 2) * win_f  # flux-weighted
    smoothed = (w * win).sum(axis=1) / w.sum(axis=1)

    out = yv.copy()
    out[sel] = np.interp(lnw[sel], grid, smoothed)
    return out


def _pixel_cells(wl_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cell_lo, cell_hi) per sorted pixel: midpoint edges, gap-capped."""
    n = wl_sorted.size
    if n == 1:
        return wl_sorted.copy(), wl_sorted.copy()
    gaps = np.diff(wl_sorted)
    g_lo = np.concatenate([[gaps[0]], gaps])  # gap toward the previous pixel
    g_hi = np.concatenate([gaps, [gaps[-1]]])  # gap toward the next pixel
    g_min = np.minimum(g_lo, g_hi)
    lo = wl_sorted - np.minimum(0.5 * g_lo, GAP_CAP * g_min)
    hi = wl_sorted + np.minimum(0.5 * g_hi, GAP_CAP * g_min)
    return lo, hi


def build_operator(
    wl_pix: np.ndarray,
    w_pix: np.ndarray,
    edges: np.ndarray,
    wl_lo: float = -np.inf,
    wl_hi: float = np.inf,
    valid: np.ndarray | None = None,
) -> dict:
    """Build the count-space measurement operator for one mode.

    A pixel belongs to the bin containing its center wavelength (indivisible extracted samples, as real reductions
    group them); the model is integrated over the pixel's full cell. Exact-duplicate pixels have zero-width cells and
    drop out here; detect.evaluate_mode masks them via degenerate_wl_mask before building the operator.

    Parameters
    ----------
    wl_pix : ndarray [n_pix]
        Pandeia pixel wavelengths, um, in any order.
    w_pix : ndarray [n_pix]
        Extracted stellar count rates, e-/s: finite and >= 0; zero-weight pixels are excluded.
    edges : ndarray [n_bins + 1]
        Final bin edges, um, finite and strictly ascending.
    wl_lo : float
        Short edge of the model wavelength span, um. Pixel cells are clipped to the span, and pixels outside it are
        dropped from both model and noise, so the two stay the same estimator.
    wl_hi : float
        Long edge of that span, um.
    valid : ndarray [n_pix] of bool or None
        Pixels allowed into the operator (saturation exclusion).

    Returns
    -------
    dict
        ``keep`` [n_bins] bins with at least one usable pixel; ``wl_center`` [n_keep] edge midpoints of the kept
        bins, um; ``n_pix`` [n_keep] usable pixels per kept bin; ``pix_idx`` [n_use] indices into the input pixel
        arrays; ``pix_w``, ``cell_lo``, ``cell_hi`` [n_use] weights and cell edges, um; ``pix_bin`` [n_use] index
        into the kept bins.

    Raises
    ------
    ValueError
        An input is invalid (never sanitized), or no usable pixel is left; the message breaks the exclusions down by
        criterion.
    """
    wl_pix, w_pix, edges = _validate_operator_inputs(wl_pix, w_pix, edges, valid)
    order = np.argsort(wl_pix)
    wl_s = wl_pix[order]
    lo_s, hi_s = _pixel_cells(wl_s)
    lo_s = np.maximum(lo_s, wl_lo)
    hi_s = np.minimum(hi_s, wl_hi)

    ok = (hi_s > lo_s) & (w_pix[order] > 0)
    if valid is not None:
        ok &= np.asarray(valid, bool)[order]
    bin_raw = np.digitize(wl_s, edges) - 1
    nb = len(edges) - 1
    ok &= (bin_raw >= 0) & (bin_raw < nb)
    if not ok.any():
        raise ValueError(
            "build_operator: no usable pixel survives -- of "
            f"{wl_pix.size} input pixels, "
            f"{int((hi_s <= lo_s).sum())} have zero-width cells after "
            f"clipping to the model span [{wl_lo:g}, {wl_hi:g}] (duplicate "
            "wavelengths or no model overlap), "
            f"{int((w_pix[order] <= 0).sum())} have zero weight, "
            f"{int((~np.asarray(valid, bool)[order]).sum()) if valid is not None else 0} "
            "are masked invalid, and the rest fall outside the bin edges "
            f"[{edges[0]:g}, {edges[-1]:g}]. Check the mode/model wavelength "
            "overlap and the requested binning"
        )

    idx = order[ok]
    bins = bin_raw[ok]
    keep = np.zeros(nb, dtype=bool)
    keep[bins] = True
    remap = np.cumsum(keep) - 1  # bin id -> kept-bin id
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_pix = np.bincount(remap[bins], minlength=int(keep.sum()))
    return dict(
        keep=keep,
        wl_center=centers[keep],
        n_pix=n_pix.astype(int),
        pix_idx=idx,
        pix_w=w_pix[idx],
        cell_lo=lo_s[ok],
        cell_hi=hi_s[ok],
        pix_bin=remap[bins],
    )


def _wsum(op: dict, values_per_pixel: np.ndarray) -> np.ndarray:
    out = np.zeros(op["wl_center"].size)
    np.add.at(out, op["pix_bin"], values_per_pixel)
    return out


def bin_values(op: dict, v_pix: np.ndarray) -> np.ndarray:
    """Count-weighted mean of a per-pixel quantity, per kept bin.

    ``v_pix`` is full-length (aligned with the arrays given to build_operator).
    """
    v = np.asarray(v_pix, float)[op["pix_idx"]]
    return _wsum(op, op["pix_w"] * v) / _wsum(op, op["pix_w"])


def bin_variance(op: dict, var_pix: np.ndarray) -> np.ndarray:
    """Variance of the count-weighted bin estimator: sum(w^2 Var)/(sum w)^2."""
    v = np.asarray(var_pix, float)[op["pix_idx"]]
    return _wsum(op, op["pix_w"] ** 2 * v) / _wsum(op, op["pix_w"]) ** 2


def _validate_model(wl: np.ndarray, y: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Strictly ascending finite wavelength grid + finite values, or ValueError: a NaN would reach every bin."""
    wl = _validate_wl(wl, f"{name}: wl_model")
    if wl.size < 2 or np.any(np.diff(wl) <= 0.0):
        raise ValueError(
            f"{name}: model wavelength grid must be strictly "
            "ascending with >= 2 points (duplicates or "
            "descending runs are upstream errors)"
        )
    y = np.asarray(y, float)
    if y.shape != wl.shape:
        raise ValueError(f"{name}: model value shape {y.shape} != wavelength shape {wl.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name}: {int((~np.isfinite(y)).sum())} non-finite model value(s) -- fix the upstream model")
    return wl, y


def _pl_cumint(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return running integral of the piecewise-linear (x, y) AT the nodes: icum[k] = int_{x[0]}^{x[k]}.

    Trapezoid is exact for a piecewise-linear integrand; sub-node evaluation is _pl_antideriv.
    """
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def _pl_antideriv(xq: np.ndarray, x: np.ndarray, y: np.ndarray, icum: np.ndarray) -> np.ndarray:
    """EXACT antiderivative of the piecewise-linear (x, y) at query points xq, given node integrals from _pl_cumint.

    The antiderivative is piecewise-QUADRATIC: never np.interp ``icum`` to
    cell edges -- only O(h^2), ~1-2 ppm off for edges between model nodes.
    For x in interval k: I(x) = icum[k] + y_k (x - x_k)
    + 0.5 slope_k (x - x_k)^2. Queries are clamped to the model span;
    ``x`` must be strictly ascending.
    """
    xq = np.asarray(xq, float)
    xc = np.clip(xq, x[0], x[-1])
    k = np.clip(np.searchsorted(x, xc, side="right") - 1, 0, x.size - 2)
    dx = xc - x[k]
    slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
    return icum[k] + y[k] * dx + 0.5 * slope * dx * dx


def bin_model(op: dict, wl_model: np.ndarray, y_model: np.ndarray) -> np.ndarray:
    """Bin a native model through the operator.

    Exact cell average of the piecewise-linear model per pixel, then the count-weighted bin mean. Linear in
    ``y_model``, so the binned Jacobian is the operator applied to each row; exact for a constant model and for cell
    edges between model nodes.

    Parameters
    ----------
    op : dict
        The operator from build_operator.
    wl_model : ndarray [n_model]
        Model wavelengths, um, strictly ascending and spanning every pixel cell.
    y_model : ndarray [n_model]
        Model values (depth or a Jacobian row).

    Returns
    -------
    ndarray [n_keep]
        The binned model on the operator's kept bins.
    """
    wl, y = _validate_model(wl_model, y_model, "bin_model")
    icum = _pl_cumint(wl, y)
    ia = _pl_antideriv(op["cell_lo"], wl, y, icum)
    ib = _pl_antideriv(op["cell_hi"], wl, y, icum)
    d_pix = (ib - ia) / (op["cell_hi"] - op["cell_lo"])
    return _wsum(op, op["pix_w"] * d_pix) / _wsum(op, op["pix_w"])
