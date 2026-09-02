"""One measurement operator per instrument mode: count-space binning shared by
the noise, the model, and every Jacobian row.

Guardrails:
* ONE operator bins noise, model, and Jacobians -- never reintroduce separate
  binning paths; the model point must be the expectation of the statistic
  whose variance is quoted.
* Count-space weights (w_i = extracted stellar count rate F_i) are the
  estimator real reductions implement: d_b = sum(w d)/sum(w),
  Var_b = sum(w^2 Var)/(sum w)^2. In the photon limit this equals
  inverse-variance weighting; with background/read noise it is slightly
  wider -- the honest variance of the estimator actually used.
* The model is averaged over each pixel's wavelength cell (exact integral of
  the piecewise-linear model), never point-sampled, so a finer model grid
  cannot alias.
* Where final bins approach the NATIVE resolving power (MIRI LRS, PRISM,
  blue SOSS), detect.evaluate_mode blurs depth, removed-molecule depth, and
  every Jacobian row to R_native via smooth_to_native_r BEFORE the cell
  average; on high-R modes the blur is a no-op. The Gaussian kernel
  approximates the full Pandeia response matrix; its width is
  R_refdata / instruments.LSF_WIDTH, the per-mode width fitted to an impulse
  through the engine (tests/parity/scripts/run_parity.py --impulse,
  parity_summary.json["lsf_impulse"]).
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
# so they are excluded loudly via n_pix_degenerate_dropped.
DEGENERATE_WL_FRAC = 0.02


def _validate_wl(wl, name: str) -> np.ndarray:
    """Finite, positive, non-empty 1-D wavelength array or ValueError;
    invalid samples are never silently dropped (a NaN would poison every
    gap statistic). Duplicates are NOT rejected here: degenerate_wl_mask
    masks them by design. Returns float array."""
    wl = np.asarray(wl, float)
    if wl.ndim != 1 or wl.size == 0:
        raise ValueError(f"{name}: wavelength must be a non-empty 1-D array")
    bad = ~np.isfinite(wl)
    if bad.any():
        raise ValueError(
            f"{name}: {int(bad.sum())}/{wl.size} non-finite wavelength(s) "
            f"(first at index {int(np.argmax(bad))}) -- fix the upstream "
            "grid; invalid samples are never silently dropped")
    if np.any(wl <= 0.0):
        raise ValueError(f"{name}: non-positive wavelength(s)")
    return wl


def _positive_gap_median(gaps: np.ndarray) -> float:
    """Median pixel spacing over STRICTLY POSITIVE gaps. Zero-width duplicate
    gaps would collapse the median to 0 and silently disable both the
    degenerate mask and the segment splitter. Returns 0.0 when every gap is
    zero."""
    pos = gaps[gaps > 0.0]
    return float(np.median(pos)) if pos.size else 0.0


def degenerate_wl_mask(wl_pix: np.ndarray) -> np.ndarray:
    """True for pixels on a degenerate wavelength solution, in INPUT order.
    Exact duplicates are always degenerate (zero spectral support); an
    all-duplicate grid returns all-True."""
    wl_pix = _validate_wl(wl_pix, "degenerate_wl_mask: wl_pix")
    order = np.argsort(wl_pix)
    wl_s = wl_pix[order]
    if wl_s.size < 3:
        return np.zeros(wl_pix.size, bool)
    gaps = np.diff(wl_s)
    med = _positive_gap_median(gaps)
    if med == 0.0:                       # every pixel shares one wavelength
        return np.ones(wl_pix.size, bool)
    local = np.minimum(np.concatenate([[gaps[0]], gaps]),
                       np.concatenate([gaps, [gaps[-1]]]))
    bad_sorted = local < DEGENERATE_WL_FRAC * med
    bad = np.zeros(wl_pix.size, bool)
    bad[order] = bad_sorted
    return bad


def segment_ids(wl_pix: np.ndarray) -> np.ndarray:
    """Detector-segment id (0, 1, ...) per pixel, in the INPUT pixel order.

    Segments split at gaps larger than SEGMENT_GAP_FACTOR x the median pixel
    spacing (the NRS1|NRS2 split; one segment for every other mode). Each
    segment gets its own calibration-offset nuisance downstream."""
    wl_pix = _validate_wl(wl_pix, "segment_ids: wl_pix")
    order = np.argsort(wl_pix)
    if wl_pix.size < 2:
        return np.zeros(wl_pix.size, int)
    gaps = np.diff(wl_pix[order])
    med = _positive_gap_median(gaps)
    if med == 0.0:                       # all-duplicate grid: one segment
        return np.zeros(wl_pix.size, int)
    split = gaps > SEGMENT_GAP_FACTOR * med
    seg_sorted = np.concatenate([[0], np.cumsum(split)])
    seg = np.empty(wl_pix.size, int)
    seg[order] = seg_sorted
    return seg


def bin_segments(op: dict, seg_pix: np.ndarray) -> np.ndarray:
    """Per-kept-bin segment id: the segment holding the bin's count weight.
    Bins never straddle a segment gap in practice (gap >> bin width); if one
    ever did, the count-majority segment is the honest assignment."""
    seg = np.asarray(seg_pix)[op["pix_idx"]].astype(int)
    n_bins = op["wl_center"].size
    n_seg = int(seg.max()) + 1 if seg.size else 1
    w = np.zeros((n_seg, n_bins))
    np.add.at(w, (seg, op["pix_bin"]), op["pix_w"])
    return np.argmax(w, axis=0)


def smooth_to_native_r(wl_model: np.ndarray, y: np.ndarray,
                       wl_r: np.ndarray, r_curve: np.ndarray,
                       band_lo: float, band_hi: float,
                       weight: np.ndarray | None = None) -> np.ndarray:
    """Blur a native transit-depth model to the instrument's Gaussian LSF of
    resolving power R(lambda) over [band_lo, band_hi]; returns a full-length copy.

    Only bites where the MODEL grid resolves the kernel, i.e. model R >=
    2.3548 x the smallest R in the band (MIRI LRS, PRISM, and SOSS order 1
    under the correlated-k default's R = 1000 grid); on higher-R modes the
    kernel is unresolved and the input is returned unchanged. NIRCam, SOSS
    order 2, and the M gratings are not resolved under that default.

    ``wl_r``/``r_curve`` are the native resolving-power table. ``wl_r`` MUST
    be strictly ascending and is validated: the kernel width comes from
    ``np.interp(lambda, wl_r, r_curve)``, which silently returns the last
    entry everywhere on a descending table. The pandeia pixel grid is
    DISPERSION order, not wavelength order (MIRI LRS ships it 13.86 -> 5.02
    um), so sort it before calling.

    ``weight`` is the stellar flux at ``wl_model``. The instrument measures
    LSF-averaged COUNTS, so d_obs = L[F d] / L[F], the flux-weighted LSF mean
    -- never revert to the flat L[d] blur. ``None`` or a constant weight
    reduces exactly to the flat blur. F is only pixel-resolved (sub-pixel
    stellar lines are a documented limit). The operator is LINEAR in d for
    fixed F, so a Jacobian row blurs with the SAME weight as the depth.

    Implementation: cell-average model and weight onto a uniform ln-lambda
    grid finer than the narrowest kernel, convolve with the
    wavelength-dependent Gaussian, interpolate back inside the band. The
    working band extends 5 sigma past the edges so one-sided kernels never
    touch the returned region.
    """
    wl, yv = _validate_model(wl_model, y, "smooth_to_native_r")
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
            "wavelength")
    if not np.all(np.isfinite(r_v)) or np.any(r_v <= 0.0):
        raise ValueError(
            "smooth_to_native_r: r_curve (native resolving power) must be "
            f"finite and > 0 -- got {int((~np.isfinite(r_v)).sum())} "
            f"non-finite / {int((r_v <= 0.0).sum())} non-positive value(s)")
    if wl_r_v.size > 1 and np.any(np.diff(wl_r_v) <= 0.0):
        _bad = int(np.argmax(np.diff(wl_r_v) <= 0.0))
        raise ValueError(
            "smooth_to_native_r: wl_r (the native-R wavelength table) must be "
            f"STRICTLY ASCENDING -- first violation at index {_bad} "
            f"({wl_r_v[_bad]:g} -> {wl_r_v[_bad + 1]:g}). The kernel width is "
            "read with np.interp, which requires increasing sample points and "
            "silently returns the end value everywhere on a descending table "
            "(MIRI LRS ships its pandeia grid in dispersion order, 13.86 -> "
            "5.02 um). Sort wl_r and r_curve together before calling.")
    # Validate the weight BEFORE any early return: the no-op paths must still
    # reject a NaN/negative stellar flux, or the same bad input raises or
    # passes depending on the mode.
    if weight is not None:
        wv = np.asarray(weight, float)
        if wv.shape != wl.shape or not np.all(np.isfinite(wv)) \
                or np.any(wv < 0.0):
            raise ValueError(
                "smooth_to_native_r: weight (stellar flux on the model grid) "
                "must match wl_model's shape and be finite and >= 0 -- a NaN "
                "weight silently corrupts the count-ratio blur")
    lnw = np.log(wl)
    x_lo, x_hi = np.log(band_lo), np.log(band_hi)
    in_band = (wl_r >= band_lo) & (wl_r <= band_hi)
    r_band = np.asarray(r_curve, float)[in_band] if in_band.any() else np.asarray(r_curve, float)
    r_min = max(5.0, float(np.min(r_band)))
    r_max = max(r_min, float(np.max(r_band)))
    s_max = 1.0 / (2.3548 * r_min)          # widest kernel sigma, in ln-lambda
    s_min = 1.0 / (2.3548 * r_max)

    sel = (lnw >= x_lo) & (lnw <= x_hi)
    if not sel.any():
        return yv.copy()
    d_model = float(np.median(np.diff(lnw[sel]))) if sel.sum() > 2 else np.inf
    if s_max < d_model:                     # kernel unresolved by the model grid
        return yv.copy()

    lo = max(float(lnw[0]), x_lo - 5.0 * s_max)
    hi = min(float(lnw[-1]), x_hi + 5.0 * s_max)
    dl = s_min / 6.0
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
    if _valid.size == 0:        # cannot happen for a validated band, be loud
        raise ValueError(
            "smooth_to_native_r: every working-grid cell collapsed to zero "
            "width after clipping to the model span -- the model grid does "
            "not overlap the requested band")
    if _valid[0] > 0:
        yg[:_valid[0]] = yg[_valid[0]]
    if _valid[-1] < yg.size - 1:
        yg[_valid[-1] + 1:] = yg[_valid[-1]]

    # Stellar-flux weight cell-averaged the same way, so the convolution forms
    # L[F d]/L[F], not the flat L[d]. A tiny positive floor keeps an all-zero-
    # flux window from dividing by zero (falls back to flat weighting there).
    if weight is None:
        Fg = np.ones_like(yg)
    else:
        wv = np.asarray(weight, float)   # already validated above
        icf = _pl_antideriv(edges, lnw, wv, _pl_cumint(lnw, wv))
        Fg = np.maximum(np.diff(icf) / widths, 0.0)
        # zero-width end cells: constant-extend the flux weight exactly like yg
        if _valid[0] > 0:
            Fg[:_valid[0]] = Fg[_valid[0]]
        if _valid[-1] < Fg.size - 1:
            Fg[_valid[-1] + 1:] = Fg[_valid[-1]]
        fmax = float(Fg.max()) if Fg.size else 0.0
        Fg = np.maximum(Fg, 1e-12 * fmax) if fmax > 0.0 else np.ones_like(yg)

    k = int(np.ceil(4.0 * s_max / dl))
    pad = np.concatenate([np.full(k, yg[0]), yg, np.full(k, yg[-1])])
    padF = np.concatenate([np.full(k, Fg[0]), Fg, np.full(k, Fg[-1])])
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1)
    winF = np.lib.stride_tricks.sliding_window_view(padF, 2 * k + 1)
    sig = 1.0 / (2.3548 * np.maximum(np.interp(np.exp(grid), wl_r, r_curve), 5.0))
    off = dl * (np.arange(2 * k + 1) - k)
    w = np.exp(-0.5 * (off[None, :] / sig[:, None]) ** 2) * winF   # flux-weighted
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
    g_lo = np.concatenate([[gaps[0]], gaps])       # gap toward the previous pixel
    g_hi = np.concatenate([gaps, [gaps[-1]]])      # gap toward the next pixel
    g_min = np.minimum(g_lo, g_hi)
    lo = wl_sorted - np.minimum(0.5 * g_lo, GAP_CAP * g_min)
    hi = wl_sorted + np.minimum(0.5 * g_hi, GAP_CAP * g_min)
    return lo, hi


def build_operator(wl_pix: np.ndarray, w_pix: np.ndarray, edges: np.ndarray,
                   wl_lo: float = -np.inf, wl_hi: float = np.inf,
                   valid: np.ndarray | None = None) -> dict:
    """Build the count-space measurement operator for one mode.

    wl_pix, w_pix : Pandeia pixel wavelengths (any order) and weights
                    (extracted stellar count rates). Weights must be finite
                    and >= 0; zero-weight pixels are excluded, negative or
                    non-finite weights raise (never sanitized).
    edges         : final bin edges (finite, strictly ascending, >= 2).
    wl_lo, wl_hi  : model wavelength span; pixel cells are clipped to it and
                    pixels falling outside are dropped from BOTH model and
                    noise, so the two stay the same estimator.
    valid         : optional per-pixel bool mask (e.g. saturation exclusion).

    A pixel belongs to the bin containing its CENTER wavelength (indivisible
    extracted samples, as real reductions group them); the MODEL is
    integrated over the pixel's full cell. Exact-duplicate pixels have
    zero-width cells and drop out here; callers count them loudly via
    degenerate_wl_mask (n_pix_degenerate_dropped). An operator left with NO
    usable pixel raises with the per-criterion exclusion breakdown.

    Returns dict:
      keep     (n_bins,)  bins with >=1 usable pixel
      wl_center(n_keep,)  edge midpoints of kept bins
      n_pix    (n_keep,)  usable pixels per kept bin
      pix_idx  (n_use,)   indices into the INPUT pixel arrays (callers pass
                          full-length per-pixel arrays; the operator selects)
      pix_w, cell_lo, cell_hi (n_use,), pix_bin (n_use,) index into kept bins
    """
    wl_pix = _validate_wl(wl_pix, "build_operator: wl_pix")
    w_pix = np.asarray(w_pix, float)
    if w_pix.shape != wl_pix.shape:
        raise ValueError(f"build_operator: w_pix shape {w_pix.shape} != "
                         f"wl_pix shape {wl_pix.shape}")
    if not np.all(np.isfinite(w_pix)) or np.any(w_pix < 0.0):
        raise ValueError(
            "build_operator: pixel weights (extracted count rates) must be "
            "finite and >= 0 -- got "
            f"{int((~np.isfinite(w_pix)).sum())} non-finite / "
            f"{int((w_pix < 0.0).sum())} negative value(s); fix the worker "
            "output, invalid weights are never silently dropped")
    edges = np.asarray(edges, float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)) \
            or np.any(np.diff(edges) <= 0.0):
        raise ValueError("build_operator: bin edges must be a 1-D, finite, "
                         f"strictly ascending array of >= 2 values, got "
                         f"shape {edges.shape}")
    if valid is not None and np.asarray(valid).shape != wl_pix.shape:
        raise ValueError(f"build_operator: valid mask shape "
                         f"{np.asarray(valid).shape} != wl_pix shape "
                         f"{wl_pix.shape}")
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
            "overlap and the requested binning")

    idx = order[ok]
    bins = bin_raw[ok]
    keep = np.zeros(nb, dtype=bool)
    keep[bins] = True
    remap = np.cumsum(keep) - 1                     # bin id -> kept-bin id
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_pix = np.bincount(remap[bins], minlength=int(keep.sum()))
    return dict(keep=keep, wl_center=centers[keep],
                n_pix=n_pix.astype(int),
                pix_idx=idx, pix_w=w_pix[idx],
                cell_lo=lo_s[ok], cell_hi=hi_s[ok],
                pix_bin=remap[bins])


def _wsum(op: dict, values_per_pixel: np.ndarray) -> np.ndarray:
    out = np.zeros(op["wl_center"].size)
    np.add.at(out, op["pix_bin"], values_per_pixel)
    return out


def bin_values(op: dict, v_pix: np.ndarray) -> np.ndarray:
    """Count-weighted mean of a per-pixel quantity, per kept bin.
    ``v_pix`` is full-length (aligned with the arrays given to build_operator)."""
    v = np.asarray(v_pix, float)[op["pix_idx"]]
    return _wsum(op, op["pix_w"] * v) / _wsum(op, op["pix_w"])


def bin_variance(op: dict, var_pix: np.ndarray) -> np.ndarray:
    """Variance of the count-weighted bin estimator: sum(w^2 Var)/(sum w)^2."""
    v = np.asarray(var_pix, float)[op["pix_idx"]]
    return _wsum(op, op["pix_w"] ** 2 * v) / _wsum(op, op["pix_w"]) ** 2


def _validate_model(wl: np.ndarray, y: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Strictly ascending finite wavelength grid + finite values, or
    ValueError (a NaN model value would propagate into every bin)."""
    wl = _validate_wl(wl, f"{name}: wl_model")
    if wl.size < 2 or np.any(np.diff(wl) <= 0.0):
        raise ValueError(f"{name}: model wavelength grid must be strictly "
                         "ascending with >= 2 points (duplicates or "
                         "descending runs are upstream errors)")
    y = np.asarray(y, float)
    if y.shape != wl.shape:
        raise ValueError(f"{name}: model value shape {y.shape} != wavelength "
                         f"shape {wl.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name}: {int((~np.isfinite(y)).sum())} non-finite "
                         "model value(s) -- fix the upstream model")
    return wl, y


def _pl_cumint(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Running integral of the piecewise-linear (x, y) AT the nodes:
    icum[k] = int_{x[0]}^{x[k]}. Trapezoid is exact for a piecewise-linear
    integrand; sub-node evaluation is _pl_antideriv."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def _pl_antideriv(xq: np.ndarray, x: np.ndarray, y: np.ndarray,
                  icum: np.ndarray) -> np.ndarray:
    """EXACT antiderivative of the piecewise-linear (x, y) at query points xq,
    given its node integrals ``icum`` (from _pl_cumint).

    The antiderivative is piecewise-QUADRATIC: never np.interp ``icum`` to
    cell edges -- only O(h^2), ~1-2 ppm off for edges between model nodes.
    For x in interval k: I(x) = icum[k] + y_k (x - x_k)
    + 0.5 slope_k (x - x_k)^2. Queries are clamped to the model span;
    ``x`` must be strictly ascending."""
    xq = np.asarray(xq, float)
    xc = np.clip(xq, x[0], x[-1])
    k = np.clip(np.searchsorted(x, xc, side="right") - 1, 0, x.size - 2)
    dx = xc - x[k]
    slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
    return icum[k] + y[k] * dx + 0.5 * slope * dx * dx


def bin_model(op: dict, wl_model: np.ndarray, y_model: np.ndarray) -> np.ndarray:
    """Bin a native model through the operator: exact cell average of the
    piecewise-linear model per pixel, then the count-weighted bin mean.

    ``wl_model`` must be ascending and span every pixel cell. Linear in
    y_model, so the binned Jacobian is the operator applied to each row;
    exact for a constant model and for cell edges between model nodes."""
    wl, y = _validate_model(wl_model, y_model, "bin_model")
    icum = _pl_cumint(wl, y)
    ia = _pl_antideriv(op["cell_lo"], wl, y, icum)
    ib = _pl_antideriv(op["cell_hi"], wl, y, icum)
    d_pix = (ib - ia) / (op["cell_hi"] - op["cell_lo"])
    return _wsum(op, op["pix_w"] * d_pix) / _wsum(op, op["pix_w"])
