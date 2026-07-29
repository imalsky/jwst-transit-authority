"""One measurement operator per instrument mode: count-space binning shared by
the noise, the model, and every Jacobian row.

Why this module exists (2026-07-11 external audit, P0.1): the tool used to
report inverse-variance combined bin errors (noise.py) while binning the model
and Jacobians with local wavelength/trapezoid weights (detect.py). Those are
two DIFFERENT estimators, so the forecast model point was not the expectation
value of the statistic whose variance was being quoted. Everything now goes
through one operator built from the Pandeia extracted-pixel grid:

    pixels -> bin     d_b   = sum_i(w_i d_i) / sum_i(w_i),   w_i = F_i
    variance          Var_b = sum_i(w_i^2 Var_i) / (sum_i w_i)^2
    model -> pixel    d_i   = average of the piecewise-linear model over the
                              pixel's wavelength cell (midpoint edges)

Count-space weights (w_i = extracted stellar count rate F_i) are the estimator
real reductions implement: the binned light curve is the SUM of extracted
counts across the bin's pixels, so the fitted bin depth is the flux-weighted
mean of the per-pixel depths. In the photon-dominated limit (Var_i ~ 1/F_i)
this coincides with inverse-variance weighting; where background or read noise
contribute, count-space is slightly wider -- the honest variance of the
estimator actually used. The same weights bin the model, the removed-molecule
model, and the Jacobians, so signal, derivative, and noise all describe the
same statistic.

The model is averaged over each pixel's wavelength cell (exact integral of the
piecewise-linear model), never point-sampled, so a model grid finer than the
pixel grid (e.g. MIRI LRS) cannot alias. Where the FINAL binning approaches
the instrument's NATIVE resolving power (MIRI LRS, PRISM, blue SOSS),
detect.evaluate_mode additionally blurs the model -- the depth, the
removed-molecule depth, and EVERY Jacobian row, with one flux weight -- to
R_native via smooth_to_native_r (the flux-weighted count-ratio Gaussian LSF
in this module) BEFORE the cell average; for high-R modes that kernel is
unresolved by the model grid and the blur is a no-op. The Gaussian R(lambda)
kernel is an approximation to the full Pandeia response matrix
(monochromatic impulses through the 3D engine); impulse-response validation
against the engine is a documented pending gate, and Pandeia's extraction
already carries PSF/throughput into F_i and sigma_i.
"""
from __future__ import annotations

import numpy as np

# A pixel's cell half-width toward a neighbor is half that gap, capped at
# GAP_CAP x the smaller of its two adjacent gaps. The cap only matters at
# detector gaps (NRS1/NRS2) and band ends, where a raw midpoint cell would
# smear the model across wavelengths the pixel never sees.
GAP_CAP = 1.5

# A gap between adjacent (usable) pixels larger than this factor times the
# mode's median pixel spacing marks a DETECTOR-SEGMENT boundary (NIRSpec
# G395H/G235H NRS1|NRS2: gap ~150x median). Real dispersion gradients (PRISM,
# MIRI LRS) vary smoothly by factors of a few, far below this. Interior holes
# carved by saturation masks can also split a segment -- that only ADDS a
# nuisance offset (conservative), never removes one.
SEGMENT_GAP_FACTOR = 20.0

# Pixels whose local wavelength spacing is below this fraction of the mode's
# median spacing sit on a DEGENERATE wavelength solution (e.g. pandeia_data
# 3.0rc3 G395H piles ~700 samples within <1e-4 um at the NRS2 red edge, spacing
# down to 3.7e-6 um vs 6.6e-4 median). Counting them as independent spectral
# samples overstates the information in that bin (~sqrt(n) too-small sigma)
# and mislocates their flux in wavelength, so they are excluded -- loudly, via
# the n_pix_degenerate count surfaced by detect.evaluate_mode. Real dispersion
# gradients (PRISM, MIRI LRS) vary smoothly by factors of a few, far above
# this cut.
DEGENERATE_WL_FRAC = 0.02


def _validate_wl(wl, name: str) -> np.ndarray:
    """Loud wavelength-grid validation (2026-07-12 re-audit, item 7): a NaN or
    infinite wavelength is an upstream data error, never a sample to drop
    silently -- np.argsort would park NaNs at the end and every gap statistic
    downstream would be poisoned or silently truncated. Returns float array."""
    wl = np.asarray(wl, float)
    if wl.size == 0:
        raise ValueError(f"{name}: empty wavelength array")
    bad = ~np.isfinite(wl)
    if bad.any():
        raise ValueError(
            f"{name}: {int(bad.sum())}/{wl.size} non-finite wavelength(s) "
            f"(first at index {int(np.argmax(bad))}) -- fix the upstream "
            "grid; invalid samples are never silently dropped")
    return wl


def _positive_gap_median(gaps: np.ndarray) -> float:
    """Median pixel spacing over STRICTLY POSITIVE gaps. Duplicate wavelengths
    contribute zero-width gaps; including them made the median collapse to 0
    on duplicate-majority grids, which silently disabled both the degenerate
    mask (`local < 0.02*0` is never true) and the segment splitter
    (2026-07-12 re-audit, item 7). Returns 0.0 when every gap is zero."""
    pos = gaps[gaps > 0.0]
    return float(np.median(pos)) if pos.size else 0.0


def degenerate_wl_mask(wl_pix: np.ndarray) -> np.ndarray:
    """True for pixels on a degenerate wavelength solution (see above).
    Returned in the INPUT pixel order. Exact-duplicate wavelengths are always
    degenerate (zero local spacing = zero spectral support); a grid whose
    every pixel is duplicated comes back all-True, never all-False."""
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

    Segments are contiguous wavelength runs separated by gaps larger than
    SEGMENT_GAP_FACTOR x the median pixel spacing -- the NRS1/NRS2 split for
    the two-detector NIRSpec gratings, one segment for every other mode. Each
    segment gets its own calibration-offset nuisance in the detection score
    and the Fisher forecasts (Moran+2023 / Madhusudhan+2023-style NRS1/NRS2
    steps of tens of ppm are universal in real G395H fits)."""
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

    Needed where the final bins approach or beat the NATIVE resolving power
    (MIRI LRS R~40-160 across 5-12 um, NIRSpec PRISM R~30-300, blue SOSS):
    there the LSF redistributes the depth signal across bins to first order.
    For high-R modes the kernel is unresolved by the model grid and this is a
    no-op (returned unchanged) -- consistent with the sub-ppm edge-effect
    estimate for e.g. G395H at R_bin=100.

    STELLAR-FLUX WEIGHTING (``weight``, 2026-07-12 re-audit item 2). The
    instrument does not measure the LSF-average of the transit depth. It
    measures LSF-averaged in- and out-of-transit COUNTS and forms their ratio:
    with stellar flux F and depth d,

        d_obs = 1 - L[F (1 - d)] / L[F] = L[F d] / L[F],

    i.e. the FLUX-WEIGHTED LSF mean of d, not the flat mean L[d]. The two
    differ wherever F varies across a kernel (a stellar line, a throughput
    gradient, the SOSS blue drop-off): the flat blur mislocated the depth
    signal by tens of ppm near structured stellar spectra. Pass the stellar
    flux at ``wl_model`` as ``weight`` to get the correct ratio; ``None`` (or a
    constant weight) reduces exactly to the flat blur. F is only resolved to
    the extracted pixel scale, so sub-pixel stellar lines cannot be recovered
    here (a documented limitation); this corrects the pixel-resolved structure,
    which is what the native-R blur redistributes. The operator stays LINEAR in
    d for fixed F, so a Jacobian row blurs with the SAME weight as the depth.

    Implementation: cell-average the piecewise-linear model (and the weight)
    onto a uniform ln-lambda grid finer than the narrowest kernel (flux-
    conserving, no aliasing of unresolved lines), convolve with the
    wavelength-dependent Gaussian, and interpolate back onto the model points
    inside the band. The working band extends 5 sigma beyond [band_lo, band_hi]
    so one-sided kernels never touch the returned region.
    """
    wl, yv = _validate_model(wl_model, y, "smooth_to_native_r")
    # Validate the weight BEFORE any early return: the no-op paths (band with
    # no model points, kernel unresolved by the model grid) used to skip this,
    # so a NaN/negative stellar flux raised on MIRI LRS / PRISM and passed
    # silently on the high-R gratings -- the same bad input diagnosed or not
    # depending on the mode (2026-07-28 audit). A check that cannot change the
    # result still has to say the input is invalid.
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

    # flux-conserving cell average of the piecewise-linear model in ln-lambda,
    # via the EXACT piecewise-quadratic antiderivative (linear interp of icum
    # would misplace flux for cell edges between nodes -- audit item 2)
    icum = _pl_cumint(lnw, yv)
    edges = np.concatenate([[grid[0] - 0.5 * dl], grid + 0.5 * dl])
    edges = np.clip(edges, lnw[0], lnw[-1])
    ic = _pl_antideriv(edges, lnw, yv, icum)
    w_raw = np.diff(edges)
    widths = np.maximum(w_raw, 1e-300)
    yg = np.diff(ic) / widths
    # Zero-width END cells (the edge clip collapses cells whose grid overshoot
    # lands past the model span -- decided by an uncontrolled grid-alignment
    # fractional part) hold NO model content: 0/1e-300 = 0 above, and that
    # spurious zero used to fill the entire constant pad below, dragging the
    # weight=None blur toward zero over the last ~4 kernel sigmas of the band
    # (~49% of a constant depth at the edge on a MIRI-like clamped grid; the
    # weighted path survived only because the flux floor demoted the cell).
    # Constant extension must carry the nearest cell WITH content instead --
    # a normalized kernel then preserves a constant for ANY weight (v17 fix).
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

    # stellar-flux weight on the same grid (cell-averaged the same way), so the
    # convolution forms L[F d]/L[F] rather than the flat L[d] (re-audit item 2).
    # None/constant weight -> Fg constant -> exactly the flat blur. A tiny
    # positive floor keeps an (unphysical) all-zero-flux window from dividing by
    # zero -- it falls back to flat weighting there.
    if weight is None:
        Fg = np.ones_like(yg)
    else:
        wv = np.asarray(weight, float)   # already validated above
        icf = _pl_antideriv(edges, lnw, wv, _pl_cumint(lnw, wv))
        Fg = np.maximum(np.diff(icf) / widths, 0.0)
        # zero-width end cells: constant-extend the flux weight exactly like
        # yg (pre-v17 they held weight 0 and only the 1e-12 floor kept the
        # spurious zero DEPTH pad from biting on the weighted path)
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
                    and >= 0; a zero-weight pixel is excluded from the
                    estimator (it carries no counts), a negative or non-finite
                    weight raises (upstream extraction error, never sanitized).
    edges         : final bin edges (finite, strictly ascending, >= 2).
    wl_lo, wl_hi  : model wavelength span; pixel cells are clipped to it and
                    pixels whose cell falls outside are dropped (their model
                    expectation is undefined) -- from BOTH model and noise,
                    so the two stay the same estimator.
    valid         : optional per-pixel bool mask (e.g. saturation exclusion).

    Bin-assignment policy: a pixel belongs to the bin containing its CENTER
    wavelength (extracted pixels are indivisible samples grouped into bins;
    no fractional pixel-response splitting across bin edges is performed),
    while the MODEL is integrated over the pixel's full cell. This matches
    how real reductions group extracted pixels; it is not an arbitrary-edge
    fractional regridding and is not claimed to be one.

    Duplicate-wavelength policy: an exact-duplicate pixel has a zero-width
    cell (no wavelength support), so it drops out of the estimator here; such
    pixels are degenerate-wavelength pixels and callers exclude + COUNT them
    via degenerate_wl_mask (surfaced as n_pix_degenerate) -- that is the loud
    channel, not this operator. An operator left with NO usable pixel raises
    with the per-criterion exclusion breakdown (2026-07-12 re-audit, item 7)
    instead of returning an empty object that fails downstream.

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


def bin_counts(op: dict, flag_pix: np.ndarray) -> np.ndarray:
    """Plain per-kept-bin sum of a per-pixel count/flag (e.g. saturation)."""
    v = np.asarray(flag_pix, float)[op["pix_idx"]]
    return _wsum(op, v)


def _validate_model(wl: np.ndarray, y: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Model-grid validation for the piecewise-linear integrators: strictly
    ascending finite wavelengths (what _pl_antideriv's searchsorted assumes)
    and finite values. A NaN model value would propagate NaN into every bin
    it touches -- raise instead (repo loud-errors rule)."""
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
    icum[k] = int_{x[0]}^{x[k]}. The trapezoid rule is EXACT for a
    piecewise-linear integrand, so this is exact at the nodes (only float64
    cumsum roundoff); the sub-node evaluation is done by _pl_antideriv."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def _pl_antideriv(xq: np.ndarray, x: np.ndarray, y: np.ndarray,
                  icum: np.ndarray) -> np.ndarray:
    """EXACT antiderivative of the piecewise-linear function (x, y) evaluated at
    query points xq, given its node integrals ``icum`` (from _pl_cumint).

    The antiderivative of a piecewise-LINEAR spectrum is piecewise-QUADRATIC,
    so linearly interpolating ``icum`` between nodes (np.interp) is wrong
    whenever a query point falls inside an interval -- exactly the case for a
    detector cell edge lying between model nodes (2026-07-12 audit, item 2:
    the y=x, [0.1,0.2] counterexample returned 0.5 instead of 0.15). The
    quadratic term is carried explicitly here: for x in interval k
    ([x_k, x_{k+1}]),

        I(x) = icum[k] + y_k (x - x_k) + 0.5 slope_k (x - x_k)^2,
        slope_k = (y_{k+1} - y_k) / (x_{k+1} - x_k).

    Query points are clamped to [x_0, x_{-1}] (the model span; callers already
    clip pixel cells to it). ``x`` must be strictly ascending."""
    xq = np.asarray(xq, float)
    xc = np.clip(xq, x[0], x[-1])
    k = np.clip(np.searchsorted(x, xc, side="right") - 1, 0, x.size - 2)
    dx = xc - x[k]
    slope = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
    return icum[k] + y[k] * dx + 0.5 * slope * dx * dx


def bin_model(op: dict, wl_model: np.ndarray, y_model: np.ndarray) -> np.ndarray:
    """Bin a native model through the operator: exact cell average of the
    piecewise-linear model per pixel, then the count-weighted bin mean.

    ``wl_model`` must be ascending and span every pixel cell (build_operator's
    wl_lo/wl_hi clipping guarantees this). Linear in y_model, so the binned
    Jacobian is the operator applied to each Jacobian row. Exact for a constant
    model (constant-depth conservation) AND, via the exact piecewise-quadratic
    antiderivative, for any pixel cell whose edges fall between model nodes."""
    wl, y = _validate_model(wl_model, y_model, "bin_model")
    icum = _pl_cumint(wl, y)
    ia = _pl_antideriv(op["cell_lo"], wl, y, icum)
    ib = _pl_antideriv(op["cell_hi"], wl, y, icum)
    d_pix = (ib - ia) / (op["cell_hi"] - op["cell_lo"])
    return _wsum(op, op["pix_w"] * d_pix) / _wsum(op, op["pix_w"])
