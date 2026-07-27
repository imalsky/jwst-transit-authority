"""Fisher-information parameter forecast from the autodiff spectrum Jacobian.

Given each instrument mode's binned Jacobian J (n_par, n_bins) and per-bin depth
sigma, the Fisher matrix is

    F_ij = sum_b J_ib J_jb / sigma_b^2

and the marginalized 1-sigma forecast on parameter i is sqrt((F^-1)_ii).

Nuisance handling (mirrors the zco_information campaign):
  * per mode: the free parameters + lnR0 (reference-radius) + one constant
    depth OFFSET per detector SEGMENT are jointly fit; the nuisances are
    marginalized out of the report. A single-detector mode has one segment
    (its offset is degenerate with lnR0 and drops out via the rank-aware
    inversion); the two-detector NIRSpec gratings (G395H, G235H) get a
    separate NRS1 and NRS2 offset, because a detector-to-detector step of
    tens of ppm is universal in real fits (Moran+2023, Madhusudhan+2023) and
    can otherwise masquerade as atmospheric structure.
  * combined (all selected modes): one SHARED lnR0 plus one constant depth
    OFFSET per SEGMENT across all modes (absolute-calibration nuisance between
    detectors/visits), all marginalized. Offsets are what make
    multi-instrument combinations honest -- within a single band an offset and
    lnR0 are nearly degenerate.

A parameter with (numerically) no spectral response, or with weight in a
numerically null Fisher direction (a degeneracy), comes back as inf, shown as
"unconstrained" by the GUI rather than a fake number. The inversion is ALWAYS
rank-aware AND unit-invariant: rank detection happens on the Jacobi-whitened
(unit-diagonal) Fisher matrix, so redefining a parameter's units cannot flip
a constraint between finite and "unconstrained" (see _marg_sigmas).
np.linalg.inv on an ill-conditioned Fisher matrix returns misleading finite
numbers without raising, so it is never used. Forecast sigmas are local
Cramer-Rao lower bounds under the quoted noise model -- best cases, not
posterior widths (the standard Fisher-forecast caveats -- local Gaussian
approximation, prior-free, degeneracy-blind beyond rank -- are Vallisneri
2008, PRD 77, 042001, arXiv:gr-qc/0703086).
"""
from __future__ import annotations

import numpy as np

_LN10 = np.log(10.0)

# report-unit conversion: sigma in internal ln-units -> display units. lnZ/lnKzz
# are natural-log internally but REPORTED in dex (log10), so divide by ln10. C/O
# is handled separately in display_sigma (its display factor is the atmosphere's
# absolute C/O, not a constant -- see below).
_TO_DISPLAY = {"lnZ": 1.0 / _LN10, "lnKzz": 1.0 / _LN10}

# Relative eigenvalue threshold, applied to the WHITENED (unit-diagonal,
# correlation-form) Fisher matrix: directions below REL_EIG_TOL x the largest
# whitened eigenvalue are treated as numerically unconstrained (null space).
# PRECISE GUARANTEE: whitening makes the rank decision invariant under
# DIAGONAL per-parameter rescalings (unit changes) -- thresholding the raw
# mixed-unit matrix flipped finite constraints to "unconstrained" (or back)
# under a pure K-vs-kK rescaling (2026-07-12 external audit, confirmed). It
# is NOT invariant under arbitrary MIXED reparameterizations of directions
# sitting near the threshold: no numerical rank cut is metric-free, and a
# publication-grade statement should quote the whitened eigenvalue spectrum
# (the ``diag`` output) and, where it matters, the constrained
# eigen-combinations rather than only coordinate-wise sigmas. eigh's noise
# floor is ~1e-16 x wmax; 1e-10 keeps 6 decades of margin either way.
REL_EIG_TOL = 1e-10
# A reported parameter whose projection ONTO the null subspace (whitened
# coordinates) exceeds this is flagged inf (it lives partly in an unconstrained
# direction). The metric is the L2 norm over the null eigenvectors -- a
# basis-invariant subspace projection, not any single eigenvector's largest
# component (that is arbitrary when the null eigenspace is degenerate;
# 2026-07-12 external audit, item 3).
NULL_LOAD_TOL = 1e-6


def _marg_sigmas(F: np.ndarray, n_report: int,
                 diag: dict | None = None) -> np.ndarray:
    """Rank-aware marginalized sigmas for the first n_report parameters of F.

    The Fisher matrix mixes parameters with unrelated units (K, ln-units,
    fractional depths), so rank detection happens in Jacobi-whitened
    coordinates: q_i = theta_i * sqrt(F_ii), giving a unit-diagonal
    (correlation-form) matrix whose eigen-spectrum is invariant under
    per-parameter rescaling. Sigmas are transformed back to physical units
    afterwards; a full-rank matrix reproduces inv(F) exactly. Parameters with
    F_ii == 0 (no response at all) and parameters loaded on null directions
    come back inf. Pass a dict as ``diag`` to receive rank / dimension /
    condition number / eigenvalues (the whitened spectrum -- scale-free).
    """
    F = np.asarray(F, float)
    F = 0.5 * (F + F.T)
    n = F.shape[0]
    out = np.full(n_report, np.inf)
    d = np.sqrt(np.clip(np.diag(F), 0.0, None))
    nz = d > 0.0
    if not nz.any():
        if diag is not None:
            diag.update(fisher_dimension=n, fisher_rank=0,
                        condition_number=float("inf"),
                        eigenvalues=np.zeros(0), rel_eig_tol=REL_EIG_TOL)
        return out
    Fw = F[np.ix_(nz, nz)] / np.outer(d[nz], d[nz])
    w, V = np.linalg.eigh(0.5 * (Fw + Fw.T))
    wmax = float(w[-1]) if w.size else 0.0
    good = w > REL_EIG_TOL * max(wmax, 1e-300)
    if diag is not None:
        diag.update(
            fisher_dimension=n,
            fisher_rank=int(good.sum()),
            condition_number=(wmax / float(w[good].min()) if good.any()
                              else float("inf")),
            eigenvalues=w.copy(),
            rel_eig_tol=REL_EIG_TOL,
        )
    if not good.any():
        return out
    cov_w = ((V[:, good] ** 2) / w[good]).sum(axis=1)
    sig_nz = np.sqrt(cov_w) / d[nz]
    if (~good).any():
        # basis-invariant projection onto the null subspace: L2 norm over the
        # null eigenvectors, not a single vector's largest component (audit 3)
        load = np.sqrt(np.sum(V[:, ~good] ** 2, axis=1))
        sig_nz[load > NULL_LOAD_TOL] = np.inf
    full = np.full(n, np.inf)
    full[nz] = sig_nz
    out[:] = full[:n_report]
    return out


def _conditional_sigmas(F: np.ndarray, n_report: int) -> np.ndarray:
    """CONDITIONAL sigmas 1/sqrt(F_ii) for the first n_report parameters:
    every other parameter AND every nuisance held fixed at truth. The
    optimistic Cramer-Rao complement of the marginalized sigma (conditional
    <= marginalized always, up to numerical-rank inf); a large gap flags
    degeneracy with other parameters/nuisances, not missing spectral
    response. F_ii == 0 (no response at all) reads inf. Pure diagonal
    read-off -- no inversion, so no rank subtleties."""
    d = np.asarray(np.diag(np.asarray(F, float)), float)[:n_report]
    out = np.full(n_report, np.inf)
    pos = d > 0.0
    out[pos] = 1.0 / np.sqrt(d[pos])
    return out


def display_sigma(name: str, sigma: float, co_eval: float | None = None) -> float:
    """Internal-unit sigma -> display-unit sigma.

    C/O (``dlnCO``) is reported as an ABSOLUTE number ratio: the internal sigma is
    in delta-ln(C/O), so to first order sigma_CO = C/O * sigma_lnCO. ``co_eval``
    is the atmosphere's C/O at the evaluation point (params_json co_ratio);
    it is REQUIRED for dlnCO (a loud error beats a silently-wrong scale)."""
    if name == "dlnCO":
        if co_eval is None:
            raise ValueError(
                "display_sigma('dlnCO', ...) needs co_eval (the atmosphere's C/O) "
                "to report an absolute C/O uncertainty: sigma_CO = C/O * sigma_lnCO.")
        return sigma * co_eval
    return sigma * _TO_DISPLAY.get(name, 1.0)


def _segment_offset_rows(result: dict) -> np.ndarray:
    """One constant-offset indicator row per detector segment INCLUDING the
    first (n_seg rows; a single-segment mode gets one global constant), from
    result["seg"] (per-bin segment id from detect.evaluate_mode).

    The pre-2026-07-12 version omitted segment 0 on the premise that the
    shared lnR0 derivative "already spans" the first segment's constant --
    WRONG: lnR0 is a physical radiative-transfer derivative, generally not
    constant in wavelength, so the calibration offset and lnR0 are distinct
    nuisance directions, and a spectrally-constant science signal could read
    as constrained (recheck item P0-A, confirmed by reproducer). Any exact
    redundancy between rows is precisely what the rank-aware _marg_sigmas
    handles. With every segment present, mode_forecast(r) implements the
    SAME statistical model as combined_forecast([r]) -- pinned by
    test_mode_forecast_equals_combined_single_result."""
    nb = np.asarray(result["sigma"]).size
    seg = np.asarray(result.get("seg", np.zeros(nb, int)), int)
    n_seg = int(seg.max()) + 1 if seg.size else 1
    return np.stack([(seg == s).astype(float) for s in range(n_seg)])


def _slope_nuisance_rows(result: dict) -> np.ndarray:
    """Per-segment slope rows the mode's noise scenario profiles (stored by
    detect.evaluate_mode; empty (0, n_bins) when the scenario floats none)."""
    nb = np.asarray(result["sigma"]).size
    return np.asarray(result.get("slope_rows", np.zeros((0, nb))), float)


def _fisher(Jn: np.ndarray, result: dict) -> np.ndarray:
    """J C^-1 J^T under the mode's noise model: the stored scenario covariance
    when present (correlated floor), else the exact diagonal fast path."""
    cov = result.get("cov")
    if cov is None:
        s = np.asarray(result["sigma"])
        return (Jn / s[None, :] ** 2) @ Jn.T
    return Jn @ np.linalg.solve(np.asarray(cov, float), Jn.T)


def mode_forecast(result: dict, free_names: list[str],
                  diag: dict | None = None,
                  conditional: dict | None = None) -> dict:
    """Per-mode marginalized sigmas. result needs jac_bins (n_par, n_bins) whose rows
    are [free..., lnR0], sigma (n_bins,), and (optionally) seg (n_bins,) for the
    per-segment offset nuisances, cov (correlated-scenario covariance) and
    slope_rows (per-segment slope nuisances). ``diag``: see _marg_sigmas.
    Pass a dict as ``conditional`` to also receive the conditional sigmas
    (others fixed; see _conditional_sigmas) from the SAME Fisher matrix."""
    J = np.asarray(result["jac_bins"])
    steps = _segment_offset_rows(result)          # (n_steps, n_bins)
    slope = _slope_nuisance_rows(result)          # (n_slopes, n_bins)
    Jn = np.vstack([J] + [x for x in (steps, slope) if x.size])
    F = _fisher(Jn, result)
    if conditional is not None:
        conditional.update(zip(free_names,
                               _conditional_sigmas(F, len(free_names))))
    sig = _marg_sigmas(F, len(free_names), diag=diag)
    return dict(zip(free_names, sig))


def combined_forecast(results: list[dict], free_names: list[str],
                      diag: dict | None = None,
                      conditional: dict | None = None) -> dict:
    """All modes jointly: shared free params + shared lnR0 + one depth offset
    per detector SEGMENT (NRS1/NRS2 counted separately for the two-detector
    gratings) + each mode's scenario slope nuisances, all marginalized. Noise
    is block-diagonal across modes (each block the mode's scenario covariance
    or diagonal sigma; no cross-mode noise correlation is modeled).
    ``conditional``: as in mode_forecast."""
    n_f = len(free_names)
    # count nuisance columns: one offset per segment + this mode's slope rows
    seg_counts, slope_counts = [], []
    for r in results:
        nb = np.asarray(r["sigma"]).size
        seg = np.asarray(r.get("seg", np.zeros(nb, int)), int)
        seg_counts.append(int(seg.max()) + 1 if seg.size else 1)
        slope_counts.append(_slope_nuisance_rows(r).shape[0])
    n_nui = int(sum(seg_counts)) + int(sum(slope_counts))
    n_tot = n_f + 1 + n_nui                        # free + lnR0 + nuisances
    F = np.zeros((n_tot, n_tot))
    col = n_f + 1
    for r, n_seg, n_sl in zip(results, seg_counts, slope_counts):
        J = np.asarray(r["jac_bins"])             # rows: free..., lnR0
        nb = J.shape[1]
        seg = np.asarray(r.get("seg", np.zeros(nb, int)), int)
        Jg = np.zeros((n_tot, nb))
        Jg[:n_f] = J[:n_f]
        Jg[n_f] = J[n_f]                          # shared lnR0
        for s_id in range(n_seg):                 # this mode's per-segment offsets
            Jg[col + s_id] = (seg == s_id).astype(float)
        col += n_seg
        if n_sl:                                  # this mode's slope nuisances
            Jg[col:col + n_sl] = _slope_nuisance_rows(r)
            col += n_sl
        F += _fisher(Jg, r)
    if conditional is not None:
        conditional.update(zip(free_names, _conditional_sigmas(F, n_f)))
    sig = _marg_sigmas(F, n_f, diag=diag)
    return dict(zip(free_names, sig))


def transits_to_target(result: dict, free_names: list[str], gp: str,
                       target_display: float, sigma_at_transits,
                       co_eval: float | None = None) -> dict:
    """Smallest transit count at which the marginalized (display-unit) forecast on
    ``gp`` reaches ``target_display`` -- with the systematic floor respected.

    ``sigma_at_transits(result, n) -> per-bin sigma`` comes from detect.py
    (random variance scales 1/N; the minimum floor is a hard lower bound at
    every N). Returns dict(n=int|None, n_last=int|None, reachable=bool,
    sig_inf=float): ``sig_inf`` is the INFINITE-TRANSIT LIMIT of the scenario
    noise model in display units. Under the diagonal "random" scenario the
    forecast is monotone in N, so sig_inf is an exact best case and a target
    below it short-circuits to unreachable. Under a correlated scenario the
    floor-EXCESS systematic grows as photon noise averages down, the forecast
    can be BEST at a finite N, and reachability comes from the full scan
    (2026-07-15 audit -- the old sig_inf gate returned false "never"s there);
    ``n_last`` is then the largest scanned count still meeting the target.
    """
    from . import detect as _detect  # local import: fisher stays numpy-only otherwise

    def _sig_with(sigma, cov):
        r2 = dict(result)
        r2["sigma"] = sigma
        r2["cov"] = cov
        return display_sigma(gp, mode_forecast(r2, free_names)[gp], co_eval=co_eval)

    cov_inf = _detect.cov_at_transits(result, 1, floor_only=True)
    sig_inf = _sig_with(np.maximum(np.asarray(result["floor"]), 1e-30), cov_inf)
    if not np.isfinite(sig_inf):
        return dict(n=None, n_last=None, reachable=False, sig_inf=sig_inf)
    diagonal = cov_inf is None         # random scenario: monotone in N
    if diagonal and target_display < sig_inf:
        return dict(n=None, n_last=None, reachable=False, sig_inf=sig_inf)
    n_first = n_last = None
    for n in range(1, _detect.N_TRANSITS_CAP + 1):
        ok = _sig_with(sigma_at_transits(result, n),
                       _detect.cov_at_transits(result, n)) <= target_display
        if ok:
            if n_first is None:
                n_first = n
                if diagonal:           # monotone: smallest n is the answer
                    return dict(n=n, n_last=None, reachable=True,
                                sig_inf=sig_inf)
            n_last = n
    return dict(n=n_first, n_last=n_last, reachable=n_first is not None,
                sig_inf=sig_inf)
