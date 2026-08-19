"""Fisher-information parameter forecast from the spectrum Jacobian.

Given each mode's binned Jacobian J (n_par, n_bins) and per-bin depth sigma,

    F_ij = sum_b J_ib J_jb / sigma_b^2

and the marginalized 1-sigma forecast on parameter i is sqrt((F^-1)_ii).

Nuisances, jointly fit and marginalized out of the report:
  * per mode: lnR0 (reference radius) + one constant depth OFFSET per detector
    SEGMENT (NRS1/NRS2 separately for the two-detector gratings -- a
    detector-to-detector step of tens of ppm is universal in real fits and can
    otherwise masquerade as atmospheric structure).
  * combined: one SHARED lnR0 + one offset per SEGMENT across all modes.

A parameter with no spectral response, loaded on a numerically null Fisher
direction, or wider than its no-information scale (UNINFORMATIVE_SIGMA),
comes back inf ("unconstrained" in the GUI). The inversion is
ALWAYS rank-aware and unit-invariant: rank detection happens on the
Jacobi-whitened (unit-diagonal) Fisher matrix (see _marg_sigmas).
np.linalg.inv returns misleading finite numbers on ill-conditioned matrices,
so it is never used. Forecast sigmas are local Cramer-Rao lower bounds under
the quoted noise model -- best cases, not posterior widths (Vallisneri 2008).
"""
from __future__ import annotations

import numpy as np

_LN10 = np.log(10.0)

# sigma in internal ln-units -> display units: lnZ/lnKzz report in dex. C/O
# is handled in display_sigma (its factor is the atmosphere's absolute C/O).
_TO_DISPLAY = {"lnZ": 1.0 / _LN10, "lnKzz": 1.0 / _LN10}

# Relative eigenvalue threshold on the WHITENED (unit-diagonal) Fisher matrix:
# directions below REL_EIG_TOL x the largest whitened eigenvalue are null.
# Whitening makes the rank decision invariant under per-parameter unit
# changes; thresholding the raw mixed-unit matrix flipped constraints under a
# pure K-vs-kK rescaling. It is NOT invariant under arbitrary MIXED
# reparameterizations near the threshold -- quote the whitened spectrum (the
# ``diag`` output) where it matters. eigh's noise floor is ~1e-16 x wmax;
# 1e-10 keeps 6 decades of margin either way.
REL_EIG_TOL = 1e-10
# A parameter whose L2 projection onto the null subspace exceeds this reads
# inf. Basis-invariant subspace norm, never a single eigenvector's component.
NULL_LOAD_TOL = 1e-6

# Widths at or past which the reported number is not a constraint, in the
# FITTED coordinate. dlnCO is the only row whose display transform is a
# LINEARIZATION (display_sigma: sigma_CO = C/O * sigma_lnCO); every other row
# converts exactly (identity, or 1/ln10 for dex) and reports its width
# verbatim however large it is. At sigma_lnCO = 1 that linearization has run
# out: the symmetric display Gaussian's minus-1-sigma edge sits exactly at the
# physical boundary C/O = 0, so "C/O = 0.55 +/- 0.55" and anything past it
# describes the boundary, not a measurement. Such a direction reads
# UNCONSTRAINED (inf), the same as a numerically null one -- never a number
# that looks like a weak measurement. The value is derived, not tuned: it is
# where sigma_CO reaches C/O.
UNINFORMATIVE_SIGMA = {"dlnCO": 1.0}


def _cut_uninformative(sigmas: dict) -> dict:
    """inf for any width at or past its parameter's no-information scale
    (UNINFORMATIVE_SIGMA); a name absent from that table passes through."""
    return {n: (np.inf if float(s) >= UNINFORMATIVE_SIGMA.get(n, np.inf)
                else float(s))
            for n, s in sigmas.items()}


def _marg_sigmas(F: np.ndarray, n_report: int,
                 diag: dict | None = None) -> np.ndarray:
    """Rank-aware marginalized sigmas for the first n_report parameters of F.

    Rank detection happens in Jacobi-whitened coordinates (q_i = theta_i *
    sqrt(F_ii), unit-diagonal correlation form), invariant under
    per-parameter rescaling; sigmas transform back afterwards and a full-rank
    matrix reproduces inv(F) exactly. F_ii == 0 or a null-direction load
    reads inf. Pass a dict as ``diag`` to receive rank / dimension /
    condition number / whitened eigenvalues.
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
        # L2 projection onto the null subspace (basis-invariant)
        load = np.sqrt(np.sum(V[:, ~good] ** 2, axis=1))
        sig_nz[load > NULL_LOAD_TOL] = np.inf
    full = np.full(n, np.inf)
    full[nz] = sig_nz
    out[:] = full[:n_report]
    return out


def _whitened_solve(F: np.ndarray, b: np.ndarray,
                    n_report: int) -> np.ndarray:
    """Rank-aware solve of F x = b for the first ``n_report`` entries, using
    the SAME Jacobi whitening and relative eigen-threshold as _marg_sigmas
    (so a direction that reads inf there reads NaN here, never a number from
    an inverted null eigenvalue). Used by the linearized single-realization
    mock recovery (posteriors.mock_recovery): x = F^+ b with b = J W n.

    Zero-diagonal parameters and parameters loaded on the null subspace
    (same NULL_LOAD_TOL projection test) come back NaN -- an unconstrained
    direction has no recovered value by design.
    """
    F = np.asarray(F, float)
    F = 0.5 * (F + F.T)
    b = np.asarray(b, float)
    n = F.shape[0]
    if b.shape != (n,):
        raise ValueError(f"_whitened_solve: b has shape {b.shape}, expected "
                         f"({n},)")
    if not np.all(np.isfinite(b)):
        raise ValueError("_whitened_solve: b contains non-finite values")
    out = np.full(n_report, np.nan)
    d = np.sqrt(np.clip(np.diag(F), 0.0, None))
    nz = d > 0.0
    if not nz.any():
        return out
    Fw = F[np.ix_(nz, nz)] / np.outer(d[nz], d[nz])
    w, V = np.linalg.eigh(0.5 * (Fw + Fw.T))
    wmax = float(w[-1]) if w.size else 0.0
    good = w > REL_EIG_TOL * max(wmax, 1e-300)
    if not good.any():
        return out
    bw = b[nz] / d[nz]
    xw = V[:, good] @ ((V[:, good].T @ bw) / w[good])
    x_nz = xw / d[nz]
    if (~good).any():
        load = np.sqrt(np.sum(V[:, ~good] ** 2, axis=1))
        x_nz[load > NULL_LOAD_TOL] = np.nan
    full = np.full(n, np.nan)
    full[nz] = x_nz
    out[:] = full[:n_report]
    return out


def _conditional_sigmas(F: np.ndarray, n_report: int) -> np.ndarray:
    """Conditional sigmas 1/sqrt(F_ii) for the first n_report parameters
    (every other parameter and nuisance held fixed). Always <= marginalized;
    a large gap flags degeneracy, not missing spectral response. F_ii == 0
    reads inf. Pure diagonal read-off, no inversion."""
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
    """One constant-offset indicator row per detector segment INCLUDING
    segment 0 (n_seg rows), from result["seg"].

    lnR0 is a physical derivative, not a constant in wavelength, so it never
    stands in for the first segment's offset; any exact redundancy between
    rows is what the rank-aware _marg_sigmas handles. With every segment
    present, mode_forecast(r) implements the same model as
    combined_forecast([r]) (pinned by test)."""
    nb = np.asarray(result["sigma"]).size
    seg = np.asarray(result.get("seg", np.zeros(nb, int)), int)
    n_seg = int(seg.max()) + 1 if seg.size else 1
    return np.stack([(seg == s).astype(float) for s in range(n_seg)])


def _fisher(Jn: np.ndarray, result: dict) -> np.ndarray:
    """J W J^T under the mode's diagonal noise model, W = diag(1/sigma^2)."""
    s = np.asarray(result["sigma"])
    return (Jn / s[None, :] ** 2) @ Jn.T


def mode_forecast(result: dict, free_names: list[str],
                  diag: dict | None = None,
                  conditional: dict | None = None) -> dict:
    """Per-mode marginalized sigmas. result needs jac_bins (n_par, n_bins) whose rows
    are [free..., lnR0], sigma (n_bins,), and (optionally) seg (n_bins,) for the
    per-segment offset nuisances. ``diag``: see _marg_sigmas.
    Pass a dict as ``conditional`` to also receive the conditional sigmas
    (others fixed; see _conditional_sigmas) from the SAME Fisher matrix.
    Both sets pass the no-information cut (_cut_uninformative)."""
    J = np.asarray(result["jac_bins"])
    steps = _segment_offset_rows(result)          # (n_steps, n_bins)
    Jn = np.vstack([J, steps]) if steps.size else J
    F = _fisher(Jn, result)
    if conditional is not None:
        conditional.update(_cut_uninformative(
            dict(zip(free_names, _conditional_sigmas(F, len(free_names))))))
    sig = _marg_sigmas(F, len(free_names), diag=diag)
    return _cut_uninformative(dict(zip(free_names, sig)))


def _combined_system(results: list[dict], n_f: int) -> tuple[int, list]:
    """The combined stacked design shared by combined_forecast and the
    linearized mock recovery (posteriors.mock_recovery): global parameter
    layout [free..., shared lnR0, per-mode per-segment offsets...].

    Returns ``(n_tot, [(Jg, r), ...])`` with one augmented Jacobian
    (n_tot, n_bins) per result, in input order. Keeping this in ONE place is
    what guarantees the recovery offsets are consistent with the
    marginalized sigmas -- never re-derive the layout elsewhere.
    """
    seg_counts = []
    for r in results:
        nb = np.asarray(r["sigma"]).size
        seg = np.asarray(r.get("seg", np.zeros(nb, int)), int)
        seg_counts.append(int(seg.max()) + 1 if seg.size else 1)
    n_tot = n_f + 1 + int(sum(seg_counts))         # free + lnR0 + offsets
    out = []
    col = n_f + 1
    for r, n_seg in zip(results, seg_counts):
        J = np.asarray(r["jac_bins"])             # rows: free..., lnR0
        nb = J.shape[1]
        seg = np.asarray(r.get("seg", np.zeros(nb, int)), int)
        Jg = np.zeros((n_tot, nb))
        Jg[:n_f] = J[:n_f]
        Jg[n_f] = J[n_f]                          # shared lnR0
        for s_id in range(n_seg):                 # this mode's per-segment offsets
            Jg[col + s_id] = (seg == s_id).astype(float)
        col += n_seg
        out.append((Jg, r))
    return n_tot, out


def combined_forecast(results: list[dict], free_names: list[str],
                      diag: dict | None = None,
                      conditional: dict | None = None) -> dict:
    """All modes jointly: shared free params + shared lnR0 + one depth offset
    per detector SEGMENT (NRS1/NRS2 counted separately for the two-detector
    gratings), all marginalized. Noise is block-diagonal across modes
    (diagonal sigma; no cross-mode noise correlation is modeled).
    ``conditional`` and the no-information cut: as in mode_forecast."""
    n_f = len(free_names)
    n_tot, system = _combined_system(results, n_f)
    F = np.zeros((n_tot, n_tot))
    for Jg, r in system:
        F += _fisher(Jg, r)
    if conditional is not None:
        conditional.update(_cut_uninformative(
            dict(zip(free_names, _conditional_sigmas(F, n_f)))))
    sig = _marg_sigmas(F, n_f, diag=diag)
    return _cut_uninformative(dict(zip(free_names, sig)))


def transits_to_target(result: dict, free_names: list[str], gp: str,
                       target_display: float, sigma_at_transits,
                       co_eval: float | None = None) -> dict:
    """Smallest transit count at which the marginalized display-unit forecast
    on ``gp`` reaches ``target_display``, with the systematic floor respected.

    ``sigma_at_transits(result, n) -> per-bin sigma`` comes from detect.py.
    Returns dict(n, reachable, sig_inf); ``sig_inf`` is the infinite-transit
    (floor-only) limit in display units. The forecast is monotone in N
    (diagonal noise), so a target below sig_inf short-circuits to
    unreachable. With no floor set anywhere, sig_inf is 0.0 and
    unreachability then means the target needs more than N_TRANSITS_CAP
    transits.
    """
    from . import detect as _detect  # local import: fisher stays numpy-only otherwise

    def _sig_with(sigma):
        r2 = dict(result)
        r2["sigma"] = sigma
        return display_sigma(gp, mode_forecast(r2, free_names)[gp], co_eval=co_eval)

    _floor = np.asarray(result["floor"])
    if not np.any(_floor > 0.0):
        # No floor: nothing caps the precision. Reporting the 1e-30-clipped
        # value here produced a spurious reachability gate.
        sig_inf = 0.0
    else:
        sig_inf = _sig_with(np.maximum(_floor, 1e-30))
        if not np.isfinite(sig_inf):
            return dict(n=None, reachable=False, sig_inf=sig_inf)
    if target_display < sig_inf:
        return dict(n=None, reachable=False, sig_inf=sig_inf)
    for n in range(1, _detect.N_TRANSITS_CAP + 1):
        if _sig_with(sigma_at_transits(result, n)) <= target_display:
            return dict(n=n, reachable=True, sig_inf=sig_inf)
    return dict(n=None, reachable=False, sig_inf=sig_inf)
