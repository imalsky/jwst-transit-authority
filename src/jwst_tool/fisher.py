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
Jacobi-whitened (unit-diagonal) Fisher matrix (see _whitened_eig).
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
# Whitening makes the rank decision invariant under per-parameter unit changes
# -- never threshold the raw mixed-unit matrix. It is NOT invariant under
# arbitrary MIXED reparameterizations near the threshold: quote the whitened
# spectrum (the ``diag`` output) where that matters. eigh's noise floor is
# ~1e-16 x wmax, so 1e-10 keeps 6 decades of margin either way.
REL_EIG_TOL = 1e-10
# A parameter whose L2 projection onto the null subspace exceeds this reads
# inf. Basis-invariant subspace norm, never a single eigenvector's component.
NULL_LOAD_TOL = 1e-6

# Widths at or past which no number is reported, in the FITTED coordinate.
# This is a CHOSEN policy threshold, not a derived one: sigma_lnCO = 1 is one
# e-fold each way, which is where a C/O width stops being worth printing. It
# also guards the numerics -- at sigma_lnCO of 41-920 (niriss_soss_ord2, see
# notes.md) a shifted mock center collapses the curve grid. It is NOT the
# reporting gate: format_co_width decides separately whether a PHYSICAL range
# can be quoted, from the solver's own supported band, and keeps the dex
# width either way.
UNINFORMATIVE_SIGMA = {"dlnCO": 1.0}


def _cut_uninformative(sigmas: dict) -> dict:
    """inf for any width at or past its parameter's no-information scale
    (UNINFORMATIVE_SIGMA); a name absent from that table passes through."""
    return {n: (np.inf if float(s) >= UNINFORMATIVE_SIGMA.get(n, np.inf)
                else float(s))
            for n, s in sigmas.items()}


def _whitened_eig(F: np.ndarray) -> tuple:
    """The ONE Jacobi whitening + rank decision behind _marg_sigmas and
    _whitened_solve, so a direction that is null in one is null in the other.

    Returns (n, d, nz, w, V, good): F's dimension, the sqrt-diagonal scales
    d_i = sqrt(F_ii), the mask of parameters with d > 0, and the eigenpairs
    of the whitened (unit-diagonal) submatrix with the REL_EIG_TOL keep mask.
    A matrix with no positive diagonal returns empty w/V and an all-False
    ``good``; callers turn that into their own unconstrained value.
    """
    F = np.asarray(F, float)
    F = 0.5 * (F + F.T)
    n = F.shape[0]
    d = np.sqrt(np.clip(np.diag(F), 0.0, None))
    nz = d > 0.0
    if not nz.any():
        return n, d, nz, np.zeros(0), np.zeros((0, 0)), np.zeros(0, bool)
    Fw = F[np.ix_(nz, nz)] / np.outer(d[nz], d[nz])
    w, V = np.linalg.eigh(0.5 * (Fw + Fw.T))
    wmax = float(w[-1]) if w.size else 0.0
    return n, d, nz, w, V, w > REL_EIG_TOL * max(wmax, 1e-300)


def _marg_sigmas(F: np.ndarray, n_report: int,
                 diag: dict | None = None) -> np.ndarray:
    """Rank-aware marginalized sigmas for the first n_report parameters of F.

    Sigmas are read in the whitened coordinates of _whitened_eig and
    transformed back, so a full-rank matrix reproduces inv(F) exactly.
    F_ii == 0 or a null-direction load reads inf. Pass a dict as ``diag`` to
    receive rank / dimension / condition number / whitened eigenvalues.
    """
    n, d, nz, w, V, good = _whitened_eig(F)
    out = np.full(n_report, np.inf)
    if diag is not None:
        wmax = float(w[-1]) if w.size else 0.0
        diag.update(
            fisher_dimension=n,
            fisher_rank=int(good.sum()),
            condition_number=(wmax / float(w[good].min()) if good.any()
                              else float("inf")),
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
    n, d, nz, w, V, good = _whitened_eig(F)
    b = np.asarray(b, float)
    if b.shape != (n,):
        raise ValueError(f"_whitened_solve: b has shape {b.shape}, expected "
                         f"({n},)")
    if not np.all(np.isfinite(b)):
        raise ValueError("_whitened_solve: b contains non-finite values")
    out = np.full(n_report, np.nan)
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


# Display SYMBOL, UNIT and friendly name per free parameter. The symbol
# MUST match the unit's log base: metallicity and Kzz are dex (log10), so
# [M/H] and log Kzz, never "ln"; C/O is the number ratio N_C/N_O (no unit).
PARAM_SYMBOLS = {"lnZ": "[M/H]", "dlnCO": "C/O", "lnKzz": "log Kzz",
                 "Tirr": "T_irr", "Tint": "T_int",
                 "log_kappa": "log κ_IR", "log_gamma": "log γ",
                 "log_kappa_cloud": "log κ_cloud", "alpha_cloud": "α_cloud"}
PARAM_UNITS = {"lnZ": "dex", "dlnCO": "", "lnKzz": "dex",
               "Tirr": "K", "Tint": "K",
               "log_kappa": "dex", "log_gamma": "dex",
               "log_kappa_cloud": "dex", "alpha_cloud": ""}
PARAM_LABELS = {"lnZ": "Metallicity", "dlnCO": "C/O ratio",
                "lnKzz": "Vertical mixing (Kzz)",
                "Tirr": "Guillot T_irr",
                "Tint": "Guillot T_int", "log_kappa": "Guillot log κ_IR",
                "log_gamma": "Guillot log γ",
                "log_kappa_cloud": "Cloud deck log κ (at 3.5 um)",
                "alpha_cloud": "Cloud deck slope α"}


def param_axis(name: str) -> str:
    """Axis/column label for a parameter: 'Symbol [unit]', or bare 'Symbol'
    when it is dimensionless (e.g. '[M/H] [dex]', 'C/O', 'T_irr [K]')."""
    u = PARAM_UNITS[name]
    return f"{PARAM_SYMBOLS[name]} [{u}]" if u else PARAM_SYMBOLS[name]


def display_sigma(name: str, sigma: float, co_eval: float | None = None) -> float:
    """Internal-unit sigma -> display-unit sigma.

    C/O (``dlnCO``) is reported as an ABSOLUTE number ratio: the internal sigma
    is in delta-ln(C/O) and the local Jacobian is exactly C/O, so
    sigma_CO = C/O * sigma_lnCO is the exact Jacobian rescaling of the local
    Fisher width -- NOT the standard deviation of the nonlinear transformed
    distribution, and not what any surface prints. It survives only because
    meta["target_prec"] is a user-entered target in absolute C/O; display goes
    through format_co_width. ``co_eval`` is the atmosphere's C/O at the
    evaluation point (params_json co_ratio); it is REQUIRED for dlnCO (a loud
    error beats a silently-wrong scale)."""
    if name == "dlnCO":
        if co_eval is None:
            raise ValueError(
                "display_sigma('dlnCO', ...) needs co_eval (the atmosphere's C/O) "
                "to report an absolute C/O uncertainty: sigma_CO = C/O * sigma_lnCO.")
        return sigma * co_eval
    return sigma * _TO_DISPLAY.get(name, 1.0)


def co_interval(center: float, sigma_ln: float, k: float = 1.0) -> tuple:
    """(lo, hi) physical C/O at k sigma: center * exp(+-k*sigma_ln).

    Positive by construction. These are the k-sigma quantiles of the
    log-coordinate Gaussian the panel draws -- exact UNDER that local
    approximation, which is a quadratic expansion at the input C/O. They are
    not a globally evaluated credible interval; see format_co_width."""
    center, w = float(center), float(sigma_ln) * float(k)
    if not (np.isfinite(center) and center > 0.0):
        raise ValueError(f"co_interval: center must be a finite C/O > 0, "
                         f"got {center!r}")
    if not (np.isfinite(w) and w > 0.0):
        raise ValueError(f"co_interval: sigma_ln*k must be finite and > 0, "
                         f"got {w!r}")
    return (center * np.exp(-w), center * np.exp(w))


def format_co_width(center: float, sigma_ln: float, bounds: tuple,
                    k: float = 1.0, qualify_coord: bool = False) -> str:
    """The C/O width string. dex is PRIMARY -- it is the coordinate the Fisher
    Gaussian lives in -- and is ALWAYS reported when finite: a wide dex width
    is a weak local constraint, not the absence of one.

    Independently, the physical C/O range is appended only when the center AND
    the interval stay inside ``bounds`` (forward.co_bounds(network,
    use_photo)). Outside it the forward model cannot be evaluated, so a mapped
    range there would extrapolate a local derivative; the width is marked
    "(local)" instead. The CENTER is not required to sit in the band --
    mock_center_co shifts a recovered center multiplicatively and a broad mode
    lands outside it legitimately, which is a local result, never an error.

    ``qualify_coord`` names the coordinate the dex belongs to, for surfaces
    with no C/O axis beside them (the constraint table). PLAIN TEXT, never
    mathtext: this string also lands in a Streamlit table cell and a CSV."""
    lo_b, hi_b = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(lo_b) and np.isfinite(hi_b) and 0.0 < lo_b < hi_b):
        raise ValueError(f"format_co_width: bounds must be finite, positive "
                         f"and ordered, got {bounds!r}")
    center = float(center)
    if not (np.isfinite(center) and center > 0.0):
        raise ValueError(f"format_co_width: center must be a finite C/O > 0, "
                         f"got {center!r}")
    sig = float(sigma_ln) * float(k) / _LN10
    if not qualify_coord:
        # Panel legend, Batalha & Line's convention: the value AND its error
        # in log10(C/O) -- "Error on log C/O", their cases tagged
        # "logC/O = -0.26" for C/O 0.55. Naming the coordinate is what
        # licenses the "+-": the Gaussian is symmetric there and skewed in
        # linear C/O. A bare number beside a C/O axis would be read as a
        # linear-C/O width, which is off by ln 10.
        return f"log10(C/O) = {np.log10(center):.3g} \u00b1 {sig:.3g}"
    lo, hi = co_interval(center, sigma_ln, k=k)
    base = f"\u00b1{sig:.3g} in log10(C/O)"
    if lo_b <= center <= hi_b and lo_b <= lo and hi <= hi_b:
        return f"{base} (C/O {lo:.3g}\u2013{hi:.3g})"
    return f"{base} (local)"


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
    Jn = np.vstack([J, _segment_offset_rows(result)])
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
                       target_display: float,
                       co_eval: float | None = None) -> dict:
    """Smallest transit count at which the marginalized display-unit forecast
    on ``gp`` reaches ``target_display``, with the systematic floor respected.

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
    if not _detect.has_floor(result):
        # No floor: nothing caps the precision, so report 0.0 -- the
        # 1e-30-clipped value would be a spurious reachability gate.
        sig_inf = 0.0
    else:
        sig_inf = _sig_with(np.maximum(_floor, 1e-30))
        if not np.isfinite(sig_inf):
            return dict(n=None, reachable=False, sig_inf=sig_inf)
    if target_display < sig_inf:
        return dict(n=None, reachable=False, sig_inf=sig_inf)
    for n in range(1, _detect.N_TRANSITS_CAP + 1):
        if _sig_with(_detect.sigma_at_transits(result, n)) <= target_display:
            return dict(n=n, reachable=True, sig_inf=sig_inf)
    return dict(n=None, reachable=False, sig_inf=sig_inf)
