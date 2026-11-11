"""The Fisher Jacobian of a run: FD stencils, warm AD chemistry rows, RT-only rows and lnR0.

Method: docs/forward_model.md, section `forward`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from jwst_tool.certify import (
    NotCertified,
    certified_solve,
    check_ad_co_margin,
    check_converged,
    check_elements,
    check_emission_thin,
    check_t_window,
    gate_unmodeled,
    merge_unmodeled,
    table_species,
)
from jwst_tool.model_defs import (
    CHEM_PARAM_NAMES,
    CLOUD_FISHER_PARAMS,
    CO_MAX,
    CO_MAX_PHOTO_OFF,
    FD_COMP_PARAMS,
    FD_STEPS,
    PICASO_MET_RANGE,
    fd_stencil,
)
from jwst_tool.spectrum import ColumnSpectrum, RtSetup

if TYPE_CHECKING:
    from collections.abc import Callable

    from jwst_tool.tool_types import AnyArray, Log

# The h-vs-2h gate, max |J_h - J_2h| / max |J_h|: measured rows on both sides of it are in notes "Jacobians and Fisher".
FD_CONSISTENCY_TOL = 0.25


FD_LNR0_STEP = 0.01  # lnR0 is RT-only (smooth, analytic)


# The build an AD row's warm re-converge runs on. count_max bounds the cost;
# dt_max (s) stops the adaptive step running away on a photolysis-driven
# column. DERIVATIVE builds only: the cold solve and the FD stencil points keep
# DT_MAX_S, so no shipped spectrum moves.
AD_BUILD_OVERRIDES = {"count_max": 6000, "dt_max": 1.0e5}


# The cadence-1 retry gets the cold step budget: a slowly settling tangent
# needs it. The first attempt keeps the 6000 cap above, so a genuine stall
# still escalates after 6000 steps, not 30000.
AD_RETRY_COUNT_MAX = 30000


def fd_estimates(offs: tuple, dvals: dict, f0: np.ndarray | None, h: float) -> tuple[np.ndarray, np.ndarray]:
    """(J_h, J_2h) from stencil values dvals[s] = f(x + s h).

    Central when offs holds +-1, else the one-sided stencil anchored on f0 = f(x). The Richardson combination (4 J_h
    - J_2h) / 3 is exact for cubics either way.
    """
    if 1 in offs and -1 in offs:
        return ((dvals[1] - dvals[-1]) / (2.0 * h), (dvals[2] - dvals[-2]) / (4.0 * h))
    d = offs[0]
    return (
        d * (-3.0 * f0 + 4.0 * dvals[d] - dvals[2 * d]) / (2.0 * h),  # type: ignore[operator]  # a one-sided stencil carries f0
        d * (-3.0 * f0 + 4.0 * dvals[2 * d] - dvals[4 * d]) / (4.0 * h),  # type: ignore[operator]  # a one-sided stencil carries f0
    )


def fd_row(name: str, j1: np.ndarray, j2: np.ndarray, h: float) -> tuple[np.ndarray, float]:
    """Richardson-combined Jacobian row and its h-vs-2h consistency error.

    j1 / j2 are the same scheme's estimate at step h and 2h. Raises when the
    two disagree beyond FD_CONSISTENCY_TOL: an uncertified derivative is never
    reported. A row with no spectral response is an exact zero, error 0.
    """
    if not (np.isfinite(j1).all() and np.isfinite(j2).all()):
        raise RuntimeError(f"FD Jacobian for {name}: non-finite entries")
    scale = float(np.max(np.abs(j1)))
    if scale == 0.0:
        # The no-response escape needs BOTH estimates to vanish: with no scale
        # to divide by there is no h-vs-2h certificate, and an uncertified
        # derivative is never reported however plausible its Richardson value.
        if np.all(j2 == 0.0):
            return j1, 0.0
        raise RuntimeError(
            f"FD Jacobian for {name} FAILED the step-size consistency check: "
            f"max|J(h)| is exactly zero but max|J(2h)| = "
            f"{float(np.max(np.abs(j2))):.3e} (h = {h:g}). The h-vs-2h "
            "certificate is undefined at zero scale, and an uncertified "
            "derivative is never reported."
        )
    err = float(np.max(np.abs(j1 - j2)) / scale)
    if err > FD_CONSISTENCY_TOL:
        raise RuntimeError(
            f"FD Jacobian for {name} FAILED the step-size consistency "
            f"check: max|J(h) - J(2h)| / max|J(h)| = {err:.3f} > "
            f"{FD_CONSISTENCY_TOL} (h = {h:g}), so the row is dominated by "
            "solver convergence noise or curvature, and an uncertified "
            f"derivative is never reported. Drop {name} from the Fisher "
            "parameters, or move its value away from a range edge; on the "
            "VULCAN-JAX engine a tighter convergence (lower yconv_min, with "
            "yconv_cri) or a larger nz can also certify it."
        )
    return (4.0 * j1 - j2) / 3.0, err  # Richardson: O(h^4) central, O(h^3) one-sided


@dataclass
class Baseline:
    """The certified baseline a run's Jacobian rows are built around."""

    cp: dict
    pic: bool
    theta: Any
    theta_names: list
    th0: Any
    build_chem: Any
    abundance_overrides: Any
    assemble: Any
    setup: RtSetup
    chem: Any
    column: ColumnSpectrum
    y_sol: Any
    species: list
    run_frq: int | None
    escalated: list
    bz_build: float
    advance: Callable[[], None]
    log: Log
    # (low, high) bar: the photosphere the baseline's absorber gate used, applied to every column the rows solve
    unmodeled_band: tuple[float, float]
    # the run's (species, VMR) absorbers with no k-table above the warning level, every column's merged
    unmodeled: list
    # PICASO only: (stage, layer solves retried at a shifted T) of every column the run solves; the T-P stencil
    # points' own assemblies add theirs here
    picaso_retried: list = field(default_factory=list)


@dataclass
class Rows:
    """The Jacobian as it fills: one entry per row, in jac_names order, with lnR0 last."""

    names: list
    jac: np.ndarray
    fd_h: list = field(default_factory=list)
    fd_err: list = field(default_factory=list)
    method: list = field(default_factory=list)
    bz_warm: float = float("nan")

    def add(self, j: int, row: Any, h: float, err: float, method: str) -> None:
        """Record row j with its step, its h-vs-2h error (NaN when unmeasured) and how it was computed."""
        self.jac[j] = row
        self.fd_h.append(h)
        self.fd_err.append(err)
        self.method.append(method)


def _gate_emis(b: Baseline, dep: AnyArray, tau: AnyArray, stage: str) -> AnyArray:
    """Thin-bottom gate for a perturbed emission spectrum.

    `dep * fs_j` is Fp times the eclipse prefactor, and the fraction is a ratio, so the scalar cancels.
    """
    check_emission_thin(np.asarray(tau), np.asarray(dep) * np.asarray(b.setup.fs_j), b.setup.rt.nu_grid, stage)
    return dep


def _check_unmodeled(b: Baseline, y: AnyArray, chem_b: Any, stage: str) -> None:
    """Apply the baseline's absorber gate to a column a row solved, and carry its warnings to the run's result.

    No absorber without a k-table may be bulk there; one above the warning level joins ``b.unmodeled``.
    """
    species = [sp for sp, _ in sorted(chem_b.sidx.items(), key=lambda kv: kv[1])]
    tables = table_species(b.setup.config.MOLECULES)
    y, p_bar = np.asarray(y), np.asarray(chem_b.p_bar)
    found = gate_unmodeled(species, y, p_bar, tables, b.unmodeled_band, stage, b.log)
    if found:
        b.log(
            f"[fwd] WARNING: {stage}: the column carries "
            + ", ".join(f"{sp} at {v:.2e}" for sp, v in found)
            + " (mean VMR over the photosphere band) with NO published k-table; recorded with the result"
        )
        merge_unmodeled(b.unmodeled, found)


def _certified_depth(
    b: Baseline, chem_b: Any, th: AnyArray, stage: str, tp_eval_b: Callable | None = None
) -> np.ndarray:
    """One stencil point: the chemistry certificate AND, in emission, the thin-bottom certificate.

    A Fisher row is never built from a spectrum neither gate has passed. A point that does not certify RAISES -- the
    escalation is a row-level retry, never per point.
    """
    import jax.numpy as jnp

    if b.pic:  # a PICASO build is an already-certified column
        y_b: AnyArray = jnp.asarray(chem_b.y)
    else:
        y_b, chem_b, _, _ = certified_solve(chem_b, th, stage, log=b.log)
        check_elements(np.asarray(y_b), chem_b, stage, b.log)
    _check_unmodeled(b, y_b, chem_b, stage)
    dep_fn = ColumnSpectrum(b.setup, chem_b, tp_eval_b).depth
    if b.setup.emis is None:
        return np.asarray(dep_fn(y_b, jnp.asarray(th)))
    dep, tau = dep_fn(y_b, jnp.asarray(th), return_tau=True)
    return np.asarray(_gate_emis(b, dep, tau, stage))


def _ad_row(b: Baseline, chem_x: Any, e_h: np.ndarray) -> tuple:
    # Warm continuation from the converged column, differentiated
    # along `e_h` (a unit direction times its FD step: the solver
    # certifies the tangent per cell in those units, together with
    # the column it belongs to; a plain jax.jvp stops when the column
    # certifies, which from a converged start is at count_min with
    # the tangent unrelaxed). The spectrum tangent follows
    # by the chain rule from (y_w, dy_w) and the direct theta
    # dependence of the RT. Returns (depth, diag, tau, y_w, d depth).
    import jax
    import jax.numpy as jnp

    y_w, dy_w, diag = chem_x.converged_y_jvp(b.th0, e_h, warm_y=b.y_sol, lnZ_ref=0.0, c_o_ref=0.0)
    e_j = jnp.asarray(e_h)
    if b.setup.emis is None:
        dep, ddep = jax.jvp(b.column.depth, (y_w, b.th0), (dy_w, e_j))
        return dep, diag, None, y_w, ddep
    (dep, tau), (ddep, _) = jax.jvp(lambda y, th: b.column.depth(y, th, return_tau=True), (y_w, b.th0), (dy_w, e_j))
    return dep, diag, tau, y_w, ddep


def _ad_chemistry_rows(b: Baseline, rows: Rows) -> dict[str, np.ndarray]:
    """Return the AD chemistry rows by name, one tangent-certified warm jvp per row."""
    jac_names = rows.names
    # AD differentiates the CHEMISTRY rows only. A T-P row takes the FD
    # branch under either method: the warm jvp runs at the baseline
    # r_anchor/g_anchor, while a T-P step moves that anchor and with it
    # the Guillot mapping, which only a rebuilt stencil point carries.
    ad_chem_rows = [n for n in jac_names if n in CHEM_PARAM_NAMES] if b.cp["jac_method"] == "ad" else []
    ad_cols = {}
    if "dlnCO" in ad_chem_rows:
        # The build-time co_bz_bound is a proxy (the equilibrium seed
        # column); the seed map recomputes the O split from the converged
        # column the tangent starts from, so gate on that column too.
        rows.bz_warm = check_ad_co_margin(
            b.chem, b.cp["co_ratio"], y=np.asarray(b.y_sol), build_margin=b.bz_build, log=b.log
        )
    if ad_chem_rows:
        # One tangent-certified jvp per chemistry-theta row, NEVER vmap
        # over the tangent directions: the batched tangent through the
        # solver's while_loop is NaN in every bin on a column with
        # clamped-zero layers, while the unbatched jvp is finite. Each row
        # certifies its own warm re-converge and its own tangent.
        # That warm re-converge is where a shielded column stalls at the
        # config's photolysis cadence, so it runs on a build
        # with AD_BUILD_OVERRIDES (step cap + dt cap). A row that does not
        # certify there is re-solved on a cadence-1 build with the cold
        # step budget (AD_RETRY_COUNT_MAX) and flagged in
        # `photo_escalated`, in either science mode. That fallback build is
        # SHARED: once one row fails to certify at the config cadence, the
        # remaining rows run on the cadence-1 build directly and are all
        # flagged. Only the certificate is inside the try: a backend
        # failure propagates.
        chem_ad = b.build_chem(tag="AD rows", photo_frq=b.run_frq, solver=AD_BUILD_OVERRIDES)
        chem_ad1 = None
        t1 = time.time()
        b.advance()
        for _n in ad_chem_rows:
            e = np.zeros(b.theta.size)
            e[b.theta_names.index(_n)] = FD_STEPS[_n]  # the certificate's units
            stage = f"AD warm re-converge ({_n})"
            out = None
            if chem_ad1 is None:  # the shared primal has not stalled yet
                out = _ad_row(b, chem_ad, e)
                try:
                    check_converged(out[1], stage, b.species, chem_ad, b.log)
                except NotCertified as exc:
                    if b.run_frq == 1:
                        raise
                    b.log(f"[fwd] {stage}: {exc}")
                    out = None
            if out is None:
                b.log(
                    f"[fwd] {stage}: re-solving with photolysis refreshed "
                    f"every accepted step (step cap {AD_RETRY_COUNT_MAX})"
                )
                if chem_ad1 is None:
                    chem_ad1 = b.build_chem(
                        tag="AD rows [photo cadence 1]",
                        photo_frq=1,
                        solver={**AD_BUILD_OVERRIDES, "count_max": AD_RETRY_COUNT_MAX},
                    )
                stage += " [photo cadence 1]"
                out = _ad_row(b, chem_ad1, e)
                check_converged(out[1], stage, b.species, chem_ad1, b.log)
                b.escalated.append(f"AD {_n} row")
            pd, _pdiag, ptau, py, dd = out
            check_elements(np.asarray(py), b.chem, stage, b.log)
            _check_unmodeled(b, py, b.chem, stage)
            # the jvp's own primal, not y_sol: certify the point the
            # tangent is actually taken at
            if b.setup.emis is not None:
                _gate_emis(b, pd, ptau, stage)
            ad_cols[_n] = np.asarray(dd) / FD_STEPS[_n]
        b.log(f"[fwd] AD Jacobian: {len(ad_chem_rows)} rows, one warm jvp each, in {time.time() - t1:.0f} s")
    return ad_cols


def _rt_deck_row(b: Baseline, name: str) -> tuple[np.ndarray, float, float, str]:
    """RT-only Jacobian row for a cloud-deck parameter (no chemistry re-solve).

    AD -> one jvp along it. FD -> a SINGLE central difference, since the analytic power-law deck carries no solver
    noise; its O(h^2) truncation error is then unmeasured, so fd_err is NaN, never 0. Returns (row, h, err, method).
    """
    import jax
    import jax.numpy as jnp

    base_vec = np.array([b.cp["log_kappa_cloud"], b.cp["alpha_cloud"]], dtype=np.float64)
    idx = CLOUD_FISHER_PARAMS.index(name)
    if b.cp["jac_method"] == "ad":
        e = np.zeros(base_vec.size)
        e[idx] = 1.0
        _, dd = jax.jvp(lambda v: b.column.depth(b.y_sol, b.th0, cloud=v), (jnp.asarray(base_vec),), (jnp.asarray(e),))
        return np.asarray(dd), 0.0, np.nan, "ad-jvp"
    h = FD_STEPS[name]

    def _d(step: float) -> np.ndarray:
        shifted = base_vec.copy()
        shifted[idx] += step
        v = jnp.asarray(shifted)
        if b.setup.emis is None:
            return np.asarray(b.column.depth(b.y_sol, b.th0, cloud=v))
        dep, tau = b.column.depth(b.y_sol, b.th0, return_tau=True, cloud=v)
        return np.asarray(_gate_emis(b, dep, tau, f"FD {name} {step:+g}"))

    j1 = (_d(h) - _d(-h)) / (2.0 * h)
    return j1, h, np.nan, "fd-rt"  # truncation unmeasured, not 0


def _stencil_points(b: Baseline, name: str, offs: tuple[int, ...], h: float, frq: int | None) -> dict:
    """Every stencil point of a row at ONE photolysis cadence.

    Never mix: the cadence moves a converged spectrum by 0.01-2.4 ppm on the presets, and in a row that offset divides
    by h and lands in the derivative.
    """
    pts = {}
    if name in FD_COMP_PARAMS:
        # equilibrium re-seed + certified cold solve per point
        for s in offs:
            f = float(np.exp(s * h))
            if name == "lnZ":  # all metals together; C/O preserved
                ab = b.abundance_overrides(b.cp["met_x_solar"] * f, b.cp["co_ratio"])
            else:  # dlnCO: carbon at fixed oxygen
                ab = b.abundance_overrides(b.cp["met_x_solar"], b.cp["co_ratio"] * f)
            chem_s = b.build_chem(ab, tag=f"FD {name} {s:+d}h", photo_frq=frq)
            if b.pic and b.setup.tp_eval is None:  # a climate point has its own T
                check_t_window(None, b.theta, chem_s.p_bar, b.log, T_base=chem_s.T_base)
            pts[s] = _certified_depth(b, chem_s, b.theta, f"FD {name} {s:+d}h")
        return pts
    # Kzz changes only the live transport coefficient. T-P also
    # changes the radius/gravity anchor and hence the Guillot
    # mapping, so rebuild it just as a full perturbed run does.
    i_par = b.theta_names.index(name)
    chem_r = (
        b.chem
        if frq == b.run_frq or i_par >= 3
        else b.build_chem(tag=f"{name} row [photo cadence {frq}]", photo_frq=frq)
    )
    for s in offs:
        if i_par >= 3:
            cp_s = dict(b.cp, **{name: b.cp[name] + s * h})
            A_s = b.assemble(cp_s, b.log)
            chem_s = A_s.build_chem(tag=f"FD {name} {s:+d}h", photo_frq=frq)
            if b.pic:
                b.picaso_retried.extend((f"FD {name} {s:+d}h {stage}", n) for stage, n in A_s.retried)
            check_t_window(A_s.tp_eval, A_s.theta, chem_s.p_bar, b.log, T_base=chem_s.T_base)
            pts[s] = _certified_depth(b, chem_s, A_s.theta, f"FD {name} {s:+d}h", A_s.tp_eval)
            continue
        th_s = b.theta.copy()
        th_s[i_par] += s * h
        pts[s] = _certified_depth(b, chem_r, th_s, f"FD {name} {s:+d}h")
    return pts


def _stencil_anchor(b: Baseline, name: str, frq: int | None) -> np.ndarray:
    """Return a one-sided stencil's anchor, the BASELINE point.

    At the run's own cadence it is the already-gated baseline spectrum; an escalated row needs the anchor at ITS
    cadence, which costs one solve.
    """
    if frq == b.run_frq:
        return np.asarray(b.column.depth(b.y_sol, b.th0))
    return _certified_depth(
        b,
        b.build_chem(tag=f"baseline [photo cadence {frq}]", photo_frq=frq),
        b.theta,
        f"FD {name} anchor [photo cadence {frq}]",
    )


def _fd_row(b: Baseline, name: str, rows: Rows, j: int, t1: float) -> None:
    """One certified FD row: every stencil point at one photolysis cadence, escalated as a whole on a stall."""
    offs: tuple[int, ...] = (1, -1, 2, -2)
    h = FD_STEPS[name]
    if name in FD_COMP_PARAMS:
        # composition direction: central, or one-sided away from the
        # C/O bound or the PICASO metallicity cap -- see fd_stencil
        offs, h = fd_stencil(
            name,
            b.cp["met_x_solar" if name == "lnZ" else "co_ratio"],
            CO_MAX_PHOTO_OFF if b.pic else CO_MAX[b.cp["network"]],
            PICASO_MET_RANGE[1] if b.pic else None,
        )
    central = len(offs) == 4
    method = "fd-central" if central else "fd-onesided"
    try:
        dvals = _stencil_points(b, name, offs, h, b.run_frq)
        f0 = None if central else _stencil_anchor(b, name, b.run_frq)
    except NotCertified as exc:
        # ONLY a stalled column escalates. Every other RuntimeError
        # this try spans -- a backend fault, the element gate, the
        # emission gate, the T-window check -- propagates: re-solving
        # it at cadence 1 pays a second full stencil and reports a
        # photolysis escalation for a failure that is not one.
        if b.run_frq == 1:  # already at the finest cadence
            raise
        b.log(f"[fwd] FD {name} row: {exc}")
        b.log(
            f"[fwd] FD {name} row: re-solving EVERY point of this row "
            "with photolysis refreshed every accepted step (a row may "
            "not mix cadences)"
        )
        dvals = _stencil_points(b, name, offs, h, 1)
        f0 = None if central else _stencil_anchor(b, name, 1)
        b.escalated.append(f"FD {name} row")
    row, err = fd_row(name, *fd_estimates(offs, dvals, f0, h), h)
    rows.add(j, row, h, err, method)
    b.log(
        f"[fwd] FD Jacobian d(depth)/d({name}) [{method}] in "
        f"{time.time() - t1:.0f} s (h-vs-2h consistency {err:.3f} < "
        f"{FD_CONSISTENCY_TOL})"
    )


def _lnr0_row(b: Baseline, rows: Rows, j: int) -> None:
    """Add the lnR0 row: RT-only and analytic in lnR0, one central difference or the RT jvp."""
    import jax
    import jax.numpy as jnp

    t1 = time.time()
    b.advance()
    if b.cp["jac_method"] == "ad":
        _, dd = jax.jvp(lambda r: b.column.depth(b.y_sol, b.th0, lnR0=r), (jnp.asarray(0.0),), (jnp.asarray(1.0),))
        rows.add(j, np.asarray(dd), 0.0, np.nan, "ad-jvp")
    else:
        # lnR0 re-hangs the column at fixed planet mass (radius and gravity move together), so in emission the flux and
        # the bottom optical depth move too: each point is gated like a deck row's.
        def _d_r(step: float) -> np.ndarray:
            if b.setup.emis is None:
                return np.asarray(b.column.depth(b.y_sol, b.th0, lnR0=step))
            dep, tau = b.column.depth(b.y_sol, b.th0, lnR0=step, return_tau=True)
            return np.asarray(_gate_emis(b, dep, tau, f"FD lnR0 {step:+g}"))

        row = (_d_r(FD_LNR0_STEP) - _d_r(-FD_LNR0_STEP)) / (2.0 * FD_LNR0_STEP)
        rows.add(j, row, FD_LNR0_STEP, np.nan, "fd-rt")  # single central difference: truncation unmeasured
    b.log(
        f"[fwd] {b.cp['jac_method'].upper()} Jacobian d(depth)/d(lnR0) [RT-only nuisance] in {time.time() - t1:.0f} s"
    )


def build_jacobian(b: Baseline, depth_size: int) -> Rows:
    """Build every requested row and lnR0; the method of each is recorded with it."""
    names = list(b.cp["fisher_params"])
    rows = Rows(names, np.zeros((len(names) + 1, depth_size)))
    ad_cols = _ad_chemistry_rows(b, rows)
    for j, name in enumerate(names):
        t1 = time.time()
        if name not in ad_cols:
            b.advance()  # the AD chemistry rows advanced once, together
        if name in CLOUD_FISHER_PARAMS:
            # No chemistry certificate: the analytic deck carries no solver noise. Its AD primal sits at the already-
            # gated baseline; the FD points are gated, since a perturbed deck is a different opacity.
            row, h, err, method = _rt_deck_row(b, name)
            rows.add(j, row, h, err, method)
            if not np.isfinite(row).all():
                raise RuntimeError(f"cloud Jacobian for {name}: non-finite entries")
            b.log(
                f"[fwd] {b.cp['jac_method'].upper()} Jacobian "
                f"d(depth)/d({name}) [RT-only cloud row] in "
                f"{time.time() - t1:.0f} s"
            )
        elif name in ad_cols:
            # Keyed on the AD columns, not on jac_method: under "ad" the T-P rows still run FD.
            rows.add(j, ad_cols[name], 0.0, np.nan, "ad-jvp")
            if not np.isfinite(ad_cols[name]).all():
                raise RuntimeError(f"AD Jacobian for {name}: non-finite entries")
            b.log(f"[fwd] AD Jacobian d(depth)/d({name}) from its warm jvp")
        else:
            _fd_row(b, name, rows, j, t1)
    _lnr0_row(b, rows, len(names))
    names.append("lnR0")
    return rows
