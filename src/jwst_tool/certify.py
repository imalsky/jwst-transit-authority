"""Certificates a solved column or spectrum must pass: convergence, element budget, absorbers, emission bottom."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from jwst_tool.model_defs import NO_EXOMOLOP_TABLE, NZ_DEFAULT, T_WINDOW

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from jwst_tool.tool_types import AnyArray, Log

# Minimum positivity margin for the AD dlnCO row. The engine's co_bz_bound =
# ln(1 + min_z(OO_z/OC_z)) is the largest ln C/O increment before some layer's
# fixed-O factor b_z turns nonpositive. Empirical, set by closure against the
# certified one-sided FD row; a literal so an FD step change cannot
# move the AD gate silently. FD has no equivalent limit: it re-initializes
# the equilibrium seed per stencil point and never uses the b_z map.
CO_BZ_MIN_AD = 0.1


# ConvDiag.conv_branch legend (0 = not certified).
CONV_BRANCH = {1: "tight (yconv_cri)", 2: "loose (yconv_min)"}


# Max |element/H drift| of a certified column vs its build column. 1.5%:
# the smallest round value above the drift the advertised C/O top costs.
# The photolysis-off ladder on sncho2025 is 0.06% at C/O 2, 0.32% at 4,
# 0.81% at 7 and 1.30% at 10 (1.21% sncho, 1.22% ncho).
ELEMENT_TOL = 0.015


def check_elements(y: AnyArray, chem: Any, stage: str, log: Log = print) -> float:
    """Refuse a certified column whose element budget drifted from its build column.

    Per layer, every atom column of ``chem.compo_array`` over the H column; the layer median (molecular diffusion
    legitimately separates elements at the very top) against the same ratio on ``chem.y0``, which carries the
    requested composition exactly. The convergence certificate watches the change per step and is blind to a slow
    uniform leak. Returns the worst relative drift.
    """
    compo = np.asarray(chem.compo_array, dtype=np.float64)  # (ni, natom)
    atoms = list(chem.atom_list)
    tot = np.asarray(y, dtype=np.float64) @ compo  # (nz, natom)
    tot0 = np.asarray(chem.y0, dtype=np.float64) @ compo
    i_h = atoms.index("H")
    r = np.median(tot / tot[:, [i_h]], axis=0)
    r0 = np.median(tot0 / tot0[:, [i_h]], axis=0)
    ok = r0 > 0.0
    dev = np.abs(r[ok] / r0[ok] - 1.0)
    worst = float(dev.max())
    el = atoms[int(np.flatnonzero(ok)[int(dev.argmax())])]
    log(f"[fwd] {stage}: element budget within {worst:.2%} of the build column (worst {el})")
    if worst > ELEMENT_TOL:
        raise RuntimeError(
            f"{stage}: the converged column's {el}/H drifted {worst:.1%} from the "
            f"requested composition (gate {ELEMENT_TOL:.1%}); the solve leaked "
            "material and is not a steady state of this atmosphere"
        )
    return worst


class NotCertified(RuntimeError):
    """A solve (or its tangent) that did not certify.

    Distinct from every other RuntimeError on the same path -- a backend fault, the element gate, the emission gate,
    the T-window check -- so a cadence-1 retry fires on a stall alone.
    """


def check_converged(
    diag: Any, stage: str, species: list[str], chem: Any, log: Log = print
) -> tuple[str, int, float, float, int, float, str]:
    """Certify one solve against the runner's CANONICAL gate and return its record.

    Certification is ``ConvDiag.conv_normal`` AND ``longdy < yconv_min``, recomputed at the exit state -- never
    loosen it. Raises naming the branch, longdydt, aflux_change and the controlling ``species@z`` cell when the
    solve does not certify.
    """
    ac = int(diag.accept_count)
    longdy = float(diag.longdy)
    branch = int(diag.conv_branch)
    z = int(diag.cell_layer)
    cell = f"{species[int(diag.cell_species)]}@z{z}"
    tl = float(diag.tangent_longdy)  # NaN: no tangent path
    detail = (
        f"longdy={longdy:.3g}, longdydt={float(diag.longdydt):.3g}/s, "
        f"branch {CONV_BRANCH.get(branch, 'none')}, "
        f"aflux_change={float(diag.aflux_change):.3g}, "
        + (f"tangent_longdy={tl:.3g}, " if not np.isnan(tl) else "")
        + f"t={float(diag.t):.3g} s, dt={float(diag.dt):.3g} s; "
        f"controlling cell {cell} at {float(chem.p_bar[z]):.3g} bar, "
        f"VMR {float(diag.cell_vmr):.3g}"
    )
    if not (bool(diag.conv_normal) and longdy < chem.yconv_min):
        how = (
            f"hit the count_max={chem.count_max} cap"
            if ac >= int(chem.count_max)
            else f"exited at {ac} accepted steps without the runner's "
            "canonical certification (stall fallback / hybrid "
            "vm_mol phase-flip / photolysis flux still changing)"
        )
        if branch and not np.isnan(tl) and longdy < chem.yconv_min:
            # the column certified; the derivative through it did not settle
            raise NotCertified(
                f"the sensitivity did NOT settle ({stage}: {detail}; the "
                f"column certified on branch {CONV_BRANCH[branch]} but its "
                f"tangent, tangent_longdy={tl:.3g} with its slope, did not "
                f"pass the same two-branch test (yconv_cri / yconv_min="
                f"{chem.yconv_min:g}); {how}). The derivative at this point "
                "is not certified; the column is."
            )
        # A column coarser than the default can settle into an exact limit cycle at the step cap, where more steps or a
        # retry cannot help: notes "Accepted limitations".
        nz = int(np.size(chem.p_bar))
        coarse = (
            f" At nz={nz} this can be a limit cycle of a coarse column: raise nz (default {NZ_DEFAULT})."
            if ac >= int(chem.count_max) and nz < NZ_DEFAULT
            else ""
        )
        raise NotCertified(
            f"chemistry did NOT converge ({stage}: {detail}; "
            f"gate yconv_min={chem.yconv_min:g}, "
            f"conv_normal={bool(diag.conv_normal)}; {how}). This "
            "parameter corner did not pass the convergence gates at exit -- adjust "
            "T-P / Kzz / composition (or the convergence settings) "
            f"rather than trusting an unconverged spectrum.{coarse}"
        )
    log(f"[fwd] {stage}: certified at {ac} accepted steps ({detail})")
    return (stage, ac, longdy, float(diag.longdydt), branch, float(diag.aflux_change), cell)


def certified_solve(
    chem_b: Any, th: AnyArray, stage: str, rebuild: Callable[[], Any] | None = None, log: Log = print
) -> tuple[AnyArray, Any, tuple, bool]:
    """Solve one column and certify it, with ONE escalation.

    At the configured photolysis cadence dt can grow by 2^frq between refreshes;
    a strongly shielded column then rejects every refresh step and never
    certifies. A column that does not certify is re-solved once
    with photolysis refreshed every accepted step, which converges it. `rebuild`
    returns the same model at that cadence; without it a column that does not
    certify raises NotCertified. Returns (y, chem_used, cert, escalated).

    Only the CERTIFICATE is inside the try, like the AD rows: an XLA runtime
    error is a RuntimeError too, and a backend failure must propagate instead
    of being retried as a stall and reported with the escalation caveat.
    """

    def _solve(c: Any) -> tuple[AnyArray, Any, list[str]]:
        species = [sp for sp, _ in sorted(c.sidx.items(), key=lambda kv: kv[1])]
        y, diag = c.converged_y(th, return_conv_diag=True)
        return y, diag, species

    y, diag, species = _solve(chem_b)
    try:
        cert = check_converged(diag, stage, species, chem_b, log)
    except NotCertified as exc:
        if rebuild is None:
            raise
        log(f"[fwd] {stage}: {exc}")
        log(f"[fwd] {stage}: re-solving with photolysis refreshed every accepted step")
    else:
        return y, chem_b, cert, False
    chem_b = rebuild()
    stage = f"{stage} [photo cadence 1]"
    y, diag, species = _solve(chem_b)
    return y, chem_b, check_converged(diag, stage, species, chem_b, log), True


def check_ad_co_margin(
    chem: Any, co_ratio: float, y: np.ndarray | None = None, build_margin: float | None = None, log: Log | None = None
) -> float:
    """Certify the AD dlnCO row's oxygen-reservoir margin, or raise.

    With ``y`` the margin is taken on that column (the converged one the
    tangent starts from); without it, on the build's own ``co_bz_bound``.
    Returns the margin. Compared with ``not m > gate`` so a NaN margin refuses
    (see ``vulcan_forward.vulcan_chem.bz_margin``).
    """
    if y is None:
        margin, where = float(chem.co_bz_bound), "build column"
    else:
        margin, where = float(chem.co_bz_margin(y)), "converged column"
        if log is not None:
            log(
                f"[fwd] AD dlnCO margin: build column {build_margin:.3g}, "
                f"converged column {margin:.3g} (gate {CO_BZ_MIN_AD:g})"
            )
    if not margin > CO_BZ_MIN_AD:
        raise RuntimeError(
            f"The AD Jacobian for dlnCO is not available at C/O = "
            f"{co_ratio:g}: the {where}'s oxygen-reservoir margin is "
            f"{margin:.3g} and it needs > {CO_BZ_MIN_AD:g} (roughly C/O below "
            f"{math.exp(-CO_BZ_MIN_AD):.2f}), the margin before the fixed-O "
            "direction exhausts the oxygen-only carriers. Use finite "
            "differences."
        )
    return margin


# A converged column can put a network species with NO published k-table at an
# abundance the RT then ignores. The contrast there carries an unsigned,
# unquantified error, so SAY so. WARN at 1e-4, an order of magnitude below the
# bulk carriers and above every trace species the RT already carries at solar
# C/O, so every carbon-rich run says so.
UNMODELED_VMR_WARN = 1.0e-4


# REFUSE, not warn, above this: the RT is blind to a bulk absorber and the
# spectrum is not a model of the requested atmosphere. The gate is coarse and
# deliberate, chosen over 1e-4. Between the two
# thresholds the run only warns, and that error is unquantified.
UNMODELED_VMR_REFUSE = 1.0e-3


# Transmission photosphere, the band the gate averages over in a transit (bar).
PHOTOSPHERE_BAR = (1.0e-5, 1.0e-2)

# In an eclipse the gate averages over the pressures the emitted flux comes from: between these flux-weighted
# quantiles of the tau = 2/3 pressure over the baseline column's bands and g-ordinates
# (spectrum.emission_photosphere_bar), the central 90 % of the flux (measured: notes "Convergence certificate and
# element gate").
EMIS_PHOTOSPHERE_QUANTILES = (0.05, 0.95)


# Absent from the k-table set for a REASON, so not missing opacity. Two
# classes: the bare atoms (ExoMolOP is a MOLECULAR line-list database -- an
# atom has no IR bands to tabulate), and the background gases, which are
# either the CIA continuum and Rayleigh scatterers (H2, He) or homonuclear
# and so dipole-free (N2, O2, S2).
_NOT_A_K_TABLE_ABSORBER = frozenset(
    {
        "H",
        "O",
        "C",
        "N",
        "S",
        "O_1",
        "N_2D",
        "H2",
        "He",
        "N2",
        "O2",
        "S2",
        # the metal atoms of PICASO's equilibrium chemistry
        "Li",
        "F",
        "Na",
        "Mg",
        "Si",
        "P",
        "Cl",
        "K",
        "Ti",
        "V",
        "Cr",
        "Fe",
        "Rb",
        "atCs",
    }
)


def table_species(molecules: dict) -> set[str]:
    """Return the network names of the species the engine can build opacity for, from its molecule table."""
    return {v["vulcan"] for k, v in molecules.items() if k not in NO_EXOMOLOP_TABLE}


def unmodeled_absorbers(
    species: list[str],
    y: AnyArray,
    p_bar: AnyArray,
    table_species: Iterable[str],
    band_bar: tuple[float, float] = PHOTOSPHERE_BAR,
    floor: float = UNMODELED_VMR_WARN,
) -> list[tuple[str, float]]:
    """Network species with no opacity table that are abundant anyway.

    Returns [(name, mean VMR over the layers inside ``band_bar``), ...] above
    ``floor`` (UNMODELED_VMR_WARN by default), most abundant first.
    ``band_bar`` is the photosphere of the observation, bar: PHOTOSPHERE_BAR in
    a transit, spectrum.emission_photosphere_bar in an eclipse.
    ``table_species`` is the set the engine can build opacity for; the
    background gases and the homonuclear diatomics are excluded because their
    absence is deliberate, not a gap.
    """
    y = np.asarray(y, dtype=np.float64)
    vmr = y / y.sum(axis=1, keepdims=True)
    p_bar = np.asarray(p_bar, dtype=np.float64)
    band = (p_bar >= band_bar[0]) & (p_bar <= band_bar[1])
    if not band.any():
        raise RuntimeError(
            f"no layer of the column ({p_bar.size} layers, {p_bar.min():.3g}-{p_bar.max():.3g} bar) has its centre in "
            f"the photosphere band {band_bar[0]:.3g}-{band_bar[1]:.3g} bar, so the absorbers with no k-table cannot be "
            "checked there. Raise nz, or move the column so it covers the band."
        )
    skip = set(table_species) | _NOT_A_K_TABLE_ABSORBER
    out = [(sp, float(vmr[band, i].mean())) for i, sp in enumerate(species) if sp not in skip]
    return sorted([o for o in out if o[1] > floor], key=lambda o: -o[1])


def merge_unmodeled(into: list[tuple[str, float]], found: list[tuple[str, float]]) -> None:
    """Add a column's (species, VMR) warnings to a run's list in place: each species' largest VMR, largest first."""
    best = dict(into)
    for sp, v in found:
        best[sp] = max(v, best.get(sp, v))
    into[:] = sorted(best.items(), key=lambda o: -o[1])


def check_unmodeled(
    species: list[str],
    y: AnyArray,
    p_bar: AnyArray,
    table_species: Iterable[str],
    band_bar: tuple[float, float],
    stage: str,
    log: Log = print,
) -> None:
    """Refuse a column whose observable layers carry an absorber with no opacity table, above UNMODELED_VMR_REFUSE.

    Below it the run proceeds and only says so (the warning at the call site). No table can be supplied for these
    species -- ExoMolOP publishes none -- so the only remedy is a composition that does not make them bulk carriers.
    ``band_bar`` is the observation's photosphere, as for unmodeled_absorbers.
    """
    over = unmodeled_absorbers(species, y, p_bar, table_species, band_bar=band_bar, floor=UNMODELED_VMR_REFUSE)
    if not over:
        return
    log(
        f"[fwd] {stage}: unmodeled absorbers over the refusal threshold: "
        + ", ".join(f"{sp} {v:.2e}" for sp, v in over)
    )
    raise RuntimeError(
        f"{stage}: the photosphere ({band_bar[0]:.2g}-{band_bar[1]:.2g} bar) carries "
        + ", ".join(f"{sp} at VMR {v:.2e}" for sp, v in over)
        + f", over the {UNMODELED_VMR_REFUSE:.0e} gate, and the RT has no "
        "opacity table for it: the spectrum would omit a bulk absorber and "
        "carry an unsigned error of several hundred ppm. Lower C/O -- the "
        "carbon-rich states are where this bites -- there is no table to "
        "supply for these species."
    )


def gate_unmodeled(
    species: list[str],
    y: AnyArray,
    p_bar: AnyArray,
    table_species: Iterable[str],
    band_bar: tuple[float, float],
    stage: str,
    log: Log = print,
) -> list[tuple[str, float]]:
    """Refuse a column as ``check_unmodeled`` does; return the absorbers above the warning level to report."""
    check_unmodeled(species, y, p_bar, table_species, band_bar, stage, log)
    return unmodeled_absorbers(species, y, p_bar, table_species, band_bar=band_bar)


# Instrument windows the emission tau-bottom report is broken down by, so a
# refusal says WHERE the column sees through. The 1-2 um entry is first: it is
# where the modeled opacity set runs out, not where the column is shallow.
_TAU_WINDOWS = (
    ("1.0-2.0 um (no modeled continuum)", 1.0, 2.0),
    ("2.0-3.0 um", 2.0, 3.0),
    ("3.0-5.2 um (NIRSpec)", 3.0, 5.2),
    ("5.0-12 um (MIRI LRS)", 5.0, 12.0),
    ("12-15 um", 12.0, 15.0),
)


# Share of the planet's EMITTED FLUX that may come from wavelengths where the
# column sees through its own bottom. Flux-weighted on purpose, so a narrow
# blue-edge notch cannot veto an otherwise opaque column. Above the tolerance
# the run is refused.
EMIS_TAU_THIN = 3.0  # per-wavelength "sees through the bottom"


EMIS_THIN_FLUX_FRAC = 0.01  # tolerated share of emitted flux from those wavelengths

# A removed-molecule eclipse whose smallest bottom optical depth is below this, while its flux passes the gate, is
# reported as leaning on the deepest layers (a log warning; no number changes). No basis recorded.
EMIS_TAU_WARN = 10.0


def tau_bottom_breakdown(wl_um: AnyArray, tau: AnyArray, flux: AnyArray) -> str:
    """Report the per-window minimum bottom optical depth, the band share below the gate and its emission share.

    A single min() cannot tell a shallow column from an opacity gap.
    """
    wl, tau = np.asarray(wl_um, float), np.asarray(tau, float)
    thin = tau < EMIS_TAU_THIN
    lines = [
        f"Where the column sees through ({100.0 * thin.mean():.1f}% of the model band is below tau = {EMIS_TAU_THIN:g}"
    ]
    if thin.any():
        lines[0] += f", spanning {wl[thin].min():.2f}-{wl[thin].max():.2f} um)"
    else:
        lines[0] += ")"
    frac = thin_flux_fraction(tau, flux, 1.0e4 / wl)
    lines.append(f"  carrying {100.0 * frac:.3f}% of the planet's emitted flux")
    for label, lo, hi in _TAU_WINDOWS:
        m = (wl >= lo) & (wl <= hi)
        if not m.any():
            continue
        lines.append(
            f"  {label:34s} min tau {tau[m].min():10.3g}   {100.0 * (tau[m] < EMIS_TAU_THIN).mean():5.1f}% thin"
        )
    return "\n".join(lines)


def thin_flux_fraction(tau: AnyArray, flux: AnyArray, nu: AnyArray) -> float:
    """Return the share of the emitted energy from wavelengths whose bottom optical depth is below the gate.

    The emission certificate. The ExoMolOP k-tables make the band grid constant-R, so dnu ~ nu: summing the samples
    unweighted is not an energy fraction and under-reports the short-wavelength share, exactly where a see-through
    bottom shows up. ``nu`` is required, never defaulted, so that sum cannot come back silently.

    Parameters
    ----------
    tau : Array [n_nu]
        Optical depth at the column bottom.
    flux : Array [n_nu]
        Spectral flux density per unit wavenumber, any positive scale.
    nu : Array [n_nu]
        Band-centre wavenumbers, cm^-1, strictly monotonic.

    Returns
    -------
    float
        The flux-weighted fraction with tau below EMIS_TAU_THIN.

    Raises
    ------
    ValueError
        The arrays are misaligned, non-finite, negative or not monotonic.
    RuntimeError
        The flux is zero or non-finite across the whole band.
    """
    tau, flux = np.asarray(tau, float), np.asarray(flux, float)
    nu = np.asarray(nu, dtype=np.float64)
    if (
        tau.ndim != 1
        or flux.ndim != 1
        or nu.ndim != 1
        or tau.shape != flux.shape
        or nu.shape != flux.shape
        or nu.size < 2
    ):
        raise ValueError(
            "thin_flux_fraction needs aligned 1-D tau, flux, and wavenumber "
            f"arrays (>= 2 bands); got {tau.shape}, {flux.shape}, {nu.shape}"
        )
    if not np.all(np.isfinite(tau)) or np.any(tau < 0.0):
        raise ValueError("bottom optical depth must be finite and non-negative")
    if not np.all(np.isfinite(flux)) or np.any(flux < 0.0):
        raise ValueError("emission flux must be finite and non-negative")
    if not np.all(np.isfinite(nu)) or np.any(nu <= 0.0):
        raise ValueError("wavenumber grid must be finite and positive")
    dnu = np.diff(nu)
    if not (np.all(dnu > 0.0) or np.all(dnu < 0.0)):
        raise ValueError("wavenumber grid must be strictly monotonic")
    # band width per sample; exact to 1e-6 against the k-table bin_edges on the
    # constant-R grid, and it needs no edges the RT namespace does not carry
    w = np.abs(np.gradient(nu))
    tot = float((flux * w).sum())
    if not np.isfinite(tot) or tot <= 0.0:
        raise RuntimeError(
            "emission flux is zero or non-finite across the whole band; the thin-bottom certificate cannot be evaluated"
        )
    return float((flux * w)[tau < EMIS_TAU_THIN].sum() / tot)


def check_emission_thin(tau: AnyArray, flux: AnyArray, nu: AnyArray, stage: str, detail: str = "") -> float:
    """Refuse an emission spectrum whose column bottom leaks too much flux.

    The gate ``thin_flux_fraction`` computes, applied to ONE spectrum and
    raising with the stage that produced it. Every emission spectrum the
    results consume goes through here -- the baseline, each removed-molecule
    spectrum, and every FD stencil point and AD primal a Fisher row is built
    from. ``flux`` may be Fp times any positive scalar: the fraction is a
    ratio, so an eclipse-depth prefactor cancels.
    """
    frac = thin_flux_fraction(tau, flux, nu)
    if frac > EMIS_THIN_FLUX_FRAC:
        raise RuntimeError(
            f"emission unreliable ({stage}): {100.0 * frac:.1f}% of the "
            "emitted flux comes from wavelengths where the RT column bottom "
            f"is optically thin (tau < {EMIS_TAU_THIN:g}), above the "
            f"{100.0 * EMIS_THIN_FLUX_FRAC:g}% tolerance -- that flux is set "
            "by the interior source term, an assumption about everything "
            "below the column." + detail
        )
    return frac


def check_t_window(
    tp_eval: Callable[[AnyArray, AnyArray], AnyArray] | None,
    theta: AnyArray,
    p_bar: AnyArray,
    log: Log,
    T_base: AnyArray | None = None,
) -> np.ndarray:
    """T-P validity on the chemistry grid: REJECT (never clip) out-of-window profiles; returns the evaluated T(P).

    In file mode (tp_eval None) the profile IS the structure, so pass T_base (chem.T_base).
    """
    import jax.numpy as jnp

    if tp_eval is None:
        if T_base is None:
            raise ValueError("check_t_window: tp_eval=None (file mode) requires the T_base array")
        T_check = np.asarray(T_base, dtype=np.float64)
    else:
        T_check = np.asarray(tp_eval(jnp.asarray(theta[3:]), jnp.asarray(p_bar)))
    tmin, tmax = float(T_check.min()), float(T_check.max())
    if tmin < T_WINDOW[0] or tmax > T_WINDOW[1]:
        raise RuntimeError(
            f"T-P profile leaves the modelable window [{T_WINDOW[0]:.0f}, "
            f"{T_WINDOW[1]:.0f}] K (min {tmin:.0f} K, max {tmax:.0f} K). "
            "Adjust the profile parameters -- out-of-window layers are rejected, "
            "not clipped (the raw k-tables span a wider temperature range)."
        )
    log(f"[fwd] T-P in window: [{tmin:.0f}, {tmax:.0f}] K")
    return T_check
