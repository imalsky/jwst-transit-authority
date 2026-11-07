"""The spectrum of a solved column: the RT setup of a run, and transit or eclipse depths behind one signature."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from jwst_tool import planets
from jwst_tool.certify import (
    EMIS_TAU_THIN,
    EMIS_THIN_FLUX_FRAC,
    check_emission_thin,
    tau_bottom_breakdown,
    thin_flux_fraction,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax

    from jwst_tool.tool_types import AnyArray, Log


@dataclass(frozen=True)
class RtSetup:
    """The RT side of one run: the models, the stellar SED and the baseline deck and T-P."""

    cp: dict
    rt: Any
    emis: Any  # None in transmission
    fs_j: Any  # stellar surface flux on the RT grid, None in transmission
    rstar_cm_sq: float
    cloud_vec: Any  # the baseline deck, None when it is off
    p_art_j: Any
    config: Any
    tp_eval: Any  # None for a tabulated profile
    art_t_const: Any  # the tabulated profile on the RT grid, None for a parametric one


def gas_mask(chem_b: Any) -> np.ndarray:
    """Return 0 for the network's condensate columns and 1 for the gas, per species."""
    # GAS-phase normalization: the network's condensate columns (*_l_s,
    # *_l, *s: vulcan3's conden.GAS_TO_CONDENSATE) are particles,
    # not gas -- counting them dilutes every gas VMR and inflates the RT
    # mean molecular weight. Aerosol OPACITY stays excluded (cloud deck).
    mask = np.ones(int(np.asarray(chem_b.species_masses).size))
    for s, i in chem_b.sidx.items():
        if s.endswith(("_l_s", "_l", "_s")):
            mask[int(i)] = 0.0
    return mask


def _t_art_from(rt: Any, chem_b: Any) -> jax.Array:
    """Tabulated-mode RT temperature: the build's T_base interpolated in ln P onto the ART grid.

    The chemistry grid covers the RT grid, so np.interp's edge clamp is inert.
    """
    import jax.numpy as jnp

    pb = np.asarray(chem_b.p_bar)
    t_base = np.asarray(chem_b.T_base, dtype=np.float64)
    order = np.argsort(pb)
    return jnp.asarray(np.interp(np.log(np.asarray(rt.p_art_bar)), np.log(pb[order]), t_base[order]))


def build_rt_setup(
    cp: dict, profile: dict, tp_eval: Any, config: Any, chem: Any, advance: Callable[[], None], log: Log
) -> RtSetup:
    """Build the RT and, in emission, the eclipse model and stellar SED; verify the engine echoed every RT setting."""
    import jax.numpy as jnp
    from vulcan_forward import exojax_rt

    t0 = time.time()
    advance()
    log("[fwd] building ExoJAX RT (opacities + CIA) ...")
    rt = exojax_rt.build_rt_model(profile)
    log(f"[fwd] RT ready in {time.time() - t0:.0f} s")
    # Echo check: an engine that ignores an unknown profile key must not cache
    # a spectrum under a key describing physics it skipped.
    echo = {
        "art_ptop_bar": cp["rt_ptop_bar"],
        "art_pbtm_bar": profile["art_pbtm_bar"],
        "rt_integration": cp["rt_integration"],
        "p_ref_bar": cp["p_ref_bar"],
        "opacity_mode": profile["opacity_mode"],
    }
    for k, want in echo.items():
        got = getattr(rt, k, None)
        if got != want:
            raise RuntimeError(
                f"RT engine did not honor {k}={want!r} (echoed {got!r}): the "
                "installed vulcan-forward engine does not read this profile "
                "key. Upgrade vulcan-forward."
            )

    emis, fs_j = None, None
    # 1 / R_star^2 only: the PLANET radius for the eclipse prefactor comes from
    # emis.emission_radius(T, mmw) at the emission anchor (~0.1 bar), NOT the
    # catalogue transit radius (~1 mbar).
    rstar_cm_sq = (cp["rstar_rsun"] * planets.R_SUN_CM) ** 2
    if cp["science_mode"] == "emission":
        advance()
        log("[fwd] building emission model + stellar SED ...")
        emis = exojax_rt.build_emis_model(rt, profile)
        from jwst_tool import stellar as stellar_mod

        fs_j = jnp.asarray(
            stellar_mod.phoenix_surface_flux(rt.nu_grid, cp["star_teff"], cp["star_logg"], cp["star_feh"], log=log)
        )

    p_art_j = jnp.asarray(rt.p_art_bar)
    # ExoJAX power-law cloud [log10 kappac0 (cm^2/g at 3.5 um), alphac]: the
    # BASELINE deck the cloud Fisher rows differentiate around (None when off)
    cloud_vec = jnp.asarray([cp["log_kappa_cloud"], cp["alpha_cloud"]]) if cp["cloud_on"] else None
    # A tabulated profile is ONE fixed profile for chemistry and RT.
    art_t_const = _t_art_from(rt, chem) if tp_eval is None else None
    return RtSetup(cp, rt, emis, fs_j, rstar_cm_sq, cloud_vec, p_art_j, config, tp_eval, art_t_const)


class ColumnSpectrum:
    """The depth of ONE chemistry build.

    The interpolation map follows that build's hydrostatic grid, so it is never shared across builds: composition moves
    the mean molecular weight and hence the pressure grid between FD re-init builds.

    science_mode picks the observable behind the same signature: transmission gives the transit depth
    (Rp(lambda)/Rstar)^2, emission the eclipse depth (Fp/Fs) * (Rp/Rs)^2. In both, lnR0 is the engine's radius change at
    fixed planet mass (radius e^lnR0, gravity e^(-2 lnR0) at p_ref).
    """

    def __init__(self, setup: RtSetup, chem_b: Any, tp_eval_b: Any = None) -> None:
        import jax.numpy as jnp
        from vulcan_forward import interp_map

        rt = setup.rt
        self._setup = setup
        self._chem = chem_b
        self._tp_eval_b = tp_eval_b
        self._to_art = interp_map.make_to_art(chem_b.p_bar, rt.p_art_bar)
        # a PICASO climate build carries its own T (a T_int stencil point solves its own profile); every VULCAN
        # file-mode build shares one
        pic = setup.cp["chem_provider"] == "picaso"
        self._t_art_build = _t_art_from(rt, chem_b) if pic and setup.tp_eval is None else None
        self._mol_cols = {k: chem_b.sidx[setup.config.MOLECULES[k]["vulcan"]] for k in rt.molecules}
        self._h2, self._he = chem_b.sidx["H2"], chem_b.sidx["He"]
        self._gas_mask = jnp.asarray(gas_mask(chem_b))

    def _art_t(self, th: AnyArray) -> AnyArray:
        setup = self._setup
        if self._tp_eval_b is not None:
            return self._tp_eval_b(th[3:], setup.p_art_j)
        if self._t_art_build is not None:
            return self._t_art_build
        return setup.art_t_const if setup.tp_eval is None else setup.tp_eval(th[3:], setup.p_art_j)

    def art_profiles(self, y: AnyArray, th: AnyArray) -> tuple:
        """Return (VMRs, H2 VMR, T, mean molecular weight, He VMR) on the RT grid."""
        import jax.numpy as jnp

        to_art = self._to_art
        y_gas = y * self._gas_mask[None, :]
        ymix = y_gas / jnp.sum(y_gas, axis=1, keepdims=True)
        t_art = self._art_t(th)
        mmw_art = to_art(ymix @ self._chem.species_masses)
        vmr = {k: to_art(ymix[:, c]) for k, c in self._mol_cols.items()}
        return (vmr, to_art(ymix[:, self._h2]), t_art, mmw_art, to_art(ymix[:, self._he]))

    def depth(
        self,
        y: AnyArray,
        th: AnyArray,
        lnR0: float | jax.Array = 0.0,
        cloud: AnyArray | None = None,
        wo_mols: list | None = None,
        return_tau: bool = False,
    ) -> Any:
        """Return the depth, or (depth, bottom optical depth) with return_tau in emission.

        cloud=None is the baseline deck; an explicit vector overrides it (the cloud rows differentiate through it).
        wo_mols is transmission only: emission's leave-one-out batch comes from emission_spectra. eclipse_flux_tau, not
        emission_flux_tau: the flux carries the tau = 2/3 photospheric radius per wavelength relative to the single
        anchor radius (Fortney et al. 2019).
        """
        import jax
        import jax.numpy as jnp

        setup = self._setup
        emis, rt = setup.emis, setup.rt
        vmr, vmr_h2, T_art, mmw_art, vmr_he = self.art_profiles(y, th)
        cl = setup.cloud_vec if cloud is None else cloud
        if emis is not None:
            if wo_mols is not None:
                raise ValueError(
                    "ColumnSpectrum.depth(wo_mols=...) is transmission-only; emission "
                    "removed-molecule spectra come from "
                    "emis.eclipse_flux_tau (emission_spectra owns that call)."
                )
            fp, tau = emis.eclipse_flux_tau(vmr, vmr_h2, T_art, mmw_art, vmr_he=vmr_he, cloud=cl, lnR0=lnR0)
            rp_em = emis.emission_radius(T_art, mmw_art, lnR0=lnR0)
            dep = (fp / setup.fs_j) * (rp_em**2 / setup.rstar_cm_sq)
            # Tau is gate evidence, not a differentiated observable.
            return (dep, jax.lax.stop_gradient(tau)) if return_tau else dep
        if return_tau:
            raise ValueError("return_tau is emission-only")
        return rt.transmission_depth_r(
            vmr, vmr_h2, T_art, mmw_art, jnp.asarray(lnR0), vmr_he=vmr_he, cloud=cl, wo_mols=wo_mols
        )


@dataclass(frozen=True)
class EmissionEvidence:
    """What detect's band-restricted gate needs from the emission spectra."""

    depth_norm: float
    tau_bottom: np.ndarray
    tau_bottom_wo: np.ndarray
    thin_flux_frac_wo: np.ndarray


def _certify_baseline_bottom(setup: RtSetup, tau_b: np.ndarray, fp_j: AnyArray, log: Log) -> None:
    """Refuse a baseline whose column bottom leaks flux, and report where the column sees through."""
    emis_tau_min = float(tau_b.min())
    wl_thin = float(setup.rt.wl_um[int(np.argmin(tau_b))])
    # FLUX-WEIGHTED, not min() over the band: the column being transparent
    # somewhere only matters to the extent the planet emits there.
    fp = np.asarray(fp_j)
    report = tau_bottom_breakdown(setup.rt.wl_um, tau_b, flux=fp)
    emis_thin_frac = check_emission_thin(
        tau_b,
        fp,
        setup.rt.nu_grid,
        f"baseline, column bottom {setup.emis.art_pbtm_bar:g} bar",
        detail="\n\n" + report + "\n\nThin only at the SHORT-wavelength edge means missing "
        "opacity, not a shallow column: below ~2 um the continuum "
        "that fills the window in a real hot atmosphere (H- bound-free "
        "and free-free, the Na and K wings, TiO/VO) is not modeled "
        "here, and deepening p_btm_bar would hide that rather than fix "
        "it. Thin across the band means the column really is too "
        f"shallow -- raise p_btm_bar (now {setup.cp['p_btm_bar']:g} bar).",
    )
    log("[fwd] " + report.replace("\n", "\n[fwd] "))
    if emis_thin_frac > 0.0:
        log(
            f"[fwd] NOTE: {100.0 * emis_thin_frac:.3f}% of the emitted "
            f"flux comes from below tau = {EMIS_TAU_THIN:g} (min "
            f"{emis_tau_min:.2f} at {wl_thin:.2f} um). Under the "
            f"{100.0 * EMIS_THIN_FLUX_FRAC:g}% tolerance, so the run "
            "proceeds; treat those wavelengths as unmodeled."
        )


def _removed_molecule_gates(
    setup: RtSetup, wo_list: list, fp_wo_j: AnyArray, tau_wo_j: AnyArray, log: Log
) -> np.ndarray:
    """Return each removed-molecule spectrum's thin-bottom fraction; only that molecule's detection is at stake."""
    emis_thin_wo = np.full(len(wo_list), np.nan)
    emis_tau_min_wo = np.full(len(wo_list), np.nan)
    # Each removed-molecule optical depth (from the batch above) feeds
    # its spectrum AND the tau-bottom gate, which must cover EVERY
    # emission spectrum the results consume: zeroing a dominant
    # absorber can open see-through windows that inflate the contrast.
    for i, mol in enumerate(wo_list):
        tau_i = np.asarray(tau_wo_j[i])
        emis_tau_min_wo[i] = float(tau_i.min())
        wl_i = float(setup.rt.wl_um[int(np.argmin(tau_i))])
        # flux-weighted, same reasoning as the baseline gate above
        emis_thin_wo[i] = thin_flux_fraction(tau_i, np.asarray(fp_wo_j[i]), setup.rt.nu_grid)
        if emis_thin_wo[i] > EMIS_THIN_FLUX_FRAC:
            # Per-molecule, never whole-run: only THIS molecule's
            # detection is unreliable, and detect refuses that target.
            log(
                f"[fwd] NOTE: emission detection of {mol} is "
                f"UNRELIABLE -- with it removed, "
                f"{100.0 * emis_thin_wo[i]:.1f}% of the emitted flux "
                f"comes from below tau = {EMIS_TAU_THIN:g} (min "
                f"{emis_tau_min_wo[i]:.2f} at {wl_i:.2f} um); detect "
                "refuses this target. The spectrum and the other "
                "molecules are unaffected."
            )
        elif emis_tau_min_wo[i] < 10.0:
            log(
                f"[fwd] WARNING: min bottom optical depth "
                f"{emis_tau_min_wo[i]:.1f} at {wl_i:.2f} um (< 10) "
                f"with {mol} removed: its detection contrast leans on "
                "the deepest layers -- treat with care."
            )
    return emis_thin_wo


def emission_spectra(
    setup: RtSetup,
    column: ColumnSpectrum,
    y_sol: AnyArray,
    th0: AnyArray,
    wo_list: list,
    advance: Callable[[], None],
    log: Log,
) -> tuple:
    """Return (depth, depth_wo, evidence) from one optical-depth build, certifying the column bottom."""
    from vulcan_forward import constants as _vfc

    cp, emis, fs_j, cloud_vec, rstar_cm_sq = setup.cp, setup.emis, setup.fs_j, setup.cloud_vec, setup.rstar_cm_sq
    t0 = time.time()
    advance()  # full (+ removed-molecule) eclipse spectra
    prof0 = column.art_profiles(y_sol, th0)
    # the SAME (R_p/R_star)^2 the baseline depth was built with, so the
    # stored Fp inverts the stored depth exactly
    depth_norm_em0 = float(np.asarray(emis.emission_radius(prof0[2], prof0[3])) ** 2 / rstar_cm_sq)
    log(
        f"[fwd] emission anchor: R_p = "
        f"{depth_norm_em0**0.5 * cp['rstar_rsun'] * planets.R_SUN_CM / planets.R_JUP_CM:.4f} "
        f"R_Jup at {_vfc.P_REF_EMISSION_BAR:g} "
        f"bar (catalogue transit radius {cp['rp_rjup']:.4f} R_Jup at "
        f"{cp['p_ref_bar']:g} bar)"
    )
    # ONE optical-depth build feeds the flux, the tau-bottom gate, the
    # stored depth AND the removed-molecule spectra; the batch's full total
    # is bitwise the standalone call.
    if wo_list:
        fp_j, tau_j, fp_wo_j, tau_wo_j = emis.eclipse_flux_tau(*prof0, cloud=cloud_vec, wo_mols=wo_list)
    else:
        fp_j, tau_j = emis.eclipse_flux_tau(*prof0, cloud=cloud_vec)
    tau_b = np.asarray(tau_j)
    _certify_baseline_bottom(setup, tau_b, fp_j, log)

    # the stored depth via ColumnSpectrum.depth's expression at lnR0 = 0
    rp_em = emis.emission_radius(prof0[2], prof0[3])

    def _emis_depth(fp: AnyArray) -> np.ndarray:
        return np.asarray((fp / fs_j) * (rp_em**2 / rstar_cm_sq))

    depth = _emis_depth(fp_j)
    if not wo_list:
        log(f"[fwd] full spectrum in {time.time() - t0:.0f} s")

    depth_wo = np.zeros((len(wo_list), depth.shape[0]))
    emis_thin_wo = np.full(len(wo_list), np.nan)
    if wo_list:
        for i in range(len(wo_list)):
            depth_wo[i] = _emis_depth(fp_wo_j[i])
        emis_thin_wo = _removed_molecule_gates(setup, wo_list, fp_wo_j, tau_wo_j, log)
        log(f"[fwd] full + {len(wo_list)} removed-molecule spectra in {time.time() - t0:.0f} s")
    tau_wo = np.asarray(tau_wo_j) if wo_list else np.zeros((0, tau_b.size))
    return depth, depth_wo, EmissionEvidence(depth_norm_em0, tau_b, tau_wo, emis_thin_wo)


def transmission_spectra(
    column: ColumnSpectrum, y_sol: AnyArray, th0: AnyArray, wo_list: list, advance: Callable[[], None], log: Log
) -> tuple:
    """Return (depth, depth_wo): the full spectrum and the leave-one-out batch."""
    t0 = time.time()
    advance()
    if wo_list:
        log("[fwd] radiative transfer: full + removed-molecule spectra ...")
        d_j, d_wo_j = column.depth(y_sol, th0, wo_mols=wo_list)
        depth = np.asarray(d_j)
        depth_wo = np.asarray(d_wo_j)
        log(f"[fwd] full + {len(wo_list)} removed-molecule spectra in {time.time() - t0:.0f} s")
    else:
        log("[fwd] radiative transfer: full spectrum ...")
        depth = np.asarray(column.depth(y_sol, th0))
        depth_wo = np.zeros((0, depth.shape[0]))
        log(f"[fwd] full spectrum in {time.time() - t0:.0f} s")
    return depth, depth_wo
