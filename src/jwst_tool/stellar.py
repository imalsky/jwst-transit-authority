"""Stellar surface flux for eclipse depths (emission mode).

``phoenix_surface_flux(nu_grid, teff, logg, feh)`` returns the stellar
SURFACE flux density per wavenumber (erg s^-1 cm^-2 / cm^-1) on the native RT
grid, from the same minimal-CDBS PHOENIX grid the Pandeia noise side uses.

Units contract: the CDBS PHOENIX models are emergent SURFACE flux densities
and ExoJax ``ArtEmisPure`` returns the planet's in the same convention, so

    eclipse depth  d_ec(nu) = (F_p(nu) / F_s(nu)) * (R_p/R_s)^2

with NO extra pi. Guarded by an energy-closure check: band-integrated F_s
must match band-integrated pi B_nu(T_eff) to within ~1.5x -- a grid that
were secretly intensity (~3.1x) or Eddington flux (~12x) fails immediately.

Heavy-path module: imports stsynphot on first call; never imported by the
GUI's light path.
"""
from __future__ import annotations

import os

import numpy as np

# CGS constants (CODATA; match exojax's planck.py values to float precision)
_H = 6.62607015e-27       # erg s
_C = 2.99792458e10        # cm/s
_KB = 1.380649e-16        # erg/K

# numpy 2 renamed trapz -> trapezoid (and removed trapz); support both
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _pi_planck_nu(nu_cm: np.ndarray, teff: float) -> np.ndarray:
    """pi * B_nu(T) per wavenumber: erg s^-1 cm^-2 / cm^-1 (surface flux
    density of a blackbody -- ArtEmisPure's optically-thick limit)."""
    x = _H * _C * nu_cm / (_KB * teff)
    return np.pi * 2.0 * _H * _C**2 * nu_cm**3 / np.expm1(x)


def phoenix_surface_flux(nu_grid: np.ndarray, teff: float, logg: float,
                         feh: float, log=print) -> np.ndarray:
    """PHOENIX stellar surface flux (erg s^-1 cm^-2 / cm^-1) on ``nu_grid``.

    Parameters: nu_grid (cm^-1, any order), teff (K), logg (log10 cgs),
    feh ([Fe/H] dex). Raises with a remedy when the PHOENIX grid is absent
    (``jwst-tool fetch`` / the data README) and on an energy-closure failure.
    """
    from jwst_tool import instruments as ins

    phoenix_dir = os.path.join(ins.PYSYN_CDBS, "grid", "phoenix")
    if not os.path.isdir(phoenix_dir):
        raise FileNotFoundError(
            f"PHOENIX grid not found at {phoenix_dir}: emission mode needs "
            "the stellar SED for the eclipse depth Fp/Fs. It is the same "
            "dataset the noise side uses -- run 'jwst-tool data' for status "
            "and the data README for the download.")
    # Pin the tool's OWN cdbs root unconditionally: an inherited shell
    # PYSYN_CDBS set elsewhere must never redirect the grid.
    os.environ["PYSYN_CDBS"] = ins.PYSYN_CDBS
    # local CALSPEC Vega so stsynphot never phones home (same file the
    # pandeia worker pins)
    vega = os.path.join(ins.PYSYN_CDBS, "calspec", "alpha_lyr_stis_011.fits")
    import synphot
    if os.path.isfile(vega):
        synphot.conf.vega_file = vega
    import stsynphot
    from synphot import units as syn_units
    stsynphot.conf.rootdir = ins.PYSYN_CDBS   # belt over the env suspenders

    spec = stsynphot.grid_to_spec("phoenix", float(teff), float(feh),
                                  float(logg))
    nu = np.asarray(nu_grid, dtype=np.float64)
    wl_A = 1.0e8 / nu                                   # Angstrom
    flam = spec(wl_A, flux_unit=syn_units.FLAM).value   # erg s^-1 cm^-2 A^-1
    fs_nu = flam * 1.0e8 / nu**2                        # per cm^-1
    if not np.all(np.isfinite(fs_nu)) or np.any(fs_nu <= 0.0):
        raise RuntimeError(
            f"PHOENIX surface flux non-finite/non-positive on the RT grid "
            f"(Teff={teff:g}, logg={logg:g}, [Fe/H]={feh:g}): the requested "
            "star is outside the grid's reliable range for this band.")
    # Energy-closure guard (pi-convention): band-integrated Fs vs pi B_nu.
    order = np.argsort(nu)
    band_fs = _trapz(fs_nu[order], nu[order])
    band_bb = _trapz(_pi_planck_nu(nu[order], float(teff)), nu[order])
    ratio = band_fs / band_bb
    if not 0.5 <= ratio <= 1.5:
        raise RuntimeError(
            f"PHOENIX flux normalization failed the energy-closure check: "
            f"band-integrated Fs / pi*B_nu(Teff) = {ratio:.3g} (expected "
            "~1). The grid's units do not match the surface-flux convention "
            "the eclipse depth needs -- refusing rather than mis-scaling "
            "every Fp/Fs.")
    log(f"[fwd] stellar SED: PHOENIX Teff={teff:g} K, log g={logg:g}, "
        f"[Fe/H]={feh:g}; band energy closure Fs/piB = {ratio:.3f}")
    return fs_nu
