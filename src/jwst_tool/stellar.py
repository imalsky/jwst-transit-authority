"""Stellar surface flux for eclipse depths (emission mode).

Method and contract: docs/forward_model.md, section `stellar`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jwst_tool.tool_types import Log

# CGS constants (CODATA; match exojax's planck.py values to float precision)
_H = 6.62607015e-27  # erg s
_C = 2.99792458e10  # cm/s
_KB = 1.380649e-16  # erg/K

# Accepted band-integrated F_s / (pi B_nu(T_eff)) of a PHOENIX spectrum on the RT band: a surface flux sits near 1
# (1.083 for WASP-39 at 1-15 um), while a unit or convention mistake is off by orders of magnitude.
FLUX_CLOSURE_RANGE = (0.5, 1.5)

# numpy 2 renamed trapz to trapezoid; support both
_trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]  # noqa: NPY201  (numpy 1 has only this name)


def _pi_planck_nu(nu_cm: np.ndarray, teff: float) -> np.ndarray:
    """Return pi * B_nu(T) per wavenumber, in erg s^-1 cm^-2 / cm^-1.

    The surface flux density of a blackbody: ArtEmisPure's optically-thick limit.
    """
    x = _H * _C * nu_cm / (_KB * teff)
    return np.pi * 2.0 * _H * _C**2 * nu_cm**3 / np.expm1(x)


def phoenix_surface_flux(nu_edges: np.ndarray, teff: float, logg: float, feh: float, log: Log = print) -> np.ndarray:
    """Return the PHOENIX stellar surface flux density averaged over each band of ``nu_edges``.

    The planet flux is a mean over each correlated-k band, so the star's is too: a point sample at a band centre lands
    on or between the star's lines and moves the eclipse depth of that band with it.

    Parameters
    ----------
    nu_edges : ndarray [n_nu + 1]
        Band edges, cm^-1, increasing (the RT model's ``nu_edges``).
    teff : float
        Effective temperature, K.
    logg : float
        Surface gravity, log10 of cm s^-2.
    feh : float
        Metallicity [Fe/H], dex.
    log : Log
        Receives the closure line.

    Returns
    -------
    ndarray [n_nu]
        Band-mean surface flux density, erg s^-1 cm^-2 / cm^-1.

    Raises
    ------
    FileNotFoundError
        The PHOENIX grid is absent (``jwst-tool fetch`` downloads it).
    RuntimeError
        The flux is non-finite or non-positive, or fails the energy-closure check.
    ValueError
        ``nu_edges`` is not a one-dimensional increasing array of at least two edges.
    """
    from jwst_tool import instruments as ins

    phoenix_dir = Path(ins.PYSYN_CDBS) / "grid" / "phoenix"
    if not phoenix_dir.is_dir():
        raise FileNotFoundError(
            f"PHOENIX grid not found at {phoenix_dir}: emission mode needs "
            "the stellar SED for the eclipse depth Fp/Fs. It is the same "
            "dataset the noise side uses -- run 'jwst-tool data' for status "
            "and 'jwst-tool fetch', which downloads it."
        )
    # Pin the tool's OWN cdbs root unconditionally: an inherited shell
    # PYSYN_CDBS set elsewhere must never redirect the grid.
    os.environ["PYSYN_CDBS"] = ins.PYSYN_CDBS
    # local CALSPEC Vega so stsynphot never phones home (same file the
    # pandeia worker pins)
    vega = Path(ins.PYSYN_CDBS) / "calspec" / "alpha_lyr_stis_011.fits"
    import synphot

    if vega.is_file():
        synphot.conf.vega_file = str(vega)
    import stsynphot
    from synphot import units as syn_units

    stsynphot.conf.rootdir = ins.PYSYN_CDBS  # config too, not only the env var

    spec = stsynphot.grid_to_spec("phoenix", float(teff), float(feh), float(logg))
    edges = np.asarray(nu_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"nu_edges must be increasing band edges (cm^-1), got shape {edges.shape}")
    nu = np.sqrt(edges[:-1] * edges[1:])  # the engine's band centres
    # The mean of F_nu over a band is the integral of F_lambda over its wavelength span, divided by its width in
    # wavenumber. The integral is a trapezoid on the grid's own samples plus the band edges.
    wl_edges = 1.0e8 / edges[::-1]  # Angstrom, increasing
    native = np.asarray(spec.waveset.to_value("angstrom"), dtype=np.float64)
    wl = np.union1d(wl_edges, native[(native > wl_edges[0]) & (native < wl_edges[-1])])
    flam = spec(wl, flux_unit=syn_units.FLAM).value  # erg s^-1 cm^-2 A^-1
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (flam[1:] + flam[:-1]) * np.diff(wl))])
    fs_nu = np.diff(cumulative[np.searchsorted(wl, wl_edges)])[::-1] / np.diff(edges)  # per cm^-1
    if not np.all(np.isfinite(fs_nu)) or np.any(fs_nu <= 0.0):
        raise RuntimeError(
            f"PHOENIX surface flux non-finite/non-positive on the RT grid "
            f"(Teff={teff:g}, logg={logg:g}, [Fe/H]={feh:g}): the requested "
            "star is outside the grid's reliable range for this band."
        )
    # Energy-closure guard (pi-convention): band-integrated Fs vs pi B_nu.
    order = np.argsort(nu)
    band_fs = _trapz(fs_nu[order], nu[order])
    band_bb = _trapz(_pi_planck_nu(nu[order], float(teff)), nu[order])
    ratio = band_fs / band_bb
    if not FLUX_CLOSURE_RANGE[0] <= ratio <= FLUX_CLOSURE_RANGE[1]:
        raise RuntimeError(
            f"PHOENIX flux normalization failed the energy-closure check: "
            f"band-integrated Fs / pi*B_nu(Teff) = {ratio:.3g} (expected "
            "~1). The grid's units do not match the surface-flux convention "
            "the eclipse depth needs -- refusing rather than mis-scaling "
            "every Fp/Fs."
        )
    log(
        f"[fwd] stellar SED: PHOENIX Teff={teff:g} K, log g={logg:g}, "
        f"[Fe/H]={feh:g}; band energy closure Fs/piB = {ratio:.3f}"
    )
    return fs_nu
