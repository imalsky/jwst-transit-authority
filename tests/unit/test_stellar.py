"""The stellar flux of the eclipse depth is a band mean, as the planet flux is.

The PHOENIX grid is replaced by a spectrum with a known answer, so only the averaging is tested; the grid directory
must still exist, because the function refuses to run without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jwst_tool import instruments, stellar

AMPLITUDE = 3.0e15  # F_lambda = AMPLITUDE / lambda^2 (erg s^-1 cm^-2 A^-1, lambda in A) is flat per wavenumber
FLAT_NU = AMPLITUDE / 1.0e8  # that flat value, erg s^-1 cm^-2 / cm^-1
TEFF = 5000.0


@pytest.fixture
def flat_star_with_a_line(monkeypatch):
    """A star flat in F_nu, with one narrow line that removes 30 % of the flux at the centre of the third band."""
    if not (Path(instruments.PYSYN_CDBS) / "grid" / "phoenix").is_dir():
        pytest.skip("no PHOENIX grid directory: phoenix_surface_flux refuses to run without it")
    import synphot

    # as phoenix_surface_flux does before it imports stsynphot, which warns when it finds no Vega file
    monkeypatch.setenv("PYSYN_CDBS", instruments.PYSYN_CDBS)
    synphot.conf.vega_file = str(Path(instruments.PYSYN_CDBS) / "calspec" / "alpha_lyr_stis_011.fits")
    import stsynphot
    from synphot import SourceSpectrum
    from synphot.models import Empirical1D

    edges = np.linspace(2000.0, 2010.0, 6)  # five bands of 2 cm^-1 near 5 um
    centre = np.sqrt(edges[2] * edges[3])
    wl = np.linspace(1.0e8 / edges[-1] - 5.0, 1.0e8 / edges[0] + 5.0, 20001)  # Angstrom
    depth = 0.3 * np.exp(-0.5 * ((1.0e8 / wl - centre) / 0.05) ** 2)
    flam = (AMPLITUDE / wl**2) * (1.0 - depth) * synphot.units.FLAM
    spec = SourceSpectrum(Empirical1D, points=wl, lookup_table=flam)
    monkeypatch.setattr(stsynphot, "grid_to_spec", lambda *_a, **_k: spec)
    # the closure check compares with a 5000 K blackbody; this toy star is scaled to pass it
    monkeypatch.setattr(stellar, "_pi_planck_nu", lambda nu, _teff: np.full_like(nu, FLAT_NU))
    return edges


def test_the_stellar_flux_is_the_mean_over_each_band(flat_star_with_a_line):
    edges = flat_star_with_a_line
    fs = stellar.phoenix_surface_flux(edges, TEFF, 4.5, 0.0, log=lambda _m: None)
    assert fs.shape == (5,)
    clean = [0, 1, 3, 4]
    np.testing.assert_allclose(fs[clean], FLAT_NU, rtol=1.0e-6)
    # the line's area is 0.3 * 0.05 * sqrt(2 pi) cm^-1 of a 2 cm^-1 band: a 1.9 % dip, not the 30 % at its centre
    np.testing.assert_allclose(fs[2] / FLAT_NU, 1.0 - 0.3 * 0.05 * np.sqrt(2.0 * np.pi) / 2.0, rtol=1.0e-4)


@pytest.mark.parametrize("edges", [np.array([2000.0]), np.array([2002.0, 2000.0]), np.zeros((2, 2))])
def test_band_edges_must_be_increasing(flat_star_with_a_line, edges):
    with pytest.raises(ValueError, match="nu_edges"):
        stellar.phoenix_surface_flux(edges, TEFF, 4.5, 0.0, log=lambda _m: None)
