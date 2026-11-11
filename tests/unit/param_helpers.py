"""Shared helpers of the canonical-parameter tests: a request builder, a T-P table writer and the reference keys."""

from __future__ import annotations

import numpy as np

from jwst_tool import model_defs


def _p(**kw):
    base = dict(planet="wasp39b", tp_mode="guillot", kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return base


def _table(tmp_path, kzz=True, name="prof.txt", tmin=800.0, tmax=1400.0, rows=8, scramble=False, p_lo_log10=6.9):
    # p_lo_log10: bottom pressure exponent. 6.9 (7.9e6 dyn/cm^2) clears the
    # 7.6 bar transmission column; an EMISSION run needs 100 bar, so pass 8.1.
    P = np.logspace(p_lo_log10, -1.0, rows)  # dyne/cm^2, descending (deep first)
    if scramble:
        P = P.copy()
        P[2], P[3] = P[3], P[2]  # break monotonicity
    T = np.linspace(tmax, tmin, rows)
    lines = ["#(dyne/cm2) (K)" + (" (cm2/s)" if kzz else ""), "Pressure\tTemp" + ("\tKzz" if kzz else "")]
    for i in range(rows):
        row = f"{P[i]:.6e}\t{T[i]:.1f}"
        if kzz:
            row += "\t1.0e9"
        lines.append(row)
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def _custom(**kw):
    """A complete custom request whose system differs from WASP-39 b's (the GUI form's seed) in every field."""
    base = dict(
        planet="custom",
        rp_rjup=1.1,
        gs_cgs=800.0,
        rstar_rsun=0.8,
        orbit_au=0.04,
        star_teff=4800.0,
        star_logg=4.4,
        star_feh=0.1,
        sflux="Gueymard_solar.txt",
        tp_mode="guillot",
        Tint=250.0,
    )
    base.update(kw)
    return base


def _pf(path, **kw):
    base = dict(
        planet="wasp39b",
        tp_mode="file",
        tp_file=model_defs.TP_FILE_UPLOAD,
        tp_file_path=str(path),
        kzz_mode="const",
        kzz_const=1.0e9,
    )
    base.update(kw)
    return base
