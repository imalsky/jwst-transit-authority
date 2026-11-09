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


# --- WASP-39 b reference state: DO NOT let this drift ------------------------
# The REFERENCE configuration (tp_mode="file", the shipped evening-terminator
# table) is the one measured against the published JWST detection, and also
# the DEFAULT. The guard anchors to the EXPLICIT file-mode
# config so it protects the reference even if the default moves.
# Re-measure against the literature before updating expected values.
W39B_REFERENCE = {
    "tp_mode": "file",  # measured evening-terminator table
    "tp_file": "shipped",
    "tp_file_sha1": "1a4ce744e65205d8",  # exact profile bytes (T AND Kzz)
    "kzz_mode": "file",  # mixing from the table, not a stand-in
    "kzz_const": 0.0,  # inert once tabulated
    "met_x_solar": 10.0,  # Tsai+2023 10x solar
    # CO_DEFAULT: the cfg C_H/O_H 0.549348 rounded onto the 0.05 widget grid
    "co_ratio": 0.55,
    "use_photo": True,  # SO2 is photochemical; non-negotiable
    "sl_angle_deg": 83.0,  # Tsai+2023 terminator slant
    "use_vm_mol": False,  # the validated baseline chemistry
    "nz": 100,
    # G395H SO2 4.23 sigma, as on sncho (notes "Composition, C/O bounds and networks")
    "network": "sncho2025",
}
