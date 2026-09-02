"""Writes validation/data/case_<name>.npz: the chemistry solves behind the
Tsai-2023 and ERS-CO2 validation figures. One WASP-39 b case per invocation
(fresh process; FastChem input is written from process-level config state).
Needs the engine data root and this repo's jax environment:

    python validation/scripts/inputs/chem_cases.py tsai_e5   # no argument lists the cases

Metallicity is set where the engine actually reads it: the config's explicit
C_H/N_H/O_H/S_H elemental knobs (ini_abun writes the FastChem element file
from those for network elements; fastchem_met_scale only scales the
non-network trace metals, which is also scaled here for consistency). He
stays at the config value. The ERS fiducial keeps 10x O/N/S and sets
C_H = 0.35 * O_H (their C/O = 0.35).

NOTE the theta route was tried first and rejected on evidence: audit_init
confirms lnZ = ln 2 doubles the initial column inventory exactly, yet the
converged column returns to the baseline inventory (closed column, so that
cannot be atom-conserving) -- recorded as an open engine question. Each case
here therefore runs at theta = 0 and ASSERTS the converged column inventory.

The morning T-P file ships without a Kzz column; Tsai et al. 2023 use one
Kzz profile, so the evening file's Kzz(P) is interpolated onto the morning
grid (written next to the outputs).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "data"
ELEMS = ("C", "N", "O", "S")


def morning_tp_kzz() -> Path:
    out = OUT / "atm_W39b_morning_TP_Kzz.txt"
    if out.exists():
        return out
    import vulcan_jax
    atm = Path(vulcan_jax.__file__).resolve().parent / "atm"
    ev = np.genfromtxt(atm / "atm_W39b_evening_TP_Kzz.txt", skip_header=2)
    mo = np.genfromtxt(atm / "atm_W39b_10Xsolar_Twhole_morning_TP_20deg.txt", skip_header=2)
    k = 10.0 ** np.interp(np.log10(mo[:, 0]), np.log10(ev[:, 0][::-1]),
                          np.log10(ev[:, 2][::-1]))
    rows = "\n".join(f"{p:.4E}\t{t:.1f}\t {kk:.4E}" for p, t, kk in zip(mo[:, 0], mo[:, 1], k))
    out.write_text("#(dyne/cm2) (K)\t (cm2/s)\nPressure\tTemp  \t Kzz\n" + rows + "\n")
    return out


def cases():
    """name -> (elemental multiplier vs the 10x baseline, C/O target or None,
    morning?, photo?)"""
    out = {}
    for met in (5, 10, 20):
        out[f"tsai_e{met}"] = (met / 10.0, None, False, True)
        out[f"tsai_m{met}"] = (met / 10.0, None, True, True)
    out["ers_photo"] = (1.0, 0.35, False, True)
    out["ers_nophoto"] = (1.0, 0.35, False, False)
    return out


def main(name):
    from vulcan_forward import vulcan_chem  # must precede any vulcan_jax import
    import vulcan_jax
    from vulcan_jax import composition as ba

    factor, co_target, morning, photo = cases()[name]
    base_cfg = vulcan_jax.load_config("W39b")
    ovr = {f"{e}_H": float(getattr(base_cfg, f"{e}_H")) * factor for e in ELEMS}
    ovr["fastchem_met_scale"] = float(base_cfg.fastchem_met_scale) * factor
    if co_target is not None:
        ovr["C_H"] = co_target * ovr["O_H"]
    if morning:
        ovr["atm_file"] = str(morning_tp_kzz())
    profile = dict(vulcan_cfg_name="W39b", use_photo=photo, yconv_cri=0.01,
                   abundance_mode="elemental", skip_warmup=True,
                   cfg_overrides=ovr)

    t0 = time.time()
    model = vulcan_chem.build_chem_model(profile)
    final, _ = model.run_diag(np.zeros(4))
    conv = bool(model.conv_normal_at_exit(final))
    y = np.asarray(final.y, dtype=np.float64)
    ymix = y / y.sum(axis=1, keepdims=True)
    species = [s for s, _ in sorted(model.sidx.items(), key=lambda kv: kv[1])]

    # The gate: the CONVERGED column's elemental inventory must match the case.
    n = {e: np.array([ba.compo[e][ba.compo_row.index(s)] for s in species], float)
         for e in ("H", "C", "O")}
    CH = float((ymix * n["C"]).sum() / (ymix * n["H"]).sum())
    OH = float((ymix * n["O"]).sum() / (ymix * n["H"]).sum())
    want_OH = float(base_cfg.O_H) * factor
    want_CH = (co_target if co_target is not None else float(base_cfg.C_H) / float(base_cfg.O_H)) * want_OH
    print(f"[{name}] column C/H {CH:.4e} (want ~{want_CH:.4e})  "
          f"O/H {OH:.4e} (want ~{want_OH:.4e})  C/O {CH/OH:.4f}")
    assert abs(OH / want_OH - 1.0) < 0.05 and abs(CH / want_CH - 1.0) < 0.05, \
        f"{name}: converged inventory does not match the case"
    assert conv, f"{name} did not certify convergence"

    np.savez_compressed(OUT / f"case_{name}.npz", p_bar=model.p_bar,
                        T=model.T_base, ymix=ymix, species=np.array(species),
                        conv_normal=conv, overrides=str(ovr),
                        accept_count=int(final.accept_count),
                        longdy=float(final.longdy))
    print(f"[{name}] steps {int(final.accept_count)}  longdy {float(final.longdy):.4f}"
          f"  conv_normal {conv}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in cases():
        print("usage: run_cases.py <case>; cases:", " ".join(cases()))
        sys.exit(2)
    main(sys.argv[1])
