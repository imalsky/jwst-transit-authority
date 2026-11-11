"""Writes validation/data/case_<name>.npz: the chemistry solves behind the Tsai-2023 and ERS-CO2 validation figures.

Method and contract: docs/validation.md, section `chem_cases`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "data"
ELEMS = ("C", "N", "O", "S")


def morning_tp_kzz() -> Path:
    """Return the path of the morning-terminator T-P/Kzz table."""
    out = OUT / "atm_W39b_morning_TP_Kzz.txt"
    if out.exists():
        return out
    import vulcan3

    atm = Path(vulcan3.__file__).resolve().parent / "atm"
    ev = np.genfromtxt(atm / "atm_W39b_evening_TP_Kzz.txt", skip_header=2)
    mo = np.genfromtxt(atm / "atm_W39b_10Xsolar_Twhole_morning_TP_20deg.txt", skip_header=2)
    k = 10.0 ** np.interp(np.log10(mo[:, 0]), np.log10(ev[:, 0][::-1]), np.log10(ev[:, 2][::-1]))
    rows = "\n".join(f"{p:.4E}\t{t:.1f}\t {kk:.4E}" for p, t, kk in zip(mo[:, 0], mo[:, 1], k, strict=True))
    out.write_text("#(dyne/cm2) (K)\t (cm2/s)\nPressure\tTemp  \t Kzz\n" + rows + "\n")
    return out


def cases():
    """Return name -> (elemental multiplier vs the 10x baseline, C/O target or None, morning?, photo?)."""
    out = {}
    for met in (5, 10, 20):
        out[f"tsai_e{met}"] = (met / 10.0, None, False, True)
        out[f"tsai_m{met}"] = (met / 10.0, None, True, True)
    out["ers_photo"] = (1.0, 0.35, False, True)
    out["ers_nophoto"] = (1.0, 0.35, False, False)
    return out


def main(name):
    """Solve one named case and save its column."""
    # isort: off
    from vulcan_forward import vulcan_chem  # first: it sets jax x64 before exojax
    from jwst_tool.model_defs import DEFAULT_NETWORK, NETWORKS, RT_PTOP_DEFAULT

    network, atom_list = NETWORKS[DEFAULT_NETWORK]  # the tool's default network, named before vulcan3 freezes one
    vulcan_chem.select_network(network, atom_list)
    import vulcan3
    from vulcan3 import composition as ba
    # isort: on

    factor, co_target, morning, photo = cases()[name]
    base_cfg = vulcan3.load_config("W39b")
    ovr = {f"{e}_H": float(getattr(base_cfg, f"{e}_H")) * factor for e in ELEMS}
    if co_target is not None:
        ovr["C_H"] = co_target * ovr["O_H"]
    if morning:
        ovr["atm_file"] = str(morning_tp_kzz())
    # The shipped model top, so the validation columns are the tool's: the engine's own default reaches higher.
    profile = dict(
        network=network,
        atom_list=atom_list,
        vulcan_cfg_name="W39b",
        use_photo=photo,
        yconv_cri=0.01,
        cfg_overrides=ovr,
        art_ptop_bar=RT_PTOP_DEFAULT,
    )

    t0 = time.time()
    model = vulcan_chem.build_chem_model(profile)
    final, _ = model.run_diag(np.zeros(4))
    conv = bool(model.conv_normal_at_exit(final))
    y = np.asarray(final.y, dtype=np.float64)
    ymix = y / y.sum(axis=1, keepdims=True)
    species = [s for s, _ in sorted(model.sidx.items(), key=lambda kv: kv[1])]

    # The gate: the CONVERGED column's elemental inventory must match the case.
    n = {e: np.array([ba.compo[e][ba.compo_row.index(s)] for s in species], float) for e in ("H", "C", "O")}
    CH = float((ymix * n["C"]).sum() / (ymix * n["H"]).sum())
    OH = float((ymix * n["O"]).sum() / (ymix * n["H"]).sum())
    want_OH = float(base_cfg.O_H) * factor
    want_CH = (co_target if co_target is not None else float(base_cfg.C_H) / float(base_cfg.O_H)) * want_OH
    print(f"[{name}] column C/H {CH:.4e} (want ~{want_CH:.4e})  O/H {OH:.4e} (want ~{want_OH:.4e})  C/O {CH / OH:.4f}")
    assert abs(OH / want_OH - 1.0) < 0.05 and abs(CH / want_CH - 1.0) < 0.05, (
        f"{name}: converged inventory does not match the case"
    )
    assert conv, f"{name} did not certify convergence"

    np.savez_compressed(
        OUT / f"case_{name}.npz",
        p_bar=model.p_bar,
        T=model.T_base,
        ymix=ymix,
        species=np.array(species),
        conv_normal=conv,
        overrides=str(ovr),
        accept_count=int(final.accept_count),
        longdy=float(final.longdy),
    )
    print(
        f"[{name}] steps {int(final.accept_count)}  longdy {float(final.longdy):.4f}"
        f"  conv_normal {conv}  ({time.time() - t0:.0f} s)",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in cases():
        print("usage: chem_cases.py <case>; cases:", " ".join(cases()))
        sys.exit(2)
    main(sys.argv[1])
