"""This work vs petitRADTRANS 3.4.0 on the JWST tool's own converged
atmospheres: four planets in transmission, two in emission, identical
P/T/mmw/VMRs/geometry and k-table files, lines only, 3.03-5.17 um. Inputs:
validation/data/atmos_*.npz (inputs/atmos_cases.py) and prt_*.npz
(inputs/prt_reference.py). numpy + matplotlib only.

    python validation/scripts/fig_rt_verification_six_atmospheres.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # validation/scripts
sys.path.insert(0, str(HERE.parent))              # figstyle
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import CYC, INK, RED, panels, legend, save  # noqa: E402
fs.use()
DATA = fs.DATA

CASES = {"transmission": ["wasp39b", "hd189733b", "wasp107b", "hd209458b"],
         "emission": ["wasp39b", "hd189733b"]}
KEY = {"transmission": "depth_cmp_ppm", "emission": "flux_cmp"}
PKEY = {"transmission": "depth_ppm", "emission": "flux_per_cm1"}

fig, axes = panels(1, 2)
for ax, (mode, plist) in zip(axes, CASES.items()):
    for planet, col in zip(plist, CYC if mode == "transmission" else (INK, RED)):
        a = np.load(DATA / f"atmos_{planet}_{mode}.npz")
        b = np.load(DATA / f"prt_{planet}_{mode}.npz")
        wl, ours = a["wl_cmp"], a[KEY[mode]]
        o = np.argsort(wl)
        wl, ours = wl[o], ours[o]
        m = (wl > 3.03 * 1.01) & (wl < 5.17 * 0.99)
        r = ours[m] / np.interp(wl[m], b["wl_um"], b[PKEY[mode]])
        ax.plot(wl[m], 100 * (r - 1), color=col, lw=1.2,
                label=f"{a['label']}, $g$ = {float(a['gs_cgs']):.0f} cm s$^{{-2}}$")
        print(f"{mode:12s} {str(a['label']):12s} mean ratio {r.mean():.4f}  rms {100 * np.std(r):.3f} %  "
              f"max |dev| {100 * np.max(np.abs(r - 1)):.3f} %")
    ax.set_xlabel(r"wavelength ($\mu$m)")
    ax.set_ylabel(f"{mode}: this work / petitRADTRANS $-$ 1 (%)")
    legend(ax)
save(fig, "rt_verification_six_atmospheres.png")
