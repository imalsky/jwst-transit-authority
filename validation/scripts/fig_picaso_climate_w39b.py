"""PICASO climate on WASP-39 b: picaso 4.1's own WASP-39 b notebook with
on-the-fly chemistry at picaso's composition and at the tool's composition,
against picaso's reference profile reference/base_cases/HJ.pt (left); and the
temperature change when picaso 4.1's stellar binning (issue #413) is replaced
by the bin-edge integral of picaso's terradev branch, around WASP-39 (G8) and
around a 3250 K M dwarf at the same T_eq (right). Inputs:
validation/data/picaso_climate_*.npz (inputs/picaso_cases.py climate_*).

    python validation/scripts/fig_picaso_climate_w39b.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import CYC, INK, RED, DASH_KW, DOT_KW, pastel, panels, legend, save  # noqa: E402
fs.use()


def load(case):
    return np.load(fs.DATA / f"picaso_climate_{case}.npz")


nb, tool = load("notebook"), load("tool")
fig, (ax, bx) = panels(1, 2)
ax.plot(nb["T_ref"], nb["P_ref"], color=pastel(INK), lw=2.5, solid_capstyle="round",
        label="picaso reference (HJ.pt)")
ax.plot(nb["T"], nb["P"], color=INK, lw=1.2, label="picaso composition", **DASH_KW)
ax.plot(tool["T"], tool["P"], color=RED, lw=1.2, label="tool composition", **DOT_KW)
ax.set_xlabel("temperature (K)")
p_all = np.r_[nb["P"], nb["P_ref"]]
ax.set_ylim(p_all.max(), p_all.min())
for name, c, lab in (("fgk", CYC[0], "WASP-39 (G8)"), ("mdwarf", CYC[1], "3250 K M dwarf")):
    s, e = load(f"{name}_stock"), load(f"{name}_edges")
    bx.plot(e["T"] - s["T"], s["P"], color=c, lw=1.5, label=lab)
    m = (s["P"] >= 1e-5) & (s["P"] <= 1.0)
    d = e["T"][m] - s["T"][m]
    print(f"#413 {lab}: dT over 1e-5..1 bar [{d.min():+.2f}, {d.max():+.2f}] K")
bx.set_xlabel("temperature change, bin edges $-$ picaso 4.1 (K)")
bx.set_ylim(s["P"].max(), s["P"].min())
for a in (ax, bx):
    a.set_yscale("log"); a.set_ylabel("pressure (bar)")
    legend(a)
m = (nb["P"] > 1e-4) & (nb["P"] < 10)
for lab, z in (("picaso composition", nb), ("tool composition", tool)):
    d = z["T"][m] - np.interp(np.log(z["P"][m]), np.log(nb["P_ref"]), nb["T_ref"])
    print(f"{lab} - HJ.pt over 1e-4..10 bar: median {np.median(d):+.1f} K, max |dT| {np.abs(d).max():.1f} K")
save(fig, "picaso_climate_w39b.png")
