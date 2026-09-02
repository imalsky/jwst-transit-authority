"""WASP-39 b chemistry: VULCAN 2.0 vs VULCAN 3.0 (JAX) on identical inputs --
six species' volume mixing ratios against pressure, and their relative
difference. Input: validation/data/chemistry_w39b_vulcan2_vs_vulcan3.npz
(extracted by inputs/extract_vul_columns.py from the two solver outputs
jax_paper/data/W39b_{master,jax}_paper.vul). numpy + matplotlib only.

    python validation/scripts/fig_chemistry_w39b_vulcan2_vs_vulcan3.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # validation/scripts
sys.path.insert(0, str(HERE.parent))              # figstyle
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import CYC, INK, DASH_KW, pastel, panels, legend, save  # noqa: E402
fs.use()
DATA = fs.DATA

SPECIES = ["H2O", "CO", "CO2", "CH4", "SO2", "H2S"]
LABEL = {"H2O": "H$_2$O", "CO": "CO", "CO2": "CO$_2$", "CH4": "CH$_4$",
         "SO2": "SO$_2$", "H2S": "H$_2$S"}
FLOOR = 1e-12

z = np.load(DATA / "chemistry_w39b_vulcan2_vs_vulcan3.npz")
p2 = p3 = z["p_bar"]      # extract_vul_columns.py asserted the two grids are equal
y2 = {s: z[f"vulcan2_{s}"] for s in SPECIES}
y3 = {s: z[f"vulcan3_{s}"] for s in SPECIES}

fig, (ax, dx) = panels(1, 2)
worst = 0.0
for s, c in zip(SPECIES, CYC):
    ax.plot(y2[s], p2, color=pastel(c), lw=2.5, solid_capstyle="round")
    ax.plot(y3[s], p3, color=c, lw=1.2, **DASH_KW)
    rel = np.where(y2[s] > FLOOR, np.abs(y3[s] - y2[s]) / y2[s], np.nan)
    dx.plot(rel, p2, color=c, lw=1.5, label=LABEL[s])
    worst = max(worst, float(np.max(np.abs(y3[s] - y2[s]))))
ax.plot([], [], color=pastel(INK), lw=2.5, label="VULCAN 2.0")
ax.plot([], [], color=INK, lw=1.2, label="VULCAN 3.0 (JAX)", **DASH_KW)

for a in (ax, dx):
    a.set_xscale("log"); a.set_yscale("log")
    a.set_ylim(p2.max(), p2.min()); a.set_ylabel("pressure (bar)")
ax.set_xlim(FLOOR, None); ax.set_xlabel("volume mixing ratio")
dx.set_xlabel(r"$|y_{3.0} - y_{2.0}|\,/\,y_{2.0}$")
legend(ax); legend(dx)
save(fig, "chemistry_w39b_vulcan2_vs_vulcan3.png")
print(f"max abs ymix difference over the six species: {worst:.3e}")
