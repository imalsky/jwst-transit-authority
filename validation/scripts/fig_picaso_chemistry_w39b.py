"""WASP-39 b equilibrium chemistry on one column: VULCAN3's ExoGibbs seed vs
the PICASO engine (picaso 4.1 / photochem on-the-fly) at the same composition
(left), and the PICASO engine with and without the Lodders 2020 metals, whose
silicate condensates take oxygen (right). Shipped WASP-39 b table, 10x solar,
C/O 0.55. Input: validation/data/picaso_chem.npz (inputs/picaso_cases.py chem).

    python validation/scripts/fig_picaso_chemistry_w39b.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import CYC, INK, DASH_KW, pastel, panels, legend, save  # noqa: E402
fs.use()

SPECIES = ["H2O", "CO", "CH4", "CO2", "NH3", "HCN"]
LABEL = {"H2O": "H$_2$O", "CO": "CO", "CH4": "CH$_4$", "CO2": "CO$_2$",
         "NH3": "NH$_3$", "HCN": "HCN"}
FLOOR = 1e-12
z = np.load(fs.DATA / "picaso_chem.npz")
p = z["p_bar"]

fig, (ax, bx) = panels(1, 2)
worst = {}
for s, c in zip(SPECIES, CYC):
    ax.plot(z[f"vulcan_{s}"], p, color=pastel(c), lw=2.5, solid_capstyle="round", label=LABEL[s])
    ax.plot(z[f"picaso_cnos_{s}"], p, color=c, lw=1.2, **DASH_KW)
    bx.plot(z[f"picaso_cnos_{s}"], p, color=pastel(c), lw=2.5, solid_capstyle="round")
    bx.plot(z[f"picaso_full_{s}"], p, color=c, lw=1.2, **DASH_KW)
    m = (p >= 1e-5) & (p <= 1.0) & (z[f"vulcan_{s}"] > 1e-8)
    worst[s] = float(np.abs(np.log10(z[f"picaso_cnos_{s}"][m] / z[f"vulcan_{s}"][m])).max())
ax.plot([], [], color=pastel(INK), lw=2.5, label="VULCAN3 seed (ExoGibbs)")
ax.plot([], [], color=INK, lw=1.2, label="PICASO (C, N, O, S)", **DASH_KW)
bx.plot([], [], color=pastel(INK), lw=2.5, label="PICASO (C, N, O, S)")
bx.plot([], [], color=INK, lw=1.2, label="PICASO (+ metals)", **DASH_KW)
for a in (ax, bx):
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylim(p.max(), p.min())
    a.set_ylabel("pressure (bar)")
    # the decade above the most abundant curve, so no curve runs along the frame
    top = max(np.max(z[k]) for k in z.files if k.endswith(tuple(SPECIES)))
    a.set_xlim(FLOOR, 10.0 ** np.ceil(np.log10(top)))
    a.set_xlabel("volume mixing ratio")
    legend(a, fontsize=8)
save(fig, "picaso_chemistry_w39b.png")
print("max |dlog10| PICASO(C,N,O,S) - seed over 1e-5..1 bar:",
      {k: round(v, 4) for k, v in worst.items()})
