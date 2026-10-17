"""WASP-39 b on its default Guillot profile at one constant Kzz = 1e9 cm^2/s:
VULCAN3 kinetics with photolysis off against PICASO's quench approximation
(Zahnle & Marley 2014 timescales, kinetic CO2) and PICASO equilibrium --
mixing ratios (left) and transmission spectra at R = 100 (right). Inputs:
validation/data/picaso_spectra_*.npz (inputs/picaso_cases.py spectra_*).

    python validation/scripts/fig_picaso_quench_vs_kinetics.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import CYC, INK, RED, DASH_KW, DOT_KW, pastel, panels, legend, save  # noqa: E402
fs.use()

SPECIES = ["H2O", "CO", "CH4", "CO2", "NH3"]
LABEL = {"H2O": "H$_2$O", "CO": "CO", "CH4": "CH$_4$", "CO2": "CO$_2$", "NH3": "NH$_3$"}
vul, q, eq = (np.load(fs.DATA / f"picaso_spectra_{k}.npz") for k in ("vul_K9", "pic_q9", "pic_eq"))


def binned(z, lo=1.0, hi=12.0, R=100.0):
    """Cell means over equal log-wavelength bins, complete bins only."""
    wl, d = z["wl_um"], z["depth"] * 1e6
    edges = lo * np.exp(np.arange(0, np.log(hi / lo) * R + 1) / R)
    i = np.digitize(wl, edges) - 1
    ok = (i >= 0) & (i < edges.size - 1)
    n = np.bincount(i[ok], minlength=edges.size - 1)
    s = np.bincount(i[ok], weights=d[ok], minlength=edges.size - 1)
    keep = n > 0
    return np.sqrt(edges[:-1] * edges[1:])[keep], (s / np.maximum(n, 1))[keep]


fig, (ax, bx) = panels(1, 2)
for s, c in zip(SPECIES, CYC):
    ax.plot(vul[f"y_{s}"], vul["p_bar"], color=pastel(c), lw=2.5, solid_capstyle="round", label=LABEL[s])
    ax.plot(q[f"y_{s}"], q["p_bar"], color=c, lw=1.2, **DASH_KW)
ax.plot([], [], color=pastel(INK), lw=2.5, label="VULCAN3 kinetics")
ax.plot([], [], color=INK, lw=1.2, label="PICASO quench", **DASH_KW)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(vul["p_bar"].max(), vul["p_bar"].min())
# whole decades around the curves, so none runs along the frame
lo = min(np.min(z[f"y_{s}"]) for z in (vul, q) for s in SPECIES)
top = max(np.max(z[f"y_{s}"]) for z in (vul, q) for s in SPECIES)
ax.set_xlim(10.0 ** np.floor(np.log10(lo)), 10.0 ** np.ceil(np.log10(top)))
ax.set_xlabel("volume mixing ratio"); ax.set_ylabel("pressure (bar)")
w, dv = binned(vul); _, dq = binned(q); _, de = binned(eq)
bx.plot(w, dv, color=pastel(INK), lw=2.5, solid_capstyle="round", label="VULCAN3 kinetics")
bx.plot(w, dq, color=INK, lw=1.2, label="PICASO quench", **DASH_KW)
bx.plot(w, de, color=RED, lw=1.2, label="PICASO equilibrium", **DOT_KW)
bx.set_xlim(w[0], w[-1]); bx.set_xlabel(r"wavelength ($\mu$m)"); bx.set_ylabel("transit depth (ppm)")
legend(ax, fontsize=9); legend(bx)
save(fig, "picaso_quench_vs_kinetics.png")
for lab, d in (("PICASO quench", dq), ("PICASO equilibrium", de)):
    r = d - dv
    print(f"{lab} - VULCAN kinetics, R=100 1-12 um: mean {r.mean():+.1f} ppm, rms {np.sqrt(np.mean(r**2)):.1f}, max |d| {np.abs(r).max():.1f}")
