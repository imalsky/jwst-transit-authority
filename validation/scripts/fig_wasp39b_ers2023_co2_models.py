"""WASP-39 b vs the JWST ERS CO2 paper's model grid (Zenodo 6959427); their
fiducial 10x solar, C/O = 0.35, grey cloud log kappa = -2.15. Our side is the
photochemical run in validation/data/case_ers_{photo,nophoto}.npz
(inputs/chem_cases.py); the published grid is
validation/data/ers2023/MODEL_FITS/, and all curves are put on the paper's bins.

    JAX_PLATFORM_NAME=cpu python validation/scripts/fig_wasp39b_ers2023_co2_models.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # validation/scripts
sys.path.insert(0, str(HERE.parent))              # figstyle
sys.path.insert(0, str(HERE))                     # rtcase
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import CYC, INK, DASH_KW, DOT_KW, panels, legend, save  # noqa: E402
fs.use()
DATA = fs.DATA

import rtcase as rc                               # noqa: E402

MODELS = DATA / "ers2023" / "MODEL_FITS"
CLOUD = [-2.15, 0.0]
SPC = ["H2O", "CO", "CH4", "H2S"]
LBL = {"H2O": "H$_2$O", "CO": "CO", "CH4": "CH$_4$", "H2S": "H$_2$S"}


def model(name):
    a = np.genfromtxt(MODELS / f"{name}.txt")
    return a[:, 0], a[:, 1] * 1e6


grid = model("ScCHIMERA_MODEL_noCloud")[0]
sc = np.interp(grid, *model("ScCHIMERA_MODEL"))
sc_wo = {m: np.interp(grid, *model(f"ScCHIMERA_MODEL_no{m}")) for m in ["CO2"] + SPC}

trt = rc.build_trt(2.95, 5.65)
wl, d_cloud, wo = rc.depth_ppm(trt, rc.case_column("ers_photo"), cloud=CLOUD, wo=["CO2"] + SPC)
_, d_np, wo_np = rc.depth_ppm(trt, rc.case_column("ers_nophoto"), cloud=CLOUD, wo=["H2S"])
b = lambda y: rc.bin_to(wl, y, grid)  # noqa: E731
ours = b(d_cloud)
contrib = {m: ours - b(wo[m]) for m in wo}
h2s_np = b(d_np) - b(wo_np["H2S"])

fig, (a, c, s) = panels(1, 3)
a.plot(grid, ours - np.nanmean(ours), color=CYC[0], lw=1.5, label="this work")
a.plot(grid, sc - np.mean(sc), color=CYC[1], lw=1.5, label="ScCHIMERA")
for name, lab, col in (("PICASO_MODEL", "PICASO", CYC[2]), ("ATMO_MODEL", "ATMO", CYC[3]),
                       ("PHOENIX_MODEL", "PHOENIX", CYC[4])):
    w, y = model(name)
    a.plot(w, y - np.mean(y), color=col, lw=1.2, label=lab)
a.set_ylabel("transit depth minus its mean (ppm)")

c.plot(grid, contrib["CO2"], color=CYC[0], lw=1.5, label="this work")
c.plot(grid, sc - sc_wo["CO2"], color=CYC[1], lw=1.5, label="ScCHIMERA")
c.set_ylabel("CO$_2$ contribution (ppm)")

for m, col in zip(SPC, CYC):
    s.plot(grid, contrib[m], color=col, lw=1.5, label=LBL[m])
    s.plot(grid, sc - sc_wo[m], color=col, lw=1.5, **DASH_KW)
s.plot(grid, h2s_np, color=CYC[3], lw=1.5, label="H$_2$S, photochemistry off", **DOT_KW)
s.plot([], [], color=INK, lw=1.5, label="ScCHIMERA", **DASH_KW)
s.set_ylabel("contribution (ppm)")

for ax in (a, c, s):
    ax.set_xlim(grid[0], grid[-1])
    ax.set_xlabel(r"wavelength ($\mu$m)")
    legend(ax)
save(fig, "wasp39b_ers2023_co2_models.png")
r = ours - sc
print(f"rms vs ScCHIMERA {np.nanstd(r - np.nanmean(r)):.0f} ppm; CO2 peak ours "
      f"{np.nanmax(contrib['CO2']):.0f} vs {np.nanmax(sc - sc_wo['CO2']):.0f} ppm")
