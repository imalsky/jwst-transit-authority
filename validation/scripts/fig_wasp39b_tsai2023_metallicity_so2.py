"""WASP-39 b vs Tsai et al. 2023 (gCMCRT spectra, Zenodo 7542781):
terminator-averaged, cloud-free, their radius anchor, 5/10/20x solar, plus the
SO2 contribution and the SO2 column at 10x solar. Our chemistry is
validation/data/case_tsai_*.npz (inputs/chem_cases.py); their spectra and
columns are validation/data/tsai2023/ and ers2023/photochem/; both
SO2-contribution curves use our RT.

    JAX_PLATFORM_NAME=cpu python validation/scripts/fig_wasp39b_tsai2023_metallicity_so2.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # validation/scripts
sys.path.insert(0, str(HERE.parent))              # figstyle
sys.path.insert(0, str(HERE))                     # rtcase
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import CYC, DASH_KW, pastel, panels, overlay, legend, save  # noqa: E402
fs.use()

import rtcase as rc                               # noqa: E402

MET = (5, 10, 20)
TERM = {"evening": "east", "morning": "west"}   # ours -> their file naming
CASE = {"evening": "e", "morning": "m"}

trt = rc.build_trt(2.95, 5.05)
grid = rc.r100_grid()

ours, theirs = {}, {}
for met in MET:
    ours[met] = np.mean([rc.bin_to(*rc.depth_ppm(trt, rc.case_column(f"tsai_{CASE[t]}{met}")), grid)
                         for t in TERM], axis=0)
    theirs[met] = np.mean([rc.bin_to(*rc.gcmcrt_depth_ppm(met, t), grid)
                           for t in TERM.values()], axis=0)

pub, pub_wo, ours10_wo = [], [], []
for term in TERM:
    wl, d, dw = rc.depth_ppm(trt, rc.published_column(term), wo=["SO2"])
    pub.append(rc.bin_to(wl, d, grid))
    pub_wo.append(rc.bin_to(wl, dw["SO2"], grid))
    wl, d, dw = rc.depth_ppm(trt, rc.case_column(f"tsai_{CASE[term]}10"), wo=["SO2"])
    ours10_wo.append(rc.bin_to(wl, d, grid) - rc.bin_to(wl, dw["SO2"], grid))
rt_on_theirs = np.mean(pub, axis=0)
so2_theirs_ourrt = rt_on_theirs - np.mean(pub_wo, axis=0)
so2_ours = np.mean(ours10_wo, axis=0)

fig, ((a, b), (c, d)) = panels(2, 2)
for met, col in zip(MET, CYC):
    overlay(a, grid, theirs[met], ours[met], col,
            (f"Tsai et al. 2023, {met}$\\times$ solar", f"this work, {met}$\\times$ solar"))
    r = ours[met] - theirs[met]
    b.plot(grid, r, color=col, lw=1.5, label=f"{met}$\\times$ solar")
    print(f"{met}x: mean diff {np.mean(r):+.0f} ppm, rms about mean {np.std(r - np.mean(r)):.0f} ppm")
r = rt_on_theirs - theirs[10]
b.plot(grid, r, color=CYC[1], lw=1.5, label="our RT, their chemistry, 10$\\times$ solar", **DASH_KW)
print(f"our RT on their 10x chemistry: rms about mean {np.std(r - np.mean(r)):.0f} ppm")
a.set_ylabel("transit depth (ppm)")
b.set_ylabel("this work $-$ Tsai et al. 2023 (ppm)")

c.plot(grid, so2_ours, color=CYC[1], lw=1.5, label="this work")
c.plot(grid, so2_theirs_ourrt, color=CYC[1], lw=1.5, label="our RT, their chemistry", **DASH_KW)
c.set_ylabel("SO$_2$ contribution at 10$\\times$ solar (ppm)")
for ax in (a, b, c):
    ax.set_xlabel(r"wavelength ($\mu$m)")

for term, col in zip(TERM, CYC):
    col_o = rc.case_column(f"tsai_{CASE[term]}10")
    pb = rc.published_column(term)
    d.plot(pb["so2"], pb["p"], color=pastel(col), lw=4, solid_capstyle="round",
           label=f"Tsai et al. 2023, {term}")
    d.plot(col_o["so2"], col_o["p"], color=col, lw=1.5, label=f"this work, {term}", **DASH_KW)
d.set_xscale("log"); d.set_yscale("log")
# The SO2 feature spans 4e-2 to 4e-6 bar in all four columns; the only other
# SO2 above 1e-9 sits at the column edges (5e-9 bar top, 27-50 bar in Tsai's).
d.set_xlim(1e-9, None); d.set_ylim(5e-2, 3e-6)
d.set_xlabel("SO$_2$ volume mixing ratio at 10$\\times$ solar")
d.set_ylabel("pressure (bar)")
for ax in (a, b, c, d):
    legend(ax)
save(fig, "wasp39b_tsai2023_metallicity_so2.png")
