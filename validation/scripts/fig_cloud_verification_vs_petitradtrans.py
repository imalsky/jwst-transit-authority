"""Power-law cloud deck vs petitRADTRANS 3.4.0 on identical inputs (clear and
cloudy WASP-39 b, same ExoMolOP H2O table), binned to R = 300. The pRT side is
cached in validation/data/prt_spectrum_w39b.npz, which carries the shared
configuration and the deck parameters in its `meta` field.

    JAX_PLATFORM_NAME=cpu python validation/scripts/fig_cloud_verification_vs_petitradtrans.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # validation/scripts
sys.path.insert(0, str(HERE.parent))              # figstyle
sys.path.insert(0, str(HERE))                     # rtcase
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import CYC, square, overlay, legend, save  # noqa: E402
fs.use()

import rtcase as rc                               # noqa: E402
import jax.numpy as jnp                           # noqa: E402

R_BIN = 300


def binned(w, d, edges, min_count=2):
    tot, _ = np.histogram(w, edges, weights=d)
    cnt, _ = np.histogram(w, edges)
    out = np.full(cnt.shape, np.nan)
    np.divide(tot, cnt, out=out, where=cnt >= min_count)
    return out


z = np.load(rc.DATA / "prt_spectrum_w39b.npz")
m = json.loads(bytes(z["meta"].tobytes()).decode())
cfg, deck = m["config"], m["deck"]
trt = rc.exojax_rt.build_rt_model(dict(
    molecules=["H2O"], nu_min=1e4 / cfg["wl_um"][1], nu_max=1e4 / cfg["wl_um"][0],
    art_nlayer=cfg["nlayer"], art_ptop_bar=cfg["p_top_bar"], art_pbtm_bar=cfg["p_btm_bar"],
    rp_cm=cfg["planet_radius_cm"], gs_cgs=cfg["reference_gravity"],
    rstar_cm=cfg["rstar_cm"], p_ref_bar=cfg["reference_pressure_bar"], use_rayleigh=True))
n = len(trt.p_art_bar)
ins = dict(vmr={"H2O": jnp.full(n, cfg["vmr"]["H2O"])}, vmr_h2=jnp.full(n, cfg["vmr"]["H2"]),
           vmr_he=jnp.full(n, cfg["vmr"]["He"]), T_art=jnp.full(n, cfg["T_iso_K"]),
           mmw_art=jnp.full(n, cfg["mmw"]))
wl = np.asarray(trt.wl_um)
ours = {"clear": np.asarray(trt.transmission_depth(cloud=None, **ins)) * 1e6,
        "cloudy": np.asarray(trt.transmission_depth(cloud=jnp.array(deck), **ins)) * 1e6}

# 1% inset at both band edges, as the committed prt_ref_*.npz fixtures mask.
lo, hi = cfg["wl_um"][0] * 1.01, cfg["wl_um"][1] * 0.99
edges = np.geomspace(lo, hi, int(np.log(hi / lo) * R_BIN) + 1)
cen = np.sqrt(edges[:-1] * edges[1:])

fig, ax = square()
for tag, c in (("clear", CYC[0]), ("cloudy", CYC[1])):
    bj = binned(wl, ours[tag], edges)
    bp = binned(z[f"wl_um_{tag}"], z[f"depth_ppm_{tag}"], edges)
    overlay(ax, cen, bp, bj, c, (f"petitRADTRANS, {tag}", f"VULCAN-JAX, {tag}"))
    good = np.isfinite(bj) & np.isfinite(bp)
    print(f"{tag:6s} rms {np.sqrt(((bj[good] - bp[good]) ** 2).mean()):5.2f} ppm | "
          f"feature amplitude {np.nanmax(bj) - np.nanmin(bj):6.0f} ppm")
ax.set_xlabel(r"wavelength ($\mu$m)")
ax.set_ylabel("transit depth (ppm)")
legend(ax)
save(fig, "cloud_verification_vs_petitradtrans.png")
