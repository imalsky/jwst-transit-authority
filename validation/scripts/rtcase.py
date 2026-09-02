"""Shared RT + reference-data helpers for the Tsai and ERS figures: the tool's
RT model at Tsai et al. 2023's radius anchor, the saved chemistry columns in
validation/data/case_*.npz, the published columns and gCMCRT spectra, and the
R = 100 binning both figures report on.

Import order is load-bearing: vulcan_chem before anything exojax.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vulcan_forward import vulcan_chem  # noqa: F401  (must precede exojax)

import jax.numpy as jnp  # noqa: E402
from vulcan_forward import exojax_rt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RSTAR = 0.932 * 6.957e10
RT_MOLS = ["H2O", "CH4", "CO", "CO2", "H2S", "SH", "SO", "SO2"]
# Tsai et al. 2023's own radius convention (their z = 0 sits at 0.8995 bar and
# equals the catalogue radius; see vulcan-forward notes 2026-08-17).
ANCHOR = dict(p_ref_bar=0.8995, rp_cm=9.1438268e9, gs_cgs=426.0, rstar_cm=RSTAR)


def build_trt(wl_lo_um: float, wl_hi_um: float):
    return exojax_rt.build_rt_model(dict(
        molecules=RT_MOLS, nu_min=1e4 / wl_hi_um, nu_max=1e4 / wl_lo_um,
        opacity_mode="exomolop", art_nlayer=80, art_ptop_bar=1e-8,
        art_pbtm_bar=7.6, **ANCHOR))


def _masses(species):
    from vulcan_jax import composition as ba
    return np.array([ba.compo["mass"][ba.compo_row.index(s)] for s in species])


def _regrid(p_from, v, p_to, log=True):
    x, xt = np.log10(p_from), np.log10(p_to)
    o = np.argsort(x)
    if log:
        return 10.0 ** np.interp(xt, x[o], np.log10(np.maximum(v[o], 1e-300)))
    return np.interp(xt, x[o], v[o])


def case_column(name):
    """A saved chemistry case -> dict of arrays on the chemistry grid."""
    z = np.load(DATA / f"case_{name}.npz", allow_pickle=True)
    species = [str(s) for s in z["species"]]
    ymix = z["ymix"]
    mmw = ymix @ _masses(species)
    i = {s: species.index(s) for s in RT_MOLS + ["H2", "He"]}
    return dict(p=z["p_bar"], T=z["T"], mmw=mmw,
                vmr={s: ymix[:, i[s]] for s in RT_MOLS},
                h2=ymix[:, i["H2"]], he=ymix[:, i["He"]],
                so2=ymix[:, species.index("SO2")])


def depth_ppm(trt, col, cloud=None, wo=None):
    """Transmission depth (ppm) for a chemistry column; optionally leave-one-out.

    Returns (wl_um_sorted, depth) or (wl, depth, {mol: depth_wo}) with wo.
    """
    p = np.asarray(trt.p_art_bar)
    vmr = {m: jnp.asarray(_regrid(col["p"], col["vmr"][m], p)) for m in RT_MOLS}
    args = (vmr, jnp.asarray(_regrid(col["p"], col["h2"], p)),
            jnp.asarray(_regrid(col["p"], col["T"], p, log=False)),
            jnp.asarray(_regrid(col["p"], col["mmw"], p, log=False)))
    kw = dict(vmr_he=jnp.asarray(_regrid(col["p"], col["he"], p)),
              cloud=None if cloud is None else jnp.asarray(cloud))
    o = np.argsort(trt.wl_um)
    wl = trt.wl_um[o]
    if wo is None:
        d = np.asarray(trt.transmission_depth(*args, **kw))
        return wl, d[o] * 1e6
    d, d_wo = trt.transmission_depth_r(*args[:2], *args[2:], 0.0, wo_mols=wo, **kw)
    return (wl, np.asarray(d)[o] * 1e6,
            {m: np.asarray(d_wo[i])[o] * 1e6 for i, m in enumerate(wo)})


def published_column(term):
    """Tsai et al. 2023's released 10x VULCAN column (evening/morning)."""
    cols = ["P", "T", "z", "mu", "H2", "H2O", "CH4", "CO", "CO2", "H2S",
            "S", "S2", "SH", "SO", "SO2"]
    a = np.genfromtxt(DATA / "ers2023" / "photochem"
                      / f"wasp39b_10Xsolar_{term}_vulcan.txt", skip_header=2)
    c = {k: a[:, j] for j, k in enumerate(cols)}
    p = c["P"] / 1e6
    he = np.full_like(p, 0.168) * c["H2"]   # He/H2 from our own run (notes 2026-08-17)
    return dict(p=p, T=c["T"], mmw=c["mu"],
                vmr={m: c[m] for m in RT_MOLS}, h2=c["H2"], he=he, so2=c["SO2"])


def gcmcrt_depth_ppm(met, term):
    """Their gCMCRT spectrum; depth = (H(1)^2 + 2 col2)/Rstar^2, H(1) from the header."""
    path = DATA / "tsai2023" / "vulcan" / f"Transmission_{met}x_{term}.txt"
    with open(path) as f:
        h1 = float(f.readline().split()[1])
    a = np.genfromtxt(path, skip_header=1)
    return a[:, 0], (h1 ** 2 + 2.0 * a[:, 1]) / RSTAR ** 2 * 1e6


def bin_to(wl, y, centers):
    """Boxcar average into bins whose edges are geometric midpoints of centers."""
    e = np.sqrt(centers[:-1] * centers[1:])
    edges = np.concatenate([[centers[0] ** 2 / e[0]], e, [centers[-1] ** 2 / e[-1]]])
    idx = np.digitize(wl, edges) - 1
    out = np.full(centers.size, np.nan)
    for i in range(centers.size):
        m = idx == i
        if m.any():
            out[i] = float(np.mean(y[m]))
    return out


def r100_grid(lo=3.02, hi=4.98):
    n = int(np.round(np.log(hi / lo) * 100.0))
    return lo * np.exp(np.arange(n + 1) / 100.0)
