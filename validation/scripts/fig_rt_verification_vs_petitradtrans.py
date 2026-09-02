"""Radiative transfer vs petitRADTRANS 3.4.0 on the engine's committed
fixtures (isothermal H2O transmission, the WASP-39 b eight-species column,
H2O emission) and a grey absorber vs the Lecavelier des Etangs et al. (2008)
closed form. Inputs: validation/data/prt_ref/prt_ref_*.npz and
wasp39b_10Xsolar_evening_vulcan.txt (vulcan-forward's own test fixtures) plus
the installed ExoMolOP k-tables.

    JAX_PLATFORM_NAME=cpu python validation/scripts/fig_rt_verification_vs_petitradtrans.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # validation/scripts
sys.path.insert(0, str(HERE.parent))              # figstyle
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import INK, RED, MS, panels, overlay2, legend, save  # noqa: E402
fs.use()
DATA = fs.DATA / "prt_ref"

from vulcan_forward import vulcan_chem  # noqa: E402,F401  (must precede exojax)
import jax.numpy as jnp  # noqa: E402
from vulcan_forward import exojax_rt  # noqa: E402

RSTAR_W39 = 0.932 * 6.957e10
RJ = 7.1492e9
KB, MH = 1.380649e-16, 1.66053907e-24


def load(name):
    z = np.load(DATA / name)
    return z, json.loads(bytes(np.asarray(z["meta"])))


def isothermal_h2o():
    z, _ = load("prt_ref_isothermal_h2o_trans.npz")
    n = 100
    trt = exojax_rt.build_rt_model(dict(
        molecules=["H2O"], nu_min=2000.0, nu_max=10000.0, opacity_mode="exomolop",
        art_nlayer=n, art_ptop_bar=1e-6, art_pbtm_bar=1e2, p_ref_bar=10.0,
        rp_cm=RJ, gs_cgs=1e3, rstar_cm=RSTAR_W39))
    zeros = jnp.zeros(n)
    depth = np.asarray(trt.transmission_depth(
        {"H2O": jnp.full(n, 1e-3)}, zeros, jnp.full(n, 1000.0), jnp.full(n, 2.33), vmr_he=zeros))
    o = np.argsort(trt.wl_um)
    wl, r = trt.wl_um[o], np.sqrt(depth[o]) * RSTAR_W39 / 1e5
    m = (wl > 1.01) & (wl < 4.95)
    return wl[m], r[m], np.interp(wl[m], z["wl_um"], z["prt_radius_cm"]) / 1e5


def w39b_8species():
    z, _ = load("prt_ref_w39b_8species_trans.npz")
    mols = ["H2O", "CH4", "CO", "CO2", "H2S", "SH", "SO", "SO2"]
    cols = ["P_dyn", "T", "z", "mu", "H2", "H2O", "CH4", "CO", "CO2", "H2S",
            "S", "S2", "SH", "SO", "SO2"]
    raw = np.genfromtxt(DATA / "wasp39b_10Xsolar_evening_vulcan.txt", skip_header=2)
    c = {k: raw[:, i] for i, k in enumerate(cols)}
    ps = c["P_dyn"] / 1e6
    order = np.argsort(ps)
    ps = ps[order]

    def regrid(ys, pd, log=True):
        if log:
            return 10.0 ** np.interp(np.log10(pd), np.log10(ps),
                                     np.log10(np.maximum(ys[order], 1e-300)))
        return np.interp(np.log10(pd), np.log10(ps), ys[order])

    n = 80
    trt = exojax_rt.build_rt_model(dict(
        molecules=mols, nu_min=1e4 / 5.0, nu_max=1e4 / 3.0, opacity_mode="exomolop",
        art_nlayer=n, art_ptop_bar=1e-8, art_pbtm_bar=50.0, p_ref_bar=0.8995,
        rp_cm=9.1438268e9, gs_cgs=426.0, rstar_cm=RSTAR_W39))
    p = trt.p_art_bar
    zeros = jnp.zeros(n)
    depth = np.asarray(trt.transmission_depth(
        {m: jnp.asarray(regrid(c[m], p)) for m in mols}, zeros,
        jnp.asarray(regrid(c["T"], p, log=False)),
        jnp.asarray(regrid(c["mu"], p, log=False)), vmr_he=zeros))
    o = np.argsort(trt.wl_um)
    wl, d = trt.wl_um[o], depth[o] * 1e6
    m = (wl > 3.0 * 1.005) & (wl < 5.0 * 0.995)
    return wl[m], d[m], np.interp(wl[m], z["wl_um"], z["prt_depth_ppm"])


def emission_h2o():
    z, _ = load("prt_ref_emission_h2o.npz")
    n = 80
    prof = dict(molecules=["H2O"], nu_min=1e4 / 5.0, nu_max=1e4 / 3.0,
                opacity_mode="exomolop", art_nlayer=n, art_ptop_bar=1e-6,
                art_pbtm_bar=1e2, rp_cm=100 * RJ, gs_cgs=1e3, rstar_cm=RSTAR_W39)
    trt = exojax_rt.build_rt_model(prof)
    emod = exojax_rt.build_emis_model(trt, prof)
    p = np.asarray(emod.p_art_bar)
    x = (np.log10(p) + 6.0) / 8.0
    zeros = jnp.zeros(n)
    flux = np.asarray(emod.emission_flux(
        {"H2O": jnp.full(n, 1e-3)}, zeros, jnp.asarray(900.0 + 1300.0 * x),
        jnp.full(n, 2.33), vmr_he=zeros))
    wl = 1e4 / np.asarray(emod.nu_grid)
    o = np.argsort(wl)
    wl, flux = wl[o], flux[o]
    m = (wl > 3.0 * 1.01) & (wl < 5.0 * 0.99)
    ref = np.interp(wl[m], z["wl_um"], z["prt_flux_per_cm1"])
    nu2 = (1e4 / wl[m]) ** 2          # F_lambda = F_nu~ * nu~^2
    return wl[m], flux[m] * nu2 / 1e11, ref * nu2 / 1e11


def grey_absorber():
    """Isothermal 1000 K, mu 2.33, g 1e3, R = 100 R_J (constant g), anchor 1e-2 bar,
    no H2/He (no CIA), Rayleigh off, the one H2O table loaded with VMR 0."""
    T, mu, g, rp, p_ref = 1000.0, 2.33, 1.0e3, 100 * RJ, 1.0e-2
    n = 120
    trt = exojax_rt.build_rt_model(dict(
        molecules=["H2O"], nu_min=1e4 / 5.0, nu_max=1e4 / 3.0, opacity_mode="exomolop",
        art_nlayer=n, art_ptop_bar=1e-10, art_pbtm_bar=1e2, p_ref_bar=p_ref,
        rp_cm=rp, gs_cgs=g, rstar_cm=RSTAR_W39))
    zeros = jnp.zeros(n)
    H = KB * T / (mu * MH * g)
    rho_ref = p_ref * 1e6 / (g * H)
    tau_eq = np.exp(-0.5772156649)
    kappas = np.logspace(-5, 2, 29)
    h_eng = []
    for k in kappas:
        depth = np.asarray(trt.transmission_depth(
            {"H2O": zeros}, zeros, jnp.full(n, T), jnp.full(n, mu),
            vmr_he=zeros, cloud=jnp.asarray([np.log10(k), 0.0])))
        h_eng.append((np.sqrt(np.median(depth)) * RSTAR_W39 - rp) / H)
    h_ana = np.log(kappas * rho_ref * np.sqrt(2.0 * np.pi * rp * H) / tau_eq)
    return kappas, np.asarray(h_eng), h_ana


fig, ((a, b), (c, d)) = panels(2, 2)
for ax, (wl, ours, ref), ylabel in (
        (a, isothermal_h2o(), "transit radius (km)"),
        (b, w39b_8species(), "transit depth (ppm)"),
        (c, emission_h2o(), "$F_\\lambda$ (10$^{11}$ erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$)")):
    overlay2(ax, wl, ref, ours, ("petitRADTRANS 3.4.0", "VULCAN-JAX"))
    ax.set_xlabel(r"wavelength ($\mu$m)"); ax.set_ylabel(ylabel)
    print(f"  {ylabel}: max |delta| {100 * np.max(np.abs(ours / ref - 1)):.3f} %")
legend(a)

kap, h_eng, h_ana = grey_absorber()
d.plot(kap, h_ana, color=INK, lw=1.5, label="Lecavelier des Etangs et al. 2008")
d.plot(kap, h_eng, "o", ms=MS, color=RED, ls="", label="VULCAN-JAX")
d.set_xscale("log")
d.set_xlabel(r"grey opacity (cm$^2$ g$^{-1}$)")
d.set_ylabel(r"transit height above $p_\mathrm{ref}$ ($H$)")
legend(d)
print(f"  grey absorber: max |delta| {np.max(np.abs(h_eng - h_ana)):.4f} H")
save(fig, "rt_verification_vs_petitradtrans.png")
