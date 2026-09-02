"""Writes validation/data/atmos_<planet>_<mode>.npz: one (planet, science_mode)
case at the JWST tool's defaults (GUI molecule set) -- solve the chemistry, map
the column onto the RT grid, and save what both RT codes need (P, T, mean
molecular weight, VMRs, geometry) plus the tool's own wide-band observable and a
lines-only 3.03-5.17 um spectrum for the petitRADTRANS comparison. Needs the
engine data root and this repo's jax environment:

    JAX_PLATFORM_NAME=cpu python validation/scripts/inputs/atmos_cases.py wasp39b transmission
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from jwst_tool import forward, planets
from vulcan_forward import vulcan_chem  # noqa: F401  (env + jax x64; must precede exojax)
from jwst_tool.stellar import phoenix_surface_flux
import jax.numpy as jnp
from vulcan_forward import exojax_rt, interp_map

OUT = Path(__file__).resolve().parents[2] / "data"
WL_CMP = (3.03, 5.17)
planet, mode = sys.argv[1], sys.argv[2]
t0 = time.time()

cp = forward.canonical_params({"planet": planet, "science_mode": mode,
                               "extra_mols": forward.EXTRA_MOLECULES_DEFAULT})
A = forward._assemble_chem(cp, print)
chem = A.build_chem()
theta = jnp.asarray(A.theta)
y, diag = chem.converged_y(theta, return_conv_diag=True)
assert bool(diag.conv_normal) and float(diag.longdy) < chem.yconv_min, \
    f"{planet} {mode}: chemistry not certified (longdy {float(diag.longdy):.3g})"
y = np.asarray(y, dtype=np.float64)
gas = np.ones(y.shape[1])
for s, i in chem.sidx.items():
    if s.endswith("_l_s"):
        gas[int(i)] = 0.0
ymix = y * gas / (y * gas).sum(axis=1, keepdims=True)
mmw_chem = ymix @ np.asarray(chem.species_masses)
print(f"[{planet} {mode}] chemistry certified: {int(diag.accept_count)} steps, "
      f"longdy {float(diag.longdy):.3g}, {time.time() - t0:.0f} s")

profile = dict(A.profile)
rt = exojax_rt.build_rt_model(profile)                      # the tool's wide band
p_art = np.asarray(rt.p_art_bar)
to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)
if A.tp_eval is None:                                       # tabulated T-P (file mode)
    pb, Tb = np.asarray(chem.p_bar), np.asarray(chem.T_base, dtype=np.float64)
    o = np.argsort(pb)
    T_art = np.interp(np.log(p_art), np.log(pb[o]), Tb[o])
else:
    T_art = np.asarray(A.tp_eval(theta[3:], jnp.asarray(p_art)))
col = lambda s: np.asarray(to_art(jnp.asarray(ymix[:, chem.sidx[s]])))  # noqa: E731
vmr = {m: col(A.config.MOLECULES[m]["vulcan"]) for m in rt.molecules}
h2, he = col("H2"), col("He")
mmw = np.asarray(to_art(jnp.asarray(mmw_chem)))
J = jnp.asarray
vmr_j = {k: J(v) for k, v in vmr.items()}
z = J(np.zeros_like(h2))

out = dict(p_art=p_art, T_art=T_art, mmw=mmw, h2=h2, he=he, mols=np.array(rt.molecules),
           rp_cm=profile["rp_cm"], gs_cgs=profile["gs_cgs"], rstar_cm=profile["rstar_cm"],
           p_ref_bar=float(rt.p_ref_bar), label=planets.PLANETS[planet]["label"],
           params_json=json.dumps(cp), **{f"vmr_{m}": v for m, v in vmr.items()})
if mode == "transmission":
    d = np.asarray(rt.transmission_depth(vmr_j, J(h2), J(T_art), J(mmw), vmr_he=J(he)))
    out.update(wl_um=rt.wl_um, depth_ppm=d * 1e6)
else:
    emis = exojax_rt.build_emis_model(rt, profile)
    fp = np.asarray(emis.emission_flux(vmr_j, J(h2), J(T_art), J(mmw), vmr_he=J(he)))
    fs = phoenix_surface_flux(rt.nu_grid, cp["star_teff"], cp["star_logg"], cp["star_feh"])
    r_em = float(emis.emission_radius(J(T_art), J(mmw)))
    out.update(wl_um=rt.wl_um, fp=fp, fs=np.asarray(fs), r_em_cm=r_em,
               depth_ppm=fp / np.asarray(fs) * (r_em / profile["rstar_cm"]) ** 2 * 1e6)
print(f"[{planet} {mode}] tool spectrum done, {time.time() - t0:.0f} s")

# The comparison spectrum: same column, 3.03-5.17 um, lines only in both codes.
prof2 = dict(A.profile, nu_min=1e4 / WL_CMP[1], nu_max=1e4 / WL_CMP[0], use_rayleigh=False)
rt2 = exojax_rt.build_rt_model(prof2)
assert np.allclose(rt2.p_art_bar, p_art)
if mode == "transmission":
    d2 = np.asarray(rt2.transmission_depth(vmr_j, z, J(T_art), J(mmw), vmr_he=z))
    out.update(wl_cmp=rt2.wl_um, depth_cmp_ppm=d2 * 1e6)
else:
    emis2 = exojax_rt.build_emis_model(rt2, prof2)
    f2 = np.asarray(emis2.emission_flux(vmr_j, z, J(T_art), J(mmw), vmr_he=z))
    r_em2, g_em = exojax_rt._radius_at(jnp.log(J(p_art)), J(T_art), J(mmw), profile["rp_cm"],
                                       profile["gs_cgs"], float(rt2.p_ref_bar),
                                       float(emis2.p_ref_emission_bar))
    out.update(wl_cmp=rt2.wl_um, flux_cmp=f2, g_em=float(g_em), r_em_cmp_cm=float(r_em2),
               p_ref_emission_bar=float(emis2.p_ref_emission_bar))
np.savez_compressed(OUT / f"atmos_{planet}_{mode}.npz", **out)
print(f"[{planet} {mode}] wrote atmos_{planet}_{mode}.npz, {time.time() - t0:.0f} s")
