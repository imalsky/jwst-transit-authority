"""Writes validation/data/prt_<planet>_<mode>.npz: petitRADTRANS 3.4.0 on the
columns saved by atmos_cases.py (same P, T, mmw, VMRs, geometry; same ExoMolOP
k-table files through the symlinked input_data tree), lines only over
3.03-5.17 um. petitRADTRANS lives in its own venv, which has neither
vulcan_forward nor jwst_tool, so the k-table tree is read straight from the
engine's data-root contract ($VULCAN_FORWARD_DATA, what
vulcan_forward.paths.exomolop_dir() resolves):

    ~/venvs/prt/bin/python validation/scripts/inputs/prt_reference.py validation/data/atmos_*.npz
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.chemistry.utils import get_species_molar_mass

PROV = json.load(open(Path(os.environ["VULCAN_FORWARD_DATA"]) / "exomolop" / "provenance.json"))
NAME = {m: f"{m}__{v['dataset']}" for m, v in PROV.items()}

for f in sys.argv[1:]:
    z = np.load(f)
    mode = "emission" if "flux_cmp" in z else "transmission"
    mols = [str(m) for m in z["mols"]]
    p, T, mmw = z["p_art"], z["T_art"], z["mmw"]
    rt = Radtrans(pressures=p, line_species=[NAME[m] for m in mols],
                  wavelength_boundaries=[3.03, 5.17], line_opacity_mode="c-k",
                  rayleigh_species=[], gas_continuum_contributors=[], cloud_species=[])
    mf = {NAME[m]: z[f"vmr_{m}"] * get_species_molar_mass(NAME[m]) / mmw for m in mols}
    if mode == "transmission":
        wl, rad, _ = rt.calculate_transit_radii(
            temperatures=T, mass_fractions=mf, mean_molar_masses=mmw,
            reference_gravity=float(z["gs_cgs"]), reference_pressure=float(z["p_ref_bar"]),
            planet_radius=float(z["rp_cm"]), variable_gravity=True)
        wl_um = wl * 1e4
        o = np.argsort(wl_um)
        res = dict(wl_um=wl_um[o], depth_ppm=(rad[o] / float(z["rstar_cm"])) ** 2 * 1e6)
    else:
        wl, fl, _ = rt.calculate_flux(
            temperatures=T, mass_fractions=mf, mean_molar_masses=mmw,
            reference_gravity=float(z["g_em"]), frequencies_to_wavelengths=True)
        wl_um = wl * 1e4
        o = np.argsort(wl_um)
        nu = 1.0 / wl[o]                        # cm^-1; pRT flux is per cm of wavelength
        res = dict(wl_um=wl_um[o], flux_per_cm1=fl[o] / nu ** 2)
    out = Path(f).with_name(Path(f).name.replace("atmos_", "prt_"))
    np.savez_compressed(out, **res)
    print("wrote", out)
