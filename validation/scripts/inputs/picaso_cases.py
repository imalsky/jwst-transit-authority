"""Inputs of the PICASO-engine validation figures. One case per call; each
writes validation/data/picaso_<case>.npz.

    # main environment (VULCAN + the tool):
    python validation/scripts/inputs/picaso_cases.py chem
    python validation/scripts/inputs/picaso_cases.py spectra_vul_K9   # VULCAN photo-off, Kzz 1e9
    python validation/scripts/inputs/picaso_cases.py spectra_pic_q9   # PICASO quench, Kzz 1e9
    python validation/scripts/inputs/picaso_cases.py spectra_pic_eq   # PICASO equilibrium
    # PICASO environment ($JWST_TOOL_PICASO_PYTHON), ~10 min per climate solve:
    python validation/scripts/inputs/picaso_cases.py climate_notebook   # picaso's own composition
    python validation/scripts/inputs/picaso_cases.py climate_tool       # the tool's composition
    python validation/scripts/inputs/picaso_cases.py climate_{fgk,mdwarf}_{stock,edges}

chem: on the shipped WASP-39 b table at 10x solar, C/O 0.55, VULCAN's
ExoGibbs equilibrium seed, PICASO with C/N/O/S/He only, and PICASO with the
production composition (plus the Lodders 2020 metals).
climate_notebook/tool: picaso 4.1's WASP-39 b climate notebook (2b) inputs
with on-the-fly chemistry (database='phoenix': ck04models is not installed),
the reference profile reference/base_cases/HJ.pt alongside.
climate_*_{stock,edges}: picaso issue #413, WASP-39 b's composition and
gravity around WASP-39 (G8) and around a 3250 K M dwarf at the same T_eq;
'edges' replaces picaso 4.1's stellar binning with the bin-edge integral from
picaso's terradev branch (justdoit.py:1496).
spectra_*: WASP-39 b transmission on the default Guillot profile.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "validation" / "data"
case = sys.argv[1]
W39 = dict(teff=5485.0, logg=4.5, feh=0.0, rstar_rsun=0.932, orbit_au=0.04828)
MDWARF = dict(teff=3250.0, logg=5.0, feh=0.0, rstar_rsun=0.3, orbit_au=0.005456)
G_W39 = 484.97        # the climate baseline's gravity anchor (cm/s^2, 0.977 bar)
SP = ["H2O", "CH4", "CO", "CO2", "NH3", "N2", "HCN", "H2S"]


def _tool():
    sys.path.insert(0, str(REPO / "src"))
    from vulcan_forward import vulcan_chem  # noqa: F401  (import order)
    from jwst_tool import forward, picaso_engine
    return forward, picaso_engine


def _worker():
    sys.path.insert(0, str(REPO / "src" / "jwst_tool"))
    import picaso_worker as pw
    return pw


def chem():
    forward, pe = _tool()
    cp = forward.canonical_params(dict(planet="wasp39b", tp_mode="file", use_photo=False))
    seed = forward._assemble_chem(cp, lambda s: None).build_chem()
    p = np.asarray(seed.p_bar)[::-1]                    # ascending
    T = np.asarray(seed.T_base, float)[::-1]
    y = np.asarray(seed.y0, float)[::-1]
    y = y / y.sum(axis=1, keepdims=True)
    out = {"p_bar": p, "T": T}
    out.update({f"vulcan_{s}": y[:, seed.sidx[s]] for s in SP})
    full = pe.composition(cp["met_x_solar"], cp["co_ratio"])
    for tag, xh in (("cnos", {k: (v if k in ("He", "C", "N", "O", "S") else 1e-30)
                             for k, v in full.items()}), ("full", full)):
        res = pe.solve({"xh": xh, "metallicity": cp["met_x_solar"], "g_cgs": G_W39,
                        "p_bar": p.tolist(), "T": T.tolist(), "quench_kzz": None,
                        "climate": None}, f"chem {tag}", print)
        names = [str(s) for s in res["gas_names"]]
        out.update({f"picaso_{tag}_{s}": np.asarray(res["vmr"])[:, names.index(s)] for s in SP})
    return out


def spectra(sub):
    forward, _ = _tool()
    base = dict(planet="wasp39b", tp_mode="guillot")
    p = {"vul_K9": dict(base, use_photo=False, kzz_mode="const", kzz_const=1e9),
         "pic_q9": dict(base, chem_provider="picaso", picaso_quench=True, kzz_const=1e9),
         "pic_eq": dict(base, chem_provider="picaso")}[sub]
    d = np.load(forward.run_model(p, log=print))
    names = [str(s) for s in d["ymix_species"]]
    out = {k: d[k] for k in ("wl_um", "depth", "p_bar", "T")}
    out.update({f"y_{s}": d["ymix"][:, names.index(s)] for s in SP if s in names})
    return out


def _bin_stellar_flux(wno_star, flux_star, wno_planet, delta_wno):
    """picaso terradev justdoit.py:1496 (the issue #413 fix), verbatim."""
    half_dw = delta_wno / 2.0
    bin_flux = np.zeros(len(wno_planet))
    for i in range(len(wno_planet)):
        wno_lo = wno_planet[i] - half_dw[i]
        wno_hi = wno_planet[i] + half_dw[i]
        in_bin = (wno_star >= wno_lo) & (wno_star <= wno_hi)
        bin_flux[i] = np.trapezoid(flux_star[in_bin], x=-1 / wno_star[in_bin])
    return bin_flux


def climate(sub):
    pw = _worker()
    refdata = os.environ.get("JWST_TOOL_PICASO_REFDATA",
                             str(REPO / "data" / "picaso_refdata-4.1"))
    job = {"refdata": refdata, "cdbs": str(REPO / "data" / "cdbs"),
           "versions": {"picaso": "4.1", "photochem": "0.9.2"}}
    jdi, u, _ = pw._bootstrap(job)
    xh = json.loads(os.environ["PICASO_CASE_XH"])     # the tool's composition, see below
    if sub in ("notebook", "tool"):
        opa = jdi.opannection(method="resortrebin")
        cl = jdi.inputs(calculation="planet", climate=True)
        cl.gravity(gravity=4.5, gravity_unit=u.Unit("m/(s**2)"))
        cl.effective_temp(200)
        cl.star(opa, temp=5326.6, metal=-0.03, logg=4.38933, radius=0.932,
                radius_unit=u.R_sun, semi_major=0.0486, semi_major_unit=u.AU,
                database="phoenix")
        pt = cl.guillot_pt(1000, nlevel=91, T_int=200, p_bottom=2, p_top=-6)
        cl.inputs_climate(temp_guess=pt["temperature"].values,
                          pressure=pt["pressure"].values, rcb_guess=85, rfacv=0.5)
        if sub == "notebook":
            cl.atmosphere(mh=10, cto_relative=1, chem_method="on-the-fly")
        else:
            thermo = os.path.join(refdata, "chemistry", "thermo_data",
                                  "thermo-sonora-component.yaml")
            cl.inputs["climate"]["chemeq_solver"] = pw._solver_class()(thermo, xh, 10.0)
            cl.atmosphere(mh=10.0, cto_absolute=jdi.LODDERS2020_C_TO_O,
                          chem_method="on-the-fly")
        out = cl.climate(opa, with_spec=False, verbose=False)
        ref = jdi.pd.read_csv(jdi.HJ_pt(), sep=r"\s+")
        return {"P": np.asarray(out["pressure"]), "T": np.asarray(out["temperature"]),
                "P_ref": ref["pressure"].values, "T_ref": ref["temperature"].values}
    star_name, variant = sub.split("_")
    star = {"fgk": W39, "mdwarf": MDWARF}[star_name]
    if variant == "edges":
        import stsynphot as sts
        orig = jdi.inputs.star

        def star_edges(self, o, temp=None, metal=None, logg=None, radius=None,
                       radius_unit=None, semi_major=None, semi_major_unit=None,
                       database="ck04models", **kw):
            orig(self, o, temp=temp, metal=metal, logg=logg, radius=radius,
                 radius_unit=radius_unit, semi_major=semi_major,
                 semi_major_unit=semi_major_unit, database=database, **kw)
            sp = sts.grid_to_spec(database, temp, metal, logg)
            wno = 1e4 / sp.waveset.to(u.um).value[::-1]
            flux = sp(sp.waveset, flux_unit=u.Unit("erg*cm^(-3)*s^(-1)")).value[::-1]
            r = (radius * radius_unit).to(u.cm).value
            a = (semi_major * semi_major_unit).to(u.cm).value
            rel = _bin_stellar_flux(wno, flux, o.wno, o.delta_wno) * (r / a) ** 2
            o.relative_flux = rel
            self.inputs["star"]["relative_flux"] = rel
        jdi.inputs.star = star_edges
    teq = star["teff"] * np.sqrt(star["rstar_rsun"] * 6.957e10 / (2 * star["orbit_au"] * 1.495978707e13))
    p_eval = np.logspace(-7, 1, 100)
    req = {"xh": xh, "metallicity": 10.0, "g_cgs": G_W39, "p_bar": p_eval.tolist(),
           "T": None, "quench_kzz": None,
           "climate": {"tint": 100.0, "rfacv": 0.5, "teq_guess": teq, "nlevel": 91,
                       "p_top_bar": 1e-6, "p_bottom_bar": 100.0, "rcb": 85, "star": star}}
    job["ck_molecules"] = json.loads(os.environ["PICASO_CASE_CK"])
    res = pw._run(jdi, u, job, req)
    return {"P": res["clim_p"], "T": res["clim_T"]}


if case.startswith("climate_") and "PICASO_CASE_XH" not in os.environ:
    # the composition and the CK list come from the tool (main environment);
    # re-run this file under the PICASO python with them in the environment
    import subprocess
    forward, pe = _tool()
    env = dict(os.environ, PICASO_CASE_XH=json.dumps(pe.composition(10.0, 0.55)),
               PICASO_CASE_CK=json.dumps(pe.CK_MOLECULES), OMP_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1")
    sys.exit(subprocess.call([os.environ["JWST_TOOL_PICASO_PYTHON"], __file__, case], env=env))
fn = chem if case == "chem" else (
    (lambda: spectra(case[len("spectra_"):])) if case.startswith("spectra_")
    else (lambda: climate(case[len("climate_"):])))
DATA.mkdir(exist_ok=True)
np.savez(DATA / f"picaso_{case}.npz", **fn())
print("wrote", DATA / f"picaso_{case}.npz")
