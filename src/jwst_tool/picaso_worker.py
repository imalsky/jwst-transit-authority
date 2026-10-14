"""PICASO worker -- runs INSIDE the PICASO environment (JWST_TOOL_PICASO_PYTHON).

Standalone on purpose: stdlib + numpy at module level, picaso/photochem/astropy
and synphot/stsynphot inside functions, nothing from the rest of the tool and
no jax. picaso 4.1 needs numpy >= 2; the tool's stack pins numpy < 2.

    python picaso_worker.py job.json out.npz

job.json:
    {"refdata": <picaso reference root>, "cdbs": <PYSYN_CDBS root>,
     "versions": {"picaso": "4.1", "photochem": "0.9.2"},
     "ck_molecules": [...],            # the climate correlated-k set, all loaded
     "request": {
        "xh": {atom: X/H for every atom except H},   # the tool's composition
        "metallicity": Z,             # linear; quench timescales + tripwire
        "g_cgs": g,                   # one constant gravity (picaso issue #384)
        "p_bar": [...],               # ascending evaluation grid
        "T": [...] | null,            # null: the climate solve supplies T
        "quench_kzz": Kzz | null,     # cm^2/s; null = equilibrium
        "climate": null | {"tint", "rfacv", "teq_guess", "nlevel",
                           "p_top_bar", "p_bottom_bar", "rcb",
                           "star": {"teff", "logg", "feh", "rstar_rsun",
                                    "orbit_au"}}}}

out.npz: the evaluation column ("T", "gas_names", "gas_mass", "vmr" (n_p,
n_gas), "converged", "atoms_names", "atoms" (n_p, n_atoms; all phases),
"target", "quench_json", "chem_fail" (every failed layer solve of the
request)) plus, for a climate request, the solve record ("clim_p", "clim_T",
"cvz", "conv_flag", "flux_net", "tidal", "chem_calls", "wall_s");
"versions_json" names the stack that ran. A failure exits with a traceback
and writes nothing. The host certifies; this file only computes.

The composition is imposed exactly: a subclass of picaso's
EquilibriumChemistry solves every layer from the requested atom mole
fractions (photochem's ``solve(molfracs_atoms=...)``). picaso's own path
(``solve_metallicity``) scales its proto-solar table and moves C and O at
fixed C+O, which is not the tool's convention. Every layer is solved cold:
picaso's layer-to-layer warm start returns stale all-phase atom totals (up to
10% off in C/N/O/S at 100x solar on WASP-39 b) while the gas phase is
unchanged, which would blind the element check.
"""
import json
import os
import sys
import threading
import time

import numpy as np

# picaso's own temperature-perturbation ladder for a layer that does not
# converge at T (picaso/photochem.py EquilibriumChemistry.equilibrate_atmosphere)
_EPS = (0.0, 1.0e-12, -1.0e-12, 1.0e-8, -1.0e-8, 1.0e-6, -1.0e-6, 1.0e-4, -1.0e-4)
_HEARTBEAT_S = 30.0


def _bootstrap(job):
    """Pin picaso's data roots BEFORE its import (justdoit checks both paths
    at import, and os.path.exists(None) raises), check the stack versions."""
    os.environ["picaso_refdata"] = job["refdata"]
    os.environ["PYSYN_CDBS"] = job["cdbs"]
    from importlib.metadata import version
    got = {k: version(k) for k in job["versions"]}
    if got != job["versions"]:
        raise RuntimeError(f"PICASO environment mismatch: installed {got}, "
                           f"required {job['versions']}")
    import synphot
    vega = os.path.join(job["cdbs"], "calspec", "alpha_lyr_stis_011.fits")
    if os.path.isfile(vega):   # offline Vega: stsynphot never phones home
        synphot.conf.vega_file = vega
    import stsynphot
    stsynphot.conf.rootdir = job["cdbs"]
    import astropy.units as u
    import picaso.justdoit as jdi
    return jdi, u, got


def _solver_class():
    from picaso.photochem import EquilibriumChemistry

    class ToolEquilibrium(EquilibriumChemistry):
        """picaso's on-the-fly solver at an explicit atom composition.

        ``method='sonora-approx'`` keeps picaso's species list and lets
        ``chemeq_on_the_fly`` reuse this instance through
        ``inputs['climate']['chemeq_solver']``."""

        def __init__(self, thermofile, xh, metallicity):
            super().__init__(thermofile, method="sonora-approx")
            names = list(self.atoms_names)
            if (set(names) - {"H"}) != set(xh):
                raise ValueError(
                    f"composition atoms {sorted(xh)} do not match the solver's "
                    f"{sorted(set(names) - {'H'})}")
            a = np.array([1.0 if n == "H" else float(xh[n]) for n in names])
            self.target = a / a.sum()
            self.metallicity = float(metallicity)
            self.calls = 0
            self.fail_layers = 0

        def solve_column(self, p_bar, T):
            n = len(p_bar)
            vmr = np.empty((n, len(self.gas_names)))
            atoms = np.empty((n, len(self.atoms_names)))
            conv = np.zeros(n, dtype=bool)
            self.use_prev_guess = False
            for i in range(n):
                for eps in _EPS:
                    conv[i] = self.solve(p_bar[i] * 1e6, T[i] + T[i] * eps,
                                         molfracs_atoms=self.target)
                    if conv[i]:
                        break
                vmr[i] = self.molfracs_species_gas
                atoms[i] = self.molfracs_atoms
            self.calls += 1
            self.fail_layers += int(np.sum(~conv))
            return vmr, atoms, conv

        def equilibrate_atmosphere(self, P, T, log10mh, CtoO_relative):
            # picaso calls this with atmosphere(mh=Z, cto_absolute=its solar
            # C/O): anything else means it did not take the tool's path
            if CtoO_relative != 1.0 or abs(10.0 ** log10mh / self.metallicity - 1.0) > 1e-12:
                raise RuntimeError(
                    f"picaso called the composition-locked solver with "
                    f"log10mh={log10mh}, CtoO_relative={CtoO_relative}")
            vmr, _, _ = self.solve_column(np.asarray(P, float), np.asarray(T, float))
            gases = {k: vmr[:, j].copy() for j, k in enumerate(self.gas_names)}
            conds = {k: np.zeros(len(P)) for k in self.condensate_names}
            return gases, conds

    return ToolEquilibrium


def _heartbeat(label):
    """One stdout line every _HEARTBEAT_S so a minutes-long solve never reads
    as a hang; the host forwards these to the run log."""
    stop = threading.Event()
    t0 = time.time()

    def beat():
        while not stop.wait(_HEARTBEAT_S):
            print(f"[pic] {label}: {time.time() - t0:.0f} s", flush=True)
    threading.Thread(target=beat, daemon=True).start()
    return stop


def _quenched_column(jdi, u, solver, req, p, T, opa):
    """picaso's own quench recipe (Zahnle & Marley 2014 timescales at one
    constant Kzz, kinetic CO2) on the evaluation column."""
    from picaso.climate import calculate_atm, update_quench_levels
    b = jdi.inputs(calculation="planet", climate=True)
    b.gravity(gravity=req["g_cgs"], gravity_unit=u.Unit("cm/(s**2)"))
    b.add_pt(T=T, P=p)
    b.inputs["climate"]["chemeq_solver"] = solver
    b.atmosphere(mh=req["metallicity"], cto_absolute=jdi.LODDERS2020_C_TO_O,
                 chem_method="on-the-fly", quench=True)
    b.inputs["star"].setdefault("radius", np.nan)   # read by calculate_atm
    atm = calculate_atm(b, opa, only_atmosphere=True)
    levels = update_quench_levels(b, atm, np.full(len(p), float(req["quench_kzz"])),
                                  0.01 * req["g_cgs"])   # picaso passes SI here
    if any(int(v) >= len(p) - 1 for v in levels.values()):
        raise RuntimeError(
            f"quench level below the column bottom ({p[-1]:.3g} bar): {levels}. "
            "The quench point needs a deeper column.")
    b.premix_atmosphere(opa=None, quench_levels=levels, verbose=False)
    if b.inputs["climate"]["chemeq_solver"] is not solver:
        raise RuntimeError("picaso replaced the composition-locked solver")
    prof = b.inputs["atmosphere"]["profile"]
    if not np.allclose(prof["pressure"].values, p, rtol=1e-12):
        raise RuntimeError("picaso changed the quench column's pressure grid")
    vmr = np.stack([prof[k].values for k in solver.gas_names], axis=1)
    quench = {k: float(p[int(v)]) for k, v in levels.items()}
    return vmr, quench


def _climate(jdi, u, solver, req, opa):
    c, s = req["climate"], req["climate"]["star"]
    cl = jdi.inputs(calculation="planet", climate=True)
    cl.gravity(gravity=req["g_cgs"], gravity_unit=u.Unit("cm/(s**2)"))
    cl.effective_temp(float(c["tint"]))      # picaso's planet T_eff is T_int
    cl.star(opa, temp=s["teff"], metal=s["feh"], logg=s["logg"],
            radius=s["rstar_rsun"], radius_unit=u.R_sun,
            semi_major=s["orbit_au"], semi_major_unit=u.AU, database="phoenix")
    pt = cl.guillot_pt(float(c["teq_guess"]), T_int=float(c["tint"]),
                       nlevel=int(c["nlevel"]), p_bottom=np.log10(c["p_bottom_bar"]),
                       p_top=np.log10(c["p_top_bar"]))
    cl.inputs_climate(temp_guess=pt["temperature"].values,
                      pressure=pt["pressure"].values, rcb_guess=int(c["rcb"]),
                      rfacv=float(c["rfacv"]))
    cl.inputs["climate"]["chemeq_solver"] = solver
    # The climate is always solved in equilibrium; a quench is applied after,
    # on the certified T (_quenched_column). picaso's quench-in-the-loop climate
    # misses the TOA flux balance by ~1e-2 of F_int on WASP-39 b while still
    # reporting converged.
    cl.atmosphere(mh=req["metallicity"], cto_absolute=jdi.LODDERS2020_C_TO_O,
                  chem_method="on-the-fly")
    stop = _heartbeat("climate solve")
    t0 = time.time()
    try:
        out = cl.climate(opa, with_spec=False, verbose=False, diseq_chem=False,
                         self_consistent_kzz=True)
    except ValueError as e:   # picaso's convective-zone failures
        raise RuntimeError(f"PICASO climate solve failed: {e}") from e
    finally:
        stop.set()
    if cl.inputs["climate"]["chemeq_solver"] is not solver:
        raise RuntimeError("picaso replaced the composition-locked solver")
    fb = out["flux_balance"]
    return {"clim_p": np.asarray(out["pressure"], float),
            "clim_T": np.asarray(out["temperature"], float),
            "cvz": np.asarray(out["cvz_locs"], dtype=np.int64),
            "conv_flag": np.array(int(out["converged"])),
            "flux_net": np.atleast_1d(np.asarray(fb["flux_net"], float)),
            "tidal": np.atleast_1d(np.asarray(fb["tidal"], float)),
            "chem_calls": np.array(solver.calls),
            "wall_s": np.array(time.time() - t0)}


def _run(jdi, u, job, req):
    thermo = os.path.join(job["refdata"], "chemistry", "thermo_data",
                          "thermo-sonora-component.yaml")
    solver = _solver_class()(thermo, req["xh"], req["metallicity"])
    p = np.asarray(req["p_bar"], float)
    if not np.all(np.diff(p) > 0):
        raise ValueError("evaluation grid must be strictly ascending in pressure")
    rec, opa = {}, None
    if req["climate"] is not None or req["quench_kzz"] is not None:
        want = sorted(job["ck_molecules"])
        opa = jdi.opannection(method="resortrebin", preload_gases=want)
        # picaso drops a missing table with only a warning, which would change
        # the climate opacity silently
        if sorted(opa.molecules) != want:
            raise RuntimeError(
                f"picaso loaded correlated-k tables {sorted(opa.molecules)}, "
                f"expected {want}")
    if req["climate"] is not None:
        rec = _climate(jdi, u, solver, req, opa)
        if p[-1] > rec["clim_p"][-1] * (1 + 1e-9):
            raise RuntimeError(f"evaluation grid bottom {p[-1]:.3g} bar lies below "
                               f"the climate grid bottom {rec['clim_p'][-1]:.3g} bar")
        # above the climate top the topmost T is held (file-mode convention)
        T = np.interp(np.log(p), np.log(rec["clim_p"]), rec["clim_T"])
    else:
        T = np.asarray(req["T"], float)
    vmr, atoms, conv = solver.solve_column(p, T)
    quench = {}
    if req["quench_kzz"] is not None:
        vmr, quench = _quenched_column(jdi, u, solver, req, p, T, opa)
    # every layer solve of this request: climate iterations, the column, and
    # the quench recipe's own equilibrium solves
    return dict(rec, T=T, vmr=vmr, atoms=atoms, converged=conv,
                chem_fail=np.array(solver.fail_layers),
                gas_names=np.array(solver.gas_names),
                gas_mass=np.asarray(solver.gas_mass, float),
                atoms_names=np.array(solver.atoms_names),
                target=np.asarray(solver.target, float),
                quench_json=np.array(json.dumps(quench)))


def main(job_path, out_path):
    with open(job_path) as fh:
        job = json.load(fh)
    jdi, u, got = _bootstrap(job)
    out = _run(jdi, u, job, job["request"])
    tmp = out_path + ".part.npz"
    np.savez(tmp, versions_json=np.array(json.dumps(got)), **out)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
