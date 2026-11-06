"""PICASO worker -- runs INSIDE the PICASO environment (JWST_TOOL_PICASO_PYTHON).

Method and contract: docs/picaso_engine.md, section `picaso_worker`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

# picaso's own temperature-perturbation ladder for a layer that does not
# converge at T (picaso/photochem.py EquilibriumChemistry.equilibrate_atmosphere)
_EPS = (0.0, 1.0e-12, -1.0e-12, 1.0e-8, -1.0e-8, 1.0e-6, -1.0e-6, 1.0e-4, -1.0e-4)
_HEARTBEAT_S = 30.0


def _bootstrap(job: dict) -> tuple:
    """Pin picaso's data roots BEFORE its import.

    (justdoit checks both paths at import, and os.path.exists(None) raises), check the stack versions.
    """
    os.environ["picaso_refdata"] = job["refdata"]
    os.environ["PYSYN_CDBS"] = job["cdbs"]
    from importlib.metadata import version

    got = {k: version(k) for k in job["versions"]}
    if got != job["versions"]:
        raise RuntimeError(f"PICASO environment mismatch: installed {got}, required {job['versions']}")
    import synphot

    vega = os.path.join(job["cdbs"], "calspec", "alpha_lyr_stis_011.fits")
    if os.path.isfile(vega):  # offline Vega: stsynphot never phones home
        synphot.conf.vega_file = vega
    import stsynphot

    stsynphot.conf.rootdir = job["cdbs"]
    import astropy.units as u
    import picaso.justdoit as jdi

    return jdi, u, got


def _solver_class() -> Any:
    from picaso.photochem import EquilibriumChemistry

    class ToolEquilibrium(EquilibriumChemistry):
        """picaso's on-the-fly solver at an explicit atom composition.

        ``method='sonora-approx'`` keeps picaso's species list and lets
        ``chemeq_on_the_fly`` reuse this instance through
        ``inputs['climate']['chemeq_solver']``.
        """

        def __init__(self, thermofile: str | Path, xh: dict, metallicity: float) -> None:
            super().__init__(thermofile, method="sonora-approx")
            names = list(self.atoms_names)
            if (set(names) - {"H"}) != set(xh):
                raise ValueError(
                    f"composition atoms {sorted(xh)} do not match the solver's {sorted(set(names) - {'H'})}"
                )
            a = np.array([1.0 if n == "H" else float(xh[n]) for n in names])
            self.target = a / a.sum()
            self.metallicity = float(metallicity)
            self.calls = 0
            self.fail_layers = 0

        def solve_column(self, p_bar: np.ndarray, T: np.ndarray) -> tuple:
            n = len(p_bar)
            vmr = np.empty((n, len(self.gas_names)))
            atoms = np.empty((n, len(self.atoms_names)))
            conv = np.zeros(n, dtype=bool)
            self.use_prev_guess = False
            for i in range(n):
                for eps in _EPS:
                    conv[i] = self.solve(p_bar[i] * 1e6, T[i] + T[i] * eps, molfracs_atoms=self.target)
                    if conv[i]:
                        break
                vmr[i] = self.molfracs_species_gas
                atoms[i] = self.molfracs_atoms
            self.calls += 1
            self.fail_layers += int(np.sum(~conv))
            return vmr, atoms, conv

        def equilibrate_atmosphere(self, P: Any, T: np.ndarray, log10mh: float, CtoO_relative: Any) -> tuple:
            # picaso calls this with atmosphere(mh=Z, cto_absolute=its solar
            # C/O): anything else means it did not take the tool's path
            if CtoO_relative != 1.0 or abs(10.0**log10mh / self.metallicity - 1.0) > 1e-12:
                raise RuntimeError(
                    f"picaso called the composition-locked solver with log10mh={log10mh}, CtoO_relative={CtoO_relative}"
                )
            vmr, _, _ = self.solve_column(np.asarray(P, float), np.asarray(T, float))
            gases = {k: vmr[:, j].copy() for j, k in enumerate(self.gas_names)}
            conds = {k: np.zeros(len(P)) for k in self.condensate_names}
            return gases, conds

    return ToolEquilibrium


def _heartbeat(label: str) -> Any:
    """One stdout line every _HEARTBEAT_S so a minutes-long solve never reads as a hang.

    The host forwards these to the run log.
    """
    stop = threading.Event()
    t0 = time.time()

    def beat() -> None:
        while not stop.wait(_HEARTBEAT_S):
            print(f"[pic] {label}: {time.time() - t0:.0f} s", flush=True)

    threading.Thread(target=beat, daemon=True).start()
    return stop


def _quenched_column(jdi: Any, u: Any, solver: Any, req: dict, p: Any, T: np.ndarray, opa: Any) -> tuple:
    """Picaso's own quench recipe.

    (Zahnle & Marley 2014 timescales at one constant Kzz, kinetic CO2) on the evaluation column.
    """
    from picaso.climate import calculate_atm, update_quench_levels

    b = jdi.inputs(calculation="planet", climate=True)
    b.gravity(gravity=req["g_cgs"], gravity_unit=u.Unit("cm/(s**2)"))
    b.add_pt(T=T, P=p)
    b.inputs["climate"]["chemeq_solver"] = solver
    b.atmosphere(mh=req["metallicity"], cto_absolute=jdi.LODDERS2020_C_TO_O, chem_method="on-the-fly", quench=True)
    b.inputs["star"].setdefault("radius", np.nan)  # read by calculate_atm
    atm = calculate_atm(b, opa, only_atmosphere=True)
    levels = update_quench_levels(
        b, atm, np.full(len(p), float(req["quench_kzz"])), 0.01 * req["g_cgs"]
    )  # picaso passes SI here
    if any(int(v) >= len(p) - 1 for v in levels.values()):
        raise RuntimeError(
            f"quench level below the column bottom ({p[-1]:.3g} bar): {levels}. The quench point needs a deeper column."
        )
    b.premix_atmosphere(opa=None, quench_levels=levels, verbose=False)
    if b.inputs["climate"]["chemeq_solver"] is not solver:
        raise RuntimeError("picaso replaced the composition-locked solver")
    prof = b.inputs["atmosphere"]["profile"]
    if not np.allclose(prof["pressure"].values, p, rtol=1e-12):
        raise RuntimeError("picaso changed the quench column's pressure grid")
    vmr = np.stack([prof[k].values for k in solver.gas_names], axis=1)
    quench = {k: float(p[int(v)]) for k, v in levels.items()}
    return vmr, quench


def _climate(jdi: Any, u: Any, solver: Any, req: dict, opa: Any) -> dict:
    c, s = req["climate"], req["climate"]["star"]
    cl = jdi.inputs(calculation="planet", climate=True)
    cl.gravity(gravity=req["g_cgs"], gravity_unit=u.Unit("cm/(s**2)"))
    cl.effective_temp(float(c["tint"]))  # picaso's planet T_eff is T_int
    cl.star(
        opa,
        temp=s["teff"],
        metal=s["feh"],
        logg=s["logg"],
        radius=s["rstar_rsun"],
        radius_unit=u.R_sun,
        semi_major=s["orbit_au"],
        semi_major_unit=u.AU,
        database="phoenix",
    )
    pt = cl.guillot_pt(
        float(c["teq_guess"]),
        T_int=float(c["tint"]),
        nlevel=int(c["nlevel"]),
        p_bottom=np.log10(c["p_bottom_bar"]),
        p_top=np.log10(c["p_top_bar"]),
    )
    cl.inputs_climate(
        temp_guess=pt["temperature"].values,
        pressure=pt["pressure"].values,
        rcb_guess=int(c["rcb"]),
        rfacv=float(c["rfacv"]),
    )
    cl.inputs["climate"]["chemeq_solver"] = solver
    # The climate is always solved in equilibrium; a quench is applied after,
    # on the certified T (_quenched_column). picaso's quench-in-the-loop climate
    # misses the TOA flux balance by ~1e-2 of F_int on WASP-39 b while still
    # reporting converged.
    cl.atmosphere(mh=req["metallicity"], cto_absolute=jdi.LODDERS2020_C_TO_O, chem_method="on-the-fly")
    stop = _heartbeat("climate solve")
    t0 = time.time()
    try:
        out = cl.climate(opa, with_spec=False, verbose=False, diseq_chem=False, self_consistent_kzz=True)
    except ValueError as e:  # picaso's convective-zone failures
        raise RuntimeError(f"PICASO climate solve failed: {e}") from e
    finally:
        stop.set()
    if cl.inputs["climate"]["chemeq_solver"] is not solver:
        raise RuntimeError("picaso replaced the composition-locked solver")
    fb = out["flux_balance"]
    return {
        "clim_p": np.asarray(out["pressure"], float),
        "clim_T": np.asarray(out["temperature"], float),
        "cvz": np.asarray(out["cvz_locs"], dtype=np.int64),
        "conv_flag": np.array(int(out["converged"])),
        "flux_net": np.atleast_1d(np.asarray(fb["flux_net"], float)),
        "tidal": np.atleast_1d(np.asarray(fb["tidal"], float)),
        "chem_calls": np.array(solver.calls),
        "wall_s": np.array(time.time() - t0),
    }


def _run(jdi: Any, u: Any, job: dict, req: dict) -> dict:
    thermo = os.path.join(job["refdata"], "chemistry", "thermo_data", "thermo-sonora-component.yaml")
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
            raise RuntimeError(f"picaso loaded correlated-k tables {sorted(opa.molecules)}, expected {want}")
    if req["climate"] is not None:
        rec = _climate(jdi, u, solver, req, opa)
        if p[-1] > rec["clim_p"][-1] * (1 + 1e-9):
            raise RuntimeError(
                f"evaluation grid bottom {p[-1]:.3g} bar lies below the climate grid bottom {rec['clim_p'][-1]:.3g} bar"
            )
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
    return dict(
        rec,
        T=T,
        vmr=vmr,
        atoms=atoms,
        converged=conv,
        chem_fail=np.array(solver.fail_layers),
        gas_names=np.array(solver.gas_names),
        gas_mass=np.asarray(solver.gas_mass, float),
        atoms_names=np.array(solver.atoms_names),
        target=np.asarray(solver.target, float),
        quench_json=np.array(json.dumps(quench)),
    )


def main(job_path: str | Path, out_path: str | Path) -> None:
    """Run a job file and write its result file."""
    with open(job_path) as fh:
        job = json.load(fh)
    jdi, u, got = _bootstrap(job)
    out = _run(jdi, u, job, job["request"])
    tmp = out_path + ".part.npz"
    np.savez(tmp, versions_json=np.array(json.dumps(got)), **out)
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
