# The PICASO engine

## `picaso_engine`

PICASO engine, host side: the alternative to VULCAN-JAX kinetics.

PICASO's radiative-convective climate solver supplies T(P) (or the run's
Guillot / tabulated profile does) and its on-the-fly equilibrium chemistry
(photochem, Gibbs minimization) supplies the abundances, optionally quenched
at one constant Kzz. The spectrum, binning, noise and Fisher are the tool's
own (ExoJAX + ExoMolOP correlated-k), so both engines share one opacity
dataset. The coupling is ONE-WAY: the climate solve uses PICASO's own
correlated-k tables, the spectrum the ExoMolOP tables; never call the pair
self-consistent.

The physics runs in ``picaso_worker.py`` inside the PICASO environment
(``JWST_TOOL_PICASO_PYTHON``; picaso 4.1 needs numpy >= 2). This module builds
the worker's requests, caches and certifies its results, and hands
``forward.run_model`` a chemistry object shaped like the VULCAN build
(``p_bar`` bottom to top, ``sidx``, ``species_masses``, ``T_base``) plus the
solved column ``y``.

## `picaso_worker`

PICASO worker -- runs INSIDE the PICASO environment (JWST_TOOL_PICASO_PYTHON).

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
