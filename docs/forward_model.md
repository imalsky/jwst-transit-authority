# The forward model

## `forward`

Forward model runner: chemistry -> ExoJAX spectrum.

The chemistry is VULCAN-JAX photochemical kinetics (the default) or, with
``chem_provider='picaso'``, PICASO climate + equilibrium chemistry
(``picaso_engine.py``). The npz cache in ``instruments.MODEL_CACHE`` holds
spectra under ``params_key`` and converged VULCAN chemistry columns under
``chem-<chem_key>``; the GUI's light path (``params_key`` / ``cache_path`` /
``load_result``) reads only the ``params_key`` entries.
``python -m jwst_tool.forward params.json`` runs the heavy pipeline and writes
the ``params_key`` entry, logging "[fwd] PROG <frac> <label>" for the GUI
bar. With ``fisher_params`` it also builds the spectrum Jacobian row by row,
recording each row's method in ``jac_row_method``. ``jac_method="fd"`` (the
default) gives certified finite differences under the h-vs-2h gate, central
or one-sided at a range edge ("fd-central", "fd-onesided"); ``"ad"`` gives
one warm-started jvp per chemistry row ("ad-jvp", VULCAN only); T-P rows take
the FD stencil under either method. The RT-only rows (cloud deck, lnR0) are
one central difference with no h-vs-2h check ("fd-rt", fd_err NaN) or an RT
jvp ("ad-jvp").
``fisher.py`` turns the Jacobian + Pandeia noise into forecasts.

A "removed" spectrum zeroes that molecule's VMR in the RT only, keeping the
structure (T, mmw): the standard nested-model comparison.

## `adjoint_diag`

Adjoint diagnostics: reverse-mode AD sensitivities the FD Fisher cannot do.

One adjoint solve replaces thousands of re-runs for the high-dimensional
questions: dL/d(ln k_r) over every reaction
(``steady_state_reaction_sensitivity``) and dL/dT per layer
(``steady_state_input_sensitivity``). L is the log10 VMR of the target
molecule at its peak-VMR layer inside the transit photosphere
(``PHOTOSPHERE_P_BAR``).

API-only (``python -m jwst_tool.adjoint_diag``); the GUI does not
read it. Contract, all recorded in the npz:

* same build path as the forecasts (``chem_build.assemble_chem``); the
  geometry/operator splice is ``make_body_terms``, never a manual one.
* the scope audit runs FIRST and an audit error refuses the run. It scans
  the upstream ``BODY_MAP_DT_CANDIDATES`` (near-zero trace cells oscillate
  under small probe steps); the gradient uses the first passing ``body_dt``
  and the scan trail is cached.
* magnitudes are trusted only when resid_median <= 0.2 AND ensemble_spread
  <= 0.15 (upstream thresholds); otherwise the result is a RANKING.
  ``pair_antisym`` is a diagnostic only, never a trust gate.
* the rate-uncertainty spread assumes a UNIFORM Agundez (2025) class-B
  0.65 dex per reaction -- a stated assumption, not per-reaction.
* dL/dT is the chemistry-path gradient (photolysis cross sections and the
  diffusion/geometry rebuild frozen, upstream contract); the
  rebuild-consistency metric is stored.

Two faces like forward.py: the light cache API (no JAX imports) and the
heavy script mode (``python -m jwst_tool.adjoint_diag params.json SO2``).
Condensation is refused up front: the pinned reservoir is a
step-sequence-dependent transient (tangents through it are ~91% wrong, see
run_params.canonical_params). The PICASO engine is refused too: it has no
reaction network to differentiate.

## `stellar`

Stellar surface flux for eclipse depths (emission mode).

``phoenix_surface_flux(nu_grid, teff, logg, feh)`` returns the stellar
SURFACE flux density per wavenumber (erg s^-1 cm^-2 / cm^-1) on the native RT
grid, from the same minimal-CDBS PHOENIX grid the Pandeia noise side uses.

Units contract: the CDBS PHOENIX models are emergent SURFACE flux densities
and ExoJAX ``ArtEmisPure`` returns the planet's in the same convention, so

    eclipse depth  d_ec(nu) = (F_p(nu) / F_s(nu)) * (R_p/R_s)^2

with NO extra pi. Guarded by an energy-closure check: band-integrated F_s
over band-integrated pi B_nu(T_eff) must lie in [0.5, 1.5] -- a grid that
were secretly intensity (~3.1x) or Eddington flux (~12x) fails immediately.

Heavy-path module: imports stsynphot on first call; never imported by the
GUI's light path.

## `planets`

Planet/system registry for the JWST instrument selector.

Pure data, importable by the light GUI path (no jax/vulcan/exojax imports).

Every planet runs on the SAME W39b-validated machinery (the WASP-39b SNCHO
photo network + 10x-solar equilibrium baseline), with the planet identity
injected through the existing hooks:

    chemistry : cfg_overrides {Mp, Rp, r_star, orbit_radius, sflux_file, ...}
                (VULCAN derives gs = G*Mp/Rp^2; gs_cgs is converted to Mp at
                the boundary in forward.py)
    RT        : profile {rp_cm, gs_cgs, rstar_cm}
    noise     : star dict -> pandeia phoenix SED + Ks normalization
    timing    : t14_hr -> in/out-of-transit integration split

T-P and Kzz are explicit modes, never silently substituted. Under Guillot the
structural baseline is isothermal at a representative temperature (it sets
only the hydrostatic grid + EQ init; tp_eval supplies T(P) for chemistry and
RT); ``file`` makes the tabulated profile the structure.
Kzz is const / Pfunc / JM16 / the table's own column.

Values are literature defaults for PLANNING (all editable in the GUI):
WASP-39b Mancini+2018/Tsai+2023; HD 189733b Torres+2008 (a: Bouchy+2005);
HD 209458b Torres+2008 (a, gs from Southworth+2010); WASP-107b Piaulet+2021.
Stellar UV: shipped VULCAN spectra, nearest available spectral type (shown in
the GUI, never silently swapped).

``star["metallicity"]`` is DELIBERATELY 0.0 for every host, not a literature
value: it only selects the PHOENIX node for the noise SED, both consumers
renormalize to the observed 2MASS Ks magnitude, and a 0.1-0.2 dex node shift
is a sub-percent effect on forecast sigmas. The GUI exposes [Fe/H] as an
editable field, so a run that cares sets it explicitly.
