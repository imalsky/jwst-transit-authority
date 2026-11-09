# The forward model

## `forward`

Forward model runner: chemistry -> ExoJAX spectrum.

The chemistry is VULCAN-JAX photochemical kinetics (the default) or, with
``chem_provider='picaso'``, PICASO climate + equilibrium chemistry
(``picaso_engine.py``). The npz cache in ``instruments.MODEL_CACHE`` holds
spectra under ``params_key`` and converged VULCAN chemistry columns under
``chem-<chem_key>``; the GUI's light path (``model_cache``: ``params_key`` /
``cache_path`` / ``load_result``) reads only the ``params_key`` entries.
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

## The stages

`forward` runs the stages below, then `jacobian` builds the rows. It re-exports, in `__all__`, the names the deploy scripts
and the research note's scripts read.

- `run_params`, `param_checks`: `canonical_params` fills every default, rounds, zeroes settings inert for the chosen
  modes so one physics maps to one key, and refuses unknown keys, out-of-range values and conflicts. A refusal costs no
  solve. The output maps to itself, so a saved payload validates by feeding it back in.
- `model_defs`: what the model offers, the FD steps and stencil, `_VERSION`.
- `model_cache`, `tp_tables`: keys and cache reads, with no JAX; T-P tables, shipped or content-addressed uploads.
- `chem_build`: assembles the VULCAN chemistry build and refuses a cfg that drifted from the constants here.
- `certify`: the gates a column passes before a spectrum is built, each raising. `certified_solve` raises `NotCertified`
  on a stall, the only failure that may escalate photolysis cadence.
- `spectrum`: builds the RT, and `ColumnSpectrum` is the depth of one chemistry build, transit or eclipse behind one
  signature.

## `jacobian`

Rows are built around the certified baseline (`Baseline`).

- An FD row solves every stencil point at one photolysis cadence: the cadence moves a converged spectrum by up to 2.4 ppm
  on the presets, which a mixed row would divide by h and report as derivative. On `NotCertified` the whole row re-solves
  at cadence 1, a one-sided anchor included, and is flagged in `photo_escalated`. Any other failure propagates.
- The stencil is central ($\pm h, \pm 2h$), or one-sided toward lower values ($-h, -2h, -4h$ at $h/2$, so its reach
  equals the central $2h$) when a composition row would leave its bound. The $h$-vs-$2h$ estimates must agree within
  `FD_CONSISTENCY_TOL` and combine by Richardson, $(4 J_h - J_{2h})/3$, fourth order central and third order one-sided.
- An AD row takes one warm jvp through the engine's `converged_y_jvp` along `FD_STEPS[name] * e_i`, and is divided by the
  step. A row that does not certify re-solves on a shared cadence-1 build, and every later row uses that build, flagged.
- Cloud-deck rows and `lnR0` are RT-only: one central difference with no $h$-vs-$2h$ check (`fd_err` NaN, meaning
  unmeasured) or an RT jvp.

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
* the rate-uncertainty spread assumes a uniform class-B 0.65 dex per reaction, attributed to Agundez (2025), whom
  this repository could not resolve to a publication (see the glossary): a stated assumption, not per-reaction.
* dL/dT is the chemistry-path gradient (photolysis cross sections and the
  diffusion/geometry rebuild frozen, upstream contract); the
  rebuild-consistency metric is stored.

Two faces: the light cache API (no JAX imports) and the heavy script mode
(``python -m jwst_tool.adjoint_diag params.json SO2``).
Condensation is refused up front: the pinned reservoir is a
step-sequence-dependent transient (tangents through it are ~91% wrong, see
param_checks.check_condensation). The PICASO engine is refused too: it has no
reaction network to differentiate.

## `stellar`

Stellar surface flux for eclipse depths (emission mode).

``phoenix_surface_flux(nu_grid, teff, logg, feh)`` returns the stellar
SURFACE flux density per wavenumber (erg s^-1 cm^-2 / cm^-1) on the native RT
grid, from the same minimal-CDBS PHOENIX grid the Pandeia noise side uses.

Units contract: the CDBS PHOENIX models are emergent SURFACE flux densities
and ExoJAX ``ArtEmisPure`` returns the planet's in the same convention, so

$$d_{ec}(\nu) = \frac{F_p(\nu)}{F_s(\nu)} \left(\frac{R_p}{R_s}\right)^2$$

with no extra $\pi$. Guarded by an energy-closure check: band-integrated F_s
over band-integrated pi B_nu(T_eff) must lie in [0.5, 1.5] -- a grid that
were secretly intensity (~3.1x) or Eddington flux (~12x) fails immediately.

Heavy-path module: imports stsynphot on first call; never imported by the
GUI's light path.

## `planets`

Planet/system registry for the JWST instrument selector.

Pure data, importable by the light GUI path (no jax/vulcan/exojax imports).

Every planet runs on the same W39b-validated machinery (a network from `model_defs.NETWORKS`, default `sncho2025`, and
the 10x-solar equilibrium baseline), with the planet identity injected through the existing hooks:

- chemistry: cfg_overrides `{Mp, Rp, r_star, orbit_radius, sflux_file, ...}`. VULCAN derives $g = G M_p / R_p^2$;
  `gs_cgs` is converted to `Mp` at the boundary in `chem_build._cfg_overrides`.
- RT: profile `{rp_cm, gs_cgs, rstar_cm}`.
- noise: star dict to the Pandeia PHOENIX SED and Ks normalization.
- timing: `t14_hr` to the in- and out-of-transit integration split.

T-P and Kzz are explicit modes, never silently substituted. Under Guillot the
structural baseline is isothermal at a representative temperature (it sets
only the hydrostatic grid + EQ init; tp_eval supplies T(P) for chemistry and
RT); ``file`` makes the tabulated profile the structure.
Kzz is const / Pfunc / JM16 / the table's own column.

Values are literature defaults for PLANNING (all editable in the GUI):
WASP-39b Mancini+2018/Tsai+2023; HD 189733b Torres+2008 (a: Bouchy+2005);
HD 209458b Torres+2008 (a, gs from Southworth+2010); WASP-107b Piaulet+2021.
Stellar UV: one shipped VULCAN spectrum per planet, its own host's where one ships,
else the closest shipped spectral type; the GUI shows which, and no run swaps it.

``star["metallicity"]`` is DELIBERATELY 0.0 for every host, not a literature
value: it only selects the PHOENIX node for the noise SED, both consumers
renormalize to the observed 2MASS Ks magnitude, and a 0.1-0.2 dex node shift
is a sub-percent effect on forecast sigmas. The GUI exposes [Fe/H] as an
editable field, so a run that cares sets it explicitly.
