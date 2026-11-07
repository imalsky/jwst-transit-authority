# Validation scripts

## `make_parity_plots`

Render PandExo-parity figures from a gate-PASS parity_summary.json and,
where available, the raw per-wavelength run outputs.

Usage:
    python validation/parity/scripts/make_parity_plots.py

These figures show the quantities that are in PARITY between this tool and
current PandExo on the same Pandeia 2026.7 engine -- the things that match
1:1: the selected groups, integration time, integration counts, and the
extracted stellar flux. The depth-uncertainty difference (a noise-model
difference, not a configuration one) is quantified in REPORT.md and
parity_summary.json, not plotted.

The summary is RE-VALIDATED through the shared gate (parity_gate.py) before
any figure gets current-release labels; the persisted `gate.passed` boolean
is never trusted. The extracted-flux figure also reads the raw
{star}_{ours,pandexo}.json that run_parity.py writes into the outputs
directory (git-ignored); it is skipped with a notice if those are absent (a
fresh clone has the committed figures already, and re-running run_parity.py
regenerates the raw JSON).

Layout under validation/parity/: scripts/ (the plot and report scripts and the harness), outputs/ (the
committed parity_summary.json + REPORT.md and the git-ignored raw run JSON),
figs/ (the committed PNG figures this writes).

Style: validation/figstyle.py, the one style of every committed figure here
(serif, Okabe-Ito cycle, square panels, axis labels and legend only). Two-code
overlays are black under, red dashed on top; the timing panels colour by
instrument and mark by star. fs.save() embeds this script in each PNG.

## `make_report`

Render the short REPORT.md from parity_summary.json (after run_parity.py).

The report is a SUMMARY, kept deliberately short: gate banner, provenance,
and the measured envelope against each declared threshold. The durable
narrative -- method, scope, and known intended differences -- lives in section
`run_parity` of this file, and the full
per-mode rows, noise-model attribution, and raw PandExo warnings stay in
`parity_summary.json`, which is committed beside it. Do not grow this
renderer into a per-mode table dump; the JSON is already the record.

The report is generated from RE-VALIDATED JSON only: it imports the shared
gate (`parity_gate.py`) and re-runs `validate_artifact()` on the summary
instead of trusting the persisted `gate.passed` boolean, so a hand-edited
artifact cannot render as a PASS. Every numerical statement is computed from
the artifact -- nothing is hard-coded, so the text cannot go stale against
the JSON.

Usage: python validation/parity/scripts/make_report.py

## `pandexo_worker`

PandExo parity worker -- runs INSIDE a current-Pandeia environment.

Standalone on purpose (mirrors src/jwst_tool/pandeia_worker.py): stdlib +
numpy + pandexo only. Executes current PandExo (master, at the commit
parity_gate.REQUIRED_PANDEXO_COMMIT pins) on matched star/instrument
configurations and dumps the raw quantities the parity comparison needs.

    python pandexo_worker.py job.json result.json

job.json:
    {"refdata": <pandeia_refdata>, "psf_dir": <pandeia PSF library>,
     "cdbs": <PYSYN_CDBS>, "vega_file": <local CALSPEC alpha_lyr fits>,
     "star": {"teff":.., "logg":.., "metal":.., "kmag":..},
     "transit_duration_s": .., "sat_level_pct": 80.0, "depth": 0.01,
     "modes": [{"key":.., "pandexo_name":..,
                "config_overrides": {"detector": {...}, "instrument": {...},
                                     "strategy": {...}}},
               ...]}

result.json: {"__provenance__": {...}, <key>: {...} | {"error": traceback}}.
Per mode key: PandExo's native-grid (R=None) results with noise_floor=0 and
baseline fraction 1.0 (out-of-transit time == in-transit time):
    wave, error (final error with floor=0), timing (PandExo timing dict),
    ngroup, config (the exact pandeia configuration PandExo ran),
    electrons_out/in, var_out/in, e_rate_out, error_no_floor, warnings,
    qy_on_grid (the quantum-yield curve PandExo's remove_QY divided out of
    every flux; multiply back to recover pandeia's electron rate).

## `run_parity`

PandExo numerical parity harness (release gate).

Runs MATCHED star/instrument configurations through BOTH noise paths on the
SAME Pandeia backend and compares them mode by mode:

  * this package's worker (src/jwst_tool/pandeia_worker.py) + its box-transit
    depth-error propagation (noise.pixel_depth_variance), and
  * current PandExo (master; validation/parity/scripts/pandexo_worker.py).

Running both sides on one engine/refdata generation isolates ESTIMATOR
differences (timing policy, in/out propagation, saturation handling) from
engine-calibration differences -- the point of the parity gate.
This is a FIXED-CONFIGURATION estimator gate: the submitted instrument
configuration is pinned identically on both sides (run_parity.PANDEXO_MODES
overrides PandExo's templates to this tool's registry), so it deliberately
does NOT test PandExo's own configuration-selection policy. The harness
points the worker at the backend under test explicitly via environment
variables. The gate (parity_gate.validate) refuses the run unless BOTH
sides are on the same SUPPORTED engine/refdata/PSF triple.

Required environment (all loud, no defaults -- machine paths stay out of git;
main() scrubs absolute paths from the provenance before writing the summary):
  JWST_TOOL_PANDEIA_PYTHON   python of an environment with pandeia.engine
                             2026.7 AND pandexo (master) installed
  JWST_TOOL_PANDEIA_REFDATA  extracted pandeia_data-2026.7-jwst tree
  JWST_TOOL_PANDEIA_PSF_DIR  extracted pandeia_psfs-2026.7-jwst tree
  JWST_TOOL_DATA_DIR         directory whose cdbs/ holds the phoenix grid,
                             calspec Vega, and comp/nonhst bandpasses
  JWST_TOOL_OUTPUT_DIR       the tool's own worker noise cache (model_cache /
                             noise_cache); unrelated to the parity artifacts

Usage: python validation/parity/scripts/run_parity.py [--impulse]
(--impulse adds the LSF impulse-response block to an existing
parity_summary.json; it reuses the full run's shared stellar spectrum.)
The raw per-run JSON (git-ignored) is written to validation/parity/outputs/,
beside the committed parity_summary.json and REPORT.md; make_parity_plots.py
writes the figures to validation/parity/figs/. The tool's Pandeia noise
cache goes under JWST_TOOL_OUTPUT_DIR (that is the app's cache, not a parity
output).

## `chem_cases`

Writes validation/data/case_<name>.npz: the chemistry solves behind the Tsai-2023 and ERS-CO2 validation figures.

One WASP-39 b case per invocation (fresh process). Needs the engine data root and this repo's jax environment:

    python validation/scripts/inputs/chem_cases.py tsai_e5   # no argument lists the cases

Metallicity is set where the engine reads it: the config's explicit
C_H/N_H/O_H/S_H elemental settings (the equilibrium seed in ini_abun reads
those for network elements). He stays at the config value. The ERS fiducial
keeps 10x O/N/S and sets C_H = 0.35 * O_H (their C/O = 0.35).

Metallicity is not set through theta: audit_init confirms lnZ = ln 2 doubles
the initial column inventory exactly, yet the converged column returns to the
baseline inventory (a closed column, so that cannot be atom-conserving). Each
case therefore runs at theta = 0 and ASSERTS the converged column inventory.

The morning T-P file ships without a Kzz column; Tsai et al. 2023 use one
Kzz profile, so the evening file's Kzz(P) is interpolated onto the morning
grid (written next to the outputs).

## `exok_ckd_values`

Writes validation/data/exok_ckd_vals.npz: the exo_k 1.3.1 side of the correlated-k cross-validation figure.

Per molecule in the engine's k-table tree ($VULCAN_FORWARD_DATA/exomolop):
probe (T, P) points (grid nodes + random interior + out-of-range clamping,
seed 42), exo_k Ktable.interpolate_kdata (log_interp) vs vulcan_forward
load_tables + ckd._interp_logk. Saves the probes (reused verbatim by the
figure script for the ExoJAX side), the per-molecule worst |dln k| (native)
and worst |dlog10 band-mean sigma| binned R=1000 -> R=100, and the H2O
band-mean spectrum at 1100 K, 1 mbar over 2000-10000 cm^-1.

exo_k pulls numpy 2 and never enters this repo's jax environment. Run OFFLINE
in a throwaway venv (same recipe as vulcan-forward's exok_ref_overlap.npz
fixture generator, plus the engine itself for the comparison side):

    python -m venv exok-venv
    exok-venv/bin/pip install "exo_k==1.3.1" jax h5py
    exok-venv/bin/pip install --no-deps -e <the vulcan-forward checkout>
    exok-venv/bin/python validation/scripts/inputs/exok_ckd_values.py

Zero-k table entries are floored to exomolop.K_FLOOR (1e-60) in exo_k's
kdata BEFORE interpolation, matching load_tables' floor, so the log-space
interpolations are the same mathematical operation everywhere (exo_k is
linear in T and log10 P on log k; ours in T and ln P -- identical weights).

# Test notes

## `test_e2e_run_model`

Full-chain end-to-end regression: chemistry -> ExoJAX RT -> spectrum, for both engines (VULCAN kinetics; PICASO climate + equilibrium chemistry).

Runs the tool's real `run_model` for the default WASP-39 b case in BOTH
science modes -- a fresh solve each (on an empty model cache, so this
validates the chain, not the cache) -- and asserts the R = 100 binned
spectrum against a committed reference fixture generated under the same
`model_defs._VERSION`. The engine side of the physics is separately pinned
against petitRADTRANS in vulcan-forward's own e2e suite; THIS test pins that
the whole tool still reproduces its own verified state end to end.

Cost: one cold solve per case, about 9 min per VULCAN mode and 20 min per
PICASO mode on daw. Gated like the FD closure test: JWST_TOOL_RUN_SLOW=1,
plus clean skips without the RT stack, the engine data or the PICASO
environment.

Tolerances: cross-run JAX nondeterminism on this chain measures under a ppm
(the committed reference reproduces bit-identically), so the band is set per
mode at 5% of the reference's own structure -- an absolute ppm would be far
too loose on emission, where a quarter of the bins sit below 100 ppm while
transmission runs near 20000. On a breach: if the change is INTENDED,
regenerate the fixture (its meta["regenerate"]) and bump model_defs._VERSION.

## `test_picaso_published`

PICASO engine vs a published PICASO spectrum: WASP-39 b, Rustamkulov et al. 2023 (Nature 614, 659), the PICASO 3.0 best fit to the NIRSpec PRISM ERS spectrum (Zenodo 10.5281/zenodo.7388032, CC-BY-4.0).

Published parameters: 10x solar, C/O 0.68, a grey cloud kappa = 10^-2.07 cm2/g
per gram of atmosphere (their eq. 3, the same deck as the tool's gray cloud),
H2-only Rayleigh, SO2 injected at log10 VMR -5.2. The geometry is the PICASO
WASP-39 b grid's (R = 1.279 RJ at 10 bar, 0.28 MJ, Teff 5326.6 K, a = 0.0486 AU;
attrs of best_fit_PICASO_grey.nc in Zenodo 7101283). The T(P) is this tool's
own PICASO climate at those parameters (T_int 200 K), frozen in the data file
so the test costs one chemistry column and one RT pass (~1 min on daw) instead
of a climate solve; the climate itself is pinned by the e2e fixture.

What this pins: composition imposition, equilibrium chemistry, ExoMolOP
correlated-k RT, the gray cloud and the radius/gravity anchor, together,
against an independent code's published spectrum. At R = 100 over 1.02-5.30 um
with the SO2 band (3.95-4.15 um) excluded and a constant offset removed, the
residual rms is 39.0 ppm (std ratio 0.937; H2O 1.4 um amplitude ratio 0.95,
CO2 4.3 um 0.95 against a 3.70-3.85 um continuum). The published spectrum's
std is 390 ppm, so the 60 ppm bound trips on a ~10% amplitude change or an
equivalent shape change: the same column clear gives 424 ppm, and a climate
run at the tool's default radius anchor (1.279 RJ at 1 mbar, the registry
star and orbit) 120 ppm. The inter-grid spread of
the four published ERS best fits on the same data is 70-122 ppm rms. Peak
memory 5.4 GB (the RT), 53 s on one daw core.

## `test_toi7169_dlnco`

The +h dlnCO stencil point of a 10x-solar, C/O 0.55, photo-on column (the TOI-7169 b CO case in data/toi7169_co_case.json) certifies via the photolysis-cadence escalation.

At the config's refresh every 5th accepted step this point does not certify:
dt can grow up to 2^5 between refreshes, and at the recorded event current
water shielding gave Ros2 error 0.43 against rtol 0.2 while the previous water
profile for radiation alone gave 0.06, so the column rejected every refresh
step. `certify.certified_solve` re-solves such a column once with photolysis
refreshed every accepted step.

The case was recorded on the sncho network and vulcan3 freezes the first network
a process imports, so the solves run in a child process.

Cost: two cold solves, about an hour on daw. Gated like the e2e:
JWST_TOOL_RUN_SLOW=1.

## `test_engine_api_contract`

The tool's calls into the shared engine must exist on the INSTALLED engine.

`deploy/pins.env` pins the sibling repositories by commit, and
`test_deploy_pins.py` checks that the Dockerfile and CI agree on those strings.
What neither can check is whether the pinned pair is COMPATIBLE. Shipping
VULCAN3 at a commit that has removed a parameter alongside vulcan-forward
at a commit that still passes it makes every fresh forward model on the
deployed Space raise TypeError, while three green suites and a passing pin
test say nothing, because no test called the engine.

This closes that at the consumer boundary: CI installs the siblings at the
pinned commits, so an incompatible pair fails HERE. It is deliberately a
signature check, not a run -- it needs no data, no k-tables and no solve.

The sibling repository carries the mirror of this test for its own calls into
VULCAN-JAX's private API (`vulcan-forward/tests/test_vulcan3_private_api.py`).

## `test_shared_constants_pin`

G, R_sun and AU are duplicated across repos; pin them.

`chem_build.assemble_chem` converts this tool's surface-gravity input into the
planet mass VULCAN-JAX wants:

    Mp = gs_cgs * rp_cm**2 / planets.G_CGS

and `vulcan3.atm_setup.surface_gravity` inverts it:

    gs = phy_const.G_grav * Mp / Rp**2

so `gs` round-trips ONLY because two independently-declared literals carry
the same digits. A one-digit edit on either side would silently rescale every
atmosphere this tool submits, with no error anywhere: the value stays
physical, just wrong.

The constants stay where they are: VULCAN-JAX sits BELOW vulcan-forward in the
dependency DAG and must keep its own `phy_const`, so moving them into
`vulcan_forward.constants` would leave the same two-copy coupling plus an
extra import edge. A pin is the fix, in the style of vulcan-forward's
`_assert_composition_tables`.

VULCAN-JAX is read from SOURCE with `ast`, never imported: the pin needs
only the literals, and importing `vulcan3` sets process-wide JAX config
(x64) and loads its config module.

## `make_app_lock`

Generate requirements-app-lock.txt from the LIVE working environment.

Walks the installed dependency closure of the app stack and pins every
package at exactly the version running here. The Docker build installs this
lock with --no-deps so it reproduces the validated environment verbatim rather
than selecting newer transitive dependencies during a later rebuild. Generate
the lock only from a resolver-clean environment and run ``pip check`` before
committing it; the deployed working set must not rely on ignored dependency
metadata.

Run in the environment that runs the tool, then commit the lock:
    python deploy/make_app_lock.py

The PICASO env is a conda env of its own (picaso 4.1 needs numpy >= 2). Conda
installs its compiled stack from picaso-conda-lock.txt; this script's
``--env picaso`` mode pins only the distributions pip installed on top. Run
both in the validated env (``pip check`` clean), then commit both locks:
    micromamba env export -p <env> --explicit --md5 > deploy/picaso-conda-lock.txt
    <env>/bin/python deploy/make_app_lock.py --env picaso
