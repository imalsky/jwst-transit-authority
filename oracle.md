# oracle.md — what JWST Transit Authority must achieve

This file states the goal of JWST Transit Authority (this repo) and what counts as success or failure. It is the
yardstick for judging whether the tool is ready to release or deploy. It holds targets and decisions, not
measurements: the evidence lives in notes.md, named here by section heading. Only the maintainer changes this file.
The forward model's own criteria are in vulcan-forward's `oracle.md`, and the chemistry solver's in vulcan3's.

## Goal: an honest detectability forecast

JWST Transit Authority tells a proposer how well a JWST transit or eclipse observation detects a molecule or
constrains metallicity and C/O. It solves the chemistry (VULCAN by default, PICASO as a second engine), computes the
spectrum with the shared forward model and the noise with Pandeia, and is used through a public web app, a local GUI,
a Python API and a command line. It is a forecast, not a retrieval. The method is the same for every planet: a planet
is a registry entry or a custom system, and nothing in the code belongs to one planet.

Correct means three things. Every number shown comes from a certified model. The model holds what the planet and the
observation need, or the run stops with the reason. Every number is in honest units and labelled for what it is.

### Success: the machinery

- **Certified columns.** Every spectrum comes from a column that passes the solver's certificate, the element gate
  and the unmodeled-absorber gate (notes "Convergence certificate and element gate").
- **Certified derivatives.** A Fisher Jacobian row is a certified finite difference or a certified forward-mode
  derivative; an uncertified derivative is never shown (notes "Jacobians and Fisher").
- **Noise matches Pandeia.** One measurement operator bins noise, model and Jacobians, and the parity matrix against
  PandExo passes its gate (notes "Parity testing").
- **A cache never serves older physics.** A change to the physics or to an engine pin changes the cache key.
- **Settings are visible.** A number that changes a result is a named setting the user can read and change in the
  configuration, with its unit and its reason. It is not a value buried in the code. A physical constant or a
  numerical guard that is fixed on purpose is a named constant with its reason beside it.
- **One definition per default.** A default that both the API and the GUI use has one definition, so the two cannot
  drift apart.

### Success: the science

- **The model follows the observation.** A transit is modelled at the terminator and an eclipse on the dayside, in
  the temperature profile and in the photolysis geometry alike.
- **The core absorbers are there.** H2O, CO2, CO, CH4 and NH3 are always in the spectrum. HCN, H2S or SO2 left out
  is shown to the user with the result. A minor species is the user's choice.
- **The star is the planet's star.** The UV spectrum that drives the photochemistry covers the network's photolysis
  range, and the stellar flux of an eclipse depth is averaged the way the planet flux is.
- **One planet, one set of numbers.** A planet this tool shares with the retrieval's cases has the same radius,
  gravity, star, orbit and UV spectrum there and here, or the difference is recorded with its reason.
- **The standard opacity terms are on.** H2-H2 and H2-He collision-induced absorption always, H2 and He Rayleigh
  scattering in transmission by default. A cloud is never a default.
- **Inside its bounds, or an error.** An input outside the range the model covers (temperature, pressure,
  composition, wavelength, stellar parameters) is refused with the reason, never clipped.

### Success: the output

- **Labelled for what it is.** A significance is a conditional template signal-to-noise, never a retrieval claim.
  A C/O uncertainty is in dex. A curve fitted to one noise realization is never called a forecast.
- **Limits are listed.** Everything the model leaves out that could move a shown number is in notes "Accepted
  limitations", and a limitation that applies to a run is said beside its result.

### Success: the checks

- **Fast enough to run.** No test takes more than 90 minutes on a CPU.
- **End to end, both engines, both modes.** The default case, solved cold through the whole chain, reproduces a
  committed reference made at the same model version (`tests/live/test_e2e_run_model.py`).
- **Against other codes and data.** The radiative transfer against petitRADTRANS on this tool's own columns, the
  noise against PandExo, and the spectra against published models and observations (`validation/`), regenerated at
  the shipped settings whenever a default spectrum moves.
- **The web app runs what was tested.** It deploys from pinned, tested commits, and the live checks (a cold transit
  and a cold eclipse run, a configuration export and reload, an out-of-range refusal) pass on the deployed version.

### Failure

- A shown number from an uncertified column or derivative, or in the wrong units.
- A cache serving a result made by older code.
- A default run that leaves out a core absorber, or uses another star's or another geometry's input, without saying
  so.
- A planet with two sets of numbers and no recorded reason.
- A clipped input, or a silent fallback or default where the code should refuse.
- A number that changes a result, hidden in the code where the user cannot see or set it.
- A limitation that could move a shown number and is not in notes "Accepted limitations".
- A validation figure or reference made at settings the tool does not ship.
- A deployed app built from a pair of engine and tool commits that was not tested together.

## Ready to use

The repo is ready when, at the commit to be released:

- the normal tier passes locally and in CI, and the slow tier passes when the change touches what it covers;
- the model version, both end-to-end references and the validation set agree with the code;
- its version floors and deploy pins name released, tested commits of vulcan3 and vulcan-forward;
- no `[owed]` or `[stale]` item in todo.md backs a claim the release makes, and nothing above under Failure applies.
