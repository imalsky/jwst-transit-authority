# todo

Open, owed or waiting on a decision, in priority order. Evidence is in notes.md, by
section name.

1. [owed] v62 (NH3 always on, survey values and UV files, dayside angle, band-mean stellar flux): the slow tier again
   once the engine's pending interface change lands.
2. [blocker] v62 needs an unreleased vulcan-forward: its floor and the deploy pins move with that release.
3. [decision] Fixed numbers outside any input: the finite-difference steps, defaults repeated in the GUI and the share
   file, and about 146 result-changing numbers in the code.
4. [science] Three gates without a recorded basis: the unmodeled-absorber band, the finite-difference tolerance and
   the radius-gravity anchor's pass count.

## 1. v62: the slow tier

Evidence: notes "Opacity and radiative transfer" and "Emission" (what v62 changed and by how much) and "Current
state". Done: the normal tier, both e2e fixtures regenerated from their embedded scripts, the validation set
(chemistry cases, atmosphere cases, pRT references, figures) regenerated at the shipped model top, and the slow tier
at v62 on the uncommitted engine. Owed:

- The slow tier again (`JWST_TOOL_RUN_SLOW=1`, the command in CLAUDE.md "Setup and commands"; about 70 min on 15
  CPUs) once the engine's pending interface change lands.
- WASP-39 b and HD 189733 b are not aligned with the sulfur survey's system values (notes "Opacity and radiative
  transfer"); the maintainer has not asked for it.

Done when the slow tier passes on the engine v62 ships with.

## 2. The engine release v62 depends on

v62 reads `nu_edges`, `unlisted_core_molecules` and `omit_core_molecules` from vulcan-forward's RT result, and its
WASP-107 b default relies on nothing else there, but a default run on the pinned engine (`deploy/pins.env`, 0accd36)
fails at the echo check. Order: vulcan-forward's version and push, then this repo's floor (`pyproject.toml`), the
pins (`deploy/pins.env`, the Dockerfile, CI) and its push. Never deploy the Space from a mixed pair.

## 3. Fixed numbers outside any input

Against the oracle's "Settings are visible" and "One definition per default". The maintainer decides which become
settings and which stay named constants with their reason.

- The finite-difference steps (`model_defs.FD_STEPS`) are fixed; a row that fails its step-size check can only be
  dropped or moved.
- Defaults are repeated: `share_config.WIDGET_RANGES` holds the widget defaults of the Pfunc and JM16 Kzz, the cloud
  deck and the diurnal factor, and `run_params._filled_payload` holds the same values again.
- About 146 numbers that change a result are fixed in the code, outside any input (an audit count, not re-measured).

Done when each is a setting or a named constant with its reason, by the maintainer's choice.

## 4. Three gates without a recorded basis

Evidence: notes "Convergence certificate and element gate", "Jacobians and Fisher" and "Structure: T-P, Kzz, gravity
anchor, model top".

- The unmodeled-absorber gate (`certify.check_unmodeled`) averages over the transmission photosphere
  (`certify.PHOTOSPHERE_BAR`, 1e-5 to 1e-2 bar) in emission too, and runs only on the baseline column, not on the
  finite-difference stencil columns, though CLAUDE.md "Certificates" says every certified column passes it.
- The finite-difference h-vs-2h tolerance (`jacobian.FD_CONSISTENCY_TOL`, 0.25) has no recorded basis.
- The radius-gravity anchor (`chem_build._radius_anchor`, and `picaso_engine._guillot_gravity` on the PICASO Guillot
  path) runs a fixed 3 passes with no convergence test; notes record two passes as enough on WASP-39 b. The PICASO
  climate path does test `ANCHOR_TOL`.

Done when each has a measured basis in notes, or is changed by a science decision.
