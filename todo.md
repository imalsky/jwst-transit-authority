# todo

Open, owed or waiting on a decision, in priority order. Evidence is in notes.md, by
section name.

1. [owed] v62 (NH3 always on, survey values and UV files, dayside angle, band-mean stellar flux): the slow tier.
2. [blocker] v62 needs an unreleased vulcan-forward: its floor and the deploy pins move with that release.

## 1. v62: the slow tier

Evidence: notes "Opacity and radiative transfer" and "Emission" (what v62 changed and by how much) and "Current
state". Done: the normal tier, both e2e fixtures regenerated from their embedded scripts, and the validation
set (chemistry cases, atmosphere cases, pRT references, figures) regenerated at the shipped model top. Owed:

- The slow tier at v62 (`JWST_TOOL_RUN_SLOW=1`, the command in CLAUDE.md "Setup and commands"; about 80 min on 15
  CPUs).
- WASP-39 b and HD 189733 b are not aligned with the sulfur survey's system values (notes "Opacity and radiative
  transfer"); the maintainer has not asked for it.
- HCN and H2S left out are a log line only; nothing in the result or the GUI shows it.

Done when the slow tier passes at v62.

## 2. The engine release v62 depends on

v62 reads `nu_edges`, `unlisted_core_molecules` and `omit_core_molecules` from vulcan-forward's RT result, and its
WASP-107 b default relies on nothing else there, but a default run on the pinned engine (`deploy/pins.env`, 0accd36)
fails at the echo check. Order: vulcan-forward's version and push, then this repo's floor (`pyproject.toml`), the
pins (`deploy/pins.env`, the Dockerfile, CI) and its push. Never deploy the Space from a mixed pair.
