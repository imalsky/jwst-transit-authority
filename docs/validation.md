# Validation and release gate

This file defines what evidence is required; a passing unit suite alone is not
a scientific release.

## Required evidence

| Claim | Independent check | Release artifact |
|---|---|---|
| Pandeia units and transit propagation | literal analytic count/rate oracles, unequal baselines, multiple transits | unit log and equation mapping |
| Count-weighted binning | hand-computed bins, descending grids, partial saturation | unit log |
| Ramp and saturation policy | native per-pixel masks and maximal-safe ramp search | fixed-source PandExo parity JSON/report |
| Instrument coverage | bright, WASP-39-like, and faint stars across all eight modes | 3 x 8 matrix |
| Operational policy | unmodified PandExo templates, warnings fail closed | default-policy report |
| Fisher implementation | analytic full-rank/singular cases, whitened-design SVD, direct likelihood curvature, reorder/rescale tests | unit log and benchmark JSON |
| Jacobians | predeclared multi-step finite-difference closure against AD | closure report |
| VULCAN-JAX chemistry | exact frozen upstream oracle plus shipped planet cases | parity report |
| Forward RT | transmission/emission golden outputs and optical-depth checks | benchmark tables/plots |
| PICASO | dependency-compatible live run and current native-RT cross-model gate | currently excluded |
| Packaging | wheel build and install into an empty environment | build/install log |

## Hard gate

A collaborator release requires all of the following:

- exact, clean repository revisions and dependency/data checksums;
- Pandeia engine, refdata, and PSF markers from one supported release;
- no failed, skipped-required, stale, truncated, or unattributed validation;
- no open critical/high scientific finding;
- no unexplained numerical change from the frozen pre-refactor golden output;
- no operational warning described as a valid recommended configuration;
- no claim whose equation, test, artifact, limitation, and provenance cannot be
  traced in the bundle.

Thresholds are declared before a run. A failed observation is investigated or
the feature is excluded; the threshold is not widened to obtain a pass.

## Current audit status

The collaborator audit begun 2026-08-14 is not yet a passing release. The
PICASO dependency conflict and stale failing native-RT artifact exclude PICASO.
The current Pandeia/PandExo matrix must be regenerated after the fixed-source
normalization correction and must pass its per-pixel saturation gate. The
strict VULCAN-JAX suite also requires one oracle revision per declared parity
case plus a built FastChem executable; a mixed-oracle run is not valid
evidence.

The release package must preserve these failures as findings rather than copy
an older green banner.
