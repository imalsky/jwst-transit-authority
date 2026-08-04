# Native-PICASO RT vs tool ExoJax RT: one-state CROSS-MODEL DISCREPANCY

**VERDICT: FAIL (outside target).** All three declared targets below are missed.
This artifact is a cross-model discrepancy record: it does NOT validate absolute
spectral agreement and must not be described as parity, or cited as evidence
that the consumers' physics is validated against real spectra. The two codes use
different opacity sources, broadening, and reference-radius conventions, so the
disagreement does not by itself identify a bug in either.

> **STALE - RERUN REQUIRED (relabeled 2026-08-03).**
> These numbers were generated 2026-07-20, BEFORE the 2026-07-28 audit put the
> tool's transmission RT on an inverse-square gravity profile
> (`vulcan_forward.exojax_rt._gravity_profile_invsq`). They therefore describe a
> code state that no longer exists, and the script docstring that produced them
> asserted the tool "uses the constant surface gravity" - false since that
> audit, and offered as part of the explanation for these residuals. The run
> also carries no code/dependency/data provenance, so it cannot be attributed to
> an exact state.
>
> Two things were fixed on 2026-08-03 WITHOUT rerunning: the heading and claim
> now follow the measurement (this file was titled "one-state parity" while
> missing all three targets), and the script now records full provenance and
> returns nonzero when a declared target fails. Regenerate on a machine with the
> PICASO reference tree:
>
> ```
> python tests/parity_picaso/scripts/run_native_rt_parity.py
> ```
>
> Until that rerun lands, treat every number below as historical.

Generated 2026-07-20 16:50 by scripts/run_native_rt_parity.py. OFFLINE
comparison only; the production path is always provider chemistry + ExoJax. See
the script docstring for the method and why exact agreement is not expected
(different opacity sources + reference-radius conventions). Gravity is NOT one
of the differences in current code: both sides integrate altitude on an
inverse-square profile.

State: W39b geometry, isothermal 1100 K, blended equilibrium at 10x solar / C/O
0.55, absorbers ['H2O', 'CO2', 'CO', 'CH4'] on H2/He. Native DB:
opacities_0.3_15_R15000.db.

| metric | value |
|---|---|
| broadband offset (removed) | -2207 ppm |
| median abs residual | 688 ppm |
| p95 abs residual | 1540 ppm |
| max abs residual | 2019 ppm |
| bins (R=100, 1.0-12.0 um) | 250 |

Declared targets:

- OUTSIDE TARGET: broadband offset |x| < 2000 ppm
- OUTSIDE TARGET: median |resid| < 150 ppm
- OUTSIDE TARGET: p95 |resid| < 400 ppm

## Provenance

Not recorded for this run - one of the defects that made it uninterpretable.
The regenerating script now writes a provenance table here: tool / engine /
ExoJAX / PICASO / NumPy / JAX versions, repository commits (marked `-dirty` when
the tree is not clean), the native opacity-database identity and size, and a
`version_metadata_consistent` flag that catches a stale editable install whose
recorded version disagrees with the code actually imported.

Figure: ../figs/parity_native_rt.png
