# todo

Open, owed or waiting on a decision, in priority order. Evidence is in notes.md, by
section name.

- Decide the constrain verdict's C/O width. `gui_results._constrain_verdict` prints `fisher.display_sigma`'s dlnCO value
  (`±{bs:.3g}`, no unit), which CLAUDE.md "Jacobians, Fisher and statistics" forbids, while the table on the same page is
  in dex. Either keep the absolute comparison against the target and print through `format_co_width`, or relax the rule.
  Display only; no number moves.
- Decide the photolysis-cadence escalation: keep it, or refuse on a stall. It is a second solve at other numerics, against
  the rule of no retry; no shipped request reaches it, and two recorded custom-planet cases do (evidence: notes "Jacobians
  and Fisher"). Removing it refuses the TOI-7169 b FD dlnCO row, whose AD row certifies at the config cadence, and needs
  `AD_BUILD_OVERRIDES` count_max 30000 to keep the LP 714-47 b AD lnZ row. A cached escalated column means a `_VERSION`
  bump and both e2e fixtures.
