# todo

Open, owed or waiting on a decision, in priority order. Evidence is in notes.md, by
section name.

- Decide the constrain verdict's C/O width. `gui_results._constrain_verdict` prints `fisher.display_sigma`'s dlnCO value
  (`±{bs:.3g}`, no unit), which CLAUDE.md "Jacobians, Fisher and statistics" forbids, while the table on the same page is
  in dex. Either keep the absolute comparison against the target and print through `format_co_width`, or relax the rule.
  Display only; no number moves.
