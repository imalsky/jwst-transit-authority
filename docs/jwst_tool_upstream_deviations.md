# jwst-tool effective chemistry config vs upstream VULCAN master (W39b)

Date: 2026-07-15. Method: field-by-field diff of `VULCAN-master/vulcan_cfg.py`
(W39b-configured, the parity oracle) against the tool's EFFECTIVE VULCAN-JAX
cfg at GUI defaults (`W39b.yaml` + `vulcan_chem` profile application + the
tool's `cfg_overrides`, forward v12). Script: session scratchpad `cfg_diff.py`.

## Fields identical in both (the reassuring bulk)

Elemental abundances (C_H 2.95e-3, O_H 5.37e-3, N_H, S_H, He_H), solver
tolerances (rtol 0.2, atol 0.1, mtol, mtol_conv), step counters (count_max
3e4, count_min 120, trun_min), convergence thresholds (yconv_cri 1e-2,
yconv_min 0.1, slope_cri, flux_cri), photolysis geometry (sl_angle 83 deg,
f_diurnal 1.0), photo update cadence (ini/final 5/5), EQ initialization +
FastChem file, stellar UV file, boundary-condition files, use_photo on,
use_moldiff on, use_vm_mol off (tool PINS it since v11; upstream YAML default
flipped on 2026-07-14), condensation/settling off in both.

## Real deviations (tool vs master), with status

| Field | Master (W39b) | Tool effective | Status |
|---|---|---|---|
| `atm_type` | `file` (GCM evening-terminator T-P) | `isothermal` structural + live `tp_eval` (iso/Guillot) | DELIBERATE, documented: GCM baseline removed 2026-07-13; every planet gets an explicit T-P. Tool answers differ from Tsai-2023-style runs by construction. |
| `Kzz_prof` | `file` (GCM Kzz(z)) | `const`, GUI default 1e9 | DELIBERATE (same removal). Master's `const_Kzz=1e10` is inert there. Note the GUI default 1e9 is a choice, not Tsai's profile; slider spans 1e6-1e12. |
| `dt_max` | 1e17 (`runtime*1e-5`) | 1e11 | DELIBERATE, documented (validated state-preserving; prevents the adaptive-dt balloon; master's own uncapped value implicated in the photo-off blow-up, see VULCAN-JAX/docs/vulcan_jax_notes.md (2026-07-15 photo-off entry)). |
| `nz` | 150 | 100 (GUI default; 150 available) | DELIBERATE fast-tier default; use 150 + yconv 1e-3 for final numbers (documented deltas: MIRI LRS halves at 100/1e-2). |
| `conver_ignore` | 13 heavy hydrocarbons (C2H2, C6H6, ...) | `['HC3N']` only | FLAGGED: the JAX side is STRICTER than master (more species must certify). Conservative, but hard corners (cool / C-rich) may fail the longdy gate where master would declare convergence. Provenance of the shorter list unclear; review whether the master list should be adopted for parity. |
| `network` | `NCHO_photo_network.txt` (current oracle parking) | `SNCHO_photo_network.txt` | Not a tool deviation: SNCHO is the Tsai 2023 W39b science network (sulfur/SO2). The workspace master cfg is parked on NCHO for the oracle tests. |
| `wall_clock_max` | 1800 s | 3600 s | Runner backstop doubled on the JAX side. Benign (a backstop, not physics). |
| `Tiso` | 1000 (inert; atm_type=file) | 1100 (GUI default near W39b Teq) | Inert in master; not a comparison. |
| `gs` vs `Mp` | gs directly | `Mp = gs*Rp^2/G` | Exactly equivalent by construction (checked 2026-07-15). |

## VULCAN-JAX-only feature toggles (confirmed inert at tool defaults)

`use_chunked_runner`, `use_fix_H2He`, `use_fix_all_bot`, `use_ini_cold_trap`,
`use_pi_controller`, `use_sat_surfaceH2O`, `use_adapt_rtol` all False;
`use_hybrid_vm_mol` pinned False by the tool (v11). No silent feature is
active that master lacks.

## Tool-level constructions with no master counterpart

- Two-stage warm-started composition continuation (lnZ/dco steps from the
  relaxed column) instead of cold re-solves; validated around the O-rich
  baseline; refused elsewhere (v12 gates).
- The differential fixed-O `dco` knob (exact delta-ln C/O through the solve,
  AD-capable). Master's own convention IS fixed-O (its cfg anchors `O_H` and
  edits `C_H`), so the tool's differential direction matches the upstream
  convention; the perturbative form adds the C/O < 1 ceiling (b_z bound).
- The structural `co_baseline` mode (v12) reproduces master's workflow
  exactly: pin `C_H = co * O_H`, FastChem re-initializes at the target C/O
  (any value incl. C-rich); detection-only.
- The longdy convergence gate refuses non-certified states loudly (master
  reports but does not refuse).

## Engine-level (VULCAN-JAX vs master) differences

Documented separately: `VULCAN-JAX/docs/validation.md` (VULCAN-JAX
parity and bug guide) and `jax_paper/paper/notes_gaps.md` (ranked gaps;
headline: reservoir projection default-on in JAX vs none in master). Those
apply to every consumer of VULCAN-JAX, not just this tool.
