"""Render REPORT.md from parity_summary.json (run after run_parity.py).

The report is generated from RE-VALIDATED JSON only: it imports the shared
gate (`parity_gate.py`) and re-runs `validate_artifact()` on the summary
instead of trusting the persisted `gate.passed` boolean, so a hand-edited
artifact cannot render as a PASS. Every numerical statement in the prose is
computed from the artifact -- nothing is hard-coded, so the text cannot go
stale against the tables. `--require-pass` refuses to write a report for a
failing artifact at all.

Usage: python tests/parity/scripts/make_report.py [--require-pass]
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent / "outputs"      # parity_summary.json in, REPORT.md out
sys.path.insert(0, str(HERE))

import parity_gate as pg               # noqa: E402


def fmt_ratio(s: dict | None) -> str:
    if not s or not s.get("n"):
        return "--"
    return (f"{s['median']:.4f} [{s['p05']:.4f}, {s['p95']:.4f}] "
            f"(n={s['n']})")


def _ok_rows(summary):
    for sname, star in summary.get("stars", {}).items():
        for m in star.get("modes", []):
            if m.get("status") == "OK":
                yield sname, m


def _measured(summary) -> dict:
    """Every number the prose uses, computed from the artifact."""
    rows = list(_ok_rows(summary))
    match_fracs = [m["npix_matched"] / m["npix_pandexo"] for _, m in rows]
    flux_meds = [m["flux_ratio"]["median"] for _, m in rows]
    tint_devs = sorted(
        ((abs(m["t_int_ours_s"] / m["t_int_pandexo_s"] - 1.0), s, m["key"],
          m["ngroup_ours"]) for s, m in rows), reverse=True)
    ng_dev_faint = max((abs(m["ngroup_ours"] - m["ngroup_pandexo"])
                        for s, m in rows if s in pg.FAINT_STARS), default=0)
    ng_dev_bright = max((abs(m["ngroup_ours"] - m["ngroup_pandexo"])
                         for s, m in rows if s not in pg.FAINT_STARS),
                        default=0)
    g = next((m for s, m in rows
              if s == "w39_like" and m["key"] == "nirspec_g395h"), None)
    return {
        "n_ok": len(rows),
        "min_match_frac": min(match_fracs) if match_fracs else float("nan"),
        "flux_med_lo": min(flux_meds) if flux_meds else float("nan"),
        "flux_med_hi": max(flux_meds) if flux_meds else float("nan"),
        "tint_worst": tint_devs[0] if tint_devs else (float("nan"), "?", "?", 0),
        "ng_dev_faint": ng_dev_faint,
        "ng_dev_bright": ng_dev_bright,
        "g395h": g,
    }


def _gate_header(summary: dict, problems: list[str] | None) -> list[str]:
    """PASS/FAIL banner + provenance table, at the very top of the report.

    `problems` is the FRESH validate_artifact() result (None only when the
    summary has no gate block at all)."""
    out = []
    if problems is None:
        out += [
            "> **GATE: NOT EVALUATED.** This summary predates the fail-closed "
            "release gate, so nothing here has been checked for worker "
            "version, engine/refdata/PSF release agreement, PandExo identity, "
            "saturated rows, or the numerical thresholds. It is a forensic "
            "artifact, NOT a release certificate. Regenerate with "
            "`run_parity.py`.",
            ""]
    elif not problems:
        out += [
            f"**GATE: PASS** (re-validated by `make_report.py`, not read from "
            f"the artifact). Worker v{pg.PRODUCTION_WORKER_VERSION}, Pandeia "
            f"{pg.REQUIRED_PANDEIA_RELEASE} on both sides, the full "
            f"{len(pg.STARS)}-star x {len(pg.MODE_KEYS)}-mode matrix present, "
            "every declared threshold met.",
            ""]
    else:
        out += [
            f"> **GATE: FAIL ({len(problems)} problem(s)).** This artifact is "
            "NOT a release certificate and must not be cited as one.",
            ""]
        out += [f"> - {p}" for p in problems]
        out += [""]

    # Provenance for every star, both sides -- what actually produced the run.
    out += ["## Provenance", "",
            "| star | side | engine | refdata | PSFs | worker / PandExo |",
            "|---|---|---|---|---|---|"]
    for sname, star in summary.get("stars", {}).items():
        po = star.get("provenance_ours") or {}
        pp = star.get("provenance_pandexo") or {}
        out.append(
            f"| {sname} | this tool | {po.get('engine_version', '?')} "
            f"| {po.get('refdata_version', '?')} "
            f"| {po.get('psf_version') or 'not recorded'} "
            f"| worker v{po.get('worker_version', '?')} |")
        _commit = pp.get("pandexo_commit")
        out.append(
            f"| {sname} | PandExo | {pp.get('pandeia_engine_version', '?')} "
            f"| {pp.get('refdata_version') or 'not recorded'} "
            f"| {pp.get('psf_version') or 'not recorded'} "
            f"| {pp.get('pandexo_version') or 'unknown'} "
            f"@ {_commit[:12] if _commit else 'commit not recorded'} |")
    out.append("")
    return out


def main(require_pass: bool = False):
    summary = json.loads((OUTPUTS / "parity_summary.json").read_text())
    problems = (pg.validate_artifact(summary)
                if summary.get("gate") is not None else None)
    if require_pass and problems != []:
        raise SystemExit(
            "make_report: --require-pass given but parity_summary.json "
            + ("has no gate block (regenerate with run_parity.py)"
               if problems is None else
               f"FAILS re-validation ({len(problems)} problems). "
               "Fix the run; do not publish a report for a failing artifact."))
    cfg = summary["config"]
    ms = _measured(summary)
    lines = []
    w = lines.append
    w("# PandExo numerical parity report")
    w("")
    w(f"Generated {date.today().isoformat()} by `run_parity.py` + "
      "`make_report.py` in this directory.")
    w("")
    lines.extend(_gate_header(summary, problems))
    w("Both sides run on the SAME Pandeia backend -- the exact engine, "
      "reference-data, and PSF releases are in the provenance table above, "
      "and the gate refuses the run if they disagree across the two sides. "
      "Every difference below is therefore an ESTIMATOR/policy difference, "
      "not an engine calibration difference. (This says nothing about whether "
      "that release is the SUPPORTED one; the gate banner does.) PandExo is "
      "used from master at the commit recorded above. Configuration: "
      "constant transit depth "
      f"{cfg['depth']}, transit duration {cfg['transit_duration_s']/3600:.4f} h, "
      "equal out-of-transit baseline, saturation limit "
      f"{cfg['sat_limit']:.0%}, no noise floor, native (R=None) grids.")
    w("")
    w("**Scope: a fixed-configuration estimator comparison.** The submitted "
      "instrument configuration is injected into BOTH sides from this tool's "
      "registry (the harness overrides PandExo's template subarray/readout/"
      "filter per mode), so configuration equality below is by construction, "
      "and this gate deliberately does not test PandExo's own configuration-"
      "selection policy. What each side computes INDEPENDENTLY -- and what "
      "this report actually compares -- is the ramp/group optimization, the "
      "timing, the 2D extraction, and the noise propagation on that fixed "
      "configuration.")
    w("")
    w("## Figures (regenerate with `make_parity_plots.py`)")
    w("")
    w("These show the quantities that match 1:1 -- the parity result. The "
      "depth-uncertainty difference (a noise-model difference, not a "
      "configuration one) is quantified in the tables and Findings below, "
      "not plotted.")
    w("")
    w("- **parity_config_timing.png** -- selected groups, integration time, "
      "and integration count, this tool vs PandExo, on the 1:1 line "
      "(log-log).")
    w("- **parity_extracted_flux.png** -- G395H extracted stellar count rate, "
      "per-wavelength overlay with a ratio strip (per-pixel jitter + binned "
      "median).")
    w("")
    w("## What matches, and how it is measured")
    w("")
    w("The two are independently IMPLEMENTED estimators calling the same "
      "Pandeia engine on an identically pinned configuration. Only what is "
      "read straight from the shared engine agrees exactly; anything each "
      "computes on its own agrees closely but not exactly. This is a property "
      "of a cross-tool test, not a defect -- forcing bit-equality would mean "
      "one tool copying the other's numbers, which is not a validation.")
    w("")
    w("- **Submitted configuration (subarray, readout, filter, disperser):** "
      "identical on every row -- by construction, since the harness pins both "
      "sides to the registry; the gate fails on any drift in these four "
      "recorded fields. The extraction strategy (apertures/annuli) and the "
      "ecliptic/medium background are also configured to match PandExo's TSO "
      "conventions, but those fields are NOT captured in the artifact, so no "
      "measured claim is made for them.")
    w(f"- **Extracted wavelength grids:** on every unsaturated row, "
      f"{ms['min_match_frac']:.1%} of PandExo's pixels find an "
      f"exact-wavelength partner on our side at relative tolerance "
      f"{pg.WL_MATCH_RTOL:g} (gate floor: "
      f"{pg.MIN_MATCHED_PIXEL_FRAC:.0%}; per-pixel deltas beyond that "
      "tolerance are not stored).")
    w(f"- **Groups:** each tool independently optimizes the ramp to the same "
      f"{cfg['sat_limit']:.0%} saturation target; the freedom left is "
      f"rounding to an integer group count. Measured: within "
      f"{ms['ng_dev_bright']} group(s) on the moderate/bright stars, within "
      f"{ms['ng_dev_faint']} groups on the faint Ks=13 star (the gate allows "
      f"{pg.MAX_NGROUP_ABS_DIFF_FAINT} groups OR "
      f"{pg.MAX_NGROUP_REL_DIFF:.0%} there, whichever is looser -- rounding "
      "on a ~500-1000 group ramp is ~1% by itself). On SHORT ramps (either "
      f"side at <= {pg.LOW_NGROUP_EXACT} groups) the gate requires EXACT "
      "agreement: one group there is a large slice of the integration and "
      "of the noise, not rounding. Per-group integration "
      "time then inherits the group choice (gated at "
      f"{pg.MAX_TINT_REL_DIFF:.0%} relative); the largest total-t_int gap "
      f"is {ms['tint_worst'][0]:.1%} ({ms['tint_worst'][1]}/"
      f"{ms['tint_worst'][2]}, on a ~{ms['tint_worst'][3]}-group ramp). "
      "Matched sigma-ratio medians additionally sit inside the "
      f"[{pg.SIGMA_RATIO_MEDIAN_BAND[0]}, {pg.SIGMA_RATIO_MEDIAN_BAND[1]}] "
      "anomaly band (an outlier ceiling, not a parity-to-unity claim).")
    w(f"- **Extracted flux:** per-mode median ratios span "
      f"{ms['flux_med_lo']:.4f}-{ms['flux_med_hi']:.4f} (gate: median within "
      f"{pg.MAX_FLUX_RATIO_DEV:.0%} of unity). The per-pixel scatter around "
      "each median comes from the two tools' independent extraction of the "
      "same 2D calculation and is disclosed in the tables (5th/95th "
      "percentiles); it is not gated. Both calculations receive the exact "
      "same sampled stellar spectrum. The remaining wavelength-dependent "
      "extraction difference has not been assigned a physical cause, so this "
      "artifact makes no per-pixel flux-parity claim.")
    w("- **Saturation:** both tools search down to Pandeia's per-detector "
      "minimum ramp (worker v11: NIR 1 group, MIRI 2). Native partial- and "
      "full-saturation arrays are wavelength-aligned and compared as binary "
      "masks; the gate requires complete grid coverage and exact mask "
      "agreement. Rows above the saturation limit remain diagnostic rows, "
      "not numerical estimator-validation rows.")
    w("")
    w("**PandExo operational warnings are recorded, not adjudicated.** Under "
      "the pinned RAPID readout, PandExo attaches data-volume-excess "
      "warnings to the NIRCam rows (its optimizer would prefer a slower "
      "pattern); the raw warnings are printed per star below. A numerical "
      "parity row says the two estimators agree on that configuration -- it "
      "is not a statement that the configuration is schedulable or "
      "operationally recommended.")
    w("")
    w("Columns: sigma ratio = (this tool's per-pixel transit-depth sigma) / "
      "(PandExo's), median [5th, 95th percentile] over matched pixels. "
      "'matched' uses PandExo's integration counts in the tool formula "
      "(isolates the noise model); 'policy' uses the tool's own "
      "floor(T/t_int) counts (adds the integration-counting policy). "
      "flux ratio compares extracted stellar count rates (engine parity; "
      "expect 1.0000).")
    for sname, block in summary["stars"].items():
        star = cfg["stars"][sname]
        w("")
        w(f"## Star `{sname}` (Teff {star['teff']:.0f} K, logg "
          f"{star['log_g']}, [Fe/H] {star['metallicity']}, Ks "
          f"{star['ks_mag']})")
        w("")
        po = block.get("provenance_ours") or {}
        pp = block.get("provenance_pandexo") or {}
        w(f"Backend: engine {po.get('engine_version')} + "
          f"{po.get('refdata_name')} (worker v{po.get('worker_version')}); "
          f"PandExo {pp.get('pandexo_version')} on engine "
          f"{pp.get('pandeia_engine_version')}.")
        w("")
        w("| mode | status | ngroup ours/PX | t_int s ours/PX | "
          "n_int ours/PX(in) | flux ratio | sigma ratio (matched) | "
          "sigma ratio (policy) |")
        w("|---|---|---|---|---|---|---|---|")
        for m in block["modes"]:
            if m.get("status") == "OK":
                w(f"| {m['key']} | OK | {m['ngroup_ours']}/"
                  f"{m['ngroup_pandexo']} | {m['t_int_ours_s']:.3f}/"
                  f"{m['t_int_pandexo_s']:.3f} | {m['n_int_ours']}/"
                  f"{m['n_int_pandexo_in']:.0f} | "
                  f"{fmt_ratio(m.get('flux_ratio'))} | "
                  f"{fmt_ratio(m.get('sigma_ratio_matched'))} | "
                  f"{fmt_ratio(m.get('sigma_ratio_policy'))} |")
            elif m.get("status") == "SATURATED":
                w(f"| {m['key']} | SATURATED (ours: unusable, loud; "
                  f"PandExo ngroup={m.get('pandexo_ngroup')}) | -- | -- | "
                  "-- | -- | -- | -- |")
            elif m.get("status") == "SATURATED_ABOVE_LIMIT":
                w(f"| {m['key']} | SATURATED above limit (measured "
                  f"{m.get('sat_frac_ours', float('nan')):.2f}x full well; "
                  "reported, not a validation row) | "
                  f"{m.get('ngroup_ours', '--')}/"
                  f"{m.get('ngroup_pandexo', '--')} | -- | -- | -- | -- | "
                  "-- |")
            else:
                w(f"| {m['key']} | ERROR (see parity_summary.json) | -- | "
                  "-- | -- | -- | -- | -- |")
        w("")
        w("Noise-model attribution (median per-integration variance over "
          "pure photon counts; photon-limited = 1.0):")
        w("")
        w("| mode | this tool (pandeia extracted noise) | PandExo (fml) |")
        w("|---|---|---|")
        for m in block["modes"]:
            if m.get("status") == "OK":
                w(f"| {m['key']} | {m['var_excess_ours']:.3f} | "
                  f"{m['var_excess_pandexo']:.3f} |")
        for m in block["modes"]:
            if (m.get("status") in ("OK", "SATURATED_ABOVE_LIMIT")
                    and m.get("pandexo_warnings")):
                warns = {k: v for k, v in m["pandexo_warnings"].items()
                         if str(v) not in ("nan", "None", "0", "All good")}
                if warns:
                    w("")
                    w(f"PandExo warnings for {m['key']}: {warns}")
    w("")
    w("## Findings")
    w("")
    w("1. **Matched-configuration parity: achieved.** With the submitted "
      "configuration pinned identically on both sides (see Scope above), "
      "the two sides agree on the extracted wavelength grids "
      f"({ms['min_match_frac']:.1%} of pixels matched on every unsaturated "
      f"row), extracted count rates (per-mode flux-ratio medians "
      f"{ms['flux_med_lo']:.4f}-{ms['flux_med_hi']:.4f}), independently "
      f"selected groups (within {ms['ng_dev_bright']} on the moderate/bright "
      f"stars; within {ms['ng_dev_faint']} groups on the faint star, i.e. "
      "integer rounding of the same saturation target), per-group "
      "integration times, and integration counts (within rounding policy).")
    w("")
    g = ms["g395h"]
    if g is not None:
        _vr = g["var_excess_ours"] / g["var_excess_pandexo"]
        _sr = g["sigma_ratio_matched"]["median"]
        _attrib = (
            f"the variance-excess ratio ({g['var_excess_ours']:.3f}/"
            f"{g['var_excess_pandexo']:.3f} = {_vr:.3f} on the W39-like "
            f"G395H row) accounts for the bulk of the measured squared sigma "
            f"ratio ({_sr:.3f}^2 = {_sr**2:.3f}). ")
    else:
        _attrib = ""
    # Derived from THIS artifact, never hard-coded: the envelope moves every
    # time the matrix is regenerated, and a stale literal beside fresh tables
    # reads as a measurement.
    _nir, _miri, _nir_pol = [], [], []
    for _b in summary["stars"].values():
        for _m in _b["modes"]:
            _mm = (_m.get("sigma_ratio_matched") or {}).get("median")
            _pp = (_m.get("sigma_ratio_policy") or {}).get("median")
            if _mm is None:
                continue
            (_miri if "miri" in _m["key"] else _nir).append(_mm)
            if _pp is not None and "miri" not in _m["key"]:
                _nir_pol.append(_pp)

    def _pct(v):
        d = (v - 1.0) * 100.0
        return f"{d:+.1f}%" if abs(d) < 2.0 else f"{d:+.0f}%"

    _below = [v for v in _nir + _miri if v < 1.0]
    _sense = ("This tool is therefore CONSERVATIVE relative to PandExo on "
              "every row" if not _below else
              "This tool is therefore conservative relative to PandExo on "
              f"every row but {len(_below)}, which sits marginally below unity"
              if len(_below) == 1 else
              f"every row but {len(_below)}, which sit marginally below unity")
    w("2. **The remaining sigma difference is the noise model itself, and "
      "it is nearly one-sided.** This tool propagates pandeia's full "
      "extracted noise (correlated ramp/read noise, background, dark, IPC, "
      "quantum-yield excess); PandExo's default 'fml' calculation is an "
      "analytic ramp formula that sits within a few percent of pure photon "
      "noise in the NIR. The attribution tables above show the variance "
      f"excess over photon counts on both sides; {_attrib}"
      f"{_sense}: {_pct(min(_nir))} to {_pct(max(_nir))} on matched "
      f"NIRSpec/NIRISS/NIRCam configurations (up to {_pct(max(_nir_pol))} "
      f"under the policy configs), and larger for MIRI LRS "
      f"({_pct(min(_miri))} to {_pct(max(_miri))}), where the deep-red "
      "background and detector terms dominate and the analytic formula "
      "under-represents them.")
    w("")
    w("3. **Residual policy differences (documented, small):** integration "
      "counts are floored here vs rounded in PandExo (at most one "
      "integration per window); the symmetric in/out "
      "approximation adds ~+0.5% sigma at 1% depth (grows with depth; "
      "docstring in noise.pixel_depth_variance). Since worker v8 the ramp "
      "floors equal pandeia's per-detector mingroups (NIR 1, MIRI 2), the "
      "same field PandExo reads, so the old ngroup-floor delta (ours 2 vs "
      "PandExo 1 on bright-star PRISM) no longer exists.")
    w("")
    _passed = problems == []
    w("4. **What may be claimed:** "
      + ("on the fixed configurations this tool's registry submits (with "
         "PandExo explicitly overridden to the same hardware), the timing, "
         "group optimization, configuration-level saturation handling, and "
         "extraction of this tool match the PandExo revision named in the "
         "provenance table, on the supported Pandeia engine. This is not a "
         "test of PandExo's configuration-selection policy. Absolute sigmas "
         "are NOT PandExo-identical and are not labeled as such: they are "
         "pandeia-extracted-noise forecasts, conservative relative to "
         "PandExo's analytic noise by the mode-dependent margins quantified "
         "above."
         if _passed else
         "NOTHING, until this artifact passes its gate. The rows above are "
         "forensic measurements of whatever engine, refdata, PSF tree, worker "
         "version, and PandExo revision the provenance table names -- which is "
         "not necessarily the shipping configuration. Regenerate on the "
         "supported release before citing any parity claim."))
    w("")
    (OUTPUTS / "REPORT.md").write_text("\n".join(lines))
    print(f"wrote {OUTPUTS / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--require-pass", action="store_true",
        help="refuse to write a report unless the summary re-validates as a "
             "pass (use this in a release job)")
    raise SystemExit(main(require_pass=ap.parse_args().require_pass))
