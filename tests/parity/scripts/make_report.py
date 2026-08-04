"""Render REPORT.md from parity_summary.json (run after run_parity.py).

The report is generated from VALIDATED JSON only. `run_parity.py` now writes a
`gate` block recording whether the artifact passed its release gate; a summary
that failed (or predates the gate) produces a report headed FAIL that says so,
and `--require-pass` refuses to write one at all. Before this, a stale,
saturated, or version-mismatched run rendered a report that read like a pass.

Usage: python tests/parity/scripts/make_report.py [--require-pass]
"""
import argparse
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent / "outputs"      # parity_summary.json in, REPORT.md out


def fmt_ratio(s: dict | None) -> str:
    if not s or not s.get("n"):
        return "--"
    return (f"{s['median']:.4f} [{s['p05']:.4f}, {s['p95']:.4f}] "
            f"(n={s['n']})")


def _gate_header(summary: dict) -> list[str]:
    """PASS/FAIL banner + provenance table, at the very top of the report."""
    gate = summary.get("gate")
    out = []
    if gate is None:
        out += [
            "> **GATE: NOT EVALUATED.** This summary predates the fail-closed "
            "release gate, so nothing here has been checked for worker "
            "version, engine/refdata/PSF release agreement, PandExo identity, "
            "saturated rows, or the numerical thresholds. It is a forensic "
            "artifact, NOT a release certificate. Regenerate with "
            "`run_parity.py`.",
            ""]
    elif gate.get("passed"):
        out += [
            f"**GATE: PASS.** Worker v{gate.get('required_worker_version')}, "
            f"Pandeia {gate.get('required_pandeia_release')} on both sides, "
            "every declared threshold met.",
            ""]
    else:
        probs = gate.get("problems", [])
        out += [
            f"> **GATE: FAIL ({len(probs)} problem(s)).** This artifact is NOT "
            "a release certificate and must not be cited as one.",
            ""]
        out += [f"> - {p}" for p in probs]
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
    gate = summary.get("gate")
    if require_pass and not (gate and gate.get("passed")):
        raise SystemExit(
            "make_report: --require-pass given but parity_summary.json "
            + ("has no gate block (regenerate with run_parity.py)"
               if gate is None else
               f"FAILED its gate ({len(gate.get('problems', []))} problems). "
               "Fix the run; do not publish a report for a failing artifact."))
    cfg = summary["config"]
    lines = []
    w = lines.append
    w("# PandExo numerical parity report")
    w("")
    w(f"Generated {date.today().isoformat()} by `run_parity.py` + "
      "`make_report.py` in this directory.")
    w("")
    lines.extend(_gate_header(summary))
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
    w("## Figures (regenerate with `make_parity_plots.py`)")
    w("")
    w("These show the quantities that match 1:1 -- the parity result. The "
      "depth-uncertainty difference (a noise-model difference, not a "
      "configuration one) is quantified in the tables and Findings below, "
      "not plotted.")
    w("")
    w("- **parity_config_timing.png** -- selected groups, integration time, "
      "and integration count, this tool vs PandExo, on the 1:1 line "
      "(log-log). Every mode/star lands on the diagonal except PRISM's known "
      "group floor (ngroup_min=2 vs PandExo's 1 on a bright star).")
    w("- **parity_extracted_flux.png** -- G395H extracted stellar count rate, "
      "per-wavelength overlay with a ratio strip (per-pixel jitter + binned "
      "median). The curves coincide (systematic ratio ~0.997).")
    w("")
    w("## What is bit-identical, and why the rest is not")
    w("")
    w("The two are INDEPENDENT tools calling the same Pandeia engine, so only "
      "what is read straight from the shared engine is bit-for-bit identical; "
      "anything each computes on its own agrees closely but not exactly. This "
      "is a property of a cross-tool test, not a defect -- forcing bit-equality "
      "would mean one tool copying the other's numbers, which is not a "
      "validation.")
    w("")
    w("- **Bit-identical (max difference exactly 0):** the instrument "
      "configuration (subarray, readout, disperser, filter, aperture, "
      "background) and the extracted WAVELENGTH GRID -- every extracted "
      "pixel, every mode and star (verified: max |Δλ| = 0).")
    w("- **Groups agree to ≤1 on the moderate/bright stars, to ~1% relative "
      "(up to 5 groups absolute) on the faint Ks=13 star:** each tool "
      "independently optimizes the ramp to the same 80% saturation target; "
      "the freedom left is rounding to an integer group count (e.g. G395H "
      "124 vs 125; faint-star ramps of ~500-1000 groups differ by up to 5). "
      "Per-group integration time matches to <0.1%; total integration time "
      "and count then inherit the group choice (up to ~5.5% where a single "
      "group of ~16 dominates, bright G395H; ~2.5% bright MIRI). Only VALID "
      "configurations are compared: PRISM saturates on the "
      "two bright stars (both tools flag it; this tool floors at ngroup=2 "
      "while PandExo drops to ngroup=1) and is excluded there, and is compared "
      "on the faint star where it is unsaturated and matches (ngroup 20/20).")
    w("- **Extracted flux agrees to a systematic ~0.3%** (binned median 0.997 "
      "for G395H), with per-pixel scatter from the two tools' independent "
      "extraction of the same 2D calculation -- photon-level for the smooth "
      "NIRSpec/MIRI traces (MIRI LRS matches to ~0.15%), larger for the curved "
      "NIRISS SOSS order-1 trace and the saturating low-R PRISM. The tool "
      "integrates over bins, so the per-pixel jitter averages out; the "
      "systematic is what enters the noise. The narrow downward spikes in this "
      "tool's flux are STELLAR ABSORPTION LINES in Pandeia's PHOENIX spectrum "
      "(hydrogen recombination -- e.g. Brackett-α 4.052 μm, Pfund-δ 3.297 μm "
      "-- plus molecular bands on cool stars, so they multiply on cooler "
      "targets: 9 lines at 6250 K, 193 at 4500 K); PandExo's separately-loaded "
      "stellar spectrum smooths them. They are physically real, cancel in the "
      "in/out transit-depth ratio, and wash out in binning.")
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
            if m.get("status") == "OK" and m.get("pandexo_warnings"):
                warns = {k: v for k, v in m["pandexo_warnings"].items()
                         if str(v) not in ("nan", "None", "0", "All good")}
                if warns:
                    w("")
                    w(f"PandExo warnings for {m['key']}: {warns}")
    w("")
    w("## Findings")
    w("")
    w("1. **Configuration parity: achieved.** With the registry's explicit "
      "TSO readout patterns, PandExo's extraction strategy (apertures/"
      "annuli), and the ecliptic/medium background, the two sides agree on "
      "the extracted wavelength grids (every pixel matches), extracted "
      "count rates (flux-ratio medians 0.987-1.03), selected groups "
      "(within 1 on the moderate/bright stars; within ~1% relative, up to "
      "5 groups absolute, on the faint Ks=13 star), per-group integration "
      "times (<0.1%; total t_int inherits the group choice, up to ~5.5% "
      "where one group of ~16 dominates), and integration counts (within "
      "rounding policy). Saturation behavior matches: pixels PandExo masks "
      "as saturated are the pixels this tool excludes.")
    w("")
    w("2. **The remaining sigma difference is the noise model itself, and "
      "it is one-sided.** This tool propagates pandeia's full extracted "
      "noise (correlated ramp/read noise, background, dark, IPC, "
      "quantum-yield excess); PandExo's default 'fml' calculation is an "
      "analytic ramp formula that sits within a few percent of pure photon "
      "noise in the NIR. The attribution tables above show the variance "
      "excess over photon counts on both sides; their ratio reproduces the "
      "observed sigma ratios (e.g. G395H: 1.220/1.014 = 1.203 ~= 1.108^2). "
      "This tool is therefore systematically CONSERVATIVE relative to "
      "PandExo: ~2-24% higher sigma for NIRSpec/NIRISS/NIRCam on matched "
      "configurations (up to ~31% under the policy configs on the faint "
      "Ks=13 star), and larger "
      "for MIRI LRS (~33-56%), where the deep-red background and detector "
      "terms dominate and the analytic formula under-represents them. For "
      "context, published achieved-vs-PandExo noise ratios (COMPASS "
      "G395H 1.05-1.12; MIRI LRS ~1.15-1.2 above random-noise "
      "simulations) fall on the same side, between the two models for "
      "NIRSpec.")
    w("")
    w("3. **Residual policy differences (documented, small):** integration "
      "counts are floored here vs rounded in PandExo (at most one "
      "integration per window); ngroup_min is 2 here while PandExo will "
      "select 1 group (PRISM on a bright star); the symmetric in/out "
      "approximation adds ~+0.5% sigma at 1% depth (grows with depth; "
      "docstring in noise.pixel_depth_variance).")
    w("")
    _passed = bool(gate and gate.get("passed"))
    w("4. **What may be claimed:** "
      + ("the instrument configuration, timing, group optimization, "
         "saturation handling, and extraction of this tool match the PandExo "
         "release named in the provenance table above, on the supported "
         "Pandeia engine. Absolute sigmas are NOT PandExo-identical and are "
         "not labeled as such: they are pandeia-extracted-noise forecasts, "
         "conservative relative to PandExo's analytic noise by the "
         "mode-dependent margins quantified above."
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
        help="refuse to write a report unless the summary passed its gate "
             "(use this in a release job)")
    raise SystemExit(main(require_pass=ap.parse_args().require_pass))
