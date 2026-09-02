"""Render the short REPORT.md from parity_summary.json (after run_parity.py).

The report is a SUMMARY, kept deliberately short: gate banner, provenance,
and the measured envelope against each declared threshold. The durable
narrative -- method, scope, and known intended differences -- lives in the
harness docstrings (`run_parity.py` in this directory), and the full
per-mode rows, noise-model attribution, and raw PandExo warnings stay in
`parity_summary.json`, which is committed beside it. Do not grow this
renderer back into a per-mode table dump; the JSON is already the record.

The report is generated from RE-VALIDATED JSON only: it imports the shared
gate (`parity_gate.py`) and re-runs `validate_artifact()` on the summary
instead of trusting the persisted `gate.passed` boolean, so a hand-edited
artifact cannot render as a PASS. Every numerical statement is computed from
the artifact -- nothing is hard-coded, so the text cannot go stale against
the JSON. `--require-pass` refuses to write a report for a failing artifact
at all.

Usage: python validation/parity/scripts/make_report.py [--require-pass]
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


def _sigma_envelope(summary) -> dict:
    """Sigma-ratio envelope, derived from THIS artifact and never hard-coded.

    The envelope moves every time the matrix is regenerated, and a stale
    literal sitting beside a fresh gate banner reads as a measurement."""
    nir, miri, nir_pol = [], [], []
    for b in summary["stars"].values():
        for m in b["modes"]:
            mm = (m.get("sigma_ratio_matched") or {}).get("median")
            pp = (m.get("sigma_ratio_policy") or {}).get("median")
            if mm is None:
                continue
            (miri if "miri" in m["key"] else nir).append(mm)
            if pp is not None and "miri" not in m["key"]:
                nir_pol.append(pp)

    def pct(v):
        d = (v - 1.0) * 100.0
        return f"{d:+.1f}%" if abs(d) < 2.0 else f"{d:+.0f}%"

    def rng(vals):
        return f"{pct(min(vals))} to {pct(max(vals))}" if vals else "--"

    below = [v for v in nir + miri if v < 1.0]
    if not below:
        sense = "This tool is CONSERVATIVE relative to PandExo on every row."
    else:
        sense = (f"This tool is conservative relative to PandExo on every row "
                 f"but {len(below)}, which "
                 + ("sits" if len(below) == 1 else "sit")
                 + " marginally below unity.")
    return {"nir": rng(nir) + (f" (up to {pct(max(nir_pol))} policy)"
                               if nir_pol else ""),
            "miri": rng(miri),
            "sense": sense}


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
    env = _sigma_envelope(summary)
    lines = []
    w = lines.append
    w("# PandExo numerical parity report")
    w("")
    w(f"Generated {date.today().isoformat()} by `run_parity.py` + "
      "`make_report.py` in this directory. This file is a SUMMARY. The "
      "method, scope, and known intended differences live in the harness "
      "docstrings (`../scripts/run_parity.py`). The full per-mode rows, the "
      "noise-model attribution, and the raw PandExo warnings live in "
      "`parity_summary.json` beside this file. Figures live in `../figs/`.")
    w("")
    lines.extend(_gate_header(summary, problems))
    w(f"Configuration: constant transit depth {cfg['depth']}, transit "
      f"duration {cfg['transit_duration_s']/3600:.4f} h, equal "
      f"out-of-transit baseline, saturation limit {cfg['sat_limit']:.0%}, "
      "no noise floor, native (R=None) grids. Both sides run the SAME "
      "Pandeia release named above, so every difference below is an "
      "estimator or policy difference, never an engine calibration "
      "difference.")
    w("")
    w(f"## Measured ({ms['n_ok']} unsaturated rows)")
    w("")
    w("| quantity | measured | gate |")
    w("|---|---|---|")
    w(f"| wavelength-grid pixels matched | {ms['min_match_frac']:.1%} "
      f"(worst row) | >= {pg.MIN_MATCHED_PIXEL_FRAC:.0%} at rtol "
      f"{pg.WL_MATCH_RTOL:g} |")
    w(f"| extracted flux ratio, per-mode medians | "
      f"{ms['flux_med_lo']:.4f}-{ms['flux_med_hi']:.4f} | median within "
      f"{pg.MAX_FLUX_RATIO_DEV:.0%} of unity |")
    w(f"| group agreement, moderate/bright | within {ms['ng_dev_bright']} "
      f"group(s) | exact when either side <= {pg.LOW_NGROUP_EXACT} groups |")
    w(f"| group agreement, faint Ks=13 | within {ms['ng_dev_faint']} "
      f"group(s) | {pg.MAX_NGROUP_ABS_DIFF_FAINT} groups OR "
      f"{pg.MAX_NGROUP_REL_DIFF:.0%}, whichever is looser |")
    w(f"| largest total t_int gap | {ms['tint_worst'][0]:.1%} "
      f"({ms['tint_worst'][1]}/{ms['tint_worst'][2]}, "
      f"~{ms['tint_worst'][3]}-group ramp) | {pg.MAX_TINT_REL_DIFF:.0%} "
      "relative |")
    w(f"| sigma ratio, NIR matched | {env['nir']} | median inside "
      f"[{pg.SIGMA_RATIO_MEDIAN_BAND[0]}, "
      f"{pg.SIGMA_RATIO_MEDIAN_BAND[1]}] |")
    w(f"| sigma ratio, MIRI LRS | {env['miri']} | same band |")
    w("")
    w(env["sense"] + " The residual sigma difference is the noise model "
      "itself, not the configuration (mechanism: notes.md, Parity "
      "testing). Saturation "
      "masks are wavelength-aligned and gated for complete coverage and "
      "exact agreement; rows above the saturation limit are diagnostic "
      "rows, not validation rows.")
    if problems:
        w("")
        w("**Nothing may be claimed from this artifact until it passes its "
          "gate.** The numbers above are forensic measurements of whatever "
          "engine, refdata, PSF tree, worker version, and PandExo revision "
          "the provenance table names, which is not necessarily the "
          "shipping configuration.")
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
