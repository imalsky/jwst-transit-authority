"""Run PandExo's unmodified instrument templates for the parity star matrix.

The fixed-hardware parity gate answers whether the estimators agree. This
experiment answers the separate policy question: what configuration does the
pinned PandExo revision choose when this tool does not override its template?
It records differences and operational warnings without calling either choice
"recommended" or silently treating a warned row as valid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_parity as fixed  # noqa: E402

OUTPUTS = HERE.parent / "outputs"


def _compact_config(config: dict) -> dict:
    detector = config.get("detector", {})
    instrument = config.get("instrument", {})
    return {
        "subarray": detector.get("subarray"),
        "readout": detector.get("readout_pattern"),
        "ngroup": detector.get("ngroup"),
        "mode": instrument.get("mode"),
        "filter": instrument.get("filter"),
        "disperser": instrument.get("disperser"),
    }


def _tool_config(key: str) -> dict:
    return _compact_config(fixed.ins.MODES[key]["config"])


def _row(key: str, result: dict) -> dict:
    if isinstance(result.get("error"), str):
        return {"key": key, "status": "ERROR", "valid": False,
                "error": result["error"][-500:]}
    # PandExo returns a complete status dictionary, including benign entries
    # whose value is literally "All good" or "All good (<measured limit>)".
    # Keep only messages that require attention. In particular, do not drop
    # "All good. Ngroups=1 ... Proceed with caution.", which is a warning
    # despite its prefix.
    warnings = {
        str(k): str(v)
        for k, v in result.get("warnings", {}).items()
        if str(v).strip() != "All good"
        and not (str(v).strip().startswith("All good (")
                 and str(v).strip().endswith(")"))
    }
    selected = _compact_config(result["config"])
    expected = _tool_config(key)
    compare_fields = ("subarray", "readout", "filter", "disperser")
    differences = {
        name: {"tool": expected[name], "pandexo_default": selected[name]}
        for name in compare_fields if expected[name] != selected[name]
    }
    full = sum(float(x) > 0 for x in result["n_full_saturated"])
    partial = sum(float(x) > 0 for x in result["n_partial_saturated"])
    status = "WARNING" if warnings else "EXECUTED"
    return {
        "key": key,
        "status": status,
        # A row carrying an operational warning is never labeled valid.
        "valid": not warnings,
        "pandexo_default": selected,
        "tool_fixed": expected,
        "differences": differences,
        "config_matches": not differences,
        "n_partial_saturated": partial,
        "n_full_saturated": full,
        "warnings": warnings,
    }


def _problems(summary: dict) -> list[str]:
    problems = []
    for star in fixed.STARS:
        rec = summary["stars"].get(star)
        if rec is None:
            problems.append(f"missing star {star}")
            continue
        prov = rec.get("provenance", {})
        if fixed.pg._release_of(prov.get("pandeia_engine_version")) \
                != fixed.pg.REQUIRED_PANDEIA_RELEASE:
            problems.append(f"{star}: wrong Pandeia engine release")
        if prov.get("pandexo_commit") != fixed.pg.REQUIRED_PANDEXO_COMMIT:
            problems.append(f"{star}: wrong or missing PandExo commit")
        rows = {row["key"]: row for row in rec.get("modes", [])}
        for key in fixed.pg.MODE_KEYS:
            if key not in rows:
                problems.append(f"{star}/{key}: missing row")
            elif rows[key]["status"] == "ERROR":
                problems.append(f"{star}/{key}: PandExo failed")
    return problems


def _config_text(config: dict) -> str:
    return "/".join(str(config.get(key) or "-")
                    for key in ("subarray", "readout", "filter", "disperser"))


def _write_report(summary: dict) -> None:
    lines = [
        "# PandExo unmodified-template policy report",
        "",
        "This is a configuration-policy experiment, not the fixed-hardware "
        "estimator parity gate. `valid = false` means the executed PandExo row "
        "carried an operational warning; it is never presented as an "
        "unqualified valid configuration.",
        "",
        "| star | mode | status | config match | PandExo default | tool fixed | warnings |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for star, rec in summary["stars"].items():
        for row in rec["modes"]:
            warnings = "; ".join(row.get("warnings", {}).values()).replace("|", "/")
            lines.append(
                f"| {star} | {row['key']} | {row['status']} | "
                f"{'yes' if row.get('config_matches') else 'no'} | "
                f"{_config_text(row.get('pandexo_default', {}))} | "
                f"{_config_text(row.get('tool_fixed', {}))} | "
                f"{warnings or '-'} |")
    lines += ["", f"Completeness gate: **{summary['gate']['status']}**.", ""]
    lines += [f"- {problem}" for problem in summary["gate"]["problems"]]
    (OUTPUTS / "POLICY_REPORT.md").write_text("\n".join(lines))


def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "pandexo_unmodified_templates",
        "stars": {},
        "config": {"stars": fixed.STARS,
                   "mode_keys": list(fixed.pg.MODE_KEYS)},
    }
    for star_name, star in fixed.STARS.items():
        # Use the same worker and pinned environment as fixed parity, but no
        # instrument-template override at all.
        original = fixed.PANDEXO_MODES
        try:
            fixed.PANDEXO_MODES = {
                key: (original[key][0], {}) for key in fixed.pg.MODE_KEYS
            }
            result = fixed.run_pandexo(
                star, list(fixed.pg.MODE_KEYS), OUTPUTS,
                tag=f"{star_name}_policy_")
        finally:
            fixed.PANDEXO_MODES = original
        summary["stars"][star_name] = {
            "provenance": fixed._scrub_paths(result.get("__provenance__") or {}),
            "modes": [_row(key, result.get(key, {"error": "missing"}))
                      for key in fixed.pg.MODE_KEYS],
        }
        (OUTPUTS / "policy_summary.json").write_text(json.dumps(summary, indent=1))
    problems = _problems(summary)
    summary["gate"] = {"status": "PASS" if not problems else "FAIL",
                       "problems": problems}
    (OUTPUTS / "policy_summary.json").write_text(json.dumps(summary, indent=1))
    _write_report(summary)
    print(f"policy summary -> {OUTPUTS / 'policy_summary.json'}")
    print(f"policy report  -> {OUTPUTS / 'POLICY_REPORT.md'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
