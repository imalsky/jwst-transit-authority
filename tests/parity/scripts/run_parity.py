"""PandExo numerical parity harness (2026-07-12 external audit, release gate).

Runs MATCHED star/instrument configurations through BOTH noise paths on the
SAME Pandeia backend and compares them mode by mode:

  * this package's worker (src/jwst_tool/pandeia_worker.py) + its box-transit
    depth-error propagation (noise.pixel_depth_variance), and
  * current PandExo (master; pandexo_worker.py in this directory).

Running both sides on one engine/refdata generation isolates ESTIMATOR
differences (timing policy, in/out propagation, saturation handling) from
engine-calibration differences -- the point of the audit's parity gate.
This is a FIXED-CONFIGURATION estimator gate: the submitted instrument
configuration is pinned identically on both sides (PANDEXO_MODES below
overrides PandExo's templates to this tool's registry), so it deliberately
does NOT test PandExo's own configuration-selection policy. The harness
points the worker at the backend under test explicitly via environment
variables. The gate (parity_gate.validate) refuses the run unless BOTH
sides are on the same SUPPORTED engine/refdata/PSF triple.

Required environment (all loud, no defaults -- machine paths stay out of git;
main() scrubs absolute paths from the provenance before writing the summary):
  JWST_TOOL_PANDEIA_PYTHON   python of a conda env with pandeia.engine 2026.7
                             AND pandexo (master) installed
  JWST_TOOL_PANDEIA_REFDATA  extracted pandeia_data-2026.7-jwst tree
  JWST_TOOL_PANDEIA_PSF_DIR  extracted pandeia_psfs-2026.7-jwst tree
  JWST_TOOL_DATA_DIR         directory whose cdbs/ holds the phoenix grid,
                             calspec Vega, and comp/nonhst bandpasses
  JWST_TOOL_OUTPUT_DIR       the tool's own worker noise cache (model_cache /
                             noise_cache); unrelated to the parity artifacts

Usage: python tests/parity/run_parity.py
Everything parity lives in THIS directory (tests/parity/): the raw per-run
JSON (git-ignored, see .gitignore) is written here alongside the committed
artifacts (parity_summary.json, REPORT.md, the figures). The tool's Pandeia
noise cache still goes under JWST_TOOL_OUTPUT_DIR (that is the app's cache,
not a parity output).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent        # tests/parity/scripts
OUTPUTS = HERE.parent / "outputs"             # raw JSON + parity_summary.json
REPO = HERE.parents[2]                         # scripts -> parity -> tests -> repo
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

for var in ("JWST_TOOL_PANDEIA_PYTHON", "JWST_TOOL_PANDEIA_REFDATA",
            "JWST_TOOL_PANDEIA_PSF_DIR", "JWST_TOOL_DATA_DIR",
            "JWST_TOOL_OUTPUT_DIR"):
    if not os.environ.get(var):
        raise SystemExit(f"run_parity: {var} must be set (see module docstring)")

from jwst_tool import instruments as ins            # noqa: E402
from jwst_tool import noise                          # noqa: E402
import parity_gate as pg                             # noqa: E402

# The declared experiment (stars, run constants, thresholds, validate) lives
# in parity_gate.py -- one import-safe module shared with the unit tests and
# the renderers.
T_TRANSIT_S = pg.T_TRANSIT_S
DEPTH = pg.DEPTH
SAT_LIMIT = pg.SAT_LIMIT
STARS = pg.STARS

# PandExo template name + the overrides that pin BOTH sides to one
# configuration (subarray / readout / filter), per mode key. PandExo's SOSS
# template defaults to substrip96 and its MIRI template to the FAST pattern;
# real TSO programs (and this tool) use substrip256 / FASTR1.
PANDEXO_MODES = {
    "nirspec_prism": ("NIRSpec Prism",
                      {"detector": {"subarray": "sub512",
                                    "readout_pattern": "nrsrapid",
                                    "readmode": "nrsrapid"}}),
    "nirspec_g395h": ("NIRSpec G395H",
                      {"detector": {"subarray": "sub2048",
                                    "readout_pattern": "nrsrapid",
                                    "readmode": "nrsrapid"}}),
    "nirspec_g235h": ("NIRSpec G235H",
                      {"detector": {"subarray": "sub2048",
                                    "readout_pattern": "nrsrapid",
                                    "readmode": "nrsrapid"}}),
    "nirspec_g395m": ("NIRSpec G395M",
                      {"detector": {"subarray": "sub2048",
                                    "readout_pattern": "nrsrapid",
                                    "readmode": "nrsrapid"}}),
    "niriss_soss": ("NIRISS SOSS",
                    {"detector": {"subarray": "substrip256",
                                  "readout_pattern": "nisrapid",
                                  "readmode": "nisrapid"}}),
    # NIRCam and MIRI were pinned only PARTIALLY until 2026-08-04 (NIRCam got
    # a filter and no detector block, MIRI got a readout and no subarray).
    # The parent 2026.2 artifact shows both sides still EXECUTED identical
    # hardware (subgrism64/rapid, slitlessprism) -- PandExo's choices matched
    # at that PandExo revision -- so the old artifact was a matched-config
    # comparison in fact, just not by construction. Current PandExo defaults
    # moved (MIRI's template default is now slitlessprism_ip; NIRCam's
    # readout is template policy, "optimize"), so the pins are now explicit
    # to keep the submitted hardware fixed on both sides:
    #   * NIRCam: 'rapid' and 'bright1' are BOTH valid for lw_tsgrism and the
    #     engine declares NO default. SUBGRISM64 + RAPID is a flight-capable
    #     grism-TSO choice (this tool's registry), not the unique flown one;
    #     under RAPID PandExo reports a data-volume excess warning, recorded
    #     per row and surfaced in REPORT.md, not adjudicated by this gate.
    #   * MIRI: 'slitlessprism' is 72 x 416 with tframe 0.15904 s (this
    #     tool's registry choice). PandExo's current default
    #     'slitlessprism_ip' is a cropped 68 x 384 variant; all three
    #     slitless subarrays are real modes.
    # Comparing configuration POLICY (what each tool would choose on its own)
    # is a separate, unbuilt harness; this one holds hardware fixed.
    "nircam_f322w2": ("NIRCam F322W2",
                      {"instrument": {"filter": "f322w2"},
                       "detector": {"subarray": "subgrism64",
                                    "readout_pattern": "rapid",
                                    "readmode": "rapid"}}),
    "nircam_f444w": ("NIRCam F444W",
                     {"instrument": {"filter": "f444w"},
                      "detector": {"subarray": "subgrism64",
                                   "readout_pattern": "rapid",
                                   "readmode": "rapid"}}),
    "miri_lrs": ("MIRI LRS",
                 {"detector": {"subarray": "slitlessprism",
                               "readout_pattern": "fastr1",
                               "readmode": "fastr1"}}),
}
assert set(PANDEXO_MODES) == set(pg.MODE_KEYS), (
    "PANDEXO_MODES does not match the declared experiment in parity_gate.py")


def run_ours(star: dict, keys: list[str], star_spectrum: dict) -> dict:
    # noise_job resolves any engine-generation mode renames via
    # instruments.engine_mode(), so parity exercises the SAME production path
    # as a normal run -- never a parity-only rename (that was the bug: a
    # parity-only patch let NIRCam pass the gate while the production path
    # silently sent a rejected token).
    job = noise.noise_job(star, keys, sat_limit=SAT_LIMIT)
    job["star_spectrum"] = star_spectrum
    eng = noise.backend_fingerprint()["engine_version"]
    if pg._release_of(eng) != pg.REQUIRED_PANDEIA_RELEASE:
        raise SystemExit(
            f"run_parity: JWST_TOOL_PANDEIA_PYTHON resolves engine {eng!r}; "
            f"the release gate requires the supported "
            f"{pg.REQUIRED_PANDEIA_RELEASE} engine. Point it at a matching "
            "environment (see the module docstring), or run the harness "
            "knowing the gate will fail.")
    return noise.run_pandeia(job, progress=lambda s: print("  " + s, flush=True))


def run_pandexo(star: dict, keys: list[str], workdir: Path,
                tag: str = "") -> dict:
    job = {
        "refdata": os.environ["JWST_TOOL_PANDEIA_REFDATA"],
        "psf_dir": os.environ["JWST_TOOL_PANDEIA_PSF_DIR"],
        "cdbs": ins.PYSYN_CDBS,
        "vega_file": str(Path(ins.PYSYN_CDBS) / "calspec"
                         / "alpha_lyr_stis_011.fits"),
        "star": {"teff": star["teff"], "logg": star["log_g"],
                 "metal": star["metallicity"], "kmag": star["ks_mag"]},
        "transit_duration_s": T_TRANSIT_S,
        "sat_level_pct": SAT_LIMIT * 100.0,
        "depth": DEPTH,
        "modes": [{"key": k, "pandexo_name": PANDEXO_MODES[k][0],
                   "config_overrides": PANDEXO_MODES[k][1]} for k in keys],
    }
    jf = workdir / f"{tag}pandexo_job.json"
    rf = workdir / f"{tag}pandexo_result.json"
    if rf.exists() and os.environ.get("PARITY_REUSE_PANDEXO") == "1":
        prev = json.loads(rf.read_text())
        prev_job = json.loads(jf.read_text()) if jf.exists() else None
        if prev_job == job:
            print(f"  [pandexo] REUSING {rf} (PARITY_REUSE_PANDEXO=1, "
                  "identical job)", flush=True)
            return prev
        print("  [pandexo] job changed; re-running despite "
              "PARITY_REUSE_PANDEXO=1", flush=True)
    jf.write_text(json.dumps(job))
    r = subprocess.run([ins.PANDEIA_PYTHON, str(HERE / "pandexo_worker.py"),
                        str(jf), str(rf)], text=True, capture_output=True)
    print(r.stdout)
    if r.returncode != 0 or not rf.exists():
        raise SystemExit(f"pandexo worker failed (rc={r.returncode}):\n"
                         f"{r.stderr[-3000:]}")
    return json.loads(rf.read_text())


def _stats(ratio: np.ndarray) -> dict:
    r = ratio[np.isfinite(ratio)]
    if r.size == 0:
        return dict(n=0)
    return dict(n=int(r.size), median=float(np.median(r)),
                p05=float(np.percentile(r, 5)),
                p95=float(np.percentile(r, 95)),
                max_abs_dev=float(np.max(np.abs(r - 1.0))))


def _saturation_mask_stats(ours: dict, px: dict) -> dict:
    """Pixel-aligned binary saturation-mask comparison on the native grids.

    Curves are compared as masks (count > 0), because the scientific decision
    is whether a channel is partially or fully saturated.  The raw count
    arrays remain in the per-run JSON for diagnosis.
    """
    required_ours = ("wl_native", "n_part_sat_native_curve",
                     "n_full_sat_native_curve")
    required_px = ("sat_wave", "n_partial_saturated", "n_full_saturated")
    missing = [f"ours.{k}" for k in required_ours if k not in ours]
    missing += [f"pandexo.{k}" for k in required_px if k not in px]
    if missing:
        return {"error": "missing native saturation arrays: " + ", ".join(missing)}

    wo = np.asarray(ours["wl_native"], float)
    wp = np.asarray(px["sat_wave"], float)
    po = np.asarray(ours["n_part_sat_native_curve"], float)
    fo = np.asarray(ours["n_full_sat_native_curve"], float)
    pp = np.asarray(px["n_partial_saturated"], float)
    fp = np.asarray(px["n_full_saturated"], float)
    if po.shape != wo.shape or fo.shape != wo.shape \
            or pp.shape != wp.shape or fp.shape != wp.shape:
        return {"error": ("unaligned saturation arrays: "
                           f"ours wave/partial/full={wo.shape}/{po.shape}/{fo.shape}, "
                           f"pandexo={wp.shape}/{pp.shape}/{fp.shape}")}
    order = np.argsort(wo)
    ws = wo[order]
    ii = np.searchsorted(ws, wp)
    ii = np.clip(ii, 0, max(ws.size - 1, 0))
    exact = (np.abs(ws[ii] - wp) <
             1e-9 * np.maximum(np.abs(wp), 1e-9)) if ws.size else np.zeros(wp.size, bool)
    io = order[ii[exact]] if ws.size else np.zeros(0, int)
    ip = np.flatnonzero(exact)
    part_equal = (po[io] > 0) == (pp[ip] > 0)
    full_equal = (fo[io] > 0) == (fp[ip] > 0)
    return {
        "n_ours": int(wo.size), "n_pandexo": int(wp.size),
        "n_matched": int(ip.size),
        "matched_frac": float(ip.size / wp.size) if wp.size else 0.0,
        "partial_equal_frac": float(part_equal.mean()) if ip.size else 0.0,
        "full_equal_frac": float(full_equal.mean()) if ip.size else 0.0,
        "partial_disagree": int((~part_equal).sum()),
        "full_disagree": int((~full_equal).sum()),
        "partial_flagged_ours": int((po > 0).sum()),
        "partial_flagged_pandexo": int((pp > 0).sum()),
        "full_flagged_ours": int((fo > 0).sum()),
        "full_flagged_pandexo": int((fp > 0).sum()),
    }


def _scrub_paths(d: dict) -> dict:
    """Machine-absolute paths stay out of the committed artifact: keep only
    the basename (release/name identity is preserved by the version fields)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _scrub_paths(v)
        elif isinstance(v, str) and os.path.isabs(v):
            out[k] = os.path.basename(os.path.normpath(v))
        else:
            out[k] = v
    return out


def compare_mode(key: str, ours: dict, px: dict) -> dict:
    out = {"key": key}
    # a FAILED side carries a traceback STRING under "error"; a successful
    # PandExo mode also has an "error" key -- its sigma ARRAY -- so the
    # failure test must be on the type, not key presence
    if isinstance(ours.get("error"), str) or isinstance(px.get("error"), str):
        out["status"] = "ERROR"
        out["ours_error"] = str(ours.get("error", ""))[-400:]
        out["pandexo_error"] = str(px.get("error", ""))[-400:]
        return out
    out["saturation_mask"] = _saturation_mask_stats(ours, px)
    if ours.get("unusable"):
        out["status"] = "SATURATED"
        out["ours_reason"] = ours["reason"]
        # the gate needs the MEASURED evidence on saturated rows too: our
        # probe fraction + both ngroups + PandExo's full-well verdict, so a
        # false "unusable" cannot hide behind the status label
        out["ngroup_ours"] = int(ours["ngroup"])
        out["sat_frac_ours"] = float(ours["sat_frac"])
        out["ngroup_pandexo"] = px.get("ngroup")
        out["pandexo_ngroup"] = px.get("ngroup")   # legacy alias
        out["pandexo_warnings"] = px.get("warnings")
        return out

    wl_o = np.asarray(ours["wl"])
    flux_o = np.asarray(ours["flux"])
    noise_o = np.asarray(ours["noise_1int"])
    o = np.argsort(wl_o)             # MIRI LRS disperses red-to-blue; sort so
    wl_o, flux_o, noise_o = wl_o[o], flux_o[o], noise_o[o]  # searchsorted works
    wl_p = np.asarray(px["wave"])
    err_p = np.asarray(px["error"])
    tim = px["timing"]

    # grid identity: match pixels by wavelength (the two sides run the same
    # engine, so the extraction grids should agree exactly; PandExo NaNs
    # fully saturated pixels, ours drops non-finite ones)
    ok_p = np.isfinite(err_p) & (err_p > 0)
    ii = np.searchsorted(wl_o, wl_p[ok_p])
    ii = np.clip(ii, 0, wl_o.size - 1)
    exact = np.abs(wl_o[ii] - wl_p[ok_p]) < 1e-9 * np.maximum(wl_p[ok_p], 1e-9)
    io, ip = ii[exact], np.where(ok_p)[0][exact]

    # integration counts: our floor policy vs PandExo's timing
    t_cyc = float(ours["t_cycle_s"])
    n_ours = int(T_TRANSIT_S / t_cyc)
    n_p_in = float(tim["Num Integrations In Transit"])
    n_p_out = float(tim["Num Integrations Out of Transit"])

    def sigma_ours(n_in, n_out):
        return (noise_o[io] / flux_o[io]) * np.sqrt(1.0 / n_in + 1.0 / n_out)

    # noise-model attribution: per-integration variance over pure photon
    # counts (photon-limited == 1.0). Ours uses pandeia's full extracted
    # noise; PandExo's default "fml" formula is analytic ramp noise.
    tm = float(tim["Measurement Time per Integration (sec)"])
    excess_ours = float(np.median(noise_o[io] ** 2 * tm / flux_o[io]))
    e_out = np.asarray(px["electrons_out"])[ip]
    v_out = np.asarray(px["var_out"])[ip]
    excess_px = float(np.median(v_out[e_out > 0] / e_out[e_out > 0])) \
        if (e_out > 0).any() else float("nan")

    # Saturation is classified from the MEASURED full-well fraction as well as
    # the worker's `unusable` flag. A configuration can return usable pixels
    # while sitting above the saturation limit -- the committed 2026-07
    # artifact carried two such rows labeled OK, one at 7.31x full well. Those
    # numbers are still reported; they just cannot count as a validation row.
    _sat_frac = float(ours["sat_frac"])
    _status = "OK" if _sat_frac <= SAT_LIMIT else "SATURATED_ABOVE_LIMIT"
    out.update(
        status=_status,
        npix_ours=int(wl_o.size), npix_pandexo=int(wl_p.size),
        npix_matched=int(io.size),
        ngroup_ours=int(ours["ngroup"]), ngroup_pandexo=int(px["ngroup"]),
        sat_frac_ours=float(ours["sat_frac"]),
        t_int_ours_s=t_cyc,
        t_int_pandexo_s=float(tim["Time/Integration incl reset (sec)"]),
        t_frame_pandexo_s=float(tim["Seconds per Frame"]),
        n_int_ours=n_ours, n_int_pandexo_in=n_p_in, n_int_pandexo_out=n_p_out,
        config_ours={
            "subarray": ins.MODES[key]["config"].get(
                "detector", {}).get("subarray"),
            "readout": ins.MODES[key]["config"].get(
                "detector", {}).get("readout_pattern"),
            "filter": ins.MODES[key]["config"].get(
                "instrument", {}).get("filter"),
            "disperser": ins.MODES[key]["config"].get(
                "instrument", {}).get("disperser")},
        config_pandexo={
            "subarray": px["config"]["detector"].get("subarray"),
            "readout": px["config"]["detector"].get("readout_pattern"),
            "mode": px["config"]["instrument"].get("mode"),
            "filter": px["config"]["instrument"].get("filter"),
            "disperser": px["config"]["instrument"].get("disperser")},
        flux_ratio=_stats(flux_o[io] / np.asarray(px["e_rate_out"])[ip]),
        sigma_ratio_matched=_stats(sigma_ours(n_p_in, n_p_out) / err_p[ip]),
        sigma_ratio_policy=_stats(sigma_ours(n_ours, n_ours) / err_p[ip]),
        var_excess_ours=excess_ours, var_excess_pandexo=excess_px,
        pandexo_warnings=px.get("warnings"),
    )
    return out


def main():
    # raw per-run JSON goes in tests/parity/outputs/ (git-ignored there,
    # alongside the committed parity_summary.json and REPORT.md)
    out_root = OUTPUTS
    out_root.mkdir(parents=True, exist_ok=True)
    keys = list(PANDEXO_MODES)
    summary = {"stars": {}, "config": dict(
        transit_duration_s=T_TRANSIT_S, depth=DEPTH, sat_limit=SAT_LIMIT,
        stars=STARS)}
    for sname, star in STARS.items():
        print(f"=== {sname}: PandExo ===", flush=True)
        px = run_pandexo(star, keys, out_root, tag=f"{sname}_")
        (out_root / f"{sname}_pandexo.json").write_text(json.dumps(px))
        shared_star = px.get("__shared_star__")
        if not isinstance(shared_star, dict):
            raise SystemExit(
                f"pandexo worker did not return the shared stellar spectrum "
                f"for {sname}; estimator parity requires identical source input")
        print(f"=== {sname}: jwst_tool worker ===", flush=True)
        ours = run_ours(star, keys, shared_star)
        (out_root / f"{sname}_ours.json").write_text(json.dumps(ours))
        rows = [compare_mode(k, ours.get(k, {"error": "missing"}),
                             px.get(k, {"error": "missing"})) for k in keys]
        summary["stars"][sname] = {
            "provenance_ours": _scrub_paths(ours.get("__provenance__") or {}),
            "provenance_pandexo": _scrub_paths(px.get("__provenance__") or {}),
            "modes": rows,
        }
        (OUTPUTS / "parity_summary.json").write_text(
            json.dumps(summary, indent=1))
        print(f"=== {sname}: done ===", flush=True)

    # FAIL CLOSED. Previously this wrote the summary and returned None (exit
    # 0), so a stale, saturated, or version-mismatched artifact still looked
    # like a passing release gate.
    problems = pg.validate(summary)
    summary["gate"] = pg.gate_block(problems)
    (OUTPUTS / "parity_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"summary -> {OUTPUTS / 'parity_summary.json'}")

    if problems:
        print(f"\nPARITY GATE: FAIL ({len(problems)} problem(s))",
              file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("\nDo NOT commit this artifact as a passing gate. Fix the cause; "
              "if a threshold is genuinely wrong, change it in parity_gate.py "
              "AND the report with a reason, not silently.", file=sys.stderr)
        return 1
    print("\nPARITY GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
