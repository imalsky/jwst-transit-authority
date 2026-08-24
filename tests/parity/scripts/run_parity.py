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

from jwst_tool import binning, instruments as ins   # noqa: E402
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
    # NIRCam and MIRI must be pinned in FULL, never partially: PandExo
    # template defaults move (MIRI's is slitlessprism_ip; NIRCam's readout is
    # template policy, "optimize"), so only explicit pins keep the submitted
    # hardware fixed on both sides:
    #   * NIRCam: 'rapid' and 'bright1' are BOTH valid for lw_tsgrism and the
    #     engine declares NO default. SUBGRISM64 + RAPID is a flight-capable
    #     grism-TSO choice (this tool's registry), not the unique flown one;
    #     under RAPID PandExo reports a data-volume excess warning (~28 GB
    #     against a 15 GB advisory) on every star tested. It is recorded per
    #     row in parity_summary.json with the rest of PandExo's raw warnings,
    #     which is where REPORT.md points; the gate does not adjudicate it,
    #     and the GUI says only that it checks no program limits.
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
    "nirspec_g140h": ("NIRSpec G140H",
                      {"detector": {"subarray": "sub2048",
                                    "readout_pattern": "nrsrapid",
                                    "readmode": "nrsrapid"}}),
    "nirspec_g235m": ("NIRSpec G235M",
                      {"detector": {"subarray": "sub2048",
                                    "readout_pattern": "nrsrapid",
                                    "readmode": "nrsrapid"}}),
    # PandExo has no order-2 SOSS template: reuse the SOSS template and pin
    # the extraction order through the strategy override channel (top-level
    # "strategy" section, applied to inst["strategy"] in pandexo_worker).
    "niriss_soss_ord2": ("NIRISS SOSS",
                         {"detector": {"subarray": "substrip256",
                                       "readout_pattern": "nisrapid",
                                       "readmode": "nisrapid"},
                          "strategy": {"order": 2}}),
    # PandExo has no F277W template: reuse the F322W2 grism template with the
    # filter pinned, the same way both NIRCam entries already pin theirs.
    "nircam_f277w": ("NIRCam F322W2",
                     {"instrument": {"filter": "f277w"},
                      "detector": {"subarray": "subgrism64",
                                   "readout_pattern": "rapid",
                                   "readmode": "rapid"}}),
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

    # Saturation is classified from the MEASURED saturation fraction (of
    # Pandeia's per-mode saturation level, NOT of the physical full well) as
    # well as the worker's `unusable` flag. A configuration can return usable
    # pixels while sitting above the saturation limit -- the committed 2026-07
    # artifact carried two such rows labeled OK, one at 7.31x that level.
    # Those numbers are still reported; they just cannot count as a
    # validation row.
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
        # like-for-like electron rates: PandExo's remove_QY divided the
        # detector quantum yield out of e_rate_out (photon convention for its
        # shot-noise formula); multiply the recorded curve back so both sides
        # are pandeia's extracted electron rate. Gated to unity in the gate.
        flux_ratio=_stats(flux_o[io] / (np.asarray(px["e_rate_out"])[ip]
                                        * np.asarray(px["qy_on_grid"])[ip])),
        # disclosed only: the photon-convention ratio (== the QY curve when
        # the electron rates agree) so the artifact keeps the divergence
        # between the two conventions visible
        flux_ratio_photon=_stats(
            flux_o[io] / np.asarray(px["e_rate_out"])[ip]),
        sigma_ratio_matched=_stats(sigma_ours(n_p_in, n_p_out) / err_p[ip]),
        sigma_ratio_policy=_stats(sigma_ours(n_ours, n_ours) / err_p[ip]),
        var_excess_ours=excess_ours, var_excess_pandexo=excess_px,
        pandexo_warnings=px.get("warnings"),
    )
    return out


# --- LSF impulse response (`--impulse`) ------------------------------------
# Inject narrow emission lines into the shared stellar SED, run the tool's own
# Pandeia path twice (continuum, continuum+lines), and compare the extracted
# per-pixel response with binning.smooth_to_native_r applied to the same lines
# on the same wl / r_native grid. Constant-depth parity cannot exercise the
# LSF operator (a flat spectrum is its fixed point); this does.
IMPULSE_STAR = "w39_like"
IMPULSE_LINES_UM = (0.75, 1.1, 1.5, 2.0, 2.6, 3.1, 3.6, 4.1, 4.6, 5.5, 7.5, 10.5)
IMPULSE_SIGMA_REL = 5e-5      # line sigma / lambda: unresolved by every mode
IMPULSE_AMP = 0.5             # line peak / continuum


def _impulse_lines(w):
    rel = np.zeros_like(w)
    for lam in IMPULSE_LINES_UM:
        rel += IMPULSE_AMP * np.exp(-0.5 * ((w - lam) / (IMPULSE_SIGMA_REL * lam)) ** 2)
    return rel


def _impulse_sed(shared: dict):
    """The SED handed to Pandeia. Pandeia convolves the spectrum with its own
    lambda/R Gaussian on a fine internal grid and then np.interp's the result
    back onto the INPUT grid (instrument.spectral_convolution), so the input
    must resolve the instrument-broadened line, not just the injected one: a
    grid dense only near the line core turns the broadened wings into linear
    ramps to the next coarse point and adds 25-150% spurious line flux on
    every R >~ 500 mode. Tiers around each line:
    sigma/4 within +-8 sigma, lambda/2e4 within +-0.5%, lambda/5e3 within
    +-2% (>= 10 lambda/R for R >= 500); the shared continuum grid
    (R ~ 250-500) resolves the R ~ 100 modes by itself."""
    w0 = np.asarray(shared["wave_um"], float)
    f0 = np.asarray(shared["flux_mjy"], float)
    fine = [w0]
    for lam in IMPULSE_LINES_UM:
        sig = IMPULSE_SIGMA_REL * lam
        fine.append(lam + sig * np.arange(-8.0, 8.01, 0.25))
        fine.append(lam * np.exp(np.arange(-5e-3, 5.01e-3, 5e-5)))
        fine.append(lam * np.exp(np.arange(-2e-2, 2.01e-2, 2e-4)))
    w = np.unique(np.concatenate(fine))
    # the tiers coincide to within an ulp at multiples of 2e-4; synphot
    # refuses exact duplicates once Pandeia has folded in its midpoints
    w = w[np.concatenate([[True], np.diff(np.log(w)) > 1e-8])]
    cont = np.exp(np.interp(np.log(w), np.log(w0), np.log(f0)))
    return w, cont, _impulse_lines(w)


def _impulse_mode(key: str, cont_res: dict, line_res: dict, shared: dict) -> dict:
    from scipy.optimize import minimize_scalar
    wl = np.asarray(cont_res["wl"], float)
    if not np.array_equal(wl, np.asarray(line_res["wl"], float)):
        raise SystemExit(f"{key}: continuum and line runs returned different pixel grids")
    po = np.argsort(wl)
    wl_s, r_s = wl[po], np.asarray(cont_res["r_native"], float)[po]
    obs = (np.asarray(line_res["flux"], float) / np.asarray(cont_res["flux"], float) - 1.0)[po]
    m = ins.MODES[key]
    lo, hi = max(m["wl_min"], wl_s.min()), min(m["wl_max"], wl_s.max())
    c_lo, c_hi = binning._pixel_cells(wl_s)
    # The prediction is evaluated on a uniform R = 1e5 grid, NOT the SED
    # grid: that one is coarse away from the lines, and a blurred line
    # sampled there loses its wings.
    w0, f0 = np.asarray(shared["wave_um"], float), np.asarray(shared["flux_mjy"], float)
    w = np.exp(np.arange(np.log(lo * 0.95), np.log(hi * 1.05), 1e-5))
    cont, rel, dw = np.exp(np.interp(np.log(w), np.log(w0), np.log(f0))), _impulse_lines(w), np.diff(w)

    def predict(r_curve, b_lo=lo * 0.97, b_hi=hi * 1.03):
        fine = binning.smooth_to_native_r(w, rel, wl_s, r_curve, b_lo, b_hi, weight=cont)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (fine[1:] + fine[:-1]) * dw)])
        return (np.interp(c_hi, w, cum) - np.interp(c_lo, w, cum)) / (c_hi - c_lo)

    def metrics(p, win, b100):
        o, q = obs[win], p[win]
        return dict(area_ratio=float(np.sum(o * (c_hi - c_lo)[win]) / np.sum(q * (c_hi - c_lo)[win])),
                    peak_ratio=float(o.max() / q.max()),
                    max_resid_over_peak=float(np.max(np.abs(o - q)) / q.max()),
                    r100_bin_ratio=float(obs[b100].mean() / p[b100].mean()))

    # `pred`: the Gaussian at the refdata R (the calibration input);
    # `pred_eff`: what detect.evaluate_mode applies, R / instruments.LSF_WIDTH.
    r_eff = ins.lsf_r(key, wl_s, r_s)
    pred, pred_eff = predict(r_s), predict(r_eff)
    lines = {}
    for lam in IMPULSE_LINES_UM:
        r_at = float(np.interp(lam, wl_s, r_s)); fw = lam / r_at
        if not (lo + 5 * fw < lam < hi - 5 * fw):
            continue
        win = np.abs(wl_s - lam) < 6.0 * fw
        b100 = np.abs(np.log(wl_s / lam)) < 0.5 / 100.0
        # width_fit: the single-Gaussian FWHM scale (x lambda/R_refdata) that
        # best fits the extracted line's SHAPE on its pixels, amplitude free
        # (solved linearly at each width, so a throughput or order-overlap
        # amplitude loss cannot masquerade as width) -- the number LSF_WIDTH
        # stores. A half-maximum width read off 2-3 pixels is not usable.
        fit = np.abs(wl_s - lam) < 12.0 * fw

        def shape_resid(s):
            p = predict(r_s / s, lam * 0.97, lam * 1.03)[fit]
            return float(np.sum((obs[fit] - p * np.dot(obs[fit], p) / np.dot(p, p)) ** 2))
        s_fit = minimize_scalar(shape_resid, bounds=(0.3, 5.0), method="bounded").x
        lines[f"{lam:g}"] = dict(
            r_native=r_at, n_pixels=int(win.sum()), width_fit=float(s_fit),
            **metrics(pred, win, b100),
            applied=dict(width=float(r_at / np.interp(lam, wl_s, r_eff)),
                         **metrics(pred_eff, win, b100)))
    return lines


def main_impulse():
    shared = json.loads((OUTPUTS / f"{IMPULSE_STAR}_pandexo.json").read_text())["__shared_star__"]
    w, cont, rel = _impulse_sed(shared)
    keys = list(PANDEXO_MODES)
    star = STARS[IMPULSE_STAR]
    print(f"=== impulse: {IMPULSE_STAR}, continuum ===", flush=True)
    cont_res = run_ours(star, keys, {"wave_um": w.tolist(), "flux_mjy": cont.tolist()})
    print(f"=== impulse: {IMPULSE_STAR}, continuum + lines ===", flush=True)
    line_res = run_ours(star, keys, {"wave_um": w.tolist(), "flux_mjy": (cont * (1.0 + rel)).tolist()})
    out = {"star": IMPULSE_STAR, "sigma_rel": IMPULSE_SIGMA_REL, "amp": IMPULSE_AMP,
           "provenance_ours": _scrub_paths(line_res.get("__provenance__") or {}), "modes": {}}
    for k in keys:
        if "error" in cont_res.get(k, {}) or "error" in line_res.get(k, {}):
            out["modes"][k] = {"error": (cont_res.get(k) or line_res.get(k)).get("error")}
            continue
        out["modes"][k] = _impulse_mode(k, cont_res[k], line_res[k], shared)
        for lam, d in out["modes"][k].items():
            a = d["applied"]
            print(f"  {k:16s} {lam:>5s} um  R={d['r_native']:6.0f}  peak {d['peak_ratio']:.3f}  "
                  f"maxres/peak {d['max_resid_over_peak']:.3f}  R100 {d['r100_bin_ratio']:.3f}  "
                  f"width_fit {d['width_fit']:.2f} | applied x{a['width']:.2f}: "
                  f"peak {a['peak_ratio']:.3f}  maxres/peak {a['max_resid_over_peak']:.3f}  "
                  f"R100 {a['r100_bin_ratio']:.3f}", flush=True)
    summary = json.loads((OUTPUTS / "parity_summary.json").read_text())
    summary["lsf_impulse"] = out
    (OUTPUTS / "parity_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"lsf_impulse -> {OUTPUTS / 'parity_summary.json'}")


def main():
    if "--impulse" in sys.argv:
        return main_impulse()
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

    # FAIL CLOSED: writing the summary and returning None (exit 0) would let
    # a stale, saturated, or version-mismatched artifact look like a passing
    # release gate.
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
