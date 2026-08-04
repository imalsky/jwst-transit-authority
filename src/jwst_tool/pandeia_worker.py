"""Pandeia ETC worker -- runs INSIDE the selected backend's conda env (DEFAULT
"current" = the matched 2026.7 engine/refdata/PSF triple; "archival_2026_2" and
"legacy" = reproducibility only; see instruments._BACKENDS).

Standalone on purpose: no imports from the rest of the tool, stdlib + pandeia only.

    python pandeia_worker.py job.json result.json

job.json:
    {"refdata": <pandeia_refdata path>, "cdbs": <PYSYN_CDBS path>,
     "star": {"teff":.., "log_g":.., "metallicity":.., "ks_mag":..},
     "sat_limit": 0.80,
     "modes": [{"key":.., "instrument":.., "mode":.., "config": {...},
                "strategy": {...}, "ngroup_min":.., "ngroup_max":..}, ...]}

result.json: one entry per mode key, plus a reserved "__provenance__" entry
(engine/refdata/python versions -- the exact backend identity of this run;
written before any mode runs so even an all-failed result is attributable).
Per mode key:
    {"wl": [...um], "flux": [...e-/s], "noise_1int": [...e-/s, sigma for 1 integ],
     "n_part_sat": [...], "n_full_sat": [...],   per-pixel saturated-group counts
     "r_native": [...] | null,      native resolving power R(lambda) on the wl
                                    grid (refdata dispersion file; null when the
                                    disperser has no such file -- host then skips
                                    the LSF blur, safe for high-R modes only),
     "r_native_source": "..",
     "t_cycle_s": ..,                per-integration CYCLE time in a TSO series
                                    (worker v6: the nint=2 minus nint=1
                                    exposure-time difference, which includes
                                    the between-integration reset -- the nint=1
                                    scalar missed it on MIRI FASTR1),
     "ngroup": .., "sat_frac": .., "sat_ngroups": ..,
     "saturated": bool, "engine_version": "..",
     "n_pix_native": ..,             NATIVE-GRID counts (worker v7), taken
     "n_pix_unusable_dropped": ..,   BEFORE the finite/positive `good` filter.
     "n_pix_part_sat_native": ..,    Fully saturated channels are exactly the
     "n_pix_full_sat_native": ..,    ones with non-finite extracted noise, so
                                     they are removed by `good` and the
                                     per-pixel n_full_sat curve below can no
                                     longer see them. These four are the
                                     reporting truth; the arrays are science.
     "warnings": {...}}      -- or {"error": "..."} if that mode failed.

Star normalization (worker_version >= 4): band-integrated synphot photsys
normalization to the 2MASS Ks magnitude in vegamag ("2mass,ks" bandpass from
the minimal CDBS comp/nonhst tree) -- the same convention as the STScI web
ETC. The Vega reference: the 2026+ engine uses its OWN refdata Vega
(sed/hst_calspec/alpha_lyr_stis_010.fits, exactly the web ETC's); the legacy
3.x engine goes through synphot's vega_file, pinned to the local CALSPEC
alpha_lyr_stis_011 copy (required for legacy only; the two Vegas agree to
0.08 mmag in Ks). The old at_lambda shortcut (monochromatic
666.7 Jy zero point AT 2.159 um) mis-scales the flux by ~1-4% depending on
spectral type (CO bandhead for cool stars, Brackett-gamma wing for warm ones;
Cohen 2003 isophotal calibration holds only for A0V shapes), which fed
saturation/ngroup choices at full amplitude.

Group selection: probe at ngroup_min, form two candidates -- the PandExo-style
linear extrapolation of the brightest-pixel full-well fraction, and pandeia's
own scalar sat_ngroups (min groups-to-saturation over extraction pixels) scaled
by sat_limit -- take the smaller, then VERIFY: run the chosen ngroup and step
down until the measured fraction_saturation is actually under sat_limit (the
linear assumption alone can overshoot). If even ngroup_min busts the limit the
mode is flagged saturated (kept, with its degraded ngroup_min numbers, so the
GUI can say WHY it's bad). Channel-level saturation comes from the report's 1d
n_partial_saturated / n_full_saturated curves so the host can exclude or flag
affected pixels instead of trusting one mode-wide boolean.
"""
import copy
import glob
import json
import math
import os
import sys
import traceback

import numpy as np


def _make_calc(build_default_calc, m, star):
    calc = build_default_calc("jwst", m["instrument"], m["mode"])
    for section, kv in (m.get("config") or {}).items():
        calc["configuration"][section].update(kv)
    calc["configuration"]["detector"]["nint"] = 1
    calc["configuration"]["detector"]["nexp"] = 1
    for k, v in (m.get("strategy") or {}).items():
        calc["strategy"][k] = v
    # sky background model (pandeia's default is "minzodi"/"benchmark"); the
    # registry pins PandExo's TSO convention ("ecliptic" at "medium"). BOTH
    # keys are required together -- the engine resolves them to one canned
    # file (<background>_<level>.fits) and fails on a partial pair.
    if m.get("background"):
        calc["background"] = m["background"]
        if not m.get("background_level"):
            raise ValueError(
                f"mode {m['key']}: background={m['background']!r} without "
                "background_level -- the engine needs both (e.g. 'medium')")
        calc["background_level"] = m["background_level"]
    calc["scene"][0]["spectrum"]["sed"] = {
        "sed_type": "phoenix", "teff": float(star["teff"]),
        "log_g": float(star["log_g"]), "metallicity": float(star["metallicity"])}
    # band-integrated 2MASS Ks normalization in vegamag (web-ETC convention);
    # see module docstring for why the at_lambda shortcut was retired
    calc["scene"][0]["spectrum"]["normalization"] = {
        "type": "photsys", "bandpass": "2mass,ks",
        "norm_flux": float(star["ks_mag"]), "norm_fluxunit": "vegamag"}
    return calc


def _native_r(refdata, m, wl):
    """Native resolving power R(lambda) interpolated onto the extracted grid,
    from the mode's refdata dispersion file. Returns (list|None, source str).

    Disperser token: config disperser where present; MIRI LRS is the p750l
    prism; SOSS picks the order-specific file. Missing file -> (None, note):
    the host then applies no LSF blur, which is only safe for high-R modes --
    every low-R mode used here (PRISM, LRS, SOSS) has a dispersion file in
    pandeia_data (verified for 3.0rc3).
    """
    disp = (m.get("config", {}).get("instrument", {}) or {}).get("disperser")
    if m["instrument"] == "miri" and m["mode"] == "lrsslitless":
        disp = "p750l"
    if m["instrument"] == "niriss" and m["mode"] == "soss":
        order = int((m.get("strategy") or {}).get("order", 1))
        disp = f"gr700xd-ord{order}"
    if not disp:
        return None, "no disperser token for this mode"
    pat = os.path.join(refdata, "jwst", m["instrument"], "dispersion",
                       f"*{disp}*disp*.fits")
    hits = sorted(glob.glob(pat))
    if not hits:
        return None, f"no dispersion file matching {pat}"
    from astropy.io import fits
    with fits.open(hits[0]) as h:
        cols = {c.upper(): c for c in h[1].columns.names}
        if "R" not in cols or "WAVELENGTH" not in cols:
            return None, f"{os.path.basename(hits[0])} lacks WAVELENGTH/R columns"
        w = np.asarray(h[1].data[cols["WAVELENGTH"]], float)
        r = np.asarray(h[1].data[cols["R"]], float)
    order_ix = np.argsort(w)
    r_i = np.interp(np.asarray(wl, float), w[order_ix], r[order_ix])
    return r_i.tolist(), os.path.basename(hits[0])


def _clamp_ngroup(ng, ng_min, ng_max):
    """Clamp a candidate group count to the mode's supported [ng_min, ng_max].

    ng_max is the PandExo-compatible hard maximum for the mode (NIRCam grism =
    100; see instruments.PANDEXO_NGROUP_MAX). This is the ONE place the group
    optimizer's output is bounded, so a selected ramp can never exceed the
    supported range on a faint target (2026-07-12 audit item 5)."""
    return max(int(ng_min), min(int(ng_max), int(ng)))


def _run(perform_calculation, calc, ngroup, nint=1):
    c = copy.deepcopy(calc)
    c["configuration"]["detector"]["ngroup"] = int(ngroup)
    c["configuration"]["detector"]["nint"] = int(nint)
    return perform_calculation(c)


def _sat_curve(rpt, key, n_pix):
    """Per-pixel saturated-group counts from the 1d report.

    LOAD-BEARING (worker v6): detect.evaluate_mode drops fully saturated
    pixels from the measurement operator via these curves, so a missing or
    renamed report key must FAIL here -- the old silent all-zeros fallback
    would have quietly stopped excluding saturated pixels after an engine
    rename (STScI does rename report keys between releases)."""
    try:
        wave, curve = rpt["1d"][key]
    except Exception as e:
        raise RuntimeError(
            f"pandeia report has no 1d {key!r} curve (available: "
            f"{sorted(rpt.get('1d', {}))}). The engine may have renamed it; "
            "the saturation mask cannot be built without it, and an unmasked "
            "saturated pixel is a silently wrong measurement.") from e
    curve = np.asarray(curve, dtype=float)
    if curve.shape[0] != n_pix:
        raise RuntimeError(
            f"pandeia 1d {key!r} curve has {curve.shape[0]} samples but the "
            f"extracted grid has {n_pix}: the saturation mask would be "
            "misaligned with the pixels it must protect.")
    return np.nan_to_num(curve, nan=0.0)


def _one_mode(build_default_calc, perform_calculation, m, star, sat_limit,
              refdata):
    calc = _make_calc(build_default_calc, m, star)
    ng_min, ng_max = int(m["ngroup_min"]), int(m["ngroup_max"])

    probe = _run(perform_calculation, calc, ng_min)
    sat_probe = float(probe["scalar"]["fraction_saturation"])
    sat_ng = probe["scalar"].get("sat_ngroups")

    # candidate group counts: linear extrapolation of the probe's full-well
    # fraction, and pandeia's own groups-to-saturation estimate; take the
    # smaller (more conservative), then verify empirically below.
    cands = []
    if sat_probe > 0:
        cands.append(int(math.floor(ng_min * sat_limit / sat_probe)))
    if sat_ng is not None and np.isfinite(sat_ng) and sat_ng > 0:
        cands.append(int(math.floor(sat_limit * float(sat_ng))))
    ng_best = min(cands) if cands else ng_max
    saturated = ng_best < ng_min      # even the shortest ramp busts the limit
    ng_best = _clamp_ngroup(ng_best, ng_min, ng_max)

    rpt = probe if ng_best == ng_min else _run(perform_calculation, calc, ng_best)
    # verify the CHOSEN ramp: the linear full-well model can overshoot, so step
    # down until the measured saturation fraction actually respects the limit
    for _ in range(4):
        frac = float(rpt["scalar"]["fraction_saturation"])
        if frac <= sat_limit or ng_best <= ng_min:
            break
        ng_best = max(ng_min, min(ng_best - 1,
                                  int(math.floor(ng_best * sat_limit / frac))))
        rpt = _run(perform_calculation, calc, ng_best)
    if float(rpt["scalar"]["fraction_saturation"]) > sat_limit:
        saturated = True

    wl, _sn = rpt["1d"]["sn"]
    flux = np.asarray(rpt["1d"]["extracted_flux"][1], dtype=float)
    noise = np.asarray(rpt["1d"]["extracted_noise"][1], dtype=float)
    wl = np.asarray(wl, dtype=float)
    good = np.isfinite(wl) & np.isfinite(flux) & np.isfinite(noise) & (flux > 0) & (noise > 0)

    # FULL-GRID saturation aggregates, computed BEFORE `good` is applied
    # (worker_version >= 7). Fully saturated channels are exactly the ones whose
    # extracted noise pandeia returns non-finite, so `good` removes them and the
    # `n_*_sat[good]` curves below can no longer see them -- a column saturated
    # everywhere used to export a full-saturation count of ZERO. The science
    # estimator is unchanged (it must keep excluding unusable pixels); these are
    # reporting-only counts on the native grid, and they are what display/export
    # must quote.
    n_part = _sat_curve(rpt, "n_partial_saturated", wl.shape[0])
    n_full = _sat_curve(rpt, "n_full_saturated", wl.shape[0])
    native_counts = {
        "n_pix_native": int(wl.shape[0]),
        "n_pix_unusable_dropped": int((~good).sum()),
        "n_pix_part_sat_native": int((n_part > 0).sum()),
        "n_pix_full_sat_native": int((n_full > 0).sum()),
    }

    if not good.any():
        # pandeia returned no usable pixels (e.g. NIRSpec PRISM on a bright star:
        # saturation within the shortest ramp NaNs the extracted noise everywhere)
        return {
            "unusable": True,
            "reason": (f"no unsaturated pixels at the shortest ramp "
                       f"(ngroup={ng_best}, max full-well fraction "
                       f"{float(rpt['scalar']['fraction_saturation']):.1f}) -- "
                       "target too bright for this mode"),
            "ngroup": int(ng_best),
            "sat_frac": float(rpt["scalar"]["fraction_saturation"]),
            "saturated": True,
            **native_counts,
            "warnings": {k: str(v) for k, v in rpt["warnings"].items()},
        }

    r_native, r_src = _native_r(refdata, m, wl[good])

    # Per-integration CYCLE time (worker v6): the nint=1 total_exposure_time
    # is NOT the marginal cost of one integration in a TSO series when the
    # first-reset and between-integration-reset counts differ. MIRI FASTR1
    # has nreset1=0, nreset2=1 (2026.2 refdata), so the nint=1 scalar was one
    # tframe SHORT per integration -- integrations were overcounted by
    # (ngroup+1)/ngroup and sigma was optimistic by up to ~9% at ngroup_min
    # (the committed parity run shows the exact one-frame gap vs PandExo at
    # identical ngroup). All four NIR rapid patterns have nreset1=nreset2=1,
    # where the two agree exactly. The engine-truth marginal cost is the
    # nint=2 minus nint=1 exposure-time difference -- one extra cheap call,
    # no refdata timing-ladder reimplementation to drift.
    t_exp_1 = float(rpt["scalar"]["total_exposure_time"])
    rpt2 = _run(perform_calculation, calc, ng_best, nint=2)
    t_cycle = float(rpt2["scalar"]["total_exposure_time"]) - t_exp_1
    if not (np.isfinite(t_cycle) and t_cycle > 0.0):
        raise RuntimeError(
            f"per-integration cycle time from the nint=2/nint=1 exposure "
            f"difference is not positive/finite ({t_cycle!r}; nint=1 total "
            f"{t_exp_1!r}): the engine's timing report changed shape and "
            "the integration count cannot be trusted.")

    import pandeia.engine
    return {
        "wl": wl[good].tolist(),
        "flux": flux[good].tolist(),
        "noise_1int": noise[good].tolist(),
        "n_part_sat": n_part[good].tolist(),
        "n_full_sat": n_full[good].tolist(),
        "r_native": r_native,
        "r_native_source": r_src,
        "t_cycle_s": t_cycle,
        "ngroup": int(ng_best),
        "sat_frac": float(rpt["scalar"]["fraction_saturation"]),
        "sat_ngroups": (float(sat_ng) if sat_ng is not None
                        and np.isfinite(sat_ng) else None),
        "saturated": bool(saturated),
        **native_counts,
        "engine_version": str(getattr(pandeia.engine, "__version__", "unknown")),
        "warnings": {k: str(v) for k, v in rpt["warnings"].items()},
    }


def _release(version):
    """Leading dotted-numeric release segment of a version string
    ("3.0rc3" -> "3.0", "2026.2" -> "2026.2"); None if it has none."""
    import re
    m = re.match(r"(\d+(?:\.\d+)*)", str(version).strip())
    return m.group(1) if m else None


def _refdata_version(refdata):
    """Best-available refdata version: the VERSION file (some pandeia_data
    releases), VERSION_DATA (the 2026+ split data trees), else VERSION_PSF's
    first line (3.0-era trees have only that), else the pandeia_data-<ver>
    directory name. Returns (version|None, source)."""
    for name in ("VERSION", "VERSION_DATA", "VERSION_PSF"):
        p = os.path.join(refdata, name)
        if os.path.isfile(p):
            with open(p) as f:
                first = f.readline().strip()
            if first:
                return first, name
    base = os.path.basename(os.path.normpath(refdata))
    if base.startswith("pandeia_data-"):
        return base[len("pandeia_data-"):], "directory name"
    return None, "no VERSION/VERSION_PSF file or pandeia_data-<ver> dir name"


def _psf_version(psf_dir):
    """PSF-library release from VERSION_PSF, else the pandeia_psfs-<ver> dir
    name. Returns (version|None, source)."""
    p = os.path.join(psf_dir, "VERSION_PSF")
    if os.path.isfile(p):
        with open(p) as f:
            first = f.readline().strip()
        if first:
            return first, "VERSION_PSF"
    base = os.path.basename(os.path.normpath(psf_dir))
    if base.startswith("pandeia_psfs-"):
        return base[len("pandeia_psfs-"):], "directory name"
    return None, "no VERSION_PSF file or pandeia_psfs-<ver> dir name"


def _check_backend_match(engine_version, refdata, psf_dir=None):
    """Enforce STScI's matching rule across the FULL TRIPLE before any
    calculation: engine release == refdata release == PSF-library release.

    A mismatched engine/refdata pair fails deep inside the engine, or worse
    runs with wrong calibrations. The PSF library is the same hazard and was
    previously only checked for EXISTENCE (`VERSION_PSF` present), so a 2026.2
    PSF tree could serve a 2026.7 engine with no version error at all, and the
    PSF release never reached provenance. `psf_dir=None`/"" means this backend
    carries its own PSFs inside the refdata tree (the 3.0-era layout), which is
    the only case where there is nothing separate to match.

    Returns the provenance fields; raises RuntimeError naming the offending
    component on any mismatch or undeterminable version.
    """
    ref_ver, source = _refdata_version(refdata)
    if ref_ver is None:
        raise RuntimeError(
            f"cannot determine the pandeia_data version of {refdata} ({source}); "
            "refusing to run against unidentifiable reference data. Point "
            "JWST_TOOL_PANDEIA_REFDATA at an intact pandeia_data tree.")
    eng_rel, ref_rel = _release(engine_version), _release(ref_ver)
    if eng_rel is None or ref_rel is None or eng_rel != ref_rel:
        raise RuntimeError(
            f"pandeia.engine {engine_version} does not match pandeia_data "
            f"{ref_ver} (from {source}) at {refdata}. STScI requires the "
            "engine, refdata, and PSF releases to be the same. Fix the set: "
            "point JWST_TOOL_PANDEIA_PYTHON at an env whose engine matches "
            "JWST_TOOL_PANDEIA_REFDATA (the supported triple is engine 2026.7 "
            "+ pandeia_data-2026.7-jwst + pandeia_psfs-2026.7-jwst).")

    prov = {"refdata_version": ref_ver, "refdata_version_source": source}
    if not psf_dir:
        prov["psf_version"] = None
        prov["psf_version_source"] = "backend carries PSFs inside refdata"
        return prov

    psf_ver, psf_source = _psf_version(psf_dir)
    if psf_ver is None:
        raise RuntimeError(
            f"cannot determine the pandeia_psfs version of {psf_dir} "
            f"({psf_source}); refusing to run against an unidentifiable PSF "
            "library. Point JWST_TOOL_PANDEIA_PSF_DIR at an intact "
            "pandeia_psfs tree.")
    psf_rel = _release(psf_ver)
    if psf_rel is None or psf_rel != eng_rel:
        raise RuntimeError(
            f"PSF library release {psf_ver} (from {psf_source}) at {psf_dir} "
            f"does not match pandeia.engine {engine_version} / pandeia_data "
            f"{ref_ver}. All three must be the same release -- a mixed PSF "
            "tree changes the point-spread function and therefore the "
            "extracted flux and noise, silently. Point "
            "JWST_TOOL_PANDEIA_PSF_DIR at the pandeia_psfs tree for release "
            f"{eng_rel}.")
    prov["psf_version"] = psf_ver
    prov["psf_version_source"] = psf_source
    return prov


def _preflight(job):
    """Fail with the offending PATH, not a deep synphot traceback, when the
    reference trees are missing (e.g. a stale job/cache from an old layout)."""
    problems = []
    if not os.path.isdir(job["refdata"]):
        problems.append(f"pandeia_refdata does not exist: {job['refdata']}")
    psf_dir = job.get("psf_dir")
    if psf_dir:
        # split-layout (pandeia_data >= 2026) PSF library; must be an intact
        # tree, not just an existing directory
        if not os.path.isfile(os.path.join(psf_dir, "VERSION_PSF")):
            problems.append(
                f"PSF_DIR has no VERSION_PSF file: {psf_dir} (point "
                "JWST_TOOL_PANDEIA_PSF_DIR at the extracted pandeia_psfs tree)")
    cdbs = job["cdbs"]
    if not os.path.isdir(cdbs):
        problems.append(f"PYSYN_CDBS does not exist: {cdbs}")
    else:
        phx = os.path.join(cdbs, "grid", "phoenix")
        if not os.path.isdir(os.path.realpath(phx)):
            problems.append(
                f"PHOENIX grid missing or dangling symlink: {phx} "
                f"-> {os.path.realpath(phx)} (the star SED cannot be built)")
        # NOTE (worker v6): the local CALSPEC Vega file is required by the
        # LEGACY 3.x engine only -- checked in main() once the engine version
        # is known. The 2026+ engine loads its OWN refdata Vega
        # (sed/hst_calspec/alpha_lyr_stis_010.fits) and never reads the pin.
        rel = os.path.join("comp", "nonhst", "2mass_ks_001_syn.fits")
        if not os.path.isfile(os.path.join(cdbs, rel)):
            problems.append(f"missing {rel} in PYSYN_CDBS -- the 2MASS Ks "
                            "bandpass (photsys normalization); fetch it from "
                            "https://ssb.stsci.edu/trds/")
    if problems:
        raise RuntimeError(
            "pandeia worker preflight failed:\n  " + "\n  ".join(problems)
            + "\n(check JWST_TOOL_PANDEIA_REFDATA / JWST_TOOL_DATA_DIR, or "
            "regenerate a stale cached job)")


def main():
    job = json.load(open(sys.argv[1]))
    _preflight(job)
    os.environ["pandeia_refdata"] = job["refdata"]
    os.environ["PYSYN_CDBS"] = job["cdbs"]
    if job.get("psf_dir"):
        os.environ["PSF_DIR"] = job["psf_dir"]
    import warnings as _w
    _w.filterwarnings("ignore")
    # synphot's default vega_file is an ssb.stsci.edu URL: point it at the
    # local CALSPEC copy so the LEGACY 3.x engine's vegamag normalization
    # (synphot from_vega, which honors this pin) works offline. The 2026+
    # engine never reads it -- it loads its OWN refdata Vega at import
    # (sed/hst_calspec/alpha_lyr_stis_010.fits; band-integrated Ks ratio to
    # the local 011 file 0.99993, 0.08 mmag), which is why the pin is set
    # when available but only REQUIRED for a 3.x engine (checked below,
    # once the engine version is known).
    _vega = os.path.join(job["cdbs"], "calspec", "alpha_lyr_stis_011.fits")
    if os.path.isfile(_vega):
        import synphot
        synphot.conf.vega_file = _vega
    from pandeia.engine.calc_utils import build_default_calc
    from pandeia.engine.perform_calculation import perform_calculation

    import pandeia.engine
    engine_version = str(getattr(pandeia.engine, "__version__", "unknown"))
    if (_release(engine_version) or "").startswith("3.") \
            and not os.path.isfile(_vega):
        raise RuntimeError(
            f"missing {_vega} -- the legacy 3.x engine normalizes vegamag "
            "through synphot's vega_file and needs the local CALSPEC copy "
            "to run offline; fetch it from https://ssb.stsci.edu/trds/")
    match = _check_backend_match(engine_version, job["refdata"],
                                 job.get("psf_dir"))
    _psf_dir = job.get("psf_dir") or ""
    out = {
        "__provenance__": {
            "engine_version": engine_version,
            "refdata_path": job["refdata"],
            "refdata_name": os.path.basename(os.path.normpath(job["refdata"])),
            "psf_path": _psf_dir,
            "psf_name": (os.path.basename(os.path.normpath(_psf_dir))
                         if _psf_dir else ""),
            **match,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "worker_version": job.get("worker_version"),
        },
    }
    print(f"[pandeia] engine {engine_version} + {out['__provenance__']['refdata_name']} "
          f"(refdata {match['refdata_version']} from {match['refdata_version_source']}"
          + (f"; PSFs {match['psf_version']} from {match['psf_version_source']}"
             if match.get("psf_version") else "; PSFs inside refdata")
          + ")", flush=True)
    for m in job["modes"]:
        key = m["key"]
        print(f"[pandeia] {key} ...", flush=True)
        try:
            out[key] = _one_mode(build_default_calc, perform_calculation,
                                 m, job["star"], float(job.get("sat_limit", 0.8)),
                                 job["refdata"])
            if out[key].get("unusable"):
                print(f"[pandeia] {key}: UNUSABLE ({out[key]['reason']})", flush=True)
            else:
                print(f"[pandeia] {key}: ngroup={out[key]['ngroup']} "
                      f"sat={out[key]['sat_frac']:.2f} npix={len(out[key]['wl'])}",
                      flush=True)
        except Exception:
            out[key] = {"error": traceback.format_exc()}
            print(f"[pandeia] {key}: FAILED", flush=True)

    with open(sys.argv[2], "w") as f:
        json.dump(out, f)
    print("[pandeia] done", flush=True)


if __name__ == "__main__":
    main()
