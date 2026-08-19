"""Pandeia ETC worker -- runs INSIDE the selected backend's conda env
(see instruments._BACKENDS). Standalone on purpose: no imports from the rest
of the tool, stdlib + pandeia only.

    python pandeia_worker.py job.json result.json

job.json:
    {"refdata": <pandeia_refdata path>, "cdbs": <PYSYN_CDBS path>,
     "star": {"teff":.., "log_g":.., "metallicity":.., "ks_mag":..},
     "sat_limit": 0.80,
     "modes": [{"key":.., "instrument":.., "mode":.., "config": {...},
                "strategy": {...}, "ngroup_min":.., "ngroup_max":..}, ...]}

result.json: one entry per mode key, plus a reserved "__provenance__" entry
(the exact backend identity, written before any mode runs). Per mode key:
    {"wl": [...um], "flux": [...e-/s], "noise_1int": [...e-/s, sigma for 1 integ],
     "n_part_sat": [...], "n_full_sat": [...],
                                    NOT group counts: for each extracted
                                    wavelength channel, the NUMBER OF DETECTOR
                                    PIXELS inside that channel's extraction
                                    box that are partially (some groups lost)
                                    or fully (fewer than mingroups usable)
                                    saturated. Pandeia builds them as
                                    sum(saturation_mask == 1) and == 2. Only
                                    "> 0" is ever used downstream, but the
                                    counts are pixels, not groups.
     "r_native": [...] | null,      native resolving power R(lambda) on the wl
                                    grid (null = host skips the LSF blur,
                                    safe for high-R modes only),
     "r_native_source": "..",
     "t_cycle_s": ..,               per-integration CYCLE time in a TSO series
                                    (nint=2 minus nint=1 exposure difference,
                                    which includes the between-integration
                                    reset),
     "ngroup": ..,
     "sat_frac": ..,                pandeia's fraction_saturation: the peak
                                    accumulated charge over the WHOLE scene,
                                    divided by the engine's per-mode
                                    saturation_fullwell. That divisor is not
                                    the physical full well and differs by
                                    mode (instruments.py docstring), and for
                                    a multi-order mode (SOSS) it is the
                                    brightest order, not the extracted one.
     "sat_ngroups": ..,             min over the detector of pandeia's
                                    groups-before-saturation map, i.e. the
                                    ramp length at which the brightest pixel
                                    reaches saturation_fullwell.
     "saturated": bool, "engine_version": "..",
     "n_pix_native": ..,             NATIVE-GRID channel counts, taken BEFORE
     "n_pix_unusable_dropped": ..,   the finite/positive `good` filter (fully
     "n_pix_part_sat_native": ..,    saturated channels have non-finite
     "n_pix_full_sat_native": ..,    extracted noise, so `good` drops them).
                                     The two *_sat_native counts are numbers
                                     of CHANNELS holding at least one such
                                     pixel; these four are the reporting truth.
     "warnings": {...}}      -- or {"error": "..."} if that mode failed.

Star normalization: band-integrated synphot photsys normalization to the
2MASS Ks magnitude in vegamag (web-ETC convention). Never revert to the
at_lambda monochromatic shortcut -- it mis-scales the flux by ~1-4% with
spectral type. Vega reference: the engine uses its OWN refdata Vega; the
local synphot vega_file pin is only an offline guard (the two agree to
0.08 mmag in Ks).

Group selection: probe at ngroup_min; "saturated" means the MEASURED probe
exceeds sat_limit, never a prediction. Otherwise a bracket search returns
the largest measured-safe group count with maximality PROVEN (complete only
when the count equals ngroup_max or the next integer measured unsafe); the
analytic formulas pick the first candidate only. A budget-exhausted search
returns the measured-safe best with ramp_search_complete=False -- reported,
never presented as maximal. A mode saturated at ngroup_min is kept,
flagged, with its degraded numbers so the GUI can say why. Channel-level
saturation comes from the report's 1d curves so the host can exclude or
flag per pixel.

Both the search and the "saturated" verdict read the SCENE-WIDE
fraction_saturation, matching PandExo. On a multi-order mode (SOSS) that is
the brightest order, so an order-2 selection inherits order 1's saturation
even though its own extraction may be clean; detect discloses that, and
instruments.MODES["niriss_soss_ord2"] records why it is not overridden here.

The seed formulas assume saturation_time grows linearly with ngroup. That is
EXACT for every shipped mode: pandeia's saturation_time is
tframe * (ndrop1 + (ngroup - 1) * (nframe + ndrop2) + nframe), and all four
readout patterns used here (NRSRAPID/NISRAPID/RAPID/FASTR1) have nframe = 1
and ndrop2 = 0. A future frame-averaging pattern would break the linearity,
not the search -- the bracket still proves maximality by measurement.
"""
import copy
import glob
import json
import math
import os
import sys
import traceback

import numpy as np


def _make_calc(build_default_calc, m, star, star_spectrum=None):
    calc = build_default_calc("jwst", m["instrument"], m["mode"])
    for section, kv in (m.get("config") or {}).items():
        calc["configuration"][section].update(kv)
    calc["configuration"]["detector"]["nint"] = 1
    calc["configuration"]["detector"]["nexp"] = 1
    for k, v in (m.get("strategy") or {}).items():
        calc["strategy"][k] = v
    # the registry pins PandExo's TSO background convention; the engine
    # needs BOTH keys (it resolves them to one canned file)
    if m.get("background"):
        calc["background"] = m["background"]
        if not m.get("background_level"):
            raise ValueError(
                f"mode {m['key']}: background={m['background']!r} without "
                "background_level -- the engine needs both (e.g. 'medium')")
        calc["background_level"] = m["background_level"]
    spectrum = calc["scene"][0]["spectrum"]
    if star_spectrum is None:
        spectrum["sed"] = {
            "sed_type": "phoenix", "teff": float(star["teff"]),
            "log_g": float(star["log_g"]),
            "metallicity": float(star["metallicity"])}
        # band-integrated 2MASS Ks vegamag normalization (module docstring)
        spectrum["normalization"] = {
            "type": "photsys", "bandpass": "2mass,ks",
            "norm_flux": float(star["ks_mag"]), "norm_fluxunit": "vegamag"}
    else:
        wave = np.asarray(star_spectrum.get("wave_um"), dtype=float)
        flux = np.asarray(star_spectrum.get("flux_mjy"), dtype=float)
        if (wave.ndim != 1 or wave.size < 2 or flux.shape != wave.shape
                or not np.all(np.isfinite(wave))
                or not np.all(np.isfinite(flux))
                or np.any(np.diff(wave) <= 0.0) or np.any(flux <= 0.0)):
            raise ValueError(
                "star_spectrum must contain aligned, finite, positive wave_um "
                "and flux_mjy arrays on a strictly increasing grid")
        spectrum["sed"] = {
            "sed_type": "input", "spectrum": [wave.tolist(), flux.tolist()]}
        spectrum["normalization"] = {"type": "none"}
    return calc


def _native_r(refdata, m, wl):
    """Native resolving power R(lambda) interpolated onto the extracted grid,
    from the mode's refdata dispersion file. Returns (list|None, source str).
    Missing file -> (None, note): the host applies no LSF blur, safe only
    for high-R modes.
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
    if m["instrument"] == "nircam" and m["mode"] == "lw_tsgrism":
        # The LW grism dispersion file carries NO disperser token in its name
        # (jwst_nircam_disp_*.fits, 2.4-5.0 um; the dhs0-ord* files are the
        # short-wave DHS), so the token pattern above matches nothing here.
        # The exact prefix is required: a bare *disp*.fits glob sorts the
        # dhs0 files first.
        pat = os.path.join(refdata, "jwst", "nircam", "dispersion",
                           "jwst_nircam_disp_*.fits")
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


def _run(perform_calculation, calc, ngroup, nint=1):
    c = copy.deepcopy(calc)
    c["configuration"]["detector"]["ngroup"] = int(ngroup)
    c["configuration"]["detector"]["nint"] = int(nint)
    return perform_calculation(c)


def _sat_curve(rpt, key, n_pix):
    """Saturated-PIXEL counts per extracted wavelength channel, from the 1d
    report (pandeia: sum(saturation_mask == 1) for partial, == 2 for full,
    over that channel's extraction box). Not group counts.

    LOAD-BEARING: a missing or renamed report key must FAIL here -- a silent
    all-zeros fallback would stop excluding saturated pixels after an engine
    key rename (STScI does rename them)."""
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
              refdata, star_spectrum=None):
    calc = _make_calc(build_default_calc, m, star, star_spectrum)
    ng_min, ng_max = int(m["ngroup_min"]), int(m["ngroup_max"])

    probe = _run(perform_calculation, calc, ng_min)
    sat_probe = float(probe["scalar"]["fraction_saturation"])
    sat_ng = probe["scalar"].get("sat_ngroups")

    # Saturation is a MEASUREMENT, never a prediction: the mode is saturated
    # exactly when the shortest permitted ramp measures above the limit. A
    # predictor falling below the floor does not prove the measured floor is
    # unsafe.
    saturated = sat_probe > sat_limit
    ng_best, rpt = ng_min, probe
    # Maximality is PROVEN, not extrapolated: a predictor-stall exit can stop
    # one integer below the true optimum on ramps with a per-integration time
    # offset. Bracket
    # invariant: ng_best is the largest MEASURED-safe count, hi the smallest
    # MEASURED-unsafe one; the search is complete only when ng_best == ng_max
    # or hi == ng_best + 1. Every candidate is strictly inside the bracket,
    # so each measurement shrinks it and the loop terminates. The analytic
    # formulas (linear extrapolation of the probe; pandeia's sat_ngroups)
    # only pick the FIRST candidate. If the calculation budget runs out
    # first, the measured-safe ng_best is returned with
    # ramp_search_complete=False -- reported, never presented as maximal.
    search_complete = True          # a measured-saturated floor is a proof
    if not saturated:
        hi = None
        seeds = []
        if sat_probe > 0:
            seeds.append(int(math.floor(ng_min * sat_limit / sat_probe)))
        if sat_ng is not None and np.isfinite(sat_ng) and sat_ng > 0:
            seeds.append(int(math.floor(sat_limit * float(sat_ng))))
        first = True
        for _ in range(12):                       # calculation budget
            if ng_best >= ng_max or (hi is not None and hi <= ng_best + 1):
                break                             # maximality proven
            frac_lo = float(rpt["scalar"]["fraction_saturation"])
            pred = (int(math.floor(ng_best * sat_limit / frac_lo))
                    if frac_lo > 0 else ng_max)
            guess = min(seeds) if (first and seeds) else pred
            first = False
            lo_bound, hi_bound = ng_best + 1, (ng_max if hi is None
                                               else hi - 1)
            guess = max(lo_bound, min(hi_bound, guess))
            r = _run(perform_calculation, calc, guess)
            frac = float(r["scalar"]["fraction_saturation"])
            if frac <= sat_limit:
                ng_best, rpt = guess, r
            else:
                hi = guess
        search_complete = (ng_best >= ng_max
                           or (hi is not None and hi <= ng_best + 1))

    wl, _sn = rpt["1d"]["sn"]
    flux = np.asarray(rpt["1d"]["extracted_flux"][1], dtype=float)
    noise = np.asarray(rpt["1d"]["extracted_noise"][1], dtype=float)
    wl = np.asarray(wl, dtype=float)
    good = np.isfinite(wl) & np.isfinite(flux) & np.isfinite(noise) & (flux > 0) & (noise > 0)

    # Native-grid saturation counts, taken BEFORE `good`: fully saturated
    # channels have non-finite extracted noise, so `good` removes them and a
    # post-filter count would read zero. Reporting-only; the estimator still
    # excludes unusable pixels. Display/export must quote these.
    n_part = _sat_curve(rpt, "n_partial_saturated", wl.shape[0])
    n_full = _sat_curve(rpt, "n_full_saturated", wl.shape[0])
    native_counts = {
        "n_pix_native": int(wl.shape[0]),
        "n_pix_unusable_dropped": int((~good).sum()),
        "n_pix_part_sat_native": int((n_part > 0).sum()),
        "n_pix_full_sat_native": int((n_full > 0).sum()),
    }
    native_masks = {
        "wl_native": wl.tolist(),
        "usable_native": good.tolist(),
        "n_part_sat_native_curve": n_part.tolist(),
        "n_full_sat_native_curve": n_full.tolist(),
    }

    if not good.any():
        # pandeia returned no usable pixels (e.g. NIRSpec PRISM on a bright star:
        # saturation within the shortest ramp NaNs the extracted noise everywhere)
        return {
            "unusable": True,
            "reason": (f"no unsaturated pixels at the shortest ramp "
                       f"(ngroup={ng_best}, peak saturation fraction "
                       f"{float(rpt['scalar']['fraction_saturation']):.1f}) -- "
                       "target too bright for this mode"),
            "ngroup": int(ng_best),
            "sat_frac": float(rpt["scalar"]["fraction_saturation"]),
            "saturated": True,
            "ramp_search_complete": bool(search_complete),
            **native_counts,
            **native_masks,
            "warnings": {k: str(v) for k, v in rpt["warnings"].items()},
        }

    r_native, r_src = _native_r(refdata, m, wl[good])

    # Per-integration CYCLE time: the nint=1 total_exposure_time misses the
    # between-integration reset when nreset1 != nreset2 (MIRI FASTR1), making
    # sigma optimistic by up to ~9% at ngroup_min. Keep the engine-truth
    # nint=2 minus nint=1 exposure difference.
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
        "ramp_search_complete": bool(search_complete),
        **native_counts,
        **native_masks,
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
    """Best-available refdata version: VERSION or VERSION_DATA first line,
    else the pandeia_data-<ver> directory name. Returns (version|None, source).

    A VERSION_PSF file is deliberately NOT accepted here: it names a PSF
    library, and with every supported backend on the split layout a misplaced
    PSF marker must not authenticate a data tree."""
    for name in ("VERSION", "VERSION_DATA"):
        p = os.path.join(refdata, name)
        if os.path.isfile(p):
            with open(p) as f:
                first = f.readline().strip()
            if first:
                return first, name
    base = os.path.basename(os.path.normpath(refdata))
    if base.startswith("pandeia_data-"):
        return base[len("pandeia_data-"):], "directory name"
    return None, "no VERSION/VERSION_DATA file or pandeia_data-<ver> dir name"


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

    A mismatched pair runs with wrong calibrations; the PSF library is the
    same hazard, so its RELEASE is checked, not just existence. Every
    supported backend uses the split-PSF layout, so an empty ``psf_dir`` is
    always refused.

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
        # every supported backend uses the split-PSF layout; a missing path
        # would silently bypass a third of the matched-triple check
        raise RuntimeError(
            f"pandeia.engine {engine_version} requires a separate PSF "
            "library, but no PSF_DIR was supplied. Point "
            "JWST_TOOL_PANDEIA_PSF_DIR at the matching pandeia_psfs tree "
            f"for release {eng_rel}.")

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
        # split-layout PSF library: must be an intact tree, not just a dir
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
    # point synphot's vega_file (default: a URL) at the local CALSPEC copy as
    # an offline guard; the engine loads its OWN refdata Vega and never reads
    # the pin
    _vega = os.path.join(job["cdbs"], "calspec", "alpha_lyr_stis_011.fits")
    if os.path.isfile(_vega):
        import synphot
        synphot.conf.vega_file = _vega
    from pandeia.engine.calc_utils import build_default_calc
    from pandeia.engine.perform_calculation import perform_calculation

    import pandeia.engine
    engine_version = str(getattr(pandeia.engine, "__version__", "unknown"))
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
                                 job["refdata"], job.get("star_spectrum"))
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
