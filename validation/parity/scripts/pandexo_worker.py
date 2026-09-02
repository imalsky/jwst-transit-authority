"""PandExo parity worker -- runs INSIDE a current-Pandeia conda env.

Standalone on purpose (mirrors src/jwst_tool/pandeia_worker.py): stdlib +
numpy + pandexo only. Executes current PandExo (master, pinned commit in the
parity report) on matched star/instrument configurations and dumps the raw
quantities the parity comparison needs.

    python pandexo_worker.py job.json result.json

job.json:
    {"refdata": <pandeia_refdata>, "psf_dir": <pandeia PSF library>,
     "cdbs": <PYSYN_CDBS>, "vega_file": <local CALSPEC alpha_lyr fits>,
     "star": {"teff":.., "logg":.., "metal":.., "kmag":..},
     "transit_duration_s": .., "sat_level_pct": 80.0, "depth": 0.01,
     "modes": [{"key":.., "pandexo_name":..,
                "config_overrides": {"detector": {...}, "instrument": {...},
                                     "strategy": {...}}},
               ...]}

result.json: {"__provenance__": {...}, <key>: {...} | {"error": traceback}}.
Per mode key: PandExo's native-grid (R=None) results with noise_floor=0 and
baseline fraction 1.0 (out-of-transit time == in-transit time):
    wave, error (final error with floor=0), timing (PandExo timing dict),
    ngroup, config (the exact pandeia configuration PandExo ran),
    electrons_out/in, var_out/in, e_rate_out, error_no_floor, warnings,
    qy_on_grid (the quantum-yield curve PandExo's remove_QY divided out of
    every flux; multiply back to recover pandeia's electron rate).
"""
import json
import os
import sys
import traceback

import numpy as np


def _version_file_release(root, name):
    """First line of `root/name`, or None. Used for the refdata/PSF releases."""
    if not root:
        return None
    p = os.path.join(root, name)
    if os.path.isfile(p):
        with open(p) as f:
            first = f.readline().strip()
        return first or None
    return None


def _pandexo_commit(pandexo_dir):
    """Exact PandExo git commit, or None when it cannot be established.

    PandExo is consumed from MASTER, whose behavior moves between releases with
    no version bump (the NIRISS SOSS 30-group cap landed on master one day
    after the 2026-07-12 parity run without changing `pandexo.engine`'s
    version). A version string alone therefore does not identify what ran, so
    the gate requires this commit.

    Two sources, in order of authority:

    1. pip's own record of a `pip install git+...` -- ``direct_url.json`` in
       the dist-info -- accepted only when the imported package lives under
       that distribution's install root (a shadowed second install must not
       pair one package's code with another's commit) and the recorded value
       is a full 40-hex commit.
    2. a real git checkout, accepted only when the IMPORTED package file is
       TRACKED by that repository and the repository ships PandExo's engine
       (``engine/justdoit.py``).

    The tracking proof is load-bearing. `git rev-parse` walks UP from its
    -C directory, so for a pip-installed package it finds whatever repository
    happens to enclose the environment and reports ITS head: measured
    2026-08-04, a conda env under /opt/homebrew returned Homebrew's own HEAD
    as "the PandExo commit", and any enclosing Python repo passed the old
    ancestor/marker checks. `git ls-files --error-unmatch` on the imported
    ``__init__.py`` cannot be satisfied by an enclosing repository that does
    not actually version this package. Worktrees are fine (no `.git`-is-a-dir
    requirement); everything unattributable returns None, and the gate then
    fails loudly on the missing commit.
    """
    import json
    import subprocess

    def _git(*args):
        return subprocess.run(["git", "-C", pandexo_dir, *args],
                              capture_output=True, text=True, timeout=10)

    def _is_commit(s):
        return len(s) == 40 and all(c in "0123456789abcdef" for c in s.lower())

    # 1. pip's record of the resolved commit (authoritative, no git needed)
    try:
        from importlib.metadata import distribution
        dist = distribution("pandexo.engine")
        raw = dist.read_text("direct_url.json")
        if raw:
            commit = str((json.loads(raw).get("vcs_info") or {})
                         .get("commit_id") or "")
            root = os.path.realpath(str(dist.locate_file("")))
            pkg = os.path.realpath(pandexo_dir)
            if (_is_commit(commit)
                    and (pkg == root or pkg.startswith(root + os.sep))):
                return commit
    except Exception:
        pass

    # 2. a genuine checkout that versions THIS package
    try:
        tracked = _git("ls-files", "--error-unmatch", "__init__.py")
        engine = _git("ls-files", "--error-unmatch", "engine/justdoit.py")
        if tracked.returncode != 0 or engine.returncode != 0:
            return None
        r = _git("rev-parse", "HEAD")
        head = r.stdout.strip()
        if r.returncode != 0 or not _is_commit(head):
            return None
        dirty = _git("status", "--porcelain").stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return None


def _provenance():
    import pandeia.engine
    import pandexo
    try:
        from importlib.metadata import version
        pandexo_version = version("pandexo.engine")
    except Exception:
        pandexo_version = str(getattr(pandexo, "__version__", "unknown"))
    pandexo_dir = os.path.dirname(pandexo.__file__)
    refdata = os.environ.get("pandeia_refdata")
    psf_dir = os.environ.get("PSF_DIR")
    return {
        "pandeia_engine_version": str(getattr(pandeia.engine, "__version__",
                                              "unknown")),
        "pandexo_version": pandexo_version,
        "pandexo_commit": _pandexo_commit(pandexo_dir),
        "pandexo_path": pandexo_dir,
        "refdata": refdata,
        "refdata_version": (_version_file_release(refdata, "VERSION")
                            or _version_file_release(refdata, "VERSION_DATA")),
        "psf_dir": psf_dir,
        # the PSF RELEASE, not just the directory: a mixed PSF tree changes the
        # point-spread function and hence the extracted flux on this side too
        "psf_version": _version_file_release(psf_dir, "VERSION_PSF"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def _shared_star(job):
    """One PHOENIX spectrum, normalized to the tool's 2MASS Ks input.

    PandExo interprets ``ref_wave=2.22`` through Bessell K, whereas the tool's
    user input is explicitly 2MASS Ks.  Convert the Ks-normalized spectrum to
    its equivalent Bessell-K magnitude, then pass it as a user spectrum so
    both Pandeia calculations receive the same wavelength/flux arrays.
    """
    import stsynphot
    from synphot import Observation, units
    from pandexo.engine.synphot_compat import (
        load_bandpass_from_file, load_phoenix_spectrum,
        renormalize_to_vegamag, sample_spectrum_micron_mjy)
    from pandexo.engine.create_input import outTrans

    star = job["star"]
    root = os.path.join(job["cdbs"], "comp", "nonhst")
    ks = load_bandpass_from_file(
        os.path.join(root, "2mass_ks_001_syn.fits"))
    bessell_k = load_bandpass_from_file(
        os.path.join(root, "bessell_k_003_syn.fits"))
    raw = load_phoenix_spectrum(
        float(star["teff"]), float(star["metal"]), float(star["logg"]))
    normalized = renormalize_to_vegamag(raw, float(star["kmag"]), ks)
    k_mag = Observation(normalized, bessell_k, force="extrap").effstim(
        units.VEGAMAG, vegaspec=stsynphot.Vega)
    wave, flux = sample_spectrum_micron_mjy(normalized)
    # Record the spectrum *after* PandExo's own user-input normalization and
    # sampling. Feeding this exact array to the tool side prevents a second
    # resampling implementation from masquerading as an estimator difference.
    pandexo_source = outTrans({
        "type": "user",
        "starpath": {"w": np.asarray(wave, float),
                     "f": np.asarray(flux, float) * 1e-3},
        "w_unit": "um", "f_unit": "Jy", "ref_wave": 2.22,
        "mag": float(k_mag.value),
    })
    return {
        "wave_um": np.asarray(pandexo_source["wave"], float).tolist(),
        "flux_mjy": np.asarray(
            pandexo_source["flux_out_trans"], float).tolist(),
        "pandexo_k_mag": float(k_mag.value),
        "normalization": "2MASS Ks vegamag",
        "ks_mag": float(star["kmag"]),
    }


def _exo_dict(jdi, job, shared_star):
    exo = jdi.load_exo_dict()
    exo["observation"]["sat_level"] = float(job["sat_level_pct"])
    exo["observation"]["sat_unit"] = "%"
    exo["observation"]["noccultations"] = 1
    exo["observation"]["R"] = None                 # native grid
    exo["observation"]["baseline"] = 1.0           # t_out == t_in
    exo["observation"]["baseline_unit"] = "frac"
    exo["observation"]["noise_floor"] = 0          # random noise only

    star = job["star"]
    exo["star"]["type"] = "user"
    exo["star"]["starpath"] = {
        "w": shared_star["wave_um"],
        # PandExo accepts Jy, while the shared artifact records mJy because
        # that is the Pandeia input-spectrum unit.
        "f": (np.asarray(shared_star["flux_mjy"], float) * 1e-3).tolist(),
    }
    exo["star"]["w_unit"] = "um"
    exo["star"]["f_unit"] = "Jy"
    exo["star"]["mag"] = float(shared_star["pandexo_k_mag"])
    exo["star"]["ref_wave"] = 2.22                 # K normalization branch
    exo["star"]["temp"] = float(star["teff"])
    exo["star"]["metal"] = float(star["metal"])
    exo["star"]["logg"] = float(star["logg"])

    # PandExo's "constant" planet derives the depth from the radii
    # (depth = (rp/r*)^2), not from a depth key: encode job["depth"] as
    # rp = sqrt(depth) stellar radii.
    exo["planet"]["type"] = "constant"
    exo["planet"]["f_unit"] = "rp^2/r*^2"
    exo["star"]["radius"] = 1.0
    exo["star"]["r_unit"] = "R_sun"
    exo["planet"]["radius"] = float(np.sqrt(job["depth"]))
    exo["planet"]["r_unit"] = "R_sun"
    exo["planet"]["transit_duration"] = float(job["transit_duration_s"])
    exo["planet"]["td_unit"] = "s"
    return exo


def _one_mode(jdi, job, m, shared_star):
    exo = _exo_dict(jdi, job, shared_star)
    inst = jdi.load_mode_dict(m["pandexo_name"])
    for section, kv in (m.get("config_overrides") or {}).items():
        # "strategy" is a TOP-LEVEL section of the PandExo mode dict (it
        # carries the SOSS extraction order); everything else lives under
        # "configuration".
        if section == "strategy":
            inst["strategy"].update(kv)
        else:
            inst["configuration"][section].update(kv)
    res = jdi.run_pandexo(exo, inst, save_file=False, verbose=False)

    fs = res["FinalSpectrum"]
    # PandExo divides pandeia's extracted electron rate by the detector
    # quantum yield (jwst.remove_QY) so its shot-noise formula runs on
    # photons; every flux it reports (e_rate_out, electrons_*) carries that
    # division, while the tool side reports pandeia's raw electron rate.
    # Record the exact curve by probing PandExo's own remove_QY with a unit
    # spectrum on the same grid -- no reimplementation to drift. Identity
    # (all 1.0) for NIRCam/MIRI and red of ~3 um on NIRSpec.
    from pandexo.engine.jwst import remove_QY
    _wave = np.asarray(fs["wave"], float)
    _probe = {"1d": {"extracted_flux": [_wave, np.ones_like(_wave)]}}
    _inst_name = m["pandexo_name"].split(" ")[0].lower()
    qy_on_grid = 1.0 / np.asarray(
        remove_QY(_probe, _inst_name)["1d"]["extracted_flux"][1], float)
    # validity only (finite, positive): the reference files themselves dip
    # marginally below 1 (NIRISS red end reaches 0.99990), so a physics
    # bound of >= 1 would reject real reference data
    if not (np.all(np.isfinite(qy_on_grid)) and np.all(qy_on_grid > 0.0)):
        raise RuntimeError(
            f"{m['key']}: probed quantum-yield curve is not finite and "
            f"positive (min {np.nanmin(qy_on_grid)}, "
            f"max {np.nanmax(qy_on_grid)})")
    raw = res["RawData"]
    pout = res["PandeiaOutTrans"]
    cfg = res["PandeiaOutTrans"]["input"]["configuration"]
    sat_wave = np.asarray(pout["1d"]["sn"][0], float)
    n_part_sat = np.asarray(pout["1d"]["n_partial_saturated"][1], float)
    n_full_sat = np.asarray(pout["1d"]["n_full_saturated"][1], float)
    pandeia_star = np.asarray(
        pout["input"]["scene"][0]["spectrum"]["sed"]["spectrum"], float)
    if pandeia_star.ndim != 2 or pandeia_star.shape[0] != 2:
        raise RuntimeError(
            "PandExo did not submit a 2 x N input spectrum to Pandeia; "
            f"got shape {pandeia_star.shape}")
    if n_part_sat.shape != sat_wave.shape or n_full_sat.shape != sat_wave.shape:
        raise RuntimeError(
            "PandExo Pandeia saturation curves are not aligned with the "
            f"native wavelength grid: wave={sat_wave.shape}, "
            f"partial={n_part_sat.shape}, full={n_full_sat.shape}")
    return {
        "wave": np.asarray(fs["wave"], float).tolist(),
        "error": np.asarray(fs["error_w_floor"], float).tolist(),
        "error_no_floor": np.asarray(raw["error_no_floor"], float).tolist(),
        "electrons_out": np.asarray(raw["electrons_out"], float).tolist(),
        "electrons_in": np.asarray(raw["electrons_in"], float).tolist(),
        "var_out": np.asarray(raw["var_out"], float).tolist(),
        "var_in": np.asarray(raw["var_in"], float).tolist(),
        "e_rate_out": np.asarray(raw["e_rate_out"], float).tolist(),
        "qy_on_grid": qy_on_grid.tolist(),
        "sat_wave": sat_wave.tolist(),
        "n_partial_saturated": n_part_sat.tolist(),
        "n_full_saturated": n_full_sat.tolist(),
        "_pandeia_star": {
            "wave_um": pandeia_star[0].tolist(),
            "flux_mjy": pandeia_star[1].tolist(),
            "normalization": "2MASS Ks vegamag; PandExo-resampled input",
            "ks_mag": float(job["star"]["kmag"]),
        },
        "timing": {k: (float(v) if isinstance(v, (int, float, np.floating))
                       else str(v)) for k, v in res["timing"].items()},
        "ngroup": int(cfg["detector"]["ngroup"]),
        "config": cfg,
        "warnings": {k: str(v) for k, v in res.get("warning", {}).items()},
    }


def main():
    job = json.load(open(sys.argv[1]))
    for var, key in (("pandeia_refdata", "refdata"), ("PSF_DIR", "psf_dir"),
                     ("PYSYN_CDBS", "cdbs")):
        path = job[key]
        if not os.path.isdir(path):
            raise RuntimeError(f"{var} path does not exist: {path}")
        os.environ[var] = path

    import warnings as _w
    _w.filterwarnings("ignore")
    import stsynphot
    vega = job["vega_file"]
    if not os.path.isfile(vega):
        raise RuntimeError(f"local Vega spectrum not found: {vega}")
    stsynphot.conf.vega_file = vega
    stsynphot.spectrum.load_vega(vega)
    if stsynphot.Vega is None:
        raise RuntimeError(f"stsynphot failed to load Vega from {vega}")
    import pandexo.engine.justdoit as jdi

    shared_star = _shared_star(job)
    out = {"__provenance__": _provenance()}
    print(f"[pandexo] engine {out['__provenance__']['pandeia_engine_version']}",
          flush=True)
    for m in job["modes"]:
        key = m["key"]
        print(f"[pandexo] {key} ({m['pandexo_name']}) ...", flush=True)
        try:
            out[key] = _one_mode(jdi, job, m, shared_star)
            submitted = out[key].pop("_pandeia_star")
            if "__shared_star__" not in out:
                out["__shared_star__"] = submitted
            elif submitted != out["__shared_star__"]:
                raise RuntimeError(
                    "PandExo submitted different stellar spectra across modes; "
                    "a single fixed-source parity comparison is impossible")
            print(f"[pandexo] {key}: ngroup={out[key]['ngroup']} "
                  f"npix={len(out[key]['wave'])}", flush=True)
        except Exception:
            out[key] = {"error": traceback.format_exc()}
            print(f"[pandexo] {key}: FAILED", flush=True)

    with open(sys.argv[2], "w") as f:
        json.dump(out, f)
    print("[pandexo] done", flush=True)


if __name__ == "__main__":
    main()
