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
                "config_overrides": {"detector": {...}, "instrument": {...}}},
               ...]}

result.json: {"__provenance__": {...}, <key>: {...} | {"error": traceback}}.
Per mode key: PandExo's native-grid (R=None) results with noise_floor=0 and
baseline fraction 1.0 (out-of-transit time == in-transit time):
    wave, error (final error with floor=0), timing (PandExo timing dict),
    ngroup, config (the exact pandeia configuration PandExo ran),
    electrons_out/in, var_out/in, e_rate_out, error_no_floor, warnings.
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
       the dist-info -- which names the commit pip actually resolved.
    2. a real git checkout, but ONLY when the discovered work tree actually
       contains this package.

    That containment check is load-bearing. `git rev-parse` walks UP from its
    -C directory, so for a pip-installed package under site-packages it finds
    whatever repository happens to enclose the environment and reports ITS
    head. Measured 2026-08-04: a conda env under /opt/homebrew returned
    Homebrew's own HEAD as "the PandExo commit". This function is supposed to
    fail closed by returning None; instead it silently attributed an unrelated
    repository's commit, and on a machine where that commit happened to match
    the gate would have PASSED having verified nothing.
    """
    import json
    import subprocess

    # 1. pip's record of the resolved commit (authoritative, no git needed)
    try:
        from importlib.metadata import distribution
        dist = distribution("pandexo.engine")
        raw = dist.read_text("direct_url.json")
        if raw:
            commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id")
            if commit:
                return str(commit)
    except Exception:
        pass

    # 2. a genuine checkout of THIS package
    try:
        top = subprocess.run(
            ["git", "-C", pandexo_dir, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10)
        if top.returncode != 0:
            return None
        root = os.path.realpath(top.stdout.strip())
        pkg = os.path.realpath(pandexo_dir)
        if not (pkg == root or pkg.startswith(root + os.sep)):
            return None                     # unrelated enclosing repository
        if not os.path.isdir(os.path.join(root, ".git")):
            return None
        # the work tree must really be PandExo, not a parent that contains it
        if not any(os.path.exists(os.path.join(root, n))
                   for n in ("pandexo", "setup.py", "pyproject.toml")):
            return None
        r = subprocess.run(["git", "-C", pandexo_dir, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        head = r.stdout.strip()
        if r.returncode != 0 or not head:
            return None
        dirty = subprocess.run(
            ["git", "-C", pandexo_dir, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10).stdout.strip()
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


def _exo_dict(jdi, job):
    exo = jdi.load_exo_dict()
    exo["observation"]["sat_level"] = float(job["sat_level_pct"])
    exo["observation"]["sat_unit"] = "%"
    exo["observation"]["noccultations"] = 1
    exo["observation"]["R"] = None                 # native grid
    exo["observation"]["baseline"] = 1.0           # t_out == t_in
    exo["observation"]["baseline_unit"] = "frac"
    exo["observation"]["noise_floor"] = 0          # random noise only

    star = job["star"]
    exo["star"]["type"] = "phoenix"
    exo["star"]["mag"] = float(star["kmag"])
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


def _one_mode(jdi, job, m):
    exo = _exo_dict(jdi, job)
    inst = jdi.load_mode_dict(m["pandexo_name"])
    for section, kv in (m.get("config_overrides") or {}).items():
        inst["configuration"][section].update(kv)
    res = jdi.run_pandexo(exo, inst, save_file=False, verbose=False)

    fs = res["FinalSpectrum"]
    raw = res["RawData"]
    cfg = res["PandeiaOutTrans"]["input"]["configuration"]
    return {
        "wave": np.asarray(fs["wave"], float).tolist(),
        "error": np.asarray(fs["error_w_floor"], float).tolist(),
        "error_no_floor": np.asarray(raw["error_no_floor"], float).tolist(),
        "electrons_out": np.asarray(raw["electrons_out"], float).tolist(),
        "electrons_in": np.asarray(raw["electrons_in"], float).tolist(),
        "var_out": np.asarray(raw["var_out"], float).tolist(),
        "var_in": np.asarray(raw["var_in"], float).tolist(),
        "e_rate_out": np.asarray(raw["e_rate_out"], float).tolist(),
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

    out = {"__provenance__": _provenance()}
    print(f"[pandexo] engine {out['__provenance__']['pandeia_engine_version']}",
          flush=True)
    for m in job["modes"]:
        key = m["key"]
        print(f"[pandexo] {key} ({m['pandexo_name']}) ...", flush=True)
        try:
            out[key] = _one_mode(jdi, job, m)
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
