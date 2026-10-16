#!/usr/bin/env python3
"""Generate requirements-app-lock.txt from the LIVE working environment.

Walks the installed dependency closure of the app stack and pins every
package at exactly the version running here. The Docker build installs this
lock with --no-deps so it reproduces the validated environment verbatim rather
than selecting newer transitive dependencies during a later rebuild. Generate
the lock only from a resolver-clean environment and run ``pip check`` before
committing it; the deployed working set must not rely on ignored dependency
metadata.

Run in the environment that runs the tool, then commit the lock:
    python deploy/make_app_lock.py

The PICASO env is a conda env of its own (picaso 4.1 needs numpy >= 2). Conda
installs its compiled stack from picaso-conda-lock.txt; this script's
``--env picaso`` mode pins only the distributions pip installed on top. Run
both in the validated env (``pip check`` clean), then commit both locks:
    micromamba env export -p <env> --explicit --md5 > deploy/picaso-conda-lock.txt
    <env>/bin/python deploy/make_app_lock.py --env picaso
"""
import argparse
import json
import re
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from packaging.requirements import Requirement

# Roots: the local packages' dependencies get walked, plus the gui extra
# (streamlit/pandas) and the Space bootstrap's huggingface_hub. The trailing
# entries are runtime imports that the roots' metadata does NOT declare; the
# walker only fills in their versions.
ROOTS = ["jwst-transit-authority", "vulcan3",
         "streamlit", "pandas", "huggingface_hub", "matplotlib",
         "pooch", "sortedcontainers", "cloudpickle",
         "platformdirs", "setuptools", "threadpoolctl", "zstandard",
         "xyzservices"]
# (jaraco.collections/jaraco.text are setuptools-vendored, not standalone
# distributions -- pinning setuptools covers them)
# Local editables: present in the closure but installed from the repo
# checkouts by the Dockerfile, never from PyPI. vulcan-forward MUST be here:
# it is a hard dependency of jwst-transit-authority with no PyPI release, so pinning
# it writes a line the --no-deps install cannot resolve and the image fails.
LOCAL = {"jwst-transit-authority", "vulcan3", "vulcan-forward"}
# --env picaso: pin only what pip installed. The conda-installed rest is
# walked, never pinned: a --no-deps pip install of conda's numpy or h5py would
# replace its HDF5-linked builds.
PIP_ONLY: set[str] | None = None

seen: set[str] = set()
pins: dict[str, str] = {}


def canon(name: str) -> str:
    return name.lower().replace("_", "-")


def pip_installed() -> set[str]:
    """Names of the distributions whose metadata dir in site-packages no
    conda package's file list owns (exact dir names, so a conda package that
    pip later replaced counts as pip's)."""
    meta = Path(sys.prefix) / "conda-meta"
    if not meta.is_dir():
        sys.exit(f"--env picaso must run with the picaso conda env's python; "
                 f"{sys.prefix} has no conda-meta/")
    owned = set()
    for f in meta.glob("*.json"):
        for path in json.loads(f.read_text()).get("files", []):
            m = re.search(r"site-packages/([^/]+\.(dist|egg)-info)(/|$)", path)
            if m:
                owned.add(m.group(1))
    site = Path(sysconfig.get_paths()["purelib"])
    return {pep503(d.name.split("-")[0]) for d in site.iterdir()
            if d.name.endswith((".dist-info", ".egg-info")) and d.name not in owned}


def pep503(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def walk(name: str) -> None:
    c = canon(name)
    if c in seen:
        return
    seen.add(c)
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        # extras and marker-false deps are skipped before the recursion, so
        # this one is required: a lock written without it would install
        # an env that fails at import
        raise SystemExit(f"{name} is required but not installed; repair the "
                         "env (pip check) and regenerate") from None
    if c not in LOCAL and (PIP_ONLY is None
                           or pep503(dist.metadata["Name"]) in PIP_ONLY):
        pins[canon(dist.metadata["Name"])] = dist.version
    for raw in dist.requires or []:
        req = Requirement(raw)
        if req.marker is not None:
            # A marker can only be evaluated for THIS platform. The case
            # that matters is a dep excluded here but needed on the Linux
            # image; it is rare, so a skipped Linux-marked dep is printed
            # for review instead of pinned.
            if not req.marker.evaluate({"extra": ""}):
                m = str(req.marker)
                if "linux" in m:
                    print(f"NOTE: skipped linux-marked dep {req.name} "
                          f"({m}) -- verify it is not needed")
                continue
        walk(req.name)


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--env", choices=("app", "picaso"), default="app")
picaso = ap.parse_args().env == "picaso"
if picaso:
    ROOTS, LOCAL, PIP_ONLY = ["picaso"], set(), pip_installed()

for r in ROOTS:
    walk(r)

out = Path(__file__).resolve().parent / "requirements-app-lock.txt"
lines = [
    "# GENERATED by make_app_lock.py from the validated working env.",
    "# Installed with --no-deps so the validated versions remain exact.",
    "# The generating environment must pass `pip check`; regenerate after",
    "# any dependency or environment change.",
]
if picaso:
    out = out.with_name("requirements-picaso-lock.txt")
    lines = [
        "# GENERATED by make_app_lock.py --env picaso from the validated picaso env.",
        "# Only what pip installed: conda installs the rest from",
        "# picaso-conda-lock.txt first, and this lock goes on with --no-deps.",
        "# The generating environment must pass `pip check`; regenerate both",
        "# locks after any change to the env.",
    ]
lines += [f"{n}=={v}" for n, v in sorted(pins.items())]
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out} ({len(pins)} pins)")
