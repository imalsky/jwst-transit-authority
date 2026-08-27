"""Compact, path-free provenance for configurations and exported results."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import subprocess
from functools import lru_cache
from pathlib import Path

from jwst_tool import forward
from jwst_tool import instruments as ins
from jwst_tool import noise


# Repository identity -> the DIRECTORY names it may sit under; the solver's
# GitHub repo is `jax-vulcan` but its working copy is usually `VULCAN-JAX`, and
# a name-only lookup dropped it from every exported block. Every entry is
# always reported, with an explicit status when absent or unversioned: an
# omitted row reads as "nothing to record" rather than "not found". The last
# two are reference oracles, not dependencies.
REPOSITORIES = {
    "vulcan-jax": ("VULCAN-JAX", "jax-vulcan"),
    "vulcan-forward": ("vulcan-forward",),
    "vulcan-jwst-tool": ("vulcan-jwst-tool",),
    "vulcan-retrieval": ("vulcan-retrieval",),
    "VULCAN-master": ("VULCAN-master",),
    "VULCAN-vm-branch": ("VULCAN-vm-branch", ".oracles/vulcan3_vm_branch"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path) -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True,
            timeout=10, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    sha = run("rev-parse", "HEAD")
    return {
        "commit": sha or "unknown",
        "branch": run("branch", "--show-current") or "detached",
        "dirty": bool(run("status", "--porcelain")),
    }


def _repo_identity(workspace: Path, directories: tuple) -> dict:
    """Resolve one repository, reporting absence instead of omitting it."""
    for name in directories:
        path = workspace / name
        if (path / ".git").exists():
            out = _git(path)
            out["directory"] = name
            return out
        if path.exists():
            # Present but unversioned: its revision CANNOT be recorded, and a
            # comparison against it is not reproducible. Say so.
            return {"commit": "unversioned", "branch": "unversioned",
                    "dirty": None, "directory": name}
    return {"commit": "absent", "branch": "absent", "dirty": None,
            "directory": None}


def _versions() -> dict:
    out = {}
    for name in ("vulcan-jwst-tool", "vulcan-forward", "vulcan-jax",
                 "exojax", "jax", "numpy"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def _marker(path: Path) -> dict:
    if not path.is_file():
        return {"name": path.name, "version": "missing", "sha256": None}
    return {
        "name": path.name,
        "version": path.read_text(errors="replace").splitlines()[0].strip(),
        "sha256": _sha256(path),
    }


def _file(path: Path) -> dict:
    if not path.is_file():
        return {"name": path.name, "bytes": None, "sha256": None}
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _pandeia_python_identity() -> dict:
    python = ins.PANDEIA_PYTHON
    if not python or not Path(python).is_file():
        return {"engine": "unavailable", "pandexo": "unavailable",
                "pandexo_commit": None}
    code = r'''import importlib.metadata as m, json
out = {}
for name in ("pandeia.engine", "pandexo.engine"):
    try: out[name] = m.version(name)
    except m.PackageNotFoundError: out[name] = "not installed"
commit = None
try:
    raw = m.distribution("pandexo.engine").read_text("direct_url.json")
    commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id") if raw else None
except Exception: pass
print(json.dumps({"engine": out["pandeia.engine"],
                  "pandexo": out["pandexo.engine"],
                  "pandexo_commit": commit}))'''
    result = subprocess.run(
        [python, "-c", code], capture_output=True, text=True, timeout=30,
        check=False)
    if result.returncode != 0:
        return {"engine": "unavailable", "pandexo": "unavailable",
                "pandexo_commit": None}
    try:
        return json.loads(result.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"engine": "unavailable", "pandexo": "unavailable",
                "pandexo_commit": None}


def ktable_sources() -> dict:
    """{molecule: record} for every ExoMolOP k-table the engine's fetcher
    recorded, merged with each table's own header. Path-free (exportable).
    Raises the engine's own error when the data root or provenance.json is
    absent -- callers decide how loud."""
    from vulcan_forward import exomolop
    out = {}
    for mol, rec in sorted(exomolop.provenance().items()):
        info = exomolop.table_info(mol)
        try:
            grid_sha, band_r = info["grid_sha256"], info["band_resolution"]
        except KeyError as exc:
            raise RuntimeError(
                f"vulcan-forward >= 0.11.1 required: table_info lacks {exc}"
            ) from exc
        out[mol] = {
            "dataset": rec["dataset"], "iso": rec["iso"],
            "natural_abundance": bool(rec["natural_abundance"]),
            "file": rec["file"], "url": rec["url"],
            # The fetcher's curated DOI wins where it exists: four ExoMolOP
            # headers ship a placeholder instead of one. The header value is
            # kept alongside, so nothing is silently substituted.
            "doi": rec.get("doi") or info.get("doi"),
            "header_doi": info.get("doi"), "date_id": info.get("date_id"),
            "ngauss": int(info["ngauss"]),
            "t_range_K": [float(v) for v in info["t_range_k"]],
            "p_range_bar": [float(v) for v in info["p_range_bar"]],
            "wl_range_um": [float(v) for v in info["wl_range_um"]],
            "grid_sha256": grid_sha, "band_resolution": float(band_r),
        }
    return out


def _ktables() -> dict:
    try:
        return {"status": "ok", "tables": ktable_sources()}
    except (ImportError, RuntimeError, FileNotFoundError):
        # explicit absence, kept PATH-FREE: the engine's message names the
        # absolute data directory, which must never enter an export
        return {"status": "absent",
                "remedy": "python -m vulcan_forward.fetch_exomolop "
                          "--molecules <species>"}


@lru_cache(maxsize=1)
def _base_snapshot() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    workspace = repo_root.parent
    repos = {
        name: _repo_identity(workspace, directories)
        for name, directories in REPOSITORIES.items()
    }
    pandeia = _pandeia_python_identity()
    pandeia.update({
        "required_release": ins.BACKEND_RELEASE,
        "refdata": _marker(Path(ins.PANDEIA_REFDATA) / "VERSION_DATA"),
        "psf": _marker(Path(ins.PANDEIA_PSF_DIR) / "VERSION_PSF")
        if ins.PANDEIA_PSF_DIR else {
            "name": "VERSION_PSF", "version": "missing", "sha256": None},
    })
    return {
        "schema": 1,
        "repositories": repos,
        "software": _versions(),
        "pandeia_stack": pandeia,
        "datasets": {
            "phoenix_catalog": _file(
                Path(ins.PYSYN_CDBS) / "grid" / "phoenix" / "catalog.fits"),
            "ktables": _ktables(),
        },
        "cache_schema": {
            "model": forward._VERSION,
            "pandeia_worker": noise.WORKER_VERSION,
        },
    }


def snapshot(random_seed: int | None = None) -> dict:
    """Return exact code/data identity without absolute workstation paths."""
    out = json.loads(json.dumps(_base_snapshot()))
    out["random_seed"] = None if random_seed is None else int(random_seed)
    return out
