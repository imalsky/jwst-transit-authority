"""PICASO reference-data bootstrap + content fingerprints (provider glue).

Light module: stdlib-only at import time; picaso itself is imported lazily by
``import_picaso()`` so the numpy-only test suite never touches it.

Contracts (fail fast, no silent fallbacks):

* The reference tree is selected ONLY by ``JWST_TOOL_PICASO_REFDATA`` -- no
  baked-in default path; unset or wrong raises with the remedy.
* ``bootstrap()`` HARD-ASSIGNS ``picaso_refdata`` and ``PYSYN_CDBS`` and must
  run before EVERY picaso import and climate run: picaso re-pins PYSYN_CDBS
  per call and the tool's stellar.py pins its own cdbs root -- both sides
  re-pin, keep it that way.
* Fingerprints are CONTENT hashes (never name/count-only), memoized
  in-process and disk-cached keyed by a stat signature, so canonicalization
  does not re-read ~30 MB of tables per call; any file change invalidates
  via the signature.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ENV_VAR = "JWST_TOOL_PICASO_REFDATA"

CHEM_GRID_REL = "chemistry/visscher_grid_2121"
PREWEIGHTED_REL = "opacities/preweighted"
STELLAR_TRDS_REL = "stellar_grids/grp/redcat/trds"
CONTINUUM_REL = "opacities/ck_cx_cont_opacities.db"
CLIMATE_INPUTS_REL = "climate_INPUTS"
NATIVE_OPACITY_REL = "opacities/opacities/opacities_0.3_15_R15000.db"
#: complete-tree expectations (checked by datacheck + the manifest builders)
CHEM_GRID_N_FILES = 78          # 13 [M/H] nodes x 6 C/O nodes
PREWEIGHTED_N_FILES = 140       # 70 (feh, C/O) nodes x {standard, _NoTiOVO}


def refdata_root() -> Path:
    """The PICASO reference root, validated. Loud on unset/missing."""
    root = os.environ.get(ENV_VAR, "").strip()
    if not root:
        raise RuntimeError(
            f"{ENV_VAR} is not set: the PICASO provider / climate T-P mode "
            "needs the PICASO v4.0 reference tree (chemistry/"
            "visscher_grid_2121 + opacities/preweighted + stellar_grids). "
            "Set the env var to the reference root; 'jwst-tool data' reports "
            "each piece.")
    p = Path(root).expanduser()
    if not (p / CHEM_GRID_REL).is_dir():
        raise RuntimeError(
            f"{ENV_VAR}={p} does not look like a PICASO v4.0 reference tree: "
            f"missing {CHEM_GRID_REL}. Point the env var at the tree that "
            "holds chemistry/, opacities/, climate_INPUTS/ and "
            "stellar_grids/.")
    return p


def bootstrap() -> Path:
    """Pin picaso's env to the validated tree. Call before every picaso use.

    HARD assignment, never ``setdefault`` (shell values are routinely stale);
    re-pins the already-imported synphot config objects too.
    """
    root = refdata_root()
    trds = root / STELLAR_TRDS_REL
    os.environ["picaso_refdata"] = str(root)
    os.environ["PYSYN_CDBS"] = str(trds)
    if "stsynphot" in sys.modules:
        import stsynphot
        stsynphot.conf.rootdir = str(trds)
    return root


def import_picaso():
    """``picaso.justdoit`` with the env pinned first. Loud on failure."""
    bootstrap()
    try:
        import picaso.justdoit as jdi
    except Exception as exc:                     # noqa: BLE001 -- re-raised loud
        raise RuntimeError(
            f"picaso import failed ({exc!r}). The PICASO provider needs "
            "picaso==4.0.1 in this environment: pip install picaso==4.0.1"
        ) from exc
    return jdi


def picaso_version() -> str:
    """Installed picaso dist version WITHOUT importing the package."""
    import importlib.metadata as _im
    try:
        return _im.version("picaso")
    except _im.PackageNotFoundError as exc:
        raise RuntimeError(
            "picaso is not installed in this environment; the PICASO "
            "provider / climate mode needs picaso==4.0.1 "
            "(pip install picaso==4.0.1)") from exc


def ck_node_string(met_x_solar: float, co_ratio: float) -> str:
    """Canonical CK-node token, e.g. 10x solar / 0.55 -> 'feh1.0_co0.55'.

    Mirrors the preweighted/chemistry FILE naming exactly (feh has one
    decimal, no '+'; C/O has two decimals).
    """
    import math
    feh = math.log10(float(met_x_solar))
    return "feh%.1f_co%.2f" % (feh, float(co_ratio))


def ck_path(node: str, tio_vo: bool, root: Path | None = None) -> Path:
    """Preweighted CK table for a node token; refuses a missing file."""
    root = refdata_root() if root is None else root
    suffix = "" if tio_vo else "_NoTiOVO"
    p = root / PREWEIGHTED_REL / f"sonora_2121grid_{node}{suffix}.hdf5"
    if not p.is_file():
        raise RuntimeError(
            f"climate CK table not found: {p}. The (met, C/O) node {node!r} "
            "is not shipped in the preweighted grid (extreme metallicities "
            "carry only the mid C/O nodes) or the tree is incomplete.")
    return p


def chem_node_path(node: str, root: Path | None = None) -> Path:
    """Visscher 2121 chemistry node file for a node token."""
    root = refdata_root() if root is None else root
    p = root / CHEM_GRID_REL / f"sonora_2121grid_{node}.txt"
    if not p.is_file():
        raise RuntimeError(
            f"chemistry grid node file not found: {p}. The tree is "
            "incomplete (expected the 13x6 Visscher 2121 node files).")
    return p


# ---------------------------------------------------------------------------
# Content fingerprints (memoized; disk-cached keyed by stat signature)
# ---------------------------------------------------------------------------

_MEMO: dict[str, object] = {}


def _stat_signature(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _content_sha1(paths: list[Path]) -> str:
    """sha1 over the sorted files' names + raw contents."""
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(p.name.encode() + b"\0")
        with open(p, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()


def _fingerprint_cache_file() -> Path:
    from jwst_tool import instruments as ins
    return Path(ins.OUTPUT_DIR) / "picaso_fingerprints.json"


def _cached_content_sha1(kind: str, paths: list[Path]) -> str:
    """Content sha1 with in-process memo + on-disk cache.

    The disk cache maps ``kind`` -> {"sig": stat signature, "sha1": hash};
    a mismatched signature re-hashes. Writes are atomic (tmp + replace).
    """
    import time as _time
    memo = _MEMO.get(kind)
    # re-verify the stat signature at most once per minute per process: each
    # verification stats every file (seconds per call on a remote volume)
    if memo is not None and _time.time() - memo.get("checked_at", 0.0) < 60.0:
        return memo["sha1"]
    sig = _stat_signature(paths)
    if memo is not None and memo["sig"] == sig:
        memo["checked_at"] = _time.time()
        return memo["sha1"]
    cache_file = _fingerprint_cache_file()
    disk: dict = {}
    if cache_file.is_file():
        try:
            disk = json.loads(cache_file.read_text())
        except (OSError, ValueError):
            disk = {}                      # unreadable cache: recompute
    entry = disk.get(kind)
    if not (isinstance(entry, dict) and entry.get("sig") == sig):
        entry = {"sig": sig, "sha1": _content_sha1(paths)}
        disk[kind] = entry
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".json.tmp.%d" % os.getpid())
        tmp.write_text(json.dumps(disk, indent=1, sort_keys=True))
        os.replace(tmp, cache_file)
    import time as _time2
    _MEMO[kind] = dict(entry, checked_at=_time2.time())
    return entry["sha1"]


def chem_fingerprint() -> dict:
    """Provider fingerprint for the model cache key.

    {"picaso_version", "chemgrid_sha1"}: version of the installed picaso
    dist + content sha1 (16 hex) over all 78 chemistry node files and
    version.md. Raises when the tree is absent/incomplete.
    """
    root = refdata_root()
    grid = sorted((root / CHEM_GRID_REL).glob("sonora_2121grid_*.txt"))
    if len(grid) != CHEM_GRID_N_FILES:
        raise RuntimeError(
            f"chemistry grid at {root / CHEM_GRID_REL} has {len(grid)} node "
            f"files, expected {CHEM_GRID_N_FILES} (13 [M/H] x 6 C/O). "
            "Refusing an incomplete grid.")
    files = grid + [root / "version.md"]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise RuntimeError(f"PICASO reference tree incomplete: {missing}")
    return {"picaso_version": picaso_version(),
            "chemgrid_sha1": _cached_content_sha1("chem_grid", files)[:16]}


def climate_refdata_fingerprint(node: str, tio_vo: bool) -> str:
    """Content fingerprint (16 hex) of EVERYTHING the climate solve reads.

    Covers: the selected CK table, both continuum DBs, the adiabat table,
    the 661 wavenumber grid, config.json, version.md, and a name+size+mtime
    manifest of the ck04models stellar grid (2.8 GB -- manifest, not full
    content; any file swap changes the manifest).
    """
    root = refdata_root()
    content_files = [
        ck_path(node, tio_vo, root),
        root / CONTINUUM_REL,
        root / CLIMATE_INPUTS_REL / "ck_cx_cont_opacities_661.db",
        root / CLIMATE_INPUTS_REL / "specific_heat_p_adiabat_grad.json",
        root / CLIMATE_INPUTS_REL / "wvno_661",
        root / "config.json",
        root / "version.md",
    ]
    missing = [str(p) for p in content_files if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"climate reference data incomplete under {root}: missing "
            f"{missing}")
    ck04 = root / STELLAR_TRDS_REL / "grid" / "ck04models"
    if not ck04.is_dir():
        raise RuntimeError(
            f"stellar grid missing: {ck04} (the climate star() irradiation "
            "reads the ck04models atlas).")
    # name+size MANIFEST (deliberately no mtime -- mtimes change on every
    # copy); documented as a manifest, never claimed as a content hash.
    # Computed ONCE per process: the rglob is hundreds of stat calls and this
    # runs inside canonical_params on every GUI rerun in climate mode.
    _st_key = str(ck04)
    stellar_manifest = _MEMO.get("stellar:" + _st_key)
    if stellar_manifest is None:
        h_st = hashlib.sha1()
        for p in sorted(ck04.rglob("*.fits")):
            h_st.update(f"{p.name}\0{p.stat().st_size}\n".encode())
        stellar_manifest = h_st.hexdigest()
        _MEMO["stellar:" + _st_key] = stellar_manifest
    kind = f"climate:{node}:{'tiovo' if tio_vo else 'notiovo'}"
    content = _cached_content_sha1(kind, content_files)
    h = hashlib.sha1((content + stellar_manifest).encode())
    return h.hexdigest()[:16]


def native_opacity_path(root: Path | None = None) -> Path:
    """The native-RT opacity DB (parity script only). Loud when absent."""
    root = refdata_root() if root is None else root
    p = root / NATIVE_OPACITY_REL
    if not p.is_file():
        raise RuntimeError(
            f"native-RT opacity DB not found: {p}. Only the offline parity "
            "script needs it; the production PICASO provider does not.")
    return p
