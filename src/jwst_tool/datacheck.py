"""Data-availability detection for JWST Transit Authority.

Knows every external dataset the tool touches, probes the filesystem, and
reports a structured status: present, missing, or fetched-on-first-use.
Consumers: the ``jwst-tool data`` CLI, the GUI data-status panel and widget
annotations, and the unit tests (every check takes explicit paths).

Detection never REPLACES the loud runtime failures (repo standing rule);
it only makes the state visible up front, with the exact remedy.

Import discipline: stdlib only (no numpy/jax/streamlit); the one non-stdlib
touch is an optional ``import jwst_tool.engine_config``, whose RuntimeError
text IS the status detail.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jwst_tool import instruments as ins
from jwst_tool import planets

# Report structure

#: OK = present; MISSING = absent, manual fetch; AUTO = fetched
#: automatically on first use (network required then)
OK, MISSING, AUTO = "ok", "missing", "on-first-use"


@dataclass
class Item:
    key: str            # stable machine key, e.g. "ktable:NH3"
    label: str          # human name, e.g. "NH3 ExoMolOP k-table"
    status: str         # OK / MISSING / AUTO
    required: bool      # required for ANY run (vs opt-in feature data)
    detail: str         # what was found / which path is missing
    remedy: str = ""    # exact command / URL / pointer to get it


def _found(path: Path) -> str:
    return f"found: {path}"


# Python stack (this environment)

_STACK = (
    ("vulcan_jax", True,
     "pip install -e <PROJECT_ROOT>/VULCAN-JAX --no-deps   (or from TestPyPI: "
     "pip install -i https://test.pypi.org/simple/ vulcan-jax)"),
    ("vulcan_forward", True,
     "pip install -e <PROJECT_ROOT>/vulcan-forward --no-deps   (dist name "
     "vulcan-forward; the shared chemistry + radiative-transfer engine)"),
    ("exojax", True, "pip install exojax"),
    ("jax", True, "pip install jax"),
    ("streamlit", False, "pip install streamlit pandas   (GUI only)"),
    ("pandas", False, "pip install streamlit pandas   (GUI only)"),
)


def check_python_stack() -> list[Item]:
    items = []
    for mod, required, remedy in _STACK:
        try:
            present = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            present = False
        items.append(Item(
            key=f"pkg:{mod}", label=f"python package {mod}",
            status=OK if present else MISSING, required=required,
            detail=("importable" if present else
                    "not importable in this environment"),
            remedy="" if present else remedy))
    return items


# Chemistry / RT engine data (the shared vulcan-forward engine's data trees)

def _engine_config():
    """The engine-config view, or an exception instance whose text is the
    status detail (an unset/wrong $VULCAN_FORWARD_DATA raises here)."""
    try:
        mod = importlib.import_module("jwst_tool.engine_config")
        _ = mod.DATA_DIR        # probe the engine's data-root contract
        return mod
    except Exception as e:      # ImportError, or the data-root RuntimeError
        return e


def exomolop_table_path(mol: str) -> Path | None:
    """The ExoMolOP k-table file the engine reads for ``mol`` (None when the
    engine config is unavailable). Filename convention is the engine's
    ``exomolop.table_path``; a data-gated test pins the two agree."""
    cfg = _engine_config()
    if isinstance(cfg, Exception):
        return None
    return Path(cfg.EXOMOLOP_DIR) / f"{mol}.ktable.h5"


def exomolop_provenance_path() -> Path | None:
    """The fetcher's provenance.json beside the k-tables (dataset,
    isotopologue, URL per species; None when the engine config is
    unavailable). The GUI keys its source cache on this file's mtime."""
    cfg = _engine_config()
    if isinstance(cfg, Exception):
        return None
    return Path(cfg.EXOMOLOP_DIR) / "provenance.json"


def exomolop_table_status(mols: list[str]) -> dict[str, str]:
    """{molecule: OK/MISSING} for the ExoMolOP k-tables.

    Never AUTO: the engine NEVER downloads a k-table at run time (a missing
    one raises with the fetch command), so absence means the run will stop.
    """
    out = {}
    for m in mols:
        p = exomolop_table_path(m)
        out[m] = OK if (p is not None and p.is_file()) else MISSING
    return out


def check_engine_data(base_mols: list[str], extra_mols: list[str]) -> list[Item]:
    cfg = _engine_config()
    if isinstance(cfg, Exception):
        return [Item(
            key="engine:config", label="forward-model engine data root",
            status=MISSING, required=True, detail=str(cfg),
            remedy="Set VULCAN_FORWARD_DATA to the directory holding the "
                   "engine's exomolop/ and opacity_cache/ trees "
                   "(vulcan-forward never bundles them; `jwst-tool fetch` "
                   "downloads the CIA tables, fetch_exomolop the "
                   "k-tables).")]

    items = [Item(
        key="engine:config", label="forward-model engine data root",
        status=OK, required=True, detail=_found(Path(cfg.DATA_DIR)))]

    h2h2 = Path(cfg.CIA_H2H2_FILE)
    items.append(Item(
        key="cia:H2-H2", label="H2-H2 collision-induced absorption table",
        status=OK if h2h2.is_file() else AUTO, required=True,
        detail=_found(h2h2) if h2h2.is_file() else f"absent: {h2h2}",
        remedy="ExoJAX fetches it on first use (~24 MB), or download "
               f"https://hitran.org/data/CIA/main/H2-H2_2011.cia to {h2h2}"))

    h2he = Path(cfg.CIA_H2HE_FILE)
    items.append(Item(
        key="cia:H2-He", label="H2-He collision-induced absorption table",
        status=OK if h2he.is_file() else MISSING, required=True,
        detail=_found(h2he) if h2he.is_file() else f"absent: {h2he}",
        remedy="Manual, one-time, ~147 MB (the RT refuses to build without "
               "it): download https://hitran.org/data/CIA/main/H2-He_2011.cia "
               f"to {h2he} (the /main/ path segment is required)."))

    # ExoMolOP k-tables: the gas opacity every run reads, ~389 MB per
    # species. Without this check `jwst-tool data` reports green while the
    # data behind every run is entirely absent. Never AUTO -- the engine
    # refuses to download at run time. Species with no published table
    # (forward._NO_EXOMOLOP_TABLE, refused by canonical_params) are skipped.
    from jwst_tool.forward import _NO_EXOMOLOP_TABLE
    ktable_mols = [m for m in base_mols + extra_mols
                   if m not in _NO_EXOMOLOP_TABLE]
    kstat = exomolop_table_status(ktable_mols)
    fetch_missing = [m for m in ktable_mols if kstat[m] != OK]
    for mol in ktable_mols:
        p = exomolop_table_path(mol)
        items.append(Item(
            key=f"ktable:{mol}", label=f"{mol} ExoMolOP k-table",
            status=kstat[mol], required=mol in base_mols,
            detail=(_found(p) if kstat[mol] == OK else f"absent: {p}"),
            remedy="" if kstat[mol] == OK else
                   "python -m vulcan_forward.fetch_exomolop --molecules "
                   + ",".join(fetch_missing)))
    prov = exomolop_provenance_path()
    items.append(Item(
        key="ktable:provenance",
        label="ExoMolOP k-table provenance (dataset, isotopologue, URL per "
              "species)",
        status=OK if prov.is_file() else MISSING, required=False,
        detail=_found(prov) if prov.is_file() else f"absent: {prov}",
        remedy="" if prov.is_file() else
               "python -m vulcan_forward.fetch_exomolop --molecules "
               + ",".join(ktable_mols)
               + " (records tables already on disk without re-downloading)"))
    return items


# VULCAN-JAX stellar UV spectra (ship inside the vulcan_jax package)

def _vulcan_pkg_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("vulcan_jax")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def vulcan_atm_dir() -> Path | None:
    pkg = _vulcan_pkg_dir()
    return None if pkg is None else pkg / "atm"


def check_fastchem() -> list[Item]:
    """The FastChem equilibrium-init binary: compiled on demand (make + C++)."""
    pkg = _vulcan_pkg_dir()
    if pkg is None:
        return []                    # covered by the python-stack section
    binary = pkg / "fastchem_vulcan" / "fastchem"
    return [Item(
        key="fastchem:binary", label="FastChem binary (equilibrium initializer)",
        status=OK if binary.is_file() else AUTO, required=True,
        detail=_found(binary) if binary.is_file() else f"absent: {binary}",
        remedy="Compiled automatically on the first equilibrium-init run "
               "(every shipped planet config); needs `make` and a C++ "
               "compiler on PATH. No download involved.")]


def uv_spectra_status() -> dict[str, bool]:
    """{sflux filename: present} for every registry stellar UV spectrum."""
    atm = vulcan_atm_dir()
    return {f: (atm is not None and (atm / "stellar_flux" / f).is_file())
            for f in planets.SFLUX_CHOICES}


def check_stellar_uv() -> list[Item]:
    atm = vulcan_atm_dir()
    if atm is None:
        return [Item(
            key="uv:package", label="stellar UV spectra (vulcan_jax/atm)",
            status=MISSING, required=True,
            detail="vulcan_jax is not importable, so its shipped spectra "
                   "cannot be located",
            remedy="Install vulcan-jax (see the python-stack section).")]
    items = []
    for fname, label in planets.SFLUX_CHOICES.items():
        p = atm / "stellar_flux" / fname
        items.append(Item(
            key=f"uv:{fname}", label=f"UV spectrum {label}",
            status=OK if p.is_file() else MISSING, required=True,
            detail=_found(p) if p.is_file() else f"absent: {p}",
            remedy="" if p.is_file() else
                   "Ships inside the vulcan-jax package (atm/stellar_flux/); "
                   "a missing file means a broken/partial install -- "
                   "reinstall vulcan-jax."))
    return items


# Pandeia noise backend + synphot CDBS

def _refdata_version(refdata: Path) -> str | None:
    for name in ("VERSION", "VERSION_DATA"):
        f = refdata / name
        if f.is_file():
            try:
                return f.read_text().strip().splitlines()[0]
            except OSError:
                return None
    return None


def check_pandeia_backend(python: str | Path = None,
                          refdata: str | Path = None,
                          psf_dir: str | Path | None = None) -> list[Item]:
    """The ACTIVE backend's three path roots (mirrors the worker preflight)."""
    # an unset backend python is a MISSING item, never a crash: this reports
    _py = python if python is not None else ins.PANDEIA_PYTHON
    python = Path(_py) if _py else None
    refdata = Path(refdata if refdata is not None else ins.PANDEIA_REFDATA)
    psf_dir = (ins.PANDEIA_PSF_DIR if psf_dir is None else psf_dir) or ""

    _py_ok = python is not None and python.exists()
    items = [Item(
        key="pandeia:python", label="Pandeia engine environment (python)",
        status=OK if _py_ok else MISSING, required=True,
        detail=(_found(python) if _py_ok else
                (f"absent: {python}" if python is not None
                 else "JWST_TOOL_PANDEIA_PYTHON not set (no default: the "
                      "backend environment is machine-specific)")),
        remedy=("Create a conda env with the matching engine, e.g.  "
                f"conda create -n pandeia_{ins.BACKEND_RELEASE.replace('.', '_')} "
                "python=3.11  then  <env>/bin/pip install "
                f"pandeia.engine=={ins.BACKEND_RELEASE}  and point "
                "JWST_TOOL_PANDEIA_PYTHON at its python."))]

    ver = _refdata_version(refdata)
    items.append(Item(
        key="pandeia:refdata", label="Pandeia JWST reference data",
        status=OK if refdata.is_dir() else MISSING, required=True,
        detail=(f"{_found(refdata)}" + (f" (version {ver})" if ver else ""))
               if refdata.is_dir() else f"absent: {refdata}",
        remedy=(
            "Download the reference tree for the SAME release as the engine "
            f"(this backend: pandeia_data-{ins.BACKEND_RELEASE}-jwst, ~20 MiB) "
            "and extract it to the path above (or point "
            "JWST_TOOL_PANDEIA_REFDATA elsewhere). Links are on STScI's "
            "installation page: https://outerspace.stsci.edu/spaces/PEN/pages/"
            "77530136/Pandeia%2BEngine%2BInstallation -- the engine, this "
            "tree, and the PSF tree must all be the same release.")))

    if psf_dir:                                  # split-PSF layout (>= 2026)
        pv = Path(psf_dir) / "VERSION_PSF"       # worker preflight checks this
        # report the PSF RELEASE, not just presence: a release mismatch is
        # exactly what the worker refuses, so it must show here
        psf_ver = None
        if pv.is_file():
            try:
                psf_ver = pv.read_text().splitlines()[0].strip() or None
            except OSError:
                psf_ver = None
        _psf_matches = (psf_ver or "").startswith(ins.BACKEND_RELEASE)
        items.append(Item(
            key="pandeia:psf", label="Pandeia PSF library (split layout)",
            status=(OK if (pv.is_file() and _psf_matches) else MISSING),
            required=True,
            detail=(
                f"{_found(Path(psf_dir))} (release {psf_ver})"
                if pv.is_file() and _psf_matches else
                (f"RELEASE MISMATCH: PSF tree is {psf_ver}, backend needs "
                 f"{ins.BACKEND_RELEASE} -- {psf_dir}" if pv.is_file() else
                 f"no VERSION_PSF under: {psf_dir}")),
            remedy=(
                f"Download pandeia_psfs-{ins.BACKEND_RELEASE}-jwst (~4 GiB), "
                "the SAME release as the engine and reference data, and "
                "extract it to the path above (or point "
                "JWST_TOOL_PANDEIA_PSF_DIR elsewhere). Links are on STScI's "
                "installation page: https://outerspace.stsci.edu/spaces/PEN/"
                "pages/77530136/Pandeia%2BEngine%2BInstallation")))
    else:
        # every supported backend uses the split-PSF layout; an empty PSF dir
        # is a misconfiguration, never "PSFs embedded in refdata"
        items.append(Item(
            key="pandeia:psf", label="Pandeia PSF library (split layout)",
            status=MISSING, required=True,
            detail="no PSF directory configured (JWST_TOOL_PANDEIA_PSF_DIR "
                   "is empty)",
            remedy=(
                f"Download pandeia_psfs-{ins.BACKEND_RELEASE}-jwst (~4 GiB) "
                "and point JWST_TOOL_PANDEIA_PSF_DIR at it. Links are on "
                "STScI's installation page: https://outerspace.stsci.edu/"
                "spaces/PEN/pages/77530136/Pandeia%2BEngine%2BInstallation")))
    return items


def probe_pandeia_engine(python: str | Path = None,
                         timeout: float = 60.0) -> str:
    """Ask the backend env for its pandeia.engine version (slow: subprocess).

    Returns the version string, or an 'unavailable (...)' explanation.
    Never raises -- this is a report, the run path has its own loud gate.
    """
    _py = python if python is not None else ins.PANDEIA_PYTHON
    if not _py:
        return ("unavailable (JWST_TOOL_PANDEIA_PYTHON is not set; the backend "
                "environment is machine-specific and has no default)")
    python = Path(_py)
    if not python.exists():
        return f"unavailable (no python at {python})"
    try:
        r = subprocess.run(
            [str(python), "-c",
             "import pandeia.engine; print(pandeia.engine.__version__)"],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"unavailable (probe timed out after {timeout:g} s)"
    if r.returncode != 0 or not r.stdout.strip():
        tail = (r.stderr or "").strip().splitlines()
        return ("unavailable (pandeia.engine not importable in that env"
                + (f": {tail[-1]}" if tail else "") + ")")
    return r.stdout.strip().splitlines()[-1]


def check_synphot_cdbs(cdbs: str | Path = None) -> list[Item]:
    """The minimal synphot tree the worker preflights (star SED + Ks norm)."""
    cdbs = Path(cdbs if cdbs is not None else ins.PYSYN_CDBS)
    items = []
    phx = cdbs / "grid" / "phoenix"
    phx_real = Path(os.path.realpath(phx))
    phx_ok = phx_real.is_dir()
    items.append(Item(
        key="cdbs:phoenix", label="PHOENIX stellar grid (synphot)",
        status=OK if phx_ok else MISSING, required=True,
        detail=(f"found: {phx} -> {phx_real}" if phx_ok else
                f"missing or dangling symlink: {phx} -> {phx_real}"),
        remedy="Fetch the STScI reference-atlases PHOENIX tarball (~1.9 GB): "
               "https://archive.stsci.edu/hlsps/reference-atlases/"
               "hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_"
               "synphot5.tar ('pheonix' spelling is STScI's own), extract, "
               f"and place its grp/redcat/trds/grid/phoenix tree at {phx} "
               "(a real directory is fine; the shipped symlink just points "
               "at an existing local copy)."))
    # the CALSPEC Vega copy is only an offline pin for the tool-side
    # synphot/stsynphot; the engine loads its own refdata Vega
    for rel, label, required, remedy in (
            (Path("comp") / "nonhst" / "2mass_ks_001_syn.fits",
             "2MASS Ks bandpass (Ks normalization)", True,
             "Ships with the repo (data/cdbs/comp/); restore it from git or "
             "fetch https://ssb.stsci.edu/trds/comp/nonhst/"
             "2mass_ks_001_syn.fits (8.6 KB)."),
            (Path("calspec") / "alpha_lyr_stis_011.fits",
             "Vega spectrum (offline pin for tool-side synphot; the engine "
             "uses its own refdata Vega)", False,
             "Fetch https://ssb.stsci.edu/trds/calspec/alpha_lyr_stis_011.fits"
             f" (288 KB) to {cdbs / 'calspec'}/")):
        p = cdbs / rel
        items.append(Item(
            key=f"cdbs:{rel.name}", label=label,
            status=OK if p.is_file() else MISSING, required=required,
            detail=_found(p) if p.is_file() else f"absent: {p}",
            remedy="" if p.is_file() else remedy))
    return items


# Generated caches (informational)

def cache_stats() -> dict:
    def _stat(d: Path, glob: str):
        files = list(d.glob(glob)) if d.is_dir() else []
        return {"n": len(files),
                "mb": round(sum(f.stat().st_size for f in files) / 2**20, 1)}
    return {"model_cache": _stat(ins.MODEL_CACHE, "*.npz"),
            "noise_cache": _stat(ins.NOISE_CACHE, "*.json")}


# Full report + rendering

def full_report(base_mols: list[str] = None, extra_mols: list[str] = None,
                deep: bool = False) -> dict:
    """Every section's items, plus cache stats.

    ``base_mols``/``extra_mols`` default to the forward model's sets (passed
    in to keep this module import-independent of forward.py for tests).
    ``deep`` adds a subprocess probe of the Pandeia env's engine version.
    """
    if base_mols is None or extra_mols is None:
        from jwst_tool import forward
        base_mols = forward.MOLECULES if base_mols is None else base_mols
        extra_mols = forward.EXTRA_MOLECULES if extra_mols is None else extra_mols

    sections = {
        "Python stack (this environment)": check_python_stack(),
        "Chemistry / RT engine data": check_engine_data(base_mols, extra_mols),
        "Chemistry equilibrium initializer": check_fastchem(),
        "Stellar UV spectra (photochemistry)": check_stellar_uv(),
        f"Pandeia noise backend ({ins.JWST_TOOL_BACKEND})":
            check_pandeia_backend(),
        "Star normalization data (synphot CDBS)": check_synphot_cdbs(),
    }
    report = {"backend": ins.JWST_TOOL_BACKEND,
              "backend_status": ins.BACKEND_STATUS,
              "sections": sections, "caches": cache_stats()}
    if deep:
        report["engine_probe"] = probe_pandeia_engine()
    return report


def all_items(report: dict) -> list[Item]:
    return [it for items in report["sections"].values() for it in items]


def required_ok(report: dict) -> bool:
    """True when every REQUIRED item is present or auto-fetches on first use."""
    return all(it.status != MISSING for it in all_items(report)
               if it.required)


def missing_required(report: dict) -> list[Item]:
    return [it for it in all_items(report)
            if it.required and it.status == MISSING]


_STATUS_TEXT = {OK: "OK", MISSING: "MISSING",
                AUTO: "downloads on first use"}


def format_report(report: dict) -> str:
    """Plain-text report for the CLI (one line per item + remedies)."""
    lines = [f"jwst-tool data status  (backend: {report['backend_status']})"]
    if "engine_probe" in report:
        lines.append(f"pandeia.engine version probe: {report['engine_probe']}")
    for name, items in report["sections"].items():
        lines += ["", name, "-" * len(name)]
        for it in items:
            flag = _STATUS_TEXT[it.status]
            opt = "" if it.required else "  [optional]"
            lines.append(f"  [{flag}]{opt} {it.label}")
            lines.append(f"      {it.detail}")
            if it.remedy and it.status != OK:
                lines.append(f"      how to get it: {it.remedy}")
    c = report["caches"]
    lines += ["", "Generated caches (safe to delete; rebuilt on demand)",
              "----------------------------------------------------",
              f"  model spectra: {c['model_cache']['n']} files, "
              f"{c['model_cache']['mb']} MB",
              f"  pandeia noise: {c['noise_cache']['n']} files, "
              f"{c['noise_cache']['mb']} MB"]
    miss = missing_required(report)
    lines += ["", ("All required data present." if not miss else
                   f"{len(miss)} required item(s) MISSING -- calculations "
                   "that need them will stop and show an error. Remedies "
                   "above.")]
    return "\n".join(lines)


# Disk-persisted report cache
# The full report walks every external dataset (slow on remote volumes), so
# it is persisted: the GUI serves the file when fresh, the Space entrypoint
# warms it in the background at boot, and the refresh button rebuilds.

REPORT_CACHE_FILE = Path(ins.OUTPUT_DIR) / "datacheck_report.json"
REPORT_CACHE_MAX_AGE_S = 3600.0


def _report_to_jsonable(report: dict) -> dict:
    from dataclasses import asdict
    out = dict(report)
    out["sections"] = {name: [asdict(it) for it in items]
                       for name, items in report["sections"].items()}
    return out


def _report_from_jsonable(d: dict) -> dict:
    out = dict(d)
    out["sections"] = {name: [Item(**it) for it in items]
                       for name, items in d["sections"].items()}
    return out


def load_cached_report(max_age_s: float = REPORT_CACHE_MAX_AGE_S):
    """The disk-cached report if present and fresh, else None."""
    import time
    try:
        if not REPORT_CACHE_FILE.is_file():
            return None
        env = json.loads(REPORT_CACHE_FILE.read_text())
        if time.time() - float(env.get("generated_at", 0.0)) > max_age_s:
            return None
        return _report_from_jsonable(env["report"])
    except (OSError, ValueError, KeyError, TypeError):
        return None                     # unreadable cache: recompute


def warm_report_cache(**kw) -> dict:
    """Compute the full report and persist it (atomic write)."""
    import os
    import time
    report = full_report(**kw)
    REPORT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_CACHE_FILE.with_suffix(".json.tmp.%d" % os.getpid())
    tmp.write_text(json.dumps({"generated_at": time.time(),
                               "report": _report_to_jsonable(report)}))
    os.replace(tmp, REPORT_CACHE_FILE)
    return report
