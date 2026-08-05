"""One-command data bootstrap: ``jwst-tool fetch``.

Downloads every missing dataset that has a public direct URL, skips whatever
is already present, and ends with exact instructions for the few pieces it
does not fetch itself: two STScI Box downloads (Box shared links serve HTML,
not the file, so they need either a browser or the markup-dependent recipe in
MANUAL) and the Pandeia conda environment. Line lists and the
ExoMol CO tables are left to their existing on-first-use fetchers, and the
FastChem binary compiles itself on the first run. Stdlib only, like
datacheck; every failure is loud and the command is idempotent.
"""
from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from jwst_tool import instruments as ins

_UA = {"User-Agent": "vulcan-jwst-tool data fetcher"}
_CHUNK = 1 << 20                       # 1 MiB read chunks
_PROGRESS_EVERY = 100                  # progress line every N chunks


def _engine_cfg():
    """The engine-config view, with the engine's data layout CREATED first.

    A setup command makes the directories it is about to download into
    (otherwise the CIA downloads fail on a correct first-run setup). The root
    itself is never guessed -- an unset $VULCAN_FORWARD_DATA raises.
    """
    import importlib

    from vulcan_forward import paths as _fwd_paths

    _fwd_paths.ensure_layout()
    return importlib.import_module("jwst_tool.engine_config")


@dataclass(frozen=True)
class Fetch:
    key: str                           # datacheck item key this satisfies
    label: str
    url: str
    dest: Callable[[], Path]           # resolved lazily (env-dependent roots)
    size: str                          # human download-size hint
    tar_subtree: str = ""              # extract only this prefix into dest


FETCHES = (
    Fetch("cia:H2-H2", "H2-H2 collision-induced absorption table",
          "https://hitran.org/data/CIA/main/H2-H2_2011.cia",
          lambda: Path(_engine_cfg().CIA_H2H2_FILE), "24 MB"),
    Fetch("cia:H2-He", "H2-He collision-induced absorption table",
          "https://hitran.org/data/CIA/main/H2-He_2011.cia",
          lambda: Path(_engine_cfg().CIA_H2HE_FILE), "147 MB"),
    Fetch("cdbs:2mass_ks_001_syn.fits", "2MASS Ks bandpass",
          "https://ssb.stsci.edu/trds/comp/nonhst/2mass_ks_001_syn.fits",
          lambda: Path(ins.PYSYN_CDBS) / "comp" / "nonhst"
          / "2mass_ks_001_syn.fits", "9 KB"),
    Fetch("cdbs:alpha_lyr_stis_011.fits", "Vega spectrum (CALSPEC)",
          "https://ssb.stsci.edu/trds/calspec/alpha_lyr_stis_011.fits",
          lambda: Path(ins.PYSYN_CDBS) / "calspec"
          / "alpha_lyr_stis_011.fits", "288 KB"),
    Fetch("cdbs:phoenix", "PHOENIX stellar grid (synphot)",
          "https://archive.stsci.edu/hlsps/reference-atlases/"
          "hlsp_reference-atlases_hst_multi_pheonix-models_multi_v3_"
          "synphot5.tar",
          lambda: Path(ins.PYSYN_CDBS) / "grid" / "phoenix",
          "1.9 GB download (STScI's own 'pheonix' spelling in the URL)",
          tar_subtree="grp/redcat/trds/grid/phoenix"),
)

# What fetch does not download itself, printed verbatim after the downloads.
# The two Pandeia pieces live on STScI Box, whose /v/ links serve HTML rather
# than the file; the scriptable recipe depends on Box's page markup, so it
# stays documented instead of automated.
MANUAL = """\
The Pandeia backend is a MATCHED TRIPLE: the engine, the reference data, and
the PSF library must all be release {release}. The worker refuses to run a
mixed set, so do not upgrade one without the others.

Two trees are not downloaded automatically. Both links are on STScI's
installation page, which lists the trees for the supported release:
  https://outerspace.stsci.edu/spaces/PEN/pages/77530136/Pandeia%2BEngine%2BInstallation

  1. Pandeia reference data, pandeia_data-{release}-jwst (~20 MB):
     extract to: {refdata}
  2. Pandeia PSF library, pandeia_psfs-{release}-jwst (~4 GB):
     extract to: {psf}

A browser is the simple path. To script them instead: the Box /v/ link
returns HTML that carries the file's "sharedName", its "typedID":"f_<id>",
and its exact "size"; feeding the first two to

  https://stsci.app.box.com/index.php?rm=box_download_shared_file\\
      &shared_name=<sharedName>&file_id=f_<id>

serves the tarball. The 4 GB transfer is cut short repeatedly, so loop with
curl -C - until the byte count on disk equals that "size". CHECK THE BYTE
COUNT: a truncated transfer still exits 0.

The Pandeia engine runs in its own conda environment (once):
  conda create -n pandeia_{env_suffix} python=3.12
  conda run -n pandeia_{env_suffix} pip install pandeia.engine=={release}
then point JWST_TOOL_PANDEIA_PYTHON at that environment's python.

Fetched automatically on first use, nothing to do now: HITRAN molecular
line lists, the ExoMol CO tables, and the FastChem binary (compiles itself;
needs `make` and a C++ compiler on PATH)."""


def _present(f: Fetch) -> bool:
    try:
        d = f.dest()
    except Exception:
        return False
    if f.tar_subtree:
        real = Path(os.path.realpath(d))
        return real.is_dir() and any(real.iterdir())
    return d.is_file()


def _download(url: str, out: Path, label: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as w:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        for i, chunk in enumerate(iter(lambda: r.read(_CHUNK), b"")):
            w.write(chunk)
            done += len(chunk)
            if i and i % _PROGRESS_EVERY == 0:
                pct = f" ({100 * done / total:.0f}%)" if total else ""
                print(f"    ... {done >> 20} MiB{pct}", flush=True)
    if total and done != total:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"{label}: truncated download ({done} of {total} bytes)")
    os.replace(tmp, out)


def _extract_subtree(tar_path: Path, subtree: str, dest: Path) -> int:
    """Extract members under ``subtree`` into ``dest`` (prefix stripped)."""
    prefix = subtree.rstrip("/") + "/"
    n = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.name.startswith(prefix) or not m.isfile():
                continue
            rel = m.name[len(prefix):]
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            with open(target, "wb") as w:
                shutil.copyfileobj(src, w)
            n += 1
    if n == 0:
        raise RuntimeError(f"no members under {subtree!r} in {tar_path}")
    return n


def _fetch_one(f: Fetch) -> None:
    dest = f.dest()
    if not f.tar_subtree:
        print(f"  downloading {f.label} ({f.size}) ...", flush=True)
        _download(f.url, dest, f.label)
        print(f"  -> {dest}", flush=True)
        return
    # tarball with a subtree to extract (PHOENIX): download to a temp file
    # next to the destination, extract, then delete the tarball
    dest_real = Path(os.path.realpath(dest))
    dest_real.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=dest_real.parent, suffix=".tar", delete=False) as t:
        tar_path = Path(t.name)
    try:
        print(f"  downloading {f.label} ({f.size}) ...", flush=True)
        _download(f.url, tar_path, f.label)
        print("  extracting ...", flush=True)
        n = _extract_subtree(tar_path, f.tar_subtree, dest_real)
        print(f"  -> {n} files under {dest_real}", flush=True)
    finally:
        tar_path.unlink(missing_ok=True)


def run_fetch() -> int:
    """Fetch every missing direct-URL dataset. Returns a shell exit code."""
    failures = []
    for f in FETCHES:
        try:
            if _present(f):
                print(f"  already present: {f.label}")
                continue
            _fetch_one(f)
        except Exception as e:
            failures.append((f, e))
            print(f"  FAILED: {f.label}: {e}", file=sys.stderr, flush=True)
    print()
    print(MANUAL.format(refdata=ins.PANDEIA_REFDATA,
                        release=ins.BACKEND_RELEASE,
                        env_suffix=ins.BACKEND_RELEASE.replace(".", "_"),
                        psf=ins.PANDEIA_PSF_DIR or "(backend has no split "
                        "PSF dir; not needed)"))
    if failures:
        print(f"\n{len(failures)} download(s) failed; re-run `jwst-tool "
              "fetch` to retry, or follow the item's remedy in "
              "`jwst-tool data`.", file=sys.stderr)
        return 1
    print("\nDone. Run `jwst-tool data` to see the full status report.")
    return 0
