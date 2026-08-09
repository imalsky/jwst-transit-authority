"""NASA Exoplanet Archive lookup for the custom-planet form.

The GUI fills the custom planet's system parameters from a SHIPPED SNAPSHOT of
the archive's PSCompPars table (one adopted row per planet), committed next to
this module and refreshed per release with ``jwst-tool archive-refresh`` --
the only step that touches the network. Runtime lookups never query the
archive: the snapshot date is disclosed in the GUI, and a planet announced
after the shipped snapshot is entered by hand like before.

Fill policy (standing loud-errors rule): values outside the custom form's
bounds (``planets.CUSTOM_FIELD_RANGES``) and values missing from the archive
row are NEVER clamped or guessed -- the field is left unchanged and the
refusal is reported by name. An out-of-range value written into Streamlit
session state would crash the widget at instantiation, so the range check is
a crash guard, not politeness.

Stdlib-only at module scope (light CI installs no pandas/astropy/requests);
``urllib`` is imported lazily inside the refresh path, fetch.py-style.
"""
from __future__ import annotations

import csv
import functools
import re
from dataclasses import dataclass
from pathlib import Path

from jwst_tool import planets

SNAPSHOT_PATH = Path(__file__).resolve().parent / "exoplanet_archive_pscomppars.csv"
TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Column order is the file format; the loader and the refresh both refuse any
# drift (schema drift means the mapping below needs review, never a silent
# reinterpretation). st_metratio / pl_bmassprov ride along purely for honest
# disclosure ([M/H] vs [Fe/H]; Msini vs true mass behind the derived gravity).
SNAPSHOT_COLUMNS = (
    "pl_name", "st_spectype", "st_teff", "st_logg", "st_met", "st_metratio",
    "st_rad", "sy_kmag", "pl_radj", "pl_bmassj", "pl_bmassprov",
    "pl_orbsmax", "pl_trandur",
)

# Transiting planets with the four fields without which a fill is meaningless;
# everything else stays nullable and is disclosed by name when missing.
SNAPSHOT_ADQL = (
    "select " + ",".join(SNAPSHOT_COLUMNS) + " from pscomppars "
    "where tran_flag=1 and st_teff is not null and st_rad is not null "
    "and pl_radj is not null and pl_orbsmax is not null order by pl_name"
)

_MIN_ROWS = 3000   # refuse a truncated/outage TAP response


class SnapshotError(RuntimeError):
    """The shipped snapshot is missing or malformed (loud, never a fallback)."""


@dataclass(frozen=True)
class Snapshot:
    fetched_utc: str            # ISO timestamp from the '# fetched:' header
    names: tuple[str, ...]      # archive pl_name values, file (= pl_name) order
    rows: dict                  # normalize_name(pl_name) -> {column: str}


def normalize_name(name: str) -> str:
    """Archive-name key: casefold, collapse whitespace, and insert the space
    before a trailing planet letter when it follows a digit or ')' -- so
    "WASP-39b", "wasp-39 B", and " WASP-39  b " all key "wasp-39 b"."""
    n = " ".join(str(name).split()).casefold()
    return re.sub(r"(?<=[0-9)])([b-i])$", r" \1", n)


@functools.lru_cache(maxsize=4)
def load_snapshot(path: str | None = None) -> Snapshot:
    """Parse the shipped snapshot (or ``path``); raises SnapshotError on a
    missing file, missing provenance, header drift, or a name collision."""
    p = Path(path) if path else SNAPSHOT_PATH
    if not p.is_file():
        raise SnapshotError(
            f"archive snapshot not found at {p}; the package ships it -- "
            "reinstall, or regenerate with `jwst-tool archive-refresh`.")
    fetched = ""
    names: list[str] = []
    rows: dict = {}
    body: list[str] = []
    with p.open(newline="", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                m = re.match(r"#\s*fetched:\s*(\S+)", line)
                if m:
                    fetched = m.group(1)
                continue
            body.append(line)
    if not fetched:
        raise SnapshotError(
            f"{p} has no '# fetched:' provenance line; regenerate with "
            "`jwst-tool archive-refresh` (never hand-edit the snapshot).")
    reader = csv.reader(body)
    header = tuple(next(reader, ()))
    if header != SNAPSHOT_COLUMNS:
        raise SnapshotError(
            f"{p} header {header} != expected {SNAPSHOT_COLUMNS}; the archive "
            "schema drifted -- update archive.py's mapping and regenerate.")
    for cells in reader:
        if not cells:
            continue
        row = dict(zip(SNAPSHOT_COLUMNS, cells))
        key = normalize_name(row["pl_name"])
        if key in rows:
            raise SnapshotError(
                f"{p}: two rows normalize to {key!r} "
                f"({rows[key]['pl_name']!r} and {row['pl_name']!r}).")
        rows[key] = row
        names.append(row["pl_name"])
    if len(names) < _MIN_ROWS:
        raise SnapshotError(
            f"{p} has only {len(names)} rows (expected >= {_MIN_ROWS}); "
            "likely a truncated refresh -- regenerate.")
    return Snapshot(fetched_utc=fetched, names=tuple(names), rows=rows)


def lookup(name: str, path: str | None = None) -> dict:
    """The snapshot row for ``name`` (normalized); KeyError names the input."""
    snap = load_snapshot(path)
    key = normalize_name(name)
    if key not in snap.rows:
        raise KeyError(
            f"planet {name!r} is not in the shipped archive snapshot "
            f"(fetched {snap.fetched_utc}); enter the parameters by hand, or "
            "refresh the snapshot for a newly announced planet.")
    return snap.rows[key]


def _cell(row: dict, col: str) -> float | None:
    v = str(row.get(col, "")).strip()
    if not v:
        return None
    return float(v)


def custom_fill(row: dict) -> tuple[dict, list[str]]:
    """Map an archive row onto the custom form's widget fields.

    Returns ``(values, notes)``: ``values`` keys are the widget-key suffixes
    (teff, logg, feh, ks, rstar, rp, g, a, t14, sflux) in widget units
    (g in m s^-2); ``notes`` are per-field disclosures (missing archive
    values, out-of-range refusals, mass provenance, metallicity basis, the
    nearest-spectral-type UV choice). Never clamps, never guesses."""
    name = row.get("pl_name", "this planet")
    direct = [("teff", "st_teff", "stellar Teff (K)"),
              ("logg", "st_logg", "stellar log g"),
              ("feh", "st_met", "stellar metallicity (dex)"),
              ("ks", "sy_kmag", "Ks magnitude"),
              ("rstar", "st_rad", "stellar radius (R_sun)"),
              ("rp", "pl_radj", "planet radius (R_jup)"),
              ("a", "pl_orbsmax", "semi-major axis (AU)"),
              ("t14", "pl_trandur", "transit duration (hours)")]
    values: dict = {}
    notes: list[str] = []

    def _take(field: str, val: float | None, label: str) -> None:
        if val is None:
            notes.append(f"the archive has no {label} for {name}; "
                         "field left unchanged.")
            return
        lo, hi = planets.CUSTOM_FIELD_RANGES[field]
        if not (lo <= val <= hi):
            notes.append(f"{label} {val:g} is outside this tool's supported "
                         f"range {lo:g}-{hi:g}; field left unchanged.")
            return
        values[field] = float(val)

    for field, col, label in direct:
        _take(field, _cell(row, col), label)

    # gravity, derived from the archive best mass + radius in widget units
    mass = _cell(row, "pl_bmassj")
    radj = _cell(row, "pl_radj")
    if mass is None or radj is None or radj <= 0:
        notes.append(f"the archive has no mass and radius pair for {name}; "
                     "surface gravity left unchanged.")
    else:
        g_ms2 = (planets.G_CGS * mass * planets.M_JUP_G
                 / (radj * planets.R_JUP_CM) ** 2) / 100.0
        _take("g", g_ms2, "surface gravity (m s^-2, from archive mass+radius)")
        prov = str(row.get("pl_bmassprov", "")).strip()
        if "g" in values and prov and prov.lower() != "mass":
            notes.append(f"gravity uses the archive best mass, provenance "
                         f"{prov!r} (not a directly measured true mass).")

    if "feh" in values:
        ratio = str(row.get("st_metratio", "")).strip()
        if ratio and ratio != "[Fe/H]":
            notes.append(f"archive metallicity is {ratio}, entered as [Fe/H] "
                         "(PHOENIX-node selector only; sub-percent effect).")

    if "teff" in values:
        sflux = planets.nearest_sflux(values["teff"])
        values["sflux"] = sflux
        spectype = str(row.get("st_spectype", "")).strip()
        stype = f"; archive spectral type {spectype}" if spectype else ""
        notes.append(
            f"UV spectrum set to {planets.SFLUX_CHOICES[sflux]}, the nearest "
            f"shipped spectral type for Teff {values['teff']:.0f} K{stype}. "
            "The archive carries no UV spectra; override freely.")
    else:
        notes.append("stellar Teff was not filled, so the UV spectrum menu "
                     "was left unchanged.")
    return values, notes


def refresh_snapshot(dest: Path | None = None, url: str = TAP_SYNC_URL) -> tuple[int, int]:
    """Fetch a fresh PSCompPars snapshot (maintainer/release step, network).

    Deterministic output (ADQL orders by pl_name; fixed column order; cells
    verbatim) so the per-release diff shows real archive changes only.
    Returns (rows, bytes) written. Loud on HTTP errors, header drift, a
    too-small response, or a normalized-name collision; atomic replace."""
    import datetime
    import os
    import tempfile
    import urllib.parse
    import urllib.request

    dest = Path(dest) if dest else SNAPSHOT_PATH
    q = urllib.parse.urlencode({"query": SNAPSHOT_ADQL, "format": "csv"})
    req = urllib.request.Request(f"{url}?{q}",
                                 headers={"User-Agent":
                                          "vulcan-jwst-tool archive-refresh"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        text = resp.read().decode("utf-8")
    rows = [r for r in csv.reader(text.splitlines()) if r]
    if not rows or tuple(c.strip() for c in rows[0]) != SNAPSHOT_COLUMNS:
        got = tuple(rows[0]) if rows else ()
        raise SnapshotError(
            f"TAP response header {got} != expected {SNAPSHOT_COLUMNS}; the "
            "archive schema drifted -- update archive.py before shipping.")
    body = rows[1:]
    if len(body) < _MIN_ROWS:
        raise SnapshotError(
            f"TAP response has only {len(body)} rows (expected >= "
            f"{_MIN_ROWS}); refusing to write a truncated snapshot.")
    seen: dict = {}
    for r in body:
        key = normalize_name(r[0])
        if key in seen:
            raise SnapshotError(
                f"two archive rows normalize to {key!r} ({seen[key]!r} and "
                f"{r[0]!r}); tighten normalize_name before shipping.")
        seen[key] = r[0]
    fetched = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            fh.write("# NASA Exoplanet Archive PSCompPars snapshot "
                     "(shipped with the release)\n")
            fh.write(f"# fetched: {fetched}\n")
            fh.write(f"# endpoint: {url}\n")
            fh.write(f"# query: {SNAPSHOT_ADQL}\n")
            fh.write(f"# rows: {len(body)}\n")
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(SNAPSHOT_COLUMNS)
            w.writerows(body)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    load_snapshot.cache_clear()   # a running GUI still needs a restart
    return len(body), dest.stat().st_size


def run_refresh(argv: list[str]) -> int:
    """``jwst-tool archive-refresh [--dest PATH]`` entry point."""
    dest = None
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "--dest" and args:
            dest = Path(args.pop(0))
        else:
            print(f"archive-refresh: unknown argument {a!r}\n"
                  "usage: jwst-tool archive-refresh [--dest PATH]")
            return 2
    try:
        n, size = refresh_snapshot(dest=dest)
    except Exception as e:                      # loud, single line + type
        print(f"archive-refresh FAILED: {type(e).__name__}: {e}")
        return 1
    out = dest if dest else SNAPSHOT_PATH
    print(f"archive-refresh: wrote {n} planets ({size} bytes) to {out}")
    print("Review the diff and commit; the GUI discloses the fetched date.")
    return 0
