"""Tabulated T-P/Kzz files: the shipped table per planet, uploads, parsing and validation."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

import numpy as np

from jwst_tool import instruments as _ins, planets
from jwst_tool.model_defs import CHEM_P_SPAN_DYN, RT_PTOP_RANGE, T_WINDOW, TP_FILE_SHIPPED, TP_FILE_UPLOAD

# Log-spaced pressures at which a table, interpolated linearly as the engine reads it, is checked against T_WINDOW
# over the run's span. No basis recorded.
T_CHECK_POINTS = 200

# Relative slack of the top-coverage gate: the canonical rt_ptop_bar keeps six significant figures, so a model top
# set to a table's top can sit up to 5e-6 (relative) above it once rounded.
TOP_RTOL = 1.0e-5

# tp_mode="file" helpers (light path: no vulcan3/jax imports)


def shipped_tp_table_name(planet: str) -> str:
    """Filename of the MEASURED structure table bundled for ``planet`` (its "tp_table"), or "" when it has none."""
    entry = planets.PLANETS.get(planet, {})
    return entry.get("tp_table") or ""


def shipped_tp_table_is_default(planet: str) -> bool:
    """Whether ``planet``'s bundled table is its DEFAULT structure.

    Separate from merely having one: a table becomes the default only once a default run on it is verified
    end-to-end here. A planet whose table the solver will not certify at default settings stays
    selectable-but-not-default rather than erroring on arrival (planets.PLANETS[...]["tp_table_note"] says why).
    """
    entry = planets.PLANETS.get(planet, {})
    return bool(entry.get("tp_table") and entry.get("tp_table_default"))


def declares_top_extension(planet: str) -> bool:
    """Whether ``planet``'s registry entry declares that its shipped table is held constant above its top."""
    return bool(planets.PLANETS.get(planet, {}).get("tp_table_extends_top"))


def shipped_tp_file(planet: str) -> Path:
    """Return the path of the shipped T-P/Kzz table for ``planet``, without importing vulcan3.

    find_spec only: importing parses the reaction network, far too heavy for the GUI's cache-key path. A planet with
    no usable table raises the reason, never substitutes another planet's atmosphere.
    """
    name = shipped_tp_table_name(planet)
    if not name:
        entry = planets.PLANETS.get(planet, {})
        why = entry.get("tp_table_note") or "no table is bundled for it."
        raise ValueError(
            f"tp_file='shipped' is not available for planet {planet!r}: {why} "
            "Choose tp_mode='guillot', or upload a table "
            "with tp_file='upload'."
        )
    import importlib.util

    spec = importlib.util.find_spec("vulcan3")
    if spec is None or not spec.origin:
        raise RuntimeError(
            "vulcan3 is not installed (or has no package origin): "
            f"tp_mode='file' with tp_file='shipped' needs its bundled "
            f"atm/{name} table."
        )
    return Path(spec.origin).parent / "atm" / name


def uploads_dir() -> Path:
    """Content-addressed home for uploaded T-P tables (sibling of model_cache)."""
    return _ins.MODEL_CACHE.parent / "uploads"


def content_key(raw: bytes) -> str:
    """Return the content-addressed name of a table's bytes."""
    return hashlib.sha1(raw).hexdigest()[:16]


def stage_upload(raw: bytes) -> tuple[Path, Path]:
    """Write a table beside its archive entry: (staged file, archive entry). os.replace commits it.

    A unique name, so two sessions staging the same bytes cannot collide, and a rejected table leaves nothing behind
    once the caller unlinks the staged file.
    """
    key = content_key(raw)
    destination = uploads_dir() / f"{key}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{key}.{uuid.uuid4().hex}.tmp")
    _ins.atomic_write(staged, lambda fh: fh.write(raw))
    return staged, destination


def _parse_table(path: Path) -> np.ndarray:
    """Return the table as a structured array; malformed content is a ValueError naming the expected layout."""
    try:
        # genfromtxt warns instead of raising on an empty file: reject it
        # before NumPy so corrupt share uploads fail cleanly.
        text = path.read_text()
        lines = [line for line in text.splitlines() if line.strip()]
        if path.stat().st_size == 0 or not text.strip():
            raise ValueError("file is empty")
        if len(lines) < 6:
            raise ValueError("file is too short for a units line, column header, and four rows")
        tab = np.genfromtxt(path, names=True, dtype=None, skip_header=1)
    except (OSError, ValueError, IndexError) as e:  # genfromtxt raises IndexError on a comments-only file
        raise ValueError(
            f"T-P table {path} is not parseable as a VULCAN atm file "
            f"(header comment line, then 'Pressure Temp [Kzz]' columns): {e}"
        ) from e
    return tab


def _read_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (P dyn/cm^2, T K, Kzz cm^2/s or None) of a table, refusing a malformed one."""
    tab = _parse_table(path)
    names = list(tab.dtype.names or [])
    if "Pressure" not in names or "Temp" not in names:
        raise ValueError(
            f"T-P table {path} needs 'Pressure' and 'Temp' columns "
            f"(found {names}). Line 1 must be a units comment (e.g. "
            "'#(dyne/cm2) (K) (cm2/s)'), line 2 the column names."
        )
    P = np.asarray(tab["Pressure"], dtype=np.float64)
    T = np.asarray(tab["Temp"], dtype=np.float64)
    if P.ndim != 1 or P.size < 4:
        raise ValueError(f"T-P table {path}: need >= 4 rows (got {P.size})")
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(T))):
        raise ValueError(f"T-P table {path}: non-finite Pressure/Temp entries")
    if np.any(P <= 0.0):
        raise ValueError(f"T-P table {path}: Pressure must be > 0 (dyne/cm^2)")
    dP = np.diff(P)
    if not (np.all(dP > 0) or np.all(dP < 0)):
        raise ValueError(f"T-P table {path}: Pressure must be strictly monotonic")
    Kzz = None
    if "Kzz" in names:
        Kzz = np.asarray(tab["Kzz"], dtype=np.float64)
        if not np.all(np.isfinite(Kzz)) or np.any(Kzz <= 0.0):
            raise ValueError(f"T-P table {path}: Kzz column must be finite and > 0")
    return P, T, Kzz


def read_tp_table(path: Path, span: tuple | None = None, extends_top: bool = False) -> dict:
    """Parse + validate a VULCAN atm table on the LIGHT path.

    ``span`` is the RUN's chemistry span (P_t, P_b) in dyn/cm^2 that the
    coverage and T-window checks are made against. Without it (an upload,
    checked before a run's top is known) the T window is checked over the
    shipped cfg's span and the top is not gated. ``extends_top``: the table's
    registry entry declares that it is held constant above its top, so the
    top is not gated (top_extension_caveat says so with the result).

    Mirrors the engine's read exactly (np.genfromtxt, names=True,
    skip_header=1: line 1 a units comment, line 2 the column names). Columns
    'Pressure' (dyne/cm^2) and 'Temp' (K) are required, 'Kzz' (cm^2/s)
    optional; returns {"P_dyn", "T", "Kzz" or None}. Malformed content raises
    ValueError at the API, never inside the engine's pre-loop.
    """
    P, T, Kzz = _read_columns(path)
    run_span = span is not None
    span = CHEM_P_SPAN_DYN if span is None else (float(span[0]), float(span[1]))
    # Bottom coverage is a HARD requirement: the engine clamp-extends outside
    # the table, so a short table would run the quench region isothermal.
    if P.max() < span[1]:
        raise ValueError(
            f"T-P table {path}: bottom pressure {P.max():.3g} dyn/cm^2 does "
            f"not reach the chemistry-grid bottom {span[1]:.3g} "
            f"dyn/cm^2 ({span[1] / 1e6:.1f} bar). The engine would "
            "clamp-extend the last tabulated temperature isothermally over "
            "the deep quench region (CO/CH4/NH3 quenching lives there), "
            "silently biasing quenched abundances. Extend the table to at "
            "least the grid bottom, lower p_btm_bar, or (in emission) use the "
            "Guillot profile, which is defined at every depth."
        )
    # Top coverage too, against a run's own top: above the table the engine holds the top tabulated temperature and
    # Kzz constant.
    if run_span and not extends_top and P.min() > span[0] * (1.0 + TOP_RTOL):
        top_bar = P.min() / 1e6
        remedy = (
            f"Extend the table to the model top, or set rt_ptop_bar to at least {top_bar:.4g} bar."
            if top_bar <= RT_PTOP_RANGE[1]
            else f"Its top lies below the deepest model top the tool supports, {RT_PTOP_RANGE[1]:g} bar, so the table "
            "cannot be used: extend it upward to that pressure or beyond, or use the Guillot profile."
        )
        raise ValueError(
            f"T-P table {path}: top pressure {top_bar:.4g} bar does not reach the model top {span[0] / 1e6:.4g} bar "
            f"(rt_ptop_bar). The engine would hold the top tabulated temperature and Kzz constant above the table. "
            + remedy
        )
    # The T window applies to what the engine evaluates -- the profile over the
    # chemistry span with the edge clamp, NOT every raw row (raw rows wrongly
    # reject a table that merely extends past the span). Sampled in log P while
    # the engine interpolates T linearly in P (VULCAN-JAX atm_setup): the two
    # agree at every tabulated row and differ only by the temperature change
    # across the single row-interval at each span edge.
    o = np.argsort(P)
    grid = np.logspace(np.log10(span[0]), np.log10(span[1]), T_CHECK_POINTS)
    T_grid = np.interp(np.log10(grid), np.log10(P[o]), T[o])
    if T_grid.min() < T_WINDOW[0] or T_grid.max() > T_WINDOW[1]:
        lo_all, hi_all = float(T.min()), float(T.max())
        extra = (
            ""
            if (lo_all >= T_WINDOW[0] and hi_all <= T_WINDOW[1])
            else f" (the file spans [{lo_all:.0f}, {hi_all:.0f}] K in total; "
            "only the part covering the chemistry grid is checked)"
        )
        raise ValueError(
            f"T-P table {path}: on the chemistry grid "
            f"({span[0]:g}-{span[1]:g} dyn/cm^2) the "
            f"profile spans [{T_grid.min():.0f}, {T_grid.max():.0f}] K, which "
            f"leaves the modelable window [{T_WINDOW[0]:.0f}, "
            f"{T_WINDOW[1]:.0f}] K{extra} (the raw k-tables span a wider "
            "range); out-of-window profiles are rejected, never clipped. Move the model top (rt_ptop_bar) or "
            "bottom (p_btm_bar) to where the table stays inside the window."
        )
    return {"P_dyn": P, "T": T, "Kzz": Kzz}


def top_extension_caveat(cp: dict) -> str | None:
    """Return the caveat of a run whose declared shipped table ends below the model top, else None.

    One sentence for the run log and the result page. It reads the shipped table only for a planet whose registry
    entry declares the extension.

    Parameters
    ----------
    cp : dict
        Canonical parameters.

    Returns
    -------
    str or None
        The sentence, naming the table's top and the model top in bar.
    """
    if cp["tp_mode"] != "file" or cp["tp_file"] != TP_FILE_SHIPPED or not declares_top_extension(cp["planet"]):
        return None
    top_bar = float(_read_columns(shipped_tp_file(cp["planet"]))[0].min()) / 1.0e6
    if top_bar <= cp["rt_ptop_bar"] * (1.0 + TOP_RTOL):
        return None
    return (
        f"Structure held constant above the T-P table's top: from {top_bar:.3g} bar to the model top "
        f"{cp['rt_ptop_bar']:.3g} bar, T and Kzz keep the table's top values (VULCAN's file-mode convention)."
    )


def resolve_tp_file(params: dict) -> tuple[Path, str]:
    """(path, content-sha1[:16]) of the requested T-P table.

    'shipped' is the vulcan3 table bundled FOR THIS PLANET; 'upload' is params['tp_file_path'], which run_model
    copies to the content-addressed uploads/<sha1>.txt so canonical params alone can re-resolve it.
    """
    src = str(params.get("tp_file", TP_FILE_SHIPPED))
    if src == TP_FILE_SHIPPED:
        path = shipped_tp_file(str(params.get("planet", "wasp39b")))
    elif src == TP_FILE_UPLOAD:
        raw = params.get("tp_file_path")
        if raw:
            path = Path(str(raw))
        else:
            # No raw path: the content-addressed archive is what makes
            # canonical params ROUND-TRIP -- the sha re-resolves the bytes.
            sha = str(params.get("tp_file_sha1", ""))
            if not re.fullmatch(r"[0-9a-f]{16}", sha):
                raise ValueError(
                    "tp_file='upload' requires tp_file_path (the saved "
                    "table; the GUI sets it on upload) or tp_file_sha1 (a "
                    "previously archived upload)."
                )
            path = uploads_dir() / f"{sha}.txt"
    else:
        raise ValueError(
            f"tp_file={src!r}: choose '{TP_FILE_SHIPPED}' (the measured table "
            f"bundled with vulcan3 for this planet) or '{TP_FILE_UPLOAD}'"
        )
    if not path.exists():
        raise ValueError(f"T-P table not found: {path}")
    sha1 = content_key(path.read_bytes())
    return path, sha1


def tp_file_from_cp(cp: dict) -> Path:
    """Re-resolve the T-P table from CANONICAL params alone.

    Shipped -> the bundled file; upload -> the content-addressed uploads/<sha1>.txt. Verifies the hash, so a table
    that changed since the key was computed is refused.
    """
    if cp["tp_file"] == TP_FILE_SHIPPED:
        path = shipped_tp_file(cp["planet"])
    else:
        path = uploads_dir() / f"{cp['tp_file_sha1']}.txt"
    if not path.exists():
        raise RuntimeError(
            f"T-P table for this run is missing: {path}. For uploads the "
            "content-addressed copy is written by run_model; re-upload the "
            "table if the output directory was cleaned."
        )
    sha1 = content_key(path.read_bytes())
    if sha1 != cp["tp_file_sha1"]:
        raise RuntimeError(
            f"T-P table {path} content drifted: sha1 {sha1} != canonical "
            f"{cp['tp_file_sha1']}. Refusing -- the cached spectrum would be "
            "keyed to different physics than the file now holds."
        )
    return path
