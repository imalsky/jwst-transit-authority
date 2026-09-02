"""Forward model runner: VULCAN-JAX photochemistry -> ExoJAX spectrum.

The GUI's light path (``params_key`` / ``cache_path`` / ``load_result``) only
reads the npz cache in ``instruments.MODEL_CACHE``: spectra under
``params_key``, converged chemistry columns under ``chem-<chem_key>``.
``python -m jwst_tool.forward params.json`` runs the heavy pipeline and writes
that entry, logging "[fwd] PROG <frac> <label>" for the GUI bar. With
``fisher_params`` it also builds the spectrum Jacobian row by row (per-row
method in ``jac_row_method``): "fd" is certified central finite differences
under the h-vs-2h gate, "ad" one warm-started jvp per row (photo-on required;
the lnZ jvp is the fixed-structural-grid
derivative and the dlnCO jvp is refused near O-exhaustion). Where the
central dlnCO stencil would straddle C/O = 1, the FD row steps one-sided
toward lower C/O instead (``fd_stencil``; refused at or above 1 in that
band); AD has no such escape.
``fisher.py`` turns the Jacobian + Pandeia noise into forecasts.

A "removed" spectrum zeroes that molecule's VMR in the RT only, keeping the
structure (T, mmw): the standard nested-model comparison.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np

# instruments is import-light (stdlib + numpy, safe on the GUI's light path)
# and owns the data/output root resolution (env-overridable, loud on failure)
from jwst_tool import instruments as _ins

MODEL_CACHE = _ins.MODEL_CACHE

from jwst_tool import planets   # installed package: works as module AND as a script

MOLECULES = ["H2O", "CO2", "CO", "CH4", "SO2"]   # always-on WIDE-profile set
# RT additions the network already solves; extra_mols exposes them to the RT.
EXTRA_MOLECULES = ["C2", "C2H2", "C2H4", "CH", "CH3", "CN", "CS",
                   "H2CO", "H2O2", "H2S", "HCN", "N2O", "NH", "NH3",
                   "NO", "NS", "OCS", "OH", "SH", "SO"]

# GUI preselection: NEVER widen without a measured leave-one-out ppm impact.
EXTRA_MOLECULES_DEFAULT = ["C2H2", "C2H4", "H2S", "HCN", "NH3", "OCS",
                           "SH", "SO"]
# Kinetics networks ("ncho" drops sulfur, 69 vs 89 species) -> the engine's
# import-frozen (VULCAN_JAX_NETWORK, VULCAN_JAX_ATOM_LIST) pair.
NETWORKS = {
    "sncho": ("thermo/SNCHO_photo_network.txt", "H,O,C,N,S"),
    "ncho": ("thermo/NCHO_photo_network.txt", "H,O,C,N"),
}
# Every S-bearing member of the lists above; network="ncho" REFUSES these
# rather than dropping them silently.
_S_MOLECULES = frozenset({"SO2", "H2S", "OCS", "SO", "SH", "CS", "NS"})
# Network species with no published ExoMolOP k-table: never offered anywhere
# (correlated-k over the published tables is the only opacity path).
_NO_EXOMOLOP_TABLE = frozenset({"CS2", "C2H6"})
_VERSION = 41  # model_cache buster (identity = canonical params + this
               # number, never a content hash); bump on any physics or
               # canonical-key-set change.

# Baseline C/O: the number ratio N_C/N_O (not [C/H]/[O/H], not a log) on the
# W39b cfg's elemental set, the ONLY valid basis. Display value only, and
# _assemble_chem refuses on drift from the loaded cfg's C_H/O_H.
CO_BASELINE = 0.00295 / 0.00537   # = 0.54935, cfg C_H/O_H (Tsai 2023 10x-solar)
# GUI/API default co_ratio: the baseline rounded onto the widgets' 0.05 grid.
CO_DEFAULT = 0.55

# FD Fisher steps. lnZ scales O/C/N/S together (C/O preserved), dlnCO scales
# C_H at fixed O, lnKzz/T-P rows perturb theta on the same build. Every row is
# evaluated at h AND 2h -- disagreement beyond FD_CONSISTENCY_TOL RAISES --
# and reported as the Richardson combination (4 J_h - J_2h)/3.
FD_STEPS = {"lnZ": 0.10, "dlnCO": 0.10, "lnKzz": 0.10,      # ln-space steps
            "Tirr": 10.0, "Tint": 10.0,                     # Kelvin
            "log_kappa": 0.05, "log_gamma": 0.05,           # dex
            "log_kappa_cloud": 0.05, "alpha_cloud": 0.05}   # dex / slope
FD_COMP_PARAMS = ("lnZ", "dlnCO")     # need a chemistry re-init per FD point
FD_CONSISTENCY_TOL = 0.25
FD_LNR0_STEP = 0.01                   # lnR0 is RT-only (smooth, analytic)
# Model top (chemistry grid AND RT top) when a request does not set it.
# 1e-7 bar is the usual published choice; the converged 1e-9 bar top the
# tool shipped before (v40) is still selectable in the same range.
RT_PTOP_DEFAULT = 1.0e-7
JAC_METHODS = ("fd", "ad")            # certified-FD default / warm-jvp opt-in
# Minimum positivity margin for the AD dlnCO row. The engine's co_bz_bound =
# ln(1 + min_z(OO_z/OC_z)) is the largest ln C/O increment before some
# layer's fixed-O factor b_z turns nonpositive (nonnegative by construction,
# so a "<= 0" refusal could never fire); it is computed on the build's
# initial column, a proxy for the warm state the tangent runs on. Below this
# margin the per-layer tangent factors are ill-conditioned. Empirical, equal
# to the central FD stencil's reach today; a literal so a step change cannot
# move the AD gate silently. FD re-initializes per stencil point and steps
# one-sided below C/O = 1 when the central stencil would cross it.
CO_BZ_MIN_AD = 0.2


def fd_stencil(name: str, value: float) -> tuple[tuple[int, ...], float]:
    """(step multiples, step) a composition row solves at. Central (+-h, +-2h)
    at h = FD_STEPS[name], unless the dlnCO stencil would straddle C/O = 1:
    the C-rich side does not certify under the shipped solver settings (W39b
    at C/O 1.087 exhausts count_max at longdy 36), so below 1 the row uses
    the one-sided stencil (-1, -2, -4) toward lower C/O -- third order after
    Richardson -- at h/2 so its reach equals the central stencil's 2h (at
    the full step the curvature toward the boundary fails the h-vs-2h gate
    on a 1600 K hot Jupiter). At or above 1 inside that band no stencil is
    certified, so the row refuses."""
    h = FD_STEPS[name]
    if name == "dlnCO":
        m = float(np.exp(2.0 * h))
        if value / m <= 1.0 <= value * m:
            if value >= 1.0:
                raise ValueError(
                    f"co_ratio={value:g}: a dlnCO FD row at or above C/O = 1 "
                    "within one stencil of the boundary is not certified "
                    "(the C-rich side does not reach a certified steady "
                    "state under the shipped settings). Drop dlnCO from "
                    "fisher_params.")
            return (-1, -2, -4), h / 2.0
    return (1, -1, 2, -2), h


def fd_estimates(offs, dvals, f0, h):
    """(J_h, J_2h) from stencil values dvals[s] = f(x + s h): central when
    offs holds +-1, else the one-sided stencil anchored on f0 = f(x). The
    Richardson combination (4 J_h - J_2h) / 3 is exact for cubics either way."""
    if 1 in offs and -1 in offs:
        return ((dvals[1] - dvals[-1]) / (2.0 * h),
                (dvals[2] - dvals[-2]) / (4.0 * h))
    d = offs[0]
    return (d * (-3.0 * f0 + 4.0 * dvals[d] - dvals[2 * d]) / (2.0 * h),
            d * (-3.0 * f0 + 4.0 * dvals[2 * d] - dvals[4 * d]) / (4.0 * h))


# Cloud-deck Fisher rows: RT-only like lnR0 (one central difference or an RT
# jvp, no chemistry re-solve, no h-vs-2h gate). Need cloud_on.
CLOUD_FISHER_PARAMS = ("log_kappa_cloud", "alpha_cloud")

# Kzz profile modes. "file" requires tp_mode="file" with a Kzz column
# (the upstream constraint: the tabulated Kzz lives in the atm table).
KZZ_MODES = ("const", "Pfunc", "JM16", "file")

# tp_mode="file" profile sources.
TP_FILE_SHIPPED = "shipped"       # the cfg's own atm_file (W39b evening terminator)
TP_FILE_UPLOAD = "upload"         # user-supplied table; content-addressed copy
                                  # under <output>/uploads/<sha1>.txt

# Chemistry-grid pressure span (dyn/cm^2) of the SHIPPED W39b cfg,
# cross-checked against the live cfg in _assemble_chem. For a RUN's span ask
# chem_p_span_dyn(cp), never this constant.
CHEM_P_SPAN_DYN = (0.1, 7.6e6)

# Column bottom, structure-aware: a measured T-P table caps chemistry at its
# own bottom, parametric profiles default to a round 10 bar. The range ceiling
# is a DATA limit (opacities are built for T_WINDOW).
P_BTM_FILE_BAR = CHEM_P_SPAN_DYN[1] / 1.0e6    # 7.6 bar, the shipped tables' bottom
P_BTM_PARAMETRIC_BAR = 10.0
P_BTM_RANGE = (1.0, 300.0)
# The RT column bottom sits just inside the chemistry bottom: interp_map refuses
# an ART grid reaching below it. Keep this 7.0 / 7.6 ratio at any depth.
ART_PBTM_FRACTION = 7.0 / 7.6


def default_p_btm_bar(params: dict) -> float:
    """Chemistry-grid bottom (bar) for this run: a measured T-P table's own
    bottom in file mode, a round 10 bar for the parametric profiles."""
    mode = str(params.get("tp_mode", _default_tp_mode(params)))
    return P_BTM_FILE_BAR if mode == "file" else P_BTM_PARAMETRIC_BAR


# Instrument windows the emission tau-bottom report is broken down by, so a
# refusal says WHERE the column sees through. The 1-2 um entry is first: it is
# where the modeled opacity set runs out, not where the column is shallow.
_TAU_WINDOWS = (("1.0-2.0 um (no modeled continuum)", 1.0, 2.0),
                ("2.0-3.0 um", 2.0, 3.0),
                ("3.0-5.2 um (NIRSpec)", 3.0, 5.2),
                ("5.0-12 um (MIRI LRS)", 5.0, 12.0),
                ("12-15 um", 12.0, 15.0))


# Share of the planet's EMITTED FLUX that may come from wavelengths where the
# column sees through its own bottom. Flux-weighted on purpose, so a narrow
# blue-edge notch cannot veto an otherwise opaque column. Above the tolerance
# the run is refused.
EMIS_TAU_THIN = 3.0            # per-wavelength "sees through the bottom"
EMIS_THIN_FLUX_FRAC = 0.01     # tolerated share of emitted flux from those


def _tau_bottom_breakdown(wl_um, tau, flux=None) -> str:
    """Per-window minimum bottom optical depth, how much of the band is below
    the gate, and (given ``flux``) how much of the planet's emission comes from
    there. A single min() cannot tell a shallow column from an opacity gap."""
    wl, tau = np.asarray(wl_um, float), np.asarray(tau, float)
    thin = tau < EMIS_TAU_THIN
    lines = [f"Where the column sees through ({100.0 * thin.mean():.1f}% of "
             f"the model band is below tau = {EMIS_TAU_THIN:g}"]
    if thin.any():
        lines[0] += f", spanning {wl[thin].min():.2f}-{wl[thin].max():.2f} um)"
    else:
        lines[0] += ")"
    if flux is not None:
        lines.append(f"  carrying {100.0 * thin_flux_fraction(tau, flux):.3f}% "
                     "of the planet's emitted flux")
    for label, lo, hi in _TAU_WINDOWS:
        m = (wl >= lo) & (wl <= hi)
        if not m.any():
            continue
        lines.append(f"  {label:34s} min tau {tau[m].min():10.3g}"
                     f"   {100.0 * (tau[m] < EMIS_TAU_THIN).mean():5.1f}% thin")
    return "\n".join(lines)


def thin_flux_fraction(tau, flux) -> float:
    """Share of the emitted flux coming from wavelengths whose bottom optical
    depth is below the gate. The emission certificate."""
    tau, flux = np.asarray(tau, float), np.abs(np.asarray(flux, float))
    tot = float(flux.sum())
    if not np.isfinite(tot) or tot <= 0.0:
        raise RuntimeError(
            "emission flux is zero or non-finite across the whole band; the "
            "thin-bottom certificate cannot be evaluated")
    return float(flux[tau < EMIS_TAU_THIN].sum() / tot)


def chem_p_span_dyn(cp: dict) -> tuple:
    """The RUN's chemistry span (P_t, P_b) in dyn/cm^2: the top follows
    rt_ptop_bar, the bottom p_btm_bar (CHEM_P_SPAN_DYN is the shipped cfg's)."""
    return (float(cp.get("rt_ptop_bar") or RT_PTOP_DEFAULT) * 1.0e6,
            float(cp.get("p_btm_bar") or default_p_btm_bar(cp)) * 1.0e6)

# Structure default: a MEASURED T-P/Kzz table VERIFIED end-to-end for a planet
# is that planet's default structure, Guillot everywhere else. File mode has no
# T-P Fisher row, so switch to Guillot when a temperature row is needed.
def _default_tp_mode(params: dict) -> str:
    """tp_mode default: the planet's measured table where a default run on it
    is VERIFIED, else the analytic Guillot profile (having a table is not
    enough -- see shipped_tp_table_is_default).

    EMISSION always defaults to Guillot: the bundled tables are TERMINATOR
    profiles, the wrong structure for a dayside eclipse. Selecting one for
    emission stays possible as the user's explicit choice.
    """
    if str(params.get("science_mode", "transmission")) == "emission":
        return "guillot"
    planet = str(params.get("planet", "wasp39b"))
    if shipped_tp_table_is_default(planet):
        return "file"
    return "guillot"


def default_tirr(planet: str, system: dict | None = None,
                 science_mode: str = "transmission") -> float:
    """Guillot T_irr default for ``planet`` -- sqrt(2) * T_eq, GUI-identical
    (planets.default_tirr is the single definition both sides call). For the
    custom planet pass ``system`` (star_teff, rstar_rsun, orbit_au)."""
    return planets.default_tirr(
        planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS),
        system=(system if planet not in planets.PLANETS else None),
        science_mode=science_mode)

def active_molecules(cp: dict) -> list[str]:
    """RT molecule set for canonical params: base set + selected extras."""
    base, extras = MOLECULES, EXTRA_MOLECULES
    if cp.get("network", "sncho") == "ncho":
        base = [m for m in base if m not in _S_MOLECULES]
        extras = [m for m in extras if m not in _S_MOLECULES]
    return base + [m for m in extras if m in cp["extra_mols"]]

# Numerical-resolution knobs layered on engine_config.WIDE (1-15 um band
# unchanged). _rt_profile_common locks the RT layer count equal to nz, so there
# is no RT-layer knob; the spectral grid is the k-tables' R = 1000 grid.
NZ_DEFAULT, YCONV_DEFAULT = 100, 1.0e-2
NZ_RANGE = (60, 150)            # equal counts on distinct chemistry and RT grids
YCONV_RANGE = (1.0e-4, 1.0e-2)  # steady-state convergence tolerance (1e-3 is the
                                # validated "high" tier; tighter costs runtime
                                # but is safe -- the longdy gate is loud)

# Modelable temperature window -- reject, never clip. Conservative next to the
# published ExoMolOP tables (100-3400 K): nothing above 2980 K is validated.
# Widening it is a physics decision backed by a run, not a constant edit.
T_WINDOW = (320.0, 2980.0)

# Parameters that can be freed in the Fisher forecast, per tp_mode. "file" is
# deliberately EMPTY: a tabulated profile has no temperature parameter, so a
# file-mode forecast conditions on the profile (optimistic).
CHEM_PARAM_NAMES = ["lnZ", "dlnCO", "lnKzz"]
TP_PARAM_NAMES = {
    "guillot": ["Tirr", "Tint", "log_kappa", "log_gamma"],
    "file": [],
}
# Display SYMBOL, UNIT and friendly name per parameter for the GUI. The symbol
# MUST match the unit's log base: metallicity and Kzz are dex (log10), so
# [M/H] and log Kzz, never "ln"; C/O is the number ratio N_C/N_O (no unit).
PARAM_SYMBOLS = {"lnZ": "[M/H]", "dlnCO": "C/O", "lnKzz": "log Kzz",
                 "Tirr": "T_irr", "Tint": "T_int",
                 "log_kappa": "log κ_IR", "log_gamma": "log γ",
                 "log_kappa_cloud": "log κ_cloud", "alpha_cloud": "α_cloud"}
PARAM_UNITS = {"lnZ": "dex", "dlnCO": "", "lnKzz": "dex",
               "Tirr": "K", "Tint": "K",
               "log_kappa": "dex", "log_gamma": "dex",
               "log_kappa_cloud": "dex", "alpha_cloud": ""}
PARAM_LABELS = {"lnZ": "Metallicity", "dlnCO": "C/O ratio",
                "lnKzz": "Vertical mixing (Kzz)",
                "Tirr": "Guillot T_irr",
                "Tint": "Guillot T_int", "log_kappa": "Guillot log κ_IR",
                "log_gamma": "Guillot log γ",
                "log_kappa_cloud": "Cloud deck log κ (at 3.5 um)",
                "alpha_cloud": "Cloud deck slope α"}


def param_axis(name: str) -> str:
    """Axis/column label for a parameter: 'Symbol [unit]', or bare 'Symbol'
    when it is dimensionless (e.g. '[M/H] [dex]', 'C/O', 'T_irr [K]')."""
    u = PARAM_UNITS[name]
    return f"{PARAM_SYMBOLS[name]} [{u}]" if u else PARAM_SYMBOLS[name]


# tp_mode="file" helpers (light path: no vulcan_jax/jax imports)

def shipped_tp_table_name(planet: str) -> str:
    """Filename of the MEASURED structure table bundled for ``planet``, or ""
    when that planet has none (planets.PLANETS[...]["tp_table"])."""
    entry = planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS)
    return entry.get("tp_table") or ""


def shipped_tp_table_is_default(planet: str) -> bool:
    """Whether ``planet``'s bundled table is its DEFAULT structure -- separate
    from merely having one: a table becomes the default only once a default run
    on it is verified end-to-end here. A planet whose table the solver will not
    certify at default settings stays selectable-but-not-default rather than
    erroring on arrival (planets.PLANETS[...]["tp_table_note"] says why)."""
    entry = planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS)
    return bool(entry.get("tp_table") and entry.get("tp_table_default"))


def _shipped_tp_file(planet: str) -> Path:
    """Path of the shipped T-P/Kzz table for ``planet``, WITHOUT importing
    vulcan_jax (find_spec only: importing parses the reaction network, far too
    heavy for the GUI's cache-key path). A planet with no usable table raises
    the reason, never substitutes another planet's atmosphere."""
    name = shipped_tp_table_name(planet)
    if not name:
        entry = planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS)
        why = entry.get("tp_table_note") or "no table is bundled for it"
        raise ValueError(
            f"tp_file='shipped' is not available for planet {planet!r}: {why} "
            "Choose tp_mode='guillot', or upload a table "
            "with tp_file='upload'.")
    import importlib.util
    spec = importlib.util.find_spec("vulcan_jax")
    if spec is None or not spec.origin:
        raise RuntimeError(
            "vulcan_jax is not installed (or has no package origin): "
            f"tp_mode='file' with tp_file='shipped' needs its bundled "
            f"atm/{name} table.")
    return Path(spec.origin).parent / "atm" / name


def _uploads_dir() -> Path:
    """Content-addressed home for uploaded T-P tables (sibling of model_cache)."""
    return MODEL_CACHE.parent / "uploads"


def _read_tp_table(path: Path, span: tuple | None = None) -> dict:
    """Parse + validate a VULCAN atm table on the LIGHT path.

    ``span`` is the RUN's chemistry span (P_t, P_b) in dyn/cm^2 that the
    coverage and T-window checks are made against; it defaults to the shipped
    cfg's own span, and an emission run passes its deeper one.

    Mirrors the engine's read exactly (np.genfromtxt, names=True,
    skip_header=1: line 1 a units comment, line 2 the column names). Columns
    'Pressure' (dyne/cm^2) and 'Temp' (K) are required, 'Kzz' (cm^2/s)
    optional; returns {"P_dyn", "T", "Kzz" or None}. Malformed content raises
    ValueError at the API, never inside the engine's pre-loop."""
    span = CHEM_P_SPAN_DYN if span is None else (float(span[0]), float(span[1]))
    try:
        # genfromtxt warns instead of raising on an empty file: reject it
        # before NumPy so corrupt share uploads fail cleanly.
        text = path.read_text()
        lines = [line for line in text.splitlines() if line.strip()]
        if path.stat().st_size == 0 or not text.strip():
            raise ValueError("file is empty")
        if len(lines) < 6:
            raise ValueError(
                "file is too short for a units line, column header, and four rows")
        tab = np.genfromtxt(path, names=True, dtype=None, skip_header=1)
    except Exception as e:                                    # noqa: BLE001
        raise ValueError(
            f"T-P table {path} is not parseable as a VULCAN atm file "
            f"(header comment line, then 'Pressure Temp [Kzz]' columns): {e}")
    names = list(tab.dtype.names or [])
    if "Pressure" not in names or "Temp" not in names:
        raise ValueError(
            f"T-P table {path} needs 'Pressure' and 'Temp' columns "
            f"(found {names}). Line 1 must be a units comment (e.g. "
            "'#(dyne/cm2) (K) (cm2/s)'), line 2 the column names.")
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
    # Bottom coverage is a HARD requirement: the engine clamp-extends outside
    # the table, so a short table would run the quench region isothermal.
    if P.max() < span[1]:
        raise ValueError(
            f"T-P table {path}: bottom pressure {P.max():.3g} dyn/cm^2 does "
            f"not reach the chemistry-grid bottom {span[1]:.3g} "
            f"dyn/cm^2 ({span[1]/1e6:.1f} bar). The engine would "
            "clamp-extend the last tabulated temperature isothermally over "
            "the deep quench region (CO/CH4/NH3 quenching lives there), "
            "silently biasing quenched abundances. Extend the table to at "
            "least the grid bottom, lower p_btm_bar, or (in emission) use the "
            "Guillot profile, which is defined at every depth.")
    # The T window applies to what the engine evaluates -- the profile over the
    # chemistry span with the edge clamp, NOT every raw row (raw rows wrongly
    # reject a table that merely extends past the span). Sampled in log P while
    # the engine interpolates T linearly in P (VULCAN-JAX atm_setup): the two
    # agree at every tabulated row and differ only by the temperature change
    # across the single row-interval at each span edge.
    _o = np.argsort(P)
    _grid = np.logspace(np.log10(span[0]), np.log10(span[1]), 200)
    T_grid = np.interp(np.log10(_grid), np.log10(P[_o]), T[_o])
    if T_grid.min() < T_WINDOW[0] or T_grid.max() > T_WINDOW[1]:
        _lo_all, _hi_all = float(T.min()), float(T.max())
        _extra = ("" if (_lo_all >= T_WINDOW[0] and _hi_all <= T_WINDOW[1]) else
                  f" (the file spans [{_lo_all:.0f}, {_hi_all:.0f}] K in total; "
                  "only the part covering the chemistry grid is checked)")
        raise ValueError(
            f"T-P table {path}: on the chemistry grid "
            f"({span[0]:g}-{span[1]:g} dyn/cm^2) the "
            f"profile spans [{T_grid.min():.0f}, {T_grid.max():.0f}] K, which "
            f"leaves the modelable window [{T_WINDOW[0]:.0f}, "
            f"{T_WINDOW[1]:.0f}] K{_extra} (the raw k-tables span a wider "
            "range); out-of-window profiles are rejected, never clipped.")
    Kzz = None
    if "Kzz" in names:
        Kzz = np.asarray(tab["Kzz"], dtype=np.float64)
        if not np.all(np.isfinite(Kzz)) or np.any(Kzz <= 0.0):
            raise ValueError(f"T-P table {path}: Kzz column must be finite and > 0")
    return {"P_dyn": P, "T": T, "Kzz": Kzz}


def _resolve_tp_file(params: dict) -> tuple[Path, str]:
    """(path, content-sha1[:16]) of the requested T-P table: 'shipped' is the
    vulcan_jax table bundled FOR THIS PLANET; 'upload' is
    params['tp_file_path'], which run_model copies to the content-addressed
    uploads/<sha1>.txt so canonical params alone can re-resolve it."""
    src = str(params.get("tp_file", TP_FILE_SHIPPED))
    if src == TP_FILE_SHIPPED:
        path = _shipped_tp_file(str(params.get("planet", "wasp39b")))
    elif src == TP_FILE_UPLOAD:
        raw = params.get("tp_file_path")
        if raw:
            path = Path(str(raw))
        else:
            # No raw path: the content-addressed archive is what makes
            # canonical params ROUND-TRIP -- the sha re-resolves the bytes.
            sha = str(params.get("tp_file_sha1", ""))
            if not sha:
                raise ValueError(
                    "tp_file='upload' requires tp_file_path (the saved "
                    "table; the GUI sets it on upload) or tp_file_sha1 (a "
                    "previously archived upload).")
            path = _uploads_dir() / f"{sha}.txt"
    else:
        raise ValueError(
            f"tp_file={src!r}: choose '{TP_FILE_SHIPPED}' (the measured table "
            f"bundled with vulcan_jax for this planet) or '{TP_FILE_UPLOAD}'")
    if not path.exists():
        raise ValueError(f"T-P table not found: {path}")
    sha1 = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
    return path, sha1


def _tp_file_from_cp(cp: dict) -> Path:
    """Re-resolve the T-P table from CANONICAL params alone: shipped -> the
    bundled file; upload -> the content-addressed uploads/<sha1>.txt. Verifies
    the hash, so a table that changed since the key was computed is refused."""
    if cp["tp_file"] == TP_FILE_SHIPPED:
        path = _shipped_tp_file(cp["planet"])
    else:
        path = _uploads_dir() / f"{cp['tp_file_sha1']}.txt"
    if not path.exists():
        raise RuntimeError(
            f"T-P table for this run is missing: {path}. For uploads the "
            "content-addressed copy is written by run_model; re-upload the "
            "table if the output directory was cleaned.")
    sha1 = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
    if sha1 != cp["tp_file_sha1"]:
        raise RuntimeError(
            f"T-P table {path} content drifted: sha1 {sha1} != canonical "
            f"{cp['tp_file_sha1']}. Refusing -- the cached spectrum would be "
            "keyed to different physics than the file now holds.")
    return path


# VULCAN condensation channel on the SNCHO network. DETECTION-ONLY: the
# whole-column fix_species pin freezes the reservoir at a step-sequence-
# dependent transient, so no derivative through it is trustworthy (the
# canonical_params raises carry the user-facing reason). One reaction,
# S8 -> S8_l_s, on rainout-sized 50 um orthorhombic sulfur (smaller radii make
# the growth term stiffer than Ros2 resolves). The cold-trap argmin degenerates
# on isothermal columns, so the pin is whole-column; without it every solve
# exhausts count_max.
CONDEN_CFG = {
    "use_condense": True,
    "condense_sp": ["S8"],
    "non_gas_sp": ["S8_l_s"],
    "r_p": {"S8_l_s": 5.0e-3},
    "rho_p": {"S8_l_s": 2.07},
    "use_relax": [],
    "use_settling": False,
    "fix_species": ["S8", "S8_l_s"],
    "fix_species_from_coldtrap_lev": False,
    "start_conden_time": 0.0,
    "stop_conden_time": 1.0e6,
    # At the 1e-20 default, glacial trace species gate longdy forever on cold
    # columns; 1e-15 is still far below RT relevance.
    "mtol_conv": 1.0e-15,
    # Heavy hydrocarbons + trace sulfur allotropes re-equilibrate on
    # unreachable timescales far below RT relevance against a pinned S8. The
    # observable sulfur species (SO2, H2S, SO) STAY in the gate.
    "conver_ignore": ["C6H6", "C2H2", "C6H5", "C2H", "C2H4", "C2H5", "C2H6",
                      "C3H2", "C3H3", "C4H5", "CH2NH", "CH3NH2", "H2CCO",
                      "S", "S2", "S3", "S4"],
    # bounded from below so the conden window + pin complete before the gate
    "trun_min": 1.0e6,
}


# Every input key canonical_params (and its helpers) actually reads. An
# AST-walking test re-derives this set from the source, so it cannot rot.
_PARAM_KEYS_READ = frozenset({
    "Tint", "Tirr", "alpha_cloud", "bot_flux", "chem_provider",
    "cloud_on", "co_ratio", "diff_esc", "extra_mols",
    "f_diurnal", "fisher_params", "gs_cgs", "jac_method", "kzz_const",
    "kzz_kdeep", "kzz_kmax", "kzz_mode", "kzz_plev", "kzz_x", "log_gamma",
    "log_kappa", "log_kappa_cloud", "met_x_solar", "network", "nz",
    "orbit_au", "p_btm_bar", "p_ref_bar", "planet", "rp_rjup",
    "rstar_rsun", "rt_integration", "rt_ptop_bar",
    "science_mode", "sflux", "sl_angle_deg", "star_feh", "star_logg",
    "star_teff", "top_flux", "tp_file", "tp_file_path",
    "tp_file_sha1", "tp_mode", "use_condense", "use_moldiff", "use_photo",
    "use_rayleigh", "use_settling", "use_vm_mol", "wo_mols", "yconv_cri",
})
# Output-only keys of the canonical dict itself: share_config validates a SAVED
# payload by feeding it back in, so echo fields must not read as unknown.
_PARAM_KEYS_ECHOED = frozenset({"version"})
_KNOWN_PARAM_KEYS = _PARAM_KEYS_READ | _PARAM_KEYS_ECHOED

# Misspellings worth a pointed hint; unknown keys are refused, never dropped
# quietly (standing fail-loud rule).
_PARAM_KEY_HINTS = {"mode": "science_mode",
                    "metallicity": "met_x_solar", "co": "co_ratio"}
# Keys of the removed line-by-line opacity mode and Mie condensate deck,
# refused BY NAME so an old config says what was dropped.
_REMOVED_PARAM_KEYS = frozenset({
    "opacity_mode", "nu_pts", "rt_dit_res", "broadening", "mie_condensate",
    "mie_log_rg", "mie_sigmag", "mie_log_mmr"})


def canonical_params(params: dict) -> dict:
    removed = sorted(set(params) & _REMOVED_PARAM_KEYS)
    if removed:
        raise ValueError(
            f"parameter key(s) {removed} were removed in 0.48.0 with the "
            "sampled line-by-line opacity mode and the Mie condensate deck; "
            "correlated-k is the only opacity path now. Drop the keys (the "
            "power-law cloud deck, cloud_on, stays).")
    unknown = sorted(set(params) - _KNOWN_PARAM_KEYS)
    if unknown:
        hints = "".join(
            f" (did you mean {_PARAM_KEY_HINTS[k]!r} instead of {k!r}?)"
            for k in unknown if k in _PARAM_KEY_HINTS)
        raise ValueError(
            f"unknown parameter key(s) {unknown}.{hints} An unknown key is "
            "refused, never dropped: the canonical key set is "
            "forward._KNOWN_PARAM_KEYS.")
    # The keys stay in the canonical payload, pinned off, so no cached
    # spectrum's key changes.
    if (params.get("use_settling", False) or params.get("diff_esc")
            or params.get("top_flux") or params.get("bot_flux")):
        raise ValueError(
            "boundary fluxes, escape and settling were removed from this "
            "tool: leave use_settling False and diff_esc/top_flux/bot_flux "
            "empty.")
    tp_mode = str(params.get("tp_mode", _default_tp_mode(params)))
    if tp_mode == "picaso_climate":
        raise ValueError(
            "tp_mode='picaso_climate' was removed with the PICASO subsystem "
            "in 0.43.0. Use a Guillot profile, or tp_mode='file' with an "
            "explicit table.")
    if tp_mode not in TP_PARAM_NAMES:
        raise ValueError(
            f"unknown tp_mode {tp_mode!r}: choose from "
            f"{list(TP_PARAM_NAMES)} -- a Guillot profile, or 'file' with an "
            "explicit table.")
    provider = str(params.get("chem_provider", "vulcan"))
    if provider != "vulcan":
        raise ValueError(
            f"chem_provider {provider!r} is not available: the PICASO "
            "equilibrium provider was removed in 0.43.0. The only engine is "
            "'vulcan' (VULCAN-JAX kinetics, the default -- drop the key).")
    network = str(params.get("network", "sncho"))
    if network not in NETWORKS:
        raise ValueError(
            f"unknown network {network!r}: choose from {list(NETWORKS)} "
            "('sncho' = the shipped S-N-C-H-O kinetics network, the default; "
            "'ncho' = the sulfur-free N-C-H-O network, a cheaper solve with "
            "no SO2/H2S/CS2/OCS).")
    # tp_mode="file": resolve + validate the table NOW (numpy parse + hash, no
    # engine imports) so a bad upload fails at the API and the cache key is
    # content-addressed, never path-addressed.
    science_mode = str(params.get("science_mode", "transmission"))
    # Resolved BEFORE the table is validated: the coverage gate runs against
    # THIS run's span, not the shipped cfg's.
    p_btm_bar = float(params.get("p_btm_bar", default_p_btm_bar(params)))
    if not P_BTM_RANGE[0] <= p_btm_bar <= P_BTM_RANGE[1]:
        raise ValueError(
            f"p_btm_bar={p_btm_bar:g} outside {P_BTM_RANGE} bar (the "
            "chemistry and RT column bottom): below it the deep quench region "
            "is cut off, above it no shipped profile stays inside the "
            f"modelable window {T_WINDOW} K.")
    _span = (float(params.get("rt_ptop_bar", RT_PTOP_DEFAULT)) * 1.0e6, p_btm_bar * 1.0e6)
    tp_file, tp_file_sha1, tp_table = "", "", None
    if tp_mode == "file":
        tp_path, tp_file_sha1 = _resolve_tp_file(params)
        tp_table = _read_tp_table(tp_path, span=_span)
        tp_file = str(params.get("tp_file", TP_FILE_SHIPPED))
    if science_mode not in ("transmission", "emission"):
        raise ValueError(
            f"unknown science_mode {science_mode!r}: choose 'transmission' "
            "(transit depth) or 'emission' (secondary-eclipse depth)")
    planet = str(params.get("planet", "wasp39b"))
    if planet not in planets.PLANETS and planet != "custom":
        raise ValueError(f"unknown planet {planet!r}")
    sysd = planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS)
    nz = int(params.get("nz", NZ_DEFAULT))
    if not NZ_RANGE[0] <= nz <= NZ_RANGE[1]:
        raise ValueError(f"nz={nz} outside the validated layer-count range {NZ_RANGE} "
                         "(applied to the distinct chemistry and RT grids)")
    yconv_cri = float(params.get("yconv_cri", YCONV_DEFAULT))
    if not YCONV_RANGE[0] <= yconv_cri <= YCONV_RANGE[1]:
        raise ValueError(f"yconv_cri={yconv_cri:g} outside the validated range "
                         f"{YCONV_RANGE} (steady-state convergence tolerance)")
    sflux = str(params.get("sflux", sysd["sflux"]))
    if sflux not in planets.SFLUX_CHOICES:
        raise ValueError(f"unknown stellar UV spectrum {sflux!r} "
                         f"(choose from {list(planets.SFLUX_CHOICES)})")
    star_ref = planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS)["star"]
    # before the literal: the custom planet's Tirr default derives from these
    # three (planets.system_teq)
    _teff = round(float(params.get("star_teff", star_ref["teff"])), 1)
    _rstar = round(float(params.get("rstar_rsun", sysd["rstar_rsun"])), 4)
    _orbit = round(float(params.get("orbit_au", sysd["orbit_au"])), 5)
    cp = {
        "planet": planet,
        "science_mode": science_mode,
        # Star identity for the eclipse normalization Fp/Fs: part of the MODEL
        # only in emission (zeroed in transmission, where it is noise-side).
        "star_teff": _teff,
        "star_logg": round(float(params.get("star_logg", star_ref["log_g"])), 2),
        "star_feh": round(float(params.get("star_feh",
                                           star_ref["metallicity"])), 2),
        "nz": nz,
        "yconv_cri": round(yconv_cri, 6),
        "rp_rjup": round(float(params.get("rp_rjup", sysd["rp_rjup"])), 4),
        "gs_cgs": round(float(params.get("gs_cgs", sysd["gs_cgs"])), 1),
        "rstar_rsun": _rstar,
        "orbit_au": _orbit,
        "sflux": sflux,
        "met_x_solar": round(float(params.get("met_x_solar", 10.0)), 4),
        "co_ratio": round(float(params.get("co_ratio", CO_DEFAULT)), 6),
        # Kzz default follows the structure: a tabulated T-P with a Kzz column
        # supplies the mixing profile too, otherwise constant.
        "kzz_mode": str(params.get(
            "kzz_mode",
            "file" if (tp_mode == "file" and tp_table is not None
                       and tp_table["Kzz"] is not None) else "const")),
        "kzz_x": round(float(params.get("kzz_x", 1.0)), 4),
        "kzz_const": round(float(params.get("kzz_const", 1.0e9)), 1),
        # parametric Kzz profiles: Pfunc (kzz_kmax cm^2/s, kzz_plev bar) and
        # JM16 (kzz_kdeep); unused knobs are zeroed below for cache hygiene
        "kzz_kmax": round(float(params.get("kzz_kmax", 1.0e5)), 1),
        "kzz_plev": float(f"{float(params.get('kzz_plev', 0.1)):.6e}"),
        "kzz_kdeep": round(float(params.get("kzz_kdeep", 1.0e5)), 1),
        "tp_mode": tp_mode,
        # file-mode identity: source label + content hash, so two tables can
        # never share a cache entry ("" outside file mode)
        "tp_file": tp_file,
        "tp_file_sha1": tp_file_sha1,
        # sqrt(2) * T_eq of the SELECTED planet (the DAYSIDE T_eq in
        # emission), via planets.default_tirr so API and GUI cannot diverge
        "Tirr": round(float(params.get("Tirr", default_tirr(
            planet, system=dict(star_teff=_teff, rstar_rsun=_rstar,
                                orbit_au=_orbit),
            science_mode=science_mode))), 2),
        "Tint": round(float(params.get("Tint", 100.0)), 2),
        # 0.01 cm^2/g, Guillot (2010); the GUI default (app.py) must agree
        "log_kappa": round(float(params.get("log_kappa", -2.0)), 3),
        "log_gamma": round(float(params.get("log_gamma", -1.0)), 3),
        # physical VULCAN knobs (all flow through cfg_overrides; the defaults
        # are the W39b cfg values)
        "use_photo": bool(params.get("use_photo", True)),
        "sl_angle_deg": round(float(params.get("sl_angle_deg", 83.0)), 1),
        "f_diurnal": round(float(params.get("f_diurnal", 1.0)), 3),
        "use_moldiff": bool(params.get("use_moldiff", True)),
        # Upwind molecular-diffusion advection: PINNED explicitly, never
        # inherited from the engine. False is the validated baseline here.
        "use_vm_mol": bool(params.get("use_vm_mol", False)),
        # Rayleigh is zero-parameter physics, ON by default (off it biases the
        # <1.5 um slope); the power-law cloud deck is OFF by default.
        "use_rayleigh": bool(params.get("use_rayleigh", True)),
        # ExoJAX RT knobs. rt_ptop_bar is the MODEL top and the chemistry grid
        # follows it (engine rule), so no RT layer is clamped above the
        # chemistry. rt_integration picks the ArtTransPure chord scheme.
        "rt_ptop_bar": float(f"{float(params.get('rt_ptop_bar', RT_PTOP_DEFAULT)):.6e}"),
        "rt_integration": str(params.get("rt_integration", "simpson")),
        # The pressure rp_rjup and gs_cgs apply at. A catalogue radius is the
        # transit radius near the terminator photosphere, NOT the RT-grid
        # bottom exojax defaults to; anchoring it there inflates every depth.
        "p_ref_bar": float(f"{float(params.get('p_ref_bar', 1.0e-3)):.6e}"),
        # chemistry + RT column bottom (see the P_BTM_* block at module top)
        "p_btm_bar": round(p_btm_bar, 4),
        "cloud_on": bool(params.get("cloud_on", False)),
        "log_kappa_cloud": round(float(params.get("log_kappa_cloud", -1.0)), 3),
        "alpha_cloud": round(float(params.get("alpha_cloud", 0.0)), 2),
        # Detection-only condensation: the raises below refuse ANY derivative
        "use_condense": bool(params.get("use_condense", False)),
        # removed and refused above; pinned off so the cache key is unchanged
        "use_settling": False,
        "diff_esc": [],
        "top_flux": [],
        "bot_flux": [],
        "extra_mols": sorted(str(m) for m in (params.get("extra_mols") or [])),
        # Leave-one-out spectrum set: None = every RT molecule (the detect
        # default), [] skips the block. Canonicalized to fold order below.
        "wo_mols": (None if params.get("wo_mols") is None
                    else [str(m) for m in params.get("wo_mols")]),
        "fisher_params": sorted(str(p) for p in (params.get("fisher_params") or [])),
        # Jacobian method: "fd" (certified FD, default, valid everywhere) or
        # "ad" (one warm-started jvp per row, photo-on only; module docstring).
        "jac_method": str(params.get("jac_method", "fd")),
        "chem_provider": provider,
        "network": network,
        "version": _VERSION,
    }
    if not 0.0 <= cp["sl_angle_deg"] <= 89.0:
        raise ValueError(f"sl_angle_deg={cp['sl_angle_deg']} outside [0, 89] deg")
    if not 0.0 < cp["f_diurnal"] <= 1.0:
        raise ValueError(f"f_diurnal={cp['f_diurnal']} outside (0, 1]")
    if not 1.0e-9 <= cp["rt_ptop_bar"] <= 1.0e-6:
        raise ValueError(
            f"rt_ptop_bar={cp['rt_ptop_bar']:g} outside [1e-9, 1e-6] bar (the "
            f"exercised RT-top range; {RT_PTOP_DEFAULT:g} is the default)")
    _art_pbtm = cp["p_btm_bar"] * ART_PBTM_FRACTION
    if not cp["rt_ptop_bar"] <= cp["p_ref_bar"] <= _art_pbtm:
        raise ValueError(
            f"p_ref_bar={cp['p_ref_bar']:g} must lie inside the RT grid "
            f"[{cp['rt_ptop_bar']:g}, {_art_pbtm:g}] bar. It is the pressure "
            "at which rp_rjup and gs_cgs are defined, so the grid has to "
            "cover it. 1e-3 bar is the validated default for a "
            "transit-derived radius.")
    if cp["rt_integration"] not in ("simpson", "trapezoid"):
        raise ValueError(
            f"rt_integration={cp['rt_integration']!r}: exojax ArtTransPure "
            "supports 'simpson' (default) or 'trapezoid'")
    no_table = sorted(set(cp["extra_mols"]) & _NO_EXOMOLOP_TABLE)
    if no_table:   # before the universe check: the pointed reason wins
        raise ValueError(
            f"extra_mols {no_table} have no published ExoMolOP k-table and "
            "cannot be modelled (correlated-k over the published tables is "
            "the only opacity path). Drop them.")
    bad_mols = set(cp["extra_mols"]) - set(EXTRA_MOLECULES)
    if bad_mols:
        raise ValueError(
            f"unknown RT molecule(s) {sorted(bad_mols)}: this tool ships "
            f"opacity for {MOLECULES} plus the opt-in extras "
            f"{EXTRA_MOLECULES}. Adding one means extending the shared "
            "vulcan-forward engine (inject profile['molecule_table'] or "
            "extend vulcan_forward.constants.MOLECULES with molmass + VULCAN "
            "species name), fetching its ExoMolOP k-table, and listing it in "
            "forward.EXTRA_MOLECULES.")
    if network == "ncho":
        _s_req = sorted(set(cp["extra_mols"]) & _S_MOLECULES)
        if _s_req:
            raise ValueError(
                f"extra_mols {_s_req} are sulfur species and do not exist in "
                "the ncho network. Drop them, or keep network='sncho'. "
                "(Sulfur species are refused rather than dropped, so the "
                "model computed is always the model asked for.)")
    # --- leave-one-out spectrum set -----------------------------------------
    # After the molecule-universe checks, so the RT set is final. Canonical form
    # is a deduped subset of active_molecules(cp) in fold order, which keeps the
    # cache key and the stored depth_wo rows in one order.
    _active = active_molecules(cp)
    if cp["wo_mols"] is None:
        cp["wo_mols"] = list(_active)
    else:
        _bad_wo = sorted(set(cp["wo_mols"]) - set(_active))
        if _bad_wo:
            raise ValueError(
                f"wo_mols {_bad_wo} are not in this run's RT molecule set "
                f"{_active}. A removed-molecule spectrum exists only for "
                "molecules the RT actually includes; fix wo_mols or add the "
                "molecule via extra_mols.")
        _req = set(cp["wo_mols"])
        cp["wo_mols"] = [m for m in _active if m in _req]
    if not 0.1 <= cp["co_ratio"] <= 2.0:
        raise ValueError(
            f"co_ratio={cp['co_ratio']} outside [0.1, 2.0] (the network was "
            "never exercised beyond this range)")
    if not 0.1 <= cp["met_x_solar"] <= 100.0:
        raise ValueError(
            f"met_x_solar={cp['met_x_solar']} outside [0.1, 100] x solar")
    # Fisher menu: chemistry + the tp_mode's T-P parameters (none in file mode)
    # + the cloud-deck parameters when the deck is in the model.
    allowed_fp = {"lnZ", "dlnCO", "lnKzz"} | set(TP_PARAM_NAMES[tp_mode])
    if cp["cloud_on"]:
        allowed_fp |= set(CLOUD_FISHER_PARAMS)
    bad_fp = set(cp["fisher_params"]) - allowed_fp
    if bad_fp:
        raise ValueError(
            f"unknown Fisher parameter(s) {sorted(bad_fp)} for tp_mode="
            f"{tp_mode!r}: choose from {CHEM_PARAM_NAMES} + "
            f"{TP_PARAM_NAMES[tp_mode]}"
            + (f" + {list(CLOUD_FISHER_PARAMS)}" if cp["cloud_on"] else "")
            + (". (tp_mode='file' has NO T-P Fisher rows by design; the "
               f"cloud parameters {list(CLOUD_FISHER_PARAMS)} require "
               "cloud_on.)"))
    if cp["jac_method"] not in JAC_METHODS:
        raise ValueError(
            f"jac_method={cp['jac_method']!r}: choose 'fd' (certified central "
            "finite differences, the default) or 'ad' (one warm-started jvp "
            "per Jacobian row)")
    if cp["jac_method"] == "ad" and not cp["fisher_params"]:
        cp["jac_method"] = "fd"   # no Jacobian requested: inert knob --
        #                           normalize so it can't fragment the cache
    if cp["jac_method"] == "ad" and not cp["use_photo"]:
        raise ValueError(
            "jac_method='ad' (warm-started jvp Jacobian rows) is validated "
            "only in the photo-on regime. Enable photochemistry, or use the "
            "default certified finite differences (jac_method='fd'), which "
            "work photo-off too.")
    # Composition FD rows solve the chemistry at every stencil point, so each
    # point must stay inside the validated envelope (a row requested AT a
    # range edge would otherwise silently solve outside it). T-P rows
    # window-check every stencil point for the same reason.
    if cp["jac_method"] == "fd":
        for name, key, rng in (("lnZ", "met_x_solar", (0.1, 100.0)),
                               ("dlnCO", "co_ratio", (0.1, 2.0))):
            if name not in cp["fisher_params"]:
                continue
            offs, h = fd_stencil(name, cp[key])
            pts = [cp[key] * float(np.exp(s * h)) for s in offs]
            if not all(rng[0] <= p <= rng[1] for p in pts):
                raise ValueError(
                    f"{key}={cp[key]:g}: the {name} FD stencil would solve "
                    f"the chemistry at {[round(p, 3) for p in pts]}, outside "
                    f"the validated range {list(rng)}. Move {key} inward or "
                    f"drop {name} from fisher_params.")
    # --- condensation: detection-only -- refuse every derivative combo -----
    # (the "91% wrong" and "not a 9% mismatch" wording is test-pinned)
    if cp["use_condense"]:
        if network == "ncho":
            raise ValueError(
                "condensation (use_condense) requires network='sncho': the "
                "certified recipe condenses S8, which does not exist in the "
                "sulfur-free ncho network. Keep the sncho network or turn "
                "condensation off.")
        if cp["fisher_params"]:
            raise ValueError(
                "condensation (use_condense) cannot be combined with a "
                "Fisher forecast under ANY Jacobian method: the pinned "
                "reservoir is frozen at a step-sequence-dependent transient "
                "and the condensing-layer set switches discretely in "
                "temperature, so the AD tangent through it is about 91% "
                "wrong (jvp-vs-FD relative error ~0.91, not a 9% mismatch) "
                "and finite differences of it are no better. Clear "
                "fisher_params (detection still works), or use the "
                "differentiable ExoJAX cloud deck (cloud_on) for aerosol "
                "opacity.")
        if not cp["use_photo"]:
            raise ValueError(
                "condensation (use_condense) requires photochemistry ON: "
                "a cold no-photo condensing column has no certifiable longdy "
                "steady state, and a runtime-capped solve is never presented "
                "as converged. Enable photochemistry, or turn condensation "
                "off.")
        if not cp["use_moldiff"]:
            raise ValueError(
                "condensation (use_condense) requires molecular diffusion "
                "(use_moldiff): the growth term IS the species' "
                "molecular-diffusion coefficient, so every condensation rate "
                "would silently be zero. Enable it, or turn condensation "
                "off.")
    if not cp["use_photo"]:            # photolysis knobs are inert without photo
        cp["sl_angle_deg"] = 0.0
        cp["f_diurnal"] = 1.0
        # The UV spectrum feeds only photolysis: normalize it photo-off so
        # identical physics never caches under two keys.
        cp["sflux"] = str(sysd["sflux"])
        cp["orbit_au"] = round(float(sysd["orbit_au"]), 5)
    if not cp["use_moldiff"]:          # upwind vm_mol is inert without moldiff
        cp["use_vm_mol"] = False       # (engine gates use_vm on both); keep the
                                       # key from fragmenting the cache
    if not cp["cloud_on"]:             # cloud knobs are inert when the deck is off
        cp["log_kappa_cloud"] = 0.0
        cp["alpha_cloud"] = 0.0
    # --- science-mode hygiene + gating --------------------------------------
    if science_mode == "emission":
        if not 3000.0 <= cp["star_teff"] <= 7000.0:
            raise ValueError(
                f"star_teff={cp['star_teff']:g} outside [3000, 7000] K (the "
                "range exercised against the PHOENIX grid for Fp/Fs)")
        if not 3.0 <= cp["star_logg"] <= 5.5:
            raise ValueError(f"star_logg={cp['star_logg']:g} outside [3.0, 5.5]")
        if not -2.5 <= cp["star_feh"] <= 0.5:
            raise ValueError(f"star_feh={cp['star_feh']:g} outside [-2.5, 0.5]")
        # Rayleigh is transmission-only physics (the pure-absorption emission
        # solver must not read scattering as thermal absorption) and the chord
        # scheme exists only in transmission: normalize both for the cache.
        cp["use_rayleigh"] = False
        cp["rt_integration"] = "simpson"
    else:
        # transmission: the star lives only on the noise side.
        cp["star_teff"] = cp["star_logg"] = cp["star_feh"] = 0.0
    # drop fields inert for the chosen modes so they don't fragment the cache
    if tp_mode != "guillot":
        cp["Tirr"] = cp["Tint"] = cp["log_kappa"] = cp["log_gamma"] = 0.0
    # --- Kzz profile mode (const / Pfunc / JM16 / file) ---------------------
    if cp["kzz_mode"] not in KZZ_MODES:
        raise ValueError(
            f"unknown kzz_mode {cp['kzz_mode']!r}: choose from "
            f"{list(KZZ_MODES)}. Kzz profiles are explicit -- the GCM-scaled "
            "'scale' mode was removed.")
    if not 0.01 <= cp["kzz_x"] <= 100.0:
        raise ValueError(
            f"kzz_x={cp['kzz_x']} outside [0.01, 100] (multiplicative scale "
            "applied on-graph to the whole Kzz profile)")
    if cp["kzz_mode"] == "const":
        if not 1.0e3 <= cp["kzz_const"] <= 1.0e13:
            raise ValueError(
                f"kzz_const={cp['kzz_const']:g} outside [1e3, 1e13] cm^2/s")
    elif cp["kzz_mode"] == "Pfunc":
        if not 1.0e3 <= cp["kzz_kmax"] <= 1.0e12:
            raise ValueError(
                f"kzz_kmax={cp['kzz_kmax']:g} outside [1e3, 1e12] cm^2/s "
                "(Pfunc deep Kzz)")
        if not 1.0e-6 <= cp["kzz_plev"] <= 1.0e3:
            raise ValueError(
                f"kzz_plev={cp['kzz_plev']:g} outside [1e-6, 1e3] bar "
                "(Pfunc transition pressure)")
    elif cp["kzz_mode"] == "JM16":
        if not 1.0e3 <= cp["kzz_kdeep"] <= 1.0e12:
            raise ValueError(
                f"kzz_kdeep={cp['kzz_kdeep']:g} outside [1e3, 1e12] cm^2/s "
                "(JM16 deep floor)")
    elif cp["kzz_mode"] == "file":
        if tp_mode != "file":
            raise ValueError(
                "kzz_mode='file' requires tp_mode='file': the tabulated Kzz "
                "lives in the Kzz column of the atm table (the upstream "
                "constraint -- Kzz_prof='file' needs atm_type='file').")
        if tp_table is None or tp_table["Kzz"] is None:
            raise ValueError(
                "kzz_mode='file' requires a 'Kzz' column in the T-P table; "
                "the selected table has none. Add the column (cm^2/s) or "
                "pick a parametric kzz_mode.")
    # inert-knob zeroing (cache hygiene): only the active mode's knobs key
    if cp["kzz_mode"] != "const":
        cp["kzz_const"] = 0.0
    if cp["kzz_mode"] != "Pfunc":
        cp["kzz_kmax"] = cp["kzz_plev"] = 0.0
    if cp["kzz_mode"] != "JM16":
        cp["kzz_kdeep"] = 0.0
    return cp


def params_key(params: dict) -> str:
    s = json.dumps(canonical_params(params), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def cache_path(params: dict) -> Path:
    return MODEL_CACHE / f"{params_key(params)}.npz"


# Canonical keys that change the SPECTRUM but never the converged chemistry
# column, so chem_key strips them and an RT-only edit reuses the solved column
# (adjoint_diag.adjoint_key's strip list plus p_ref_bar). Dual-use keys -- nz,
# p_btm_bar, rt_ptop_bar (the chemistry top follows it), rp_rjup, gs_cgs, T-P,
# composition, Kzz -- stay in the key, as does "version" via canonical_params.
CHEM_IRRELEVANT_PARAMS = (
    "fisher_params", "jac_method", "use_rayleigh",
    "cloud_on", "log_kappa_cloud", "alpha_cloud", "extra_mols", "wo_mols",
    "rt_integration",
    "science_mode", "star_teff", "star_logg", "star_feh", "p_ref_bar",
)


def chem_key(params: dict) -> str:
    cp = canonical_params(params)
    payload = {k: v for k, v in cp.items()
               if k not in CHEM_IRRELEVANT_PARAMS}
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def chem_cache_path(params: dict) -> Path:
    return MODEL_CACHE / f"chem-{chem_key(params)}.npz"


def _load_cached_npz(p: Path):
    """Cached npz as a dict, or None when absent. A file that exists but does
    not read back as a complete npz (torn write, damaged disk) is quarantined
    to ``<name>.corrupt-<t>`` and treated as a miss, so the entry recomputes
    and the damaged bytes stay on disk. Also serves adjoint_diag's cache."""
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (zipfile.BadZipFile, zlib.error, OSError, EOFError,
            ValueError) as e:
        quarantine = p.with_name(f"{p.name}.corrupt-{int(time.time())}")
        try:
            p.rename(quarantine)
        except OSError:
            pass                # a concurrent reader already moved it
        print(f"[cache] {p.name} unreadable ({e!r}); quarantined -> "
              f"{quarantine.name}, treating as a cache miss", flush=True)
        return None


def load_result(params: dict):
    """Cached spectrum dict or None.

    Always present: wl_um, depth, depth_wo (n_wo, n_nu), wo_mols (the
    leave-one-out set depth_wo rows align with), mols, ymix, p_bar, T, theta,
    theta_names, params_json, chem_provider, and the convergence certificate
    (conv_stages, conv_accept, conv_longdy, conv_gate). With Fisher requested:
    jac (n_par, n_nu), jac_names, jac_row_method, fd_h, fd_err.
    """
    return _load_cached_npz(cache_path(params))


# Heavy path (script mode only below this line)

def _build_tp(cp: dict, gs_cgs: float):
    """(tp_eval, n_tp, tp_values, theta_names) for the chosen T-P mode.

    tp_eval(tp_params, p_bar) is pure JAX (differentiable) for the parametric
    modes. In file mode tp_eval is None: the engine's temperature path is
    T = T_base + theta[3] over the pre-loop's re-grid of the tabulated profile
    (atm_type="file"), and theta[3] is PINNED to 0, so there is no dT
    parameter and no T-P Fisher row.
    """
    import jax.numpy as jnp

    mode = cp["tp_mode"]
    if mode == "guillot":
        from exojax.atm.atmprof import atmprof_Guillot

        def tp_eval(tp, p_bar):
            p = jnp.asarray(p_bar)
            Tirr, Tint = tp[0], tp[1]
            kappa, gamma = 10.0 ** tp[2], 10.0 ** tp[3]
            return atmprof_Guillot(p, gs_cgs, kappa, gamma, Tint, Tirr, 0.25)
        vals = [cp["Tirr"], cp["Tint"], cp["log_kappa"], cp["log_gamma"]]
        return tp_eval, 4, vals, CHEM_PARAM_NAMES + TP_PARAM_NAMES["guillot"]
    if mode == "file":
        # theta keeps its 4th slot for the engine's uniform-shift path, pinned
        # to 0.0 and named "dT"
        return None, 0, [0.0], CHEM_PARAM_NAMES + ["dT"]
    raise ValueError(f"unknown tp_mode {mode!r}")


def _make_progress(cp: dict, log):
    """Sequential stage tracker: emits "[fwd] PROG <frac> <label>" lines.

    The stage list MUST mirror run_model's actual stage order (same
    conditionals); weights are rough wall-clock seconds. advance() is called
    at the START of each stage.
    """
    _emis = cp.get("science_mode") == "emission"
    stages = [("building chemistry model", 5.0)]
    stages += [("building radiative transfer (opacities + CIA)",
                10.0 + 3.0 * len(cp["extra_mols"]))]
    if _emis:
        stages += [("emission model + stellar SED", 6.0)]
    stages += [("solving photochemistry", 35.0)]
    # The leave-one-out block is one engine batch, not one stage per molecule;
    # each wo spectrum refolds from its position to the end of the set.
    _n_wo = len(cp["wo_mols"])
    _n_active = len(active_molecules(cp))
    _wo_w = 1.5 * _n_wo * (_n_active + 1)
    # ONE stage either way: baseline and removed-molecule spectra come out of a
    # single engine batch in both science modes.
    if _n_wo:
        stages += [(f"full + removed-molecule spectra ({_n_wo} molecules)",
                    8.0 + _wo_w)]
    else:
        stages += [(f"full {'eclipse' if _emis else 'transmission'} spectrum",
                    8.0)]
    # Jacobian rows: fd = one build+solve per stencil point, cloud rows are
    # RT-only, ad = one warm jvp per chemistry row (one stage for all)
    _ad = cp["jac_method"] == "ad"

    def _row_stage(n):
        if n in CLOUD_FISHER_PARAMS:
            return (f"{'AD' if _ad else 'FD'} Jacobian d/d({n})", 8.0)
        return (f"FD Jacobian d/d({n})",
                280.0 if n in FD_COMP_PARAMS else 260.0)

    if _ad:
        # chemistry-theta rows run back to back in one stage; RT-only deck
        # rows keep their own per-row stages
        _chem_rows = [n for n in cp["fisher_params"]
                      if n not in CLOUD_FISHER_PARAMS]
        if _chem_rows:
            stages += [(f"AD Jacobian ({len(_chem_rows)} rows)",
                        110.0 * len(_chem_rows))]
    stages += [_row_stage(n) for n in cp["fisher_params"]
               if not _ad or n in CLOUD_FISHER_PARAMS]
    if cp["fisher_params"]:
        stages += [(("AD" if _ad else "FD") + " Jacobian d/d(lnR0)", 8.0)]
    total = sum(w for _, w in stages)
    state = {"i": 0, "done": 0.0}

    def advance():
        label, w = stages[state["i"]]
        log(f"[fwd] PROG {state['done'] / total:.3f} {label}")
        state["i"] += 1
        state["done"] += w

    def finish():
        log("[fwd] PROG 1.000 done")

    return advance, finish


def _rt_profile_common(cp: dict, config) -> dict:
    """The RT-facing profile keys (exojax_rt / build_emis_model read exactly
    these); pinned by tests/unit/test_rt_profile_golden.py."""
    profile = dict(config.WIDE)
    # The COUNTS are locked equal for one resolution control; ExoJAX builds its
    # own pressure coordinates and interp_map maps chemistry onto them.
    profile["nz"] = cp["nz"]
    profile["art_nlayer"] = cp["nz"]
    # The engine ECHOES these on the built rt namespace and run_model verifies
    # the echo, so an engine ignoring a profile key cannot return a spectrum
    # differing from what the cache key claims.
    profile["art_ptop_bar"] = cp["rt_ptop_bar"]
    # RT bottom, just inside the chemistry bottom (interp_map refuses an ART
    # grid reaching below it)
    profile["art_pbtm_bar"] = cp["p_btm_bar"] * ART_PBTM_FRACTION
    profile["rt_integration"] = cp["rt_integration"]
    profile["p_ref_bar"] = cp["p_ref_bar"]
    # Correlated-k over the published ExoMolOP k-tables is the engine's only
    # opacity path; set explicitly so the echo check can verify it.
    profile["opacity_mode"] = "exomolop"
    profile["molecules"] = active_molecules(cp)
    profile["use_photo"] = cp["use_photo"]        # build_chem_model reads this key
    profile["use_rayleigh"] = cp["use_rayleigh"]  # exojax_rt reads this flag
    # --- planet identity ----------------------------------------------------
    rp_cm = cp["rp_rjup"] * planets.R_JUP_CM
    rstar_cm = cp["rstar_rsun"] * planets.R_SUN_CM
    profile["rp_cm"] = rp_cm            # RT geometry (exojax_rt reads these)
    profile["gs_cgs"] = cp["gs_cgs"]
    profile["rstar_cm"] = rstar_cm
    return profile


def _assemble_chem(cp: dict, log):
    """Shared heavy-path assembly (run_model AND adjoint_diag): the resolved
    run profile with the structural composition pinned into cfg_overrides, the
    on-graph T-P hook, theta, and a chemistry-build factory. One code path, so
    the adjoint diagnostics analyze exactly the model the forecasts ran."""
    # import order is load-bearing: vulcan_chem (env + x64) before jax/exojax
    from types import SimpleNamespace

    from jwst_tool import engine_config as config

    # Non-default kinetics network: the engine freezes network/atom_list at ITS
    # first import, so the selection must land before vulcan_chem arrives; a
    # conflicting in-process import raises.
    _net_path, _net_atoms = NETWORKS[cp["network"]]
    if cp["network"] != "sncho":
        os.environ["VULCAN_JAX_NETWORK"] = _net_path
        os.environ["VULCAN_JAX_ATOM_LIST"] = _net_atoms
        log(f"[fwd] kinetics network: {cp['network']} ({_net_path}, "
            f"atoms {_net_atoms})")
    from vulcan_forward import vulcan_chem
    import jax

    # Persistent XLA compile cache: ESSENTIAL for adjoint_diag, whose step-VJP
    # is a multi-hour cold compile on CPU.
    jax.config.update("jax_compilation_cache_dir",
                      str(Path.home() / ".cache" / "jax_vulcan"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    tp_eval, n_tp, tp_vals, theta_names = _build_tp(cp, cp["gs_cgs"])
    # Composition is set STRUCTURALLY in the cfg abundances (below), never as a
    # parameter perturbation, so lnZ and c_o are exactly 0; only lnKzz and the
    # T-P parameters are live theta directions.
    theta = np.asarray(
        vulcan_chem.ChemParams(lnZ=0.0, c_o=0.0, lnKzz=np.log(cp["kzz_x"]),
                               tp=tuple(tp_vals)).to_vector(),
        dtype=np.float64)
    log(f"[fwd] params {cp}")
    log(f"[fwd] theta {dict(zip(theta_names, np.round(theta, 4)))}")

    profile = _rt_profile_common(cp, config)
    profile["yconv_cri"] = cp["yconv_cri"]
    # exact-elemental abundance map (see the vulcan_chem docstring)
    profile["abundance_mode"] = "elemental"
    profile["co_mode"] = "fixed_O"
    profile["reanchor_atom_ini"] = True   # finite-Z steps must re-anchor atom totals
    # step-size cap: prevents the adaptive-dt non-convergence at high Kzz
    profile["dt_max"] = 1.0e11
    rp_cm = profile["rp_cm"]
    ovr = {                              # chemistry side (applied pre-pre-loop)
        # VULCAN derives g = G*Mp/Rp^2, so convert gs_cgs to a planet mass
        "Mp": cp["gs_cgs"] * rp_cm**2 / planets.G_CGS,
        # the cfg network must agree with the import-frozen one; S_H stays in
        # the cfg under ncho, harmless with an S-free atom_list
        **({"network": _net_path,
            "atom_list": _net_atoms.split(",")} if cp["network"] != "sncho"
           else {}),
        "Rp": rp_cm, "r_star": cp["rstar_rsun"],
        "orbit_radius": cp["orbit_au"],
        "sflux_file": f"atm/stellar_flux/{cp['sflux']}",
        "use_moldiff": cp["use_moldiff"],
        # pin the vm_mol scheme EXPLICITLY, never inherit the upstream YAML
        # default; the two flags travel together (hybrid is how vm_mol runs)
        "use_vm_mol": cp["use_vm_mol"],
        "use_hybrid_vm_mol": cp["use_vm_mol"],
    }
    if cp["use_photo"]:                  # photolysis geometry/averaging knobs
        ovr["sl_angle"] = float(np.deg2rad(cp["sl_angle_deg"]))
        ovr["f_diurnal"] = cp["f_diurnal"]
    if cp["use_condense"]:
        # canonical_params already confirmed detection-only, photo on and
        # moldiff on; the engine rebuilds the arrays on-graph per solve
        ovr.update(CONDEN_CFG)
    # Structural baseline. Guillot mode: isothermal -- the on-graph tp_eval
    # supplies the actual T(P), the structural profile only sets the hydrostatic
    # grid + EQ init. File mode: the table IS the structure, hash-verified.
    if cp["tp_mode"] == "file":
        tp_path = _tp_file_from_cp(cp)   # sha-verified re-resolution
        ovr.update({"atm_type": "file", "atm_file": str(tp_path)})
        log(f"[fwd] planet {cp['planet']}: tabulated T-P structure from "
            f"{tp_path.name} (sha1 {cp['tp_file_sha1']}), UV = {cp['sflux']}")
    else:
        T_struct = cp["Tirr"] / np.sqrt(2.0)   # guillot: ~equilibrium T at f=0.25
        ovr.update({"atm_type": "isothermal", "Tiso": float(T_struct)})
        log(f"[fwd] planet {cp['planet']}: isothermal structural baseline "
            f"{T_struct:.0f} K, UV = {cp['sflux']}")
    # Kzz profile (lnKzz = theta[2] scales ANY of them on-graph).
    if cp["kzz_mode"] == "const":
        ovr.update({"Kzz_prof": "const", "const_Kzz": cp["kzz_const"]})
        log(f"[fwd] Kzz: const {cp['kzz_const']:.1e} cm2/s")
    elif cp["kzz_mode"] == "Pfunc":
        ovr.update({"Kzz_prof": "Pfunc", "K_max": cp["kzz_kmax"],
                    "K_p_lev": cp["kzz_plev"]})
        log(f"[fwd] Kzz: Pfunc deep {cp['kzz_kmax']:.1e} cm2/s rising as "
            f"P^-0.4 above {cp['kzz_plev']:g} bar")
    elif cp["kzz_mode"] == "JM16":
        ovr.update({"Kzz_prof": "JM16", "K_deep": cp["kzz_kdeep"]})
        log(f"[fwd] Kzz: JM16 (deep floor {cp['kzz_kdeep']:.1e} cm2/s, "
            "1e5 (300 mbar/P)^0.5 above)")
    else:                                # "file": gated to tp_mode="file"
        ovr.update({"Kzz_prof": "file"})
        log("[fwd] Kzz: tabulated column of the T-P table")
    # Chemistry-grid bottom: a run needing a deeper column overrides P_b here
    # rather than editing the YAML the siblings were validated on.
    if abs(cp["p_btm_bar"] - P_BTM_FILE_BAR) > 1e-9:
        ovr["P_b"] = cp["p_btm_bar"] * 1.0e6
        log(f"[fwd] chemistry grid bottom {cp['p_btm_bar']:g} bar "
            f"(cfg default {P_BTM_FILE_BAR:g} bar), RT bottom "
            f"{cp['p_btm_bar'] * ART_PBTM_FRACTION:g} bar; "
            f"{cp['nz'] / np.log10(cp['p_btm_bar'] / cp['rt_ptop_bar']):.0f} "
            "layers per decade")
    profile["cfg_overrides"] = ovr

    # CO_BASELINE must equal the loaded cfg's C_H/O_H: a wrong basis here
    # mislabels every C/O display.
    import vulcan_jax as _vj
    _cfg_chk = _vj.load_config(profile.get("vulcan_cfg_name") or config.W39B_CFG_NAME)
    _co_cfg = float(_cfg_chk.C_H) / float(_cfg_chk.O_H)
    if abs(_co_cfg / CO_BASELINE - 1.0) > 1e-9:
        raise RuntimeError(
            f"forward.CO_BASELINE={CO_BASELINE:.5f} no longer matches the "
            f"network cfg's C_H/O_H={_co_cfg:.5f}: the C/O display baseline "
            "would be mislabeled. Update CO_BASELINE to the cfg value (and "
            "bump _VERSION).")
    # CHEM_P_SPAN_DYN must match the live cfg too: the light path's
    # bottom-coverage refusal keys on it.
    _span_cfg = (float(_cfg_chk.P_t), float(_cfg_chk.P_b))
    if any(abs(a / b - 1.0) > 1e-9
           for a, b in zip(_span_cfg, CHEM_P_SPAN_DYN)):
        raise RuntimeError(
            f"forward.CHEM_P_SPAN_DYN={CHEM_P_SPAN_DYN} no longer matches "
            f"the network cfg's (P_t, P_b)={_span_cfg}: the T-P table "
            "span validation would gate against the wrong grid. Update the "
            "constant (and bump _VERSION).")
    if cp["tp_mode"] == "file":
        # Log the conventional TOP clamp (the bottom is hard-gated at the API)
        _P_tab = _read_tp_table(tp_path, span=chem_p_span_dyn(cp))["P_dyn"]
        _p_top_run = chem_p_span_dyn(cp)[0]
        _dec = float(np.log10(_P_tab.min() / _p_top_run))
        if _dec > 0.0:
            log(f"[fwd] NOTE: T-P table top ({_P_tab.min():.3g} dyn/cm^2) "
                f"sits {_dec:.1f} decades below the chemistry-grid top "
                f"({_p_top_run:g}): the topmost tabulated T is held constant "
                "over that range (the upstream file-mode convention).")

    def _abundance_overrides(met_x_solar: float, co_ratio: float) -> dict:
        # Structural composition: scale the cfg metals together (He fixed), then
        # set carbon from the requested C/O at the scaled oxygen. FastChem
        # re-initializes here; fastchem_met_scale follows for trace metals.
        m = met_x_solar / 10.0                 # cfg abundances ARE 10x solar
        o_h = float(_cfg_chk.O_H) * m
        return {"O_H": o_h, "C_H": co_ratio * o_h,
                "N_H": float(_cfg_chk.N_H) * m,
                "S_H": float(_cfg_chk.S_H) * m,
                "fastchem_met_scale": float(met_x_solar)}

    ovr.update(_abundance_overrides(cp["met_x_solar"], cp["co_ratio"]))
    log(f"[fwd] structural composition: {cp['met_x_solar']:g}x solar metals, "
        f"C/O = {cp['co_ratio']:.3f} (C_H {ovr['C_H']:.3e}, O_H {ovr['O_H']:.3e})")

    def _build_chem(extra_abun: dict | None = None, tag: str = "baseline"):
        prof = dict(profile)
        prof["cfg_overrides"] = ({**ovr, **extra_abun} if extra_abun else ovr)
        # skip the engine's build-time warm-up SOLVE: this tool certifies its
        # own solves and never reads baseline_conv_normal.
        prof["skip_warmup"] = True
        t_b = time.time()
        chem_b = vulcan_chem.build_chem_model(prof, tp_eval=tp_eval,
                                              n_tp_params=n_tp)
        log(f"[fwd] chemistry model ({tag}) ready in {time.time()-t_b:.0f} s")
        return chem_b

    return SimpleNamespace(
        profile=profile, theta=theta, theta_names=theta_names,
        tp_eval=tp_eval, n_tp=n_tp, build_chem=_build_chem,
        abundance_overrides=_abundance_overrides, config=config)


def _check_t_window(tp_eval, theta, p_bar, log, T_base=None):
    """T-P validity on the chemistry grid: REJECT (never clip) out-of-window
    profiles; returns the evaluated T(P). In file mode (tp_eval None) the
    profile IS the structure, so pass T_base (chem.T_base)."""
    import jax.numpy as jnp

    if tp_eval is None:
        if T_base is None:
            raise ValueError("_check_t_window: tp_eval=None (file mode) "
                             "requires the T_base array")
        T_check = np.asarray(T_base, dtype=np.float64)
    else:
        T_check = np.asarray(tp_eval(jnp.asarray(theta[3:]), jnp.asarray(p_bar)))
    tmin, tmax = float(T_check.min()), float(T_check.max())
    if tmin < T_WINDOW[0] or tmax > T_WINDOW[1]:
        raise RuntimeError(
            f"T-P profile leaves the modelable window [{T_WINDOW[0]:.0f}, "
            f"{T_WINDOW[1]:.0f}] K (min {tmin:.0f} K, max {tmax:.0f} K). "
            "Adjust the profile parameters -- out-of-window layers are rejected, "
            "not clipped (the raw k-tables span a wider temperature range).")
    log(f"[fwd] T-P in window: [{tmin:.0f}, {tmax:.0f}] K")
    return T_check


def run_model(params: dict, log=print) -> Path:
    cp = canonical_params(params)
    # Uploaded T-P tables become a CONTENT-ADDRESSED copy under
    # <output>/uploads/<sha1>.txt before anything heavy runs, so canonical
    # params alone re-resolve the exact bytes later.
    if cp["tp_mode"] == "file" and cp["tp_file"] == TP_FILE_UPLOAD:
        src_path, sha1 = _resolve_tp_file(params)
        dst = _uploads_dir() / f"{sha1}.txt"
        if not dst.exists():
            _uploads_dir().mkdir(parents=True, exist_ok=True)
            # atomic: the exists() guard would make a torn copy permanent and
            # _tp_file_from_cp would then refuse this sha1 forever
            _ins.atomic_write(dst,
                              lambda fh: fh.write(src_path.read_bytes()))
            log(f"[fwd] uploaded T-P table archived -> {dst}")
    advance, finish = _make_progress(cp, log)
    A = _assemble_chem(cp, log)
    # heavy imports AFTER the assembler: vulcan_chem must init env/x64 first
    import jax
    import jax.numpy as jnp
    from vulcan_forward import exojax_rt
    from vulcan_forward import interp_map

    config = A.config
    profile, theta, theta_names = A.profile, A.theta, A.theta_names
    tp_eval, _abundance_overrides, _build_chem = (
        A.tp_eval, A.abundance_overrides, A.build_chem)
    mols_active = list(profile["molecules"])

    t0 = time.time()
    advance()
    log("[fwd] building chemistry model ...")
    chem = _build_chem()
    if cp["jac_method"] == "ad" and "dlnCO" in cp["fisher_params"]:
        # Refuse the AD dlnCO row within one FD stencil of O-exhaustion (see
        # CO_BZ_MIN_AD at module top) -- here, straight after the build that
        # sets co_bz_bound, so the refusal costs no solve. A missing engine
        # attribute means the check cannot run: refuse, never pass silently.
        _bz = getattr(chem, "co_bz_bound", None)
        if _bz is None:
            raise RuntimeError(
                "the sibling forward engine does not expose co_bz_bound "
                "(the fixed-O direction's oxygen-reservoir bound): the "
                "AD dlnCO row cannot be certified. Upgrade "
                "vulcan-forward or use jac_method='fd'.")
        if float(_bz) <= CO_BZ_MIN_AD:
            raise RuntimeError(
                f"The AD Jacobian for dlnCO is not available at C/O = "
                f"{cp['co_ratio']:g}: it needs co_bz_bound > {CO_BZ_MIN_AD:g} "
                f"(here {float(_bz):.3g}; roughly C/O below "
                f"{math.exp(-CO_BZ_MIN_AD):.2f}), the margin before the "
                "fixed-O direction exhausts the oxygen-only carriers. Use "
                "finite differences.")

    T_check = _check_t_window(tp_eval, theta, chem.p_bar, log,
                              T_base=getattr(chem, "T_base", None))

    t0 = time.time()
    advance()
    log("[fwd] building ExoJAX RT (opacities + CIA) ...")
    rt = exojax_rt.build_rt_model(profile)
    log(f"[fwd] RT ready in {time.time()-t0:.0f} s")
    # Echo check: an engine that ignores an unknown profile key must not cache
    # a spectrum under a key describing physics it skipped.
    _echo = {"art_ptop_bar": cp["rt_ptop_bar"],
             "art_pbtm_bar": profile["art_pbtm_bar"],
             "rt_integration": cp["rt_integration"],
             "p_ref_bar": cp["p_ref_bar"],
             "opacity_mode": profile["opacity_mode"]}
    for k, want in _echo.items():
        got = getattr(rt, k, None)
        if got != want:
            raise RuntimeError(
                f"RT engine did not honor {k}={want!r} (echoed {got!r}): the "
                "installed vulcan-forward engine does not read this profile "
                "key. Upgrade vulcan-forward.")

    # --- emission mode: day-side model + stellar SED ------------------------
    emis, fs_j = None, None
    # 1 / R_star^2 only: the PLANET radius for the eclipse prefactor comes from
    # emis.emission_radius(T, mmw) at the emission anchor (~0.1 bar), NOT the
    # catalogue transit radius (~1 mbar).
    _rstar_cm_sq = (cp["rstar_rsun"] * planets.R_SUN_CM) ** 2
    if cp["science_mode"] == "emission":
        advance()
        log("[fwd] building emission model + stellar SED ...")
        emis = exojax_rt.build_emis_model(rt, profile)
        from jwst_tool import stellar as stellar_mod
        fs_j = jnp.asarray(stellar_mod.phoenix_surface_flux(
            rt.nu_grid, cp["star_teff"], cp["star_logg"], cp["star_feh"],
            log=log))

    p_art_j = jnp.asarray(rt.p_art_bar)

    def _t_art_const_from(chem_b):
        """Tabulated-mode RT temperature: the build's T_base interpolated in
        ln P onto the ART grid. The chemistry grid covers the RT grid, so
        np.interp's edge clamp is inert."""
        _pb = np.asarray(chem_b.p_bar)
        _Tb = np.asarray(chem_b.T_base, dtype=np.float64)
        _order = np.argsort(_pb)
        return jnp.asarray(np.interp(
            np.log(np.asarray(rt.p_art_bar)),
            np.log(_pb[_order]), _Tb[_order]))

    if tp_eval is None:
        # tabulated file mode: ONE fixed profile for chemistry and RT.
        _T_art_const = _t_art_const_from(chem)

        def art_T(th):
            return _T_art_const
    else:
        def art_T(th):
            return tp_eval(th[3:], p_art_j)

    # ExoJAX power-law cloud [log10 kappac0 (cm^2/g at 3.5 um), alphac]: the
    # BASELINE deck the cloud Fisher rows differentiate around (None when off)
    cloud_vec = (jnp.asarray([cp["log_kappa_cloud"], cp["alpha_cloud"]])
                 if cp["cloud_on"] else None)

    def make_depth_fn(chem_b):
        """Depth function bound to ONE chemistry build: the interpolation map
        follows that build's hydrostatic grid, so it is NEVER shared across
        builds (composition moves the mean molecular weight and hence the
        pressure grid between the FD re-init builds).

        science_mode picks the observable behind the SAME signature:
        transmission -> transit depth (Rp(lambda)/Rstar)^2; emission ->
        eclipse depth (Fp/Fs) * (Rp/Rs)^2 * e^{2 lnR0}. Every downstream
        consumer is observable-agnostic through this function."""
        to_art_b = interp_map.make_to_art(chem_b.p_bar, rt.p_art_bar)
        mol_cols = {k: chem_b.sidx[config.MOLECULES[k]["vulcan"]]
                    for k in rt.molecules}
        h2_b, he_b = chem_b.sidx["H2"], chem_b.sidx["He"]
        # GAS-phase normalization: the network's *_l_s columns are particles,
        # not gas -- counting them dilutes every gas VMR and inflates the RT
        # mean molecular weight. Aerosol OPACITY stays excluded (cloud deck).
        _gas = np.ones(int(np.asarray(chem_b.species_masses).size))
        for _s, _i in chem_b.sidx.items():
            if _s.endswith("_l_s"):
                _gas[int(_i)] = 0.0
        gas_mask = jnp.asarray(_gas)

        def _art_profiles(y, th):
            y_gas = y * gas_mask[None, :]
            ymix = y_gas / jnp.sum(y_gas, axis=1, keepdims=True)
            T_art = art_T(th)
            mmw_art = to_art_b(ymix @ chem_b.species_masses)
            vmr = {k: to_art_b(ymix[:, c]) for k, c in mol_cols.items()}
            return (vmr, to_art_b(ymix[:, h2_b]), T_art, mmw_art,
                    to_art_b(ymix[:, he_b]))

        def depth_fn(y, th, lnR0=0.0, cloud=None, wo_mols=None):
            # cloud=None -> the baseline deck; an explicit vector overrides it
            # (the cloud Fisher rows differentiate through this argument).
            # wo_mols is transmission only: emission gets its leave-one-out
            # batch from emis.emission_flux_tau in run_model, whose tau-bottom
            # gate needs the same optical depth.
            vmr, vmr_h2, T_art, mmw_art, vmr_he = _art_profiles(y, th)
            cl = cloud_vec if cloud is None else cloud
            if emis is not None:
                if wo_mols is not None:
                    raise ValueError(
                        "depth_fn(wo_mols=...) is transmission-only; emission "
                        "removed-molecule spectra come from "
                        "emis.emission_flux_tau (run_model owns that call).")
                fp = emis.emission_flux(vmr, vmr_h2, T_art, mmw_art,
                                        vmr_he=vmr_he, cloud=cl)
                rp_em = emis.emission_radius(T_art, mmw_art)
                return (fp / fs_j) * (rp_em ** 2 / _rstar_cm_sq) * jnp.exp(
                    2.0 * jnp.asarray(lnR0))
            return rt.transmission_depth_r(
                vmr, vmr_h2, T_art, mmw_art,
                jnp.asarray(lnR0), vmr_he=vmr_he, cloud=cl,
                wo_mols=wo_mols)
        depth_fn._art_profiles = _art_profiles   # reused by the tau-bottom gate
        return depth_fn

    depth_from_y = make_depth_fn(chem)

    # --- chemistry: certified cold solves (no warm continuation) ------------
    t0 = time.time()
    th0 = jnp.asarray(theta)
    conv_cert = []   # (stage, accept_count, longdy) for every PASSED gate
    # Certification is the runner's own CANONICAL gate (ConvDiag.conv_normal AND
    # longdy < yconv_min), recomputed at the exit state. Never loosen it.

    def _check_converged(diag, stage):
        ac = int(diag.accept_count)
        longdy = float(diag.longdy)
        if not (bool(diag.conv_normal) and longdy < chem.yconv_min):
            how = (f"hit the count_max={chem.count_max} cap" if ac >= int(chem.count_max)
                   else f"exited at {ac} accepted steps without the runner's "
                        "canonical certification (stall fallback / hybrid "
                        "vm_mol phase-flip / photolysis flux still changing)")
            raise RuntimeError(
                f"chemistry did NOT converge ({stage}: longdy={longdy:.3g}, "
                f"gate yconv_min={chem.yconv_min:g}, "
                f"conv_normal={bool(diag.conv_normal)}; {how}). This "
                "parameter corner has no certified steady state -- adjust "
                "T-P / Kzz / composition (or the convergence settings) "
                "rather than trusting an unconverged spectrum.")
        conv_cert.append((stage, ac, longdy))

    # Single certified cold solve, unless an identical chemistry-relevant set
    # already solved: the chem-level cache stores the RAW column, so an RT-only
    # edit re-renders from the bits a fresh solve would produce.
    advance()
    _species_now = [s for s, _ in sorted(chem.sidx.items(),
                                         key=lambda kv: kv[1])]
    _chem_out = chem_cache_path(params)
    _chem_art = _load_cached_npz(_chem_out)
    if _chem_art is not None:
        # validate loudly: a mismatched entry raises, never silently re-solves
        if [str(s) for s in _chem_art["species"]] != _species_now:
            raise RuntimeError(
                f"chem-cache {_chem_out.name}: species set does not match "
                "the built network -- the cache key failed to separate two "
                "different chemistry configurations. Delete the entry and "
                "report this; do not ignore it.")
        if not np.array_equal(np.asarray(_chem_art["theta"]), theta):
            raise RuntimeError(
                f"chem-cache {_chem_out.name}: stored theta differs from "
                "this run's theta under an equal chem_key -- the key is "
                "missing a chemistry-relevant parameter. Delete the entry "
                "and report this; do not ignore it.")
        y_np = np.asarray(_chem_art["y_raw"], dtype=np.float64)
        if not np.all(np.isfinite(y_np)):
            raise RuntimeError(
                f"chem-cache {_chem_out.name}: non-finite stored column")
        _ac = int(_chem_art["conv_accept"][0])
        _ld = float(_chem_art["conv_longdy"][0])
        if not _ld < float(chem.yconv_min):
            raise RuntimeError(
                f"chem-cache {_chem_out.name}: stored certificate "
                f"(longdy={_ld:.3g}) fails the current gate "
                f"yconv_min={float(chem.yconv_min):g}")
        conv_cert.append(("baseline solve (chem-cache)", _ac, _ld))
        y_sol = jnp.asarray(y_np)
        log(f"[fwd] chemistry column from chem-cache ({_chem_out.name}, "
            f"{_ac} accepted steps at write); solve skipped")
    else:
        log("[fwd] solving photochemistry (cold, certified) ...")
        y_sol, _cdiag = chem.converged_y(th0, return_conv_diag=True)
        _check_converged(_cdiag, "baseline solve")
        y_np = np.asarray(y_sol)
        if not np.all(np.isfinite(y_np)):
            raise RuntimeError(
                "chemistry solve returned non-finite abundances -- "
                "parameter set outside the modelable range")
        log(f"[fwd] chemistry solved in {time.time()-t0:.0f} s total")
        # persist the certified RAW column (atomic, as for the flat cache)
        _stage, _ac, _ld = conv_cert[-1]
        _chem_arrays = dict(
            y_raw=y_np,
            species=np.array(_species_now, dtype="U16"),
            theta=theta,
            theta_names=np.array(theta_names, dtype="U16"),
            p_bar=np.asarray(chem.p_bar),
            conv_accept=np.array([_ac], dtype=np.int64),
            conv_longdy=np.array([_ld], dtype=np.float64),
        )
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        _ins.atomic_write(
            _chem_out, lambda fh: np.savez_compressed(fh, **_chem_arrays))
        log(f"[fwd] chem column cached -> {_chem_out.name}")

    # Emission bottom-boundary certification. The interior source term is a
    # blackbody at the extrapolated bottom temperature, an assumption about
    # everything below the grid: the fix for a thin bottom is a deeper column
    # (p_btm_bar), never a wider tolerance.
    emis_tau_min = float("nan")
    _depth_norm_em0 = float("nan")
    wo_list = list(cp["wo_mols"])
    depth_wo = None
    emis_tau_min_wo = np.full(len(wo_list), np.nan)
    emis_thin_wo = np.full(len(wo_list), np.nan)
    if emis is not None:
        t0 = time.time()
        advance()          # full (+ removed-molecule) eclipse spectra
        _prof0 = depth_from_y._art_profiles(y_sol, th0)
        # the SAME (R_p/R_star)^2 the baseline depth was built with, so the
        # stored Fp inverts the stored depth exactly
        _depth_norm_em0 = float(
            np.asarray(emis.emission_radius(_prof0[2], _prof0[3])) ** 2
            / _rstar_cm_sq)
        log(f"[fwd] emission anchor: R_p = "
            f"{_depth_norm_em0 ** 0.5 * cp['rstar_rsun'] * planets.R_SUN_CM / planets.R_JUP_CM:.4f} "
            f"R_Jup at {getattr(emis, 'p_ref_emission_bar', float('nan')):g} "
            f"bar (catalogue transit radius {cp['rp_rjup']:.4f} R_Jup at "
            f"{cp['p_ref_bar']:g} bar)")
        # ONE optical-depth build feeds the flux, the tau-bottom gate, the
        # stored depth AND the removed-molecule spectra; the batch's full total
        # is bitwise the standalone call.
        if wo_list:
            _fp_j, _tau_j, _fp_wo_j, _tau_wo_j = emis.emission_flux_tau(
                *_prof0, cloud=cloud_vec, wo_mols=wo_list)
        else:
            _fp_j, _tau_j = emis.emission_flux_tau(*_prof0, cloud=cloud_vec)
        _tau_b = np.asarray(_tau_j)
        emis_tau_min = float(_tau_b.min())
        _wl_thin = float(rt.wl_um[int(np.argmin(_tau_b))])
        # FLUX-WEIGHTED, not min() over the band: the column being transparent
        # somewhere only matters to the extent the planet emits there.
        _fp = np.asarray(_fp_j)
        emis_thin_frac = thin_flux_fraction(_tau_b, _fp)
        _report = _tau_bottom_breakdown(rt.wl_um, _tau_b, flux=_fp)
        if emis_thin_frac > EMIS_THIN_FLUX_FRAC:
            raise RuntimeError(
                f"emission unreliable: {100.0 * emis_thin_frac:.1f}% of this "
                f"planet's emitted flux comes from wavelengths where the RT "
                f"column bottom ({emis.art_pbtm_bar:g} bar) is optically thin "
                f"(tau < {EMIS_TAU_THIN:g}), above the "
                f"{100.0 * EMIS_THIN_FLUX_FRAC:g}% tolerance -- that flux is "
                "set by the interior source term, an assumption about "
                "everything below the column.\n\n"
                + _report +
                "\n\nThin only at the SHORT-wavelength edge means missing "
                "opacity, not a shallow column: below ~2 um the continuum "
                "that fills the window in a real hot atmosphere (H- bound-free "
                "and free-free, the Na and K wings, TiO/VO) is not modeled "
                "here, and deepening p_btm_bar would hide that rather than fix "
                "it. Thin across the band means the column really is too "
                f"shallow -- raise p_btm_bar (now {cp['p_btm_bar']:g} bar).")
        log("[fwd] " + _report.replace("\n", "\n[fwd] "))
        if emis_thin_frac > 0.0:
            log(f"[fwd] NOTE: {100.0 * emis_thin_frac:.3f}% of the emitted "
                f"flux comes from below tau = {EMIS_TAU_THIN:g} (min "
                f"{emis_tau_min:.2f} at {_wl_thin:.2f} um). Under the "
                f"{100.0 * EMIS_THIN_FLUX_FRAC:g}% tolerance, so the run "
                "proceeds; treat those wavelengths as unmodeled.")

        # the stored depth via the exact expression depth_fn uses (bitwise)
        _rp_em = emis.emission_radius(_prof0[2], _prof0[3])
        depth = np.asarray((_fp_j / fs_j) * (_rp_em ** 2 / _rstar_cm_sq)
                           * jnp.exp(2.0 * jnp.asarray(0.0)))
        if not wo_list:
            log(f"[fwd] full spectrum in {time.time()-t0:.0f} s")

        depth_wo = np.zeros((len(wo_list), depth.shape[0]))
        if wo_list:
            # Each removed-molecule optical depth (from the batch above) feeds
            # its spectrum AND the tau-bottom gate, which must cover EVERY
            # emission spectrum the results consume: zeroing a dominant
            # absorber can open see-through windows that inflate the contrast.
            _fp_wo, _tau_wo = _fp_wo_j, _tau_wo_j
            for i, mol in enumerate(wo_list):
                depth_wo[i] = np.asarray(
                    (_fp_wo[i] / fs_j) * (_rp_em ** 2 / _rstar_cm_sq)
                    * jnp.exp(2.0 * jnp.asarray(0.0)))
                _tau_i = np.asarray(_tau_wo[i])
                emis_tau_min_wo[i] = float(_tau_i.min())
                _wl_i = float(rt.wl_um[int(np.argmin(_tau_i))])
                # flux-weighted, same reasoning as the baseline gate above
                emis_thin_wo[i] = thin_flux_fraction(_tau_i,
                                                     np.asarray(_fp_wo[i]))
                if emis_thin_wo[i] > EMIS_THIN_FLUX_FRAC:
                    # Per-molecule, never whole-run: only THIS molecule's
                    # detection is unreliable, and detect refuses that target.
                    log(f"[fwd] NOTE: emission detection of {mol} is "
                        f"UNRELIABLE -- with it removed, "
                        f"{100.0 * emis_thin_wo[i]:.1f}% of the emitted flux "
                        f"comes from below tau = {EMIS_TAU_THIN:g} (min "
                        f"{emis_tau_min_wo[i]:.2f} at {_wl_i:.2f} um); detect "
                        "refuses this target. The spectrum and the other "
                        "molecules are unaffected.")
                elif emis_tau_min_wo[i] < 10.0:
                    log(f"[fwd] WARNING: min bottom optical depth "
                        f"{emis_tau_min_wo[i]:.1f} at {_wl_i:.2f} um (< 10) "
                        f"with {mol} removed: its detection contrast leans on "
                        "the deepest layers -- treat with care.")
            log(f"[fwd] full + {len(wo_list)} removed-molecule spectra in "
                f"{time.time()-t0:.0f} s")
    else:
        # --- RT: transmission full spectrum (+ leave-one-out batch) ----------
        t0 = time.time()
        advance()
        if wo_list:
            log("[fwd] radiative transfer: full + removed-molecule spectra ...")
            _d_j, _d_wo_j = depth_from_y(y_sol, th0, wo_mols=wo_list)
            depth = np.asarray(_d_j)
            depth_wo = np.asarray(_d_wo_j)
            log(f"[fwd] full + {len(wo_list)} removed-molecule spectra in "
                f"{time.time()-t0:.0f} s")
        else:
            log("[fwd] radiative transfer: full spectrum ...")
            depth = np.asarray(depth_from_y(y_sol, th0))
            depth_wo = np.zeros((0, depth.shape[0]))
            log(f"[fwd] full spectrum in {time.time()-t0:.0f} s")

    # Fisher Jacobian: certified FD (default) / warm-jvp AD (opt-in); per-row
    # method recorded in jac_row_method.
    jac_names = list(cp["fisher_params"])
    jac = np.zeros((len(jac_names) + 1, depth.shape[0])) if jac_names else None
    fd_h, fd_err, row_method = [], [], []
    if jac_names:
        def _certified_depth(chem_b, th, stage):
            y_b, diag_b = chem_b.converged_y(jnp.asarray(th),
                                             return_conv_diag=True)
            _check_converged(diag_b, stage)
            return np.asarray(make_depth_fn(chem_b)(y_b, jnp.asarray(th)))

        def _fd_row(name, j1, j2, h):
            # j1 / j2: the same scheme's estimate at step h and 2h
            if not (np.isfinite(j1).all() and np.isfinite(j2).all()):
                raise RuntimeError(
                    f"FD Jacobian for {name}: non-finite entries")
            scale = float(np.max(np.abs(j1)))
            if scale == 0.0:
                return j1, 0.0     # no spectral response: exact zero row
            err = float(np.max(np.abs(j1 - j2)) / scale)
            if err > FD_CONSISTENCY_TOL:
                raise RuntimeError(
                    f"FD Jacobian for {name} FAILED the step-size consistency "
                    f"check: max|J(h) - J(2h)| / max|J(h)| = {err:.3f} > "
                    f"{FD_CONSISTENCY_TOL} (h = {h:g}). The row is dominated "
                    "by solver convergence noise or curvature -- tighten "
                    "yconv_cri (1e-3 or 1e-4), raise nz, or adjust "
                    "forward.FD_STEPS. An uncertified derivative is never "
                    "reported.")
            return (4.0 * j1 - j2) / 3.0, err   # Richardson: O(h^4) central,
            #                                      O(h^3) one-sided

        def _ad_theta_depth_diag(th):
            # warm continuation from the converged column: the primal is a
            # warm re-converge plus the full spectrum, the jvp the validated
            # steady-state tangent (photo ON, gated in canonical_params). The
            # convergence certificate is a second output, so the batch
            # certifies the point it differentiates from that same solve.
            y_w, diag = chem.converged_y(th, warm_y=y_sol, lnZ_ref=0.0,
                                         c_o_ref=0.0, return_conv_diag=True)
            return depth_from_y(y_w, th), diag

        _ad_chem_rows = ([n for n in jac_names
                          if n not in CLOUD_FISHER_PARAMS]
                         if cp["jac_method"] == "ad" else [])
        _ad_cols = {}
        if _ad_chem_rows:
            # One plain jvp per chemistry-theta row, NEVER vmap over the
            # tangent directions: the batched tangent through the solver's
            # while_loop is NaN in every bin on a column with clamped-zero
            # layers (TOI-7169 b, 10x solar, C/O 0.55; even a batch of one),
            # while the unbatched jvp is finite. Each row certifies its own
            # warm re-converge.
            t1 = time.time()
            advance()
            for _n in _ad_chem_rows:
                _e = np.zeros(theta.size)
                _e[theta_names.index(_n)] = 1.0
                (_pd, _pdiag), (_dd, _) = jax.jvp(
                    _ad_theta_depth_diag, (th0,), (jnp.asarray(_e),))
                _check_converged(_pdiag, f"AD warm re-converge ({_n})")
                _ad_cols[_n] = np.asarray(_dd)
            log(f"[fwd] AD Jacobian: {len(_ad_chem_rows)} rows, one warm jvp "
                f"each, in {time.time()-t1:.0f} s")

        def _rt_deck_row(name, base_vec, idx, kwarg):
            """RT-only Jacobian row for a cloud-deck parameter (no chemistry
            re-solve). AD -> one jvp along `idx`. FD -> a SINGLE central
            difference, since the analytic power-law deck carries no solver
            noise; its O(h^2) truncation error is then unmeasured, so fd_err
            is NaN, never 0. Returns (row, h, err, method)."""
            base_vec = np.asarray(base_vec, dtype=np.float64)
            if cp["jac_method"] == "ad":
                e = np.zeros(base_vec.size)
                e[idx] = 1.0
                _, dd = jax.jvp(
                    lambda v: depth_from_y(y_sol, th0, **{kwarg: v}),
                    (jnp.asarray(base_vec),), (jnp.asarray(e),))
                return np.asarray(dd), 0.0, np.nan, "ad-jvp"
            h = FD_STEPS[name]

            def _d(step):
                v = base_vec.copy()
                v[idx] += step
                return np.asarray(depth_from_y(y_sol, th0,
                                               **{kwarg: jnp.asarray(v)}))
            j1 = (_d(h) - _d(-h)) / (2.0 * h)
            return j1, h, np.nan, "fd-rt"   # truncation unmeasured, not 0

        for j, name in enumerate(jac_names):
            t1 = time.time()
            if name not in _ad_cols:
                advance()        # batched AD rows advanced once, above
            if name in CLOUD_FISHER_PARAMS:
                # RT-only deck row: the power-law deck is smooth, so ungated
                jac[j], _h, _err, _m = _rt_deck_row(
                    name, [cp["log_kappa_cloud"], cp["alpha_cloud"]],
                    CLOUD_FISHER_PARAMS.index(name), "cloud")
                fd_h.append(_h)
                fd_err.append(_err)
                row_method.append(_m)
                if not np.isfinite(jac[j]).all():
                    raise RuntimeError(
                        f"cloud Jacobian for {name}: non-finite entries")
                log(f"[fwd] {cp['jac_method'].upper()} Jacobian "
                    f"d(depth)/d({name}) [RT-only cloud row] in "
                    f"{time.time()-t1:.0f} s")
                continue
            if cp["jac_method"] == "ad":
                # AD row: warm-started jvp along this theta direction; lnZ is
                # the fixed-structural-grid derivative and is not cross-checked
                # against FD. Computed row by row above.
                jac[j] = _ad_cols[name]
                if not np.isfinite(jac[j]).all():
                    raise RuntimeError(
                        f"AD Jacobian for {name}: non-finite entries")
                fd_h.append(0.0)          # no FD step: AD row
                fd_err.append(np.nan)     # no h-vs-2h metric: AD row
                row_method.append("ad-jvp")
                log(f"[fwd] AD Jacobian d(depth)/d({name}) from its warm jvp")
                continue
            offs, h = (1, -1, 2, -2), FD_STEPS[name]
            dvals = {}
            if name in FD_COMP_PARAMS:
                # composition direction: FastChem re-init + certified cold
                # solve per stencil point (central, or one-sided away from
                # C/O = 1 -- see fd_stencil)
                offs, h = fd_stencil(name, cp["met_x_solar" if name == "lnZ"
                                              else "co_ratio"])
                for s in offs:
                    f = float(np.exp(s * h))
                    if name == "lnZ":      # all metals together; C/O preserved
                        ab = _abundance_overrides(cp["met_x_solar"] * f,
                                                  cp["co_ratio"])
                    else:                  # dlnCO: carbon at fixed oxygen
                        ab = _abundance_overrides(cp["met_x_solar"],
                                                  cp["co_ratio"] * f)
                    chem_s = _build_chem(ab, tag=f"FD {name} {s:+d}h")
                    dvals[s] = _certified_depth(chem_s, theta,
                                                f"FD {name} {s:+d}h")
            else:
                # theta direction (lnKzz / T-P): baseline build, certified
                # points at theta +- h, +- 2h
                i_par = theta_names.index(name)
                for s in (1, -1, 2, -2):
                    th_s = theta.copy()
                    th_s[i_par] += s * h
                    # T-P step must stay in the window (tp_eval is None only
                    # in file mode, which has no theta T-P rows)
                    if i_par >= 3 and tp_eval is not None:
                        T_s = np.asarray(tp_eval(jnp.asarray(th_s[3:]),
                                                 jnp.asarray(chem.p_bar)))
                        if T_s.min() < T_WINDOW[0] or T_s.max() > T_WINDOW[1]:
                            raise RuntimeError(
                                f"FD step for {name} ({s:+d}h = {s * h:+g}) "
                                f"leaves the modelable T window {T_WINDOW}: "
                                "move the profile away from the window edge "
                                "or reduce forward.FD_STEPS for it.")
                    dvals[s] = _certified_depth(chem, th_s,
                                                f"FD {name} {s:+d}h")
            method = "fd-central" if len(offs) == 4 else "fd-onesided"
            f0 = (None if method == "fd-central"
                  else np.asarray(depth_from_y(y_sol, th0)))
            jac[j], err = _fd_row(name, *fd_estimates(offs, dvals, f0, h), h)
            fd_h.append(h)
            fd_err.append(err)
            row_method.append(method)
            log(f"[fwd] FD Jacobian d(depth)/d({name}) [{method}] in "
                f"{time.time()-t1:.0f} s (h-vs-2h consistency {err:.3f} < "
                f"{FD_CONSISTENCY_TOL})")

        t1 = time.time()
        advance()
        # lnR0 is RT-only (smooth, analytic in lnR0). "fd": one central
        # difference through the radiative transfer, no chemistry and no gate
        # needed; "ad": the RT jvp.
        if cp["jac_method"] == "ad":
            _, dd = jax.jvp(lambda r: depth_from_y(y_sol, th0, lnR0=r),
                            (jnp.asarray(0.0),), (jnp.asarray(1.0),))
            jac[-1] = np.asarray(dd)
            fd_h.append(0.0)
            fd_err.append(np.nan)
            row_method.append("ad-jvp")
        else:
            d_rp = np.asarray(depth_from_y(y_sol, th0, lnR0=+FD_LNR0_STEP))
            d_rm = np.asarray(depth_from_y(y_sol, th0, lnR0=-FD_LNR0_STEP))
            jac[-1] = (d_rp - d_rm) / (2.0 * FD_LNR0_STEP)
            fd_h.append(FD_LNR0_STEP)
            fd_err.append(np.nan)   # single central diff: truncation unmeasured
            row_method.append("fd-rt")
        jac_names.append("lnR0")
        log(f"[fwd] {cp['jac_method'].upper()} Jacobian d(depth)/d(lnR0) "
            f"[RT-only nuisance] in {time.time()-t1:.0f} s")

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    out = cache_path(params)
    # npz ymix must use the SAME gas normalization the RT applies (*_l_s
    # columns excluded), or the saved ymix disagrees with the spectra.
    _gas_np = np.ones(y_np.shape[1])
    for _s, _i in chem.sidx.items():
        if _s.endswith("_l_s"):
            _gas_np[int(_i)] = 0.0
    _y_gas_np = y_np * _gas_np[None, :]
    ymix_np = _y_gas_np / _y_gas_np.sum(axis=1, keepdims=True)
    arrays = dict(
        wl_um=np.asarray(rt.wl_um, dtype=np.float64),
        depth=depth, depth_wo=depth_wo,
        mols=np.array(mols_active, dtype="U8"),
        # the set depth_wo (and the emis_*_wo certificates) align with: a
        # subset of mols in fold order. Readers index by THIS array, not mols.
        wo_mols=np.array(wo_list, dtype="U8"),
        ymix=ymix_np, p_bar=np.asarray(chem.p_bar),
        # THE COLUMN NAMES OF ymix, which is the FULL network state while
        # `mols` is the shorter RT list. Never infer these from `mols`.
        ymix_species=np.array(
            [s for s, _ in sorted(chem.sidx.items(), key=lambda kv: kv[1])],
            dtype="U16"),
        T=np.asarray(T_check), theta=theta,
        theta_names=np.array(theta_names, dtype="U16"),
        # auto-sized U dtype: a fixed width silently truncated long JSON
        params_json=np.array(json.dumps(cp)),
        # convergence certificate: the runner's own longdy per gated stage
        conv_stages=np.array([s for s, _, _ in conv_cert], dtype="U48"),
        conv_accept=np.array([a for _, a, _ in conv_cert], dtype=np.int64),
        conv_longdy=np.array([l for _, _, l in conv_cert], dtype=np.float64),
        conv_gate=np.array([float(getattr(chem, "yconv_min", np.nan))],
                           dtype=np.float64),
        science_mode=np.array(cp["science_mode"], dtype="U16"),
        chem_provider=np.array(cp["chem_provider"], dtype="U16"),
    )
    if emis is not None:
        arrays["fs_flux"] = np.asarray(fs_j, dtype=np.float64)
        # Fp derived exactly from the stored eclipse depth (lnR0 = 0 baseline)
        arrays["fp_flux"] = depth * np.asarray(fs_j) / _depth_norm_em0
        arrays["emis_tau_bottom_min"] = np.array([emis_tau_min])
        # per-removed-molecule bottom-tau certificate (aligned with wo_mols)
        arrays["emis_tau_bottom_min_wo"] = emis_tau_min_wo
        # The FLUX-WEIGHTED certificate is what the gates use; the min-tau
        # arrays above are provenance only.
        arrays["emis_thin_flux_frac"] = np.array([emis_thin_frac])
        arrays["emis_thin_flux_frac_wo"] = emis_thin_wo
    if jac is not None:
        arrays["jac"] = jac
        arrays["jac_names"] = np.array(jac_names, dtype="U16")
        # Per-row provenance. fd_err NaN = unmeasured (AD rows, ungated
        # single-difference RT rows), never 0; 0.0 means a zero-response row.
        arrays["jac_row_method"] = np.array(row_method, dtype="U16")
        arrays["fd_h"] = np.array(fd_h, dtype=np.float64)
        arrays["fd_err"] = np.array(fd_err, dtype=np.float64)
    # atomic: two same-key runs can write at once and a direct savez would let
    # a reader see a partial zip, or poison the key on a kill mid-write
    _ins.atomic_write(out, lambda fh: np.savez_compressed(fh, **arrays))
    finish()
    log(f"[fwd] cached -> {out.name}")
    return out


def main():
    from jwst_tool import proc
    params = json.load(open(sys.argv[1]))
    proc.worker_prologue(_ins.OUTPUT_DIR)
    run_model(params, log=lambda *a: print(*a, flush=True))
    print("[fwd] DONE", flush=True)


if __name__ == "__main__":
    main()
