"""PICASO equilibrium-chemistry provider: composition-blended Visscher grid.

Pure numpy + stdlib at import time (the fast test suite exercises the math on
synthetic fixtures without picaso installed); the reference tree is touched
only inside the loaders, resolved through :mod:`jwst_tool.picaso_env`.

Scientific contract (2026-07-20 review, rev 3):

* ``chemeq_visscher_2121`` picks the NEAREST node file in ([M/H], C/O) with no
  composition interpolation -- naive FD composition rows through the stock API
  would be exactly zero inside a cell. This module instead blends the 2x2
  bracketing node files bilinearly in (feh = log10 met, C/O) of the LOG10
  abundances, then interpolates (T, P) within the blended table using exactly
  picaso's own ``chem_interp`` convention (bilinear in 1/T and log10 P of the
  log10 abundance; verified to 4e-15 dex against the native implementation).
  Unlike picaso we REFUSE outside the table instead of extrapolating.
* Composition derivatives are therefore SYMMETRIC TWO-CELL INTERPOLANT
  SECANTS: the default baselines sit ON grid nodes, where the interpolant is
  continuous with a kinked derivative, so a central difference spans the two
  adjacent cells and no unique local derivative exists. The forward model
  computes left/right one-sided secants and HARD-ERRORS when their
  disagreement exceeds ``FD_KINK_TOL`` (never report a Fisher row whose
  one-sided slopes differ materially).
* Gas accounting: the gas total includes EVERY gas-phase column -- ions and
  electrons too (they contribute number density; the electron mass is
  5.49e-4 amu) -- and excludes only the true condensate column (graphite),
  which is renamed ``C-gr_l_s`` so the shared RT path's condensate mask
  handles it exactly like VULCAN's ``*_l_s`` reservoirs. The tables do NOT
  sum to 1 everywhere (documented upstream; species outside the reported set
  are missing, and low-T cells floor uncalculated species at 1e-50): the
  shared path renormalizes the retained gas per layer, and the provider
  certificate records the per-layer pre-normalization sum -- refusing below
  ``GAS_SUM_MIN``, flagging below ``GAS_SUM_WARN``.
* Known data defect (measured 2026-07-20/21): feh1.0_co0.55 carries ONE
  corrupted row at (900 K, logP = -5.523). Anatomy: EVERY species in the
  row is uniformly deflated by ~x0.747 (a spurious ~25% phantom abundance
  entered the row's normalization at generation), plus two junk residues:
  VO ~9.9e6x and CrH ~4.8e4x too high (both spectroscopically inert at
  ~1e-12 and not RT species). The same cell is clean in all four
  neighboring node files. Handling (v18.1 review decision): a VERSIONED,
  CONTENT-GUARDED correction -- ``KNOWN_TABLE_CORRECTIONS`` replaces the
  row by the log-mean of its T-neighbors ONLY while the file's row still
  hashes to the registered corrupt bytes (an upstream fix makes the entry
  a no-op); every application is recorded in the certificate/npz. The
  measured spectral difference between the correction and the previous
  renormalize-through treatment is <= 2.2 ppm worst-case (900 K profile).
  Any OTHER isolated anomaly (a row below ``SUSPECT_SUM`` whose T-neighbor
  rows are clean) inside the evaluated span REFUSES loudly -- unvetted
  corruption is never renormalized through. The extreme-metallicity files
  (|feh| >= 1.5) systematically sum to 0.86-0.98 at T <~ 500 K with
  equally-low neighbor rows -- expected missing-species behavior, NOT
  isolated corruption: renormalized and certificate-flagged, never
  refused.
* Realized composition: gas-phase C/O from the blended abundances matches the
  file label only where nothing has condensed (verified: 0.458 vs label 0.46
  at 2000 K; at 800-1200 K silicate condensation sequesters O and the
  gas-phase C/O rises to ~0.55 -- real physics, not an interpolation error).
  The certificate records the per-layer realized gas C/O for exactly this
  reason, plus a hot-layer summary (``realized_gas_co_hotT``: the median
  over layers with T >= ``CO_CHECK_T_K``, where condensation cannot shift
  it) that the GUI surfaces for comparison against the label. No code path
  refuses on a label mismatch -- it is a reported diagnostic, not a gate.
"""
from __future__ import annotations

import functools
from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# Grid geometry + provider constants
# ---------------------------------------------------------------------------

FEH_NODES = (-2.0, -1.5, -1.0, -0.7, -0.5, -0.3, 0.0,
             0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
CO_NODES = (0.14, 0.27, 0.46, 0.55, 0.82, 1.10)
#: extreme-metallicity nodes ship only the mid C/O columns (both the chemistry
#: files and the preweighted CK tables)
_EXTREME_FEH = (-2.0, -1.5, 1.5, 2.0)
_EXTREME_CO = (0.27, 0.46, 0.55, 0.82)
CK_NODES_AVAILABLE = tuple(
    "feh%.1f_co%.2f" % (f, c)
    for f in FEH_NODES for c in CO_NODES
    if not (f in _EXTREME_FEH and c not in _EXTREME_CO))
assert len(CK_NODES_AVAILABLE) == 70

#: RT molecules the provider supplies (NO SO2/S2/S8: equilibrium sulfur sits
#: in H2S/OCS -- the W39b photochemical-sulfur science stays VULCAN-only).
#: H2S is BASE here (v20): it is the dominant equilibrium sulfur reservoir at
#: 700-1500 K (and in picaso's own default species set), and leaving it
#: opt-in made the default eq-vs-kinetics comparison asymmetric on sulfur
#: (the vulcan base carries SO2 while the picaso base carried no S at all).
PICASO_MOLECULES = ["H2O", "CO2", "CO", "CH4", "H2S"]
PICASO_EXTRA_MOLECULES = ["C2H2", "HCN", "NH3", "OCS"]

#: registry vulcan-token -> table column, for species where the SNCHO
#: network and the Visscher tables disagree on the name; the picaso chem
#: adapter aliases sidx so the shared depth path's registry-token lookup
#: works under both engines
VULCAN_TO_TABLE = {"COS": "OCS"}

#: table span (validated by the loader) and provider pressure policy: the
#: tables start at 1e-6 bar, so the provider chemistry grid spans exactly
#: [1e-6 bar, chemistry bottom]; ABOVE 1e-6 bar the RT interpolation map
#: constant-extends the top layer (stated policy, certificate-recorded).
TABLE_T_K = (75.0, 6000.0)
TABLE_P_LOGBAR = (-6.0, 4.0)
N_T, N_P = 101, 21

GAS_SUM_MIN = 0.70     # refuse: pre-normalization gas sum below this anywhere
GAS_SUM_WARN = 0.98    # certificate flag threshold
SUSPECT_SUM = 0.90     # loader-level suspect-row threshold (catches the
                       # feh1.0_co0.55 glitch cell at 0.746)
FLOOR_LOG10 = -45.0    # post-blend values at/below this are exact zeros
                       # (the files floor uncalculated species at 1e-50)
FD_KINK_TOL = 0.5      # max |j_right - j_left| / max|j_sym| before a
                       # composition Fisher row hard-errors (node kink gate)
CO_CHECK_T_K = 2000.0  # realized-C/O vs label comparisons only above this
                       # (below it, condensate O-sequestration shifts gas C/O)

GRAPHITE = "C-gr"
GRAPHITE_OUT = "C-gr_l_s"   # renamed so the shared condensate mask catches it

#: Versioned, content-guarded corrections for CATALOGUED table defects
#: (module docstring has the vetting evidence). Keyed by node token; each
#: entry names the cell, the sha1-16 of the exact corrupt ROW bytes (the
#: np.loadtxt float64 row including the T/logP columns -- if upstream fixes
#: the file the hash no longer matches and the correction is a NO-OP), and
#: the correction method. Never a general repair heuristic.
KNOWN_TABLE_CORRECTIONS = {
    "feh1.0_co0.55": (
        {"T": 900.0, "logP": -5.523,
         "corrupt_row_sha1": "29c1396fb5d35401",
         "method": "T-neighbor log-mean",
         "why": "uniform x0.747 row deflation + junk VO/CrH residues "
                "(2026-07-20 anatomy; <= 2.2 ppm spectral impact bound)"},
    ),
}

# ---------------------------------------------------------------------------
# Species masses + stoichiometry (one source for mmw AND realized composition)
# ---------------------------------------------------------------------------

_ELEMENT_MASS = {  # standard atomic weights, amu
    "H": 1.008, "He": 4.002602, "C": 12.011, "N": 14.007, "O": 15.999,
    "Na": 22.98976928, "Mg": 24.305, "Si": 28.085, "P": 30.973761998,
    "S": 32.06, "K": 39.0983, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
    "Fe": 55.845, "Li": 6.94, "Rb": 85.4678, "Cs": 132.90545196,
    "F": 18.998403163, "Cl": 35.45,
}
_M_E = 5.48579909e-4     # electron mass, amu

#: element counts per table column (charge ignored for mass -- an ion's mass
#: differs from its parent by one electron mass, far below the 5-digit
#: standard-atomic-weight precision used here; e- carries its own mass)
SPECIES_ELEMENTS = {
    "e-": {}, "H2": {"H": 2}, "H": {"H": 1}, "H+": {"H": 1}, "H-": {"H": 1},
    "H2-": {"H": 2}, "H2+": {"H": 2}, "H3+": {"H": 3}, "He": {"He": 1},
    "H2O": {"H": 2, "O": 1}, "CH4": {"C": 1, "H": 4}, "CO": {"C": 1, "O": 1},
    "NH3": {"N": 1, "H": 3}, "N2": {"N": 2}, "PH3": {"P": 1, "H": 3},
    "H2S": {"H": 2, "S": 1}, "TiO": {"Ti": 1, "O": 1}, "VO": {"V": 1, "O": 1},
    "Fe": {"Fe": 1}, "FeH": {"Fe": 1, "H": 1}, "CrH": {"Cr": 1, "H": 1},
    "Na": {"Na": 1}, "K": {"K": 1}, "Rb": {"Rb": 1}, "Cs": {"Cs": 1},
    "CO2": {"C": 1, "O": 2}, "HCN": {"H": 1, "C": 1, "N": 1},
    "C2H2": {"C": 2, "H": 2}, "C2H4": {"C": 2, "H": 4},
    "C2H6": {"C": 2, "H": 6}, "SiO": {"Si": 1, "O": 1},
    "MgH": {"Mg": 1, "H": 1}, "OCS": {"O": 1, "C": 1, "S": 1},
    "Li": {"Li": 1}, "LiOH": {"Li": 1, "O": 1, "H": 1},
    "LiH": {"Li": 1, "H": 1}, "LiCl": {"Li": 1, "Cl": 1},
    "OH": {"O": 1, "H": 1}, GRAPHITE: {"C": 1}, "Li+": {"Li": 1},
    "LiF": {"Li": 1, "F": 1}, "C": {"C": 1}, "O": {"O": 1}, "Mg": {"Mg": 1},
    "Mg+": {"Mg": 1}, "Si": {"Si": 1}, "Fe+": {"Fe": 1}, "Ti": {"Ti": 1},
    "Ti+": {"Ti": 1}, "C+": {"C": 1},
}


def species_mass(name: str) -> float:
    if name == "e-":
        return _M_E
    if name in (GRAPHITE, GRAPHITE_OUT):
        name = GRAPHITE
    counts = SPECIES_ELEMENTS[name]
    return float(sum(_ELEMENT_MASS[el] * n for el, n in counts.items()))


def _atom_counts(species: list[str], element: str) -> np.ndarray:
    key = [GRAPHITE if s == GRAPHITE_OUT else s for s in species]
    return np.array([SPECIES_ELEMENTS[s].get(element, 0) for s in key],
                    dtype=float)


# ---------------------------------------------------------------------------
# Exact composition transforms (the lnZ / dlnCO step definitions)
# ---------------------------------------------------------------------------

def comp_step(met_x_solar: float, co_ratio: float, param: str,
              s: float) -> tuple[float, float]:
    """(met', co') for a signed log step ``s``: lnZ means Z*exp(s) (all
    metals together, C/O preserved); dlnCO means (C/O)*exp(s) at fixed Z."""
    if param == "lnZ":
        return float(met_x_solar) * float(np.exp(s)), float(co_ratio)
    if param == "dlnCO":
        return float(met_x_solar), float(co_ratio) * float(np.exp(s))
    raise ValueError(f"unknown composition parameter {param!r}")


def kink_metric(j_left: np.ndarray, j_right: np.ndarray,
                j_sym: np.ndarray) -> float:
    """Normalized one-sided-secant disagreement for the node-kink gate."""
    scale = float(np.max(np.abs(j_sym)))
    if scale == 0.0:
        return 0.0 if np.allclose(j_left, j_right) else np.inf
    return float(np.max(np.abs(np.asarray(j_right) - np.asarray(j_left)))
                 / scale)


# ---------------------------------------------------------------------------
# Node tables
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def load_node_table(node: str) -> SimpleNamespace:
    """One Visscher node file as a validated (nT, nP, n_sp) log10 cube.

    Refuses (never repairs) shape/monotonicity violations; records rows whose
    gas sum falls below ``SUSPECT_SUM`` (see the module docstring's known
    single-cell defect).
    """
    from jwst_tool import picaso_env as pe

    path = pe.chem_node_path(node)
    with open(path) as fh:
        header = fh.readline().split()
    if header[:2] != ["T(K)", "P(bar)"]:
        raise RuntimeError(f"unexpected header in {path}: {header[:2]}")
    species = header[2:]
    raw = np.loadtxt(path, skiprows=1)
    if raw.shape != (N_T * N_P, 2 + len(species)):
        raise RuntimeError(
            f"{path}: shape {raw.shape}, expected ({N_T * N_P}, "
            f"{2 + len(species)}) -- refusing a non-rectangular or "
            "truncated node file.")
    T_col, P_col, ab = raw[:, 0], raw[:, 1], raw[:, 2:]
    T = np.unique(T_col)
    Plog = np.unique(P_col)
    if len(T) != N_T or len(Plog) != N_P:
        raise RuntimeError(
            f"{path}: {len(T)} temperatures x {len(Plog)} pressures, "
            f"expected {N_T} x {N_P}.")
    if not (np.isclose(T[0], TABLE_T_K[0]) and np.isclose(T[-1], TABLE_T_K[1])
            and np.isclose(Plog[0], TABLE_P_LOGBAR[0])
            and np.isclose(Plog[-1], TABLE_P_LOGBAR[1])):
        raise RuntimeError(
            f"{path}: table span T [{T[0]}, {T[-1]}] K / logP "
            f"[{Plog[0]}, {Plog[-1]}] differs from the validated "
            f"{TABLE_T_K} K / {TABLE_P_LOGBAR}.")
    if np.any(ab <= 0.0) or not np.all(np.isfinite(ab)):
        raise RuntimeError(f"{path}: non-positive or non-finite abundances.")
    order = np.lexsort((P_col, T_col))
    raw_sorted = raw[order].reshape(N_T, N_P, 2 + len(species))
    cube = np.log10(ab[order]).reshape(N_T, N_P, len(species))
    # catalogued corrections (content-guarded; module docstring + registry)
    corrections_applied = []
    for entry in KNOWN_TABLE_CORRECTIONS.get(node, ()):
        iT = int(np.argmin(np.abs(T - entry["T"])))
        iP = int(np.argmin(np.abs(Plog - entry["logP"])))
        import hashlib
        row_sha = hashlib.sha1(
            raw_sorted[iT, iP].tobytes()).hexdigest()[:16]
        if row_sha != entry["corrupt_row_sha1"]:
            continue          # upstream fixed (or changed) the row: no-op
        if not 0 < iT < N_T - 1:
            raise RuntimeError(
                f"{path}: registered correction cell at a T edge -- the "
                "T-neighbor method does not apply; update the registry.")
        cube[iT, iP, :] = 0.5 * (cube[iT - 1, iP, :] + cube[iT + 1, iP, :])
        corrections_applied.append(
            {"node": node, "T": float(T[iT]), "logP": float(Plog[iP]),
             "method": entry["method"], "row_sha1": row_sha})
    # suspect-row bookkeeping (gas total = everything but graphite),
    # AFTER corrections; ``isolated`` = both T-neighbor rows are clean, the
    # signature of point corruption rather than systematic missing species
    igr = species.index(GRAPHITE)
    gas_cols = [j for j in range(len(species)) if j != igr]
    sums = (10.0 ** cube)[:, :, gas_cols].sum(axis=2)
    it, ip = np.where(sums < SUSPECT_SUM)
    suspect = []
    for a, b in zip(it, ip):
        neigh = [sums[a + d, b] for d in (-1, 1) if 0 <= a + d < N_T]
        suspect.append((float(T[a]), float(Plog[b]), float(sums[a, b]),
                        bool(neigh and min(neigh) >= GAS_SUM_WARN)))
    return SimpleNamespace(node=node, T=T, Plog=Plog, cube=cube,
                           species=species, suspect_cells=suspect,
                           corrections_applied=corrections_applied)


def _bracket(nodes: tuple, x: float, what: str) -> tuple[int, int, float]:
    """(i_lo, i_hi, weight of i_hi) on a node axis; exact-edge tolerant."""
    arr = np.asarray(nodes, dtype=float)
    if x < arr[0] - 1e-9 or x > arr[-1] + 1e-9:
        raise ValueError(
            f"{what}={x:g} outside the grid span [{arr[0]:g}, {arr[-1]:g}]")
    x = float(np.clip(x, arr[0], arr[-1]))
    hi = int(np.clip(np.searchsorted(arr, x, side="right"), 1, len(arr) - 1))
    lo = hi - 1
    w = (x - arr[lo]) / (arr[hi] - arr[lo])
    return lo, hi, float(w)


def bracketing_cells(met_x_solar: float, co_ratio: float) -> dict:
    """The 2x2 bracketing nodes + bilinear weights for a composition."""
    feh = float(np.log10(met_x_solar))
    flo, fhi, wf = _bracket(FEH_NODES, feh, "[M/H] (dex)")
    clo, chi, wc = _bracket(CO_NODES, float(co_ratio), "C/O")
    nodes = [["feh%.1f_co%.2f" % (FEH_NODES[f], CO_NODES[c])
              for c in (clo, chi)] for f in (flo, fhi)]
    return dict(nodes=nodes, wf=wf, wc=wc, feh=feh, co=float(co_ratio))


# ---------------------------------------------------------------------------
# Blended evaluation on a T-P profile
# ---------------------------------------------------------------------------

def _eval_cube_tp(cube: np.ndarray, T_grid: np.ndarray, Plog_grid: np.ndarray,
                  T_prof: np.ndarray, p_bar: np.ndarray) -> np.ndarray:
    """picaso's chem_interp convention: bilinear in (1/T, log10 P) of log10
    abundance. REFUSES outside the table (picaso would extrapolate)."""
    T_prof = np.asarray(T_prof, dtype=float)
    pl = np.log10(np.asarray(p_bar, dtype=float))
    if np.any(T_prof < T_grid[0]) or np.any(T_prof > T_grid[-1]):
        raise ValueError(
            f"profile temperature [{T_prof.min():.1f}, {T_prof.max():.1f}] K "
            f"outside the equilibrium table [{T_grid[0]:g}, {T_grid[-1]:g}] "
            "K -- refusing to extrapolate table chemistry.")
    if np.any(pl < Plog_grid[0] - 1e-9) or np.any(pl > Plog_grid[-1] + 1e-9):
        raise ValueError(
            f"profile pressure log10 range [{pl.min():.2f}, {pl.max():.2f}] "
            f"outside the table [{Plog_grid[0]:g}, {Plog_grid[-1]:g}] -- "
            "refusing to extrapolate table chemistry.")
    it_hi = np.clip(np.searchsorted(T_grid, T_prof, side="left"),
                    1, len(T_grid) - 1)
    it_lo = it_hi - 1
    ti = 1.0 / T_prof
    w_t = ((ti - 1.0 / T_grid[it_lo])
           / (1.0 / T_grid[it_hi] - 1.0 / T_grid[it_lo]))
    ip_hi = np.clip(np.searchsorted(Plog_grid, pl, side="right"),
                    1, len(Plog_grid) - 1)
    ip_lo = ip_hi - 1
    w_p = (pl - Plog_grid[ip_lo]) / (Plog_grid[ip_hi] - Plog_grid[ip_lo])
    wt = w_t[:, None]
    wp = w_p[:, None]
    return ((1 - wt) * (1 - wp) * cube[it_lo, ip_lo, :]
            + wt * (1 - wp) * cube[it_hi, ip_lo, :]
            + wt * wp * cube[it_hi, ip_hi, :]
            + (1 - wt) * wp * cube[it_lo, ip_hi, :])


def blend_cubes(tabs: list, wf: float, wc: float) -> np.ndarray:
    """Bilinear blend of the 2x2 node log10 cubes in (feh, C/O)."""
    ref = tabs[0][0].species
    for row in tabs:
        for t in row:
            if t.species != ref:
                raise RuntimeError(
                    f"node files disagree on species columns: {t.node} vs "
                    f"{tabs[0][0].node} -- refusing to blend.")
    return ((1 - wf) * (1 - wc) * tabs[0][0].cube
            + (1 - wf) * wc * tabs[0][1].cube
            + wf * (1 - wc) * tabs[1][0].cube
            + wf * wc * tabs[1][1].cube)


def evaluate(met_x_solar: float, co_ratio: float, T_prof: np.ndarray,
             p_bar: np.ndarray) -> SimpleNamespace:
    """Blended equilibrium state on a T-P profile.

    Returns raw (unnormalized) linear VMRs for all 50 columns with graphite
    renamed ``C-gr_l_s`` -- the shared RT path's condensate mask + per-layer
    gas normalization then treats it exactly like a VULCAN reservoir column.
    The certificate block carries the normalization / realized-composition /
    suspect-cell diagnostics; GAS_SUM_MIN violations raise here.
    """
    cell = bracketing_cells(met_x_solar, co_ratio)
    tabs = [[load_node_table(n) for n in row] for row in cell["nodes"]]
    ref = tabs[0][0]
    blended = blend_cubes(tabs, cell["wf"], cell["wc"])
    logab = _eval_cube_tp(blended, ref.T, ref.Plog, T_prof, p_bar)
    y = 10.0 ** logab
    n_floored = int(np.count_nonzero(logab <= FLOOR_LOG10))
    y[logab <= FLOOR_LOG10] = 0.0

    species = list(ref.species)
    igr = species.index(GRAPHITE)
    species_out = [GRAPHITE_OUT if s == GRAPHITE else s for s in species]
    gas = np.ones(len(species))
    gas[igr] = 0.0
    pre_norm_sum = (y * gas[None, :]).sum(axis=1)
    if np.any(pre_norm_sum < GAS_SUM_MIN):
        i = int(np.argmin(pre_norm_sum))
        raise RuntimeError(
            f"blended gas abundances sum to {pre_norm_sum[i]:.3f} at layer "
            f"{i} (T={float(np.asarray(T_prof)[i]):.0f} K, "
            f"p={float(np.asarray(p_bar)[i]):.2e} bar), below "
            f"GAS_SUM_MIN={GAS_SUM_MIN}: the tabulated species miss too much "
            "of the gas for a trustworthy renormalization here.")

    # realized gas-phase elemental composition (diagnostics; see docstring)
    nC = _atom_counts(species_out, "C") * gas
    nO = _atom_counts(species_out, "O") * gas
    denomO = y @ nO
    realized_co = np.where(denomO > 0.0, (y @ nC) / np.maximum(denomO, 1e-300),
                           np.nan)

    # suspect table rows that can actually influence this profile: any
    # loader-flagged cell within the (T, P) bounding box of the profile.
    # ISOLATED anomalies (clean T-neighbors -> point corruption, not the
    # systematic missing-species deficits) REFUSE unless a registered
    # correction already handled them -- unvetted corruption is never
    # renormalized through (v18.1).
    tmin, tmax = float(np.min(T_prof)), float(np.max(T_prof))
    plmin = float(np.min(np.log10(p_bar)))
    plmax = float(np.max(np.log10(p_bar)))
    dT = np.diff(ref.T).max()
    dP = np.diff(ref.Plog).max()
    flat_tabs = (tabs[0][0], tabs[0][1], tabs[1][0], tabs[1][1])
    suspect_hit = sorted({
        c for t in flat_tabs for c in t.suspect_cells
        if (tmin - dT) <= c[0] <= (tmax + dT)
        and (plmin - dP) <= c[1] <= (plmax + dP)})
    isolated_hit = [c for c in suspect_hit if c[3]]
    if isolated_hit:
        c0 = isolated_hit[0]
        raise RuntimeError(
            f"equilibrium table cell at (T={c0[0]:g} K, logP={c0[1]:g}) "
            f"has an ISOLATED anomalous gas sum {c0[2]:.3f} (clean "
            "T-neighbors) inside the evaluated profile span: point "
            "corruption is refused, never renormalized through. If the "
            "defect is vetted, register it in "
            "picaso_chem.KNOWN_TABLE_CORRECTIONS (see the catalogued "
            "feh1.0_co0.55 entry for the pattern); otherwise replace the "
            "reference table.")
    corrections = [c for t in flat_tabs for c in t.corrections_applied]

    cert = dict(
        nodes=cell["nodes"], wf=round(cell["wf"], 6), wc=round(cell["wc"], 6),
        feh=round(cell["feh"], 6), co=round(cell["co"], 6),
        # the full per-layer pre-normalization gas sums (v18.1: the summary
        # stats alone overstated what was recorded)
        gas_sum_layers=[round(float(v), 6) for v in pre_norm_sum],
        gas_sum_min=float(pre_norm_sum.min()),
        gas_sum_median=float(np.median(pre_norm_sum)),
        n_layers_below_warn=int(np.count_nonzero(pre_norm_sum < GAS_SUM_WARN)),
        gas_sum_warn=GAS_SUM_WARN, gas_sum_floor=GAS_SUM_MIN,
        n_floored_entries=n_floored,
        realized_gas_co_hotT=(
            float(np.nanmedian(realized_co[np.asarray(T_prof) >= CO_CHECK_T_K]))
            if np.any(np.asarray(T_prof) >= CO_CHECK_T_K) else None),
        suspect_cells_in_span=[list(c) for c in suspect_hit],
        corrections_applied=corrections,
    )
    return SimpleNamespace(
        y=y, species=species_out,
        species_masses=np.array([species_mass(s) for s in species_out]),
        pre_norm_sum=pre_norm_sum, realized_co=realized_co, cert=cert)
