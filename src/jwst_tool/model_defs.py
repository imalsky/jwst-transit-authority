"""What the forward model offers: networks, molecule sets, ranges, defaults and the model-cache version."""

from __future__ import annotations

import numpy as np

from jwst_tool import planets

MOLECULES = ["H2O", "CO2", "CO", "CH4", "SO2"]  # always-on RT set

# RT additions the network already solves; extra_mols exposes them to the RT.
EXTRA_MOLECULES = [
    "C2",
    "C2H2",
    "C2H4",
    "CH",
    "CH3",
    "CN",
    "CS",
    "H2CO",
    "H2O2",
    "H2S",
    "HCN",
    "N2O",
    "NH",
    "NH3",
    "NO",
    "NS",
    "OCS",
    "OH",
    "SH",
    "SO",
]

# GUI preselection: NEVER widen without a measured leave-one-out ppm impact.
EXTRA_MOLECULES_DEFAULT = ["C2H2", "C2H4", "H2S", "HCN", "NH3", "OCS", "SH", "SO"]

# Kinetics networks -> the engine's import-frozen (VULCAN3_NETWORK,
# VULCAN3_ATOM_LIST) pair. "sncho2025" (93 species) is the default: what
# upstream cfg_wasp39b.py ships, the Tsai 2023 lineage plus C3 isomers, C3H5
# and H2CS. "sncho" (89) is the 2021 file the Tsai 2023 WASP-39 b work used,
# kept for condensation; "ncho" drops sulfur (69 species). The first entry is
# the GUI's default selection.
NETWORKS = {
    "sncho2025": ("thermo/SNCHO_photo_network_2025.txt", "H,O,C,N,S"),
    "sncho": ("thermo/SNCHO_photo_network.txt", "H,O,C,N,S"),
    "ncho": ("thermo/NCHO_photo_network.txt", "H,O,C,N"),
}

DEFAULT_NETWORK = "sncho2025"

# Every S-bearing member of the lists above; network="ncho" REFUSES these
# rather than dropping them silently.
S_MOLECULES = frozenset({"SO2", "H2S", "OCS", "SO", "SH", "CS", "NS"})

# Network species with no published ExoMolOP k-table: never offered anywhere
# (correlated-k over the published tables is the only opacity path).
NO_EXOMOLOP_TABLE = frozenset({"CS2", "C2H6"})

DT_MAX_S = 1.0e13  # chemistry step-size cap (s); prevents the adaptive-dt balloon

# model_cache buster (identity = canonical params + this number, never a content
# hash); bump on any physics or canonical-key-set change.
_VERSION = 61

# The W39b cfg's elemental set (number ratios to H, 10x solar): Lodders 2020
# Table 8 present-day solar (log eps C 8.47, N 7.85, O 8.73, S 7.15; He is
# not scaled and stays at the abundance preset file's 10.9232) times 10, the
# numbers upstream VULCAN's vulcan_cfg.py carries. assemble_chem refuses on
# drift from the loaded cfg, so they are usable without the engine (share
# files, display).
CFG_ABUNDANCES = {"O_H": 0.00537, "C_H": 0.00295, "N_H": 0.000708, "S_H": 0.000141}

# GUI/API default co_ratio: the baseline C/O, the number ratio N_C/N_O on that
# set (C_H / O_H = 0.54935; not [C/H]/[O/H], not a log), rounded onto the
# widgets' 0.05 grid.
CO_DEFAULT = 0.55


def elemental_abundances(met_x_solar: float, co_ratio: float) -> dict:
    """Return the run's O/H, C/H, N/H, S/H.

    The cfg metals scaled by met_x_solar/10 (the cfg is 10x solar; He fixed), carbon from co_ratio at the scaled
    oxygen. Structural: the equilibrium seed re-initializes at exactly this set, so at C/O away from the baseline
    carbon is NOT at met_x_solar times solar.
    """
    m = float(met_x_solar) / 10.0
    o_h = CFG_ABUNDANCES["O_H"] * m
    return {
        "O_H": o_h,
        "C_H": float(co_ratio) * o_h,
        "N_H": CFG_ABUNDANCES["N_H"] * m,
        "S_H": CFG_ABUNDANCES["S_H"] * m,
    }


# Model top (chemistry grid AND RT top) when a request does not set it.
# 1e-7 bar is the usual published choice; 1e-9 bar is inside the allowed
# range.
RT_PTOP_DEFAULT = 1.0e-7

JAC_METHODS = ("fd", "ad")  # certified-FD default / warm-jvp opt-in

# Accepted co_ratio range, INCLUSIVE at both ends. TWO gate axes, network and
# photolysis; do NOT add temperature (non-monotonic) or Kzz (a tabulated column
# is not a scalar to gate on). Ten is the highest value SAMPLED, never a
# ceiling; the bound is deliberately permissive and the convergence certificate
# refuses the cases that do not converge. "sncho" is held at 0.99: across
# 0.99-1.02 the FD dlnCO row fails its h-vs-2h gate and the AD row is refused
# at build (notes "Decision records").
CO_MIN = 0.1

CO_MAX = {"sncho": 0.99, "sncho2025": 10.0, "ncho": 10.0}  # photolysis ON

CO_MAX_PHOTO_OFF = 10.0

# Accepted met_x_solar range, inclusive. The top is a decision (notes
# "Decision records") under the demonstrated envelope: the default case certifies through 300x
# and the lnZ stencil's 122x sits under the certified 150x.
# Convergence is certified per run; a value outside is refused before any
# solve.
MET_RANGE = (0.1, 100.0)

# Chemistry engines. "picaso": PICASO climate + equilibrium chemistry, run in
# its own environment (picaso_engine.py); finite-difference Jacobians only.
CHEM_PROVIDERS = ("vulcan", "picaso")

# picaso's climate heat redistribution (rfacv): 0.5 = planet average (a
# terminator), 1.0 = dayside (an eclipse). 0 (no stellar heating) is not offered.
RFACV_RANGE = (0.05, 1.0)

RFACV_DEFAULT = {"transmission": 0.5, "emission": 1.0}

# Climate T_int range: the GUI's T_int widget (picaso refuses T_int = 0).
TINT_RANGE = (50.0, 500.0)

# The climate grid and the quench search end at 100 bar (picaso_engine).
PICASO_P_BTM_MAX = 100.0

# PICASO metallicity: its thermochemical data are stated for +-1 dex and its
# climate continuum lacks H2O/CO2 self-CIA, so the top is set at 10x; the lnZ
# stencil turns one-sided at the cap (fd_stencil).
PICASO_MET_RANGE = (0.1, 10.0)

# One WASP-39 b climate solve: 497-570 s on one core across nine variants
# (numba warm; serial: four threads give bit-identical T in the same time)
PICASO_CLIMATE_SOLVE_S = 550.0

# One quenched chemistry-only PICASO column (~5 min: its atmosphere tuple
# needs the 2.2 GB correlated-k set loaded)
PICASO_QUENCH_COLUMN_S = 300.0

# RT extras missing from picaso's equilibrium species list
PICASO_NO_SPECIES = frozenset({"N2O"})


def co_bounds(network: str, use_photo: bool) -> tuple[float, float]:
    """(min, max) co_ratio for a network + photolysis state, both inclusive.

    canonical_params reads it, so the API and share_config refuse the same values.
    """
    return CO_MIN, (CO_MAX[network] if use_photo else CO_MAX_PHOTO_OFF)


# Cloud-deck Fisher rows: RT-only like lnR0 (one central difference or an RT
# jvp, no chemistry re-solve, no h-vs-2h gate). Need cloud_on.
CLOUD_FISHER_PARAMS = ("log_kappa_cloud", "alpha_cloud")

# Kzz profile modes. "file" requires tp_mode="file" with a Kzz column
# (the upstream constraint: the tabulated Kzz lives in the atm table).
KZZ_MODES = ("const", "Pfunc", "JM16", "file")

# tp_mode="file" profile sources.
TP_FILE_SHIPPED = "shipped"  # the vulcan3 table bundled for the planet

TP_FILE_UPLOAD = "upload"  # user-supplied table; content-addressed copy under <output>/uploads/<sha1>.txt

# Chemistry-grid pressure span (dyn/cm^2) of the SHIPPED W39b cfg,
# cross-checked against the live cfg in assemble_chem. For a RUN's span ask
# chem_p_span_dyn(cp), never this constant.
CHEM_P_SPAN_DYN = (0.1, 7.6e6)

# Column bottom, structure-aware: file mode defaults to the shipped W39b cfg
# and table bottom, parametric profiles to a round 10 bar. The range ceiling
# is a DATA limit (opacities are built for T_WINDOW).
P_BTM_FILE_BAR = CHEM_P_SPAN_DYN[1] / 1.0e6  # 7.6 bar, the shipped W39b cfg bottom

P_BTM_PARAMETRIC_BAR = 10.0

P_BTM_RANGE = (1.0, 300.0)

# The RT column bottom sits just inside the chemistry bottom: interp_map refuses
# an ART grid reaching below it. Keep this 7.0 / 7.6 ratio at any depth.
ART_PBTM_FRACTION = 7.0 / 7.6


def chem_p_span_dyn(cp: dict) -> tuple:
    """Return the RUN's chemistry span (P_t, P_b) in dyn/cm^2.

    The top follows rt_ptop_bar, the bottom p_btm_bar (CHEM_P_SPAN_DYN is the shipped cfg's).
    """
    return float(cp["rt_ptop_bar"]) * 1.0e6, float(cp["p_btm_bar"]) * 1.0e6


def default_tirr(planet: str, system: dict | None = None, science_mode: str = "transmission") -> float:
    """Guillot T_irr default for ``planet``.

    Sqrt(2) * T_eq, GUI-identical (planets.default_tirr is the single definition both sides call). For the custom
    planet pass ``system`` (star_teff, rstar_rsun, orbit_au).
    """
    return planets.default_tirr(
        planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS),
        system=(system if planet not in planets.PLANETS else None),
        science_mode=science_mode,
    )


def active_molecules(cp: dict) -> list[str]:
    """RT molecule set for canonical params: base set + selected extras."""
    base, extras = MOLECULES, EXTRA_MOLECULES
    if cp["network"] == "ncho":
        base = [m for m in base if m not in S_MOLECULES]
        extras = [m for m in extras if m not in S_MOLECULES]
    return base + [m for m in extras if m in cp["extra_mols"]]


# Numerical-resolution knobs layered on engine_config.WIDE (1-15 um band
# unchanged). rt_profile_common locks the RT layer count equal to nz, so there
# is no RT-layer knob; the spectral grid is the k-tables' R = 1000 grid.
NZ_DEFAULT, YCONV_DEFAULT = 100, 1.0e-2

NZ_RANGE = (60, 150)  # equal counts on distinct chemistry and RT grids

# Steady-state convergence tolerance: tighter costs runtime but is safe, the longdy gate is loud.
YCONV_RANGE = (1.0e-4, 1.0e-2)

# The runner certifies on EITHER branch: tight (yconv_cri, slope_cri) or loose
# (yconv_min, slope_min). Every cold baseline solve exits on the loose one, so
# yconv_cri alone cannot tighten a run; this is the knob that can. Its cfg
# default is the upstream 0.1, and it may only tighten (never above it).
YCONV_MIN_DEFAULT = 0.1

YCONV_MIN_RANGE = (1.0e-4, 0.1)

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
    # PICASO's radiative-convective climate solver (chem_provider='picaso'):
    # no T-P row -- the T_int row does not certify (canonical_params)
    "picaso_climate": [],
}

# FD Fisher steps. lnZ scales O/C/N/S together (C/O preserved), dlnCO scales
# C_H at fixed O, lnKzz/T-P rows perturb theta on the same build. Every row is
# evaluated at h AND 2h -- disagreement beyond FD_CONSISTENCY_TOL RAISES --
# and reported as the Richardson combination (4 J_h - J_2h)/3.
FD_STEPS = {
    "lnZ": 0.10,
    "dlnCO": 0.10,
    "lnKzz": 0.10,  # ln-space steps
    "Tirr": 10.0,
    "Tint": 10.0,  # Kelvin
    "log_kappa": 0.05,
    "log_gamma": 0.05,  # dex
    "log_kappa_cloud": 0.05,
    "alpha_cloud": 0.05,
}  # dex / slope


FD_COMP_PARAMS = ("lnZ", "dlnCO")  # need a chemistry re-init per FD point


def fd_stencil(name: str, value: float, co_max: float, met_max: float | None = None) -> tuple[tuple[int, ...], float]:
    """(step multiples, step) a composition row solves at.

    Central (+-h, +-2h) at h = FD_STEPS[name], unless the dlnCO stencil would leave co_max (the network's
    photolysis-ON ceiling, the conservative choice photo-off too) or the lnZ stencil would leave met_max (the PICASO
    cap; None = no cap): then the row uses the one-sided stencil (-1, -2, -4) toward lower values -- third order
    after Richardson -- at h/2 so its reach equals the central stencil's 2h (at the full step the curvature toward
    the boundary fails the h-vs-2h gate on a 1600 K hot Jupiter).
    """
    h = FD_STEPS[name]
    top = {"dlnCO": co_max, "lnZ": met_max}.get(name)
    if top is not None and value * float(np.exp(2.0 * h)) > top:
        return (-1, -2, -4), h / 2.0
    return (1, -1, 2, -2), h
