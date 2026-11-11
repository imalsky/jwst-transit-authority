"""What the forward model offers: networks, molecule sets, ranges, defaults and the model-cache version."""

from __future__ import annotations

import numpy as np

from jwst_tool import planets

# Always-on RT set. vulcan-forward refuses a band that lacks H2O, CO2, CO, CH4 or NH3 where each absorbs
# (constants.REQUIRED_MOLECULES), so those five are never optional here; SO2 stays on with them.
MOLECULES = ["H2O", "CO2", "CO", "CH4", "SO2", "NH3"]
# Extras of earlier versions that are now always on. `extra_mols` may still name one (an old share file or script):
# it is dropped from the list, and the run is the model that was asked for. This is the one legacy reading the tool
# keeps, so that old share files that list NH3 as an extra still load.
FORMER_EXTRA_MOLECULES = frozenset({"NH3"})

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
    "NO",
    "NS",
    "OCS",
    "OH",
    "SH",
    "SO",
]

# GUI preselection: NEVER widen without a measured leave-one-out ppm impact.
EXTRA_MOLECULES_DEFAULT = ["C2H2", "C2H4", "H2S", "HCN", "OCS", "SH", "SO"]

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
_VERSION = 64

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

# Request defaults. One definition each: canonical_params fills a missing key from here, and the sidebar widgets
# (directly or through share_config.WIDGET_RANGES) start from the same value. "No basis recorded" marks a default
# chosen without a recorded measurement or citation.
MET_DEFAULT = 10.0  # x solar: the W39b cfg's set (CFG_ABUNDANCES), the metallicity Tsai et al. 2023 fit for WASP-39 b
KZZ_X_DEFAULT = 1.0  # multiplier on the whole Kzz profile: the profile as given
KZZ_CONST_DEFAULT = 1.0e9  # cm^2/s, constant Kzz: the log-centre of KZZ_CONST_RANGE; no basis recorded
KZZ_KMAX_DEFAULT = 1.0e5  # cm^2/s, Pfunc deep Kzz: vulcan3's config value (K_max)
KZZ_PLEV_DEFAULT_BAR = 0.1  # bar, Pfunc transition level: vulcan3's config value (K_p_lev)
KZZ_KDEEP_DEFAULT = 1.0e5  # cm^2/s, JM16 deep floor: equal to the Pfunc deep value; no basis recorded
LOG_KAPPA_DEFAULT = -2.0  # log10 cm^2/g, Guillot kappa_IR: 0.01 cm^2/g, Guillot (2010)'s thermal opacity
LOG_GAMMA_DEFAULT = -1.0  # log10 kappa_vis / kappa_IR: no basis recorded
F_DIURNAL_DEFAULT = 1.0  # diurnal photolysis factor: vulcan3's config value (f_diurnal)
# bar: a catalogue transit radius belongs near the transmission photosphere, not at the RT grid bottom; 1 mbar is not
# tuned to data (notes "Structure: T-P, Kzz, gravity anchor, model top")
P_REF_DEFAULT_BAR = 1.0e-3
LOG_KAPPA_CLOUD_DEFAULT = -1.0  # log10 cm^2/g at 3.5 um, cloud deck opacity when the deck is on: no basis recorded
ALPHA_CLOUD_DEFAULT = 0.0  # cloud deck slope: 0 is a grey deck, the engine's powerlaw_clouds convention
RT_INTEGRATION_DEFAULT = "simpson"  # the chord scheme exojax's ArtTransPure uses by default
# Switches. Photolysis and molecular diffusion are VULCAN physics (off under PICASO, which has neither): on, as in the
# W39b cfg. Upwind advection off: the validated baseline (notes "Structure: T-P, Kzz, gravity anchor, model top").
# Rayleigh on: zero-parameter physics, without which the < 1.5 um slope is biased. No cloud and no condensation by
# default: a cloud is never a default (oracle), and condensation is detection-only. No PICASO quench: equilibrium.
USE_PHOTO_DEFAULT = True
USE_MOLDIFF_DEFAULT = True
USE_VM_MOL_DEFAULT = False
USE_RAYLEIGH_DEFAULT = True
CLOUD_ON_DEFAULT = False
USE_CONDENSE_DEFAULT = False
PICASO_QUENCH_DEFAULT = False
# The API's Jacobian method: certified finite differences. The sidebar starts from "ad" (faster); the maintainer
# decides whether the two become one (todo).
JAC_METHOD_DEFAULT = "fd"

# Fraction of Pandeia's per-mode saturation_fullwell a ramp may reach: PandExo's default saturation level, 80 % (docs/
# noise_and_binning.md says what that divisor is).
SAT_LIMIT_DEFAULT = 0.80

# Gravity anchor: the catalogue (Rp, g) pair is moved to the chemistry's anchor level, a fixed point with the profile
# that reads g (Guillot's tau = kappa P / g, or PICASO's climate). Passes run until one moves g by at most ANCHOR_TOL
# (relative); the Guillot anchors run at least ANCHOR_MIN_PASSES (cheap, and the passes every default has run), and an
# anchor still moving after ANCHOR_MAX_PASSES is refused. Measured, the third Guillot pass moves g by 1e-7 to 7e-5
# (notes "Structure: T-P, Kzz, gravity anchor, model top"), so a map not settled by the eighth is not contracting.
ANCHOR_TOL = 5.0e-3
ANCHOR_MIN_PASSES = 3
PICASO_ANCHOR_MIN_PASSES = 1  # a climate pass is a full climate solve: none runs past the first that settles
ANCHOR_MAX_PASSES = 8

# Heat redistribution factor f of the Guillot (2010) profile: 1/4 spreads the irradiation over the whole planet. The
# dayside an eclipse sees is carried by its T_irr instead (planets.DAYSIDE_TEQ_FACTOR).
GUILLOT_F_REDIST = 0.25


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


# Model top (chemistry grid AND RT top) of a run with no T-P table that does not set it, bar: the usual published
# choice. A tp_mode="file" run defaults to its table's own top (run_params.default_rt_ptop_bar).
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

# T_int range (K), Guillot and climate alike: the GUI's T_int widget (picaso refuses T_int = 0). It holds every
# planets.default_tint value: the Thorngren, Gao & Fortney (2019) rule's least is 45 K at T_eq 700 K (the low edge of
# its fit) and its most about 671 K near T_eq 1883 K, under the "up to 700 K" the paper quotes for its grid.
TINT_RANGE = (40.0, 700.0)

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

# Accepted ranges, inclusive. canonical_params and the sidebar widgets (share_config.WIDGET_RANGES) both read these,
# so the API and the GUI refuse the same values. Kzz in cm^2/s.
KZZ_CONST_RANGE = (1.0e6, 1.0e12)
KZZ_KMAX_RANGE = (1.0e4, 1.0e11)  # Pfunc deep Kzz
KZZ_PLEV_RANGE = (1.0e-5, 1.0e2)  # Pfunc transition pressure, bar
KZZ_KDEEP_RANGE = (1.0e4, 1.0e11)  # JM16 deep floor
KZZ_X_RANGE = (0.1, 10.0)  # multiplier on the whole Kzz profile
SL_ANGLE_RANGE = (0.0, 89.0)  # photolysis zenith angle, deg
F_DIURNAL_RANGE = (0.1, 1.0)  # diurnal photolysis factor
# Model top, bar. The top end admits the shipped WASP-39 b table's top (5.3e-6 bar); above 1e-5 bar the
# molecular-diffusion advection puts a 2dz oscillation on atomic H (notes "Structure: T-P, Kzz, gravity anchor, model
# top").
RT_PTOP_RANGE = (1.0e-9, 1.0e-5)
LOG_KAPPA_RANGE = (-4.0, 0.0)  # Guillot log10 kappa_IR, cm^2/g
LOG_GAMMA_RANGE = (-2.0, 0.3)  # Guillot log10 kappa_vis / kappa_IR
LOG_KAPPA_CLOUD_RANGE = (-7.0, 2.0)  # cloud deck log10 kappa at 3.5 um, cm^2/g
ALPHA_CLOUD_RANGE = (0.0, 4.0)  # cloud deck power-law slope

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

# bar: where rp and g may be quoted, the widest RT grid any run builds. canonical_params also requires it inside the
# run's own grid [rt_ptop_bar, p_btm_bar * ART_PBTM_FRACTION]; the sidebar widget offers this range.
P_REF_RANGE_BAR = (RT_PTOP_RANGE[0], P_BTM_RANGE[1] * ART_PBTM_FRACTION)


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
# and reported as the Richardson combination (4 J_h - J_2h)/3. Fixed, not
# settings. A smaller step lets the solver's convergence noise, which a
# difference divides by h, fail the gate: dlnCO at h = 0.05 reads 0.258
# (notes "Jacobians and Fisher"), and the PICASO T_int row 1.190 at 7 K against
# 0.541 at 10 K (notes "PICASO engine"). The Guillot kappa and gamma and the
# cloud-deck steps have no basis recorded.
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
