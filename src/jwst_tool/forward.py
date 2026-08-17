"""Forward model runner: VULCAN-JAX photochemistry -> ExoJax transmission spectrum.

Two faces:

* Imported by the GUI (light): ``params_key`` / ``cache_path`` / ``load_result``
  touch only the disk cache -- no JAX, no VULCAN, no ExoJax imports.
* Run as a script (heavy): ``python -m jwst_tool.forward params.json`` runs the
  live pipeline and writes the npz cache entry. Progress goes to stdout as
  "[fwd] ..." lines; "[fwd] PROG <frac> <label>" lines drive the GUI bar.

Every planet in ``planets.PLANETS`` (plus "custom") runs the same
W39b-validated SNCHO machinery -- one shared code path, not per-planet
validation (see notes.md, Decision records section). The planet identity
enters via cfg_overrides (chemistry) and profile rp_cm/gs_cgs/rstar_cm (RT).
A GCM profile is never silently substituted.

Resolution knobs: nz (chemistry layers; the RT grid is locked equal in
run_model), nu_pts (native wavenumber points over 1-15 um, native
R ~ nu_pts/2.7 -- LINE-BY-LINE mode only; the default correlated-k mode
takes its grid from the published k-tables at R = 1000 and ignores it),
yconv_cri (steady-state convergence tolerance). Defaults are
the old "fast" tier; the validated ceiling is the old "high" tier
(nz=150, nu_pts=8000, yconv 1e-3).

Structure and physics knobs: tp_mode ("guillot"; "file" = explicit tabulated
T(P) [+ Kzz(P)], cache-keyed by content hash, no T-P Fisher rows -- file-mode
forecasts condition on the profile and are stated optimistic;
"picaso_climate"); kzz_mode (const / Pfunc / JM16 / file); structural
composition (met_x_solar scales the cfg O/C/N/S together, co_ratio sets
C_H = co_ratio * O_H, FastChem re-initializes at the requested values -- one
path for every composition, including C-rich); use_photo, sl_angle_deg,
f_diurnal, use_moldiff, use_rayleigh; the power-law and Mie cloud decks
(deck parameters can be freed as RT-only Fisher rows); boundary conditions
(use_settling, diff_esc, top_flux, bot_flux); extra_mols; and the
echo-checked RT knobs rt_ptop_bar / rt_integration / rt_dit_res.

Any T-P outside the modelable opacity window [320, 2980] K is REJECTED with
a clear error, never clipped -- same rule as the retrieval.

Condensation is DETECTION-ONLY (certified S8 recipe, CONDEN_CFG). The
fix-species pin freezes the reservoir at a step-sequence-dependent transient,
so the state is not a reproducible function of the inputs; measured jvp-vs-FD
relative error ~0.91 (the tangent is order-unity WRONG), and FD of pinned
transients is equally untrustworthy. ``canonical_params`` refuses
use_condense with fisher_params (any jac_method), with photo off, and with
use_moldiff off; adjoint_diag refuses condensing states too. Documented
footgun: a column too hot to condense still pins S8 at its end-of-window
value -- conden-on does not reduce to conden-off. For aerosol opacity in a
forecast use the differentiable ExoJax cloud deck, never condensation.

Fisher machinery: with ``fisher_params`` set, the runner computes the
spectrum Jacobian row by row; the method is recorded per row in the npz
(``jac_row_method``). jac_method="fd" (the API default -- it works
everywhere; the GUI defaults to "ad" since 2026-08-09) is certified central
finite differences: composition rows (lnZ, dlnCO) re-initialize the
chemistry per FD point, lnKzz/T-P rows perturb theta on the baseline build,
every row must pass the h-vs-2h consistency gate, lnR0 is RT-only.
jac_method="ad" is one
warm-started forward-mode jvp per row (photo-on required, ~1.7-4x faster;
cross-validated vs FD to 0.07-1.6% on the then-default W39b configuration --
method validation, not a per-row guarantee). Two stated AD caveats: the lnZ
jvp is the fixed-structural-grid derivative, and the dlnCO jvp uses the
fixed-O direction, refused near O-exhaustion (b_z bound); FD is valid
everywhere. ``fisher.py`` turns the Jacobian + Pandeia noise into forecasts.

Per-molecule "removed" spectra zero that molecule's VMR in the RT only --
structure (T, mmw) is kept: the standard nested-model comparison.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# instruments is import-light (os + pathlib only, safe on the GUI's light path)
# and owns the data/output root resolution (env-overridable, loud on failure)
from jwst_tool import instruments as _ins

MODEL_CACHE = _ins.MODEL_CACHE

from jwst_tool import planets   # installed package: works as module AND as a script

MOLECULES = ["H2O", "CO2", "CO", "CH4", "SO2"]   # always-on WIDE-profile set
# RT additions beyond the base set. The SNCHO network already solves these;
# extra_mols only exposes them to the RT and the detection scores. COMPLETE
# since v34: every network species with a published ExoMolOP table on the
# R1000 0.3-50um grid. What is absent cannot be added (reasons:
# vulcan_forward.constants.MOLECULES). CS2/C2H6 stay listed so selecting one
# raises the explanatory refusal below rather than a bare KeyError.
EXTRA_MOLECULES = ["C2", "C2H2", "C2H4", "C2H6", "CH", "CH3", "CN", "CS",
                   "CS2", "H2CO", "H2O2", "H2S", "HCN", "N2O", "NH", "NH3",
                   "NO", "NS", "OCS", "OH", "SH", "SO"]

# Which of them the GUI preselects. MEASURED (2026-08-17), not assumed: max
# |depth - depth_without| over the 2706 bands on the converged W39b and HD189
# reference atmospheres, best of the two. Preselected: H2S 309, NH3 301,
# HCN 241, OCS 60.5, SO 48.1, SH 10.0 ppm (C2H2 0.149 / C2H4 0.024 are held
# over for the C/O ~ 1 case neither planet tests). Left off: CN 2.35, OH 0.46,
# and 10 more at or below 0.08. Full table: notes.md, v34.
# NEVER widen this without the ppm measurement behind it -- each default-on
# molecule adds a leave-one-out spectrum (block ~ n(n+3)/2 folds).
EXTRA_MOLECULES_DEFAULT = ["C2H2", "C2H4", "H2S", "HCN", "NH3", "OCS",
                           "SH", "SO"]
# VULCAN kinetics network menu (v33). "sncho" is the shipped default; "ncho"
# drops sulfur (69 vs 89 species) for a cheaper solve when no S species is
# needed. Values: (VULCAN_JAX_NETWORK, VULCAN_JAX_ATOM_LIST) -- import-frozen
# in the engine, so the solver subprocess sets both env vars from the
# canonical parameter before the first engine import.
NETWORKS = {
    "sncho": ("thermo/SNCHO_photo_network.txt", "H,O,C,N,S"),
    "ncho": ("thermo/NCHO_photo_network.txt", "H,O,C,N"),
}
# Sulfur-bearing entries of the molecule lists above: absent from the NCHO
# network, so network="ncho" refuses them rather than dropping them silently.
# MUST list every S-bearing member of MOLECULES + EXTRA_MOLECULES -- a species
# missing here would be offered under a network that cannot produce it, and
# would then be modelled at whatever the elemental repair left behind rather
# than refused. SO/SH/CS/NS joined at v34 with the species sweep.
_S_MOLECULES = frozenset({"SO2", "H2S", "CS2", "OCS", "SO", "SH", "CS", "NS"})
# Species ExoMolOP publishes no k-table for: refused EARLY under the default
# opacity_mode (canonical_params), instead of failing minutes later inside
# exomolop.load_tables. Cross-checked against exomolop.available() by a
# data-gated test so this set cannot rot when ExoMolOP adds a species.
_NO_EXOMOLOP_TABLE = frozenset({"CS2", "C2H6"})
_VERSION = 34  # model_cache buster: bump whenever the physics or the canonical
               # key set changes. Per-version history lives in notes.md.
               # DELIBERATE (reviews keep re-finding it): the cache identity
               # is canonical params + this hand-bumped version, NOT content
               # hashes or commit pins of vulcan-forward/vulcan-jax/exojax or
               # the line lists. That trade (maintainer discipline over
               # content addressing, for a single-maintainer research tool)
               # is recorded as S2-05 in notes.md, Decision records section;
               # the PICASO subsystem is the exception and DOES fingerprint
               # its reference tables.

# Baseline C/O of the shipped network: the number ratio N_C/N_O (not
# [C/H]/[O/H], not a log). Basis is the W39b cfg's elemental set
# (C_H 2.95e-3, O_H 5.37e-3, Tsai et al. 2023 10x-solar) -- the ONLY valid
# basis: the FastChem .dat set (0.458) only seeds the equilibrium initial
# guess and never survives into the converged column. This constant is only
# the GUI default co_ratio and display baseline; run_model cross-checks it
# against the loaded cfg's C_H/O_H and refuses on drift.
CO_BASELINE = 0.00295 / 0.00537   # = 0.54935, cfg C_H/O_H (Tsai 2023 10x-solar)
# The GUI/API default co_ratio (v32): the baseline rounded onto the widgets'
# 0.05 grid. Default runs SOLVE at this value; CO_BASELINE stays the display
# baseline and the live-cfg cross-check constant.
CO_DEFAULT = 0.55

# FD Fisher Jacobians: every composition/structure row is a CENTRAL difference
# of fully re-initialized, longdy-certified cold solves (the upstream-VULCAN
# workflow). Directions: lnZ scales O/C/N/S together (C/O preserved); dlnCO
# scales C_H at fixed O; lnKzz/T-P rows perturb theta on the same build (Kzz/T
# enter on-graph). Each row is evaluated at h AND 2h; disagreement beyond
# FD_CONSISTENCY_TOL RAISES (a row dominated by convergence noise is never
# reported); the reported row is the Richardson combination (4 J_h - J_2h)/3.
# AD (jac_method="ad") is the warm-jvp opt-in, ~1.7-4x faster, photo-on only;
# FD stays the default: certified, valid everywhere.
FD_STEPS = {"lnZ": 0.10, "dlnCO": 0.10, "lnKzz": 0.10,      # ln-space steps
            "Tirr": 10.0, "Tint": 10.0,                     # Kelvin
            # Tint_cl: full certified climate re-run per FD point
            # (bit-deterministic; h-vs-2h of dT/dTint is 1.6% at h=15 K)
            "Tint_cl": 15.0,                                # Kelvin
            "log_kappa": 0.05, "log_gamma": 0.05,           # dex
            "log_kappa_cloud": 0.05, "alpha_cloud": 0.05,   # dex / slope
            # Mie rows: rg/sigmag steps stay inside one miegrid cell (grid
            # spacing ~0.1 dex / ~0.33), so the FD row is the local slope.
            # MMR is not gridded (dtau linear in it).
            "mie_log_rg": 0.03, "mie_sigmag": 0.05, "mie_log_mmr": 0.05}
FD_COMP_PARAMS = ("lnZ", "dlnCO")     # need a chemistry re-init per FD point
FD_CONSISTENCY_TOL = 0.25
FD_LNR0_STEP = 0.01                   # lnR0 is RT-only (smooth, analytic)
JAC_METHODS = ("fd", "ad")            # certified-FD default / warm-jvp opt-in
# Minimum oxygen-reservoir bound b_z for the AD dlnCO row. co_bz_bound =
# ln(1 + min_z(OO_z/OC_z)) is nonnegative by construction, so a "<= 0" refusal
# can never fire; the reachable criterion is one FD stencil width (2h) from
# the O-exhaustion boundary, below which the per-layer tangent factors are
# ill-conditioned. FD re-initializes and is valid at any composition.
CO_BZ_MIN_AD = 2.0 * FD_STEPS["dlnCO"]   # = 0.2
# Cloud-deck Fisher parameters: RT-only rows, evaluated like the lnR0 nuisance
# (single central difference or RT jvp; no chemistry re-solve, no h-vs-2h
# gate). Only available with cloud_on.
CLOUD_FISHER_PARAMS = ("log_kappa_cloud", "alpha_cloud")

# Mie cloud deck: condensate cloud from the exojax PdbCloud/OpaMie miegrid,
# an alternative (or addition) to the analytic power-law deck. mie_condensate
# selects the species ("" = off); the three knobs are one column-uniform
# lognormal size distribution (mie_log_rg / mie_sigmag ride the miegrid
# interp; mie_log_mmr is not gridded -- dtau is linear in it). Curated to the
# condensates exojax 2.2.3 ships indices AND a density for (must match
# tools/generate_miegrid.py SUPPORTED); grids are generated once, loaded at
# run time.
MIE_CONDENSATES = ("NH3", "H2O", "MgSiO3", "Mg2SiO4", "Fe", "Al2O3", "TiO2")
MIE_FISHER_PARAMS = ("mie_log_rg", "mie_sigmag", "mie_log_mmr")
# Ranges for the gridded knobs stay inside the miegrid edges with margin for
# the +-2h FD stencil (getix edge-clamps; a clamped stencil would silently
# zero the row). MMR is not gridded (generous physical envelope).
MIE_LOG_RG_RANGE = (-6.5, -3.5)      # ~3 nm to ~3 um mean radius
MIE_SIGMAG_RANGE = (1.15, 3.85)      # inset from [1.0001, 4.0] by > 2h (0.10)
MIE_LOG_MMR_RANGE = (-12.0, -2.0)
MIE_DATA_SUBDIR = "exojax_mie"       # under DATA_DIR (miegrids + virga archive)

# Kzz profile modes (v16). "file" requires tp_mode="file" with a Kzz column
# (the upstream constraint: the tabulated Kzz lives in the atm table).
KZZ_MODES = ("const", "Pfunc", "JM16", "file")

# Diffusion-limited-escape species choices (v16). Curated to the light species
# the TOA escape formula is meant for AND guaranteed present in the SNCHO
# network -- an arbitrary species would only fail deep inside the engine.
DIFF_ESC_CHOICES = ("H", "H2", "He")

# Boundary-condition flux sanity bounds (v16): generous physical envelopes so a
# typo (1e30) fails at the API instead of producing a silently absurd column.
BC_FLUX_MAX = 1.0e15              # |flux| ceiling, molecules cm^-2 s^-1
BC_VDEP_MAX = 1.0e3               # deposition-velocity ceiling, cm s^-1

# tp_mode="file" profile sources (v16).
TP_FILE_SHIPPED = "shipped"       # the cfg's own atm_file (W39b evening terminator)
TP_FILE_UPLOAD = "upload"         # user-supplied table; content-addressed copy
                                  # under <output>/uploads/<sha1>.txt

# Chemistry-grid pressure span (dyn/cm^2) of the shipped W39b cfg: (P_t, P_b)
# = (0.1, 7.6e6), i.e. 1e-7 to 7.6 bar (vulcan_jax configs/W39b.yaml). The
# engine re-grids a tabulated T-P onto this FIXED span with a constant-value
# clamp outside the table, so an uploaded table that stops above P_b would
# silently run the entire CO/CH4/NH3 quench region isothermal at its last
# tabulated temperature -- _read_tp_table REFUSES that (v17). A clamped TOP is
# the standard upstream convention (the shipped profile itself extends its
# topmost T over ~1.7 decades) and is logged loudly, not refused. run_model
# cross-checks these constants against the LIVE cfg, CO_BASELINE-style.
CHEM_P_SPAN_DYN = (0.1, 7.6e6)

# The column bottom is a per-run parameter (v27) with STRUCTURE-AWARE defaults
# (v32): a measured T-P table caps honest chemistry at its own bottom -- the
# shipped tables end at 7.6 bar, and chemistry below a table's end is refused
# (the quench-region clamp, v17) -- while the parametric profiles (Guillot /
# isothermal) default to a round 10 bar, both geometries.
#
# Emission does NOT need a deeper default. It briefly defaulted to 100 bar,
# retracted on measurement: on HD 189733 b at 7.6 bar the bottom optical depth
# is 30-370 across the real instrument windows; the only tau < 3 sliver is a
# 60 nm notch at 1.02-1.08 um where the modeled opacity set (lines + H2 CIA)
# runs out and the real continuum there (H- bound-free/free-free, Na/K wings,
# TiO/VO) is unmodeled. Deepening buys opacity the real planet does not get,
# at 4x the chemistry cost, in the least-trusted opacity temperatures -- so
# the tau-bottom gate is flux-weighted, not min() (see run_model).
#
# The RANGE stays open to 300 bar for a genuinely deep case. The ceiling is a
# DATA limit, not a preference: every opacity here is built for T_WINDOW and
# the shipped H2-H2 CIA table physically ends at 3000 K, so a column hotter
# than that deeper down is refused rather than extrapolated.
P_BTM_FILE_BAR = CHEM_P_SPAN_DYN[1] / 1.0e6    # 7.6 bar, the shipped tables' bottom
P_BTM_PARAMETRIC_BAR = 10.0
P_BTM_RANGE = (1.0, 300.0)
# The RT column bottom sits just inside the chemistry bottom: interp_map refuses
# an ART grid reaching below the chemistry (deep clamping would fabricate
# chemistry). The shipped pair was 7.0 / 7.6 bar; keep that ratio at any depth.
ART_PBTM_FRACTION = 7.0 / 7.6


def default_opacity_mode(params: dict) -> str:
    """The published ExoMolOP opacities, correlated-k, for both observables.

    Two modes exist. "exomolop" (the default) is correlated-k over the
    published ExoMol/HITEMP high-temperature k-tables. "lbl" samples the
    cross section directly on the output grid, far below exojax's critical
    resolution, and is measurably biased in BOTH observables -- worst in
    emission, where it suppressed more than half the emergent flux
    (measurements: vulcan-forward README, the two Opacity sections; the
    engine's interim HITRAN-built "ckd" mode was removed with forward 0.8.0).

    ONE case keeps "lbl", and it is physics, not preference: a MIE CONDENSATE
    DECK. Mie extinction has structure across a band, so it cannot be folded
    into a k-distribution built from line opacity.

    Choosing the mode from the inputs is a default, not a fallback: asking
    for a correlated-k mode explicitly alongside a Mie deck still raises.
    """
    if str(params.get("mie_condensate", "") or ""):
        return "lbl"
    return "exomolop"


def default_p_btm_bar(params: dict) -> float:
    """Chemistry-grid bottom (bar) for this run, both geometries: a measured
    T-P table's own bottom in file mode (the shipped tables end at 7.6 bar,
    and chemistry below a table's end is refused), a round 10 bar for the
    parametric profiles. See the P_BTM_* block above."""
    mode = str(params.get("tp_mode", _default_tp_mode(params)))
    return P_BTM_FILE_BAR if mode == "file" else P_BTM_PARAMETRIC_BAR


# Instrument windows the emission tau-bottom report is broken down by, so a
# refusal says WHERE the column sees through rather than only that it does.
# The 1-2 um entry is first on purpose: it is where the modeled opacity set
# runs out, and a min-only report cannot distinguish that from a shallow column.
_TAU_WINDOWS = (("1.0-2.0 um (no modeled continuum)", 1.0, 2.0),
                ("2.0-3.0 um", 2.0, 3.0),
                ("3.0-5.2 um (NIRSpec)", 3.0, 5.2),
                ("5.0-12 um (MIRI LRS)", 5.0, 12.0),
                ("12-15 um", 12.0, 15.0))


# Share of the planet's EMITTED FLUX that may come from wavelengths where the
# column sees through its own bottom. This is the emission gate, and it is
# flux-weighted on purpose: a min() over the whole band let a 60 nm notch at the
# extreme blue edge, carrying under a thousandth of the planet's emission, veto
# an otherwise thoroughly opaque column (measured min tau 30 to 370 across every
# instrument window). 1% is the tolerance -- below it the interior-source
# assumption cannot move a reported number; above it the run is refused.
EMIS_TAU_THIN = 3.0            # per-wavelength "sees through the bottom"
EMIS_THIN_FLUX_FRAC = 0.01     # tolerated share of emitted flux from those


def _tau_bottom_breakdown(wl_um, tau, flux=None) -> str:
    """Per-window minimum bottom optical depth, how much of the band is below
    the gate, and (given ``flux``) how much of the planet's emission comes from
    there. A single min() cannot tell a shallow column from a band edge with no
    modeled opacity; this can."""
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
    """The RUN's chemistry span (P_t, P_b) in dyn/cm^2.

    CHEM_P_SPAN_DYN is the shipped cfg's OWN span and stays the constant the
    live-cfg cross-check compares against; the bottom is overridden per run
    through cfg_overrides["P_b"], so every gate that asks "does this profile
    cover the chemistry grid" must ask THIS, not the module constant.
    """
    return (CHEM_P_SPAN_DYN[0],
            float(cp.get("p_btm_bar") or default_p_btm_bar(cp)) * 1.0e6)

# Structure default (2026-08-11 maintainer decision, reversing the structure
# half of the 2026-08-09 speed-first flip): where vulcan_jax bundles a
# MEASURED T-P/Kzz table VERIFIED end-to-end for a planet, that table (and
# its Kzz column) is the default structure; the analytic Guillot profile is
# the default everywhere else and under the picaso provider. Rationale: the
# default W39b run must be the literature-comparable SO2 state; on the Guillot
# + constant-Kzz stand-in it read ~2 sigma (measured 2026-07-21: ~100 K hot
# through the SO2 formation zone, Kzz 4-33x low). The table run measured 4.16
# sigma against Alderson+2023 4.8 / Tsai+2023 4.5 -- but that was BEFORE the
# v26 radius anchoring, whose 1.42x excess contrast inflated it. Re-measured
# at v27: G395H SO2 sigma_detect 2.89, i.e. the tool now forecasts BELOW the
# published detection. The table is still the right default (it beats Guillot
# by the same margin); the absolute agreement is an OPEN gap, see README. The table also converges
# FASTER (28 s vs ~8 min), so this costs no speed; the stated trade-off is
# that file mode has no T-P Fisher row (Fisher forecasts are conditional on
# the profile -- switch to Guillot when a temperature row is needed). The
# AD-first GUI defaults of 2026-08-09 are kept and compatible (AD gates on
# photo/conden/provider, not on tp_mode). Which planets have a table is
# data: planets.PLANETS[...]["tp_table"].
def _default_tp_mode(params: dict) -> str:
    """tp_mode default: the planet's measured table where a default run on it
    is VERIFIED, else the analytic Guillot profile. Having a table is not
    enough -- see shipped_tp_table_is_default.

    EMISSION always defaults to Guillot (v27), whatever the planet ships. The
    bundled tables are TERMINATOR profiles -- WASP-39 b's is an exo-FMS GCM
    average over +/-10 deg about the eastern terminator -- and a terminator
    profile is the wrong structure for a dayside eclipse. Selecting a table
    for emission stays possible and is then the user's explicit choice.
    """
    if str(params.get("science_mode", "transmission")) == "emission":
        return "guillot"
    planet = str(params.get("planet", "wasp39b"))
    if (shipped_tp_table_is_default(planet)
            and str(params.get("chem_provider", "vulcan")) == "vulcan"):
        return "file"
    return "guillot"


def default_tirr(planet: str, system: dict | None = None,
                 science_mode: str = "transmission") -> float:
    """Guillot T_irr default for ``planet`` -- sqrt(2) * T_eq, GUI-identical
    (planets.default_tirr is the single definition both sides call). For the
    custom planet pass ``system`` (star_teff, rstar_rsun, orbit_au) so T_eq
    derives from the entered star and orbit instead of the WASP-39 b seed."""
    return planets.default_tirr(
        planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS),
        system=(system if planet not in planets.PLANETS else None),
        science_mode=science_mode)

# PICASO equilibrium provider + climate T-P mode. chem_provider selects the
# atmospheric-state engine; everything downstream of the chemistry (RT,
# binning, noise, detect/fisher) is SHARED. The picaso provider is FD-only
# (not differentiable), equilibrium-only (no photochemistry -> no SO2/S2/S8/
# CS2: the W39b sulfur science stays VULCAN-only), composition axes = the
# Visscher grid (C/O hard-capped at 1.10). Its composition Jacobians are
# symmetric two-cell interpolant secants with a one-sided-secant kink gate
# (picaso_chem.FD_KINK_TOL): the defaults sit ON grid nodes with no unique
# local derivative, and a row whose left/right secants disagree hard-errors.
CHEM_PROVIDERS = ("vulcan", "picaso")
PICASO_EXPERIMENTAL_ENV = "JWST_TOOL_ENABLE_UNCERTIFIED_PICASO"


def picaso_experimental_enabled() -> bool:
    """Whether the explicitly uncertified PICASO research path is enabled."""
    return os.environ.get(PICASO_EXPERIMENTAL_ENV) == "1"


PICASO_MET_RANGE = (0.1, 100.0)       # = [M/H] in [-1, 2], inside the grid
PICASO_CO_RANGE = (0.14, 1.10)        # the Visscher grid span (hard cap)
# ln-space steps chosen so 2h stays inside the cells adjacent to the defaults
PICASO_FD_STEPS = {"lnZ": 0.10, "dlnCO": 0.04}

# tp_mode="picaso_climate": a PICASO radiative-convective-equilibrium T-P,
# post-processed with either provider's chemistry and the ExoJax RT. ONE-WAY
# coupling -- never present this as "self-consistent with VULCAN".
# Composition under climate mode is exact-CK-node only (per-node tables, no
# interpolation); the converged profile is cached per (climate inputs +
# refdata fingerprint) in picaso_climate.py. climate_rcb is a NUMERICAL SEED
# (PICASO's initial convective-zone guess), not a physical parameter: PICASO
# grows convective zones upward but cannot shrink one seeded too shallow, so
# a shallow seed imposes a spurious convective region that passes every
# a-posteriori check. Default is the deep seed per PICASO's own guidance; the
# value is cache-keyed and the Tint_cl FD row differentiates at fixed seed.
TINT_CL_RANGE = (50.0, 500.0)
TINT_CL_DEFAULT = 200.0
RFACV_CHOICES = (0.0, 0.5, 1.0)       # no / full-redistribution / dayside
CLIMATE_N_LEVELS = 91
# The deep seed is a layer index, derived from CLIMATE_N_LEVELS so a grid
# change keeps it near the bottom. The floor stays a shallow 10 so seed
# sensitivity can still be probed (an in-window shallow seed runs and carries
# the spurious-convective-zone bias; the GUI help says so).
CLIMATE_RCB_DEEP_MARGIN = 6                            # convective layers below
CLIMATE_RCB_DEFAULT = CLIMATE_N_LEVELS - CLIMATE_RCB_DEEP_MARGIN   # 85 @ 91
CLIMATE_RCB_RANGE = (10, CLIMATE_N_LEVELS - 3)        # (10, 88) @ 91
CLIMATE_P_SPAN_BAR = (1.0e-6, 300.0)  # solve grid (equilibrium tables start
                                      # at 1e-6 bar; policy in picaso_chem)


def _picaso_fingerprint() -> dict:
    """Provider content fingerprint (test seam -- monkeypatched by the fast
    suite, which runs without picaso or its reference tree)."""
    from jwst_tool import picaso_env
    return picaso_env.chem_fingerprint()


def _picaso_climate_fingerprint(node: str, tio_vo: bool) -> str:
    """Climate refdata content fingerprint (test seam, same contract)."""
    from jwst_tool import picaso_env
    return picaso_env.climate_refdata_fingerprint(node, tio_vo)


def active_molecules(cp: dict) -> list[str]:
    """RT molecule set for canonical params: provider base set + extras."""
    if cp.get("chem_provider", "vulcan") == "picaso":
        from jwst_tool import picaso_chem as _pc
        return list(_pc.PICASO_MOLECULES) + [
            m for m in _pc.PICASO_EXTRA_MOLECULES if m in cp["extra_mols"]]
    base, extras = MOLECULES, EXTRA_MOLECULES
    if cp.get("network", "sncho") == "ncho":
        base = [m for m in base if m not in _S_MOLECULES]
        extras = [m for m in extras if m not in _S_MOLECULES]
    return base + [m for m in extras if m in cp["extra_mols"]]

# Numerical-resolution knobs layered on the base RT profile in
# engine_config.WIDE (1-15 um band unchanged). The RT layer count is locked
# equal to nz in run_model, so there is no separate RT-layer knob.
# Measured tier sensitivity (notes.md): near-IR modes move <= 6% low vs high
# tier, but weak mid-IR SO2 bands nearly double, and the entire movement is
# native sampling (nu_pts), not nz/yconv. Raise nu_pts for final mid-IR
# numbers and treat MIRI-LRS weak-band sigmas as sampling-limited lower bounds.
#
# The grid is ESLOG over 667-10000 cm^-1, so the resolving power is constant:
# R = (nu_pts - 1) / ln(10000/667) = nu_pts / 2.708. The cap was 8000 (R 2954),
# which was BELOW what the highest-R shipped mode needs -- NIRSpec G395H's
# line-spread function is unresolvable by the model grid there, so the LSF
# operator silently no-ops on it (detect.py now says so out loud). Resolving a
# Gaussian kernel needs R_model >= 2.3548 * R_native, i.e. nu_pts >= ~12,300
# for G395H. The cap is 32,000 so that convergence can at least be DEMONSTRATED
# in range; it is not a claim that 32,000 converges. It does not: PreMODIT
# renders any line narrower than a grid cell as a ~2-cell feature, and these
# line cores are Doppler-limited at R = 3.4e5 to 6.7e5, which would need
# nu_pts ~ 1e6 and ~127 GB of broadening arrays. This model cannot be
# line-core converged as built; correlated-k (exojax.opacity.ckd) is the
# durable answer. Cost measured on the way up: broadening arrays 0.267 GB at
# 4000, 1.07 GB at 16,000, 2.14 GB at 32,000, with cross-section evaluation
# ~8x slower at 16,000 than at 4000.
NZ_DEFAULT, NU_PTS_DEFAULT, YCONV_DEFAULT = 100, 4000, 1.0e-2
NZ_RANGE = (60, 150)            # chemistry (= RT) layers
NU_PTS_RANGE = (4000, 32000)    # native wavenumber points (R = nu_pts / 2.708)
YCONV_RANGE = (1.0e-4, 1.0e-2)  # steady-state convergence tolerance (1e-3 is the
                                # validated "high" tier; below it costs runtime
                                # but is safe -- the longdy gate rejects loudly)

# Modelable temperature window -- reject, never clip. The numbers are the
# PreMODIT table range with a 20 K inset, which set the limit when
# line-by-line was the only opacity mode. Under the default correlated-k
# mode the published ExoMolOP tables span 100-3400 K, so this window is
# CONSERVATIVE there; it is kept as-is because nothing above 2980 K has been
# validated (the H2-H2 CIA table also ends near 3000 K). Widening it is a
# physics decision backed by a run, not a constant edit.
T_WINDOW = (320.0, 2980.0)

# Parameters that can be freed in the Fisher forecast, per tp_mode. "file" is
# deliberately EMPTY: a tabulated profile has no temperature parameter, so
# file-mode forecasts condition on the profile (documented as optimistic).
CHEM_PARAM_NAMES = ["lnZ", "dlnCO", "lnKzz"]
TP_PARAM_NAMES = {
    "guillot": ["Tirr", "Tint", "log_kappa", "log_gamma"],
    "file": [],
    # climate mode: ONE structure parameter, the internal temperature. It is
    # a REBUILD row (each FD point re-runs the certified climate solve +
    # chemistry + RT), never a theta row -- named Tint_cl so its FD step and
    # nuisance registration never collide with guillot's on-graph Tint.
    "picaso_climate": ["Tint_cl"],
}
# Display SYMBOL, UNIT, and friendly name per parameter for the GUI's constraint
# table / science goals. The symbol is what a reader recognizes and MUST match the
# unit's log base: metallicity and Kzz are reported in dex (log10), so their
# symbols are [M/H] and log Kzz (never "ln", which would mislabel the base); C/O
# is the absolute number ratio N_C/N_O (dimensionless, so no unit bracket).
PARAM_SYMBOLS = {"lnZ": "[M/H]", "dlnCO": "C/O", "lnKzz": "log Kzz",
                 "Tirr": "T_irr", "Tint": "T_int",
                 "Tint_cl": "T_int (climate)",
                 "log_kappa": "log κ_IR", "log_gamma": "log γ",
                 "log_kappa_cloud": "log κ_cloud", "alpha_cloud": "α_cloud",
                 "mie_log_rg": "log r_g", "mie_sigmag": "σ_g",
                 "mie_log_mmr": "log MMR"}
PARAM_UNITS = {"lnZ": "dex", "dlnCO": "", "lnKzz": "dex",
               "Tirr": "K", "Tint": "K", "Tint_cl": "K",
               "log_kappa": "dex", "log_gamma": "dex",
               "log_kappa_cloud": "dex", "alpha_cloud": "",
               "mie_log_rg": "dex(cm)", "mie_sigmag": "", "mie_log_mmr": "dex"}
PARAM_LABELS = {"lnZ": "Metallicity", "dlnCO": "C/O ratio",
                "lnKzz": "Vertical mixing (Kzz)",
                "Tirr": "Guillot T_irr",
                "Tint": "Guillot T_int", "log_kappa": "Guillot log κ_IR",
                "log_gamma": "Guillot log γ",
                "Tint_cl": "Climate internal T (full re-solve)",
                "log_kappa_cloud": "Cloud deck log κ (at 3.5 um)",
                "alpha_cloud": "Cloud deck slope α",
                "mie_log_rg": "Mie particle radius (log r_g)",
                "mie_sigmag": "Mie size dispersion (σ_g)",
                "mie_log_mmr": "Mie condensate abundance (log MMR)"}


def param_axis(name: str) -> str:
    """Axis/column label for a parameter: 'Symbol [unit]', or bare 'Symbol' when
    it is dimensionless (C/O). Keeps every user-facing header on the standard
    representation (e.g. '[M/H] [dex]', 'C/O', 'T_irr [K]')."""
    u = PARAM_UNITS[name]
    return f"{PARAM_SYMBOLS[name]} [{u}]" if u else PARAM_SYMBOLS[name]


# ---------------------------------------------------------------------------
# tp_mode="file" helpers (light path: no vulcan_jax/jax imports)
# ---------------------------------------------------------------------------

def shipped_tp_table_name(planet: str) -> str:
    """Filename of the MEASURED structure table bundled for ``planet``, or ""
    when that planet has none (planets.PLANETS[...]["tp_table"])."""
    entry = planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS)
    return entry.get("tp_table") or ""


def shipped_tp_table_is_default(planet: str) -> bool:
    """Whether ``planet``'s bundled table is its DEFAULT structure.

    Separate from merely having one: a table only becomes the default once a
    default run on it has been verified end-to-end here (default-selecting
    again since 2026-08-11; between 2026-08-09 and then this was a pure
    verification record under the Guillot-everywhere default). HD 189733 b
    ships a good profile the solver will not certify at default settings, so
    it stays selectable-but-not-default rather than making that planet error
    on arrival (planets.PLANETS[...]["tp_table_note"] carries the
    measurement)."""
    entry = planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS)
    return bool(entry.get("tp_table") and entry.get("tp_table_default"))


def _shipped_tp_file(planet: str) -> Path:
    """Path of the shipped T-P/Kzz table for ``planet``, WITHOUT importing
    vulcan_jax (find_spec only -- importing parses the reaction network, far
    too heavy for the GUI's cache-key path). A planet with no usable table
    raises with the reason, never substitutes another planet's atmosphere."""
    name = shipped_tp_table_name(planet)
    if not name:
        entry = planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS)
        why = entry.get("tp_table_note") or "no table is bundled for it"
        raise ValueError(
            f"tp_file='shipped' is not available for planet {planet!r}: {why} "
            "Choose tp_mode='guillot' (or 'picaso_climate'), or upload a table "
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

    ``span``: the RUN's chemistry span (P_t, P_b) in dyn/cm^2, which the
    coverage and T-window checks are made against. Defaults to the shipped
    cfg's own span, which is the transmission depth; an emission run passes
    its deeper one (see chem_p_span_dyn).

    Mirrors the engine's read exactly (np.genfromtxt, names=True,
    skip_header=1: line 1 is a units comment, line 2 the column names):
    columns 'Pressure' (dyne/cm^2) and 'Temp' (K) are required, 'Kzz'
    (cm^2/s) is optional. Returns {"P_dyn", "T", "Kzz" or None}. Raises
    ValueError with the offending detail on any malformed content -- a bad
    table must fail at the API, never inside the engine's pre-loop."""
    span = CHEM_P_SPAN_DYN if span is None else (float(span[0]), float(span[1]))
    try:
        # genfromtxt emits a warning, rather than an exception, for an empty
        # file. Reject it before NumPy so corrupt share uploads fail cleanly.
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
    # the table, so a short table would silently run the deep quench region
    # isothermal. A clamped TOP is the standard upstream convention (logged,
    # not refused).
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
    # The T window applies to what the engine actually evaluates: T re-gridded
    # onto the chemistry span (log-P interp, edge clamp -- mirror it exactly),
    # NOT every raw row. Checking raw rows wrongly rejects tables that merely
    # extend past the span (e.g. a thermosphere above it).
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
            f"{T_WINDOW[1]:.0f}] K{_extra}. Opacity tables end there; "
            "out-of-window profiles are rejected, never clipped.")
    Kzz = None
    if "Kzz" in names:
        Kzz = np.asarray(tab["Kzz"], dtype=np.float64)
        if not np.all(np.isfinite(Kzz)) or np.any(Kzz <= 0.0):
            raise ValueError(f"T-P table {path}: Kzz column must be finite and > 0")
    return {"P_dyn": P, "T": T, "Kzz": Kzz}


def _resolve_tp_file(params: dict) -> tuple[Path, str]:
    """(path, content-sha1[:16]) of the requested T-P table.

    'shipped' resolves to the vulcan_jax table bundled FOR THIS PLANET;
    'upload' takes params['tp_file_path'] (any readable path -- run_model
    copies it to the content-addressed uploads/<sha1>.txt so later
    re-resolution from the canonical params alone always works)."""
    src = str(params.get("tp_file", TP_FILE_SHIPPED))
    if src == TP_FILE_SHIPPED:
        path = _shipped_tp_file(str(params.get("planet", "wasp39b")))
    elif src == TP_FILE_UPLOAD:
        raw = params.get("tp_file_path")
        if raw:
            path = Path(str(raw))
        else:
            # No raw path: fall back to the content-addressed archive. This
            # is what makes canonical params ROUND-TRIP: the GUI hands the
            # subprocess canonical_params(params) (no tp_file_path in it),
            # and the sha alone re-resolves the exact bytes.
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
    """Re-resolve the T-P table from CANONICAL params alone (no raw path):
    shipped -> the bundled file; upload -> the content-addressed copy
    uploads/<sha1>.txt (run_model wrote it). Verifies the content hash --
    a table that changed since the cache key was computed is refused."""
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


# ---------------------------------------------------------------------------
# Boundary-condition helpers (v16)
# ---------------------------------------------------------------------------

def _canon_bc_entries(raw, *, kind: str) -> list:
    """Canonicalize boundary-flux entries for the cache key.

    kind='top': entries [species, flux]; kind='bot': [species, flux, vdep].
    Accepts lists/tuples (or dicts with species/flux/vdep keys); returns a
    sorted, rounded, duplicate-free list of lists (JSON-stable). Zero-flux
    top rows and zero-flux+zero-vdep bottom rows are dropped (inert -- they
    must not fragment the cache). Loud on any malformed entry."""
    if not raw:
        return []
    want = 2 if kind == "top" else 3
    out = {}
    for i, e in enumerate(raw):
        if isinstance(e, dict):
            e = ([e.get("species"), e.get("flux")] if kind == "top" else
                 [e.get("species"), e.get("flux"), e.get("vdep", 0.0)])
        e = list(e)
        if len(e) != want:
            raise ValueError(
                f"{kind}_flux entry {i}: expected {want} fields "
                f"({'species, flux' if kind == 'top' else 'species, flux, vdep'}), "
                f"got {e!r}")
        sp = str(e[0]).strip()
        if not sp or not sp.replace("_", "").isalnum():
            raise ValueError(f"{kind}_flux entry {i}: bad species token {e[0]!r}")
        try:
            flux = float(e[1])
            vdep = float(e[2]) if kind == "bot" else 0.0
        except (TypeError, ValueError):
            raise ValueError(f"{kind}_flux entry {i} ({sp}): non-numeric value")
        if not np.isfinite(flux) or abs(flux) > BC_FLUX_MAX:
            raise ValueError(
                f"{kind}_flux for {sp}: flux {flux:g} not finite or beyond "
                f"|flux| <= {BC_FLUX_MAX:g} molecules cm^-2 s^-1")
        if kind == "bot" and (not np.isfinite(vdep) or not 0.0 <= vdep <= BC_VDEP_MAX):
            raise ValueError(
                f"bot_flux for {sp}: vdep {vdep:g} outside [0, {BC_VDEP_MAX:g}] cm/s")
        if sp in out:
            raise ValueError(f"{kind}_flux: duplicate species {sp!r}")
        if flux == 0.0 and (kind == "top" or vdep == 0.0):
            continue                      # inert row: keep the cache key clean
        row = [sp, float(f"{flux:.6e}")]
        if kind == "bot":
            row.append(float(f"{vdep:.6e}"))
        out[sp] = row
    return [out[sp] for sp in sorted(out)]


def _write_bc_file(kind: str, entries: list) -> Path:
    """Write the VULCAN BC token file for the canonical entries and return its
    path (content-addressed under <output>/bc_files/, idempotent). Format is
    upstream's: comment header, then 'species flux' (top) or
    'species flux vdep' (bottom) token rows."""
    lines = ["# generated by jwst_tool.forward (v16 boundary conditions)",
             ("# species  flux(cm^-2 s^-1)" if kind == "top"
              else "# species  flux(cm^-2 s^-1)  vdep(cm/s)")]
    for row in entries:
        lines.append("  ".join(f"{v:.6e}" if isinstance(v, float) else str(v)
                               for v in row))
    text = "\n".join(lines) + "\n"
    tag = hashlib.sha1(text.encode()).hexdigest()[:12]
    bc_dir = MODEL_CACHE.parent / "bc_files"
    bc_dir.mkdir(parents=True, exist_ok=True)
    path = bc_dir / f"{kind}_{tag}.txt"
    if not path.exists():
        path.write_text(text)
    return path


# VULCAN condensation channel on the SNCHO network (detection-only; the
# module docstring says why it can never meet a derivative). One reaction:
# S8 -> S8_l_s (no H2O_l_s/NH3_l_s species on this network). Particles are
# rainout-sized 50 um orthorhombic sulfur (smaller radii make the growth term
# stiffer than Ros2 resolves). Methodology is upstream VULCAN's conden window
# + whole-column fix_species pin (from_coldtrap_lev=False: the cold-trap
# argmin degenerates on isothermal columns). Without the pin the steady state
# is transport-limited and every solve exhausts count_max. Documented caveat:
# on planets too hot to condense the pin still freezes the end-of-window
# transient.
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
    # At the 1e-20 default, kinetically-glacial trace species gate longdy
    # forever on cold columns; 1e-15 is still far below RT relevance.
    "mtol_conv": 1.0e-15,
    # Heavy hydrocarbons + trace sulfur allotropes: against a pinned S8 they
    # re-equilibrate on unreachable timescales (>=1e15 s) at abundances far
    # below RT relevance; none is an RT molecule. The observable sulfur
    # species (SO2, H2S, SO) STAY in the gate.
    "conver_ignore": ["C6H6", "C2H2", "C6H5", "C2H", "C2H4", "C2H5", "C2H6",
                      "C3H2", "C3H3", "C4H5", "CH2NH", "CH3NH2", "H2CCO",
                      "S", "S2", "S3", "S4"],
    # Certification bounded from below so the conden window + pin always
    # complete before the convergence gate may fire.
    "trun_min": 1.0e6,
}


# Every input key canonical_params (and the helpers it calls) actually reads.
# Pinned by an AST-walking test (test_forward_params) that re-derives this set
# from the source, so it cannot rot when a parameter is added.
_PARAM_KEYS_READ = frozenset({
    "Tint", "Tirr", "alpha_cloud", "bot_flux", "broadening", "chem_provider",
    "climate_rcb", "cloud_on", "co_ratio", "diff_esc", "extra_mols",
    "f_diurnal", "fisher_params", "gs_cgs", "jac_method", "kzz_const",
    "kzz_kdeep", "kzz_kmax", "kzz_mode", "kzz_plev", "kzz_x", "log_gamma",
    "log_kappa", "log_kappa_cloud", "met_x_solar", "mie_condensate",
    "mie_log_mmr", "mie_log_rg", "mie_sigmag", "network", "nu_pts", "nz",
    "opacity_mode",
    "orbit_au", "p_btm_bar", "p_ref_bar", "planet", "rfacv", "rp_rjup",
    "rstar_rsun", "rt_dit_res", "rt_integration", "rt_ptop_bar",
    "science_mode", "sflux", "sl_angle_deg", "star_feh", "star_logg",
    "star_teff", "tint_cl", "tio_vo", "top_flux", "tp_file", "tp_file_path",
    "tp_file_sha1", "tp_mode", "use_condense", "use_moldiff", "use_photo",
    "use_rayleigh", "use_settling", "use_vm_mol", "wo_mols", "yconv_cri",
})
# Output-only keys of the canonical dict itself: a SAVED canonical payload
# round-trips through canonical_params for validation (share_config), so its
# echo fields must not be rejected as unknown.
_PARAM_KEYS_ECHOED = frozenset({
    "version", "picaso_version", "picaso_chemgrid_sha1",
    "picaso_climate_sha1", "picaso_ck_node",
})
_KNOWN_PARAM_KEYS = _PARAM_KEYS_READ | _PARAM_KEYS_ECHOED

# Misspellings worth a pointed hint. "mode" is not hypothetical: a validation
# driver passed {"mode": "emission"}, the key was silently ignored, and a
# TRANSMISSION spectrum was scored against eclipse data (chi2/N ~ 5e5). An
# unknown key must never be dropped quietly (standing fail-loud rule).
_PARAM_KEY_HINTS = {"mode": "science_mode", "opacity": "opacity_mode",
                    "metallicity": "met_x_solar", "co": "co_ratio"}


def canonical_params(params: dict) -> dict:
    unknown = sorted(set(params) - _KNOWN_PARAM_KEYS)
    if unknown:
        hints = "".join(
            f" (did you mean {_PARAM_KEY_HINTS[k]!r} instead of {k!r}?)"
            for k in unknown if k in _PARAM_KEY_HINTS)
        raise ValueError(
            f"unknown parameter key(s) {unknown}.{hints} Unknown keys are "
            "refused rather than ignored, because a silently dropped key "
            "means the model computed is not the model you asked for. The "
            "canonical key set is forward._KNOWN_PARAM_KEYS.")
    tp_mode = str(params.get("tp_mode", _default_tp_mode(params)))
    if tp_mode not in TP_PARAM_NAMES:
        raise ValueError(
            f"unknown tp_mode {tp_mode!r} (choose from {list(TP_PARAM_NAMES)}). "
            "The WASP-39b GCM 'baseline' and the isothermal profile were removed "
            "-- use a Guillot profile, tp_mode='file' with an explicit table "
            "(v16), or tp_mode='picaso_climate' (v18).")
    provider = str(params.get("chem_provider", "vulcan"))
    if provider not in CHEM_PROVIDERS:
        raise ValueError(
            f"unknown chem_provider {provider!r}: choose from "
            f"{list(CHEM_PROVIDERS)} ('vulcan' = the VULCAN-JAX kinetics "
            "engine, the default; 'picaso' = PICASO equilibrium chemistry, "
            "FD-only, no photochemistry/SO2, C/O <= 1.10).")
    network = str(params.get("network", "sncho"))
    if network not in NETWORKS:
        raise ValueError(
            f"unknown network {network!r}: choose from {list(NETWORKS)} "
            "('sncho' = the shipped S-N-C-H-O kinetics network, the default; "
            "'ncho' = the sulfur-free N-C-H-O network, a cheaper solve with "
            "no SO2/H2S/CS2/OCS).")
    if network != "sncho" and provider == "picaso":
        raise ValueError(
            "network='ncho' selects the VULCAN kinetics network and has no "
            "meaning under chem_provider='picaso' (equilibrium chemistry, no "
            "kinetics network). Drop the network key or use the vulcan "
            "provider.")
    needs_picaso = provider == "picaso" or tp_mode == "picaso_climate"
    if needs_picaso and not picaso_experimental_enabled():
        raise RuntimeError(
            "PICASO chemistry/climate is uncertified in this release and is "
            "disabled by default: PICASO 4.0.1 requires NumPy >=2 while the "
            "validated ExoJAX 2.2.3 stack requires NumPy <2, and the native-RT "
            "cross-model artifact does not pass. Use the VULCAN path for "
            "collaborator results. Maintainers may set "
            f"{PICASO_EXPERIMENTAL_ENV}=1 only for isolated investigation; "
            "such output is not release-certified.")
    if provider == "picaso" and tp_mode == "file":
        raise ValueError(
            "tp_mode='file' is not implemented under chem_provider='picaso': "
            "file mode rides the kinetics engine's default-temperature path "
            "(tp_eval=None), which the equilibrium provider does not have -- "
            "its T(P) comes from the Guillot profile or the climate solve. "
            "Use tp_mode='guillot' or 'picaso_climate', or "
            "chem_provider='vulcan' for tabulated profiles "
            "(README.md, PICASO engine section).")
    # tp_mode="file": resolve + validate the table NOW (numpy parse + content
    # hash, no engine imports) so a bad upload fails at the API and the cache
    # key is content-addressed, never path-addressed. tp_table is reused by
    # the kzz_mode="file" gate below.
    science_mode = str(params.get("science_mode", "transmission"))
    # Chemistry-grid bottom: emission needs an optically thick column, so it is
    # resolved BEFORE the table is validated -- the coverage gate is against
    # THIS run's span, not the shipped cfg's.
    p_btm_bar = float(params.get("p_btm_bar", default_p_btm_bar(params)))
    if not P_BTM_RANGE[0] <= p_btm_bar <= P_BTM_RANGE[1]:
        raise ValueError(
            f"p_btm_bar={p_btm_bar:g} outside {P_BTM_RANGE} bar. It is the "
            "chemistry and RT column bottom; below the low end the deep "
            "quench region is cut off, and above the high end no shipped "
            "profile stays inside the modelable temperature window "
            f"{T_WINDOW} K, where the opacity tables end.")
    _span = (CHEM_P_SPAN_DYN[0], p_btm_bar * 1.0e6)
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
    sysd = planets.system_fields(planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS))
    nz = int(params.get("nz", NZ_DEFAULT))
    if not NZ_RANGE[0] <= nz <= NZ_RANGE[1]:
        raise ValueError(f"nz={nz} outside the validated layer range {NZ_RANGE} "
                         "(chemistry layers, also used for the RT grid)")
    nu_pts = int(params.get("nu_pts", NU_PTS_DEFAULT))
    if not NU_PTS_RANGE[0] <= nu_pts <= NU_PTS_RANGE[1]:
        raise ValueError(f"nu_pts={nu_pts} outside the validated range {NU_PTS_RANGE} "
                         "(native wavenumber points; native R ~ nu_pts/2.7)")
    yconv_cri = float(params.get("yconv_cri", YCONV_DEFAULT))
    if not YCONV_RANGE[0] <= yconv_cri <= YCONV_RANGE[1]:
        raise ValueError(f"yconv_cri={yconv_cri:g} outside the validated range "
                         f"{YCONV_RANGE} (steady-state convergence tolerance)")
    sflux = str(params.get("sflux", sysd["sflux"]))
    if sflux not in planets.SFLUX_CHOICES:
        raise ValueError(f"unknown stellar UV spectrum {sflux!r} "
                         f"(choose from {list(planets.SFLUX_CHOICES)})")
    star_ref = planets.PLANETS.get(planet, planets.CUSTOM_DEFAULTS)["star"]
    # resolved BEFORE the literal: the custom planet's Tirr default derives
    # from these three (planets.system_teq), so they must exist first
    _teff = round(float(params.get("star_teff", star_ref["teff"])), 1)
    _rstar = round(float(params.get("rstar_rsun", sysd["rstar_rsun"])), 4)
    _orbit = round(float(params.get("orbit_au", sysd["orbit_au"])), 5)
    cp = {
        "planet": planet,
        "science_mode": science_mode,
        # Star identity for the eclipse normalization Fp/Fs (v16 emission):
        # part of the MODEL only in emission mode (zeroed in transmission --
        # there the star lives purely on the noise side).
        "star_teff": _teff,
        "star_logg": round(float(params.get("star_logg", star_ref["log_g"])), 2),
        "star_feh": round(float(params.get("star_feh",
                                           star_ref["metallicity"])), 2),
        "nz": nz,
        "nu_pts": nu_pts,
        "yconv_cri": round(yconv_cri, 6),
        "rp_rjup": round(float(params.get("rp_rjup", sysd["rp_rjup"])), 4),
        "gs_cgs": round(float(params.get("gs_cgs", sysd["gs_cgs"])), 1),
        "rstar_rsun": _rstar,
        "orbit_au": _orbit,
        "sflux": sflux,
        "met_x_solar": round(float(params.get("met_x_solar", 10.0)), 4),
        # C/O default is provider/mode-aware: climate CK tables exist only at
        # exact nodes (0.55 = the 10x-solar default node), and the picaso
        # provider defaults mid-cell (0.50) so the constraint stencil stays
        # inside one table cell.
        "co_ratio": round(float(params.get(
            "co_ratio",
            0.50 if (provider == "picaso" and tp_mode != "picaso_climate")
            else CO_DEFAULT)), 6),
        # Kzz default follows the structure: a tabulated T-P that carries a
        # Kzz column supplies the mixing profile too; no column -> constant.
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
        # file-mode identity: source label + content hash; the hash keys the
        # cache so two tables can never share an entry ("" outside file mode)
        "tp_file": tp_file,
        "tp_file_sha1": tp_file_sha1,
        # Guillot T_irr default = sqrt(2) * T_eq of the SELECTED planet, and
        # the DAYSIDE T_eq in emission (planets.default_tirr is the single
        # definition the GUI calls too -- a constant here would make API and
        # GUI defaults diverge per planet)
        "Tirr": round(float(params.get("Tirr", default_tirr(
            planet, system=dict(star_teff=_teff, rstar_rsun=_rstar,
                                orbit_au=_orbit),
            science_mode=science_mode))), 2),
        "Tint": round(float(params.get("Tint", 100.0)), 2),
        "log_kappa": round(float(params.get("log_kappa", -2.3)), 3),
        "log_gamma": round(float(params.get("log_gamma", -1.0)), 3),
        # physical VULCAN knobs (all flow through the validated cfg_overrides hook;
        # defaults reproduce the previous hard-coded behavior = the W39b cfg values)
        "use_photo": bool(params.get("use_photo", True)),
        "sl_angle_deg": round(float(params.get("sl_angle_deg", 83.0)), 1),
        "f_diurnal": round(float(params.get("f_diurnal", 1.0)), 3),
        "use_moldiff": bool(params.get("use_moldiff", True)),
        # Upwind molecular-diffusion advection (Shami's vm_branch hybrid scheme).
        # PINNED explicitly since v11: VULCAN-JAX flipped its own default to True
        # on 2026-07-14 and the tool inherited it silently. Default False = the
        # tool's validated pre-flip baseline; True = the upwind scheme, not yet
        # re-baselined for this tool. Only meaningful with use_moldiff on
        # (the engine gates use_vm on use_vm_mol AND use_moldiff).
        "use_vm_mol": bool(params.get("use_vm_mol", False)),
        # RT physics: Rayleigh is known zero-parameter physics, ON by default
        # (v3 and earlier ran without it -- that biased the <1.5 um slope);
        # the cloud deck is the ExoJax power-law retrieval cloud, OFF by default.
        "use_rayleigh": bool(params.get("use_rayleigh", True)),
        # line-broadening perturber: "air" (HITRAN terrestrial widths, the
        # validated default) or "h2he" (planetary H2/He blend; downloads
        # separate h2he/<db> line-list caches on first use, and exojax_rt
        # RAISES for a molecule with no H2/He coverage rather than silently
        # falling back)
        "broadening": str(params.get("broadening", "air")),
        # ExoJAX RT knobs (v15). rt_ptop_bar: the RT column top; above
        # VULCAN's chemistry top the topmost VMR/T are clamped constant
        # (standard transmission convention). Too low a top saturates strong
        # bands into a flat wall (W39b 4.2-5.2 um: ~4.8% of pixels saturated
        # at 1e-6 bar vs 0.1% at the 1e-8 default -- the sibling repo's
        # validation/top_pressure_ladder.py quantifies it). rt_integration:
        # exojax ArtTransPure chord-integration scheme. rt_dit_res: PreMODIT
        # broadening-grid spacing (1.0 = the validated default here, 0.2 =
        # exojax's own default; smaller = finer line wings, slower build).
        "rt_ptop_bar": float(f"{float(params.get('rt_ptop_bar', 1.0e-8)):.6e}"),
        "rt_integration": str(params.get("rt_integration", "simpson")),
        "rt_dit_res": round(float(params.get("rt_dit_res", 1.0)), 3),
        # opacity_mode (v28 transmission, v29 emission, v30/v31 ExoMolOP):
        # "exomolop" = correlated-k over the published k-tables, "lbl" = the
        # old direct sampling, kept only for the Mie deck. The sampled path
        # was measurably biased in both observables -- the decision and every
        # measured number live in the vulcan-forward README (Opacity
        # sections); the history is in notes.md (2026-08-15 entry).
        "opacity_mode": str(params.get("opacity_mode",
                                       default_opacity_mode(params))),
        # p_ref_bar (v26): the pressure at which rp_rjup and gs_cgs are taken to
        # apply. A catalogue radius comes from a white-light TRANSIT fit, so it
        # is the transit radius at roughly the terminator photosphere, NOT the
        # radius at the bottom of the RT grid where exojax wants it. Before v26
        # the literature pair was handed straight to exojax as its
        # radius_btm/gravity_btm at 7 bar, which stacked the whole 7 bar -> mbar
        # column on top of a radius that already was the photospheric one:
        # WASP-39 b came out at 26,853 ppm against a measured 21,381, with 1.42x
        # too much spectral contrast. 1e-3 bar is the validated default
        # (re-anchored median 21,299 ppm, 0.4% from the JWST ERS spectrum).
        "p_ref_bar": float(f"{float(params.get('p_ref_bar', 1.0e-3)):.6e}"),
        # chemistry + RT column bottom (v27; v32: structure-aware default --
        # see the P_BTM_* block at module top)
        "p_btm_bar": round(p_btm_bar, 4),
        "cloud_on": bool(params.get("cloud_on", False)),
        "log_kappa_cloud": round(float(params.get("log_kappa_cloud", -1.0)), 3),
        "alpha_cloud": round(float(params.get("alpha_cloud", 0.0)), 2),
        # Mie condensate deck (v16): "" = off. The three continuous knobs key
        # the cache only when a condensate is set (zeroed below otherwise).
        "mie_condensate": str(params.get("mie_condensate", "") or ""),
        "mie_log_rg": round(float(params.get("mie_log_rg", -5.0)), 3),
        "mie_sigmag": round(float(params.get("mie_sigmag", 2.0)), 3),
        "mie_log_mmr": round(float(params.get("mie_log_mmr", -6.0)), 3),
        # Detection-only condensation (v14): the certified S8 forward recipe.
        # The compatibility matrix below refuses it with ANY derivative.
        "use_condense": bool(params.get("use_condense", False)),
        # Boundary conditions / transport (v16): all default OFF = the
        # validated baseline; upstream VULCAN machinery via cfg_overrides.
        "use_settling": bool(params.get("use_settling", False)),
        "diff_esc": sorted(set(str(s) for s in (params.get("diff_esc") or []))),
        "top_flux": _canon_bc_entries(params.get("top_flux"), kind="top"),
        "bot_flux": _canon_bc_entries(params.get("bot_flux"), kind="bot"),
        "extra_mols": sorted(str(m) for m in (params.get("extra_mols") or [])),
        # Leave-one-out spectrum set (v32): which molecules get a removed-
        # molecule spectrum. None = every RT molecule (the detect default);
        # [] skips the whole block (the constrain goal reads none of them).
        # Canonicalized to fold order below, once the RT set is validated.
        "wo_mols": (None if params.get("wo_mols") is None
                    else [str(m) for m in params.get("wo_mols")]),
        "fisher_params": sorted(str(p) for p in (params.get("fisher_params") or [])),
        # Jacobian method: "fd" (certified central FD, default, valid
        # everywhere) or "ad" (one warm-started jvp per row, photo-on only;
        # see the module docstring for the per-row caveats).
        "jac_method": str(params.get("jac_method", "fd")),
        # --- PICASO provider / climate T-P identity (v18) ------------------
        # Content fingerprints of the installed picaso + its reference tables
        # ("" when inactive -- cache hygiene; filled by the matrix below, so a
        # picaso request without its data fails at the API, and any table
        # change self-invalidates every cached spectrum built on it).
        "chem_provider": provider,
        "network": network,
        "picaso_version": "",
        "picaso_chemgrid_sha1": "",
        "picaso_climate_sha1": "",
        # climate-mode knobs (validated + zeroed below unless
        # tp_mode="picaso_climate"; climate_rcb caveat in the constants block)
        "tint_cl": round(float(params.get("tint_cl", TINT_CL_DEFAULT)), 2),
        "rfacv": round(float(params.get("rfacv", 0.5)), 3),
        "tio_vo": bool(params.get("tio_vo", False)),
        "climate_rcb": int(params.get("climate_rcb", CLIMATE_RCB_DEFAULT)),
        "picaso_ck_node": "",
        "version": _VERSION,
    }
    if not 0.0 <= cp["sl_angle_deg"] <= 89.0:
        raise ValueError(f"sl_angle_deg={cp['sl_angle_deg']} outside [0, 89] deg")
    if not 0.0 < cp["f_diurnal"] <= 1.0:
        raise ValueError(f"f_diurnal={cp['f_diurnal']} outside (0, 1]")
    if cp["broadening"] not in ("air", "h2he"):
        raise ValueError(f"broadening={cp['broadening']!r} (choose 'air' or 'h2he')")
    if not 1.0e-9 <= cp["rt_ptop_bar"] <= 1.0e-6:
        raise ValueError(
            f"rt_ptop_bar={cp['rt_ptop_bar']:g} outside [1e-9, 1e-6] bar (the "
            "exercised RT-top range; 1e-8 is the validated default)")
    _art_pbtm = cp["p_btm_bar"] * ART_PBTM_FRACTION
    if not cp["rt_ptop_bar"] <= cp["p_ref_bar"] <= _art_pbtm:
        raise ValueError(
            f"p_ref_bar={cp['p_ref_bar']:g} must lie inside the RT grid "
            f"[{cp['rt_ptop_bar']:g}, {_art_pbtm:g}] bar. It is the pressure "
            "at which rp_rjup and gs_cgs are defined, so the grid has to "
            "cover it. 1e-3 bar is the validated default for a "
            "transit-derived radius.")
    if cp["opacity_mode"] not in ("exomolop", "lbl"):
        raise ValueError(
            f"opacity_mode={cp['opacity_mode']!r}: choose 'exomolop' "
            "(correlated-k over the published ExoMolOP high-temperature "
            "tables -- the default and the only one whose line data suits a "
            "hot hydrogen atmosphere) or 'lbl' (direct sampling, kept for "
            "the Mie deck and pre-v28 results). The interim 'ckd' mode was "
            "removed with forward 0.8.0 (v31).")
    if cp["opacity_mode"] == "exomolop" and cp["mie_condensate"]:
        raise ValueError(
            "a Mie condensate deck needs opacity_mode='lbl': Mie extinction "
            "has structure across a band, so folding it into a k-distribution "
            "built from line opacity would be wrong. Set opacity_mode='lbl' "
            "knowingly, and read the resolution warning it prints.")
    if cp["rt_integration"] not in ("simpson", "trapezoid"):
        raise ValueError(
            f"rt_integration={cp['rt_integration']!r}: exojax ArtTransPure "
            "supports 'simpson' (default) or 'trapezoid'")
    if not 0.1 <= cp["rt_dit_res"] <= 1.0:
        raise ValueError(
            f"rt_dit_res={cp['rt_dit_res']:g} outside [0.1, 1.0] (PreMODIT "
            "broadening-grid spacing; 1.0 = this tool's validated default, "
            "0.2 = exojax's own default)")
    if cp["opacity_mode"] == "exomolop":
        # Inert-knob normalization (cache hygiene, Kzz precedent): under
        # correlated-k the wavelength grid comes from the published k-tables
        # and PreMODIT never runs, so nu_pts and rt_dit_res cannot reach the
        # physics -- pin them to the defaults so editing either does not force
        # a recompute of a bit-identical spectrum. Both stay live in lbl mode.
        cp["nu_pts"] = NU_PTS_DEFAULT
        cp["rt_dit_res"] = 1.0
    if provider == "picaso":
        # provider-specific menu (NO SO2 anywhere in the picaso sets): the
        # detailed refusal with the sulfur explanation lives in the provider
        # matrix below; this generic gate just uses the right universe.
        from jwst_tool import picaso_chem as _pc0
        bad_mols = set(cp["extra_mols"]) - set(_pc0.PICASO_EXTRA_MOLECULES)
    else:
        bad_mols = set(cp["extra_mols"]) - set(EXTRA_MOLECULES)
    if bad_mols and provider == "picaso":
        from jwst_tool import picaso_chem as _pc0
        raise ValueError(
            f"extra_mols {sorted(bad_mols)} are not available under "
            f"chem_provider='picaso': it supplies {_pc0.PICASO_MOLECULES} + "
            f"optional {_pc0.PICASO_EXTRA_MOLECULES}. There is NO "
            "SO2/S2/S8/CS2 -- equilibrium sulfur sits in H2S/OCS; "
            "photochemical sulfur science needs chem_provider='vulcan'.")
    if bad_mols:
        raise ValueError(
            f"unknown RT molecule(s) {sorted(bad_mols)}. This tool ships opacity "
            f"for the always-on base set {MOLECULES} plus the opt-in extras "
            f"{EXTRA_MOLECULES}. To add another molecule you must extend the "
            "forward engine (a cross-repo change in the shared vulcan-forward engine): "
            "add an entry to the engine's molecule table (pass your own via "
            "profile['molecule_table'], or extend "
            "vulcan_forward.constants.MOLECULES) "
            "(HITRAN db id, molmass, VULCAN species name), make sure the SNCHO "
            "network actually solves that species, then list it here in "
            "forward.EXTRA_MOLECULES.")
    if network == "ncho":
        _s_req = sorted(set(cp["extra_mols"]) & _S_MOLECULES)
        if _s_req:
            raise ValueError(
                f"extra_mols {_s_req} are sulfur species and do not exist in "
                "the ncho network. Drop them, or keep network='sncho'. "
                "(Sulfur species are refused rather than dropped, so the "
                "model computed is always the model asked for.)")
    if cp["opacity_mode"] == "exomolop":
        # Fail HERE, not deep inside the RT build: ExoMolOP publishes no
        # k-table for these species, so a run selecting them under the
        # default mode would die minutes later in exomolop.load_tables.
        # (After the molecule-universe checks above, so an unknown or
        # provider-refused species keeps its more fundamental error.) The
        # frozenset is cross-checked against exomolop.available() by a
        # data-gated test, so it cannot rot silently. They stay in
        # EXTRA_MOLECULES because opacity_mode="lbl" CAN run them; the GUI
        # drops them from its options (v34) because it has no opacity_mode
        # control and defaults to "exomolop", so a picker entry would be a
        # dead end. (A Mie deck DERIVES "lbl", where they would run -- an
        # accepted narrowing of the GUI, not of this API. See app.py.)
        no_table = sorted(set(cp["extra_mols"]) & _NO_EXOMOLOP_TABLE)
        if no_table:
            raise ValueError(
                f"extra_mols {no_table} have no published ExoMolOP k-table, "
                "so they cannot run under opacity_mode='exomolop' (the "
                "default). Drop them, or set opacity_mode='lbl' knowingly "
                "(sampled line-by-line, measurably biased -- vulcan-forward "
                "README, Opacity sections).")
    # --- leave-one-out spectrum set (v32) -----------------------------------
    # After the molecule-universe checks, so the RT set is final. Canonical
    # form: a subset of active_molecules(cp) in fold order (deduped), which
    # keeps the cache key and the stored depth_wo rows in one order.
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
    # --- PICASO provider compatibility matrix (v18) -------------------------
    # Refuse EXPLICIT requests for physics equilibrium cannot model (a knob
    # the caller actually set would otherwise be silently dropped); normalize
    # the untouched defaults (inert-knob pattern -- the GUI hides these
    # widgets under picaso and always submits the normalized values).
    if needs_picaso:
        _fp = _picaso_fingerprint()
        cp["picaso_version"] = str(_fp["picaso_version"])
    if provider == "picaso":
        cp["picaso_chemgrid_sha1"] = str(_picaso_fingerprint()["chemgrid_sha1"])
        if cp["jac_method"] == "ad":
            raise ValueError(
                "jac_method='ad' is unavailable under chem_provider='picaso': "
                "the PICASO equilibrium tables are numpy/numba, not "
                "differentiable. Use jac_method='fd' -- the composition rows "
                "are symmetric two-cell interpolant secants with a one-sided "
                "kink gate (see README.md, PICASO engine section).")
        if cp["use_condense"]:
            raise ValueError(
                "use_condense is a VULCAN kinetics feature (the certified S8 "
                "rainout recipe); the PICASO equilibrium provider has no "
                "condensation channel. Turn it off or use "
                "chem_provider='vulcan'.")
        if cp["use_settling"] or cp["diff_esc"] or cp["top_flux"] or cp["bot_flux"]:
            raise ValueError(
                "boundary-condition / transport knobs (use_settling, "
                "diff_esc, top_flux, bot_flux) are kinetics features with no "
                "equilibrium counterpart: the PICASO provider refuses them "
                "rather than silently ignoring them. Clear them or use "
                "chem_provider='vulcan'.")
        if "lnKzz" in cp["fisher_params"]:
            raise ValueError(
                "lnKzz has no effect in equilibrium chemistry (no transport), "
                "so a Fisher row for it would be identically zero. The "
                "quench-approximation lnKzz row is a deferred feature -- see "
                "README.md, PICASO engine section. Drop lnKzz from fisher_params.")
        for _knob, _label in (("use_photo", "photochemistry"),
                              ("use_moldiff", "molecular diffusion"),
                              ("use_vm_mol", "upwind molecular-diffusion "
                                             "advection")):
            if _knob in params and bool(params[_knob]):
                raise ValueError(
                    f"{_knob}=True requests {_label}, which the PICASO "
                    "equilibrium provider cannot model (no kinetics, no "
                    "transport). Set it False, or use chem_provider='vulcan' "
                    "for disequilibrium physics.")
            cp[_knob] = False
        if "kzz_mode" in params and str(params["kzz_mode"]) != "const":
            raise ValueError(
                f"kzz_mode={params['kzz_mode']!r} requests a mixing profile, "
                "which equilibrium chemistry cannot consume (no transport; "
                "the quench machinery is deferred -- README.md, PICASO engine section). "
                "Leave kzz_mode unset or 'const'.")
        cp["kzz_mode"] = "const"
        cp["kzz_x"] = 1.0
        cp["kzz_const"] = 0.0          # inert sentinel: no transport at all
        cp["yconv_cri"] = YCONV_DEFAULT   # no iterative solver -> inert knob
        if not PICASO_CO_RANGE[0] <= cp["co_ratio"] <= PICASO_CO_RANGE[1]:
            raise ValueError(
                f"co_ratio={cp['co_ratio']:g} outside the Visscher "
                f"equilibrium grid span {list(PICASO_CO_RANGE)}: the PICASO "
                "provider is HARD-CAPPED at C/O 1.10 by its tables (VULCAN "
                "handles up to 2.0 structurally -- use "
                "chem_provider='vulcan' for C-rich atmospheres).")
        # (extra_mols is NOT re-checked here: the generic gate above already
        # differenced it against PICASO_EXTRA_MOLECULES under this provider
        # and raised, and nothing writes cp["extra_mols"] in between.)
    # --- climate T-P mode matrix (v18; both providers) ----------------------
    if tp_mode == "picaso_climate":
        from jwst_tool import picaso_chem as _pc
        from jwst_tool import picaso_env as _pe
        if cp["jac_method"] == "ad":
            raise ValueError(
                "jac_method='ad' is not certified with "
                "tp_mode='picaso_climate' (the climate T-P is a tabulated "
                "solver output; its Tint_cl row is a full-re-solve FD row). "
                "Use jac_method='fd'.")
        if not TINT_CL_RANGE[0] <= cp["tint_cl"] <= TINT_CL_RANGE[1]:
            raise ValueError(
                f"tint_cl={cp['tint_cl']:g} K outside {list(TINT_CL_RANGE)} "
                "(the exercised internal-temperature range).")
        if cp["rfacv"] not in RFACV_CHOICES:
            raise ValueError(
                f"rfacv={cp['rfacv']:g}: choose from {list(RFACV_CHOICES)} "
                "(0 = no irradiation, 0.5 = full redistribution, 1 = "
                "dayside-only).")
        if not CLIMATE_RCB_RANGE[0] <= cp["climate_rcb"] <= CLIMATE_RCB_RANGE[1]:
            raise ValueError(
                f"climate_rcb={cp['climate_rcb']} outside "
                f"{list(CLIMATE_RCB_RANGE)} (a layer index on the "
                f"{CLIMATE_N_LEVELS}-level climate grid).")
        # EXACT-CK-NODE composition (v18): the correlated-k tables are
        # per-node files with NO composition interpolation, so climate mode
        # accepts only compositions sitting exactly on a shipped node.
        _feh = float(np.log10(cp["met_x_solar"]))
        _feh_ok = min(abs(_feh - f) for f in _pc.FEH_NODES) <= 5.0e-4
        _co_ok = min(abs(cp["co_ratio"] - c) for c in _pc.CO_NODES) <= 5.0e-4
        _node = _pe.ck_node_string(cp["met_x_solar"], cp["co_ratio"])
        if not (_feh_ok and _co_ok) or _node not in _pc.CK_NODES_AVAILABLE:
            raise ValueError(
                f"tp_mode='picaso_climate' needs (met_x_solar, co_ratio) "
                f"exactly ON a shipped correlated-k node; got [M/H]="
                f"{_feh:+.3f}, C/O={cp['co_ratio']:g} -> {_node!r}. The CK "
                "tables carry no composition interpolation (off-node blending "
                "exists only for the chemistry under analytic T-P modes). "
                f"Metallicity nodes (dex): {list(_pc.FEH_NODES)}; C/O nodes: "
                f"{list(_pc.CO_NODES)}; extreme metallicities (+-1.5, +-2.0) "
                f"ship only C/O {list(_pc._EXTREME_CO)}.")
        cp["picaso_ck_node"] = _node
        cp["picaso_climate_sha1"] = str(
            _picaso_climate_fingerprint(_node, cp["tio_vo"]))
        if provider == "picaso" and "dlnCO" in cp["fisher_params"]:
            # v18.1 review finding 20: climate composition is exact-node
            # only, so the picaso C/O stencil ALWAYS straddles a table kink
            # -- the node-kink gate would fire mid-run after the expensive
            # climate + opacity builds (measured 1.52 at the 0.55 node).
            # Refuse at the API instead of shipping a poisoned default.
            raise ValueError(
                "a C/O (dlnCO) constraint row is unavailable under the "
                "PICASO engine in climate mode: climate composition sits "
                "exactly ON a chemistry-table node, where the one-sided "
                "table slopes disagree and no trustworthy derivative "
                "exists (the run would refuse mid-way; measured at the "
                "C/O = 0.55 node). Constrain C/O with the PICASO engine "
                "under an analytic T-P at a mid-cell value (default 0.50), "
                "or with the VULCAN engine (its own chemistry "
                "differentiates fine at any C/O).")
        if "Tint_cl" in cp["fisher_params"]:
            _h2 = 2.0 * FD_STEPS["Tint_cl"]
            if not (TINT_CL_RANGE[0] + _h2 <= cp["tint_cl"]
                    <= TINT_CL_RANGE[1] - _h2):
                raise ValueError(
                    f"tint_cl={cp['tint_cl']:g} K is within one FD stencil "
                    f"(2h = {_h2:g} K) of the range edge "
                    f"{list(TINT_CL_RANGE)}: the Tint_cl row would re-solve "
                    "the climate outside the exercised range. Move tint_cl "
                    "inward or drop Tint_cl from fisher_params.")
    else:
        cp["tint_cl"] = 0.0
        cp["rfacv"] = 0.0
        cp["tio_vo"] = False
        cp["climate_rcb"] = 0
    # Fisher parameter menu: chemistry + the tp_mode's T-P parameters (none
    # in file mode) + the cloud-deck parameters when the deck is in the model.
    allowed_fp = {"lnZ", "dlnCO", "lnKzz"} | set(TP_PARAM_NAMES[tp_mode])
    if provider == "picaso":
        allowed_fp -= {"lnKzz"}        # no transport in equilibrium (refused
        #                                with its own message above)
    if cp["cloud_on"]:
        allowed_fp |= set(CLOUD_FISHER_PARAMS)
    if cp["mie_condensate"]:
        allowed_fp |= set(MIE_FISHER_PARAMS)
    bad_fp = set(cp["fisher_params"]) - allowed_fp
    if bad_fp:
        _chem_menu = (["lnZ", "dlnCO"] if provider == "picaso"
                      else ["lnZ", "dlnCO", "lnKzz"])
        raise ValueError(
            f"unknown Fisher parameter(s) {sorted(bad_fp)} for tp_mode="
            f"{tp_mode!r}: choose from {_chem_menu} + "
            f"{TP_PARAM_NAMES[tp_mode]}"
            + (f" + {list(CLOUD_FISHER_PARAMS)}" if cp["cloud_on"] else "")
            + (f" + {list(MIE_FISHER_PARAMS)}" if cp["mie_condensate"] else "")
            + (". (tp_mode='file' has NO T-P Fisher rows by design; the "
               f"cloud parameters {list(CLOUD_FISHER_PARAMS)} require cloud_on; "
               f"the Mie parameters {list(MIE_FISHER_PARAMS)} require a "
               "mie_condensate.)"))
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
    # Composition FD stencils must stay inside the validated envelope: the
    # central stencil evaluates at +-2h in ln-space, so a row requested AT a
    # range edge would silently solve the chemistry outside it. T-P rows
    # window-check every stencil point and the Mie ranges are inset by > 2h
    # for the same reason.
    if cp["jac_method"] == "fd" and cp["fisher_params"]:
        # provider-dependent steps + envelopes (picaso stencils must stay on
        # its tables)
        _steps = PICASO_FD_STEPS if provider == "picaso" else FD_STEPS
        _met_rng = PICASO_MET_RANGE if provider == "picaso" else (0.1, 100.0)
        _co_rng = PICASO_CO_RANGE if provider == "picaso" else (0.1, 2.0)
        if "lnZ" in cp["fisher_params"]:
            _m = float(np.exp(2.0 * _steps["lnZ"]))
            if not _met_rng[0] * _m <= cp["met_x_solar"] <= _met_rng[1] / _m:
                raise ValueError(
                    f"met_x_solar={cp['met_x_solar']:g} is within one FD "
                    f"stencil (2h = {2.0 * _steps['lnZ']:g} in ln) of the "
                    f"validated range edge {list(_met_rng)}: the lnZ Fisher "
                    f"row would evaluate the chemistry outside the envelope. "
                    f"Keep met_x_solar in [{_met_rng[0] * _m:.3g}, "
                    f"{_met_rng[1] / _m:.3g}] for an lnZ row, or drop lnZ "
                    "from fisher_params.")
        if "dlnCO" in cp["fisher_params"]:
            _m = float(np.exp(2.0 * _steps["dlnCO"]))
            if not _co_rng[0] * _m <= cp["co_ratio"] <= _co_rng[1] / _m:
                raise ValueError(
                    f"co_ratio={cp['co_ratio']:g} is within one FD stencil "
                    f"(2h = {2.0 * _steps['dlnCO']:g} in ln) of the "
                    f"validated range edge {list(_co_rng)}: the dlnCO Fisher "
                    f"row would evaluate the chemistry outside the envelope. "
                    f"Keep co_ratio in [{_co_rng[0] * _m:.3g}, "
                    f"{_co_rng[1] / _m:.3g}] for a dlnCO row, or drop dlnCO "
                    "from fisher_params.")
    # --- condensation: detection-only -- refuse every derivative combo -----
    # (why: module docstring; the raises below carry the full user-facing
    # explanation, and the '91% wrong' wording is test-pinned)
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
                "condensed reservoir is frozen at a step-sequence-dependent "
                "transient (the state is not a reproducible function of the "
                "parameters) and the condensing-layer set switches "
                "discretely in temperature. The AD tangent through it is "
                "about 91% wrong (jvp-vs-FD relative error ~0.91 -- an "
                "order-unity failure, not a 9% mismatch), and finite "
                "differences of pinned transients are equally "
                "untrustworthy. Clear the Fisher parameter list (detection "
                "works), or turn condensation off. For aerosol opacity in a "
                "forecast use the differentiable ExoJax cloud deck "
                "(cloud_on) instead.")
        if not cp["use_photo"]:
            raise ValueError(
                "condensation (use_condense) requires photochemistry ON: a "
                "cold no-photo condensing column has no certifiable longdy "
                "steady state (well-mixed CO2 creeps toward equilibrium on "
                ">= 1e17 s -- the quench regime; upstream integrates those "
                "to a runtime cap, which this tool refuses to present as a "
                "converged spectrum). Enable photochemistry or turn "
                "condensation off.")
        if not cp["use_moldiff"]:
            raise ValueError(
                "condensation (use_condense) requires molecular diffusion "
                "(use_moldiff): the condensation growth term IS the species' "
                "molecular-diffusion coefficient, so with it off every "
                "condensation rate would silently be zero. Enable molecular "
                "diffusion, or turn condensation off.")
    # --- boundary conditions / transport (v16) -----------------------------
    bad_esc = set(cp["diff_esc"]) - set(DIFF_ESC_CHOICES)
    if bad_esc:
        raise ValueError(
            f"diff_esc species {sorted(bad_esc)} not supported: choose from "
            f"{list(DIFF_ESC_CHOICES)} (the light species the TOA "
            "diffusion-limited escape formula applies to, all present in the "
            "SNCHO network).")
    if cp["diff_esc"] and not cp["use_moldiff"]:
        raise ValueError(
            "diff_esc requires use_moldiff: the diffusion-limited escape flux "
            "is proportional to the top-of-atmosphere molecular-diffusion "
            "coefficient, so with moldiff off every escape flux would silently "
            "be zero. Enable molecular diffusion or clear the escape species.")
    if cp["use_settling"]:
        if not cp["use_moldiff"]:
            raise ValueError(
                "use_settling requires use_moldiff: the settling velocity "
                "enters through the molecular-diffusion operator, so with "
                "moldiff off settling would be silently inert. Enable "
                "molecular diffusion or turn settling off.")
        if cp["use_condense"]:
            raise ValueError(
                "use_settling cannot be combined with use_condense: the "
                "certified S8 condensation recipe (CONDEN_CFG) pins settling "
                "OFF -- the conden-window + fix-species convergence "
                "methodology was validated without gravitational settling, "
                "and enabling both would silently override one of them. "
                "Choose one.")
    if not cp["use_photo"]:            # photolysis knobs are inert without photo
        cp["sl_angle_deg"] = 0.0
        cp["f_diurnal"] = 1.0
        # The UV spectrum feeds only photolysis: normalize it photo-off so
        # identical physics never caches under different keys. orbit_au gets
        # the same treatment EXCEPT under picaso_climate, where the climate
        # solve consumes it for the stellar irradiation.
        cp["sflux"] = str(sysd["sflux"])
        if tp_mode != "picaso_climate":
            cp["orbit_au"] = round(float(sysd["orbit_au"]), 5)
    if not cp["use_moldiff"]:          # upwind vm_mol is inert without moldiff
        cp["use_vm_mol"] = False       # (engine gates use_vm on both); keep the
                                       # key from fragmenting the cache
    if not cp["cloud_on"]:             # cloud knobs are inert when the deck is off
        cp["log_kappa_cloud"] = 0.0
        cp["alpha_cloud"] = 0.0
    # --- Mie condensate deck (v16) -----------------------------------------
    if cp["mie_condensate"]:
        if cp["mie_condensate"] not in MIE_CONDENSATES:
            raise ValueError(
                f"mie_condensate={cp['mie_condensate']!r} not supported: "
                f"choose '' (off) or one of {list(MIE_CONDENSATES)} (the "
                "condensates exojax ships refractive indices + a substance "
                "density for; a miegrid is generated once per condensate with "
                "tools/generate_miegrid.py).")
        if not MIE_LOG_RG_RANGE[0] <= cp["mie_log_rg"] <= MIE_LOG_RG_RANGE[1]:
            raise ValueError(
                f"mie_log_rg={cp['mie_log_rg']} outside {MIE_LOG_RG_RANGE} "
                "(log10 mean radius in cm, kept inside the miegrid edges so the "
                "derivative is never a silently edge-clamped zero)")
        if not MIE_SIGMAG_RANGE[0] <= cp["mie_sigmag"] <= MIE_SIGMAG_RANGE[1]:
            raise ValueError(
                f"mie_sigmag={cp['mie_sigmag']} outside {MIE_SIGMAG_RANGE} "
                "(lognormal geometric std dev, inside the miegrid edges)")
        if not MIE_LOG_MMR_RANGE[0] <= cp["mie_log_mmr"] <= MIE_LOG_MMR_RANGE[1]:
            raise ValueError(
                f"mie_log_mmr={cp['mie_log_mmr']} outside {MIE_LOG_MMR_RANGE} "
                "(log10 condensate mass mixing ratio)")
        if science_mode == "emission":
            # The ExoJax emission solver (ArtEmisPure) is pure-absorption, so a
            # Mie deck's scattering extinction would be counted as thermal
            # absorption -- a conservative-scattering cloud would radiate like a
            # blackbody instead of zero, faking a thermal source in a cloudy
            # eclipse. Refuse upfront (loud) rather than return a wrong flux.
            raise ValueError(
                "mie_condensate is not supported with science_mode='emission': "
                "Mie scattering cannot be treated by the pure-absorption "
                "emission solver (it violates the conservative-scattering "
                "zero-emission limit). Use transmission, or the absorbing "
                "power-law cloud for emission.")
    else:                              # deck off: zero the knobs (cache hygiene)
        cp["mie_log_rg"] = cp["mie_sigmag"] = cp["mie_log_mmr"] = 0.0
    # --- science-mode hygiene + gating (v16 emission) ----------------------
    if science_mode == "emission":
        if not 3000.0 <= cp["star_teff"] <= 7000.0:
            raise ValueError(
                f"star_teff={cp['star_teff']:g} outside [3000, 7000] K (the "
                "range exercised against the PHOENIX grid for Fp/Fs)")
        if not 3.0 <= cp["star_logg"] <= 5.5:
            raise ValueError(f"star_logg={cp['star_logg']:g} outside [3.0, 5.5]")
        if not -2.5 <= cp["star_feh"] <= 0.5:
            raise ValueError(f"star_feh={cp['star_feh']:g} outside [-2.5, 0.5]")
        # Rayleigh scattering is transmission-only physics (the pure-
        # absorption emission solver must not count scattering as thermal
        # absorption -- engine contract), and the chord-integration scheme
        # only exists in transmission: normalize both so they cannot
        # fragment the emission cache.
        cp["use_rayleigh"] = False
        cp["rt_integration"] = "simpson"
    elif tp_mode != "picaso_climate":
        # transmission: the star normally lives only on the noise side -- but
        # under the climate T-P mode the climate solve CONSUMES the stellar
        # irradiation (star + rfacv set the T-P), so the star identity is
        # model physics there and must stay in the cache key (v18).
        cp["star_teff"] = cp["star_logg"] = cp["star_feh"] = 0.0
    # drop fields inert for the chosen modes so they don't fragment the cache
    if tp_mode != "guillot":
        cp["Tirr"] = cp["Tint"] = cp["log_kappa"] = cp["log_gamma"] = 0.0
    # --- Kzz profile mode (v16: const / Pfunc / JM16 / file) ----------------
    if cp["kzz_mode"] not in KZZ_MODES:
        raise ValueError(
            f"unknown kzz_mode {cp['kzz_mode']!r}: choose from "
            f"{list(KZZ_MODES)}. The WASP-39b GCM-scaled 'scale' mode was "
            "removed and stays removed -- profiles are explicit.")
    if not 0.01 <= cp["kzz_x"] <= 100.0:
        raise ValueError(
            f"kzz_x={cp['kzz_x']} outside [0.01, 100] (multiplicative scale "
            "applied on-graph to the whole Kzz profile)")
    if cp["kzz_mode"] == "const" and provider != "picaso":
        # (picaso normalized kzz_const to the 0.0 inert sentinel above --
        # equilibrium has no transport, so there is no value to validate)
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


def load_result(params: dict):
    """Cached spectrum dict or None.

    Always present: wl_um, depth, depth_wo (n_wo, n_nu), wo_mols (the
    leave-one-out set depth_wo rows align with -- a subset of mols in fold
    order, empty for a wo_mols=[] run), mols, ymix, p_bar,
    T, theta, theta_names, params_json, chem_provider, and (vulcan provider)
    the convergence certificate (conv_stages, conv_accept, conv_longdy,
    conv_gate -- EMPTY arrays under the picaso provider, which emits
    picaso_cert_json instead; climate mode adds climate_provenance_json).
    With Fisher requested: jac (n_par, n_nu), jac_names, jac_row_method,
    fd_h, fd_err, fd_kink, fd_grid_cell.
    """
    p = cache_path(params)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# Heavy path (script mode only below this line)
# ---------------------------------------------------------------------------

def _build_tp(cp: dict, gs_cgs: float):
    """(tp_eval, n_tp, tp_values, theta_names) for the chosen T-P mode.

    tp_eval(tp_params, p_bar) is pure JAX (differentiable) for the parametric
    modes. In file mode tp_eval is None: the engine's default temperature
    path is T = T_base + theta[3] with T_base = the tabulated profile the
    pre-loop re-gridded (atm_type="file"), and theta[3] is PINNED to 0 --
    there is no dT parameter and no T-P Fisher row (documented conditional).
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
    if mode in ("file", "picaso_climate"):
        # Tabulated-structure modes: theta keeps its 4th slot for the
        # engine's uniform-shift path, pinned to 0.0 and named "dT" so the
        # theta log stays self-describing. The climate mode's ONE structure
        # parameter (Tint_cl) is a full-re-solve REBUILD row handled in
        # run_model's Jacobian loop, never a theta direction.
        return None, 0, [0.0], CHEM_PARAM_NAMES + ["dT"]
    raise ValueError(f"unknown tp_mode {mode!r}")


def _make_progress(cp: dict, log):
    """Sequential stage tracker: emits "[fwd] PROG <frac> <label>" lines.

    The stage list MUST mirror run_model's actual stage order (same
    conditionals); weights are rough wall-clock seconds so the GUI bar moves
    honestly. advance() is called at the START of each stage.
    """
    _emis = cp.get("science_mode") == "emission"
    _pic = cp.get("chem_provider") == "picaso"
    stages = []
    if cp.get("tp_mode") == "picaso_climate":
        # first stage in run_model (a cache hit completes it instantly --
        # weights are rough wall-seconds, not promises)
        stages += [("PICASO climate solve (radiative-convective)", 80.0)]
    if _pic:
        stages += [("loading + blending equilibrium tables", 4.0)]
    else:
        stages += [("building chemistry model (compile + warm-up)", 45.0)]
    stages += [("building radiative transfer (opacities + CIA)",
                10.0 + 3.0 * len(cp["extra_mols"]))]
    if _emis:
        stages += [("emission model + stellar SED", 6.0)]
    stages += [("equilibrium state ready", 1.0) if _pic
               else ("solving photochemistry", 35.0)]
    # wo_mols (v32): the leave-one-out block is one engine batch, not one
    # stage per molecule -- transmission folds it into the full-spectrum
    # call. Cost ~ n(n+3)/2 overlap folds at ~3 s (measured: 42 s at n=5).
    _n_wo = len(cp["wo_mols"])
    _wo_w = 1.5 * _n_wo * (_n_wo + 3)
    if _emis:
        stages += [("full eclipse spectrum", 8.0)]
        if _n_wo:
            stages += [(f"removed-molecule spectra ({_n_wo} molecules)",
                        _wo_w)]
    elif _n_wo:
        stages += [(f"full + removed-molecule spectra ({_n_wo} molecules)",
                    8.0 + _wo_w)]
    else:
        stages += [("full transmission spectrum", 8.0)]
    # Jacobian rows: fd = 4 re-init build+solve cycles per composition row /
    # 4 cold solves per lnKzz/T-P row (picaso: 4 table re-evaluations, the RT
    # call dominates); Tint_cl = 4 full climate re-solves (+ chemistry);
    # cloud rows are RT-only (~seconds); ad = one warm jvp per row
    _ad = cp["jac_method"] == "ad"

    def _row_stage(n):
        if n in CLOUD_FISHER_PARAMS or n in MIE_FISHER_PARAMS:
            return (f"{'AD' if _ad else 'FD'} Jacobian d/d({n})", 8.0)
        if n == "Tint_cl":
            return ("FD Jacobian d/d(Tint_cl) (4 climate re-solves)",
                    340.0 if _pic else 1500.0)
        if _ad:
            return (f"AD Jacobian d/d({n})", 110.0)
        if _pic:
            return (f"FD Jacobian d/d({n})", 35.0)
        return (f"FD Jacobian d/d({n})",
                420.0 if n in FD_COMP_PARAMS else 260.0)

    stages += [_row_stage(n) for n in cp["fisher_params"]]
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
    """The RT-facing profile keys BOTH providers share (exojax_rt /
    build_emis_model read exactly these): pure extraction from the original
    _assemble_chem (v18 refactor) -- the vulcan profile dict is bit-identical
    to the pre-refactor one, pinned by the golden regression test."""
    profile = dict(config.WIDE)
    # numerical resolution (was the fidelity tier): the ExoJax RT layer count is
    # LOCKED equal to the chemistry layer count -- chemistry and RT share one grid.
    profile["nz"] = cp["nz"]
    profile["art_nlayer"] = cp["nz"]
    profile["nu_pts"] = cp["nu_pts"]
    profile["broadening"] = cp["broadening"]   # canonical (cache-keyed) knob
    # ExoJAX RT knobs (v15, canonical): the engine validates and ECHOES them
    # on the built rt namespace; run_model verifies the echo so an older
    # engine that ignores unknown profile keys can never return a spectrum
    # that differs from what the cache key claims.
    profile["art_ptop_bar"] = cp["rt_ptop_bar"]
    # RT bottom, just inside the chemistry bottom (interp_map refuses an ART
    # grid reaching below the chemistry). Emission is why this moves at all.
    profile["art_pbtm_bar"] = cp["p_btm_bar"] * ART_PBTM_FRACTION
    profile["rt_integration"] = cp["rt_integration"]
    profile["dit_grid_resolution"] = cp["rt_dit_res"]
    profile["p_ref_bar"] = cp["p_ref_bar"]
    profile["opacity_mode"] = cp["opacity_mode"]
    # Mie condensate deck (v16): the engine builds the OpaMie deck from a
    # pre-generated miegrid under DATA_DIR/exojax_mie and ECHOES mie_condensate
    # on the rt namespace; run_model verifies that echo below (an engine too old
    # to know the key would ignore it silently, caching a deck-less spectrum
    # under a Mie key). The absolute data dir is pinned so exojax never scatters
    # caches into the launch directory.
    if cp["mie_condensate"]:
        profile["mie_condensate"] = cp["mie_condensate"]
        profile["mie_data_dir"] = str(_ins.DATA_DIR / MIE_DATA_SUBDIR)
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


def _assemble_chem(cp: dict, log, clim=None):
    """Shared heavy-path assembly (run_model AND adjoint_diag): the resolved
    run profile with the structural composition pinned into cfg_overrides,
    the on-graph T-P hook, theta, and a chemistry-build factory. One code
    path -- the adjoint diagnostics must analyze exactly the model the
    forecasts ran. Imports the engine (import order load-bearing).

    ``clim``: the certified picaso_climate result (run_model passes it when
    tp_mode='picaso_climate'; when None it is fetched from the climate cache
    here so adjoint_diag keeps the one-assembly contract)."""
    # import order is load-bearing: vulcan_chem (env + x64) before jax/exojax
    from types import SimpleNamespace

    from jwst_tool import engine_config as config

    # Non-default kinetics network (v33): the engine freezes network/atom_list
    # at ITS first import (setdefault on the env vars), so the selection must
    # land before vulcan_chem arrives. In the run subprocess this IS the first
    # engine import; an in-process caller who already imported the engine on a
    # different network gets vulcan_chem's loud conflict raise, which is the
    # intended failure.
    _net_path, _net_atoms = NETWORKS[cp["network"]]
    if cp["network"] != "sncho":
        os.environ["VULCAN_JAX_NETWORK"] = _net_path
        os.environ["VULCAN_JAX_ATOM_LIST"] = _net_atoms
        log(f"[fwd] kinetics network: {cp['network']} ({_net_path}, "
            f"atoms {_net_atoms})")
    from vulcan_forward import vulcan_chem
    import jax

    # Persistent XLA compile cache (shared with the jax_paper adjoint
    # campaign's artifacts): saves the ~40 s runner warm-up on repeat runs
    # and is ESSENTIAL for adjoint_diag, whose step-VJP is a multi-hour
    # cold compile on CPU (measured 2026-07-16).
    jax.config.update("jax_compilation_cache_dir",
                      str(Path.home() / ".cache" / "jax_vulcan"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    if cp["tp_mode"] == "picaso_climate" and clim is None:
        from jwst_tool import picaso_climate
        clim = picaso_climate.get_or_run(cp, log)

    tp_eval, n_tp, tp_vals, theta_names = _build_tp(cp, cp["gs_cgs"])
    # Composition is set STRUCTURALLY in the cfg abundances (below), never as
    # a parameter perturbation, so lnZ and c_o are exactly 0; only lnKzz (an
    # on-graph multiplier) and the T-P parameters are live directions. The
    # vector form is what jax.jvp differentiates against.
    theta = np.asarray(
        vulcan_chem.ChemParams(lnZ=0.0, c_o=0.0, lnKzz=np.log(cp["kzz_x"]),
                               tp=tuple(tp_vals)).to_vector(),
        dtype=np.float64)
    log(f"[fwd] params {cp}")
    log(f"[fwd] theta {dict(zip(theta_names, np.round(theta, 4)))}")

    profile = _rt_profile_common(cp, config)
    profile["yconv_cri"] = cp["yconv_cri"]
    # exact-elemental abundance map (see vulcan_chem docstring);
    # reanchor_atom_ini is moot in this mode but kept for a masks-mode fallback
    profile["abundance_mode"] = "elemental"
    profile["co_mode"] = "fixed_O"
    profile["reanchor_atom_ini"] = True   # finite-Z steps must re-anchor atom totals
    # step-size cap, validated state-preserving (retrieval case.py): prevents the
    # adaptive-dt ballooning non-convergence at high Kzz the GUI inputs can reach
    profile["dt_max"] = 1.0e11
    rp_cm = profile["rp_cm"]
    ovr = {                              # chemistry side (applied pre-pre-loop)
        # VULCAN derives gravity as g = G*Mp/Rp^2; convert the tool's gs_cgs knob
        # to the equivalent planet mass at this radius.
        "Mp": cp["gs_cgs"] * rp_cm**2 / planets.G_CGS,
        # the cfg network must agree with the import-frozen one (the engine
        # fail-fasts on a mismatch); S_H stays in the cfg under ncho, which
        # is harmless with an S-free atom_list (the shipped HD189 precedent)
        **({"network": _net_path,
            "atom_list": _net_atoms.split(",")} if cp["network"] != "sncho"
           else {}),
        "Rp": rp_cm, "r_star": cp["rstar_rsun"],
        "orbit_radius": cp["orbit_au"],
        "sflux_file": f"atm/stellar_flux/{cp['sflux']}",
        "use_moldiff": cp["use_moldiff"],
        # pin the vm_mol scheme EXPLICITLY -- never inherit the upstream YAML
        # default; the two flags travel together (hybrid is how vm_mol runs)
        "use_vm_mol": cp["use_vm_mol"],
        "use_hybrid_vm_mol": cp["use_vm_mol"],
    }
    if cp["use_photo"]:                  # photolysis geometry/averaging knobs
        ovr["sl_angle"] = float(np.deg2rad(cp["sl_angle_deg"]))
        ovr["f_diurnal"] = cp["f_diurnal"]
    if cp["use_condense"]:
        # canonical_params already confirmed detection-only, photo on,
        # moldiff on; the engine rebuilds the condensation arrays on-graph
        # from the live T(P) per solve. Channel config = CONDEN_CFG.
        ovr.update(CONDEN_CFG)
    # Structural baseline. Guillot mode: isothermal structural baseline for
    # every planet (the on-graph tp_eval supplies the actual T(P); the
    # structural profile only sets the hydrostatic grid + EQ init). File
    # mode: the tabulated profile IS the structure (atm_type="file" re-grid,
    # tp_eval=None), content-hash verified -- never a silent substitution.
    if cp["tp_mode"] == "file":
        tp_path = _tp_file_from_cp(cp)   # sha-verified re-resolution
        ovr.update({"atm_type": "file", "atm_file": str(tp_path)})
        log(f"[fwd] planet {cp['planet']}: tabulated T-P structure from "
            f"{tp_path.name} (sha1 {cp['tp_file_sha1']}), UV = {cp['sflux']}")
    elif cp["tp_mode"] == "picaso_climate":
        # the certified climate profile IS the structure, through the exact
        # file-mode machinery (atm_type="file" re-grid; T_base = the profile;
        # tp_eval=None). One-way coupling: this chemistry never feeds back
        # into the climate opacity.
        tp_path = clim.atm_table
        ovr.update({"atm_type": "file", "atm_file": str(tp_path)})
        log(f"[fwd] planet {cp['planet']}: PICASO RCE climate T-P "
            f"(key {clim.key}, Tint={cp['tint_cl']:g} K, "
            f"rfacv={cp['rfacv']:g}, node {cp['picaso_ck_node']}), "
            f"UV = {cp['sflux']}")
    else:
        T_struct = cp["Tirr"] / np.sqrt(2.0)   # guillot: ~equilibrium T at f=0.25
        ovr.update({"atm_type": "isothermal", "Tiso": float(T_struct)})
        log(f"[fwd] planet {cp['planet']}: isothermal structural baseline "
            f"{T_struct:.0f} K, UV = {cp['sflux']}")
    # Kzz profile (v16 modes; lnKzz = theta[2] scales ANY of them on-graph).
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
    # Boundary conditions / transport (v16). canonical_params already refused
    # the conflicting combos (settling+conden, settling w/o moldiff), so these
    # never fight the CONDEN_CFG overrides above.
    if cp["use_settling"]:
        ovr["use_settling"] = True
        log("[fwd] BC: gravitational settling ON")
    if cp["diff_esc"]:
        ovr["diff_esc"] = list(cp["diff_esc"])
        log(f"[fwd] BC: diffusion-limited escape for {cp['diff_esc']}")
    if cp["top_flux"]:
        p_top = _write_bc_file("top", cp["top_flux"])
        ovr.update({"use_topflux": True, "top_BC_flux_file": str(p_top)})
        log(f"[fwd] BC: top flux rows {[e[0] for e in cp['top_flux']]} "
            f"-> {p_top.name}")
    if cp["bot_flux"]:
        p_bot = _write_bc_file("bot", cp["bot_flux"])
        ovr.update({"use_botflux": True, "bot_BC_flux_file": str(p_bot)})
        log(f"[fwd] BC: bottom flux rows {[e[0] for e in cp['bot_flux']]} "
            f"-> {p_bot.name}")
    # Chemistry-grid bottom. The cfg ships the shipped-planet span; a run that
    # needs a deeper column (emission -- see the P_BTM_* block at module top)
    # overrides P_b here rather than editing the YAML, so VULCAN-JAX and
    # vulcan-retrieval keep the span they were validated on.
    if abs(cp["p_btm_bar"] - P_BTM_FILE_BAR) > 1e-9:
        ovr["P_b"] = cp["p_btm_bar"] * 1.0e6
        log(f"[fwd] chemistry grid bottom {cp['p_btm_bar']:g} bar "
            f"(cfg default {P_BTM_FILE_BAR:g} bar), RT bottom "
            f"{cp['p_btm_bar'] * ART_PBTM_FRACTION:g} bar; "
            f"{cp['nz'] / np.log10(cp['p_btm_bar'] * 1e6 / CHEM_P_SPAN_DYN[0]):.0f} "
            "layers per decade")
    profile["cfg_overrides"] = ovr

    # CO_BASELINE must equal the loaded cfg's C_H/O_H -- refuse on drift (a
    # wrong-basis constant here mislabels every C/O display).
    import vulcan_jax as _vj
    _cfg_chk = _vj.load_config(profile.get("vulcan_cfg_name") or config.W39B_CFG_NAME)
    _co_cfg = float(_cfg_chk.C_H) / float(_cfg_chk.O_H)
    if abs(_co_cfg / CO_BASELINE - 1.0) > 1e-9:
        raise RuntimeError(
            f"forward.CO_BASELINE={CO_BASELINE:.5f} no longer matches the "
            f"network cfg's C_H/O_H={_co_cfg:.5f}: the C/O display baseline "
            "would be mislabeled. Update CO_BASELINE to the cfg value (and "
            "bump _VERSION).")
    # CHEM_P_SPAN_DYN must match the live cfg the same way: the light path's
    # bottom-coverage refusal keys on it.
    _span_cfg = (float(_cfg_chk.P_t), float(_cfg_chk.P_b))
    if any(abs(a / b - 1.0) > 1e-9
           for a, b in zip(_span_cfg, CHEM_P_SPAN_DYN)):
        raise RuntimeError(
            f"forward.CHEM_P_SPAN_DYN={CHEM_P_SPAN_DYN} no longer matches "
            f"the network cfg's (P_t, P_b)={_span_cfg}: the T-P table "
            "span validation would gate against the wrong grid. Update the "
            "constant (and bump _VERSION).")
    if cp["tp_mode"] in ("file", "picaso_climate"):
        # Log the conventional TOP clamp loudly (the bottom was hard-gated
        # at the API).
        _P_tab = _read_tp_table(tp_path, span=chem_p_span_dyn(cp))["P_dyn"]
        _dec = float(np.log10(_P_tab.min() / CHEM_P_SPAN_DYN[0]))
        if _dec > 0.0:
            log(f"[fwd] NOTE: T-P table top ({_P_tab.min():.3g} dyn/cm^2) "
                f"sits {_dec:.1f} decades below the chemistry-grid top "
                f"({CHEM_P_SPAN_DYN[0]:g}): the topmost tabulated T is held "
                "constant over that range (the standard upstream file-mode "
                "convention; the shipped profile does the same).")

    def _abundance_overrides(met_x_solar: float, co_ratio: float) -> dict:
        # Structural composition: scale the cfg metals together (He fixed),
        # then set carbon from the requested C/O at the scaled oxygen.
        # FastChem re-initializes at exactly this composition;
        # fastchem_met_scale follows for the non-network trace metals.
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
        t_b = time.time()
        chem_b = vulcan_chem.build_chem_model(prof, tp_eval=tp_eval,
                                              n_tp_params=n_tp)
        log(f"[fwd] chemistry model ({tag}) ready in {time.time()-t_b:.0f} s")
        return chem_b

    return SimpleNamespace(
        profile=profile, theta=theta, theta_names=theta_names,
        tp_eval=tp_eval, n_tp=n_tp, build_chem=_build_chem,
        abundance_overrides=_abundance_overrides, config=config)


def _assemble_chem_picaso(cp: dict, log, clim=None):
    """PICASO-provider counterpart of _assemble_chem: same namespace shape,
    with ``build_chem`` returning a table-equilibrium ADAPTER that satisfies
    the make_depth_fn contract (p_bar, sidx, species_masses, solved column)
    -- the RT, removed-molecule loop, cloud/Mie/lnR0 rows, and emission
    machinery run UNCHANGED on it.

    vulcan_chem is imported for its env + jax-x64 INIT side effects only
    (skipping it would silently run the shared RT in float32 -- the sibling
    sets x64 at import); no kinetics model is built. The provider pressure
    grid spans exactly the equilibrium tables' top (1e-6 bar) down to the
    chemistry bottom; above the table top the RT interpolation map constant-
    extends the top layer (the stated pressure policy, certificate-recorded).
    """
    # import order is load-bearing: vulcan_chem (env + x64) before jax/exojax
    from types import SimpleNamespace

    from jwst_tool import engine_config as config
    from vulcan_forward import vulcan_chem   # env + jax x64, and ChemParams
    import jax.numpy as jnp

    from jwst_tool import picaso_chem as pc

    if cp["tp_mode"] == "picaso_climate" and clim is None:
        from jwst_tool import picaso_climate
        clim = picaso_climate.get_or_run(cp, log)

    tp_eval, n_tp, tp_vals, theta_names = _build_tp(cp, cp["gs_cgs"])
    # Same named primitive as the VULCAN path; here the lnKzz slot is inert too
    # (kzz_x is normalized to 1 under the equilibrium provider).
    theta = np.asarray(
        vulcan_chem.ChemParams(lnZ=0.0, c_o=0.0, lnKzz=np.log(cp["kzz_x"]),
                               tp=tuple(tp_vals)).to_vector(),
        dtype=np.float64)
    log(f"[fwd] params {cp}")
    log(f"[fwd] theta {dict(zip(theta_names, np.round(theta, 4)))}")
    profile = _rt_profile_common(cp, config)

    # provider chemistry grid: table top (1e-6 bar) .. chemistry bottom
    p_bar = np.logspace(pc.TABLE_P_LOGBAR[0],
                        np.log10(chem_p_span_dyn(cp)[1] / 1.0e6), cp["nz"])
    if clim is not None:
        from jwst_tool import picaso_climate as _pcl
        T_base = _pcl.interp_T(clim, p_bar)
    else:
        T_base = None

    def _T_of(th):
        if T_base is not None:
            return np.asarray(T_base, dtype=np.float64)
        return np.asarray(tp_eval(jnp.asarray(np.asarray(th)[3:]),
                                  jnp.asarray(p_bar)), dtype=np.float64)

    def _solve_state(met, co, th, tag, T_prof=None):
        T_prof = _T_of(th) if T_prof is None else np.asarray(T_prof, float)
        t_b = time.time()
        state = pc.evaluate(met, co, T_prof, p_bar)
        c = state.cert
        log(f"[fwd] picaso equilibrium ({tag}) in {time.time() - t_b:.1f} s: "
            f"nodes {c['nodes'][0]}|{c['nodes'][1]} wf={c['wf']:.3f} "
            f"wc={c['wc']:.3f}, gas-sum min {c['gas_sum_min']:.4f} "
            f"({c['n_layers_below_warn']} layers < {c['gas_sum_warn']}), "
            f"suspect cells in span: {len(c['suspect_cells_in_span'])}")
        return state

    def _build_chem(extra_abun: dict | None = None, tag: str = "baseline"):
        if extra_abun:
            raise RuntimeError(
                "picaso adapter: composition perturbs through solve_at, "
                "never through cfg overrides")
        state0 = _solve_state(cp["met_x_solar"], cp["co_ratio"], theta, tag)
        sidx = {s: i for i, s in enumerate(state0.species)}
        for _need in ("H2", "He"):
            if _need not in sidx:
                raise RuntimeError(
                    f"picaso tables miss required species {_need}")
        # registry-token aliases where SNCHO and the tables disagree on a
        # species name (OCS: SNCHO "COS", table "OCS") -- the shared depth
        # path looks RT molecules up by the registry's vulcan token
        for _tok, _tab in pc.VULCAN_TO_TABLE.items():
            if _tab in sidx:
                sidx.setdefault(_tok, sidx[_tab])
        return SimpleNamespace(
            p_bar=p_bar, sidx=sidx,
            species_masses=np.asarray(state0.species_masses),
            T_base=T_base, y0=np.asarray(state0.y), cert=state0.cert,
            # (met, co, theta[, T_prof]) -> raw VMR matrix; the shared depth
            # path does the gas-mask + per-layer normalization
            solve_at=lambda met, co, th, tag="solve", T_prof=None: np.asarray(
                _solve_state(met, co, th, tag, T_prof=T_prof).y))

    return SimpleNamespace(
        profile=profile, theta=theta, theta_names=theta_names,
        tp_eval=tp_eval, n_tp=n_tp, build_chem=_build_chem,
        abundance_overrides=None, config=config, clim=clim)


def _check_t_window(tp_eval, theta, p_bar, log, T_base=None):
    """T-P validity on the chemistry grid: REJECT (never clip) out-of-window
    profiles. Returns the evaluated T(P) as a numpy array. In file mode
    (tp_eval None) the profile IS the structure: pass T_base (chem.T_base,
    the pre-loop's re-grid of the tabulated file)."""
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
            "not clipped (opacity tables end there).")
    log(f"[fwd] T-P in window: [{tmin:.0f}, {tmax:.0f}] K")
    return T_check


def run_model(params: dict, log=print) -> Path:
    cp = canonical_params(params)
    # Uploaded T-P tables become a CONTENT-ADDRESSED copy under
    # <output>/uploads/<sha1>.txt before anything heavy runs, so later
    # re-resolution from the canonical params alone (adjoint_diag, cache
    # inspection) always finds the exact bytes the key was computed from.
    if cp["tp_mode"] == "file" and cp["tp_file"] == TP_FILE_UPLOAD:
        src_path, sha1 = _resolve_tp_file(params)
        dst = _uploads_dir() / f"{sha1}.txt"
        if not dst.exists():
            _uploads_dir().mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src_path.read_bytes())
            log(f"[fwd] uploaded T-P table archived -> {dst}")
    advance, finish = _make_progress(cp, log)
    _is_picaso = cp["chem_provider"] == "picaso"
    clim = None
    if cp["tp_mode"] == "picaso_climate":
        # climate FIRST: the certified RCE T-P is an input to EITHER
        # provider's chemistry (cache-shared between them; picaso_climate
        # refuses anything uncertified)
        advance()
        from jwst_tool import picaso_climate
        clim = picaso_climate.get_or_run(cp, log)
    A = (_assemble_chem_picaso(cp, log, clim=clim) if _is_picaso
         else _assemble_chem(cp, log, clim=clim))
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
    if _is_picaso:
        log("[fwd] building PICASO equilibrium state (blended Visscher "
            "tables) ...")
    else:
        log("[fwd] building chemistry model (VULCAN-JAX warm-up ~40 s) ...")
    chem = _build_chem()

    # BC flux species must exist in the solved network: the upstream
    # read_bc_flux SILENTLY skips unknown tokens, which would turn a typo'd
    # boundary condition into a no-op. Loud rule: verify against the built
    # network and refuse.
    _bc_sp = [e[0] for e in cp["top_flux"]] + [e[0] for e in cp["bot_flux"]]
    _bad_sp = sorted(set(s for s in _bc_sp if s not in chem.sidx))
    if _bad_sp:
        raise RuntimeError(
            f"boundary-condition species {_bad_sp} not in the solved SNCHO "
            "network (the upstream BC reader would silently ignore them -- "
            "the run would quietly compute WITHOUT your boundary condition). "
            "Check spelling/capitalization (e.g. 'H2O', 'SO2', 'CH4').")

    T_check = _check_t_window(tp_eval, theta, chem.p_bar, log,
                              T_base=getattr(chem, "T_base", None))

    t0 = time.time()
    advance()
    log("[fwd] building ExoJax RT (opacities + CIA) ...")
    rt = exojax_rt.build_rt_model(profile)
    log(f"[fwd] RT ready in {time.time()-t0:.0f} s")
    # Echo check on the v15 RT knobs: an engine too old to know these
    # profile keys ignores them silently -- refuse rather than cache a
    # spectrum under a key describing physics the engine did not apply.
    _echo = {"art_ptop_bar": cp["rt_ptop_bar"],
             "art_pbtm_bar": profile["art_pbtm_bar"],
             "rt_integration": cp["rt_integration"],
             "dit_grid_resolution": cp["rt_dit_res"],
             "p_ref_bar": cp["p_ref_bar"],
             "opacity_mode": cp["opacity_mode"]}
    for k, want in _echo.items():
        got = getattr(rt, k, None)
        if got != want:
            raise RuntimeError(
                f"RT engine did not honor {k}={want!r} (echoed {got!r}). "
                "The installed vulcan-forward engine predates the "
                "profile-overridable RT knobs -- upgrade to >= 0.10.1.")
    # Mie deck echo (v16): the engine echoes mie_condensate ("" when no deck).
    # A mismatch means the engine ignored the profile key (too old to know it),
    # so the spectrum would be cached under a Mie key without the Mie opacity.
    _mie_echo = getattr(rt, "mie_condensate", None)
    if _mie_echo != cp["mie_condensate"]:
        raise RuntimeError(
            f"RT engine did not honor mie_condensate="
            f"{cp['mie_condensate']!r} (echoed {_mie_echo!r}). The installed "
            "vulcan-forward predates the Mie cloud deck -- upgrade to "
            ">= 0.11.0.")

    # --- emission mode (v16): day-side model + stellar SED ------------------
    emis, fs_j = None, None
    # 1 / R_star^2 only. The PLANET radius for the eclipse prefactor comes from
    # emis.emission_radius(T, mmw) -- the radius at the emission anchor
    # (~0.1 bar), NOT the catalogue transit radius (~1 mbar). Pairing the
    # transit radius with the emission column's gravity implied a GM 8.9% off
    # on WASP-39 b and inflated every eclipse depth by the same factor.
    _rstar_cm_sq = (cp["rstar_rsun"] * planets.R_SUN_CM) ** 2
    if cp["science_mode"] == "emission":
        advance()
        log("[fwd] building emission model + stellar SED ...")
        if not hasattr(exojax_rt, "build_emis_model"):
            raise RuntimeError(
                "the installed vulcan-forward engine has no "
                "build_emis_model: emission mode needs the >= 0.11 sibling. "
                "Upgrade vulcan-forward.")
        emis = exojax_rt.build_emis_model(rt, profile)
        if not hasattr(emis, "tau_bottom"):
            raise RuntimeError(
                "the installed vulcan-retrieval engine predates the emission "
                "tau_bottom diagnostic (>= 0.11): without it an optically-"
                "thin RT bottom silently underestimates the day-side flux. "
                "Upgrade vulcan-forward.")
        from jwst_tool import stellar as stellar_mod
        fs_j = jnp.asarray(stellar_mod.phoenix_surface_flux(
            rt.nu_grid, cp["star_teff"], cp["star_logg"], cp["star_feh"],
            log=log))

    p_art_j = jnp.asarray(rt.p_art_bar)

    def _t_art_const_from(chem_b):
        """Tabulated-mode RT temperature: the build's T_base interpolated in
        ln P onto the ART grid, constant-extended at the edges (the standard
        clamp above the chemistry top -- same convention the VMR
        interpolation uses). Also used by the Tint_cl rebuild rows, whose
        perturbed builds carry their own T_base."""
        _pb = np.asarray(chem_b.p_bar)
        _Tb = np.asarray(chem_b.T_base, dtype=np.float64)
        _order = np.argsort(_pb)
        return jnp.asarray(np.interp(
            np.log(np.asarray(rt.p_art_bar)),
            np.log(_pb[_order]), _Tb[_order]))

    if tp_eval is None:
        # tabulated modes (file / picaso_climate): ONE fixed profile for
        # chemistry and RT.
        _T_art_const = _t_art_const_from(chem)

        def art_T(th):
            return _T_art_const
    else:
        def art_T(th):
            return tp_eval(th[3:], p_art_j)

    # ExoJax power-law retrieval cloud [log10 kappac0 (cm^2/g at 3.5 um), alphac]:
    # the BASELINE deck. Since v16 it can be marginalized in the Fisher forecast
    # via the cloud= override in depth_fn (RT-only Jacobian rows); this vector is
    # the point the rows differentiate around (None when the deck is off).
    cloud_vec = (jnp.asarray([cp["log_kappa_cloud"], cp["alpha_cloud"]])
                 if cp["cloud_on"] else None)
    # Mie condensate deck [log10 rg (cm), sigmag, log10 MMR]: same pattern -- the
    # baseline vector the mie= Fisher rows differentiate around (None when off).
    # Independent of the power-law deck; the engine sums both if both are set.
    mie_vec = (jnp.asarray([cp["mie_log_rg"], cp["mie_sigmag"],
                            cp["mie_log_mmr"]])
               if cp["mie_condensate"] else None)

    def make_depth_fn(chem_b, T_art_override=None):
        """Depth function bound to ONE chemistry build: the interpolation map
        follows that build's hydrostatic grid (composition moves the mean
        molecular weight and hence the pressure grid between the FD re-init
        builds, so the map is never shared across builds).

        ``T_art_override`` (v18): a fixed ART-grid temperature replacing
        art_T(th) -- the Tint_cl rebuild rows bind perturbed-climate builds
        to their OWN temperature (the outer art_T closes over the BASELINE
        table in tabulated modes and would silently mix baseline RT
        temperature with perturbed chemistry).

        science_mode picks the observable behind the SAME signature:
        transmission -> transit depth (Rp(lambda)/Rstar)^2; emission ->
        eclipse depth (Fp/Fs) * (Rp/Rs)^2 * e^{2 lnR0}. Every downstream
        consumer (removed-molecule spectra, FD/AD Jacobian rows, cloud rows)
        is observable-agnostic through this function."""
        to_art_b = interp_map.make_to_art(chem_b.p_bar, rt.p_art_bar)
        mol_cols = {k: chem_b.sidx[config.MOLECULES[k]["vulcan"]]
                    for k in rt.molecules}
        h2_b, he_b = chem_b.sidx["H2"], chem_b.sidx["He"]
        # GAS-phase normalization (v17): condensed-phase reservoir species
        # (the network's *_l_s columns, e.g. S8_l_s) are particles, not gas --
        # counting them diluted every gas VMR and inflated the RT mean
        # molecular weight as if the condensate were vapor (~0.5% of mmw at
        # 10x solar with full sulfur rainout, growing with metallicity).
        # Conden-off solves never populate *_l_s, so those spectra are
        # bit-identical; the condensate's aerosol OPACITY stays deliberately
        # excluded (use the cloud/Mie decks for that -- module docstring).
        _gas = np.ones(int(np.asarray(chem_b.species_masses).size))
        for _s, _i in chem_b.sidx.items():
            if _s.endswith("_l_s"):
                _gas[int(_i)] = 0.0
        gas_mask = jnp.asarray(_gas)

        def _art_profiles(y, th):
            y_gas = y * gas_mask[None, :]
            ymix = y_gas / jnp.sum(y_gas, axis=1, keepdims=True)
            T_art = art_T(th) if T_art_override is None else T_art_override
            mmw_art = to_art_b(ymix @ chem_b.species_masses)
            vmr = {k: to_art_b(ymix[:, c]) for k, c in mol_cols.items()}
            return (vmr, to_art_b(ymix[:, h2_b]), T_art, mmw_art,
                    to_art_b(ymix[:, he_b]))

        def depth_fn(y, th, lnR0=0.0, cloud=None, mie=None, wo_mols=None):
            # cloud=/mie=None -> the baseline decks (cloud_vec/mie_vec; None
            # when that deck is off). An explicit vector overrides it -- the
            # v16 cloud and Mie Fisher rows differentiate the depth through
            # these two arguments (one deck at a time).
            # wo_mols (v32, transmission only): a list of molecules returns
            # (depth, depth_wo) from the engine's leave-one-out batch instead
            # of one solve per removed molecule; emission gets its batch via
            # emis.emission_flux_tau in run_model (the tau-bottom gate needs
            # the same optical depth, so the batch lives there).
            vmr, vmr_h2, T_art, mmw_art, vmr_he = _art_profiles(y, th)
            cl = cloud_vec if cloud is None else cloud
            mi = mie_vec if mie is None else mie
            if emis is not None:
                if wo_mols is not None:
                    raise ValueError(
                        "depth_fn(wo_mols=...) is transmission-only; emission "
                        "removed-molecule spectra come from "
                        "emis.emission_flux_tau (run_model owns that call).")
                fp = emis.emission_flux(vmr, vmr_h2, T_art, mmw_art,
                                        vmr_he=vmr_he, cloud=cl, mie=mi)
                rp_em = emis.emission_radius(T_art, mmw_art)
                return (fp / fs_j) * (rp_em ** 2 / _rstar_cm_sq) * jnp.exp(
                    2.0 * jnp.asarray(lnR0))
            return rt.transmission_depth_r(
                vmr, vmr_h2, T_art, mmw_art,
                jnp.asarray(lnR0), vmr_he=vmr_he, cloud=cl, mie=mi,
                wo_mols=wo_mols)
        depth_fn._art_profiles = _art_profiles   # reused by the tau-bottom gate
        return depth_fn

    depth_from_y = make_depth_fn(chem)

    # --- chemistry: certified cold solves (v13: no warm continuation) --------
    t0 = time.time()
    th0 = jnp.asarray(theta)
    conv_cert = []   # (stage, accept_count, longdy) for every PASSED gate
    # Certification is the runner's own CANONICAL gate (ConvDiag.conv_normal
    # AND longdy < yconv_min), recomputed at the exit state. longdy alone
    # accepted budget-exhausted photo-on solves with drifting UV flux;
    # accept_count alone is weaker still. Never loosen this.
    picaso_cert = None
    if _is_picaso:
        # equilibrium provider: the state was solved (and its certificate
        # gates -- gas-sum floor, table span -- enforced) inside the build;
        # no iterative solver, so no conv_* certificate is emitted.
        advance()
        log("[fwd] PICASO equilibrium state (certificate enforced at "
            "build) ...")
        y_np = np.asarray(chem.y0, dtype=np.float64)
        y_sol = jnp.asarray(y_np)
        picaso_cert = dict(chem.cert)
        _check_converged = None
        if not np.all(np.isfinite(y_np)):
            raise RuntimeError(
                "picaso equilibrium state returned non-finite abundances -- "
                "parameter set outside the modelable range")
    else:
        import inspect as _inspect
        if "return_conv_diag" not in _inspect.signature(
                chem.converged_y).parameters:
            raise RuntimeError(
                "the sibling forward engine's converged_y() does not support "
                "return_conv_diag (ConvDiag canonical certification): "
                "vulcan-retrieval is too old for this tool version. Upgrade the "
                "sibling install -- an uncertifiable solve is never presented "
                "as converged.")

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

        # Single certified cold solve, always: composition is baked into the
        # build (structural), so there is no composition continuation and no
        # stage 2.
        advance()
        log("[fwd] solving photochemistry (cold, certified) ...")
        y_sol, _cdiag = chem.converged_y(th0, return_conv_diag=True)
        _check_converged(_cdiag, "baseline solve")
        y_np = np.asarray(y_sol)
        if not np.all(np.isfinite(y_np)):
            raise RuntimeError(
                "chemistry solve returned non-finite abundances -- "
                "parameter set outside the modelable range")
        log(f"[fwd] chemistry solved in {time.time()-t0:.0f} s total")

    # Emission bottom-boundary certification. Since engine v0.5.0 the solver
    # DOES carry an interior source term (exojax ibased_linsap), so a
    # see-through window is no longer silently zero -- but the term is a
    # blackbody at the extrapolated bottom-boundary temperature, i.e. an
    # ISOTHERMAL-INTERIOR ASSUMPTION standing in for everything below the grid.
    # A column that leans on it is reporting an assumption, not a model. Keep
    # refusing the thin case: the fix is a deeper column (p_btm_bar), which is
    # cheap, not a wider tolerance.
    emis_tau_min = float("nan")
    _depth_norm_em0 = float("nan")
    wo_list = list(cp["wo_mols"])
    depth_wo = None
    emis_tau_min_wo = np.full(len(wo_list), np.nan)
    emis_thin_wo = np.full(len(wo_list), np.nan)
    if emis is not None:
        t0 = time.time()
        advance()                    # full eclipse spectrum
        _prof0 = depth_from_y._art_profiles(y_sol, th0)
        # the SAME (R_p/R_star)^2 the baseline depth was built with, so the
        # stored Fp inverts the stored depth exactly. Recomputing it from the
        # catalogue radius would reintroduce the transit-vs-emission radius
        # mismatch in the exported flux only.
        _depth_norm_em0 = float(
            np.asarray(emis.emission_radius(_prof0[2], _prof0[3])) ** 2
            / _rstar_cm_sq)
        log(f"[fwd] emission anchor: R_p = "
            f"{_depth_norm_em0 ** 0.5 * cp['rstar_rsun'] * planets.R_SUN_CM / planets.R_JUP_CM:.4f} "
            f"R_Jup at {getattr(emis, 'p_ref_emission_bar', float('nan')):g} "
            f"bar (catalogue transit radius {cp['rp_rjup']:.4f} R_Jup at "
            f"{cp['p_ref_bar']:g} bar)")
        # ONE optical-depth build feeds the flux, the tau-bottom gate, and the
        # stored depth (emission_flux_tau is bitwise the separate calls; the
        # old path built the same dtau three times)
        _fp_j, _tau_j = emis.emission_flux_tau(*_prof0, cloud=cloud_vec,
                                               mie=mie_vec)
        _tau_b = np.asarray(_tau_j)
        emis_tau_min = float(_tau_b.min())
        _wl_thin = float(rt.wl_um[int(np.argmin(_tau_b))])
        # FLUX-WEIGHTED, not min() over the band. The column being transparent
        # somewhere only matters to the extent the planet emits there. Measured
        # on HD 189733 b at the shipped 7.6 bar bottom: min tau 96.7 at 2-3 um,
        # 30.5 across NIRSpec, 140 across MIRI LRS, 370 beyond 12 um, and one
        # 60 nm notch at 1.02-1.08 um below 3 -- the extreme blue edge, where
        # the modeled opacity set runs out (no H- continuum, no Na/K wings, no
        # TiO/VO). A min() let that notch veto the whole run, and the "obvious"
        # fix of deepening the column would have bought opacity from pressure
        # that the real planet does not get, while hiding the actual limitation.
        _fp = np.asarray(_fp_j)
        emis_thin_frac = thin_flux_fraction(_tau_b, _fp)
        _report = _tau_bottom_breakdown(rt.wl_um, _tau_b, flux=_fp)
        if emis_thin_frac > EMIS_THIN_FLUX_FRAC:
            raise RuntimeError(
                f"emission unreliable: {100.0 * emis_thin_frac:.1f}% of this "
                f"planet's emitted flux comes from wavelengths where the RT "
                f"column bottom ({emis.art_pbtm_bar:g} bar) is optically thin "
                f"(tau < {EMIS_TAU_THIN:g}), above the "
                f"{100.0 * EMIS_THIN_FLUX_FRAC:g}% tolerance. The flux there is "
                "set by the interior source term, a blackbody at the "
                "extrapolated bottom-layer temperature -- an assumption about "
                "everything below the column, not a model of it.\n\n"
                + _report +
                "\n\nIf the thin part is at the SHORT-wavelength edge, the "
                "cause is missing opacity rather than a shallow column: below "
                "~2 um the modeled sources (molecular lines plus H2 "
                "collision-induced absorption) thin out, and the continuum "
                "that fills that window in a real hot atmosphere (H- "
                "bound-free and free-free, the Na and K wings, TiO/VO) is not "
                "modeled here. Deepening p_btm_bar would hide that, not fix "
                "it. If the thin part is spread across the band, the column "
                f"really is too shallow and p_btm_bar (now "
                f"{cp['p_btm_bar']:g} bar) is the setting to raise.")
        log("[fwd] " + _report.replace("\n", "\n[fwd] "))
        if emis_thin_frac > 0.0:
            log(f"[fwd] NOTE: {100.0 * emis_thin_frac:.3f}% of the emitted "
                f"flux comes from below tau = {EMIS_TAU_THIN:g} (min "
                f"{emis_tau_min:.2f} at {_wl_thin:.2f} um). Under the "
                f"{100.0 * EMIS_THIN_FLUX_FRAC:g}% tolerance, so the run "
                "proceeds; treat those wavelengths as unmodeled.")

        # the stored depth via the exact expression depth_fn uses (bitwise);
        # the flux is already in hand, so no separate depth solve remains
        _rp_em = emis.emission_radius(_prof0[2], _prof0[3])
        depth = np.asarray((_fp_j / fs_j) * (_rp_em ** 2 / _rstar_cm_sq)
                           * jnp.exp(2.0 * jnp.asarray(0.0)))
        log(f"[fwd] full spectrum in {time.time()-t0:.0f} s")

        depth_wo = np.zeros((len(wo_list), depth.shape[0]))
        if wo_list:
            t0 = time.time()
            advance()
            # ONE engine batch: each removed-molecule optical depth is built
            # once and feeds the spectrum AND the tau-bottom certification,
            # which must cover EVERY emission spectrum the results consume --
            # zeroing a dominant absorber can open see-through windows that
            # inflate the full-minus-removed contrast.
            _, _, _fp_wo, _tau_wo = emis.emission_flux_tau(
                *_prof0, cloud=cloud_vec, mie=mie_vec, wo_mols=wo_list)
            for i, mol in enumerate(wo_list):
                depth_wo[i] = np.asarray(
                    (_fp_wo[i] / fs_j) * (_rp_em ** 2 / _rstar_cm_sq)
                    * jnp.exp(2.0 * jnp.asarray(0.0)))
                _tau_i = np.asarray(_tau_wo[i])
                emis_tau_min_wo[i] = float(_tau_i.min())
                _wl_i = float(rt.wl_um[int(np.argmin(_tau_i))])
                # flux-weighted, same reasoning as the baseline gate above:
                # the blue-edge notch must not refuse every molecule in turn
                emis_thin_wo[i] = thin_flux_fraction(_tau_i,
                                                     np.asarray(_fp_wo[i]))
                if emis_thin_wo[i] > EMIS_THIN_FLUX_FRAC:
                    # Per-molecule, never whole-run: a thin bottom with THIS
                    # molecule removed makes only THIS molecule's detection
                    # unreliable (detect refuses that target at scoring time);
                    # the baseline spectrum and other molecules stay usable.
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
            log(f"[fwd] {len(wo_list)} removed-molecule spectra in "
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

    # Fisher Jacobian: certified FD (default) / warm-jvp AD (opt-in) -- see
    # the FD_STEPS block at module top; per-row method in jac_row_method.
    jac_names = list(cp["fisher_params"])
    jac = np.zeros((len(jac_names) + 1, depth.shape[0])) if jac_names else None
    fd_h, fd_err, row_method = [], [], []
    # per-row node-kink provenance (picaso composition rows only; NaN / ""
    # elsewhere): the one-sided-secant disagreement and the bracketing cells
    fd_kink, fd_grid_cell = [], []
    if jac_names:
        def _certified_depth(chem_b, th, stage):
            y_b, diag_b = chem_b.converged_y(jnp.asarray(th),
                                             return_conv_diag=True)
            _check_converged(diag_b, stage)
            return np.asarray(make_depth_fn(chem_b)(y_b, jnp.asarray(th)))

        def _picaso_comp_depth(met, co, tag):
            # composition point on the SAME adapter/grid (the provider grid
            # is fixed -- these rows are fixed-grid two-cell secants)
            y_b = chem.solve_at(met, co, theta, tag)
            return np.asarray(depth_from_y(jnp.asarray(y_b), th0))

        def _picaso_theta_depth(th_s, tag):
            y_b = chem.solve_at(cp["met_x_solar"], cp["co_ratio"], th_s, tag)
            return np.asarray(depth_from_y(jnp.asarray(y_b),
                                           jnp.asarray(th_s)))

        def _fd_row(name, d_p1, d_m1, d_p2, d_m2, h):
            j1 = (d_p1 - d_m1) / (2.0 * h)
            j2 = (d_p2 - d_m2) / (4.0 * h)
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
            return (4.0 * j1 - j2) / 3.0, err   # Richardson, O(h^4)

        def _ad_theta_depth(th):
            # warm continuation from the converged column: the primal is a
            # no-op re-converge, the jvp is the validated steady-state
            # tangent (photo ON -- gated in canonical_params)
            y_w = chem.converged_y(th, warm_y=y_sol, lnZ_ref=0.0, c_o_ref=0.0)
            return depth_from_y(y_w, th)

        if cp["jac_method"] == "ad" and "dlnCO" in jac_names:
            # Refuse the AD dlnCO row within one FD stencil of O-exhaustion
            # (see CO_BZ_MIN_AD at module top). A missing engine attribute
            # means the check cannot run: refuse, never pass silently.
            _bz = getattr(chem, "co_bz_bound", None)
            if _bz is None:
                raise RuntimeError(
                    "the sibling forward engine does not expose co_bz_bound "
                    "(the fixed-O direction's oxygen-reservoir bound): the "
                    "AD dlnCO row cannot be certified. Upgrade "
                    "vulcan-retrieval or use jac_method='fd'.")
            if float(_bz) <= CO_BZ_MIN_AD:
                raise RuntimeError(
                    "AD Jacobian for dlnCO refused at this composition: the "
                    f"fixed-O differential direction's oxygen-reservoir "
                    f"bound b_z = {float(_bz):.3g} <= {CO_BZ_MIN_AD:g} "
                    f"(C-rich composition, C/O = {cp['co_ratio']:g}: O-only "
                    "carriers are within one FD stencil of exhaustion, so "
                    "the tangent direction is ill-conditioned). Use "
                    "jac_method='fd' -- the certified FD row re-initializes "
                    "the chemistry and is valid at any composition.")

        if cp["jac_method"] == "ad":
            # Certify the AD rows' shared warm primal: one cheap warm solve
            # verifies the linearization point actually certifies (asserting
            # it without checking let stall exits pass silently).
            _, _diag_w = chem.converged_y(
                th0, warm_y=y_sol, lnZ_ref=0.0, c_o_ref=0.0,
                return_conv_diag=True)
            _check_converged(_diag_w, "AD warm re-converge (primal)")

        def _rt_deck_row(name, base_vec, idx, kwarg, gated):
            """RT-only Jacobian row for a cloud/Mie deck parameter (no chemistry
            re-solve). AD -> one jvp along `idx`. FD -> central difference: the
            analytic power-law deck is smooth (no solver noise), so gated=False
            rows take a SINGLE central difference -- which still carries the
            usual O(h^2) truncation error (~(h ln10)^2/6 ~ 0.2% at h=0.05 dex),
            just no h-vs-2h measurement of it, so fd_err is reported NaN
            (unmeasured), never 0 (v17). The Mie rg/sigmag rows ride the
            piecewise-linear miegrid, so they carry the same h-vs-2h consistency
            gate the theta rows use (gated=True) and REFUSE a step straddling a
            grid knot. Returns (row, h, err, method)."""
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
            if not gated:
                return j1, h, np.nan, "fd-rt"   # truncation unmeasured, not 0
            j2 = (_d(2.0 * h) - _d(-2.0 * h)) / (4.0 * h)
            scale = float(np.max(np.abs(j1)))
            if scale == 0.0:
                return j1, h, 0.0, "fd-rt"       # no spectral response
            err = float(np.max(np.abs(j1 - j2)) / scale)
            if err > FD_CONSISTENCY_TOL:
                raise RuntimeError(
                    f"Mie Jacobian for {name} FAILED the step-size consistency "
                    f"check: max|J(h) - J(2h)| / max|J(h)| = {err:.3f} > "
                    f"{FD_CONSISTENCY_TOL} (h = {h:g}). The FD step straddles a "
                    "miegrid knot (rg/sigmag interpolation is piecewise linear), "
                    "so the row is not a clean local slope. Move the parameter "
                    "off the grid node, shrink forward.FD_STEPS, or use "
                    "jac_method='ad' (the exact local-cell tangent). An "
                    "uncertified derivative is never reported.")
            return (4.0 * j1 - j2) / 3.0, h, err, "fd-rt"

        for j, name in enumerate(jac_names):
            t1 = time.time()
            advance()
            if name in CLOUD_FISHER_PARAMS or name in MIE_FISHER_PARAMS:
                # RT-only deck row, no chemistry re-solve: power-law deck and
                # MMR are smooth (ungated); Mie rg/sigmag ride the miegrid
                # (gated).
                if name in CLOUD_FISHER_PARAMS:
                    jac[j], _h, _err, _m = _rt_deck_row(
                        name, [cp["log_kappa_cloud"], cp["alpha_cloud"]],
                        CLOUD_FISHER_PARAMS.index(name), "cloud", gated=False)
                    _kind = "cloud"
                else:
                    jac[j], _h, _err, _m = _rt_deck_row(
                        name, [cp["mie_log_rg"], cp["mie_sigmag"],
                               cp["mie_log_mmr"]],
                        MIE_FISHER_PARAMS.index(name), "mie",
                        gated=(name != "mie_log_mmr"))
                    _kind = "Mie"
                fd_h.append(_h)
                fd_err.append(_err)
                row_method.append(_m)
                fd_kink.append(np.nan)
                fd_grid_cell.append("")
                if not np.isfinite(jac[j]).all():
                    raise RuntimeError(
                        f"{_kind} Jacobian for {name}: non-finite entries")
                log(f"[fwd] {cp['jac_method'].upper()} Jacobian "
                    f"d(depth)/d({name}) [RT-only {_kind} row] in "
                    f"{time.time()-t1:.0f} s")
                continue
            if name == "Tint_cl":
                # climate-mode structure row ("fd-climate"): each FD point
                # re-runs the certified climate at fixed rcb / CK node,
                # rebuilds the chemistry on the perturbed profile, and binds
                # the RT to that profile's OWN temperature. The climate solve
                # is bit-deterministic, so the h-vs-2h gate sees pure physics.
                h = FD_STEPS[name]
                from jwst_tool import picaso_climate as _pcl
                dvals = {}
                for s in (1, -1, 2, -2):
                    _tag = f"FD Tint_cl {s:+d}h"
                    clim_s = _pcl.get_or_run(cp, log,
                                             tint_override=cp["tint_cl"] + s * h)
                    if _is_picaso:
                        T_s = _pcl.interp_T(clim_s, chem.p_bar)
                        if (T_s.min() < T_WINDOW[0]
                                or T_s.max() > T_WINDOW[1]):
                            raise RuntimeError(
                                f"FD step for Tint_cl ({s:+d}h) leaves the "
                                f"modelable T window {T_WINDOW}: move "
                                "tint_cl away from the window edge or "
                                "reduce forward.FD_STEPS['Tint_cl'].")
                        y_s = chem.solve_at(cp["met_x_solar"],
                                            cp["co_ratio"], theta, _tag,
                                            T_prof=T_s)
                        _T_art_s = jnp.asarray(np.interp(
                            np.log(np.asarray(rt.p_art_bar)),
                            np.log(np.asarray(chem.p_bar)), T_s))
                        _dfn = make_depth_fn(chem, T_art_override=_T_art_s)
                        dvals[s] = np.asarray(_dfn(jnp.asarray(y_s), th0))
                    else:
                        chem_s = _build_chem(
                            {"atm_type": "file",
                             "atm_file": str(clim_s.atm_table)}, tag=_tag)
                        y_s, diag_s = chem_s.converged_y(
                            th0, return_conv_diag=True)
                        _check_converged(diag_s, _tag)
                        _dfn = make_depth_fn(
                            chem_s, T_art_override=_t_art_const_from(chem_s))
                        dvals[s] = np.asarray(_dfn(y_s, th0))
                jac[j], err = _fd_row(name, dvals[1], dvals[-1],
                                      dvals[2], dvals[-2], h)
                fd_h.append(h)
                fd_err.append(err)
                row_method.append("fd-climate")
                fd_kink.append(np.nan)
                fd_grid_cell.append("")
                log(f"[fwd] FD Jacobian d(depth)/d(Tint_cl) [climate "
                    f"re-solve row] in {time.time()-t1:.0f} s (h-vs-2h "
                    f"consistency {err:.3f} < {FD_CONSISTENCY_TOL})")
                continue
            if cp["jac_method"] == "ad":
                # AD row: one warm-started forward-mode jvp along this
                # theta direction (composition directions included -- the
                # cross-validated differential map; lnZ is the fixed-
                # structural-grid derivative, see the module docstring)
                i_par = theta_names.index(name)
                e = np.zeros_like(theta)
                e[i_par] = 1.0
                _, dd = jax.jvp(_ad_theta_depth, (th0,), (jnp.asarray(e),))
                jac[j] = np.asarray(dd)
                if not np.isfinite(jac[j]).all():
                    raise RuntimeError(
                        f"AD Jacobian for {name}: non-finite entries")
                fd_h.append(0.0)          # no FD step: AD row
                fd_err.append(np.nan)     # no h-vs-2h metric: AD row
                row_method.append("ad-jvp")
                fd_kink.append(np.nan)
                fd_grid_cell.append("")
                log(f"[fwd] AD Jacobian d(depth)/d({name}) in "
                    f"{time.time()-t1:.0f} s (warm-started jvp)")
                continue
            h = (PICASO_FD_STEPS[name]
                 if _is_picaso and name in FD_COMP_PARAMS else FD_STEPS[name])
            dvals = {}
            _row_kink, _row_cell = np.nan, ""
            if name in FD_COMP_PARAMS and _is_picaso:
                # picaso composition row: 4 cheap table re-evaluations on the
                # fixed provider grid -- a symmetric two-cell secant,
                # kink-gated below.
                from jwst_tool import picaso_chem as _pck
                for s in (1, -1, 2, -2):
                    met_s, co_s = _pck.comp_step(
                        cp["met_x_solar"], cp["co_ratio"], name, s * h)
                    dvals[s] = _picaso_comp_depth(met_s, co_s,
                                                  f"FD {name} {s:+d}h")
            elif name in FD_COMP_PARAMS:
                # composition direction: FastChem re-init + certified cold
                # solve per FD point (4x build+solve)
                for s in (1, -1, 2, -2):
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
                # theta direction (lnKzz / T-P): baseline build/adapter,
                # certified points at theta +- h, +- 2h (picaso: table
                # re-equilibration at the perturbed temperature)
                i_par = theta_names.index(name)
                for s in (1, -1, 2, -2):
                    th_s = theta.copy()
                    th_s[i_par] += s * h
                    # T-P step must stay in the window (tp_eval is None only
                    # in tabulated modes, which have no theta T-P rows)
                    if i_par >= 3 and tp_eval is not None:
                        T_s = np.asarray(tp_eval(jnp.asarray(th_s[3:]),
                                                 jnp.asarray(chem.p_bar)))
                        if T_s.min() < T_WINDOW[0] or T_s.max() > T_WINDOW[1]:
                            raise RuntimeError(
                                f"FD step for {name} ({s:+d}h = {s * h:+g}) "
                                f"leaves the modelable T window {T_WINDOW}: "
                                "move the profile away from the window edge "
                                "or reduce forward.FD_STEPS for it.")
                    if _is_picaso:
                        dvals[s] = _picaso_theta_depth(th_s,
                                                       f"FD {name} {s:+d}h")
                    else:
                        dvals[s] = _certified_depth(chem, th_s,
                                                    f"FD {name} {s:+d}h")
            jac[j], err = _fd_row(name, dvals[1], dvals[-1],
                                  dvals[2], dvals[-2], h)
            if name in FD_COMP_PARAMS and _is_picaso:
                # Node-kink gate: default compositions sit ON grid nodes with
                # no unique local derivative; a row whose one-sided secants
                # disagree materially hard-errors. The h-vs-2h gate alone
                # cannot see a kink (symmetric averages look stable across it).
                from jwst_tool import picaso_chem as _pck
                _j_left = (depth - dvals[-1]) / h
                _j_right = (dvals[1] - depth) / h
                _row_kink = _pck.kink_metric(_j_left, _j_right, jac[j])
                # provenance covers BOTH cells the stencil traverses
                _m_lo, _c_lo = _pck.comp_step(cp["met_x_solar"],
                                              cp["co_ratio"], name, -2.0 * h)
                _m_hi, _c_hi = _pck.comp_step(cp["met_x_solar"],
                                              cp["co_ratio"], name, +2.0 * h)
                _row_cell = json.dumps({
                    "param": name, "h": h,
                    "nodes_minus2h": _pck.bracketing_cells(_m_lo,
                                                           _c_lo)["nodes"],
                    "nodes_plus2h": _pck.bracketing_cells(_m_hi,
                                                          _c_hi)["nodes"]})
                if _row_kink > _pck.FD_KINK_TOL:
                    raise RuntimeError(
                        f"composition Jacobian for {name} FAILED the node-"
                        f"kink gate: |J_right - J_left| / max|J_sym| = "
                        f"{_row_kink:.3f} > {_pck.FD_KINK_TOL} (one-sided "
                        f"secant scales: max|J_left| = "
                        f"{float(np.max(np.abs(_j_left))):.3e}, "
                        f"max|J_right| = "
                        f"{float(np.max(np.abs(_j_right))):.3e}) across "
                        f"{json.loads(_row_cell)['nodes_minus2h']} | "
                        f"{json.loads(_row_cell)['nodes_plus2h']}. The "
                        "one-sided table secants disagree materially, so no "
                        "single derivative honestly summarizes this row. "
                        "Move the baseline off the node (or accept the "
                        "row's absence) -- an uncertified derivative is "
                        "never reported.")
            fd_h.append(h)
            fd_err.append(err)
            row_method.append("fd-central")
            fd_kink.append(_row_kink)
            fd_grid_cell.append(_row_cell)
            log(f"[fwd] FD Jacobian d(depth)/d({name}) in "
                f"{time.time()-t1:.0f} s (h-vs-2h consistency {err:.3f} < "
                f"{FD_CONSISTENCY_TOL}"
                + (f"; node kink {_row_kink:.3f}" if np.isfinite(_row_kink)
                   else "") + ")")

        t1 = time.time()
        advance()
        # lnR0 is RT-only (smooth, analytic in lnR0). "fd": one central
        # difference through the radiative transfer, no chemistry and no
        # gate needed; "ad": the RT jvp (the two agree to 0.9999, measured).
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
        fd_kink.append(np.nan)
        fd_grid_cell.append("")
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
        # the leave-one-out set depth_wo (and the emis_*_wo certificates) are
        # aligned with -- a subset of mols in fold order; readers must index
        # by THIS array, never by mols (v32)
        wo_mols=np.array(wo_list, dtype="U8"),
        ymix=ymix_np, p_bar=np.asarray(chem.p_bar),
        # THE COLUMN NAMES OF ymix. ymix is the FULL network state (89 species
        # for SNCHO), while `mols` is the much shorter RT molecule list, and
        # nothing recorded which was which. Every reader that zipped `mols`
        # against ymix's columns positionally was therefore reading the wrong
        # species: the GUI's mixing-ratio panel labelled H2 as CO2, and its CSV
        # export carried the same wrong headers. Never infer these from `mols`.
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
    if picaso_cert is not None:
        arrays["picaso_cert_json"] = np.array(json.dumps(picaso_cert))
    if clim is not None:
        arrays["climate_provenance_json"] = np.array(json.dumps(
            {"cert": clim.cert, **clim.provenance}))
    if emis is not None:
        arrays["fs_flux"] = np.asarray(fs_j, dtype=np.float64)
        # Fp derived exactly from the stored eclipse depth (lnR0 = 0 baseline)
        arrays["fp_flux"] = depth * np.asarray(fs_j) / _depth_norm_em0
        arrays["emis_tau_bottom_min"] = np.array([emis_tau_min])
        # per-removed-molecule bottom-tau certificate (aligned with wo_mols)
        arrays["emis_tau_bottom_min_wo"] = emis_tau_min_wo
        # v27: the FLUX-WEIGHTED certificate, which is what the gates use.
        # The min-tau arrays above are kept for provenance and for readers of
        # older files; a min() over the band cannot tell a shallow column from
        # a band edge with no modeled opacity (see the P_BTM_* block).
        arrays["emis_thin_flux_frac"] = np.array([emis_thin_frac])
        arrays["emis_thin_flux_frac_wo"] = emis_thin_wo
    if jac is not None:
        arrays["jac"] = jac
        arrays["jac_names"] = np.array(jac_names, dtype="U16")
        # Per-row provenance. fd_err NaN = unmeasured (AD rows, ungated
        # single-central-difference RT rows), never 0; a literal 0.0 appears
        # only for an exactly-zero-response row.
        arrays["jac_row_method"] = np.array(row_method, dtype="U16")
        arrays["fd_h"] = np.array(fd_h, dtype=np.float64)
        arrays["fd_err"] = np.array(fd_err, dtype=np.float64)
        # node-kink provenance (picaso composition rows; NaN/"" elsewhere)
        arrays["fd_kink"] = np.array(fd_kink, dtype=np.float64)
        arrays["fd_grid_cell"] = np.array(fd_grid_cell, dtype="U160")
    np.savez_compressed(out, **arrays)
    finish()
    log(f"[fwd] cached -> {out.name}")
    return out


def main():
    params = json.load(open(sys.argv[1]))
    # line-buffer stdout: the GUI pipes this process, which makes Python
    # BLOCK-buffer prints from libraries (picaso's climate iteration lines
    # would sit invisible in the buffer while the GUI shows nothing)
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    # vulcan_jax's legacy IO creates RELATIVE output/ + plot/ directories in
    # the process CWD (legacy_io.py) -- junk wherever the app was launched
    # from. Run the subprocess from a dedicated scratch cwd instead (the
    # same fix the Space entrypoint uses). Library callers of run_model are
    # unaffected: only this subprocess entrypoint changes directory.
    import os
    _cwd = Path(_ins.OUTPUT_DIR) / "cwd"
    _cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(_cwd)
    run_model(params, log=lambda *a: print(*a, flush=True))
    print("[fwd] DONE", flush=True)


if __name__ == "__main__":
    main()
