"""Planet/system registry for the JWST instrument selector.

Pure data, importable by the light GUI path (no jax/vulcan/exojax imports).

Every planet runs on the SAME W39b-validated machinery: the WASP-39b SNCHO
photo network + 10x-solar FastChem baseline (the import-locked network), with
the planet identity injected through the existing hooks. Shared code path, not
per-planet validation -- the committed parity/live evidence is W39b-centered
(docs/audit_decisions_2026-07-21.md) --

    chemistry : cfg_overrides {Mp, Rp, r_star, orbit_radius, sflux_file, ...}
                (VULCAN derives gs = G*Mp/Rp^2; gs_cgs is converted to Mp at the
                boundary in forward.py. vulcan_chem.build_chem_model applies them
                before the pre-loop)
    RT        : profile {rp_cm, gs_cgs, rstar_cm}
                (exojax_rt.build_rt_model reads them for geometry/normalization)
    noise     : star dict -> pandeia phoenix SED + Ks normalization
    timing    : t14_hr -> in/out-of-transit integration split

T-P and Kzz are explicit modes, never silently substituted (the WASP-39b GCM
baseline modes were removed 2026-07-13; the globally isothermal T-P mode was
removed 2026-07-21 -- it held the CO/CH4/NH3 quench region at one temperature).
Under Guillot the structural baseline is isothermal at a representative
temperature (it sets only the hydrostatic grid + EQ init; the on-graph tp_eval
supplies T(P) for chemistry and RT), while ``file`` / ``picaso_climate`` make
the tabulated profile itself the structure. Kzz is const / Pfunc / JM16 / the
table's own column. Per-planet defaults live in ``tp_table_default`` below and
are tabulated in docs/physics_and_conventions.md.

Values are literature defaults for PLANNING (all editable in the GUI):
WASP-39b Mancini+2018/Tsai+2023; HD 189733b Torres+2008 (a: Bouchy+2005);
HD 209458b Torres+2008 (a, gs from Southworth+2010); WASP-107b Piaulet+2021.
(Provenance audited against the NASA Exoplanet Archive 2026-07-15: every
numeric field within the literature spread.) Stellar UV: shipped VULCAN spectra,
nearest available spectral type (shown in the GUI, never silently swapped).

``star["metallicity"]`` is DELIBERATELY 0.0 for every host, and is not a
literature value (WASP-39 is published near -0.12). It selects the PHOENIX
node for the noise SED and the emission-mode stellar surface flux, and both
consumers renormalize the SED to the star's observed 2MASS Ks magnitude, so a
0.1-0.2 dex node shift changes only the SED SHAPE across the JWST bands -- a
sub-percent effect on forecast sigmas, far below the ~2-56% pandeia-vs-PandExo
noise-model envelope (tests/parity/outputs/REPORT.md). Keeping every host on
the solar node keeps the four registry entries on one interpolation cell of
the minimal shipped CDBS grid. The GUI exposes [Fe/H] as an editable field
(-2.0 to 0.5), so a run that cares sets it explicitly.
"""
from __future__ import annotations

import math

# IAU 2015 Resolution B3 nominal values (R_jup equatorial, R_sun) and
# CODATA 2018 G -- identical to astropy.constants to the digits printed.
R_JUP_CM = 7.1492e9
R_SUN_CM = 6.957e10
G_CGS = 6.67430e-8  # gravitational constant (cm^3 g^-1 s^-2); for gs_cgs -> Mp

# Shipped stellar UV spectra usable as photochemistry input (VULCAN-JAX
# atm/stellar_flux/, all same two-column surface-flux format), labeled by type.
SFLUX_CHOICES = {
    "sflux-W39b_Tsai2023.txt": "WASP-39 (G8V, Tsai 2023)",
    "Gueymard_solar.txt": "Sun (G2V, Gueymard 2003)",
    "sflux-HD189_Moses11.txt": "HD 189733 (K1.5V, Moses 2011)",
    "sflux-epseri.txt": "eps Eridani (K2V, MUSCLES)",
    "sflux-GJ436.txt": "GJ 436 (M2.5V, MUSCLES)",
    "sflux-GJ1214.txt": "GJ 1214 (M4.5V, MUSCLES)",
}

# Per-planet MEASURED structure tables bundled with vulcan_jax (atm/).
#
# TWO SEPARATE FACTS, deliberately not conflated:
#   tp_table          the table exists and is SELECTABLE for this planet
#   tp_table_default  a default run on it has been VERIFIED end-to-end here
#
# Shipping a table is not evidence that the tool converges on it. HD 189733 b
# has a perfectly good bundled profile that the solver does NOT certify at the
# default settings (measured 2026-07-22: longdy 0.0998 against the 0.1 gate but
# conv_normal False -- a stall exit, so the run correctly hard-refuses), while
# its analytic default converges in 36 s. Making that table the default would
# have turned a working planet into one that errors on arrival, so it is
# offered rather than imposed. `tp_table_note` carries the reason in both
# directions so neither the gap nor the caveat is silent.
#
# The tool runs every planet on the W39b cfg's FIXED chemistry grid
# (forward.CHEM_P_SPAN_DYN = 0.1 to 7.6e6 dyn/cm^2), so a table only has to be
# valid ACROSS THAT SPAN; rows deeper or higher are never evaluated. What
# disqualifies HD 209458 b is that its thermosphere is already inside the span.
PLANETS = {
    "wasp39b": dict(
        label="WASP-39 b",
        star=dict(teff=5485.0, log_g=4.5, metallicity=0.0, ks_mag=10.20),
        rstar_rsun=0.932, rp_rjup=1.279, gs_cgs=422.0,
        orbit_au=0.04828, teq_k=1120.0, t14_hr=2.80,
        sflux="sflux-W39b_Tsai2023.txt",
        tp_table="atm_W39b_evening_TP_Kzz.txt", tp_table_default=True,
        tp_table_note="",
        note="Inflated ~Saturn-mass planet (Ks = 10.2, G8V host); a well-observed "
             "warm transmission-spectroscopy target.",
    ),
    "hd189733b": dict(
        label="HD 189733 b",
        star=dict(teff=5040.0, log_g=4.5, metallicity=0.0, ks_mag=5.54),
        rstar_rsun=0.756, rp_rjup=1.138, gs_cgs=2190.0,
        orbit_au=0.0313, teq_k=1200.0, t14_hr=1.80,
        sflux="sflux-HD189_Moses11.txt",
        tp_table="atm_HD189_Kzz.txt", tp_table_default=False,
        tp_table_note=(
            "selectable, but NOT the default: at the tool's default settings "
            "the solver does not certify a steady state on it (measured "
            "2026-07-22: longdy 0.0998 vs the 0.1 gate with conv_normal False "
            "-- a stall exit, which the run refuses rather than presents as "
            "converged), whereas the analytic default converges in ~36 s. "
            "Tightening yconv_cri to 1e-3 does NOT help: same longdy, same "
            "1445 accepted steps, so the blocker is the stall exit and not the "
            "tolerance. Its tabulated Kzz is 10-17x the constant 1e9 default "
            "over the chemistry grid at p < 1 mbar (measured 2026-07-28; the "
            "earlier 15-17x did not reproduce at any pressure cut), so it is "
            "the better structure on paper; certify a run before trusting one."),
        note="Very bright host (Ks = 5.5) with a high-gravity planet: expect "
             "most modes to saturate and small spectral features.",
    ),
    "hd209458b": dict(
        label="HD 209458 b",
        star=dict(teff=6065.0, log_g=4.4, metallicity=0.0, ks_mag=6.31),
        rstar_rsun=1.155, rp_rjup=1.359, gs_cgs=930.0,
        orbit_au=0.0475, teq_k=1450.0, t14_hr=3.07,
        sflux="Gueymard_solar.txt",
        tp_table=None, tp_table_default=False,
        tp_table_note=(
            "vulcan_jax bundles atm_HD209_Kzz.txt, but it is a full "
            "thermosphere model: re-gridded onto the chemistry span it reaches "
            "2997 K at the grid top, above the 2980 K premodit opacity ceiling. "
            "Out-of-window profiles are refused, never clipped, so this planet "
            "keeps the analytic default."),
        note="The classic inflated hot Jupiter (G0V host; solar UV spectrum, "
             "same proxy the VULCAN HD209 config uses).",
    ),
    "wasp107b": dict(
        label="WASP-107 b",
        star=dict(teff=4430.0, log_g=4.6, metallicity=0.0, ks_mag=8.64),
        rstar_rsun=0.67, rp_rjup=0.94, gs_cgs=270.0,
        orbit_au=0.0553, teq_k=740.0, t14_hr=2.74,
        sflux="sflux-epseri.txt",
        tp_table=None, tp_table_default=False,
        tp_table_note="vulcan_jax bundles no measured T-P/Kzz table for this planet.",
        note="Warm Neptune-mass super-puff: very low gravity means huge "
             "spectral features (K6V host; eps Eri UV proxy).",
    ),
}

# The "custom" planet starts from these (WASP-39b) values; everything editable.
CUSTOM_DEFAULTS = PLANETS["wasp39b"]


def system_fields(planet: dict) -> dict:
    """The forward-model parameter fields carried by a registry entry."""
    return dict(rp_rjup=planet["rp_rjup"], gs_cgs=planet["gs_cgs"],
                rstar_rsun=planet["rstar_rsun"], orbit_au=planet["orbit_au"],
                sflux=planet["sflux"])


AU_CM = 1.496e13


def system_teq(star_teff: float, rstar_rsun: float, orbit_au: float) -> float:
    """Zero-albedo full-redistribution equilibrium temperature:
    T_eq = T_eff * sqrt(R_star / (2 a)). The sqrt(2) * T_eq irradiation
    temperature this implies equals T_eff * sqrt(R_star / a), the same
    expression the PICASO climate solve uses for its guess (picaso_climate)."""
    return float(star_teff) * math.sqrt(
        float(rstar_rsun) * R_SUN_CM / (2.0 * float(orbit_au) * AU_CM))


def default_tirr(planet: dict, system: dict | None = None) -> float:
    """Guillot T_irr default: sqrt(2) * T_eq, on the GUI's 20 K step grid and
    clipped to the widget range.

    Registry entries use their literature ``teq_k``. For the CUSTOM planet the
    caller passes ``system`` (star_teff, rstar_rsun, orbit_au) and T_eq is
    DERIVED from the entered star and orbit -- without this, an untouched
    custom T_irr silently kept WASP-39 b's temperature no matter what system
    the user typed in (fixed 2026-08-04).

    ONE definition, used by BOTH canonical_params and the sidebar widget. It
    used to be a hard-coded constant on the API side and a T_eq expression in
    the GUI, so a "default" API run and a "default" GUI run built different
    profiles for every planet whose T_eq was not WASP-39 b's (HD 209458 b
    disagreed by 470 K)."""
    teq = (system_teq(system["star_teff"], system["rstar_rsun"],
                      system["orbit_au"])
           if system is not None else planet["teq_k"])
    return min(max(round(teq * math.sqrt(2.0) / 10.0) * 10.0, 800.0), 2500.0)
