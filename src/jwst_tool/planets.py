"""Planet/system registry for the JWST instrument selector.

Pure data, importable by the light GUI path (no jax/vulcan/exojax imports).

Every planet runs on the SAME W39b-validated machinery (the WASP-39b SNCHO
photo network + 10x-solar FastChem baseline), with the planet identity
injected through the existing hooks:

    chemistry : cfg_overrides {Mp, Rp, r_star, orbit_radius, sflux_file, ...}
                (VULCAN derives gs = G*Mp/Rp^2; gs_cgs is converted to Mp at
                the boundary in forward.py)
    RT        : profile {rp_cm, gs_cgs, rstar_cm}
    noise     : star dict -> pandeia phoenix SED + Ks normalization
    timing    : t14_hr -> in/out-of-transit integration split

T-P and Kzz are explicit modes, never silently substituted. Under Guillot the
structural baseline is isothermal at a representative temperature (it sets
only the hydrostatic grid + EQ init; tp_eval supplies T(P) for chemistry and
RT); ``file`` makes the tabulated profile the structure.
Kzz is const / Pfunc / JM16 / the table's own column.

Values are literature defaults for PLANNING (all editable in the GUI):
WASP-39b Mancini+2018/Tsai+2023; HD 189733b Torres+2008 (a: Bouchy+2005);
HD 209458b Torres+2008 (a, gs from Southworth+2010); WASP-107b Piaulet+2021.
Stellar UV: shipped VULCAN spectra, nearest available spectral type (shown in
the GUI, never silently swapped).

``star["metallicity"]`` is DELIBERATELY 0.0 for every host, not a literature
value: it only selects the PHOENIX node for the noise SED, both consumers
renormalize to the observed 2MASS Ks magnitude, and a 0.1-0.2 dex node shift
is a sub-percent effect on forecast sigmas. The GUI exposes [Fe/H] as an
editable field, so a run that cares sets it explicitly.
"""
from __future__ import annotations

import math

# IAU 2015 Resolution B3 nominal values (R_jup equatorial, R_sun, and
# M_jup = nominal GM_J / CODATA G) and CODATA 2018 G -- identical to
# astropy.constants to the digits printed.
R_JUP_CM = 7.1492e9
R_SUN_CM = 6.957e10
G_CGS = 6.67430e-8  # gravitational constant (cm^3 g^-1 s^-2); for gs_cgs -> Mp
M_JUP_G = 1.89813e30  # Jupiter mass (g); for archive mass -> surface gravity

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

# Where each shipped spectrum comes from: (data set, DOI, page). Keyed exactly
# like SFLUX_CHOICES -- a spectrum without a citation cannot be listed in the
# GUI's data-sources table, and a test pins the two key sets equal. Every DOI
# resolved against doi.org.
SFLUX_SOURCES = {
    "sflux-W39b_Tsai2023.txt": (
        "Tsai et al. 2023", "10.1038/s41586-023-05902-2",
        "https://www.nature.com/articles/s41586-023-05902-2"),
    "Gueymard_solar.txt": (
        "Gueymard 2004", "10.1016/j.solener.2003.08.039",
        "https://doi.org/10.1016/j.solener.2003.08.039"),
    "sflux-HD189_Moses11.txt": (
        "Moses et al. 2011", "10.1088/0004-637X/737/1/15",
        "https://doi.org/10.1088/0004-637X/737/1/15"),
    "sflux-epseri.txt": (
        "MUSCLES Treasury Survey", "10.3847/0004-637X/820/2/89",
        "https://archive.stsci.edu/prepds/muscles/"),
}
SFLUX_SOURCES["sflux-GJ436.txt"] = SFLUX_SOURCES["sflux-epseri.txt"]
SFLUX_SOURCES["sflux-GJ1214.txt"] = SFLUX_SOURCES["sflux-epseri.txt"]

# Host Teff per shipped UV spectrum, powering a SUGGESTION caption on custom
# planets only ("nearest-Teff shipped template: ..."). It is never applied:
# neither the GUI nor the archive fill selects a UV spectrum for the user
# (standing rule; registry planets keep their curated sflux).
# Values:
# WASP-39 / HD 189733 = this registry; Sun = IAU nominal; eps Eri ~5084 K
# (interferometric, Baines & Armstrong 2012); GJ 436 ~3416 K (von Braun 2012);
# GJ 1214 ~3026 K (Cloutier 2021). Only the ORDERING matters for
# nearest-neighbor selection, so literature spread within ~100 K is harmless.
SFLUX_TEFF_ANCHORS = {
    "sflux-W39b_Tsai2023.txt": 5485.0,
    "Gueymard_solar.txt": 5772.0,
    "sflux-HD189_Moses11.txt": 5040.0,
    "sflux-epseri.txt": 5084.0,
    "sflux-GJ436.txt": 3416.0,
    "sflux-GJ1214.txt": 3026.0,
}
if set(SFLUX_TEFF_ANCHORS) != set(SFLUX_CHOICES):
    raise RuntimeError(
        "SFLUX_TEFF_ANCHORS is out of sync with SFLUX_CHOICES: "
        f"{sorted(set(SFLUX_TEFF_ANCHORS) ^ set(SFLUX_CHOICES))}; the "
        "nearest-spectral-type default needs one Teff anchor per spectrum.")


def nearest_sflux(teff_k: float) -> str:
    """Filename of the shipped UV spectrum whose host Teff is closest to
    ``teff_k`` (ties resolve by SFLUX_TEFF_ANCHORS insertion order)."""
    t = float(teff_k)
    return min(SFLUX_TEFF_ANCHORS, key=lambda f: abs(SFLUX_TEFF_ANCHORS[f] - t))

# Per-planet MEASURED structure tables bundled with vulcan_jax (atm/).
#
# TWO SEPARATE FACTS, deliberately not conflated:
#   tp_table          the table exists and is SELECTABLE for this planet
#   tp_table_default  a default run on it has been VERIFIED end-to-end here
#
# Shipping a table is not evidence the tool converges on it (HD 189733 b's
# bundled profile stall-exits at the default settings, so it is offered, not
# imposed); tp_table_note carries the reason in both directions. A table only
# has to be valid across the RUN's chemistry span (forward.chem_p_span_dyn:
# the top follows rt_ptop_bar); HD 209458 b's thermosphere is already inside
# the shipped span, which disqualifies it, and HD 189733 b's enters the grid
# below 1e-7 bar, so its file mode needs rt_ptop_bar >= 1e-7.
PLANETS = {
    "wasp39b": dict(
        label="WASP-39 b",
        star=dict(teff=5485.0, log_g=4.5, metallicity=0.0, ks_mag=10.20),
        rstar_rsun=0.932, rp_rjup=1.279, gs_cgs=422.0,
        orbit_au=0.04828, t14_hr=2.80,
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
        orbit_au=0.0313, t14_hr=1.80,
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
        orbit_au=0.0475, t14_hr=3.07,
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
        orbit_au=0.0553, t14_hr=2.74,
        sflux="sflux-epseri.txt",
        tp_table=None, tp_table_default=False,
        tp_table_note="vulcan_jax bundles no measured T-P/Kzz table for this planet.",
        note="Warm Neptune-mass super-puff: very low gravity means huge "
             "spectral features (K6V host; eps Eri UV proxy).",
    ),
}

# The "custom" planet starts from these (WASP-39b) values; everything editable.
CUSTOM_DEFAULTS = PLANETS["wasp39b"]

# Single source of truth for the custom-planet form bounds, in WIDGET units
# (g in m s^-2 = gs_cgs / 100; everything else as displayed). app.py unpacks
# these into its number_inputs and archive.custom_fill refuses values outside
# them -- Streamlit hard-errors at widget instantiation on an out-of-range
# session-state value, so the two consumers must never drift. Widening a range
# is a physics decision (PHOENIX grid nodes, premodit opacity ceiling,
# validated chemistry envelope), not a UI tweak.
CUSTOM_FIELD_RANGES = {
    "teff": (3000.0, 7000.0),   # K; PHOENIX + chemistry validity
    "logg": (3.5, 5.5),         # log10 cm s^-2
    "feh": (-2.0, 0.5),         # dex
    "ks": (4.0, 16.0),          # 2MASS Ks vegamag
    "rstar": (0.2, 3.0),        # R_sun
    "rp": (0.1, 2.5),           # R_jup
    "g": (1.0, 100.0),          # m s^-2 (widget unit; registry stores cgs)
    "a": (0.005, 1.0),          # AU
    "t14": (0.5, 10.0),         # hours
}


def system_fields(planet: dict) -> dict:
    """The forward-model parameter fields carried by a registry entry."""
    return dict(rp_rjup=planet["rp_rjup"], gs_cgs=planet["gs_cgs"],
                rstar_rsun=planet["rstar_rsun"], orbit_au=planet["orbit_au"],
                sflux=planet["sflux"])


AU_CM = 1.496e13


def system_teq(star_teff: float, rstar_rsun: float, orbit_au: float) -> float:
    """Zero-albedo full-redistribution equilibrium temperature:
    T_eq = T_eff * sqrt(R_star / (2 a)). The sqrt(2) * T_eq irradiation
    temperature this implies equals T_eff * sqrt(R_star / a)."""
    return float(star_teff) * math.sqrt(
        float(rstar_rsun) * R_SUN_CM / (2.0 * float(orbit_au) * AU_CM))


# Dayside-average versus planet-average redistribution, in temperature:
# T_eq(dayside) = T_eq(global) * 2**0.25. A published equilibrium temperature is
# the GLOBAL one (f = 1/4), which is the right structure for a terminator and
# the wrong one for an eclipse.
DAYSIDE_TEQ_FACTOR = 2.0 ** 0.25


def default_tirr(planet: dict, system: dict | None = None,
                 science_mode: str = "transmission") -> float:
    """Guillot T_irr default: sqrt(2) * T_eq, on the GUI's 20 K step grid and
    clipped to the widget range.

    T_eq is always DERIVED from the star and orbit -- for registry entries from
    their own ``star["teff"]``/``rstar_rsun``/``orbit_au``, for the CUSTOM
    planet from the ``system`` the caller passes (otherwise an untouched custom
    T_irr silently keeps WASP-39 b's temperature). A stored literature value is
    a second source of truth that goes stale: WASP-39 b's carried 1120 K from
    the Faedi et al. 2011 stellar parameters after Teff and R_star had been
    refreshed to the JWST-ERS values, 3.8 % below what its own entry implies.

    EMISSION takes the DAYSIDE value, T_eq * 2**0.25. A published
    equilibrium temperature assumes full redistribution, which is a
    planet-average profile: correct for a terminator, too cool for the dayside
    an eclipse actually sees. Measured on HD 189733 b against the published
    MIRI/LRS eclipse spectrum (Inglis et al. 2024), switching only this default
    moved the eclipse depth from 1.35x LOW to within 3%, and chi2 per point from
    934 to 31. It is the single largest error in the emission path.

    ONE definition, used by BOTH canonical_params and the sidebar widget; a
    split default built different profiles per planet (history: notes.md)."""
    teq = (system_teq(system["star_teff"], system["rstar_rsun"],
                      system["orbit_au"])
           if system is not None else
           system_teq(planet["star"]["teff"], planet["rstar_rsun"],
                      planet["orbit_au"]))
    if str(science_mode) == "emission":
        teq *= DAYSIDE_TEQ_FACTOR
    return min(max(round(teq * math.sqrt(2.0) / 20.0) * 20.0, 800.0), 2500.0)
