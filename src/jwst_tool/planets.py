"""Planet/system registry for the JWST instrument selector.

Method and contract: docs/forward_model.md, section `planets`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# IAU 2015 Resolution B3 nominal values (R_jup equatorial, R_sun, and
# M_jup = nominal GM_J / CODATA G), the IAU 2012 exact au, and CODATA 2018 G
# -- identical to astropy.constants to the digits printed.
R_JUP_CM = 7.1492e9
R_SUN_CM = 6.957e10
AU_CM = 1.495978707e13  # only converts orbit_au in system_teq
G_CGS = 6.67430e-8  # gravitational constant (cm^3 g^-1 s^-2); for gs_cgs -> Mp
M_JUP_G = 1.89813e30  # Jupiter mass (g); for archive mass -> surface gravity
CM_PER_M = 100.0  # exact; surface gravity m s^-2 <-> cm s^-2

# Stellar UV spectra usable as photochemistry input, all in one two-column surface-flux format, labeled by type. Most
# ship with vulcan3 (atm/stellar_flux/); OWN_SFLUX ship with this package. vulcan3's sflux-epseri.txt is not offered:
# it covers 115-283 nm, and vulcan-forward refuses a spectrum shorter than the network's photolysis thresholds.
SFLUX_CHOICES = {
    "sflux-W39b_Tsai2023.txt": "WASP-39 (G8V, Tsai 2023)",
    "Gueymard_solar.txt": "Sun (G2V, Gueymard 2004)",
    "sflux-solar_hd209b.txt": "HD 209458 proxy (Sun scaled to 6091 K)",
    "sflux-HD189_Moses11.txt": "HD 189733 (K1.5V, Moses 2011)",
    "sflux-HD85512_W107b.txt": "WASP-107 proxy (HD 85512 scaled to 4425 K, MUSCLES)",
    "sflux-GJ436.txt": "GJ 436 (M2.5V, MUSCLES)",
    "sflux-GJ1214.txt": "GJ 1214 (M4.5V, MUSCLES)",
}
# The sulfur survey's proxies (vulcan-retrieval data/survey/), shipped here in stellar_flux/ so the planner and the
# survey read the same file for the same star (tests/unit/test_survey_parity.py compares them).
OWN_SFLUX = frozenset({"sflux-solar_hd209b.txt", "sflux-HD85512_W107b.txt"})


def sflux_file(name: str) -> str:
    """Return the path vulcan3 reads ``name`` from: this package's file, or its own ``atm/stellar_flux`` entry."""
    if name in OWN_SFLUX:
        return str(Path(__file__).resolve().parent / "stellar_flux" / name)
    return f"atm/stellar_flux/{name}"


# Where each shipped spectrum comes from: (data set, DOI, page). Keyed exactly
# like SFLUX_CHOICES -- a spectrum without a citation cannot be listed in the
# GUI's data-sources table, and a test pins the two key sets equal. Every DOI
# resolved against doi.org.
SFLUX_SOURCES = {
    "sflux-W39b_Tsai2023.txt": (
        "Tsai et al. 2023",
        "10.1038/s41586-023-05902-2",
        "https://www.nature.com/articles/s41586-023-05902-2",
    ),
    "Gueymard_solar.txt": (
        "Gueymard 2004",
        "10.1016/j.solener.2003.08.039",
        "https://doi.org/10.1016/j.solener.2003.08.039",
    ),
    "sflux-HD189_Moses11.txt": (
        "Moses et al. 2011",
        "10.1088/0004-637X/737/1/15",
        "https://doi.org/10.1088/0004-637X/737/1/15",
    ),
    "sflux-GJ436.txt": (
        "MUSCLES Treasury Survey",
        "10.3847/0004-637X/820/2/89",
        "https://archive.stsci.edu/prepds/muscles/",
    ),
}
SFLUX_SOURCES["sflux-GJ1214.txt"] = SFLUX_SOURCES["sflux-GJ436.txt"]
SFLUX_SOURCES["sflux-HD85512_W107b.txt"] = SFLUX_SOURCES["sflux-GJ436.txt"]
SFLUX_SOURCES["sflux-solar_hd209b.txt"] = SFLUX_SOURCES["Gueymard_solar.txt"]

# Per-planet MEASURED structure tables bundled with vulcan3 (atm/).
#
# TWO SEPARATE FACTS, deliberately not conflated:
#   tp_table          the table exists and is SELECTABLE for this planet
#   tp_table_default  a default run on it has been VERIFIED end-to-end here
#
# Shipping a table is not evidence the tool converges on it (HD 189733 b's
# bundled profile stall-exits at the default settings, so it is offered, not
# imposed); tp_table_note carries the reason in both directions. A table only
# has to be valid across the RUN's chemistry span (model_defs.chem_p_span_dyn:
# the top follows rt_ptop_bar); HD 209458 b's thermosphere is already inside
# the shipped span, which disqualifies it, and HD 189733 b's enters the grid
# below 1e-7 bar, so its file mode needs rt_ptop_bar >= 1e-7.
PLANETS: dict[str, dict[str, Any]] = {
    "wasp39b": dict(
        label="WASP-39 b",
        # The sulfur survey's system (vulcan-retrieval runs/survey/wasp39b.yaml): the 1 mbar transit radius
        # 9.546962e9 cm, g = G Mp / rp^2 with Mp 0.281 M_J, R* 0.939 R_Sun.
        star=dict(teff=5485.0, log_g=4.5, metallicity=0.0, ks_mag=10.20),
        rstar_rsun=0.939,
        rp_rjup=1.33539,
        gs_cgs=390.58,
        orbit_au=0.04828,
        t14_hr=2.80,
        sflux="sflux-W39b_Tsai2023.txt",
        tp_table="atm_W39b_evening_TP_Kzz.txt",
        tp_table_default=False,
        tp_table_note=(
            "Selectable, not the default: the table ends at 5.3e-6 bar, below the deepest model top the tool supports "
            "(1e-6 bar), and a table must reach the model top, so a run on it is refused. Extend the table upward to "
            "use it."
        ),
        note="Inflated ~Saturn-mass planet (Ks = 10.2, G8V host); a well-observed "
        "warm transmission-spectroscopy target.",
    ),
    "hd189733b": dict(
        label="HD 189733 b",
        # The sulfur survey's system (vulcan-retrieval runs/survey/hd189733b.yaml): the 1 mbar transit radius
        # 8.105826e9 cm, g = G Mp / rp^2 with Mp 1.13 M_J, R* 0.75 R_Sun.
        star=dict(teff=5040.0, log_g=4.5, metallicity=0.0, ks_mag=5.54),
        rstar_rsun=0.75,
        rp_rjup=1.13381,
        gs_cgs=2178.79,
        orbit_au=0.03126,
        t14_hr=1.80,
        sflux="sflux-HD189_Moses11.txt",
        tp_table="atm_HD189_Kzz.txt",
        tp_table_default=False,
        tp_table_note=(
            "Selectable, not the default: the solver does not certify a "
            "steady state on it at the default settings, while the analytic "
            "profile converges. Certify a run before trusting one."
        ),
        note="Very bright host (Ks = 5.5) with a high-gravity planet: expect "
        "most modes to saturate and small spectral features.",
    ),
    "hd209458b": dict(
        label="HD 209458 b",
        # The sulfur survey's system (vulcan-retrieval runs/survey/hd209458b.yaml): the 1 mbar transit radius
        # 9.986325e9 cm, g = G Mp / rp^2 with Mp 0.73 M_J, and the solar UV spectrum scaled to the host's Teff.
        star=dict(teff=6091.0, log_g=4.4, metallicity=0.0, ks_mag=6.31),
        rstar_rsun=1.19,
        rp_rjup=1.39685,
        gs_cgs=927.35,
        orbit_au=0.04707,
        t14_hr=3.07,
        sflux="sflux-solar_hd209b.txt",
        tp_table=None,
        tp_table_default=False,
        tp_table_note=(
            "vulcan3 bundles atm_HD209_Kzz.txt, but it is a full "
            "thermosphere model: re-gridded onto the chemistry span it reaches "
            "2997 K at the grid top, above the tool's 2980 K temperature window. "
            "Out-of-window profiles are refused, never clipped, so this planet "
            "keeps the analytic default."
        ),
        note="The classic inflated hot Jupiter (G0V host; the solar UV spectrum scaled to its temperature).",
    ),
    "wasp107b": dict(
        label="WASP-107 b",
        # The sulfur survey's system (vulcan-retrieval runs/survey/wasp107b.yaml): the 1 mbar transit radius
        # 6.567906e9 cm, g = G Mp / rp^2 with Mp 0.1039 M_J, and the MUSCLES HD 85512 UV spectrum scaled to the
        # host's Teff (the proxy of Dyrek et al. 2024).
        star=dict(teff=4425.0, log_g=4.6, metallicity=0.0, ks_mag=8.64),
        rstar_rsun=0.657,
        rp_rjup=0.91869,
        gs_cgs=305.14,
        orbit_au=0.055,
        t14_hr=2.74,
        sflux="sflux-HD85512_W107b.txt",
        tp_table=None,
        tp_table_default=False,
        tp_table_note="vulcan3 bundles no measured T-P/Kzz table for this planet.",
        # Guillot T_int: Sing et al. 2024 (Nature 630, 831) retrieve
        # T_int = 460 +/- 40 K from the CH4 depletion; Welbanks et al. 2024
        # (Nature 630, 836) give > 345 K. The generic 100 K puts the quench
        # level, and with it this planet's headline CH4 result, in the wrong
        # place.
        tint_k=460.0,
        note="Warm Neptune-mass super-puff: very low gravity means huge "
        "spectral features (K6V host; HD 85512 UV proxy).",
    ),
}

# Interior temperature default, K, of a planet the registry gives no published value (a maintainer's choice; a request
# should state its own, and a run that does not carries a caveat with its result).
TINT_DEFAULT_K = 300.0

# Photolysis zenith angle, degrees. A transit sees the terminator: 83 deg is the slant of vulcan3's WASP-39 b config
# (Tsai et al. 2023). An eclipse sees the dayside: 58 deg is VULCAN's dayside-average angle (Tsai et al. 2021).
SL_ANGLE_DEG = {"transmission": 83.0, "emission": 58.0}


def default_tint(planet: dict) -> float:
    """Return the default interior temperature (K), the Guillot and PICASO climate T_int.

    The planet's published value where the registry carries one, else TINT_DEFAULT_K. One definition for
    canonical_params and the sidebar widget.
    """
    return float(planet.get("tint_k", TINT_DEFAULT_K))


# The seed of the GUI's custom-planet form only (WASP-39 b's values, every field editable). The API never reads it: a
# custom request states its own system (run_params.CUSTOM_REQUIRED_KEYS).
CUSTOM_FORM_SEED = PLANETS["wasp39b"]

# Single source of truth for the custom-planet form bounds, in WIDGET units
# (g in m s^-2 = gs_cgs / 100; everything else as displayed). the sidebar unpacks
# these into its number_inputs and archive.custom_fill refuses values outside
# them -- Streamlit does NOT reject an out-of-range session-state value, it
# silently discards it and the widget falls back to its default, so the two
# consumers must never drift. Widening a range is a physics decision (PHOENIX
# grid nodes, the forward model's temperature window, validated chemistry
# envelope), not a UI tweak.
CUSTOM_FIELD_RANGES = {
    "teff": (3000.0, 7000.0),  # K; PHOENIX + chemistry validity
    "logg": (3.5, 5.5),  # log10 cm s^-2
    "feh": (-2.0, 0.5),  # dex
    "ks": (4.0, 16.0),  # 2MASS Ks vegamag
    "rstar": (0.2, 3.0),  # R_sun
    "rp": (0.1, 2.5),  # R_jup
    "g": (1.0, 100.0),  # m s^-2 (widget unit; registry stores cgs)
    "a": (0.005, 1.0),  # AU
    "t14": (0.5, 10.0),  # hours
}


def system_teq(star_teff: float, rstar_rsun: float, orbit_au: float) -> float:
    """Zero-albedo full-redistribution equilibrium temperature: T_eq = T_eff * sqrt(R_star / (2 a)).

    The sqrt(2) * T_eq irradiation temperature this implies equals T_eff * sqrt(R_star / a).
    """
    return float(star_teff) * math.sqrt(float(rstar_rsun) * R_SUN_CM / (2.0 * float(orbit_au) * AU_CM))


# Dayside-average versus planet-average redistribution, in temperature:
# T_eq(dayside) = T_eq(global) * 2**0.25. A published equilibrium temperature is
# the GLOBAL one (f = 1/4), which is the right structure for a terminator and
# the wrong one for an eclipse.
DAYSIDE_TEQ_FACTOR = 2.0**0.25


# Guillot T_irr the tool accepts (K), from the API and the sidebar widget alike. No physical basis is recorded for the
# ends; the profile built from it is then held to model_defs.T_WINDOW. A value outside is refused, never clipped.
TIRR_RANGE = (800.0, 2500.0)

# K: the default T_irr is rounded onto the T_irr widget's step, so an untouched widget holds the API's value exactly.
TIRR_STEP_K = 20.0


def default_tirr(planet: dict, system: dict | None = None, science_mode: str = "transmission") -> float:
    """Guillot T_irr default: sqrt(2) * T_eq on the GUI's 20 K step grid, never clipped (check_tirr refuses it).

    T_eq is always DERIVED from the star and orbit, never stored: for registry
    entries from their own ``star["teff"]``/``rstar_rsun``/``orbit_au``, for
    the CUSTOM planet from the ``system`` the caller passes (otherwise an
    untouched custom T_irr silently keeps WASP-39 b's temperature).

    EMISSION takes the DAYSIDE value, T_eq * 2**0.25: a published equilibrium
    temperature assumes full redistribution, which is right for a terminator
    and too cool for the dayside an eclipse sees.

    ONE definition, used by BOTH canonical_params and the sidebar widget.
    """
    teq = (
        system_teq(system["star_teff"], system["rstar_rsun"], system["orbit_au"])
        if system is not None
        else system_teq(planet["star"]["teff"], planet["rstar_rsun"], planet["orbit_au"])
    )
    if str(science_mode) == "emission":
        teq *= DAYSIDE_TEQ_FACTOR
    return round(teq * math.sqrt(2.0) / TIRR_STEP_K) * TIRR_STEP_K


def tirr_refusal(tirr_k: float, derived: bool) -> str | None:
    """Return the refusal sentence of a Guillot T_irr outside TIRR_RANGE, None inside; the API and the sidebar share it.

    Parameters
    ----------
    tirr_k : float
        Irradiation temperature, K.
    derived : bool
        True when ``tirr_k`` is the default computed from the star and orbit (default_tirr), not a value the user set.

    Returns
    -------
    str or None
        The sentence, or None when the value is accepted. A non-finite value is refused.
    """
    lo, hi = TIRR_RANGE
    if lo <= tirr_k <= hi:
        return None
    if derived:
        return (
            f"the default Tirr of this star and orbit is {tirr_k:g} K (sqrt(2) x T_eq), outside the supported Guillot "
            f"range [{lo:g}, {hi:g}] K. Set Tirr explicitly inside that range."
        )
    return f"Tirr={tirr_k:g} K is outside the supported Guillot range [{lo:g}, {hi:g}] K."


def check_tirr(tirr_k: float, derived: bool) -> None:
    """Raise ValueError with ``tirr_refusal``'s sentence when ``tirr_k`` is outside TIRR_RANGE or not finite."""
    refusal = tirr_refusal(tirr_k, derived)
    if refusal:
        raise ValueError(refusal)


# Gray cloud: opacity <-> the pressure it blocks down to.
#
# The engine's cloud is a uniformly mixed per-gram opacity, dtau = kappa dP/g
# (vulcan_forward._rt_opacity._ckd_continuum), so the SLANT optical depth through
# an exponential atmosphere is tau(P) = kappa (P/g) sqrt(2 pi Rp/H) -- the
# closed form vulcan-forward gates its own RT against (its
# tests/test_cloud_deck.py, Heng & Kitzmann 2017; worst case 1.04e-2 H). Setting
# tau = 1 gives the level the cloud blocks down to:
#
#     P_top = (1/kappa) sqrt(g kB T / (2 pi Rp mu m_u))
#
# The two coordinates differ by a constant in log space, so a Fisher width in
# dex is the SAME number for log kappa and for log10 P_top.
MMW_CLOUD = 2.33  # H2/He mean molecular weight assumed by the mapping
_K_B_CGS = 1.380649e-16  # erg/K
_M_U_CGS = 1.66053907e-24  # g


def gray_cloud_log_kappa(p_top_bar: float, gs_cgs: float, rp_rjup: float, t_eq_k: float) -> float:
    """Return log10 kappa of the gray cloud whose slant tau reaches 1 at ``p_top_bar``.

    log10(kappa * P_top_bar) is the constant of the mapping (the section comment above derives it).

    Parameters
    ----------
    p_top_bar : float
        Pressure the cloud blocks down to, bar.
    gs_cgs : float
        Surface gravity, cm s^-2.
    rp_rjup : float
        Planet radius, R_jup.
    t_eq_k : float
        Temperature of the scale height, K.

    Returns
    -------
    float
        log10 of the cloud opacity, cm^2/g.

    Raises
    ------
    ValueError
        An input is not positive.
    """
    if not float(p_top_bar) > 0.0:
        raise ValueError(f"cloud top pressure must be > 0 bar (got {p_top_bar!r})")
    g, rp = float(gs_cgs), float(rp_rjup) * R_JUP_CM
    if not (g > 0.0 and rp > 0.0 and float(t_eq_k) > 0.0):
        raise ValueError(
            f"gray cloud mapping needs positive gs_cgs/rp_rjup/t_eq_k (got {gs_cgs!r}, {rp_rjup!r}, {t_eq_k!r})"
        )
    h = _K_B_CGS * float(t_eq_k) / (MMW_CLOUD * _M_U_CGS * g)
    return math.log10(g / (1.0e6 * math.sqrt(2.0 * math.pi * rp / h))) - math.log10(float(p_top_bar))
