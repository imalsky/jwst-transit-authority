"""PICASO-native RT vs the tool's ExoJax RT on ONE identical atmosphere.

Offline validation ONLY (the production path is always provider chemistry +
ExoJax; decision 2026-07-20). Requires the native opacity DB
(opacities/opacities/opacities_0.3_15_R15000.db) in the reference tree and
the full tool stack. Writes tests/parity_picaso/outputs/REPORT.md and a PNG.

Method: the SAME state -- W39b geometry, isothermal 1100 K, blended
equilibrium chemistry at 10x solar / C/O 0.55, absorbers restricted to the
shared set {H2O, CO2, CO, CH4} on an H2/He background -- runs through
(a) the tool's ExoJax transmission RT and (b) picaso's get_transit_1d.
Both spectra are binned to R = 100 over 1-12 um.

STATED TOLERANCE TARGETS (why exact agreement is NOT expected):
* different opacity sources (native: the zenodo R=15000 resampled DB
  'default_3.3'; tool: HITRAN line lists through exojax PreMODIT) and
  different broadening treatments;
* different reference-radius conventions (picaso anchors the transit radius
  at a reference pressure; the tool anchors Rp at the RT bottom): a BROADBAND
  OFFSET is expected and removed (reported separately) before comparing;
* different gravity treatments: picaso's altitude integration uses
  g(z) = GM/z^2 (mass+radius are REQUIRED -- passing gravity alone leaves
  planet.mass NaN and the native transmission silently returns all-NaN),
  while the tool's RT uses the constant surface gravity.
Targets: |offset| < 2000 ppm; median |residual| after offset removal
< 150 ppm; p95 < 400 ppm in 1-10 um. Violations are findings to report,
not necessarily bugs -- this is a cross-model comparison, never a CI gate.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
R_BIN = 100.0
WL_MIN, WL_MAX = 1.0, 12.0
MOLS = ["H2O", "CO2", "CO", "CH4"]
T_ISO = 1100.0
MET, CO = 10.0, 0.55


def bin_to_r(wl, y, r=R_BIN, lo=WL_MIN, hi=WL_MAX):
    edges = [lo]
    while edges[-1] < hi:
        edges.append(edges[-1] * (1.0 + 1.0 / r))
    edges = np.asarray(edges)
    idx = np.digitize(wl, edges)
    wl_b, y_b = [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() >= 3:
            wl_b.append(wl[m].mean())
            y_b.append(y[m].mean())
    return np.asarray(wl_b), np.asarray(y_b)


def main():
    from jwst_tool import forward, planets
    from jwst_tool import picaso_chem as pc
    from jwst_tool import picaso_env as pe

    # OFFLINE diagnostic: the isothermal tp_mode was removed from the tool, so
    # canonical_params runs under guillot here. The native/tool RT below is
    # still compared on the SAME manually-built isothermal T_ISO column
    # (p_bar + T arrays constructed directly), so the parity state is unchanged;
    # only the canonical_params tp_mode label differs. (REPORT.md still describes
    # the isothermal comparison state.)
    cp = forward.canonical_params(dict(
        chem_provider="picaso", tp_mode="guillot",
        met_x_solar=MET, co_ratio=CO))

    # --- the ONE shared state ----------------------------------------------
    p_bar = np.logspace(-6.0, 1.0, 90)
    T = np.full_like(p_bar, T_ISO)
    state = pc.evaluate(MET, CO, T, p_bar)
    sid = {s: i for i, s in enumerate(state.species)}
    gas = np.ones(len(state.species))
    gas[sid[pc.GRAPHITE_OUT]] = 0.0
    ymix = state.y * gas[None, :]
    ymix = ymix / ymix.sum(axis=1, keepdims=True)

    # --- (a) tool ExoJax RT -------------------------------------------------
    from jwst_tool import engine_config as rf_config
    from vulcan_forward import vulcan_chem  # noqa: F401 (x64 init)
    from vulcan_forward import exojax_rt, interp_map
    import jax.numpy as jnp

    profile = forward._rt_profile_common(cp, rf_config)
    rt = exojax_rt.build_rt_model(profile)
    to_art = interp_map.make_to_art(p_bar, rt.p_art_bar)
    vmr = {k: to_art(jnp.asarray(ymix[:, sid[k]])) for k in MOLS}
    mmw = to_art(jnp.asarray(ymix @ state.species_masses))
    T_art = jnp.full(rt.p_art_bar.shape, T_ISO)
    d_tool = np.asarray(rt.transmission_depth_r(
        vmr, to_art(jnp.asarray(ymix[:, sid["H2"]])), T_art, mmw,
        jnp.asarray(0.0), vmr_he=to_art(jnp.asarray(ymix[:, sid["He"]]))))
    wl_tool = np.asarray(rt.wl_um)

    # --- (b) picaso native RT ----------------------------------------------
    jdi = pe.import_picaso()
    import astropy.units as u
    import pandas as pd

    db = pe.native_opacity_path()
    opa = jdi.opannection(filename_db=str(db),
                          wave_range=[WL_MIN - 0.2, WL_MAX + 0.5])
    case = jdi.inputs(calculation="planet")
    case.approx(p_reference=1.0)
    case.phase_angle(0)
    # mass + radius, NEVER bare gravity: the native altitude integration
    # needs planet.mass (see the docstring; bare gravity -> all-NaN depths)
    rp_cm = cp["rp_rjup"] * planets.R_JUP_CM
    mp_g = cp["gs_cgs"] * rp_cm**2 / planets.G_CGS
    case.gravity(mass=mp_g, mass_unit=u.g,
                 radius=cp["rp_rjup"], radius_unit=u.R_jup)
    pe.bootstrap()
    case.star(opa, temp=cp["star_teff"] or 5485.0, metal=0.0, logg=4.5,
              radius=0.932, radius_unit=u.R_sun,
              semi_major=0.04828, semi_major_unit=u.AU,
              database="ck04models")
    df = pd.DataFrame({"pressure": p_bar, "temperature": T})
    for m in MOLS:
        df[m] = ymix[:, sid[m]]
    df["H2"] = ymix[:, sid["H2"]]
    df["He"] = ymix[:, sid["He"]]
    case.atmosphere(df=df)
    out = case.spectrum(opa, calculation="transmission", full_output=False)
    wl_nat = 1e4 / np.asarray(out["wavenumber"], float)
    d_nat = np.asarray(out["transit_depth"], float)

    # --- compare ------------------------------------------------------------
    wt, dt = bin_to_r(wl_tool, d_tool)
    wn, dn = bin_to_r(wl_nat, d_nat)
    dn_i = np.interp(wt, wn[np.argsort(wn)], dn[np.argsort(wn)])
    offset = float(np.median(dn_i - dt))
    resid = (dn_i - dt - offset) * 1e6                     # ppm
    stats = dict(
        offset_ppm=offset * 1e6,
        median_abs_ppm=float(np.median(np.abs(resid))),
        p95_abs_ppm=float(np.percentile(np.abs(resid), 95)),
        max_abs_ppm=float(np.max(np.abs(resid))),
        n_bins=int(wt.size))
    print(json.dumps(stats, indent=1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "parity_native_rt.npz",
                        wl=wt, depth_tool=dt, depth_native=dn_i,
                        resid_ppm=resid, stats_json=json.dumps(stats))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(wt, dt * 100, label="tool (ExoJax RT)", lw=1)
    ax[0].plot(wt, (dn_i - offset) * 100, label="picaso native RT "
               f"(offset {offset * 1e6:+.0f} ppm removed)", lw=1)
    ax[0].set_ylabel("transit depth [%]")
    ax[0].legend()
    ax[1].plot(wt, resid, lw=1)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("wavelength [um]")
    ax[1].set_ylabel("residual [ppm]")
    fig.suptitle("Native-PICASO vs ExoJax transmission on one identical "
                 f"state (R={R_BIN:.0f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "parity_native_rt.png", dpi=200)

    verdicts = [
        ("broadband offset |x| < 2000 ppm", abs(stats["offset_ppm"]) < 2000),
        ("median |resid| < 150 ppm", stats["median_abs_ppm"] < 150),
        ("p95 |resid| < 400 ppm", stats["p95_abs_ppm"] < 400)]
    lines = [
        "# Native-PICASO RT vs tool ExoJax RT: one-state parity",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by "
        "scripts/run_native_rt_parity.py. OFFLINE validation only; the "
        "production path is always provider chemistry + ExoJax. See the "
        "script docstring for the method and why exact agreement is not "
        "expected (different opacity sources + reference-radius "
        "conventions).",
        "",
        f"State: W39b geometry, isothermal {T_ISO:.0f} K, blended "
        f"equilibrium at {MET:g}x solar / C/O {CO:g}, absorbers "
        f"{MOLS} on H2/He. Native DB: {db.name}.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| broadband offset (removed) | {stats['offset_ppm']:+.0f} ppm |",
        f"| median abs residual | {stats['median_abs_ppm']:.0f} ppm |",
        f"| p95 abs residual | {stats['p95_abs_ppm']:.0f} ppm |",
        f"| max abs residual | {stats['max_abs_ppm']:.0f} ppm |",
        f"| bins (R={R_BIN:.0f}, {WL_MIN}-{WL_MAX} um) | {stats['n_bins']} |",
        "",
        "Targets (stated in the script docstring, findings not CI gates):",
        ""]
    lines += [f"- {'PASS' if ok else 'OUTSIDE TARGET'}: {label}"
              for label, ok in verdicts]
    lines += ["", "Figure: ../figs/parity_native_rt.png"]
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT_DIR / "REPORT.md")


if __name__ == "__main__":
    sys.exit(main())
