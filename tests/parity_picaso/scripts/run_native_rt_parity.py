"""PICASO-native RT vs the tool's ExoJax RT on ONE identical atmosphere.

Offline CROSS-MODEL COMPARISON ONLY (the production path is always provider
chemistry + ExoJax; decision 2026-07-20). Requires the native opacity DB
(opacities/opacities/opacities_0.3_15_R15000.db) in the reference tree and
the full tool stack. Writes tests/parity_picaso/outputs/REPORT.json and a
PNG (a machine-readable artifact, not a doc -- see the writer's comment).

WHAT THIS IS NOT: it is not a parity check unless every declared target below
passes, and it does not validate absolute spectral agreement. The two codes use
different opacity sources, different broadening, and different reference-radius
conventions, so a disagreement here does not by itself identify a bug in either
-- and equally, a disagreement outside target must never be reported as parity.
`main()` returns nonzero when a target fails so a release gate can consume it;
`--diagnostic` writes the report and returns 0 for exploratory runs.

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
* gravity is NOT a difference between the two sides (corrected 2026-08-03;
  the previous text here claimed the tool used constant surface gravity, which
  stopped being true at the 2026-07-28 audit and left this docstring asserting
  a false explanation for the residuals). BOTH sides now integrate altitude on
  an inverse-square profile: picaso uses g(z) = GM/z^2 (mass+radius are
  REQUIRED -- passing gravity alone leaves planet.mass NaN and the native
  transmission silently returns all-NaN), and the tool's RT uses
  `vulcan_forward.exojax_rt._gravity_profile_invsq`, g(r) = g_btm*(R_btm/r)^2,
  consistently for both the chord heights and the pressure-to-column-mass
  conversion (pinned by vulcan-forward's tests/test_gravity_profile.py).
Targets: |offset| < 2000 ppm; median |residual| after offset removal
< 150 ppm; p95 < 400 ppm in 1-10 um.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
REPO_DIR = Path(__file__).resolve().parents[3]
R_BIN = 100.0
WL_MIN, WL_MAX = 1.0, 12.0
MOLS = ["H2O", "CO2", "CO", "CH4"]
T_ISO = 1100.0
MET, CO = 10.0, 0.55


def _git_head(path):
    """Short HEAD of the checkout containing `path`, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10)
        head = out.stdout.strip()
        if out.returncode != 0 or not head:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return None


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def collect_provenance(db_path):
    """Exact identity of everything that can move these numbers.

    Without this the archived artifact cannot be attributed to a code state,
    which is how the committed report survived the gravity change while still
    describing the pre-change behavior.
    """
    import platform

    prov = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vulcan_jwst_tool_commit": _git_head(REPO_DIR),
        "vulcan_jwst_tool_version": _pkg_version("vulcan-jwst-tool"),
        "vulcan_forward_version": _pkg_version("vulcan-forward"),
        "vulcan_jax_version": _pkg_version("vulcan-jax"),
        "exojax_version": _pkg_version("exojax"),
        "picaso_version": _pkg_version("picaso"),
        "numpy_version": _pkg_version("numpy"),
        "jax_version": _pkg_version("jax"),
        "native_opacity_db": str(db_path),
        "native_opacity_db_name": Path(db_path).name,
        "gravity_profile": "inverse-square on BOTH sides (see module docstring)",
    }
    try:
        from vulcan_forward import paths as _vf_paths
        prov["vulcan_forward_commit"] = _git_head(
            Path(_vf_paths.__file__).resolve().parents[3])
    except Exception:
        prov["vulcan_forward_commit"] = None

    # Installed metadata can be STALE relative to the checkout actually being
    # exercised (an editable install keeps the old dist-info version). Record
    # the imported package's own __version__ next to it so a mismatch is
    # visible in the artifact instead of silently mislabeling it.
    for pkg, key in (("jwst_tool", "vulcan_jwst_tool"),
                     ("vulcan_forward", "vulcan_forward"),
                     ("vulcan_jax", "vulcan_jax")):
        try:
            mod = __import__(pkg)
            prov[f"{key}_imported_version"] = getattr(mod, "__version__", None)
        except Exception:
            prov[f"{key}_imported_version"] = None
    prov["version_metadata_consistent"] = all(
        prov.get(f"{k}_imported_version") in (None, prov.get(f"{k}_version"))
        for k in ("vulcan_jwst_tool", "vulcan_forward", "vulcan_jax"))
    try:
        prov["native_opacity_db_bytes"] = Path(db_path).stat().st_size
    except OSError:
        prov["native_opacity_db_bytes"] = None
    return prov


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


def main(diagnostic: bool = False):
    from jwst_tool import forward, planets
    from jwst_tool import picaso_chem as pc
    from jwst_tool import picaso_env as pe

    # OFFLINE diagnostic: the isothermal tp_mode was removed from the tool, so
    # canonical_params runs under guillot here. The native/tool RT below is
    # still compared on the SAME manually-built isothermal T_ISO column
    # (p_bar + T arrays constructed directly), so the parity state is unchanged;
    # only the canonical_params tp_mode label differs. (REPORT.json still
    # records the isothermal comparison state.)
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
    prov = collect_provenance(db)
    print(json.dumps(stats, indent=1))
    print(json.dumps(prov, indent=1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "parity_native_rt.npz",
                        wl=wt, depth_tool=dt, depth_native=dn_i,
                        resid_ppm=resid, stats_json=json.dumps(stats),
                        provenance_json=json.dumps(prov))
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
    all_pass = all(ok for _label, ok in verdicts)

    # The VERDICT and the claim follow the measurement, not the other way
    # round. An artifact outside target is a cross-model discrepancy, and
    # saying "parity" on top of failing numbers is exactly what change 7 of the
    # 2026-08 handoff was written to stop.
    #
    # This writes REPORT.json, not a markdown doc (2026-08-16, 3-doc policy):
    # the repo carries README.md + notes.md + CLAUDE.md and nothing else, and
    # everything a reader needs about this comparison -- verdict, the measured
    # envelope, and the never-cite-it-as-validation rule -- is a README.md
    # section ("PICASO engine" + "Open gaps and accepted limitations"). Do not
    # reintroduce a .md artifact here; put new prose in the README instead.
    if all_pass:
        claim = (
            "PASS. Every declared target is met on this one state, so this "
            "artifact supports a one-state numerical parity claim between "
            "the native PICASO RT and the tool's ExoJax RT.")
    else:
        claim = (
            "FAIL (outside target). At least one declared target is not met. "
            "This artifact is a cross-model discrepancy record: it does NOT "
            "validate absolute spectral agreement and must not be described "
            "as parity, or cited as evidence that the consumers' physics is "
            "validated against real spectra. The two codes use different "
            "opacity sources, broadening, and reference-radius conventions, "
            "so the disagreement does not by itself identify a bug in "
            "either.")

    report = {
        "verdict": "PASS" if all_pass else "FAIL",
        "claim": claim,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "generated_by": "tests/parity_picaso/scripts/run_native_rt_parity.py",
        "note": (
            "OFFLINE comparison only; the production path is always provider "
            "chemistry + ExoJax. Method and why exact agreement is not "
            "expected: this script's docstring. Gravity is NOT one of the "
            "differences: both sides integrate altitude on an inverse-square "
            "profile. Scope and the standing rule: README.md."),
        "state": {
            "geometry": "W39b",
            "T_iso_K": T_ISO,
            "metallicity_x_solar": MET,
            "co_ratio": CO,
            "absorbers": list(MOLS),
            "background": "H2/He",
            "native_db": db.name,
        },
        "metrics_ppm": {
            "broadband_offset_removed": round(stats["offset_ppm"], 1),
            "median_abs_residual": round(stats["median_abs_ppm"], 1),
            "p95_abs_residual": round(stats["p95_abs_ppm"], 1),
            "max_abs_residual": round(stats["max_abs_ppm"], 1),
        },
        "binning": {"R": R_BIN, "wl_min_um": WL_MIN, "wl_max_um": WL_MAX,
                    "n_bins": stats["n_bins"]},
        "targets": [{"target": label, "met": bool(ok)}
                    for label, ok in verdicts],
        "provenance": {k: str(v) for k, v in sorted(prov.items())},
        "figure": "../figs/parity_native_rt.png",
    }
    (OUT_DIR / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n")
    print("wrote", OUT_DIR / "REPORT.json")

    if not all_pass:
        failed = [label for label, ok in verdicts if not ok]
        print("FAIL: outside declared target(s): " + "; ".join(failed),
              file=sys.stderr)
        if diagnostic:
            print("(--diagnostic: reporting 0 anyway; NOT a release gate pass)",
                  file=sys.stderr)
            return 0
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--diagnostic", action="store_true",
        help="write the report for a failing comparison and still exit 0. For "
             "exploration only -- a release gate must NOT pass this flag.")
    sys.exit(main(diagnostic=ap.parse_args().diagnostic))
