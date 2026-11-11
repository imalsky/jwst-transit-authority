"""Shared helpers of the GUI smoke tests: the app's path and a synthetic cached run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "src" / "jwst_tool" / "app.py"
GUI_SOURCES = [APP, *sorted(APP.parent.glob("gui_*.py"))]


def run_app():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    return at


def synthetic_out(science_mode="transmission", saturated=False, sigma_detect=0.0, n_transits=1, with_jac=False):
    """Minimal cached-result pair (out, out_meta) for post-run rendering.

    sigma_detect must stay 0 or above target so the render never calls
    detect.transits_to_target, which needs the full evaluate_mode payload.
    ``with_jac=True`` adds a two-parameter Jacobian (lnZ, dlnCO + lnR0) and a
    SECOND mode, so the Fisher table, combo builder, forecast posteriors and
    summary figure all render. ymix is the FULL network state, wider than the
    RT molecule list and in a different order -- the shape that catches a
    positional mixing-ratio mislabelling; it also drives the one log-x panel.
    """
    import json

    n = 40
    wl = np.linspace(1.0, 5.0, n)
    model = {
        "wl_um": wl,
        "depth": np.full(n, 0.021) + 1e-4 * np.sin(wl),
        "mols": np.array(["H2O", "CO2", "CO", "CH4", "SO2"], dtype="U8"),
        # depth_wo rows align with wo_mols, not mols
        "wo_mols": np.array(["H2O", "CO2", "CO", "CH4", "SO2"], dtype="U8"),
        "depth_wo": np.tile(np.full(n, 0.0208), (5, 1)),
        "T": np.full(30, 1100.0),
        "p_bar": np.logspace(-7, 0.8, 30),
        "ymix": np.tile(np.array([0.85, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0.14]), (30, 1)),
        "ymix_species": np.array(["H2", "CO", "H2O", "CH4", "CO2", "SO2", "He"], dtype="U16"),
        "science_mode": science_mode,
        "params_json": json.dumps(
            {"co_ratio": 0.549348, "tp_mode": "guillot", "met_x_solar": 10.0, "science_mode": science_mode}
        ),
        # every cached result carries the caveat fields (forward.py writes
        # them unconditionally); empty = nothing to report
        "unmodeled": np.array([], dtype="U32"),
        "photo_escalated": np.array([], dtype="U48"),
        "rate_rows_outside": np.array([0]),
        "rate_rows_total": np.array([0]),
        "unlisted_core": np.array([], dtype="U8"),
        "chem_provider": np.array("vulcan", dtype="U16"),
    }
    nb = 12
    rng = np.random.default_rng(0)

    def _mode(key, label, wl_lo, wl_hi):
        return {
            "mode_key": key,
            "label": label,
            "saturated": saturated,
            "sat_frac": 0.97,
            "sigma_detect": sigma_detect,
            "sigma_detect_proj": float("nan"),
            "wl": np.linspace(wl_lo, wl_hi, nb),
            "wl_eff": np.linspace(wl_lo, wl_hi, nb),
            "depth": np.full(nb, 0.021),
            "sigma": np.full(nb, 1.5e-4),
            "floor": np.zeros(nb),
            "seg": np.zeros(nb, int),
            "median_sigma_ppm": 150.0,
            "n_bins": nb,
            "ngroup": 12,
            "t_cycle_s": 11.0,
            "jac_bins": (rng.standard_normal((3, nb)) * 1e-4 if with_jac else None),
        }

    results = [_mode("nirspec_g395h", "NIRSpec G395H", 2.9, 5.1)]
    if with_jac:
        results.append(_mode("nirspec_prism", "NIRSpec PRISM", 0.7, 5.2))
        model["jac"] = np.zeros((3, n))
        model["jac_names"] = np.array(["lnZ", "dlnCO", "lnR0"], dtype="U8")
    out = dict(model=model, results=results, failed=[], unusable=[])
    # run_sig left empty: every non-canonical field reads as changed
    out_meta = dict(
        goal="detect",
        target="SO2",
        goal_param=None,
        target_prec=None,
        target_sig=3.0,
        n_transits=n_transits,
        r_bin=100,
        planet="WASP-39 b",
        sat_limit=0.8,
        run_sig=dict(goal={}, observation={}),
    )
    return out, out_meta


def run_with_result(out, out_meta, **state):
    """AppTest primed with a cached result (plus optional session state), run
    once. Post-run rendering only; no forward model is launched.
    """
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["out"] = out
    at.session_state["out_meta"] = out_meta
    for key, val in state.items():
        at.session_state[key] = val
    at.run()
    return at
