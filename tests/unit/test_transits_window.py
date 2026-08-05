"""transits_to_target under correlated scenarios.

The floor-excess systematic grows as photon noise averages down, so
sigma_detect(N) can PEAK at a finite N; the old `target > sig_inf ->
unreachable` gate returned false "never"s. Pinned: the correlated path scans
and reports the window; the random path still short-circuits on its ceiling.
"""
from __future__ import annotations

import numpy as np

from jwst_tool import detect


def _result(scenario: str) -> dict:
    n = 80
    wl = np.linspace(3.0, 5.0, n)
    floor = np.full(n, 100e-6)
    sigma1 = np.full(n, 300e-6)                      # sigma at n_transits = 1
    bump = 150e-6 * np.exp(-0.5 * ((np.log(wl) - np.log(4.0)) / 0.10) ** 2)
    return dict(
        wl=wl, depth=0.02 + bump, depth_wo=np.full(n, 0.02),
        floor=floor, var_phot=sigma1 ** 2, n_transits_eval=1,
        scenario=scenario, seg=np.zeros(n, int),
        slope_rows=np.zeros((0, n)),
    )


def _score(r, n):
    return detect.detection_significance(
        np.asarray(r["depth"]) - np.asarray(r["depth_wo"]),
        detect.sigma_at_transits(r, n),
        nuisance=detect._result_nuisance(r),
        cov=detect.cov_at_transits(r, n))


def test_correlated_scenario_reachable_window_not_gated_by_sig_inf():
    r = _result("conservative")
    # pick a target above the floor-only limit but below the finite-N peak
    sig_inf = detect.detection_significance(
        np.asarray(r["depth"]) - np.asarray(r["depth_wo"]),
        np.maximum(np.asarray(r["floor"]), 1e-30),
        nuisance=detect._result_nuisance(r),
        cov=detect.cov_at_transits(r, 1, floor_only=True))
    peak = max(_score(r, n) for n in (1, 2, 4, 9, 16, 32, 64))
    assert peak > sig_inf, "reproducer must be non-monotone (peak above limit)"
    target = 0.5 * (sig_inf + peak)
    tt = detect.transits_to_target(r, target)
    # the OLD gate returned reachable=False here because target > sig_inf
    assert target > tt["sig_inf"]
    assert tt["reachable"] and tt["n"] is not None
    assert _score(r, tt["n"]) >= target
    if tt["n"] > 1:
        assert _score(r, tt["n"] - 1) < target       # smallest such n
    # finite window: the largest scanned count still meeting the target
    assert tt["n_last"] is not None and tt["n_last"] >= tt["n"]
    if tt["n_last"] < detect.N_TRANSITS_CAP:
        assert _score(r, tt["n_last"] + 1) < target


def test_random_scenario_gate_still_short_circuits():
    r = _result("random")
    sig_inf = detect.transits_to_target(r, 1e-9)["sig_inf"]
    tt = detect.transits_to_target(r, sig_inf * 1.01)
    assert tt == dict(n=None, n_last=None, reachable=False, sig_inf=tt["sig_inf"])
    # a reachable target straddles: score(n) >= target > score(n-1)
    target = _score(r, 6) * 0.999
    tt2 = detect.transits_to_target(r, target)
    assert tt2["reachable"] and tt2["n_last"] is None
    assert _score(r, tt2["n"]) >= target
    if tt2["n"] > 1:
        assert _score(r, tt2["n"] - 1) < target
