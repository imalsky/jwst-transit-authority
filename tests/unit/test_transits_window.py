"""transits_to_target under the diagonal noise model.

With no correlated-floor scenarios, the score is monotone in N (sigma_N =
max(sigma_random_N, floor)), so with a positive floor on every bin sig_inf is
an exact ceiling: a floored target above it short-circuits to unreachable,
and a reachable target returns the smallest count. No-floor runs report
sig_inf = inf (0 for a signal inside the nuisance span) and never a
floor-proven "unreachable"; a mixed floor (some bins at 0 ppm) reports
sig_inf = nan and only the scan decides.
"""
from __future__ import annotations

import numpy as np

from jwst_tool import detect


def _result(floor_ppm: float = 100.0) -> dict:
    n = 80
    wl = np.linspace(3.0, 5.0, n)
    sigma1 = np.full(n, 300e-6)                      # sigma at n_transits = 1
    bump = 150e-6 * np.exp(-0.5 * ((np.log(wl) - np.log(4.0)) / 0.10) ** 2)
    return dict(
        wl=wl, depth=0.02 + bump, depth_wo=np.full(n, 0.02),
        floor=np.full(n, floor_ppm * 1e-6), var_phot=sigma1 ** 2,
        n_transits_eval=1, seg=np.zeros(n, int),
    )


def _score(r, n):
    return detect.detection_significance(
        np.asarray(r["depth"]) - np.asarray(r["depth_wo"]),
        detect.sigma_at_transits(r, n),
        nuisance=detect._result_nuisance(r))


def test_floor_ceiling_short_circuits_and_smallest_n_is_returned():
    r = _result()
    sig_inf = detect.transits_to_target(r, 1e-9)["sig_inf"]
    tt = detect.transits_to_target(r, sig_inf * 1.01)
    assert tt == dict(n=None, reachable=False, sig_inf=sig_inf)
    # a reachable target straddles: score(n) >= target > score(n-1)
    target = _score(r, 6) * 0.999
    tt2 = detect.transits_to_target(r, target)
    assert tt2["reachable"]
    assert _score(r, tt2["n"]) >= target
    if tt2["n"] > 1:
        assert _score(r, tt2["n"] - 1) < target
    # a MIXED floor (one bin at 0 ppm): the floor-only limit is not
    # evaluated (nan) and the scan finds the one-transit answer (8.7 sigma)
    rm = dict(depth=np.array([1e-3, 0, 0, 0]), depth_wo=np.zeros(4),
              sigma=np.full(4, 1e-4), var_phot=np.full(4, 1e-8),
              floor=np.array([0.0, 1e-5, 1e-5, 1e-5]), n_transits_eval=1,
              seg=np.zeros(4, int))
    tt3 = detect.transits_to_target(rm, 5.0)
    assert tt3["reachable"] and tt3["n"] == 1 and np.isnan(tt3["sig_inf"])


def test_score_is_monotone_in_n_with_a_floor():
    r = _result()
    sc = [_score(r, n) for n in (1, 2, 4, 9, 16, 32, 64, 200, 500)]
    assert np.all(np.diff(sc) >= 0.0)


def test_no_floor_never_reports_floor_proven_unreachable():
    r = _result(floor_ppm=0.0)
    tt = detect.transits_to_target(r, 8.0)
    assert tt["sig_inf"] == float("inf")
    assert tt["reachable"] and tt["n"] is not None
    # a signal entirely inside the offset span scores 0 at every N: the
    # limit is 0, never "inf" with a failed 500-transit scan
    r0 = _result(floor_ppm=0.0)
    r0["depth"] = r0["depth_wo"] + 1e-3
    assert detect.transits_to_target(r0, 5.0) == \
        dict(n=None, reachable=False, sig_inf=0.0)
