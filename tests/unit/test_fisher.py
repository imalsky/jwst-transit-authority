"""Rank-aware Fisher tests: duplicated Jacobian columns must be reported as
degenerate, near-singular matrices must not return arbitrary finite sigmas,
the well-conditioned case must match independent analytic oracles, and the
no-floor transits-to-target limit must be exact."""
import numpy as np
import pytest

from jwst_tool import fisher, posteriors


def test_marg_sigmas_matches_independent_oracles():
    """Marginalized sigmas against three independent references: the analytic
    inverse of a well-conditioned Fisher matrix, an SVD of the noise-whitened
    design, and (for the Fisher construction itself) the finite curvature of
    the Gaussian likelihood."""
    # analytic inverse + rank/condition diagnostics
    J = np.array([[1.0, 0.5, 0.2, 0.9],
                  [0.1, 1.3, 0.7, 0.2]])
    s = np.array([1.0, 2.0, 0.5, 1.5]) * 1e-4
    F = (J / s[None, :] ** 2) @ J.T
    diag = {}
    sig = fisher._marg_sigmas(F, 2, diag=diag)
    assert np.allclose(sig, np.sqrt(np.diag(np.linalg.inv(F))), rtol=1e-10)
    assert diag["fisher_rank"] == 2
    assert diag["condition_number"] < 1e6
    # SVD oracle on the noise-whitened design matrix
    rng = np.random.default_rng(42)
    design = rng.normal(size=(4, 30))
    sigma = np.linspace(0.7, 1.3, 30)
    whitened = (design / sigma[None, :]).T
    _u, singular, vt = np.linalg.svd(whitened, full_matrices=False)
    covariance = (vt.T / singular[None, :] ** 2) @ vt
    got = fisher._marg_sigmas(whitened.T @ whitened, 4)
    assert np.allclose(got, np.sqrt(np.diag(covariance)), rtol=1e-11)
    # the (J / sigma^2) J^T construction IS the likelihood curvature
    design2 = np.array([[0.4, -0.2, 0.7, 0.1],
                        [0.3, 0.8, -0.1, 0.5]])
    sigma2 = np.array([0.8, 1.1, 0.9, 1.3])

    def nll(theta):
        return 0.5 * np.sum(((theta @ design2) / sigma2) ** 2)

    step = 1e-4
    curvature = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.eye(2)[i] * step
            ej = np.eye(2)[j] * step
            curvature[i, j] = (nll(ei + ej) - nll(ei - ej)
                               - nll(-ei + ej) + nll(-ei - ej)) / (4.0 * step ** 2)
    expected = (design2 / sigma2[None, :] ** 2) @ design2.T
    assert np.allclose(curvature, expected, rtol=1e-9, atol=1e-10)
    # adding a known Gaussian prior as explicit precision reproduces the
    # analytic posterior covariance
    Jp = np.array([[1.0, 0.3], [0.2, 1.1], [0.5, 0.5]]).T
    sp = np.array([1e-4, 2e-4, 1.5e-4])
    Fp = (Jp / sp[None, :] ** 2) @ Jp.T
    prior_sig = np.array([0.05, 0.2])
    F_post = Fp + np.diag(1.0 / prior_sig ** 2)
    sig_post = fisher._marg_sigmas(F_post, 2)
    assert np.allclose(sig_post, np.sqrt(np.diag(np.linalg.inv(F_post))),
                       rtol=1e-10)


def test_degenerate_directions_read_inf_never_garbage():
    """Two identical Jacobian rows are a perfect degeneracy: both must come
    back inf, never a finite number. An ALMOST-duplicated row (relative
    difference 1e-8) is numerically unconstrained too; a plain np.linalg.inv
    would return a huge-but-finite 'constraint' without any error."""
    rng = np.random.default_rng(0)
    row = rng.standard_normal(50)
    other = rng.standard_normal(50)
    s = np.full(50, 1e-4)
    J = np.stack([row, row, other])
    F = (J / s[None, :] ** 2) @ J.T
    diag = {}
    sig = fisher._marg_sigmas(F, 3, diag=diag)
    assert np.isinf(sig[0]) and np.isinf(sig[1]) and np.isfinite(sig[2])
    assert diag["fisher_rank"] == 2
    assert diag["fisher_dimension"] == 3
    # near-duplicate row
    rng1 = np.random.default_rng(1)
    row1 = rng1.standard_normal(50)
    J2 = np.stack([row1, row1 * (1 + 1e-8), rng1.standard_normal(50)])
    F2 = (J2 / s[None, :] ** 2) @ J2.T
    sig2 = fisher._marg_sigmas(F2, 3)
    assert np.isinf(sig2[0]) and np.isinf(sig2[1]) and np.isfinite(sig2[2])
    # regression guard: plain inv would have "succeeded" silently
    assert np.all(np.isfinite(np.linalg.inv(F2)))


def test_uninformative_width_reads_unconstrained():
    """A C/O width past the no-information scale is not a constraint. The C/O
    display transform sigma_CO = C/O * sigma_lnCO is a linearization; at
    sigma_lnCO = 1 its minus-1-sigma edge sits exactly on the physical
    boundary C/O = 0. Past that the forecast must read unconstrained (inf)
    end to end -- reported width AND posterior curve -- never a finite number
    that looks like a weak measurement. Sigma scales linearly with the per-bin
    noise, so one design brackets the cut from both sides. Rows with an exact
    display transform (lnZ) are never cut, however wide."""
    rng = np.random.default_rng(7)
    nb = 40
    free = ["dlnCO", "lnZ"]
    base = dict(jac_bins=rng.standard_normal((3, nb)),   # [dlnCO, lnZ, lnR0]
                sigma=np.full(nb, 1e-4), mode_key="nirspec_prism",
                depth=np.full(nb, 0.02))
    s0 = fisher.mode_forecast(base, free)["dlnCO"]
    assert np.isfinite(s0)
    for target, constrained in ((0.9, True), (1.1, False)):
        r = dict(base, sigma=base["sigma"] * (target / s0))
        sig = fisher.mode_forecast(r, free)
        assert bool(np.isfinite(sig["dlnCO"])) is constrained
        if constrained:
            assert sig["dlnCO"] == pytest.approx(target, rel=1e-9)
        assert np.isfinite(sig["lnZ"])
        out = posteriors.marginalized_posteriors(
            r, free, {"dlnCO": 0.55, "lnZ": 1.0}, co_eval=0.55)
        assert out["params"]["dlnCO"]["constrained"] is constrained
        assert (out["params"]["dlnCO"]["theta"] is None) is not constrained


def test_combined_forecast_structure_and_order_invariance():
    """Combined Fisher must allocate one offset column per segment of every
    mode (dimension check via the diag), and the result is invariant to mode
    order and parameter order."""
    rng = np.random.default_rng(4)
    r1 = dict(jac_bins=rng.standard_normal((2, 25)), sigma=np.full(25, 1e-4),
              seg=np.array([0] * 12 + [1] * 13))   # 2 segments
    r2 = dict(jac_bins=rng.standard_normal((2, 20)), sigma=np.full(20, 1e-4),
              seg=np.zeros(20, int))               # 1 segment
    diag = {}
    fisher.combined_forecast([r1, r2], ["p0"], diag=diag)
    # 1 free + lnR0 + (2 + 1) segment offsets = 5
    assert diag["fisher_dimension"] == 1 + 1 + 3
    rng = np.random.default_rng(14)
    r3 = dict(jac_bins=rng.normal(size=(3, 25)), sigma=np.linspace(1, 2, 25),
              seg=np.zeros(25, int))
    r4 = dict(jac_bins=rng.normal(size=(3, 31)), sigma=np.linspace(2, 1, 31),
              seg=np.array([0] * 15 + [1] * 16))
    base = fisher.combined_forecast([r3, r4], ["p0", "p1"])
    assert fisher.combined_forecast([r4, r3], ["p0", "p1"]) == \
        pytest.approx(base)
    swapped = [dict(r, jac_bins=np.asarray(r["jac_bins"])[[1, 0, 2]])
               for r in (r3, r4)]
    reordered = fisher.combined_forecast(swapped, ["p1", "p0"])
    assert reordered["p0"] == pytest.approx(base["p0"])
    assert reordered["p1"] == pytest.approx(base["p1"])


def test_mode_forecast_equals_combined_single_result():
    """mode_forecast(r) and combined_forecast([r]) must implement the SAME
    statistical model (free params + shared lnR0 + one constant offset per
    segment), INCLUDING the first segment's offset. A single-segment mode
    and a no-seg call agree too: both carry exactly one constant-offset
    nuisance, because lnR0 is a physical derivative, not a constant, and
    never stands in for the offset. Diag dimension = free + lnR0 + the
    always-present constant offset."""
    rng = np.random.default_rng(7)
    nb = 45
    seg = np.array([0] * 22 + [1] * 23)
    jac = rng.standard_normal((3, nb))             # 2 free + lnR0
    r = dict(jac_bins=jac, sigma=np.full(nb, 1e-4), seg=seg)
    a = fisher.mode_forecast(dict(r), ["p0", "p1"])
    b = fisher.combined_forecast([dict(r)], ["p0", "p1"])
    for k in ("p0", "p1"):
        if np.isinf(a[k]) or np.isinf(b[k]):
            assert np.isinf(a[k]) and np.isinf(b[k])
        else:
            assert a[k] == pytest.approx(b[k], rel=1e-10)
    # and the single-segment / no-seg variants agree too
    r1 = dict(jac_bins=jac, sigma=np.full(nb, 1e-4))
    a1 = fisher.mode_forecast(dict(r1), ["p0", "p1"])
    b1 = fisher.combined_forecast([dict(r1)], ["p0", "p1"])
    for k in ("p0", "p1"):
        assert a1[k] == pytest.approx(b1[k], rel=1e-10)
    r2 = dict(jac_bins=jac, sigma=np.full(nb, 1e-4), seg=np.zeros(nb, int))
    a2 = fisher.mode_forecast(dict(r2), ["p0", "p1"])
    for k in ("p0", "p1"):
        assert a2[k] == pytest.approx(a1[k], rel=1e-12)
    # diag passthrough on a minimal mode
    diag = {}
    out = fisher.mode_forecast(
        dict(jac_bins=np.array([[1.0, 2.0, 0.5], [0.2, 0.1, 0.9]]),
             sigma=np.array([1e-4, 1e-4, 1e-4])), ["p0"], diag=diag)
    assert set(out) == {"p0"} and np.isfinite(out["p0"])
    assert diag["fisher_dimension"] == 3 and diag["fisher_rank"] == 3


def test_constant_science_derivative_unconstrained():
    """An exactly CONSTANT science derivative must be absorbed by the constant
    calibration offset (unconstrained), even when the lnR0 derivative is NOT
    constant: lnR0 cannot stand in for the offset. The same holds per
    segment: a derivative constant WITHIN each segment (any step pattern,
    including a pure detector step) lies in the span of the per-segment
    offsets -- but only once segment info is supplied."""
    nb = 40
    x = np.linspace(0.0, 1.0, nb)
    jac = np.stack([np.ones(nb), 0.3 + 0.2 * x])   # [free=const, lnR0 nonconst]
    r = dict(jac_bins=jac, sigma=np.full(nb, 1e-4))
    assert np.isinf(fisher.mode_forecast(r, ["p0"])["p0"])
    # a science derivative with real shape stays constrained
    jac2 = np.stack([np.sin(6 * x), 0.3 + 0.2 * x])
    assert np.isfinite(fisher.mode_forecast(
        dict(jac_bins=jac2, sigma=np.full(nb, 1e-4)), ["p0"])["p0"])
    # per-segment constant derivative (general step pattern)
    seg = np.array([0] * 20 + [1] * 20)
    jac3 = np.stack([np.where(seg == 0, 0.7, -0.2), np.linspace(1, 2, nb)])
    r3 = dict(jac_bins=jac3, sigma=np.full(nb, 1e-4), seg=seg)
    assert np.isinf(fisher.mode_forecast(r3, ["p0"])["p0"])
    # a pure detector step: well constrained WITHOUT segment info, absorbed
    # (unconstrained) as soon as the two segment offsets are floated
    jac4 = np.stack([(seg == 1).astype(float), np.ones(nb)])
    base = dict(jac_bins=jac4, sigma=np.full(nb, 1e-4))
    assert np.isfinite(fisher.mode_forecast(dict(base), ["p0"])["p0"])
    assert np.isinf(fisher.mode_forecast(dict(base, seg=seg), ["p0"])["p0"])


def test_mode_forecast_matches_schur_complement():
    """Marginalized sigma against an independent Schur-complement GLS
    calculation on the full nuisance-augmented design."""
    rng = np.random.default_rng(11)
    nb = 60
    seg = np.array([0] * 30 + [1] * 30)
    jac = rng.standard_normal((3, nb))             # 2 free + lnR0
    sigma = np.full(nb, 2e-4)
    r = dict(jac_bins=jac, sigma=sigma, seg=seg)
    got = fisher.mode_forecast(dict(r), ["p0", "p1"])
    # independent construction: rows = [free(2), lnR0, seg0, seg1]
    rows = np.vstack([jac, (seg == 0).astype(float), (seg == 1).astype(float)])
    F = (rows / sigma[None, :] ** 2) @ rows.T
    n_f = 2
    A = F[:n_f, :n_f]
    B = F[:n_f, n_f:]
    D = F[n_f:, n_f:]
    S = A - B @ np.linalg.solve(D, B.T)            # Schur complement
    cov = np.linalg.inv(S)
    assert got["p0"] == pytest.approx(np.sqrt(cov[0, 0]), rel=1e-9)
    assert got["p1"] == pytest.approx(np.sqrt(cov[1, 1]), rel=1e-9)


def test_display_sigma_units():
    """Report-unit conventions: metallicity/Kzz in dex (log10), C/O as an
    ABSOLUTE number ratio (sigma_CO = C/O * sigma_lnCO), temperature in K."""
    # internal natural-log sigma of ln(10) -> exactly 1 dex
    assert fisher.display_sigma("lnZ", np.log(10.0)) == pytest.approx(1.0)
    assert fisher.display_sigma("lnKzz", np.log(10.0)) == pytest.approx(1.0)
    # C/O: absolute ratio, scaled by the atmosphere's C/O
    assert fisher.display_sigma("dlnCO", 0.1, co_eval=0.5) == pytest.approx(0.05)
    # dlnCO WITHOUT co_eval is refused loudly -- never a silent wrong scale
    with pytest.raises(ValueError):
        fisher.display_sigma("dlnCO", 0.1)
    # a plain temperature is unit-1 (K in, K out)
    assert fisher.display_sigma("Tirr", 42.0) == 42.0


def test_conditional_sigmas():
    """Conditional (others fixed) <= marginalized (others free), always; the
    two coincide when the parameter rows are orthogonal; a zero-response
    parameter reads inf; a second mode never worsens the conditional. All
    read off the SAME nuisance-augmented Fisher matrix."""
    rng = np.random.default_rng(7)
    nb = 40
    jac = np.vstack([rng.standard_normal((2, nb)),
                     rng.standard_normal(nb) + 1.0])   # p0, p1, lnR0
    r = dict(jac_bins=jac, sigma=np.full(nb, 1e-4))
    cond = {}
    marg = fisher.mode_forecast(r, ["p0", "p1"], conditional=cond)
    for n in ("p0", "p1"):
        assert np.isfinite(cond[n]) and cond[n] > 0
        assert cond[n] <= marg[n] * (1 + 1e-12)
    # orthogonal, zero-mean parameter rows (also orthogonal to the constant
    # offset and to lnR0): F is diagonal on the report block, so conditional
    # and marginalized coincide
    p0 = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    p1 = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    ln_r0 = np.array([1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
    r2 = dict(jac_bins=np.vstack([p0, p1, ln_r0]), sigma=np.ones(8))
    cond2 = {}
    marg2 = fisher.mode_forecast(r2, ["p0", "p1"], conditional=cond2)
    for n in ("p0", "p1"):
        assert np.isclose(cond2[n], marg2[n], rtol=1e-12)
    # a dead parameter (zero response) reads inf
    jac3 = np.vstack([np.zeros(10), np.ones(10)])       # p0 dead, lnR0 alive
    cond3 = {}
    fisher.mode_forecast(dict(jac_bins=jac3, sigma=np.ones(10)), ["p0"],
                         conditional=cond3)
    assert cond3["p0"] == np.inf
    # two modes must beat (or match) either alone, conditionally too
    rng = np.random.default_rng(3)
    ra = dict(jac_bins=rng.standard_normal((2, 25)), sigma=np.full(25, 1e-4))
    rb = dict(jac_bins=rng.standard_normal((2, 20)), sigma=np.full(20, 1e-4))
    c1, c12 = {}, {}
    fisher.combined_forecast([ra], ["p0"], conditional=c1)
    fisher.combined_forecast([ra, rb], ["p0"], conditional=c12)
    assert c12["p0"] <= c1["p0"] * (1 + 1e-12)


# --- fisher no-floor limits ---------------------------------------------------

def _fisher_result(floor_ppm: float, n=60) -> dict:
    wl = np.linspace(3.0, 5.0, n)
    bump = 150e-6 * np.exp(-0.5 * ((np.log(wl) - np.log(4.0)) / 0.10) ** 2)
    rng = np.random.default_rng(0)
    jac = np.vstack([1e-4 * np.sin(2.0 * wl), 1e-4 * np.cos(3.0 * wl),
                     1e-4 * rng.normal(size=n)])
    r = dict(wl=wl, depth=0.02 + bump, depth_wo=np.full(n, 0.02),
             floor=np.full(n, floor_ppm * 1e-6),
             var_phot=np.full(n, 300e-6) ** 2, n_transits_eval=1,
             seg=np.zeros(n, int),
             jac_bins=jac, jac_names=["lnZ", "lnKzz", "lnR0"])
    r["sigma"] = np.maximum(np.sqrt(r["var_phot"]), r["floor"])
    return r


def test_fisher_transits_to_target_limits():
    """Precision improves without bound with no floor, so the display-unit
    limit is exactly 0.0 (never a finite clip artifact like 1e-26) and the
    scan stays monotone; a floored result keeps a finite positive limit."""
    names = ["lnZ", "lnKzz", "lnR0"]
    tt = fisher.transits_to_target(_fisher_result(0.0), names, "lnZ", 1e9)
    assert tt["sig_inf"] == 0.0
    assert tt["reachable"]
    tt2 = fisher.transits_to_target(_fisher_result(100.0), names, "lnZ", 1e9)
    assert np.isfinite(tt2["sig_inf"]) and tt2["sig_inf"] > 0.0


def test_co_width_keeps_the_dex_and_gates_only_the_physical_range():
    """dex is always reported when finite -- a 0.4 dex width IS a weak local
    constraint, not an absence of one. Only the physical range is gated, on
    whether the center AND the interval stay inside the network's supported
    C/O band."""
    bounds = (0.1, 0.99)                     # sncho, photolysis on
    lo, hi = fisher.co_interval(0.55, 0.0745, k=1.0)
    assert (lo, hi) == pytest.approx((0.55 * np.exp(-0.0745),
                                      0.55 * np.exp(0.0745)), rel=1e-12)
    # informative mode: range fits at 1 and 3 sigma
    assert fisher.format_co_width(0.55, 0.0745, bounds) == \
        "±0.0324 dex (C/O 0.511–0.593)"
    assert fisher.format_co_width(0.55, 0.0745, bounds, k=3.0,
                                  qualify_coord=True) == \
        "±0.0971 dex in log10(C/O) (C/O 0.44–0.688)"
    # MIRI LRS: the width survives, the range does not
    assert fisher.format_co_width(0.55, 0.9218, bounds) == "±0.4 dex (local)"
    assert fisher.format_co_width(0.55, 0.9218, bounds, k=3.0,
                                  qualify_coord=True) == \
        "±1.2 dex in log10(C/O) (local)"
    # plain text only: this string also lands in a Streamlit cell and a CSV
    assert "$" not in fisher.format_co_width(0.55, 0.9218, bounds)


def test_format_co_width_accepts_an_out_of_domain_mock_center():
    """mock_center_co shifts a recovered center MULTIPLICATIVELY, so a broad
    mode legitimately lands outside the solver band. That is a local result,
    never an error -- raising there would crash a valid MIRI realization."""
    assert fisher.format_co_width(1.27, 0.9218, (0.1, 0.99)) == \
        "±0.4 dex (local)"
    assert fisher.format_co_width(0.05, 0.05, (0.1, 0.99)) == \
        "±0.0217 dex (local)"


@pytest.mark.parametrize("bounds,center", [((0.99, 0.1), 0.55),
                                           ((0.0, 0.99), 0.55),
                                           ((0.1, np.inf), 0.55),
                                           ((0.1, 0.99), 0.0),
                                           ((0.1, 0.99), np.nan)])
def test_format_co_width_validates_bounds_and_center(bounds, center):
    """bounds finite, positive and ordered; center finite and positive. The
    center is NOT required to lie inside the bounds."""
    with pytest.raises(ValueError):
        fisher.format_co_width(center, 0.1, bounds)
