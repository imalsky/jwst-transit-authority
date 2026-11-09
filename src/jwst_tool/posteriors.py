"""Marginalized Fisher-Gaussian forecast densities + named mode combos.

Method and contract: docs/forecast_statistics.md, section `posteriors`.
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

import numpy as np

from jwst_tool import fisher

if TYPE_CHECKING:
    from typing import Any

# Auto curve grid: +/- _AUTO_SIGMA widths, _AUTO_POINTS samples. Shared by
# both curve builders and by the off-scale test in mock_center_co.
_AUTO_SIGMA, _AUTO_POINTS = 5.0, 201

# Machine token + human sentence, carried on every record this module
# returns, so every downstream surface (GUI, exports) labels the curves
# with the same wording instead of re-deriving it.
FORECAST_KIND = "fisher_gaussian_forecast"
FORECAST_LABEL = (
    "Linearized Fisher (Cramer-Rao) forecast: a Gaussian approximation "
    "in the fitted parameter coordinate, centered on the input model with "
    "the marginalized 1-sigma lower bound as width, under the quoted noise "
    "model. C/O is fitted in ln(C/O) and shown as a density per d ln(C/O) "
    "against a log C/O axis, with its width in dex; that width is a LOCAL "
    "quadratic expansion at the input C/O. Not a sampled posterior; a "
    "retrieval freeing more parameters under the same assumptions usually "
    "reports lower significance."
)

_SATURATED_REASON = "saturated at the shortest ramp tried (unusable data)"


def _param_unit(name: str) -> str:
    """Display unit for ``name``, from the single source of truth (fisher.PARAM_UNITS).

    dlnCO reports "C/O ratio" (its display transform is the absolute number
    ratio). A name absent from the table is a caller-defined row already in
    its own display units (display_sigma passes it through unchanged), so it
    reports unit "" -- same convention, not a fallback.
    """
    if name == "dlnCO":
        return "C/O ratio"
    return fisher.PARAM_UNITS.get(name, "")


def _validate_result(r: dict, n_free: int, where: str) -> None:
    """Fail fast on a malformed/non-finite mode result (house style)."""
    for key in ("jac_bins", "sigma"):
        if key not in r:
            raise ValueError(f"{where}: result is missing {key!r} -- pass the evaluate_mode output (with Jacobians)")
    J = np.asarray(r["jac_bins"], float)
    s = np.asarray(r["sigma"], float)
    if J.ndim != 2:
        raise ValueError(f"{where}: jac_bins must be 2-D (n_par, n_bins), got shape {J.shape}")
    if J.shape[0] != n_free + 1:
        raise ValueError(
            f"{where}: jac_bins has {J.shape[0]} rows but free_names has "
            f"{n_free} entries -- rows must be [free..., lnR0] "
            f"({n_free + 1} rows)"
        )
    if s.ndim != 1 or s.size != J.shape[1]:
        raise ValueError(f"{where}: sigma shape {s.shape} does not match jac_bins columns ({J.shape[1]})")
    if not np.all(np.isfinite(J)):
        raise ValueError(f"{where}: jac_bins contains non-finite values")
    if not np.all(np.isfinite(s)) or not np.all(s > 0.0):
        raise ValueError(f"{where}: sigma must be finite and > 0 everywhere")


def gaussian_curve(center: float, sigma: float) -> dict:
    """Return normalized Gaussian pdf N(center, sigma) on the auto grid center +/- _AUTO_SIGMA*sigma.

    Returns dict(theta, pdf); pdf integrates to ~1 over the grid. All
    inputs must be finite (sigma > 0) -- an inf/NaN sigma is an unconstrained
    direction and must be handled by the caller, never plotted.
    """
    center = float(center)
    sigma = float(sigma)
    if not np.isfinite(center):
        raise ValueError(f"gaussian_curve: center must be finite, got {center!r}")
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError(
            f"gaussian_curve: sigma must be finite and > 0, got {sigma!r} -- "
            "an unconstrained (inf) direction has no curve by design"
        )
    theta = np.linspace(center - _AUTO_SIGMA * sigma, center + _AUTO_SIGMA * sigma, _AUTO_POINTS)
    # The grid collapses (repeats a value) when the window is thinner than
    # one float64 step at the center -- which is what a center shifted far
    # from the point where its sigma was linearized produces.
    if np.any(np.diff(theta) == 0.0):
        raise ValueError(
            f"gaussian_curve: the window around center {center!r} is "
            f"unresolvable in float64 at sigma {sigma!r} -- the grid repeats "
            "values. A center shifted far from where its sigma was "
            "linearized is the usual cause."
        )
    z = (theta - center) / sigma
    pdf = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
    return dict(theta=theta, pdf=pdf)


def ln_gaussian_curve(center: float, sigma_ln: float, bounds: tuple | None = None) -> dict:
    """Return the normalized marginalized Fisher forecast density in ln(theta), on a geometric theta grid.

    The curve for a LOG-scaled physical axis.

    We CHOOSE to show density per unit d ln(theta) -- a logarithmic axis does
    not by itself fix the measure -- and therefore apply NO 1/theta Jacobian
    to the ordinate. The curve peaks at ``center`` and is symmetric in
    ln theta, on the grid center*exp(+-_AUTO_SIGMA*sigma_ln). This builder
    returns a UNIT-AREA density in d ln(theta); summary_figure divides by
    the peak to render it as relative forecast density.

    ``bounds`` (lo, hi) clips the window to the range the caller can
    actually produce -- for C/O, the range the forward model solves. A wide
    forecast otherwise spans decades of theta the model cannot reach, and past
    sigma_ln 142 the auto grid overflows float64 outright. The density is
    unchanged: only the drawn window moves, so the curve runs off the panel
    edge instead of being rescaled. Bounds that exclude ``center`` are ignored
    (the peak must stay in the window) -- callers keep their centers in range
    instead, which is what mock_center_co's range test is for.
    """
    center = float(center)
    sigma_ln = float(sigma_ln)
    if not (np.isfinite(center) and center > 0.0):
        raise ValueError(f"ln_gaussian_curve: center must be finite and > 0, got {center!r}")
    if not (np.isfinite(sigma_ln) and sigma_ln > 0.0):
        raise ValueError(
            f"ln_gaussian_curve: sigma_ln must be finite and > 0, got "
            f"{sigma_ln!r} -- an unconstrained (inf) direction has no curve "
            "by design"
        )
    u_lo, u_hi = -_AUTO_SIGMA * sigma_ln, _AUTO_SIGMA * sigma_ln
    clip = None
    if bounds is not None:
        b_lo, b_hi = float(bounds[0]), float(bounds[1])
        if b_lo <= center <= b_hi:
            lnc = float(np.log(center))
            u_lo = max(u_lo, float(np.log(b_lo)) - lnc)
            u_hi = min(u_hi, float(np.log(b_hi)) - lnc)
            clip = (b_lo, b_hi)
    with np.errstate(over="ignore"):
        theta = center * np.exp(np.linspace(u_lo, u_hi, _AUTO_POINTS))
    if clip is not None:
        # exp(log(hi) - log(center)) * center lands a few ulp outside;
        # the window is a promise, so make the endpoints exact
        theta = np.clip(theta, *clip)
    if not np.all(np.isfinite(theta)) or np.any(theta <= 0.0):
        raise ValueError(
            f"ln_gaussian_curve: the grid around center {center!r} at sigma_ln "
            f"{sigma_ln!r} is unresolvable in float64 or not positive"
        )
    if np.any(np.diff(theta) <= 0.0):
        raise ValueError("ln_gaussian_curve: grid must be strictly ascending")
    z = np.log(theta / center) / sigma_ln
    pdf = np.exp(-0.5 * z * z) / (sigma_ln * np.sqrt(2.0 * np.pi))
    return dict(theta=theta, pdf=pdf)


def _marginal_record(name: str, center: float, s_int: float, c_int: float, co_eval: float | None) -> dict:
    """One parameter's record; a null Fisher direction (inf sigma) is explicitly unconstrained, with no curve."""
    s_disp = fisher.display_sigma(name, s_int, co_eval=co_eval)
    rec: dict[str, Any] = dict(
        center=float(center),
        sigma_display=float(s_disp),
        sigma_conditional_display=float(fisher.display_sigma(name, c_int, co_eval=co_eval)),
        unit=_param_unit(name),
    )
    if not np.isfinite(s_disp):
        rec.update(constrained=False, theta=None, pdf=None, curve_family=None, sigma_ln=None)
        return rec
    if name == "dlnCO":
        # the density in its OWN coordinate, for the panel's log C/O axis; sigma_display stays the absolute-C/O width
        # that meta["target_prec"] is compared against
        curve = ln_gaussian_curve(rec["center"], s_int, bounds=co_curve_bounds())
        rec.update(curve_family="ln_gaussian", sigma_ln=s_int)
    else:
        curve = gaussian_curve(rec["center"], s_disp)
        rec.update(curve_family="gaussian", sigma_ln=None)
    rec.update(constrained=True, theta=curve["theta"], pdf=curve["pdf"])
    return rec


def marginalized_posteriors(
    results: dict | list[dict],
    free_names: list[str],
    centers: dict,
    params: list[str] | None = None,
    co_eval: float | None = None,
) -> dict:
    """1D marginalized Fisher-Gaussian forecast densities for chosen parameters.

    ``results`` is one mode result dict or a list treated as a combination, through ``fisher.combined_forecast``.
    ``centers`` maps every requested parameter to its finite input value in DISPLAY units. ``co_eval`` is the
    atmosphere's C/O, required when dlnCO is requested. Returned record: docs/forecast_statistics.md, section
    `posteriors`.
    """
    rlist = [results] if isinstance(results, dict) else list(results)
    if not rlist:
        raise ValueError("marginalized_posteriors: results is empty")
    if not free_names:
        raise ValueError("marginalized_posteriors: free_names is empty")
    for i, r in enumerate(rlist):
        _validate_result(r, len(free_names), f"marginalized_posteriors results[{i}]")
    if params is None:
        params = list(free_names)
    unknown = [p for p in params if p not in free_names]
    if unknown:
        raise ValueError(f"marginalized_posteriors: params {unknown} are not in free_names {list(free_names)}")
    missing = [p for p in params if p not in centers]
    if missing:
        raise ValueError(f"marginalized_posteriors: centers is missing {missing} (theta_input in display units)")
    for p in params:
        if not np.isfinite(float(centers[p])):
            raise ValueError(f"marginalized_posteriors: centers[{p!r}] must be finite, got {centers[p]!r}")

    cond: dict = {}
    sig = fisher.combined_forecast(rlist, list(free_names), conditional=cond)

    out_params = {
        name: _marginal_record(name, float(centers[name]), float(sig[name]), float(cond[name]), co_eval)
        for name in params
    }

    return dict(
        kind=FORECAST_KIND,
        label=FORECAST_LABEL,
        n_modes=len(rlist),
        co_eval=co_eval,
        sigma_marginalized={n: float(sig[n]) for n in free_names},
        sigma_conditional={n: float(cond[n]) for n in free_names},
        params=out_params,
    )


# Named mode combinations


def combo_forecast(
    name: str, mode_keys: list[str], results_by_mode: dict, free_names: list[str], co_eval: float | None = None
) -> dict:
    """Forecast one named combination of instrument modes.

    Saturation policy is exactly the "All usable modes" row's: a saturated mode is unusable data, excluded from the
    Fisher sum, and the exclusion is disclosed in the record. Goes through ``fisher.combined_forecast`` for any
    usable-mode count, so a single-mode combo equals ``fisher.mode_forecast``. Curves are the caller's job
    (``marginalized_posteriors``).

    Parameters
    ----------
    name : str
        The user's label for the combination (e.g. "SOSS + G395H").
    mode_keys : list of str
        Registry keys (instruments.MODES), without repeats.
    results_by_mode : dict
        {mode_key: evaluate_mode result with Jacobians}; every requested key must be present.
    free_names : list of str
        The free parameters, in Jacobian row order.
    co_eval : float or None
        The atmosphere's C/O; required when dlnCO is free.

    Returns
    -------
    dict
        ``name``, ``kind``, ``label``, ``mode_keys``, ``usable_modes``, ``excluded`` (mode and reason),
        ``n_modes_usable``, ``co_eval``, ``sigma_marginalized_display`` and ``sigma_conditional_display`` (display
        units, per parameter) and ``units``.

    Raises
    ------
    ValueError
        A blank name, an empty, repeated or unknown mode key, a missing result, or no usable mode.
    """
    from jwst_tool import instruments as ins  # data-root resolution stays lazy

    if not isinstance(name, str) or not name.strip():  # type: ignore[redundant-expr]  # a runtime guard for untyped callers
        raise ValueError(f"combo_forecast: combo name must be a non-empty string, got {name!r}")
    mode_keys = list(mode_keys)
    if not mode_keys:
        raise ValueError(f"combo {name!r}: mode_keys is empty")
    if len(set(mode_keys)) != len(mode_keys):
        raise ValueError(
            f"combo {name!r}: duplicate mode keys in {mode_keys} -- a mode cannot enter the Fisher sum twice"
        )
    bad = [k for k in mode_keys if k not in ins.MODES]
    if bad:
        raise ValueError(f"combo {name!r}: unknown mode key(s) {bad}; valid keys are {sorted(ins.MODES)}")
    missing = [k for k in mode_keys if k not in results_by_mode]
    if missing:
        raise ValueError(
            f"combo {name!r}: no result for mode(s) {missing} -- run the modes first and pass their result dicts"
        )

    usable, excluded = [], []
    for k in mode_keys:
        r = results_by_mode[k]
        if bool(r["saturated"]):
            excluded.append(dict(mode_key=k, reason=_SATURATED_REASON))
        else:
            usable.append(k)
    if not usable:
        raise ValueError(
            f"combo {name!r}: no usable mode -- all of {mode_keys} are "
            "saturated on this star. There is no forecast to report; pick "
            "different modes or a different target."
        )
    rlist = [results_by_mode[k] for k in usable]
    for k, r in zip(usable, rlist, strict=True):
        _validate_result(r, len(free_names), f"combo {name!r} mode {k!r}")

    cond: dict = {}
    sig = fisher.combined_forecast(rlist, list(free_names), conditional=cond)
    sig_disp = {n: float(fisher.display_sigma(n, float(sig[n]), co_eval=co_eval)) for n in free_names}
    cond_disp = {n: float(fisher.display_sigma(n, float(cond[n]), co_eval=co_eval)) for n in free_names}

    return dict(
        name=name,
        kind=FORECAST_KIND,
        label=FORECAST_LABEL,
        mode_keys=mode_keys,
        usable_modes=usable,
        excluded=excluded,
        n_modes_usable=len(usable),
        co_eval=co_eval,
        sigma_marginalized_display=sig_disp,
        sigma_conditional_display=cond_disp,
        units={n: _param_unit(n) for n in free_names},
    )


# Mock-observation layer: generated AFTER the forward model and the noise
# model, never inside them. mock_realization draws it; mock_recovery fits it
# (the forecast itself never sees the draw).

MOCK_KIND = "mock_observation_realization"
# The disclosure exists in exactly two forms, reused everywhere it is
# needed (a short one for controls and exports, a full one for provenance
# records) so the wording cannot drift between call sites.
MOCK_SHORT_LABEL = (
    "One seeded realization of the adopted effective diagonal noise model; "
    "the forecast numbers are realization-independent."
)

MOCK_LABEL = (
    "One simulated noise realization: the binned noiseless model plus one "
    "seeded Gaussian draw per bin at each mode's final per-bin uncertainty "
    "(floor included). A single realization can be lucky or unlucky. The "
    "draw never enters the detection or Fisher scores, the forecast widths, "
    "the caches, or the noiseless result exports -- every quoted precision "
    "and conditional template S/N is computed from the noiseless model and "
    "is independent of the draw by construction -- but it does drive the "
    "optional linearized recovery overlay (mock_recovery)."
)

MOCK_RECOVERY_KIND = "mock_recovery"
MOCK_RECOVERY_LABEL = (
    "Recovered parameter shift for one simulated noise realization under "
    "the same linearization as the forecast (delta_theta = F^+ J^T "
    "Sigma^-1 n, with the shared lnR0 and per-segment offset nuisances in "
    "the fit). The shift has zero mean and the marginalized Fisher "
    "covariance over realizations; it is not a retrieval result."
)


# Version tag for the seed -> stream mapping below. Bump if the mapping ever
# changes (different hash, different generator), so an archived seed +
# scheme version pins the exact realization.
SEED_SCHEME = "v1:default_rng([seed, crc32(mode_key)])"


def _mode_stream_seed(seed: int, mode_key: str) -> list[int]:
    """Deterministic per-mode RNG seed sequence.

    The draw for a mode depends only on (seed, mode_key), never on which other modes were run or on the selection
    order. crc32 is platform-stable.
    """
    return [int(seed), zlib.crc32(mode_key.encode())]


def mock_realization(results: list, seed: int) -> dict:
    """Draw one seeded mock observation over the run's evaluated modes.

    The same seed reproduces the identical realization (numpy ``default_rng`` with a per-mode substream; global
    ``np.random`` state is never touched). The draw is always at the mode's own per-bin sigma -- the same
    uncertainty the error bars, the conditional template S/N and the Fisher
    forecast use. There is deliberately no scale factor: a 0.5x or 2x draw
    is not a realization of the model whose error bars are plotted beside
    it. To change the assumed uncertainty, change the noise model (the
    per-mode random-noise multiplier), which propagates to every number.

    Parameters
    ----------
    results : list of dict
        ``detect.evaluate_mode`` results, each with ``mode_key``, ``depth`` [n_bins] and ``sigma`` [n_bins].
    seed : int
        Non-negative seed.

    Returns
    -------
    dict
        ``kind``, ``label`` (what the draw may and may not enter: MOCK_LABEL), ``seed``, ``seed_scheme``,
        ``numpy_version`` and ``modes``: per mode ``noise`` [n_bins], the N(0, sigma_i) draw at the mode's final
        per-bin depth error (floor included), ``depth_mock = depth + noise`` and ``sigma``.

    Raises
    ------
    ValueError
        A negative seed, no results, a missing or repeated mode key, or a misshapen or non-finite depth or sigma.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"mock_realization: seed must be >= 0, got {seed}")
    rlist = list(results)
    if not rlist:
        raise ValueError("mock_realization: results is empty")
    modes = {}
    for i, r in enumerate(rlist):
        key = r.get("mode_key")
        if not key:
            raise ValueError(f"mock_realization: results[{i}] has no mode_key -- pass evaluate_mode outputs")
        if key in modes:
            raise ValueError(f"mock_realization: duplicate mode_key {key!r}")
        d = np.asarray(r["depth"], float)
        s = np.asarray(r["sigma"], float)
        if d.ndim != 1 or d.shape != s.shape:
            raise ValueError(
                f"mock_realization: mode {key!r} depth/sigma shapes {d.shape}/{s.shape} must match and be 1-D"
            )
        if not np.all(np.isfinite(d)):
            raise ValueError(f"mock_realization: mode {key!r} depth has non-finite values")
        if not np.all(np.isfinite(s)) or not np.all(s > 0.0):
            raise ValueError(f"mock_realization: mode {key!r} sigma must be finite and > 0 everywhere")
        rng = np.random.default_rng(_mode_stream_seed(seed, key))
        noise = rng.normal(0.0, s)
        modes[key] = dict(noise=noise, depth_mock=d + noise, sigma=s)
    # Provenance for archival reproducibility: default_rng's bitstream is
    # stable in practice but not contractually guaranteed across numpy
    # versions, so the generating scheme and numpy version ride the record.
    return dict(
        kind=MOCK_KIND, label=MOCK_LABEL, seed=seed, modes=modes, seed_scheme=SEED_SCHEME, numpy_version=np.__version__
    )


def mock_recovery(
    results: dict | list[dict], free_names: list[str], realization: dict, co_eval: float | None = None
) -> dict:
    """Linearized parameter recovery for ONE mock realization.

    Under the tool's linearization a noise draw ``n`` shifts the recovered
    parameters by delta_theta = F^+ J^T Sigma^-1 n, evaluated on the SAME
    stacked nuisance-augmented system as ``fisher.combined_forecast``
    (shared lnR0 + per-segment offsets jointly fit), so the shifts are
    consistent with the marginalized sigmas: E[delta] = 0 and
    Cov[delta] = (F^+)_free over realizations (pinned by test).

    ``results``: the USABLE (unsaturated) evaluate_mode dicts entering the
    forecast; ``realization``: a ``mock_realization`` record covering every
    mode in ``results``. Returns dict(kind, label, seed, delta,
    delta_display, units, recovered): ``delta`` is the internal-unit shift
    per free parameter; ``delta_display`` the display-unit shift (same
    linear transforms as the fisher table; dlnCO needs ``co_eval``). A
    parameter on a null Fisher direction has recovered=False and None
    shifts -- an unconstrained direction has no recovered value, never a
    fake one.
    """
    rlist = list(results)
    if not rlist:
        raise ValueError("mock_recovery: results is empty")
    if not free_names:
        raise ValueError("mock_recovery: free_names is empty")
    if not isinstance(realization, dict) or "modes" not in realization:  # type: ignore[redundant-expr]  # a runtime guard for untyped callers
        raise ValueError("mock_recovery: realization must be a mock_realization record (with 'modes')")
    n_f = len(free_names)
    for i, r in enumerate(rlist):
        _validate_result(r, n_f, f"mock_recovery results[{i}]")
        key = r.get("mode_key")
        if key not in realization["modes"]:
            raise ValueError(
                f"mock_recovery: realization has no draw for mode {key!r} -- "
                "pass the mock_realization record for this run"
            )
        nb = np.asarray(r["sigma"]).size
        noise = np.asarray(realization["modes"][key]["noise"], float)
        if noise.shape != (nb,):
            raise ValueError(f"mock_recovery: mode {key!r} noise shape {noise.shape} does not match its {nb} bins")

    n_tot, system = fisher._combined_system(rlist, n_f)
    F = np.zeros((n_tot, n_tot))
    b = np.zeros(n_tot)
    for Jg, r in system:
        F += fisher._fisher(Jg, r)
        noise = np.asarray(realization["modes"][r["mode_key"]]["noise"], float)
        b += Jg @ (noise / np.asarray(r["sigma"], float) ** 2)
    delta = fisher._whitened_solve(F, b, n_f)

    out_delta: dict[str, float | None] = {}
    out_disp: dict[str, float | None] = {}
    recovered: dict[str, bool] = {}
    for i, name in enumerate(free_names):
        d_i = float(delta[i])
        if np.isfinite(d_i):
            recovered[name] = True
            out_delta[name] = d_i
            # display_sigma's transforms are linear, so they apply to signed
            # shifts exactly as to sigmas (dlnCO: co_eval * delta_ln)
            out_disp[name] = float(fisher.display_sigma(name, d_i, co_eval=co_eval))
        else:
            recovered[name] = False
            out_delta[name] = None
            out_disp[name] = None
    return dict(
        kind=MOCK_RECOVERY_KIND,
        label=MOCK_RECOVERY_LABEL,
        seed=realization.get("seed"),
        delta=out_delta,
        delta_display=out_disp,
        units={n: _param_unit(n) for n in free_names},
        recovered=recovered,
    )


def param_center(name: str, cpj: dict) -> float | None:
    """Input-model value of ``name`` in DISPLAY units, the Gaussian's center.

    None when the run's stored parameters define no single value (lnKzz under a
    non-constant or zero mixing profile, or a T-P/cloud parameter the run does not carry). A None center draws no
    curve -- the panel says so explicitly instead of guessing a center. The canonical payload always carries
    met_x_solar, co_ratio, kzz_mode and kzz_const.
    """
    if name == "lnZ":
        return float(np.log10(float(cpj["met_x_solar"])))
    if name == "dlnCO":
        return float(cpj["co_ratio"])
    if name == "lnKzz":
        kzz = float(cpj["kzz_const"])
        return float(np.log10(kzz)) if cpj["kzz_mode"] == "const" and kzz > 0 else None
    v = cpj.get(name)
    return None if v is None else float(v)


def co_curve_bounds() -> tuple:
    """Return the C/O window every forecast curve is drawn in.

    The range the forward model solves (the span the GUI C/O widget offers), not the +-5 sigma the width alone
    implies.
    """
    from jwst_tool import model_defs  # light path: no JAX/VULCAN imports

    return float(model_defs.CO_MIN), float(model_defs.CO_MAX_PHOTO_OFF)


def mock_center_co(center: float, sigma_display: float, delta_ln: float) -> float | None:
    """Recovered absolute C/O for ONE draw, or None when it is off scale.

    C/O lives on (0, inf), so the recovered center is shifted
    MULTIPLICATIVELY by the internal ln-space shift (center +
    delta_display can land below zero for a weakly constrained C/O).

    The drawn curve keeps the FORECAST width, and that width is linearized
    at the INPUT C/O (sigma_CO = C/O * sigma_lnCO). The pairing only holds
    while the shifted center stays inside the forecast's own auto-grid
    window: on a weakly constrained direction exp(delta_ln) runs to many
    orders of magnitude, and a center that far out carries no width
    information -- its curve degenerates to one float64 value and the panel
    axis stretches to the shifted center. Such a draw returns None; the
    caller reports it and draws the unshifted forecast, never an off-scale
    spike. A center outside the range the model solves (co_curve_bounds) is
    off scale for the same reason and returns None too.
    """
    center = float(center)
    sigma_display = float(sigma_display)
    delta_ln = float(delta_ln)
    if not (np.isfinite(center) and center > 0.0):
        raise ValueError(f"mock_center_co: center must be finite and > 0, got {center!r}")
    if not (np.isfinite(sigma_display) and sigma_display > 0.0):
        raise ValueError(f"mock_center_co: sigma_display must be finite and > 0, got {sigma_display!r}")
    if not np.isfinite(delta_ln):
        raise ValueError(f"mock_center_co: delta_ln must be finite, got {delta_ln!r}")
    with np.errstate(over="ignore"):
        mu = center * float(np.exp(delta_ln))
    if not (np.isfinite(mu) and mu > 0.0):
        return None  # exp() overflowed, or underflowed to a finite 0.0
    if abs(mu - center) > _AUTO_SIGMA * sigma_display:
        return None
    # A recovered C/O the forward model cannot produce is off scale in the
    # other sense: its curve would have to be drawn outside the window every
    # C/O panel uses, so the caller draws the unshifted forecast instead.
    lo, hi = co_curve_bounds()
    return mu if lo <= mu <= hi else None
