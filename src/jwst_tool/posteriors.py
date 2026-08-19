"""Fisher-Gaussian (Laplace) marginalized posterior curves + named mode combos.

Pure numpy, GUI-independent. Everything here is a LINEARIZED forecast read off
the same nuisance-augmented Fisher matrices as the constraint table
(``fisher.mode_forecast`` / ``fisher.combined_forecast``): each 1D curve is
the Gaussian N(theta_input, sigma_marg) with sigma_marg the marginalized
Cramer-Rao bound in the fitted coordinate. Curves are rendered in DISPLAY
units (the fisher-table convention:
dex for lnZ/lnKzz, absolute C/O number ratio for dlnCO via
``fisher.display_sigma``, K for temperatures; a parameter unknown to
``forward.PARAM_UNITS`` is already in display units and reports unit "").
The dlnCO curve is a symmetric Gaussian in absolute C/O, clipped at the
physical boundary C/O = 0 (curves render peak-normalized, so clipping and
clip-plus-renormalize are identical on screen).

HONESTY (baked into every returned record, keys ``kind``/``label``): these
are linearized Fisher/Cramer-Rao forecasts around the input model under the
quoted noise model -- Gaussian by construction, local, best-case. They are
NOT sampled posteriors, and a retrieval freeing more parameters under the
same model and noise assumptions usually reports lower significance. Never
present a curve from this module as a retrieval result.

Rank-deficient directions: a parameter on a numerically null Fisher direction
comes back from ``fisher._marg_sigmas`` as inf; here it is an explicit
``constrained=False`` record with NO curve (theta/pdf None) -- never a fake
finite Gaussian. Non-finite inputs raise (house style: fail fast and loud).

Named combinations (``combo_forecast``): a combo is a
user-facing name plus a list of registry mode keys, evaluated against the
per-mode result dicts of a run through the EXISTING ``combined_forecast``
(single-mode combos equal ``mode_forecast`` -- that identity is pinned
upstream). Saturated modes are excluded with the same policy as the
"All usable modes" row (a saturated mode is unusable data) and the
exclusion is disclosed per combo; a combo with zero usable modes raises.
"""
from __future__ import annotations

import zlib

import numpy as np

from jwst_tool import fisher

# Machine token + human sentence, carried on every record this module
# returns, so any downstream surface (GUI phase, exports) can label the
# curves honestly without re-deriving the wording.
FORECAST_KIND = "fisher_gaussian_forecast"
FORECAST_LABEL = (
    "Linearized Fisher (Cramer-Rao) forecast: a Gaussian approximation "
    "in the fitted parameter coordinate, centered on the input model with "
    "the marginalized 1-sigma lower bound as width, under the quoted noise "
    "model. The absolute C/O curve is clipped at the physical boundary "
    "C/O = 0. Not a sampled posterior; a "
    "retrieval freeing more parameters under the same assumptions usually "
    "reports lower significance.")

_SATURATED_REASON = "saturated at the shortest ramp tried (unusable data)"


def _param_unit(name: str) -> str:
    """Display unit for ``name``, from the single source of truth
    (forward.PARAM_UNITS; local import keeps this module numpy-light).

    dlnCO reports "C/O ratio" (its display transform is the absolute number
    ratio). A name absent from the table is a caller-defined row already in
    its own display units (display_sigma passes it through unchanged), so it
    reports unit "" -- same convention, not a fallback.
    """
    if name == "dlnCO":
        return "C/O ratio"
    from jwst_tool import forward  # light path: no JAX/VULCAN imports
    unit = forward.PARAM_UNITS.get(name)
    return unit if unit is not None else ""


def _validate_result(r: dict, n_free: int, where: str) -> None:
    """Fail fast on a malformed/non-finite mode result (house style)."""
    for key in ("jac_bins", "sigma"):
        if key not in r:
            raise ValueError(f"{where}: result is missing {key!r} -- pass "
                             "the evaluate_mode output (with Jacobians)")
    J = np.asarray(r["jac_bins"], float)
    s = np.asarray(r["sigma"], float)
    if J.ndim != 2:
        raise ValueError(f"{where}: jac_bins must be 2-D (n_par, n_bins), "
                         f"got shape {J.shape}")
    if J.shape[0] != n_free + 1:
        raise ValueError(
            f"{where}: jac_bins has {J.shape[0]} rows but free_names has "
            f"{n_free} entries -- rows must be [free..., lnR0] "
            f"({n_free + 1} rows)")
    if s.ndim != 1 or s.size != J.shape[1]:
        raise ValueError(f"{where}: sigma shape {s.shape} does not match "
                         f"jac_bins columns ({J.shape[1]})")
    if not np.all(np.isfinite(J)):
        raise ValueError(f"{where}: jac_bins contains non-finite values")
    if not np.all(np.isfinite(s)) or not np.all(s > 0.0):
        raise ValueError(f"{where}: sigma must be finite and > 0 everywhere")


def gaussian_curve(center: float, sigma: float, grid=None,
                   n_sigma: float = 5.0, n_points: int = 201) -> dict:
    """Normalized Gaussian pdf N(center, sigma) on ``grid`` (caller-supplied,
    1-D ascending) or an auto grid center +/- n_sigma*sigma (n_points).

    Returns dict(theta, pdf); pdf integrates to ~1 over an auto grid. All
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
            "an unconstrained (inf) direction has no curve by design")
    if grid is None:
        n_sigma = float(n_sigma)
        n_points = int(n_points)
        if not (np.isfinite(n_sigma) and n_sigma > 0.0):
            raise ValueError(f"gaussian_curve: n_sigma must be finite and > 0, "
                             f"got {n_sigma!r}")
        if n_points < 3:
            raise ValueError(f"gaussian_curve: n_points must be >= 3, got {n_points}")
        theta = np.linspace(center - n_sigma * sigma,
                            center + n_sigma * sigma, n_points)
    else:
        theta = np.asarray(grid, float)
        if theta.ndim != 1 or theta.size < 2:
            raise ValueError("gaussian_curve: grid must be a 1-D array with "
                             f"at least 2 points, got shape {theta.shape}")
        if not np.all(np.isfinite(theta)):
            raise ValueError("gaussian_curve: grid contains non-finite values")
    # Both paths: a grid that repeats a value carries no curve. An AUTO grid
    # collapses that way when the +/- n_sigma window is thinner than one
    # float64 step at the center -- which is what a center shifted far from
    # the point where its sigma was linearized produces. Name that cause
    # rather than reporting it as a bad grid.
    step = np.diff(theta)
    if np.any(step < 0.0):
        raise ValueError("gaussian_curve: grid must be strictly ascending")
    if np.any(step == 0.0):
        raise ValueError(
            f"gaussian_curve: the window around center {center!r} is "
            f"unresolvable in float64 at sigma {sigma!r} -- the grid repeats "
            "values. A center shifted far from where its sigma was "
            "linearized is the usual cause.")
    z = (theta - center) / sigma
    pdf = np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))
    return dict(theta=theta, pdf=pdf)


def truncated_gaussian_curve(center: float, sigma: float,
                             n_sigma: float = 5.0,
                             n_points: int = 201) -> dict:
    """Gaussian N(center, sigma) clipped to theta >= 0 (positive quantity,
    e.g. C/O). The maintainer's chosen display for the C/O panel: a familiar
    symmetric bell whose width matches the quoted +/- sigma, with the
    unphysical negative part simply cut. Panels draw curves peak-normalized,
    so clipping and clip-plus-renormalize render identically.
    """
    center = float(center)
    sigma = float(sigma)
    if not (np.isfinite(center) and center > 0.0):
        raise ValueError(
            f"truncated_gaussian_curve: center must be finite and > 0, "
            f"got {center!r}")
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError(
            f"truncated_gaussian_curve: sigma must be finite and > 0, "
            f"got {sigma!r} -- an unconstrained (inf) direction has no "
            "curve by design")
    lo = max(center - float(n_sigma) * sigma, 0.0)
    grid = np.linspace(lo, center + float(n_sigma) * sigma, int(n_points))
    return gaussian_curve(center, sigma, grid=grid)


def marginalized_posteriors(results, free_names: list[str], centers: dict,
                            params: list[str] | None = None,
                            co_eval: float | None = None,
                            grids: dict | None = None,
                            n_sigma: float = 5.0,
                            n_points: int = 201) -> dict:
    """1D marginalized Fisher-Gaussian posterior curves for chosen parameters.

    ``results``: one mode result dict, or a list of them (>= 1) treated as a
    combination -- a single dict goes through ``fisher.mode_forecast``, a
    list through ``fisher.combined_forecast`` (identical for one mode; the
    identity is pinned upstream). Each result needs ``jac_bins`` (rows
    [free..., lnR0]), ``sigma``, and optionally ``seg``.

    ``centers``: {param: theta_input in DISPLAY units} -- the input-model
    value each Gaussian is centered on (e.g. lnZ: log10(met_x_solar) in dex;
    dlnCO: the atmosphere's absolute C/O, i.e. co_eval). Required for every
    requested parameter; must be finite.

    ``params``: subset of free_names to render (default: all).
    ``co_eval``: the atmosphere's C/O at the evaluation point; REQUIRED when
    dlnCO is requested (fisher.display_sigma raises without it).
    ``grids``: optional {param: 1-D ascending display-unit grid}; parameters
    not listed get the auto grid center +/- n_sigma * sigma_marg.

    Returns dict(kind, label, n_modes, co_eval, sigma_marginalized,
    sigma_conditional, params) where params[name] is a record with
    center / sigma_display / sigma_conditional_display / unit /
    constrained / theta / pdf. An unconstrained direction (inf sigma from
    the rank-aware inversion) has constrained=False and theta=pdf=None --
    explicitly flagged, never a fake finite curve.
    """
    single = isinstance(results, dict)
    rlist = [results] if single else list(results)
    if not rlist:
        raise ValueError("marginalized_posteriors: results is empty")
    if not free_names:
        raise ValueError("marginalized_posteriors: free_names is empty")
    for i, r in enumerate(rlist):
        _validate_result(r, len(free_names),
                         f"marginalized_posteriors results[{i}]")
    if params is None:
        params = list(free_names)
    unknown = [p for p in params if p not in free_names]
    if unknown:
        raise ValueError(f"marginalized_posteriors: params {unknown} are not "
                         f"in free_names {list(free_names)}")
    missing = [p for p in params if p not in centers]
    if missing:
        raise ValueError(f"marginalized_posteriors: centers is missing "
                         f"{missing} (theta_input in display units)")
    for p in params:
        if not np.isfinite(float(centers[p])):
            raise ValueError(f"marginalized_posteriors: centers[{p!r}] must "
                             f"be finite, got {centers[p]!r}")
    if grids is not None:
        bad = [k for k in grids if k not in params]
        if bad:
            raise ValueError(f"marginalized_posteriors: grids has entries for "
                             f"{bad}, which are not requested params")

    cond: dict = {}
    if single:
        sig = fisher.mode_forecast(rlist[0], list(free_names), conditional=cond)
    else:
        sig = fisher.combined_forecast(rlist, list(free_names), conditional=cond)

    out_params = {}
    for name in params:
        s_int = float(sig[name])
        c_int = float(cond[name])
        s_disp = fisher.display_sigma(name, s_int, co_eval=co_eval)
        c_disp = fisher.display_sigma(name, c_int, co_eval=co_eval)
        rec = dict(center=float(centers[name]),
                   sigma_display=float(s_disp),
                   sigma_conditional_display=float(c_disp),
                   unit=_param_unit(name))
        if np.isfinite(s_disp):
            grid = grids.get(name) if grids is not None else None
            if name == "dlnCO":
                # C/O display: symmetric Gaussian in the display units,
                # clipped at the physical boundary C/O = 0; a caller grid
                # is honored verbatim
                if grid is not None:
                    curve = gaussian_curve(rec["center"], s_disp, grid=grid)
                else:
                    curve = truncated_gaussian_curve(
                        rec["center"], s_disp,
                        n_sigma=n_sigma, n_points=n_points)
                rec["curve_family"] = "gaussian_truncated_positive"
            else:
                curve = gaussian_curve(rec["center"], s_disp, grid=grid,
                                       n_sigma=n_sigma, n_points=n_points)
                rec["curve_family"] = "gaussian"
            rec.update(constrained=True, theta=curve["theta"],
                       pdf=curve["pdf"])
        else:
            # null Fisher direction: explicitly unconstrained, no curve
            rec.update(constrained=False, theta=None, pdf=None,
                       curve_family=None)
        out_params[name] = rec

    return dict(kind=FORECAST_KIND, label=FORECAST_LABEL,
                n_modes=len(rlist), co_eval=co_eval,
                sigma_marginalized={n: float(sig[n]) for n in free_names},
                sigma_conditional={n: float(cond[n]) for n in free_names},
                params=out_params)


# ---------------------------------------------------------------------------
# Named mode combinations
# ---------------------------------------------------------------------------

def combo_forecast(name: str, mode_keys: list[str], results_by_mode: dict,
                   free_names: list[str], centers: dict | None = None,
                   params: list[str] | None = None,
                   co_eval: float | None = None,
                   grids: dict | None = None, n_sigma: float = 5.0,
                   n_points: int = 201) -> dict:
    """Forecast one NAMED combination of instrument modes.

    ``name``: the user's label for the combination (e.g. "SOSS + G395H").
    ``mode_keys``: registry keys (instruments.MODES); an unknown key raises.
    ``results_by_mode``: {mode_key: evaluate_mode result with Jacobians} from
    a run; every requested key must be present.

    Saturation policy is EXACTLY the "All usable modes" row's: a
    saturated mode is unusable data, excluded from the Fisher sum, and the
    exclusion is disclosed in the returned record (``excluded``, with the
    reason). A combo whose every mode is excluded raises -- there is no
    forecast to report.

    Goes through ``fisher.combined_forecast`` for any usable-mode count, so a
    single-mode combo equals ``fisher.mode_forecast`` (identity pinned
    upstream). Marginalized AND conditional display-unit sigmas are reported;
    posterior curves (via ``marginalized_posteriors``) are attached when
    ``centers`` is given, else ``posteriors`` is None and the record says so.
    """
    from jwst_tool import instruments as ins  # data-root resolution stays lazy

    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"combo_forecast: combo name must be a non-empty "
                         f"string, got {name!r}")
    mode_keys = list(mode_keys)
    if not mode_keys:
        raise ValueError(f"combo {name!r}: mode_keys is empty")
    if len(set(mode_keys)) != len(mode_keys):
        raise ValueError(f"combo {name!r}: duplicate mode keys in "
                         f"{mode_keys} -- a mode cannot enter the Fisher sum "
                         "twice")
    bad = [k for k in mode_keys if k not in ins.MODES]
    if bad:
        raise ValueError(f"combo {name!r}: unknown mode key(s) {bad}; valid "
                         f"keys are {sorted(ins.MODES)}")
    missing = [k for k in mode_keys if k not in results_by_mode]
    if missing:
        raise ValueError(f"combo {name!r}: no result for mode(s) {missing} -- "
                         "run the modes first and pass their result dicts")

    usable, excluded = [], []
    for k in mode_keys:
        r = results_by_mode[k]
        if bool(r.get("saturated", False)):
            excluded.append(dict(mode_key=k, reason=_SATURATED_REASON))
        else:
            usable.append(k)
    if not usable:
        raise ValueError(
            f"combo {name!r}: no usable mode -- all of {mode_keys} are "
            "saturated on this star. There is no forecast to report; pick "
            "different modes or a different target.")
    rlist = [results_by_mode[k] for k in usable]
    for k, r in zip(usable, rlist):
        _validate_result(r, len(free_names), f"combo {name!r} mode {k!r}")

    cond: dict = {}
    sig = fisher.combined_forecast(rlist, list(free_names), conditional=cond)
    report = list(free_names) if params is None else list(params)
    unknown = [p for p in report if p not in free_names]
    if unknown:
        raise ValueError(f"combo {name!r}: params {unknown} are not in "
                         f"free_names {list(free_names)}")
    sig_disp = {n: float(fisher.display_sigma(n, float(sig[n]), co_eval=co_eval))
                for n in report}
    cond_disp = {n: float(fisher.display_sigma(n, float(cond[n]), co_eval=co_eval))
                 for n in report}

    posteriors = None
    if centers is not None:
        posteriors = marginalized_posteriors(
            rlist, list(free_names), centers, params=report, co_eval=co_eval,
            grids=grids, n_sigma=n_sigma, n_points=n_points)

    return dict(
        name=name, kind=FORECAST_KIND, label=FORECAST_LABEL,
        mode_keys=mode_keys, usable_modes=usable, excluded=excluded,
        n_modes_usable=len(usable), co_eval=co_eval,
        sigma_marginalized_display=sig_disp,
        sigma_conditional_display=cond_disp,
        units={n: _param_unit(n) for n in report},
        posteriors=posteriors,
        posteriors_note=(None if centers is not None else
                         "no centers supplied: sigmas only, no curves"),
    )


# ---------------------------------------------------------------------------
# Mock-observation layer: generated AFTER the forward model and the noise
# model, never inside them. mock_realization draws it; mock_recovery fits it
# (the forecast itself never sees the draw).
# ---------------------------------------------------------------------------

MOCK_KIND = "mock_observation_realization"
# The disclosure exists in exactly two forms, reused everywhere it is
# needed (a short one for controls and exports, a full one for provenance
# records) so the wording cannot drift between call sites.
MOCK_SHORT_LABEL = (
    "One seeded realization of the adopted effective diagonal noise model; "
    "the forecast numbers are realization-independent.")

MOCK_LABEL = (
    "One simulated noise realization: the binned noiseless model plus one "
    "seeded Gaussian draw per bin at each mode's final per-bin uncertainty "
    "(floor included). A single realization can be lucky or unlucky. The "
    "draw never enters the detection or Fisher scores, the forecast widths, "
    "the caches, or the noiseless result exports -- every quoted precision "
    "and conditional template S/N is computed from the noiseless model and "
    "is independent of the draw by construction -- but it does drive the "
    "optional linearized recovery overlay (mock_recovery).")

MOCK_RECOVERY_KIND = "mock_recovery"
MOCK_RECOVERY_LABEL = (
    "Recovered parameter shift for one simulated noise realization under "
    "the same linearization as the forecast (delta_theta = F^+ J^T "
    "Sigma^-1 n, with the shared lnR0 and per-segment offset nuisances in "
    "the fit). The shift has zero mean and the marginalized Fisher "
    "covariance over realizations; it is not a retrieval result.")


# Version tag for the seed -> stream mapping below. Bump if the mapping ever
# changes (different hash, different generator), so an archived seed +
# scheme version pins the exact realization.
SEED_SCHEME = "v1:default_rng([seed, crc32(mode_key)])"


def _mode_stream_seed(seed: int, mode_key: str) -> list[int]:
    """Deterministic per-mode RNG seed sequence: the draw for a mode depends
    only on (seed, mode_key), never on which other modes were run or on the
    selection order. crc32 is platform-stable."""
    return [int(seed), zlib.crc32(mode_key.encode())]


def _assert_mode_streams_distinct(mode_keys) -> None:
    """crc32 is a 32-bit identifier; a collision between two registered mode
    keys would silently give them identical noise draws. The registry is
    small, so checking the stream ids on each mock draw is negligible and
    prevents a silent collision."""
    crcs = {}
    for k in mode_keys:
        c = zlib.crc32(str(k).encode())
        if c in crcs:
            raise ValueError(
                f"mock-observation seed streams collide: mode keys "
                f"{crcs[c]!r} and {k!r} share crc32 {c:#010x}; change one "
                "key or move the stream id to a wider hash")
        crcs[c] = k


def mock_realization(results, seed: int) -> dict:
    """One seeded mock observation over the run's evaluated modes.

    ``results``: list of ``detect.evaluate_mode`` result dicts (each needs
    ``mode_key``, ``depth``, ``sigma``). ``seed``: non-negative int; the same
    seed reproduces the identical realization (numpy ``default_rng`` with a
    per-mode substream -- global ``np.random`` state is never touched).
    The draw is ALWAYS at the mode's own per-bin sigma -- the same
    uncertainty the error bars, the conditional template S/N and the Fisher
    forecast use. There is deliberately no scale factor: a 0.5x or 2x draw
    is not a realization of the model whose error bars are plotted beside
    it. To change the assumed uncertainty, change the noise model (the
    per-mode random-noise multiplier), which propagates to every number.

    Returns dict(kind, label, seed, modes) where modes[mode_key] is
    dict(noise, depth_mock, sigma): ``noise`` is the N(0, sigma_i) draw per
    bin (sigma_i = that mode's FINAL per-bin depth error, floor included)
    and ``depth_mock = depth + noise``.

    This is a mock-observation layer, generated AFTER the forward model and
    the noise model, never inside them. The invariant is one-directional, NOT
    "display only": the draw may never enter the detection or Fisher scores,
    the forecast widths, the caches, or the noiseless result exports -- but it
    IS fitted by ``mock_recovery``, whose recovered-parameter shift the GUI
    overlays on the posterior panels.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"mock_realization: seed must be >= 0, got {seed}")
    rlist = list(results)
    if not rlist:
        raise ValueError("mock_realization: results is empty")
    _assert_mode_streams_distinct(
        r.get("mode_key") for r in rlist if r.get("mode_key"))
    modes = {}
    for i, r in enumerate(rlist):
        key = r.get("mode_key")
        if not key:
            raise ValueError(f"mock_realization: results[{i}] has no "
                             "mode_key -- pass evaluate_mode outputs")
        if key in modes:
            raise ValueError(f"mock_realization: duplicate mode_key {key!r}")
        d = np.asarray(r["depth"], float)
        s = np.asarray(r["sigma"], float)
        if d.ndim != 1 or d.shape != s.shape:
            raise ValueError(
                f"mock_realization: mode {key!r} depth/sigma shapes "
                f"{d.shape}/{s.shape} must match and be 1-D")
        if not np.all(np.isfinite(d)):
            raise ValueError(f"mock_realization: mode {key!r} depth has "
                             "non-finite values")
        if not np.all(np.isfinite(s)) or not np.all(s > 0.0):
            raise ValueError(f"mock_realization: mode {key!r} sigma must be "
                             "finite and > 0 everywhere")
        rng = np.random.default_rng(_mode_stream_seed(seed, key))
        noise = rng.normal(0.0, s)
        modes[key] = dict(noise=noise, depth_mock=d + noise, sigma=s)
    # Provenance for archival reproducibility: default_rng's bitstream is
    # stable in practice but not contractually guaranteed across numpy
    # versions, so the generating scheme and numpy version ride the record.
    return dict(kind=MOCK_KIND, label=MOCK_LABEL, seed=seed, modes=modes,
                seed_scheme=SEED_SCHEME, numpy_version=np.__version__)


def mock_recovery(results, free_names: list[str], realization: dict,
                  co_eval: float | None = None) -> dict:
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
    if not isinstance(realization, dict) or "modes" not in realization:
        raise ValueError("mock_recovery: realization must be a "
                         "mock_realization record (with 'modes')")
    n_f = len(free_names)
    for i, r in enumerate(rlist):
        _validate_result(r, n_f, f"mock_recovery results[{i}]")
        key = r.get("mode_key")
        if key not in realization["modes"]:
            raise ValueError(
                f"mock_recovery: realization has no draw for mode {key!r} -- "
                "pass the mock_realization record for this run")
        nb = np.asarray(r["sigma"]).size
        noise = np.asarray(realization["modes"][key]["noise"], float)
        if noise.shape != (nb,):
            raise ValueError(
                f"mock_recovery: mode {key!r} noise shape {noise.shape} does "
                f"not match its {nb} bins")

    n_tot, system = fisher._combined_system(rlist, n_f)
    F = np.zeros((n_tot, n_tot))
    b = np.zeros(n_tot)
    for Jg, r in system:
        s2 = np.asarray(r["sigma"], float) ** 2
        F += (Jg / s2[None, :]) @ Jg.T
        noise = np.asarray(realization["modes"][r["mode_key"]]["noise"],
                           float)
        b += Jg @ (noise / s2)
    delta = fisher._whitened_solve(F, b, n_f)

    out_delta, out_disp, recovered = {}, {}, {}
    for i, name in enumerate(free_names):
        d_i = float(delta[i])
        if np.isfinite(d_i):
            recovered[name] = True
            out_delta[name] = d_i
            # display_sigma's transforms are linear, so they apply to signed
            # shifts exactly as to sigmas (dlnCO: co_eval * delta_ln)
            out_disp[name] = float(fisher.display_sigma(name, d_i,
                                                        co_eval=co_eval))
        else:
            recovered[name] = False
            out_delta[name] = None
            out_disp[name] = None
    return dict(kind=MOCK_KIND, label=MOCK_RECOVERY_LABEL,
                seed=realization.get("seed"),
                delta=out_delta, delta_display=out_disp,
                units={n: _param_unit(n) for n in free_names},
                recovered=recovered)


def mock_center_co(center: float, sigma_display: float, delta_ln: float,
                   n_sigma: float = 5.0):
    """Recovered absolute C/O for ONE draw, or None when it is off scale.

    C/O lives on (0, inf), so the recovered center is shifted
    MULTIPLICATIVELY by the internal ln-space shift (center +
    delta_display can land below zero for a weakly constrained C/O).

    The drawn curve keeps the FORECAST width, and that width is linearized
    at the INPUT C/O (sigma_CO = C/O * sigma_lnCO). The pairing only holds
    while the shifted center stays inside the forecast's own +/- n_sigma
    window: on a weakly constrained direction exp(delta_ln) runs to many
    orders of magnitude, and a center that far out carries no width
    information -- its curve degenerates to one float64 value and the panel
    axis stretches to the shifted center. Such a draw returns None; the
    caller reports it and draws the unshifted forecast, never an off-scale
    spike.
    """
    center = float(center)
    sigma_display = float(sigma_display)
    delta_ln = float(delta_ln)
    if not (np.isfinite(center) and center > 0.0):
        raise ValueError(f"mock_center_co: center must be finite and > 0, "
                         f"got {center!r}")
    if not (np.isfinite(sigma_display) and sigma_display > 0.0):
        raise ValueError(f"mock_center_co: sigma_display must be finite and "
                         f"> 0, got {sigma_display!r}")
    if not np.isfinite(delta_ln):
        raise ValueError(f"mock_center_co: delta_ln must be finite, got "
                         f"{delta_ln!r}")
    with np.errstate(over="ignore"):
        mu = center * float(np.exp(delta_ln))
    if not np.isfinite(mu):
        return None
    return mu if abs(mu - center) <= float(n_sigma) * sigma_display else None
