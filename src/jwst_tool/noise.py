"""Host-side noise interface: subprocess to the Pandeia worker + transit-depth error math.

The worker gives, per instrument mode, the per-native-pixel extracted stellar
flux and its 1-integration sigma. This module turns that into a transit-depth
uncertainty per spectral bin:

    depth = 1 - F_in/F_out
    var(depth)_pixel = (sigma_1int/flux)^2 * (1/n_int_in + 1/n_int_out) / n_transits
    var(depth)_bin   = sum(w^2 var_pixel) / (sum w)^2,  w = flux   (count-space)
    sigma_bin        = max(sqrt(var_bin), floor)         (PandExo floor convention)

The count-space combination is the variance of the SAME estimator detect.py
uses to bin the model (binning.build_operator) -- never revert to the
inverse-variance combination, which quotes a different (optimal-weights)
estimator than the one the model is binned with.

The minimum floor follows PandExo semantics exactly (resolve_floor): a HARD
MINIMUM on the final binned uncertainty -- never added in quadrature, never
scaled with the requested resolving power, never averaged down by transits.

`floor_spec` has NO DEFAULT anywhere and must be passed explicitly: a default
floor would silently set the headline precision, and a default of zero would
claim precision no program has demonstrated. The caller chooses, and the
choice is recorded in the run provenance. The per-mode values in
`instruments.MODES` are named `floor_ppm_suggested` and only prefill a GUI
widget.

Scope: a Pandeia-extracted-noise BOX-TRANSIT planning forecast under the
selected extraction/detector configuration. NEVER label sigmas
PandExo-identical; parity status is per-backend and lives in
tests/parity/outputs/REPORT.md, kept honest by the fail-closed gate
(tests/parity/scripts/run_parity.py). Pandeia's extracted noise is not a
time-series systematics model (no 1/f residuals, visit-long trends,
detrending covariance, or stellar heterogeneity).

Results are cached by a hash of (star, modes, sat_limit, engine + refdata
versions) so the ETC runs once per star/instrument set and stale caches are
invalidated when the Pandeia backend changes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from . import binning
from . import instruments as ins
from . import proc as proc_mod

_BACKEND_FINGERPRINT = None

# The production Pandeia-worker version: part of every noise-cache key AND the
# identity the parity gate (tests/parity/scripts/parity_gate.py) requires of a
# committed artifact. Bump whenever pandeia_worker.py output changes (history:
# notes.md); a bump without a fresh parity run fails the gate test.
# v8 (2026-08-09): ramp-search floors adopt pandeia mingroups (NIR modes drop
# 2 -> 1, MIRI LRS 5 -> 2), matching PandExo's search space; not required for
# cache correctness (ngroup_min is in the job key) but the search-policy
# change is part of the parity-artifact identity.
# v9 (2026-08-09, same-day review fix): group selection returns the LARGEST
# measured-safe count via a bidirectional measured search, and "saturated"
# means the measured floor exceeds the limit. v8's min-then-verify-down
# search under-selected (a needless 1-group SOSS ramp with ~7x the noise);
# v8 cached results are wrong on short ramps, so the version bump is
# REQUIRED for cache correctness here, not just artifact identity.
# v10 (2026-08-09, review round 2): maximality is PROVEN by a bracket
# search (complete only when the next integer measured unsafe or the cap is
# reached; v9's predictor-stall exit could sit one integer low on offset
# ramps), and the payload gains ramp_search_complete (budget exhaustion is
# reported, never presented as maximal).
WORKER_VERSION = 10


def backend_fingerprint() -> dict:
    """Pandeia backend identity baked into every cache key (queried once per
    process): engine version from the backend python + refdata version-file
    contents. "unavailable" (still cache-key-stable) when the env is missing --
    run_pandeia raises loudly on an actual run attempt."""
    global _BACKEND_FINGERPRINT
    if _BACKEND_FINGERPRINT is not None:
        return _BACKEND_FINGERPRINT
    engine = "unavailable"
    # a missing backend keeps the fingerprint "unavailable" (stable cache
    # key); the failure belongs at RUN time, not while building a key
    py = Path(ins.PANDEIA_PYTHON) if ins.PANDEIA_PYTHON else None
    if py is not None and py.exists():
        try:
            r = subprocess.run(
                [str(py), "-c",
                 "import pandeia.engine; print(pandeia.engine.__version__)"],
                capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and r.stdout.strip():
                engine = r.stdout.strip().splitlines()[-1]
        except Exception:
            pass
    refver = []
    # refdata identifies itself via VERSION/VERSION_DATA; a VERSION_PSF file
    # inside a refdata tree is a misplaced PSF marker, never a data version
    # (the retired 3.0-era fallback let one authenticate the tree)
    for root, names in ((Path(ins.PANDEIA_REFDATA),
                         ("VERSION", "VERSION_DATA")),
                        (Path(ins.PANDEIA_PSF_DIR) if ins.PANDEIA_PSF_DIR
                         else None, ("VERSION_PSF",))):
        if root is None:
            continue
        for name in names:
            f = root / name
            if f.exists():
                refver.append(
                    f"{name}:{hashlib.sha1(f.read_bytes()).hexdigest()[:12]}")
    ref = Path(ins.PANDEIA_REFDATA)
    # CDBS identity: the PHOENIX grid is a machine-specific symlink and a
    # repointed tree changes every result, so the resolved real path + the
    # catalog's (size, mtime_ns) must key the cache (cheap; no gigabyte
    # hashing)
    cdbs_id = []
    cdbs = Path(ins.PYSYN_CDBS)
    phx = cdbs / "grid" / "phoenix"
    if phx.exists():
        real = phx.resolve()
        cdbs_id.append(f"phoenix:{real}")
        cat = real / "catalog.fits"
        if cat.is_file():
            st = cat.stat()
            cdbs_id.append(f"catalog:{st.st_size}:{st.st_mtime_ns}")
    _BACKEND_FINGERPRINT = {
        "engine_version": engine,
        "refdata_name": ref.name,
        # PSF tree identity is a FIRST-CLASS key: a mixed PSF release changes
        # the extracted flux and noise (the worker also refuses a mismatch)
        "psf_name": (Path(ins.PANDEIA_PSF_DIR).name
                     if ins.PANDEIA_PSF_DIR else ""),
        "backend_token": ins.JWST_TOOL_BACKEND,
        "backend_release": ins.BACKEND_RELEASE,
        "refdata_version_files": sorted(refver),
        "cdbs_identity": cdbs_id,
    }
    return _BACKEND_FINGERPRINT


def noise_job(star: dict, mode_keys: list[str], sat_limit: float = 0.80) -> dict:
    modes = []
    for key in mode_keys:
        m = dict(ins.MODES[key])
        modes.append({
            # engine_mode() applies any per-backend mode renames (none at
            # present). The parity harness relies on the same resolution --
            # never a parity-only rename.
            "key": key, "instrument": m["instrument"],
            "mode": ins.engine_mode(m["instrument"], m["mode"]),
            "config": m.get("config", {}), "strategy": m.get("strategy", {}),
            "background": m.get("background"),
            "background_level": m.get("background_level"),
            "ngroup_min": m["ngroup_min"], "ngroup_max": m["ngroup_max"],
        })
    job_extra = {}
    if ins.PANDEIA_PSF_DIR:
        # split-layout PSF library: part of the cache key (every supported
        # backend uses the split layout; the worker refuses an empty psf_dir)
        job_extra["psf_dir"] = ins.PANDEIA_PSF_DIR
    return {
        "refdata": ins.PANDEIA_REFDATA, "cdbs": ins.PYSYN_CDBS,
        **job_extra,
        "backend": backend_fingerprint(),
        "star": {k: float(star[k]) for k in ("teff", "log_g", "metallicity", "ks_mag")},
        "sat_limit": float(sat_limit),
        "modes": modes,
        "worker_version": WORKER_VERSION,   # cache-buster; see the constant
    }


def job_key(job: dict) -> str:
    return hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]


def _mode_key(star: dict, mode_key: str, sat_limit: float) -> str:
    """Per-mode cache identity: the single-mode job's key, so it embeds the
    star, the mode's full submitted configuration, sat_limit, the backend
    fingerprint, and WORKER_VERSION -- everything that changes the result."""
    return job_key(noise_job(star, [mode_key], sat_limit=sat_limit))


def missing_modes(star: dict, mode_keys: list[str],
                  sat_limit: float = 0.80) -> list[str]:
    """The subset of ``mode_keys`` with no per-mode cache entry (in order)."""
    return [k for k in mode_keys
            if not (ins.NOISE_CACHE
                    / f"{_mode_key(star, k, sat_limit)}.json").exists()]


def run_modes(star: dict, mode_keys: list[str], sat_limit: float = 0.80,
              progress=None, force: bool = False) -> dict:
    """The production ETC path (0.27.0): per-mode cache, one worker batch.

    Each mode is cached under its own single-mode job key, so a run computes
    ONLY the modes it needs and a later selection change costs exactly the
    newly added modes -- the pre-0.27.0 design computed all seven registry
    modes on every first run so that selection changes were free, which made
    the default run ~2.5x slower than its three-mode selection required.
    All cache misses go to the worker in ONE batch job (one subprocess, one
    pandeia import), and the result is split back into per-mode files.
    Returns {mode_key: payload, "__provenance__": {...}} like the worker.
    """
    ins.NOISE_CACHE.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    todo = list(mode_keys) if force else missing_modes(star, mode_keys,
                                                       sat_limit)
    for k in mode_keys:
        if k in todo:
            continue
        cached = json.loads(
            (ins.NOISE_CACHE
             / f"{_mode_key(star, k, sat_limit)}.json").read_text())
        out[k] = cached[k]
        out.setdefault("__provenance__", cached.get("__provenance__"))
    if todo:
        result = _run_worker(noise_job(star, todo, sat_limit=sat_limit),
                             progress)
        prov = result.get("__provenance__")
        for k in todo:
            single = {k: result[k], "__provenance__": prov}
            (ins.NOISE_CACHE
             / f"{_mode_key(star, k, sat_limit)}.json").write_text(
                json.dumps(single))
            out[k] = result[k]
        out["__provenance__"] = prov
    return out


def run_pandeia(job: dict, progress=None, force: bool = False) -> dict:
    """Run the worker on a whole job (or return the whole-job cached result).

    The parity harness's path: its artifact identity is the complete job.
    The GUI uses ``run_modes`` (per-mode cache) instead.
    ``progress``: optional callable(str) receiving worker stdout lines live.
    Raises RuntimeError (loudly, with stderr) if the worker process itself dies;
    per-mode pandeia failures come back as {"error": traceback} entries.
    """
    ins.NOISE_CACHE.mkdir(parents=True, exist_ok=True)
    cache = ins.NOISE_CACHE / f"{job_key(job)}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    result = _run_worker(job, progress)
    cache.write_text(json.dumps(result))
    return result


def _run_worker(job: dict, progress=None) -> dict:
    """Run pandeia_worker.py on ``job`` in the backend env; no caching."""
    ins.NOISE_CACHE.mkdir(parents=True, exist_ok=True)
    py = Path(ins.require_pandeia_python())
    if not py.exists():
        raise RuntimeError(
            f"Pandeia backend python not found at {py} (the '{ins.JWST_TOOL_BACKEND}' "
            f"backend: {ins.BACKEND_STATUS}). The noise model cannot run without it; "
            "set JWST_TOOL_PANDEIA_PYTHON to a python with the matching pandeia.engine.")

    in_json = ins.NOISE_CACHE / f"{job_key(job)}.job.json"
    out_json = ins.NOISE_CACHE / f"{job_key(job)}.out.json"
    in_json.write_text(json.dumps(job))
    worker = ins.TOOL_DIR / "pandeia_worker.py"

    proc = subprocess.Popen([str(py), str(worker), str(in_json), str(out_json)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # drain stderr on a thread while streaming stdout: reading only stdout
    # deadlocks once the worker fills the ~64 KB stderr pipe buffer
    import io
    import threading
    err_buf = io.StringIO()
    t_err = threading.Thread(target=lambda: err_buf.write(proc.stderr.read()),
                             daemon=True)
    t_err.start()
    # ``progress`` can raise Streamlit's cancel exception (a BaseException);
    # the worker must not outlive the run that started it
    with proc_mod.terminating(proc):
        for line in proc.stdout:
            if progress:
                progress(line.rstrip())
        proc.wait()
    t_err.join(timeout=30)
    if proc.returncode != 0 or not out_json.exists():
        err = err_buf.getvalue()
        raise RuntimeError(f"pandeia worker failed (rc={proc.returncode}):\n{err[-3000:]}")

    return json.loads(out_json.read_text())


def make_bins(wl_lo: float, wl_hi: float, R: float) -> np.ndarray:
    """Log-spaced bin EDGES at resolving power R over [wl_lo, wl_hi]."""
    if not (np.isfinite(wl_lo) and np.isfinite(wl_hi) and 0.0 < wl_lo < wl_hi):
        raise ValueError(f"make_bins: need finite 0 < wl_lo < wl_hi, got "
                         f"[{wl_lo!r}, {wl_hi!r}]")
    if not (np.isfinite(R) and R > 0.0):
        raise ValueError(f"make_bins: resolving power must be finite and > 0, "
                         f"got {R!r}")
    n = max(2, int(np.ceil(np.log(wl_hi / wl_lo) * R)))
    return np.geomspace(wl_lo, wl_hi, n + 1)


def resolve_floor(wl_um: np.ndarray, floor_spec) -> np.ndarray:
    """PandExo-compatible minimum-floor evaluation on the FINAL bin wavelengths.

    ``floor_spec`` is one of:
      * None            -- no minimum floor (zeros),
      * a scalar        -- constant minimum uncertainty in ppm for every bin,
      * an (n, 2) array -- [wavelength_um, floor_ppm] rows, linearly
        interpolated with constant edge extension; duplicate wavelengths raise.

    Returns the per-bin floor as a FRACTIONAL depth (ppm * 1e-6). Non-finite
    or negative values raise, never silently sanitized. Applied downstream as
    sigma_final = max(sigma_random, floor): a hard minimum, never quadrature,
    never rescaled by the binning R.
    """
    wl = np.asarray(wl_um, float)
    if floor_spec is None:
        return np.zeros(wl.size)
    spec = np.asarray(floor_spec, float)
    if spec.ndim == 0:
        val = float(spec)
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(f"constant noise floor must be a finite value "
                             f">= 0 ppm, got {val!r}")
        return np.full(wl.size, val * 1e-6)
    if spec.ndim != 2 or spec.shape[1] != 2 or spec.shape[0] < 2:
        raise ValueError(
            "wavelength-dependent noise floor must be an (n>=2, 2) array of "
            f"[wavelength_um, floor_ppm] rows, got shape {spec.shape}")
    if not np.all(np.isfinite(spec)):
        raise ValueError("noise-floor table contains non-finite values")
    order = np.argsort(spec[:, 0], kind="stable")
    w, f = spec[order, 0], spec[order, 1]
    if np.any(f < 0.0):
        raise ValueError("noise-floor table contains negative floor values")
    if np.any(np.diff(w) == 0.0):
        raise ValueError("noise-floor table contains duplicate wavelengths -- "
                         "resolve them explicitly before passing the table")
    return np.interp(wl, w, f, left=f[0], right=f[-1]) * 1e-6


# --- correlated-noise scenarios (EXPERIMENTAL) --------------------------------
# The floor EXCESS, excess^2 = max(0, floor^2 - var_phot), is the systematic
# budget. A scenario re-allocates it between a white part and a smooth part
# with a squared-exponential kernel in ln(lambda):
#
#     C = diag(var_phot + f_white * excess^2)
#         + (1 - f_white) * excess_i excess_j exp(-(ln wl_i - ln wl_j)^2 / 2 ell^2)
#
# PSD by construction. INVARIANT: diag(C) = max(var_phot, floor^2) =
# sigma_final^2 in EVERY scenario, so ranking changes between scenarios are
# correlation structure alone, never a bigger error bar. Photon-dominated
# bins carry no correlated part; f_white = 1 recovers the exact diagonal
# (build_cov returns None: fast sigma path).
#
# The presets are stated ASSUMPTIONS, not calibrated JWST covariances, and
# are excluded from headline results (default scenario "random"). ell is in
# ln-wavelength (0.05 ~ 5% wavelength scale). "conservative" also profiles a
# per-detector-segment SLOPE nuisance on top of the offsets.
# NOTE: because the correlated budget is the floor EXCESS, it GROWS as
# transits average the photon term down, so scores are NOT monotone in
# n_transits under the correlated presets -- transits_to_target scans
# instead of gating on the limit. A PSD-monotone additive model would be a
# NEW scenario family with documented diagonal semantics, never a silent
# swap.
SCENARIOS = {
    "random": dict(f_white=1.0, ell=None, slopes=False,
                   label="random-only: diagonal noise, offsets profiled "
                         "(default)"),
    "moderate": dict(f_white=0.5, ell=0.05, slopes=False,
                     label="moderate: half the floor excess smooth "
                           "(5% wl scale) [experimental]"),
    "conservative": dict(f_white=0.2, ell=0.15, slopes=True,
                         label="conservative: mostly-smooth floor excess "
                               "(15% wl scale) + per-segment slopes "
                               "[experimental]"),
}


def build_cov(wl_center: np.ndarray, var_phot: np.ndarray, floor: np.ndarray,
              scenario: str) -> np.ndarray | None:
    """Per-bin depth covariance under a named scenario (see SCENARIOS).

    ``floor`` is the resolved per-bin minimum floor (fractional depth); the
    correlated budget is the floor EXCESS, so diag(C) always equals
    max(var_phot, floor^2) -- the same total the diagonal path quotes (and
    scores on this covariance are not monotone in the transit count).
    Returns None for a fully-white scenario or when the floor binds nowhere,
    so callers keep the exact diagonal fast path; otherwise a
    positive-definite (n_bins, n_bins) matrix. Unknown scenario names raise
    (KeyError); non-finite or negative variances/floors and non-positive
    wavelengths raise.
    """
    sc = SCENARIOS[scenario]
    wl = np.asarray(wl_center, float)
    var = np.asarray(var_phot, float)
    fl = np.asarray(floor, float)
    if wl.ndim != 1 or var.shape != wl.shape or fl.shape != wl.shape:
        raise ValueError(
            f"build_cov: wl_center/var_phot/floor must be matching 1-D "
            f"arrays, got shapes {wl.shape}/{var.shape}/{fl.shape}")
    if not np.all(np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError("build_cov: wavelengths must be finite and > 0")
    if not np.all(np.isfinite(var)) or np.any(var < 0.0):
        raise ValueError("build_cov: var_phot must be finite and >= 0")
    if not np.all(np.isfinite(fl)) or np.any(fl < 0.0):
        raise ValueError("build_cov: floor must be finite and >= 0")
    a = np.sqrt(np.maximum(fl ** 2 - var, 0.0))
    if sc["f_white"] >= 1.0 or not np.any(a > 0.0):
        return None
    lnl = np.log(np.asarray(wl_center, float))
    d = (lnl[:, None] - lnl[None, :]) / float(sc["ell"])
    K = (1.0 - sc["f_white"]) * (a[:, None] * a[None, :]) * np.exp(-0.5 * d ** 2)
    return K + np.diag(var + sc["f_white"] * a ** 2)


def n_transits_int(n_transits) -> int:
    """A positive-INTEGER transit count, or a loud error.

    ONE definition for the whole stack (``detect._n_transits`` is this
    function): a fractional count has no meaning and must never be silently
    floored.
    """
    n = int(n_transits)
    if n < 1 or n != n_transits:
        raise ValueError(f"n_transits must be a positive integer, got "
                         f"{n_transits!r}")
    return n


def pixel_depth_variance(mode_result: dict, t_in_s: float, t_out_s: float,
                         n_transits: int) -> np.ndarray:
    """Per-native-pixel transit-depth variance (box-depth approximation).

    var = (sigma_1int/flux)^2 (1/n_in + 1/n_out) / n_transits, with the
    integration counts from the mode's measured cycle time. Neglects
    ingress/egress, limb darkening, and depth-detrending covariance.

    SYMMETRIC in/out approximation: the OUT-of-transit flux and sigma stand
    in for both terms. This is CONSERVATIVE and the excess grows with depth
    (~3d/4 for equal baselines, NOT d/2: +0.76% at d=1%, +8.2% at 10%; d/2
    to d for unequal baselines). Kept deliberately model-INDEPENDENT so ONE
    measurement operator bins noise and model; exact separate in/out
    propagation is a remaining refinement, NOT a same-answer refactor.
    """
    flux = np.asarray(mode_result["flux"], float)
    noise = np.asarray(mode_result["noise_1int"], float)
    t_cycle = float(mode_result["t_cycle_s"])
    # fail fast on inputs the worker normally guarantees but the public API
    # does not: NaN/zero flux or cycle time would propagate silently
    if not (np.isfinite(t_cycle) and t_cycle > 0.0):
        raise ValueError(f"t_cycle_s must be finite and > 0, got {t_cycle!r}")
    if not (np.isfinite(t_in_s) and t_in_s > 0.0
            and np.isfinite(t_out_s) and t_out_s > 0.0):
        raise ValueError(f"observation windows must be finite and > 0, got "
                         f"t_in_s={t_in_s!r}, t_out_s={t_out_s!r}")
    if flux.size == 0 or not np.all(np.isfinite(flux)) or np.any(flux <= 0.0):
        raise ValueError("extracted flux must be non-empty, finite and > 0 "
                         "(the worker drops unusable pixels; do not pass raw "
                         "grids with NaN/zero flux here)")
    if noise.shape != flux.shape or not np.all(np.isfinite(noise)) \
            or np.any(noise < 0.0):
        raise ValueError("noise_1int must match flux's shape and be finite "
                         "and >= 0")
    # int() floors to whole integrations (conservative); a window shorter
    # than one cycle yields NO usable integration -- never pretend one fits
    n_in = int(t_in_s / t_cycle)
    n_out = int(t_out_s / t_cycle)
    if n_in < 1 or n_out < 1:
        raise ValueError(
            f"observation window shorter than one integration cycle "
            f"(t_cycle={t_cycle:.1f} s, in-transit {t_in_s:.1f} s -> {n_in} "
            f"integrations, out-of-transit {t_out_s:.1f} s -> {n_out}): "
            "this mode cannot produce a usable depth measurement as configured")
    n_tr = n_transits_int(n_transits)
    return (noise / flux) ** 2 * (1.0 / n_in + 1.0 / n_out) / n_tr


def depth_error_bins(mode_result: dict, edges: np.ndarray,
                     t_in_s: float, t_out_s: float, n_transits: int,
                     floor_spec, op: dict | None = None,
                     noise_inflation: float = 1.0) -> dict:
    """Per-bin transit-depth sigma from a worker mode result.

    ``op`` is the mode's count-space measurement operator
    (binning.build_operator); pass the SAME operator used to bin the model.
    Built here from the pixel grid alone when omitted (noise-only use).

    ``floor_spec`` is the PandExo-style minimum floor (resolve_floor). Order
    of operations matches PandExo: (1) random variance, (2) binned in count
    space, (3) transits combined, (4) floor on the final bin wavelengths,
    (5) sigma = max(sigma_random, floor).

    Returns dict(wl_center, sigma, n_pix, var_phot, floor, n_transits) over
    the operator's kept bins. ``var_phot`` is the random bin variance AT the
    evaluated ``n_transits`` (scales as 1/N, inflation included); ``floor``
    is N-independent. The two components let callers extrapolate to other
    transit counts correctly.

    ``noise_inflation``: optional factor on the random sigma, default 1.0
    (the Pandeia prediction as-is). Literature achieved-vs-predicted ratios
    (instruments.LITERATURE_NOISE_FACTORS) are reference points only, never
    a default calibration. Proportional: averages down with transits, unlike
    the floor.
    """
    ninf = float(noise_inflation)
    if not (np.isfinite(ninf) and ninf > 0.0):
        # squared below: a negative factor would silently act like its
        # absolute value, and 0/NaN would zero/poison the sigma
        raise ValueError(f"noise_inflation must be finite and > 0, got "
                         f"{noise_inflation!r}")
    if op is None:
        op = binning.build_operator(np.asarray(mode_result["wl"]),
                                    np.asarray(mode_result["flux"]), edges)
    var_pix = pixel_depth_variance(mode_result, t_in_s, t_out_s, n_transits)
    var_phot = binning.bin_variance(op, var_pix) * ninf ** 2
    floor = resolve_floor(op["wl_center"], floor_spec)
    sigma = np.maximum(np.sqrt(var_phot), floor)
    return dict(wl_center=op["wl_center"], sigma=sigma, n_pix=op["n_pix"],
                var_phot=var_phot, floor=floor,
                n_transits=n_transits_int(n_transits))
