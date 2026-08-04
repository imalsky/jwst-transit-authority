"""Host-side noise interface: subprocess to the Pandeia worker + transit-depth error math.

The worker gives, per instrument mode, the per-native-pixel extracted stellar
flux and its 1-integration sigma. This module turns that into a transit-depth
uncertainty per spectral bin:

    depth = 1 - F_in/F_out
    var(depth)_pixel = (sigma_1int/flux)^2 * (1/n_int_in + 1/n_int_out) / n_transits
    var(depth)_bin   = sum(w^2 var_pixel) / (sum w)^2,  w = flux   (count-space)
    sigma_bin        = max(sqrt(var_bin), floor)         (PandExo floor convention)

The count-space combination is the variance of the SAME estimator detect.py
uses to bin the model (binning.build_operator) -- the sum-of-extracted-counts
bin depth every real reduction produces. The old inverse-variance combination
quoted the variance of a different (optimal-weights) estimator than the one
the model was binned with; in the photon-dominated limit the two agree.

The minimum floor follows PandExo semantics exactly (resolve_floor): a HARD
MINIMUM on the final binned uncertainty -- none, a constant ppm value, or a
user-supplied wavelength-vs-ppm model evaluated on the final bin wavelengths
with constant edge extension. It is NOT added in quadrature, NOT scaled with
the requested resolving power, and does NOT average down when transits are
added (the random term does; the floor caps it from below).

`floor_spec` has NO DEFAULT anywhere and must be passed explicitly. Because the
floor is a hard minimum, a 15-40 ppm value SETS the reported precision for any
well-observed target -- so a default floor would silently make an editorial
planning number the headline result, while a default of zero would claim a
precision no program has demonstrated (Pandeia models pixel-level and
correlated read noise, but no 1/f drift and no visit-long systematics). Neither
is neutral; the caller chooses, and the choice is recorded in the run
provenance. The per-mode values in `instruments.MODES` are named
`floor_ppm_suggested` and are only ever read to prefill a GUI widget.

What this is, and is not: a Pandeia-extracted-noise BOX-TRANSIT APPROXIMATION
under the selected extraction/detector configuration -- an instrument-model
planning forecast. PandExo parity status is per-backend and lives in
tests/parity/outputs/REPORT.md, which the fail-closed gate
(tests/parity/scripts/run_parity.py) keeps honest: a parity claim exists only
for a backend triple whose committed artifact PASSES the gate. The archival
2026.2 backend was matched on configuration, timing, wavelength grids, and
extracted flux, with the sigma difference attributed to the noise model
(Pandeia full extracted noise vs PandExo's analytic fml) and this tool
conservative; the 2026.7 triple has NO committed parity artifact yet -- rerun
the suite before treating its output as validated. Absolute sigmas are still
pandeia-extracted-noise forecasts, NOT labeled PandExo-identical. Pandeia's
extracted noise includes photon (with the HgCdTe quantum-yield/Fano excess
variance below ~2 um), background, dark, and correlated read noise + IPC -- it
is NOT a time-series systematics model (1/f residuals, visit-long trends,
pointing/tilt events, limb-darkening and detrending covariance, stellar
heterogeneity). Real reductions can differ in either direction depending on
extraction and analysis choices, though unmodeled systematics commonly degrade
precision.

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


def backend_fingerprint() -> dict:
    """Pandeia backend identity baked into every cache key (queried once per
    process): engine version from the picaso_base python, refdata version-file
    contents. "unavailable" (still cache-key-stable) when the env is missing --
    run_pandeia raises loudly on an actual run attempt."""
    global _BACKEND_FINGERPRINT
    if _BACKEND_FINGERPRINT is not None:
        return _BACKEND_FINGERPRINT
    engine = "unavailable"
    # PICASO_PYTHON is None until the machine-specific backend env is set; the
    # fingerprint stays "unavailable" rather than raising here (this runs on the
    # cache-key path, and a missing backend must fail at RUN time with the one
    # actionable message, not while building a key).
    py = Path(ins.PICASO_PYTHON) if ins.PICASO_PYTHON else None
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
    for root, names in ((Path(ins.PANDEIA_REFDATA),
                         ("VERSION", "VERSION_DATA", "VERSION_PSF")),
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
    # CDBS identity (worker v6): the star SED comes from the PYSYN_CDBS
    # PHOENIX grid, which is a machine-specific external symlink here -- a
    # repointed/updated tree changes every flux/noise result, so its identity
    # must key the cache. Contents-hashing gigabytes is off the table; the
    # resolved real path + the PHOENIX catalog's (size, mtime_ns) is a cheap
    # fingerprint that changes whenever the grid is swapped or regenerated.
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
        # PSF tree identity is a FIRST-CLASS key, not just a hashed VERSION
        # file: a mixed PSF release changes the point-spread function and hence
        # the extracted flux and noise. The worker now also REFUSES a release
        # that differs from the engine/refdata (item 3).
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
            # engine_mode() maps the registry's canonical (3.0) token to the
            # name the ACTIVE backend accepts (e.g. NIRCam ssgrism->lw_tsgrism on
            # the 2026 engine); no-op on the legacy backend. Without this the
            # 2026 engine raises "Invalid mode: ssgrism" and both NIRCam modes
            # fail. The parity harness relies on the same resolution.
            "key": key, "instrument": m["instrument"],
            "mode": ins.engine_mode(m["instrument"], m["mode"]),
            "config": m.get("config", {}), "strategy": m.get("strategy", {}),
            "background": m.get("background"),
            "background_level": m.get("background_level"),
            "ngroup_min": m["ngroup_min"], "ngroup_max": m["ngroup_max"],
        })
    job_extra = {}
    if ins.PANDEIA_PSF_DIR:
        # split-layout (2026+) PSF library: passed to the worker and part of
        # the cache key; absent entirely under the 3.0-era combined layout so
        # existing cache keys are untouched
        job_extra["psf_dir"] = ins.PANDEIA_PSF_DIR
    return {
        "refdata": ins.PANDEIA_REFDATA, "cdbs": ins.PYSYN_CDBS,
        **job_extra,
        "backend": backend_fingerprint(),
        "star": {k: float(star[k]) for k in ("teff", "log_g", "metallicity", "ks_mag")},
        "sat_limit": float(sat_limit),
        "modes": modes,
        # cache-buster: bump whenever pandeia_worker.py output changes
        # (output semantics in the worker docstring, per-version history
        # in notes.md). v7 = native-grid saturation counts taken before the
        # `good` filter + the matched engine/refdata/PSF release triple in
        # __provenance__.
        "worker_version": 7,
    }


def job_key(job: dict) -> str:
    return hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]


def run_pandeia(job: dict, progress=None, force: bool = False) -> dict:
    """Run the worker in the selected backend's env (or return the cached result).

    ``progress``: optional callable(str) receiving worker stdout lines live.
    Raises RuntimeError (loudly, with stderr) if the worker process itself dies;
    per-mode pandeia failures come back as {"error": traceback} entries.
    """
    ins.NOISE_CACHE.mkdir(parents=True, exist_ok=True)
    cache = ins.NOISE_CACHE / f"{job_key(job)}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

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
    # Drain stderr on a thread while streaming stdout (v6): reading only
    # stdout to EOF deadlocks both processes once the worker writes more than
    # the OS pipe buffer (~64 KB) of engine/library chatter to stderr -- the
    # worker blocks on the stderr write, stdout never reaches EOF.
    import io
    import threading
    err_buf = io.StringIO()
    t_err = threading.Thread(target=lambda: err_buf.write(proc.stderr.read()),
                             daemon=True)
    t_err.start()
    # ``progress`` is the GUI's callback, so it can raise the exception
    # Streamlit uses to cancel a script run mid-stream (a BaseException).
    # The worker must not outlive the run that started it -- it would keep a
    # CPU busy on shared hardware after the caller's concurrency slot was
    # released.
    with proc_mod.terminating(proc):
        for line in proc.stdout:
            if progress:
                progress(line.rstrip())
        proc.wait()
    t_err.join(timeout=30)
    if proc.returncode != 0 or not out_json.exists():
        err = err_buf.getvalue()
        raise RuntimeError(f"pandeia worker failed (rc={proc.returncode}):\n{err[-3000:]}")

    result = json.loads(out_json.read_text())
    cache.write_text(json.dumps(result))
    return result


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
      * an (n, 2) array -- columns wavelength (micron), floor (ppm); linearly
        interpolated to ``wl_um`` with the endpoint values continued outside
        the supplied range (PandExo's constant edge extension). Rows are
        sorted by wavelength internally; duplicate wavelengths raise.

    Returns the per-bin floor as a FRACTIONAL depth (ppm * 1e-6). All values
    must be finite and nonnegative -- anything else raises (loudly, per the
    repo rule), never silently sanitized. The result is applied downstream as
    sigma_final = max(sigma_random, floor): a hard minimum, never a quadrature
    term, and never rescaled by the binning R.
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
# The floor is a hard minimum: wherever it binds, it lifts the per-bin variance
# from var_phot to floor^2. That LIFT -- the floor EXCESS,
#
#     excess^2 = max(0, floor^2 - var_phot)
#
# -- is a systematic budget, and treating it as white (bin-to-bin
# uncorrelated) is only one limiting assumption: real JWST transit-spectroscopy
# residuals are spectrally smooth (the generic finding of independent-pipeline
# comparisons, e.g. Holmberg & Madhusudhan 2023). A scenario RE-ALLOCATES the
# excess between a white part and a smooth part with a squared-exponential
# kernel in ln(lambda):
#
#     C = diag(var_phot + f_white * excess^2)
#         + (1 - f_white) * excess_i excess_j exp(-(ln wl_i - ln wl_j)^2 / 2 ell^2)
#
# PSD by construction (SE kernel Gram matrix, congruence-scaled by the per-bin
# excess amplitudes; the photon diagonal is strictly positive). The per-bin
# TOTAL variance is IDENTICAL in every scenario -- diag(C) = var_phot +
# excess^2 = max(var_phot, floor^2) = sigma_final^2 always -- so ranking
# changes between scenarios are attributable to correlation structure alone,
# never to a bigger error bar. Photon-dominated bins (floor not binding) carry
# no correlated part. f_white = 1 recovers the exact diagonal model
# (build_cov returns None: fast sigma path).
#
# These presets are EXPERIMENTAL stated ASSUMPTIONS bracketing the correlation
# structure, not measured or calibrated JWST covariances. They are excluded
# from headline results: the default scenario is "random" (exact diagonal,
# PandExo-style), and the GUI/README label the tier accordingly. ell is in
# ln-wavelength (0.05 ~ 5% wavelength scale; the G395H band spans ~0.6).
# "conservative" additionally profiles a per-detector-segment SLOPE nuisance
# (real per-visit fits float linear trends), on top of the per-segment
# offsets every scenario profiles.
# NOTE (2026-07-15 audit): because the correlated budget is the floor EXCESS
# max(0, floor^2 - var_phot), the correlated systematic is absent where photon
# noise dominates and fully present at N -> infinity -- it GROWS as transits
# average the photon term down. Template S/N (and Fisher forecasts) are
# therefore NOT monotone in n_transits under the correlated presets: they can
# peak at a finite transit count (where the floor just binds) and decline
# toward the floor-only limit. This is a property of the stated assumption,
# kept deliberately (it preserves the PandExo per-bin totals exactly, so
# scenario ranking differences are attributable to correlation structure
# alone); transits_to_target scans instead of gating on the limit there. A
# PSD-monotone additive-systematic model would need a NEW scenario family
# with different (documented) diagonal semantics -- never a silent swap.
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
    correlated budget is the floor EXCESS over the random variance, so
    diag(C) always equals max(var_phot, floor^2) -- the same total the
    diagonal path quotes. Consequence (see the SCENARIOS note): the
    correlated part grows as the random variance shrinks, so scores built on
    this covariance are not monotone in the transit count. Returns None for
    a fully-white scenario OR when the
    floor binds nowhere (no excess to correlate), so callers keep the exact
    diagonal fast path; otherwise a positive-definite (n_bins, n_bins)
    matrix. Unknown scenario names raise (KeyError) rather than defaulting;
    non-finite or negative variances/floors and non-positive wavelengths
    raise (2026-07-12 recheck, P2-E: a negative variance used to come back
    as a plausible positive diagonal, NaNs propagated silently).
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
    function): a fractional count has no meaning -- you cannot observe 2.7
    transits -- and silently flooring it reported a 2-transit sigma under a
    2.7-transit label. ``pixel_depth_variance`` used to do exactly that via
    ``int(n_transits)`` while its sibling scalers already refused, so the same
    input raised or was accepted depending on which entry point saw it.
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
    ingress/egress, limb darkening, and depth-detrending covariance -- the
    fast lower-bound mode; a time-domain information calculation would need
    the full light-curve derivative set.

    SYMMETRIC in/out approximation (2026-07-12 re-audit item 3): this uses the
    OUT-of-transit extracted flux and sigma for BOTH the in- and out-of-transit
    terms -- i.e. F_in ~ F_out and sigma_in ~ sigma_out. For pure source
    Poisson noise the exact box-transit propagation of depth = 1 - F_in/F_out
    (with F_in = (1-d) F_out) gives sigma_sym/sigma_exact =
    sqrt[(a+b) / ((1-d)a + (1-d)^2 b)] where a=1/n_in, b=1/n_out. For EQUAL
    in/out baselines this is 1 + 3d/4 + O(d^2) (2026-07-13 recheck item 1: the
    first-order coefficient is 3d/4, NOT d/2; for unequal baselines it ranges
    d/2 to d). The approximation is therefore CONSERVATIVE (never
    under-predicts the random sigma from this effect) and the excess grows with
    depth: about +0.075% at d=0.1%, +0.76% at 1%, +1.5% at 2%, +8.2% at 10%.
    It is kept deliberately model-INDEPENDENT (the noise never sees the depth
    spectrum, so one measurement operator bins noise and model consistently);
    exact separate in/out flux/variance propagation is a remaining refinement
    (see the module docstring and README), not a same-answer refactor.
    """
    flux = np.asarray(mode_result["flux"], float)
    noise = np.asarray(mode_result["noise_1int"], float)
    t_cycle = float(mode_result["t_cycle_s"])
    # fail-fast on inputs the worker normally guarantees but the public API
    # does not (2026-07-12 recheck, P2-E): a NaN/zero flux or cycle time
    # otherwise propagates NaN/inf variance silently.
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
    # int() floors to whole integrations (conservative); a window shorter than
    # one integration cycle yields NO usable integration -- say so, never
    # silently pretend one fits (the old max(1, ...) did exactly that).
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

    ``op`` is the mode's count-space measurement operator (binning.build_operator);
    pass the SAME operator used to bin the model so noise and model describe one
    estimator. Built here from the pixel grid alone when omitted (noise-only use).

    ``floor_spec`` is the PandExo-style minimum floor (resolve_floor: None,
    constant ppm, or a wavelength-vs-ppm table). Order of operations matches
    PandExo: (1) random variance for the requested observation, (2) binned in
    count space, (3) transits combined, (4) the floor evaluated on the final
    bin wavelengths, (5) sigma = max(sigma_random, floor).

    Returns dict(wl_center, sigma, n_pix, var_phot, floor, n_transits) over
    the operator's kept bins. ``var_phot`` is the random (photon/detector) bin
    variance AT the evaluated ``n_transits`` (it scales as 1/N, inflation
    included); ``floor`` is the resolved per-bin minimum (N-independent).
    Returning the two components separately is what lets callers extrapolate
    to other transit counts CORRECTLY -- sigma approaches the floor from
    above as N grows, never below it.

    ``noise_inflation`` is an OPTIONAL empirical sensitivity factor on the
    random (Pandeia) sigma, default 1.0 -- the Pandeia prediction as-is.
    Published achieved-vs-predicted ratios (COMPASS/Gordon+2025, Radica+2023,
    Bouwman+2023; see instruments.LITERATURE_NOISE_FACTORS) are
    program-specific, NOT a transferable calibration, so nothing is applied by
    default. Proportional noise: it averages down with transits, unlike the
    floor.
    """
    ninf = float(noise_inflation)
    if not (np.isfinite(ninf) and ninf > 0.0):
        # squared below, so a negative factor would silently act like its
        # absolute value and 0/NaN would zero/poison the sigma (recheck P2-E)
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
