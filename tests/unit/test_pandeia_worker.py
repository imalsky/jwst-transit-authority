"""pandeia_worker backend-identity helpers (no pandeia needed: the worker's
pandeia imports are function-local, so the module imports with numpy alone).

The release-match gate compares leading numeric release segments: 3.0 +
3.0rc3 passes; a mixed engine/refdata pair is refused BEFORE a deep error.
"""
import os

import pytest

from jwst_tool import instruments as ins
from jwst_tool import pandeia_worker as pw


# --- native-R dispersion lookup ----------------------------------------------

def test_native_r_finds_the_tokenless_nircam_grism_file(tmp_path):
    """Worker v12: the NIRCam LW grism dispersion file carries NO disperser
    token in its name (jwst_nircam_disp_*.fits), so the *grismr* token
    pattern matched nothing and NIRCam ran without a native-R export. The
    lookup must pick that file and never the alphabetically-earlier
    short-wave dhs0-ord* files."""
    fits = pytest.importorskip("astropy.io.fits")
    import numpy as np

    ddir = tmp_path / "jwst" / "nircam" / "dispersion"
    ddir.mkdir(parents=True)

    def _disp_fits(path, r_value):
        cols = fits.ColDefs([
            fits.Column(name="WAVELENGTH", format="D",
                        array=np.array([2.4, 5.0])),
            fits.Column(name="R", format="D",
                        array=np.array([r_value, r_value], float)),
        ])
        fits.HDUList([fits.PrimaryHDU(),
                      fits.BinTableHDU.from_columns(cols)]).writeto(path)

    # decoy sorts BEFORE the grism file; a bare *disp*.fits glob would pick it
    _disp_fits(ddir / "jwst_nircam_dhs0-ord1_disp_20240607150902.fits", 300.0)
    _disp_fits(ddir / "jwst_nircam_disp_20170901102005.fits", 1400.0)

    for key in ("nircam_f322w2", "nircam_f444w"):
        r, src = pw._native_r(str(tmp_path), ins.MODES[key], [4.0])
        assert src == "jwst_nircam_disp_20170901102005.fits", (key, src)
        assert r == [1400.0], (key, r)


# --- ngroup limits (PandExo compatibility) -----------------------------------

def test_group_caps():
    """NIRCam grism must not permit more than PandExo's hard 100-group max,
    and no mode exceeds its instrument's cap (the import-time guard mirror)."""
    assert ins.PANDEXO_NGROUP_MAX["nircam"] == 100
    assert any(m["instrument"] == "nircam" for m in ins.MODES.values())
    for key, m in ins.MODES.items():
        cap = ins.PANDEXO_NGROUP_MAX.get(m["instrument"])
        if cap is not None:
            assert m["ngroup_max"] <= cap, key


def test_ramp_floors_equal_pandeia_mingroups():
    """Since worker v8 the search floor is pandeia 2026.7's per-detector
    mingroups (jwst/<instrument>/config.json: nirspec 1, niriss 1,
    nircam 1, miri 2) -- the same field PandExo reads, so both tools search
    the same ramp space. The warn thresholds are instrument-specific
    (jwst-docs, verified 2026-08-09): NIRSpec/NIRISS warn at 1 group;
    NIRCam TSO guidance says avoid saturating in fewer than 4 groups
    (linearity-correction reliance); MIRI calls 2-5 group ramps very
    difficult to calibrate (5+ recommended)."""
    expected = {"nirspec": (1, 2), "niriss": (1, 2),
                "nircam": (1, 4), "miri": (2, 6)}
    for key, m in ins.MODES.items():
        floor, warn = expected[m["instrument"]]
        assert m["ngroup_min"] == floor, key
        assert m["ngroup_warn_below"] == warn, key
        assert 1 <= m["ngroup_min"] <= m["ngroup_warn_below"] \
            <= m["ngroup_max"], key


def test_release_segment():
    """Leading numeric release segment; rc/dev suffixes drop, non-numeric
    strings read None."""
    for raw, expected in (("3.0", "3.0"), ("3.0rc3", "3.0"),
                          ("2026.2.dev1", "2026.2"), ("  4.1 \n", "4.1"),
                          ("rc3", None), ("", None)):
        assert pw._release(raw) == expected, raw


# --- _refdata_version -------------------------------------------------------

def test_refdata_version_sources(tmp_path):
    """VERSION wins when present; a misplaced VERSION_PSF must never
    authenticate refdata (a VERSION_PSF file names a PSF library, never a
    data tree -- the retired 3.0-era fallback allowed that) while the
    pandeia_data-<ver> dir-name convention still applies; an unmarked tree
    is undeterminable."""
    (tmp_path / "VERSION").write_text("2026.2\nextra\n")
    (tmp_path / "VERSION_PSF").write_text("9.9\n")
    assert pw._refdata_version(str(tmp_path)) == ("2026.2", "VERSION")

    tree = tmp_path / "pandeia_data-2026.7-jwst"
    tree.mkdir()
    (tree / "VERSION_PSF").write_text("2026.7\n\nPSF provenance text\n")
    assert pw._refdata_version(str(tree)) == ("2026.7-jwst", "directory name")
    (tree / "VERSION_DATA").write_text("2026.7\n")
    assert pw._refdata_version(str(tree)) == ("2026.7", "VERSION_DATA")

    bare = tmp_path / "bare"
    bare.mkdir()
    assert pw._refdata_version(str(bare))[0] is None


# --- matched engine/refdata/PSF triple ---------------------------------------

def _triple(tmp_path, data_ver, psf_ver):
    """Build (refdata, psf_dir) version-file stubs at the requested releases."""
    ref = tmp_path / f"pandeia_data-{data_ver}-jwst"
    ref.mkdir()
    (ref / "VERSION_DATA").write_text(f"{data_ver}\n")
    psf = tmp_path / f"pandeia_psfs-{psf_ver}-jwst"
    psf.mkdir()
    (psf / "VERSION_PSF").write_text(f"{psf_ver}\n")
    return str(ref), str(psf)


def test_matched_triple_and_psf_identity(tmp_path):
    """A matched triple passes and records all three versions; PSF identity
    comes from VERSION_PSF, then the directory name; an unidentifiable PSF
    tree is refused."""
    ref, psf = _triple(tmp_path, "2026.7", "2026.7")
    prov = pw._check_backend_match("2026.7", ref, psf)
    assert prov["refdata_version"] == "2026.7"
    assert prov["psf_version"] == "2026.7"
    assert prov["psf_version_source"] == "VERSION_PSF"

    os.remove(os.path.join(psf, "VERSION_PSF"))
    prov = pw._check_backend_match("2026.7", ref, psf)
    assert (prov["psf_version"], prov["psf_version_source"]) == (
        "2026.7-jwst", "directory name")

    blank = tmp_path / "psfs_somewhere"
    blank.mkdir()
    with pytest.raises(RuntimeError, match="cannot determine the pandeia_psfs"):
        pw._check_backend_match("2026.7", ref, str(blank))


def test_any_component_out_of_step_is_refused(tmp_path):
    """Any component out of step must fail BEFORE a calculation; the PSF
    release is the one that used to go unchecked. Unidentifiable refdata is
    refused too."""
    for engine, data_ver, psf_ver, offender in (
            ("2026.2", "2026.7", "2026.7", "does not match pandeia_data"),
            ("2026.7", "2026.2", "2026.7", "does not match pandeia_data"),
            ("2026.7", "2026.7", "2026.2", "PSF library release"),
            ("2026.2", "2026.2", "2026.7", "PSF library release")):
        sub = tmp_path / f"{engine}-{data_ver}-{psf_ver}"
        sub.mkdir()
        ref, psf = _triple(sub, data_ver, psf_ver)
        with pytest.raises(RuntimeError, match=offender):
            pw._check_backend_match(engine, ref, psf)

    unmarked = tmp_path / "unmarked_refdata"
    unmarked.mkdir()
    with pytest.raises(RuntimeError, match="cannot determine"):
        pw._check_backend_match("3.0", str(unmarked))


def test_missing_psf_tree_is_refused(tmp_path):
    """Every backend uses the split-PSF layout since the legacy (3.0)
    backend was removed: a missing PSF dir is always an error, and the
    split 2026+ layout must never masquerade as embedded-PSF data."""
    ref, _ = _triple(tmp_path, "2026.7", "2026.7")
    for psf_dir in (None, ""):
        with pytest.raises(RuntimeError, match="requires a separate PSF"):
            pw._check_backend_match("2026.7", ref, psf_dir)


# --- backend registry --------------------------------------------------------

def test_backend_registry_is_the_single_supported_triple():
    """One backend: current = 2026.7 matched triple; unvalidated archival
    backends are not selectable."""
    assert ins.JWST_TOOL_BACKEND in ins._BACKENDS
    cur = ins._BACKENDS["current"]
    assert cur["release"] == ins._SUPPORTED_PANDEIA_RELEASE == "2026.7"
    assert cur["supported"] is True
    assert "pandeia_data-2026.7-jwst" in cur["refdata"]
    assert "pandeia_psfs-2026.7-jwst" in cur["psf"]
    assert set(ins._BACKENDS) == {"current"}
    assert set(ins._MODE_RENAMES) == {"current"}


def test_no_backend_carries_a_personal_absolute_path():
    """No checked-in SOURCE literal may point into one person's home (checks
    source text, not resolved values: refdata/psf legitimately resolve under
    a developer's home in an editable checkout)."""
    import pathlib

    src = pathlib.Path(ins.__file__).read_text()
    offenders = [ln.strip() for ln in src.splitlines()
                 if "/Users/" in ln and not ln.strip().startswith("#")]
    assert not offenders, offenders

    for token, be in ins._BACKENDS.items():
        assert be["python"] is None, (
            f"{token}: the backend interpreter is machine-specific and must "
            "come from JWST_TOOL_PANDEIA_PYTHON, not a baked-in path")


def test_missing_backend_python_gives_one_actionable_error(monkeypatch):
    monkeypatch.setattr(ins, "PANDEIA_PYTHON", None)
    with pytest.raises(RuntimeError, match="JWST_TOOL_PANDEIA_PYTHON"):
        ins.require_pandeia_python()
    monkeypatch.setattr(ins, "PANDEIA_PYTHON", "/some/env/bin/python")
    assert ins.require_pandeia_python() == "/some/env/bin/python"


# --- native-grid saturation census -------------------------------------------

def _stub_report(sat_frac, wl, flux, noise, n_full, n_part, t_exp=100.0,
                 sat_ngroups=50.0):
    """Minimal pandeia report shaped the way `_one_mode` reads it."""
    return {
        "scalar": {"fraction_saturation": sat_frac, "sat_ngroups": sat_ngroups,
                   "total_exposure_time": t_exp},
        "1d": {
            "sn": (list(wl), [0.0] * len(wl)),
            "extracted_flux": (list(wl), list(flux)),
            "extracted_noise": (list(wl), list(noise)),
            "n_full_saturated": (list(wl), list(n_full)),
            "n_partial_saturated": (list(wl), list(n_part)),
        },
        "warnings": {},
    }


def _run_one_mode(wl, flux, noise, n_full, n_part, sat_frac=0.5,
                  sat_by_ngroup=None, sat_ngroups=50.0,
                  ngroup_min=2, ngroup_max=10, call_log=None):
    """Drive `_one_mode` with stub pandeia callables (no engine involved).

    ``sat_by_ngroup`` (ngroup -> measured full-well fraction) makes
    saturation depend on the requested ramp, which the group-search
    regression tests need; the default constant ``sat_frac`` keeps the
    older census tests unchanged. ``call_log`` collects the ngroup of every
    nint=1 stub calculation, in order."""
    mode = {"key": "stub", "instrument": "nirspec", "mode": "prism",
            "ngroup_min": ngroup_min, "ngroup_max": ngroup_max}
    star = {"teff": 5400.0, "log_g": 4.45, "metallicity": 0.0, "ks_mag": 10.0}

    def build_default_calc(_tel, _inst, _mode):
        return {"configuration": {"detector": {}}, "strategy": {},
                "scene": [{"spectrum": {}}]}

    calls = {"n": 0}

    def perform_calculation(calc):
        # nint=2 costs one extra frame, so t_cycle_s comes out positive
        calls["n"] += 1
        det = calc["configuration"]["detector"]
        ng = int(det.get("ngroup", ngroup_min))
        frac = float(sat_by_ngroup(ng)) if sat_by_ngroup else sat_frac
        if call_log is not None and det.get("nint", 1) == 1:
            call_log.append(ng)
        t = 100.0 + 10.0 * (det.get("nint", 1) - 1)
        return _stub_report(frac, wl, flux, noise, n_full, n_part, t_exp=t,
                            sat_ngroups=sat_ngroups)

    # The usable-pixel return path records pandeia.engine.__version__. Stub it
    # so this suite stays engine-free (the rest of the file's contract).
    import sys
    import types

    stub = types.ModuleType("pandeia")
    stub_engine = types.ModuleType("pandeia.engine")
    stub_engine.__version__ = "2026.7"
    stub.engine = stub_engine
    added = "pandeia" not in sys.modules
    if added:
        sys.modules["pandeia"] = stub
        sys.modules["pandeia.engine"] = stub_engine
    try:
        # _native_r reads refdata dispersion files; point it at nothing so it
        # reports "unavailable" rather than reaching for a 20 MiB tree.
        return pw._one_mode(build_default_calc, perform_calculation, mode, star,
                            sat_limit=0.80, refdata="/nonexistent-refdata")
    finally:
        if added:
            sys.modules.pop("pandeia.engine", None)
            sys.modules.pop("pandeia", None)


def test_full_saturation_is_counted_before_the_good_filter():
    """Saturation is counted on NATIVE curves, before the good-pixel filter.

    A fully saturated channel has non-finite extracted noise, so counting
    from the filtered curves reported zero saturated pixels.
    """
    out = _run_one_mode(
        wl=[1.0, 2.0],
        flux=[1.0e6, 5.0e5],
        noise=[1.0e3, float("nan")],       # saturated channel -> non-finite
        n_full=[0.0, 3.0],                 # ...and it is the saturated one
        n_part=[0.0, 3.0],
    )
    assert out.get("unusable") is not True
    assert len(out["wl"]) == 1                      # one-pixel science grid
    assert out["wl"] == [1.0]
    assert out["n_full_sat"] == [0.0]               # filtered curve sees none

    assert out["n_pix_native"] == 2
    assert out["n_pix_unusable_dropped"] == 1
    assert out["n_pix_full_sat_native"] == 1        # the count that used to be 0
    assert out["n_pix_part_sat_native"] == 1


def test_native_census_survives_an_all_unusable_mode():
    """An all-unusable mode still reports its native and saturated counts."""
    out = _run_one_mode(
        wl=[1.0, 2.0],
        flux=[1.0e6, 5.0e5],
        noise=[float("nan"), float("nan")],
        n_full=[2.0, 3.0],
        n_part=[2.0, 3.0],
        sat_frac=7.31,                     # the real bright_hot PRISM value
    )
    assert out["unusable"] is True
    assert out["n_pix_native"] == 2
    assert out["n_pix_unusable_dropped"] == 2
    assert out["n_pix_full_sat_native"] == 2


def test_detect_never_substitutes_the_post_filter_count():
    """A pre-v7 payload must read UNMEASURED, not a falsely-clean zero."""
    from jwst_tool import detect

    assert detect._native_pixel_counts({}) == {
        "n_pix_native": None, "n_pix_unusable_dropped": None,
        "n_pix_part_sat_native": None, "n_pix_full_sat_native": None}
    passed = detect._native_pixel_counts(
        {"n_pix_native": 2048, "n_pix_unusable_dropped": 7,
         "n_pix_part_sat_native": 31, "n_pix_full_sat_native": 4})
    assert passed["n_pix_native"] == 2048
    assert passed["n_pix_full_sat_native"] == 4


def test_sat_curve_is_loud_on_missing_or_misaligned_keys():
    """The saturation curves are load-bearing: a missing/renamed report key or
    a grid-length mismatch raises, never the old silent all-zeros fallback."""
    import numpy as np
    rpt = {"1d": {"n_full_saturated": [[1.0, 2.0, 3.0], [0.0, 1.0, np.nan]]}}
    curve = pw._sat_curve(rpt, "n_full_saturated", 3)
    assert curve.tolist() == [0.0, 1.0, 0.0]        # NaN -> 0, values kept
    with pytest.raises(RuntimeError, match="renamed"):
        pw._sat_curve(rpt, "n_partial_saturated", 3)
    with pytest.raises(RuntimeError, match="misaligned"):
        pw._sat_curve(rpt, "n_full_saturated", 4)


# --- group search selects the largest MEASURED-safe ramp (2026-08-09 review) --

_CLEAN = dict(wl=[1.0, 2.0], flux=[1.0e6, 5.0e5], noise=[1.0e3, 2.0e3],
              n_full=[0.0, 0.0], n_part=[0.0, 0.0])

# One case per group-search invariant; each is a synthetic saturation curve.
# Fields: sat_by_ngroup, sat_ngroups, expected (ngroup, saturated,
# ramp_search_complete or None = not asserted), in_log (ngroups that must
# have been MEASURED, i.e. appear in the nint=1 call log).
_GROUP_SEARCH_CASES = {
    # committed bright_hot/niriss_soss regression: the largest measured-safe
    # count (2, PandExo's choice), not the conservative predictor seed (1)
    "soss_two_groups": (lambda ng: 0.392 * ng, 2.04, (2, False, None), [2]),
    # a predictor below the floor does not prove the floor saturated
    "predictor_below_safe_floor": (lambda ng: 0.79 * ng, 0.9,
                                   (1, False, None), []),
    # nonlinear curve: an overshooting seed comes back down to the largest
    # measured-safe count
    "overshoot_converges": (lambda ng: {1: 0.10, 7: 0.79}.get(ng, 0.90),
                            None, (7, False, None), []),
    # everything above the floor unsafe -> the measured-safe floor, never a
    # saturation flag
    "bracket_collapse": (lambda ng: 0.40 if ng == 1 else 0.90, None,
                         (1, False, None), []),
    # saturation is a MEASUREMENT of the shortest permitted ramp
    "saturated_floor": (lambda ng: 0.85 * ng, None, (1, True, None), []),
    # the upward search stops at the APT/PandExo cap
    "cap_respected": (lambda ng: 0.005 * ng, None, (30, False, None), []),
    # review round 2 counterexample f(n)=0.1n+0.1: the v9 predictor stalled
    # at 6; the bracket search must PROVE 7 by measuring 8 unsafe
    "affine_offset_maximum": (lambda ng: 0.1 * ng + 0.1, None,
                              (7, False, True), []),
    # completeness = the boundary neighbor was measured (3 in the call log)
    "maximality_proven": (lambda ng: 0.392 * ng, 2.04, (2, False, True), [3]),
}


@pytest.mark.parametrize("case", list(_GROUP_SEARCH_CASES),
                         ids=list(_GROUP_SEARCH_CASES))
def test_group_search_selects_largest_measured_safe(case):
    sat_by_ngroup, sat_ngroups, want, in_log = _GROUP_SEARCH_CASES[case]
    log = []
    out = _run_one_mode(**_CLEAN, sat_by_ngroup=sat_by_ngroup,
                        sat_ngroups=sat_ngroups, ngroup_min=1,
                        ngroup_max=30, call_log=log)
    ngroup, saturated, complete = want
    assert out["ngroup"] == ngroup
    assert out["saturated"] is saturated
    if complete is not None:
        assert out["ramp_search_complete"] is complete
    for ng in in_log:
        assert ng in log            # the choice/disproof was MEASURED
    assert not log or max(log) <= 30


def test_budget_exhaustion_is_reported_not_hidden():
    """f(n) = 0.8*n/(n+1) climbs one group per iteration forever; the
    budget runs out with no unsafe neighbor measured, so the result is the
    measured-safe best WITH ramp_search_complete=False."""
    out = _run_one_mode(**_CLEAN,
                        sat_by_ngroup=lambda ng: 0.8 * ng / (ng + 1.0),
                        sat_ngroups=None, ngroup_min=1, ngroup_max=10_000)
    assert out["saturated"] is False
    assert out["ramp_search_complete"] is False
    assert out["ngroup"] >= 10          # it climbed, measured-safe
