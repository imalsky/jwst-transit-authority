"""pandeia_worker backend-identity helpers (no pandeia needed: the worker's
pandeia imports are function-local, so the module imports with numpy alone).

The engine/refdata release-match gate is the STScI rule "engine and refdata
versions must be the same", applied to the leading numeric release segment so
the validated 3.0 + 3.0rc3 pair passes while 2026.1-engine-on-3.0rc3-refdata
(the documented base-env failure mode) is refused BEFORE a deep engine error.
"""
import os

import pytest

from jwst_tool import instruments as ins
from jwst_tool import pandeia_worker as pw


# --- ngroup limits (PandExo compatibility, 2026-07-12 audit item 5) ----------

def test_nircam_modes_respect_pandexo_group_cap():
    """NIRCam grism must not permit more than PandExo's hard 100-group max."""
    cap = ins.PANDEXO_NGROUP_MAX["nircam"]
    assert cap == 100
    nircam = [k for k, m in ins.MODES.items() if m["instrument"] == "nircam"]
    assert nircam                                        # the modes exist
    for key in nircam:
        assert ins.MODES[key]["ngroup_max"] <= cap, key


def test_every_mode_respects_its_instrument_cap():
    """The import-time guard mirror: no mode exceeds its instrument's cap."""
    for key, m in ins.MODES.items():
        cap = ins.PANDEXO_NGROUP_MAX.get(m["instrument"])
        if cap is not None:
            assert m["ngroup_max"] <= cap, key


def test_optimizer_clamp_never_exceeds_ngroup_max():
    """The worker's group selection is bounded by _clamp_ngroup at every step;
    a faint-target candidate far above ng_max still returns <= ng_max, and a
    saturated candidate below ng_min still returns >= ng_min."""
    for key, m in ins.MODES.items():
        lo, hi = m["ngroup_min"], m["ngroup_max"]
        for cand in (-5, 0, 1, lo, lo + 1, hi - 1, hi, hi + 50, 10_000):
            got = pw._clamp_ngroup(cand, lo, hi)
            assert lo <= got <= hi, (key, cand, got)

@pytest.mark.parametrize("raw, expected", [
    ("3.0", "3.0"),
    ("3.0rc3", "3.0"),
    ("2026.2", "2026.2"),
    ("2026.2.dev1", "2026.2"),
    ("  4.1 \n", "4.1"),
    ("rc3", None),
    ("", None),
])
def test_release_segment(raw, expected):
    assert pw._release(raw) == expected


# --- _refdata_version -------------------------------------------------------

def test_refdata_version_prefers_version_file(tmp_path):
    (tmp_path / "VERSION").write_text("2026.2\nextra\n")
    (tmp_path / "VERSION_PSF").write_text("9.9\n")
    ver, src = pw._refdata_version(str(tmp_path))
    assert (ver, src) == ("2026.2", "VERSION")


def test_refdata_version_falls_back_to_psf_then_dirname(tmp_path):
    # the 3.0rc3 tree ships VERSION_PSF only
    tree = tmp_path / "pandeia_data-3.0rc3"
    tree.mkdir()
    (tree / "VERSION_PSF").write_text("3.0\n\nPSF provenance text\n")
    assert pw._refdata_version(str(tree)) == ("3.0", "VERSION_PSF")
    os.remove(tree / "VERSION_PSF")
    ver, src = pw._refdata_version(str(tree))
    assert (ver, src) == ("3.0rc3", "directory name")


def test_refdata_version_undeterminable(tmp_path):
    ver, _src = pw._refdata_version(str(tmp_path))
    assert ver is None


# --- _check_backend_match ---------------------------------------------------

def test_match_accepts_validated_pair(tmp_path):
    tree = tmp_path / "pandeia_data-3.0rc3"
    tree.mkdir()
    (tree / "VERSION_PSF").write_text("3.0\n")
    prov = pw._check_backend_match("3.0", str(tree))
    assert prov["refdata_version"] == "3.0"
    assert prov["refdata_version_source"] == "VERSION_PSF"
    # no separate PSF tree on this backend: PSFs live inside refdata
    assert prov["psf_version"] is None


def test_match_refuses_mismatched_engine(tmp_path):
    tree = tmp_path / "pandeia_data-3.0rc3"
    tree.mkdir()
    (tree / "VERSION_PSF").write_text("3.0\n")
    with pytest.raises(RuntimeError, match="does not match"):
        pw._check_backend_match("2026.1", str(tree))


def test_match_refuses_unidentifiable_refdata(tmp_path):
    with pytest.raises(RuntimeError, match="cannot determine"):
        pw._check_backend_match("3.0", str(tmp_path))


# --- matched engine/refdata/PSF triple (item 3) ------------------------------

def _triple(tmp_path, data_ver, psf_ver):
    """Build a (refdata, psf_dir) pair at the requested releases.

    Version files only -- the real PSF tree is ~4 GiB and the check is a
    release comparison, so the test needs neither.
    """
    ref = tmp_path / f"pandeia_data-{data_ver}-jwst"
    ref.mkdir()
    (ref / "VERSION_DATA").write_text(f"{data_ver}\n")
    psf = tmp_path / f"pandeia_psfs-{psf_ver}-jwst"
    psf.mkdir()
    (psf / "VERSION_PSF").write_text(f"{psf_ver}\n")
    return str(ref), str(psf)


def test_matched_triple_runs_and_records_all_three_versions(tmp_path):
    ref, psf = _triple(tmp_path, "2026.7", "2026.7")
    prov = pw._check_backend_match("2026.7", ref, psf)
    assert prov["refdata_version"] == "2026.7"
    assert prov["psf_version"] == "2026.7"
    assert prov["psf_version_source"] == "VERSION_PSF"


@pytest.mark.parametrize("engine, data_ver, psf_ver, offender", [
    ("2026.2", "2026.7", "2026.7", "does not match pandeia_data"),   # engine odd
    ("2026.7", "2026.2", "2026.7", "does not match pandeia_data"),   # data odd
    ("2026.7", "2026.7", "2026.2", "PSF library release"),           # PSFs odd
    ("2026.2", "2026.2", "2026.7", "PSF library release"),           # PSFs odd
])
def test_every_pairwise_mismatch_is_refused(tmp_path, engine, data_ver,
                                            psf_ver, offender):
    """Any component out of step must fail BEFORE a calculation.

    The PSF row is the one that used to slip through: the worker only checked
    that VERSION_PSF existed, never its release, so a 2026.2 PSF tree could
    serve a 2026.7 engine and silently change the extracted flux and noise.
    """
    ref, psf = _triple(tmp_path, data_ver, psf_ver)
    with pytest.raises(RuntimeError, match=offender):
        pw._check_backend_match(engine, ref, psf)


def test_psf_release_read_from_directory_name_when_version_file_absent(tmp_path):
    ref, psf = _triple(tmp_path, "2026.7", "2026.7")
    os.remove(os.path.join(psf, "VERSION_PSF"))
    prov = pw._check_backend_match("2026.7", ref, psf)
    assert (prov["psf_version"], prov["psf_version_source"]) == (
        "2026.7-jwst", "directory name")


def test_unidentifiable_psf_tree_is_refused(tmp_path):
    ref, _ = _triple(tmp_path, "2026.7", "2026.7")
    blank = tmp_path / "psfs_somewhere"
    blank.mkdir()
    with pytest.raises(RuntimeError, match="cannot determine the pandeia_psfs"):
        pw._check_backend_match("2026.7", ref, str(blank))


def test_backend_without_separate_psf_tree_is_allowed(tmp_path):
    """The 3.0-era layout carries PSFs inside refdata; nothing to match."""
    ref, _ = _triple(tmp_path, "3.0", "3.0")
    for psf_dir in (None, ""):
        prov = pw._check_backend_match("3.0", ref, psf_dir)
        assert prov["psf_version"] is None
        assert "inside refdata" in prov["psf_version_source"]


# --- backend registry (item 3) ----------------------------------------------

def test_current_backend_is_the_supported_release_triple():
    assert ins.JWST_TOOL_BACKEND in ins._BACKENDS
    cur = ins._BACKENDS["current"]
    assert cur["release"] == ins._SUPPORTED_PANDEIA_RELEASE == "2026.7"
    assert cur["supported"] is True
    assert "pandeia_data-2026.7-jwst" in cur["refdata"]
    assert "pandeia_psfs-2026.7-jwst" in cur["psf"]


def test_archival_backend_is_named_and_labeled_unsupported():
    """The 2026.2 token was renamed, not silently repointed: caches and
    committed artifacts recorded as 'current' must not look current-release."""
    arch = ins._BACKENDS["archival_2026_2"]
    assert arch["release"] == "2026.2"
    assert arch["supported"] is False
    assert "ARCHIVAL" in arch["status"]
    assert "NOT suitable for planning new proposals" in arch["status"]
    assert ins._BACKENDS["legacy"]["supported"] is False


def test_no_backend_carries_a_personal_absolute_path():
    """No checked-in SOURCE literal may point into one person's home.

    Checks the source text, not the resolved values: refdata/psf are built from
    DATA_DIR, so in an editable checkout they legitimately resolve under the
    developer's home. The defect was a hardcoded
    /Users/imalsky/Documents/Important_Docs/... refdata for the legacy backend,
    which simply does not exist anywhere else.
    """
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
    monkeypatch.setattr(ins, "PICASO_PYTHON", None)
    with pytest.raises(RuntimeError, match="JWST_TOOL_PANDEIA_PYTHON"):
        ins.require_pandeia_python()
    monkeypatch.setattr(ins, "PICASO_PYTHON", "/some/env/bin/python")
    assert ins.require_pandeia_python() == "/some/env/bin/python"


# --- native-grid saturation census (worker v7) ------------------------------

def _stub_report(sat_frac, wl, flux, noise, n_full, n_part, t_exp=100.0):
    """Minimal pandeia report shaped the way `_one_mode` reads it."""
    return {
        "scalar": {"fraction_saturation": sat_frac, "sat_ngroups": 50.0,
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


def _run_one_mode(wl, flux, noise, n_full, n_part, sat_frac=0.5):
    """Drive `_one_mode` with stub pandeia callables (no engine involved)."""
    import numpy as np

    mode = {"key": "stub", "instrument": "nirspec", "mode": "prism",
            "ngroup_min": 2, "ngroup_max": 10}
    star = {"teff": 5400.0, "log_g": 4.45, "metallicity": 0.0, "ks_mag": 10.0}

    def build_default_calc(_tel, _inst, _mode):
        return {"configuration": {"detector": {}}, "strategy": {},
                "scene": [{"spectrum": {}}]}

    calls = {"n": 0}

    def perform_calculation(calc):
        # nint=2 costs one extra frame, so t_cycle_s comes out positive
        calls["n"] += 1
        t = 100.0 + 10.0 * (calc["configuration"]["detector"].get("nint", 1) - 1)
        return _stub_report(sat_frac, wl, flux, noise, n_full, n_part, t_exp=t)

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
    """The item-6 acceptance case: one good pixel + one fully saturated pixel.

    A fully saturated channel is exactly where pandeia returns non-finite
    extracted noise, so the `good` filter removes it. Counting saturation from
    the FILTERED curves therefore reported zero saturated pixels on a column
    that was in fact saturated. The science grid must still be the one good
    pixel; the census must still say one fully saturated pixel.
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
    """A mode with no usable pixel returns unusable=True -- and still reports
    how many native pixels were there and how many were saturated."""
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
    """worker v6: the saturation curves are LOAD-BEARING (detect drops fully
    saturated pixels through them), so a missing/renamed report key or a
    grid-length mismatch must raise -- the old silent all-zeros fallback would
    have quietly stopped masking saturated pixels after an engine rename."""
    import numpy as np
    rpt = {"1d": {"n_full_saturated": [[1.0, 2.0, 3.0], [0.0, 1.0, np.nan]]}}
    curve = pw._sat_curve(rpt, "n_full_saturated", 3)
    assert curve.tolist() == [0.0, 1.0, 0.0]        # NaN -> 0, values kept
    with pytest.raises(RuntimeError, match="renamed"):
        pw._sat_curve(rpt, "n_partial_saturated", 3)
    with pytest.raises(RuntimeError, match="misaligned"):
        pw._sat_curve(rpt, "n_full_saturated", 4)
