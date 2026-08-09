"""pandeia_worker backend-identity helpers (no pandeia needed: the worker's
pandeia imports are function-local, so the module imports with numpy alone).

The release-match gate compares leading numeric release segments: 3.0 +
3.0rc3 passes; a mixed engine/refdata pair is refused BEFORE a deep error.
"""
import os

import pytest

from jwst_tool import instruments as ins
from jwst_tool import pandeia_worker as pw


# --- ngroup limits (PandExo compatibility) -----------------------------------

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
    """_clamp_ngroup bounds any candidate into [ngroup_min, ngroup_max]."""
    for key, m in ins.MODES.items():
        lo, hi = m["ngroup_min"], m["ngroup_max"]
        for cand in (-5, 0, 1, lo, lo + 1, hi - 1, hi, hi + 50, 10_000):
            got = pw._clamp_ngroup(cand, lo, hi)
            assert lo <= got <= hi, (key, cand, got)


def test_ramp_floors_equal_pandeia_mingroups():
    """Since worker v8 the search floor is pandeia 2026.7's per-detector
    mingroups (jwst/<instrument>/config.json: nirspec 1, niriss 1,
    nircam 1, miri 2) -- the same field PandExo reads, so both tools search
    the same ramp space. The warn threshold is the STScI-recommended
    minimum (jwst-docs, verified 2026-08-09): 2 for the NIR modes (1-group
    ramps are new since Cycle 4), 5 for MIRI."""
    expected = {"nirspec": (1, 2), "niriss": (1, 2),
                "nircam": (1, 2), "miri": (2, 5)}
    for key, m in ins.MODES.items():
        floor, warn = expected[m["instrument"]]
        assert m["ngroup_min"] == floor, key
        assert m["ngroup_warn_below"] == warn, key
        assert 1 <= m["ngroup_min"] <= m["ngroup_warn_below"] \
            <= m["ngroup_max"], key

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


def test_refdata_version_never_reads_a_misplaced_psf_marker(tmp_path):
    # A VERSION_PSF file names a PSF library, never a data tree: with every
    # supported backend on the split layout, a misplaced PSF marker must not
    # authenticate refdata (the retired 3.0-era fallback allowed that). The
    # dir-name convention still applies.
    tree = tmp_path / "pandeia_data-2026.7-jwst"
    tree.mkdir()
    (tree / "VERSION_PSF").write_text("2026.7\n\nPSF provenance text\n")
    ver, src = pw._refdata_version(str(tree))
    assert (ver, src) == ("2026.7-jwst", "directory name")
    (tree / "VERSION_DATA").write_text("2026.7\n")
    assert pw._refdata_version(str(tree)) == ("2026.7", "VERSION_DATA")


def test_refdata_version_undeterminable(tmp_path):
    ver, _src = pw._refdata_version(str(tmp_path))
    assert ver is None


# --- _check_backend_match ---------------------------------------------------

def test_match_accepts_validated_pair(tmp_path):
    # a matched engine/refdata release with a matching PSF tree passes and
    # records the provenance
    ref, psf = _triple(tmp_path, "3.0", "3.0")
    prov = pw._check_backend_match("3.0", ref, psf)
    assert prov["refdata_version"] == "3.0"
    assert prov["psf_version"] == "3.0"


def test_match_refuses_mismatched_engine(tmp_path):
    tree = tmp_path / "pandeia_data-3.0rc3"
    tree.mkdir()
    (tree / "VERSION_PSF").write_text("3.0\n")
    with pytest.raises(RuntimeError, match="does not match"):
        pw._check_backend_match("2026.1", str(tree))


def test_match_refuses_unidentifiable_refdata(tmp_path):
    with pytest.raises(RuntimeError, match="cannot determine"):
        pw._check_backend_match("3.0", str(tmp_path))


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
    """Any component out of step must fail BEFORE a calculation; the PSF
    release is the one that used to go unchecked."""
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


def test_any_backend_without_separate_psf_tree_is_refused(tmp_path):
    """Every backend uses the split-PSF layout since the legacy (3.0)
    backend was removed: a missing PSF dir is always an error."""
    ref, _ = _triple(tmp_path, "3.0", "3.0")
    for psf_dir in (None, ""):
        with pytest.raises(RuntimeError, match="requires a separate PSF"):
            pw._check_backend_match("3.0", ref, psf_dir)


@pytest.mark.parametrize("psf_dir", [None, ""])
def test_current_backend_without_separate_psf_tree_is_refused(tmp_path,
                                                               psf_dir):
    """The split 2026+ layout must never masquerade as embedded-PSF data."""
    ref, _ = _triple(tmp_path, "2026.7", "2026.7")
    with pytest.raises(RuntimeError, match="requires a separate PSF library"):
        pw._check_backend_match("2026.7", ref, psf_dir)


# --- backend registry --------------------------------------------------------

def test_current_backend_is_the_supported_release_triple():
    assert ins.JWST_TOOL_BACKEND in ins._BACKENDS
    cur = ins._BACKENDS["current"]
    assert cur["release"] == ins._SUPPORTED_PANDEIA_RELEASE == "2026.7"
    assert cur["supported"] is True
    assert "pandeia_data-2026.7-jwst" in cur["refdata"]
    assert "pandeia_psfs-2026.7-jwst" in cur["psf"]


def test_archival_backend_is_named_and_labeled_unsupported():
    """The 2026.2 token was renamed, not silently repointed."""
    arch = ins._BACKENDS["archival_2026_2"]
    assert arch["release"] == "2026.2"
    assert arch["supported"] is False
    assert "ARCHIVAL" in arch["status"]
    assert "NOT suitable for planning new proposals" in arch["status"]
    # the Pandeia 3.0 "legacy" backend was removed outright
    assert "legacy" not in ins._BACKENDS


def test_no_backend_carries_a_personal_absolute_path():
    """No checked-in SOURCE literal may point into one person's home.

    Checks the source text, not resolved values: refdata/psf legitimately
    resolve under a developer's home in an editable checkout.
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
    monkeypatch.setattr(ins, "PANDEIA_PYTHON", None)
    with pytest.raises(RuntimeError, match="JWST_TOOL_PANDEIA_PYTHON"):
        ins.require_pandeia_python()
    monkeypatch.setattr(ins, "PANDEIA_PYTHON", "/some/env/bin/python")
    assert ins.require_pandeia_python() == "/some/env/bin/python"


# --- native-grid saturation census -------------------------------------------

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
