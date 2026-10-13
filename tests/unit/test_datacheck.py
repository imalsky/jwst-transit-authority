"""datacheck: data-availability detection (pure stdlib, path-injectable).

Everything here runs without the chemistry stack, pandeia, or streamlit --
the path-based checks take explicit tmp paths, and the engine-config-backed
checks are exercised only when the shared engine's data root is
importable (they skip cleanly on the dependency-light CI).
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from jwst_tool import datacheck
from jwst_tool import forward


# Pandeia backend path checks (tmp-path injected; mirrors the worker preflight)

def _backend_release():
    from jwst_tool import instruments as ins
    return ins.BACKEND_RELEASE


def _matched_backend(root, rel, psf_rel=None):
    """python + refdata + PSF triple at ``rel`` (PSF optionally at another)."""
    py = root / "env" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.touch()
    ref = root / f"pandeia_data-{rel}-jwst"
    ref.mkdir()
    (ref / "VERSION").write_text(f"{rel}\n")
    psf = root / "psfs"
    psf.mkdir()
    (psf / "VERSION_PSF").write_text(f"{psf_rel or rel}\n")
    return py, ref, psf


def test_pandeia_backend_present_reads_version(tmp_path):
    """A MATCHED triple at the active backend's release reports OK."""
    rel = _backend_release()
    py, ref, psf = _matched_backend(tmp_path, rel)
    items = datacheck.check_pandeia_backend(python=py, refdata=ref,
                                            psf_dir=str(psf))
    by = {it.key: it for it in items}
    assert by["pandeia:python"].status == datacheck.OK
    assert by["pandeia:refdata"].status == datacheck.OK
    assert rel in by["pandeia:refdata"].detail           # version surfaced
    assert by["pandeia:psf"].status == datacheck.OK
    assert rel in by["pandeia:psf"].detail               # PSF release surfaced


def test_pandeia_backend_missing_and_misconfigured(tmp_path):
    """Grouped failure modes; every failing item must carry a remedy.

    The PSF blocks pin the worker-preflight rules: a PSF tree from a
    DIFFERENT release must not read as OK (the worker refuses the set, so
    the status panel has to agree); VERSION_PSF is required, not just an
    existing directory; and an empty psf_dir is a misconfiguration that
    must surface, never read as "embedded PSFs".
    """
    # all three paths absent
    items = datacheck.check_pandeia_backend(
        python=tmp_path / "nope" / "python",
        refdata=tmp_path / "nope" / "refdata",
        psf_dir=str(tmp_path / "nope" / "psfs"))
    by = {it.key: it for it in items}
    assert by["pandeia:python"].status == datacheck.MISSING
    assert by["pandeia:refdata"].status == datacheck.MISSING
    assert by["pandeia:psf"].status == datacheck.MISSING
    assert all(it.required for it in items)
    assert all(it.remedy for it in items)          # every failure has a remedy

    # PSF tree from a different release than the matched refdata
    rel = _backend_release()
    other = "2026.2" if rel != "2026.2" else "2026.7"
    py, ref, psf = _matched_backend(tmp_path / "mismatch", rel, psf_rel=other)
    by = {it.key: it for it in datacheck.check_pandeia_backend(
        python=py, refdata=ref, psf_dir=str(psf))}
    assert by["pandeia:psf"].status == datacheck.MISSING
    assert "RELEASE MISMATCH" in by["pandeia:psf"].detail
    assert other in by["pandeia:psf"].detail and rel in by["pandeia:psf"].detail

    # psf dir that exists but has no VERSION_PSF file
    bare = tmp_path / "bare_psfs"
    bare.mkdir()
    by = {it.key: it for it in datacheck.check_pandeia_backend(
        python=tmp_path / "python", refdata=tmp_path / "ref",
        psf_dir=str(bare))}
    assert by["pandeia:psf"].status == datacheck.MISSING
    assert "VERSION_PSF" in by["pandeia:psf"].detail

    # empty psf_dir string (every supported backend uses the split-PSF layout)
    by = {it.key: it for it in datacheck.check_pandeia_backend(
        python=tmp_path / "python", refdata=tmp_path / "ref", psf_dir="")}
    assert by["pandeia:psf"].status == datacheck.MISSING
    assert by["pandeia:psf"].required is True


# synphot CDBS checks

def _make_cdbs(tmp_path):
    cdbs = tmp_path / "cdbs"
    (cdbs / "grid" / "phoenix").mkdir(parents=True)
    (cdbs / "comp" / "nonhst").mkdir(parents=True)
    (cdbs / "comp" / "nonhst" / "2mass_ks_001_syn.fits").touch()
    (cdbs / "calspec").mkdir()
    (cdbs / "calspec" / "alpha_lyr_stis_011.fits").touch()
    return cdbs


def test_cdbs_tree_states(tmp_path):
    # complete tree: all OK
    cdbs = _make_cdbs(tmp_path)
    items = datacheck.check_synphot_cdbs(cdbs)
    assert [it.status for it in items] == [datacheck.OK] * 3

    # missing everything: reported with remedies and recognizable labels
    items = datacheck.check_synphot_cdbs(tmp_path / "empty_cdbs")
    assert [it.status for it in items] == [datacheck.MISSING] * 3
    assert all(it.remedy for it in items)
    labels = " ".join(it.label for it in items)
    assert "PHOENIX" in labels and "Vega" in labels and "Ks" in labels

    # dangling phoenix symlink (like a fresh clone) must read MISSING
    phx = cdbs / "grid" / "phoenix"
    phx.rmdir()
    os.symlink(tmp_path / "gone", phx)
    by = {it.key: it for it in datacheck.check_synphot_cdbs(cdbs)}
    assert by["cdbs:phoenix"].status == datacheck.MISSING
    assert "dangling" in by["cdbs:phoenix"].detail


# PICASO engine checks (optional section; tmp trees, sparse CK files)

def _picaso_tree(root, version="4.1"):
    ref = root / "picaso_refdata"
    for rel in datacheck.PICASO_REFDATA_FILES:
        (ref / rel).parent.mkdir(parents=True, exist_ok=True)
        (ref / rel).write_text("x")
    (ref / "config.json").write_text(json.dumps({"version": version}))
    ck = ref / datacheck.PICASO_CK_DIR
    ck.mkdir(parents=True)
    for name in datacheck.PICASO_CK_MD5:
        with open(ck / name, "wb") as w:
            w.truncate(datacheck.PICASO_CK_BYTES)
    return ref


@pytest.mark.parametrize("case, expect, fragment", [
    ("complete", ("ok", "ok", "ok"), "version 4.1"),
    ("absent", ("missing",) * 3, "absent: all"),
    ("wrong_version", ("ok", "missing", "ok"), "version 4.0, need 4.1"),
    ("missing_file", ("ok", "missing", "ok"), "ck_cx_cont_opacities_661.db"),
    ("short_table", ("ok", "ok", "missing"), "not 61806680 bytes: H3+_1460"),
    ("empty_config", ("ok", "missing", "ok"), "version unreadable, need 4.1"),
    ("list_config", ("ok", "missing", "ok"), "version unreadable, need 4.1"),
])
def test_check_picaso_states(tmp_path, monkeypatch, case, expect, fragment):
    from jwst_tool import instruments as ins
    from jwst_tool import provenance
    py = tmp_path / "python"
    if case == "absent":
        ref = tmp_path / "nowhere"
    else:
        py.touch()
        ref = _picaso_tree(tmp_path, "4.0" if case == "wrong_version" else "4.1")
    if case in ("empty_config", "list_config"):
        (ref / "config.json").write_text("" if case == "empty_config" else "[]")
    if case == "missing_file":
        (ref / "climate_INPUTS" / "ck_cx_cont_opacities_661.db").unlink()
    if case == "short_table":
        with open(ref / datacheck.PICASO_CK_DIR / "H3+_1460.hdf5", "r+b") as w:
            w.truncate(1000)
    items = datacheck.check_picaso(python=py, refdata=ref)
    assert [it.key for it in items] == ["picaso:python", "picaso:refdata",
                                        "picaso:ck"]
    assert tuple(it.status for it in items) == expect
    assert any(fragment in it.detail for it in items)
    assert not any(it.required for it in items)
    assert "JWST_TOOL_PICASO_PYTHON" in items[0].remedy
    assert all("jwst-tool fetch --picaso" in it.remedy for it in items[1:])
    # optional: a machine without PICASO keeps the required-data banner
    rep = {"sections": {datacheck.PICASO_SECTION: items}}
    assert datacheck.missing_required(rep) == []
    assert datacheck.picaso_missing(rep) == [
        it for it in items if it.status == datacheck.MISSING]

    # the export's PICASO record never raises on the same broken setups: the
    # touched "python" is not executable, and empty_config's one hangs
    monkeypatch.setattr(ins, "PICASO_PYTHON", str(py))
    monkeypatch.setattr(ins, "PICASO_REFDATA", str(ref))
    if case == "empty_config":
        def hang(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        monkeypatch.setattr(provenance.subprocess, "run", hang)
    rec = provenance._picaso_identity()
    assert rec["picaso"] == "unavailable"
    assert rec["refdata"]["version"] == {
        "absent": "missing", "wrong_version": "4.0", "empty_config":
        "unreadable", "list_config": "unreadable"}.get(case, "4.1")
    assert rec["climate_tables"] == {"zenodo_record": "18644980",
                                     "count": 36, "status": expect[2]}
    assert str(tmp_path) not in json.dumps(rec)


# Report plumbing (no external deps at all)

def test_full_report_structure_formatting_and_cache_stats():
    rep = datacheck.full_report(base_mols=forward.MOLECULES,
                                extra_mols=forward.EXTRA_MOLECULES)
    items = datacheck.all_items(rep)
    assert items, "report must not be empty"
    assert all(it.status in (datacheck.OK, datacheck.MISSING, datacheck.AUTO)
               for it in items)
    # consistency of the two summary helpers
    assert datacheck.required_ok(rep) == (not datacheck.missing_required(rep))
    # every missing/auto item must tell the user how to fix it
    assert all(it.remedy for it in items if it.status != datacheck.OK)
    # the rendered report mentions every item
    text = datacheck.format_report(rep)
    for it in items:
        assert it.label in text
    assert "Generated caches" in text
    assert datacheck.PICASO_SECTION in rep["sections"]
    assert datacheck.picaso_missing({"sections": {}}) == []   # older cache
    stats = datacheck.cache_stats()
    for key in ("model_cache", "noise_cache"):
        assert set(stats[key]) == {"n", "mb"}
        assert stats[key]["n"] >= 0 and stats[key]["mb"] >= 0


def test_cli_data_subcommand(capsys):
    from jwst_tool import cli
    rc = cli._data_status([])
    out = capsys.readouterr().out
    assert rc in (0, 1)
    assert "jwst-tool data status" in out
    rc_bad = cli._data_status(["--bogus"])
    assert rc_bad == 2


# Engine-config-backed checks (skip when the engine data root is unset)

_engine = datacheck._engine_config()
needs_engine = pytest.mark.skipif(
    isinstance(_engine, Exception),
    reason="engine data root not configured here (set VULCAN_FORWARD_DATA)")


def test_engine_data_unavailable_is_one_loud_item(monkeypatch):
    monkeypatch.setattr(datacheck, "_engine_config",
                        lambda: RuntimeError("tree not found"))
    items = datacheck.check_engine_data(["H2O"], [])
    assert len(items) == 1
    assert items[0].status == datacheck.MISSING and items[0].required
    assert "tree not found" in items[0].detail


# --- ExoMolOP k-table coverage -----------------------------------------------

class _FakeEngineCfg:
    """Just enough engine-config surface for check_engine_data."""

    def __init__(self, root):
        self.DATA_DIR = root
        self.EXOMOLOP_DIR = root / "exomolop"
        self.CIA_H2H2_FILE = root / "opacity_cache" / "H2-H2_2011.cia"
        self.CIA_H2HE_FILE = root / "opacity_cache" / "H2-He_2011.cia"


def _fake_root(tmp_path, ktables=()):
    (tmp_path / "exomolop").mkdir()
    for m in ktables:
        # datacheck only stats paths, so an empty file stands in for 389 MB
        (tmp_path / "exomolop" / f"{m}.ktable.h5").touch()
    return tmp_path


def test_exomolop_status_never_auto(tmp_path, monkeypatch):
    cfg = _FakeEngineCfg(_fake_root(tmp_path, ["H2O"]))
    monkeypatch.setattr(datacheck, "_engine_config", lambda: cfg)
    status = datacheck.exomolop_table_status(["H2O", "CO2"])
    assert status == {"H2O": datacheck.OK, "CO2": datacheck.MISSING}


def test_missing_ktables_are_required_items_with_the_fetch_command(
        tmp_path, monkeypatch):
    # The false green this check exists to prevent: `jwst-tool data`
    # reporting green while the data behind every run is entirely absent.
    cfg = _FakeEngineCfg(_fake_root(tmp_path, ["H2O"]))
    monkeypatch.setattr(datacheck, "_engine_config", lambda: cfg)
    items = {i.key: i for i in datacheck.check_engine_data(
        ["H2O", "CO2"], ["HCN", "CS2"])}
    assert items["ktable:H2O"].status == datacheck.OK
    assert items["ktable:H2O"].required
    assert items["ktable:CO2"].status == datacheck.MISSING
    assert items["ktable:CO2"].required
    assert items["ktable:HCN"].status == datacheck.MISSING
    assert not items["ktable:HCN"].required
    # CS2 has no published table and is refused upstream: no item for it
    assert "ktable:CS2" not in items
    assert "fetch_exomolop --molecules CO2,HCN" in items["ktable:CO2"].remedy
    # the fetcher's provenance record is reported (optional, with the fetch
    # command that backfills it without re-downloading)
    prov = items["ktable:provenance"]
    assert prov.status == datacheck.MISSING and not prov.required
    assert "fetch_exomolop" in prov.remedy


@needs_engine
def test_exomolop_path_matches_the_engine_convention():
    # datacheck rebuilds <MOL>.ktable.h5 from EXOMOLOP_DIR without importing
    # the RT stack; pin it against the engine's own table_path so the two
    # conventions cannot drift.
    from vulcan_forward import exomolop
    assert datacheck.exomolop_table_path("H2O") == exomolop.table_path("H2O")
