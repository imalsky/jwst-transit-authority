"""datacheck: data-availability detection (pure stdlib, path-injectable).

Everything here runs without the chemistry stack, pandeia, or streamlit --
the path-based checks take explicit tmp paths, and the engine-config-backed
checks are exercised only when the sibling vulcan-retrieval install is
importable (they skip cleanly on the dependency-light CI).
"""
from __future__ import annotations

import os

import pytest

from jwst_tool import datacheck
from jwst_tool import forward


# ---------------------------------------------------------------------------
# Pandeia backend path checks (tmp-path injected; mirrors the worker preflight)
# ---------------------------------------------------------------------------

def test_pandeia_backend_all_missing(tmp_path):
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


def test_pandeia_backend_present_reads_version(tmp_path):
    py = tmp_path / "env" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.touch()
    ref = tmp_path / "pandeia_data-2026.2-jwst"
    ref.mkdir()
    (ref / "VERSION").write_text("2026.2\n")
    psf = tmp_path / "psfs"
    psf.mkdir()
    (psf / "VERSION_PSF").write_text("2026.2\n")
    items = datacheck.check_pandeia_backend(python=py, refdata=ref,
                                            psf_dir=str(psf))
    by = {it.key: it for it in items}
    assert by["pandeia:python"].status == datacheck.OK
    assert by["pandeia:refdata"].status == datacheck.OK
    assert "2026.2" in by["pandeia:refdata"].detail       # version surfaced
    assert by["pandeia:psf"].status == datacheck.OK


def test_pandeia_psf_dir_without_version_file_is_missing(tmp_path):
    # worker preflight requires VERSION_PSF, not just an existing directory
    psf = tmp_path / "psfs"
    psf.mkdir()
    items = datacheck.check_pandeia_backend(
        python=tmp_path / "python", refdata=tmp_path / "ref",
        psf_dir=str(psf))
    by = {it.key: it for it in items}
    assert by["pandeia:psf"].status == datacheck.MISSING
    assert "VERSION_PSF" in by["pandeia:psf"].detail


def test_pandeia_no_psf_dir_configured_yields_no_psf_item(tmp_path):
    # legacy (3.0-era) backend: PSFs live inside refdata, psf_dir is empty
    items = datacheck.check_pandeia_backend(
        python=tmp_path / "python", refdata=tmp_path / "ref", psf_dir="")
    assert not any(it.key == "pandeia:psf" for it in items)


# ---------------------------------------------------------------------------
# synphot CDBS checks
# ---------------------------------------------------------------------------

def _make_cdbs(tmp_path):
    cdbs = tmp_path / "cdbs"
    (cdbs / "grid" / "phoenix").mkdir(parents=True)
    (cdbs / "comp" / "nonhst").mkdir(parents=True)
    (cdbs / "comp" / "nonhst" / "2mass_ks_001_syn.fits").touch()
    (cdbs / "calspec").mkdir()
    (cdbs / "calspec" / "alpha_lyr_stis_011.fits").touch()
    return cdbs


def test_cdbs_complete_tree_is_ok(tmp_path):
    items = datacheck.check_synphot_cdbs(_make_cdbs(tmp_path))
    assert [it.status for it in items] == [datacheck.OK] * 3


def test_cdbs_missing_pieces_reported_with_remedies(tmp_path):
    items = datacheck.check_synphot_cdbs(tmp_path / "empty_cdbs")
    assert [it.status for it in items] == [datacheck.MISSING] * 3
    assert all(it.remedy for it in items)
    labels = " ".join(it.label for it in items)
    assert "PHOENIX" in labels and "Vega" in labels and "Ks" in labels


def test_cdbs_dangling_phoenix_symlink_is_missing(tmp_path):
    cdbs = _make_cdbs(tmp_path)
    phx = cdbs / "grid" / "phoenix"
    phx.rmdir()
    os.symlink(tmp_path / "gone", phx)           # dangling, like a fresh clone
    items = datacheck.check_synphot_cdbs(cdbs)
    by = {it.key: it for it in items}
    assert by["cdbs:phoenix"].status == datacheck.MISSING
    assert "dangling" in by["cdbs:phoenix"].detail


# ---------------------------------------------------------------------------
# Report plumbing (no external deps at all)
# ---------------------------------------------------------------------------

def test_full_report_structure_and_flags():
    rep = datacheck.full_report(base_mols=forward.MOLECULES,
                                extra_mols=forward.EXTRA_MOLECULES)
    items = datacheck.all_items(rep)
    assert items, "report must not be empty"
    assert all(it.section for it in items)        # section filled in
    assert all(it.status in (datacheck.OK, datacheck.MISSING, datacheck.AUTO)
               for it in items)
    # consistency of the two summary helpers
    assert datacheck.required_ok(rep) == (not datacheck.missing_required(rep))
    # every missing/auto item must tell the user how to fix it
    assert all(it.remedy for it in items if it.status != datacheck.OK)


def test_format_report_mentions_every_item():
    rep = datacheck.full_report(base_mols=forward.MOLECULES,
                                extra_mols=forward.EXTRA_MOLECULES)
    text = datacheck.format_report(rep)
    for it in datacheck.all_items(rep):
        assert it.label in text
    assert "Generated caches" in text


def test_cache_stats_shape():
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


# ---------------------------------------------------------------------------
# Engine-config-backed checks (skip when vulcan-retrieval is not installed)
# ---------------------------------------------------------------------------

_engine = datacheck._engine_config()
needs_engine = pytest.mark.skipif(
    isinstance(_engine, Exception),
    reason="vulcan-retrieval engine config not importable here")


@needs_engine
def test_molecule_linelist_status_statuses():
    status = datacheck.molecule_linelist_status(
        forward.MOLECULES + forward.EXTRA_MOLECULES)
    assert set(status) == set(forward.MOLECULES + forward.EXTRA_MOLECULES)
    assert all(v in (datacheck.OK, datacheck.AUTO, datacheck.MISSING)
               for v in status.values())
    # an unknown molecule is MISSING, never silently OK
    assert datacheck.molecule_linelist_status(["NOT_A_MOL"]) == {
        "NOT_A_MOL": datacheck.MISSING}


@needs_engine
def test_linelist_path_broadening_layout():
    p_air = datacheck.linelist_path("H2O", "air")
    p_h2he = datacheck.linelist_path("H2O", "h2he")
    assert p_air is not None and p_air.name == "H2O.h5"
    # h2he caches live in an h2he/ SUBDIR with a radis-parseable stem (the
    # "<db>_h2he" suffix layout broke MdbHitran; pinned in the engine repo)
    assert p_h2he is not None and p_h2he.name == "H2O.h5"
    assert p_h2he.parent.name == "h2he"
    assert datacheck.linelist_path("CO") is None      # cached ExoMol, not HITRAN


def test_engine_data_unavailable_is_one_loud_item(monkeypatch):
    monkeypatch.setattr(datacheck, "_engine_config",
                        lambda: RuntimeError("tree not found"))
    items = datacheck.check_engine_data(["H2O"], [])
    assert len(items) == 1
    assert items[0].status == datacheck.MISSING and items[0].required
    assert "tree not found" in items[0].detail
