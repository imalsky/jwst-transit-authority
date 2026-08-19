"""Archive snapshot / custom-fill unit tests (stdlib + planets only, no
network, light-CI safe). The shipped snapshot is a test fixture here: its
provenance, schema, and required-field guarantees are part of the contract.
Fill-refusal assertions are merged into one test per behavior (maintainer:
fewer, stronger tests).
"""
from __future__ import annotations

import re

import pytest

from jwst_tool import archive, planets


def test_snapshot_contract_lookup_and_loud_error_paths(tmp_path):
    """The shipped snapshot loads with provenance and its ADQL-guaranteed
    fields; lookup normalizes name variants and raises on unknowns; every
    malformed snapshot file raises SnapshotError, never loads partially."""
    snap = archive.load_snapshot()
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", snap.fetched_utc)
    assert len(snap.names) >= 3000
    assert len(snap.rows) == len(snap.names)
    # the ADQL WHERE guarantees these four fields on every shipped row
    for row in list(snap.rows.values())[::200]:
        for col in ("st_teff", "st_rad", "pl_radj", "pl_orbsmax"):
            assert str(row[col]).strip(), (row["pl_name"], col)

    for variant in ("WASP-39 b", "WASP-39b", " wasp-39  B ", "wasp-39 b"):
        assert archive.lookup(variant)["pl_name"] == "WASP-39 b", variant
    with pytest.raises(KeyError, match="Totally Fake Planet 9x"):
        archive.lookup("Totally Fake Planet 9x")

    with pytest.raises(archive.SnapshotError, match="not found"):
        archive.load_snapshot(str(tmp_path / "missing.csv"))
    p = tmp_path / "noprov.csv"
    p.write_text(",".join(archive.SNAPSHOT_COLUMNS) + "\n")
    with pytest.raises(archive.SnapshotError, match="fetched"):
        archive.load_snapshot(str(p))
    q = tmp_path / "drift.csv"
    q.write_text("# fetched: 2026-08-09T00:00:00Z\npl_name,bogus\n")
    with pytest.raises(archive.SnapshotError, match="header"):
        archive.load_snapshot(str(q))
    r = tmp_path / "short_row.csv"
    r.write_text("# fetched: 2026-08-09T00:00:00Z\n"
                 + ",".join(archive.SNAPSHOT_COLUMNS) + "\n"
                 + "Only-Two b,G8 V\n")
    with pytest.raises(archive.SnapshotError, match="cells"):
        archive.load_snapshot(str(r))


def _row(**over):
    """A hand-built archive row with every column present and in range."""
    base = {"pl_name": "Testplanet-1 b", "st_spectype": "G8 V",
            "st_teff": "5485.0", "st_logg": "4.50", "st_met": "0.10",
            "st_metratio": "[Fe/H]", "st_rad": "0.932", "sy_kmag": "10.20",
            "pl_radj": "1.279", "pl_bmassj": "0.281", "pl_bmassprov": "Mass",
            "pl_orbsmax": "0.04828", "pl_trandur": "2.80",
            "pl_radjlim": "0", "pl_bmassjlim": "0"}
    base.update(over)
    return base


def test_custom_fill_maps_derives_gravity_and_never_selects_uv():
    """A complete row maps every field and derives gravity from mass +
    radius, with exactly two disclosures. Standing maintainer rule: the
    fill NEVER writes the UV menu -- no substitute is ever selected; the note
    names the nearest-Teff shipped template as a suggestion only."""
    values, notes = archive.custom_fill(_row())
    assert values["teff"] == 5485.0 and values["logg"] == 4.5
    assert values["feh"] == 0.10 and values["ks"] == 10.20
    assert values["rstar"] == 0.932 and values["rp"] == 1.279
    assert values["a"] == 0.04828 and values["t14"] == 2.80
    g_expect = (planets.G_CGS * 0.281 * planets.M_JUP_G
                / (1.279 * planets.R_JUP_CM) ** 2) / 100.0
    assert values["g"] == pytest.approx(g_expect, rel=1e-12)
    assert len(notes) == 2
    assert any("nominal composite planning value" in n for n in notes)
    uv = [n for n in notes if "Nearest-Teff shipped template" in n]
    assert len(uv) == 1 and "G8 V" in uv[0]
    assert "sflux" not in values
    assert any("was NOT changed" in n and "never substitutes" in n
               for n in notes)


def test_custom_fill_refusals_and_disclosures():
    """Every refusal path, all fail-closed with a named note: out-of-range
    values are never clamped, missing cells are named, non-[Fe/H]
    metallicities are never re-based (the never-guess rule), one-sided limits
    never become values, malformed cells raise SnapshotError."""
    # out of range: never clamped, partial fill disclosed
    values, notes = archive.custom_fill(_row(st_teff="8850.0"))
    assert "teff" not in values                       # never clamped
    hits = [n for n in notes if "8850" in n and "3000" in n and "7000" in n
            and "left unchanged" in n]
    assert len(hits) == 1
    # no Teff filled -> not even a nearest-template suggestion
    assert "sflux" not in values
    assert any("UV spectrum menu was not changed" in n for n in notes)
    assert values["rp"] == 1.279          # in-range fields still fill

    # missing values are named; Msini provenance disclosed
    values, notes = archive.custom_fill(
        _row(sy_kmag="", pl_bmassj="", st_met=""))
    for absent in ("ks", "g", "feh"):
        assert absent not in values
    assert any("no Ks magnitude" in n for n in notes)
    assert any("no mass and radius pair" in n for n in notes)
    assert any("no stellar metallicity" in n for n in notes)
    v2, n2 = archive.custom_fill(_row(pl_bmassprov="Msini"))
    assert "g" in v2
    assert any("mass provenance 'Msini'" in n for n in n2)

    # [M/H] and [Fe/H] are different archive quantities: refused with a note
    v, n = archive.custom_fill(_row(st_metratio="[M/H]"))
    assert "feh" not in v
    assert any("[M/H]" in x and "not [Fe/H]" in x for x in n)
    v2, n2 = archive.custom_fill(_row(st_metratio=""))
    assert "feh" not in v2
    assert any("unstated" in x for x in n2)

    # one-sided limits never become values
    v, n = archive.custom_fill(_row(pl_bmassjlim="1"))
    assert "g" not in v and "rp" in v
    assert any("one-sided limit" in x and "gravity" in x for x in n)
    v2, n2 = archive.custom_fill(_row(pl_radjlim="-1"))
    assert "rp" not in v2 and "g" not in v2
    assert any("planet radius" in x and "one-sided limit" in x for x in n2)

    # malformed cells raise
    with pytest.raises(archive.SnapshotError, match="unreadable"):
        archive.custom_fill(_row(st_teff="five thousand"))
    with pytest.raises(archive.SnapshotError, match="non-finite"):
        archive.custom_fill(_row(st_teff="nan"))


def test_nearest_sflux_anchors():
    assert set(planets.SFLUX_TEFF_ANCHORS) == set(planets.SFLUX_CHOICES)
    for fname, teff in planets.SFLUX_TEFF_ANCHORS.items():
        assert planets.nearest_sflux(teff) == fname
    assert planets.nearest_sflux(5772.0) == "Gueymard_solar.txt"
    assert planets.nearest_sflux(3100.0) == "sflux-GJ1214.txt"
    assert planets.nearest_sflux(7000.0) == "Gueymard_solar.txt"


def test_archive_fill_feeds_canonical_params():
    """End-to-end: a real snapshot row, mapped by custom_fill, builds a valid
    custom-planet canonical parameter set -- pinning the g (m s^-2) -> gs_cgs
    x100 widget-unit conversion and the derived-T_irr path."""
    from jwst_tool import forward

    values, _ = archive.custom_fill(archive.lookup("HD 189733 b"))
    assert "sflux" not in values          # the fill never selects one
    cp = forward.canonical_params(dict(
        planet="custom", tp_mode="guillot",
        star_teff=values["teff"], star_logg=values["logg"],
        star_feh=values["feh"], rstar_rsun=values["rstar"],
        orbit_au=values["a"], rp_rjup=values["rp"],
        gs_cgs=values["g"] * 100.0,
        sflux="sflux-HD189_Moses11.txt"))   # the USER's explicit choice
    # canonical_params quantizes gs_cgs; the mapping must survive the round
    assert cp["gs_cgs"] == pytest.approx(values["g"] * 100.0, rel=1e-4)
    assert cp["sflux"] == "sflux-HD189_Moses11.txt"
    teq = planets.system_teq(values["teff"], values["rstar"], values["a"])
    expect_tirr = min(max(round(teq * 2.0 ** 0.5 / 20.0) * 20.0, 800.0),
                      2500.0)
    assert cp["Tirr"] == expect_tirr
