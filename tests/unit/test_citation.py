"""CITATION.cff stays in lock-step with the package version (it is a manual
second copy of _version.py; regex parse, no yaml dependency in light CI)."""
from __future__ import annotations

import re
from pathlib import Path

import jwst_tool

CFF = Path(__file__).resolve().parents[2] / "CITATION.cff"


def test_citation_file_version_matches_package():
    text = CFF.read_text(encoding="utf-8")
    m = re.search(r"^version:\s*\"?([0-9][0-9a-z.]*)\"?\s*$", text,
                  re.MULTILINE)
    assert m, "CITATION.cff has no version: line"
    assert m.group(1) == jwst_tool.__version__, (
        "CITATION.cff version out of step with jwst_tool._version -- bump "
        "both together on every release")


def test_citation_file_names_the_license_and_repo():
    text = CFF.read_text(encoding="utf-8")
    assert "GPL-3.0-only" in text
    assert "github.com/imalsky/vulcan-jwst-tool" in text
