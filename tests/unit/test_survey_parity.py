"""The planner and the sulfur survey describe a shared planet with the same numbers and the same UV file.

The survey lives in vulcan-retrieval, which this package never imports: its files are read as text from the sibling
checkout, and the test says so when that checkout is absent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from jwst_tool import planets

SURVEY = Path(__file__).resolve().parents[3] / "vulcan-retrieval"
# planner registry key -> (survey spec, survey vulcan3 config). WASP-39 b and HD 189733 b are not aligned yet.
SHARED = {
    "wasp107b": ("runs/survey/wasp107b.yaml", "data/survey/w107b/W107b.yaml"),
    "hd209458b": ("runs/survey/hd209458b.yaml", "data/survey/hd209b/HD209b.yaml"),
}
REL = 1.0e-4  # the registry carries five significant figures


@pytest.mark.parametrize("planet", sorted(SHARED))
def test_registry_matches_the_survey(planet):
    if not SURVEY.is_dir():
        pytest.skip(f"no vulcan-retrieval checkout at {SURVEY}: the survey's values cannot be compared")
    spec_file, cfg_file = SHARED[planet]
    spec = yaml.safe_load((SURVEY / spec_file).read_text())["config"]
    cfg = yaml.safe_load((SURVEY / cfg_file).read_text())
    reg = planets.PLANETS[planet]
    assert reg["rp_rjup"] * planets.R_JUP_CM == pytest.approx(spec["rp_cm"], rel=REL)
    assert reg["rstar_rsun"] * planets.R_SUN_CM == pytest.approx(spec["rstar_cm"], rel=REL)
    assert reg["gs_cgs"] == pytest.approx(spec["tp_gravity_cgs"], rel=REL)
    assert reg["rstar_rsun"] == pytest.approx(cfg["r_star"], rel=REL)
    assert reg["orbit_au"] == pytest.approx(cfg["orbit_radius"], rel=REL)
    ours = Path(planets.sflux_file(reg["sflux"]))
    theirs = SURVEY / cfg["sflux_file"]
    assert ours.name == theirs.name
    assert hashlib.sha256(ours.read_bytes()).hexdigest() == hashlib.sha256(theirs.read_bytes()).hexdigest()
