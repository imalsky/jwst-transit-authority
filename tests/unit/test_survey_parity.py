"""The planner and the sulfur survey describe a shared planet with the same numbers and the same UV file.

The survey lives in vulcan-retrieval, which this package never imports: its files are read as text from the sibling
checkout, and the test says so when that checkout is absent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from jwst_tool import planets, run_params

SURVEY = Path(__file__).resolve().parents[3] / "vulcan-retrieval"
# planner registry key -> (survey spec, survey vulcan3 config)
SHARED = {
    "wasp39b": ("runs/survey/wasp39b.yaml", "data/survey/w39b/W39b.yaml"),
    "hd189733b": ("runs/survey/hd189733b.yaml", "data/survey/hd189b/HD189b.yaml"),
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
    # the model's own values: canonical_params keeps enough digits for every registry value
    cp = run_params.canonical_params({"planet": planet, "science_mode": "emission"})
    assert cp["rp_rjup"] * planets.R_JUP_CM == pytest.approx(spec["rp_cm"], rel=REL)
    assert cp["rstar_rsun"] * planets.R_SUN_CM == pytest.approx(spec["rstar_cm"], rel=REL)
    assert cp["gs_cgs"] == pytest.approx(spec["tp_gravity_cgs"], rel=REL)
    assert cp["rstar_rsun"] == pytest.approx(cfg["r_star"], rel=REL)
    assert cp["orbit_au"] == pytest.approx(cfg["orbit_radius"], rel=REL)
    # a relative path is vulcan3's: the working directory first, then its packaged data (the survey runs from its root)
    resolve_data_path = pytest.importorskip("vulcan3.paths").resolve_data_path
    ours = resolve_data_path(planets.sflux_file(reg["sflux"]))
    theirs = SURVEY / cfg["sflux_file"]
    theirs = theirs if theirs.exists() else resolve_data_path(cfg["sflux_file"])
    assert ours.name == theirs.name
    assert hashlib.sha256(ours.read_bytes()).hexdigest() == hashlib.sha256(theirs.read_bytes()).hexdigest()
