"""The TOI-7169 b child reaches its solve: run in a fresh interpreter, nothing else would catch a missing import."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

CHILD = Path(__file__).resolve().parents[1] / "live" / "toi7169_child.py"

pytest.importorskip("vulcan_forward")


def _load():
    spec = importlib.util.spec_from_file_location("toi7169_child", CHILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("escalated", [True, False], ids=["escalated", "certified-at-cadence"])
def test_the_child_solves_the_plus_h_point_and_requires_the_escalation(tmp_path, monkeypatch, escalated):
    child = _load()
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"canonical_params": {"co_ratio": 0.5, "met_x_solar": 10.0}}))
    abun = []
    assembly = SimpleNamespace(
        abundance_overrides=lambda met, co: abun.append((met, co)) or {"co": co},
        build_chem=lambda ab, tag, photo_frq=None: SimpleNamespace(ab=ab, frq=photo_frq),
        theta=np.zeros(4),
    )
    gated = []
    monkeypatch.setattr(child.chem_build, "assemble_chem", lambda cp, log: assembly)
    monkeypatch.setattr(
        child.certify,
        "certified_solve",
        lambda chem, theta, stage, rebuild: (np.ones((2, 2)), rebuild() if escalated else chem, (), escalated),
    )
    monkeypatch.setattr(child.certify, "check_elements", lambda y, chem, stage: gated.append(chem.frq))
    if escalated:
        child.main(str(case))
        assert gated == [1], "the element gate must see the cadence-1 column"
    else:
        with pytest.raises(RuntimeError, match="re-measure"):
            child.main(str(case))
    np.testing.assert_allclose(abun, [(10.0, 0.5 * np.exp(child.model_defs.FD_STEPS["dlnCO"]))])
