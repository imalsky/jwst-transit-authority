"""The TOI-7169 b +h dlnCO stencil point certifies via the photolysis-cadence escalation.

Why, cost and tolerances: docs/validation.md, section `test_toi7169_dlnco`.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("JWST_TOOL_RUN_SLOW") != "1", reason="two cold chemistry solves (~10 min): set JWST_TOOL_RUN_SLOW=1"
)

if importlib.util.find_spec("exojax") is None or importlib.util.find_spec("vulcan3") is None:
    pytest.skip("RT stack not installed", allow_module_level=True)

CASE = Path(__file__).parent / "data" / "toi7169_co_case.json"
CHILD = Path(__file__).parent / "toi7169_child.py"


def test_plus_h_dlnco_point_certifies_after_escalation():
    env = {k: v for k, v in os.environ.items() if not k.startswith("VULCAN3_")}
    res = subprocess.run(
        [sys.executable, str(CHILD), str(CASE)], env=env, capture_output=True, text=True, timeout=3 * 3600
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
