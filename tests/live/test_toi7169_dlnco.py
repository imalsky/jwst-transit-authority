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


_CHILD = r"""
import json, sys
from pathlib import Path
import numpy as np
from vulcan_forward import vulcan_chem  # noqa: F401  (env + x64 before exojax)
from jwst_tool import forward

cp = json.loads(Path(sys.argv[1]).read_text())["canonical_params"]
A = chem_build.assemble_chem(cp, lambda *a: None)
co_plus_h = cp["co_ratio"] * float(np.exp(jacobian.FD_STEPS["dlnCO"]))
abun = A.abundance_overrides(cp["met_x_solar"], co_plus_h)
stage = "TOI-7169 b dlnCO +h"

y, chem, _cert, escalated = certify.certified_solve(
    A.build_chem(abun, tag="toi7169 +h"), A.theta, stage,
    rebuild=lambda: A.build_chem(abun, tag="toi7169 +h cadence 1",
                                 photo_frq=1))

certify.check_elements(np.asarray(y), chem, stage)   # raises otherwise
assert escalated, (
    "this column certified at the configured cadence: the stall this "
    "escalation exists for is gone, so re-measure before trusting the "
    "retry path")
"""


def test_plus_h_dlnco_point_certifies_after_escalation():
    env = {k: v for k, v in os.environ.items() if not k.startswith("VULCAN3_")}
    res = subprocess.run(
        [sys.executable, "-c", _CHILD, str(CASE)], env=env, capture_output=True, text=True, timeout=3 * 3600
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
