"""The +h dlnCO stencil point of a 10x-solar, C/O 0.55, photo-on column
(the TOI-7169 b CO case in data/toi7169_co_case.json) certifies via the
photolysis-cadence escalation.

At the config's refresh every 5th accepted step this point does not certify:
dt can grow up to 2^5 between refreshes, and at the recorded event current
water shielding gave Ros2 error 0.43 against rtol 0.2 while the previous water
profile for radiation alone gave 0.06, so the column rejected every refresh
step (vulcan-forward notes S1.6). `forward.certified_solve` re-solves such a
column once with photolysis refreshed every accepted step.

Cost: two cold solves, ~10 min. Gated like the e2e: JWST_TOOL_RUN_SLOW=1.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("JWST_TOOL_RUN_SLOW") != "1",
    reason="two cold chemistry solves (~10 min): set JWST_TOOL_RUN_SLOW=1")

if importlib.util.find_spec("exojax") is None or \
        importlib.util.find_spec("vulcan_jax") is None:
    pytest.skip("RT stack not installed", allow_module_level=True)

CASE = Path(__file__).parent / "data" / "toi7169_co_case.json"


def test_plus_h_dlnco_point_certifies_after_escalation():
    from vulcan_forward import vulcan_chem  # noqa: F401  (env + x64 before exojax)
    from jwst_tool import forward

    cp = json.loads(CASE.read_text())["canonical_params"]
    A = forward._assemble_chem(cp, lambda *a: None)
    co_plus_h = cp["co_ratio"] * float(np.exp(forward.FD_STEPS["dlnCO"]))
    abun = A.abundance_overrides(cp["met_x_solar"], co_plus_h)
    stage = "TOI-7169 b dlnCO +h"

    y, chem, _cert, escalated = forward.certified_solve(
        A.build_chem(abun, tag="toi7169 +h"), A.theta, stage,
        rebuild=lambda: A.build_chem(abun, tag="toi7169 +h cadence 1",
                                     photo_frq=1))

    forward.check_elements(np.asarray(y), chem, stage)   # raises otherwise
    assert escalated, (
        "this column certified at the configured cadence: the stall this "
        "escalation exists for is gone, so re-measure before trusting the "
        "retry path (vulcan-forward todo 4)")
