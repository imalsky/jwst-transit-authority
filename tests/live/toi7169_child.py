"""Solve the TOI-7169 b +h dlnCO stencil point in its own process.

The case is on sncho and a process freezes its network at import, so the suite's sncho2025 process cannot run it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from vulcan_forward import vulcan_chem  # noqa: F401  (env + x64 before exojax)

from jwst_tool import certify, chem_build, model_defs

STAGE = "TOI-7169 b dlnCO +h"


def main(case_path: str) -> None:
    """Raise unless the point certifies, and certifies through the cadence escalation."""
    cp = json.loads(Path(case_path).read_text())["canonical_params"]
    assembly = chem_build.assemble_chem(cp, lambda *a: None)
    co_plus_h = cp["co_ratio"] * float(np.exp(model_defs.FD_STEPS["dlnCO"]))
    abun = assembly.abundance_overrides(cp["met_x_solar"], co_plus_h)
    y, chem, _cert, escalated = certify.certified_solve(
        assembly.build_chem(abun, tag="toi7169 +h"),
        assembly.theta,
        STAGE,
        rebuild=lambda: assembly.build_chem(abun, tag="toi7169 +h cadence 1", photo_frq=1),
    )
    certify.check_elements(np.asarray(y), chem, STAGE)
    if not escalated:
        raise RuntimeError(
            "this column certified at the configured cadence: the stall this escalation exists for is gone, "
            "so re-measure before trusting the retry path"
        )


if __name__ == "__main__":
    main(sys.argv[1])
