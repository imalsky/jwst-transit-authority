"""Hold the tool's view of the shared forward engine's configuration.

Holds what the planner needs from the ``vulcan-forward`` engine:

- physics constants re-exported from ``vulcan_forward.constants`` (one source
  of truth, so the two applications cannot drift apart),
- the planner's OWN base radiative-transfer profile, and
- the engine's data locations, resolved lazily so importing this module never
  touches the filesystem.

The attribute surface is a contract: ``chem_build``, ``picaso_engine``, ``spectrum``,
``datacheck``, ``fetch`` and ``gui_results`` read these names, so keep them working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vulcan_forward import constants as _fwd, paths as _fwd_paths

if TYPE_CHECKING:
    from pathlib import Path

# Shared physics: re-exported, never redefined
MOLECULES = _fwd.MOLECULES

# The vulcan3 config every VULCAN run builds on: the tool overrides its planet, star, orbit, UV file and composition
# per run, and model_defs.CFG_ABUNDANCES and CHEM_P_SPAN_DYN are checked against it (chem_build._check_cfg_constants).
W39B_CFG_NAME = "W39b"

# The planner's own base RT profile
# Only the band edges: run_model sets every other key from the canonical
# parameter set. The engine owns the supported 1-15 um window (short edge =
# the H2-H2 CIA table).
WIDE = {
    "nu_min": _fwd.WIDE_BAND_NU_MIN,  # 667 cm^-1 = 15 um
    "nu_max": _fwd.WIDE_BAND_NU_MAX,  # 10000 cm^-1 = 1 um (H2-H2 CIA edge)
}

# Data locations
# Resolved on ACCESS, not at import: the tool must stay importable on a
# machine with no data installed ($VULCAN_FORWARD_DATA contract). Touching
# one without a configured root raises RuntimeError naming the remedy.
_LAZY = {
    "DATA_DIR": _fwd_paths.data_root,
    "CIA_H2H2_FILE": _fwd_paths.cia_h2h2_file,
    "CIA_H2HE_FILE": _fwd_paths.cia_h2he_file,
    # ExoMolOP k-table tree (the gas opacity every run reads). No existence
    # check in the engine accessor, so datacheck can report per-molecule
    # MISSING items even when the whole tree is absent.
    "EXOMOLOP_DIR": _fwd_paths.exomolop_dir,
}


def __getattr__(name: str) -> Path:
    """PEP 562 lazy data paths (see the _LAZY note above)."""
    if name in _LAZY:
        return _LAZY[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
