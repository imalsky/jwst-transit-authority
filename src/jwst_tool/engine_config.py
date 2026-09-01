"""This tool's view of the shared forward engine's configuration.

Holds what the planner needs from the ``vulcan-forward`` engine:

- physics constants re-exported from ``vulcan_forward.constants`` (one source
  of truth, so the two applications cannot drift apart),
- the planner's OWN base radiative-transfer profile, and
- the engine's data locations, resolved lazily so importing this module never
  touches the filesystem.

The attribute surface is a contract shared with the retrieval framework's
config module: every ``config.MOLECULES`` / ``cfg.CIA_H2HE_FILE`` call site
must keep working.
"""
from __future__ import annotations

from pathlib import Path

from vulcan_forward import constants as _fwd
from vulcan_forward import paths as _fwd_paths

# --- shared physics: re-exported, never redefined --------------------------
MOLECULES = _fwd.MOLECULES
W39B_CFG_NAME = _fwd.DEFAULT_CFG_NAME

# --- the planner's own base RT profile -------------------------------------
# run_model overrides the resolution knobs from the canonical parameter set;
# the band edges come from the engine, which owns the supported 1-15 um window
# (short edge = the H2-H2 CIA table). Bit-identical to the pre-extraction
# profile, pinned by tests/unit/test_rt_profile_golden.py.
WIDE = {
    "use_photo": True,
    "nz": 150,
    "yconv_cri": 1.0e-3,
    "molecules": ["H2O", "CO2", "CO", "CH4", "SO2"],
    "nu_min": _fwd.WIDE_BAND_NU_MIN,    # 667 cm^-1 = 15 um
    "nu_max": _fwd.WIDE_BAND_NU_MAX,    # 10000 cm^-1 = 1 um (H2-H2 CIA edge)
    "art_nlayer": 60,
}

# --- data locations --------------------------------------------------------
# Resolved on ACCESS, not at import: the tool must stay importable on a
# machine with no data installed ($VULCAN_FORWARD_DATA contract). Touching
# one without a configured root raises RuntimeError naming the remedy.
_LAZY = {
    "DATA_DIR": lambda: _fwd_paths.data_root(),
    "CIA_H2H2_FILE": lambda: _fwd_paths.cia_h2h2_file(),
    "CIA_H2HE_FILE": lambda: _fwd_paths.cia_h2he_file(),
    # ExoMolOP k-table tree (the gas opacity every run reads). No existence
    # check in the engine accessor, so datacheck can report per-molecule
    # MISSING items even when the whole tree is absent.
    "EXOMOLOP_DIR": lambda: _fwd_paths.exomolop_dir(),
}


def __getattr__(name: str) -> Path:
    """PEP 562 lazy data paths (see the _LAZY note above)."""
    if name in _LAZY:
        return _LAZY[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
