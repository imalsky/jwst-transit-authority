"""This tool's view of the shared forward engine's configuration.

Until 2026-07-29 the planner imported ``retrieval_framework.forward.config``:
the retrieval framework's own configuration module. That was the root of the
dependency on a sibling APPLICATION -- the planner read another app's demo run
profile, its molecule table, and its repo-relative data paths, and could not
even be imported without a retrieval checkout whose observed-spectra directory
existed. The engine is now the ``vulcan-forward`` distribution, so this module
holds what the planner actually needs:

- physics constants re-exported from ``vulcan_forward.constants`` (one source of
  truth, so the two applications cannot drift apart on molecule masses or
  opacity sources),
- the planner's OWN base radiative-transfer profile, and
- the engine's data locations, resolved lazily through the engine's env
  contract so importing this module never touches the filesystem.

The attribute surface is deliberately unchanged from the module it replaced, so
every ``config.MOLECULES`` / ``cfg.CIA_H2HE_FILE`` call site keeps working.
"""
from __future__ import annotations

from pathlib import Path

from vulcan_forward import constants as _fwd
from vulcan_forward import paths as _fwd_paths

# --- shared physics: re-exported, never redefined --------------------------
MOLECULES = _fwd.MOLECULES
W39B_CFG_NAME = _fwd.DEFAULT_CFG_NAME
BULK_H2_VULCAN = _fwd.BULK_H2_VULCAN

# --- the planner's own base RT profile -------------------------------------
# Values are the ones this tool has always run with (they were inherited from
# the retrieval's "WIDE" demo profile, which is exactly the coupling being
# removed -- an observation planner should not be pinned to another
# application's demo settings). run_model overrides the resolution knobs from
# the canonical parameter set; the band edges come from the engine, which owns
# the supported window: 1-15 um, whose short edge is set by the H2-H2 CIA
# table. Bit-identical to the pre-extraction profile, pinned by
# tests/unit/test_rt_profile_golden.py.
WIDE = {
    "use_photo": True,
    "nz": 150,
    "yconv_cri": 1.0e-3,
    "molecules": ["H2O", "CO2", "CO", "CH4", "SO2"],
    "nu_min": _fwd.WIDE_BAND_NU_MIN,    # 667 cm^-1 = 15 um
    "nu_max": _fwd.WIDE_BAND_NU_MAX,    # 10000 cm^-1 = 1 um (H2-H2 CIA edge)
    "nu_pts": 8000,                     # native R ~ 2950; binned to display_R
    "art_nlayer": 60,
    "display_R": 100,
}

# --- data locations --------------------------------------------------------
# Resolved on ACCESS, not at import: the engine's contract is an explicit data
# root ($VULCAN_FORWARD_DATA), and a tool that only wants to read its cache or
# render a page must still be importable on a machine with no data installed.
# Touching one of these without a configured root raises RuntimeError naming
# the remedy, which is exactly what the data-status report displays.
_LAZY = {
    "DATA_DIR": lambda: _fwd_paths.data_root(),
    "DEMO_DATABASE": lambda: _fwd_paths.linelist_dir(),
    "OPACITY_CACHE": lambda: _fwd_paths.opacity_cache_dir(),
    "CO_CACHED_DIR": lambda: (_fwd_paths.opacity_cache_dir()
                              / MOLECULES["CO"]["db"]),
    "CIA_H2H2_FILE": lambda: _fwd_paths.cia_h2h2_file(),
    "CIA_H2HE_FILE": lambda: _fwd_paths.cia_h2he_file(),
}


def __getattr__(name: str) -> Path:
    """PEP 562 lazy data paths (see the _LAZY note above)."""
    if name in _LAZY:
        return _LAZY[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY))
