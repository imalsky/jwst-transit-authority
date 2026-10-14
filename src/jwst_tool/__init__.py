"""JWST Transit Authority: a JWST instrument-mode selector.

Chemistry (VULCAN-JAX kinetics or the PICASO engine) -> ExoJAX RT (the
shared vulcan-forward engine) -> Pandeia noise; modes are ranked by
conditional matched-template S/N and Fisher forecasts. Entry point: the
console script ``jwst-tool``.
"""
from jwst_tool._version import __version__  # noqa: F401
