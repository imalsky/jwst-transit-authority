"""Console entry point: ``jwst-tool`` launches the Streamlit GUI.

``jwst-tool``              launch the GUI (preflight checks first, loud)
``jwst-tool fetch``        download every missing dataset with a public URL,
                           then print the short manual list (two STScI Box
                           downloads + the Pandeia conda environment)
``jwst-tool data``         print the full data-availability report + remedies
``jwst-tool data --deep``  also probe the Pandeia env for its engine version
``jwst-tool archive-refresh``  refresh the shipped NASA Exoplanet Archive
                           snapshot (maintainer/release step; network)

The GUI launch is equivalent to ``streamlit run src/jwst_tool/app.py``, but
works from anywhere once the package is installed. Preflight checks catch the
two external requirements with actionable messages instead of a mid-run stack
trace: an importable ``vulcan_jax`` package and the Pandeia backend env
(``$JWST_TOOL_PANDEIA_PYTHON``).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _data_status(argv: list[str]) -> int:
    """Print the data-availability report (the ``jwst-tool data`` subcommand)."""
    deep = "--deep" in argv
    unknown = [a for a in argv if a not in ("--deep",)]
    if unknown:
        print(f"jwst-tool data: unknown argument(s) {unknown}; the only flag "
              "is --deep (probe the Pandeia env's engine version).",
              file=sys.stderr)
        return 2
    from jwst_tool import datacheck
    report = datacheck.full_report(deep=deep)
    print(datacheck.format_report(report))
    return 0 if datacheck.required_ok(report) else 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "data":
        return _data_status(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        from jwst_tool import fetch
        return fetch.run_fetch()
    if len(sys.argv) > 1 and sys.argv[1] == "archive-refresh":
        from jwst_tool import archive
        return archive.run_refresh(sys.argv[2:])

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("jwst-tool: streamlit is not installed in this environment.\n"
              "Install the GUI extra:  pip install 'vulcan-jwst-tool[gui]'  "
              "(or, in a checkout:  pip install -e '.[gui]' --no-deps "
              "plus pip install streamlit pandas)", file=sys.stderr)
        return 2

    import importlib.util
    if importlib.util.find_spec("vulcan_jax") is None:
        print("jwst-tool: the vulcan_jax package is not installed in this "
              "environment.\n"
              "Install it from TestPyPI "
              "(pip install -i https://test.pypi.org/simple/ vulcan-jax), or "
              "from a checkout:\n"
              "  pip install -e <PROJECT_ROOT>/VULCAN-JAX --no-deps\n"
              "Run `jwst-tool data` for the full data-availability report.",
              file=sys.stderr)
        return 2

    from jwst_tool import instruments as ins
    if not (ins.PANDEIA_PYTHON and Path(ins.PANDEIA_PYTHON).exists()):
        print(f"jwst-tool: Pandeia backend python not found at {ins.PANDEIA_PYTHON} "
              f"(backend '{ins.JWST_TOOL_BACKEND}': {ins.BACKEND_STATUS}).\n"
              "Point JWST_TOOL_PANDEIA_PYTHON at a python with the matching "
              "pandeia.engine (and JWST_TOOL_PANDEIA_REFDATA at the matching refdata). "
              "The interface will start, but noise calculations cannot run "
              "until this environment is available. "
              "Run `jwst-tool data` for the full data-availability report.",
              file=sys.stderr)

    app = Path(__file__).parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)] + sys.argv[1:]
    return subprocess.call(cmd, env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main())
