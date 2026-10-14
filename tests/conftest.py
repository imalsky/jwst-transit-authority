import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

# run from a checkout without requiring the editable install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The fast suite writes refusals.log, uploads/ and the datacheck report under
# the output root, so give it a throwaway one (set before any jwst_tool
# import: instruments resolves the root at import). The slow tier keeps the
# real caches it reuses.
if ("JWST_TOOL_OUTPUT_DIR" not in os.environ
        and os.environ.get("JWST_TOOL_RUN_SLOW") != "1"):
    os.environ["JWST_TOOL_OUTPUT_DIR"] = tempfile.mkdtemp(prefix="jwst-tool-tests-")
    atexit.register(shutil.rmtree, os.environ["JWST_TOOL_OUTPUT_DIR"], True)

# Import order is load-bearing (vulcan_forward.vulcan_chem freezes the
# VULCAN_JAX_* env vars and jax x64 before vulcan_jax loads), and a unit test
# may reach a helper that imports vulcan_jax on its own -- which then freezes
# the WRONG network for every engine test after it in the process. Take the
# order here, once. Guarded: the light suite runs with no engine installed.
try:
    import vulcan_forward.vulcan_chem  # noqa: F401
except ImportError:
    pass
