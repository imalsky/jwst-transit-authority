import sys
from pathlib import Path

# run from a checkout without requiring the editable install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import order is load-bearing (vulcan_forward.vulcan_chem freezes the
# VULCAN_JAX_* env vars and jax x64 before vulcan_jax loads), and a unit test
# may reach a helper that imports vulcan_jax on its own -- which then freezes
# the WRONG network for every engine test after it in the process. Take the
# order here, once. Guarded: the light suite runs with no engine installed.
try:
    import vulcan_forward.vulcan_chem  # noqa: F401
except ImportError:
    pass
