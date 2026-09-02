"""The README example: the app's summary figure for the default run (WASP-39 b,
SO2, one transit per mode), exported exactly as the app's PNG download does
(200 dpi, tight crop). Runs the app headlessly through Streamlit's AppTest and
needs the default model and noise caches under output/ (a first run fills them).

    python validation/scripts/fig_readme_example.py      # writes assets/w39b_so2_forecast.png"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jwst_tool import plotting, summary_figure  # noqa: E402

captured = []
_compose = summary_figure.compose_summary_figure


def _keep(*a, **k):
    fig = _compose(*a, **k)
    captured.append(fig)
    return fig


summary_figure.compose_summary_figure = _keep
at = AppTest.from_file(str(ROOT / "src" / "jwst_tool" / "app.py"), default_timeout=600)
at.run()
assert not at.exception, at.exception
(run,) = [b for b in at.button if b.label == "Run"]
run.click().run()
assert not at.exception, [e.value for e in at.exception]
assert captured, "the app did not render the summary figure"
buf = io.BytesIO()
with plotting.render_lock:
    captured[-1].savefig(buf, format="png", facecolor="white", bbox_inches="tight", dpi=200)
out = ROOT / "assets" / "w39b_so2_forecast.png"
out.write_bytes(buf.getvalue())
print("wrote", out)
