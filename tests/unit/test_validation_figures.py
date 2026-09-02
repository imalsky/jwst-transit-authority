"""Every committed figure was made by the committed script: the PNG embeds its
generator (figstyle.save writes it into a tEXt chunk) and that text equals the
script of the same name. Covers validation/figures and validation/parity/figs."""
from pathlib import Path
import sys

import matplotlib
import pytest

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / "validation"
sys.path.insert(0, str(VAL))
import figstyle  # noqa: E402

FIGURE_DIRS = {VAL / "figures": VAL / "scripts",
               VAL / "parity" / "figs": VAL / "parity" / "scripts"}


def test_use_applies_the_serif_style():
    figstyle.use()
    assert matplotlib.rcParams["font.family"] == ["serif"]
    assert [c["color"] for c in matplotlib.rcParams["axes.prop_cycle"]] == figstyle.CYC


@pytest.mark.parametrize("png", sorted(p for d in FIGURE_DIRS for p in d.glob("*.png")),
                         ids=lambda p: p.name)
def test_every_committed_figure_embeds_its_committed_generator(png):
    meta = figstyle.embedded_source(png)
    script = FIGURE_DIRS[png.parent] / meta["Source"]
    assert script.exists(), f"{png.name}: generator {meta['Source']} is not committed"
    assert meta["Generator-Source"] == script.read_text(), \
        f"{png.name} was not made by the committed {script.name}; regenerate it"
