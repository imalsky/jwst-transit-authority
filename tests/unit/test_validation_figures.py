"""Every committed figure was made by the committed script: the PNG embeds its
generator (figstyle.save writes it into a tEXt chunk) and that code equals the
script of the same name. Compared as parsed code with docstrings stripped, so
a comment or docstring edit does not demand a new PNG (each regeneration adds
the file to git history). Covers validation/figures and validation/parity/figs."""
import ast
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


def _code(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


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
    assert _code(meta["Generator-Source"]) == _code(script.read_text()), \
        f"{png.name} was not made by the committed {script.name}; regenerate it"
