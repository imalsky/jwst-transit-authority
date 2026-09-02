"""Regenerate every committed validation figure. fig_readme_example.py is left
out: it runs the whole app, so run that one directly.

    python validation/scripts/make_figures.py            # all fig_*.py in this directory
    python validation/scripts/make_figures.py rt_verification_six_atmospheres   # one, by PNG stem"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
names = sys.argv[1:] or sorted(p.stem[4:] for p in HERE.glob("fig_*.py") if p.stem != "fig_readme_example")
for name in names:
    subprocess.run([sys.executable, str(HERE / f"fig_{name}.py")], check=True)
