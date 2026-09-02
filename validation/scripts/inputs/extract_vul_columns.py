"""P and volume mixing ratios of six species from the two VULCAN outputs behind
the chemistry validation figure (VULCAN 2.0 vs VULCAN 3.0 on identical inputs).

    python validation/scripts/inputs/extract_vul_columns.py <master.vul> <jax.vul>

Sources: jax_paper/data/W39b_master_paper.vul and W39b_jax_paper.vul (28 MB of
solver pickles, not committed). Writes validation/data/chemistry_w39b_vulcan2_vs_vulcan3.npz."""
import pickle
import sys
from pathlib import Path

import numpy as np

SPECIES = ["H2O", "CO", "CO2", "CH4", "SO2", "H2S"]
OUT = Path(__file__).resolve().parents[2] / "data" / "chemistry_w39b_vulcan2_vs_vulcan3.npz"


def columns(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    sp = list(d["variable"]["species"])
    ymix = np.asarray(d["variable"]["ymix"])
    return np.asarray(d["atm"]["pco"]) / 1e6, {s: ymix[:, sp.index(s)] for s in SPECIES}


p2, y2 = columns(sys.argv[1])
p3, y3 = columns(sys.argv[2])
assert np.array_equal(p2, p3)
np.savez(OUT, p_bar=p2, species=np.array(SPECIES),
         **{f"vulcan2_{s}": y2[s] for s in SPECIES}, **{f"vulcan3_{s}": y3[s] for s in SPECIES},
         source=np.array([Path(sys.argv[1]).name, Path(sys.argv[2]).name]),
         generator=np.array(Path(__file__).read_text()))
print("wrote", OUT)
