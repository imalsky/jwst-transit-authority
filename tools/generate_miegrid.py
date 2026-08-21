"""Generate ExoJAX Mie lookup grids (miegrid_lognorm_<condensate>.mg.npz).

One-time per condensate; the forward model only LOADS grids (pure
numpy/jax, differentiable) and refuses loudly when one is missing, pointing
here. Generation needs PyMieScatt, whose 1.8.x release is broken under
scipy >= 1.14 (scipy.integrate.trapz was removed) -- shimmed below BEFORE
the import, the accepted workaround until PyMieScatt ships a fix.

Usage:
    python tools/generate_miegrid.py MgSiO3 [Fe Al2O3 ...]

Writes into <JWST_TOOL_DATA_DIR or repo data>/exojax_mie/ (also the home of
the virga refractive-index archive, downloaded automatically from Zenodo on
first use, ~4 MB). Supported condensates (refractive indices AND substance
density in exojax 2.2.3): NH3, H2O, MgSiO3, Mg2SiO4, Fe, Al2O3, TiO2.
Default grid (rg 1e-7..1e-3 cm x 40, sigmag 1.0001..4 x 10) takes on the
order of an hour per condensate on one CPU core.
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.integrate as _si

if not hasattr(_si, "trapz"):          # scipy >= 1.14 removed trapz
    _si.trapz = getattr(np, "trapezoid", None) or np.trapz

from jwst_tool import instruments as ins  # noqa: E402  (path roots)

SUPPORTED = ("NH3", "H2O", "MgSiO3", "Mg2SiO4", "Fe", "Al2O3", "TiO2")


def main(argv):
    conds = argv or ["MgSiO3"]
    bad = [c for c in conds if c not in SUPPORTED]
    if bad:
        raise SystemExit(f"unsupported condensate(s) {bad}: exojax 2.2.3 has "
                         f"refractive indices + substance density only for "
                         f"{list(SUPPORTED)}")
    import PyMieScatt  # noqa: F401  -- fail here, loudly, if the shim wasn't enough
    from exojax.database.pardb import PdbCloud

    mie_dir = Path(ins.DATA_DIR) / "exojax_mie"
    mie_dir.mkdir(parents=True, exist_ok=True)
    for cond in conds:
        t0 = time.time()
        pdb = PdbCloud(cond, path=str(mie_dir))
        if pdb.miegrid_path.exists():
            print(f"[miegrid] {cond}: already present at {pdb.miegrid_path}")
            continue
        print(f"[miegrid] {cond}: generating (default rg/sigmag grid; "
              "on the order of an hour) ...", flush=True)
        pdb.generate_miegrid()
        print(f"[miegrid] {cond}: done in {(time.time()-t0)/60.0:.1f} min "
              f"-> {pdb.miegrid_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
