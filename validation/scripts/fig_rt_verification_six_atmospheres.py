"""This work vs petitRADTRANS 3.4.0 on JWST Transit Authority's own converged
atmospheres: four planets in transmission, two in emission, identical
P/T/mmw/VMRs/geometry and k-table files, lines only, 3.03-5.17 um. Inputs:
validation/data/atmos_*.npz (inputs/atmos_cases.py) and prt_*.npz
(inputs/prt_reference.py). numpy + matplotlib only.

    python validation/scripts/fig_rt_verification_six_atmospheres.py
"""
import hashlib
from io import BytesIO
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # validation/scripts
sys.path.insert(0, str(HERE.parent))              # figstyle
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import CYC, INK, RED, panels, legend, save  # noqa: E402
from jwst_tool.binning import _pl_antideriv, _pl_cumint  # noqa: E402
fs.use()
DATA = fs.DATA

CASES = {"transmission": ["wasp39b", "hd189733b", "wasp107b", "hd209458b"],
         "emission": ["wasp39b", "hd189733b"]}
KEY = {"transmission": "depth_cmp_ppm", "emission": "flux_cmp"}
PKEY = {"transmission": "depth_ppm", "emission": "flux_per_cm1"}

def load_pair(planet, mode):
    atmosphere = DATA / f"atmos_{planet}_{mode}.npz"
    reference = DATA / f"prt_{planet}_{mode}.npz"
    source = atmosphere.read_bytes()
    with np.load(reference) as b:
        if ("source_atmosphere_sha256" not in b
                or b["source_atmosphere_sha256"].shape != ()
                or str(b["source_atmosphere_sha256"]) != hashlib.sha256(source).hexdigest()):
            raise ValueError(f"{reference.name}: stale or mismatched atmosphere; regenerate with inputs/prt_reference.py")
        if "prt_version" not in b or b["prt_version"].shape != () or not str(b["prt_version"]).strip():
            raise ValueError(f"{reference.name}: missing pRT version provenance; regenerate with inputs/prt_reference.py")
        ref = dict(b)
    with np.load(BytesIO(source)) as a:
        return dict(a), ref


# Equal log-wavelength bins, R=100, complete bins within 3.06-5.12 um.
# Exact piecewise-linear wavelength averages, with identical edges for both
# spectra; no instrument LSF or stellar-count weighting in this RT-only test.
EDGES = 3.06 * np.exp(np.arange(int(100 * np.log(5.12 / 3.06)) + 1) / 100)


def binned(wl, values):
    if wl[0] > EDGES[0] or wl[-1] < EDGES[-1]:
        raise ValueError("spectrum does not cover the comparison bins")
    return np.diff(_pl_antideriv(EDGES, wl, values, _pl_cumint(wl, values))) / np.diff(EDGES)


# Validate every pair before constructing or saving the comparison.
PAIRS = {(p, m): load_pair(p, m) for m, planets in CASES.items() for p in planets}
fig, axes = panels(1, 2)
for ax, (mode, plist) in zip(axes, CASES.items()):
    for planet, col in zip(plist, CYC if mode == "transmission" else (INK, RED)):
        a, b = PAIRS[planet, mode]
        wl, ours = a["wl_cmp"], a[KEY[mode]]
        o = np.argsort(wl)
        wl, ours = wl[o], ours[o]
        m = (wl > 3.03 * 1.01) & (wl < 5.17 * 0.99)
        label = f"{a['label']}, $g$ = {float(a['gs_cgs']):.0f} cm s$^{{-2}}$"
        if mode == "transmission":
            diff = binned(wl, ours) - binned(b["wl_um"], b[PKEY[mode]])
            residual = diff - diff.mean()
            ax.plot(np.sqrt(EDGES[:-1] * EDGES[1:]), residual, color=col, lw=1.2, label=label)
            print(f"{mode:12s} {str(a['label']):12s} R=100 offset {diff.mean():.2f} ppm  "
                  f"residual std {diff.std():.2f} ppm  max |residual| {np.max(np.abs(residual)):.2f} ppm  "
                  f"pRT {b['prt_version']}")
        else:
            r = ours[m] / np.interp(wl[m], b["wl_um"], b[PKEY[mode]])
            ax.plot(wl[m], 100 * (r - 1), color=col, lw=1.2, label=label)
            print(f"{mode:12s} {str(a['label']):12s} mean ratio {r.mean():.4f}  rms {100 * np.std(r):.3f} %  "
                  f"max |dev| {100 * np.max(np.abs(r - 1)):.3f} %  pRT {b['prt_version']}")
    ax.set_xlabel(r"wavelength ($\mu$m)")
    ax.set_ylabel("transmission residual, offset removed (ppm)" if mode == "transmission"
                  else "emission: this work / petitRADTRANS $-$ 1 (%)")
    legend(ax)
save(fig, "rt_verification_six_atmospheres.png")
