"""HD 189733 b MIRI LRS eclipse depths (Inglis et al. 2024, Eureka! reduction,
two eclipses combined by inverse variance), for the observed-spectra figure.

    python validation/scripts/inputs/extract_inglis2024.py <HD189733b_MIRI_results_final dir>

Source: the authors' results archive (DOI recorded in the npz as `doi`), files
Eureka_Reduction/eclipse{1,2}_eclipse_depths_eureka.nc. Writes
validation/data/hd189733b_miri_eclipse_inglis2024.npz with the bins in um and
depths in ppm."""
import sys
from pathlib import Path

import h5py
import numpy as np

# The archive ships no README; this is the `DOI` global attribute the two .nc
# files carry, beside reference "Quartz Clouds in Quintessential Hot Jupiter
# HD 189733 b (Inglis. 2024b, ApJL)".
DOI = "10.3847/2041-8213/ad725e"
OUT = Path(__file__).resolve().parents[2] / "data" / "hd189733b_miri_eclipse_inglis2024.npz"
root = Path(sys.argv[1]) / "Eureka_Reduction"
ds, ss = [], []
for i in (1, 2):
    with h5py.File(root / f"eclipse{i}_eclipse_depths_eureka.nc") as h:
        cen, hw = h["central_wavelength"][()], h["bin_half_width"][()]
        ds.append(h["eclipse_depth"][()] * 1e4)         # stored in percent
        ss.append(h["eclipse_depth_error"][()] * 1e4)
w = 1 / np.asarray(ss) ** 2
np.savez(OUT, wl_lo_um=cen - hw, wl_hi_um=cen + hw,
         depth_ppm=(np.asarray(ds) * w).sum(0) / w.sum(0), sigma_ppm=1 / np.sqrt(w.sum(0)),
         doi=np.array(DOI), generator=np.array(Path(__file__).read_text()))
print("wrote", OUT)
