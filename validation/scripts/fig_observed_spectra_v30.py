"""The tool's default models against published spectra: WASP-39 b NIRSpec PRISM
(FIREFLy reduction, Rustamkulov et al. 2023, Zenodo 6959427), WASP-39 b NIRSpec
G395H (Carter & May 2024, R = 100), HD 189733 b MIRI LRS eclipse (Inglis et al.
2024, DOI 10.3847/2041-8213/ad725e, Eureka!, both eclipses combined by inverse
variance). The model is blurred to the instrument line-spread function (refdata
R(lambda) / the tool's fitted width) and integrated exactly over every published
bin. Transmission carries one fitted depth offset (the free reference radius;
the estimator without it recovers the ERS team's own 1.3 for their best fit on
the FIREFLy reduction); emission is the absolute prediction.
contrast = std(model) / std(data). Inputs: validation/data/
atmos_wasp39b_transmission.npz and atmos_hd189733b_emission.npz
(inputs/atmos_cases.py), ers2023/FIREFly_final.txt, cm24_wasp39b/G395H_*.csv,
hd189733b_miri_eclipse_inglis2024.npz (inputs/extract_inglis2024.py), and the
Pandeia refdata dispersion curves.

    JAX_PLATFORM_NAME=cpu python validation/scripts/fig_observed_spectra_v30.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # validation/scripts
sys.path.insert(0, str(HERE.parent))              # figstyle
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import CYC, INK, MS, panels, legend, save  # noqa: E402
fs.use()
DATA = fs.DATA

from astropy.io import fits                       # noqa: E402
from jwst_tool import binning, instruments        # noqa: E402

PANDEIA = Path(instruments.PANDEIA_REFDATA) / "jwst"
DISP = {"nirspec_prism": "nirspec/dispersion/jwst_nirspec_prism_disp_20160902193401.fits",
        "nirspec_g395h": "nirspec/dispersion/jwst_nirspec_g395h_disp_20240607131021.fits",
        "miri_lrs": "miri/dispersion/jwst_miri_p750l_disp_20170404135013.fits"}


def r_curve(key):
    d = fits.open(PANDEIA / DISP[key])[1].data
    wl, r = np.asarray(d["WAVELENGTH"], float), np.asarray(d["R"], float)
    o = np.argsort(wl)
    return wl[o], instruments.lsf_r(key, wl[o], r[o])


def blur(wl_m, y_m, key, lo, hi):
    o = np.argsort(wl_m)
    wl_m, y_m = wl_m[o], y_m[o]
    wl_r, r = r_curve(key)
    return wl_m, binning.smooth_to_native_r(wl_m, y_m, wl_r, r, float(lo.min()), float(hi.max()))


def bin_means(wl_m, y, lo, hi):
    icum = binning._pl_cumint(wl_m, y)
    return (binning._pl_antideriv(hi, wl_m, y, icum) - binning._pl_antideriv(lo, wl_m, y, icum)) / (hi - lo)


def prism():
    a = np.genfromtxt(DATA / "ers2023" / "FIREFly_final.txt")
    cen, w, d, s = a.T
    keep = cen - w / 2 >= 1.0                       # the model band starts at 1 um
    return (cen - w / 2)[keep], (cen + w / 2)[keep], d[keep] * 1e6, s[keep] * 1e6


def g395h():
    rows = [np.genfromtxt(DATA / "cm24_wasp39b" / f, delimiter=",", skip_header=1)[:, 1:7]
            for f in ("G395H_NRS1_R100.csv", "G395H_NRS2_R100.csv")]
    a = np.vstack(rows)
    wl, lo, hi, k, e1, e2 = a.T
    return lo, hi, k ** 2 * 1e6, 2 * k * 0.5 * (e1 + e2) * 1e6


def hd189():
    z = np.load(DATA / "hd189733b_miri_eclipse_inglis2024.npz")
    return z["wl_lo_um"], z["wl_hi_um"], z["depth_ppm"], z["sigma_ppm"]


tr = np.load(DATA / "atmos_wasp39b_transmission.npz")
em = np.load(DATA / "atmos_hd189733b_emission.npz")
PANELS = [
    ("nirspec_prism", prism(), tr, True, "WASP-39 b PRISM, Rustamkulov et al. 2023", "transit depth (ppm)"),
    ("nirspec_g395h", g395h(), tr, True, "WASP-39 b G395H, Carter & May 2024", "transit depth (ppm)"),
    ("miri_lrs", hd189(), em, False, "HD 189733 b MIRI LRS, Inglis et al. 2024", "eclipse depth (ppm)"),
]

fig, axes = panels(1, 3)
for ax, (key, (lo, hi, d, s), mod, fit_offset, dlabel, ylabel) in zip(axes, PANELS):
    wl_m, y_b = blur(mod["wl_um"], mod["depth_ppm"], key, lo, hi)
    m = bin_means(wl_m, y_b, lo, hi)
    w = 1 / s ** 2
    off = np.sum(w * (d - m)) / np.sum(w) if fit_offset else 0.0
    chi2 = np.mean(((d - m - off) / s) ** 2)
    print(f"{dlabel:45s} N={d.size:3d}  chi2/N {chi2:6.2f}  contrast {np.std(m) / np.std(d):.3f}  "
          f"offset {off:+.0f} ppm" + ("" if fit_offset else
          f"  (with a fitted offset {np.sum(w * (d - m)) / np.sum(w):+.0f} ppm: chi2/N "
          f"{np.mean(((d - m - np.sum(w * (d - m)) / np.sum(w)) / s) ** 2):.2f})"))
    cen = 0.5 * (lo + hi)
    ax.errorbar(cen, d, yerr=s, fmt="o", color=INK, ms=MS, elinewidth=0.8, capsize=0, ls="", label=dlabel)
    inside = (wl_m >= lo.min()) & (wl_m <= hi.max())
    ax.plot(wl_m[inside], y_b[inside] + off, color=CYC[0], lw=1.2,
            label="this work, fitted offset" if fit_offset else "this work")
    ax.set_xlim(lo.min(), hi.max())
    ax.set_xlabel(r"wavelength ($\mu$m)"); ax.set_ylabel(ylabel)
    legend(ax)
save(fig, "observed_spectra_v30.png")
