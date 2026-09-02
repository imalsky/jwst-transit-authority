"""Correlated-k cross-validation: vulcan-forward vs ExoJAX vs exo_k on the same
ExoMolOP tables. H2O band-mean cross section at 1100 K, 1 mbar (native R = 1000
and binned to R = 100) and the worst per-molecule |dlog10 k| over the shared
(T, P) probes. The exo_k side is validation/data/exok_ckd_vals.npz
(inputs/exok_ckd_values.py, exo_k never enters this environment); the ExoJAX
side reads the same k-tables through the vendored upstream reader
exojax_master_provider_exomolop.py.

    JAX_PLATFORM_NAME=cpu python validation/scripts/fig_ckd_verification_vs_exojax_exok.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # validation/scripts
sys.path.insert(0, str(HERE.parent))              # figstyle
sys.path.insert(0, str(HERE))                     # the vendored provider
import numpy as np                                # noqa: E402
import figstyle as fs                             # noqa: E402
from figstyle import INK, RED, DASH_KW, DOT_KW, pastel, panels, lollipop, legend, save  # noqa: E402
fs.use()
DATA = fs.DATA

import jax                                        # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                           # noqa: E402

from jwst_tool import engine_config               # noqa: E402  (the tool's k-table root)
import exojax_master_provider_exomolop as prov    # noqa: E402
from vulcan_forward import ckd, exomolop          # noqa: E402
from exojax.opacity.ckd.core import interpolate_log_k_2d  # noqa: E402

EK = np.load(DATA / "exok_ckd_vals.npz")
KTABLES = engine_config.EXOMOLOP_DIR
FLOOR = exomolop.K_FLOOR
T_SHOW, P_SHOW = 1100.0, 1e-3
NB = 10
SUB = {m: "".join(c if not c.isdigit() else f"$_{c}$" for c in m) for m in EK["mols"]}


def binned(a):
    n = a.size - a.size % NB
    return a[:n].reshape(-1, NB).mean(axis=1)


def his_interp(logk, t, p, T, P):
    return np.stack([np.asarray(interpolate_log_k_2d(
        logk, t, p, float(T[i]), float(P[i]))) for i in range(len(T))])


def worst_vs_exojax():
    out = {}
    for m in EK["mols"]:
        pack = exomolop.load_tables([m], 0.0, 1e9, verbose=False)
        xs, *_, t_h, p_h, _, _, _ = prov.load_ckd(KTABLES / f"{m}.ktable.h5")
        T, P = EK[f"T_{m}"], EK[f"P_{m}"]
        ours = np.asarray(ckd._interp_logk(
            jnp.asarray(pack.logk[m]), jnp.asarray(pack.t_grid),
            jnp.asarray(pack.p_grid), jnp.asarray(T), jnp.asarray(P)))
        his = his_interp(jnp.asarray(np.log(np.maximum(xs, FLOOR))),
                         jnp.asarray(t_h), jnp.asarray(p_h), T, P)
        d_nat = float(np.max(np.abs(ours - his))) / np.log(10.0)
        gw = np.asarray(pack.gw)
        s_ours = np.einsum("ngb,g->nb", np.exp(ours), gw)
        s_his = np.einsum("ngb,g->nb", np.exp(his), gw)
        nb = s_ours.shape[1] - s_ours.shape[1] % NB
        b = lambda a: a[:, :nb].reshape(a.shape[0], -1, NB).mean(-1)  # noqa: E731
        out[m] = (d_nat, float(np.max(np.abs(np.log10(b(s_ours)) - np.log10(b(s_his))))))
    return out


def h2o_spectrum():
    pack = exomolop.load_tables(["H2O"], 2000.0, 10000.0, verbose=False)
    xs, _, _, t_h, p_h, nu_h, _, _ = prov.load_ckd(KTABLES / "H2O.ktable.h5")
    j0 = int(np.argmin(np.abs(nu_h - pack.nu_bands[0])))
    xs = xs[:, :, :, j0:j0 + pack.nu_bands.size]
    T, P = jnp.asarray([T_SHOW]), jnp.asarray([P_SHOW])
    ours_lk = np.asarray(ckd._interp_logk(
        jnp.asarray(pack.logk["H2O"]), jnp.asarray(pack.t_grid),
        jnp.asarray(pack.p_grid), T, P))[0]
    his_lk = np.asarray(interpolate_log_k_2d(
        jnp.asarray(np.log(np.maximum(xs, FLOOR))), jnp.asarray(t_h),
        jnp.asarray(p_h), T_SHOW, P_SHOW))
    gw = np.asarray(pack.gw)
    wl = 1e4 / np.asarray(pack.nu_bands)
    o = np.argsort(wl)
    ours_k = (np.exp(ours_lk).T @ gw)[o]
    assert np.allclose(wl[o], EK["h2o_wl_um"][o], rtol=1e-12)
    assert np.allclose(ours_k, EK["h2o_sigma_ours_venv"][o], rtol=1e-12)
    return wl[o], ours_k, (np.exp(his_lk).T @ gw)[o], np.asarray(EK["h2o_sigma_exok"])[o]


def spectrum(ax, wl, ours, ej, ek, ylabel, thin=False):
    w_under, w_top = (1.5, 0.7) if thin else (2.5, 1.2)
    ax.plot(wl, ours, color=pastel(INK), lw=w_under, solid_capstyle="round", label="vulcan-forward")
    ax.plot(wl, ej, color=INK, lw=w_top, label="ExoJAX", **DASH_KW)
    ax.plot(wl, ek, color=RED, lw=w_top, label="exo_k 1.3.1", **DOT_KW)
    ax.set_yscale("log")
    ax.set_xlabel(r"wavelength ($\mu$m)"); ax.set_ylabel(ylabel)
    print(f"  {ylabel}: max |dsigma/sigma| ExoJAX {np.max(np.abs(ours / ej - 1)):.1e}, "
          f"exo_k {np.max(np.abs(ours / ek - 1)):.1e}")


wl, ours, ej, ek = h2o_spectrum()
both = worst_vs_exojax()
w_ej_nat = {m: v[0] for m, v in both.items()}
w_ej_bin = {m: v[1] for m, v in both.items()}
w_ek_nat = {m: v / np.log(10.0) for m, v in zip(EK["mols"], EK["worst_exok"])}
w_ek_bin = dict(zip(EK["mols"], EK["worst_exok_binned"]))
mols = sorted(w_ej_nat, key=lambda m: max(w_ej_nat[m], w_ek_nat[m]))

fig, ((a, b), (c, d)) = panels(2, 2)
spectrum(a, wl, ours, ej, ek, "H$_2$O cross section at $R$ = 1000 (cm$^2$)", thin=True)
spectrum(c, binned(wl), binned(ours), binned(ej), binned(ek),
         "H$_2$O cross section at $R$ = 100 (cm$^2$)")
names = [SUB[m] for m in mols]
lollipop(b, names, [w_ej_nat[m] for m in mols], [w_ek_nat[m] for m in mols], ("ExoJAX", "exo_k 1.3.1"), colors=(INK, RED))
b.set_xlim(2e-15, 2e-13); b.set_xlabel(r"worst $|\Delta \log_{10} k|$ per g-point, $R$ = 1000")
lollipop(d, names, [w_ej_bin[m] for m in mols], [w_ek_bin[m] for m in mols], ("ExoJAX", "exo_k 1.3.1"), colors=(INK, RED))
lo = min(min(w_ej_bin.values()), min(w_ek_bin.values())); hi = max(max(w_ej_bin.values()), max(w_ek_bin.values()))
d.set_xlim(lo / 3, hi * 3); d.set_xlabel(r"worst $|\Delta \log_{10} \bar{\sigma}|$ binned to $R$ = 100")
legend(a); b.legend(loc="upper left"); d.legend(loc="upper left")
save(fig, "ckd_verification_vs_exojax_exok.png")
