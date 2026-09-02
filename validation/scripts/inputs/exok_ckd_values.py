"""Writes validation/data/exok_ckd_vals.npz: the exo_k 1.3.1 side of the
correlated-k cross-validation figure.

Per molecule in the engine's k-table tree ($VULCAN_FORWARD_DATA/exomolop):
probe (T, P) points (grid nodes + random interior + out-of-range clamping,
seed 42), exo_k Ktable.interpolate_kdata (log_interp) vs vulcan_forward
load_tables + ckd._interp_logk. Saves the probes (reused verbatim by the
figure script for the ExoJAX side), the per-molecule worst |dln k| (native)
and worst |dlog10 band-mean sigma| binned R=1000 -> R=100, and the H2O
band-mean spectrum at 1100 K, 1 mbar over 2000-10000 cm^-1.

exo_k pulls numpy 2 and never enters this repo's jax environment. Run OFFLINE
in a throwaway venv (same recipe as the exok_ref_overlap.npz fixture
generator, plus the engine itself for the comparison side):

    python -m venv exok-venv
    exok-venv/bin/pip install "exo_k==1.3.1" jax h5py
    exok-venv/bin/pip install --no-deps -e <the vulcan-forward checkout>
    exok-venv/bin/python validation/scripts/inputs/exok_ckd_values.py

Zero-k table entries are floored to exomolop.K_FLOOR (1e-60) in exo_k's
kdata BEFORE interpolation, matching load_tables' floor, so the log-space
interpolations are the same mathematical operation everywhere (exo_k is
linear in T and log10 P on log k; ours in T and ln P -- identical weights).
"""
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import exo_k as xk

from vulcan_forward import ckd, exomolop, paths

OUT = Path(__file__).resolve().parents[2] / "data"
DATA = paths.exomolop_dir()
FLOOR = exomolop.K_FLOOR
T_SHOW, P_SHOW = 1100.0, 1e-3
WN_SHOW = (2000.0, 10000.0)
NB = 10                                   # R=1000 -> R=100 binning factor

xk.Settings().set_mks(False)
mols = sorted(p.name.split(".")[0] for p in DATA.glob("*.ktable.h5"))
rng = np.random.default_rng(42)
out = {"mols": np.array(mols)}
worst, worst_b = [], []

for m in mols:
    kt = xk.Ktable(filename=str(DATA / f"{m}.ktable.h5"), mol=m,
                   remove_zeros=False)
    assert kt.p_unit == "bar" and kt.kdata_unit == "cm^2/molecule", m
    kt.kdata = np.maximum(kt.kdata, FLOOR)

    pack = exomolop.load_tables([m], 0.0, 1e9, verbose=False)
    t_np, p_np = np.asarray(pack.t_grid), np.asarray(pack.p_grid)
    assert np.array_equal(kt.tgrid, t_np) and np.allclose(kt.pgrid, p_np)

    T = np.concatenate([t_np, rng.uniform(t_np[0], t_np[-1], 12),
                        [0.5 * t_np[0], 2.0 * t_np[-1]]])
    P = np.concatenate([p_np[rng.integers(0, p_np.size, t_np.size)],
                        np.exp(rng.uniform(np.log(p_np[0]),
                                           np.log(p_np[-1]), 12)),
                        [0.1 * p_np[0], 10.0 * p_np[-1]]])

    k_ek = kt.interpolate_kdata(logp_array=np.log10(P), t_array=T,
                                log_interp=True)      # (n, band, g)
    ek_logk = np.log(np.swapaxes(k_ek, 1, 2))          # (n, g, band)
    ours = np.asarray(ckd._interp_logk(
        jnp.asarray(pack.logk[m]), jnp.asarray(pack.t_grid),
        jnp.asarray(pack.p_grid), jnp.asarray(T), jnp.asarray(P)))
    d = float(np.max(np.abs(ours - ek_logk)))
    worst.append(d)
    out[f"T_{m}"], out[f"P_{m}"] = T, P

    # band-mean sigma at each probe, binned R=1000 -> R=100
    gw = np.asarray(pack.gw)
    s_ours = np.einsum("ngb,g->nb", np.exp(ours), gw)
    s_ek = np.einsum("ngb,g->nb", np.exp(ek_logk), gw)
    nb = s_ours.shape[1] - s_ours.shape[1] % NB
    b = lambda a: a[:, :nb].reshape(a.shape[0], -1, NB).mean(-1)  # noqa: E731
    db = float(np.max(np.abs(np.log10(b(s_ours)) - np.log10(b(s_ek)))))
    worst_b.append(db)
    print(f"{m:>6}  max|dlnk| ours vs exo_k = {d:.2e}   "
          f"binned R=100 max|dlog10 sigma| = {db:.2e}", flush=True)

    if m == "H2O":
        kt.clip_spectral_range(wn_range=list(WN_SHOW))
        pk = exomolop.load_tables(["H2O"], *WN_SHOW, verbose=False)
        assert kt.kdata.shape[2] == pk.nu_bands.size
        ks = kt.interpolate_kdata(logp_array=np.log10([P_SHOW]),
                                  t_array=np.array([T_SHOW]),
                                  log_interp=True)[0]          # (band, g)
        gw = np.asarray(pk.gw)
        out["h2o_wl_um"] = 1e4 / np.asarray(pk.nu_bands)
        out["h2o_sigma_exok"] = ks @ gw
        ol = np.asarray(ckd._interp_logk(
            jnp.asarray(pk.logk["H2O"]), jnp.asarray(pk.t_grid),
            jnp.asarray(pk.p_grid), jnp.asarray([T_SHOW]),
            jnp.asarray([P_SHOW])))[0]
        out["h2o_sigma_ours_venv"] = np.exp(ol).T @ gw          # sanity copy
    del kt, pack

out["worst_exok"] = np.array(worst)
out["worst_exok_binned"] = np.array(worst_b)
np.savez_compressed(OUT / "exok_ckd_vals.npz", **out)
print("wrote", OUT / "exok_ckd_vals.npz")
