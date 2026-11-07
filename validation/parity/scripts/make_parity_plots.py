"""Render PandExo-parity figures from a gate-PASS parity_summary.json and,
where available, the raw per-wavelength run outputs.

Method and contract: docs/validation.md, section `make_parity_plots`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent  # validation/parity/scripts
OUTPUTS = HERE.parent / "outputs"  # parity_summary.json + raw JSON
FIGS = HERE.parent / "figs"  # committed PNG figures
sys.path.insert(0, str(HERE))

import parity_gate as pg

from jwst_tool import instruments as _ins

sys.path.insert(0, str(HERE.parents[1]))  # validation/figstyle
import figstyle as fs
from figstyle import CYC, INK, RED, MS

fs.use()

# The plotted mode set is the GATE's declared experiment, never a local copy.
# A hand-maintained list here silently drops modes from the committed
# config-parity figure while REPORT.md advertises the full matrix;
# test_instruments_registry pins MODE_KEYS against the
# registry, so deriving from it inherits that guard. Colours and short labels
# likewise come from the one validated registry palette.
MODES = list(pg.MODE_KEYS)
INSTRUMENT_LABEL = {"nirspec": "NIRSpec", "niriss": "NIRISS", "nircam": "NIRCam", "miri": "MIRI"}
INSTRUMENT_COLOR = {ins: CYC[i] for i, ins in enumerate(INSTRUMENT_LABEL)}
LABEL = {
    k: (
        _ins.MODES[k]["label"]
        .replace("NIRSpec ", "")
        .replace("NIRCam ", "")
        .replace("NIRISS ", "")
        .replace(" (ord 1)", "")
        .replace(" (slitless)", "")
    )
    for k in MODES
}
STAR_MARK = {"w39_like": "o", "bright_hot": "s", "faint_k": "^"}
# gate-owned data, never re-typed here (the MODES rule above applies)
STAR_LABEL = {
    "w39_like": f"W39-like (Ks {pg.STARS['w39_like']['ks_mag']:.1f})",
    "bright_hot": f"bright (Ks {pg.STARS['bright_hot']['ks_mag']:.1f})",
    "faint_k": f"faint (Ks {pg.STARS['faint_k']['ks_mag']:.0f})",
}
SAT_LIMIT = pg.SAT_LIMIT  # a mode above this is saturated (unusable)


def load_summary():
    return json.loads((OUTPUTS / "parity_summary.json").read_text())


def require_passing_summary(summary):
    """Re-validate through the shared gate.

    Runs parity_gate.validate_artifact() rather than trusting the persisted
    `gate.passed` boolean, so a hand-edited artifact cannot get
    current-release labels."""
    if summary.get("gate") is None:
        raise RuntimeError(
            "make_parity_plots: parity_summary.json has no gate block "
            "(regenerate with run_parity.py); refusing to put "
            "current-release labels on an unvalidated artifact"
        )
    problems = pg.validate_artifact(summary)
    if problems:
        raise RuntimeError(
            "make_parity_plots: parity_summary.json failed its gate "
            f"({len(problems)} problems; first: {problems[0]}); refusing to "
            "put current-release labels on an unvalidated artifact"
        )


def ok_rows(summary, star):
    return {m["key"]: m for m in summary["stars"][star]["modes"] if m.get("status") == "OK"}


# Configuration & timing parity: ours vs PandExo on the 1:1 line
def fig_config_parity(summary):
    quantities = [
        ("ngroup_pandexo", "ngroup_ours", "groups per integration"),
        ("t_int_pandexo_s", "t_int_ours_s", "integration time (s)"),
        ("n_int_pandexo_in", "n_int_ours", "integrations in transit"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, (kx, ky, name) in zip(axes, quantities):
        xs, ys = [], []
        for star in summary["stars"]:
            rows = ok_rows(summary, star)
            for key in MODES:
                # saturated configurations are unusable, not a parity case
                if key not in rows or rows[key].get("sat_frac_ours", 0.0) > SAT_LIMIT:
                    continue
                x, y = rows[key][kx], rows[key][ky]
                xs.append(x)
                ys.append(y)
                ax.plot(
                    x,
                    y,
                    STAR_MARK[star],
                    ms=MS,
                    ls="",
                    clip_on=False,
                    color=INSTRUMENT_COLOR[_ins.MODES[key]["instrument"]],
                    zorder=3,
                )
        lo, hi = min(xs + ys), max(xs + ys)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.0, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        ax.set_xlabel(f"PandExo {name}")
        ax.set_ylabel(f"this tool {name}")
    handles = (
        [
            Line2D([], [], marker="o", ls="", ms=MS, color=c, label=INSTRUMENT_LABEL[k])
            for k, c in INSTRUMENT_COLOR.items()
        ]
        + [Line2D([], [], marker=STAR_MARK[st], ls="", ms=MS, color=INK, label=STAR_LABEL[st]) for st in STAR_MARK]
        + [Line2D([], [], color=INK, ls="--", lw=1.0, label="1:1")]
    )
    axes[0].legend(handles=handles, loc="upper left")
    out = fs.save(fig, "parity_config_timing.png", out_dir=FIGS)
    plt.close(fig)
    return out


# Extracted stellar flux parity (the engine product agreeing 1:1)
def fig_extracted_flux(summary, out_root, mode="nirspec_g395h", star="w39_like"):
    of = out_root / f"{star}_ours.json"
    pf = out_root / f"{star}_pandexo.json"
    if not (of.exists() and pf.exists()):
        print(f"  [flux fig] raw run outputs not in {out_root} -- skipping (re-run run_parity.py to regenerate them)")
        return None
    o_all = json.loads(of.read_text())
    p_all = json.loads(pf.read_text())
    # Plotting another run's raw files would re-label stale numbers with the committed artifact's banner.
    if problems := pg.raw_run_problems(summary["stars"][star], o_all, p_all, mode):
        raise SystemExit(
            "make_parity_plots: the raw run outputs do not belong to the "
            "committed parity_summary.json, so the extracted-flux figure "
            "would misrepresent the gated run:\n  "
            + "\n  ".join(problems)
            + "\n\nEither re-run validation/parity/scripts/run_parity.py to "
            "regenerate the raw outputs for THIS experiment, or move the "
            "stale *_ours.json / *_pandexo.json aside and keep the "
            "committed figure."
        )
    o = o_all[mode]
    p = p_all[mode]
    wl_o = np.asarray(o["wl"])
    flux_o = np.asarray(o["flux"])
    order = np.argsort(wl_o)
    wl_o, flux_o = wl_o[order], flux_o[order]
    wl_p = np.asarray(p["wave"])
    # back to pandeia's electron rate: PandExo's remove_QY divided the
    # quantum yield out of e_rate_out (see run_parity.compare_mode)
    erate = np.asarray(p["e_rate_out"]) * np.asarray(p["qy_on_grid"])
    # pair on the shared extraction grid (identical wavelengths)
    idx = np.clip(np.searchsorted(wl_o, wl_p), 0, wl_o.size - 1)
    ex = np.abs(wl_o[idx] - wl_p) < 1e-9 * np.maximum(wl_p, 1e-9)
    io, ip = idx[ex], np.where(ex)[0]
    ratio = flux_o[io] / erate[ip]
    med = float(np.median(ratio))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(wl_p, erate, color=INK, lw=1.5, solid_capstyle="round", label=f"PandExo, {LABEL[mode]}, {STAR_LABEL[star]}")
    ax.plot(wl_o, flux_o, color=RED, lw=0.7, ls="--", label=f"this tool (median ratio {med:.4f})")
    ax.set_xlabel("wavelength ($\\mu$m)")
    ax.set_ylabel("extracted stellar count rate (e$^-$ s$^{-1}$)")
    ax.legend(loc="upper right")
    out = fs.save(fig, "parity_extracted_flux.png", out_dir=FIGS)
    plt.close(fig)
    return out


def main():
    summary = load_summary()
    require_passing_summary(summary)
    fig_config_parity(summary)
    # The flux figure ALSO needs the raw per-wavelength JSON (written by
    # run_parity.py, git-ignored). A stale raw set is refused rather than
    # plotted, but that must not discard the config figure, which is built
    # from the committed summary alone: report both, then exit non-zero.
    stale = None
    try:
        fig_extracted_flux(summary, OUTPUTS)
    except SystemExit as exc:
        stale = str(exc)
    if stale:
        print(f"\n{stale}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
