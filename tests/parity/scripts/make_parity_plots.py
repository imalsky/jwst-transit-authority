"""Render PandExo-parity figures from a gate-PASS parity_summary.json and,
where available, the raw per-wavelength run outputs.

Usage:
    python tests/parity/scripts/make_parity_plots.py

These figures show the quantities that are in PARITY between this tool and
current PandExo on the same Pandeia 2026.7 engine -- the things that match
1:1: the selected groups, integration time, integration counts, and the
extracted stellar flux. The depth-uncertainty difference (a noise-model
difference, not a configuration one) is quantified in REPORT.md, not plotted.

The committed historical summary does not pass the current gate and therefore
cannot be plotted with current-release labels. The extracted-flux figure also
reads the raw
{star}_{ours,pandexo}.json that run_parity.py writes into this same directory
(git-ignored); it is skipped with a notice if those are absent (a fresh clone
has the committed figures already, and re-running run_parity.py regenerates
the raw JSON).

Layout under tests/parity/: scripts/ (this + the harness), outputs/ (the
committed parity_summary.json + REPORT.md and the git-ignored raw run JSON),
figs/ (the committed PNG figures this writes).

Design: validated categorical palette (dataviz skill). Overlays use blue =
this tool, orange = PandExo; per-mode panels color by mode in the fixed
palette order. One axis per panel, thin marks, recessive grid, PNG @ 200 dpi.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent        # tests/parity/scripts
OUTPUTS = HERE.parent / "outputs"             # parity_summary.json + raw JSON
FIGS = HERE.parent / "figs"                   # committed PNG figures

# --- validated categorical palette (light mode) ------------------------------
TOOL = "#2a78d6"       # this tool (blue, slot 1)
PANDEXO = "#eb6834"    # PandExo (orange, slot 8) -- CVD-safe against blue
MODE_HUES = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
             "#4a3aa7", "#e34948", "#e87ba4"]
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e2"

MODES = ["nirspec_prism", "nirspec_g395h", "nirspec_g235h", "niriss_soss",
         "nircam_f322w2", "nircam_f444w", "miri_lrs"]
LABEL = {"nirspec_prism": "PRISM", "nirspec_g395h": "G395H",
         "nirspec_g235h": "G235H", "niriss_soss": "SOSS",
         "nircam_f322w2": "F322W2", "nircam_f444w": "F444W",
         "miri_lrs": "MIRI LRS"}
MCOL = dict(zip(MODES, MODE_HUES))
STAR_MARK = {"w39_like": "o", "bright_hot": "s", "faint_k": "^"}
STAR_LABEL = {"w39_like": "W39-like (Ks 10.7)", "bright_hot": "bright (Ks 8.5)",
              "faint_k": "faint (Ks 13)"}
SAT_LIMIT = 0.80   # a mode with sat_frac_ours above this is saturated (unusable)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 200,
})


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_summary():
    return json.loads((OUTPUTS / "parity_summary.json").read_text())


def require_passing_summary(summary):
    """Return the validated Pandeia release or refuse a stale/failed input."""
    gate = summary.get("gate")
    if not gate or not gate.get("passed"):
        detail = (
            "has no gate block (regenerate with run_parity.py)"
            if gate is None else
            f"failed its gate ({len(gate.get('problems', []))} problems)"
        )
        raise RuntimeError(
            "make_parity_plots: parity_summary.json " + detail
            + "; refusing to put current-release labels on an unvalidated "
              "artifact")
    release = gate.get("required_pandeia_release")
    if not release:
        raise RuntimeError(
            "make_parity_plots: passing gate lacks required_pandeia_release")
    return str(release)


def ok_rows(summary, star):
    return {m["key"]: m for m in summary["stars"][star]["modes"]
            if m.get("status") == "OK"}


# =============================================================================
# Configuration & timing parity: ours vs PandExo on the 1:1 line
# =============================================================================
def fig_config_parity(summary, release):
    quantities = [
        ("ngroup_pandexo", "ngroup_ours", "groups / integration"),
        ("t_int_pandexo_s", "t_int_ours_s", "integration time"),
        ("n_int_pandexo_in", "n_int_ours", "integrations in transit"),
    ]
    unit = {"integration time": " (s)"}
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.7))
    for ax, (kx, ky, title) in zip(axes, quantities):
        xs, ys = [], []
        for star in summary["stars"]:
            rows = ok_rows(summary, star)
            for key in MODES:
                if key not in rows:
                    continue
                # skip saturated (unusable) configs entirely -- only VALID
                # configurations are a meaningful parity comparison. PRISM
                # saturates on the bright stars and is dropped there; it is
                # shown from the faint star, where it is usable.
                if rows[key].get("sat_frac_ours", 0.0) > SAT_LIMIT:
                    continue
                x, y = rows[key][kx], rows[key][ky]
                xs.append(x)
                ys.append(y)
                ax.scatter(x, y, s=46, marker=STAR_MARK[star],
                           color=MCOL[key], edgecolor="white", linewidth=0.7,
                           zorder=3)
        lo = min(xs + ys) * 0.7
        hi = max(xs + ys) * 1.4
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.0, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        u = unit.get(title, "")
        ax.set_xlabel(f"PandExo  {title}{u}")
        ax.set_ylabel(f"this tool  {title}{u}")
        ax.set_title(title)
        _style(ax)
    mode_handles = [Line2D([], [], marker="o", ls="", color=MCOL[m],
                           markeredgecolor="white", label=LABEL[m])
                    for m in MODES]
    star_handles = [Line2D([], [], marker=STAR_MARK[s], ls="", color=INK2,
                           markeredgecolor="white", label=STAR_LABEL[s])
                    for s in STAR_MARK]
    line_handle = [Line2D([], [], color=INK2, ls="--", label="1:1 parity")]
    fig.legend(handles=mode_handles + star_handles + line_handle,
               loc="lower center", ncol=6, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Configuration & timing parity: this tool vs pinned PandExo "
                 f"on the same Pandeia {release} engine", fontsize=11.5,
                 y=0.99)
    fig.text(0.5, 0.91, "Valid configurations only (saturated/unusable configs "
             "are excluded). Config + wavelength grid are bit-identical "
             "(max |Δλ| = 0); groups are independently optimized and agree to "
             "≤1 (integer rounding of the same 80% saturation target); "
             "integration time and count follow.", ha="center", fontsize=8.2,
             color=INK2, wrap=True)
    fig.tight_layout(rect=[0, 0.15, 1, 0.87])
    out = FIGS / "parity_config_timing.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# Extracted stellar flux parity (the engine product agreeing 1:1)
# =============================================================================
def fig_extracted_flux(summary, out_root, mode="nirspec_g395h",
                       star="w39_like", release="unknown"):
    of = out_root / f"{star}_ours.json"
    pf = out_root / f"{star}_pandexo.json"
    if not (of.exists() and pf.exists()):
        print(f"  [flux fig] raw run outputs not in {out_root} -- skipping "
              "(re-run run_parity.py to regenerate them)")
        return None
    o = json.loads(of.read_text())[mode]
    p = json.loads(pf.read_text())[mode]
    wl_o = np.asarray(o["wl"])
    flux_o = np.asarray(o["flux"])
    order = np.argsort(wl_o)
    wl_o, flux_o = wl_o[order], flux_o[order]
    wl_p = np.asarray(p["wave"])
    erate = np.asarray(p["e_rate_out"])
    # pair on the shared extraction grid (identical wavelengths)
    idx = np.clip(np.searchsorted(wl_o, wl_p), 0, wl_o.size - 1)
    ex = np.abs(wl_o[idx] - wl_p) < 1e-9 * np.maximum(wl_p, 1e-9)
    io, ip = idx[ex], np.where(ex)[0]
    wl_pair = wl_o[io]
    ratio = flux_o[io] / erate[ip]
    med = float(np.median(ratio))
    # binned running median: the real systematic agreement, with the
    # per-pixel photon-level extraction jitter averaged out (the tool never
    # uses per-pixel flux -- it integrates over bins)
    nb = 24
    bedges = np.linspace(wl_pair.min(), wl_pair.max(), nb + 1)
    bc = 0.5 * (bedges[:-1] + bedges[1:])
    bmed = np.array([
        np.median(ratio[(wl_pair >= bedges[k]) & (wl_pair < bedges[k + 1])])
        if ((wl_pair >= bedges[k]) & (wl_pair < bedges[k + 1])).any() else np.nan
        for k in range(nb)])

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.15]})
    ax = axes[0]
    ax.plot(wl_p, erate, color=PANDEXO, lw=1.4, label="PandExo", zorder=2)
    ax.plot(wl_o, flux_o, color=TOOL, lw=1.4, ls=(0, (4, 2)),
            label="this tool", zorder=3)
    ax.set_ylabel("extracted stellar\ncount rate  (e$^-$/s)")
    ax.set_title(f"Extracted stellar flux parity, {LABEL[mode]} on a "
                 f"{STAR_LABEL[star]} star\n(the ETC engine product, "
                 f"Pandeia {release} both sides; wavelength grid bit-identical)")
    ax.legend(frameon=False, fontsize=9.5)
    ax.annotate("narrow dips = stellar absorption lines in this tool's\n"
                "PHOENIX spectrum (e.g. Br-α 4.05, Pf-δ 3.30 μm); PandExo's\n"
                "separate stellar spectrum smooths them. They cancel in the\n"
                "transit-depth ratio and wash out in binning.",
                xy=(0.985, 0.97), xycoords="axes fraction", ha="right",
                va="top", fontsize=7.6, color=INK2)
    _style(ax)
    axr = axes[1]
    axr.plot(wl_pair, ratio, color="#c3c2bd", lw=0.6, alpha=0.9, zorder=2,
             label="per-pixel (independent-extraction jitter)")
    axr.plot(bc, bmed, color=TOOL, lw=2.0, zorder=3,
             label=f"binned median = {med:.4f} (the systematic)")
    axr.axhline(1.0, color=PANDEXO, lw=1.0, ls=":", zorder=1)
    axr.set_ylim(0.9, 1.1)
    axr.set_ylabel("ratio\ntool / PandExo")
    axr.set_xlabel("wavelength (micron)")
    axr.legend(frameon=False, fontsize=8, loc="lower left", ncol=1)
    _style(axr)
    fig.tight_layout()
    out = FIGS / "parity_extracted_flux.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    summary = load_summary()
    release = require_passing_summary(summary)
    # raw per-run JSON lives in this dir (written by run_parity.py, git-ignored)
    made = [fig_config_parity(summary, release),
            fig_extracted_flux(summary, OUTPUTS, release=release)]
    for pth in made:
        if pth is not None:
            print(f"wrote {pth}")


if __name__ == "__main__":
    main()
