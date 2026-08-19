"""Render PandExo-parity figures from a gate-PASS parity_summary.json and,
where available, the raw per-wavelength run outputs.

Usage:
    python tests/parity/scripts/make_parity_plots.py

These figures show the quantities that are in PARITY between this tool and
current PandExo on the same Pandeia 2026.7 engine -- the things that match
1:1: the selected groups, integration time, integration counts, and the
extracted stellar flux. The depth-uncertainty difference (a noise-model
difference, not a configuration one) is quantified in REPORT.md and
parity_summary.json, and explained in README.md, not plotted.

The summary is RE-VALIDATED through the shared gate (parity_gate.py) before
any figure gets current-release labels; the persisted `gate.passed` boolean
is never trusted. The extracted-flux figure also reads the raw
{star}_{ours,pandexo}.json that run_parity.py writes into the outputs
directory (git-ignored); it is skipped with a notice if those are absent (a
fresh clone has the committed figures already, and re-running run_parity.py
regenerates the raw JSON).

Layout under tests/parity/: scripts/ (this + the harness), outputs/ (the
committed parity_summary.json + REPORT.md and the git-ignored raw run JSON),
figs/ (the committed PNG figures this writes).

Design: validated categorical palette (dataviz skill). Overlays use blue =
this tool, orange = PandExo; per-mode panels color by mode in the fixed
palette order. One axis per panel, thin marks, recessive grid, PNG @ 200 dpi.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent        # tests/parity/scripts
OUTPUTS = HERE.parent / "outputs"             # parity_summary.json + raw JSON
FIGS = HERE.parent / "figs"                   # committed PNG figures
sys.path.insert(0, str(HERE))

import parity_gate as pg                       # noqa: E402

from jwst_tool import instruments as _ins       # noqa: E402

# --- validated categorical palette (light mode) ------------------------------
TOOL = "#2a78d6"       # this tool (blue, slot 1)
PANDEXO = "#eb6834"    # PandExo (orange, slot 8) -- CVD-safe against blue
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e2"

# The plotted mode set is the GATE's declared experiment, never a local copy.
# A hand-maintained list here silently drops modes from the committed
# config-parity figure while REPORT.md advertises the full matrix;
# test_instruments_registry pins MODE_KEYS against the
# registry, so deriving from it inherits that guard. Colours and short labels
# likewise come from the one validated registry palette.
MODES = list(pg.MODE_KEYS)
MCOL = {k: _ins.MODE_COLOR[k] for k in MODES}
LABEL = {k: (_ins.MODES[k]["label"]
             .replace("NIRSpec ", "").replace("NIRCam ", "")
             .replace("NIRISS ", "").replace(" (ord 1)", "")
             .replace(" (slitless)", ""))
         for k in MODES}
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
    """Re-validate through the shared gate and return the Pandeia release.

    Runs parity_gate.validate_artifact() rather than trusting the persisted
    `gate.passed` boolean, so a hand-edited artifact cannot get
    current-release labels."""
    if summary.get("gate") is None:
        raise RuntimeError(
            "make_parity_plots: parity_summary.json has no gate block "
            "(regenerate with run_parity.py); refusing to put "
            "current-release labels on an unvalidated artifact")
    problems = pg.validate_artifact(summary)
    if problems:
        raise RuntimeError(
            "make_parity_plots: parity_summary.json failed its gate "
            f"({len(problems)} problems; first: {problems[0]}); refusing to "
            "put current-release labels on an unvalidated artifact")
    return str(pg.REQUIRED_PANDEIA_RELEASE)


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
    fig.text(0.5, 0.91, "Unsaturated configurations only (saturated rows are "
             "excluded). The submitted configuration is pinned identically on "
             "both sides; wavelength grids match pixel-for-pixel at rtol "
             "1e-9; groups are independently optimized against the same 80% "
             "saturation target (integer rounding is the residual); "
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
def _require_raw_matches_summary(summary, star, o_all, p_all, of, pf):
    """Refuse to plot raw per-wavelength data from a DIFFERENT run.

    The raw JSON is git-ignored, so a checkout carries the committed figures
    beside whatever raw files happen to be on that machine -- possibly from an
    older experiment. Plotting those silently re-labels stale numbers with the
    committed artifact's release banner.

    This is not hypothetical: an outputs directory holding a worker-v10,
    7-mode run beside a committed worker-v11, 8-mode summary drew a 0.9973
    binned-median flux ratio over an artifact that records exactly 1.0
    (max_abs_dev 2.2e-16). House rule: a check that cannot run must say so,
    and a mismatch is a hard error.
    """
    s_star = summary["stars"][star]
    s_modes = {r["key"] for r in s_star["modes"]}
    problems = []
    for label, raw, path in (("ours", o_all, of), ("pandexo", p_all, pf)):
        r_modes = {k for k in raw if not k.startswith("__")}
        if missing := sorted(s_modes - r_modes):
            problems.append(
                f"{path.name}: missing mode(s) {missing} that the summary "
                f"declares -- this raw file predates the current experiment")
        s_prov = s_star.get(f"provenance_{label}", {})
        r_prov = raw.get("__provenance__", {})
        for key in ("worker_version", "engine_version", "refdata_version",
                    "psf_version"):
            if key in s_prov and key in r_prov and s_prov[key] != r_prov[key]:
                problems.append(
                    f"{path.name}: {key}={r_prov[key]!r} but the summary "
                    f"records {s_prov[key]!r}")
    if problems:
        raise SystemExit(
            "make_parity_plots: the raw run outputs do not belong to the "
            "committed parity_summary.json, so the extracted-flux figure "
            "would misrepresent the gated run:\n  "
            + "\n  ".join(problems)
            + "\n\nEither re-run tests/parity/scripts/run_parity.py to "
              "regenerate the raw outputs for THIS experiment, or move the "
              "stale *_ours.json / *_pandexo.json aside and keep the "
              "committed figure.")


def fig_extracted_flux(summary, out_root, mode="nirspec_g395h",
                       star="w39_like", release="unknown"):
    of = out_root / f"{star}_ours.json"
    pf = out_root / f"{star}_pandexo.json"
    if not (of.exists() and pf.exists()):
        print(f"  [flux fig] raw run outputs not in {out_root} -- skipping "
              "(re-run run_parity.py to regenerate them)")
        return None
    o_all = json.loads(of.read_text())
    p_all = json.loads(pf.read_text())
    _require_raw_matches_summary(summary, star, o_all, p_all, of, pf)
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
                 f"Pandeia {release} both sides; grids matched at rtol 1e-9)")
    ax.legend(frameon=False, fontsize=9.5)
    # both sides receive the identical PandExo-resampled spectrum, and after
    # the quantum-yield unfold the two extractions agree per pixel, so the
    # curves are the same line -- never label a residual here as a stellar-
    # spectrum difference
    ax.annotate("both sides receive the identical resampled stellar\n"
                "spectrum; after the quantum-yield unfold the two\n"
                "extractions agree per pixel (median ratio 1.0000)",
                xy=(0.985, 0.97), xycoords="axes fraction", ha="right",
                va="top", fontsize=7.6, color=INK2)
    _style(ax)
    axr = axes[1]
    axr.plot(wl_pair, ratio, color="#c3c2bd", lw=0.6, alpha=0.9, zorder=2,
             label="per-pixel")
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
    made = [fig_config_parity(summary, release)]
    # The flux figure ALSO needs the raw per-wavelength JSON (written by
    # run_parity.py, git-ignored). A stale raw set is refused rather than
    # plotted, but that must not discard the config figure, which is built
    # from the committed summary alone: report both, then exit non-zero.
    stale = None
    try:
        made.append(fig_extracted_flux(summary, OUTPUTS, release=release))
    except SystemExit as exc:
        stale = str(exc)
    for pth in made:
        if pth is not None:
            print(f"wrote {pth}")
    if stale:
        print(f"\n{stale}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
