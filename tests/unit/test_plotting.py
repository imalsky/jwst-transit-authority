"""Pure figure builders: the render lock, square panels, and tick hygiene.

Numpy + matplotlib only (no streamlit, no engine), so this stays in the fast
suite. The point of the extraction in ``plotting.py`` is that these tests
drive the SAME builders the GUI draws with -- a parallel reimplementation here
would prove nothing.
"""
from __future__ import annotations

import ast
import inspect
import io

import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pytest                        # noqa: E402

from jwst_tool import plotting, summary_figure   # noqa: E402

APP = Path(__file__).resolve().parents[2] / "src" / "jwst_tool" / "app.py"

MOLS = ["H2O", "CO", "CO2", "CH4", "SO2", "NH3", "HCN", "C2H2"]


def _tp():
    p = np.logspace(-8.0, 2.0, 120)
    return p, 900.0 + 700.0 * np.tanh(np.log10(p) + 4.0)


def _vmr_cols(p):
    return [(m, 10.0 ** (-2.5 - i * 1.2) * (1.0 + 0.4 * np.sin(np.log10(p))))
            for i, m in enumerate(MOLS)]


def _summary_fig(n_points: int = 3, with_sigma: bool = False):
    """The composed summary figure. ``n_points`` sets how many mode series
    are drawn, which is what makes the spectrum legend grow. ``with_sigma``
    attaches mu/sigma to each posterior curve, which is what makes the panel
    quote its width and shade the 1-sigma interval."""
    wl = np.linspace(1.0, 12.0, 300)
    _palette = ["#1f4e9c", "#8c2d04", "#0f6b4f", "#6a3d9a", "#117733",
                "#882255"]
    pts = [dict(label=f"mode {i}: {4.2 - i * 0.4:.1f}s",
                color=_palette[i % len(_palette)], marker="^",
                wl_um=np.linspace(1.0 + 1.5 * i, 4.0 + 1.5 * i, 25),
                depth_ppm=np.full(25, 20000.0 + 50 * i),
                sigma_ppm=np.full(25, 60.0))
           for i in range(n_points)]
    th = np.linspace(-1.0, 1.0, 200)
    _extra1 = dict(mu=0.02, sigma=0.1) if with_sigma else {}
    _extra2 = dict(mu=-0.01, sigma=0.2) if with_sigma else {}
    pans = [dict(axis_label="log Z",
                 curves=[dict(label="PRISM 0.044", theta=th,
                              pdf=np.exp(-th ** 2 / 0.02), **_extra1)],
                 center=0.0),
            dict(axis_label="C/O",
                 curves=[dict(label="G395H 0.11", theta=th,
                              pdf=np.exp(-th ** 2 / 0.08), **_extra2)],
                 center=0.0)]
    spec = dict(wl_um=wl, depth_ppm=20000.0 + 300.0 * np.sin(wl * 3),
                depth_label="transit depth (ppm)",
                model_label="model (smoothed for display)",
                legend_title="SO2 S/N per mode, 1 transit", points=pts)
    return summary_figure.compose_summary_figure(
        spec, posterior_panels=pans, title="test",
        footnote="one seeded realization")


def _visible_major_label_bboxes(ax, renderer):
    out = {}
    for axis in (ax.xaxis, ax.yaxis):
        out[axis.axis_name] = [
            t.label1.get_window_extent(renderer)
            for t in axis.get_major_ticks()
            if t.label1.get_visible() and t.label1.get_text()]
    return out


# ---------------------------------------------------------------------------
# The render lock: the ParseException crash of 2026-08-13
# ---------------------------------------------------------------------------

def test_concurrent_renders_do_not_raise():
    """Eight threads through the real builders: no mathtext ParseException.

    matplotlib guards its process-global mathtext parser with
    Figure._render_lock for the duration of Figure.draw, but tight_layout
    measures tick labels OUTSIDE that lock -- and on a log axis those labels
    are mathtext. Streamlit gives each session its own script-runner thread,
    so two users rendering at once raced into the shared parser and the loser
    got 'ValueError: ParseException: exception raised in parse action'.

    Measured before the fix: 7/8 threads raised on the deployed pin
    (matplotlib 3.10.0), 8/8 locally on 3.11.0.
    """
    p, T = _tp()
    cols = _vmr_cols(p)
    errors: list[str] = []

    def work():
        try:
            for _ in range(10):
                with plotting.render_lock:
                    fig, ylim = plotting.build_tp_figure(p, T)
                    fig.savefig(io.BytesIO(), format="png")
                    plt.close(fig)
                    g = plotting.build_vmr_figure(p, cols, ylim=ylim)
                    g.savefig(io.BytesIO(), format="png")
                    plt.close(g)
                    s = _summary_fig()
                    s.savefig(io.BytesIO(), format="png")
                    plt.close(s)
        except BaseException as exc:                      # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:3]


@pytest.mark.parametrize("builder", ["build_tp_figure", "build_vmr_figure"])
def test_builders_block_while_the_render_lock_is_held(builder):
    """BEHAVIORAL: each builder actually ACQUIRES the lock -- it does not
    merely tolerate the lock being held around it.

    The threaded test above is probabilistic: a future matplotlib could stop
    reproducing the race and silently void it. This one holds the lock in
    another thread and asserts the builder cannot finish, so it fails the
    moment a builder stops serializing, whatever matplotlib does.
    """
    p, T = _tp()
    cols = _vmr_cols(p)
    calls = {"build_tp_figure": lambda: plotting.build_tp_figure(p, T)[0],
             "build_vmr_figure": lambda: plotting.build_vmr_figure(p, cols)}
    held = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    box = {}

    def holder():
        with plotting.render_lock:
            held.set()
            release.wait(10.0)

    def builder_thread():
        box["fig"] = calls[builder]()
        finished.set()

    h = threading.Thread(target=holder)
    h.start()
    assert held.wait(10.0), "holder thread never acquired the lock"
    b = threading.Thread(target=builder_thread)
    b.start()
    try:
        # the lock is held elsewhere, so the builder must not complete
        assert not finished.wait(1.5), \
            f"{builder} completed while render_lock was held by another thread"
    finally:
        release.set()
        h.join(10.0)
        b.join(10.0)
    assert finished.is_set(), f"{builder} never completed after release"
    plt.close(box["fig"])


def test_summary_figure_composition_is_locked():
    """compose_summary_figure serializes too: it mutates global rcParams via
    plt.style.context AND lays out mathtext axes."""
    src = inspect.getsource(summary_figure.compose_summary_figure)
    assert "plotting.render_lock" in src
    # the lock must be entered no later than the style context, which is the
    # other global-state hazard in that function
    assert src.index("render_lock") < src.index("style.context")


def test_app_materializes_figures_only_through_locked_helpers():
    """STRUCTURAL: no unlocked layout/export/render call survives in app.py.

    app.py must go through plotting.build_* / _fig_png / _fig_pdf / _show_fig,
    all of which hold the lock. A bare fig.tight_layout(), fig.savefig() or
    st.pyplot() there would reintroduce the crash, so parse the module and
    fail on one.
    """
    src = APP.read_text()
    tree = ast.parse(src)
    locked_helpers = {"_fig_png", "_fig_pdf", "_show_fig"}

    # every materialization call must be lexically INSIDE one of the locked
    # helpers (which each open `with plotting.render_lock`)
    exempt_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in locked_helpers:
            body = ast.get_source_segment(src, node) or ""
            assert "render_lock" in body, \
                f"{node.name} no longer holds plotting.render_lock"
            exempt_lines.update(range(node.lineno, node.end_lineno + 1))

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func,
                                                            ast.Attribute):
            continue
        attr = node.func.attr
        if attr in ("tight_layout", "savefig", "pyplot") \
                and node.lineno not in exempt_lines:
            offenders.append(f"line {node.lineno}: .{attr}()")
    assert not offenders, (
        "unlocked figure layout/export/render in app.py -- route it through "
        f"{sorted(locked_helpers)}: {offenders}")


# ---------------------------------------------------------------------------
# Square panels + tick hygiene
# ---------------------------------------------------------------------------

def test_tp_and_vmr_panels_share_one_geometry():
    """Both panels: square AXES BOX on an identical (non-square) canvas.

    2026-08-13 (maintainer: "make the pt figure and the volume mixing figure
    the same size"). The canvas is 5.6 x 4.0 in and only the axes box is
    square; the surplus width is the mixing-ratio legend strip, which the T-P
    figure leaves blank so the two render at the same on-page size. An earlier
    revision made the whole canvas square -- that is what this replaces.

    Rendered size is checked in test_paired_panels_render_at_identical_size;
    here we pin the geometry the builders declare.
    """
    p, T = _tp()
    fig, ylim = plotting.build_tp_figure(p, T)
    g = plotting.build_vmr_figure(p, _vmr_cols(p), ylim=ylim)
    try:
        for name, f in (("tp", fig), ("vmr", g)):
            ax = f.axes[0]
            assert ax.get_box_aspect() == 1.0, name
            assert tuple(f.get_size_inches()) == (plotting.FIG_W_IN,
                                                  plotting.FIG_H_IN), name
        # identical canvas AND identical axes rectangle in pixels
        fig.canvas.draw()
        g.canvas.draw()
        a = fig.axes[0].get_window_extent()
        b = g.axes[0].get_window_extent()
        assert (round(a.width, 3), round(a.height, 3)) == \
               (round(b.width, 3), round(b.height, 3)), (a, b)
        # the rect must ALLOCATE a square box, or set_box_aspect silently
        # shrinks the axes and the panels stop matching
        alloc_w = (plotting.AXES_RECT["right"]
                   - plotting.AXES_RECT["left"]) * plotting.FIG_W_IN
        alloc_h = (plotting.AXES_RECT["top"]
                   - plotting.AXES_RECT["bottom"]) * plotting.FIG_H_IN
        assert abs(alloc_w - alloc_h) < 0.01, (alloc_w, alloc_h)
        assert abs(a.width - a.height) < 1.0, (a.width, a.height)
    finally:
        plt.close(fig)
        plt.close(g)


@pytest.mark.parametrize("n_species", [3, 8])
def test_paired_panels_render_at_identical_size(n_species):
    """The rendered PNGs must match, whatever the legend contains.

    This is the invariant the maintainer actually asked for. It only holds on
    the uncropped path: bbox_inches="tight" crops each figure to its own ink,
    so the legend strip makes the mixing-ratio panel wider. app.py renders
    both with tight=False for exactly this reason.
    """
    p, T = _tp()
    fig, ylim = plotting.build_tp_figure(p, T)
    g = plotting.build_vmr_figure(p, _vmr_cols(p)[:n_species], ylim=ylim)
    try:
        sizes = []
        for f in (fig, g):
            buf = io.BytesIO()
            with plotting.render_lock:
                # f.bbox_inches, NOT None: science.mplstyle sets
                # savefig.bbox=tight and None means "use the rcParam"
                f.savefig(buf, format="png", dpi=100, bbox_inches=f.bbox_inches)
            sizes.append(len(buf.getvalue()) and _png_size(buf.getvalue()))
        assert sizes[0] == sizes[1], sizes
    finally:
        plt.close(fig)
        plt.close(g)


def _png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG IHDR -- avoids a Pillow dependency."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return (int.from_bytes(data[16:20], "big"),
            int.from_bytes(data[20:24], "big"))


@pytest.mark.parametrize("which", ["tp", "vmr", "summary"])
def test_major_tick_labels_never_overlap(which):
    """No two visible MAJOR tick labels may intersect on any axis.

    The mixing-ratio panel is the reproducer: a decade-per-tick log x axis
    over 1e-12..1 plus the full pressure grid, in a narrow square panel.
    Drawn under the render lock (the draw is what measures the text).
    """
    p, T = _tp()
    with plotting.render_lock:
        if which == "tp":
            fig, _ = plotting.build_tp_figure(p, T)
        elif which == "vmr":
            _, ylim = plotting.build_tp_figure(p, T)
            plt.close("all")
            fig, ylim2 = plotting.build_tp_figure(p, T)
            plt.close(fig)
            fig = plotting.build_vmr_figure(p, _vmr_cols(p), ylim=ylim2)
        else:
            fig = _summary_fig()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bad = []
        for i, ax in enumerate(fig.axes):
            for axis_name, boxes in _visible_major_label_bboxes(
                    ax, renderer).items():
                for a, b in zip(boxes, boxes[1:]):
                    if a.overlaps(b):
                        bad.append(f"axes{i}.{axis_name}")
        plt.close(fig)
    assert not bad, f"overlapping tick labels on {sorted(set(bad))}"


def test_minor_tick_labels_are_suppressed_on_log_axes():
    """Minor decade labels would collide with the majors on a small panel."""
    p, T = _tp()
    g = plotting.build_vmr_figure(p, _vmr_cols(p))
    g.canvas.draw()
    ax = g.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels(minor=True)
            if t.get_text()] == []
    assert [t.get_text() for t in ax.get_yticklabels(minor=True)
            if t.get_text()] == []
    plt.close(g)


# ---------------------------------------------------------------------------
# Legends outside the axes
# ---------------------------------------------------------------------------

def test_summary_legends_sit_inside_their_axes_and_cover_no_data():
    """Summary figure legends are INSIDE the axes (maintainer, 2026-08-13).

    This SUPERSEDES the earlier outside-the-axes rule for the summary figure
    only -- the paired T-P/mixing-ratio panels keep their external legends
    (see test_vmr_legend_clears_the_axes_ticks_and_axis_label).

    "Inside" alone is not the requirement: the reason the legends went
    outside originally was that they landed on the data. Both panels reserve
    headroom sized from the legend's row count and pin the legend into it, so
    the real invariant is that no legend rectangle contains a plotted vertex.
    Checked across 1, 3 and 6 series, since the spectrum legend grows.
    """
    for n_pts in (1, 3, 6):
        fig = _summary_fig(n_points=n_pts)
        try:
            with plotting.render_lock:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                legends = [ax for ax in fig.axes
                           if ax.get_legend() is not None]
                # spectrum + one per posterior panel (2 in the fixture)
                assert len(legends) == 3, \
                    f"expected 3 legends, got {len(legends)}"
                for ax in legends:
                    lb = ax.get_legend().get_window_extent(renderer)
                    ab = ax.get_window_extent()
                    assert (lb.x0 >= ab.x0 - 2 and lb.x1 <= ab.x1 + 2
                            and lb.y0 >= ab.y0 - 2 and lb.y1 <= ab.y1 + 2), \
                        f"legend escaped its axes (n_pts={n_pts})"
                    covered = 0
                    for ln in ax.get_lines():
                        xd, yd = ln.get_data()
                        if len(xd) < 2:
                            continue
                        pix = ax.transData.transform(
                            np.column_stack([xd, yd]))
                        covered += int(((pix[:, 0] >= lb.x0)
                                        & (pix[:, 0] <= lb.x1)
                                        & (pix[:, 1] >= lb.y0)
                                        & (pix[:, 1] <= lb.y1)).sum())
                    assert covered == 0, \
                        f"legend covers {covered} vertices (n_pts={n_pts})"
        finally:
            plt.close(fig)


def test_vmr_legend_clears_the_axes_ticks_and_axis_label():
    """Outside the axes is not enough: the first attempt landed the legend on
    top of the x tick labels and the 'volume mixing ratio' axis label, because
    tight_layout does not reserve space for an artist anchored outside the
    axes. Check all three.
    """
    p, _ = _tp()
    g = plotting.build_vmr_figure(p, _vmr_cols(p))
    with plotting.render_lock:
        g.canvas.draw()
        renderer = g.canvas.get_renderer()
        ax = g.axes[0]
        lb = ax.get_legend().get_window_extent(renderer)
        assert not lb.overlaps(ax.get_window_extent()), "legend on the data"
        assert not lb.overlaps(ax.xaxis.label.get_window_extent(renderer)), \
            "legend on the x axis label"
        ticks = [t.label1.get_window_extent(renderer)
                 for t in ax.xaxis.get_major_ticks() if t.label1.get_text()]
        assert not any(lb.overlaps(t) for t in ticks), \
            "legend on the x tick labels"
    plt.close(g)


@pytest.mark.parametrize("which", ["tp", "vmr"])
def test_log_axes_carry_no_minor_ticks(which):
    """Over 10+ decades the 8 subdecade marks per decade merge into a solid
    black band along the spine (the house style ticks all four sides), which
    reads as a rendering artifact rather than as data."""
    p, T = _tp()
    fig = (plotting.build_tp_figure(p, T)[0] if which == "tp"
           else plotting.build_vmr_figure(p, _vmr_cols(p)))
    with plotting.render_lock:
        fig.canvas.draw()
        ax = fig.axes[0]
        assert len(ax.yaxis.get_minor_ticks()) == 0
        if which == "vmr":
            assert len(ax.xaxis.get_minor_ticks()) == 0
    plt.close(fig)


def test_legends_carry_no_title_and_no_multiline_entry():
    """No legend TITLES (maintainer, 2026-08-13): the entries carry their own
    numbers, so a title was a second caption inside the legend.

    The multi-line check stays and is the older, load-bearing half: the broken
    legend spacing originally came from folding a note into the model label,
    which made that entry multi-line. It must not come back in either form.
    """
    fig = _summary_fig()
    try:
        for i, ax in enumerate(fig.axes):
            leg = ax.get_legend()
            if leg is None:
                continue
            assert not leg.get_title().get_text(), \
                f"axes{i} legend regained a title: " \
                f"{leg.get_title().get_text()!r}"
            for txt in leg.get_texts():
                assert "\n" not in txt.get_text(), \
                    f"multi-line legend entry: {txt.get_text()!r}"
    finally:
        plt.close(fig)


def test_no_y_limit_inflation_for_legend_headroom():
    """The panels must show the data range, not a range padded to park a
    legend inside the axes."""
    src = inspect.getsource(summary_figure)
    assert "set_ylim(_y0, _y0 + (_y1 - _y0)" not in src, \
        "y-limit headroom hack is back"


def test_summary_figure_exports_png_and_pdf():
    fig = _summary_fig()
    with plotting.render_lock:
        png, pdf = io.BytesIO(), io.BytesIO()
        fig.savefig(png, format="png", bbox_inches="tight")
        fig.savefig(pdf, format="pdf", bbox_inches="tight")
    assert png.getvalue()[:4] == b"\x89PNG"
    assert pdf.getvalue()[:4] == b"%PDF"
    plt.close(fig)


def test_builders_validate_their_inputs_loudly():
    p, _ = _tp()
    with pytest.raises(ValueError, match="matching non-empty"):
        plotting.build_tp_figure(p, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="columns is empty"):
        plotting.build_vmr_figure(p, [])
    with pytest.raises(ValueError, match="does not match the pressure grid"):
        plotting.build_vmr_figure(p, [("H2O", np.array([1e-3, 1e-4]))])


def test_summary_layout_is_wide_spectrum_plus_one_panel_per_parameter():
    """Spectrum at 2x width, then ONE SQUARE PANEL PER PARAMETER.

    The single merged twin-axis box this replaced could not work: each
    parameter's grid is center +/- 5 sigma, so its axis auto-scaled to its own
    width and peak-normalizing left exactly one possible outline -- the two
    curves landed within 2e-12 px of each other, one hidden under the other.
    Separate panels give each parameter its own axis back.
    """
    fig = _summary_fig()
    try:
        fig.canvas.draw()
        assert len(fig.axes) == 3, [ax.get_ylabel() for ax in fig.axes]
        spec, p1, p2 = (ax.get_window_extent() for ax in fig.axes)
        assert abs(spec.width / p1.width - 2.0) < 0.06, (spec.width, p1.width)
        # EQUAL HEIGHT (maintainer, 2026-08-13), superseding square panels:
        # set_box_aspect(1.0) shrank each panel to its own width -- half the
        # spectrum's -- so they sat short and vertically centered against it.
        for bb in (p1, p2):
            assert abs(bb.height - spec.height) < 1.0, \
                f"panel height {bb.height:.1f} != spectrum {spec.height:.1f}"
        assert (round(p1.width, 3), round(p1.height, 3)) == \
               (round(p2.width, 3), round(p2.height, 3)), (p1, p2)
        assert abs(p1.y0 - p2.y0) < 1.0, "panels must share a baseline"
        for ax in fig.axes[1:]:
            assert ax.get_box_aspect() is None, \
                "a fixed box aspect would break the equal-height rule"
        # NOT superimposed: the whole point of the revert
        curves = [ax.get_lines()[0] for ax in fig.axes[1:]]
        px = [ax.transData.transform(
                  np.column_stack(ln.get_data()))
              for ax, ln in zip(fig.axes[1:], curves)]
        assert np.abs(px[0] - px[1]).max() > 1.0, \
            "posterior curves are pixel-identical -- the merged-box bug is back"
    finally:
        plt.close(fig)


def test_each_posterior_panel_reports_its_width():
    """The width cannot be read from a normalized curve's shape, so it is
    QUOTED in the panel title and shaded as a 1-sigma band -- how corner.py
    and published marginalized posteriors report it."""
    fig = _summary_fig(with_sigma=True)
    try:
        for ax in fig.axes[1:]:
            title = ax.get_title()
            assert "±" in title, f"panel title quotes no width: {title!r}"
            assert ax.collections, "no shaded 1-sigma band"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("x_log,y_log", [(True, False), (True, True),
                                         (False, False), (False, True)])
def test_spectrum_axis_scales_all_label_and_never_collide(x_log, y_log):
    """Both spectrum axes take log or linear (GUI checkboxes), and EVERY
    combination must produce labelled, non-overlapping ticks.

    Two defects found by rendering rather than by assertion:
    a transit depth spans ~0.02 decades, so the decade locator put all four
    of its ticks OUTSIDE the view and the log y axis drew with no labels at
    all; and the 6%-of-range padding was additive, which on a wide span
    reached below zero and tripped the positive-range guard on data that was
    entirely positive.
    """
    wl = np.linspace(0.6, 12.0, 400)
    depth = 20000.0 + 320.0 * np.sin(wl * 2.6)
    w = np.linspace(0.85, 5.18, 25)
    pts = [dict(label="PRISM: 4.2s", color="#1f4e9c", marker="^", wl_um=w,
                depth_ppm=np.interp(w, wl, depth),
                sigma_ppm=np.full(25, 55.0))]
    fig = summary_figure.compose_summary_figure(
        dict(wl_um=wl, depth_ppm=depth, depth_label="transit depth (ppm)",
             model_label="model", points=pts, x_log=x_log, y_log=y_log))
    try:
        with plotting.render_lock:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax = fig.axes[0]
            assert ax.get_xscale() == ("log" if x_log else "linear")
            assert ax.get_yscale() == ("log" if y_log else "linear")
            for axis in (ax.xaxis, ax.yaxis):
                lo, hi = (ax.get_xlim() if axis is ax.xaxis
                          else ax.get_ylim())
                shown = [t for t, v in zip(axis.get_ticklabels(),
                                           axis.get_ticklocs())
                         if lo <= v <= hi and t.get_text()]
                assert len(shown) >= 3, (
                    f"{axis.axis_name} axis labels only {len(shown)} ticks "
                    f"inside [{lo:.4g}, {hi:.4g}] "
                    f"(x_log={x_log}, y_log={y_log})")
                boxes = [t.get_window_extent(renderer) for t in shown]
                for i in range(len(boxes) - 1):
                    assert not boxes[i].overlaps(boxes[i + 1]), \
                        f"{axis.axis_name} tick labels overlap"
    finally:
        plt.close(fig)


def test_log_depth_axis_refuses_a_non_positive_range():
    """A log depth axis with a non-positive visible range fails LOUDLY rather
    than rendering an empty panel (matplotlib drops those points silently).
    Eclipse depths and low-S/N jitter draws can go negative."""
    wl = np.linspace(0.6, 12.0, 200)
    depth = np.linspace(-400.0, 900.0, 200)
    with pytest.raises(ValueError, match="positive depth range"):
        summary_figure.compose_summary_figure(
            dict(wl_um=wl, depth_ppm=depth, depth_label="eclipse depth (ppm)",
                 model_label="model", points=[], y_log=True))


@pytest.mark.parametrize("n_src", [1, 2, 4, 6])
def test_posterior_headroom_scales_with_the_legend(n_src):
    """The posterior panel's legend headroom is DERIVED from its row count,
    not a constant.

    It was a hardcoded 1.42, which only held by coincidence: the legend sits
    upper-left while the curves peak centrally. With one curve per selected
    series the legend grows with the selection, so the headroom has to grow
    with it -- and CLAUDE.md documents the row-count rule for BOTH panels.
    """
    th = np.linspace(0.6, 1.4, 200)
    palette = ["#1f4e9c", "#8c2d04", "#0f6b4f", "#6a3d9a", "#117733",
               "#444444"]
    curves = []
    for i in range(n_src):
        sig = 0.02 + 0.006 * i
        curves.append(dict(label=f"Source {i} with a longish mode name",
                           theta=th,
                           pdf=np.exp(-(th - 1.0) ** 2 / (2 * sig ** 2)),
                           mu=1.0, sigma=sig, color=palette[i],
                           ls="-", lw=1.8))
    wl = np.linspace(1.0, 12.0, 200)
    fig = summary_figure.compose_summary_figure(
        dict(wl_um=wl, depth_ppm=20000.0 + 300.0 * np.sin(wl),
             depth_label="transit depth (ppm)", model_label="model",
             points=[]),
        posterior_panels=[dict(axis_label="[M/H] [dex]", center=1.0,
                               curves=curves)])
    try:
        with plotting.render_lock:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax = fig.axes[1]
            top = ax.get_ylim()[1]
            # curves are peak-normalized to 1.0, so the top IS the headroom
            assert top > 1.0 + 0.10 * n_src - 0.02, \
                f"{n_src} sources got only {top:.2f} of headroom"
            lb = ax.get_legend().get_window_extent(renderer)
            covered = 0
            for ln in ax.get_lines():
                xd, yd = ln.get_data()
                if len(xd) < 2:
                    continue
                pix = ax.transData.transform(np.column_stack([xd, yd]))
                covered += int(((pix[:, 0] >= lb.x0) & (pix[:, 0] <= lb.x1)
                                & (pix[:, 1] >= lb.y0)
                                & (pix[:, 1] <= lb.y1)).sum())
            assert covered == 0, \
                f"legend covers {covered} vertices with {n_src} sources"
    finally:
        plt.close(fig)
