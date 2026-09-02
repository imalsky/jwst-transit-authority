"""Pure figure builders: the render lock, square panels, and tick hygiene.

Numpy + matplotlib only (no streamlit, no engine), so this stays in the fast
suite. These tests drive the SAME builders the GUI draws with -- a parallel
reimplementation here would prove nothing.
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
import numpy as np
from matplotlib.ticker import LogLocator                   # noqa: E402
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
    """Composed summary figure. ``n_points`` = mode series count (grows the
    spectrum legend); ``with_sigma`` attaches mu/sigma to each posterior
    curve, which is what makes the panel quote its width."""
    wl = np.linspace(1.0, 12.0, 300)
    pal = ["#1f4e9c", "#8c2d04", "#0f6b4f", "#6a3d9a", "#117733", "#882255"]
    pts = [dict(label=f"mode {i}: {4.2 - i * 0.4:.1f}s", color=pal[i % 6],
                marker="^", wl_um=np.linspace(1.0 + 1.5 * i, 4.0 + 1.5 * i, 25),
                depth_ppm=np.full(25, 20000.0 + 50 * i),
                sigma_ppm=np.full(25, 60.0))
           for i in range(n_points)]
    th = np.linspace(-1.0, 1.0, 200)
    e1 = dict(mu=0.02, sigma=0.1) if with_sigma else {}
    e2 = dict(mu=-0.01, sigma=0.2) if with_sigma else {}
    pans = [dict(axis_label="log Z", center=0.0,
                 curves=[dict(label="PRISM 0.044", theta=th,
                              pdf=np.exp(-th ** 2 / 0.02), **e1)]),
            dict(axis_label="C/O", center=0.0,
                 curves=[dict(label="G395H 0.11", theta=th,
                              pdf=np.exp(-th ** 2 / 0.08), **e2)])]
    spec = dict(wl_um=wl, depth_ppm=20000.0 + 300.0 * np.sin(wl * 3),
                depth_label="transit depth (ppm)",
                model_label="model (smoothed for display)",
                legend_title="SO2 S/N per mode, 1 transit", points=pts)
    return summary_figure.compose_summary_figure(
        spec, posterior_panels=pans, title="test",
        footnote="one seeded realization")


def _vertices_inside(ax, bbox):
    """Count plotted line vertices whose display coordinates fall in bbox
    (the legend-covers-no-data invariant)."""
    n = 0
    for ln in ax.get_lines():
        xd, yd = ln.get_data()
        if len(xd) < 2:
            continue
        pix = ax.transData.transform(np.column_stack([xd, yd]))
        n += int(((pix[:, 0] >= bbox.x0) & (pix[:, 0] <= bbox.x1)
                  & (pix[:, 1] >= bbox.y0) & (pix[:, 1] <= bbox.y1)).sum())
    return n


def _visible_major_label_bboxes(ax, renderer):
    out = {}
    for axis in (ax.xaxis, ax.yaxis):
        out[axis.axis_name] = [
            t.label1.get_window_extent(renderer)
            for t in axis.get_major_ticks()
            if t.label1.get_visible() and t.label1.get_text()]
    return out


def _assert_no_overlaps(boxes, what):
    """No two adjacent visible major tick labels may intersect."""
    for u, v in zip(boxes, boxes[1:]):
        assert not u.overlaps(v), f"{what}: tick labels overlap"



# The render lock: mathtext ParseException under concurrent renders

def test_concurrent_renders_do_not_raise():
    """Eight threads through the real builders: no mathtext ParseException.

    matplotlib's process-global mathtext parser is locked only during
    Figure.draw; tight_layout measures tick labels outside that lock, and on
    a log axis those labels are mathtext. Streamlit runs each session on its
    own thread, so two users rendering at once raced into the parser and the
    loser got 'ValueError: ParseException'. Measured before the fix: 7/8
    threads raised on the deployed pin (matplotlib 3.10.0), 8/8 on 3.11.0.
    """
    p, T = _tp()
    cols = _vmr_cols(p)
    errors: list[str] = []

    def work():
        try:
            for _ in range(10):
                with plotting.render_lock:
                    fig = plotting.build_structure_figure(p, T, cols)
                    fig.savefig(io.BytesIO(), format="png")
                    plt.close(fig)
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


def test_builders_block_while_the_render_lock_is_held():
    """BEHAVIORAL: each builder actually ACQUIRES the lock. The threaded
    test above serializes externally, so it cannot catch a builder that
    stops locking internally; this one holds the lock in another thread and
    asserts the builder cannot finish, whatever matplotlib does."""
    p, T = _tp()
    cols = _vmr_cols(p)
    calls = {"build_structure_figure":
             lambda: plotting.build_structure_figure(p, T, cols)}
    for builder, call in calls.items():
        held, release, finished = (threading.Event() for _ in range(3))
        box = {}

        def holder():
            with plotting.render_lock:
                held.set()
                release.wait(10.0)

        def builder_thread(call=call):
            box["fig"] = call()
            finished.set()

        h = threading.Thread(target=holder)
        h.start()
        assert held.wait(10.0), "holder thread never acquired the lock"
        b = threading.Thread(target=builder_thread)
        b.start()
        try:
            assert not finished.wait(1.5), \
                f"{builder} completed while render_lock was held elsewhere"
        finally:
            release.set()
            h.join(10.0)
            b.join(10.0)
        assert finished.is_set(), f"{builder} never completed after release"
        plt.close(box["fig"])


def test_structure_panels_share_one_pressure_axis_drawn_on_both():
    """The two structure panels lock their pressure axis together (sharey),
    but the axis must be VISIBLE on both: plt.subplots hides the right
    panel's tick labels, which left the mixing-ratio panel floating with no
    pressure scale across the gap between the panels."""
    p, T = _tp()
    fig = plotting.build_structure_figure(p, T, _vmr_cols(p))
    try:
        ax_t, ax_v = fig.axes[:2]
        assert ax_t.get_shared_y_axes().joined(ax_t, ax_v)
        lo, hi = ax_t.get_ylim()
        assert lo > hi, "pressure must increase downward"
        for ax in (ax_t, ax_v):
            assert ax.yaxis.get_tick_params()["labelleft"], \
                "pressure tick labels hidden on a panel"
        # ONE axis label, at the far left -- never between the panels
        assert ax_t.get_ylabel() == "pressure (bar)", ax_t.get_ylabel()
        assert ax_v.get_ylabel() == "", ax_v.get_ylabel()
        # T-P in black; every species a distinct colour, and the species the
        # research note draws in the SAME colour as its figure
        assert ax_t.get_lines()[0].get_color() == plotting.TP_COLOR
        colors = {ln.get_label(): ln.get_color() for ln in ax_v.get_lines()}
        assert len(set(colors.values())) == len(colors), colors
        for name, c in plotting.VMR_COLORS.items():
            if name in colors:
                assert colors[name].upper() == c.upper(), (name, colors[name])
    finally:
        plt.close(fig)


def test_app_materializes_figures_only_through_locked_helpers():
    """STRUCTURAL: no unlocked layout/export/render call survives in app.py.

    app.py must go through _fig_bytes / _show_fig, which hold the lock; a
    bare fig.tight_layout(), fig.savefig() or st.pyplot() would reintroduce
    the crash, so parse the module and fail on one.
    compose_summary_figure serializes too (it mutates global rcParams via
    plt.style.context and lays out mathtext axes): its source must enter the
    lock no later than the style context.
    """
    src = APP.read_text()
    tree = ast.parse(src)
    locked_helpers = {"_fig_bytes", "_show_fig"}
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
        if node.func.attr in ("tight_layout", "savefig", "pyplot") \
                and node.lineno not in exempt_lines:
            offenders.append(f"line {node.lineno}: .{node.func.attr}()")
    assert not offenders, (
        "unlocked figure layout/export/render in app.py -- route it through "
        f"{sorted(locked_helpers)}: {offenders}")
    ssrc = inspect.getsource(summary_figure.compose_summary_figure)
    assert "plotting.render_lock" in ssrc, \
        "compose_summary_figure no longer holds plotting.render_lock"
    assert ssrc.index("render_lock") < ssrc.index("style.context"), \
        "the lock must be entered no later than the style context"


def test_summary_legends_sit_inside_their_axes_and_cover_no_data():
    """Summary-figure legends sit INSIDE the axes (maintainer; the paired
    T-P/mixing-ratio panels keep external legends). "Inside" alone is not
    the requirement -- a legend that lands on the data is the failure -- so
    the real invariant is that no legend rectangle contains a plotted
    vertex, checked across 1/3/6 series since the spectrum legend grows
    (maintainer rule).

    Also pinned on the same figures: no legend TITLES (entries carry their
    own numbers) and no multi-line entries (folding a note into the model
    label breaks the row spacing); no overlapping tick labels; width QUOTED
    as a number with no shaded band (overlapping fills muddy the panel);
    headroom DERIVED from the legend row count, never a constant (which
    holds only by coincidence); and no y-limit inflation hack in the source.
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
                    leg = ax.get_legend()
                    lb = leg.get_window_extent(renderer)
                    ab = ax.get_window_extent()
                    assert (lb.x0 >= ab.x0 - 2 and lb.x1 <= ab.x1 + 2
                            and lb.y0 >= ab.y0 - 2 and lb.y1 <= ab.y1 + 2), \
                        f"legend escaped its axes (n_pts={n_pts})"
                    covered = _vertices_inside(ax, lb)
                    assert covered == 0, \
                        f"legend covers {covered} vertices (n_pts={n_pts})"
                    assert not leg.get_title().get_text(), \
                        f"legend regained a title: " \
                        f"{leg.get_title().get_text()!r}"
                    for txt in leg.get_texts():
                        assert "\n" not in txt.get_text(), \
                            f"multi-line legend entry: {txt.get_text()!r}"
                if n_pts == 3:
                    for i, ax in enumerate(fig.axes):
                        for axis_name, boxes in _visible_major_label_bboxes(
                                ax, renderer).items():
                            _assert_no_overlaps(boxes, f"axes{i}.{axis_name}")
        finally:
            plt.close(fig)
    assert "set_ylim(_y0, _y0 + (_y1 - _y0)" not in \
        inspect.getsource(summary_figure), "y-limit headroom hack is back"

    # width quoted, band absent (a normalized curve's shape cannot show it)
    fig = _summary_fig(with_sigma=True)
    try:
        for i, ax in enumerate(fig.axes[1:], 1):
            assert "±" in ax.get_title(), \
                f"panel title quotes no width: {ax.get_title()!r}"
            assert not ax.collections, \
                f"axes{i} regained a shaded band ({len(ax.collections)})"
    finally:
        plt.close(fig)

    # headroom derived from the row count: 1 and 6 sources, the range ends
    th = np.linspace(0.6, 1.4, 200)
    pal = ["#1f4e9c", "#8c2d04", "#0f6b4f", "#6a3d9a", "#117733", "#444444"]
    wl = np.linspace(1.0, 12.0, 200)
    for n_src in (1, 6):
        curves = [dict(label=f"Source {i} with a longish mode name",
                       theta=th,
                       pdf=np.exp(-(th - 1.0) ** 2
                                  / (2 * (0.02 + 0.006 * i) ** 2)),
                       mu=1.0, sigma=0.02 + 0.006 * i, color=pal[i],
                       ls="-", lw=1.8)
                  for i in range(n_src)]
        fig = summary_figure.compose_summary_figure(
            dict(wl_um=wl, depth_ppm=20000.0 + 300.0 * np.sin(wl),
                 depth_label="transit depth (ppm)", model_label="model",
                 points=[]),
            posterior_panels=[dict(axis_label="[M/H] [dex]", center=1.0,
                                   curves=curves)])
        try:
            with plotting.render_lock:
                fig.canvas.draw()
                ax = fig.axes[1]
                # curves are peak-normalized to 1.0: the top IS the headroom
                top = ax.get_ylim()[1]
                assert top > 1.0 + 0.10 * n_src - 0.02, \
                    f"{n_src} sources got only {top:.2f} of headroom"
                lb = ax.get_legend().get_window_extent(
                    fig.canvas.get_renderer())
                covered = _vertices_inside(ax, lb)
                assert covered == 0, \
                    f"legend covers {covered} vertices with {n_src} sources"
        finally:
            plt.close(fig)


def test_a_fitted_panel_is_not_labelled_a_forecast():
    """A panel whose curves are FITS to the jitter draw must not call itself
    a forecast (external review). A Fisher forecast is centered on the input
    by construction, so a curve 2.7 sigma away under a "forecast density"
    axis reads as a bug; it is not one, so the axis must name which of the
    two it is."""
    from jwst_tool import posteriors
    wl = np.linspace(0.6, 12.0, 200)
    base = dict(wl_um=wl, depth_ppm=20000.0 + 300.0 * np.sin(wl),
                depth_label="transit depth (ppm)", model_label="model",
                points=[])

    def _panel(kind):
        c = posteriors.gaussian_curve(1.75 if kind else 1.0, 0.28)
        return dict(axis_label="[M/H] [dex]", center=1.0,
                    density_label=("relative density" if kind
                                   else "relative forecast density"),
                    curves=[dict(label="PRISM", theta=c["theta"],
                                 pdf=c["pdf"], mu=1.75 if kind else 1.0,
                                 sigma=0.28, color="#1f4e9c", ls="-", lw=1.8,
                                 kind=kind)])

    fig = summary_figure.compose_summary_figure(
        base, posterior_panels=[_panel(posteriors.MOCK_RECOVERY_KIND),
                                _panel(None)])
    try:
        fitted, forecast = fig.axes[1].get_ylabel(), fig.axes[2].get_ylabel()
        assert fitted == "relative density", fitted
        assert "forecast" not in fitted, \
            f"a fitted panel still calls itself a forecast: {fitted!r}"
        assert forecast == "relative forecast density", forecast
    finally:
        plt.close(fig)


def _log_spec(depth, points=()):
    wl = np.linspace(0.6, 12.0, 200)
    return dict(wl_um=wl, depth_ppm=depth, depth_label="eclipse depth (ppm)",
                model_label="model", points=list(points), y_log=True)


def _log_points(depth, sigma):
    w = np.linspace(0.85, 5.2, 25)
    return [dict(label="PRISM: 1.4s", color="#1f4e9c", marker="o", wl_um=w,
                 depth_ppm=depth, sigma_ppm=np.full(25, float(sigma)))]


def test_log_depth_axis_refusals_whisker_clip_and_tick_spacing():
    """Three pins on the y_log depth axis.

    1. Non-positive DEPTHS fail LOUDLY rather than rendering a partial curve
       (matplotlib drops the non-positive points silently); filtering to the
       positives renders a -400..900 ppm model as its positive 69% with no
       indication -- never do that.
    2. A negative WHISKER must not refuse the axis (maintainer-reported:
       "Log depth axis unavailable ... starts at -288.9 ppm").
       50 +/- 340 ppm is ordinary at low S/N; the whisker is clipped at the
       spine and the limits follow the positive DEPTHS.
    3. Ticks come from a subdivided LogLocator, never a linear locator: a
       MaxNLocator branch once placed LINEARLY spaced ticks below 1.5
       decades (equal steps at unequal distances misread on a log axis).
       subs=(1, 2, 5) covers both the narrow and the wide regime.
    """
    wl = np.linspace(0.6, 12.0, 200)      # must match _log_spec's grid
    for name, depth, points in (
            ("mixed-sign model", np.linspace(-400.0, 900.0, 200), ()),
            ("all-negative model", np.linspace(-900.0, -100.0, 200), ()),
            ("mixed-sign point centers", 50.0 + 900.0 * (wl / 12.0) ** 3,
             _log_points(np.linspace(-200.0, 600.0, 25), 20.0)),
    ):
        try:
            summary_figure.compose_summary_figure(_log_spec(depth, points))
        except ValueError as e:
            assert "positive depth range" in str(e), name
        else:
            pytest.fail(f"{name}: non-positive depths did not raise")

    w = np.linspace(0.85, 5.2, 25)
    depth = 50.0 + 900.0 * (wl / 12.0) ** 3
    pts = _log_points(50.0 + 900.0 * (w / 12.0) ** 3, 340.0)
    assert (pts[0]["depth_ppm"] - pts[0]["sigma_ppm"]).min() < 0.0, \
        "fixture no longer exercises a negative whisker"
    fig = summary_figure.compose_summary_figure(_log_spec(depth, pts))
    try:
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"
        y0, y1 = ax.get_ylim()
        assert y0 > 0.0, y0
        # the limits follow the positive DEPTHS, not the clipped whisker
        assert y0 < depth.min() and y1 > depth.max(), (y0, y1)
        with plotting.render_lock:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            shown = [t.label1 for t in ax.yaxis.get_major_ticks()
                     if t.label1.get_visible() and t.label1.get_text()
                     and y0 <= t.label1.get_position()[1] <= y1]
            assert len(shown) >= 4, \
                f"only {len(shown)} y labels on a 1.4-decade log axis"
            _assert_no_overlaps(
                [t.get_window_extent(renderer) for t in shown], "log y")
    finally:
        plt.close(fig)

    for name, depth in (("narrow transit",
                         20000.0 + 320.0 * np.sin(wl * 2.6)),
                        ("mid span", 300.0 + 600.0 * (wl / 12.0)),
                        ("wide emission", 50.0 + 900.0 * (wl / 12.0) ** 3)):
        fig = summary_figure.compose_summary_figure(_log_spec(depth))
        try:
            ax = fig.axes[0]
            assert isinstance(ax.yaxis.get_major_locator(), LogLocator), name
            y0, y1 = ax.get_ylim()
            with plotting.render_lock:
                fig.canvas.draw()
                n = len([t for t in ax.yaxis.get_majorticklocs()
                         if y0 <= t <= y1])
            assert n >= 4, f"{name}: only {n} ticks inside the view"
        finally:
            plt.close(fig)
