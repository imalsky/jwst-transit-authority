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


def _summary_fig():
    wl = np.linspace(1.0, 12.0, 300)
    pts = [dict(label=f"mode {i}: {4.2 - i:.1f}s", color=c, marker=m,
                wl_um=np.linspace(1.0 + 3 * i, 4.0 + 3 * i, 25),
                depth_ppm=np.full(25, 20000.0 + 50 * i),
                sigma_ppm=np.full(25, 60.0))
           for i, (c, m) in enumerate([("#1f4e9c", "o"), ("#8c2d04", "s"),
                                       ("#0f6b4f", "^")])]
    th = np.linspace(-1.0, 1.0, 200)
    pans = [dict(axis_label="log Z",
                 curves=[dict(label="PRISM 0.044", theta=th,
                              pdf=np.exp(-th ** 2 / 0.02))], center=0.0),
            dict(axis_label="C/O",
                 curves=[dict(label="G395H 0.11", theta=th,
                              pdf=np.exp(-th ** 2 / 0.08))], center=0.0)]
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

def test_tp_and_vmr_panels_are_square():
    p, T = _tp()
    fig, ylim = plotting.build_tp_figure(p, T)
    assert fig.axes[0].get_box_aspect() == 1.0
    w, h = fig.get_size_inches()
    assert w == h, (w, h)
    plt.close(fig)
    g = plotting.build_vmr_figure(p, _vmr_cols(p), ylim=ylim)
    assert g.axes[0].get_box_aspect() == 1.0
    assert g.get_size_inches()[0] == g.get_size_inches()[1]
    plt.close(g)


def test_summary_panels_stay_square():
    fig = _summary_fig()
    assert [ax.get_box_aspect() for ax in fig.axes] == [1.0, 1.0, 1.0]
    plt.close(fig)


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

def test_every_legend_sits_outside_its_axes_and_inside_the_figure():
    """The invariant that replaced the y-limit headroom hack.

    Data are clipped to their axes, so 'legend bbox outside the axes bbox'
    IS the no-data-overlap rule; 'inside the figure bbox' catches clipping.
    """
    fig = _summary_fig()
    with plotting.render_lock:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = []
        for i, ax in enumerate(fig.axes):
            leg = ax.get_legend()
            assert leg is not None, f"axes{i} lost its legend"
            lb = leg.get_window_extent(renderer)
            ab = ax.get_window_extent()
            assert not lb.overlaps(ab), f"axes{i} legend overlaps its axes"
            fb = fig.bbox
            assert (lb.x0 >= fb.x0 - 1 and lb.x1 <= fb.x1 + 1
                    and lb.y0 >= fb.y0 - 1 and lb.y1 <= fb.y1 + 1), \
                f"axes{i} legend is clipped by the figure edge"
            boxes.append(lb)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not boxes[i].overlaps(boxes[j]), \
                    f"legends {i} and {j} overlap"
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


def test_legend_note_is_a_title_not_a_multiline_label():
    """The broken legend spacing came from folding the note into the model
    label, which made that entry multi-line. It belongs in the title."""
    fig = _summary_fig()
    leg = fig.axes[0].get_legend()
    assert leg.get_title().get_text() == "SO2 S/N per mode, 1 transit"
    for txt in leg.get_texts():
        assert "\n" not in txt.get_text(), \
            f"multi-line legend entry: {txt.get_text()!r}"
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
