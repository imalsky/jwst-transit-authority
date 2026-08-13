"""summary_figure: the proposal summary figure (spectrum + up to two posterior panels) composes headless.

Pure matplotlib (Agg): no Streamlit, no engine, no noise backend. Pins the
loud input validation, the no-silent-empty-panel rule, and that the figure
saves to both PNG and vector PDF.
"""
import io

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from jwst_tool import summary_figure


def _spectrum(n=60, with_points=True):
    wl = np.linspace(1.0, 5.0, n)
    spec = dict(wl_um=wl,
                depth_ppm=21000.0 + 100.0 * np.sin(wl),
                depth_label="transit depth (ppm)",
                model_label="model")
    if with_points:
        pw = np.linspace(2.9, 5.0, 12)
        spec["points"] = [dict(label="NIRSpec G395H", color="#199e70",
                               marker="s", wl_um=pw,
                               depth_ppm=np.full(12, 21000.0),
                               sigma_ppm=np.full(12, 120.0))]
    return spec


def _panel():
    theta = np.linspace(0.5, 1.5, 101)
    pdf = np.exp(-0.5 * ((theta - 1.0) / 0.1) ** 2)
    return dict(axis_label="[M/H] [dex]",
                curves=[dict(label="best", theta=theta, pdf=pdf,
                             color="#2a78d6")],
                notes=[], center=1.0)


def test_full_figure_composes_and_exports_png_and_pdf():
    fig = summary_figure.compose_summary_figure(
        _spectrum(), posterior_panels=[_panel(), _panel()],
        title="WASP-39 b -- transmission forecast",
        footnote="Linearized Fisher (Cramer-Rao) forecast; not a sampled "
                 "posterior.")
    try:
        for fmt in ("png", "pdf"):
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt)
            assert buf.getbuffer().nbytes > 1000, fmt
        # 1 spectrum + 2 posterior panels
        assert len(fig.axes) == 3
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_minimal_figure_spectrum_only():
    fig = summary_figure.compose_summary_figure(_spectrum(with_points=False))
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        assert buf.getbuffer().nbytes > 1000
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_unconstrained_panel_renders_note_without_curve():
    pan = dict(axis_label="C/O", curves=[],
               notes=["G395H: unconstrained -- no curve, by design"],
               center=None)
    fig = summary_figure.compose_summary_figure(
        _spectrum(with_points=False), posterior_panels=[pan])
    try:
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any("unconstrained" in t for t in texts)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_inputs_are_never_mutated():
    spec = _spectrum()
    wl_before = spec["wl_um"].copy()
    d_before = spec["depth_ppm"].copy()
    fig = summary_figure.compose_summary_figure(spec)
    import matplotlib.pyplot as plt
    plt.close(fig)
    assert np.array_equal(spec["wl_um"], wl_before)
    assert np.array_equal(spec["depth_ppm"], d_before)


def test_validation_is_loud():
    with pytest.raises(ValueError, match="wl_um"):
        summary_figure.compose_summary_figure(dict(depth_ppm=[1.0, 2.0]))
    bad = _spectrum()
    bad["depth_ppm"] = bad["depth_ppm"][:-1]
    with pytest.raises(ValueError, match="shapes differ"):
        summary_figure.compose_summary_figure(bad)
    nonfinite = _spectrum()
    nonfinite["depth_ppm"] = nonfinite["depth_ppm"].copy()
    nonfinite["depth_ppm"][3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        summary_figure.compose_summary_figure(nonfinite)
    # an empty posterior panel says nothing -- refused, never silent
    with pytest.raises(ValueError, match="curves or notes"):
        summary_figure.compose_summary_figure(
            _spectrum(with_points=False),
            posterior_panels=[dict(axis_label="x", curves=[], notes=[])])
    with pytest.raises(ValueError, match="at most two"):
        summary_figure.compose_summary_figure(
            _spectrum(with_points=False),
            posterior_panels=[_panel(), _panel(), _panel()])
