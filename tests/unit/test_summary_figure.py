"""summary_figure: the proposal summary figure (golden-ratio spectrum + up to three square posterior panels) composes headless.

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


def _panel_sized(mu=1.0, sigma=0.1):
    """A panel whose curve carries mu/sigma, which is what sets the x window."""
    pan = _panel()
    pan["curves"][0].update(mu=mu, sigma=sigma)
    return pan


def test_forecast_panel_width_follows_xlim_sigma():
    """The panel x window is +/-xlim_sigma about the curve, and nothing else.

    The curve is NOT resampled: posteriors.gaussian_curve's grid is MC-pinned
    elsewhere, so this must stay a pure window. Width scaling exactly with the
    setting is what proves that.
    """
    import matplotlib.pyplot as plt
    mu, sigma = 1.0, 0.1
    widths = {}
    for s in (2.0, 5.0):
        fig = summary_figure.compose_summary_figure(
            _spectrum(with_points=False),
            posterior_panels=[_panel_sized(mu, sigma)], xlim_sigma=s)
        try:
            lo, hi = fig.axes[1].get_xlim()
            # centered on the curve, half-width = s * sigma
            assert lo == pytest.approx(mu - s * sigma, rel=1e-12)
            assert hi == pytest.approx(mu + s * sigma, rel=1e-12)
            widths[s] = hi - lo
        finally:
            plt.close(fig)
    assert widths[5.0] == pytest.approx(2.5 * widths[2.0], rel=1e-12)


def test_default_xlim_sigma_is_unchanged_by_the_new_parameter():
    """None must reproduce the 3.5-sigma default exactly, byte for byte."""
    import matplotlib.pyplot as plt
    figs = [summary_figure.compose_summary_figure(
        _spectrum(with_points=False), posterior_panels=[_panel_sized()],
        xlim_sigma=x) for x in (None, 3.5)]
    try:
        assert figs[0].axes[1].get_xlim() == figs[1].axes[1].get_xlim()
    finally:
        for f in figs:
            plt.close(f)


def test_explicit_depth_range_is_used_verbatim():
    """An explicit depth window is never overridden -- not even for legend
    headroom, which the auto-fit path adds and this path deliberately does
    not (summary_figure._plot_spectrum). The GUI exposes this, so a silent
    inflation would mean the user's typed bounds did not appear."""
    import matplotlib.pyplot as plt
    spec = _spectrum()                      # WITH points, so a legend is drawn
    spec["depth_range"] = (20500.0, 21500.0)
    fig = summary_figure.compose_summary_figure(spec)
    try:
        assert fig.axes[0].get_ylim() == pytest.approx((20500.0, 21500.0))
    finally:
        plt.close(fig)


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
    # the cap is THREE (2026-08-13), matching the GUI's _MAX_POST_PANELS --
    # it was 2 here while the widget already allowed 3, so a three-parameter
    # selection raised instead of rendering
    with pytest.raises(ValueError, match="at most three"):
        summary_figure.compose_summary_figure(
            _spectrum(with_points=False),
            posterior_panels=[_panel() for _ in range(4)])
    # axis windows: a reversed or non-finite pair is refused, never silently
    # swapped or ignored. The GUI validates before calling, so these are the
    # backstop for API callers.
    for bad_range in ((21500.0, 20500.0), (np.nan, 1.0)):
        spec = _spectrum(with_points=False)
        spec["depth_range"] = bad_range
        with pytest.raises(ValueError, match="depth_range"):
            summary_figure.compose_summary_figure(spec)
    for bad_sigma in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="xlim_sigma"):
            summary_figure.compose_summary_figure(
                _spectrum(with_points=False), xlim_sigma=bad_sigma)
