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
import matplotlib.pyplot as plt      # noqa: E402

from jwst_tool import summary_figure  # noqa: E402


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


def test_panel_xlim_verbatim_when_given_automatic_otherwise():
    """An explicit panel window is applied exactly as typed, and only to the
    panel it belongs to; None, [] and a short list all keep the automatic
    +/- N sigma window.

    The curve is NOT resampled: posteriors.gaussian_curve's grid is MC-pinned
    elsewhere, so this must stay a pure window -- the same curve, a different
    frame. (There is no sigma-multiple slider: the axis
    controls are typed min/max numbers, so the panel window is an absolute
    pair in the parameter's own units, not a width.)
    """
    mu, sigma = 1.0, 0.1
    auto_want = (mu - summary_figure._XLIM_SIGMA * sigma,
                 mu + summary_figure._XLIM_SIGMA * sigma)
    fig = summary_figure.compose_summary_figure(
        _spectrum(with_points=False),
        posterior_panels=[_panel_sized(mu, sigma), _panel_sized(mu, sigma)],
        panel_xlims=[(0.42, 1.77), None])
    try:
        assert fig.axes[1].get_xlim() == pytest.approx((0.42, 1.77), rel=1e-12)
        # the second panel got None and keeps the automatic window
        assert fig.axes[2].get_xlim() == pytest.approx(auto_want, rel=1e-12)
    finally:
        plt.close(fig)
    # absent forms must all leave the panel automatic
    for xlims in (None, [], [None]):
        fig = summary_figure.compose_summary_figure(
            _spectrum(with_points=False),
            posterior_panels=[_panel_sized(mu, sigma)], panel_xlims=xlims)
        try:
            assert fig.axes[1].get_xlim() == pytest.approx(
                auto_want, rel=1e-12), f"panel_xlims={xlims!r}"
        finally:
            plt.close(fig)


def test_truncated_co_panel_window_clamps_at_zero():
    """A zero-clipped (positive-quantity) curve keeps the symmetric
    mu +/- 3.5 sigma window on the right but never extends the axis below
    zero; a plain Gaussian panel of the same width does go negative."""
    from jwst_tool import posteriors
    center, sigma = 0.55, 0.479          # weakly constrained C/O
    curve = posteriors.truncated_gaussian_curve(center, sigma)
    pan = dict(axis_label="C/O", notes=[], center=center,
               curves=[dict(label="one draw", theta=curve["theta"],
                            pdf=curve["pdf"], mu=center, sigma=sigma,
                            curve_family="gaussian_truncated_positive",
                            color="#2a78d6")])
    fig = summary_figure.compose_summary_figure(
        _spectrum(), posterior_panels=[pan])
    try:
        lo, hi = fig.axes[1].get_xlim()
        assert lo >= 0.0
        assert hi == pytest.approx(center + summary_figure._XLIM_SIGMA * sigma,
                                   rel=1e-6)
        assert np.all(np.asarray(curve["theta"]) >= 0.0)
        # the same width through the plain Gaussian branch DOES go negative,
        # which is exactly what the clipped family exists to avoid
        gfig = summary_figure.compose_summary_figure(
            _spectrum(),
            posterior_panels=[_panel_sized(mu=center, sigma=sigma)])
        try:
            assert gfig.axes[1].get_xlim()[0] < 0.0
        finally:
            plt.close(gfig)
    finally:
        plt.close(fig)


def test_explicit_depth_range_is_used_verbatim():
    """An explicit depth window is never overridden -- not even for legend
    headroom, which the auto-fit path adds and this path deliberately does
    not (summary_figure._plot_spectrum). The GUI exposes this, so a silent
    inflation would mean the user's typed bounds did not appear."""
    spec = _spectrum()                      # WITH points, so a legend is drawn
    spec["depth_range"] = (20500.0, 21500.0)
    fig = summary_figure.compose_summary_figure(spec)
    try:
        assert fig.axes[0].get_ylim() == pytest.approx((20500.0, 21500.0))
    finally:
        plt.close(fig)


def test_figure_composes_exports_and_never_mutates_inputs():
    """The full spectrum+panels figure and the minimal spectrum-only figure
    both compose and export (PNG and vector PDF), and the caller's arrays
    come back untouched."""
    spec = _spectrum()
    wl_before = spec["wl_um"].copy()
    d_before = spec["depth_ppm"].copy()
    fig = summary_figure.compose_summary_figure(
        spec, posterior_panels=[_panel(), _panel()],
        title="WASP-39 b -- transmission forecast",
        footnote="Linearized Fisher (Cramer-Rao) forecast; not a sampled "
                 "posterior.")
    try:
        for fmt, magic in (("png", b"\x89PNG"), ("pdf", b"%PDF")):
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt)
            assert buf.getbuffer().nbytes > 1000, fmt
            assert buf.getvalue()[:4] == magic, fmt
        # 1 spectrum + 2 posterior panels
        assert len(fig.axes) == 3
    finally:
        plt.close(fig)
    assert np.array_equal(spec["wl_um"], wl_before)
    assert np.array_equal(spec["depth_ppm"], d_before)
    # spectrum-only still composes
    fig = summary_figure.compose_summary_figure(_spectrum(with_points=False))
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        assert buf.getbuffer().nbytes > 1000
    finally:
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
        plt.close(fig)


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
    # the cap is THREE, matching the GUI's _MAX_POST_PANELS: a lower cap
    # here makes a selection the widget allows raise instead of rendering
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
    for bad_pair in ((1.0, 0.0), (np.nan, 1.0), (1.0,), 3.5):
        with pytest.raises(ValueError, match="panel_xlims"):
            summary_figure.compose_summary_figure(
                _spectrum(with_points=False),
                posterior_panels=[_panel_sized()], panel_xlims=[bad_pair])
