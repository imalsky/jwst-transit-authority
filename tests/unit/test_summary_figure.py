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

from jwst_tool import posteriors, summary_figure  # noqa: E402


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


def test_ln_gaussian_co_panel_window_is_multiplicative():
    """An ln_gaussian (multiplicative-width) curve takes its automatic window
    in ln theta, mu * exp(+-3.5 sigma_ln), so the axis stays positive on the
    left and reaches further on the right than a symmetric window would; a
    plain Gaussian panel of the same width does go negative."""
    from jwst_tool import posteriors
    center, sigma = 0.55, 0.479          # weakly constrained C/O
    curve = posteriors.ln_gaussian_curve(center, sigma / center)
    pan = dict(axis_label="C/O", notes=[], center=center,
               curves=[dict(label="fitted", theta=curve["theta"],
                            pdf=curve["pdf"], mu=center, sigma=sigma,
                            sigma_ln=sigma / center,
                            curve_family="ln_gaussian",
                            color="#2a78d6")])
    fig = summary_figure.compose_summary_figure(
        _spectrum(), posterior_panels=[pan])
    try:
        lo, hi = fig.axes[1].get_xlim()
        s = summary_figure._XLIM_SIGMA * sigma / center
        assert lo == pytest.approx(center * np.exp(-s), rel=1e-6)
        assert hi == pytest.approx(center * np.exp(s), rel=1e-6)
        assert lo > 0.0
        assert np.all(np.asarray(curve["theta"]) > 0.0)
        # the same width through the plain Gaussian branch DOES go negative,
        # which is exactly what the ln_gaussian family exists to avoid
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
        spec, posterior_panels=[_panel(), _panel()])
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


def test_unconstrained_panel_draws_no_curve_and_no_text():
    """An unconstrained direction renders an EMPTY panel: the figure never
    carries prose (the page says it, under the figure), and a caller's
    ``notes`` key cannot put any there."""
    pan = dict(axis_label="C/O", curves=[],
               notes=["G395H: unconstrained -- no curve, by design"],
               center=None)
    fig = summary_figure.compose_summary_figure(
        _spectrum(with_points=False), posterior_panels=[pan])
    try:
        texts = [t.get_text() for ax in fig.axes for t in ax.texts
                 if t.get_text().strip()]
        assert texts == []
    finally:
        plt.close(fig)


def test_validation_is_loud():
    """Every refusal carries a DISTINGUISHING token: these are all the same
    ValueError from the same callable, so a bare `raises` would let a deleted
    validator pass by hiding behind another one's message."""
    with pytest.raises(ValueError, match="missing required entry 'wl_um'"):
        summary_figure.compose_summary_figure(dict(depth_ppm=[1.0, 2.0]))
    bad = _spectrum()
    bad["depth_ppm"] = bad["depth_ppm"][:-1]
    with pytest.raises(ValueError, match="shapes differ"):
        summary_figure.compose_summary_figure(bad)
    nonfinite = _spectrum()
    nonfinite["depth_ppm"] = nonfinite["depth_ppm"].copy()
    nonfinite["depth_ppm"][3] = np.nan
    with pytest.raises(ValueError, match="depth_ppm contains non-finite"):
        summary_figure.compose_summary_figure(nonfinite)
    # one cap, exported for the GUI multiselect: if the two drifted, a
    # selection the widget allows would raise here instead of rendering
    assert summary_figure.MAX_POST_PANELS == 3
    with pytest.raises(ValueError, match="at most 3 posterior panels"):
        summary_figure.compose_summary_figure(
            _spectrum(with_points=False),
            posterior_panels=[_panel() for _ in range(4)])
    # axis windows: a reversed or non-finite pair is refused, never silently
    # swapped or ignored. The GUI validates before calling, so these are the
    # backstop for API callers.
    for bad_range, msg in (((21500.0, 20500.0), "depth_range needs lo < hi"),
                           ((np.nan, 1.0), "depth_range must be finite")):
        spec = _spectrum(with_points=False)
        spec["depth_range"] = bad_range
        with pytest.raises(ValueError, match=msg):
            summary_figure.compose_summary_figure(spec)
    for bad_pair, msg in (((1.0, 0.0), "needs finite lo < hi"),
                          ((np.nan, 1.0), "needs finite lo < hi"),
                          ((1.0,), r"must be a \(lo, hi\) pair"),
                          (3.5, r"must be a \(lo, hi\) pair")):
        with pytest.raises(ValueError, match=msg):
            summary_figure.compose_summary_figure(
                _spectrum(with_points=False),
                posterior_panels=[_panel_sized()], panel_xlims=[bad_pair])


def test_ln_gaussian_panel_is_log_scaled_with_physical_ticks():
    """Physical C/O values on a LOG axis (house style: never plot logged
    values on a linear axis, never 'ln' in a label), readable at both a
    narrow and a wide width. The y label names the measure we chose."""
    from jwst_tool import posteriors
    for sigma_ln, want in ((0.0745, ["0.5", "0.6", "0.7"]),
                           (0.9218, ["0.1", "1", "10"])):
        c = posteriors.ln_gaussian_curve(0.55, sigma_ln)
        pan = dict(axis_label="C/O", center=0.55, notes=[],
                   density_label="relative forecast density per d ln(C/O)",
                   curves=[dict(label="m", theta=c["theta"], pdf=c["pdf"],
                                mu=0.55, sigma=0.55 * sigma_ln,
                                sigma_ln=sigma_ln, curve_family="ln_gaussian",
                                color="#333333")])
        fig = summary_figure.compose_summary_figure(_spectrum(),
                                                    posterior_panels=[pan])
        try:
            ax = fig.axes[1]
            assert ax.get_xscale() == "log"
            lo, hi = ax.get_xlim()
            got = [t.get_text() for t, v in zip(ax.get_xticklabels(),
                                                ax.get_xticks())
                   if lo <= v <= hi and t.get_text()]
            assert got == want, got
            assert "ln" not in ax.get_xlabel()
            assert ax.get_ylabel() == "relative forecast density per d ln(C/O)"
        finally:
            plt.close(fig)


def test_window_uses_the_curves_own_ln_width():
    """mu * exp(+-3.5*sigma_ln) at the width the curve was BUILT with -- not
    sigma/mu, which equals it only when the mock draw left mu unmoved."""
    from jwst_tool import posteriors
    center, sigma_ln = 0.55, 0.9218
    mu = center * np.exp(1.4)                  # a draw that shifted C/O up
    c = posteriors.ln_gaussian_curve(mu, sigma_ln)
    pan = dict(axis_label="C/O", center=center, notes=[],
               curves=[dict(label="m", theta=c["theta"], pdf=c["pdf"], mu=mu,
                            sigma=center * sigma_ln, sigma_ln=sigma_ln,
                            curve_family="ln_gaussian", color="#333333")])
    fig = summary_figure.compose_summary_figure(_spectrum(),
                                                posterior_panels=[pan])
    try:
        lo, hi = fig.axes[1].get_xlim()
        s = summary_figure._XLIM_SIGMA * sigma_ln
        assert hi == pytest.approx(mu * np.exp(s), rel=1e-9)
        assert lo == pytest.approx(min(mu * np.exp(-s), center), rel=1e-9)
    finally:
        plt.close(fig)


def test_panel_renders_caller_supplied_width_text():
    """The C/O width string is built once in fisher.format_co_width and
    passed in; summary_figure stays parameter-agnostic. A curve without
    width_text keeps the plain +-sigma."""
    from jwst_tool import posteriors
    c = posteriors.ln_gaussian_curve(0.55, 0.0745)
    base = dict(theta=c["theta"], pdf=c["pdf"], mu=0.55, sigma=0.041,
                sigma_ln=0.0745, curve_family="ln_gaussian")
    pan = dict(axis_label="C/O", center=0.55, notes=[], curves=[
        dict(base, label="A", color="#333333",
             width_text="\u00b10.0324 dex (C/O 0.511\u20130.593)"),
        dict(base, label="B", color="#1f4e9c",
             width_text="\u00b10.4 dex (local)")])
    fig = summary_figure.compose_summary_figure(_spectrum(),
                                                posterior_panels=[pan])
    try:
        labs = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
        assert "A: \u00b10.0324 dex (C/O 0.511\u20130.593)" in labs, labs
    finally:
        plt.close(fig)


@pytest.mark.parametrize("sigma_ln", [1.3259, 41.0, 920.0])
def test_wide_co_panel_frames_the_curve_it_drew(sigma_ln):
    """A wide C/O forecast must RENDER, and inside the range the model solves.

    The panel window is recomputed from mu and sigma_ln rather than read off
    the grid, and mu*exp(3.5*sigma_ln) overflows to inf long before the grid
    does -- set_xlim(inf) is a hard matplotlib error, and a merely large 1e62
    frames the panel on empty decades. So the window is clamped to the curve
    that was actually drawn. 1.33 is a real HD 149026 b run; 41 and 920 are
    niriss_soss_ord2 (notes.md), the widths that used to read "unconstrained".
    """
    lo, hi = posteriors.co_curve_bounds()
    curve = posteriors.ln_gaussian_curve(0.55, sigma_ln, bounds=(lo, hi))
    assert lo <= curve["theta"][0] and curve["theta"][-1] <= hi
    pan = dict(axis_label="C/O", curves=[dict(
        label="G395H", theta=curve["theta"], pdf=curve["pdf"], color="#2a78d6",
        mu=0.55, sigma=0.55 * sigma_ln, sigma_ln=sigma_ln,
        curve_family="ln_gaussian")], center=0.55)
    fig = summary_figure.compose_summary_figure(
        _spectrum(), posterior_panels=[pan])
    x0, x1 = fig.axes[1].get_xlim()
    assert np.isfinite([x0, x1]).all(), (x0, x1)
    assert lo <= x0 < x1 <= hi, (x0, x1)
    plt.close(fig)


def test_co_panel_accepts_a_center_on_the_solvable_boundary():
    """CO_MIN and CO_MAX are legal C/O values, so a forecast centred on one
    must still get a bounded grid -- the clamp is inclusive at both ends."""
    lo, hi = posteriors.co_curve_bounds()
    for center in (lo, hi):
        theta = posteriors.ln_gaussian_curve(center, 920.0,
                                             bounds=(lo, hi))["theta"]
        assert np.all(np.isfinite(theta)) and np.all(np.diff(theta) > 0.0)
        assert theta[0] >= lo and theta[-1] <= hi


def test_mock_center_outside_the_solvable_range_is_refused():
    """A draw recovering a C/O the forward model cannot produce is off scale:
    it would have to be drawn outside every C/O panel's window, so the caller
    draws the unshifted forecast instead. Before this, such a centre reached
    ln_gaussian_curve, which ignores bounds that exclude the centre, and the
    unbounded grid then overflowed."""
    lo, hi = posteriors.co_curve_bounds()
    assert posteriors.mock_center_co(0.55, 506.0, 6.0) is None   # -> 221.9
    assert posteriors.mock_center_co(0.55, 506.0, -3.0) is None  # -> 0.027
    inside = posteriors.mock_center_co(0.55, 506.0, 1.0)
    assert inside is not None and lo <= inside <= hi
