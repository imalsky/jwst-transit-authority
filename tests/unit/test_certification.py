"""Certification gates on the heavy path: the convergence certificate, the
FD h-vs-2h consistency gate, and the AD dlnCO oxygen-reservoir refusal.

All three used to live inside ``run_model``'s closure, so nothing could reach
them without a full solve and none of them was covered. They are module-level
now; the stubs here stand in for the engine's ConvDiag and chem model.
"""
import re
from types import SimpleNamespace

import numpy as np
import pytest

from jwst_tool import forward


def _diag(*, longdy=0.05, longdydt=1e-11, branch=2, conv_normal=True,
          accept=1400, aflux=0.004, species=0, layer=42, vmr=6.1e-18,
          t=4.1e8, dt=5.3e5):
    return SimpleNamespace(
        accept_count=accept, longdy=longdy, longdydt=longdydt,
        conv_branch=branch, conv_normal=conv_normal, aflux_change=aflux,
        cell_species=species, cell_layer=layer, cell_vmr=vmr, t=t, dt=dt)


def _chem(yconv_min=0.1, count_max=30000, nz=100):
    return SimpleNamespace(yconv_min=yconv_min, count_max=count_max,
                           p_bar=np.logspace(-7, 0.88, nz))


SPECIES = ["N2O", "C", "S4"]


def test_certificate_records_every_field_and_a_failure_names_them():
    """A passing stage returns the CONV_FIELDS record and logs the branch,
    longdydt, aflux_change and the controlling species@z cell; a failure
    raises naming all of them. Certification is the runner's canonical gate
    (conv_normal AND longdy < yconv_min) -- never loosened here."""
    lines = []
    rec = forward.check_converged(_diag(), "baseline solve", SPECIES,
                                  _chem(), lines.append)
    assert len(rec) == len(forward.CONV_FIELDS)
    stage, accept, longdy, longdydt, branch, flux, cell = rec
    assert (stage, accept, branch, cell) == ("baseline solve", 1400, 2, "N2O@z42")
    assert (longdy, longdydt, flux) == (0.05, 1e-11, 0.004)
    assert "loose (yconv_min)" in lines[0] and "N2O@z42" in lines[0]

    # conv_normal False -> refuse, and say which cell and how it exited
    with pytest.raises(RuntimeError) as e:
        forward.check_converged(_diag(conv_normal=False, longdy=2.73,
                                      longdydt=5.9e-6, branch=0, aflux=341.0,
                                      accept=30001, species=1, layer=47),
                                "baseline solve", SPECIES, _chem())
    msg = str(e.value)
    for needle in ("longdy=2.73", "longdydt=5.9e-06", "aflux_change=341",
                   "branch none", "C@z47", "count_max=30000"):
        assert needle in msg, needle

    # certified by the runner but over the tool's own gate -> still refused,
    # and the wording distinguishes a stall from the step cap
    with pytest.raises(RuntimeError, match="without the runner's canonical"):
        forward.check_converged(_diag(longdy=0.5, accept=900), "FD dlnCO -1h",
                                SPECIES, _chem())


def test_fd_row_certifies_richardson_and_refuses_an_inconsistent_row():
    """The h-vs-2h gate is the certification half of the FD contract (the
    Richardson algebra is pinned in test_forward_params). A row with no
    spectral response is an exact zero, not a division by zero."""
    j1 = np.array([100.0, -50.0, 25.0])
    row, err = forward.fd_row("dlnCO", j1, j1 * 1.05, 0.1)
    assert err == pytest.approx(0.05) and err < forward.FD_CONSISTENCY_TOL
    assert row == pytest.approx((4.0 * j1 - j1 * 1.05) / 3.0)

    zero = np.zeros(3)
    assert forward.fd_row("dlnCO", zero, zero, 0.1) == (pytest.approx(zero), 0.0)

    with pytest.raises(RuntimeError, match="step-size consistency"):
        forward.fd_row("dlnCO", j1, j1 * 1.4, 0.1)
    # the advice names yconv_min: a loose-branch exit ignores yconv_cri
    with pytest.raises(RuntimeError, match="yconv_min"):
        forward.fd_row("dlnCO", j1, j1 * 1.4, 0.1)
    with pytest.raises(RuntimeError, match="non-finite"):
        forward.fd_row("dlnCO", np.array([np.nan, 1.0, 1.0]), j1, 0.1)


def test_ad_dlnco_margin_refuses_on_both_columns_and_on_a_missing_engine():
    """The AD dlnCO row needs the oxygen-reservoir margin above CO_BZ_MIN_AD
    on the BUILD column and again on the CONVERGED column the tangent starts
    from. An engine that cannot report the margin refuses rather than passing
    silently, and a NaN margin refuses too (the `not m > gate` form)."""
    gate = forward.CO_BZ_MIN_AD
    y = np.ones((4, 3))

    ok = SimpleNamespace(co_bz_bound=0.12, co_bz_margin=lambda c: 0.11)
    assert forward.check_ad_co_margin(ok, 0.89) == pytest.approx(0.12)
    assert forward.check_ad_co_margin(ok, 0.89, y=y, build_margin=0.12,
                                      log=lambda _: None) == pytest.approx(0.11)

    # build column below the gate: refused before any solve
    tight = SimpleNamespace(co_bz_bound=0.05, co_bz_margin=lambda c: 0.5)
    with pytest.raises(RuntimeError, match="build column"):
        forward.check_ad_co_margin(tight, 0.95)

    # build column passes, converged column does not -- the carbon-rich case:
    # a fixed-O column with no oxygen-only carriers left has margin ~0
    drifted = SimpleNamespace(co_bz_bound=0.5, co_bz_margin=lambda c: 0.0)
    with pytest.raises(RuntimeError, match="converged column"):
        forward.check_ad_co_margin(drifted, 2.0, y=y, build_margin=0.5,
                                   log=lambda _: None)

    nan = SimpleNamespace(co_bz_bound=float("nan"),
                          co_bz_margin=lambda c: float("nan"))
    with pytest.raises(RuntimeError, match="oxygen-reservoir margin"):
        forward.check_ad_co_margin(nan, 0.55)

    for missing in (SimpleNamespace(co_bz_margin=lambda c: 0.5),
                    SimpleNamespace(co_bz_bound=0.5)):
        with pytest.raises(RuntimeError, match="does not expose"):
            forward.check_ad_co_margin(missing, 0.55)
    assert f"{gate:g}" in str(
        pytest.raises(RuntimeError,
                      lambda: forward.check_ad_co_margin(tight, 0.95)).value)


def test_ad_chemistry_rows_are_never_vmapped_over_tangent_directions():
    """VULCAN-JAX < 0.3.5 returned NaN batched tangents on columns with
    sub-atol densities, and the fixed batched rows still differ from the
    per-row ones by up to 3% per bin. Structural guard, like the ones in
    test_posteriors / test_plotting."""
    import ast
    import importlib.util
    from pathlib import Path

    src = Path(importlib.util.find_spec("jwst_tool.forward").origin).read_text()
    calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call)]
    assert not [c for c in calls if c.endswith("vmap")], (
        "forward.py must take one plain jvp per chemistry Jacobian row")
    assert "jax.jvp(" in src


def test_picaso_is_removed_from_the_api_the_share_path_and_the_deploy_lock():
    """PICASO was removed in 0.43.0. Both refusal sites say so by name, and
    no deploy artifact may pin it (or virga / pysynphot) back in."""
    from pathlib import Path

    from jwst_tool import share_config

    base = dict(planet="wasp39b", tp_mode="guillot", kzz_mode="const",
                kzz_const=1.0e9)
    for kw in (dict(chem_provider="picaso"), dict(tp_mode="picaso_climate")):
        with pytest.raises(ValueError, match="PICASO|picaso"):
            forward.canonical_params({**base, **kw})
    cp = forward.canonical_params(base)
    with pytest.raises(ValueError, match="PICASO|picaso"):
        share_config.widget_state({**cp, "chem_provider": "picaso"},
                                  "n0_{}".format)

    deploy = Path(__file__).resolve().parents[2] / "deploy"
    banned = re.compile(r"^\s*(picaso|virga|pysynphot)\b", re.IGNORECASE | re.MULTILINE)
    for f in sorted(deploy.rglob("*")):
        if f.is_file() and f.suffix in (".txt", ".py", ".env", ""):
            assert not banned.search(f.read_text(errors="ignore")), f


def test_emission_tau_bottom_gate_measures_the_flux_that_leaks_through():
    """The emission bottom-boundary certification: the fix for a thin bottom
    is a deeper column, never a wider tolerance, so the gate has to be a
    flux-weighted fraction and its report has to say WHERE."""
    wl = np.array([1.5, 2.5, 4.0, 8.0, 13.0])
    thick = np.full(5, 50.0)
    flux = np.ones(5)
    assert forward.thin_flux_fraction(thick, flux) == 0.0

    tau = np.array([0.5, 50.0, 50.0, 50.0, 50.0])   # only the 1-2 um bin leaks
    assert forward.thin_flux_fraction(tau, flux) == pytest.approx(0.2)
    # flux-weighted, not bin-counted
    assert forward.thin_flux_fraction(
        tau, np.array([9.0, 1.0, 1.0, 1.0, 1.0])) == pytest.approx(9.0 / 13.0)
    assert forward.EMIS_THIN_FLUX_FRAC < 0.2      # that column would refuse

    report = forward._tau_bottom_breakdown(wl, tau, flux)
    assert "1.0-2.0 um (no modeled continuum)" in report
    assert "12-15 um" in report


def test_an_abundant_species_with_no_k_table_is_named_not_swallowed():
    """A network species the RT cannot see is missing ABSORPTION, not a
    missing trace. Measured at C/O 10 on sncho2025: C6H6 reaches 3.6e-3 over
    the transmission photosphere -- comparable to CO -- and ExoMolOP publishes
    no benzene table, so the spectrum is a lower bound on the true contrast.
    The run has to say so."""
    sp = ["H2O", "CO", "C6H6", "C2H2", "H2", "H"]
    # two layers inside the photosphere band, one far below it
    p = np.array([1.0e-3, 1.0e-4, 5.0])
    y = np.array([[1.0, 1.0e-2, 3.6e-3, 1.0e-6, 0.77, 1.2e-2]] * 3)
    tables = {"H2O", "CO", "C2H2"}

    found = forward.unmodeled_absorbers(sp, y, p, tables)
    assert [n for n, _ in found] == ["C6H6"]
    assert found[0][1] == pytest.approx(3.6e-3 / y[0].sum(), rel=1e-6)

    # H2 (CIA continuum) and atomic H (no IR molecular bands) are absent from
    # the table set on purpose and must NEVER be reported as missing opacity,
    # however abundant -- they dominate this column
    assert "H2" not in dict(found) and "H" not in dict(found)
    # a species the RT DOES carry is never flagged, however abundant
    assert forward.unmodeled_absorbers(sp, y, p, tables | {"C6H6"}) == []
    # and neither is one below the threshold
    y_low = y.copy()
    y_low[:, 2] = forward.UNMODELED_VMR_WARN * y[0].sum() * 0.5
    assert forward.unmodeled_absorbers(sp, y_low, p, tables) == []
    # most abundant first
    y2 = np.array([[1.0, 1.0e-2, 3.6e-3, 1.0e-2, 0.77, 1.2e-2]] * 3)
    assert [n for n, _ in forward.unmodeled_absorbers(sp, y2, p, {"H2O", "CO"})] \
        == ["C2H2", "C6H6"]
