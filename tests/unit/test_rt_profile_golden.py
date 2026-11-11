"""Golden pin on rt_profile_common: the RT-facing profile keys must be
EXACTLY these for a default vulcan request (numpy-only; the engine config is
stubbed).
"""

from types import SimpleNamespace

import pytest

from jwst_tool import chem_build, forward


def _cp(**kw):
    base = dict(planet="wasp39b", tp_mode="guillot", kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return forward.canonical_params(base)


_WIDE = {"nu_min": 666.0, "nu_max": 10000.0, "art_pbtm_bar": 10.0, "some_engine_default": "kept"}


def test_rt_profile_common_golden_default():
    cp = _cp()
    prof = chem_build.rt_profile_common(cp, SimpleNamespace(WIDE=_WIDE))
    # config.WIDE passes through untouched...
    assert prof["nu_min"] == 666.0
    assert prof["some_engine_default"] == "kept"
    # ...plus exactly these RT-facing overrides
    assert prof["nz"] == 100 and prof["art_nlayer"] == 100
    assert prof["art_ptop_bar"] == 1.0e-7
    assert prof["rt_integration"] == "simpson"
    assert "opacity_mode" not in prof  # the engine removed the key: correlated-k is its only opacity path
    assert prof["molecules"] == ["H2O", "CO2", "CO", "CH4", "SO2", "NH3"]
    assert prof["omit_core_molecules"] == []
    assert prof["use_photo"] is True
    assert prof["use_rayleigh"] is True
    assert prof["rp_cm"] == pytest.approx(9.546962e9, rel=1e-5)
    assert prof["gs_cgs"] == 390.58
    assert prof["rstar_cm"] == pytest.approx(0.939 * 6.957e10, rel=1e-6)
    # none of the chemistry-only keys leak in here (they are added by
    # assemble_chem on top): the vulcan profile stays bit-identical because
    # the union of this dict and the chemistry-only block IS the full profile
    for k in ("yconv_cri", "co_mode", "dt_max", "cfg_overrides", "network", "atom_list", "vulcan_cfg_name"):
        assert k not in prof


def test_ncho_names_every_sulfur_core_absorber_as_left_out():
    """The sulfur-free network cannot carry them, so none may reach the user as a missing absorber."""
    constants = pytest.importorskip("vulcan_forward.constants")
    from jwst_tool import model_defs

    prof = chem_build.rt_profile_common(_cp(network="ncho"), SimpleNamespace(WIDE=_WIDE))
    assert set(prof["omit_core_molecules"]) == model_defs.S_MOLECULES & set(constants.CORE_MOLECULE_BANDS_UM)
