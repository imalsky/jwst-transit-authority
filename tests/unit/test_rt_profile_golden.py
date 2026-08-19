"""Golden pin on _rt_profile_common: the RT-facing profile keys must be
EXACTLY these for a default vulcan request (numpy-only; the engine config is
stubbed)."""
from types import SimpleNamespace

import pytest

from jwst_tool import forward


def _cp(**kw):
    base = dict(planet="wasp39b", tp_mode="guillot",
                kzz_mode="const", kzz_const=1.0e9)
    base.update(kw)
    return forward.canonical_params(base)


_WIDE = {"nu_min": 666.0, "nu_max": 10000.0, "art_pbtm_bar": 10.0,
         "some_engine_default": "kept"}


def test_rt_profile_common_golden_default():
    cp = _cp()
    prof = forward._rt_profile_common(cp, SimpleNamespace(WIDE=_WIDE))
    # config.WIDE passes through untouched...
    assert prof["nu_min"] == 666.0
    assert prof["some_engine_default"] == "kept"
    # ...plus exactly the pre-refactor RT-facing overrides
    assert prof["nz"] == 100 and prof["art_nlayer"] == 100
    assert prof["nu_pts"] == 4000
    assert prof["broadening"] == "air"
    assert prof["art_ptop_bar"] == 1.0e-8
    assert prof["rt_integration"] == "simpson"
    assert prof["dit_grid_resolution"] == 1.0
    assert prof["molecules"] == ["H2O", "CO2", "CO", "CH4", "SO2"]
    assert prof["use_photo"] is True
    assert prof["use_rayleigh"] is True
    assert prof["rp_cm"] == pytest.approx(1.279 * 7.1492e9, rel=1e-6)
    assert prof["gs_cgs"] == 422.0
    assert prof["rstar_cm"] == pytest.approx(0.932 * 6.957e10, rel=1e-6)
    # no Mie keys when the deck is off (pre-refactor behavior)
    assert "mie_condensate" not in prof
    # and none of the chemistry-only keys leak in here (they are added by
    # _assemble_chem on top): the vulcan profile stays bit-identical because
    # the union of this dict and the chemistry-only block IS the full profile
    for k in ("yconv_cri", "abundance_mode", "co_mode", "reanchor_atom_ini",
              "dt_max", "cfg_overrides"):
        assert k not in prof


def test_rt_profile_common_mie_keys_when_deck_on():
    cp = _cp(mie_condensate="MgSiO3")
    prof = forward._rt_profile_common(cp, SimpleNamespace(WIDE=_WIDE))
    assert prof["mie_condensate"] == "MgSiO3"
    assert prof["mie_data_dir"].endswith("exojax_mie")
