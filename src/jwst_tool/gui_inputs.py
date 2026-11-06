"""What the sidebar hands to the rest of the page, one record per step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jwst_tool.gui_common import K

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Target:
    """Step 1: the planet, the star and the observation type."""

    planet_key: str
    pdef: dict
    science_mode: str
    evw: str  # event word for observation-facing labels: "transit" or "eclipse"
    teff: float
    logg: float
    feh: float
    ks_mag: float
    rstar: float
    rp: float
    g_ms2: float
    orbit_au: float
    t14: float
    sflux: str
    teq: float

    def k(self, name: str) -> str:
        """Widget key of a per-planet setting."""
        return K(f"{self.planet_key}_{name}")


@dataclass(frozen=True)
class Atmosphere:
    """Step 2: chemistry engine, structure, composition, mixing, opacity and clouds."""

    pic: bool
    chem_provider: str
    tp_mode: str
    tp_kwargs: dict
    tp_file: str
    tp_file_path: str | None
    tp_file_ok: bool
    rt_ptop_bar: float
    p_ref_bar: float
    p_btm_bar: float
    met: float
    co_ratio: float
    picaso_quench: bool
    kzz_mode: str
    kzz_x: float
    kzz_const: float
    kzz_kmax: float
    kzz_plev: float
    kzz_kdeep: float
    network: str
    use_photo: bool
    use_moldiff: bool
    use_vm_mol: bool
    sl_angle_deg: float
    f_diurnal: float
    base_set: list
    extra_set: list
    extra_mols: list
    use_rayleigh: bool
    cloud_on: bool
    cloud_mode: str
    cloud_top_bar: float | None
    log_kappa_cloud: float
    alpha_cloud: float


@dataclass(frozen=True)
class Goal:
    """Step 3: the science goal and the Jacobian it needs."""

    goal: str
    target_mol: str | None
    target_sig: float
    goal_param: str | None
    target_prec: float | None
    marginalize: bool
    do_fisher: bool
    fisher_params: list
    jac_method: str


@dataclass(frozen=True)
class Observation:
    """Step 4: modes, event count, mock draw, timing, binning and noise."""

    mode_keys: list
    n_transits: int
    show_noise: bool
    seed: int
    noise_scale: float
    t_base: float
    sat_limit: float
    r_bin: int
    floor_mode: str
    floor_table: np.ndarray | None
    floors: dict
    floor_choice_made: bool
    infl_mode: dict
    infl: dict


@dataclass(frozen=True)
class Solver:
    """More settings: the grid, the solver gates and the chord integration."""

    nz: int
    yconv_cri: float
    yconv_min: float
    rt_integration: str
