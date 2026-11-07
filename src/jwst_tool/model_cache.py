"""Cache keys of a canonical run, and reading its model_cache entries."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
import zipfile
import zlib
from typing import TYPE_CHECKING

import numpy as np

from jwst_tool import instruments as _ins
from jwst_tool.run_params import canonical_params

if TYPE_CHECKING:
    from pathlib import Path


def params_key(params: dict) -> str:
    """Return the hash of the canonical parameters: the spectrum's cache key."""
    s = json.dumps(canonical_params(params), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def cache_path(params: dict) -> Path:
    """Return the model_cache entry of a run."""
    return _ins.MODEL_CACHE / f"{params_key(params)}.npz"


# Canonical keys that change the SPECTRUM but never the converged chemistry
# column, so chem_key strips them and an RT-only edit reuses the solved column
# (adjoint_diag.adjoint_key's strip list). Dual-use keys -- nz, p_btm_bar,
# rt_ptop_bar (the chemistry top follows it), rp_rjup, gs_cgs, T-P,
# composition, Kzz -- stay in the key, as does "version" via canonical_params.
# p_ref_bar is NOT irrelevant: it is the pressure the catalogue radius and
# gravity apply at, so it sets the chemistry's own radius anchor and moves the
# converged column (assemble_chem, "ONE gravity per run").
CHEM_IRRELEVANT_PARAMS = (
    "fisher_params",
    "jac_method",
    "use_rayleigh",
    "cloud_on",
    "log_kappa_cloud",
    "alpha_cloud",
    "extra_mols",
    "wo_mols",
    "rt_integration",
    "science_mode",
    "star_teff",
    "star_logg",
    "star_feh",
)


def chem_key(params: dict) -> str:
    """Return the hash of the chemistry-relevant canonical parameters: the column's cache key."""
    cp = canonical_params(params)
    payload = {k: v for k, v in cp.items() if k not in CHEM_IRRELEVANT_PARAMS}
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def chem_cache_path(params: dict) -> Path:
    """Return the chem-cache entry of a run."""
    return _ins.MODEL_CACHE / f"chem-{chem_key(params)}.npz"


def load_cached_npz(p: Path) -> dict | None:
    """Return cached npz as a dict, or None when absent.

    A file that exists but does not read back as a complete npz (torn write, damaged disk) is quarantined to
    ``<name>.corrupt-<t>`` and treated as a miss, so the entry recomputes and the damaged bytes stay on disk. Also
    serves adjoint_diag's cache.
    """
    if not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (zipfile.BadZipFile, zlib.error, OSError, EOFError, ValueError) as e:
        quarantine = p.with_name(f"{p.name}.corrupt-{int(time.time())}")
        with contextlib.suppress(OSError):  # a concurrent reader may have moved it
            p.rename(quarantine)
        logging.getLogger(__name__).warning(
            "[cache] %s unreadable (%r); quarantined -> %s, treating as a cache miss", p.name, e, quarantine.name
        )
        return None


def load_result(params: dict) -> dict | None:
    """Return cached spectrum dict or None.

    Always present: wl_um, depth, depth_wo (n_wo, n_nu), wo_mols (the
    leave-one-out set depth_wo rows align with), mols, ymix, ymix_species
    (the column names of ymix), p_bar, T, theta, theta_names, params_json,
    science_mode, chem_provider, unmodeled (network species with no k-table,
    "<species>|<vmr>"), rate_rows_outside / rate_rows_total (the rate-law
    T-range advisory), co_bz_margin (the AD dlnCO margins, NaN when unused)
    and photo_escalated (stages that needed photolysis cadence 1). Under
    PICASO also picaso_cert (certificate, quench levels, gravity anchor). With
    Fisher requested:
    jac (n_par, n_nu), jac_names, jac_row_method, fd_h, fd_err. In emission,
    the thin-bottom evidence detect REQUIRES and refuses without: fp_flux,
    fs_flux, emis_depth_norm, emis_thin_flux_frac_wo (n_wo), and the
    per-wavelength emis_tau_bottom (n_nu) / emis_tau_bottom_wo (n_wo, n_nu)
    the band-restricted gate re-evaluates.
    Also ``producer``: the "repo=commit" strings that SOLVED the column
    (provenance.producer_commits; empty when no checkout could be read; under
    PICASO followed by the worker's stack versions), absent when the column
    came from the chem cache.
    """
    return load_cached_npz(cache_path(params))
