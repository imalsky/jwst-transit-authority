"""Forward model runner: chemistry -> ExoJAX spectrum.

Method and contract: docs/forward_model.md, section `forward`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from jwst_tool import instruments as _ins, picaso_engine as _pe, provenance as _prov
from jwst_tool.certify import (
    PHOTOSPHERE_BAR,
    certified_solve,
    check_ad_co_margin,
    check_elements,
    check_t_window,
    check_unmodeled,
    table_species,
    unmodeled_absorbers,
)
from jwst_tool.chem_build import assemble_chem
from jwst_tool.instruments import MODEL_CACHE
from jwst_tool.jacobian import (
    Baseline,
    Rows,
    build_jacobian,
)
from jwst_tool.model_cache import cache_path, chem_cache_path, load_cached_npz, load_result, params_key
from jwst_tool.model_defs import (
    _VERSION,
    CHEM_PARAM_NAMES,
    CLOUD_FISHER_PARAMS,
    EXTRA_MOLECULES,
    EXTRA_MOLECULES_DEFAULT,
    FD_COMP_PARAMS,
    MOLECULES,
    NETWORKS,
    NO_EXOMOLOP_TABLE,
    PICASO_CLIMATE_SOLVE_S,
    PICASO_QUENCH_COLUMN_S,
    TP_FILE_UPLOAD,
    active_molecules,
)
from jwst_tool.run_params import canonical_params
from jwst_tool.spectrum import (
    ColumnSpectrum,
    EmissionEvidence,
    RtSetup,
    build_rt_setup,
    emission_photosphere_bar,
    emission_spectra,
    gas_mask,
    transmission_spectra,
)
from jwst_tool.tp_tables import resolve_tp_file, stage_upload, uploads_dir

if TYPE_CHECKING:
    from collections.abc import Callable

    from jwst_tool.tool_types import AnyArray, Log

_NO_EXOMOLOP_TABLE = NO_EXOMOLOP_TABLE  # the deployed Space's scripts read this name, at a pinned tool commit

# The run entry point, and the names the deploy scripts and the research note's figure scripts read from this module.
__all__ = [
    "EXTRA_MOLECULES",
    "EXTRA_MOLECULES_DEFAULT",
    "MODEL_CACHE",
    "MOLECULES",
    "_NO_EXOMOLOP_TABLE",
    "_VERSION",
    "cache_path",
    "canonical_params",
    "load_result",
    "params_key",
    "run_model",
]


def _make_progress(cp: dict, log: Log) -> tuple[Callable[[], None], Callable[[], None]]:
    """Sequential stage tracker: emits "[fwd] PROG <frac> <label>" lines.

    The stage list MUST mirror run_model's actual stage order (same
    conditionals); weights are rough wall-clock seconds. advance() is called
    at the START of each stage.
    """
    emis = cp["science_mode"] == "emission"
    pic = cp["chem_provider"] == "picaso"
    # a PICASO climate solve is ~550 s on one core (two for the baseline's
    # gravity fixed point); its equilibrium chemistry alone is seconds
    clim_s = (
        PICASO_CLIMATE_SOLVE_S
        if cp["tp_mode"] == "picaso_climate"
        else PICASO_QUENCH_COLUMN_S
        if cp.get("picaso_quench")
        else 3.0
    )
    stages = [
        (
            "PICASO climate + equilibrium chemistry"
            if cp["tp_mode"] == "picaso_climate"
            else "PICASO equilibrium chemistry",
            2.0 * clim_s,
        )
        if pic
        else ("building chemistry model", 5.0)
    ]
    stages += [("building radiative transfer (opacities + CIA)", 10.0 + 3.0 * len(cp["extra_mols"]))]
    if emis:
        stages += [("emission model + stellar SED", 6.0)]
    stages += [("PICASO column", 1.0) if pic else ("solving photochemistry", 35.0)]
    # The leave-one-out block is one engine batch, not one stage per molecule;
    # each wo spectrum refolds from its position to the end of the set.
    n_wo = len(cp["wo_mols"])
    n_active = len(active_molecules(cp))
    wo_w = 1.5 * n_wo * (n_active + 1)
    # ONE stage either way: baseline and removed-molecule spectra come out of a
    # single engine batch in both science modes.
    if n_wo:
        stages += [(f"full + removed-molecule spectra ({n_wo} molecules)", 8.0 + wo_w)]
    else:
        stages += [(f"full {'eclipse' if emis else 'transmission'} spectrum", 8.0)]
    # Jacobian rows: fd = one build+solve per stencil point, cloud rows are
    # RT-only, ad = one warm jvp per chemistry row (one stage for all)
    ad = cp["jac_method"] == "ad"

    def _row_stage(n: str) -> tuple:
        if n in CLOUD_FISHER_PARAMS:
            return (f"{'AD' if ad else 'FD'} Jacobian d/d({n})", 8.0)
        if pic:
            return (f"FD Jacobian d/d({n})", 4.0 * clim_s + 8.0)
        return (f"FD Jacobian d/d({n})", 280.0 if n in FD_COMP_PARAMS else 260.0)

    if ad:
        # chemistry rows run back to back in one stage; RT-only deck rows and
        # the FD T-P rows keep their own per-row stages
        chem_rows = [n for n in cp["fisher_params"] if n in CHEM_PARAM_NAMES]
        if chem_rows:
            stages += [(f"AD Jacobian ({len(chem_rows)} rows)", 110.0 * len(chem_rows))]
    stages += [_row_stage(n) for n in cp["fisher_params"] if not ad or n not in CHEM_PARAM_NAMES]
    if cp["fisher_params"]:
        stages += [(("AD" if ad else "FD") + " Jacobian d/d(lnR0)", 8.0)]
    total = sum(w for _, w in stages)
    state: dict[str, Any] = {"i": 0, "done": 0.0}

    def advance() -> None:
        label, w = stages[state["i"]]
        log(f"[fwd] PROG {state['done'] / total:.3f} {label}")
        state["i"] += 1
        state["done"] += w

    def finish() -> None:
        log("[fwd] PROG 1.000 done")

    return advance, finish


@dataclass
class _Column:
    """A run's certified baseline column and how it was obtained."""

    y_sol: Any
    y_np: Any
    chem: Any
    column: ColumnSpectrum
    species: list
    escalated: list
    run_frq: int | None
    solved_here: bool


@dataclass
class _RunResult:
    """What a run computed on top of its column, ready to store."""

    T_check: np.ndarray
    depth: np.ndarray
    depth_wo: np.ndarray | None
    evidence: EmissionEvidence | None
    rows: Rows | None
    unmodeled: list
    rate_ex: dict
    bz_build: float
    wo_list: list


def _archive_uploaded_table(params: dict, cp: dict, log: Log) -> None:
    # Uploaded T-P tables become a CONTENT-ADDRESSED copy under
    # <output>/uploads/<sha1>.txt before anything heavy runs, so canonical
    # params alone re-resolve the exact bytes later.
    if cp["tp_mode"] == "file" and cp["tp_file"] == TP_FILE_UPLOAD:
        src_path, sha1 = resolve_tp_file(params)
        if not (uploads_dir() / f"{sha1}.txt").exists():
            # atomic: the exists() guard would make a torn copy permanent and
            # tp_file_from_cp would then refuse this sha1 forever
            staged, dst = stage_upload(src_path.read_bytes())
            os.replace(staged, dst)
            log(f"[fwd] uploaded T-P table archived -> {dst}")


def _column_from_cache(
    chem_art: dict, chem_out: Path, A: Any, species_now: list, chem: Any, setup: RtSetup, log: Log
) -> _Column:
    """Validate a chem-cache entry loudly and rebind the run to its cadence."""
    import jax.numpy as jnp

    column = ColumnSpectrum(setup, chem)
    escalated: list[str] = []
    run_frq = None
    # validate loudly: a mismatched entry raises, never silently re-solves
    if [str(s) for s in chem_art["species"]] != species_now:
        raise RuntimeError(
            f"chem-cache {chem_out.name}: species set does not match "
            "the built network -- the cache key failed to separate two "
            "different chemistry configurations. Delete the entry and "
            "report this; do not ignore it."
        )
    if not np.array_equal(np.asarray(chem_art["theta"]), A.theta):
        raise RuntimeError(
            f"chem-cache {chem_out.name}: stored theta differs from "
            "this run's theta under an equal chem_key -- the key is "
            "missing a chemistry-relevant parameter. Delete the entry "
            "and report this; do not ignore it."
        )
    y_np = np.asarray(chem_art["y_raw"], dtype=np.float64)
    if not np.all(np.isfinite(y_np)):
        raise RuntimeError(f"chem-cache {chem_out.name}: non-finite stored column")
    ac = int(chem_art["conv_accept"][0])
    ld = float(chem_art["conv_longdy"][0])
    if not ld < float(chem.yconv_min):
        raise RuntimeError(
            f"chem-cache {chem_out.name}: stored certificate "
            f"(longdy={ld:.3g}) fails the current gate "
            f"yconv_min={float(chem.yconv_min):g}"
        )
    check_elements(y_np, chem, "chem-cache column", log)
    y_sol = jnp.asarray(y_np)
    if int(chem_art["photo_escalated"][0]):
        # The cached column was solved at cadence 1, so the rest of the run
        # must be too: a row solved at the config cadence against this
        # baseline (or anchored on it) would mix cadences, which is the one
        # thing the row rule exists to prevent. Same rebinding as the fresh
        # path below; the rebuild costs a model build, not a solve.
        escalated.append("baseline solve (whole run at photolysis cadence 1)")
        run_frq = 1
        chem = A.build_chem(tag="baseline [photo cadence 1]", photo_frq=1)
        column = ColumnSpectrum(setup, chem)
    log(f"[fwd] chemistry column from chem-cache ({chem_out.name}, {ac} accepted steps at write); solve skipped")
    return _Column(y_sol, y_np, chem, column, species_now, escalated, run_frq, False)


def _solve_cold(
    chem: Any, A: Any, th0: AnyArray, species_now: list, chem_out: Path, setup: RtSetup, t0: float, log: Log
) -> _Column:
    """Solve the baseline cold, certify it and persist the raw column."""
    column = ColumnSpectrum(setup, chem)
    escalated: list[str] = []
    run_frq = None
    log("[fwd] solving photochemistry (cold, certified) ...")
    y_sol, chem, cert, esc = certified_solve(
        chem,
        th0,
        "baseline solve",
        log=log,
        rebuild=lambda: A.build_chem(tag="baseline [photo cadence 1]", photo_frq=1),
    )
    if esc:  # the model was rebuilt: rebind what binds it
        escalated.append("baseline solve (whole run at photolysis cadence 1)")
        run_frq = 1
        column = ColumnSpectrum(setup, chem)
    y_np = np.asarray(y_sol)
    if not np.all(np.isfinite(y_np)):
        raise RuntimeError(
            "chemistry solve returned non-finite abundances -- parameter set outside the modelable range"
        )
    check_elements(y_np, chem, "baseline solve", log)
    log(f"[fwd] chemistry solved in {time.time() - t0:.0f} s total")
    # persist the certified RAW column (atomic, as for the flat cache)
    _, ac, ld, lddt, br, fl, cell = cert
    chem_arrays = dict(
        y_raw=y_np,
        species=np.array(species_now, dtype="U16"),
        theta=A.theta,
        theta_names=np.array(A.theta_names, dtype="U16"),
        p_bar=np.asarray(chem.p_bar),
        conv_accept=np.array([ac], dtype=np.int64),
        conv_longdy=np.array([ld], dtype=np.float64),
        conv_longdydt=np.array([lddt], dtype=np.float64),
        conv_branch=np.array([br], dtype=np.int64),
        conv_flux=np.array([fl], dtype=np.float64),
        conv_cell=np.array([cell], dtype="U48"),
        photo_escalated=np.array([1 if esc else 0], dtype=np.int64),
    )
    chem_out.parent.mkdir(parents=True, exist_ok=True)
    _ins.atomic_write(chem_out, lambda fh: np.savez_compressed(fh, **chem_arrays))
    log(f"[fwd] chem column cached -> {chem_out.name}")
    return _Column(y_sol, y_np, chem, column, species_now, escalated, run_frq, True)


def _baseline_column(
    params: dict,
    cp: dict,
    chem: Any,
    A: Any,
    th0: AnyArray,
    setup: RtSetup,
    advance: Callable[[], None],
    log: Log,
) -> _Column:
    """Return the baseline column: certified cold, unless an identical chemistry-relevant set already solved.

    The chem-level cache stores the raw column, so an RT-only edit re-renders from the bits a fresh solve would produce.
    """
    import jax.numpy as jnp

    t0 = time.time()
    advance()
    species_now = [s for s, _ in sorted(chem.sidx.items(), key=lambda kv: kv[1])]
    if cp["chem_provider"] == "picaso":  # assemble and build already certified the PICASO column
        y_np = np.asarray(chem.y, dtype=np.float64)
        return _Column(jnp.asarray(y_np), y_np, chem, ColumnSpectrum(setup, chem), species_now, [], None, True)
    chem_out = chem_cache_path(params)
    chem_art = load_cached_npz(chem_out)
    if chem_art is not None:
        return _column_from_cache(chem_art, chem_out, A, species_now, chem, setup, log)
    return _solve_cold(chem, A, th0, species_now, chem_out, setup, t0, log)


def _opacity_advisory(col: _Column, setup: RtSetup, th0: AnyArray, log: Log) -> tuple[list, tuple[float, float]]:
    """Gate and report the baseline's absorbers with no k-table; return them and the photosphere band, bar, it used."""
    # Missing opacity: a species the network solves but the RT cannot see is
    # missing absorption, not a missing trace. Loud, never silent. The error it
    # leaves is UNSIGNED: the tool scores depth - depth_wo, which in the
    # isothermal limit is H ln(1 + k_X/k_bkg), so an omitted background
    # absorber INFLATES the target's apparent contrast as readily as an
    # omitted overlapping band deflates it.
    # Above UNMODELED_VMR_REFUSE the run stops here, before any spectrum or
    # Jacobian row is built: a missing bulk absorber is not a caveat.
    # The band is where the observation forms: the transit photosphere, or the eclipse's own (certify).
    band = PHOTOSPHERE_BAR if setup.emis is None else emission_photosphere_bar(setup, col.column, col.y_sol, th0)
    log(f"[fwd] absorber gate band: {band[0]:.3g}-{band[1]:.3g} bar")
    tables = table_species(setup.config.MOLECULES)
    y, p_bar = np.asarray(col.y_sol), np.asarray(col.chem.p_bar)
    check_unmodeled(col.species, y, p_bar, tables, band, "baseline solve", log)
    unmodeled = unmodeled_absorbers(col.species, y, p_bar, tables, band_bar=band)
    if unmodeled:
        log(
            "[fwd] WARNING: this column carries "
            + ", ".join(f"{sp} at {v:.2e}" for sp, v in unmodeled)
            + f" (mean VMR, {band[0]:.2g}-{band[1]:.2g} bar) with NO published k-table, "
            "so the RT omits them: the modeled feature contrast carries an "
            "UNQUANTIFIED error that can run in EITHER direction -- omitted "
            "background opacity inflates it, an omitted overlapping band "
            "deflates it. Carbon-rich compositions are where this bites."
        )
    return unmodeled, band


def _rate_advisory(cp: dict, T_check: np.ndarray, log: Log) -> dict:
    # Rate-law temperature-range exposure of this run's network on this run's
    # T grid: ADVISORY, never a gate -- VULCAN evaluates every rate everywhere
    # and parity requires the same. The engine prints its own advisory to the
    # worker log, which the GUI shows only when a run FAILS, so the counts are
    # stored for the app to state on a successful run. `thermal_rows` is the
    # denominator: every thermal row, including the ones with no documented
    # range (those are never counted as outside).
    if cp["chem_provider"] == "picaso":  # no kinetic rate is evaluated under PICASO
        rate_ex = {"any_outside": 0, "thermal_rows": 0}
    else:
        from vulcan3 import runtime_validation as _rv
        from vulcan3.network import parse_network as _parse_network

        rate_ex = _rv.rate_temp_range_exposure(_parse_network(NETWORKS[cp["network"]][0]), T_check)
        log(
            f"[fwd] rate T-range exposure: {rate_ex['any_outside']} of "
            f"{rate_ex['thermal_rows']} thermal rows are evaluated outside "
            f"their documented range in >= 1 layer ({rate_ex['no_range']} rows "
            "carry no parseable range); advisory, no rate is altered"
        )
    return rate_ex


def _write_result(params: dict, cp: dict, A: Any, setup: RtSetup, col: _Column, res: _RunResult) -> Path:
    """Write the run's model_cache entry atomically; every gate has passed by now."""
    pic = cp["chem_provider"] == "picaso"
    mols_active = list(A.profile["molecules"])
    bz_warm = res.rows.bz_warm if res.rows is not None else float("nan")
    out = cache_path(params)
    out.parent.mkdir(parents=True, exist_ok=True)
    # npz ymix must use the SAME gas normalization the RT applies (condensate
    # columns excluded), or the saved ymix disagrees with the spectra.
    y_gas_np = col.y_np * gas_mask(col.chem)[None, :]
    ymix_np = y_gas_np / y_gas_np.sum(axis=1, keepdims=True)
    arrays = dict(
        wl_um=np.asarray(setup.rt.wl_um, dtype=np.float64),
        depth=res.depth,
        depth_wo=res.depth_wo,
        mols=np.array(mols_active, dtype="U8"),
        # the set depth_wo (and the emis_*_wo certificates) align with: a
        # subset of mols in fold order. Readers index by THIS array, not mols.
        wo_mols=np.array(res.wo_list, dtype="U8"),
        ymix=ymix_np,
        p_bar=np.asarray(col.chem.p_bar),
        # THE COLUMN NAMES OF ymix, which is the FULL network state while
        # `mols` is the shorter RT list. Never infer these from `mols`.
        ymix_species=np.array(col.species, dtype="U16"),
        T=np.asarray(res.T_check),
        theta=A.theta,
        theta_names=np.array(A.theta_names, dtype="U16"),
        # auto-sized U dtype: a fixed width silently truncates long JSON
        params_json=np.array(json.dumps(cp)),
        # Network species with no k-table (unmodeled_absorbers). Stored, not
        # just logged: a model-cache hit never solves, so the GUI has no other
        # way to say the contrast carries an unsigned opacity error.
        unmodeled=np.array([f"{sp}|{v:.3e}" for sp, v in res.unmodeled], dtype="U32"),
        # Rate-law T-range advisory, this network on this T grid (above).
        rate_rows_outside=np.array([res.rate_ex["any_outside"]], dtype=np.int64),
        rate_rows_total=np.array([res.rate_ex["thermal_rows"]], dtype=np.int64),
        # AD dlnCO oxygen-reservoir margin on the build column and on the
        # converged column the tangent starts from (NaN when no AD dlnCO row)
        co_bz_margin=np.array([res.bz_build, bz_warm], dtype=np.float64),
        science_mode=np.array(cp["science_mode"], dtype="U16"),
        chem_provider=np.array(cp["chem_provider"], dtype="U16"),
        # Stages that needed the photolysis-cadence escalation to certify;
        # empty on the normal path. See certified_solve.
        # auto-sized U dtype: a fixed width truncates the longest label
        photo_escalated=np.array(col.escalated),
        # Core absorbers of the band the RT leaves out without naming them
        # (constants.CORE_MOLECULE_BANDS_UM outside REQUIRED_MOLECULES): their
        # features are missing from the spectrum.
        unlisted_core=np.array(list(setup.rt.unlisted_core_molecules), dtype="U8"),
    )
    if col.solved_here:
        # what SOLVED this column; exports read it, the cache key does not.
        # A chem-cache hit was solved by an unrecorded stack: no stamp, and
        # the export says "installed now" instead of asserting one.
        arrays["producer"] = np.array(
            _prov.producer_commits() + ([f"{k}={v}" for k, v in sorted(col.chem.versions.items())] if pic else [])
        )
    if pic:
        # layer solves that converged only at picaso's shifted temperatures,
        # summed over every column the run solved: assemble's columns and the
        # T-P stencil assemblies the Jacobian adds (run_model logs each stage)
        arrays["picaso_retried_layers"] = np.array([sum(n for _, n in A.retried)], dtype=np.int64)
        arrays["picaso_cert"] = np.array(
            json.dumps({"cert": col.chem.cert, "quench_bar": col.chem.quench, "g_anchor_cgs": A.g_anchor})
        )
    if setup.emis is not None:
        depth_norm_em0 = res.evidence.depth_norm  # type: ignore[union-attr]  # set in an emission run
        arrays["fs_flux"] = np.asarray(setup.fs_j, dtype=np.float64)
        # Fp derived exactly from the stored eclipse depth (lnR0 = 0 baseline)
        arrays["fp_flux"] = res.depth * np.asarray(setup.fs_j) / depth_norm_em0
        # ... and the same scalar lets detect recover every removed-molecule
        # Fp from its stored depth, which the band-restricted gate needs.
        arrays["emis_depth_norm"] = np.array([depth_norm_em0])
        # PER-WAVELENGTH bottom optical depth. The band-global fraction below
        # cannot certify a narrower observing band -- a mode measuring 8% of
        # the emitted energy amplifies a local leak ~12x -- so detect
        # re-evaluates the gate over the wavelengths its operator actually
        # covers, which needs tau, not a reduction of it.
        arrays["emis_tau_bottom"] = res.evidence.tau_bottom  # type: ignore[union-attr]  # set in an emission run
        arrays["emis_tau_bottom_wo"] = res.evidence.tau_bottom_wo  # type: ignore[union-attr]  # set in an emission run
        # the BAND-GLOBAL per-molecule certificate detect enforces per target
        arrays["emis_thin_flux_frac_wo"] = res.evidence.thin_flux_frac_wo  # type: ignore[union-attr]  # set in an emission run
    if res.rows is not None:
        arrays["jac"] = res.rows.jac
        arrays["jac_names"] = np.array(res.rows.names, dtype="U16")
        # Per-row provenance. fd_err NaN = unmeasured (AD rows, and the
        # single-difference RT rows with no h-vs-2h check), never 0; 0.0
        # means a zero-response row.
        arrays["jac_row_method"] = np.array(res.rows.method, dtype="U16")
        arrays["fd_h"] = np.array(res.rows.fd_h, dtype=np.float64)
        arrays["fd_err"] = np.array(res.rows.fd_err, dtype=np.float64)
    # atomic: two same-key runs can write at once and a direct savez would let
    # a reader see a partial zip, or poison the key on a kill mid-write
    _ins.atomic_write(out, lambda fh: np.savez_compressed(fh, **arrays))
    return out


def run_model(params: dict, log: Log = print) -> Path:
    """Run the heavy pipeline for ``params`` and write its model_cache entry.

    Every certificate and gate raises before that entry is written: an uncertified spectrum or Jacobian row is never
    cached.

    Parameters
    ----------
    params : dict
        Run settings; ``canonical_params`` refuses unknown keys and fills the rest.
    log : Log
        Receives progress lines in the "[fwd] PROG" protocol.

    Returns
    -------
    Path
        The written npz; ``model_cache.load_result`` reads it.
    """
    cp = canonical_params(params)
    _archive_uploaded_table(params, cp, log)
    advance, finish = _make_progress(cp, log)
    pic = cp["chem_provider"] == "picaso"
    if pic:
        advance()  # stage 0 is the PICASO solve, which runs in assemble
    assemble = _pe.assemble if pic else assemble_chem
    A = assemble(cp, log)
    # heavy imports AFTER the assembler: vulcan_chem must init env/x64 first
    import jax.numpy as jnp

    if not pic:
        advance()
    log("[fwd] building chemistry model ...")
    chem = A.build_chem()
    bz_build = float("nan")
    if cp["jac_method"] == "ad" and "dlnCO" in cp["fisher_params"]:
        # Straight after the build that sets co_bz_bound, so the refusal costs no solve; re-checked on the converged
        # column the tangent actually starts from.
        bz_build = check_ad_co_margin(chem, cp["co_ratio"])
    T_check = check_t_window(A.tp_eval, A.theta, chem.p_bar, log, T_base=chem.T_base)
    setup = build_rt_setup(cp, A.profile, A.tp_eval, A.config, chem, advance, log)
    th0 = jnp.asarray(A.theta)
    col = _baseline_column(params, cp, chem, A, th0, setup, advance, log)
    unmodeled, unmodeled_band = _opacity_advisory(col, setup, th0, log)
    rate_ex = _rate_advisory(cp, T_check, log)
    wo_list = list(cp["wo_mols"])
    evidence = None
    if setup.emis is not None:
        depth, depth_wo, evidence = emission_spectra(setup, col.column, col.y_sol, th0, wo_list, advance, log)
    else:
        depth, depth_wo = transmission_spectra(col.column, col.y_sol, th0, wo_list, advance, log)
    rows = None
    if cp["fisher_params"]:
        baseline = Baseline(
            cp=cp,
            pic=pic,
            theta=A.theta,
            theta_names=A.theta_names,
            th0=th0,
            build_chem=A.build_chem,
            abundance_overrides=A.abundance_overrides,
            assemble=assemble,
            setup=setup,
            chem=col.chem,
            column=col.column,
            y_sol=col.y_sol,
            species=col.species,
            run_frq=col.run_frq,
            escalated=col.escalated,
            bz_build=bz_build,
            advance=advance,
            log=log,
            picaso_retried=A.retried if pic else [],
            unmodeled_band=unmodeled_band,
            unmodeled=unmodeled,
        )
        rows = build_jacobian(baseline, depth.shape[0])
    if pic:
        retried = [f"{stage} {n}" for stage, n in A.retried if n]
        log(
            f"[fwd] PICASO: {sum(n for _, n in A.retried)} layer solves converged only at a shifted temperature "
            "(picaso's retry ladder, at most 1e-4 of T)" + (f": {', '.join(retried)}" if retried else "")
        )
    res = _RunResult(T_check, depth, depth_wo, evidence, rows, unmodeled, rate_ex, bz_build, wo_list)
    out = _write_result(params, cp, A, setup, col, res)
    finish()
    log(f"[fwd] cached -> {out.name}")
    return out


def main() -> None:
    from jwst_tool import proc

    params = json.loads(Path(sys.argv[1]).read_text())
    proc.worker_prologue()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    run_model(params, log=lambda *a: print(*a, flush=True))
    print("[fwd] DONE", flush=True)


if __name__ == "__main__":
    main()
