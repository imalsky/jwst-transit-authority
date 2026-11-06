"""The data panel: what is installed, and the citation table of every data source the tool reads."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import streamlit as st

from jwst_tool import datacheck, instruments as ins, model_defs, planets, provenance

if TYPE_CHECKING:
    from jwst_tool.gui_inputs import Atmosphere, Target


# Data availability -- detected live. The GUI reports only what BLOCKS a run;
# the full install inventory is `jwst-tool data`.
def data_label(it: datacheck.Item) -> str:
    """Public name without installation details or parenthetical clauses."""
    if it.key.startswith("pkg:"):
        return f"{it.key.removeprefix('pkg:')} Python package"
    labels = {
        "engine:config": "Opacity data folder",
        "cia:H2-H2": "H2-H2 absorption table",
        "cia:H2-He": "H2-He absorption table",
        "ktable:provenance": "ExoMolOP source record",
        "uv:package": "Stellar UV data folder",
        "pandeia:python": "Pandeia engine",
        "pandeia:refdata": "Pandeia JWST reference data",
        "pandeia:psf": "Pandeia PSF data",
        "cdbs:phoenix": "PHOENIX stellar spectra",
        "picaso:python": "PICASO engine",
        "picaso:refdata": "PICASO reference data",
        "picaso:ck": "PICASO climate opacity tables",
    }
    if it.key in labels:
        return labels[it.key]
    if it.key.startswith("ktable:"):
        return f"{it.key.removeprefix('ktable:')} ExoMolOP opacity table"
    return it.label.split(" (")[0]


@st.cache_data(ttl=3600, show_spinner="Checking installed data ...")
def _cached_full_report() -> dict:
    # Disk-persisted (the Space entrypoint warms it at boot) on top of the
    # in-process st.cache. Invalidation: the two 1-hour TTLs (this one and
    # datacheck.REPORT_CACHE_MAX_AGE_S).
    cached = datacheck.load_cached_report()
    if cached is not None:
        return cached
    return datacheck.warm_report_cache(base_mols=model_defs.MOLECULES, extra_mols=model_defs.EXTRA_MOLECULES)


# Opacity sources: the fetcher's provenance record + each k-table's header
@st.cache_data(show_spinner=False)
def _ktable_sources(stamp: int) -> dict:  # noqa: ARG001  (the cache key)
    # stamp = provenance.json mtime_ns, so a re-fetch invalidates the cache;
    # exceptions are not cached, so an absent file re-raises on every rerun.
    # NO leading underscore: Streamlit skips underscore-prefixed args when
    # hashing, which would make this cache permanent for the process lifetime.
    return provenance.ktable_sources()


# fixed prefix: the smoke test recognises the warning by it, whatever the
# engine's wording (no data root, no provenance.json, no table)
KSRC_WARN = "Opacity sources unavailable: "


# Every non-k-table source the tool reads, with the paper that defines it.
# DOI + page are both REQUIRED here: a row with either one blank is a citation
# the user cannot follow, and test_app_smoke pins that none is. Every DOI was
# resolved against doi.org. The k-table rows get theirs from the engine.
_CIA_SRC = ("HITRAN CIA 2011", "10.1016/j.jqsrt.2011.11.004", "https://hitran.org/cia/")


_EXOJAX_SRC = ("ExoJAX", "10.3847/1538-4365/ac3b4d", "https://github.com/HajimeKawahara/exojax")


def data_status() -> tuple[list, dict, str | None]:
    """Report what blocks a run, and read the opacity sources: (missing items, k-table sources, the read's error)."""
    data_report = _cached_full_report()
    missing_req = datacheck.missing_required(data_report)
    if missing_req:
        st.error(
            f"**Required data missing ({len(missing_req)} item"
            f"{'s' if len(missing_req) > 1 else ''}).** A run stops if it needs "
            "one of these files:\n\n"
            + "\n".join(
                f"- **{data_label(it)}**  \n  Location: {it.detail}  \n  Install: {it.remedy}" for it in missing_req
            )
            # on the hosted Space the visitor cannot install anything; the CLI
            # remedies are for local installs (SPACE_ID is set by Hugging Face)
            + (
                "\n\nThis hosted deployment is missing data it should ship with; "
                "please report it to isaacmalsky@gmail.com."
                if os.environ.get("SPACE_ID")
                else "\n\nUse `jwst-tool data` to check files and `jwst-tool fetch` to download supported data."
            )
        )

    prov_path = datacheck.exomolop_provenance_path()
    try:
        ksrc = _ktable_sources(prov_path.stat().st_mtime_ns if prov_path and prov_path.is_file() else 0)
        ksrc_err = None
    except (ImportError, RuntimeError, FileNotFoundError) as exc:
        ksrc, ksrc_err = {}, str(exc)  # the engine's text carries the remedy
    return missing_req, ksrc, ksrc_err


# size-limit-ok: one literal citation table
def _source_rows(
    ksrc: dict,
    base_set: Any,
    extra_set: list,
    extra_on: list,
    rayleigh_on: bool,
    cloud_on: Any,
    photo_on: bool,
    sflux_sel: str,
    engine: str,
    tp_mode: str,
) -> Any:
    """One row per data source the tool reads.

    The offered k-tables (base first), the continuum terms, and the non-opacity data files. Every row carries a
    resolvable DOI and page; `used in this setup` marks the ones this configuration actually reads. The k-table band
    grid is one line in the Opacity panel rather than a column repeating 25 identical values.
    """
    rows = []
    for m in list(base_set) + list(extra_set):
        r = ksrc.get(m, {})
        iso = "natural abundance" if r.get("natural_abundance") else r.get("iso", "")
        rows.append(
            {
                "component": m,
                "used in this setup": m in base_set or m in extra_on,
                "data set": f"ExoMol {r['dataset']} ({iso})" if r else "",
                "source DOI": r.get("doi") or "",
                "source page": r.get("url", ""),
            }
        )
    rows += [
        {
            "component": "H2-H2 CIA",
            "used in this setup": True,
            "data set": _CIA_SRC[0],
            "source DOI": _CIA_SRC[1],
            "source page": _CIA_SRC[2],
        },
        {
            "component": "H2-He CIA",
            "used in this setup": True,
            "data set": _CIA_SRC[0],
            "source DOI": _CIA_SRC[1],
            "source page": _CIA_SRC[2],
        },
        {
            "component": "Rayleigh scattering (H2, He)",
            "used in this setup": bool(rayleigh_on),
            "data set": "ExoJAX H2/He polarizabilities",
            "source DOI": _EXOJAX_SRC[1],
            "source page": _EXOJAX_SRC[2],
        },
        # Cites the code that runs: exojax.atm.simple_clouds.powerlaw_clouds.
        # Its alpha is -gamma of the petitRADTRANS definition
        # (Molliere et al. 2019, 10.1051/0004-6361/201935470).
        {
            "component": "Power-law cloud/haze",
            "used in this setup": bool(cloud_on),
            "data set": "ExoJAX powerlaw_clouds (kappa_cloud, alpha)",
            "source DOI": _EXOJAX_SRC[1],
            "source page": _EXOJAX_SRC[2],
        },
    ]
    for fname, label in planets.SFLUX_CHOICES.items():
        ds, doi, url = planets.SFLUX_SOURCES[fname]
        rows.append(
            {
                "component": f"Stellar UV — {label}",
                "used in this setup": (bool(photo_on) and engine == "vulcan" and fname == sflux_sel),
                "data set": ds,
                "source DOI": doi,
                "source page": url,
            }
        )
    climate = engine == "picaso" and tp_mode == "picaso_climate"
    rows += [
        {
            "component": "Climate model (radiative-convective equilibrium)",
            "used in this setup": climate,
            "data set": "PICASO 4.1 climate (Mukherjee et al. 2023)",
            "source DOI": "10.3847/1538-4357/ac9f48",
            "source page": "https://github.com/natashabatalha/picaso/tree/v4.1",
        },
        {
            "component": "Equilibrium chemistry (PICASO engine)",
            "used in this setup": engine == "picaso",
            "data set": "photochem 0.9.2 (Wogan et al. 2025)",
            "source DOI": "10.3847/PSJ/ae0e1c",
            "source page": "https://github.com/Nicholaswogan/photochem",
        },
        {
            "component": "Climate opacities (correlated-k)",
            "used in this setup": climate,
            "data set": "PICASO resort-rebin tables, 661 bins",
            "source DOI": "10.5281/zenodo.18644980",
            "source page": "https://zenodo.org/records/18644980",
        },
        {
            "component": "Solar abundances (PICASO engine)",
            "used in this setup": engine == "picaso",
            "data set": "Lodders (2020) Table 8, present-day solar system",
            "source DOI": "10.1093/acrefore/9780190647926.013.145",
            "source page": "https://doi.org/10.1093/acrefore/9780190647926.013.145",
        },
    ]
    rows += [
        {
            "component": "Stellar spectrum grid",
            "used in this setup": True,
            "data set": "PHOENIX (Allard et al., synphot CDBS)",
            "source DOI": "10.1098/rsta.2011.0269",
            "source page": "https://www.stsci.edu/hst/instrumentation/"
            "reference-data-for-calibration-and-tools/"
            "astronomical-catalogs/"
            "phoenix-models-available-in-synphot",
        },
        {
            "component": "Ks bandpass (flux normalization)",
            "used in this setup": True,
            "data set": "2MASS",
            "source DOI": "10.1086/498708",
            "source page": "https://irsa.ipac.caltech.edu/Missions/2mass.html",
        },
        {
            "component": "Flux standard (Vega)",
            "used in this setup": True,
            "data set": "CALSPEC alpha_lyr_stis_011",
            "source DOI": "10.1086/677655",
            "source page": "https://www.stsci.edu/hst/instrumentation/"
            "reference-data-for-calibration-and-tools/"
            "astronomical-catalogs/calspec",
        },
        {
            "component": "JWST exposure & noise model; instrument bands, dispersion and native resolving power",
            "used in this setup": True,
            "data set": f"Pandeia {ins.BACKEND_RELEASE} (incl. dispersion files)",
            "source DOI": "10.1117/12.2231768",
            "source page": "https://jwst-docs.stsci.edu/"
            "jwst-exposure-time-calculator-overview/"
            "jwst-etc-pandeia-engine-tutorial",
        },
    ]
    return rows


def render_source_table(opacity_slot: Any, ksrc: dict, ksrc_err: str | None, t: Target, a: Atmosphere) -> None:
    """Fill the data panel's citation table for the current settings."""
    with opacity_slot:
        st.markdown("#### Data sources")
        if ksrc_err:
            st.warning(KSRC_WARN + ksrc_err)
        else:
            st.dataframe(
                _source_rows(
                    ksrc,
                    a.base_set,
                    a.extra_set,
                    a.extra_mols,
                    a.use_rayleigh and t.science_mode == "transmission",
                    a.cloud_on,
                    a.use_photo,
                    t.sflux,
                    a.chem_provider,
                    a.tp_mode,
                ),
                column_config={
                    "used in this setup": st.column_config.CheckboxColumn("used in this setup"),
                    # No display_text: it is a regex over the URL, and these
                    # pages span ExoMol, HITRAN, STScI and others.
                    "source page": st.column_config.LinkColumn("source page"),
                },
                width="stretch",
                hide_index=True,
            )
            # Never turn incompatible table domains into a plausible union.
            if ksrc and len({r["grid_sha256"] for r in ksrc.values()}) != 1:
                st.error("Incompatible k-table grids installed.")
