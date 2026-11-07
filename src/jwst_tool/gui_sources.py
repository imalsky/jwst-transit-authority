"""The data panel: what is installed, and the citation table of every data source the tool reads."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import streamlit as st

from jwst_tool import datacheck, instruments as ins, model_defs, planets, provenance

if TYPE_CHECKING:
    from jwst_tool.gui_inputs import Atmosphere, Target
    from jwst_tool.tool_types import Slot


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


def _row(component: str, used: bool, data_set: str, doi: str, page: str) -> dict:
    return {
        "component": component,
        "used in this setup": used,
        "data set": data_set,
        "source DOI": doi,
        "source page": page,
    }


_STSCI_CATALOGS = (
    "https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/"
)
# Read by every run.
_ALWAYS_READ = (
    (
        "Stellar spectrum grid",
        "PHOENIX (Allard et al., synphot CDBS)",
        "10.1098/rsta.2011.0269",
        _STSCI_CATALOGS + "phoenix-models-available-in-synphot",
    ),
    (
        "Ks bandpass (flux normalization)",
        "2MASS",
        "10.1086/498708",
        "https://irsa.ipac.caltech.edu/Missions/2mass.html",
    ),
    (
        "Flux standard (Vega)",
        "CALSPEC alpha_lyr_stis_011",
        "10.1086/677655",
        _STSCI_CATALOGS + "calspec",
    ),
    (
        "JWST exposure & noise model; instrument bands, dispersion and native resolving power",
        f"Pandeia {ins.BACKEND_RELEASE} (incl. dispersion files)",
        "10.1117/12.2231768",
        "https://jwst-docs.stsci.edu/jwst-exposure-time-calculator-overview/jwst-etc-pandeia-engine-tutorial",
    ),
)


def _picaso_rows(picaso: bool, climate: bool) -> list[dict]:
    return [
        _row(
            "Climate model (radiative-convective equilibrium)",
            climate,
            "PICASO 4.1 climate (Mukherjee et al. 2023)",
            "10.3847/1538-4357/ac9f48",
            "https://github.com/natashabatalha/picaso/tree/v4.1",
        ),
        _row(
            "Equilibrium chemistry (PICASO engine)",
            picaso,
            "photochem 0.9.2 (Wogan et al. 2025)",
            "10.3847/PSJ/ae0e1c",
            "https://github.com/Nicholaswogan/photochem",
        ),
        _row(
            "Climate opacities (correlated-k)",
            climate,
            "PICASO resort-rebin tables, 661 bins",
            "10.5281/zenodo.18644980",
            "https://zenodo.org/records/18644980",
        ),
        _row(
            "Solar abundances (PICASO engine)",
            picaso,
            "Lodders (2020) Table 8, present-day solar system",
            "10.1093/acrefore/9780190647926.013.145",
            "https://doi.org/10.1093/acrefore/9780190647926.013.145",
        ),
    ]


def _source_rows(ksrc: dict, t: Target, a: Atmosphere) -> list[dict]:
    """One row per data source the tool reads.

    The offered k-tables (base first), the continuum terms, and the non-opacity data files. Every row carries a
    resolvable DOI and page; `used in this setup` marks the ones this configuration actually reads. The k-table band
    grid is one line in the Opacity panel rather than a column repeating 25 identical values.
    """
    rows = []
    for m in list(a.base_set) + list(a.extra_set):
        r = ksrc.get(m, {})
        iso = "natural abundance" if r.get("natural_abundance") else r.get("iso", "")
        rows.append(
            _row(
                m,
                m in a.base_set or m in a.extra_mols,
                f"ExoMol {r['dataset']} ({iso})" if r else "",
                r.get("doi") or "",
                r.get("url", ""),
            )
        )
    rayleigh_on = a.clouds.use_rayleigh and t.science_mode == "transmission"
    rows += [
        _row("H2-H2 CIA", True, *_CIA_SRC),
        _row("H2-He CIA", True, *_CIA_SRC),
        _row("Rayleigh scattering (H2, He)", bool(rayleigh_on), "ExoJAX H2/He polarizabilities", *_EXOJAX_SRC[1:]),
        # Cites the code that runs: exojax.atm.simple_clouds.powerlaw_clouds.
        # Its alpha is -gamma of the petitRADTRANS definition
        # (Molliere et al. 2019, 10.1051/0004-6361/201935470).
        _row(
            "Power-law cloud/haze",
            bool(a.clouds.cloud_on),
            "ExoJAX powerlaw_clouds (kappa_cloud, alpha)",
            *_EXOJAX_SRC[1:],
        ),
    ]
    picaso = a.chem_provider == "picaso"
    for fname, label in planets.SFLUX_CHOICES.items():
        used = bool(a.photo.use_photo) and a.chem_provider == "vulcan" and fname == t.sflux
        rows.append(_row(f"Stellar UV — {label}", used, *planets.SFLUX_SOURCES[fname]))
    rows += _picaso_rows(picaso, picaso and a.profile.tp_mode == "picaso_climate")
    rows += [_row(component, True, *rest) for component, *rest in _ALWAYS_READ]
    return rows


def render_source_table(opacity_slot: Slot, ksrc: dict, ksrc_err: str | None, t: Target, a: Atmosphere) -> None:
    """Fill the data panel's citation table for the current settings."""
    with opacity_slot:
        st.markdown("#### Data sources")
        if ksrc_err:
            st.warning(KSRC_WARN + ksrc_err)
        else:
            st.dataframe(
                _source_rows(ksrc, t, a),
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
