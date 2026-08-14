"""Materialize every opacity line-list used by the shipped molecule table.

Run in the core environment with ``VULCAN_FORWARD_DATA`` set. Constructing the
database objects exercises ExoJAX's normal downloader and parser without
building the much more expensive PreMODIT kernels.
"""
from __future__ import annotations

import numpy as np
from exojax.database.api import MdbExomol, MdbHitran

from vulcan_forward import constants, paths


def main() -> None:
    paths.ensure_layout()
    full_band = np.array(
        [constants.WIDE_BAND_NU_MIN, constants.WIDE_BAND_NU_MAX], dtype=float)
    for molecule, spec in constants.MOLECULES.items():
        source = spec["source"]
        location = paths.resolve_db(spec["db"], source)
        print(f"{molecule}: {source} -> {location}", flush=True)
        if source in {"exomol", "exomol_cached"}:
            database = MdbExomol(location, nurange=full_band)
        elif source == "hitran":
            database = MdbHitran(location, nurange=full_band, isotope=1)
        else:
            raise ValueError(f"unsupported opacity source {source!r}")
        n_lines = int(np.asarray(database.nu_lines).size)
        if n_lines < 1:
            raise RuntimeError(f"{molecule}: database contains no in-band lines")
        print(f"{molecule}: {n_lines} in-band lines", flush=True)


if __name__ == "__main__":
    main()
