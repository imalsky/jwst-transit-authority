# Vendored from https://raw.githubusercontent.com/HajimeKawahara/exojax/master/src/exojax/provider/exomolop.py
# at a revision after PR #730 (bugfix batch); the reader only, the downloader
# is dropped. The independent ExoMolOP reader of fig_ckd_verification_vs_exojax_exok.py.
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from exojax.utils.molname import e2s


def infer_molecule_name_from_path(path: Path):
    """Infer a simple molecule name from an ExoMolOP table path."""
    path = Path(path)
    try:
        return e2s(path.parent.parent.name)
    except Exception:
        return path.parent.parent.name


def _decode_scalar_text_dataset(dataset):
    """Decode a scalar or one-element byte/string HDF5 dataset."""
    value = np.asarray(dataset[()]).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_ckd(path: Path):
    """Load a correlated-k opacity file and return metadata and the cross-section grid."""
    path = Path(path)
    with h5py.File(path, "r") as fh5:
        if "mol_name" in fh5:
            molecule = _decode_scalar_text_dataset(fh5["mol_name"])
        else:
            molecule = infer_molecule_name_from_path(path)
        mol_mass = float(fh5["mol_mass"][()][0])
        wavenumber = fh5["bin_centers"][:]  # cm-1
        samples = fh5["samples"][:]  # g-ordinates
        weights = fh5["weights"][:]
        temperatures = fh5["t"][:]  # K
        pressures = fh5["p"][:]  # bar
        kcoeff = np.asarray(fh5["kcoeff"], dtype=float)

    # reshape to (T, P, g, wavenumber)
    xsgrid = np.swapaxes(kcoeff, 0, 1)
    xsgrid = np.swapaxes(xsgrid, 2, 3)

    # Replace exact zeros so downstream log/positivity checks stay well-defined.
    tiny = np.finfo(xsgrid.dtype).tiny
    xsgrid = np.where(xsgrid == 0, tiny, xsgrid)

    return xsgrid, samples, weights, temperatures, pressures, wavenumber, molecule, mol_mass
