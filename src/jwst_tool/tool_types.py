"""Type aliases shared by signatures across the package. Import under TYPE_CHECKING."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import jax
    import numpy as np

Log: TypeAlias = Callable[[str], None]  # one line of worker progress
KeyFn: TypeAlias = Callable[[str], str]  # a widget-key namespacer
AnyArray: TypeAlias = "np.ndarray | jax.Array"  # the forward model passes both across a jvp
Slot: TypeAlias = "DeltaGenerator"  # a reserved place on the Streamlit page

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator
