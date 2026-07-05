"""Signal interface.

A signal is a pure function of data available at as_of. It returns one score per
asset, higher meaning more attractive. It receives a DataBundle and MUST read each
source only via source.history(as_of). If you find yourself wanting the full frame,
stop: that is the look-ahead the whole framework is built to prevent.

A price-only signal may be handed a bare PriceHistory instead of a bundle; call
as_bundle(data) at the top of score() so both work. Signals that need carry or macro
must be handed a bundle that supplies them.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Union

import pandas as pd

from taa.data.panel import PriceHistory, DataBundle

Data = Union[PriceHistory, DataBundle]


@runtime_checkable
class Signal(Protocol):
    name: str

    def score(self, data: Data, as_of: pd.Timestamp) -> pd.Series:
        """Return a per-asset score using only data dated on or before as_of."""
        ...
