"""Constructor interface.

A constructor turns per-asset signal scores into target weights. The engine calls
constructor.weights(scores, policy, prices, as_of). It receives prices and as_of so
a constructor that needs a covariance estimate (Black-Litterman) can build one from
data through as_of only. Reading anything after as_of is look-ahead; the no-look-ahead
harness poisons the future and checks that weights do not move.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from taa.data.panel import PriceHistory


@runtime_checkable
class Constructor(Protocol):
    name: str

    def weights(self, scores: pd.Series, policy: pd.Series, prices: PriceHistory, as_of: pd.Timestamp) -> pd.Series:
        ...
