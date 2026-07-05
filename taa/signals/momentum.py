"""The pre-committed signal set.

Two signals, both with economic rationale and literature behind them, chosen ex
ante. Do NOT quietly add a third, tune it, and report the winner. If you test
more, report all of them, and pay for the search with a deflated Sharpe.

  AbsoluteMomentum : trailing total return over `lookback` months
                     (time-series momentum, cf. Moskowitz-Ooi-Pedersen 2012).
  TrendMA          : price relative to its `window`-month moving average
                     (Faber 2007 trend rule).
"""
from __future__ import annotations

import pandas as pd

from taa.data.panel import as_bundle


class AbsoluteMomentum:
    def __init__(self, lookback: int = 12, name: str | None = None):
        self.lookback = int(lookback)
        self.name = name or f"abs_mom_{lookback}m"

    def score(self, data, as_of: pd.Timestamp) -> pd.Series:
        prices = as_bundle(data).prices
        h = prices.history(as_of)  # rows on or before as_of only
        if len(h) <= self.lookback:
            return pd.Series(0.0, index=prices.assets)
        now = h.iloc[-1]
        past = h.iloc[-(self.lookback + 1)]
        return (now / past - 1.0).astype(float)


class TrendMA:
    def __init__(self, window: int = 10, name: str | None = None):
        self.window = int(window)
        self.name = name or f"trend_ma_{window}m"

    def score(self, data, as_of: pd.Timestamp) -> pd.Series:
        prices = as_bundle(data).prices
        h = prices.history(as_of)
        if len(h) < self.window:
            return pd.Series(0.0, index=prices.assets)
        ma = h.rolling(self.window).mean().iloc[-1]
        now = h.iloc[-1]
        return (now / ma - 1.0).astype(float)
