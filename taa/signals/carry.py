"""Carry signal: hold the sleeves that pay you to wait.

Carry is the return you earn if prices don't move: a bond's yield, a credit
sleeve's spread, cash's bill rate. It reads a dedicated per-asset carry panel
(bundle.carry), NOT prices, because carry is a yield/spread level, not a price
move. Higher carry = more attractive, so the raw level is the score.

Honest scope. Carry is cleanly defined for fixed income, credit, and cash, where a
published yield or spread exists. For equities (earnings/dividend yield) and
commodities (roll yield) a clean, free, point-in-time series is harder to get, so
on the real-data path those sleeves may be absent from the carry panel; this signal
scores them NaN and lets the tilt layer demean/ignore them rather than inventing a
number. On the offline synthetic path every sleeve has a carry level so the
machinery is exercised end to end.

Two conventions are supported:
  level : score = latest carry level (the pure cross-sectional carry bet).
  change: score = latest level minus its `change_window`-month-ago level
          (carry momentum / "is carry improving"), a milder, less static tilt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taa.data.panel import as_bundle


class Carry:
    def __init__(self, mode: str = "level", change_window: int = 12, name: str | None = None):
        if mode not in ("level", "change"):
            raise ValueError("mode must be 'level' or 'change'")
        self.mode = mode
        self.change_window = int(change_window)
        self.name = name or f"carry_{mode}"

    def score(self, data, as_of: pd.Timestamp) -> pd.Series:
        bundle = as_bundle(data)
        assets = bundle.assets
        if bundle.carry is None:
            raise ValueError("Carry signal needs a carry panel; got a bundle with carry=None")

        h = bundle.carry.history(as_of)               # yields/spreads on or before as_of
        if len(h) == 0:
            return pd.Series(0.0, index=assets)

        if self.mode == "level":
            level = h.iloc[-1]
            return level.reindex(assets).astype(float)

        if len(h) <= self.change_window:
            return pd.Series(0.0, index=assets)
        now = h.iloc[-1]
        past = h.iloc[-(self.change_window + 1)]
        return (now - past).reindex(assets).astype(float)
