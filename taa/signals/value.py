"""Valuation signal: long-horizon reversal.

Cross-asset value is hard to measure cleanly from prices alone, because the true
fundamental anchor (earnings yield, real yield, roll yield) differs by sleeve and is
not in a total-return price panel. The standard price-based stand-in, and the one
Asness-Moskowitz-Pedersen ("Value and Momentum Everywhere", 2013) use for asset
classes, is the negative of the multi-year trailing return: what has been cheap
tends to have fallen, what is rich has run up, and both mean-revert over long
horizons. So value here is:

    score_asset = -(trailing `lookback`-month total return)

skipping the most recent `skip` months so it does not collide with (and partly
cancel) 12-month momentum. Higher score = cheaper = more attractive. A five-year
(60-month) window with a 12-month skip is the conventional choice.

This is a genuine, documented signal with a real economic rationale, but be honest
about what it is: a coarse reversal proxy, not a fundamental valuation. On the
real-data path, prefer a true carry/yield anchor (see carry.py) where one exists.
"""
from __future__ import annotations

import pandas as pd

from taa.data.panel import as_bundle


class Valuation:
    def __init__(self, lookback: int = 60, skip: int = 12, name: str | None = None):
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.name = name or f"value_rev_{lookback}m"

    def score(self, data, as_of: pd.Timestamp) -> pd.Series:
        prices = as_bundle(data).prices
        h = prices.history(as_of)                     # rows on or before as_of only
        need = self.lookback + self.skip + 1
        if len(h) < need:
            return pd.Series(0.0, index=prices.assets)
        recent = h.iloc[-(self.skip + 1)]             # end the window `skip` months back
        past = h.iloc[-(self.lookback + self.skip + 1)]
        long_run_return = recent / past - 1.0
        return (-long_run_return).astype(float)       # cheap (down a lot) scores high
