"""Transaction cost model.

Flat linear cost on traded notional. cost = turnover * cost_bps / 1e4, where
turnover is one-way sum of absolute weight changes at a rebalance. A flat 5 to
15 bps one-way is a defensible placeholder for liquid ETFs. Always run the whole
backtest at several cost levels and show the sensitivity; monthly signals can
generate enough turnover to eat the entire edge.
"""
from __future__ import annotations

import pandas as pd


def linear_cost(target: pd.Series, drifted: pd.Series, cost_bps: float) -> float:
    turnover = (target - drifted).abs().sum()
    return turnover * cost_bps / 1e4
