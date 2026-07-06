"""Transaction cost model.

Linear cost on traded notional. cost = sum_i |dw_i| * bps_i / 1e4, where dw is the
one-way weight change at a rebalance. `cost_bps` may be a single number (the same
cost for every sleeve) or a per-sleeve Series, because a real book pays far more to
trade high-yield or commodities than Treasuries or cash. A flat 5-15 bps one-way is
a defensible placeholder for liquid ETFs; a per-sleeve vector is more honest. Always
run the whole backtest at several cost levels and show the sensitivity; monthly
signals can generate enough turnover to eat the entire edge.
"""
from __future__ import annotations

import pandas as pd


def linear_cost(target: pd.Series, drifted: pd.Series, cost_bps) -> float:
    trade = (target - drifted).abs()
    if isinstance(cost_bps, pd.Series):
        bps = cost_bps.reindex(trade.index)
        if bps.isna().any():
            # refuse to guess a missing sleeve's cost — a silent fill would make the
            # cost assumption unauditable
            missing = list(bps.index[bps.isna()])
            raise ValueError(f"per-sleeve cost_bps has no entry for {missing}")
        return float((trade * bps).sum() / 1e4)
    return float(trade.sum() * cost_bps / 1e4)
