"""Rebalance loop with weight drift, explicit costs, and turnover limits.

Timing convention (no look-ahead): at each rebalance date t we build weights from
data through t, then earn the NEXT period's asset returns on those weights, minus
the cost of trading from the drifted book into the new target. The policy
benchmark is run through the identical loop and cost model, so the difference
between them is a clean read on the overlay.

Turnover limits: a real desk cannot trade its full target every month. Two optional
governors sit between the constructor's target and what actually trades:
  - no_trade_band: if the required one-way turnover is below the band, hold the
    drifted book and skip the rebalance (don't churn on noise).
  - max_turnover : if the required turnover exceeds the cap, only partially
    rebalance toward the target (a proportional step), which bounds cost drag.
Both preserve long-only, sum-to-one weights and default to off so the plain backtest
is unchanged. Realized turnover/cost are reported AFTER these limits apply. The
governors apply to the POLICY BENCHMARK too: a benchmark forced to churn every tiny
drift correction the strategy is allowed to skip would pay costs the strategy avoids,
quietly flattering the overlay's active return.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taa.construction.tilt import SimpleTilt
from taa.data.panel import as_bundle
from .costs import linear_cost


@dataclass
class BacktestResult:
    strat_returns: pd.Series
    policy_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series

    @property
    def active_returns(self) -> pd.Series:
        return (self.strat_returns - self.policy_returns).dropna()


def _drift(weights: pd.Series, period_return: pd.Series) -> pd.Series:
    grown = weights * (1.0 + period_return)
    total = grown.sum()
    return grown / total if total > 0 else weights


def _limit_turnover(target, drifted, max_turnover, no_trade_band):
    """Return the weights actually held after applying the turnover governors.

    target and drifted both sum to one, so any convex step between them does too,
    and long-only is preserved because both endpoints are long-only.
    """
    gross = float((target - drifted).abs().sum())
    if no_trade_band and gross < no_trade_band:
        return drifted.copy()                       # below the band: don't rebalance
    if max_turnover is not None and gross > max_turnover and gross > 0:
        lam = max_turnover / gross                  # partial step toward target
        return drifted + lam * (target - drifted)
    return target.copy()


def run_backtest(
    data,
    signal,
    policy,
    max_tilt: float = 0.10,
    scale: float = 0.05,
    cost_bps: float = 10.0,
    warmup: int = 12,
    constructor=None,
    max_turnover: float | None = None,
    no_trade_band: float = 0.0,
    rebalance_every: int = 1,
) -> BacktestResult:
    """Run the overlay backtest.

    data: a PriceHistory or a DataBundle. A bare PriceHistory is treated as a bundle
    whose only source is prices, so price-only signals and existing call sites are
    unchanged; carry/macro signals need a bundle that supplies those panels.

    constructor: any object exposing weights(scores, policy, prices, as_of). If None,
    a SimpleTilt is built from max_tilt/scale so existing call sites are unchanged.
    Pass a BlackLitterman instance to swap in constrained mean-variance construction;
    max_tilt/scale are then ignored (the constructor owns those choices).

    cost_bps: a scalar (same cost per sleeve) or a per-sleeve pd.Series.

    max_turnover / no_trade_band: optional turnover governors (see module docstring).

    rebalance_every: rebalance the overlay (and the policy benchmark) only every k
    months; hold and drift in between. k=1 is monthly (unchanged). Quarterly (k=3)
    cuts turnover, which matters because much monthly turnover is uncompensated.
    """
    if constructor is None:
        constructor = SimpleTilt(max_tilt=max_tilt, scale=scale)

    bundle = as_bundle(data)
    prices = bundle.prices
    dates = prices.dates
    assets = prices.assets
    pol = policy.target(assets)

    strat_w = pol.copy()
    pol_w = pol.copy()

    strat_ret, pol_ret, turn, cost_series = {}, {}, {}, {}
    weights_log = {}

    # Rebalance at t, earn return over (t, t+1]. Stop one short of the end.
    for i in range(warmup, len(dates) - 1):
        t = dates[i]
        t_next = dates[i + 1]
        nxt_ret = (prices.history(t_next).iloc[-1] / prices.history(t).iloc[-1] - 1.0)
        if nxt_ret.isna().any():
            # pandas .sum() would silently skip a NaN sleeve while its weight is still
            # held, understating the book. Fail loudly instead.
            bad = list(nxt_ret.index[nxt_ret.isna()])
            raise ValueError(
                f"NaN return over ({t.date()} -> {t_next.date()}] for {bad}; "
                f"increase warmup past those assets' first valid dates or clean the panel"
            )
        do_rebalance = ((i - warmup) % rebalance_every == 0)

        # strategy: on a rebalance month build the target and trade into it (subject to
        # the turnover limits); otherwise hold the drifted book and pay nothing.
        if do_rebalance:
            scores = signal.score(bundle, t)
            target = constructor.weights(scores, pol, prices, t)
            held = _limit_turnover(target, strat_w, max_turnover, no_trade_band)
        else:
            held = strat_w
        c = linear_cost(held, strat_w, cost_bps)
        turn[t] = (held - strat_w).abs().sum()      # realized turnover, after the limits
        cost_series[t] = c
        weights_log[t] = held
        strat_ret[t_next] = float((held * nxt_ret).sum()) - c
        strat_w = _drift(held, nxt_ret)

        # policy benchmark: identical loop, cost model, rebalance schedule, AND
        # turnover governors, so the two legs face the same trading frictions
        pol_held = _limit_turnover(pol, pol_w, max_turnover, no_trade_band) if do_rebalance else pol_w
        pc = linear_cost(pol_held, pol_w, cost_bps)
        pol_ret[t_next] = float((pol_held * nxt_ret).sum()) - pc
        pol_w = _drift(pol_held, nxt_ret)

    return BacktestResult(
        strat_returns=pd.Series(strat_ret).sort_index(),
        policy_returns=pd.Series(pol_ret).sort_index(),
        weights=pd.DataFrame(weights_log).T.sort_index(),
        turnover=pd.Series(turn).sort_index(),
        costs=pd.Series(cost_series).sort_index(),
    )
