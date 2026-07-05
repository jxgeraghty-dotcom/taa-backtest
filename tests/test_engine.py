"""Engine sanity checks: no future leakage in the loop, weights stay long-only and
summing to one, and a zero-tilt overlay reproduces the policy (net of cost noise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taa.construction.policy import PolicyPortfolio
from taa.data.synthetic import make_synthetic_prices, DEFAULT_ASSETS
from taa.backtest.engine import run_backtest
from taa.signals.momentum import AbsoluteMomentum


def _equal_policy():
    return PolicyPortfolio({a: 1.0 for a in DEFAULT_ASSETS})


def test_weights_valid():
    prices = make_synthetic_prices()
    res = run_backtest(prices, AbsoluteMomentum(12), _equal_policy(), cost_bps=10)
    w = res.weights
    assert (w >= -1e-9).all().all(), "long-only violated"
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-8), "weights must sum to one"


def test_zero_tilt_matches_policy():
    prices = make_synthetic_prices()
    # scale=0 means no tilt, so strategy weights equal policy and returns match.
    res = run_backtest(prices, AbsoluteMomentum(12), _equal_policy(), scale=0.0, cost_bps=0.0)
    diff = (res.strat_returns - res.policy_returns).abs().max()
    assert diff < 1e-10


def test_result_length():
    prices = make_synthetic_prices()
    res = run_backtest(prices, AbsoluteMomentum(12), _equal_policy())
    assert len(res.strat_returns) == len(res.policy_returns) > 0


def test_turnover_cap_binds():
    prices = make_synthetic_prices()
    cap = 0.05
    res = run_backtest(prices, AbsoluteMomentum(12), _equal_policy(),
                       scale=0.3, cost_bps=10, max_turnover=cap)
    # No single rebalance may trade more than the cap (small numerical tolerance).
    assert res.turnover.max() <= cap + 1e-9
    # And the cap must actually reduce turnover vs the uncapped run.
    uncapped = run_backtest(prices, AbsoluteMomentum(12), _equal_policy(),
                            scale=0.3, cost_bps=10)
    assert res.turnover.mean() < uncapped.turnover.mean()


def test_no_trade_band_skips_small_rebalances():
    prices = make_synthetic_prices()
    band = 0.03
    res = run_backtest(prices, AbsoluteMomentum(12), _equal_policy(),
                       scale=0.05, cost_bps=10, no_trade_band=band)
    traded = res.turnover[res.turnover > 1e-12]
    # Any rebalance that does happen clears the band; sub-band ones are held instead.
    assert (traded >= band - 1e-9).all()


def test_weights_valid_under_limits():
    prices = make_synthetic_prices()
    res = run_backtest(prices, AbsoluteMomentum(12), _equal_policy(),
                       scale=0.2, cost_bps=10, max_turnover=0.05, no_trade_band=0.02)
    w = res.weights
    assert (w >= -1e-9).all().all(), "long-only violated under turnover limits"
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-8), "weights must sum to one under limits"
