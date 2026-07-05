"""Black-Litterman construction checks.

Most important property: with no view (flat scores) the posterior collapses to the
equilibrium and the optimizer returns the policy. That is what makes it an overlay
rather than a free-running optimizer. Also verify constraints and no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taa.construction.black_litterman import BlackLitterman
from taa.construction.policy import PolicyPortfolio
from taa.data.panel import PriceHistory
from taa.data.synthetic import make_synthetic_prices, DEFAULT_ASSETS

POLICY = PolicyPortfolio({
    "EQ_US": 0.35, "EQ_INTL": 0.15, "GOVT": 0.20, "CREDIT": 0.10,
    "TIPS": 0.05, "CMDTY": 0.05, "CASH": 0.10,
})
PRICES = make_synthetic_prices()
AS_OF = PRICES.dates[120]


def _policy_series():
    return POLICY.target(DEFAULT_ASSETS)


def test_no_view_returns_policy():
    bl = BlackLitterman(max_tilt=0.10)
    flat = pd.Series(0.0, index=DEFAULT_ASSETS)     # no view
    w = bl.weights(flat, _policy_series(), PRICES, AS_OF)
    assert np.allclose(w.values, _policy_series().values, atol=1e-3)


def test_constraints_respected():
    bl = BlackLitterman(max_tilt=0.10)
    rng = np.random.default_rng(0)
    scores = pd.Series(rng.standard_normal(len(DEFAULT_ASSETS)), index=DEFAULT_ASSETS)
    w = bl.weights(scores, _policy_series(), PRICES, AS_OF)
    pol = _policy_series()
    assert abs(w.sum() - 1.0) < 1e-6, "weights must sum to one"
    assert (w >= -1e-9).all(), "long-only violated"
    dev = (w - pol).abs()
    assert (dev <= 0.10 + 1e-6).all(), "per-sleeve tilt cap violated"


def test_insufficient_history_falls_back_to_policy():
    bl = BlackLitterman(cov_lookback=36)
    early = PRICES.dates[5]                          # not enough history yet
    scores = pd.Series(1.0, index=DEFAULT_ASSETS)
    w = bl.weights(scores, _policy_series(), PRICES, early)
    assert np.allclose(w.values, _policy_series().values)


def _poison_future(prices, as_of):
    df = prices.raw_copy()
    mask = df.index > as_of
    df.loc[mask] = df.loc[mask] * 1e6 + 12345.0
    return PriceHistory(df)


def test_black_litterman_ignores_future():
    bl = BlackLitterman()
    rng = np.random.default_rng(1)
    scores = pd.Series(rng.standard_normal(len(DEFAULT_ASSETS)), index=DEFAULT_ASSETS)
    base = bl.weights(scores, _policy_series(), PRICES, AS_OF)
    pois = bl.weights(scores, _policy_series(), _poison_future(PRICES, AS_OF), AS_OF)
    pd.testing.assert_series_equal(base, pois)
