from taa.backtest.costs import linear_cost
import pandas as pd
import pytest


def test_linear_cost_zero_when_no_trade():
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert linear_cost(w, w, 10.0) == 0.0


def test_linear_cost_scales_with_turnover():
    a = pd.Series({"A": 0.6, "B": 0.4})
    b = pd.Series({"A": 0.5, "B": 0.5})
    # one-way turnover = 0.1 + 0.1 = 0.2 -> 0.2 * 10bps
    assert abs(linear_cost(a, b, 10.0) - 0.2 * 10 / 1e4) < 1e-12


def test_linear_cost_per_sleeve():
    a = pd.Series({"A": 0.6, "B": 0.4})
    b = pd.Series({"A": 0.5, "B": 0.5})
    bps = pd.Series({"A": 20.0, "B": 5.0})
    # trade 0.1 in each: cost = (0.1*20 + 0.1*5) / 1e4
    assert abs(linear_cost(a, b, bps) - (0.1 * 20 + 0.1 * 5) / 1e4) < 1e-12


def test_linear_cost_missing_sleeve_raises():
    # a per-sleeve schedule that misses a sleeve must raise, not silently fill a guess
    a = pd.Series({"A": 0.6, "B": 0.4})
    b = pd.Series({"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError, match="no entry"):
        linear_cost(a, b, pd.Series({"A": 20.0}))
