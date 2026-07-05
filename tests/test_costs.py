from taa.backtest.costs import linear_cost
import pandas as pd


def test_linear_cost_zero_when_no_trade():
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert linear_cost(w, w, 10.0) == 0.0


def test_linear_cost_scales_with_turnover():
    a = pd.Series({"A": 0.6, "B": 0.4})
    b = pd.Series({"A": 0.5, "B": 0.5})
    # one-way turnover = 0.1 + 0.1 = 0.2 -> 0.2 * 10bps
    assert abs(linear_cost(a, b, 10.0) - 0.2 * 10 / 1e4) < 1e-12
