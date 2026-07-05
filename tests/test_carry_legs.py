"""Offline checks for the equity/commodity carry-leg computations.

These are the pure functions behind the real-data carry panel. They need no network,
so we can pin their arithmetic and their sign with constructed price paths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taa.data.loaders import trailing_income_yield, roll_yield_proxy


def _months(n):
    return pd.date_range("2007-01-31", periods=n, freq="ME")


def test_trailing_dividend_yield_recovers_the_income():
    d = _months(24)
    # Price flat; total return compounds at 0.5%/month => pure 0.5%/month dividend.
    pr = pd.DataFrame({"X": np.full(24, 100.0)}, index=d)
    tr = pd.DataFrame({"X": 100.0 * (1.005 ** np.arange(24))}, index=d)
    dy = trailing_income_yield(tr, pr, window=12)
    assert abs(dy["X"].iloc[-1] - 0.06) < 1e-6          # 12 * 0.5% = 6% annual
    assert np.isnan(dy["X"].iloc[10])                   # not enough history before month 12


def test_roll_yield_proxy_sign_and_size():
    d = _months(24)
    cash = pd.Series(0.024, index=d)                    # 2.4%/yr collateral => 0.2%/mo
    # Backwardation: futures outrun spot beyond collateral => positive roll.
    fut = pd.Series(100.0 * (1.010 ** np.arange(24)), index=d)
    spot = pd.Series(100.0 * (1.007 ** np.arange(24)), index=d)
    roll = roll_yield_proxy(fut, spot, cash, window=12)
    assert abs(roll.iloc[-1] - 0.012) < 1e-6            # (1.0% - 0.7% - 0.2%) * 12

    # Contango: spot outruns futures => negative roll.
    fut2 = pd.Series(100.0 * (1.005 ** np.arange(24)), index=d)
    spot2 = pd.Series(100.0 * (1.010 ** np.arange(24)), index=d)
    roll2 = roll_yield_proxy(fut2, spot2, cash, window=12)
    assert roll2.iloc[-1] < 0
