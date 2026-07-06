"""Loader plumbing checks that run offline (no network).

The one that matters: sleeve names must be attached to price columns BY TICKER.
yfinance returns multi-ticker columns in alphabetical order, not request order, so
positional assignment scrambles the universe (CASH gets VTI's prices). This bug
poisoned a real snapshot once; the test exists so it cannot come back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taa.data.loaders import rename_ticker_columns, DEFAULT_ETFS


def _frame(columns):
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    return pd.DataFrame({c: np.arange(3.0) + i * 100 for i, c in enumerate(columns)}, index=idx)


def test_rename_is_by_ticker_not_position():
    # simulate yfinance's alphabetical column order, which differs from dict order
    frame = _frame(sorted(DEFAULT_ETFS.values()))
    out = rename_ticker_columns(frame, DEFAULT_ETFS)
    assert list(out.columns) == list(DEFAULT_ETFS.keys())
    for sleeve, ticker in DEFAULT_ETFS.items():
        pd.testing.assert_series_equal(out[sleeve], frame[ticker], check_names=False)


def test_rename_raises_on_missing_ticker():
    frame = _frame(["VTI"])
    with pytest.raises(ValueError, match="missing"):
        rename_ticker_columns(frame, {"EQ_US": "VTI", "CASH": "BIL"})
