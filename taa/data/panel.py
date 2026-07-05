"""Point-in-time data containers.

The single design goal of this module is to make look-ahead bias hard to write
by accident. Signal code is expected to read data ONLY through history(as_of),
which returns rows dated on or before as_of. Nothing in a signal should ever
touch the full frame. The no-look-ahead test harness enforces this by poisoning
future rows and asserting signal output does not move.

Convention used throughout the framework:
  data through as_of (inclusive) determines the weights that are TRADED at as_of
  and HELD across the following period. The backtest applies those weights to the
  next period's return, so the trade itself never sees the return it earns.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class PriceHistory:
    """Total-return price levels indexed by date (rows) and asset (columns)."""

    def __init__(self, prices: pd.DataFrame):
        if not isinstance(prices.index, pd.DatetimeIndex):
            raise TypeError("prices must be indexed by a DatetimeIndex")
        prices = prices.sort_index()
        self._prices = prices

    def history(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Return price rows dated on or before as_of. The only sanctioned read for signals."""
        return self._prices.loc[:as_of]

    def returns(self, as_of: pd.Timestamp, periods: int = 1) -> pd.DataFrame:
        return self.history(as_of).pct_change(periods)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._prices.index

    @property
    def assets(self) -> list[str]:
        return list(self._prices.columns)

    def raw_copy(self) -> pd.DataFrame:
        """Full frame copy. For plumbing and tests only, NEVER for signal logic."""
        return self._prices.copy()


class MacroHistory:
    """Macro series with a crude release lag.

    A release_lag of k means that at as_of you may only use observations dated
    on or before (as_of shifted back k index positions). This is the poor-man's
    substitute for true vintages. For anything you would defend to a desk, pull
    vintage data (FRED/ALFRED) instead of trusting a fixed lag.
    """

    def __init__(self, series: pd.DataFrame, release_lag: int = 1):
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("series must be indexed by a DatetimeIndex")
        self._series = series.sort_index()
        self.release_lag = release_lag

    def history(self, as_of: pd.Timestamp) -> pd.DataFrame:
        h = self._series.loc[:as_of]
        if self.release_lag > 0:
            h = h.iloc[: max(len(h) - self.release_lag, 0)]
        return h

    def raw_copy(self) -> pd.DataFrame:
        return self._series.copy()


@dataclass
class DataBundle:
    """All the point-in-time inputs a signal may read, in one container.

    A signal receives a DataBundle and reads each source only through its
    history(as_of), which returns rows dated on or before as_of. Price-only
    signals touch `prices`; a carry signal touches `carry`; a macro nowcast
    touches `macro`. Keeping every source behind the same PIT gate is what lets
    the no-look-ahead harness poison ALL futures at once and prove no signal peeks.

      prices : total-return levels per asset (always present).
      carry  : per-asset carry/yield levels (bond yields, HY OAS, T-bill, ...).
               Higher means more carry. Optional; None if not supplied.
      macro  : macro series with a release lag (curve slope, spreads, ...).
               Optional; None if not supplied.
    """

    prices: PriceHistory
    carry: PriceHistory | None = None
    macro: MacroHistory | None = None

    @property
    def assets(self) -> list[str]:
        return self.prices.assets

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.dates


def as_bundle(data) -> DataBundle:
    """Coerce a PriceHistory or DataBundle into a DataBundle.

    The engine and every price-only signal accept either, so existing call sites
    that pass a bare PriceHistory keep working: they are treated as a bundle whose
    only source is prices. A signal that genuinely needs carry/macro must be handed
    a bundle that carries them, or it raises a clear error at score time.
    """
    if isinstance(data, DataBundle):
        return data
    if isinstance(data, PriceHistory):
        return DataBundle(prices=data)
    raise TypeError(f"expected PriceHistory or DataBundle, got {type(data).__name__}")
