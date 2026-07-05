"""Deterministic synthetic data so the whole pipeline runs offline.

This exists to let you verify the machinery (and the no-look-ahead test) without
any network. It is NOT a market simulation and must never be used to make a claim
about a real strategy. Swap in taa.data.loaders for real ETF/FRED history locally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import PriceHistory, MacroHistory, DataBundle

DEFAULT_ASSETS = ["EQ_US", "EQ_INTL", "GOVT", "CREDIT", "TIPS", "CMDTY", "CASH"]

# Plausible starting carry levels per sleeve (annualized): equity earnings yields,
# bond/credit yields, TIPS real yield, a ~0 commodity roll, a cash bill rate. These
# are only anchors for the offline machinery; they are NOT a market forecast.
CARRY_BASE = {
    "EQ_US": 0.045, "EQ_INTL": 0.055, "GOVT": 0.025, "CREDIT": 0.060,
    "TIPS": 0.010, "CMDTY": 0.000, "CASH": 0.020,
}


def make_synthetic_prices(
    assets: list[str] | None = None,
    start: str = "2005-01-31",
    months: int = 240,
    seed: int = 7,
) -> PriceHistory:
    assets = assets or DEFAULT_ASSETS
    rng = np.random.default_rng(seed)
    n = len(assets)
    dates = pd.date_range(start=start, periods=months, freq="ME")

    # Rough monthly drift/vol per sleeve. CASH is a low-vol positive drift proxy.
    base_mu = np.array([0.006, 0.005, 0.002, 0.004, 0.003, 0.004, 0.0015])[:n]
    base_sd = np.array([0.045, 0.050, 0.015, 0.025, 0.020, 0.055, 0.002])[:n]

    # A shared risk factor to induce cross-asset correlation, with two regime breaks.
    factor = rng.standard_normal(months)
    regime = np.ones(months)
    regime[40:52] = -3.0    # a drawdown regime
    regime[180:190] = -2.0  # a second stress window
    loadings = np.array([1.0, 1.1, -0.2, 0.6, 0.1, 0.7, 0.0])[:n]

    idio = rng.standard_normal((months, n))
    rets = (
        base_mu[None, :]
        + loadings[None, :] * (base_sd[None, :] * factor[:, None] * 0.6 * regime[:, None])
        + base_sd[None, :] * idio * 0.8
    )
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    frame = pd.DataFrame(prices, index=dates, columns=assets)
    return PriceHistory(frame)


def make_synthetic_carry(
    assets: list[str] | None = None,
    start: str = "2005-01-31",
    months: int = 240,
    seed: int = 13,
) -> PriceHistory:
    """Deterministic per-asset carry (yield/spread) levels, as a PIT panel.

    Each sleeve starts at its CARRY_BASE anchor and wanders on a slow random walk,
    clipped to stay non-silly. Container is a PriceHistory only because all a carry
    signal needs is history(as_of); the values are yields/spreads, not prices. Like
    everything synthetic here, this exercises the machinery and nothing more.
    """
    assets = assets or DEFAULT_ASSETS
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=months, freq="ME")
    base = np.array([CARRY_BASE.get(a, 0.02) for a in assets])
    steps = rng.standard_normal((months, len(assets))) * 0.0008     # ~8bps monthly wander
    levels = base[None, :] + np.cumsum(steps, axis=0)
    levels = np.clip(levels, -0.02, 0.12)
    return PriceHistory(pd.DataFrame(levels, index=dates, columns=assets))


def make_synthetic_macro(start: str = "2005-01-31", months: int = 240, seed: int = 11) -> MacroHistory:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=months, freq="ME")
    slope = np.cumsum(rng.standard_normal(months) * 0.05) + 1.5  # a fake curve-slope series
    frame = pd.DataFrame({"curve_slope": slope}, index=dates)
    return MacroHistory(frame, release_lag=1)


def make_synthetic_bundle(
    assets: list[str] | None = None,
    start: str = "2005-01-31",
    months: int = 240,
) -> DataBundle:
    """Prices + carry + macro in one PIT bundle, for the offline end-to-end demo."""
    assets = assets or DEFAULT_ASSETS
    return DataBundle(
        prices=make_synthetic_prices(assets=assets, start=start, months=months),
        carry=make_synthetic_carry(assets=assets, start=start, months=months),
        macro=make_synthetic_macro(start=start, months=months),
    )
