"""Behavioral checks for the new signal families.

Not look-ahead (that is test_no_lookahead) but the economic sign and the graceful
handling of missing data sources: value ranks the cheap sleeve above the rich one,
carry needs a carry panel, macro vanishes on a flat panel, and the composite still
runs when carry/macro are absent by renormalizing over the families that remain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taa.data.panel import PriceHistory, MacroHistory, DataBundle
from taa.data.synthetic import make_synthetic_bundle, DEFAULT_ASSETS
from taa.signals.value import Valuation
from taa.signals.carry import Carry
from taa.signals.macro import MacroNowcast
from taa.signals.composite import Composite


def _dates(n):
    return pd.date_range("2005-01-31", periods=n, freq="ME")


def test_value_prefers_the_cheaper_sleeve():
    # A: rises 5x over the window (rich). B: falls to a third (cheap). C: flat.
    d = _dates(80)
    A = np.linspace(100, 500, 80)
    B = np.linspace(300, 100, 80)
    C = np.full(80, 100.0)
    prices = PriceHistory(pd.DataFrame({"A": A, "B": B, "C": C}, index=d))
    s = Valuation(lookback=60, skip=12).score(prices, d[-1])
    assert s["B"] > s["C"] > s["A"], "cheap sleeve should score above rich sleeve"


def test_carry_ranks_by_yield_and_needs_a_panel():
    d = _dates(40)
    carry = PriceHistory(pd.DataFrame(
        {"HI": np.full(40, 0.08), "LO": np.full(40, 0.01)}, index=d))
    bundle = DataBundle(prices=carry, carry=carry)   # prices unused by carry signal here
    s = Carry("level").score(bundle, d[-1])
    assert s["HI"] > s["LO"]

    # Without a carry panel the signal must fail loudly, not silently return zeros.
    try:
        Carry("level").score(DataBundle(prices=carry), d[-1])
        assert False, "expected an error when carry panel is missing"
    except ValueError:
        pass


def test_macro_nowcast_is_flat_on_a_flat_panel():
    d = _dates(40)
    prices = PriceHistory(pd.DataFrame({a: np.full(40, 100.0) for a in DEFAULT_ASSETS}, index=d))
    flat_macro = MacroHistory(pd.DataFrame({"curve_slope": np.full(40, 1.5)}, index=d), release_lag=1)
    bundle = DataBundle(prices=prices, macro=flat_macro)
    s = MacroNowcast(window=6).score(bundle, d[-1])
    assert np.allclose(s.values, 0.0), "no macro move should mean no tilt"


def test_composite_runs_with_sources_missing():
    b = make_synthetic_bundle()
    as_of = b.dates[120]
    full = Composite()
    s_full = full.score(b, as_of)
    assert set(full.active_components) == {"momentum", "value", "carry", "macro"}

    price_only = DataBundle(prices=b.prices)          # no carry, no macro
    partial = Composite()
    s_partial = partial.score(price_only, as_of)
    assert set(partial.active_components) == {"momentum", "value"}
    assert s_full.notna().all() and s_partial.notna().all()
    assert np.isfinite(s_partial.values).all()
