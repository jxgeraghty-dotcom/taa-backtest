"""THE test. Written first, on purpose.

A signal must depend only on data dated on or before as_of. We verify this by
poisoning every future row of every bundle source (prices, carry, macro) with absurd
values and asserting the signal's output at as_of does not move. If a signal peeks
ahead, its score changes and this fails.

Every new signal you add must appear in `signals_under_test`. No exceptions. This is
the discipline the rest of the framework is built to protect. Carry and macro signals
read their own bundle sources, so poisoning prices alone would not test them; that is
why the poison here hits all three panels at once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taa.data.panel import PriceHistory, MacroHistory, DataBundle
from taa.data.synthetic import make_synthetic_bundle
from taa.signals.momentum import AbsoluteMomentum, TrendMA
from taa.signals.value import Valuation
from taa.signals.carry import Carry
from taa.signals.macro import MacroNowcast
from taa.signals.composite import Composite

POISON = 1e6  # multiplier a look-ahead bug could not survive silently
SHIFT = 12345.0


def _poison_prices(ph: PriceHistory, as_of: pd.Timestamp) -> PriceHistory:
    df = ph.raw_copy()
    mask = df.index > as_of
    df.loc[mask] = df.loc[mask] * POISON + SHIFT
    return PriceHistory(df)


def _poison_macro(mh: MacroHistory, as_of: pd.Timestamp) -> MacroHistory:
    df = mh.raw_copy()
    mask = df.index > as_of
    df.loc[mask] = df.loc[mask] * POISON + SHIFT
    return MacroHistory(df, release_lag=mh.release_lag)


def _poison_bundle(bundle: DataBundle, as_of: pd.Timestamp) -> DataBundle:
    return DataBundle(
        prices=_poison_prices(bundle.prices, as_of),
        carry=_poison_prices(bundle.carry, as_of) if bundle.carry is not None else None,
        macro=_poison_macro(bundle.macro, as_of) if bundle.macro is not None else None,
    )


signals_under_test = [
    AbsoluteMomentum(12),
    AbsoluteMomentum(6),
    TrendMA(10),
    Valuation(60),
    Carry("level"),
    Carry("change"),
    MacroNowcast(6),
    Composite(),
]

# A spread of as_of dates, avoiding the warmup edge and the final row.
_bundle = make_synthetic_bundle()
as_of_dates = [_bundle.dates[i] for i in (24, 60, 120, 200)]


@pytest.mark.parametrize("signal", signals_under_test, ids=lambda s: s.name)
@pytest.mark.parametrize("as_of", as_of_dates, ids=lambda d: str(d.date()))
def test_signal_ignores_future(signal, as_of):
    bundle = make_synthetic_bundle()
    baseline = signal.score(bundle, as_of)
    poisoned = signal.score(_poison_bundle(bundle, as_of), as_of)
    pd.testing.assert_series_equal(baseline, poisoned)


def test_history_never_returns_future():
    bundle = make_synthetic_bundle()
    as_of = bundle.dates[100]
    for source in (bundle.prices, bundle.carry):
        assert source.history(as_of).index.max() <= as_of
    # macro also respects its release lag: last usable date is strictly before as_of
    assert bundle.macro.history(as_of).index.max() < as_of
