"""Composite signal: blend the pre-committed families into one view.

The four signal families measure different things on different scales: momentum and
value in return units (~0.1), carry in yield units (~0.03), the macro nowcast in
beta*state units (~1). Summing them raw would let macro swamp carry. So the composite
z-scores EACH family cross-sectionally (across assets) first, then takes a weighted
average. That puts every family on a comparable "standard deviations of cross-asset
spread" footing before combining.

Weights are pre-committed and equal by default. This is deliberate: with ~220 monthly
points you cannot reliably estimate the optimal blend, and tuning the weights on the
backtest is exactly the overfitting the framework exists to resist. Equal-weight is
the honest prior. If you change the weights, say so in the note and pay for the search
with a higher deflated-Sharpe trial count.

Robust to missing sources: a family that needs carry/macro the bundle doesn't carry is
skipped and the remaining weights are renormalized (recorded in `.active_components`),
so the same composite runs on the offline bundle and on a real bundle where, say, the
FRED macro pull failed. Per-asset NaNs within a family (e.g. no equity carry) are
z-scored over the assets that do have a value and contribute zero elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taa.data.panel import as_bundle, DataBundle
from .momentum import AbsoluteMomentum
from .value import Valuation
from .carry import Carry
from .macro import MacroNowcast


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, computed over non-NaN entries; NaN stays NaN."""
    v = s.astype(float)
    valid = v.dropna()
    sd = valid.std(ddof=0)
    if len(valid) < 2 or not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (v - valid.mean()) / sd


def _needs(sig) -> str | None:
    """Which optional bundle source a family requires, if any."""
    if isinstance(sig, Carry):
        return "carry"
    if isinstance(sig, MacroNowcast):
        return "macro"
    return None


class Composite:
    DEFAULT_WEIGHTS = {"momentum": 1.0, "value": 1.0, "carry": 1.0, "macro": 1.0}

    def __init__(
        self,
        weights: dict | None = None,
        momentum_lookback: int = 12,
        value_lookback: int = 60,
        carry_mode: str = "level",
        macro_window: int = 6,
        name: str = "composite",
    ):
        self.name = name
        w = dict(self.DEFAULT_WEIGHTS if weights is None else weights)
        self._components = {
            "momentum": AbsoluteMomentum(momentum_lookback),
            "value": Valuation(value_lookback),
            "carry": Carry(carry_mode),
            "macro": MacroNowcast(macro_window),
        }
        self._weights = {k: float(w.get(k, 0.0)) for k in self._components}
        self.active_components: list[str] = []

    def _available(self, bundle: DataBundle) -> dict:
        """Components whose required source is present and whose weight is non-zero."""
        have = {"carry": bundle.carry is not None, "macro": bundle.macro is not None}
        out = {}
        for key, sig in self._components.items():
            if self._weights[key] <= 0:
                continue
            need = _needs(sig)
            if need is not None and not have[need]:
                continue
            out[key] = sig
        return out

    def score(self, data, as_of: pd.Timestamp) -> pd.Series:
        bundle = as_bundle(data)
        assets = bundle.assets
        comps = self._available(bundle)
        self.active_components = list(comps)
        if not comps:
            return pd.Series(0.0, index=assets)

        total_w = sum(self._weights[k] for k in comps)
        blended = pd.Series(0.0, index=assets)
        for key, sig in comps.items():
            z = _zscore(sig.score(bundle, as_of).reindex(assets)).fillna(0.0)
            blended = blended + (self._weights[key] / total_w) * z
        return blended.astype(float)
