"""The strategic policy portfolio.

Everything is measured against this. The value-add of a tactical overlay is its
contribution OVER the policy, net of costs, not its total return. Anchor here.
"""
from __future__ import annotations

import pandas as pd


class PolicyPortfolio:
    def __init__(self, weights: dict[str, float]):
        s = pd.Series(weights, dtype=float)
        if s.lt(0).any():
            raise ValueError("policy weights must be non-negative (long-only)")
        total = s.sum()
        if total <= 0:
            raise ValueError("policy weights must sum to a positive number")
        self.weights = s / total  # renormalize

    def target(self, assets: list[str]) -> pd.Series:
        return self.weights.reindex(assets).fillna(0.0)
