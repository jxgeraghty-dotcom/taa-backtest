"""Map signal scores to target weights: a simple, bounded, self-funding overlay.

Deliberately simple. A real desk uses a constrained optimizer (see
black_litterman.py). This is a transparent stand-in:

  1. z-score the signal across assets
  2. scale to raw tilts, then DEMEAN so active weights sum to zero
     (the overlay funds itself; it does not change the invested total)
  3. clip each tilt to +/- max_tilt (the mandate deviation cap)
  4. add to policy, clip to [0, 1] (long-only), renormalize

CASH is treated as the funding sleeve and is not tilted on its own signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taa.data.panel import PriceHistory


def score_to_weights(
    scores: pd.Series,
    policy: pd.Series,
    max_tilt: float = 0.10,
    scale: float = 0.05,
    funding_sleeve: str = "CASH",
) -> pd.Series:
    assets = list(policy.index)
    scores = scores.reindex(assets).astype(float)

    tiltable = [a for a in assets if a != funding_sleeve]
    s = scores.loc[tiltable]
    sd = s.std(ddof=0)
    z = (s - s.mean()) / sd if sd > 1e-12 else pd.Series(0.0, index=tiltable)

    raw = scale * z
    raw = raw - raw.mean()                      # net-zero active weight
    tilt = raw.clip(-max_tilt, max_tilt)

    active = pd.Series(0.0, index=assets)
    active.loc[tiltable] = tilt
    active.loc[funding_sleeve] = -tilt.sum()    # funding sleeve absorbs the offset

    target = (policy + active).clip(lower=0.0)
    total = target.sum()
    return target / total if total > 0 else policy.copy()


class SimpleTilt:
    """Constructor wrapper around score_to_weights (ignores prices/as_of)."""

    name = "simple_tilt"

    def __init__(self, max_tilt: float = 0.10, scale: float = 0.05, funding_sleeve: str = "CASH"):
        self.max_tilt = max_tilt
        self.scale = scale
        self.funding_sleeve = funding_sleeve

    def weights(self, scores: pd.Series, policy: pd.Series, prices: PriceHistory, as_of) -> pd.Series:
        return score_to_weights(scores, policy, self.max_tilt, self.scale, self.funding_sleeve)
