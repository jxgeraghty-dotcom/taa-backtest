"""Map signal scores to target weights: a simple, bounded, self-funding overlay.

Deliberately simple. A real desk uses a constrained optimizer (see
black_litterman.py). This is a transparent stand-in:

  1. z-score the signal across the sleeves that HAVE a view (NaN = no view)
  2. scale to raw tilts, optionally risk-scale by inverse sleeve vol, then DEMEAN
     across the viewed sleeves so active weights sum to zero (the overlay funds
     itself; it does not change the invested total)
  3. clip each tilt to +/- max_tilt (the mandate deviation cap)
  4. add to policy, clip to [0, 1] (long-only), renormalize

A sleeve with a NaN score (a signal that holds no view on it, e.g. carry before
its trailing window fills) keeps its POLICY weight. It is never dropped, and it
never causes the remaining sleeves to be silently over-weighted. CASH is the
funding sleeve and is not tilted on its own signal. The mandate cap applies to the
funding sleeve too: it absorbs the residual the per-sleeve clips leave behind, but
only up to +/- max_tilt — a "cap" the funding sleeve could blow through would not
be a cap. When it binds, the final renormalization spreads the small remainder
proportionally, so weights still sum to one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taa.data.panel import as_bundle, PriceHistory


def score_to_weights(
    scores: pd.Series,
    policy: pd.Series,
    max_tilt: float = 0.10,
    scale: float = 0.05,
    funding_sleeve: str = "CASH",
    risk_scale: pd.Series | None = None,
) -> pd.Series:
    assets = list(policy.index)
    scores = scores.reindex(assets).astype(float)

    tiltable = [a for a in assets if a != funding_sleeve]
    s = scores.loc[tiltable]
    viewed = s.dropna()                         # only sleeves the signal has a view on
    sd = viewed.std(ddof=0)
    if len(viewed) >= 2 and sd > 1e-12:
        z = (viewed - viewed.mean()) / sd
    else:
        z = pd.Series(0.0, index=viewed.index)

    raw = scale * z
    if risk_scale is not None:                  # shrink high-vol sleeves' tilts, grow low-vol
        raw = raw * risk_scale.reindex(raw.index).fillna(1.0)
    raw = raw - raw.mean()                       # net-zero active across the viewed sleeves
    tilt = raw.clip(-max_tilt, max_tilt)

    active = pd.Series(0.0, index=assets)        # no-view sleeves stay at policy (active 0)
    active.loc[tilt.index] = tilt
    # funding sleeve absorbs the clip residual, but the mandate cap binds it too
    active.loc[funding_sleeve] = float(np.clip(-tilt.sum(), -max_tilt, max_tilt))

    target = (policy + active).clip(lower=0.0)
    total = target.sum()
    return target / total if total > 0 else policy.copy()


class SimpleTilt:
    """Constructor wrapper around score_to_weights.

    With vol_scale=True it reads trailing sleeve volatility from prices.history(as_of)
    (point-in-time) and scales each tilt by median_vol / sleeve_vol, so a 10% tilt to a
    quiet sleeve and a 10% tilt to a wild one contribute comparable active risk rather
    than comparable weight. Bounded to [0.25, 4] so one calm sleeve cannot dominate.
    """

    name = "simple_tilt"

    def __init__(
        self,
        max_tilt: float = 0.10,
        scale: float = 0.05,
        funding_sleeve: str = "CASH",
        vol_scale: bool = False,
        vol_lookback: int = 36,
    ):
        self.max_tilt = max_tilt
        self.scale = scale
        self.funding_sleeve = funding_sleeve
        self.vol_scale = vol_scale
        self.vol_lookback = int(vol_lookback)

    def _risk_scale(self, assets, prices: PriceHistory, as_of) -> pd.Series | None:
        rets = prices.history(as_of)[list(assets)].pct_change().dropna()
        if len(rets) < 6:
            return None
        vol = rets.iloc[-self.vol_lookback:].std(ddof=0).replace(0.0, np.nan)
        rs = (vol.median() / vol)
        return rs.clip(0.25, 4.0)

    def weights(self, scores: pd.Series, policy: pd.Series, prices, as_of) -> pd.Series:
        risk_scale = None
        if self.vol_scale and prices is not None:
            risk_scale = self._risk_scale(policy.index, as_bundle(prices).prices, as_of)
        return score_to_weights(scores, policy, self.max_tilt, self.scale, self.funding_sleeve, risk_scale)
