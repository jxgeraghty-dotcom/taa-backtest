"""Macro nowcast signal.

A nowcast reads the state of the cycle from data available today and tilts toward
the assets that tend to do well in that state. It has two documented pieces, both of
which you must be able to defend:

1. A growth/risk-appetite state, built point-in-time from the macro panel:
     - if a yield-curve slope is available (`curve_slope`, or `dgs10` - `dgs2`),
       a STEEPENING curve over the trailing window reads as pro-growth / risk-on;
     - if a credit spread is available (`credit_spread`), TIGHTENING spreads reinforce it.
   The state is the standardized trailing change of that proxy, clipped to [-2, 2].
   Using the *change* keeps it stationary and avoids betting on the level of rates.

2. A fixed, pre-committed asset-response vector (BETAS below): who benefits when the
   state is risk-on. Growth assets load positively, duration and cash negatively.
   These betas are an economic prior, not fit to the backtest. If you change them,
   say so in the note.

    score_asset = state * beta_asset

With a flat macro panel (no move) the state is ~0 and the tilt vanishes, which is
the property that keeps this an overlay rather than a permanent bet. The macro panel
carries its own release lag (MacroHistory.release_lag), so the nowcast never uses an
observation before it would have been published.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taa.data.panel import as_bundle

# Pre-committed response to a risk-on (positive-growth) state. Economic prior, not fit.
BETAS = {
    "EQ_US": 1.0,
    "EQ_INTL": 1.0,
    "CREDIT": 0.7,
    "CMDTY": 0.5,
    "TIPS": -0.2,
    "CASH": -0.3,
    "GOVT": -0.8,
}


class MacroNowcast:
    def __init__(self, window: int = 6, betas: dict | None = None, name: str | None = None):
        self.window = int(window)
        self.betas = dict(betas) if betas is not None else dict(BETAS)
        self.name = name or f"macro_nowcast_{window}m"

    def _growth_proxy(self, h: pd.DataFrame) -> pd.Series:
        """A single pro-growth series from whatever macro columns are present."""
        cols = h.columns
        if "curve_slope" in cols:
            g = h["curve_slope"].astype(float)
        elif "dgs10" in cols and "dgs2" in cols:
            g = (h["dgs10"] - h["dgs2"]).astype(float)
        else:
            g = h.iloc[:, 0].astype(float)           # fall back to the first series
        if "credit_spread" in cols:
            # wider spreads are risk-off: subtract the spread so tightening adds to growth
            g = g - h["credit_spread"].astype(float)
        return g

    def score(self, data, as_of: pd.Timestamp) -> pd.Series:
        bundle = as_bundle(data)
        assets = bundle.assets
        if bundle.macro is None:
            raise ValueError("MacroNowcast needs a macro panel; got a bundle with macro=None")

        h = bundle.macro.history(as_of)               # already release-lagged, PIT-safe
        if len(h) <= self.window + 1:
            return pd.Series(0.0, index=assets)

        g = self._growth_proxy(h)
        change = g.diff(self.window)                  # trailing move in the growth proxy
        sd = change.dropna().std(ddof=0)
        if not np.isfinite(sd) or sd < 1e-12:
            return pd.Series(0.0, index=assets)
        state = float(np.clip(change.iloc[-1] / sd, -2.0, 2.0))

        beta = pd.Series(self.betas).reindex(assets).fillna(0.0)
        return (state * beta).astype(float)
