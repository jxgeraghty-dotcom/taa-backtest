"""Construction (SimpleTilt) checks: NaN-robustness and vol scaling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from taa.construction.tilt import score_to_weights, SimpleTilt
from taa.data.synthetic import make_synthetic_prices, DEFAULT_ASSETS

POLICY = pd.Series({"EQ_US": 0.35, "EQ_INTL": 0.15, "GOVT": 0.20,
                    "CREDIT": 0.10, "TIPS": 0.05, "CMDTY": 0.05, "CASH": 0.10})


def test_nan_scores_keep_policy_weight():
    # No view on EQ_US / EQ_INTL / CMDTY: they must stay at policy, not be dropped.
    scores = pd.Series({"EQ_US": np.nan, "EQ_INTL": np.nan, "GOVT": 0.045,
                        "CREDIT": 0.06, "TIPS": 0.02, "CMDTY": np.nan, "CASH": 0.02})
    w = score_to_weights(scores, POLICY, max_tilt=0.10, scale=0.05)
    assert not w.isna().any(), "no-view sleeves must not produce NaN weights"
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= -1e-9).all()
    # the un-viewed sleeves keep essentially their policy weight (no active tilt)
    for a in ("EQ_US", "EQ_INTL", "CMDTY"):
        assert abs(w[a] - POLICY[a]) < 0.02


def test_all_nan_scores_return_policy():
    scores = pd.Series(np.nan, index=DEFAULT_ASSETS)
    w = score_to_weights(scores, POLICY, max_tilt=0.10, scale=0.05)
    assert np.allclose(w.reindex(POLICY.index).values, POLICY.values, atol=1e-9)


def test_vol_scaling_shrinks_high_vol_tilts():
    prices = make_synthetic_prices()
    as_of = prices.dates[120]
    scores = pd.Series({a: 1.0 if a in ("EQ_US", "CMDTY") else -0.2 for a in DEFAULT_ASSETS})
    plain = SimpleTilt(max_tilt=0.10, scale=0.05, vol_scale=False)
    scaled = SimpleTilt(max_tilt=0.10, scale=0.05, vol_scale=True, vol_lookback=36)
    wp = plain.weights(scores, POLICY, prices, as_of)
    ws = scaled.weights(scores, POLICY, prices, as_of)
    assert abs(ws.sum() - 1.0) < 1e-9 and (ws >= -1e-9).all()
    # CMDTY is the highest-vol sleeve in the synthetic set, so its tilt should shrink
    # under vol scaling relative to the plain tilt.
    assert (ws["CMDTY"] - POLICY["CMDTY"]) < (wp["CMDTY"] - POLICY["CMDTY"])
