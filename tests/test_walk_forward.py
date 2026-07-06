"""Walk-forward selection checks.

With a single candidate there is nothing to select, so the stitched OOS backtest
must equal a plain backtest of that candidate on the same dates. That pins down the
plumbing. We also check selections are recorded and the OOS window is non-empty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taa.construction.policy import PolicyPortfolio
from taa.data.synthetic import make_synthetic_prices, DEFAULT_ASSETS
from taa.backtest.engine import run_backtest
from taa.signals.momentum import AbsoluteMomentum
from taa.evaluation.robustness import walk_forward_select, reality_check

POLICY = PolicyPortfolio({a: 1.0 for a in DEFAULT_ASSETS})
PRICES = make_synthetic_prices()


def test_single_candidate_matches_plain_backtest():
    wf = walk_forward_select(PRICES, AbsoluteMomentum, POLICY, [12],
                             min_train=60, step=12, cost_bps=10.0, warmup=12)
    plain = run_backtest(PRICES, AbsoluteMomentum(12), POLICY, cost_bps=10.0, warmup=12)
    s_wf, _ = wf.oos_returns()
    s_plain = plain.strat_returns.loc[wf.oos_start:]
    pd.testing.assert_series_equal(s_wf, s_plain)


def test_multi_candidate_runs_and_selects():
    wf = walk_forward_select(PRICES, AbsoluteMomentum, POLICY, [3, 6, 12, 18],
                             min_train=60, step=12, cost_bps=10.0, warmup=12)
    assert len(wf.selections) > 0
    assert set(wf.selections["chosen"]).issubset({3, 6, 12, 18})
    strat, pol = wf.oos_returns()
    assert len(strat) > 0 and len(strat) == len(pol)


def test_boundaries_deploy_after_training_window():
    """Training window iloc[:k] ends with the return realized at dates[warmup + k],
    so the boundary rebalance must sit AT that date, never one earlier — deploying
    at dates[warmup + k - 1] would let the selection see the very return that
    rebalance earns (a one-month look-ahead per boundary)."""
    warmup, min_train, step = 12, 60, 12
    wf = walk_forward_select(PRICES, AbsoluteMomentum, POLICY, [3, 6, 12, 18],
                             min_train=min_train, step=step, cost_bps=10.0, warmup=warmup)
    dates = PRICES.dates
    ks = [min_train + i * step for i in range(len(wf.selections))]
    assert list(wf.selections.index) == [dates[warmup + k] for k in ks]
    # first OOS return is the one that boundary rebalance produces
    assert wf.oos_start == dates[warmup + min_train + 1]


def test_reality_check_returns_valid_pvalue():
    rc = reality_check(PRICES, AbsoluteMomentum, POLICY, [3, 6, 12, 18],
                       block=6, n_boot=200, cost_bps=10.0, warmup=12)
    assert rc["best_param"] in {3, 6, 12, 18}
    assert 0.0 <= rc["reality_check_pvalue"] <= 1.0
    assert rc["n_candidates"] == 4
