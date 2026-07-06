"""End-to-end entry point.

Offline demo (synthetic data, no network):
    python scripts/run_backtest.py

Real data (needs yfinance / pandas-datareader locally):
    python scripts/run_backtest.py --real

This prints, in order: the pre-committed multi-signal comparison (each family and the
composite, judged on IR vs policy net of costs); a detailed summary for the composite
overlay; cost- and turnover-sensitivity grids; the regime table; the robustness block
(bootstrap CI, deflated Sharpe, random-tilt null); simple-tilt vs Black-Litterman;
walk-forward OOS for the momentum lookback family; and the current positioning tilt.

The point of running it is to see how small and fragile an honest edge is, and how
much of it turnover and costs remove, not to find a big Sharpe. On synthetic data all
numbers prove machinery only.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from taa.construction.policy import PolicyPortfolio
from taa.construction.black_litterman import BlackLitterman
from taa.construction.tilt import SimpleTilt
from taa.signals.registry import build_signal, REGISTRY
from taa.signals.momentum import AbsoluteMomentum
from taa.backtest.engine import run_backtest
from taa.evaluation.metrics import summary, information_ratio, tracking_error
from taa.evaluation.regimes import regime_table, REAL_REGIMES, DEFAULT_REGIMES
from taa.evaluation.robustness import (
    param_sensitivity,
    block_bootstrap_ir,
    deflated_sharpe,
    random_tilt_null,
    reality_check,
    walk_forward_select,
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_data(cfg, args):
    if args.snapshot_load:
        from taa.data.snapshot import load_snapshot
        return load_snapshot(args.snapshot_load)
    if args.real:
        from taa.data.loaders import load_bundle
        return load_bundle(start=cfg["data"]["start"])
    from taa.data.synthetic import make_synthetic_bundle
    return make_synthetic_bundle(assets=cfg["universe"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
    ap.add_argument("--real", action="store_true", help="use real ETF/FRED data (needs network)")
    ap.add_argument("--snapshot-load", default=None, help="load a saved data snapshot dir instead of pulling")
    ap.add_argument("--snapshot-save", default=None, help="save the loaded bundle to this snapshot dir")
    args = ap.parse_args()

    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    cfg = load_config(args.config)

    data = get_data(cfg, args)
    if args.snapshot_save:
        from taa.data.snapshot import save_snapshot
        save_snapshot(data, args.snapshot_save)
        print(f"[snapshot] saved bundle to {args.snapshot_save}")
    policy = PolicyPortfolio(cfg["policy"])
    pol = policy.target(data.assets)
    con, cost, bt = cfg["construction"], cfg["costs"], cfg["backtest"]

    # Common backtest kwargs: construction, costs, turnover governors, rebalance schedule.
    bt_kwargs = dict(
        max_tilt=con["max_tilt"], scale=con["scale"],
        cost_bps=cost["cost_bps"], warmup=bt["warmup"],
        max_turnover=con.get("max_turnover"), no_trade_band=con.get("no_trade_band", 0.0),
        rebalance_every=bt.get("rebalance_every", 1),
    )

    # A loaded snapshot IS real data (a frozen vintage), not the synthetic demo.
    is_real = args.real or bool(args.snapshot_load)
    kind = ("REAL (ETF/FRED, frozen snapshot)" if args.snapshot_load
            else "REAL (ETF/FRED)" if args.real else "SYNTHETIC (machinery only)")
    print(f"\nData: {kind}. Turnover cap={bt_kwargs['max_turnover']}, "
          f"no-trade band={bt_kwargs['no_trade_band']}, cost={cost['cost_bps']}bps one-way.")

    # --- 1. Pre-committed multi-signal comparison (the thesis: four documented families) ---
    print("\n=== MULTI-SIGNAL COMPARISON (IR vs policy, net of costs) ===")
    rows = {}
    for spec in cfg["signal_set"]:
        sig = build_signal(dict(spec))
        try:
            r = run_backtest(data, sig, policy, **bt_kwargs)
        except ValueError as e:                      # e.g. carry/macro absent on real data
            print(f"  skip {sig.name}: {e}")
            continue
        rows[sig.name] = {
            "IR_vs_policy": information_ratio(r.strat_returns, r.policy_returns),
            "tracking_err": tracking_error(r.strat_returns, r.policy_returns),
            "ann_turnover": r.turnover.mean() * 12,
        }
    print(pd.DataFrame(rows).T)

    # The composite is the flagship overlay for the detailed robustness work below.
    composite = build_signal({"type": "composite"})
    res = run_backtest(data, composite, policy, **bt_kwargs)

    print("\n=== COMPOSITE OVERLAY SUMMARY (lead with IR vs policy) ===")
    print(summary(res))

    print("\n=== COST SENSITIVITY (composite IR vs policy) ===")
    grid = {}
    for c in cost["sensitivity_bps"]:
        kw = {**bt_kwargs, "cost_bps": c}
        r = run_backtest(data, composite, policy, **kw)
        grid[c] = information_ratio(r.strat_returns, r.policy_returns)
    print(pd.Series(grid, name="IR").rename_axis("cost_bps"))

    print("\n=== TURNOVER-CAP SENSITIVITY (composite IR vs policy; null = uncapped) ===")
    tgrid = {}
    for cap in cost.get("turnover_grid", [None]):
        kw = {**bt_kwargs, "max_turnover": cap}
        r = run_backtest(data, composite, policy, **kw)
        tgrid[str(cap)] = {
            "IR_vs_policy": information_ratio(r.strat_returns, r.policy_returns),
            "ann_turnover": r.turnover.mean() * 12,
        }
    print(pd.DataFrame(tgrid).T.rename_axis("max_turnover"))

    print("\n=== REBALANCE FREQUENCY (composite IR vs policy) ===")
    fgrid = {}
    for k in (1, 3):
        kw = {**bt_kwargs, "rebalance_every": k}
        r = run_backtest(data, composite, policy, **kw)
        fgrid[f"every_{k}m"] = {
            "IR_vs_policy": information_ratio(r.strat_returns, r.policy_returns),
            "ann_turnover": r.turnover.mean() * 12,
        }
    print(pd.DataFrame(fgrid).T)

    print("\n=== PER-SLEEVE vs FLAT COSTS (composite IR vs policy) ===")
    per_sleeve = pd.Series(cost["per_sleeve_bps"]).reindex(data.assets)
    r_flat = run_backtest(data, composite, policy, **bt_kwargs)
    r_ps = run_backtest(data, composite, policy, **{**bt_kwargs, "cost_bps": per_sleeve})
    print(pd.Series({
        f"flat_{cost['cost_bps']}bps": information_ratio(r_flat.strat_returns, r_flat.policy_returns),
        "per_sleeve": information_ratio(r_ps.strat_returns, r_ps.policy_returns),
    }, name="IR"))

    print("\n=== VOL-SCALED vs PLAIN TILT (composite IR vs policy) ===")
    r_plain = run_backtest(data, composite, policy, **bt_kwargs)
    vs = SimpleTilt(max_tilt=con["max_tilt"], scale=con["scale"], vol_scale=True)
    r_vs = run_backtest(data, composite, policy, constructor=vs,
                        cost_bps=cost["cost_bps"], warmup=bt["warmup"],
                        max_turnover=con.get("max_turnover"), no_trade_band=con.get("no_trade_band", 0.0),
                        rebalance_every=bt.get("rebalance_every", 1))
    print(pd.Series({
        "plain_tilt": information_ratio(r_plain.strat_returns, r_plain.policy_returns),
        "vol_scaled": information_ratio(r_vs.strat_returns, r_vs.policy_returns),
    }, name="IR"))

    print("\n=== REGIME TABLE (composite) ===")
    print(regime_table(res, REAL_REGIMES if is_real else DEFAULT_REGIMES))

    print("\n=== BOOTSTRAP CI ON IR (composite) ===")
    print(block_bootstrap_ir(res.active_returns))

    print("\n=== DEFLATED SHARPE (n_trials = families + composite tried) ===")
    print(deflated_sharpe(res.active_returns, n_trials=len(cfg["signal_set"])))

    print("\n=== RANDOM-TILT NULL (composite) ===")
    print(random_tilt_null(data, policy,
                           information_ratio(res.strat_returns, res.policy_returns),
                           n_sims=200, **bt_kwargs))

    # --- Black-Litterman construction, composite signal, constrained mean-variance ---
    bl = BlackLitterman(**cfg["black_litterman"])
    bl_res = run_backtest(data, composite, policy, constructor=bl,
                          cost_bps=cost["cost_bps"], warmup=bt["warmup"],
                          max_turnover=bt_kwargs["max_turnover"], no_trade_band=bt_kwargs["no_trade_band"])
    print("\n=== BLACK-LITTERMAN CONSTRUCTION (composite signal, constrained MV) ===")
    print("simple-tilt IR vs policy:", round(information_ratio(res.strat_returns, res.policy_returns), 4))
    print("black-litterman IR vs policy:", round(information_ratio(bl_res.strat_returns, bl_res.policy_returns), 4))
    print("Note: BL is a principled way to turn views into weights, not a source of alpha.")

    # --- Walk-forward OOS for the momentum lookback family (parameter selection is honest) ---
    wf = cfg["walk_forward"]
    print("\n=== WALK-FORWARD, OUT OF SAMPLE (momentum lookback family) ===")
    print("Naive best IN-SAMPLE IR across lookbacks (optimistic, do not trust):",
          round(param_sensitivity(data, AbsoluteMomentum, policy, wf["candidate_lookbacks"],
                                  **bt_kwargs).max(), 4))
    wf_res = walk_forward_select(data, AbsoluteMomentum, policy, wf["candidate_lookbacks"],
                                 min_train=wf["min_train"], step=wf["step"], objective=wf["objective"],
                                 **bt_kwargs)
    s_ret, p_ret = wf_res.oos_returns()
    print("OOS IR, walk-forward selected:", round(information_ratio(s_ret, p_ret), 4))
    print("Gap between best in-sample and OOS is the honest cost of parameter selection.")
    rc = reality_check(data, AbsoluteMomentum, policy, wf["candidate_lookbacks"], **bt_kwargs)
    print(f"Reality-check p-value for best lookback (lb={rc['best_param']}, {rc['n_candidates']} tried): "
          f"{rc['reality_check_pvalue']:.3f}  (high = the best lookback is explainable by the search)")

    # --- Current positioning tilt implied by the composite on the latest data ---
    last = data.dates[-1]
    scores = composite.score(data, last)
    target = SimpleTilt(max_tilt=con["max_tilt"], scale=con["scale"]).weights(scores, pol, data.prices, last)
    active = (target - pol).sort_values(ascending=False)
    print(f"\n=== CURRENT POSITIONING TILT (composite, as of {last.date()}) ===")
    print("Active components:", composite.active_components)
    tbl = pd.DataFrame({"policy": pol, "target": target, "active_tilt": target - pol}).loc[active.index]
    print(tbl.round(4))

    if not is_real:
        print("\nReminder: synthetic data proves the machinery only. Every number above is")
        print("random by construction. Run with --real for an actual positioning read, and")
        print("expect the OOS IR to sit well below the best in-sample IR.")


if __name__ == "__main__":
    main()
