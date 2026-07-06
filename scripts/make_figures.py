"""Generate the two figures embedded in the research note.

  1. Cumulative active return of the composite overlay vs the policy (net of costs).
  2. Rolling 24-month annualized information ratio, to make the regime concentration
     visible — the story the aggregate 0.04 IR hides.

Reads a saved snapshot for reproducibility (default), or pulls live with --real:

    python scripts/make_figures.py --snapshot data/snapshot_2026-07
    python scripts/make_figures.py --real
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from taa.construction.policy import PolicyPortfolio
from taa.signals.registry import build_signal
from taa.backtest.engine import run_backtest
from taa.evaluation.metrics import PERIODS_PER_YEAR

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "notes", "figures")


def _get_data(args):
    if args.snapshot:
        from taa.data.snapshot import load_snapshot
        return load_snapshot(args.snapshot)
    if args.real:
        from taa.data.loaders import load_bundle
        return load_bundle()
    from taa.data.synthetic import make_synthetic_bundle
    return make_synthetic_bundle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None, help="snapshot dir to load (reproducible)")
    ap.add_argument("--real", action="store_true", help="pull live real data instead")
    ap.add_argument("--config", default=os.path.join(HERE, "..", "config", "default.yaml"))
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data = _get_data(args)
    policy = PolicyPortfolio(cfg["policy"])
    con, cost, bt = cfg["construction"], cfg["costs"], cfg["backtest"]

    res = run_backtest(
        data, build_signal({"type": "composite"}), policy,
        max_tilt=con["max_tilt"], scale=con["scale"], cost_bps=cost["cost_bps"],
        warmup=bt["warmup"], max_turnover=con.get("max_turnover"),
        no_trade_band=con.get("no_trade_band", 0.0), rebalance_every=bt.get("rebalance_every", 1),
    )
    active = res.active_returns
    os.makedirs(FIG_DIR, exist_ok=True)

    # Figure 1: cumulative active return.
    cum = (1.0 + active).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(cum.index, 100 * cum.values, color="#1f4e79", lw=1.6)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title("Composite overlay: cumulative active return vs policy (net of costs)")
    ax.set_ylabel("cumulative active return (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    f1 = os.path.join(FIG_DIR, "composite_active_return.png")
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # Figure 2: rolling 24m annualized IR.
    win = 24
    roll_mean = active.rolling(win).mean()
    roll_sd = active.rolling(win).std(ddof=1)
    roll_ir = (roll_mean / roll_sd) * np.sqrt(PERIODS_PER_YEAR)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(roll_ir.index, roll_ir.values, color="#7a3b1f", lw=1.6)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title(f"Composite overlay: rolling {win}-month annualized information ratio")
    ax.set_ylabel("rolling IR vs policy")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    f2 = os.path.join(FIG_DIR, "composite_rolling_ir.png")
    fig.savefig(f2, dpi=130)
    plt.close(fig)

    print("wrote:", os.path.relpath(f1), "and", os.path.relpath(f2))


if __name__ == "__main__":
    main()
