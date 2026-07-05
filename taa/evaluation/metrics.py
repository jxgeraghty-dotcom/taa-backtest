"""Performance metrics. Lead with information ratio versus the policy, not total
return versus cash or equities. The overlay's job is active return per unit of
active risk; total return is mostly the policy's beta wearing a costume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12


def _ann_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return (1.0 + r).prod() ** (PERIODS_PER_YEAR / len(r)) - 1.0


def _ann_vol(r: pd.Series) -> float:
    return r.dropna().std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)


def sharpe(r: pd.Series, rf_monthly: float = 0.0) -> float:
    ex = r.dropna() - rf_monthly
    sd = ex.std(ddof=1)
    return np.nan if sd == 0 else (ex.mean() / sd) * np.sqrt(PERIODS_PER_YEAR)


def sortino(r: pd.Series, rf_monthly: float = 0.0) -> float:
    ex = r.dropna() - rf_monthly
    downside = ex[ex < 0].std(ddof=1)
    return np.nan if downside == 0 or np.isnan(downside) else (ex.mean() / downside) * np.sqrt(PERIODS_PER_YEAR)


def max_drawdown(r: pd.Series) -> float:
    curve = (1.0 + r.dropna()).cumprod()
    peak = curve.cummax()
    return (curve / peak - 1.0).min()


def tracking_error(strat: pd.Series, policy: pd.Series) -> float:
    active = (strat - policy).dropna()
    return active.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)


def information_ratio(strat: pd.Series, policy: pd.Series) -> float:
    active = (strat - policy).dropna()
    sd = active.std(ddof=1)
    return np.nan if sd == 0 else (active.mean() / sd) * np.sqrt(PERIODS_PER_YEAR)


def summary(result, rf_monthly: float = 0.0) -> pd.DataFrame:
    strat, pol = result.strat_returns, result.policy_returns
    rows = {}
    for name, r in [("strategy", strat), ("policy", pol)]:
        rows[name] = {
            "cagr": _ann_return(r),
            "ann_vol": _ann_vol(r),
            "sharpe": sharpe(r, rf_monthly),
            "sortino": sortino(r, rf_monthly),
            "max_drawdown": max_drawdown(r),
            "calmar": (_ann_return(r) / abs(max_drawdown(r))) if max_drawdown(r) != 0 else np.nan,
            "hit_rate": (r.dropna() > 0).mean(),
        }
    df = pd.DataFrame(rows).T
    df.loc["strategy", "ann_turnover"] = result.turnover.mean() * PERIODS_PER_YEAR
    df.loc["strategy", "tracking_error_vs_policy"] = tracking_error(strat, pol)
    df.loc["strategy", "information_ratio_vs_policy"] = information_ratio(strat, pol)
    return df
