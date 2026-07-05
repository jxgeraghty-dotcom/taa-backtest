"""Regime breakdown. With ~220 monthly points the aggregate number hides that the
edge is usually concentrated in a couple of stress windows. Show that explicitly.
Define regimes by DATE so the split is not itself fit to the result.
"""
from __future__ import annotations

import pandas as pd

from .metrics import _ann_return, information_ratio

# Dates below suit the offline synthetic set (five generic buckets).
DEFAULT_REGIMES = {
    "early": ("2005-01-01", "2008-06-30"),
    "stress_1": ("2008-07-01", "2009-12-31"),
    "mid": ("2010-01-01", "2019-12-31"),
    "stress_2": ("2020-01-01", "2020-12-31"),
    "late": ("2021-01-01", "2099-12-31"),
}

# Named macro regimes for the real 2007-2026 ETF sample. Defined by DATE, not fit to
# the result. Used by the --real run so the note's regime table reproduces exactly.
REAL_REGIMES = {
    "pre_gfc": ("2007-01-01", "2008-06-30"),
    "gfc": ("2008-07-01", "2009-06-30"),
    "recovery_qe": ("2009-07-01", "2019-12-31"),
    "covid": ("2020-01-01", "2020-12-31"),
    "inflation_2022": ("2021-01-01", "2022-12-31"),
    "higher_for_longer": ("2023-01-01", "2030-12-31"),
}


def regime_table(result, regimes: dict | None = None) -> pd.DataFrame:
    regimes = regimes or DEFAULT_REGIMES
    strat, pol = result.strat_returns, result.policy_returns
    rows = {}
    for name, (lo, hi) in regimes.items():
        s = strat.loc[lo:hi]
        p = pol.loc[lo:hi]
        if len(s.dropna()) < 3:
            continue
        rows[name] = {
            "months": len(s.dropna()),
            "strat_ann": _ann_return(s),
            "policy_ann": _ann_return(p),
            "info_ratio": information_ratio(s, p),
        }
    return pd.DataFrame(rows).T
