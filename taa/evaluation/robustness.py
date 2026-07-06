"""Robustness protocol. This is the actual differentiator, not the equity curve.

With ~220 monthly points an IR gap of 0.2 is statistically indistinguishable, so
treat every result with visible humility. Implemented here:
  param_sensitivity : is the result a knife-edge or a plateau?
  block_bootstrap_ir: a confidence interval on the IR that respects autocorrelation
  deflated_sharpe   : penalize the Sharpe for the number of trials you ran
  random_tilt_null  : does a turnover-matched random overlay do just as well?

Still to add (documented stubs): expanding-window / walk-forward parameter
selection. Interface given so you can drop it in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from taa.backtest.engine import BacktestResult, run_backtest
from .metrics import information_ratio, PERIODS_PER_YEAR


def param_sensitivity(prices, signal_cls, policy, lookbacks, **bt_kwargs) -> pd.Series:
    """IR over policy across a grid of the signal's main parameter."""
    out = {}
    for lb in lookbacks:
        sig = signal_cls(lb)
        res = run_backtest(prices, sig, policy, **bt_kwargs)
        out[lb] = information_ratio(res.strat_returns, res.policy_returns)
    return pd.Series(out, name="information_ratio")


def block_bootstrap_ir(active: pd.Series, block: int = 6, n_boot: int = 1000, seed: int = 3):
    """Circular block bootstrap CI for the annualized IR of an active-return series."""
    a = active.dropna().to_numpy()
    n = len(a)
    if n < block * 2:
        return {"ir": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    irs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
        sample = a[idx]
        sd = sample.std(ddof=1)
        if sd > 0:
            irs.append((sample.mean() / sd) * np.sqrt(PERIODS_PER_YEAR))
    irs = np.array(irs)
    point = (a.mean() / a.std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR)
    return {
        "ir": float(point),
        "ci_low": float(np.percentile(irs, 2.5)),
        "ci_high": float(np.percentile(irs, 97.5)),
    }


def deflated_sharpe(returns: pd.Series, n_trials: int = 1) -> dict:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado 2014).

    Penalizes the observed Sharpe for the number of configurations you tried.
    Returns per-observation SR, the deflation benchmark, and the probability the
    true Sharpe exceeds zero after accounting for the trials.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 10:
        return {"sr_obs": np.nan, "sr_deflated_benchmark": np.nan, "dsr": np.nan}
    sr = r.mean() / r.std(ddof=1)  # per-observation, not annualized
    sk = stats.skew(r)
    ku = stats.kurtosis(r, fisher=False)

    # Expected max Sharpe under n_trials independent trials.
    if n_trials > 1:
        gamma = 0.5772156649
        e = np.e
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * e))
        var_sr = (1.0 / (n - 1)) * (1 - sk * sr + ((ku - 1) / 4.0) * sr ** 2)
        sr_star = np.sqrt(max(var_sr, 0)) * ((1 - gamma) * z1 + gamma * z2)
    else:
        sr_star = 0.0

    denom = np.sqrt(max(1 - sk * sr + ((ku - 1) / 4.0) * sr ** 2, 1e-12))
    dsr = stats.norm.cdf(((sr - sr_star) * np.sqrt(n - 1)) / denom)
    return {"sr_obs": float(sr), "sr_deflated_benchmark": float(sr_star), "dsr": float(dsr)}


def reality_check(prices, signal_cls, policy, candidate_params, block: int = 6,
                  n_boot: int = 1000, seed: int = 7, **bt_kwargs) -> dict:
    """White's Reality Check p-value for the best lookback (data-snooping adjusted).

    The walk-forward's tempting OOS number rides a lookback picked from a grid, so its
    naive t-stat overstates significance. This bootstraps the null that NO candidate has
    skill: it studentizes each candidate's active-return mean, takes the max across
    candidates as the statistic, then circular-block-bootstraps the DEMEANED returns to
    build the null distribution of that max (White 2000). The p-value is the share of
    bootstrap maxima that beat the observed one. A large p-value means the best lookback's
    edge is explainable by having searched several. Autocorrelation is respected via the
    block length.
    """
    active = {}
    for p in candidate_params:
        res = run_backtest(prices, signal_cls(p), policy, **bt_kwargs)
        active[p] = res.active_returns
    A = pd.DataFrame(active).dropna()
    T, k = A.shape
    if T < block * 2 or k < 1:
        return {"best_param": None, "max_stat": np.nan, "reality_check_pvalue": np.nan, "n_candidates": k}

    scale = np.sqrt(T)
    mu = A.mean().to_numpy()
    sd = A.std(ddof=1).to_numpy()
    stat = np.where(sd > 0, scale * mu / sd, -np.inf)
    obs_max = float(np.max(stat))
    best = list(A.columns)[int(np.argmax(stat))]

    demeaned = (A - A.mean()).to_numpy()      # impose the no-skill null
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    boot_max = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % T) for s in starts])[:T]
        samp = demeaned[idx]
        bmu = samp.mean(axis=0)
        bsd = samp.std(axis=0, ddof=1)
        bstat = np.where(bsd > 0, scale * bmu / bsd, -np.inf)
        boot_max[b] = np.max(bstat)
    pval = float(np.mean(boot_max >= obs_max))
    return {"best_param": best, "max_stat": obs_max, "reality_check_pvalue": pval, "n_candidates": k}


def random_tilt_null(prices, policy, actual_ir, n_sims: int = 200, seed: int = 5, **bt_kwargs) -> dict:
    """Turnover-matched random overlay. If random tilts match your IR, you have none.

    Uses a random-score signal each sim, so the tilt machinery and cost drag match
    the real strategy. Reports the share of random runs whose IR beats the actual.
    """
    rng = np.random.default_rng(seed)
    assets = prices.assets

    class _RandomSignal:
        name = "random"

        def __init__(self, r):
            self._r = r

        def score(self, data, as_of):
            return pd.Series(self._r.standard_normal(len(assets)), index=assets)

    irs = []
    for _ in range(n_sims):
        res = run_backtest(prices, _RandomSignal(rng), policy, **bt_kwargs)
        irs.append(information_ratio(res.strat_returns, res.policy_returns))
    irs = np.array(irs)
    return {
        "actual_ir": float(actual_ir),
        "random_ir_mean": float(np.nanmean(irs)),
        "pct_random_beating_actual": float(np.mean(irs >= actual_ir)),
    }


@dataclass
class WalkForwardResult:
    """Out-of-sample result from expanding-window selection.

    `oos` is a single stitched backtest that switches to the selected parameter at
    each boundary and carries the book across, so it includes the real transition
    cost of switching. `oos_start` marks the first out-of-sample date; metrics should
    be computed from there, since everything before it used the default candidate.
    """
    oos: BacktestResult
    selections: pd.DataFrame
    oos_start: pd.Timestamp

    def active_returns(self) -> pd.Series:
        return self.oos.active_returns.loc[self.oos_start:]

    def oos_returns(self):
        return (
            self.oos.strat_returns.loc[self.oos_start:],
            self.oos.policy_returns.loc[self.oos_start:],
        )


class _ScheduledSignal:
    """Dispatches score(as_of) to whichever candidate the schedule selected for that
    date. The schedule is a step function over dates built only from past-only
    information, so this stays point-in-time correct."""

    name = "walk_forward"

    def __init__(self, chosen_by_date: pd.Series, candidates: dict):
        self._chosen = chosen_by_date.sort_index()   # date -> candidate key
        self._candidates = candidates                # key -> signal instance

    def score(self, data, as_of):
        key = self._chosen.asof(as_of)
        if pd.isna(key):
            key = self._chosen.iloc[0]
        return self._candidates[key].score(data, as_of)


def walk_forward_select(
    prices,
    signal_cls,
    policy,
    candidate_params,
    min_train: int = 60,
    step: int = 12,
    objective: str = "ir",
    warmup: int = 12,
    **bt_kwargs,
) -> WalkForwardResult:
    """Expanding-window out-of-sample parameter selection.

    At each boundary, pick the candidate parameter that maximized the objective on
    realized returns up to that point, deploy it for the next `step` months, then
    expand the window and re-select. This removes the in-sample selection bias that
    makes a fixed-lookback backtest look better than it is.

    Efficiency: each candidate is backtested once to get its realized active-return
    series (used only for selection). The reported OOS is then one stitched backtest
    that follows the selection schedule and pays to switch at boundaries.

    Honest limits: with ~220 monthly points and step=12 you get only a handful of OOS
    blocks, so the OOS estimate is itself noisy. This reduces overfitting bias; it
    does not make a short sample long.
    """
    candidates = {p: signal_cls(p) for p in candidate_params}

    # 1. realized series per candidate, for selection only (all share length/index)
    active, strat = {}, {}
    for p, sig in candidates.items():
        res = run_backtest(prices, sig, policy, warmup=warmup, **bt_kwargs)
        active[p] = res.active_returns
        strat[p] = res.strat_returns
    active_df = pd.DataFrame(active)
    L = len(active_df)
    if min_train >= L:
        raise ValueError(f"min_train={min_train} but only {L} active-return observations exist")

    def in_sample_metric(df_slice, key):
        if objective == "ir":
            a = df_slice[key].dropna()
        elif objective == "sharpe":
            a = pd.DataFrame(strat)[key].reindex(df_slice.index).dropna()
        else:
            raise ValueError("objective must be 'ir' or 'sharpe'")
        sd = a.std(ddof=1)
        return -np.inf if sd == 0 or len(a) < 3 else a.mean() / sd

    # 2. selection boundaries in units of active-return observations
    ret_dates = active_df.index                       # these are return dates (t+1)
    selections = []
    schedule = {}                                     # rebalance date -> chosen key
    k = min_train
    while k < L:
        window = active_df.iloc[:k]                   # strictly-past realized returns
        scored = {p: in_sample_metric(window, p) for p in candidate_params}
        best = max(scored, key=scored.get)
        # the block starting at return-date index k is produced by the rebalance one
        # step earlier; map to that rebalance date so the engine picks it up in time.
        reb_date = prices.dates[warmup + k - 1]
        schedule[reb_date] = best
        selections.append({"boundary": reb_date, "chosen": best, "in_sample_obj": scored[best]})
        k += step

    chosen_by_date = pd.Series(schedule).sort_index()
    # default (pre-first-boundary) candidate = first listed, for the warmup region only
    default_key = candidate_params[0]
    first_reb = prices.dates[warmup]
    if first_reb not in chosen_by_date.index:
        chosen_by_date = pd.concat([pd.Series({first_reb: default_key}), chosen_by_date]).sort_index()

    # 3. one stitched OOS backtest that pays to switch parameters at boundaries
    sched_signal = _ScheduledSignal(chosen_by_date, candidates)
    stitched = run_backtest(prices, sched_signal, policy, warmup=warmup, **bt_kwargs)

    oos_start = ret_dates[min_train]                  # first genuinely OOS-selected return
    return WalkForwardResult(
        oos=stitched,
        selections=pd.DataFrame(selections).set_index("boundary"),
        oos_start=oos_start,
    )
