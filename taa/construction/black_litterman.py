"""Black-Litterman construction.

What this buys you, stated plainly: a principled, mandate-constrained way to turn
signal views into weights, anchored on the strategic policy as the equilibrium
portfolio. What it does NOT buy you: alpha. It adds an estimated covariance and
several free parameters (tau, delta, view_scale, view_conf, cov_lookback, shrink),
each of which is estimation error and overfitting surface. Judge it out of sample
against the simple tilt (walk_forward_select does this), do not assume it helps.

Mechanics:
  equilibrium (reverse optimization):  Pi = delta * Sigma * w_policy
  views (absolute, one per asset):     P = I,  Q = Pi + view_scale * z(score) * vol
  view uncertainty (He-Litterman):     Omega = diag(P (tau Sigma) P') / view_conf
  posterior returns:                   mu = [(tauSigma)^-1 + P'Omega^-1 P]^-1
                                             [(tauSigma)^-1 Pi + P'Omega^-1 Q]
  weights (constrained mean-variance): max mu'w - (delta/2) w'Sigma w
                                       s.t. sum w = 1, policy +/- max_tilt, w >= 0

With no view (all scores equal) Q collapses to Pi, so mu = Pi and the optimum is the
policy itself. That property is what makes it a proper overlay, and it is tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from taa.data.panel import PriceHistory


def shrink_covariance(sample_cov: np.ndarray, shrink: float) -> np.ndarray:
    """Linear shrinkage toward a diagonal (variances-only) target. Ledoit-Wolf is
    the principled version; this fixed-intensity shrink keeps dependencies light."""
    diag = np.diag(np.diag(sample_cov))
    return (1.0 - shrink) * sample_cov + shrink * diag


class BlackLitterman:
    name = "black_litterman"

    def __init__(
        self,
        tau: float = 0.05,
        delta: float = 3.0,
        view_scale: float = 0.5,
        view_conf: float = 0.5,
        cov_lookback: int = 36,
        shrink: float = 0.2,
        max_tilt: float = 0.10,
    ):
        self.tau = tau
        self.delta = delta
        self.view_scale = view_scale
        self.view_conf = view_conf
        self.cov_lookback = cov_lookback
        self.shrink = shrink
        self.max_tilt = max_tilt

    def weights(self, scores: pd.Series, policy: pd.Series, prices: PriceHistory, as_of) -> pd.Series:
        assets = list(policy.index)
        h = prices.history(as_of)                       # PIT: data through as_of only
        rets = h[assets].pct_change().dropna()
        min_obs = max(self.cov_lookback, len(assets) + 2)
        if len(rets) < min_obs:
            return policy.copy()                        # not enough history yet, stay neutral
        rets = rets.iloc[-self.cov_lookback:]

        Sigma = shrink_covariance(np.cov(rets.values, rowvar=False, ddof=1), self.shrink)
        Sigma = Sigma + 1e-8 * np.eye(len(assets))      # ridge for numerical safety
        w_eq = policy.reindex(assets).values

        Pi = self.delta * Sigma @ w_eq                  # implied equilibrium excess returns

        s = scores.reindex(assets).astype(float).values
        sd = np.std(s)
        z = (s - s.mean()) / sd if sd > 1e-12 else np.zeros_like(s)
        vol = np.sqrt(np.diag(Sigma))
        P = np.eye(len(assets))
        Q = Pi + self.view_scale * z * vol              # nudge equilibrium toward the signal

        tau_sigma = self.tau * Sigma
        omega = np.diag(np.diag(P @ tau_sigma @ P.T)) / self.view_conf
        A = np.linalg.inv(tau_sigma)
        O_inv = np.linalg.inv(omega)
        M = np.linalg.inv(A + P.T @ O_inv @ P)
        mu = M @ (A @ Pi + P.T @ O_inv @ Q)

        w = self._optimize(mu, Sigma, w_eq)
        return pd.Series(w, index=assets)

    def _optimize(self, mu, Sigma, w_eq):
        n = len(w_eq)
        delta = self.delta
        mt = self.max_tilt

        def neg_obj(w):
            return -(mu @ w - 0.5 * delta * w @ Sigma @ w)

        bounds = [(max(0.0, w_eq[i] - mt), min(1.0, w_eq[i] + mt)) for i in range(n)]
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        res = minimize(neg_obj, w_eq, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
        w = res.x if res.success else w_eq
        w = np.clip(w, 0.0, None)
        total = w.sum()
        return w / total if total > 0 else w_eq
