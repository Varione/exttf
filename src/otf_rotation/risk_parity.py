"""Rolling Risk Parity Engine — B3 benchmark.

Per planning.md §8:
- Uses rolling volatility/covariance from signal date history
- Equal risk contribution (ERC) optimization
- Non-negative weights, sum <= 100%, no leverage
- Asset caps and covariance regularization
- Falls back to static equal weight when training window insufficient
- Saves marginal risk contributions per period
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


class RollingRiskParityEngine:
    """Rolling Risk Parity benchmark using historical asset returns."""

    def __init__(
        self,
        nav_df: pd.DataFrame,
        asset_columns: list[str],
        rolling_window: int = 252,
        min_window: int = 60,
        asset_cap: float = 0.40,
        regularization: float = 1e-4,
        fallback_weights: Optional[dict[str, float]] = None,
    ):
        """
        Args:
            nav_df: DataFrame with date index and NAV columns for each asset
            asset_columns: list of column names representing assets
            rolling_window: number of trading days for rolling covariance
            min_window: minimum history required before using risk parity
            asset_cap: maximum weight per asset
            regularization: covariance matrix regularization strength
            fallback_weights: static weights when window insufficient (default: equal weight)
        """
        self.nav_df = nav_df
        self.asset_columns = asset_columns
        self.rolling_window = rolling_window
        self.min_window = min_window
        self.asset_cap = asset_cap
        self.regularization = regularization
        self.fallback_weights = fallback_weights or {
            col: 1.0 / len(asset_columns) for col in asset_columns
        }

    def get_returns_window(self, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Get rolling returns window ending at signal date."""
        mask = self.nav_df.index <= date
        if not mask.any():
            return None
        recent = self.nav_df.loc[mask, self.asset_columns].dropna()
        if len(recent) < self.min_window:
            return None
        returns = recent.pct_change().dropna()
        if len(returns) < self.min_window:
            return None
        return returns.tail(self.rolling_window)

    def _project_weights(self, w: np.ndarray) -> np.ndarray:
        """Iteratively project weights to satisfy non-negativity and asset_cap.

        Clip-then-normalize can violate caps again. This method iterates until
        stable or max iterations reached.
        """
        for _ in range(20):
            w_clipped = np.clip(w, 0, self.asset_cap)
            w_sum = w_clipped.sum()
            if w_sum <= 0:
                return np.ones(len(w)) / len(w)
            w_new = w_clipped / w_sum
            if np.max(np.abs(w_new - w)) < 1e-10:
                return w_new
            w = w_new
        return np.clip(w, 0, self.asset_cap)

    def compute_risk_parity_weights(
        self, returns: pd.DataFrame
    ) -> tuple[dict[str, float], dict[str, float], str]:
        """Compute equal risk contribution weights via iterative optimization.

        Args:
            returns: DataFrame of asset returns for the rolling window

        Returns:
            weights: dict of asset -> weight
            risk_contributions: dict of asset -> marginal risk contribution fraction
            status: optimization status string for audit
        """
        n = len(self.asset_columns)
        cov = returns.cov().values + self.regularization * np.eye(n)

        # Ensure positive definite
        eigen_method = "none"
        try:
            L, _ = np.linalg.eigh(cov)
            if L.min() < 0:
                cov += (-L.min() + 1e-8) * np.eye(n)
                eigen_method = "shifted"
        except np.linalg.LinAlgError:
            cov = np.diag(np.diag(cov)) + self.regularization * np.eye(n)
            eigen_method = "diagonal_fallback"

        # Iterative ERC algorithm (de Vries 2017) with proper cap projection
        w = np.ones(n) / n
        max_iter = 500
        tol = 1e-8
        converged = False

        for _ in range(max_iter):
            port_var = w @ cov @ w
            if port_var < 1e-12:
                break
            marginal_risk = cov @ w
            risk_contrib = w * marginal_risk
            total_rc = risk_contrib.sum()

            # Risk contribution as fraction of total portfolio risk
            risk_contrib_frac = risk_contrib / (total_rc + 1e-12)

            # Target equal risk contribution
            target_rc = np.ones(n) / n
            w_new = w * (target_rc / (risk_contrib_frac + 1e-12))

            # Use iterative projection to enforce caps correctly
            w_new = self._project_weights(w_new)

            # Check convergence
            if np.max(np.abs(w_new - w)) < tol:
                w = w_new
                converged = True
                break
            w = w_new

        # Final risk contributions (normalized to sum to 1)
        marginal_risk = cov @ w
        risk_contrib = w * marginal_risk
        total_rc = risk_contrib.sum()
        risk_contrib_frac = risk_contrib / (total_rc + 1e-12)

        weights = dict(zip(self.asset_columns, w))
        risk_contributions = dict(zip(self.asset_columns, risk_contrib_frac))
        status = "converged" if converged else "max_iter_reached"

        return weights, risk_contributions, status

    def get_weights(self, date: pd.Timestamp) -> dict[str, float]:
        """Get risk parity weights for a given signal date.

        Uses only history up to and including the signal date.
        Falls back to static equal weight when window insufficient.
        """
        returns = self.get_returns_window(date)
        if returns is None or len(returns) < self.min_window:
            return self.fallback_weights.copy()

        try:
            weights, _, _ = self.compute_risk_parity_weights(returns)
            # Ensure weights sum to 1 and are non-negative using projection
            w_arr = np.array(list(weights.values()))
            w_arr = self._project_weights(w_arr)
            return dict(zip(self.asset_columns, w_arr))
        except Exception:
            return self.fallback_weights.copy()

    def get_weights_with_audit(
        self, date: pd.Timestamp
    ) -> tuple[dict[str, float], dict]:
        """Get weights with full audit information.

        Returns:
            weights: asset weights
            audit: dict with window_length, method, risk_contributions, etc.
        """
        returns = self.get_returns_window(date)

        if returns is None or len(returns) < self.min_window:
            return (
                self.fallback_weights.copy(),
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "method": "fallback_static",
                    "fallback_reason": "insufficient_window",
                    "window_length": len(returns) if returns is not None else 0,
                    "risk_contributions": {},
                    "covariance_condition_number": None,
                },
            )

        try:
            weights, risk_contributions, opt_status = self.compute_risk_parity_weights(returns)

            # Final projection to guarantee cap compliance
            w_arr = np.array(list(weights.values()))
            w_arr = self._project_weights(w_arr)
            weights = dict(zip(self.asset_columns, w_arr))

            # Compute covariance condition number for audit
            cov = returns.cov().values + self.regularization * np.eye(len(self.asset_columns))
            cond_num = float(np.linalg.cond(cov))

            return (
                weights,
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "method": "rolling_risk_parity",
                    "fallback_reason": None,
                    "window_length": len(returns),
                    "risk_contributions": risk_contributions,
                    "covariance_condition_number": cond_num,
                    "optimization_status": opt_status,
                },
            )
        except Exception as e:
            return (
                self.fallback_weights.copy(),
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "method": "fallback_static_error",
                    "fallback_reason": str(e),
                    "window_length": len(returns),
                    "error": str(e),
                    "risk_contributions": {},
                    "covariance_condition_number": None,
                },
            )
