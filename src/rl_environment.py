"""Leakage-controlled environment for adaptive strategy allocation.

Supports two modes:
1. Legacy mode (respect_regime=False): All strategies always active
2. Meta allocator mode (respect_regime=True): Strategies only active in preferred regime
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from backtest_engine import BacktestEngine
from strategy_library import STRATEGIES, get_all_strategy_names


class StrategyWeightEnv:
    """Allocate among precomputed, net-of-cost strategy return streams.

    When ``respect_regime=True``, each strategy only generates returns
    when the current regime matches its preferred regime. Otherwise,
    the strategy returns zero (cash) for that date.
    """

    def __init__(
        self,
        factor_path: str = "data/processed/factors_all_repaired.csv",
        regime_path: str = "data/processed/regime_predictions.csv",
        db_path: str = "data/processed/etf.sqlite",
        start: str = "2018-01-01",
        end: str = "2026-07-17",
        lookback: int = 5,
        n_hold: int = 20,
        max_weight: float = 0.05,
        signal_to_return_lag: int = 2,
        rebalance_every: int = 5,
        allocation_cost_rate_per_side: float = 0.0005,
        engine: BacktestEngine | None = None,
        respect_regime: bool = True,
        max_single_weight: float = 0.5,
        turnover_penalty_coef: float = 5.0,
        drawdown_penalty_coef: float = 10.0,
        concentration_penalty_coef: float = 2.0,
    ):
        self.lookback = int(lookback)
        self.n_hold = int(n_hold)
        self.max_weight = float(max_weight)
        self.allocation_cost_rate_per_side = float(allocation_cost_rate_per_side)
        self.respect_regime = respect_regime
        self.max_single_weight = float(max_single_weight)
        self.turnover_penalty_coef = float(turnover_penalty_coef)
        self.drawdown_penalty_coef = float(drawdown_penalty_coef)
        self.concentration_penalty_coef = float(concentration_penalty_coef)

        self.engine = engine or BacktestEngine(
            factor_path=factor_path,
            regime_path=regime_path,
            db_path=db_path,
            data_mode="etf",
            price_mode="total_return_proxy",
        )

        self.strategy_names = get_all_strategy_names()
        self.n_strategies = len(self.strategy_names)

        # Map strategy name -> preferred regime
        self.strategy_regime_map = {
            name: STRATEGIES[name].regime for name in self.strategy_names
        }

        self.factors = self.engine.factors.loc[
            self.engine.factors["date"].between(start, end)
        ].copy()

        regime_df = self.engine.regime_preds.loc[
            self.engine.regime_preds["date"].between(start, end)
        ].copy()
        prob_cols = ["prob_r0", "prob_r1", "prob_r2"]
        if not set(prob_cols).issubset(regime_df.columns):
            for idx in range(3):
                regime_df[f"prob_r{idx}"] = (regime_df["regime"] == idx).astype(float)
        regime_df = regime_df.set_index("date")[prob_cols].sort_index()
        self.regime_map = dict(zip(regime_df.index.strftime("%Y-%m-%d"), regime_df.values))

        # Also store discrete regime map for respect_regime logic
        if "regime" in regime_df.columns:
            self.regime_discrete_map = dict(
                zip(regime_df.index.strftime("%Y-%m-%d"), regime_df["regime"].astype(int))
            )
        else:
            self.regime_discrete_map = dict(
                zip(
                    regime_df.index.strftime("%Y-%m-%d"),
                    regime_df.values.argmax(axis=1),
                )
            )

        self.key_factors = [
            "mom_5", "mom_20", "mom_60", "real_vol_10", "vol_ratio_5_20",
            "r2_20", "slope_20", "adx_proxy_14", "choppiness_14",
            "omega_20", "skew_20", "kurt_20", "autocorr_20",
            "win_rate_20", "down_capture_20", "tail_ratio_60",
            "vp_corr_20", "amount_vol_ratio", "ibias_20", "hurst_60",
        ]
        available_factors = [f for f in self.key_factors if f in self.factors.columns]
        self._factor_col_names = available_factors
        self._factors_by_date: dict[str, pd.Series] = {}
        for date, group in self.factors.groupby("date"):
            if "pit_eligible" in group:
                group = group.loc[group["pit_eligible"].astype(bool)]
            self._factors_by_date[date.strftime("%Y-%m-%d")] = group[available_factors].mean()

        all_dates = self.engine.dates[
            (self.engine.dates >= start) & (self.engine.dates <= end)
        ]
        self.dates = pd.DatetimeIndex(all_dates)

        print("Pre-computing net strategy returns...")
        strategy_results = {}
        for name in self.strategy_names:
            daily = self.engine.run_backtest(
                name,
                start=start,
                end=end,
                n_hold=n_hold,
                max_weight=max_weight,
                signal_to_return_lag=signal_to_return_lag,
                rebalance_every=rebalance_every,
                respect_regime=False,
            ).set_index("date")
            strategy_results[name] = daily["return"].reindex(self.dates).fillna(0.0)
        self.strategy_returns = pd.DataFrame(strategy_results, index=self.dates)

        # Compute per-strategy metrics for state space
        self._strategy_vols = self.strategy_returns.rolling(20, min_periods=5).std().fillna(0.0)
        self._strategy_drawdowns = self._compute_rolling_drawdown()

        # State dimension:
        # 3 (regime probs) + n_strategies * lookback (recent returns)
        # + n_strategies (volatility) + n_strategies (drawdown)
        # + len(key_factors) (macro factors)
        self.state_dim = (
            3
            + self.n_strategies * self.lookback
            + self.n_strategies
            + self.n_strategies
            + len(self._factor_col_names)
        )
        self.reset()

    def _compute_rolling_drawdown(self) -> pd.DataFrame:
        """Compute rolling 20-day max drawdown for each strategy."""
        result = np.zeros_like(self.strategy_returns.values)
        for i in range(len(self.strategy_returns)):
            start = max(0, i - 19)
            window = self.strategy_returns.iloc[start:i + 1].values
            wealth = np.cumprod(1 + window, axis=0)
            peak = np.maximum.accumulate(wealth, axis=0)
            dd = (wealth / peak - 1.0).min(axis=0)
            result[i] = dd
        return pd.DataFrame(result, index=self.strategy_returns.index,
                           columns=self.strategy_returns.columns)

    def _get_state(self) -> np.ndarray:
        """Build state vector with regime probs, strategy metrics, and macro factors."""
        observable_idx = self.current_idx - 1
        if observable_idx < 0:
            return np.zeros(self.state_dim, dtype=np.float32)

        observable_date = self.dates[observable_idx]
        date_str = observable_date.strftime("%Y-%m-%d")

        # Regime probabilities (3)
        regime_probs = self.regime_map.get(date_str, np.array([0.33, 0.33, 0.34]))

        # Recent strategy returns (n_strategies * lookback)
        recent_returns: list[float] = []
        for idx in range(self.current_idx - self.lookback, self.current_idx):
            if 0 <= idx < len(self.dates):
                recent_returns.extend(self.strategy_returns.iloc[idx].values.tolist())
            else:
                recent_returns.extend([0.0] * self.n_strategies)

        # Strategy volatilities (n_strategies)
        strat_vols = self._strategy_vols.iloc[self.current_idx - 1].values.tolist() if self.current_idx > 0 else [0.0] * self.n_strategies

        # Strategy drawdowns (n_strategies)
        strat_dd = self._strategy_drawdowns.iloc[self.current_idx - 1].values.tolist() if self.current_idx > 0 else [0.0] * self.n_strategies

        # Macro factors
        means = self._factors_by_date.get(date_str)
        factor_vals = []
        for factor in self._factor_col_names:
            value = means.get(factor, 0.0) if means is not None else 0.0
            factor_vals.append(float(value) if np.isfinite(value) else 0.0)

        return np.concatenate([
            regime_probs.tolist(),
            recent_returns,
            strat_vols,
            strat_dd,
            factor_vals,
        ]).astype(np.float32)

    def _get_regime_mask(self, date_str: str) -> np.ndarray:
        """Return binary mask: 1 if strategy is active in current regime."""
        probs = self.regime_map.get(date_str, np.array([0.33, 0.33, 0.34]))
        current_regime = int(probs.argmax())

        mask = np.ones(self.n_strategies, dtype=float)
        if self.respect_regime:
            for i, name in enumerate(self.strategy_names):
                pref_regime = self.strategy_regime_map.get(name, -1)
                if pref_regime >= 0 and pref_regime != current_regime:
                    mask[i] = 0.0
        return mask

    def reset(self) -> np.ndarray:
        self.current_idx = max(self.lookback, 1)
        self._prev_weights = np.zeros(self.n_strategies, dtype=np.float64)
        self._last_weights = np.zeros(self.n_strategies, dtype=np.float64)
        self._history_returns: list[float] = []
        return self._get_state()

    def step(
        self, weights: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict]:
        if self.current_idx >= len(self.dates):
            raise RuntimeError("Episode is complete; call reset()")

        weights = np.asarray(weights, dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        weights = weights / total if total > 0 else np.zeros_like(weights)
        weights = np.minimum(weights, self.max_single_weight)

        # Apply regime mask: zero out returns for strategies not in their regime
        date_str = self.dates[self.current_idx].strftime("%Y-%m-%d")
        regime_mask = self._get_regime_mask(date_str)
        effective_weights = weights * regime_mask

        # Renormalize if regime masking changed total weight
        eff_total = float(effective_weights.sum())
        if eff_total > 0:
            effective_weights = effective_weights / eff_total

        strategy_rets = self.strategy_returns.iloc[self.current_idx].to_numpy(dtype=float)
        gross_return = float(np.dot(effective_weights, strategy_rets))
        traded_notional = float(np.abs(weights - self._prev_weights).sum())
        transaction_cost = traded_notional * self.allocation_cost_rate_per_side
        port_return = gross_return - transaction_cost

        # Update weights for drift
        denominator = 1.0 + gross_return
        if denominator > 0:
            self._prev_weights = effective_weights * (1.0 + strategy_rets) / denominator
        else:
            self._prev_weights = np.zeros_like(effective_weights)
        self._last_weights = weights.copy()
        self._history_returns.append(port_return)

        # Enhanced reward: risk-adjusted return with penalties
        reward = self._compute_reward(port_return, traded_notional)

        wealth = np.cumprod(1.0 + np.asarray(self._history_returns))
        drawdown = float(wealth[-1] / np.maximum.accumulate(wealth)[-1] - 1.0)

        info = {
            "date": self.dates[self.current_idx],
            "return": port_return,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "traded_notional": traded_notional,
            "turnover": traded_notional / 2.0,
            "weights": weights.copy(),
            "effective_weights": effective_weights.copy(),
            "regime_mask": regime_mask.copy(),
            "cum_20d": float(np.sum(self._history_returns[-20:])),
            "drawdown": drawdown,
        }

        self.current_idx += 1
        done = self.current_idx >= len(self.dates)
        next_state = np.zeros(self.state_dim, dtype=np.float32) if done else self._get_state()
        return next_state, reward, done, info

    def _compute_reward(self, port_return: float, turnover: float) -> float:
        """Compute risk-adjusted reward with penalties.

        reward = sharpe_proxy - turnover_penalty - drawdown_penalty - concentration_penalty
        """
        # Base return signal (scaled)
        base_reward = port_return * 100.0

        # Turnover penalty
        turnover_penalty = self.turnover_penalty_coef * turnover

        # Drawdown penalty (based on recent drawdown)
        if self._history_returns:
            wealth = np.cumprod(1.0 + np.asarray(self._history_returns))
            current_dd = float(wealth[-1] / np.maximum.accumulate(wealth)[-1] - 1.0)
            dd_penalty = self.drawdown_penalty_coef * max(0, -current_dd) * 100.0
        else:
            dd_penalty = 0.0

        # Concentration penalty (entropy of weights)
        w = self._last_weights
        w_safe = w[w > 1e-8]
        if len(w_safe) > 0:
            w_norm = w_safe / w_safe.sum()
            entropy = -float(np.sum(w_norm * np.log(w_norm + 1e-8)))
            max_entropy = np.log(len(w_safe)) if len(w_safe) > 1 else 1.0
            concentration = 1.0 - entropy / max_entropy
        else:
            concentration = 1.0
        conc_penalty = self.concentration_penalty_coef * concentration

        return base_reward - turnover_penalty - dd_penalty - conc_penalty


class LegacyStrategyWeightEnv(StrategyWeightEnv):
    """Backward compatible wrapper that uses respect_regime=False."""

    def __init__(self, **kwargs):
        kwargs.setdefault("respect_regime", False)
        super().__init__(**kwargs)
