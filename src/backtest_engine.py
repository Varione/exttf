"""Bias-aware backtesting engine for ETF signals and OTC NAV execution."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_price_series
from strategy_library import STRATEGIES, get_all_strategy_names


class BacktestEngine:
    """Daily accounting engine with explicit data, signal and cost semantics."""

    def __init__(
        self,
        factor_path: str = "data/processed/factors_all_repaired.csv",
        regime_path: str = "data/processed/regime_predictions.csv",
        db_path: str = "data/processed/etf.sqlite",
        *,
        data_mode: str = "etf",
        price_mode: str = "total_return_proxy",
        factors_df: pd.DataFrame | None = None,
        regime_df: pd.DataFrame | None = None,
        prices_df: pd.DataFrame | None = None,
        fee_rate_per_side: float = 0.0003,
        slippage_rate_per_side: float = 0.0002,
        cash_daily_return: float = 0.0,
        max_abs_asset_return: float | None = 0.50,
        require_pit: bool = True,
        require_full_pit: bool = False,
    ):
        self.data_mode = data_mode
        self.price_mode = price_mode
        self.db_path = db_path
        self.fee_rate_per_side = float(fee_rate_per_side)
        self.slippage_rate_per_side = float(slippage_rate_per_side)
        self.cash_daily_return = float(cash_daily_return)

        self.regime_preds = (
            regime_df.copy()
            if regime_df is not None
            else pd.read_csv(regime_path, parse_dates=["date"])
        )
        self.regime_preds["date"] = pd.to_datetime(self.regime_preds["date"])

        prob_cols = ["prob_r0", "prob_r1", "prob_r2"]
        has_probs = set(prob_cols).issubset(self.regime_preds.columns)

        if "regime" not in self.regime_preds:
            if not has_probs:
                raise ValueError("Regime data must contain regime or prob_r0..prob_r2")
            self.regime_preds["regime"] = (
                self.regime_preds[prob_cols]
                .to_numpy()
                .argmax(axis=1)
            )

        self.regime_map = dict(
            zip(
                self.regime_preds["date"].dt.strftime("%Y-%m-%d"),
                self.regime_preds["regime"].astype(int),
            )
        )

        if has_probs:
            self.regime_prob_map = dict(
                zip(
                    self.regime_preds["date"].dt.strftime("%Y-%m-%d"),
                    self.regime_preds[prob_cols].to_numpy(),
                )
            )
        else:
            self.regime_prob_map = {}

        self.factors = (
            factors_df.copy()
            if factors_df is not None
            else pd.read_csv(factor_path, parse_dates=["date"])
        )
        self.factors["date"] = pd.to_datetime(self.factors["date"])
        self.factors["symbol"] = self.factors["symbol"].astype(str).str.zfill(6)
        lifecycle_tables = {
            "etf_lifecycle",
            "etf_listing_history",
            "etf_termination",
            "etf_delisting",
        }
        try:
            with sqlite3.connect(db_path) as conn:
                available_tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                lifecycle_columns = (
                    {
                        row[1]
                        for row in conn.execute(
                            "PRAGMA table_info(etf_lifecycle)"
                        ).fetchall()
                    }
                    if "etf_lifecycle" in available_tables
                    else set()
                )
                lifecycle_is_complete = False
                if {
                    "source_independent",
                    "universe_scope",
                }.issubset(lifecycle_columns):
                    incomplete = conn.execute(
                        """
                        SELECT COUNT(*) FROM etf_lifecycle
                        WHERE COALESCE(source_independent, 0) != 1
                           OR universe_scope != 'historical_including_inactive'
                        """
                    ).fetchone()[0]
                    lifecycle_is_complete = incomplete == 0
        except sqlite3.Error:
            available_tables = set()
            lifecycle_is_complete = False
        has_pit_history = "pit_eligible" in self.factors
        has_lifecycle = bool(available_tables & lifecycle_tables)
        self.pit_status = (
            "PIT_COMPLETE"
            if has_pit_history and has_lifecycle and lifecycle_is_complete
            else "PIT_PARTIAL"
            if has_pit_history
            else "NO_PIT"
        )
        if require_pit and self.pit_status == "NO_PIT":
            raise RuntimeError(
                "PIT_UNIVERSE_REQUIRED: rebuild factors with factor_engine.py before backtesting"
            )
        if require_full_pit and self.pit_status != "PIT_COMPLETE":
            raise RuntimeError(
                "FULL_PIT_REQUIRED: historical listing/termination records are missing"
            )

        prices = (
            prices_df.copy()
            if prices_df is not None
            else load_price_series(db_path, data_mode=data_mode, price_mode=price_mode)
        )
        if "price" not in prices and "close" in prices:
            prices = prices.rename(columns={"close": "price"})
        required = {"symbol", "date", "price"}
        if not required.issubset(prices.columns):
            raise ValueError(f"Price data is missing columns: {sorted(required - set(prices.columns))}")

        prices["date"] = pd.to_datetime(prices["date"])
        prices["symbol"] = prices["symbol"].astype(str).str.zfill(6)
        prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
        prices = prices.loc[prices["price"] > 0].sort_values(["symbol", "date"])
        prices = prices.drop_duplicates(["symbol", "date"], keep="last")
        prices["ret"] = prices.groupby("symbol")["price"].pct_change(fill_method=None)

        if max_abs_asset_return is not None:
            bad = prices["ret"].abs().gt(max_abs_asset_return)
            if bad.any():
                examples = prices.loc[bad, ["symbol", "date", "ret"]].head(5)
                raise RuntimeError(
                    f"EXTREME_RETURN_GATE: {int(bad.sum())} returns exceed "
                    f"{max_abs_asset_return:.0%}; examples={examples.to_dict('records')}"
                )

        self.price_verification_ratio = (
            float(prices["reference_verified"].mean())
            if "reference_verified" in prices
            else 0.0
        )
        self.corporate_action_repair_count = (
            int(prices["corporate_action_repaired"].sum())
            if "corporate_action_repaired" in prices
            else 0
        )
        all_price_dates = pd.DatetimeIndex(prices["date"].drop_duplicates().sort_values())
        self.rets_wide = prices.pivot_table(
            index="date", columns="symbol", values="ret"
        ).sort_index().reindex(all_price_dates)
        # pct_change is computed before the pivot. A missing quote therefore
        # contributes zero on that date and the next observed quote captures
        # the cumulative move. Delisting remains a separate data-quality gate.
        self.rets_wide = self.rets_wide.fillna(0.0)
        self.dates = pd.DatetimeIndex(self.rets_wide.index)
        self._date_to_index = {date: idx for idx, date in enumerate(self.dates)}

        print(
            f"Loaded {len(self.dates)} trading days, {self.rets_wide.shape[1]} symbols | "
            f"data_mode={self.data_mode} price_mode={self.price_mode} pit={self.pit_status} "
            f"corporate_action_repairs={self.corporate_action_repair_count}"
        )
        print(f"Period: {self.dates[0].date()} ~ {self.dates[-1].date()}")

    @staticmethod
    def _gross_traded_notional(
        before: dict[str, float], after: dict[str, float]
    ) -> float:
        keys = set(before) | set(after)
        return float(sum(abs(after.get(k, 0.0) - before.get(k, 0.0)) for k in keys))

    @staticmethod
    def _drift_weights(
        positions: dict[str, float], daily_rets: pd.Series, net_return: float
    ) -> dict[str, float]:
        denominator = 1.0 + net_return
        if denominator <= 0:
            return {}
        drifted = {
            symbol: weight * (1.0 + float(daily_rets.get(symbol, 0.0))) / denominator
            for symbol, weight in positions.items()
            if weight > 0
        }
        return {symbol: weight for symbol, weight in drifted.items() if weight > 1e-12}

    @classmethod
    def _self_financing_rebalance(
        cls,
        before: dict[str, float],
        desired: dict[str, float],
        cost_rate: float,
        *,
        tolerance: float = 1e-12,
        max_iterations: int = 100,
    ) -> tuple[dict[str, float], float, float, float]:
        """Convert desired post-cost proportions into executable weights.

        All values are fractions of wealth immediately before the rebalance.
        Transaction costs must be funded by that same wealth, so a fully
        invested target is scaled down instead of implicitly borrowing fees.

        Returns ``(executed_weights, traded_notional, cost, post_cost_wealth)``.
        """
        if cost_rate < 0:
            raise ValueError("cost_rate must be non-negative")
        desired_exposure = float(sum(desired.values()))
        if desired_exposure > 1.0 + 1e-8:
            raise RuntimeError(
                f"LEVERAGE_NOT_ALLOWED: exposure={desired_exposure:.6f}"
            )
        if any(weight < 0 for weight in desired.values()):
            raise RuntimeError("SHORT_POSITIONS_NOT_ALLOWED")

        post_cost_wealth = 1.0
        for _ in range(max_iterations):
            executed = {
                symbol: weight * post_cost_wealth
                for symbol, weight in desired.items()
                if weight > 0
            }
            traded_notional = cls._gross_traded_notional(before, executed)
            transaction_cost = traded_notional * cost_rate
            updated_wealth = 1.0 - transaction_cost
            if updated_wealth <= 0:
                raise RuntimeError("TRANSACTION_COST_EXHAUSTED_CAPITAL")
            if abs(updated_wealth - post_cost_wealth) <= tolerance:
                return executed, traded_notional, transaction_cost, updated_wealth
            post_cost_wealth = updated_wealth
        raise RuntimeError("SELF_FINANCING_REBALANCE_DID_NOT_CONVERGE")

    def run_backtest(
        self,
        strategy_name: str,
        start: str = "2018-01-01",
        end: str = "2026-07-17",
        n_hold: int = 20,
        max_weight: float = 0.05,
        *,
        signal_to_return_lag: int = 2,
        rebalance_every: int = 5,
        respect_regime: bool = True,
        use_regime_probabilities: bool = False,
    ) -> pd.DataFrame:
        """Run a backtest with explicit signal-to-execution delay.

        With daily NAV data, ``signal_to_return_lag=2`` means NAV at T is
        observed after the cutoff, the order executes at NAV T+1, and the first
        held return is NAV T+2 / NAV T+1.

        When ``use_regime_probabilities=True``, strategy weights are scaled by
        P(strategy's preferred regime) instead of hard regime gating.
        """
        if signal_to_return_lag < 1:
            raise ValueError("signal_to_return_lag must be at least 1")
        if rebalance_every < 1:
            raise ValueError("rebalance_every must be at least 1")

        StrategyClass = STRATEGIES[strategy_name]
        strategy = StrategyClass(self.factors)
        test_dates = self.dates[(self.dates >= start) & (self.dates <= end)]
        if test_dates.empty:
            raise ValueError("No trading dates in requested backtest period")

        rows: list[dict] = []
        live_weights: dict[str, float] = {}
        initialized = False
        last_rebalance_index: int | None = None
        active_days = 0
        regime_transitions = 0
        prev_regime_for_strat: int | None = None

        for date in test_dates:
            full_idx = self._date_to_index[date]
            signal_idx = full_idx - signal_to_return_lag
            signal_date = self.dates[signal_idx] if signal_idx >= 0 else pd.NaT
            should_rebalance = (
                signal_idx >= 0
                and not (StrategyClass.buy_and_hold and initialized)
                and (
                    last_rebalance_index is None
                    or full_idx - last_rebalance_index >= rebalance_every
                )
            )

            target_weights = dict(live_weights)
            active_regime = None
            regime_scale = 1.0
            if should_rebalance:
                signal_date_str = signal_date.strftime("%Y-%m-%d")
                active_regime = self.regime_map.get(signal_date_str)

                if use_regime_probabilities and self.regime_prob_map:
                    probs = self.regime_prob_map.get(signal_date_str)
                    if probs is not None and StrategyClass.regime >= 0:
                        regime_scale = float(probs[StrategyClass.regime])
                    else:
                        regime_scale = 1.0 if StrategyClass.regime < 0 else 0.0

                    if StrategyClass.regime >= 0 and probs is not None:
                        effective_regime = int(probs.argmax())
                        if prev_regime_for_strat is not None and effective_regime != prev_regime_for_strat:
                            regime_transitions += 1
                        prev_regime_for_strat = effective_regime
                else:
                    regime_allowed = (
                        not respect_regime
                        or StrategyClass.regime < 0
                        or active_regime == StrategyClass.regime
                    )
                    regime_scale = 1.0 if regime_allowed else 0.0

                raw_positions = strategy.get_positions(
                    signal_date, n_hold=n_hold, max_weight=max_weight
                )
                if use_regime_probabilities and StrategyClass.regime >= 0:
                    target_weights = {
                        k: v * regime_scale for k, v in raw_positions.items()
                    }
                else:
                    target_weights = raw_positions if regime_scale > 0 else {}

                last_rebalance_index = full_idx
                if StrategyClass.buy_and_hold and target_weights:
                    initialized = True

            cost_rate = self.fee_rate_per_side + self.slippage_rate_per_side
            (
                executed_weights,
                traded_notional,
                transaction_cost,
                post_cost_wealth,
            ) = self._self_financing_rebalance(
                live_weights, target_weights, cost_rate
            )

            daily_rets = self.rets_wide.loc[date]
            target_exposure = float(sum(target_weights.values()))
            exposure = float(sum(executed_weights.values()))
            cash_weight = max(0.0, post_cost_wealth - exposure)
            gross_return = float(
                sum(
                    weight * float(daily_rets.get(symbol, 0.0))
                    for symbol, weight in executed_weights.items()
                )
                + cash_weight * self.cash_daily_return
            )
            net_return = gross_return - transaction_cost
            if executed_weights:
                active_days += 1

            live_weights = self._drift_weights(executed_weights, daily_rets, net_return)
            rows.append(
                {
                    "date": date,
                    "signal_date": signal_date,
                    "return": net_return,
                    "gross_return": gross_return,
                    "transaction_cost": transaction_cost,
                    "traded_notional": traded_notional,
                    "turnover": traded_notional / 2.0,
                    "exposure": exposure,
                    "target_exposure": target_exposure,
                    "cash_weight": cash_weight,
                    "position_count": len(executed_weights),
                    "rebalance": bool(should_rebalance),
                    "regime": active_regime,
                    "regime_scale": regime_scale if use_regime_probabilities else None,
                    "pit_status": self.pit_status,
                    "data_mode": self.data_mode,
                    "price_mode": self.price_mode,
                    "corporate_action_repairs": self.corporate_action_repair_count,
                }
            )

        print(f"  Active days: {active_days}/{len(test_dates)}")
        if use_regime_probabilities:
            print(f"  Regime transitions: {regime_transitions}")
        return pd.DataFrame(rows)

    def calculate_metrics(
        self,
        returns: pd.Series,
        turnover: pd.Series | None = None,
        transaction_cost: pd.Series | None = None,
        gross_returns: pd.Series | None = None,
        exposures: pd.Series | None = None,
        cash_weights: pd.Series | None = None,
        benchmark_returns: pd.Series | None = None,
        dates: pd.Series | None = None,
    ) -> dict:
        """Calculate standard performance and cost metrics.

        All percentage-type metrics have a ``%`` suffix in the key name.

        If ``benchmark_returns`` is provided, excess return metrics are
        computed: excess return, information ratio, tracking error,
        and alpha/beta relative to the benchmark.
        """
        if isinstance(returns, pd.Series):
            returns = returns.astype(float).fillna(0.0)
        else:
            returns = pd.Series(returns, dtype=float).fillna(0.0)
        if returns.empty:
            return {}

        wealth = (1.0 + returns).cumprod()
        total_return = wealth.iloc[-1] - 1.0
        n_days = len(returns)
        ann_return = wealth.iloc[-1] ** (252.0 / n_days) - 1.0
        ann_std = returns.std(ddof=1) * np.sqrt(252) if n_days > 1 else 0.0
        sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252) if ann_std > 0 else 0.0

        # Sortino ratio (downside deviation)
        downside = returns[returns < 0]
        downside_std = downside.std(ddof=1) * np.sqrt(252) if len(downside) > 1 else 0.0
        sortino = returns.mean() / downside.std(ddof=1) * np.sqrt(252) if downside_std > 0 else 0.0

        drawdown = wealth / wealth.cummax() - 1.0
        max_dd = float(drawdown.min())

        # Longest drawdown duration
        in_drawdown = (drawdown < -1e-12).astype(int)
        longest_dd_days = 0
        if in_drawdown.any():
            # Group consecutive drawdown periods
            groups = in_drawdown.ne(in_drawdown.shift(1)).cumsum()
            dd_lengths = in_drawdown.groupby(groups).sum()
            longest_dd_days = int(dd_lengths.max())

        metrics: dict[str, float] = {
            "CAGR%": ann_return * 100,
            "Total_Return%": total_return * 100,
            "Annualized_Volatility%": ann_std * 100,
            "Sharpe": float(sharpe),
            "Sortino": float(sortino),
            "Max_Drawdown%": max_dd * 100,
            "Calmar": float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0,
            "Daily_Win_Rate%": float((returns > 0).mean() * 100),
            "Skewness": float(returns.skew()),
            "Kurtosis": float(returns.kurtosis()),
            "Longest_DD_Duration_days": longest_dd_days,
        }

        # VaR and CVaR (95%)
        if n_days > 1:
            var_95 = float(np.percentile(returns, 5)) * 100
            cvar_95 = float(returns[returns <= np.percentile(returns, 5)].mean()) * 100
            metrics["VaR_95%"] = var_95
            metrics["CVaR_95%"] = cvar_95

        # Rolling Sharpe (12-month) and Rolling Drawdown (12-month)
        window_1y = min(252, n_days - 1)
        if window_1y > 20:
            rolling_sharpe = (
                returns.rolling(window_1y, min_periods=20)
                .apply(
                    lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252) if x.std(ddof=1) > 0 else 0.0,
                    raw=True,
                )
            )
            metrics["Rolling_Sharpe_12m_mean"] = float(rolling_sharpe.dropna().mean())
            metrics["Rolling_Sharpe_12m_min"] = float(rolling_sharpe.dropna().min())

            rolling_wealth = (1.0 + returns).rolling(window_1y, min_periods=20).apply(
                lambda x: np.prod(1.0 + x), raw=True
            )
            metrics["Rolling_Return_12m_mean%"] = float((rolling_wealth.dropna() - 1).mean() * 100)

        # Monthly win rate (requires date information)
        try:
            date_idx = dates if dates is not None else returns.index
            if hasattr(date_idx, "dt"):
                year_vals = date_idx.dt.year
                month_vals = date_idx.dt.month
            elif hasattr(date_idx, "year"):
                year_vals = date_idx.year
                month_vals = date_idx.month
            else:
                year_vals = None
                month_vals = None
            if year_vals is not None and month_vals is not None:
                monthly = returns.groupby([year_vals, month_vals]).apply(
                    lambda x: float((1.0 + x).prod() - 1.0)
                )
                if len(monthly) > 0:
                    metrics["Monthly_Win_Rate%"] = float((monthly > 0).mean() * 100)
        except Exception:
            pass

        # Exposure and cash ratio
        if exposures is not None:
            exp_series = pd.Series(exposures, dtype=float)
            metrics["Exposure%"] = float(exp_series.mean() * 100)
            active_returns = returns[exp_series > 0.1]
            if len(active_returns) > 0:
                metrics["Active_Win_Rate%"] = float((active_returns > 0).mean() * 100)
        if cash_weights is not None:
            cw_series = pd.Series(cash_weights, dtype=float)
            metrics["Cash_Ratio%"] = float(cw_series.mean() * 100)

        # Turnover and cost
        if turnover is not None:
            to_series = pd.Series(turnover, dtype=float)
            metrics["Turnover%"] = float(to_series.mean() * 100)
            metrics["Total_Turnover"] = float(to_series.sum())
        if transaction_cost is not None:
            tc_series = pd.Series(transaction_cost, dtype=float)
            metrics["Cumulative_Cost_Ratio%"] = float(tc_series.sum() * 100)
        if gross_returns is not None:
            gross_wealth = (1.0 + pd.Series(gross_returns, dtype=float).fillna(0.0)).cumprod()
            gross_ann = gross_wealth.iloc[-1] ** (252.0 / n_days) - 1.0
            metrics["Gross_Total_Return%"] = float(
                (gross_wealth.iloc[-1] - 1.0) * 100
            )
            metrics["Gross_CAGR%"] = gross_ann * 100
            metrics["Annualized_Cost_Drag%"] = (gross_ann - ann_return) * 100
            metrics["Transaction_Cost_Drag%"] = float(
                (gross_wealth.iloc[-1] - wealth.iloc[-1]) * 100
            )
        elif transaction_cost is not None:
            # Compatibility for callers that do not supply a gross series.
            # This is explicitly only a sum of daily cost ratios.
            metrics["Transaction_Cost_Drag%"] = metrics[
                "Cumulative_Cost_Ratio%"
            ]

        # Benchmark comparison
        if benchmark_returns is not None:
            bench = pd.Series(benchmark_returns, dtype=float).fillna(0.0)
            aligned_days = min(len(returns), len(bench))
            excess = returns.iloc[-aligned_days:] - bench.iloc[-aligned_days:]
            excess_std = excess.std(ddof=1) * np.sqrt(252) if aligned_days > 1 else 0.0
            metrics["Excess_Return%"] = float(excess.sum() * 100)
            metrics["Information_Ratio"] = (
                float(excess.mean() / excess.std(ddof=1) * np.sqrt(252))
                if excess_std > 0 else 0.0
            )
            metrics["Tracking_Error%"] = float(excess_std * 100)
            cov_mat = np.cov(
                returns.iloc[-aligned_days:].values,
                bench.iloc[-aligned_days:].values,
            )
            if cov_mat.shape == (2, 2) and cov_mat[1, 1] > 0:
                beta = float(cov_mat[0, 1] / cov_mat[1, 1])
                rf_daily = self.cash_daily_return
                alpha_daily = returns.mean() - beta * (bench.mean() - rf_daily)
                metrics["Alpha%"] = float(alpha_daily * 252 * 100)
                metrics["Beta"] = beta
            else:
                metrics["Alpha%"] = 0.0
                metrics["Beta"] = 1.0

        return metrics

    def run_all(
        self,
        start: str = "2018-01-01",
        end: str = "2026-07-17",
        n_hold: int = 20,
        max_weight: float = 0.05,
        *,
        output_path: str = "data/processed/backtest_results_repaired.csv",
        signal_to_return_lag: int = 2,
        rebalance_every: int = 5,
        respect_regime: bool = True,
        use_regime_probabilities: bool = False,
        strategy_names: list[str] | None = None,
    ) -> pd.DataFrame:
        results_all = []
        names = strategy_names if strategy_names is not None else get_all_strategy_names()
        for name in names:
            StrategyClass = STRATEGIES[name]
            print(f"Running {name} (regime={StrategyClass.regime})...")
            daily = self.run_backtest(
                name,
                start=start,
                end=end,
                n_hold=n_hold,
                max_weight=max_weight,
                signal_to_return_lag=signal_to_return_lag,
                rebalance_every=rebalance_every,
                respect_regime=respect_regime,
                use_regime_probabilities=use_regime_probabilities,
            )
            metrics = self.calculate_metrics(
                daily["return"],
                daily["turnover"],
                daily["transaction_cost"],
                daily["gross_return"],
                daily["exposure"],
                daily.get("cash_weight", None),
                dates=pd.to_datetime(daily["date"]),
            )
            metrics.update(
                {
                    "strategy": name,
                    "regime": StrategyClass.regime,
                    "n_trades": int((daily["traded_notional"] > 1e-12).sum()),
                    "pit_status": self.pit_status,
                    "data_mode": self.data_mode,
                    "price_mode": self.price_mode,
                    "signal_to_return_lag": signal_to_return_lag,
                    "rebalance_every": rebalance_every,
                    "use_regime_probabilities": use_regime_probabilities,
                }
            )
            results_all.append(metrics)

        result = pd.DataFrame(results_all)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        display_cols = [c for c in ["strategy", "CAGR%", "Sharpe", "Max_Drawdown%", "Exposure%"] if c in result.columns]
        print(result[display_cols].to_string(index=False))
        print(f"Results saved to {output_path}")
        return result


if __name__ == "__main__":
    engine = BacktestEngine()
    engine.run_all()
