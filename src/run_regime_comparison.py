"""Compare regime prediction methods for strategy allocation.

Compares four approaches:
1. Cluster baseline (KMeans hard labels)
2. MLP probabilities
3. LSTM probabilities
4. Transformer probabilities

Metrics: OOS Sharpe, Turnover reduction, Drawdown reduction, State persistence.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from backtest_engine import BacktestEngine
from regime_predictor import RegimePredictor, save_regime_probabilities
from strategy_library import STRATEGIES, get_strategies_for_regime


def compute_state_persistence(regime_series: pd.Series) -> float:
    """Compute state persistence: fraction of days where regime stays same."""
    if len(regime_series) < 2:
        return 0.0
    changes = (regime_series != regime_series.shift(1)).sum()
    return 1.0 - changes / len(regime_series)


def compute_regime_transitions(probs_df: pd.DataFrame, oos_start: str) -> int:
    """Count regime transitions in OOS period."""
    oos_mask = probs_df.index >= oos_start
    oos_regimes = probs_df.loc[oos_mask, "regime"]
    if len(oos_regimes) < 2:
        return 0
    return int((oos_regimes != oos_regimes.shift(1)).sum())


def run_strategy_backtest(
    engine: BacktestEngine,
    strategy_name: str,
    start: str,
    end: str,
    use_prob: bool = False,
) -> dict:
    """Run single strategy backtest and return metrics."""
    daily = engine.run_backtest(
        strategy_name,
        start=start,
        end=end,
        n_hold=20,
        max_weight=0.05,
        signal_to_return_lag=2,
        rebalance_every=5,
        respect_regime=True,
        use_regime_probabilities=use_prob,
    )
    if daily.empty:
        return {}

    metrics = engine.calculate_metrics(
        daily["return"],
        daily["turnover"],
        daily["transaction_cost"],
        daily["gross_return"],
        daily["exposure"],
        daily.get("cash_weight", None),
        dates=pd.to_datetime(daily["date"]),
    )
    metrics["strategy"] = strategy_name
    return metrics


def combined_strategy_returns(
    engine: BacktestEngine,
    start: str,
    end: str,
    use_prob: bool = False,
) -> pd.DataFrame:
    """Run all strategies and combine returns using regime-aware weighting."""
    all_daily = {}
    for name in STRATEGIES:
        daily = engine.run_backtest(
            name,
            start=start,
            end=end,
            n_hold=20,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
            respect_regime=True,
            use_regime_probabilities=use_prob,
        )
        if not daily.empty:
            all_daily[name] = daily.set_index("date")["return"]

    if not all_daily:
        return pd.DataFrame()

    combined = pd.DataFrame(all_daily)
    return combined.fillna(0.0)


def equal_weight_combine(combined: pd.DataFrame) -> pd.Series:
    """Equal weight combination of strategy returns."""
    n = len(combined.columns)
    if n == 0:
        return pd.Series(dtype=float)
    return combined.mean(axis=1)


def regime_prob_weighted_combine(
    combined: pd.DataFrame,
    probs_df: pd.DataFrame,
) -> pd.Series:
    """Weight strategies by regime probabilities.

    For each date, weight = sum_k P(regime=k) * (1/n_k) for strategies in regime k.
    """
    if combined.empty or probs_df.empty:
        return pd.Series(dtype=float)

    common_idx = combined.index.intersection(probs_df.index)
    if len(common_idx) < 2:
        return pd.Series(dtype=float)

    combined = combined.loc[common_idx]
    probs = probs_df.loc[common_idx, ["prob_r0", "prob_r1", "prob_r2"]]

    strategy_regimes = {name: STRATEGIES[name].regime for name in combined.columns}

    rows = []
    for date in common_idx:
        p = probs.loc[date].values
        weights = np.zeros(len(combined.columns))

        regime_counts = {0: 0, 1: 0, 2: 0}
        strat_to_regime = {}
        for i, name in enumerate(combined.columns):
            r = strategy_regimes.get(name, -1)
            if r >= 0:
                regime_counts[r] += 1
                strat_to_regime[i] = r

        for r in range(3):
            if regime_counts[r] > 0 and p[r] > 0:
                w_per = p[r] / regime_counts[r]
                for i, name in enumerate(combined.columns):
                    if strat_to_regime.get(i) == r:
                        weights[i] += w_per

        total = weights.sum()
        if total > 0:
            weights /= total

        row_ret = float(combined.loc[date].values @ weights)
        rows.append(row_ret)

    return pd.Series(rows, index=common_idx)


def compare_methods(
    oos_start: str = "2018-01-01",
    oos_end: str = "2026-07-17",
):
    """Run full comparison of four regime prediction methods."""
    results: list[dict] = []

    model_configs = {
        "Cluster_Hard": None,
        "MLP_Probs": "MLP",
        "LSTM_Probs": "LSTM",
        "Transformer_Probs": "Transformer",
    }

    base_engine = BacktestEngine(
        factor_path="data/processed/factors_all_repaired.csv",
        regime_path="data/processed/regime_predictions.csv",
        data_mode="etf",
        price_mode="total_return_proxy",
    )

    prob_cols = ["prob_r0", "prob_r1", "prob_r2"]

    for method_name, model_type in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        if model_type is not None:
            predictor = RegimePredictor(model_type=model_type)
            probs_df = predictor.predict_probabilities()
            regime_path = f"data/processed/regime_predictions_{model_type}.csv"
            save_regime_probabilities(predictor, regime_path)

            engine = BacktestEngine(
                factor_path="data/processed/factors_all_repaired.csv",
                regime_path=regime_path,
                data_mode="etf",
                price_mode="total_return_proxy",
            )
            use_prob = True
        else:
            engine = base_engine
            probs_df = pd.read_csv("data/processed/regime_predictions.csv", parse_dates=["date"])
            if "prob_r0" not in probs_df.columns:
                for r in range(3):
                    probs_df[f"prob_r{r}"] = (probs_df["regime"] == r).astype(float)
            use_prob = False

        # Combined strategy returns
        combined = combined_strategy_returns(
            engine, start=oos_start, end=oos_end, use_prob=use_prob
        )

        if combined.empty:
            print(f"  Skipping {method_name}: no data")
            continue

        # Equal weight baseline
        ew_returns = equal_weight_combine(combined)

        # Regime-weighted returns
        rw_returns = regime_prob_weighted_combine(combined, probs_df)

        # Metrics for equal weight
        ew_metrics = engine.calculate_metrics(ew_returns)

        # Metrics for regime-weighted
        rw_metrics = engine.calculate_metrics(rw_returns)

        # State persistence
        persistence = compute_state_persistence(probs_df["regime"])

        # Regime transitions in OOS
        n_transitions = compute_regime_transitions(probs_df, oos_start)

        row = {
            "method": method_name,
            "model_type": model_type if model_type else "KMeans",
            "use_probabilities": use_prob,
            # Equal weight metrics
            "EW_CAGR%": ew_metrics.get("CAGR%", 0),
            "EW_Sharpe": ew_metrics.get("Sharpe", 0),
            "EW_MaxDD%": ew_metrics.get("Max_Drawdown%", 0),
            "EW_Vol%": ew_metrics.get("Annualized_Volatility%", 0),
            "EW_Turnover%": ew_metrics.get("Turnover%", 0),
            # Regime-weighted metrics
            "RW_CAGR%": rw_metrics.get("CAGR%", 0),
            "RW_Sharpe": rw_metrics.get("Sharpe", 0),
            "RW_MaxDD%": rw_metrics.get("Max_Drawdown%", 0),
            "RW_Vol%": rw_metrics.get("Annualized_Volatility%", 0),
            "RW_Turnover%": rw_metrics.get("Turnover%", 0),
            # Regime metrics
            "state_persistence": persistence,
            "oos_transitions": n_transitions,
        }
        results.append(row)

        print(f"  EW Sharpe: {ew_metrics.get('Sharpe', 0):.4f}")
        print(f"  RW Sharpe: {rw_metrics.get('Sharpe', 0):.4f}")
        print(f"  State persistence: {persistence:.4f}")
        print(f"  OOS transitions: {n_transitions}")

    result_df = pd.DataFrame(results)
    output_path = "data/processed/regime_comparison_results.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    display_cols = [
        "method", "RW_Sharpe", "RW_CAGR%", "RW_MaxDD%",
        "state_persistence", "oos_transitions",
    ]
    print(result_df[display_cols].to_string(index=False))
    print(f"\nResults saved to {output_path}")

    return result_df


if __name__ == "__main__":
    t0 = time.time()
    compare_methods(oos_start="2018-01-01", oos_end="2026-07-17")
    print(f"\nTotal time: {time.time()-t0:.1f}s")
