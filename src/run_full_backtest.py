"""Full backtest verification with new repaired data."""

import os
import sys

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from backtest_engine import BacktestEngine

STRATEGIES = ["S04_VolTarget_Trend", "S23_Oversold_Long", "B0_BuyHold_EW"]
OUTPUT_PATH = "data/processed/backtest_results_v2.csv"


def main():
    print("=== Initializing BacktestEngine ===")
    engine = BacktestEngine(
        factor_path="data/processed/factors_all_repaired.csv",
        regime_path="data/processed/regime_predictions.csv",
        db_path="data/processed/etf.sqlite",
        data_mode="etf",
        price_mode="total_return_proxy",
    )

    results_all = []
    for name in STRATEGIES:
        print(f"\n=== Running {name} ===")
        daily = engine.run_backtest(
            name,
            start="2018-01-01",
            end="2026-07-17",
            n_hold=20,
            max_weight=0.05,
        )

        # Sanity checks
        ret = daily["return"]
        if ret.isna().any():
            print(f"  WARNING: {ret.isna().sum()} NaN returns detected!")
        if (ret == float("inf")).any() or (ret == float("-inf")).any():
            print(f"  WARNING: Infinite return values detected!")

        metrics = engine.calculate_metrics(
            daily["return"],
            daily["turnover"],
            daily["transaction_cost"],
            daily["gross_return"],
        )
        metrics.update(
            {
                "strategy": name,
                "regime": pd.NA,
                "avg_turnover": daily["turnover"].mean() * 100,
                "n_trades": int((daily["traded_notional"] > 1e-12).sum()),
                "avg_exposure": daily["exposure"].mean() * 100,
                "pit_status": engine.pit_status,
                "data_mode": engine.data_mode,
                "price_mode": engine.price_mode,
                "signal_to_return_lag": 2,
                "rebalance_every": 5,
            }
        )
        results_all.append(metrics)

        print(
            f"  AnnRet={metrics['ann_return']:.2f}%, "
            f"MaxDD={metrics['max_drawdown']:.2f}%, "
            f"Sharpe={metrics['sharpe']:.2f}, "
            f"WinRate={metrics['win_rate']:.1f}%"
        )

    result_df = pd.DataFrame(results_all)
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Compare with old results if available
    old_path = "data/processed/backtest_results.csv"
    if Path(old_path).exists():
        print("\n=== Comparing with old results ===")
        old_df = pd.read_csv(old_path)
        common = set(STRATEGIES) & set(old_df["strategy"])
        for name in sorted(common):
            old_row = old_df[old_df["strategy"] == name].iloc[0]
            new_row = result_df[result_df["strategy"] == name].iloc[0]
            print(f"\n  {name}:")
            for col in ["ann_return", "sharpe", "max_drawdown", "total_turnover"]:
                old_val = old_row.get(col, float("nan"))
                new_val = new_row.get(col, float("nan"))
                diff = new_val - old_val if not (pd.isna(old_val) or pd.isna(new_val)) else float("nan")
                print(f"    {col}: old={old_val:.2f}, new={new_val:.2f}, diff={diff:+.2f}")


if __name__ == "__main__":
    main()
