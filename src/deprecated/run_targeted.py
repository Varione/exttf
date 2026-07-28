"""Run only S04, S23, B0 strategies."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

t0 = time.time()

from backtest_engine import BacktestEngine
from strategy_library import STRATEGIES

engine = BacktestEngine()

targets = ["S04_VolTarget_Trend", "S23_Oversold_Long", "B0_BuyHold_EW"]
results_all = []

for name in targets:
    StrategyClass = STRATEGIES[name]
    print(f"\nRunning {name} (regime={StrategyClass.regime})...")

    bt_results = engine.run_backtest(name, start="2018-01-01", end="2026-07-17",
                                     n_hold=20, max_weight=0.05)

    metrics = engine.calculate_metrics(bt_results["return"], bt_results["turnover"])
    metrics["strategy"] = name
    metrics["regime"] = StrategyClass.regime
    metrics["avg_turnover"] = bt_results["turnover"].mean() * 100
    metrics["n_trades"] = (bt_results["return"] != 0).sum()

    results_all.append(metrics)

import pandas as pd
df = pd.DataFrame(results_all)

print("\n" + "=" * 110)
print(f"Backtest Results: 2018-01-01 ~ 2026-07-17")
print("=" * 110)
print(f"{'Strategy':<25} {'Regime':>6} {'TotalRet%':>9} {'AnnRet%':>8} "
      f"{'Sharpe':>7} {'MaxDD%':>8} {'WinRate%':>8} {'Turnover%':>9}")
print("-" * 110)

for _, row in df.iterrows():
    print(f"{row['strategy']:<25} {row['regime']:>6} {row['total_return']:>+8.2f}% "
          f"{row['ann_return']:>+7.2f}% {row['sharpe']:>7.2f} "
          f"{row['max_drawdown']:>7.2f}% {row['win_rate']:>7.1f}% "
          f"{row['avg_turnover']:>8.1f}%")

df.to_csv("data/processed/backtest_results.csv", index=False)
print(f"\nResults saved to data/processed/backtest_results.csv")
print(f"\nTotal time: {time.time()-t0:.1f}s")
