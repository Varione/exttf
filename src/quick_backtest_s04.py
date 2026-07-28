"""Quick backtest for S04_VolTarget_Trend + benchmarks."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

t0 = time.time()
from backtest_engine import BacktestEngine

engine = BacktestEngine()
targets = ["S04_VolTarget_Trend", "S23_Oversold_Long", "B0_BuyHold_EW"]

results_all = []
for name in targets:
    print(f"Running {name}...")
    bt_results = engine.run_backtest(name, start="2018-01-01", end="2026-07-17",
                                     n_hold=20, max_weight=0.05)
    metrics = engine.calculate_metrics(bt_results["return"], bt_results["turnover"])
    metrics["strategy"] = name
    metrics["avg_turnover"] = bt_results["turnover"].mean() * 100
    metrics["n_trades"] = (bt_results["return"] != 0).sum()
    results_all.append(metrics)

import pandas as pd
df = pd.DataFrame(results_all)

print("\n" + "=" * 110)
print(f"Backtest Results: 2018-01-01 ~ 2026-07-17")
print("=" * 110)
print(f"{'Strategy':<25} {'TotalRet%':>9} {'AnnRet%':>8} "
      f"{'Sharpe':>7} {'MaxDD%':>8} {'WinRate%':>8} {'Turnover%':>9}")
print("-" * 110)

for _, row in df.iterrows():
    print(f"{row['strategy']:<25} {row['total_return']:>+8.2f}% "
          f"{row['ann_return']:>+7.2f}% {row['sharpe']:>7.2f} "
          f"{row['max_drawdown']:>7.2f}% {row['win_rate']:>7.1f}% "
          f"{row['avg_turnover']:>8.1f}%")

print(f"\nTotal time: {time.time()-t0:.1f}s")

# Target check for S04
s04 = df[df["strategy"] == "S04_VolTarget_Trend"].iloc[0]
print(f"\n--- S04 Target Check ---")
print(f"AnnRet > 5%:     {s04['ann_return']:.2f}% -> {'PASS' if s04['ann_return'] > 5 else 'FAIL'}")
print(f"MaxDD < -15%:    {s04['max_drawdown']:.2f}% -> {'PASS' if s04['max_drawdown'] > -15 else 'FAIL'}")
print(f"Sharpe > 0.5:    {s04['sharpe']:.2f}   -> {'PASS' if s04['sharpe'] > 0.5 else 'FAIL'}")
