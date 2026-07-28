"""Run backtest after P1 fixes on complete data."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from backtest_engine import BacktestEngine
from strategy_library import STRATEGIES, get_all_strategy_names
import pandas as pd

t0 = time.time()

engine = BacktestEngine(
    factor_path="data/processed/factors_all_repaired.csv",
    regime_path="data/processed/regime_predictions.csv",
    db_path="data/processed/etf.sqlite",
    data_mode="etf",
    price_mode="total_return_proxy",
)

# Run key strategies
targets = ["S04_VolTarget_Trend", "S23_Oversold_Long", "B0_BuyHold_EW",
           "S01_CS_Momentum", "S12_Low_Volatility", "S13_Quality"]

results_all = []
for name in targets:
    StrategyClass = STRATEGIES[name]
    print(f"\nRunning {name} (regime={StrategyClass.regime})...")
    
    bt = engine.run_backtest(name, start="2018-01-01", end="2026-07-17",
                              n_hold=20, max_weight=0.05)
    
    metrics = engine.calculate_metrics(
        bt["return"], bt["turnover"], bt["transaction_cost"], bt["gross_return"],
        exposures=bt["exposure"]
    )
    metrics["strategy"] = name
    metrics["regime"] = StrategyClass.regime
    metrics["avg_turnover"] = bt["turnover"].mean() * 100
    metrics["avg_exposure"] = bt["exposure"].mean() * 100
    results_all.append(metrics)

df = pd.DataFrame(results_all)

print("\n" + "=" * 130)
print("P1 Backtest Results: 2018-01-01 ~ 2026-07-17 (Complete Data)")
print("=" * 130)
print(f"{'Strategy':<25} {'Regime':>6} {'AnnRet%':>8} {'Sharpe':>7} "
      f"{'MaxDD%':>8} {'WinRate%':>8} {'ActWin%':>7} {'Exposure%':>9}")
print("-" * 130)

for _, row in df.iterrows():
    act_win = row.get("active_win_rate", "N/A")
    act_str = f"{act_win:>6.1f}%" if isinstance(act_win, (int, float)) else f"{'N/A':>7}"
    print(f"{row['strategy']:<25} {row['regime']:>6} {row['ann_return']:>+7.2f}% "
          f"{row['sharpe']:>7.2f} {row['max_drawdown']:>7.2f}% "
          f"{row['win_rate']:>7.1f}% {act_str} {row['avg_exposure']:>8.1f}%")

df.to_csv("data/processed/backtest_results_p1.csv", index=False)
print(f"\nResults saved to data/processed/backtest_results_p1.csv")
print(f"Total time: {time.time()-t0:.1f}s")
