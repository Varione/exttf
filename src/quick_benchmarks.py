"""Quick backtest for benchmarks S23 and B0."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

t0 = time.time()
from backtest_engine import BacktestEngine

engine = BacktestEngine()

for name in ["S23_Oversold_Long", "B0_BuyHold_EW"]:
    print(f"\nRunning {name}...")
    bt_results = engine.run_backtest(name, start="2018-01-01", end="2026-07-17",
                                     n_hold=20, max_weight=0.05)
    metrics = engine.calculate_metrics(bt_results["return"], bt_results["turnover"])
    
    print(f"  Total Return:   {metrics['total_return']:+.2f}%")
    print(f"  Annual Return:  {metrics['ann_return']:+.2f}%")
    print(f"  Sharpe Ratio:   {metrics['sharpe']:.4f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.2f}%")
    print(f"  Win Rate:       {metrics['win_rate']:.1f}%")

print(f"\nTotal time: {time.time()-t0:.1f}s")
