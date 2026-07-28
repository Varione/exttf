"""Quick backtest for S04_VolTarget_Trend only."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

t0 = time.time()
from backtest_engine import BacktestEngine

engine = BacktestEngine()
bt_results = engine.run_backtest("S04_VolTarget_Trend", start="2018-01-01", end="2026-07-17",
                                 n_hold=20, max_weight=0.05)
metrics = engine.calculate_metrics(bt_results["return"], bt_results["turnover"])

print(f"\n--- S04_VolTarget_Trend Results ---")
print(f"Total Return:    {metrics['total_return']:+.2f}%")
print(f"Annual Return:   {metrics['ann_return']:+.2f}%")
print(f"Sharpe Ratio:    {metrics['sharpe']:.4f}")
print(f"Max Drawdown:    {metrics['max_drawdown']:.2f}%")
print(f"Win Rate:        {metrics['win_rate']:.1f}%")
print(f"Avg Turnover:    {bt_results['turnover'].mean()*100:.1f}%")
print(f"Est Txn Cost:    {metrics.get('est_transaction_cost', 'N/A')}%")
print(f"\nTotal time: {time.time()-t0:.1f}s")

# Target check
print(f"\n--- Target Check ---")
print(f"AnnRet > 5%:     {metrics['ann_return']:.2f}% -> {'PASS' if metrics['ann_return'] > 5 else 'FAIL'}")
print(f"MaxDD < -15%:    {metrics['max_drawdown']:.2f}% -> {'PASS' if metrics['max_drawdown'] > -15 else 'FAIL'}")
print(f"Sharpe > 0.5:    {metrics['sharpe']:.4f}   -> {'PASS' if metrics['sharpe'] > 0.5 else 'FAIL'}")
