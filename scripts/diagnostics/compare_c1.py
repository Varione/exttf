import json, sys
from pathlib import Path

old_c1 = Path("reports/strategy_research/core_satellite/core_satellite_20260729_151927")
new_c1 = Path("reports/strategy_research/core_satellite/core_satellite_20260730_114756")

for strategy in ["C1_CORE_SATELLITE_MOMENTUM", "B2_Static_EW_4Asset", "B2_LT_Static_EW_4Asset"]:
    old_metrics = json.loads((old_c1 / strategy / "metrics.json").read_text())
    new_metrics = json.loads((new_c1 / strategy / "metrics.json").read_text())
    
    print(f"\n{strategy}:")
    for key in ["Net_CAGR%", "Sharpe", "Max_Drawdown%", "Annualized_Turnover", "Cumulative_Cost_Ratio%"]:
        old_val = old_metrics.get(key, "N/A")
        new_val = new_metrics.get(key, "N/A")
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            diff = round(new_val - old_val, 4)
            print(f"  {key}: OLD={old_val:.4f} NEW={new_val:.4f} DIFF={diff:+.4f}")
        else:
            print(f"  {key}: OLD={old_val} NEW={new_val}")