import json, sys
from pathlib import Path

old_c2 = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260729_163519")
new_c2 = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_115448")

keys = ["net_cagr_pct", "sharpe", "mdd_pct", "calmar", "total_fee_amount", "worst_two_year_cagr_pct"]
for strategy in ["C2_LOW_TURNOVER_CORE_SATELLITE", "B2_Static_EW_4Asset", "B2_LT_Static_EW_4Asset"]:
    old_metrics = json.loads((old_c2 / strategy / "metrics.json").read_text())
    new_metrics = json.loads((new_c2 / strategy / "metrics.json").read_text())
    
    print(f"\n{strategy}:")
    for key in keys:
        old_val = old_metrics.get(key, "N/A")
        new_val = new_metrics.get(key, "N/A")
        if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
            diff = round(new_val - old_val, 4)
            print(f"  {key}: OLD={old_val:.4f} NEW={new_val:.4f} DIFF={diff:+.4f}")
        else:
            print(f"  {key}: OLD={old_val} NEW={new_val}")