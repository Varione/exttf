import json, sys
from pathlib import Path

old_c2 = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260729_163519")
new_c2 = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_115448")

for label, d in [("OLD", old_c2), ("NEW", new_c2)]:
    metrics = json.loads((d / "C2_LOW_TURNOVER_CORE_SATELLITE" / "metrics.json").read_text())
    orders_path = d / "C2_LOW_TURNOVER_CORE_SATELLITE" / "orders.csv"
    if orders_path.exists():
        import pandas as pd
        orders = pd.read_csv(orders_path)
        print(f"\n{label} C2:")
        print(f"  Orders count: {len(orders)}")
        print(f"  Subscribe orders: {(orders['side']=='subscribe').sum()}")
        print(f"  Redeem orders: {(orders['side']=='redeem').sum()}")
        print(f"  Confirmed: {(orders['status'].isin(['confirmed','settled'])).sum()}")
        print(f"  Pending: {(orders['status']=='pending').sum()}")
        print(f"  Cancelled: {(orders['status']=='cancelled').sum()}")
        # Check for any orders around holiday dates
        holidays = ['2022-12-31', '2023-01-02']
        for h in holidays:
            count = (orders['submit_date'].str.contains(h) | orders['confirmation_date'].str.contains(h)).sum()
            print(f"  Orders touching {h}: {count}")
    else:
        print(f"\n{label} C2: No orders.csv")

print("\n--- KEY METRICS COMPARISON ---")
old_m = json.loads((old_c2 / "C2_LOW_TURNOVER_CORE_SATELLITE" / "metrics.json").read_text())
new_m = json.loads((new_c2 / "C2_LOW_TURNOVER_CORE_SATELLITE" / "metrics.json").read_text())
for key in ["net_cagr_pct", "sharpe", "mdd_pct", "total_fee_amount"]:
    print(f"{key}: OLD={old_m.get(key, 'N/A'):.4f} NEW={new_m.get(key, 'N/A'):.4f} DIFF={new_m.get(key, 0) - old_m.get(key, 0):+.4f}")