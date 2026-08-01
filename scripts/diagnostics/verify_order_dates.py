import pandas as pd
from pathlib import Path

holiday_dates = {"2022-03-05", "2022-03-20", "2022-12-31", "2023-01-02", "2023-05-02", "2023-12-31", "2024-06-30"}

c1_dir = Path("reports/strategy_research/core_satellite/core_satellite_20260730_124538")
c2_dir = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_125734")

for label, strategy_dir in [("C1", c1_dir / "C1_C1_CORE_SATELLITE_MOMENTUM"), ("C2", c2_dir / "C2_LOW_TURNOVER_CORE_SATELLITE")]:
    orders_path = strategy_dir / "orders.csv"
    if not orders_path.exists():
        print(f"{label}: orders.csv not found at {orders_path}")
        continue
    
    orders = pd.read_csv(orders_path)
    orders["submit_date"] = pd.to_datetime(orders["submit_date"]).dt.strftime("%Y-%m-%d")
    orders["confirmation_date"] = pd.to_datetime(orders["confirmation_date"]).dt.strftime("%Y-%m-%d")
    orders["arrival_date"] = pd.to_datetime(orders.get("arrival_date", "")).dt.strftime("%Y-%m-%d")
    
    bad_submit = set(orders["submit_date"]) & holiday_dates
    bad_confirm = set(orders["confirmation_date"].dropna()) & holiday_dates
    bad_arrival = set(orders["arrival_date"].dropna()) & holiday_dates
    
    print(f"\n{label} orders:")
    print(f"  Total orders: {len(orders)}")
    print(f"  Bad submit dates (on holidays): {bad_submit}")
    print(f"  Bad confirmation dates (on holidays): {bad_confirm}")
    print(f"  Bad arrival dates (on holidays): {bad_arrival}")

# Check C2 signal date 2022-12-30 -> next execution day 2023-01-03
c2_orders = pd.read_csv(c2_dir / "C2_LOW_TURNOVER_CORE_SATELLITE" / "orders.csv")
c2_orders["signal_date"] = pd.to_datetime(c2_orders["signal_date"])
c2_orders["submit_date"] = pd.to_datetime(c2_orders["submit_date"])
dec30_signals = c2_orders[c2_orders["signal_date"].dt.strftime("%Y-%m-%d") == "2022-12-30"]
print(f"\nC2 orders with signal_date 2022-12-30: {len(dec30_signals)}")
if len(dec30_signals) > 0:
    submit_dates = set(dec30_signals["submit_date"].dt.strftime("%Y-%m-%d"))
    print(f"  Submit dates: {submit_dates}")
    if "2023-01-03" in submit_dates:
        print("  PASS: Next execution day after 2022-12-30 is correctly 2023-01-03")
    else:
        print(f"  FAIL: Expected submit on 2023-01-03, got {submit_dates}")