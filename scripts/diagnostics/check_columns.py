import sys, pandas as pd
sys.path.insert(0, "src")
import sqlite3, tempfile
from pathlib import Path
from otf_backtest_engine import OTFBacktestEngine
from otf_rotation.execution_calendar import ExecutionCalendar

tmp = Path(tempfile.mkdtemp())
db_path = tmp / "test.sqlite"
conn = sqlite3.connect(str(db_path))
conn.execute("""CREATE TABLE otf_fund_catalog (fund_code TEXT PRIMARY KEY, share_class TEXT, fund_name TEXT, asset_class TEXT, benchmark TEXT, inception_date TEXT, termination_date TEXT)""")
conn.execute("""CREATE TABLE otf_fund_nav (fund_code TEXT, nav_date TEXT, unit_nav REAL, cumulative_nav REAL, daily_growth_pct REAL, distribution_per_share REAL, share_adjustment_factor REAL, total_return_factor REAL)""")
conn.execute("INSERT INTO otf_fund_catalog VALUES ('B001','A','Fund B','equity','CSI300','2024-01-01','')")
conn.executemany("INSERT INTO otf_fund_nav VALUES (?,?,?,?,?,?,?,?)", [
    ("B001", "2024-06-28", 1.0, 1.0, 0.0, 0.0, 1.0, 1.0),
    ("B001", "2024-07-01", 1.02, 1.0, 2.0, 0.0, 1.0, 1.0),
])
conn.commit(); conn.close()

calendar = ExecutionCalendar(
    dates=pd.DatetimeIndex(["2024-06-28", "2024-07-01"]),
    path="test.csv", source="TEST", consensus_count=0,
    primary_source_max_date="2024-07-01", input_hashes={}, extension_codes=(),
)

engine = OTFBacktestEngine(db_path=str(db_path), execution_calendar=calendar, initial_cash=100_000.0, subscription_fee_rate=0.0)
targets = pd.DataFrame([[1.0], [1.0]], index=pd.to_datetime(["2024-06-28", "2024-07-01"]), columns=["B001"])
daily = engine.run_backtest(targets, start="2024-06-28", end="2024-07-01", rebalance_every=1)
print("Columns:", list(daily.columns))
print(daily[["date"] + [c for c in daily.columns if "equity" in c.lower() or "cash" in c.lower()]])