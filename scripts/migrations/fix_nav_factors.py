"""Compute share_adjustment_factor for expanded DB NAV."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sqlite3
import pandas as pd
import numpy as np

DB = "data/processed/otf_expanded.sqlite"
conn = sqlite3.connect(DB)

# Load NAV data (only needed columns)
nav = pd.read_sql("""
    SELECT fund_code, nav_date, unit_nav, daily_growth_pct
    FROM otf_fund_nav
    WHERE unit_nav > 0 AND daily_growth_pct IS NOT NULL
    ORDER BY fund_code, nav_date
""", conn)
print(f"Loaded {len(nav)} rows for {nav['fund_code'].nunique()} funds")

# Compute share_adjustment_factor per fund
prev_nav = nav.groupby("fund_code")["unit_nav"].shift(1)
nav_change = nav["unit_nav"] / prev_nav

# share_adjustment_factor makes: (nav_change * saf - 1) == published_return
# => saf = (1 + published_return) / nav_change
published_return = nav["daily_growth_pct"] / 100.0
nav["share_adjustment_factor"] = np.where(
    prev_nav.notna() & nav_change.notna() & (nav_change > 0),
    (1.0 + published_return) / nav_change,
    1.0
)

# Clip extreme values (should be near 1.0)
nav["share_adjustment_factor"] = nav["share_adjustment_factor"].clip(0.5, 2.0)

# Verify: compute reconstructed return
nav["reconstructed"] = (nav_change * nav["share_adjustment_factor"] - 1.0)
valid = nav["reconstructed"].notna() & published_return.notna()
max_err = (nav.loc[valid, "reconstructed"] - published_return[valid]).abs().max()
print(f"Max reconciliation error after fix: {max_err:.8f}")

# Update DB
cur = conn.cursor()
update_data = nav[["share_adjustment_factor", "fund_code", "nav_date"]].values.tolist()

cur.executemany("""
    UPDATE otf_fund_nav SET share_adjustment_factor = ?
    WHERE fund_code = ? AND nav_date = ?
""", update_data)
conn.commit()
print(f"Updated {len(update_data)} rows")

# Also reset total_return_factor and cumulative_nav_imputed
cur.execute("UPDATE otf_fund_nav SET total_return_factor = 1.0")
print(f"Reset total_return_factor: {cur.rowcount}")

# Check funds with still-bad data
bad = nav[valid & (nav["reconstructed"] - published_return[valid]).abs() > 1e-4]
if len(bad) > 0:
    print(f"\nFunds with remaining error > 1e-4:")
    print(bad.groupby("fund_code").apply(
        lambda x: pd.Series({
            "rows": len(x),
            "max_err": (x["reconstructed"] - published_return[x.index]).abs().max()
        })
    ).to_string())

conn.close()
print("\nDone")
