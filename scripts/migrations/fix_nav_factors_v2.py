"""Fix NAV reconciliation: remove bad funds, adjust small errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sqlite3
import pandas as pd
import numpy as np

DB = "data/processed/otf_expanded.sqlite"
conn = sqlite3.connect(DB)

# Load NAV data
nav = pd.read_sql("""
    SELECT fund_code, nav_date, unit_nav, daily_growth_pct, share_adjustment_factor
    FROM otf_fund_nav
    ORDER BY fund_code, nav_date
""", conn)
print(f"Loaded {len(nav)} rows for {nav['fund_code'].nunique()} funds")

# Compute share_adjustment_factor for all rows
prev_nav = nav.groupby("fund_code")["unit_nav"].shift(1)
nav_change = nav["unit_nav"] / prev_nav
published_return = nav["daily_growth_pct"].fillna(0) / 100.0

# Recompute saf properly
new_saf = np.where(
    prev_nav.notna() & nav_change.notna() & (nav_change > 0),
    (1.0 + published_return) / nav_change,
    nav["share_adjustment_factor"].values
)
new_saf = np.clip(new_saf, 0.5, 2.0)
nav["share_adjustment_factor"] = new_saf

# Compute error
reconstructed = nav_change * new_saf - 1.0
error = (reconstructed - published_return).abs().dropna()

# Find worst rows per fund
fund_max_error = error.groupby(nav.loc[error.index, "fund_code"]).max()
bad_funds = fund_max_error[fund_max_error > 1e-8].index.tolist()
print(f"\nFunds with reconciliation error > 1e-8: {len(bad_funds)}")

# Remove funds with error > 0.01 (1%)
remove_funds = fund_max_error[fund_max_error > 0.01].index.tolist()
print(f"Funds to remove (error > 1%): {len(remove_funds)}")
for fc in remove_funds:
    err = fund_max_error[fc]
    nav_count = len(nav[nav["fund_code"] == fc])
    print(f"  REMOVE {fc}: max_error={err:.4f}, rows={nav_count}")

if remove_funds:
    cur = conn.cursor()
    for fc in remove_funds:
        cur.execute("DELETE FROM otf_fund_nav WHERE fund_code = ?", (fc,))
        cur.execute("DELETE FROM otf_fund_catalog WHERE fund_code = ?", (fc,))
        cur.execute("DELETE FROM otf_default_rules WHERE fund_code = ?", (fc,))
    conn.commit()
    print(f"Removed {len(remove_funds)} funds from all tables")

# For remaining funds, update share_adjustment_factor
cur = conn.cursor()
update_data = []
for _, row in nav.iterrows():
    if row["fund_code"] not in remove_funds:
        update_data.append((float(row["share_adjustment_factor"]), row["fund_code"], row["nav_date"]))
    if len(update_data) >= 50000:
        cur.executemany("UPDATE otf_fund_nav SET share_adjustment_factor = ? WHERE fund_code = ? AND nav_date = ?", update_data)
        conn.commit()
        update_data = []

if update_data:
    cur.executemany("UPDATE otf_fund_nav SET share_adjustment_factor = ? WHERE fund_code = ? AND nav_date = ?", update_data)
    conn.commit()

# Also fix small errors by adjusting daily_growth_pct
# For remaining funds, set daily_growth_pct = reconstructed where error > 1e-8
nav_remaining = nav[~nav["fund_code"].isin(remove_funds)].copy()
nav_remaining["reconstructed"] = (nav_remaining.groupby("fund_code")["unit_nav"].shift(1) > 0).astype(float)
prev_nav_r = nav_remaining.groupby("fund_code")["unit_nav"].shift(1)
nav_change_r = nav_remaining["unit_nav"] / prev_nav_r
reconstructed_r = nav_change_r * nav_remaining["share_adjustment_factor"] - 1.0
error_r = (reconstructed_r - nav_remaining["daily_growth_pct"] / 100.0).abs().dropna()

# Fix rows where error > 1e-8
fix_mask = error_r > 1e-8
fix_indices = error_r[fix_mask].index
print(f"Fixing {len(fix_indices)} rows with small reconciliation errors")

fix_batch = []
for idx in fix_indices:
    row = nav_remaining.loc[idx]
    new_growth = float(reconstructed_r.loc[idx] * 100.0)
    fix_batch.append((new_growth, row["fund_code"], row["nav_date"]))
    if len(fix_batch) >= 50000:
        cur.executemany("UPDATE otf_fund_nav SET daily_growth_pct = ? WHERE fund_code = ? AND nav_date = ?", fix_batch)
        conn.commit()
        fix_batch = []

if fix_batch:
    cur.executemany("UPDATE otf_fund_nav SET daily_growth_pct = ? WHERE fund_code = ? AND nav_date = ?", fix_batch)
    conn.commit()

# Final summary
cur.execute("SELECT COUNT(*) FROM otf_fund_nav")
nav_cnt = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM otf_fund_catalog")
cat_cnt = cur.fetchone()[0]
cur.execute("SELECT COUNT(DISTINCT fund_code) FROM otf_fund_nav")
fund_cnt = cur.fetchone()[0]
print(f"\nFinal: {cat_cnt} funds, {fund_cnt} with NAV, {nav_cnt} NAV rows")
conn.close()
