"""Generate default rules for all 1000 funds in expanded DB."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import pandas as pd
import akshare as ak
from src.build_expanded_pool import generate_default_rules

DB = Path("data/processed/otf_expanded.sqlite")
conn = sqlite3.connect(str(DB))
cur = conn.cursor()

# Get all catalog codes
cats = pd.read_sql("SELECT fund_code FROM otf_fund_catalog", conn)
codes = set(cats["fund_code"].astype(str).str.strip().str.zfill(6))

# Get existing rules
rules_existing = pd.read_sql("SELECT fund_code FROM otf_default_rules", conn)
existing = set(rules_existing["fund_code"].astype(str).str.strip().str.zfill(6))

missing_codes = codes - existing
print(f"Missing rules: {len(missing_codes)} funds")

# Fetch fee data
print("Fetching fee data...")
fee_df = ak.fund_purchase_em()
fee_map = {}
for _, r in fee_df.iterrows():
    code = str(r["基金代码"]).strip().zfill(6)
    fee_map[code] = {
        "subscription_fee_pct": float(r["手续费"]) if pd.notna(r["手续费"]) else 0.0,
        "fund_type": str(r["基金类型"]),
        "sub_status": str(r["申购状态"]),
        "red_status": str(r["赎回状态"]),
    }

print(f"Fee data loaded: {len(fee_map)}")

# Get fund types from catalog
cat_df = pd.read_sql("SELECT fund_code, fund_type FROM otf_fund_catalog", conn)
type_map = dict(zip(cat_df["fund_code"], cat_df["fund_type"]))

inserted = 0
no_fee = 0
for code in sorted(missing_codes):
    ftype = type_map.get(code, "")
    if code in fee_map:
        fee_info = fee_map[code]
        rules = generate_default_rules(fee_info, ftype)
        cur.execute("""INSERT OR REPLACE INTO otf_default_rules
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (code,
             rules["subscription_fee_rate"],
             rules["subscription_confirmation_days"],
             rules["redemption_confirmation_days"],
             rules["redemption_settlement_days"],
             rules["minimum_holding_calendar_days"],
             rules["subscription_open"],
             rules["redemption_open"],
             fee_info.get("fund_type", ftype),
             fee_info.get("sub_status", ""),
             fee_info.get("red_status", "")))
        inserted += 1
    else:
        cur.execute("""INSERT OR REPLACE INTO otf_default_rules
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (code, 0.0, 1, 1, 7, 0, 1, 1, ftype, "", ""))
        no_fee += 1
        inserted += 1

conn.commit()

cur.execute("SELECT COUNT(*) FROM otf_default_rules")
total = cur.fetchone()[0]
conn.close()

print(f"Inserted: {inserted} (no fee data: {no_fee})")
print(f"Total rules: {total}")
