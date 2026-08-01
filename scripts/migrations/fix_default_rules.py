"""Insert default rules for expanded pool + add 40 rule funds to catalog."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import pandas as pd
import numpy as np
import akshare as ak
from src.build_expanded_pool import generate_default_rules

DB = Path("data/processed/otf_expanded.sqlite")
conn = sqlite3.connect(str(DB))
cur = conn.cursor()

# --- Add existing 40 rule funds to catalog ---
rules = pd.read_csv("config/otf_product_rules.csv", dtype={"fund_code": str})
existing_codes = set()
try:
    df = pd.read_sql("SELECT fund_code FROM otf_fund_catalog", conn)
    existing_codes = set(df["fund_code"].astype(str).str.strip().str.zfill(6))
except Exception:
    pass

# Fetch fund names
all_funds = ak.fund_name_em()
all_funds["基金代码"] = all_funds["基金代码"].astype(str).str.strip().str.zfill(6)
name_map = dict(zip(all_funds["基金代码"], all_funds["基金简称"]))
type_map = dict(zip(all_funds["基金代码"], all_funds["基金类型"]))

from datetime import datetime
now = datetime.now().isoformat()
for code in rules["fund_code"].str.strip().str.zfill(6):
    if code not in existing_codes:
        name = name_map.get(code, "")
        ftype = type_map.get(code, "")
        cur.execute("""INSERT OR REPLACE INTO otf_fund_catalog
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (code, name, ftype, "", "", "", "", "product_rules", now))
        print(f"  Added to catalog: {code} {name}")

conn.commit()

# --- Generate default rules for all catalog funds without existing rules ---
# Also load the expanded 448 funds rules
db_448 = Path("data/processed/otf_research.sqlite")
conn448 = sqlite3.connect(str(db_448))
try:
    existing_rules = set()
    df = pd.read_sql("SELECT fund_code FROM otf_default_rules", conn)
    existing_rules = set(df["fund_code"].astype(str).str.strip().str.zfill(6))
except:
    existing_rules = set()

# Get 448 rules
rules_448 = {}
try:
    df = pd.read_sql("SELECT fund_code FROM otf_fund_catalog", conn448)
    for c in df["fund_code"]:
        rules_448[c] = None  # marker
except:
    pass
conn448.close()

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

print(f"Fee data: {len(fee_map)} funds")

# Get all codes from catalog that don't have rules yet
codes_missing = set()
df = pd.read_sql("SELECT fund_code FROM otf_fund_catalog", conn)
for c in df["fund_code"]:
    if c not in existing_rules:
        codes_missing.add(c)

print(f"Missing rules: {len(codes_missing)} funds")

inserted = 0
errors = 0
for code in codes_missing:
    if code in fee_map:
        fee_info = fee_map[code]
        ftype = fee_info.get("fund_type", "")
        rules = generate_default_rules(fee_info, ftype)
        try:
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
                 ftype,
                 fee_info.get("sub_status", ""),
                 fee_info.get("red_status", "")))
            inserted += 1
        except Exception as e:
            errors += 1
            print(f"  Error {code}: {e}")
    else:
        # No fee data, use defaults
        try:
            cur.execute("""INSERT OR REPLACE INTO otf_default_rules
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (code,
                 0.0, 1, 1, 7, 0, 1, 1,
                 type_map.get(code, ""),
                 "", ""))
            inserted += 1
        except Exception as e:
            errors += 1

conn.commit()

# Summary
cur.execute("SELECT COUNT(*) FROM otf_fund_catalog")
cat_cnt = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM otf_default_rules")
rule_cnt = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM otf_fund_nav")
nav_cnt = cur.fetchone()[0]

conn.close()

print(f"\nSummary:")
print(f"  Catalog: {cat_cnt}")
print(f"  Rules: {rule_cnt}")
print(f"  NAV rows: {nav_cnt}")
print(f"  Rules inserted: {inserted}, errors: {errors}")
