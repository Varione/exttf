"""Consolidate: add 448 research-db funds into expanded pool catalog."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import pandas as pd
from datetime import datetime

DB_EXPANDED = Path("data/processed/otf_expanded.sqlite")
DB_RESEARCH = Path("data/processed/otf_research.sqlite")

conn_old = sqlite3.connect(str(DB_RESEARCH))
conn_new = sqlite3.connect(str(DB_EXPANDED))

# --- Catalog: add missing research pool funds ---
cats_old = pd.read_sql("SELECT * FROM otf_fund_catalog", conn_old)
cats_new = pd.read_sql("SELECT fund_code FROM otf_fund_catalog", conn_new)
existing = set(cats_new["fund_code"].astype(str).str.strip().str.zfill(6))

now = datetime.now().isoformat()
added_catalog = 0
cur = conn_new.cursor()
for _, r in cats_old.iterrows():
    code = str(r.get("fund_code", "")).strip().zfill(6)
    if code not in existing:
        cur.execute("""INSERT OR REPLACE INTO otf_fund_catalog VALUES (?,?,?,?,?,?,?,?,?)""",
            (code,
             r.get("fund_name", ""),
             r.get("fund_type", ""),
             r.get("asset_class", ""),
             r.get("underlying_name", ""),
             r.get("inception_date", ""),
             r.get("termination_date", ""),
             "otf_research.sqlite",
             now))
        added_catalog += 1
        existing.add(code)

conn_new.commit()
print(f"Added to catalog: {added_catalog}")

# --- NAV: copy missing data ---
nav_old = pd.read_sql("SELECT fund_code, nav_date, unit_nav, cumulative_nav, daily_growth_pct FROM otf_fund_nav", conn_old)
nav_new = pd.read_sql("SELECT fund_code, nav_date FROM otf_fund_nav", conn_new)
new_keys = set(zip(nav_new["fund_code"].astype(str), nav_new["nav_date"].astype(str)))

batch = []
for _, r in nav_old.iterrows():
    code = str(r.get("fund_code", "")).strip().zfill(6)
    dt = str(r.get("nav_date", ""))[:10]
    key = (code, dt)
    if key not in new_keys:
        batch.append((code, dt,
            float(r["unit_nav"]) if pd.notna(r.get("unit_nav")) else None,
            float(r["cumulative_nav"]) if pd.notna(r.get("cumulative_nav")) else None,
            float(r["daily_growth_pct"]) if pd.notna(r.get("daily_growth_pct")) else None))
        new_keys.add(key)

    if len(batch) >= 5000:
        cur.executemany("""INSERT OR REPLACE INTO otf_fund_nav (fund_code, nav_date, unit_nav, cumulative_nav, daily_growth_pct) VALUES (?,?,?,?,?)""", batch)
        conn_new.commit()
        batch = []

if batch:
    cur.executemany("""INSERT OR REPLACE INTO otf_fund_nav (fund_code, nav_date, unit_nav, cumulative_nav, daily_growth_pct) VALUES (?,?,?,?,?)""", batch)
    conn_new.commit()

print(f"Added NAV rows: {len(nav_old)} batch")

# We skip copying rules from research DB since we already generated
# default rules for all funds. The research DB rules may or may not
# be compatible.

# --- Summary ---
cur.execute("SELECT COUNT(*) FROM otf_fund_catalog")
cat_cnt = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM otf_fund_nav")
nav_cnt = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM otf_default_rules")
rule_cnt = cur.fetchone()[0]

conn_old.close()
conn_new.close()

print(f"\nFinal summary:")
print(f"  Catalog: {cat_cnt} funds")
print(f"  NAV: {nav_cnt} rows")
print(f"  Default rules: {rule_cnt}")
