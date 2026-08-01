"""Build expanded 1000-fund research pool from akshare.

Stratified sampling from 27k fund universe, download NAV + fees,
generate default product rules, store in otf_expanded.sqlite.
"""

from __future__ import annotations
import sys
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import akshare as ak

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from src.fetch_otf_funds import fetch_fund_nav

# ================================================================
# Config
# ================================================================

TARGET_TOTAL = 1000
EXPANDED_DB = Path("data/processed/otf_expanded.sqlite")
CHECKPOINT = Path("data/processed/otf_expanded_checkpoint.json")

# Fund type categories for stratified sampling
CATEGORIES: dict[str, list[str]] = {
    "指数型-股票": ["指数型-股票"],
    "指数型-海外/固收": ["指数型-海外股票", "指数型-固收", "指数型-其他"],
    "混合型-偏股": ["混合型-偏股", "混合型-灵活"],
    "混合型-偏债": ["混合型-偏债", "混合型-平衡", "混合型-绝对收益"],
    "股票型": ["股票型"],
    "债券型-利率/信用": ["债券型-长债", "债券型-利率债", "债券型-信用债"],
    "债券型-混合/短债": ["债券型-混合二级", "债券型-混合一级", "债券型-中短债"],
    "货币型": ["货币型-普通货币", "货币型-浮动净值"],
    "QDII": ["QDII-混合偏股", "QDII-普通股票", "QDII-纯债", "QDII-混合灵活",
             "QDII-混合债", "QDII-商品", "QDII-FOF", "QDII-混合平衡", "QDII-REITs"],
    "FOF": ["FOF-稳健型", "FOF-均衡型", "FOF-进取型"],
    "REITs/商品": ["Reits", "商品"],
}


# ================================================================
# 1. Stratified Sampling
# ================================================================

def stratified_sample(fund_df: pd.DataFrame, target: int,
                      random_state: int = 42) -> pd.DataFrame:
    """Proportional stratified sampling from fund_list by fund_type."""
    rng = np.random.default_rng(random_state)
    sample_dfs = []

    # Build type->category mapping
    type_to_cat: dict[str, str] = {}
    for cat_name, types in CATEGORIES.items():
        for t in types:
            type_to_cat[t] = cat_name

    fund_df = fund_df.copy()
    fund_df["_category"] = fund_df["基金类型"].map(type_to_cat).fillna("其他")
    total = len(fund_df)

    allocated = {}
    for cat_name, types in CATEGORIES.items():
        cat_count = int(fund_df["基金类型"].isin(types).sum())
        raw = target * cat_count / total
        allocated[cat_name] = max(1, round(raw))

    # Adjust rounding to hit exactly target
    diff = target - sum(allocated.values())
    sorted_cats = sorted(allocated.items(), key=lambda x: -x[1])
    for i in range(abs(diff)):
        idx = i % len(sorted_cats)
        cat = sorted_cats[idx][0]
        allocated[cat] += 1 if diff > 0 else -1

    for cat_name in CATEGORIES:
        types = CATEGORIES[cat_name]
        pool = fund_df[fund_df["基金类型"].isin(types)]
        n = min(allocated[cat_name], len(pool))
        if n <= 0:
            continue
        chosen = pool.sample(n=n, random_state=rng, replace=False)
        sample_dfs.append(chosen)

    result = pd.concat(sample_dfs, ignore_index=True)
    return result.drop(columns=["_category"])


# ================================================================
# 2. Fee & Default Rules
# ================================================================

def fetch_fee_data() -> dict[str, dict]:
    """Fetch purchase fee data from akshare."""
    df = ak.fund_purchase_em()
    fee_map: dict[str, dict] = {}
    for _, r in df.iterrows():
        code = str(r["基金代码"]).strip().zfill(6)
        fee_map[code] = {
            "subscription_fee_pct": float(r["手续费"]) if pd.notna(r["手续费"]) else 0.0,
            "fund_type": str(r["基金类型"]),
            "sub_status": str(r["申购状态"]),
            "red_status": str(r["赎回状态"]),
        }
    return fee_map


def generate_default_rules(fee_info: dict, fund_type: str) -> dict:
    """Generate default product rules based on fee data and fund type."""
    sub_fee_pct = fee_info.get("subscription_fee_pct", 0.0)

    # Determine confirmation days
    type_lower = fund_type.lower()
    if "货币" in type_lower:
        conf_days = 0
        settle_days = 1
        min_hold = 0
    elif "qdii" in type_lower or "海外" in type_lower:
        conf_days = 2
        settle_days = 7
        min_hold = 0
    else:
        conf_days = 1
        settle_days = 7
        min_hold = 0

    return {
        "subscription_fee_rate": round(sub_fee_pct / 100, 6),
        "subscription_confirmation_days": conf_days,
        "redemption_confirmation_days": conf_days,
        "redemption_settlement_days": settle_days,
        "minimum_holding_calendar_days": min_hold,
        "subscription_open": 1,
        "redemption_open": 1,
    }


# ================================================================
# 3. NAV Download (with checkpointing)
# ================================================================

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "skipped": []}


def save_checkpoint(cp: dict):
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)


def download_nav(code: str, max_retries: int = 3) -> pd.DataFrame | None:
    """Download NAV data with retries."""
    for attempt in range(max_retries):
        try:
            return fetch_fund_nav(code)
        except Exception as e:
            if "Data_netWorthTrend" in str(e):
                # Money market funds don't have standard NAV
                return None
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


# ================================================================
# 4. Build Database
# ================================================================

def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS otf_fund_catalog (
            fund_code TEXT PRIMARY KEY,
            fund_name TEXT,
            fund_type TEXT,
            asset_class TEXT,
            underlying_name TEXT,
            inception_date TEXT,
            termination_date TEXT,
            source TEXT,
            fetched_at TEXT
        );

        CREATE TABLE IF NOT EXISTS otf_fund_nav (
            fund_code TEXT,
            nav_date TEXT,
            unit_nav REAL,
            cumulative_nav REAL,
            daily_growth_pct REAL,
            cumulative_nav_imputed INTEGER,
            distribution_per_share REAL,
            share_adjustment_factor REAL,
            total_return_factor REAL,
            PRIMARY KEY (fund_code, nav_date)
        );

        CREATE TABLE IF NOT EXISTS otf_default_rules (
            fund_code TEXT PRIMARY KEY,
            subscription_fee_rate REAL,
            subscription_confirmation_days INTEGER,
            redemption_confirmation_days INTEGER,
            redemption_settlement_days INTEGER,
            minimum_holding_calendar_days INTEGER,
            subscription_open INTEGER,
            redemption_open INTEGER,
            fund_type TEXT,
            sub_status TEXT,
            red_status TEXT
        );
    """)
    conn.commit()


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 60)
    print("Build Expanded 1000-Fund Research Pool")
    print("=" * 60)

    # --- Step 1: Get full fund list from akshare ---
    print("\n[1/5] Getting fund list from akshare...")
    all_funds = ak.fund_name_em()
    all_funds["基金代码"] = all_funds["基金代码"].astype(str).str.strip().str.zfill(6)
    print(f"  Total: {len(all_funds)} funds")

    # --- Step 2: Stratified sample 1000 ---
    print("\n[2/5] Stratified sampling...")

    # Load existing 448 research DB to exclude
    existing = set()
    try:
        conn = sqlite3.connect("data/processed/otf_research.sqlite")
        existing = set(pd.read_sql("SELECT fund_code FROM otf_fund_catalog", conn)["fund_code"].astype(str).str.strip().str.zfill(6))
        conn.close()
        print(f"  Existing research pool: {len(existing)} funds")
    except Exception:
        print("  No existing research DB, starting fresh")

    # Also add current 40 rule funds
    try:
        rules = pd.read_csv("config/otf_product_rules.csv", dtype={"fund_code": str})
        for c in rules["fund_code"]:
            existing.add(c.strip().zfill(6))
    except Exception:
        pass

    # Filter out existing, only sample new funds
    new_pool = all_funds[~all_funds["基金代码"].isin(existing)]
    target_new = max(TARGET_TOTAL - len(existing), 1)
    print(f"  Need to sample: {target_new} new funds")

    sampled = stratified_sample(new_pool, target_new)
    print(f"  Sampled: {len(sampled)} funds")

    # Distribution
    print("\n  Sample distribution:")
    for cat_name in CATEGORIES:
        types = CATEGORIES[cat_name]
        count = int(sampled["基金类型"].isin(types).sum())
        bar = "#" * (count // 5)
        print(f"    {cat_name:25s} {count:4d} {bar}")

    # --- Step 3: Fetch fee data ---
    print("\n[3/5] Fetching fee data...")
    fee_map = fetch_fee_data()
    fee_df = pd.DataFrame.from_dict(fee_map, orient="index")
    # Match fees with sampled funds
    sampled_fees = fee_df[fee_df.index.isin(sampled["基金代码"])]
    print(f"  Fee data matched: {len(sampled_fees)}/{len(sampled)}")

    # --- Step 4: Download NAV ---
    print("\n[4/5] Downloading NAV data (this takes ~10 minutes)...")
    cp = load_checkpoint()

    conn = sqlite3.connect(str(EXPANDED_DB))
    init_db(conn)
    cur = conn.cursor()

    # Insert catalog entries
    now = datetime.now().isoformat()
    inserted_catalog = 0
    for _, r in sampled.iterrows():
        code = r["基金代码"]
        if code in existing:
            continue
        cur.execute("""INSERT OR REPLACE INTO otf_fund_catalog
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (code, str(r.get("基金简称", "")), str(r.get("基金类型", "")),
             "", "", "", "", "akshare_fund_name_em", now))
        inserted_catalog += 1
    conn.commit()
    print(f"  Catalog entries: {inserted_catalog}")

    # Download NAV for each new fund
    nav_count = 0
    success_count = 0
    skip_count = 0
    fail_count = 0

    for _, r in sampled.iterrows():
        code = r["基金代码"]
        if code in existing or code in cp.get("completed", []):
            skip_count += 1
            continue
        if code in cp.get("failed", []):
            fail_count += 1
            continue

        # Skip money market funds (NAV API doesn't work)
        ftype = str(r.get("基金类型", ""))
        if "货币" in ftype:
            cp["skipped"].append(code)
            save_checkpoint(cp)
            skip_count += 1
            continue

        df = download_nav(code)
        if df is None or len(df) == 0:
            cp["failed"].append(code)
            save_checkpoint(cp)
            fail_count += 1
            print(f"    FAIL {code} ({ftype})")
            continue

        # Insert NAV records
        nav_batch = []
        for _, row in df.iterrows():
            nav_batch.append((
                code, str(row["date"])[:10],
                float(row["nav"]) if pd.notna(row["nav"]) else None,
                float(row["cumulative_nav"]) if pd.notna(row["cumulative_nav"]) else None,
                float(row["daily_growth_pct"]) if pd.notna(row["daily_growth_pct"]) else None,
                int(row.get("cumulative_nav_imputed", False)),
                0.0, 1.0, 1.0,
            ))
        cur.executemany("INSERT OR REPLACE INTO otf_fund_nav VALUES (?,?,?,?,?,?,?,?,?)", nav_batch)
        conn.commit()
        nav_count += len(nav_batch)
        success_count += 1
        cp["completed"].append(code)
        save_checkpoint(cp)

        if success_count % 50 == 0:
            print(f"    Progress: {success_count}/{target_new} funds, {nav_count} NAV rows")

    print(f"\n  NAV download complete: {success_count} success, {fail_count} fail, {skip_count} skip")
    print(f"  Total NAV rows: {nav_count}")

    # --- Step 5: Generate default rules ---
    print("\n[5/5] Generating default product rules...")
    rules_inserted = 0
    for code in fee_map:
        if code not in sampled["基金代码"].values:
            continue
        if code in existing:
            continue
        fee_info = fee_map[code]
        ftype = fee_info.get("fund_type", "")
        rules = generate_default_rules(fee_info, ftype)
        cur.execute("""INSERT OR REPLACE INTO otf_default_rules VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (code, rules["subscription_fee_rate"],
             rules["subscription_confirmation_days"],
             rules["redemption_confirmation_days"],
             rules["redemption_settlement_days"],
             rules["minimum_holding_calendar_days"],
             rules["subscription_open"],
             rules["redemption_open"],
             ftype, fee_info.get("sub_status", ""), fee_info.get("red_status", "")))
        rules_inserted += 1
    conn.commit()
    print(f"  Default rules generated: {rules_inserted}")

    # Summary
    cur.execute("SELECT COUNT(*) FROM otf_fund_catalog")
    cat_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM otf_fund_nav")
    nav_total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM otf_default_rules")
    rule_count = cur.fetchone()[0]
    conn.close()

    print(f"\n{'=' * 60}")
    print(f"Database built: {EXPANDED_DB}")
    print(f"  Catalog: {cat_count} funds")
    print(f"  NAV: {nav_total} rows")
    print(f"  Default rules: {rule_count}")
    print(f"{'=' * 60}")

    # Clean checkpoint
    CHECKPOINT.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
