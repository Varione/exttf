"""P1-1 扩充场外基金数据池：akshare 抓取 + SQLite 存储 + 质量审计"""
import argparse
import os
import csv
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import akshare as ak
import requests

DATA_DIR = Path("data/raw/otf_nav")
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/data_validation")
OTF_DB = PROCESSED_DIR / "otf.sqlite"
MANIFEST_FILE = DATA_DIR / "otf_nav_manifest.csv"

TARGET_FUNDS = {
    # Existing funds (from original manifest)
    "110020": {"name": "易方达沪深300ETF联接A", "asset_class": "宽基股票", "benchmark": "沪深300"},
    "110003": {"name": "易方达上证50增强A", "asset_class": "宽基股票", "benchmark": "上证50"},
    "160119": {"name": "南方中证500ETF联接(LOF)A", "asset_class": "宽基股票", "benchmark": "中证500"},
    "000248": {"name": "汇添富中证主要消费ETF联接A", "asset_class": "行业", "benchmark": "中证主要消费"},
    "001594": {"name": "天弘中证银行ETF联接A", "asset_class": "行业", "benchmark": "中证银行"},
    "000216": {"name": "华安黄金ETF联接A", "asset_class": "黄金", "benchmark": "黄金"},
    "000071": {"name": "华夏恒生ETF联接A", "asset_class": "QDII港股", "benchmark": "恒生指数"},
    "000948": {"name": "华夏沪港通恒生ETF联接A", "asset_class": "QDII港股", "benchmark": "恒生指数"},
    "000015": {"name": "华夏纯债债券A", "asset_class": "债券", "benchmark": "中债总全价"},
    "000395": {"name": "汇添富安心中国债券A", "asset_class": "债券", "benchmark": "中债总全价"},
    # Verified replacements for the previously misclassified fund codes.
    "090010": {"name": "大成中证红利指数A", "asset_class": "红利", "benchmark": "中证红利"},
    "007466": {"name": "华泰柏瑞中证红利低波ETF联接A", "asset_class": "红利", "benchmark": "中证红利低波"},
    "002610": {"name": "博时黄金ETF联接A", "asset_class": "黄金", "benchmark": "黄金"},
    "006662": {"name": "易方达安悦超短债A", "asset_class": "债券短久期", "benchmark": "短期债券"},
    "270042": {"name": "广发纳斯达克100ETF联接人民币(QDII)A", "asset_class": "纳指100", "benchmark": "纳斯达克100"},
    "040046": {"name": "华安纳斯达克100ETF联接(QDII)A", "asset_class": "纳指100", "benchmark": "纳斯达克100"},
    "050025": {"name": "博时标普500ETF联接A", "asset_class": "标普500", "benchmark": "标普500"},
    "161125": {"name": "易方达标普500指数人民币A", "asset_class": "标普500", "benchmark": "标普500"},
}


def fetch_provider_name(fund_code: str) -> str:
    """Read the provider's own fund name and fail on identity mismatch."""
    url = f"https://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    text = response.content.decode("utf-8", errors="replace")
    match = re.search(r'var fS_name = "([^"]+)"', text)
    if not match:
        raise ValueError("Provider fund name is unavailable")
    return match.group(1).strip()

def fetch_fund_nav(fund_code: str) -> pd.DataFrame:
    """Fetch unit NAV and cumulative NAV as two independent series.

    AkShare's ``单位净值走势`` third column is the daily growth rate, not
    cumulative NAV.  The old loader mislabeled that column and later replaced
    it with unit NAV in SQLite.  Fetching ``累计净值走势`` separately prevents
    that silent data corruption.
    """
    unit = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
    cumulative = ak.fund_open_fund_info_em(
        symbol=fund_code, indicator="累计净值走势"
    )
    if unit.empty or cumulative.empty:
        raise ValueError("Unit NAV or cumulative NAV is unavailable")

    unit = unit.iloc[:, :3].copy()
    unit.columns = ["date", "nav", "daily_growth_pct"]
    cumulative = cumulative.iloc[:, :2].copy()
    cumulative.columns = ["date", "cumulative_nav"]

    for frame in (unit, cumulative):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    unit["nav"] = pd.to_numeric(unit["nav"], errors="coerce")
    unit["daily_growth_pct"] = pd.to_numeric(
        unit["daily_growth_pct"], errors="coerce"
    )
    cumulative["cumulative_nav"] = pd.to_numeric(
        cumulative["cumulative_nav"], errors="coerce"
    )

    df = unit.merge(cumulative, on="date", how="left", validate="one_to_one")
    df = (
        df.dropna(subset=["date", "nav"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    df["cumulative_nav_imputed"] = df["cumulative_nav"].isna()
    first_cumulative_date = df.loc[
        df["cumulative_nav"].notna(), "date"
    ].min()
    if pd.isna(first_cumulative_date):
        raise ValueError("Cumulative NAV has no valid observations")
    # Some converted/renamed funds publish cumulative NAV only from the
    # conversion date.  Earlier unit-NAV-only history cannot support dividend
    # accounting, so exclude it rather than silently backfilling.
    df = df.loc[df["date"] >= first_cumulative_date].reset_index(drop=True)
    raw_coverage = float(df["cumulative_nav"].notna().mean())
    # Reconstruct missing cumulative NAV from the latest observed
    # cumulative-minus-unit gap. This is auditable and preserves dividend
    # increments; it is safer than treating daily growth as cumulative NAV.
    cumulative_gap = (df["cumulative_nav"] - df["nav"]).ffill().bfill()
    df["cumulative_nav"] = df["cumulative_nav"].fillna(
        df["nav"] + cumulative_gap
    )
    if df["cumulative_nav"].isna().any():
        raise ValueError(
            f"Cumulative NAV reconstruction failed; raw coverage={raw_coverage:.2%}"
        )
    df.insert(1, "fund_code", fund_code)
    return df

def fetch_all_funds(force: bool = False):
    """Fetch NAV data for all target funds and save CSV + update manifest."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    errors = []

    existing_manifest = {}
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_manifest[row["fund_code"]] = row

    for fund_code, info in TARGET_FUNDS.items():
        # Skip if already successfully fetched
        if (
            not force
            and fund_code in existing_manifest
            and existing_manifest[fund_code].get("status") == "success"
        ):
            print(f"  SKIP {fund_code} (already fetched)")
            manifest_rows.append(existing_manifest[fund_code])
            continue

        try:
            print(f"  FETCH {fund_code} {info['name']}...")
            provider_name = fetch_provider_name(fund_code)
            if provider_name != info["name"]:
                raise ValueError(
                    f"FUND_IDENTITY_MISMATCH: expected={info['name']!r}, "
                    f"provider={provider_name!r}"
                )
            df = fetch_fund_nav(fund_code)
            if df.empty or len(df) == 0:
                raise ValueError("Empty data returned by akshare")
            csv_path = DATA_DIR / f"{fund_code}.csv"
            df[
                [
                    "date", "fund_code", "nav", "cumulative_nav",
                    "daily_growth_pct", "cumulative_nav_imputed",
                ]
            ].to_csv(csv_path, index=False, encoding="utf-8")

            manifest_rows.append({
                "fund_code": fund_code,
                "fund_name": info["name"],
                "status": "success",
                "rows": len(df),
                "start_date": df["date"].min().strftime("%Y-%m-%d"),
                "end_date": df["date"].max().strftime("%Y-%m-%d"),
                "source": "akshare_fund_open_fund_info_em",
                "error": "",
            })
            print(f"    OK: {len(df)} rows, {df['date'].min().date()} ~ {df['date'].max().date()}")
        except Exception as e:
            print(f"    FAIL: {e}")
            errors.append(fund_code)
            manifest_rows.append({
                "fund_code": fund_code,
                "fund_name": info["name"],
                "status": "error",
                "rows": 0,
                "start_date": "",
                "end_date": "",
                "source": "akshare_fund_open_fund_info_em",
                "error": str(e)[:200],
            })

    with open(MANIFEST_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fund_code", "fund_name", "status", "rows", "start_date", "end_date", "source", "error"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return len(errors), errors

def build_sqlite_db():
    """Build otf_fund_catalog and otf_fund_nav tables in SQLite."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    temp_db = OTF_DB.with_suffix(".sqlite.tmp")
    if temp_db.exists():
        temp_db.unlink()
    conn = sqlite3.connect(str(temp_db))
    cur = conn.cursor()

    cur.execute("""CREATE TABLE IF NOT EXISTS otf_fund_catalog (
        fund_code TEXT PRIMARY KEY,
        share_class TEXT,
        fund_name TEXT,
        asset_class TEXT,
        benchmark TEXT,
        inception_date TEXT,
        termination_date TEXT,
        source TEXT,
        fetched_at TEXT
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS otf_fund_nav (
        fund_code TEXT,
        nav_date TEXT,
        unit_nav REAL,
        cumulative_nav REAL,
        daily_growth_pct REAL,
        cumulative_nav_imputed INTEGER,
        distribution_per_share REAL,
        share_adjustment_factor REAL,
        total_return_factor REAL,
        PRIMARY KEY (fund_code, nav_date),
        FOREIGN KEY (fund_code) REFERENCES otf_fund_catalog(fund_code)
    )""")

    # Load manifest
    with open(MANIFEST_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "success":
                continue
            fund_code = row["fund_code"]
            info = TARGET_FUNDS.get(fund_code, {})

            cur.execute("INSERT OR REPLACE INTO otf_fund_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                fund_code,
                info.get("share_class", "A"),
                row["fund_name"],
                info.get("asset_class", ""),
                info.get("benchmark", ""),
                row["start_date"],
                "",
                row["source"],
                datetime.now().isoformat(),
            ))

            # Load NAV CSV
            csv_path = DATA_DIR / f"{fund_code}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                required = {
                    "date", "nav", "cumulative_nav", "daily_growth_pct"
                }
                missing = required - set(df.columns)
                if missing:
                    raise RuntimeError(
                        f"{fund_code}: NAV CSV missing columns {sorted(missing)}"
                    )
                df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
                df["cumulative_nav"] = pd.to_numeric(
                    df["cumulative_nav"], errors="coerce"
                )
                df["daily_growth_pct"] = pd.to_numeric(
                    df["daily_growth_pct"], errors="coerce"
                )
                source_return = df["daily_growth_pct"] / 100.0
                # Infer the cash-equivalent distribution/share adjustment that
                # reconciles unit NAV with the source's published total daily
                # growth.  Cumulative-minus-unit cannot be differenced because
                # share conversions make that gap move every day.
                implied_distribution = (
                    df["nav"].shift(1) * (1.0 + source_return) - df["nav"]
                )
                df["distribution_per_share"] = (
                    implied_distribution.clip(lower=0.0).fillna(0.0)
                )
                df["share_adjustment_factor"] = (
                    df["nav"].shift(1) * (1.0 + source_return) / df["nav"]
                ).fillna(1.0)
                if (df["share_adjustment_factor"] <= 0).any():
                    raise RuntimeError(f"{fund_code}: non-positive share adjustment")
                fallback_return = (
                    (df["nav"] + df["distribution_per_share"])
                    / df["nav"].shift(1)
                    - 1.0
                )
                total_return = source_return.where(
                    source_return.notna(), fallback_return
                ).fillna(0.0)
                if (total_return <= -1.0).any():
                    raise RuntimeError(f"{fund_code}: invalid total return <= -100%")
                df["total_return_factor"] = (1.0 + total_return).cumprod()
                for _, r in df.iterrows():
                    cur.execute("INSERT OR REPLACE INTO otf_fund_nav VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                        fund_code,
                        str(r["date"])[:10],
                        float(r["nav"]),
                        float(r["cumulative_nav"]),
                        None if pd.isna(r["daily_growth_pct"]) else float(r["daily_growth_pct"]),
                        int(bool(r.get("cumulative_nav_imputed", False))),
                        float(r["distribution_per_share"]),
                        float(r["share_adjustment_factor"]),
                        float(r["total_return_factor"]),
                    ))

    conn.commit()

    # Stats
    cur.execute("SELECT COUNT(*) FROM otf_fund_catalog")
    catalog_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM otf_fund_nav")
    nav_count = cur.fetchone()[0]
    cur.execute("SELECT DISTINCT asset_class FROM otf_fund_catalog ORDER BY asset_class")
    classes = [r[0] for r in cur.fetchall()]

    conn.close()
    os.replace(temp_db, OTF_DB)
    return catalog_count, nav_count, classes

def audit_nav_quality():
    """Audit NAV identity, completeness and total-return inputs."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    quality_rows = []

    for fund_code in TARGET_FUNDS.keys():
        csv_path = DATA_DIR / f"{fund_code}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        failures = []
        warnings = []
        # Duplicate dates
        dupes = df["date"].duplicated().sum()
        if dupes > 0:
            failures.append(f"duplicate_dates:{dupes}")

        # Abnormal jumps (>10% single day)
        df["daily_return"] = pd.to_numeric(
            df.get("daily_growth_pct"), errors="coerce"
        ) / 100.0
        abnormal = (df["daily_return"].abs() > 0.10).sum()
        if abnormal > 0:
            warnings.append(f"source_growth_over_10pct:{abnormal}")

        # Missing values
        nulls = df["nav"].isna().sum()
        if nulls > 0:
            failures.append(f"unit_nav_nulls:{nulls}")

        cumulative_nulls = df.get("cumulative_nav", pd.Series(dtype=float)).isna().sum()
        if cumulative_nulls > 0:
            failures.append(f"cumulative_nav_nulls:{cumulative_nulls}")
        imputed = int(
            pd.Series(df.get("cumulative_nav_imputed", False)).astype(bool).sum()
        )
        if imputed > 0:
            warnings.append(f"cumulative_nav_imputed:{imputed}")

        date_diffs = df["date"].diff().dt.days.dropna()
        gaps_over_30 = int((date_diffs > 30).sum())
        max_gap_days = int(date_diffs.max()) if not date_diffs.empty else 0
        backtest_mask = df["date"] >= pd.Timestamp("2018-01-01")
        backtest_diffs = df.loc[backtest_mask, "date"].diff().dt.days.dropna()
        gaps_over_30_backtest = int((backtest_diffs > 30).sum())
        if gaps_over_30_backtest > 0:
            failures.append(
                f"backtest_publication_gaps_over_30d:{gaps_over_30_backtest}"
            )
        elif gaps_over_30 > 0:
            warnings.append(f"pre_backtest_publication_gaps_over_30d:{gaps_over_30}")

        non_positive = int((pd.to_numeric(df["nav"], errors="coerce") <= 0).sum())
        if non_positive > 0:
            failures.append(f"non_positive_nav:{non_positive}")

        asset = TARGET_FUNDS[fund_code]["asset_class"]
        status = "FAIL" if failures else "WARN" if warnings else "PASS"

        quality_rows.append({
            "fund_code": fund_code,
            "fund_name": TARGET_FUNDS[fund_code]["name"],
            "asset_class": asset,
            "total_rows": len(df),
            "start_date": str(df["date"].min().date()),
            "end_date": str(df["date"].max().date()),
            "duplicate_dates": dupes,
            "abnormal_jumps_10pct": abnormal,
            "null_values": nulls,
            "cumulative_nav_imputed": imputed,
            "max_publication_gap_days": max_gap_days,
            "publication_gaps_over_30d": gaps_over_30,
            "backtest_publication_gaps_over_30d": gaps_over_30_backtest,
            "status": status,
            "failures": ";".join(failures) if failures else "NONE",
            "warnings": ";".join(warnings) if warnings else "NONE",
        })

    quality_df = pd.DataFrame(quality_rows)
    out_path = REPORT_DIR / "otf_nav_quality.csv"
    quality_df.to_csv(out_path, index=False, encoding="utf-8")
    summary = {
        "generated_at": datetime.now().isoformat(),
        "fund_count": len(quality_rows),
        "pass_count": int((quality_df["status"] == "PASS").sum()),
        "warn_count": int((quality_df["status"] == "WARN").sum()),
        "fail_count": int((quality_df["status"] == "FAIL").sum()),
        "data_gate_passed": bool((quality_df["status"] != "FAIL").all()),
        "price_mode": "published_daily_growth_total_return_reinvested",
        "source_independent": False,
    }
    with open(
        REPORT_DIR / "otf_data_gate.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return len(quality_rows), summary["fail_count"]

def update_config():
    """Add OTF fields to unified_experiment.json."""
    config_path = Path("config/unified_experiment.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "otf_data" not in config:
        config["otf_data"] = {
            "canonical_db": "data/processed/otf.sqlite",
            "nav_dir": "data/raw/otf_nav",
            "manifest": "data/raw/otf_nav/otf_nav_manifest.csv",
            "price_mode": "unit_nav",
            "confirmation_rule": "T+1",
            "qdii_confirmation_rule": "T+3",
            "fee_rate_subscription": 0.001,
            "fee_rate_redemption": 0.0015,
        }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="Refetch successful cached funds"
    )
    args = parser.parse_args()
    print("=" * 60)
    print("P1-1: 扩充场外基金数据池")
    print("=" * 60)

    print("\n[1/4] Fetching NAV data...")
    err_count, err_list = fetch_all_funds(force=args.force)
    if err_count > 0:
        print(f"  Errors: {err_list}")

    print("\n[2/4] Building SQLite DB...")
    catalog_n, nav_n, classes = build_sqlite_db()
    print(f"  Catalog: {catalog_n} funds, NAV rows: {nav_n}")
    print(f"  Asset classes: {classes}")

    print("\n[3/4] Auditing NAV quality...")
    total_funds, issue_count = audit_nav_quality()
    print(f"  Audited: {total_funds} funds, {issue_count} failed data gate")

    print("\n[4/4] Updating unified_experiment.json...")
    update_config()
    print("  Config updated with OTF fields")

    print("\n" + "=" * 60)
    print(f"COMPLETE: {catalog_n} funds, {nav_n} NAV rows, {len(classes)} asset classes")
    print("=" * 60)

if __name__ == "__main__":
    main()
