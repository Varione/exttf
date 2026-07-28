"""Fetch and standardize non-ETF OTC defensive assets.

The provider's money-fund wrapper currently assumes an obsolete response
shape.  This module reads the public endpoint directly and converts
per-10,000-unit income into a reinvested wealth index.  Calendar-day income is
compounded before the wealth index is sampled on the common OTC valuation
calendar, so weekend accrual is not lost and weekends do not become trading
days.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from fetch_otf_funds import fetch_fund_nav


ASSET_CONFIG = Path("config/extended_otf_assets.csv")
FULL_CATALOG_DB = Path("data/processed/otf_full_catalog.sqlite")
REFERENCE_DB = Path("data/processed/otf_mapped.sqlite")
RAW_DIR = Path("data/raw/otf_extended_assets")
OUTPUT_DB = Path("data/processed/otf_extended_assets.sqlite")
SUMMARY = Path("reports/data_validation/otf_extended_assets_summary.json")
MONEY_URL = "https://api.fund.eastmoney.com/f10/lsjz"


def money_rows_to_nav(
    rows: pd.DataFrame, reference_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Convert calendar-day per-10,000 income to trading-calendar wealth."""
    required = {"date", "per_10000_income"}
    if missing := required - set(rows.columns):
        raise ValueError(f"MONEY_ROWS_MISSING_COLUMNS:{sorted(missing)}")
    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["per_10000_income"] = pd.to_numeric(
        frame["per_10000_income"], errors="coerce"
    )
    frame = (
        frame.dropna(subset=["date", "per_10000_income"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    daily_return = frame["per_10000_income"] / 10000.0
    if frame.empty or (daily_return <= -1.0).any():
        raise ValueError("INVALID_MONEY_FUND_INCOME")
    frame["wealth"] = (1.0 + daily_return).cumprod()

    calendar = pd.DataFrame(
        {"date": pd.DatetimeIndex(reference_dates).sort_values().unique()}
    )
    calendar = calendar.loc[
        calendar["date"].between(frame["date"].min(), frame["date"].max())
    ]
    sampled = pd.merge_asof(calendar, frame, on="date", direction="backward")
    sampled = sampled.dropna(subset=["wealth"]).copy()
    sampled["daily_growth_pct"] = sampled["wealth"].pct_change().fillna(0.0) * 100
    sampled["unit_nav"] = sampled["wealth"]
    sampled["cumulative_nav"] = sampled["wealth"]
    sampled["cumulative_nav_imputed"] = 0
    sampled["distribution_per_share"] = 0.0
    sampled["share_adjustment_factor"] = 1.0
    sampled["total_return_factor"] = sampled["wealth"] / sampled["wealth"].iloc[0]
    return sampled


def _get_json(session: requests.Session, params: dict, retries: int = 4) -> dict:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(
                MONEY_URL,
                params=params,
                headers={"Referer": "https://fundf10.eastmoney.com/"},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            if int(payload.get("ErrCode", -1)) != 0:
                raise RuntimeError(payload.get("ErrMsg", "provider error"))
            return payload
        except Exception as exc:  # provider retry boundary
            error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"MONEY_PROVIDER_UNAVAILABLE:{error}")


def fetch_money_history(fund_code: str, workers: int = 6) -> pd.DataFrame:
    """Fetch complete money-fund history with bounded parallel pagination."""
    with requests.Session() as session:
        first = _get_json(
            session,
            {"fundCode": fund_code, "pageIndex": 1, "pageSize": 20},
        )
    total = int(first.get("TotalCount", 0))
    page_size = int(first.get("PageSize", 20))
    page_count = max(1, math.ceil(total / page_size))
    payloads = {1: first}

    def fetch_page(page: int) -> tuple[int, dict]:
        with requests.Session() as worker_session:
            return page, _get_json(
                worker_session,
                {"fundCode": fund_code, "pageIndex": page, "pageSize": 20},
            )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(fetch_page, page) for page in range(2, page_count + 1)]
        for future in as_completed(futures):
            page, payload = future.result()
            payloads[page] = payload

    records: list[dict] = []
    for page in sorted(payloads):
        for row in (payloads[page].get("Data") or {}).get("LSJZList", []):
            records.append(
                {
                    "date": row.get("FSRQ"),
                    "per_10000_income": row.get("DWJZ"),
                    "seven_day_annualized_pct": row.get("LJJZ"),
                    "purchase_status": row.get("SGZT"),
                    "redemption_status": row.get("SHZT"),
                }
            )
    frame = pd.DataFrame(records)
    if len(frame) != total:
        raise RuntimeError(f"MONEY_HISTORY_INCOMPLETE:{fund_code}:{len(frame)}/{total}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["per_10000_income"] = pd.to_numeric(frame["per_10000_income"], errors="coerce")
    frame["seven_day_annualized_pct"] = pd.to_numeric(
        frame["seven_day_annualized_pct"], errors="coerce"
    )
    return frame.dropna(subset=["date", "per_10000_income"]).sort_values("date")


def _regular_nav_to_standard(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.rename(columns={"date": "nav_date", "nav": "unit_nav"}).copy()
    source_return = pd.to_numeric(result["daily_growth_pct"], errors="coerce") / 100.0
    previous = result["unit_nav"].shift(1)
    implied_distribution = previous * (1.0 + source_return) - result["unit_nav"]
    result["distribution_per_share"] = implied_distribution.clip(lower=0.0).fillna(0.0)
    result["share_adjustment_factor"] = (
        previous * (1.0 + source_return) / result["unit_nav"]
    ).fillna(1.0)
    fallback = (
        (result["unit_nav"] + result["distribution_per_share"]) / previous - 1.0
    )
    total_return = source_return.where(source_return.notna(), fallback).fillna(0.0)
    result["daily_growth_pct"] = total_return * 100.0
    result["total_return_factor"] = (1.0 + total_return).cumprod()
    return result


def build_extended_assets(
    config_path: Path = ASSET_CONFIG,
    catalog_db: Path = FULL_CATALOG_DB,
    reference_db: Path = REFERENCE_DB,
    output_db: Path = OUTPUT_DB,
    raw_dir: Path = RAW_DIR,
    summary_path: Path = SUMMARY,
    force: bool = False,
) -> dict:
    config = pd.read_csv(config_path, dtype={"fund_code": str})
    config["fund_code"] = config["fund_code"].str.zfill(6)
    if config["fund_code"].duplicated().any():
        raise ValueError("DUPLICATE_EXTENDED_ASSET_CODES")
    with sqlite3.connect(catalog_db) as connection:
        catalog = pd.read_sql_query("SELECT * FROM otf_full_catalog", connection)
    catalog["fund_code"] = catalog["fund_code"].astype(str).str.zfill(6)
    merged = config.merge(
        catalog,
        on=["fund_code", "fund_name"],
        how="left",
        validate="one_to_one",
        suffixes=("_configured", "_catalog"),
    )
    if merged["fund_type"].isna().any():
        raise ValueError(
            "EXTENDED_ASSET_IDENTITY_MISMATCH:"
            + ",".join(merged.loc[merged["fund_type"].isna(), "fund_code"])
        )
    with sqlite3.connect(reference_db) as connection:
        reference_dates = pd.DatetimeIndex(
            pd.read_sql_query(
                "SELECT DISTINCT nav_date FROM otf_fund_nav ORDER BY nav_date", connection
            )["nav_date"]
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    nav_frames: list[pd.DataFrame] = []
    money_raw_frames: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    for row in merged.itertuples(index=False):
        raw_path = raw_dir / f"{row.fund_code}.csv"
        if raw_path.exists() and not force:
            raw = pd.read_csv(raw_path)
        elif row.data_model_configured == "money_yield":
            raw = fetch_money_history(row.fund_code)
            raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
        else:
            raw = fetch_fund_nav(row.fund_code)
            raw.to_csv(raw_path, index=False, encoding="utf-8-sig")

        if row.data_model_configured == "money_yield":
            raw["fund_code"] = row.fund_code
            money_raw_frames.append(raw.copy())
            standard = money_rows_to_nav(raw, reference_dates).rename(columns={"date": "nav_date"})
        else:
            standard = _regular_nav_to_standard(raw)
        standard["fund_code"] = row.fund_code
        standard = standard[
            ["fund_code", "nav_date", "unit_nav", "cumulative_nav", "daily_growth_pct",
             "cumulative_nav_imputed", "distribution_per_share",
             "share_adjustment_factor", "total_return_factor"]
        ]
        nav_frames.append(standard)
        audit_rows.append(
            {
                "fund_code": row.fund_code,
                "asset_class": row.asset_class,
                "data_model": row.data_model_configured,
                "rows": len(standard),
                "first_date": str(pd.to_datetime(standard["nav_date"]).min().date()),
                "last_date": str(pd.to_datetime(standard["nav_date"]).max().date()),
                "approved_for_strategy": int(row.approved_for_strategy),
            }
        )

    nav = pd.concat(nav_frames, ignore_index=True)
    nav["nav_date"] = pd.to_datetime(nav["nav_date"]).dt.strftime("%Y-%m-%d")
    if nav.duplicated(["fund_code", "nav_date"]).any():
        raise RuntimeError("DUPLICATE_EXTENDED_NAV_KEYS")
    if nav[["unit_nav", "daily_growth_pct", "share_adjustment_factor", "total_return_factor"]].isna().any().any():
        raise RuntimeError("NULL_EXTENDED_NAV_FIELDS")

    output_db.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_db.with_suffix(output_db.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        output_catalog = pd.DataFrame(
            {
                "fund_code": merged["fund_code"],
                "fund_name": merged["fund_name"],
                "fund_family": merged["fund_family"],
                "share_class": merged["share_class"],
                "fund_type": merged["fund_type"],
                "etf_symbol": "",
                "etf_name": "",
                "underlying_name": merged["asset_class"],
                "asset_class": merged["asset_class"],
                "mapping_score": None,
                "mapping_confidence": "DIRECT",
                "mapping_method": "approved_direct_otf_extended_v1",
                "inception_date": nav.groupby("fund_code")["nav_date"].min().reindex(merged["fund_code"]).to_numpy(),
                "termination_date": "",
                "source": merged["data_model_configured"].map(
                    {"money_yield": "eastmoney_f10_lsjz_money", "unit_nav": "akshare_fund_open_fund_info_em"}
                ),
                "fetched_at": datetime.now().astimezone().isoformat(),
            }
        )
        output_catalog.to_sql("otf_fund_catalog", connection, index=False, if_exists="replace")
        nav.to_sql("otf_fund_nav", connection, index=False, if_exists="replace")
        if money_raw_frames:
            money_raw = pd.concat(money_raw_frames, ignore_index=True)
            money_raw["date"] = pd.to_datetime(money_raw["date"]).dt.strftime("%Y-%m-%d")
            money_raw.to_sql("otf_money_income", connection, index=False, if_exists="replace")
        pd.DataFrame(audit_rows).to_sql("extended_asset_audit", connection, index=False, if_exists="replace")
        connection.execute("CREATE UNIQUE INDEX idx_extended_nav ON otf_fund_nav(fund_code,nav_date)")
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, output_db)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "catalog_scope": "full_catalog_discovery_curated_systematic_execution",
        "fund_count": int(len(merged)),
        "approved_fund_count": int(merged["approved_for_strategy"].sum()),
        "nav_rows": int(len(nav)),
        "asset_class_counts": merged["asset_class"].value_counts().to_dict(),
        "pit_status": "PIT_PARTIAL",
        "historical_inactive_coverage": False,
        "source_independent": False,
        "selection_is_return_optimized": False,
        "audit": audit_rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_extended_assets(force=args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
