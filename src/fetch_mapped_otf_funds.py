"""Fetch and build the expanded ETF-feeder NAV database."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from fetch_otf_funds import fetch_fund_nav, fetch_provider_name


MAPPING_PATH = Path("config/etf_otf_mapping.csv")
NAV_DIR = Path("data/raw/otf_mapped_nav")
MANIFEST_PATH = NAV_DIR / "manifest.csv"
DB_PATH = Path("data/processed/otf_mapped.sqlite")
QUALITY_PATH = Path("reports/data_validation/otf_mapped_nav_quality.csv")
GATE_PATH = Path("reports/data_validation/otf_mapped_data_gate.json")


def load_mapping(confidence: str = "HIGH") -> pd.DataFrame:
    mapping = pd.read_csv(
        MAPPING_PATH,
        encoding="utf-8-sig",
        dtype={"fund_code": str, "etf_symbol": str},
    )
    allowed = {item.strip().upper() for item in confidence.split(",")}
    mapping = mapping.loc[
        mapping["mapping_confidence"].str.upper().isin(allowed)
    ].copy()
    mapping["fund_code"] = mapping["fund_code"].str.zfill(6)
    mapping["etf_symbol"] = mapping["etf_symbol"].str.zfill(6)
    return mapping.drop_duplicates("fund_code").reset_index(drop=True)


def _fetch_one(row: dict, force: bool, retries: int = 3) -> dict:
    code = row["fund_code"]
    path = NAV_DIR / f"{code}.csv"
    if path.exists() and not force:
        cached = pd.read_csv(path, parse_dates=["date"])
        if not cached.empty:
            return {
                **row,
                "status": "success",
                "rows": len(cached),
                "start_date": str(cached["date"].min().date()),
                "end_date": str(cached["date"].max().date()),
                "source": "cached_akshare_eastmoney",
                "error": "",
            }

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            provider_name = fetch_provider_name(code)
            if provider_name != row["fund_name"]:
                raise ValueError(
                    "FUND_IDENTITY_MISMATCH: "
                    f"catalog={row['fund_name']!r}, provider={provider_name!r}"
                )
            nav = fetch_fund_nav(code)
            if nav.empty:
                raise ValueError("EMPTY_NAV")
            nav[
                [
                    "date", "fund_code", "nav", "cumulative_nav",
                    "daily_growth_pct", "cumulative_nav_imputed",
                ]
            ].to_csv(path, index=False, encoding="utf-8")
            return {
                **row,
                "status": "success",
                "rows": len(nav),
                "start_date": str(nav["date"].min().date()),
                "end_date": str(nav["date"].max().date()),
                "source": "akshare_eastmoney",
                "error": "",
            }
        except Exception as exc:  # network/provider errors are recorded per fund
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(attempt * 1.5)
    return {
        **row,
        "status": "error",
        "rows": 0,
        "start_date": "",
        "end_date": "",
        "source": "akshare_eastmoney",
        "error": last_error[:500],
    }


def fetch_all(
    mapping: pd.DataFrame,
    *,
    force: bool = False,
    workers: int = 6,
) -> pd.DataFrame:
    NAV_DIR.mkdir(parents=True, exist_ok=True)
    rows = mapping.to_dict("records")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, row, force): row for row in rows}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(rows)}] {result['fund_code']} "
                f"{result['status']} rows={result['rows']}"
            )
            if index % 20 == 0:
                pd.DataFrame(results).sort_values("fund_code").to_csv(
                    MANIFEST_PATH, index=False, encoding="utf-8-sig"
                )
    manifest = pd.DataFrame(results).sort_values("fund_code")
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    return manifest


def _prepare_nav(code: str) -> pd.DataFrame:
    frame = pd.read_csv(NAV_DIR / f"{code}.csv")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("nav", "cumulative_nav", "daily_growth_pct"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "nav", "daily_growth_pct"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    source_return = frame["daily_growth_pct"] / 100.0
    implied_distribution = (
        frame["nav"].shift(1) * (1.0 + source_return) - frame["nav"]
    )
    frame["distribution_per_share"] = implied_distribution.clip(lower=0).fillna(0)
    frame["share_adjustment_factor"] = (
        frame["nav"].shift(1) * (1.0 + source_return) / frame["nav"]
    ).fillna(1.0)
    frame["total_return_factor"] = (1.0 + source_return.fillna(0.0)).cumprod()
    return frame


def audit(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for item in manifest.loc[manifest["status"].eq("success")].itertuples(index=False):
        frame = _prepare_nav(item.fund_code)
        failures: list[str] = []
        warnings: list[str] = []
        duplicates = int(frame["date"].duplicated().sum())
        non_positive = int((frame["nav"] <= 0).sum())
        invalid_adjustment = int((frame["share_adjustment_factor"] <= 0).sum())
        abnormal = int((frame["daily_growth_pct"].abs() > 10).sum())
        gaps = frame.loc[frame["date"] >= "2018-01-01", "date"].diff().dt.days
        gaps_over_30 = int((gaps > 30).sum())
        if duplicates:
            failures.append(f"duplicates:{duplicates}")
        if non_positive:
            failures.append(f"non_positive_nav:{non_positive}")
        if invalid_adjustment:
            failures.append(f"invalid_share_adjustment:{invalid_adjustment}")
        if gaps_over_30:
            failures.append(f"publication_gaps_over_30d:{gaps_over_30}")
        if abnormal:
            warnings.append(f"source_growth_over_10pct:{abnormal}")
        status = "FAIL" if failures else "WARN" if warnings else "PASS"
        rows.append(
            {
                "fund_code": item.fund_code,
                "fund_name": item.fund_name,
                "etf_symbol": item.etf_symbol,
                "rows": len(frame),
                "start_date": frame["date"].min(),
                "end_date": frame["date"].max(),
                "status": status,
                "failures": ";".join(failures) or "NONE",
                "warnings": ";".join(warnings) or "NONE",
            }
        )
    quality = pd.DataFrame(rows)
    QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(QUALITY_PATH, index=False, encoding="utf-8-sig")
    fetch_failures = int(manifest["status"].ne("success").sum())
    success_ratio = float(manifest["status"].eq("success").mean())
    gate = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "requested_funds": int(len(manifest)),
        "fetch_success": int(manifest["status"].eq("success").sum()),
        "fetch_failures": fetch_failures,
        "fetch_success_ratio": success_ratio,
        "pass_count": int(quality["status"].eq("PASS").sum()),
        "warn_count": int(quality["status"].eq("WARN").sum()),
        "fail_count": int(quality["status"].eq("FAIL").sum()),
        "data_gate_passed": bool(
            success_ratio >= 0.98 and quality["status"].ne("FAIL").all()
        ),
        "source_independent": False,
        "mapping_confidence": "HIGH",
    }
    GATE_PATH.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    return quality, gate


def build_db(manifest: pd.DataFrame, quality: pd.DataFrame) -> tuple[int, int]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = DB_PATH.with_suffix(".sqlite.tmp")
    if temp.exists():
        temp.unlink()
    good = set(quality.loc[quality["status"].ne("FAIL"), "fund_code"].astype(str))
    successful = manifest.loc[
        manifest["status"].eq("success") & manifest["fund_code"].astype(str).isin(good)
    ]
    with sqlite3.connect(temp) as conn:
        conn.execute(
            """CREATE TABLE otf_fund_catalog (
                fund_code TEXT PRIMARY KEY, fund_name TEXT, fund_family TEXT,
                share_class TEXT, fund_type TEXT, etf_symbol TEXT, etf_name TEXT,
                underlying_name TEXT, asset_class TEXT, mapping_score REAL,
                mapping_confidence TEXT, mapping_method TEXT, inception_date TEXT,
                termination_date TEXT, source TEXT, fetched_at TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE otf_fund_nav (
                fund_code TEXT, nav_date TEXT, unit_nav REAL,
                cumulative_nav REAL, daily_growth_pct REAL,
                cumulative_nav_imputed INTEGER, distribution_per_share REAL,
                share_adjustment_factor REAL, total_return_factor REAL,
                PRIMARY KEY (fund_code, nav_date)
            )"""
        )
        for item in successful.itertuples(index=False):
            nav = _prepare_nav(item.fund_code)
            conn.execute(
                "INSERT INTO otf_fund_catalog VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item.fund_code, item.fund_name, item.fund_family,
                    item.share_class, item.fund_type, item.etf_symbol, item.etf_name,
                    item.underlying_name, item.asset_class, float(item.mapping_score),
                    item.mapping_confidence, item.mapping_method,
                    str(nav["date"].min().date()), "", item.source,
                    datetime.now().astimezone().isoformat(),
                ),
            )
            records = [
                (
                    item.fund_code, str(row.date.date()), float(row.nav),
                    float(row.cumulative_nav), float(row.daily_growth_pct),
                    int(bool(row.cumulative_nav_imputed)),
                    float(row.distribution_per_share),
                    float(row.share_adjustment_factor),
                    float(row.total_return_factor),
                )
                for row in nav.itertuples(index=False)
            ]
            conn.executemany(
                "INSERT INTO otf_fund_nav VALUES (?,?,?,?,?,?,?,?,?)", records
            )
        conn.execute("CREATE INDEX idx_otf_nav_date ON otf_fund_nav(nav_date)")
        fund_count = conn.execute("SELECT COUNT(*) FROM otf_fund_catalog").fetchone()[0]
        nav_count = conn.execute("SELECT COUNT(*) FROM otf_fund_nav").fetchone()[0]
    conn.close()
    os.replace(temp, DB_PATH)
    return int(fund_count), int(nav_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", default="HIGH")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-funds", type=int)
    args = parser.parse_args()
    mapping = load_mapping(args.confidence)
    if args.max_funds:
        mapping = mapping.head(args.max_funds)
    manifest = fetch_all(mapping, force=args.force, workers=args.workers)
    quality, gate = audit(manifest)
    fund_count, nav_count = build_db(manifest, quality)
    print(json.dumps({**gate, "db_funds": fund_count, "db_nav_rows": nav_count}, indent=2))


if __name__ == "__main__":
    main()
