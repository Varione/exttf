from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS etf_catalog (symbol TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS etf_daily (
    symbol TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL, low REAL,
    close REAL, volume REAL, amount REAL, PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS etf_quality (
    symbol TEXT PRIMARY KEY, name TEXT NOT NULL, asset_class TEXT NOT NULL,
    rows_raw INTEGER NOT NULL, rows_valid INTEGER NOT NULL,
    duplicate_dates INTEGER NOT NULL, null_close INTEGER NOT NULL,
    non_positive_close INTEGER NOT NULL, first_date TEXT, last_date TEXT,
    span_days INTEGER, median_amount_60d REAL, zero_volume_ratio_60d REAL
);
CREATE INDEX IF NOT EXISTS idx_etf_daily_date ON etf_daily(date);
CREATE INDEX IF NOT EXISTS idx_etf_daily_symbol_date ON etf_daily(symbol, date);
"""


def classify_etf(name: str) -> str:
    text = str(name).lower()
    if any(k in text for k in ("货币", "快线", "添益", "零钱")):
        return "cash"
    if any(k in text for k in ("债", "国债", "信用", "短融", "中短")):
        return "bond"
    if any(k in text for k in ("黄金", "有色", "商品", "原油", "油气")):
        return "commodity"
    if any(k in text for k in ("恒生", "港股", "纳指", "纳斯达克", "标普", "美国", "全球", "qdii")):
        return "overseas"
    if any(k in text for k in ("红利", "股息", "价值")):
        return "equity_dividend"
    if any(k in text for k in ("沪深", "中证", "上证", "深证", "创业板", "科创", "50etf", "300", "500", "1000")):
        return "equity_broad"
    return "equity_thematic"


def _read_history(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_csv(path)
    raw_rows = len(raw)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        raw[column] = pd.to_numeric(raw.get(column), errors="coerce")
    duplicate_dates = int(raw["date"].duplicated().sum())
    null_close = int(raw["close"].isna().sum())
    non_positive_close = int((raw["close"].fillna(0) <= 0).sum())
    clean = raw[["date", "open", "high", "low", "close", "volume", "amount"]].dropna(
        subset=["date", "close"]
    ).drop_duplicates("date", keep="last").sort_values("date")
    last_60 = clean.tail(60)
    amount = last_60["amount"].dropna()
    stats = {
        "rows_raw": raw_rows, "rows_valid": len(clean),
        "duplicate_dates": duplicate_dates, "null_close": null_close,
        "non_positive_close": non_positive_close,
        "first_date": clean["date"].min().strftime("%Y-%m-%d") if len(clean) else None,
        "last_date": clean["date"].max().strftime("%Y-%m-%d") if len(clean) else None,
        "span_days": int((clean["date"].max() - clean["date"].min()).days) if len(clean) else 0,
        "median_amount_60d": float(amount.median()) if len(amount) else 0.0,
        "zero_volume_ratio_60d": float((last_60["volume"].fillna(0) <= 0).mean()) if len(last_60) else 1.0,
    }
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    return clean, stats


def build_database(
    raw_dir: str | Path = "data/raw/all_etf",
    database: str | Path = "data/processed/etf.sqlite",
    quality_csv: str | Path = "reports/all_etf_quality.csv",
    summary_json: str | Path = "reports/all_etf_quality_summary.json",
) -> dict[str, object]:
    raw_dir, database, quality_csv, summary_json = map(Path, (raw_dir, database, quality_csv, summary_json))
    database.parent.mkdir(parents=True, exist_ok=True)
    quality_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(raw_dir / "etf_catalog.csv", dtype=str).fillna("")
    files = sorted((raw_dir / "history").glob("*.csv"))
    conn = sqlite3.connect(database)
    # SQLite 是派生缓存；重复运行时重建行情表，避免重复主键或旧文件残留。
    conn.executescript("DROP TABLE IF EXISTS etf_daily; DROP TABLE IF EXISTS etf_quality;")
    conn.executescript(SCHEMA)
    catalog[["symbol", "name"]].to_sql("etf_catalog", conn, if_exists="replace", index=False)
    quality_rows: list[dict[str, object]] = []
    for position, path in enumerate(files, start=1):
        symbol = path.stem
        match = catalog.loc[catalog["symbol"] == symbol, "name"]
        name = str(match.iloc[0]) if len(match) else symbol
        clean, stats = _read_history(path)
        if len(clean):
            clean.insert(0, "symbol", symbol)
            clean.to_sql("etf_daily", conn, if_exists="append", index=False, chunksize=5000)
        quality_rows.append({"symbol": symbol, "name": name, "asset_class": classify_etf(name), **stats})
        if position % 100 == 0 or position == len(files):
            print(f"processed {position}/{len(files)}", flush=True)
    quality = pd.DataFrame(quality_rows).sort_values("symbol")
    quality.to_sql("etf_quality", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    summary = {
        "catalog_count": int(len(catalog)), "history_file_count": int(len(files)),
        "valid_etf_count": int((quality["rows_valid"] > 0).sum()),
        "total_daily_rows": int(quality["rows_valid"].sum()),
        "asset_class_counts": quality["asset_class"].value_counts().to_dict(),
        "median_history_rows": float(quality["rows_valid"].median()),
        "median_amount_60d_quantiles": quality["median_amount_60d"].quantile([0.25, 0.5, 0.75]).to_dict(),
    }
    quality.to_csv(quality_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_universe_snapshot(
    quality_csv: str | Path = "reports/all_etf_quality.csv",
    output_csv: str | Path = "reports/strategy_universe_snapshot.csv",
    min_history_rows: int = 252,
    max_stale_days: int = 30,
    min_median_amount_60d: float = 10_000_000,
    exclude_asset_classes: tuple[str, ...] = ("cash",),
) -> dict[str, Any]:
    """按研究参数生成一份可审计的 ETF 可交易池快照。"""
    quality = pd.read_csv(quality_csv)
    quality["last_date"] = pd.to_datetime(quality["last_date"], errors="coerce")
    as_of = quality["last_date"].max()
    quality["stale_days"] = (as_of - quality["last_date"]).dt.days
    quality["eligible"] = True
    reasons: list[list[str]] = [[] for _ in range(len(quality))]
    checks = [
        (quality["rows_valid"] < min_history_rows, f"history<{min_history_rows}"),
        (quality["stale_days"] > max_stale_days, f"stale>{max_stale_days}d"),
        (quality["median_amount_60d"] < min_median_amount_60d, f"amount<{min_median_amount_60d:g}"),
        (quality["asset_class"].isin(exclude_asset_classes), "excluded_asset_class"),
    ]
    for mask, reason in checks:
        quality.loc[mask, "eligible"] = False
        for index in quality.index[mask]:
            reasons[quality.index.get_loc(index)].append(reason)
    quality["exclusion_reason"] = [";".join(x) for x in reasons]
    quality["last_date"] = quality["last_date"].dt.strftime("%Y-%m-%d")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    quality.sort_values(["eligible", "asset_class", "median_amount_60d"], ascending=[False, True, False]).to_csv(
        output_csv, index=False, encoding="utf-8-sig"
    )
    return {
        "as_of": str(as_of.date()),
        "total": int(len(quality)),
        "eligible": int(quality["eligible"].sum()),
        "excluded": int((~quality["eligible"]).sum()),
        "eligible_by_asset_class": quality.loc[quality["eligible"]].groupby("asset_class").size().to_dict(),
    }


def build_research_universe(
    quality_csv: str | Path = "reports/all_etf_quality.csv",
    output_csv: str | Path = "config/research_universe_top200.csv",
    per_category: int = 50,
    cash_count: int = 5,
    min_history_rows: int = 252,
    max_stale_days: int = 30,
    min_median_amount_60d: float = 10_000_000,
) -> dict[str, object]:
    """生成小型、可快速迭代的研究池：各风险类别按流动性取前 N，另保留少量现金资产。"""
    quality = pd.read_csv(quality_csv)
    quality["last_date"] = pd.to_datetime(quality["last_date"], errors="coerce")
    as_of = quality["last_date"].max()
    quality["stale_days"] = (as_of - quality["last_date"]).dt.days
    base = quality[
        (quality["rows_valid"] >= min_history_rows)
        & (quality["stale_days"] <= max_stale_days)
        & (quality["median_amount_60d"] >= min_median_amount_60d)
    ].copy()
    risk = base[base["asset_class"] != "cash"].sort_values("median_amount_60d", ascending=False)
    selected = risk.groupby("asset_class", group_keys=False).head(per_category)
    cash = base[base["asset_class"] == "cash"].sort_values("median_amount_60d", ascending=False).head(cash_count)
    selected = pd.concat([selected, cash], ignore_index=True).sort_values(["asset_class", "median_amount_60d"], ascending=[True, False])
    columns = ["symbol", "name", "asset_class", "rows_valid", "first_date", "last_date", "median_amount_60d"]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected[columns].to_csv(output_csv, index=False, encoding="utf-8-sig")
    return {"as_of": str(as_of.date()), "selected": int(len(selected)), "by_asset_class": selected["asset_class"].value_counts().to_dict()}
