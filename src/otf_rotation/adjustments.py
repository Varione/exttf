from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time
import sqlite3

import pandas as pd


def fetch_sina_hfq(symbol: str) -> pd.DataFrame:
    """读取新浪 ETF 后复权因子接口，保留现金分红累计值和拆分字段。"""
    import requests

    symbol = str(symbol).zfill(6)
    market = "sz" if symbol.startswith(("15", "16")) else "sh"
    url = f"https://finance.sina.com.cn/realstock/company/{market}{symbol}/hfq.js"
    text = requests.get(url, timeout=20).text
    if "=" not in text:
        raise ValueError(f"未找到复权因子: {symbol}")
    payload_text = text.split("=", 1)[1].split("\n/*", 1)[0].strip().rstrip(";")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"复权因子格式异常: {symbol}") from exc
    data = pd.DataFrame(payload.get("data", []))
    if data.empty:
        return pd.DataFrame(columns=["date", "f", "s", "u"])
    data = data.rename(columns={"d": "date", "f": "f", "s": "s", "u": "u"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("f", "s", "u"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.loc[data["date"].notna() & (data["date"] > "1900-01-01"), ["date", "f", "s", "u"]].sort_values("date")


def _fetch_adjustment_job(symbol: str, name: str, output_dir: str, retries: int, sleep_seconds: float) -> dict[str, object]:
    path = Path(output_dir) / f"{symbol}.csv"
    error = ""
    for attempt in range(retries + 1):
        try:
            factors = fetch_sina_hfq(symbol)
            factors.to_csv(path, index=False, date_format="%Y-%m-%d")
            return {"symbol": symbol, "name": name, "rows": len(factors), "status": "ok", "error": ""}
        except Exception as exc:
            error = str(exc)
            if attempt < retries:
                time.sleep(sleep_seconds * (attempt + 1))
    return {"symbol": symbol, "name": name, "rows": 0, "status": "failed", "error": error}


def fetch_all_adjustments(
    catalog_csv: str | Path = "data/raw/all_etf/etf_catalog.csv",
    output_dir: str | Path = "data/raw/all_etf/adjustments",
    workers: int = 4,
    retries: int = 2,
    sleep_seconds: float = 0.2,
    resume: bool = True,
) -> dict[str, int]:
    catalog = pd.read_csv(catalog_csv, dtype=str).fillna("")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = catalog[~catalog["symbol"].map(lambda x: (output_dir / f"{str(x).zfill(6)}.csv").exists())] if resume else catalog
    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_fetch_adjustment_job, str(row.symbol).zfill(6), row.name, str(output_dir), retries, sleep_seconds): row.symbol
            for row in pending.itertuples(index=False)
        }
        for position, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(f"[{position}/{len(futures)}] {record['symbol']} {record['status']} rows={record['rows']}", flush=True)
    result = pd.DataFrame(records, columns=["symbol", "name", "rows", "status", "error"])
    if len(result):
        result = result.sort_values("symbol")
    result.to_csv(output_dir.parent / "adjustment_fetch_status.csv", index=False, encoding="utf-8-sig")
    result[result["status"] == "failed"].to_csv(output_dir.parent / "adjustment_failed.csv", index=False, encoding="utf-8-sig")
    return {"catalog": len(catalog), "pending": len(pending), "success": int((result["status"] == "ok").sum()), "failed": int((result["status"] == "failed").sum())}


def build_total_return_proxy(
    raw_dir: str | Path = "data/raw/all_etf",
    output_dir: str | Path = "data/processed/total_return",
    database: str | Path | None = "data/processed/etf.sqlite",
) -> dict[str, int]:
    """用现金分红和拆分因子构建可审计的 total-return proxy：TR = close * split_factor + cumulative_dividend。

    这是现金分红调整层，不等同于基金官方复权净值；拆分字段单独保留供审计。
    """
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir, adjustment_dir = raw_dir / "history", raw_dir / "adjustments"
    success, failed = 0, 0
    for path in sorted(history_dir.glob("*.csv")):
        adjustment_path = adjustment_dir / path.name
        if not adjustment_path.exists():
            failed += 1
            continue
        history = pd.read_csv(path, parse_dates=["date"])
        factors = pd.read_csv(adjustment_path, parse_dates=["date"])
        if factors.empty:
            history["cumulative_dividend"] = 0.0
            history["split_factor"] = 1.0
        else:
            factors = factors.sort_values("date")
            history = history.sort_values("date")
            history = pd.merge_asof(history, factors[["date", "s", "u"]], on="date", direction="backward")
            history["cumulative_dividend"] = history["u"].fillna(0.0)
            history["split_factor"] = history["s"].fillna(1.0)
        history["total_return_proxy"] = history["close"] * history["split_factor"] + history["cumulative_dividend"]
        history[["date", "close", "total_return_proxy", "cumulative_dividend", "split_factor"]].to_csv(
            output_dir / path.name, index=False, date_format="%Y-%m-%d"
        )
        success += 1
    if database is not None:
        build_total_return_table(database, output_dir)
    return {"success": success, "failed": failed}


def build_total_return_table(database: str | Path, proxy_dir: str | Path) -> None:
    """把总回报代理写入现有 SQLite，供研究器按 price-mode 读取。"""
    conn = sqlite3.connect(database)
    conn.executescript(
        "DROP TABLE IF EXISTS etf_daily_total_return;"
        "CREATE TABLE etf_daily_total_return ("
        "symbol TEXT NOT NULL, date TEXT NOT NULL, close REAL, total_return_proxy REAL, "
        "cumulative_dividend REAL, split_factor REAL, PRIMARY KEY(symbol, date));"
    )
    files = sorted(Path(proxy_dir).glob("*.csv"))
    for path in files:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.insert(0, "symbol", path.stem)
        frame.to_sql("etf_daily_total_return", conn, if_exists="append", index=False, chunksize=5000)
    conn.execute("CREATE INDEX idx_etf_total_return_date ON etf_daily_total_return(date)")
    conn.commit()
    conn.close()
