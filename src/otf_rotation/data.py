from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sqlite3
import time

import pandas as pd


def read_universe(path: str | Path) -> pd.DataFrame:
    """读取并校验资产池配置。"""
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"asset", "signal_symbol", "signal_name", "defensive"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"universe 缺少字段: {sorted(missing)}")
    if df["asset"].duplicated().any():
        raise ValueError("universe 的 asset 必须唯一")
    df["defensive"] = df["defensive"].astype(int)
    return df


def normalize_price_frame(frame: pd.DataFrame, name: str = "close") -> pd.Series:
    """把数据源返回的行情表标准化为 DateTimeIndex 的收盘价序列。"""
    date_col = next((c for c in ("date", "日期", "交易日期", "净值日期") if c in frame.columns), None)
    price_col = next((c for c in ("close", "收盘", "单位净值", "复权单位净值") if c in frame.columns), None)
    if date_col is None or price_col is None:
        raise ValueError(f"无法识别 {name} 数据列，实际列为: {list(frame.columns)}")
    out = frame[[date_col, price_col]].rename(columns={date_col: "date", price_col: name}).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.dropna().drop_duplicates("date").sort_values("date")
    return out.set_index("date")[name]


def normalize_etf_history(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """统一东财/新浪 ETF 日线字段，保留 OHLCV 和成交额。"""
    aliases = {
        "date": ("date", "日期", "交易日期"),
        "open": ("open", "开盘"),
        "high": ("high", "最高"),
        "low": ("low", "最低"),
        "close": ("close", "收盘"),
        "volume": ("volume", "成交量"),
        "amount": ("amount", "成交额"),
    }
    selected: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        source = next((c for c in candidates if c in frame.columns), None)
        if source is not None:
            selected[canonical] = source
    if "date" not in selected or "close" not in selected:
        raise ValueError(f"无法识别 {symbol} ETF 历史行情字段: {list(frame.columns)}")
    out = frame[[selected[k] for k in selected]].rename(
        columns={source: canonical for canonical, source in selected.items()}
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in out.columns:
        if column != "date":
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    return out.set_index("date")


def load_price_csv(path: str | Path) -> pd.DataFrame:
    """读取宽表行情：date + 多个资产列。"""
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "date"}).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce").sort_index()
    return df[~df.index.duplicated(keep="last")].dropna(how="all")


def load_prices_sqlite(
    database: str | Path,
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """从全量 SQLite 按需读取 close 宽表，避免一次读取所有字段。"""
    clauses: list[str] = []
    params: list[object] = []
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        clauses.append(f"symbol IN ({placeholders})")
        params.extend(symbols)
    if start_date:
        clauses.append("date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("date <= ?")
        params.append(end_date)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    conn = sqlite3.connect(database)
    query = f"SELECT date, symbol, close FROM etf_daily{where} ORDER BY date, symbol"
    long = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
    conn.close()
    if long.empty:
        return pd.DataFrame()
    return long.pivot(index="date", columns="symbol", values="close").sort_index()


def save_price_csv(prices: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.sort_index().to_csv(path, index_label="date", date_format="%Y-%m-%d")


class AkshareProvider:
    """AKShare 适配器；导入延迟到调用时，便于离线回测。"""

    def __init__(self) -> None:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("未安装 AKShare，请执行: python -m pip install -e '.[data]'") from exc
        self.ak = ak

    def _fetch_etf_raw(self, symbol: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, str]:
        symbol = str(symbol)
        try:
            raw = self.ak.fund_etf_hist_em(
                symbol=symbol, start_date=start_date, end_date=end_date, adjust=""
            )
            source = "eastmoney"
        except Exception as eastmoney_error:
            # 东财在部分网络环境会拒绝请求；新浪接口返回全历史行情，作为免费回退。
            market = "sz" if symbol.startswith(("15", "16")) else "sh"
            try:
                raw = self.ak.fund_etf_hist_sina(symbol=f"{market}{symbol}")
                source = "sina"
            except Exception as sina_error:
                raise RuntimeError(
                    f"东财与新浪 ETF 行情接口均失败；东财={eastmoney_error}; 新浪={sina_error}"
                ) from sina_error
        return normalize_etf_history(raw, symbol), source

    def fetch_etf_history(
        self, symbol: str, start_date: str, end_date: str, source_preference: str = "auto"
    ) -> tuple[pd.DataFrame, str]:
        if source_preference == "sina":
            symbol = str(symbol)
            market = "sz" if symbol.startswith(("15", "16")) else "sh"
            raw = self.ak.fund_etf_hist_sina(symbol=f"{market}{symbol}")
            history, source = normalize_etf_history(raw, symbol), "sina"
        else:
            history, source = self._fetch_etf_raw(symbol, start_date, end_date)
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return history.loc[(history.index >= start) & (history.index <= end)], source

    def fetch_etf_close(self, symbol: str, start_date: str, end_date: str) -> pd.Series:
        history, _ = self.fetch_etf_history(symbol, start_date, end_date)
        return history["close"].rename(str(symbol))

    def list_etfs(self) -> pd.DataFrame:
        raw = self.ak.fund_etf_spot_em()
        code_col = next(c for c in ("代码", "code") if c in raw.columns)
        name_col = next(c for c in ("名称", "name") if c in raw.columns)
        out = raw[[code_col, name_col]].rename(columns={code_col: "symbol", name_col: "name"}).copy()
        out["symbol"] = out["symbol"].astype(str).str.zfill(6)
        return out.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)

    def fetch_open_fund_nav(self, symbol: str) -> pd.Series:
        raw = self.ak.fund_open_fund_info_em(symbol=str(symbol), indicator="单位净值走势")
        return normalize_price_frame(raw, name=str(symbol))


def fetch_etf_prices(
    universe: pd.DataFrame, start_date: str, end_date: str, output: str | Path
) -> pd.DataFrame:
    """按资产池抓取 ETF 收盘价并保存；单个代码失败会明确报错。"""
    provider = AkshareProvider()
    series: list[pd.Series] = []
    errors: list[str] = []
    for row in universe.itertuples(index=False):
        try:
            s = provider.fetch_etf_close(row.signal_symbol, start_date, end_date)
            s.name = row.asset
            series.append(s)
        except Exception as exc:  # 数据源异常需保留资产名
            errors.append(f"{row.asset}({row.signal_symbol}): {exc}")
    if errors:
        raise RuntimeError("以下 ETF 抓取失败:\n" + "\n".join(errors))
    prices = pd.concat(series, axis=1).sort_index()
    save_price_csv(prices, output)
    return prices


def fetch_all_etf_history(
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    workers: int = 4,
    retries: int = 2,
    sleep_seconds: float = 0.2,
    resume: bool = True,
    refresh_catalog: bool = False,
) -> dict[str, int]:
    """抓取当前 ETF 清单的全部日线历史，按代码分文件保存并支持断点续传。"""
    root = Path(output_dir)
    history_dir = root / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "etf_catalog.csv"
    failed_path = root / "etf_failed.csv"

    if catalog_path.exists() and not refresh_catalog:
        catalog = pd.read_csv(catalog_path, dtype={"symbol": str}).fillna("")
        catalog["symbol"] = catalog["symbol"].astype(str).str.zfill(6)
    else:
        catalog = AkshareProvider().list_etfs()
        catalog.to_csv(catalog_path, index=False, encoding="utf-8-sig")
    pending = catalog[~catalog["symbol"].map(lambda s: (history_dir / f"{s}.csv").exists())] if resume else catalog

    records: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _fetch_one_etf_job,
                row.symbol,
                row.name,
                start_date,
                end_date,
                str(history_dir),
                retries,
                sleep_seconds,
            ): row.symbol
            for row in pending.itertuples(index=False)
        }
        completed = 0
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            completed += 1
            print(f"[{completed}/{len(futures)}] {record['symbol']} {record['status']} rows={record['rows']}", flush=True)

    record_map = {str(record["symbol"]): record for record in records}
    all_records: list[dict[str, object]] = []
    for row in catalog.itertuples(index=False):
        symbol = str(row.symbol).zfill(6)
        path = history_dir / f"{symbol}.csv"
        if symbol in record_map:
            all_records.append(record_map[symbol])
        elif path.exists() and path.stat().st_size > 0:
            with path.open("r", encoding="utf-8-sig") as handle:
                rows = max(sum(1 for _ in handle) - 1, 0)
            all_records.append({"symbol": symbol, "name": row.name, "rows": rows, "source": "existing", "status": "ok", "error": ""})
        else:
            all_records.append({"symbol": symbol, "name": row.name, "rows": 0, "source": "", "status": "failed", "error": "未找到历史文件"})
    result = pd.DataFrame(all_records).sort_values("symbol")
    result.to_csv(root / "etf_fetch_status.csv", index=False, encoding="utf-8-sig")
    result[result["status"] == "failed"].to_csv(failed_path, index=False, encoding="utf-8-sig")
    return {
        "catalog": len(catalog),
        "pending": len(pending),
        "success": int((result["status"] == "ok").sum()),
        "failed": int((result["status"] == "failed").sum()),
    }


def _fetch_one_etf_job(
    symbol: str,
    name: str,
    start_date: str,
    end_date: str,
    history_dir: str,
    retries: int,
    sleep_seconds: float,
) -> dict[str, object]:
    """进程池任务：每个进程独立初始化 AKShare，避免 MiniRacer/requests 线程竞争。"""
    path = Path(history_dir) / f"{symbol}.csv"
    last_error = ""
    for attempt in range(retries + 1):
        try:
            history, source = AkshareProvider().fetch_etf_history(
                symbol, start_date, end_date, source_preference="sina"
            )
            if history.empty:
                raise ValueError("返回空数据")
            history.to_csv(path, index_label="date", date_format="%Y-%m-%d")
            return {"symbol": symbol, "name": name, "rows": len(history), "source": source, "status": "ok", "error": ""}
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(sleep_seconds * (attempt + 1))
    return {"symbol": symbol, "name": name, "rows": 0, "source": "", "status": "failed", "error": last_error}
