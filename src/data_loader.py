"""Point-in-time ETF and fund price loaders."""

from __future__ import annotations

import sqlite3
import warnings

import numpy as np
import pandas as pd


ETF_PRICE_MODES = {"raw_close", "total_return_proxy", "hfq_reference"}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {row[0] for row in rows}


def load_price_series(
    db_path: str = "data/processed/etf.sqlite",
    data_mode: str = "etf",
    price_mode: str = "total_return_proxy",
) -> pd.DataFrame:
    """Load a total-return-capable price series.

    ``data_mode='otc_nav'`` deliberately fails closed unless a ``fund_nav``
    table with an adjusted/total-return NAV is present. This prevents an ETF
    close-price backtest from being mislabeled as an OTC fund backtest.
    """
    with sqlite3.connect(db_path) as conn:
        tables = _table_names(conn)

        if data_mode == "otc_nav":
            if "fund_nav" not in tables:
                raise RuntimeError(
                    "OTC_NAV_UNAVAILABLE: fund_nav table is missing; "
                    "ETF prices cannot be used silently as OTC NAV."
                )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(fund_nav)")}
            symbol_col = next((c for c in ("symbol", "fund_code", "code") if c in cols), None)
            date_col = next((c for c in ("date", "nav_date") if c in cols), None)
            nav_col = next(
                (c for c in ("total_return_nav", "adjusted_nav") if c in cols),
                None,
            )
            if not symbol_col or not date_col or not nav_col:
                raise RuntimeError(
                    "OTC_NAV_SCHEMA_INVALID: fund_nav must contain a symbol/code, "
                    "date/nav_date and total_return_nav/adjusted_nav column."
                )
            query = (
                f'SELECT "{symbol_col}" AS symbol, "{date_col}" AS date, '
                f'"{nav_col}" AS price FROM fund_nav'
            )
            prices = pd.read_sql_query(query, conn)
            prices["price_mode"] = "otc_total_return_nav"
            prices["reference_verified"] = True
        elif data_mode == "etf":
            if price_mode not in ETF_PRICE_MODES:
                raise ValueError(f"Unsupported ETF price mode: {price_mode}")
            if price_mode == "raw_close":
                prices = pd.read_sql_query(
                    "SELECT symbol, date, close AS price, open AS raw_open, "
                    "close AS raw_close FROM etf_daily",
                    conn,
                )
                prices["reference_verified"] = False
            else:
                if "etf_daily_price_modes" not in tables:
                    raise RuntimeError("PRICE_MODE_UNAVAILABLE: etf_daily_price_modes is missing")
                prices = pd.read_sql_query(
                    f"SELECT pm.symbol, pm.date, pm.{price_mode} AS price, "
                    "pm.validation_status, d.open AS raw_open, d.close AS raw_close "
                    "FROM etf_daily_price_modes pm "
                    "JOIN etf_daily d ON d.symbol=pm.symbol AND d.date=pm.date",
                    conn,
                )
                prices["reference_verified"] = prices["validation_status"].eq("PASS")
                prices = prices.drop(columns="validation_status")
            prices["price_mode"] = price_mode
        else:
            raise ValueError(f"Unsupported data mode: {data_mode}")

    prices["date"] = pd.to_datetime(prices["date"])
    prices["symbol"] = prices["symbol"].astype(str).str.zfill(6)
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices = prices.loc[prices["price"] > 0].copy()
    prices = prices.sort_values(["symbol", "date"]).drop_duplicates(
        ["symbol", "date"], keep="last"
    )

    if data_mode == "etf":
        # ETF share conversions can create multi-hundred-percent mechanical
        # price jumps. Replace only those gated events with the same-day
        # open-to-close move, then rebuild a continuous level series. The
        # event is retained explicitly for audit instead of silently clipping.
        prices["raw_open"] = pd.to_numeric(prices["raw_open"], errors="coerce")
        prices["raw_close"] = pd.to_numeric(prices["raw_close"], errors="coerce")

        returns = prices.groupby("symbol")["price"].pct_change(fill_method=None)
        repair = returns.abs().ge(0.50)
        intraday = prices["raw_close"] / prices["raw_open"] - 1.0
        invalid = repair & (~np.isfinite(intraday) | intraday.abs().gt(0.30))
        if invalid.any():
            rows = prices.loc[invalid, ["symbol", "date"]].to_dict("records")
            raise RuntimeError(f"CORPORATE_ACTION_REPAIR_FAILED: {rows[:5]}")
        returns.loc[repair] = intraday.loc[repair]
        first_price = prices.groupby("symbol")["price"].transform("first")
        growth = (1.0 + returns.fillna(0.0)).groupby(prices["symbol"]).cumprod()
        prices["price"] = first_price * growth
        prices["corporate_action_repaired"] = repair.to_numpy()
    else:
        prices["corporate_action_repaired"] = False

    return prices


def load_etf_daily(
    db_path: str = "data/processed/etf.sqlite",
    price_mode: str = "total_return_proxy",
    min_history: int = 250,
    min_median_amount_60d: float = 1e6,
) -> pd.DataFrame:
    """Load ETF OHLCV and compute a point-in-time eligibility gate.

    Eligibility uses only observations and turnover known through each row.
    No current-date quality table is joined into historical data.
    """
    with sqlite3.connect(db_path) as conn:
        daily = pd.read_sql_query(
            "SELECT symbol, date, open, high, low, close, volume, amount FROM etf_daily",
            conn,
        )

    daily["date"] = pd.to_datetime(daily["date"])
    daily["symbol"] = daily["symbol"].astype(str).str.zfill(6)
    daily = daily.sort_values(["symbol", "date"]).drop_duplicates(
        ["symbol", "date"], keep="last"
    )
    daily = daily.loc[pd.to_numeric(daily["close"], errors="coerce") > 0].copy()
    daily["raw_close"] = daily["close"]

    if price_mode != "raw_close":
        prices = load_price_series(db_path, data_mode="etf", price_mode=price_mode)
        prices = prices[["symbol", "date", "price", "reference_verified"]]
        daily = daily.merge(prices, on=["symbol", "date"], how="left", validate="one_to_one")
        if daily["price"].isna().any():
            missing = int(daily["price"].isna().sum())
            raise RuntimeError(f"PRICE_MODE_INCOMPLETE: {missing} adjusted prices are missing")
        scale = daily["price"] / daily["raw_close"]
        for col in ("open", "high", "low", "close"):
            daily[col] = pd.to_numeric(daily[col], errors="coerce") * scale
    else:
        daily["reference_verified"] = False

    grouped = daily.groupby("symbol", sort=False)
    daily["pit_observations"] = grouped.cumcount() + 1
    daily["pit_median_amount_60d"] = grouped["amount"].transform(
        lambda values: pd.to_numeric(values, errors="coerce")
        .rolling(60, min_periods=20)
        .median()
    )
    daily["pit_eligible"] = (
        daily["pit_observations"].ge(min_history)
        & daily["pit_median_amount_60d"].ge(min_median_amount_60d)
    )
    daily["price_mode"] = price_mode
    return daily


def load_etf_lifecycle(
    db_path: str = "data/processed/etf.sqlite",
) -> pd.DataFrame:
    """Load ETF lifecycle table for point-in-time universe filtering."""
    with sqlite3.connect(db_path) as conn:
        tables = _table_names(conn)
        if "etf_lifecycle" not in tables:
            raise RuntimeError(
                "LIFECYCLE_UNAVAILABLE: etf_lifecycle table is missing; "
                "run build_etf_lifecycle.py first."
            )
        lifecycle = pd.read_sql_query(
            "SELECT symbol, name, asset_class, list_date, last_date, "
            "is_active, total_trading_days, first_year, data_quality_score "
            "FROM etf_lifecycle",
            conn,
        )
    lifecycle["list_date"] = pd.to_datetime(lifecycle["list_date"])
    lifecycle["last_date"] = pd.to_datetime(lifecycle["last_date"])
    lifecycle["symbol"] = lifecycle["symbol"].astype(str).str.zfill(6)
    return lifecycle


def apply_lifecycle_filter(
    df: pd.DataFrame,
    db_path: str = "data/processed/etf.sqlite",
) -> pd.DataFrame:
    """Filter a DataFrame to only include rows within each ETF's lifecycle.

    An observation at date t for symbol s is kept only if
    list_date(s) <= t <= last_date(s).
    This prevents survivorship bias by excluding ETFs that had not yet
    listed or had already delisted at date t.
    """
    lifecycle = load_etf_lifecycle(db_path)

    df = df.copy()
    if "date" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    merged = df.merge(lifecycle[["symbol", "list_date", "last_date"]], on="symbol", how="left")
    mask = (merged["date"] >= merged["list_date"]) & (merged["date"] <= merged["last_date"])
    filtered = merged.loc[mask].drop(columns=["list_date", "last_date"])

    n_dropped = len(df) - len(filtered)
    if n_dropped > 0:
        warnings.warn(
            f"LIFECYCLE_FILTER: dropped {n_dropped} rows outside ETF lifecycle window"
        )
    return filtered


def get_valid_symbols(
    db_path: str = "data/processed/etf.sqlite",
    as_of: pd.Timestamp | None = None,
) -> list[str]:
    """Return all symbols eligible as of a given date.

    If ``as_of`` is None, returns every symbol that has ever existed.
    If ``as_of`` is provided, only returns symbols whose lifecycle
    covers that date (list_date <= as_of <= last_date).
    """
    with sqlite3.connect(db_path) as conn:
        tables = _table_names(conn)

    if as_of is not None and "etf_lifecycle" in tables:
        with sqlite3.connect(db_path) as conn:
            lifecycle = pd.read_sql_query(
                "SELECT symbol FROM etf_lifecycle",
                conn,
            )
        lifecycle["list_date"] = pd.to_datetime(lifecycle["list_date"])
        lifecycle["last_date"] = pd.to_datetime(lifecycle["last_date"])
        mask = (lifecycle["list_date"] <= as_of) & (lifecycle["last_date"] >= as_of)
        symbols = lifecycle.loc[mask, "symbol"].tolist()
    else:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM etf_daily ORDER BY symbol"
            ).fetchall()
        symbols = [str(row[0]).zfill(6) for row in rows]

    return sorted(set(symbols))
