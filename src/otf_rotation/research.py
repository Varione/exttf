from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def load_research_frames(
    database: str | Path,
    price_mode: str = "raw",
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(database)
    if price_mode == "total-return":
        price_query = "SELECT date, symbol, total_return_proxy AS close FROM etf_daily_total_return ORDER BY date, symbol"
    else:
        price_query = "SELECT date, symbol, close FROM etf_daily ORDER BY date, symbol"
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
    prices = pd.read_sql_query(price_query.replace(" ORDER BY", f"{where} ORDER BY"), conn, params=params, parse_dates=["date"])
    amount = pd.read_sql_query(
        f"SELECT date, symbol, amount FROM etf_daily{where} ORDER BY date, symbol",
        conn,
        params=params,
        parse_dates=["date"],
    )
    quality = pd.read_sql_query("SELECT * FROM etf_quality", conn)
    conn.close()
    return (
        prices.pivot(index="date", columns="symbol", values="close").sort_index(),
        amount.pivot(index="date", columns="symbol", values="amount").sort_index(),
        quality,
    )


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _select_assets(
    scores: pd.Series,
    volatility: pd.Series,
    categories: pd.Series,
    top_n: int,
    max_per_category: int,
) -> list[str]:
    selected: list[str] = []
    counts: Counter[str] = Counter()
    for symbol in scores.sort_values(ascending=False).index:
        category = str(categories.get(symbol, "unknown"))
        if counts[category] >= max_per_category:
            continue
        selected.append(symbol)
        counts[category] += 1
        if len(selected) >= top_n:
            break
    return selected


def _inverse_vol_weights(
    selected: list[str], volatility: pd.Series, categories: pd.Series, single_cap: float, category_cap: float
) -> pd.Series:
    raw = (1.0 / volatility[selected].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    if raw.empty:
        return pd.Series(dtype=float)
    weights = raw / raw.sum()
    for _ in range(10):
        changed = False
        over_single = weights[weights > single_cap]
        if not over_single.empty:
            weights.loc[over_single.index] = single_cap
            changed = True
        group_weight = weights.groupby(categories.reindex(weights.index)).sum()
        over_groups = group_weight[group_weight > category_cap]
        for group in over_groups.index:
            members = categories[categories == group].index.intersection(weights.index)
            if len(members):
                weights.loc[members] *= category_cap / weights.loc[members].sum()
                changed = True
        if not changed:
            break
        remaining = 1.0 - weights.sum()
        if remaining <= 1e-9:
            break
        eligible = weights[weights < single_cap - 1e-9].index
        if len(eligible):
            add = raw[eligible] / raw[eligible].sum() * remaining
            weights.loc[eligible] += add
    return weights / weights.sum()


def _apply_drawdown_stop(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    breadth: pd.Series,
    stop_drawdown: float,
    reentry_breadth: float,
) -> pd.DataFrame:
    """按昨日收盘后的组合回撤决定今日是否空仓，避免使用当日未来收益。"""
    if stop_drawdown <= 0:
        return weights
    output = weights.copy()
    nav, high_water = 1.0, 1.0
    stopped = False
    for position, date in enumerate(weights.index):
        if position > 0:
            drawdown = nav / high_water - 1.0
            if not stopped and drawdown <= -stop_drawdown:
                stopped = True
            elif stopped and breadth.loc[date] >= reentry_breadth:
                stopped = False
        if stopped:
            output.iloc[position, :] = 0.0
        day_return = float((output.loc[date] * asset_returns.loc[date]).sum())
        nav *= 1.0 + day_return
        high_water = max(high_water, nav)
    return output


def run_all_etf_strategy(
    prices: pd.DataFrame,
    amount: pd.DataFrame,
    quality: pd.DataFrame,
    min_history_rows: int = 252,
    max_stale_days: int = 30,
    min_median_amount_60d: float = 10_000_000,
    top_n: int = 5,
    max_per_category: int = 2,
    single_cap: float = 0.30,
    category_cap: float = 0.40,
    cost_bps: float = 15.0,
    trend_window: int = 120,
    breadth_threshold: float = 0.0,
    target_vol: float = 0.0,
    stop_drawdown: float = 0.0,
    stop_reentry_breadth: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = prices.sort_index()
    amount = amount.reindex(index=prices.index, columns=prices.columns)
    categories = quality.set_index("symbol")["asset_class"].reindex(prices.columns).fillna("unknown")
    risk_assets = ~categories.isin(["cash"])

    returns = prices.pct_change(fill_method=None)
    vol20 = returns.rolling(20, min_periods=20).std() * np.sqrt(252)
    momentum60 = prices / prices.shift(60) - 1
    momentum120 = prices / prices.shift(120) - 1
    momentum252 = prices / prices.shift(252) - 1
    trend = prices > prices.rolling(trend_window, min_periods=trend_window).mean()
    liquidity = amount.rolling(60, min_periods=60).median()
    history_count = prices.notna().cumsum()
    last_seen = pd.DataFrame(index=prices.index, columns=prices.columns)
    for symbol in prices.columns:
        last_seen[symbol] = pd.Series(prices.index.where(prices[symbol].notna()), index=prices.index).ffill()
    stale_days = last_seen.apply(lambda col: (prices.index.to_series(index=prices.index) - col).dt.days)
    ratio60 = (momentum60 / vol20).replace([np.inf, -np.inf], np.nan)
    ratio120 = (momentum120 / vol20).replace([np.inf, -np.inf], np.nan)
    ratio252 = (momentum252 / vol20).replace([np.inf, -np.inf], np.nan)
    score = 0.5 * _zscore(ratio60) + 0.3 * _zscore(ratio120) + 0.2 * _zscore(ratio252)
    score = score.where(trend)
    score.loc[:, ~risk_assets] = np.nan

    week_key = prices.index.to_period("W-FRI")
    rebalance_dates = [group.index[-1] for _, group in prices.groupby(week_key, sort=True)]
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    orders: list[dict[str, object]] = []
    for signal_date in rebalance_dates:
        signal_pos = prices.index.get_loc(signal_date)
        execution_pos = signal_pos + 1
        if execution_pos >= len(prices.index):
            continue
        eligible = (
            (history_count.loc[signal_date] >= min_history_rows)
            & (liquidity.loc[signal_date] >= min_median_amount_60d)
            & (stale_days.loc[signal_date] <= max_stale_days)
            & trend.loc[signal_date]
            & risk_assets
        )
        ranked = score.loc[signal_date].where(eligible).dropna()
        selected = _select_assets(ranked, vol20.loc[signal_date], categories, top_n, max_per_category)
        if not selected:
            defensive = categories[categories.isin(["cash", "bond"])].index.intersection(prices.columns)
            selected = [symbol for symbol in defensive if prices.loc[signal_date, symbol] == prices.loc[signal_date, symbol]][:1]
        target = _inverse_vol_weights(selected, vol20.loc[signal_date], categories, single_cap, category_cap)
        next_rebalances = [prices.index.get_loc(d) for d in rebalance_dates if prices.index.get_loc(d) > signal_pos]
        next_execution_pos = min(next_rebalances) + 1 if next_rebalances else len(prices.index)
        if len(target):
            weights.iloc[execution_pos:next_execution_pos, :] = 0.0
            weights.loc[weights.index[execution_pos:next_execution_pos], target.index] = target.to_numpy()
        orders.append({
            "signal_date": signal_date, "execution_date": prices.index[execution_pos],
            "selected": ",".join(selected), "selected_count": len(selected),
        })

    asset_returns = returns.fillna(0.0)
    effective_weights = weights.copy()
    breadth = trend.loc[:, risk_assets].mean(axis=1)
    if breadth_threshold > 0:
        regime_ok = breadth.shift(1) >= breadth_threshold
        effective_weights.loc[~regime_ok.fillna(False), :] = 0.0
    gross = (effective_weights * asset_returns).sum(axis=1)
    if target_vol > 0:
        realized_vol = gross.rolling(60, min_periods=20).std() * np.sqrt(252)
        scale = (target_vol / realized_vol.replace(0, np.nan)).shift(1).clip(lower=0.0, upper=1.0).fillna(1.0)
        effective_weights = effective_weights.mul(scale, axis=0)
    if stop_drawdown > 0:
        effective_weights = _apply_drawdown_stop(
            effective_weights, asset_returns, breadth, stop_drawdown, stop_reentry_breadth
        )
    gross = (effective_weights * asset_returns).sum(axis=1)
    turnover = effective_weights.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * cost_bps / 10_000
    net = gross - cost
    daily = pd.DataFrame({
        "gross_return": gross, "turnover": turnover, "cost": cost,
        "net_return": net, "nav": (1 + net).cumprod(),
    }, index=prices.index)
    return daily, pd.DataFrame(orders)


def summarize_strategy(daily: pd.DataFrame, orders: pd.DataFrame) -> dict[str, object]:
    nav = daily["nav"]
    returns = daily["net_return"]
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1 / 365.25)
    drawdown = nav / nav.cummax() - 1
    annualized_return = float(nav.iloc[-1] ** (1 / years) - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    return {
        "start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
        "total_return": float(nav.iloc[-1] - 1), "annualized_return": annualized_return,
        "annualized_volatility": volatility, "max_drawdown": float(drawdown.min()),
        "calmar": float(annualized_return / abs(drawdown.min())) if drawdown.min() < 0 else None,
        "average_daily_turnover": float(daily["turnover"].mean()),
        "total_cost": float(daily["cost"].sum()), "rebalance_count": int(len(orders)),
    }


def run_research_report(
    database: str | Path,
    output_dir: str | Path,
    price_mode: str = "raw",
    symbols: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    **kwargs: object,
) -> dict[str, object]:
    prices, amount, quality = load_research_frames(database, price_mode=price_mode, symbols=symbols, start_date=start_date, end_date=end_date)
    daily, orders = run_all_etf_strategy(prices, amount, quality, **kwargs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "all_etf_strategy_daily.csv", index_label="date")
    orders.to_csv(output_dir / "all_etf_strategy_orders.csv", index=False)
    result = summarize_strategy(daily, orders)
    result["price_mode"] = price_mode
    result["parameters"] = kwargs
    (output_dir / "all_etf_strategy_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
