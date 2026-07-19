from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_signals(
    prices: pd.DataFrame,
    momentum_window: int = 60,
    volatility_window: int = 20,
    trend_window: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 score 和趋势资格矩阵。所有信号只使用当日及以前价格。"""
    prices = prices.astype(float).sort_index()
    returns = prices.pct_change(fill_method=None)
    momentum = prices / prices.shift(momentum_window) - 1
    volatility = returns.rolling(volatility_window, min_periods=volatility_window).std() * np.sqrt(252)
    moving_average = prices.rolling(trend_window, min_periods=trend_window).mean()
    trend = (prices > moving_average).where(moving_average.notna())
    score = momentum.div(volatility.replace(0, np.nan))
    score = score.where(trend)
    return score, trend


def weekly_targets(
    prices: pd.DataFrame,
    score: pd.DataFrame,
    defensive_asset: str,
    top_n: int = 3,
    rebalance_weekday: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在每周最后一个可用交易日计算目标权重，并从下一交易日执行。"""
    if defensive_asset not in prices.columns:
        raise ValueError(f"防御资产 {defensive_asset!r} 不在价格表中")
    dates = prices.index
    week_key = dates.to_period("W-FRI")
    rebalance_dates = pd.DatetimeIndex(
        [group.index[-1] for _, group in prices.groupby(week_key, sort=True)]
    )
    weights = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    orders: list[dict[str, object]] = []

    for signal_date in rebalance_dates:
        signal_pos = dates.get_loc(signal_date)
        exec_pos = signal_pos + 1
        if exec_pos >= len(dates):
            continue
        row = score.loc[signal_date].dropna().sort_values(ascending=False)
        selected = list(row.head(top_n).index)
        if not selected:
            selected = [defensive_asset]
        target = pd.Series(0.0, index=prices.columns)
        target[selected] = 1.0 / len(selected)
        next_signal_positions = [dates.get_loc(d) for d in rebalance_dates if dates.get_loc(d) > signal_pos]
        next_exec_pos = (min(next_signal_positions) + 1) if next_signal_positions else len(dates)
        weights.iloc[exec_pos:next_exec_pos] = target.to_numpy()
        orders.append({"signal_date": signal_date, "execution_date": dates[exec_pos], "selected": ",".join(selected)})

    return weights, pd.DataFrame(orders)
