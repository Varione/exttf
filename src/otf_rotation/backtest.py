from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import calculate_signals, weekly_targets


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    orders: pd.DataFrame
    scores: pd.DataFrame


def run_backtest(
    prices: pd.DataFrame,
    defensive_asset: str,
    top_n: int = 3,
    cost_bps: float = 10.0,
    momentum_window: int = 60,
    volatility_window: int = 20,
    trend_window: int = 120,
) -> BacktestResult:
    prices = prices.dropna(how="all").sort_index()
    scores, _ = calculate_signals(prices, momentum_window, volatility_window, trend_window)
    weights, orders = weekly_targets(prices, scores, defensive_asset, top_n=top_n)
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    gross = (weights * asset_returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * cost_bps / 10_000
    net = gross - costs
    daily = pd.DataFrame(
        {
            "gross_return": gross,
            "turnover": turnover,
            "cost": costs,
            "net_return": net,
            "nav": (1.0 + net).cumprod(),
        },
        index=prices.index,
    )
    return BacktestResult(daily=daily, orders=orders, scores=scores)


def summary(result: BacktestResult) -> dict[str, float]:
    nav = result.daily["nav"]
    net = result.daily["net_return"]
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1 / 365.25)
    drawdown = nav / nav.cummax() - 1
    return {
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "total_return": float(nav.iloc[-1] - 1),
        "annualized_return": float(nav.iloc[-1] ** (1 / years) - 1),
        "annualized_volatility": float(net.std(ddof=1) * np.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "rebalance_count": int(len(result.orders)),
    }
