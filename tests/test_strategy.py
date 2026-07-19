import numpy as np
import pandas as pd

from otf_rotation.backtest import run_backtest
from otf_rotation.strategy import calculate_signals, weekly_targets


def make_prices(days=320):
    dates = pd.bdate_range("2020-01-01", periods=days)
    t = np.arange(days)
    return pd.DataFrame(
        {
            "RISK": 100 + t * 0.2,
            "RISK2": 100 + np.sin(t / 7) * 3,
            "BOND": 100 + t * 0.03,
        },
        index=dates,
    )


def test_signal_is_nan_before_lookback():
    prices = make_prices()
    score, trend = calculate_signals(prices)
    # 120 日均线在第 120 个观测日（位置 119）才首次可用。
    assert score.iloc[:119].isna().all().all()
    assert trend.iloc[:119].isna().all().all()


def test_execution_starts_after_signal_date():
    prices = make_prices()
    score, _ = calculate_signals(prices)
    weights, orders = weekly_targets(prices, score, "BOND")
    for row in orders.itertuples(index=False):
        assert row.execution_date > row.signal_date
        signal_pos = prices.index.get_loc(row.signal_date)
        if signal_pos == 0:
            assert (weights.iloc[signal_pos] == 0).all()
        else:
            # 信号日仍使用旧权重，新权重从下一可用交易日开始。
            assert weights.iloc[signal_pos].equals(weights.iloc[signal_pos - 1])


def test_backtest_has_finite_nav_and_costs():
    result = run_backtest(make_prices(), "BOND")
    assert np.isfinite(result.daily["nav"]).all()
    assert (result.daily["cost"] >= 0).all()
    assert result.daily.index.is_monotonic_increasing
