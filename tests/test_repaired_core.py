from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest_engine import BacktestEngine
from build_dataset import build_classification_dataset
from data_loader import load_price_series
from detect_regimes import assign_regime_to_date
from factor_definitions import _down_capture, _up_capture
from rl_environment import StrategyWeightEnv
from strategy_library import S04_VolTargetTrend, S23_OversoldLong


def make_signal_factors(dates: pd.DatetimeIndex, n_symbols: int = 60) -> pd.DataFrame:
    rows = []
    for date in dates:
        for idx in range(n_symbols):
            rows.append(
                {
                    "date": date,
                    "symbol": f"{idx:06d}",
                    "pit_eligible": True,
                    "ma_dist_60": -0.05,
                    "mom_20": -0.02,
                    "real_vol_10": 0.01,
                    "rsi_14": 50.0,
                    "stoch_k_14": 50.0,
                }
            )
    return pd.DataFrame(rows)


def make_engine_inputs(n_days: int = 6, n_symbols: int = 20):
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    factors = []
    prices = []
    for day_idx, date in enumerate(dates):
        for symbol_idx in range(n_symbols):
            symbol = f"{symbol_idx:06d}"
            factors.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "pit_eligible": True,
                    "mom_20": 1.0,
                }
            )
            prices.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "price": 100.0 * (1.10 ** day_idx),
                    "reference_verified": True,
                }
            )
    regime = pd.DataFrame({"date": dates, "regime": np.zeros(n_days, dtype=int)})
    return dates, pd.DataFrame(factors), regime, pd.DataFrame(prices)


def test_zero_signal_strategies_hold_cash():
    date = pd.Timestamp("2024-01-02")
    factors = make_signal_factors(pd.DatetimeIndex([date]))
    assert S04_VolTargetTrend(factors).get_positions(date, 20, 0.05) == {}
    assert S23_OversoldLong(factors).get_positions(date, 20, 0.05) == {}


def test_s04_portfolio_variance_includes_pairwise_factor_two():
    vols = [0.02, 0.03, 0.04]
    n = len(vols)
    w = 1.0 / n
    expected = np.sqrt(
        sum((w * v) ** 2 for v in vols)
        + 2.0 * 0.3 * w**2 * sum(
            vols[i] * vols[j] for i in range(n) for j in range(i + 1, n)
        )
    )
    actual = S04_VolTargetTrend._estimate_portfolio_vol_daily(vols, rho=0.3)
    assert actual == pytest.approx(expected)


def test_pit_gate_excludes_ineligible_symbol():
    date = pd.Timestamp("2024-01-02")
    factors = make_signal_factors(pd.DatetimeIndex([date]), n_symbols=61)
    factors.loc[factors.index[0], "pit_eligible"] = False
    factors.loc[factors.index[-1], ["ma_dist_60", "mom_20"]] = [0.1, 0.2]
    positions = S04_VolTargetTrend(factors).get_positions(date, 20, 0.05)
    assert list(positions) == ["000060"]


def test_nav_lag_full_cost_and_true_buy_hold():
    dates, factors, regime, prices = make_engine_inputs()
    engine = BacktestEngine(
        factors_df=factors,
        regime_df=regime,
        prices_df=prices,
        fee_rate_per_side=0.0003,
        slippage_rate_per_side=0.0002,
        max_abs_asset_return=None,
    )
    daily = engine.run_backtest(
        "B0_BuyHold_EW",
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        signal_to_return_lag=2,
        rebalance_every=1,
    )
    assert daily.loc[0:1, "exposure"].eq(0).all()
    assert daily.loc[2, "signal_date"] == dates[0]
    executed = 1.0 / 1.0005
    cost = executed * 0.0005
    assert daily.loc[2, "traded_notional"] == pytest.approx(executed)
    assert daily.loc[2, "transaction_cost"] == pytest.approx(cost)
    assert daily.loc[2, "return"] == pytest.approx(executed * 0.10 - cost)
    assert daily.loc[3:, "traded_notional"].eq(0).all()


def test_inferred_lifecycle_table_is_only_partial_pit():
    dates, factors, regime, prices = make_engine_inputs()
    engine = BacktestEngine(
        factors_df=factors,
        regime_df=regime,
        prices_df=prices,
        max_abs_asset_return=None,
    )
    assert engine.pit_status == "PIT_PARTIAL"


def test_regime_gate_moves_strategy_to_cash():
    dates, factors, regime, prices = make_engine_inputs()
    regime["regime"] = 1  # S01 is only permitted in regime 0.
    engine = BacktestEngine(
        factors_df=factors,
        regime_df=regime,
        prices_df=prices,
        max_abs_asset_return=None,
    )
    daily = engine.run_backtest(
        "S01_CS_Momentum",
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        signal_to_return_lag=2,
        rebalance_every=1,
        respect_regime=True,
    )
    assert daily["exposure"].eq(0).all()
    assert daily["return"].eq(0).all()


def test_otc_mode_fails_closed_without_nav_table(tmp_path: Path):
    db_path = tmp_path / "empty.sqlite"
    sqlite3.connect(db_path).close()
    with pytest.raises(RuntimeError, match="OTC_NAV_UNAVAILABLE"):
        load_price_series(str(db_path), data_mode="otc_nav")


def test_capture_factors_do_not_change_when_future_is_appended():
    close = pd.Series(np.linspace(100, 120, 50) + np.sin(np.arange(50)))
    prefix = pd.DataFrame({"close": close.iloc[:35]})
    extended = pd.DataFrame({"close": pd.concat([close, pd.Series([1000.0, 1.0])], ignore_index=True)})
    pd.testing.assert_series_equal(
        _up_capture(prefix, 20), _up_capture(extended, 20).iloc[:35], check_names=False
    )
    pd.testing.assert_series_equal(
        _down_capture(prefix, 20), _down_capture(extended, 20).iloc[:35], check_names=False
    )


def test_regime_lookup_never_uses_future_date():
    regimes = pd.Series(
        [0, 2], index=pd.to_datetime(["2025-01-03", "2025-01-06"])
    )
    assert assign_regime_to_date(pd.Timestamp("2025-01-04"), regimes) == 0


def test_classification_forward_is_next_trading_row():
    dates = pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"])
    factors = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["000001"] * 4,
            "pit_eligible": [True] * 4,
            "mom_5": [1.0, 2.0, 3.0, 4.0],
        }
    )
    regimes = pd.Series([0, 1, 2, 0], index=dates)
    _, labels, sample_dates = build_classification_dataset(
        factors, ["mom_5"], regimes, lookback=1, forward=1
    )
    assert sample_dates[0] == dates[1]
    assert labels[0] == regimes.loc[dates[2]]


class FakeEngine:
    def __init__(self):
        self.dates = pd.bdate_range("2024-01-02", periods=6)
        self.factors = pd.DataFrame(
            {
                "date": self.dates,
                "symbol": ["000001"] * 6,
                "pit_eligible": [True] * 6,
                "mom_5": np.arange(6, dtype=float),
            }
        )
        self.regime_preds = pd.DataFrame(
            {
                "date": self.dates,
                "regime": [0] * 6,
                "prob_r0": [1.0] * 6,
                "prob_r1": [0.0] * 6,
                "prob_r2": [0.0] * 6,
            }
        )

    def run_backtest(self, strategy_name, start, end, **kwargs):
        dates = self.dates[(self.dates >= start) & (self.dates <= end)]
        return pd.DataFrame({"date": dates, "return": np.zeros(len(dates))})


def test_rl_state_is_lagged_and_cost_is_deducted():
    env = StrategyWeightEnv(
        start="2024-01-02",
        end="2024-01-09",
        lookback=2,
        engine=FakeEngine(),
        allocation_cost_rate_per_side=0.0005,
    )
    state = env.reset()
    # New state layout: [regime_probs(3), recent_returns(n*L), strat_vols(n), strat_dd(n), factors(F)]
    factor_offset = 3 + env.n_strategies * env.lookback + env.n_strategies + env.n_strategies
    # current_idx=2, so observable factor date is index 1, not index 2.
    assert state[factor_offset] == pytest.approx(1.0)
    _, _, _, info = env.step(np.ones(env.n_strategies))
    assert info["traded_notional"] == pytest.approx(1.0)
    assert info["transaction_cost"] == pytest.approx(0.0005)
    assert info["return"] == pytest.approx(-0.0005)
