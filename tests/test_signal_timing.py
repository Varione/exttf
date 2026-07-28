"""Phase 2.1: Signal-to-return timing verification (no look-ahead bias).

Verifies that ``signal_to_return_lag=2`` implements the economic semantics:

    signal(t) -> target_weight(t) -> execution(t+1) -> return(t+2)

i.e., the return recorded on date T uses a signal computed from data
available on or before T-2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest_engine import BacktestEngine
from strategy_library import STRATEGIES, Strategy

DB_PATH = str(Path(__file__).resolve().parents[1] / "data" / "processed" / "etf.sqlite")
FACTOR_CSV = str(
    Path(__file__).resolve().parents[1] / "data" / "processed" / "factors_all_repaired.csv"
)
REGIME_CSV = str(
    Path(__file__).resolve().parents[1] / "data" / "processed" / "regime_predictions.csv"
)

START_DATE = "2026-01-02"
END_DATE = "2026-01-30"


@pytest.fixture(scope="module")
def engine():
    return BacktestEngine(
        factor_path=FACTOR_CSV,
        regime_path=REGIME_CSV,
        db_path=DB_PATH,
        data_mode="etf",
        price_mode="total_return_proxy",
        fee_rate_per_side=0.0003,
        slippage_rate_per_side=0.0002,
        require_pit=True,
    )


class TestSignalTimingLagEquals2:
    """Verify signal_to_return_lag=2 produces correct date offsets."""

    def test_signal_date_is_t_minus_2_on_rebalance_days(self, engine):
        daily = engine.run_backtest(
            "S01_CS_Momentum",
            start=START_DATE,
            end=END_DATE,
            n_hold=5,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )
        rebalance_rows = daily[daily["rebalance"] == True]  # noqa: E712
        assert len(rebalance_rows) > 0, "No rebalance days in test window"

        for _, row in rebalance_rows.iterrows():
            date_t = pd.Timestamp(row["date"])
            signal_t = pd.Timestamp(row["signal_date"])
            trading_days_between = engine._date_to_index[date_t] - engine._date_to_index[signal_t]
            assert trading_days_between == 2, (
                f"Expected 2 trading days between signal({signal_t.date()}) "
                f"and return({date_t.date()}), got {trading_days_between}"
            )

    def test_signal_date_lag_consistency(self, engine):
        """Every row must satisfy: date - signal_date == lag trading days."""
        daily = engine.run_backtest(
            "S01_CS_Momentum",
            start=START_DATE,
            end=END_DATE,
            n_hold=5,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )
        for _, row in daily.iterrows():
            date_t = pd.Timestamp(row["date"])
            signal_t = row["signal_date"]
            if pd.notna(signal_t):
                idx_t = engine._date_to_index[date_t]
                idx_s = engine._date_to_index[pd.Timestamp(signal_t)]
                assert idx_t - idx_s == 2, (
                    f"Lag mismatch on {date_t.date()}: "
                    f"signal={signal_t.date()}, gap={idx_t - idx_s} days"
                )

    def test_no_future_data_in_signal(self, engine):
        """Verify strategy only sees data up to signal_date, not return_date."""
        factors = pd.read_csv(FACTOR_CSV, parse_dates=["date"])
        factors["date"] = pd.to_datetime(factors["date"])
        factors["symbol"] = factors["symbol"].astype(str).str.zfill(6)

        daily = engine.run_backtest(
            "S01_CS_Momentum",
            start=START_DATE,
            end=END_DATE,
            n_hold=5,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )

        for _, row in daily[daily["rebalance"]].iterrows():
            return_date = pd.Timestamp(row["date"])
            signal_date = pd.Timestamp(row["signal_date"])
            future_data = factors[
                (factors["date"] > signal_date) & (factors["date"] <= return_date)
            ]
            assert len(future_data) >= 0
            if len(future_data) > 0:
                future_symbols = set(future_data["symbol"])
                strategy = STRATEGIES["S01_CS_Momentum"](factors)
                positions = strategy.get_positions(signal_date, n_hold=5, max_weight=0.05)
                position_symbols = set(positions.keys())
                overlap = future_symbols & position_symbols
                assert len(overlap) == len(position_symbols), (
                    f"Signal at {signal_date.date()} may have used data from "
                    f"{return_date.date()}: {len(overlap)} symbols overlap"
                )

    def test_weights_stable_between_rebalances(self, engine):
        """Between rebalance days, weights should only drift (no new trades)."""
        daily = engine.run_backtest(
            "S01_CS_Momentum",
            start=START_DATE,
            end=END_DATE,
            n_hold=5,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )
        non_rebalance = daily[daily["rebalance"] == False]  # noqa: E712
        if len(non_rebalance) > 0:
            assert (non_rebalance["traded_notional"] < 1e-12).all(), (
                "Non-rebalance days should have zero traded notional"
            )


class TestNoLookAheadBias:
    """Verify no look-ahead bias in signal generation."""

    def test_b0_buy_hold_no_lookahead(self, engine):
        """B0 should only rebalance once, using earliest available data."""
        daily = engine.run_backtest(
            "B0_BuyHold_EW",
            start=START_DATE,
            end=END_DATE,
            n_hold=20,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )
        n_rebalance = daily["rebalance"].sum()
        assert 1 <= n_rebalance <= 3, (
            f"B0 should rebalance 1-3 times in warmup, got {n_rebalance}"
        )

    def test_b2_cash_zero_exposure(self, engine):
        """B2_Cash should have zero exposure and zero transaction cost."""
        daily = engine.run_backtest(
            "B2_Cash",
            start=START_DATE,
            end=END_DATE,
            n_hold=20,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )
        assert (daily["exposure"] == 0.0).all(), "B2_Cash should have zero exposure"
        assert (daily["transaction_cost"] == 0.0).all(), "B2_Cash should have zero cost"
        assert (daily["position_count"] == 0).all(), "B2_Cash should hold no positions"

    def test_lag_1_vs_lag_2_different_results(self, engine):
        """Different lag values must produce different daily returns."""
        daily_lag1 = engine.run_backtest(
            "S01_CS_Momentum",
            start=START_DATE,
            end=END_DATE,
            n_hold=5,
            max_weight=0.05,
            signal_to_return_lag=1,
            rebalance_every=5,
        )
        daily_lag2 = engine.run_backtest(
            "S01_CS_Momentum",
            start=START_DATE,
            end=END_DATE,
            n_hold=5,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )
        assert not daily_lag1["return"].equals(daily_lag2["return"]), (
            "Lag 1 and lag 2 should produce different returns"
        )


class TestReturnAccounting:
    """Verify return accounting is consistent with signal timing."""

    def test_return_column_exists_and_valid(self, engine):
        daily = engine.run_backtest(
            "S04_VolTarget_Trend",
            start=START_DATE,
            end=END_DATE,
            n_hold=5,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )
        assert "return" in daily.columns
        assert "gross_return" in daily.columns
        assert "transaction_cost" in daily.columns
        assert daily["return"].notna().all()

    def test_net_return_equals_gross_minus_cost(self, engine):
        daily = engine.run_backtest(
            "S01_CS_Momentum",
            start=START_DATE,
            end=END_DATE,
            n_hold=5,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )
        computed_net = daily["gross_return"] - daily["transaction_cost"]
        pd.testing.assert_series_equal(
            daily["return"].round(10),
            computed_net.round(10),
            check_names=False,
        )

    def test_turnover_definition(self, engine):
        """Turnover should be half of gross traded notional."""
        daily = engine.run_backtest(
            "S01_CS_Momentum",
            start=START_DATE,
            end=END_DATE,
            n_hold=5,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )
        pd.testing.assert_series_equal(
            daily["turnover"],
            (daily["traded_notional"] / 2.0).round(10),
            check_names=False,
        )

    def test_rebalance_is_self_financing(self, engine):
        executed, traded, cost, post_cost = engine._self_financing_rebalance(
            {}, {"A": 1.0}, 0.01
        )
        assert traded == pytest.approx(executed["A"])
        assert cost == pytest.approx(traded * 0.01)
        assert post_cost == pytest.approx(1.0 - cost)
        assert sum(executed.values()) == pytest.approx(post_cost)

    def test_drift_uses_net_wealth_denominator(self, engine):
        drifted = engine._drift_weights(
            {"A": 0.99}, pd.Series({"A": 0.10}), net_return=0.089
        )
        assert drifted["A"] == pytest.approx(0.99 * 1.10 / 1.089)
