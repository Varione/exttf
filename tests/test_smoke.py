"""Smoke test: full pipeline with a small time window."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest_engine import BacktestEngine
from data_loader import load_price_series


DB_PATH = str(Path(__file__).resolve().parents[1] / "data" / "processed" / "etf.sqlite")
FACTOR_CSV = str(
    Path(__file__).resolve().parents[1] / "data" / "processed" / "factors_all_repaired.csv"
)
REGIME_CSV = str(
    Path(__file__).resolve().parents[1] / "data" / "processed" / "regime_predictions.csv"
)

# 2026-01 small window (~22 trading days)
START_DATE = "2026-01-02"
END_DATE = "2026-01-30"


class TestSmokeDataLoading:
    """Verify data loads without error."""

    def test_price_series_loads(self):
        df = load_price_series(
            DB_PATH,
            data_mode="etf",
            price_mode="total_return_proxy",
        )
        df["date"] = pd.to_datetime(df["date"])
        window = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
        assert len(window) > 0
        assert "date" in df.columns
        assert "symbol" in df.columns
        assert "price" in df.columns

    def test_factors_load(self):
        df = pd.read_csv(FACTOR_CSV, parse_dates=["date"])
        window = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
        assert len(window) > 0
        assert "pit_eligible" in df.columns


class TestSmokeBacktest:
    """Verify backtest produces valid output."""

    def test_buy_hold_runs(self):
        engine = BacktestEngine(
            factor_path=FACTOR_CSV,
            regime_path=REGIME_CSV,
            db_path=DB_PATH,
            data_mode="etf",
            price_mode="total_return_proxy",
            fee_rate_per_side=0.0003,
            slippage_rate_per_side=0.0002,
            require_pit=True,
        )
        daily = engine.run_backtest(
            "B0_BuyHold_EW",
            start=START_DATE,
            end=END_DATE,
            n_hold=20,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )
        assert len(daily) > 0
        assert "return" in daily.columns
        assert "exposure" in daily.columns
        assert "transaction_cost" in daily.columns

    def test_strategy_signal_runs(self):
        engine = BacktestEngine(
            factor_path=FACTOR_CSV,
            regime_path=REGIME_CSV,
            db_path=DB_PATH,
            data_mode="etf",
            price_mode="total_return_proxy",
            fee_rate_per_side=0.0003,
            slippage_rate_per_side=0.0002,
            require_pit=True,
        )
        daily = engine.run_backtest(
            "S01_CS_Momentum",
            start=START_DATE,
            end=END_DATE,
            n_hold=20,
            max_weight=0.05,
            signal_to_return_lag=2,
            rebalance_every=5,
        )
        assert len(daily) > 0
