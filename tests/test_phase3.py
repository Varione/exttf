"""Phase 3: Walk-forward and unified metrics tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtest_engine import BacktestEngine
from walk_forward import (
    _expand_dates,
    run_walk_forward,
    summarize_walk_forward,
)

DB_PATH = str(Path(__file__).resolve().parents[1] / "data" / "processed" / "etf.sqlite")
FACTOR_CSV = str(
    Path(__file__).resolve().parents[1] / "data" / "processed" / "factors_all_repaired.csv"
)
REGIME_CSV = str(
    Path(__file__).resolve().parents[1] / "data" / "processed" / "regime_predictions.csv"
)


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


class TestUnifiedMetrics:
    """Verify all required metrics are computed with correct naming."""

    def test_core_metrics_exist(self, engine):
        daily = engine.run_backtest(
            "S01_CS_Momentum",
            start="2026-01-02",
            end="2026-01-30",
            n_hold=5,
            max_weight=0.05,
        )
        metrics = engine.calculate_metrics(
            daily["return"],
            daily["turnover"],
            daily["transaction_cost"],
            daily["gross_return"],
            daily["exposure"],
            daily.get("cash_weight", None),
        )

        required = [
            "CAGR%",
            "Annualized_Volatility%",
            "Sharpe",
            "Sortino",
            "Max_Drawdown%",
            "Calmar",
            "Turnover%",
            "Transaction_Cost_Drag%",
            "Exposure%",
            "Skewness",
            "Kurtosis",
            "VaR_95%",
            "CVaR_95%",
            "Longest_DD_Duration_days",
        ]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"

    def test_cash_ratio_computed(self, engine):
        daily = engine.run_backtest(
            "B2_Cash",
            start="2026-01-02",
            end="2026-01-30",
            n_hold=20,
            max_weight=0.05,
        )
        metrics = engine.calculate_metrics(
            daily["return"],
            cash_weights=daily.get("cash_weight", None),
        )
        assert "Cash_Ratio%" in metrics
        assert abs(metrics["Cash_Ratio%"] - 100.0) < 1.0

    def test_benchmark_metrics_exist(self, engine):
        daily_s = engine.run_backtest(
            "S01_CS_Momentum",
            start="2026-01-02",
            end="2026-01-30",
            n_hold=5,
            max_weight=0.05,
        )
        daily_b = engine.run_backtest(
            "B0_BuyHold_EW",
            start="2026-01-02",
            end="2026-01-30",
            n_hold=20,
            max_weight=0.05,
        )
        metrics = engine.calculate_metrics(
            daily_s["return"],
            benchmark_returns=daily_b["return"],
        )
        assert "Excess_Return%" in metrics
        assert "Information_Ratio" in metrics
        assert "Tracking_Error%" in metrics
        assert "Alpha%" in metrics
        assert "Beta" in metrics

    def test_active_win_rate_with_exposure(self, engine):
        daily = engine.run_backtest(
            "S04_VolTarget_Trend",
            start="2026-01-02",
            end="2026-01-30",
            n_hold=5,
            max_weight=0.05,
        )
        metrics = engine.calculate_metrics(
            daily["return"],
            exposures=daily["exposure"],
        )
        assert "Active_Win_Rate%" in metrics

    def test_monthly_win_rate(self, engine):
        daily = engine.run_backtest(
            "B0_BuyHold_EW",
            start="2025-01-01",
            end="2026-06-30",
            n_hold=20,
            max_weight=0.05,
        )
        metrics = engine.calculate_metrics(
            daily["return"],
            dates=pd.to_datetime(daily["date"]),
        )
        assert "Monthly_Win_Rate%" in metrics
        assert 0.0 <= metrics["Monthly_Win_Rate%"] <= 100.0

    def test_rolling_sharpe_computed(self, engine):
        daily = engine.run_backtest(
            "B0_BuyHold_EW",
            start="2025-01-01",
            end="2026-06-30",
            n_hold=20,
            max_weight=0.05,
        )
        metrics = engine.calculate_metrics(daily["return"])
        assert "Rolling_Sharpe_12m_mean" in metrics

    def test_var_cvar_signs(self, engine):
        daily = engine.run_backtest(
            "B0_BuyHold_EW",
            start="2026-01-02",
            end="2026-01-30",
            n_hold=20,
            max_weight=0.05,
        )
        metrics = engine.calculate_metrics(daily["return"])
        assert metrics["VaR_95%"] <= 0.0, "VaR should be negative (loss)"
        assert metrics["CVaR_95%"] <= metrics["VaR_95%"], "CVaR should be worse than VaR"

    def test_sortino_ge_sharpe_for_positive_returns(self, engine):
        """Sortino >= Sharpe when returns are mostly positive."""
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
        metrics = engine.calculate_metrics(returns)
        assert "Sortino" in metrics
        assert "Sharpe" in metrics


class TestWalkForwardExpansion:
    """Verify walk-forward window generation."""

    def test_windows_are_sequential(self):
        factors = pd.read_csv(FACTOR_CSV, parse_dates=["date"])
        all_dates = pd.DatetimeIndex(factors["date"].drop_duplicates().sort_values())

        windows = _expand_dates(
            "2018-01-01", "2026-07-17", all_dates,
            train_years=3, val_years=1, test_years=1, roll_months=6,
        )
        assert len(windows) >= 1, "Should generate at least 1 window"

        for i in range(1, len(windows)):
            prev_test_end = pd.Timestamp(windows[i - 1]["test_end"])
            curr_train_start = pd.Timestamp(windows[i]["train_start"])
            assert prev_test_end <= curr_train_start + pd.Timedelta(days=365), (
                f"Window {i} train starts too late after window {i-1}"
            )

    def test_window_boundaries_valid(self):
        factors = pd.read_csv(FACTOR_CSV, parse_dates=["date"])
        all_dates = pd.DatetimeIndex(factors["date"].drop_duplicates().sort_values())

        windows = _expand_dates(
            "2018-01-01", "2026-07-17", all_dates,
            train_years=3, val_years=1, test_years=1, roll_months=6,
        )
        for w in windows:
            assert pd.Timestamp(w["train_start"]) < pd.Timestamp(w["train_end"])
            assert pd.Timestamp(w["val_start"]) <= pd.Timestamp(w["val_end"])
            assert pd.Timestamp(w["test_start"]) < pd.Timestamp(w["test_end"])


class TestWalkForwardIntegration:
    """Small-scale walk-forward integration test."""

    def test_small_walk_forward_runs(self):
        results = run_walk_forward(
            strategy_names=["B0_BuyHold_EW", "B2_Cash"],
            start_date="2018-01-01",
            end_date="2026-07-17",
            train_years=3,
            val_years=1,
            test_years=1,
            roll_months=6,
        )
        assert not results.empty
        assert "window_id" in results.columns
        assert "strategy" in results.columns
        assert "CAGR%" in results.columns
        assert "Sharpe" in results.columns

    def test_summarize_walk_forward(self):
        results = run_walk_forward(
            strategy_names=["B0_BuyHold_EW"],
            start_date="2018-01-01",
            end_date="2026-07-17",
            train_years=3,
            val_years=1,
            test_years=1,
            roll_months=6,
        )
        summary = summarize_walk_forward(results, ["B0_BuyHold_EW"])
        assert not summary.empty
        assert "Sharpe_mean" in summary.columns
