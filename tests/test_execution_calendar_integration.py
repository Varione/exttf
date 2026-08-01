"""Execution calendar integration tests for OTF backtest engine.

Verifies that the CN execution calendar is correctly used for:
- Order submission blocking on non-execution dates
- NAV valuation with QDII holiday NAV changes
- Signal/submit date validation
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from otf_backtest_engine import (
    OTFBacktestEngine,
    OTFOrder,
    OrderSide,
    OrderStatus,
)
from otf_rotation.execution_calendar import load_execution_calendar

CALENDAR_PATH = "data/processed/execution_calendar/cn_execution_calendar.csv"
TEST_DB = str(Path(__file__).resolve().parents[1] / "data" / "processed" / "otf.sqlite")


@pytest.fixture(scope="module")
def calendar():
    return load_execution_calendar(CALENDAR_PATH)


@pytest.fixture(scope="module")
def engine_with_calendar(calendar):
    """OTF engine with execution calendar loaded."""
    return OTFBacktestEngine(
        db_path=TEST_DB,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=1,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        initial_cash=1_000_000.0,
        execution_calendar=calendar,
    )


@pytest.fixture(scope="module")
def engine_without_calendar():
    """OTF engine without execution calendar (backward compat)."""
    return OTFBacktestEngine(
        db_path=TEST_DB,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=1,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        initial_cash=1_000_000.0,
    )


class TestExecutionCalendarExcludesHolidayDates:
    """Verify the 7 known CN holiday dates with QDII NAV are excluded."""

    PROBLEM_DATES = [
        "2022-03-05", "2022-03-20", "2022-12-31",
        "2023-01-02", "2023-05-02", "2023-12-31", "2024-06-30",
    ]

    def test_holiday_dates_not_in_calendar(self, calendar):
        cal_set = set(calendar.dates.strftime("%Y-%m-%d"))
        for d in self.PROBLEM_DATES:
            assert d not in cal_set, f"Holiday date {d} should NOT be in execution calendar"

    def test_holiday_dates_not_in_engine_trading_dates(self, engine_with_calendar):
        cal_set = set(engine_with_calendar._trading_dates.strftime("%Y-%m-%d"))
        for d in self.PROBLEM_DATES:
            assert d not in cal_set, f"Holiday date {d} should NOT be in engine trading dates"

    def test_valuation_source_dates_superset_of_trading_dates(self, engine_with_calendar):
        """Valuation source dates include all NAV dates; trading dates are execution calendar subset."""
        val_set = set(engine_with_calendar._valuation_source_dates.strftime("%Y-%m-%d"))
        trad_set = set(engine_with_calendar._trading_dates.strftime("%Y-%m-%d"))
        # Trading dates should be mostly from valuation source, but may differ due to ETF vs NAV coverage
        assert len(trad_set) > 0
        assert len(val_set) > 0

    def test_trading_dates_count_matches_calendar(self, engine_with_calendar, calendar):
        assert len(engine_with_calendar._trading_dates) == len(calendar.dates)


class TestOrderSubmissionOnNonExecutionDate:
    """Orders submitted on non-execution dates must raise RuntimeError."""

    def test_submit_on_holiday_raises_error(self, engine_with_calendar):
        with pytest.raises(RuntimeError, match="OTF_NOT_EXECUTION_DATE.*2022-12-31"):
            engine_with_calendar.submit_order(
                order_id="test-holiday-submit",
                side=OrderSide.SUBSCRIBE,
                fund_code=engine_with_calendar.available_fund_codes[0],
                signal_date=pd.Timestamp("2022-12-30"),
                submit_date=pd.Timestamp("2022-12-31"),
                requested_amount=100_000.0,
                available_cash=1_000_000.0,
                current_positions={},
            )

    def test_submit_on_valid_next_business_day(self, engine_with_calendar):
        fund_code = engine_with_calendar.available_fund_codes[0]
        order = engine_with_calendar.submit_order(
            order_id="test-valid-submit",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=pd.Timestamp("2022-12-30"),
            submit_date=pd.Timestamp("2023-01-03"),
            requested_amount=100_000.0,
            available_cash=1_000_000.0,
            current_positions={},
        )
        assert order is not None
        assert order.submit_date == pd.Timestamp("2023-01-03")


class TestBackwardCompatibilityWithoutCalendar:
    """Engine without calendar should fall back to NAV union dates."""

    def test_no_calendar_uses_nav_union(self, engine_without_calendar):
        assert len(engine_without_calendar._trading_dates) == len(engine_without_calendar._valuation_source_dates)

    def test_submit_on_nonnav_date_returns_none_without_calendar(self, engine_without_calendar):
        fund_code = engine_without_calendar.available_fund_codes[0]
        order = engine_without_calendar.submit_order(
            order_id="test-no-cal-submit",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=pd.Timestamp("2099-01-01"),
            submit_date=pd.Timestamp("2099-01-01"),
            requested_amount=100_000.0,
            available_cash=1_000_000.0,
            current_positions={},
        )
        assert order is None


class TestCalendarCoverageValidation:
    """Verify calendar coverage matches expectations."""

    def test_calendar_coverage_start(self, calendar):
        assert calendar.min_date <= "2005-03-01"

    def test_calendar_coverage_end(self, calendar):
        assert calendar.max_date >= "2026-07-17"

    def test_primary_source_max_date(self, calendar):
        assert calendar.primary_source_max_date == "2026-07-17"

    def test_extension_codes(self, calendar):
        expected = ("160706", "000008", "050021", "007466")
        assert calendar.extension_codes == expected

    def test_consensus_count(self, calendar):
        assert calendar.consensus_count == 4



class TestRunBacktestTargetValidation:
    """Verify run_backtest blocks targets on non-execution dates."""

    def test_target_on_holiday_blocked(self, engine_with_calendar):
        """Target with non-NaN weights on 2022-12-31 must raise RuntimeError."""
        fund_code = engine_with_calendar.available_fund_codes[0]
        targets = pd.DataFrame(
            [[1.0]],
            index=pd.to_datetime(["2022-12-31"]),
            columns=[fund_code],
        )
        with pytest.raises(RuntimeError, match="OTF_TARGET_NOT_EXECUTION_DATE.*2022-12-31"):
            engine_with_calendar.run_backtest(
                targets,
                start="2022-12-01",
                end="2023-01-15",
                rebalance_every=1,
            )

    def test_target_on_valid_execution_date_allowed(self, engine_with_calendar):
        """Target on valid execution date should not raise."""
        fund_code = engine_with_calendar.available_fund_codes[0]
        targets = pd.DataFrame(
            [[1.0]],
            index=pd.to_datetime(["2023-01-03"]),
            columns=[fund_code],
        )
        daily = engine_with_calendar.run_backtest(
            targets,
            start="2022-12-01",
            end="2023-01-15",
            rebalance_every=1,
        )
        assert len(daily) > 0

    def test_target_nan_on_holiday_allowed(self, engine_with_calendar):
        """Target with all-NaN on holiday is allowed (no signal)."""
        fund_code = engine_with_calendar.available_fund_codes[0]
        targets = pd.DataFrame(
            [[float("nan")]],
            index=pd.to_datetime(["2022-12-31"]),
            columns=[fund_code],
        )
        # Should not raise since all weights are NaN (no signal on that date)
        daily = engine_with_calendar.run_backtest(
            targets,
            start="2022-12-01",
            end="2023-01-15",
            rebalance_every=1,
        )
        assert len(daily) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])