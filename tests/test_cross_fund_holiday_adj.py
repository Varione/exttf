"""Test share adjustment aggregation when fund has NAV on holiday but NOT on next execution date."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from otf_backtest_engine import OTFBacktestEngine, OrderSide
from otf_rotation.execution_calendar import ExecutionCalendar


@pytest.fixture
def cross_fund_holiday_db(tmp_path):
    """DB where FundA has NAV on holiday but NOT next execution date; FundB makes that date exist."""
    db_path = tmp_path / "cross_fund.sqlite"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE otf_fund_catalog (
            fund_code TEXT PRIMARY KEY, share_class TEXT, fund_name TEXT,
            asset_class TEXT, benchmark TEXT, inception_date TEXT, termination_date TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE otf_fund_nav (
            fund_code TEXT, nav_date TEXT, unit_nav REAL, cumulative_nav REAL,
            daily_growth_pct REAL, distribution_per_share REAL,
            share_adjustment_factor REAL, total_return_factor REAL
        )
    """)

    conn.executemany(
        "INSERT INTO otf_fund_catalog VALUES (?,?,?,?,?,?,?)",
        [
            ("A001", "A", "Fund A", "equity", "CSI300", "2024-01-01", ""),
            ("B001", "A", "Fund B", "equity", "CSI300", "2024-01-01", ""),
        ],
    )

    # Execution calendar: 2024-06-28 (Fri), 2024-07-01 (Mon)
    # Saturday 2024-06-29 is a holiday in the calendar.
    # FundA: NAV on Fri(1.0), Sat(holiday, NAV=1/1.05, adj=1.05), NO NAV on Mon
    # FundB: NAV on Fri(1.0), NO NAV on Sat, NAV on Mon(1.02)
    nav_h = 1.0 / 1.05
    nav_rows = [
        # Friday: both funds publish
        ("A001", "2024-06-28", 1.0, 1.0, 0.0, 0.0, 1.0, 1.0),
        ("B001", "2024-06-28", 1.0, 1.0, 0.0, 0.0, 1.0, 1.0),
        # Saturday (holiday): only FundA publishes with adj=1.05
        ("A001", "2024-06-29", nav_h, 1.0, 0.0, 0.0, 1.05, 1.0),
        # Monday: only FundB publishes; FundA has NO raw NAV row here
        ("B001", "2024-07-01", 1.02, 1.0, 2.0, 0.0, 1.0, 1.0),
    ]
    conn.executemany(
        "INSERT INTO otf_fund_nav VALUES (?,?,?,?,?,?,?,?)",
        nav_rows,
    )
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def calendar_fri_mon():
    """Execution calendar with Fri and Mon only; Saturday is excluded."""
    dates = pd.DatetimeIndex(["2024-06-28", "2024-07-01"])
    return ExecutionCalendar(
        dates=dates,
        path="test_cal.csv",
        source="TEST",
        consensus_count=0,
        primary_source_max_date="2024-07-01",
        input_hashes={},
        extension_codes=(),
    )


class TestCrossFundHolidayAggregation:
    """Verify aggregation when fund has NAV on holiday but not on next execution date."""

    def test_friday_has_no_holiday_factor(self, cross_fund_holiday_db, calendar_fri_mon):
        """Friday's adjustment for FundA must be 1.0 (Saturday factor not yet visible)."""
        engine = OTFBacktestEngine(
            db_path=cross_fund_holiday_db,
            execution_calendar=calendar_fri_mon,
            subscription_fee_rate=0.0,
        )
        idx_fri = engine._date_to_index[pd.Timestamp("2024-06-28")]
        adj = engine.get_share_adjustment("A001", idx_fri)
        assert abs(adj - 1.0) < 1e-9, f"Fri adj should be 1.0, got {adj}"

    def test_monday_gets_saturday_factor_for_fund_without_raw_nav(self, cross_fund_holiday_db, calendar_fri_mon):
        """FundA has NO raw NAV on Monday but Saturday's factor must map to Monday via searchsorted."""
        engine = OTFBacktestEngine(
            db_path=cross_fund_holiday_db,
            execution_calendar=calendar_fri_mon,
            subscription_fee_rate=0.0,
        )
        idx_mon = engine._date_to_index[pd.Timestamp("2024-07-01")]
        adj = engine.get_share_adjustment("A001", idx_mon)
        assert abs(adj - 1.05) < 1e-9, f"Mon adj for A001 should be 1.05 (from Sat), got {adj}"

    def test_valuation_nav_on_monday_reflects_saturday_for_fund_without_raw_nav(self, cross_fund_holiday_db, calendar_fri_mon):
        """FundA valuation NAV on Monday should ffill Saturday's NAV."""
        engine = OTFBacktestEngine(
            db_path=cross_fund_holiday_db,
            execution_calendar=calendar_fri_mon,
            subscription_fee_rate=0.0,
        )
        nav_h = 1.0 / 1.05
        idx_mon = engine._date_to_index[pd.Timestamp("2024-07-01")]
        vnav = engine.get_valuation_nav("A001", idx_mon)
        assert abs(vnav - nav_h) < 1e-6, f"Mon valuation NAV for A001 should be {nav_h:.6f}, got {vnav}"

    def test_friday_cannot_see_saturday_change(self, cross_fund_holiday_db, calendar_fri_mon):
        """Friday valuation NAV must not reflect Saturday's change."""
        engine = OTFBacktestEngine(
            db_path=cross_fund_holiday_db,
            execution_calendar=calendar_fri_mon,
            subscription_fee_rate=0.0,
        )
        idx_fri = engine._date_to_index[pd.Timestamp("2024-06-28")]
        vnav = engine.get_valuation_nav("A001", idx_fri)
        assert abs(vnav - 1.0) < 1e-6, f"Fri valuation NAV should be 1.0, got {vnav}"

    def test_run_backtest_fundb_confirms_correctly(self, cross_fund_holiday_db, calendar_fri_mon):
        """Verify FundB (which has NAV on both Fri and Mon) confirms orders correctly."""
        engine = OTFBacktestEngine(
            db_path=cross_fund_holiday_db,
            execution_calendar=calendar_fri_mon,
            initial_cash=100_000.0,
            subscription_fee_rate=0.0,
        )

        # Subscribe to FundB on Friday at NAV=1.0
        targets = pd.DataFrame(
            [[1.0], [1.0]],
            index=pd.to_datetime(["2024-06-28", "2024-07-01"]),
            columns=["B001"],
        )
        daily = engine.run_backtest(targets, start="2024-06-28", end="2024-07-01", rebalance_every=1)

        # Monday: order confirmed at NAV=1.02; check equity reflects position value
        mon_row = daily[daily["date"] == "2024-07-01"].iloc[0]
        # Shares = 100_000 / 1.02, position value = shares * NAV = 100_000
        assert abs(mon_row["equity"] - 100_000.0) < 1, (
            f"Mon end_equity should be ~100k, got {mon_row['end_equity']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])