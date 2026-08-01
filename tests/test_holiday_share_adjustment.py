"""Test share adjustment aggregation across holiday gaps."""

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
def holiday_gap_db(tmp_path):
    """DB with NAV on a non-execution date having share_adjustment_factor != 1."""
    db_path = tmp_path / "holiday_gap.sqlite"
    conn = sqlite3.connect(str(db_path))

    conn.execute("""
        CREATE TABLE otf_fund_catalog (
            fund_code TEXT PRIMARY KEY,
            share_class TEXT,
            fund_name TEXT,
            asset_class TEXT,
            benchmark TEXT,
            inception_date TEXT,
            termination_date TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE otf_fund_nav (
            fund_code TEXT,
            nav_date TEXT,
            unit_nav REAL,
            cumulative_nav REAL,
            daily_growth_pct REAL,
            distribution_per_share REAL,
            share_adjustment_factor REAL,
            total_return_factor REAL
        )
    """)

    conn.execute(
        "INSERT INTO otf_fund_catalog VALUES ('H001', 'A', 'Holiday Test Fund', 'equity', 'CSI300', '2024-01-01', '')"
    )

    # NAV dates: execution date T, then holiday H (not in calendar), then execution date T+1
    # On holiday H: NAV changes and share_adjustment_factor = 1.05 (simulating a split/dividend)
    # Use exact values for reconciliation: NAV(H) = NAV(T)/adj(H) so total return unchanged
    nav_h = 1.0 / 1.05
    daily_growth_t1 = (0.98 / nav_h * 1.0 - 1.0) * 100.0
    nav_rows = [
        # Date T: normal, NAV=1.0
        ("H001", "2024-06-28", 1.0, 1.0, 0.0, 0.0, 1.0, 1.0),
        # Holiday H: NAV drops but adj=1.05 means total return unchanged
        ("H001", "2024-06-29", nav_h, 1.0, 0.0, 0.0, 1.05, 1.0),
        # Date T+1: NAV grows from holiday NAV, adj=1.0
        ("H001", "2024-06-30", 0.98, 1.0, daily_growth_t1, 0.0, 1.0, 1.0),
    ]
    conn.executemany(
        "INSERT INTO otf_fund_nav VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        nav_rows,
    )
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def calendar_excluding_holiday():
    """Execution calendar with 2024-06-28 and 2024-06-30 but NOT 2024-06-29."""
    dates = pd.DatetimeIndex(["2024-06-28", "2024-06-30"])
    return ExecutionCalendar(
        dates=dates,
        path="test_calendar.csv",
        source="TEST",
        consensus_count=0,
        primary_source_max_date="2024-06-30",
        input_hashes={},
        extension_codes=(),
    )


class TestShareAdjustmentAggregation:
    """Verify share_adjustment_factor on non-execution dates is aggregated."""

    def test_holiday_adj_aggregated_to_next_execution(self, holiday_gap_db, calendar_excluding_holiday):
        engine = OTFBacktestEngine(
            db_path=holiday_gap_db,
            execution_calendar=calendar_excluding_holiday,
        )

        # Index 0 = 2024-06-28, Index 1 = 2024-06-30
        idx_t = engine._date_to_index[pd.Timestamp("2024-06-28")]
        idx_tp1 = engine._date_to_index[pd.Timestamp("2024-06-30")]

        # T should have adj=1.0 (its own factor)
        adj_t = engine.get_share_adjustment("H001", idx_t)
        assert abs(adj_t - 1.0) < 1e-9, f"T adjustment should be 1.0, got {adj_t}"

        # T+1 should have the holiday's factor aggregated (1.05 * 1.0 = 1.05)
        adj_tp1 = engine.get_share_adjustment("H001", idx_tp1)
        assert abs(adj_tp1 - 1.05) < 1e-9, f"T+1 adjustment should be 1.05 (aggregated), got {adj_tp1}"

    def test_valuation_nav_includes_holiday_change(self, holiday_gap_db, calendar_excluding_holiday):
        engine = OTFBacktestEngine(
            db_path=holiday_gap_db,
            execution_calendar=calendar_excluding_holiday,
        )

        idx_t = engine._date_to_index[pd.Timestamp("2024-06-28")]
        idx_tp1 = engine._date_to_index[pd.Timestamp("2024-06-30")]

        # Valuation NAV at T should be 1.0
        nav_t = engine.get_valuation_nav("H001", idx_t)
        assert abs(nav_t - 1.0) < 1e-6, f"NAV at T should be 1.0, got {nav_t}"

        # Valuation NAV at T+1 should reflect the holiday NAV (0.98) via ffill
        nav_tp1 = engine.get_valuation_nav("H001", idx_tp1)
        assert abs(nav_tp1 - 0.98) < 1e-6, f"NAV at T+1 should be 0.98 (holiday NAV), got {nav_tp1}"

    def test_accounting_with_holiday_adj(self, holiday_gap_db, calendar_excluding_holiday):
        """Verify positions are adjusted by aggregated factor on next execution date."""
        engine = OTFBacktestEngine(
            db_path=holiday_gap_db,
            execution_calendar=calendar_excluding_holiday,
            initial_cash=200_000.0,
            subscription_fee_rate=0.0,  # No fee to simplify test
        )

        fund_code = "H001"
        idx_t = engine._date_to_index[pd.Timestamp("2024-06-28")]
        idx_tp1 = engine._date_to_index[pd.Timestamp("2024-06-30")]

        # Subscribe on T at NAV=1.0, get 100_000 shares (fee=0)
        order = engine.submit_order(
            order_id="test-hol-adj",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=pd.Timestamp("2024-06-28"),
            submit_date=pd.Timestamp("2024-06-28"),
            requested_amount=100_000.0,
            available_cash=200_000.0,
            current_positions={},
        )
        assert order is not None

        # Confirm on T+1 (T+1 confirmation)
        engine.confirm_order(order, idx_tp1)
        assert order.shares_confirmed is not None

        # Shares should reflect the aggregated adjustment factor
        # NAV at confirmation = 0.98, so shares = 100_000 / 0.98 ≈ 102040.82
        expected_shares = 100_000.0 / 0.98
        assert abs(order.shares_confirmed - expected_shares) < 1e-3, (
            f"Shares should be ~{expected_shares:.2f}, got {order.shares_confirmed}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])