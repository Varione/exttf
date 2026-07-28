"""Tests for OTF (over-the-counter fund) NAV-based backtest engine.

Verifies:
1. NAV same-day execution prevention (T+1 minimum confirmation)
2. T+1 / T+3 confirmation rules
3. Fee calculations (subscription and redemption)
4. Cash conservation (available + frozen + positions = initial + returns)
5. Order lifecycle tracking
6. Redemption arrival delay
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from otf_backtest_engine import (
    PositionLot,
    OTFBacktestEngine,
    OTFOrder,
    OTFStrategySignal,
    OrderSide,
    OrderStatus,
)
from otf_trading_rules import (
    FundTradingRule,
    ProductRuleBook,
    RedemptionFeeTier,
    TradingRestrictionEvent,
)

TEST_DB = str(Path(__file__).resolve().parents[1] / "data" / "processed" / "otf.sqlite")


def _create_test_db(tmp_path: Path) -> str:
    """Create a minimal test database with known NAV data."""
    db_path = str(tmp_path / "test_otf.sqlite")
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE otf_fund_catalog (
            fund_code TEXT PRIMARY KEY,
            share_class TEXT,
            fund_name TEXT,
            asset_class TEXT,
            benchmark TEXT,
            inception_date TEXT,
            termination_date TEXT,
            source TEXT,
            fetched_at TEXT
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

    funds = [
        ("F001", "A", "Fund Alpha", "equity", "CSI300", "2020-01-02", "", "test", "2024-01-01"),
        ("F002", "A", "Fund Beta", "bond", "CSI_Bond", "2020-01-02", "", "test", "2024-01-01"),
        ("F003", "A", "Fund Gamma QDII", "QDII_equity", "NASDAQ", "2020-01-02", "", "test", "2024-01-01"),
    ]
    conn.executemany(
        "INSERT INTO otf_fund_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        funds,
    )

    dates = pd.bdate_range("2024-01-01", "2024-03-31")
    nav_rows = []
    for fund_number, (fund_code, _, _, _, _, _, _, _, _) in enumerate(funds):
        base_nav = 1.0
        total_return_factor = 1.0
        previous_published_nav = None
        for day_number, date in enumerate(dates):
            published_nav = round(base_nav, 6)
            daily_growth = (
                0.0
                if previous_published_nav is None
                else published_nav / previous_published_nav - 1.0
            )
            if previous_published_nav is not None:
                total_return_factor *= 1.0 + daily_growth
            nav_rows.append((
                fund_code,
                date.strftime("%Y-%m-%d"),
                published_nav,
                published_nav,
                daily_growth * 100.0,
                0.0,
                1.0,
                total_return_factor,
            ))
            previous_published_nav = published_nav
            daily_ret = 0.001 * (fund_number + 1) + 0.00001 * day_number
            base_nav *= (1 + daily_ret)

    conn.executemany(
        "INSERT INTO otf_fund_nav VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        nav_rows,
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture(scope="module")
def engine():
    """Production OTF engine for integration tests."""
    if not os.path.exists(TEST_DB):
        pytest.skip("OTF database not available")
    return OTFBacktestEngine(
        db_path=TEST_DB,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=1,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        initial_cash=1_000_000.0,
    )


@pytest.fixture
def test_db(tmp_path):
    """Minimal test database with deterministic NAV data."""
    return _create_test_db(tmp_path)


@pytest.fixture
def clean_engine(test_db):
    """OTF engine with test database."""
    return OTFBacktestEngine(
        db_path=test_db,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=1,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        initial_cash=1_000_000.0,
    )


class TestNAVSameDayExecution:
    """Verify NAV cannot be used for same-day execution."""

    def test_subscription_confirms_at_t_plus_1(self, clean_engine):
        """Subscription submitted at T should confirm at T+1 NAV."""
        engine = clean_engine
        first_date = engine._trading_dates[0]
        second_date = engine._trading_dates[1]

        fund_code = engine.available_fund_codes[0]
        nav_t = engine.get_nav(fund_code, 0)
        nav_t1 = engine.get_nav(fund_code, 1)

        order = engine.submit_order(
            order_id="test-001",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=first_date,
            submit_date=first_date,
            requested_amount=100_000.0,
            available_cash=1_000_000.0,
            current_positions={},
        )

        assert order is not None
        assert order.nav_at_submit == nav_t

        engine.confirm_order(order, 1)
        assert order.confirmed_nav == nav_t1
        assert order.confirmed_nav != nav_t

        expected_shares = 100_000.0 / nav_t1
        assert abs(order.shares_confirmed - expected_shares) < 1e-6

    def test_no_same_day_nav_execution(self, clean_engine):
        """Confirming at same index as submit should still use next day NAV."""
        engine = clean_engine
        fund_code = engine.available_fund_codes[0]

        order = engine.submit_order(
            order_id="test-002",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=engine._trading_dates[0],
            submit_date=engine._trading_dates[0],
            requested_amount=50_000.0,
            available_cash=1_000_000.0,
            current_positions={},
        )

        assert order is not None
        nav_submit = engine.get_nav(fund_code, 0)

        confirm_result = engine.confirm_order(order, 0)
        assert confirm_result is False
        assert order.status == OrderStatus.PENDING
        assert order.confirmed_nav is None

    def test_subscription_fee_deducted_before_shares(self, clean_engine):
        """Subscription fee is deducted from amount before share calculation."""
        engine = clean_engine
        fund_code = engine.available_fund_codes[0]

        order = engine.submit_order(
            order_id="test-003",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=engine._trading_dates[0],
            submit_date=engine._trading_dates[0],
            requested_amount=100_000.0,
            available_cash=1_000_000.0,
            current_positions={},
        )

        assert order is not None
        assert order.fee_paid == pytest.approx(100.0, abs=0.01)
        assert order.cash_frozen == pytest.approx(100_100.0, abs=0.01)


class TestConfirmationRules:
    """Verify T+1 and T+3 confirmation rules."""

    def test_domestic_t_plus_1(self, clean_engine):
        """Domestic fund confirms after 1 trading day."""
        engine = clean_engine
        domestic_fund = [f for f in engine.available_fund_codes if not engine.is_qdii(f)][0]

        conf_days = engine.get_confirmation_days(domestic_fund, OrderSide.SUBSCRIBE)
        assert conf_days == 1

    def test_qdii_t_plus_3(self, clean_engine):
        """QDII fund confirms after 3 trading days."""
        engine = clean_engine
        qdii_funds = [f for f in engine.available_fund_codes if engine.is_qdii(f)]
        if not qdii_funds:
            pytest.skip("No QDII funds in test data")

        fund_code = qdii_funds[0]
        conf_days_sub = engine.get_confirmation_days(fund_code, OrderSide.SUBSCRIBE)
        conf_days_red = engine.get_confirmation_days(fund_code, OrderSide.REDEEM)
        assert conf_days_sub == 3
        assert conf_days_red == 3

    def test_confirmation_delay_in_backtest(self, clean_engine):
        """Orders in backtest respect confirmation delays."""
        engine = clean_engine
        fund_code = engine.available_fund_codes[0]

        submit_idx = 5
        submit_date = engine._trading_dates[submit_idx]

        order = engine.submit_order(
            order_id="test-004",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=submit_date,
            submit_date=submit_date,
            requested_amount=100_000.0,
            available_cash=1_000_000.0,
            current_positions={},
        )

        assert order is not None
        assert order.status == OrderStatus.PENDING

        engine.confirm_order(order, submit_idx + 1)
        assert order.status == OrderStatus.CONFIRMED
        assert order.confirmation_date == engine._trading_dates[submit_idx + 1]


class TestFeeCalculations:
    """Verify subscription and redemption fee calculations."""

    def test_subscription_fee_rate(self, clean_engine):
        """Subscription fee = amount * rate."""
        engine = clean_engine
        fund_code = engine.available_fund_codes[0]

        order = engine.submit_order(
            order_id="test-fee-sub",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=engine._trading_dates[0],
            submit_date=engine._trading_dates[0],
            requested_amount=100_000.0,
            available_cash=1_000_000.0,
            current_positions={},
        )

        assert order is not None
        expected_fee = 100_000.0 * 0.001
        assert order.fee_paid == pytest.approx(expected_fee, abs=0.01)

    def test_redemption_fee_on_confirm(self, clean_engine):
        """Redemption fee calculated at confirmation NAV."""
        engine = clean_engine
        fund_code = engine.available_fund_codes[0]

        nav_submit = engine.get_nav(fund_code, 5)
        nav_confirm = engine.get_nav(fund_code, 6)

        amount = 50_000.0
        shares = amount / nav_submit

        order = engine.submit_order(
            order_id="test-fee-red",
            side=OrderSide.REDEEM,
            fund_code=fund_code,
            signal_date=engine._trading_dates[5],
            submit_date=engine._trading_dates[5],
            requested_amount=amount,
            available_cash=1_000_000.0,
            current_positions={fund_code: shares * 2},
        )

        assert order is not None
        engine.confirm_order(order, 6)

        expected_fee = shares * nav_confirm * 0.0015
        assert order.fee_paid == pytest.approx(expected_fee, abs=0.01)


class TestCashConservation:
    """Verify cash conservation throughout backtest."""

    def test_insufficient_cash_rejected(self, clean_engine):
        """Subscription exceeding available cash is rejected."""
        engine = clean_engine
        fund_code = engine.available_fund_codes[0]

        order = engine.submit_order(
            order_id="test-cash-reject",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=engine._trading_dates[0],
            submit_date=engine._trading_dates[0],
            requested_amount=2_000_000.0,
            available_cash=1_000_000.0,
            current_positions={},
        )

        assert order is None

    def test_insufficient_shares_rejected(self, clean_engine):
        """Redemption exceeding holdings is rejected."""
        engine = clean_engine
        fund_code = engine.available_fund_codes[0]

        nav = engine.get_nav(fund_code, 0)
        shares_held = 100.0

        order = engine.submit_order(
            order_id="test-shares-reject",
            side=OrderSide.REDEEM,
            fund_code=fund_code,
            signal_date=engine._trading_dates[0],
            submit_date=engine._trading_dates[0],
            requested_amount=nav * 200,
            available_cash=1_000_000.0,
            current_positions={fund_code: shares_held},
        )

        assert order is None

    def test_cash_frozen_during_pending(self, clean_engine):
        """Cash is frozen when subscription order is pending."""
        engine = clean_engine
        fund_code = engine.available_fund_codes[0]

        initial_cash = 1_000_000.0
        amount = 100_000.0
        fee = amount * 0.001

        order = engine.submit_order(
            order_id="test-frozen",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=engine._trading_dates[0],
            submit_date=engine._trading_dates[0],
            requested_amount=amount,
            available_cash=initial_cash,
            current_positions={},
        )

        assert order is not None
        remaining = initial_cash - (amount + fee)
        assert remaining == pytest.approx(899_900.0, abs=0.01)


class TestRedemptionDelay:
    """Verify redemption proceeds arrive with delay."""

    def test_redemption_arrival_delay(self, clean_engine):
        """Redemption cash arrives after settlement delay."""
        engine = clean_engine
        fund_code = engine.available_fund_codes[0]

        nav = engine.get_nav(fund_code, 10)
        shares = 1000.0

        order = engine.submit_order(
            order_id="test-red-delay",
            side=OrderSide.REDEEM,
            fund_code=fund_code,
            signal_date=engine._trading_dates[10],
            submit_date=engine._trading_dates[10],
            requested_amount=shares * nav,
            available_cash=0.0,
            current_positions={fund_code: shares},
        )

        assert order is not None

        engine.confirm_order(order, 11)
        assert order.status == OrderStatus.CONFIRMED
        assert order.redemption_arrival_date is None

        engine.settle_order(order, 12)
        assert order.status == OrderStatus.SETTLED
        assert order.redemption_arrival_date is not None


class TestBacktestIntegration:
    """Integration tests for full backtest run."""

    def test_equal_weight_backtest_runs(self, clean_engine):
        """Equal weight strategy produces valid results."""
        engine = clean_engine
        signal = OTFStrategySignal(engine)

        target_weights = signal.generate_target_weights(
            lambda d: signal.equal_weight_signal(),
            start="2024-01-01",
            end="2024-02-29",
        )

        daily = engine.run_backtest(
            target_weights,
            start="2024-01-01",
            end="2024-02-29",
            rebalance_every=5,
        )

        assert len(daily) > 0
        assert "daily_return" in daily.columns
        assert "available_cash" in daily.columns
        assert "frozen_cash" in daily.columns
        assert "position_count" in daily.columns
        assert not daily["daily_return"].isna().any()

    def test_backtest_returns_valid_metrics(self, clean_engine):
        """Backtest results produce valid performance metrics."""
        engine = clean_engine
        signal = OTFStrategySignal(engine)

        target_weights = signal.generate_target_weights(
            lambda d: signal.equal_weight_signal(),
            start="2024-01-01",
            end="2024-02-29",
        )

        daily = engine.run_backtest(
            target_weights,
            start="2024-01-01",
            end="2024-02-29",
            rebalance_every=5,
        )

        metrics = engine.calculate_metrics(daily)
        assert "CAGR%" in metrics
        assert "Sharpe" in metrics
        assert "Max_Drawdown%" in metrics
        assert isinstance(metrics["Sharpe"], float)
        net_wealth = (1.0 + daily["daily_return"]).prod()
        gross_wealth = (1.0 + daily["gross_return"]).prod()
        assert metrics["Transaction_Cost_Drag%"] == pytest.approx(
            (gross_wealth - net_wealth) * 100
        )
        assert "Cumulative_Cost_Ratio%" in metrics

    def test_momentum_signal_backtest(self, clean_engine):
        """Momentum strategy produces valid results."""
        engine = clean_engine
        signal = OTFStrategySignal(engine, lookback_days=10)

        mom_weights = signal.generate_target_weights(
            lambda d: signal.momentum_signal(d, n_hold=2),
            start="2024-01-15",
            end="2024-02-29",
        )

        mom_daily = engine.run_backtest(mom_weights, "2024-01-15", "2024-02-29", 5)
        assert len(mom_daily) > 0
        assert not mom_daily["daily_return"].isna().any()


class TestOrderLifecycle:
    """Verify complete order lifecycle from signal to settlement."""

    def test_subscription_full_lifecycle(self, clean_engine):
        """Subscription: submit -> confirm -> settle with correct NAVs."""
        engine = clean_engine
        fund_code = engine.available_fund_codes[0]

        nav_0 = engine.get_nav(fund_code, 0)
        nav_1 = engine.get_nav(fund_code, 1)

        order = engine.submit_order(
            order_id="lifecycle-sub",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=engine._trading_dates[0],
            submit_date=engine._trading_dates[0],
            requested_amount=10_000.0,
            available_cash=100_000.0,
            current_positions={},
        )

        assert order is not None
        assert order.status == OrderStatus.PENDING
        assert order.cash_frozen == pytest.approx(10_010.0, abs=0.01)

        engine.confirm_order(order, 1)
        assert order.status == OrderStatus.CONFIRMED
        assert order.confirmed_nav == nav_1
        expected_shares = 10_000.0 / nav_1
        assert order.shares_confirmed == pytest.approx(expected_shares, abs=1e-4)

        engine.settle_order(order, 1)
        assert order.status == OrderStatus.SETTLED

    def test_redemption_full_lifecycle(self, clean_engine):
        """Redemption: submit -> confirm -> settle with cash release."""
        engine = clean_engine
        fund_code = engine.available_fund_codes[0]

        nav_5 = engine.get_nav(fund_code, 5)
        nav_6 = engine.get_nav(fund_code, 6)

        shares = 1000.0
        amount = shares * nav_5

        order = engine.submit_order(
            order_id="lifecycle-red",
            side=OrderSide.REDEEM,
            fund_code=fund_code,
            signal_date=engine._trading_dates[5],
            submit_date=engine._trading_dates[5],
            requested_amount=amount,
            available_cash=0.0,
            current_positions={fund_code: shares},
        )

        assert order is not None
        assert order.status == OrderStatus.PENDING

        engine.confirm_order(order, 6)
        assert order.status == OrderStatus.CONFIRMED
        assert order.confirmed_nav == nav_6
        assert order.fee_paid > 0

        cash_flow = engine.settle_order(order, 7)
        assert order.status == OrderStatus.SETTLED
        assert cash_flow > 0

        expected_proceeds = shares * nav_6 - order.fee_paid
        assert abs(cash_flow - expected_proceeds) < 0.01


class TestAccountingCorrectness:
    """Regression tests for economically material accounting invariants."""

    def test_frozen_cash_remains_in_equity(self, clean_engine):
        signal = OTFStrategySignal(clean_engine)
        targets = signal.generate_target_weights(
            lambda d: {"F001": 0.5}, "2024-01-01", "2024-01-05"
        )
        daily = clean_engine.run_backtest(
            targets, "2024-01-01", "2024-01-05", rebalance_every=5
        )
        first = daily.iloc[0]
        assert first["frozen_cash"] > 0
        assert first["equity"] == pytest.approx(1_000_000.0, abs=0.02)
        assert first["daily_return"] == pytest.approx(0.0, abs=1e-12)

    def test_nav_change_is_reflected_in_daily_return(self, clean_engine):
        signal = OTFStrategySignal(clean_engine)
        targets = signal.generate_target_weights(
            lambda d: {"F001": 0.5}, "2024-01-01", "2024-01-12"
        )
        daily = clean_engine.run_backtest(
            targets, "2024-01-01", "2024-01-12", rebalance_every=20
        )
        assert (daily.loc[daily["position_count"] > 0, "daily_return"].abs() > 0).any()

    def test_dropped_target_is_redeemed(self, clean_engine):
        dates = clean_engine._trading_dates[:12]
        targets = pd.DataFrame(0.0, index=dates, columns=clean_engine.available_fund_codes)
        targets.loc[dates[0], "F001"] = 0.5
        targets.loc[dates[5], "F002"] = 0.5
        clean_engine.run_backtest(
            targets, str(dates[0].date()), str(dates[-1].date()), rebalance_every=5
        )
        redemptions = [
            order for order in clean_engine.last_orders
            if order.side == OrderSide.REDEEM and order.fund_code == "F001"
        ]
        assert redemptions, "A fund removed from target weights must be redeemed"

    def test_explicit_signal_date_is_preserved_on_orders(self, clean_engine):
        submit_date = clean_engine._trading_dates[2]
        source_signal_date = clean_engine._trading_dates[1]
        targets = pd.DataFrame(
            {"F001": [0.5]}, index=pd.DatetimeIndex([submit_date])
        )

        clean_engine.run_backtest(
            targets,
            str(submit_date.date()),
            str(clean_engine._trading_dates[6].date()),
            rebalance_every=1,
            signal_dates={submit_date: source_signal_date},
        )

        assert clean_engine.last_orders
        assert all(
            order.signal_date == source_signal_date
            for order in clean_engine.last_orders
        )
        assert all(order.submit_date == submit_date for order in clean_engine.last_orders)

    def test_redemption_proceeds_complete_deferred_subscription(self, clean_engine):
        dates = clean_engine._trading_dates[:12]
        targets = pd.DataFrame(
            0.0, index=pd.DatetimeIndex([dates[0], dates[5]]), columns=["F001", "F002"]
        )
        targets.loc[dates[0], "F001"] = 1.0
        targets.loc[dates[5], "F002"] = 1.0

        clean_engine.run_backtest(
            targets, str(dates[0].date()), str(dates[-1].date()), rebalance_every=1
        )

        old_fund_redemptions = [
            order
            for order in clean_engine.last_orders
            if order.side == OrderSide.REDEEM and order.fund_code == "F001"
        ]
        new_fund_subscriptions = [
            order
            for order in clean_engine.last_orders
            if order.side == OrderSide.SUBSCRIBE and order.fund_code == "F002"
        ]
        assert old_fund_redemptions
        assert new_fund_subscriptions
        assert new_fund_subscriptions[-1].submit_date > dates[5]
        assert new_fund_subscriptions[-1].signal_date == dates[5]

    def test_minimum_trade_ratio_skips_small_rebalance(self, test_db):
        engine = OTFBacktestEngine(
            test_db,
            subscription_fee_rate=0.0,
            redemption_fee_rate=0.0,
            minimum_trade_ratio=0.01,
        )
        dates = engine._trading_dates[:12]
        targets = pd.DataFrame(
            {"F001": [0.50, 0.505]},
            index=pd.DatetimeIndex([dates[0], dates[5]]),
        )

        engine.run_backtest(
            targets, str(dates[0].date()), str(dates[-1].date()), rebalance_every=1
        )

        subscriptions = [
            order for order in engine.last_orders if order.side == OrderSide.SUBSCRIBE
        ]
        assert len(subscriptions) == 1

    def test_fund_specific_fee_override(self, test_db):
        engine = OTFBacktestEngine(
            test_db,
            subscription_fee_rate=0.01,
            redemption_fee_rate=0.02,
            fund_subscription_fee_rates={"F001": 0.0},
            fund_redemption_fee_rates={"F001": 0.0},
        )
        assert engine.get_subscription_fee_rate("F001") == 0.0
        assert engine.get_redemption_fee_rate("F001") == 0.0
        assert engine.get_subscription_fee_rate("F002") == 0.01
        assert engine.get_redemption_fee_rate("F002") == 0.02


class TestProductLevelExecution:
    def test_fifo_lots_apply_holding_period_fee_tiers(self, clean_engine):
        engine = clean_engine
        code = "F001"
        engine.product_rule_book = ProductRuleBook(
            rules={code: FundTradingRule(code, 0.0, 1, 1, 3)},
            fee_tiers={
                code: [
                    RedemptionFeeTier(0, 7, 0.015),
                    RedemptionFeeTier(7, None, 0.0),
                ]
            },
        )
        submit_idx = 10
        submit_date = engine._trading_dates[submit_idx]
        nav = engine.get_nav(code, submit_idx)
        lots = {
            code: [
                PositionLot("old", submit_date - pd.Timedelta(days=10), 100.0),
                PositionLot("young", submit_date - pd.Timedelta(days=3), 100.0),
            ]
        }
        order = engine.submit_order(
            "fifo", OrderSide.REDEEM, code, submit_date, submit_date,
            150.0 * nav, 0.0, {code: 200.0}, position_lots=lots,
        )
        assert order is not None
        assert [item.lot_id for item in order.lot_allocations] == ["old", "young"]
        engine.confirm_order(order, submit_idx + 1)
        expected = 50.0 * engine.get_nav(code, submit_idx + 1) * 0.015
        assert order.fee_paid == pytest.approx(expected)

    def test_minimum_holding_period_blocks_immature_redemption(self, clean_engine):
        engine = clean_engine
        code = "F001"
        engine.product_rule_book = ProductRuleBook(
            rules={
                code: FundTradingRule(
                    code, 0.0, 1, 1, 3,
                    minimum_holding_calendar_days=7,
                )
            }
        )
        submit_idx = 10
        submit_date = engine._trading_dates[submit_idx]
        lots = {
            code: [PositionLot("young", submit_date - pd.Timedelta(days=3), 100.0)]
        }
        order = engine.submit_order(
            "immature", OrderSide.REDEEM, code, submit_date, submit_date,
            100.0 * engine.get_nav(code, submit_idx), 0.0,
            {code: 100.0}, position_lots=lots,
        )
        assert order is None
        assert engine._rejection_log[-1]["reason"] == (
            "INSUFFICIENT_MATURE_UNRESERVED_LOTS"
        )

    def test_suspension_and_subscription_limit_are_enforced(self, clean_engine):
        engine = clean_engine
        code = "F001"
        date = engine._trading_dates[5]
        event = TradingRestrictionEvent(
            code, date, date, False, True, 1000.0,
            "official temporary suspension", "notice",
        )
        engine.product_rule_book = ProductRuleBook(
            rules={code: FundTradingRule(code, 0.0, 1, 1, 1)},
            events={code: [event]},
        )
        order = engine.submit_order(
            "closed", OrderSide.SUBSCRIBE, code, date, date,
            500.0, 1000.0, {},
        )
        assert order is None
        assert engine._rejection_log[-1]["reason"] == "official temporary suspension"

    def test_order_audit_exposes_fifo_and_effective_fee(self, clean_engine):
        dates = clean_engine._trading_dates[:12]
        targets = pd.DataFrame(
            {"F001": [1.0, 0.0]}, index=[dates[0], dates[5]]
        )
        clean_engine.run_backtest(
            targets, str(dates[0].date()), str(dates[-1].date()), rebalance_every=1
        )
        audit = clean_engine.order_audit_frame()
        redemption = audit.loc[audit["side"] == "redeem"].iloc[0]
        assert redemption["fifo_lot_count"] == 1
        assert redemption["minimum_holding_days"] >= 0
        assert redemption["effective_fee_rate"] == pytest.approx(0.0015)

    def test_strict_rule_mode_blocks_uncovered_fund(self, test_db):
        engine = OTFBacktestEngine(
            test_db,
            product_rule_book=ProductRuleBook(),
            strict_product_rules=True,
        )
        date = engine._trading_dates[0]
        with pytest.raises(RuntimeError, match="MISSING_PRODUCT_RULE:F001"):
            engine.submit_order(
                "missing", OrderSide.SUBSCRIBE, "F001", date, date,
                100.0, 1000.0, {},
            )


class TestProductionData:
    """Tests against production OTF database."""

    def test_loads_funds(self, engine):
        """Engine loads all available funds."""
        assert len(engine.available_fund_codes) >= 10

    def test_nav_data_available(self, engine):
        """NAV data is available for each fund."""
        for fund_code in engine.available_fund_codes[:5]:
            fund_nav = engine._nav_df.loc[engine._nav_df["fund_code"] == fund_code]
            assert len(fund_nav) > 0
            nav_first = fund_nav["unit_nav"].iloc[0]
            assert nav_first > 0

    def test_equal_weight_runs_on_production(self, engine):
        """Equal weight strategy runs on production data."""
        signal = OTFStrategySignal(engine)

        target_weights = signal.generate_target_weights(
            lambda d: signal.equal_weight_signal(),
            start="2024-06-01",
            end="2024-07-31",
        )

        daily = engine.run_backtest(
            target_weights,
            start="2024-06-01",
            end="2024-07-31",
            rebalance_every=10,
        )

        assert len(daily) > 0
        metrics = engine.calculate_metrics(daily)
        assert "Sharpe" in metrics


class TestNetReturnIdentity:
    """Verify net return = gross return - transaction cost."""

    def test_return_identity(self, clean_engine):
        """daily_return + transaction_cost should equal gross_return."""
        engine = clean_engine
        signal = OTFStrategySignal(engine)

        target_weights = signal.generate_target_weights(
            lambda d: signal.equal_weight_signal(),
            start="2024-01-01",
            end="2024-02-29",
        )

        daily = engine.run_backtest(
            target_weights,
            start="2024-01-01",
            end="2024-02-29",
            rebalance_every=5,
        )

        computed_net = daily["gross_return"] - daily["transaction_cost"]
        np.testing.assert_allclose(
            daily["daily_return"].values,
            computed_net.values,
            atol=1e-10,
            err_msg="Net return identity violated",
        )
