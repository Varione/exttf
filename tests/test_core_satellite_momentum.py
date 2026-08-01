"""Behavioral tests for the frozen C1 core/satellite momentum research path."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from otf_backtest_engine import OTFBacktestEngine, OrderSide
from otf_rotation.core_satellite_momentum import (
    CORE_WEIGHTS,
    SATELLITE_POOL,
    CoreSatelliteMomentumSignal,
    build_total_return_index,
    clone_rule_book_with_delay_override,
    estimate_total_return_covariance,
    select_top2_with_buffer,
    scale_satellite_weights_to_vol,
    sustained_turnover_excluding_initial,
)
from otf_rotation.schedule import build_quarter_end_schedule, build_signal_submit_map
from otf_trading_rules import ProductRuleBook
from run_core_satellite_momentum import (
    PUBLICATION_TIMING_STATUS,
    SIGNAL_PIT_TRUTH_STATUS,
    _c1_gate,
    publication_timing_audit,
)


def _growth_rows(
    dates: pd.DatetimeIndex,
    rates: dict[str, float],
    *,
    unit_nav: float = 1.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code, rate in rates.items():
        for date in dates:
            rows.append(
                {
                    "fund_code": code,
                    "nav_date": date,
                    "daily_growth_pct": rate,
                    # Deliberately unrelated to the total-return signal inputs.
                    "unit_nav": unit_nav,
                }
            )
    return pd.DataFrame(rows)


def _default_rates() -> dict[str, float]:
    return {
        **{code: 0.01 for code in CORE_WEIGHTS},
        "160706": 0.12,
        "000008": 0.08,
        "007466": 0.04,
        "050021": -0.02,
        "050025": 0.06,
        "000071": 0.03,
    }


def test_total_return_index_and_momentum_are_point_in_time_and_growth_based():
    dates = pd.bdate_range("2020-01-01", periods=260)
    frame = _growth_rows(dates, _default_rates())
    signal_date = dates[-1]
    index = build_total_return_index(frame)
    signal = CoreSatelliteMomentumSignal(frame)
    before = signal(signal_date)

    future = _growth_rows(
        pd.bdate_range(signal_date + pd.Timedelta(days=1), periods=5),
        {code: 50.0 for code in {**CORE_WEIGHTS, **{c: 0 for c in SATELLITE_POOL}}},
    )
    after = CoreSatelliteMomentumSignal(pd.concat([frame, future], ignore_index=True))(
        signal_date
    )
    pd.testing.assert_frame_equal(
        index.loc[:signal_date], build_total_return_index(pd.concat([frame, future])) .loc[:signal_date]
    )
    assert before == pytest.approx(after)
    audit = signal.last_signal_audit
    top = audit["evaluations_by_code"]["160706"]
    assert top["published_observations"] == 260
    assert top["total_return_126"] > 0
    assert top["total_return_252"] > 0
    assert top["ma200"] == pytest.approx(
        signal.total_return_index.loc[:signal_date, "160706"].tail(200).mean()
    )
    assert top["eligible"] is True


def test_top2_and_three_percentage_point_buffer_are_frozen():
    evaluations = {
        "a": {"eligible": True, "momentum_score": 0.50},
        "b": {"eligible": True, "momentum_score": 0.10},
        "c": {"eligible": True, "momentum_score": 0.08},
        "d": {"eligible": False, "momentum_score": 0.90},
    }
    selected, audit = select_top2_with_buffer(evaluations, ["c", "d"])
    assert selected == ["a", "c"]
    assert audit["c"]["retained_by_buffer"] is True
    assert audit["c"]["buffer_margin_pct_points"] == pytest.approx(-2.0)
    assert audit["b"]["rank"] == 2


def test_history_shortfall_moves_satellite_budget_to_cash_and_no_future_covariance():
    dates = pd.bdate_range("2020-01-01", periods=260)
    complete = _growth_rows(dates, _default_rates())
    short = complete[
        ~((complete["fund_code"] == "050021") & (complete["nav_date"] < dates[-100]))
    ]
    signal = CoreSatelliteMomentumSignal(short)
    target = signal(dates[-1])
    evaluation = signal.last_signal_audit["evaluations_by_code"]["050021"]
    assert evaluation["eligible"] is False
    assert "INSUFFICIENT_PUBLISHED_OBSERVATIONS_253" in evaluation["reason"]
    assert target.get("050021", 0.0) == 0.0
    assert signal.last_signal_audit["cash_transfer_reasons"]["050021"]

    covariance, audit = estimate_total_return_covariance(
        complete, list(CORE_WEIGHTS) + ["160706"], dates[-1], window=60
    )
    future = _growth_rows(
        pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=10),
        _default_rates(),
    )
    future.loc[future["fund_code"] == "160706", "daily_growth_pct"] = 10.0
    future_covariance, future_audit = estimate_total_return_covariance(
        pd.concat([complete, future], ignore_index=True),
        list(CORE_WEIGHTS) + ["160706"],
        dates[-1],
        window=60,
    )
    assert covariance is not None
    assert audit["common_observations"] == 60
    assert future_audit["common_observations"] == 60
    pd.testing.assert_frame_equal(covariance, future_covariance)


def test_volatility_scaling_is_bounded_and_hits_the_ten_percent_constraint():
    codes = ["001512", "000148", "000218", "260102", "160706", "050025"]
    rng = np.random.default_rng(7)
    daily = pd.DataFrame(rng.normal(0, 0.02, size=(60, len(codes))), columns=codes)
    covariance = daily.cov() * 252.0
    scale, predicted, audit = scale_satellite_weights_to_vol(
        CORE_WEIGHTS,
        {"160706": 0.25, "050025": 0.25},
        covariance,
        target_vol=0.10,
    )
    assert 0.0 <= scale <= 1.0
    assert predicted <= 10.0 + 1e-8
    assert audit["bisection_applied"] is True
    assert sum(CORE_WEIGHTS.values()) + 0.5 * scale <= 1.0 + 1e-12


def test_quarter_signal_submits_next_day_and_empty_signal_does_not_clear():
    dates = pd.bdate_range("2020-01-01", periods=520)
    signal_dates = build_quarter_end_schedule(dates, "2021-01-01", "2021-12-31")
    submit_map = build_signal_submit_map(dates, signal_dates, "2021-12-31")
    assert submit_map
    assert all(pd.Timestamp(submit) > pd.Timestamp(signal) for submit, signal in submit_map.items())

    frame = _growth_rows(dates, _default_rates())
    signal = CoreSatelliteMomentumSignal(frame)
    first = signal(signal_dates[0])
    assert first
    # A one-day, sub-5pp natural drift does not create a sparse target row.
    second = signal(signal_dates[1])
    assert second == {}
    assert signal.last_signal_audit["rebalance"] is False
    assert "NO_REBALANCE" in signal.last_signal_audit["decision"]


def test_five_percentage_point_boundary_is_inclusive():
    assert CoreSatelliteMomentumSignal.should_rebalance_for_drift(0.05) is True
    assert CoreSatelliteMomentumSignal.should_rebalance_for_drift(0.049999999) is False


def test_stress_overrides_actual_product_rule_delays_and_fees_on_orders():
    root = Path(__file__).resolve().parents[1]
    rules = ProductRuleBook.from_csv(root / "config/otf_product_rules.csv")
    stress_rules = clone_rule_book_with_delay_override(rules, ["050025"], increment=1)
    assert rules.rule_for("050025").subscription_confirmation_days == 2
    assert stress_rules.rule_for("050025").subscription_confirmation_days == 3
    assert stress_rules.rule_for("050025").redemption_settlement_days == 8
    assert stress_rules.subscription_fee_amount("050025", 100_000, 0.0) == pytest.approx(
        rules.subscription_fee_amount("050025", 100_000, 0.0)
    )

    engine = OTFBacktestEngine(
        db_path=root / "data/processed/otf_expanded.sqlite",
        initial_cash=1_000_000,
        product_rule_book=rules,
        strict_product_rules=True,
        minimum_trade_ratio=0.0,
    )
    # Keep enough cash headroom that doubling the fee does not change the
    # affordable principal, isolating the fee-rate stress in the order.
    targets = pd.DataFrame({"050025": [0.5]}, index=pd.to_datetime(["2021-01-04"]))
    kwargs = {"start": "2021-01-04", "end": "2021-01-20", "rebalance_every": 1}
    engine.run_backtest(targets, **kwargs)
    base_order = engine.order_audit_frame().iloc[0]
    # The engine's NAV cache is immutable; swapping the rule book between
    # sequential account runs exercises the exact same order state machine
    # without loading the 1.2M-row database three times in one test process.
    engine.product_rule_book = rules.scaled_fees(2.0)
    engine.run_backtest(targets, **kwargs)
    fee_order = engine.order_audit_frame().iloc[0]
    engine.product_rule_book = stress_rules
    engine.run_backtest(targets, **kwargs)
    delay_order = engine.order_audit_frame().iloc[0]
    assert fee_order["fee_paid"] == pytest.approx(base_order["fee_paid"] * 2.0)
    assert pd.Timestamp(delay_order["confirmation_date"]) > pd.Timestamp(base_order["confirmation_date"])


def test_sustained_turnover_excludes_initial_build_orders():
    daily = pd.DataFrame(
        {"date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-07-01", "2022-01-04"]), "equity": [100, 100, 100, 100]}
    )
    orders = pd.DataFrame(
        [
            {"signal_date": "2020-12-31", "confirmation_date": "2021-01-05", "status": "settled", "side": "subscribe", "filled_notional": 100},
            {"signal_date": "2021-06-30", "confirmation_date": "2021-07-01", "status": "settled", "side": "redeem", "filled_notional": 20},
        ]
    )
    result = sustained_turnover_excluding_initial(daily, orders, "2020-12-31")
    assert result["excluded_initial_order_count"] == 1
    assert result["max_annual_confirmed_turnover"] == pytest.approx(0.2)


def test_publication_timing_disclosure_fails_gate_without_masking_existing_failures():
    audit = publication_timing_audit()
    assert audit["status"] == PUBLICATION_TIMING_STATUS
    assert audit["signal_point_in_time_truth"] == SIGNAL_PIT_TRUTH_STATUS
    assert audit["publication_timestamp_available"] is False
    assert audit["nav_date_filter_is_verified_pit"] is False
    assert "050025" in audit["affected_or_potentially_affected_products"]

    metrics = {
        "net_cagr_pct": 7.0,
        "sharpe": 1.0,
        "mdd_pct": -8.0,
        "calmar": 1.0,
        "worst_two_year_cagr_pct": -1.0,
        "max_sustained_annual_confirmed_turnover": 2.0,
        "fee_reconciliation_passed": True,
        "subperiod_metrics": {
            "2021_2023": {"net_cagr_pct": 1.0, "sharpe": 0.6},
            "2024_2026": {"net_cagr_pct": 1.0, "sharpe": 0.6},
        },
    }
    gate = _c1_gate(
        metrics,
        {"net_cagr_pct": 5.0},
        {"net_cagr_pct": 7.0, "sharpe": 1.0, "mdd_pct": -8.0},
        {"net_cagr_pct": 6.9, "mdd_pct": -8.0},
        {"data_gate": True, "rules_gate": True, "mapping_gate": True},
    )
    assert gate["checks"]["publication_timing_gate"] is False
    assert gate["publication_timing_status"] == PUBLICATION_TIMING_STATUS
    assert gate["signal_point_in_time_truth"] == SIGNAL_PIT_TRUTH_STATUS
    assert {
        "net_cagr",
        "worst_two_year_cagr",
        "sustained_confirmed_turnover",
        "double_fee_net_cagr",
        "publication_timing_gate",
    }.issubset(set(gate["failed_checks"]))
