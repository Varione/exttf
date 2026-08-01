"""Behavior-level regressions for the frozen continuous OOS runner."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from otf_backtest_engine import OTFBacktestEngine
from otf_rotation.artifact_validation import (
    CSV_SCHEMAS,
    REQUIRED_ARTIFACTS,
    validate_artifact_bundle,
)
from otf_rotation.experiment_artifacts import sha256_file, write_manifest
from run_b1_b2_b3_walkforward import (
    ACCOUNT_MODE,
    build_static_data_gates,
    build_fee_reconciliation,
    compute_metrics,
    compute_turnover,
    derive_research_status,
    make_s1_signal,
    metrics_gate,
)
from run_s1_experiment import StateRotationSignal


def _daily() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]),
            "equity": [1_000_000.0, 995_000.0, 1_005_000.0],
            "daily_return": [0.0, -0.005, 0.01005],
            "gross_return": [0.0, -0.004, 0.011],
            "subscription_fee_amount": [0.0, 10.0, 0.0],
            "redemption_fee_amount": [0.0, 0.0, 0.0],
            "total_fee_amount": [0.0, 10.0, 0.0],
        }
    )


def _write_valid_b2_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "B2_Static_EW_4Asset"
    bundle.mkdir(parents=True)
    date = "2021-01-04"
    csv_frames = {
        "daily_account.csv": pd.DataFrame([{
            "date": date, "equity": 1_000_000.0, "daily_return": 0.0,
            "gross_return": 0.0, "total_fee_amount": 1.0,
        }]),
        "daily_returns.csv": pd.DataFrame([{
            "date": date, "equity": 1_000_000.0, "daily_return": 0.0,
            "gross_return": 0.0,
        }]),
        "target_weights.csv": pd.DataFrame([{"date": date, "F001": 1.0}]),
        "actual_weights.csv": pd.DataFrame([{"date": date, "F001": 1.0}]),
        "orders.csv": pd.DataFrame([{
            "order_id": "ord-1", "fund_code": "F001", "side": "subscribe",
            "status": "settled", "signal_date": date, "submit_date": date,
            "confirmation_date": date, "redemption_arrival_date": date,
            "requested_amount": 1_000.0, "cash_frozen": 1_001.0,
            "confirmed_nav": 1.0, "shares_confirmed": 1_000.0,
            "filled_notional": 1_000.0, "settled_cash_notional": 1_001.0,
            "fee_paid": 1.0, "effective_fee_rate": 0.001,
        }]),
        "order_rejections.csv": pd.DataFrame(columns=CSV_SCHEMAS["order_rejections.csv"]),
        "position_lots.csv": pd.DataFrame(columns=CSV_SCHEMAS["position_lots.csv"]),
        "fees.csv": pd.DataFrame([{
            "date": date, "daily_subscription_fee_amount": 1.0,
            "daily_redemption_fee_amount": 0.0, "daily_total_fee_amount": 1.0,
            "order_subscription_fee_amount": 1.0,
            "order_redemption_fee_amount": 0.0, "order_total_fee_amount": 1.0,
            "subscription_fee_delta": 0.0, "redemption_fee_delta": 0.0,
            "total_fee_delta": 0.0, "reconciled": True,
        }]),
        "fee_reconciliation_orders.csv": pd.DataFrame(columns=CSV_SCHEMAS["fee_reconciliation_orders.csv"]),
        "turnover.csv": pd.DataFrame([{
            "date": date, "buy_notional": 1_000.0, "sell_notional": 0.0,
            "gross_traded_notional": 1_000.0, "bilateral_turnover": 0.001,
            "submitted_buy_notional": 1_000.0, "submitted_sell_notional": 0.0,
            "submitted_gross_notional": 1_000.0,
            "submitted_bilateral_turnover": 0.001,
            "settled_cash_buy_notional": 1_001.0,
            "settled_cash_sell_notional": 0.0,
            "settled_cash_gross_notional": 1_001.0,
            "settled_cash_turnover": 0.001001, "pending_requested_amount": 0.0,
        }]),
    }
    for filename, frame in csv_frames.items():
        frame.to_csv(bundle / filename, index=False, encoding="utf-8-sig")
    for filename, columns in CSV_SCHEMAS.items():
        if filename not in csv_frames:
            pd.DataFrame(columns=columns).to_csv(
                bundle / filename, index=False, encoding="utf-8-sig"
            )

    db_path = tmp_path / "db.bin"
    rules_path = tmp_path / "rules.csv"
    mapping_path = tmp_path / "mapping.csv"
    config_source = tmp_path / "config_source.json"
    for path in (db_path, rules_path, mapping_path):
        path.write_text("test-input", encoding="utf-8")
    config_source.write_text("{}", encoding="utf-8")
    config = {
        "run_id": "test_run",
        "account_mode": ACCOUNT_MODE,
        "requested_oos_period": ["2021-01-01", "2021-01-04"],
        "actual_oos_period": [date, date],
        "config_source_path": str(config_source),
    }
    (bundle / "config_snapshot.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    input_hashes = {
        "db_sha256": sha256_file(str(db_path)),
        "rules_sha256": sha256_file(str(rules_path)),
        "mapping_sha256": sha256_file(str(mapping_path)),
        "config_sha256": sha256_file(str(config_source)),
    }
    (bundle / "input_hashes.json").write_text(
        json.dumps(input_hashes, indent=2), encoding="utf-8"
    )
    metrics = {
        "account_mode": ACCOUNT_MODE, "n_days": 1,
        "oos_start": date, "oos_end": date, "net_cagr_pct": 0.0,
        "gross_cagr_pct": 0.0, "annualized_cost_drag_pct": 0.0,
        "sharpe": 0.0, "mdd_pct": 0.0, "total_fee_amount": 1.0,
        "max_annual_bilateral_turnover": 0.001,
        "steady_state_max_annual_bilateral_turnover": 0.001,
        "fee_reconciliation_passed": True,
        "artifact_schema_gate": True, "artifact_content_gate": True,
    }
    gate = {
        "gate_passed": True,
        "checks": {"artifact_schema": True, "artifact_content": True},
        "failed_checks": [],
    }
    (bundle / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (bundle / "gate_result.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (bundle / "parameter_freeze.json").write_text(
        json.dumps({
            "mode": ACCOUNT_MODE, "strategy": "B2_Static_EW_4Asset",
            "parameter_freeze_id": "freeze-test",
        }, indent=2), encoding="utf-8"
    )
    write_manifest(
        run_dir=str(bundle), strategy_name="B2_Static_EW_4Asset", config=config,
        db_path=str(db_path), rules_path=str(rules_path),
        exposure_mapping_path=str(mapping_path), metrics=metrics, gate_result=gate,
    )
    return bundle


def test_confirmed_turnover_excludes_pending_and_uses_confirmation_notional():
    orders = pd.DataFrame(
        [
            {
                "order_id": "confirmed",
                "side": "subscribe",
                "status": "settled",
                "submit_date": "2021-01-04",
                "confirmation_date": "2021-01-05",
                "requested_amount": 900_000.0,
                "filled_notional": 800_000.0,
                "settled_cash_notional": 800_800.0,
                "fee_paid": 800.0,
            },
            {
                "order_id": "pending",
                "side": "subscribe",
                "status": "pending",
                "submit_date": "2021-01-06",
                "confirmation_date": None,
                "requested_amount": 50_000.0,
                "filled_notional": 0.0,
                "settled_cash_notional": 0.0,
                "fee_paid": 0.0,
            },
        ]
    )
    turnover = compute_turnover(_daily(), orders)
    confirmed_row = turnover.loc[turnover["date"] == pd.Timestamp("2021-01-05")].iloc[0]
    assert confirmed_row["buy_notional"] == 800_000.0
    assert confirmed_row["submitted_buy_notional"] == 0.0
    assert turnover["buy_notional"].sum() == 800_000.0
    assert turnover["pending_requested_amount"].sum() == 50_000.0


def test_confirmed_redemption_turnover_uses_shares_times_confirmation_nav():
    orders = pd.DataFrame(
        [
            {
                "order_id": "redeem-confirmed",
                "side": "redeem",
                "status": "confirmed",
                "submit_date": "2021-01-04",
                "confirmation_date": "2021-01-05",
                "requested_amount": 900.0,
                "confirmed_nav": 12.0,
                "shares_confirmed": 100.0,
                "filled_notional": 1_200.0,
                "settled_cash_notional": 0.0,
            },
            {
                "order_id": "redeem-cancelled",
                "side": "redeem",
                "status": "cancelled",
                "submit_date": "2021-01-05",
                "confirmation_date": None,
                "requested_amount": 1_000.0,
                "confirmed_nav": None,
                "shares_confirmed": None,
                "filled_notional": 0.0,
                "settled_cash_notional": 0.0,
            },
        ]
    )
    turnover = compute_turnover(_daily(), orders)
    confirmed = turnover.loc[turnover["date"] == pd.Timestamp("2021-01-05")].iloc[0]
    assert confirmed["sell_notional"] == pytest.approx(1_200.0)
    assert confirmed["bilateral_turnover"] == pytest.approx(1_200.0 / 995_000.0)
    assert turnover["sell_notional"].sum() == pytest.approx(1_200.0)


def test_fee_reconciliation_matches_daily_account_and_order_fees():
    orders = pd.DataFrame(
        [
            {
                "order_id": "confirmed",
                "side": "subscribe",
                "status": "settled",
                "confirmation_date": "2021-01-05",
                "fee_paid": 10.0,
                "requested_amount": 10_000.0,
            }
        ]
    )
    reconciliation, order_audit, summary = build_fee_reconciliation(_daily(), orders)
    assert summary["passed"] is True
    assert summary["order_total_fee_amount"] == 10.0
    assert bool(order_audit.iloc[0]["reconciled_in_daily_fees"])
    assert bool(reconciliation.loc[reconciliation["date"] == pd.Timestamp("2021-01-05"), "reconciled"].iloc[0])


def test_fee_reconciliation_rejects_aggregate_rounding_drift():
    daily = _daily().copy()
    daily["total_fee_amount"] = [0.01, 0.01, 0.0]
    daily["subscription_fee_amount"] = daily["total_fee_amount"]
    orders = pd.DataFrame(
        columns=["order_id", "side", "status", "confirmation_date", "fee_paid", "requested_amount"]
    )
    _, _, summary = build_fee_reconciliation(daily, orders)
    assert summary["max_abs_delta"] == pytest.approx(0.01)
    assert summary["aggregate_abs_delta"] == pytest.approx(0.02)
    assert summary["passed"] is False


def test_continuous_metrics_have_frozen_account_mode():
    metrics = compute_metrics(_daily(), pd.DataFrame())
    assert metrics["account_mode"] == ACCOUNT_MODE
    assert metrics["oos_start"] == "2021-01-04"
    assert metrics["oos_end"] == "2021-01-06"


def test_gate_cannot_pass_without_artifact_content_checks():
    metrics = {
        "sharpe": 1.0,
        "mdd_pct": -1.0,
        "net_cagr_pct": 2.0,
        "annualized_cost_drag_pct": 0.1,
        "max_annual_bilateral_turnover": 0.1,
        "steady_state_max_annual_bilateral_turnover": 0.1,
        "rolling_two_year_positive_ratio_pct": 80.0,
        "worst_two_year_cagr_pct": 1.0,
        "confirmed_turnover_gate": True,
        "fee_reconciliation_passed": True,
    }
    gate = metrics_gate(
        metrics,
        {"data_gate": True, "rules_gate": True, "mapping_gate": True},
    )
    assert gate["gate_passed"] is False
    assert "artifact_schema" in gate["failed_checks"]
    assert "artifact_content" in gate["failed_checks"]


def test_any_failed_gate_cannot_become_paper_trade_candidate():
    gate = {
        "B2_Static_EW_4Asset": {
            "gate_passed": True,
            "failed_checks": [],
        },
        "S1_State_Rotation_Fixed": {
            "gate_passed": False,
            "failed_checks": ["artifact_content"],
        },
    }
    status, failed_phases = derive_research_status(
        gate, historical_rules_passed=True, test_result={"all_passed": True}
    )
    assert status == "OOS_GATE_FAILED"
    assert "COMPLETE_GATE_IMPLEMENTATION" in failed_phases


def test_historical_truth_gate_is_independent_from_conservative_candidate_gate():
    fake_engine = SimpleNamespace(
        data_quality={
            "row_count": 1,
            "duplicate_keys": 0,
            "max_total_return_reconciliation_error": 0.0,
        }
    )
    facts = {
        "input_hashes": {},
        "rule_counts_by_status": {},
        "mapping_counts_by_status": {},
    }
    static_gates = build_static_data_gates(fake_engine, facts, {"160706"})
    assert static_gates["historical_rules_gate"] is False
    metrics = {
        "sharpe": 1.0,
        "mdd_pct": -1.0,
        "net_cagr_pct": 2.0,
        "annualized_cost_drag_pct": 0.1,
        "max_annual_bilateral_turnover": 0.1,
        "steady_state_max_annual_bilateral_turnover": 0.1,
        "rolling_two_year_positive_ratio_pct": 80.0,
        "worst_two_year_cagr_pct": 1.0,
        "confirmed_turnover_gate": True,
        "fee_reconciliation_passed": True,
        "artifact_schema_gate": True,
        "artifact_content_gate": True,
    }
    gate = metrics_gate(metrics, static_gates)
    assert gate["historical_truth_gate"] is False
    assert gate["historical_truth_status"] == "NOT_ESTABLISHED"
    assert gate["candidate_gate_passed"] is True
    assert gate["gate_passed"] is True
    assert "historical_rules_gate" not in gate["failed_checks"]

    status, failed_phases = derive_research_status(
        {"B2_Static_EW_4Asset": gate},
        historical_rules_passed=False,
        test_result={"all_passed": True},
    )
    assert status == "PAPER_TRADE_CANDIDATE"
    assert "HISTORICAL_RULE_TIME_GATE" not in failed_phases

    # A real candidate Gate failure remains decisive even though historical
    # truth is disclosed independently.
    metrics["confirmed_turnover_gate"] = False
    failed_candidate_gate = metrics_gate(metrics, static_gates)
    assert failed_candidate_gate["historical_truth_gate"] is False
    assert failed_candidate_gate["gate_passed"] is False
    assert "confirmed_turnover" in failed_candidate_gate["failed_checks"]
    status, _ = derive_research_status(
        {"B2_Static_EW_4Asset": failed_candidate_gate},
        historical_rules_passed=False,
        test_result={"all_passed": True},
    )
    assert status == "OOS_GATE_FAILED"


def _create_cross_year_db(tmp_path: Path) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "cross_year.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE otf_fund_catalog (
                fund_code TEXT PRIMARY KEY, share_class TEXT, fund_name TEXT,
                asset_class TEXT, benchmark TEXT, inception_date TEXT,
                termination_date TEXT, source TEXT, fetched_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE otf_fund_nav (
                fund_code TEXT, nav_date TEXT, unit_nav REAL, cumulative_nav REAL,
                daily_growth_pct REAL, distribution_per_share REAL,
                share_adjustment_factor REAL, total_return_factor REAL
            )
            """
        )
        funds = [
            ("F001", "A", "Domestic Alpha", "equity", "CSI300", "2020-01-01", "", "test", "2021-01-01"),
            ("F003", "A", "Overseas QDII", "QDII_equity", "NASDAQ", "2020-01-01", "", "test", "2021-01-01"),
        ]
        conn.executemany("INSERT INTO otf_fund_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", funds)
        dates = pd.bdate_range("2020-12-21", "2021-01-12")
        nav_rows = []
        for fund_index, fund in enumerate(funds):
            previous = None
            total_factor = 1.0
            nav = 1.0
            for day_index, date in enumerate(dates):
                if day_index:
                    nav *= 1.0 + (0.001 * (fund_index + 1))
                    total_factor *= 1.0 + (0.001 * (fund_index + 1))
                growth = 0.0 if previous is None else nav / previous - 1.0
                nav_rows.append(
                    (
                        fund[0], date.strftime("%Y-%m-%d"), nav, nav,
                        growth * 100.0, 0.0, 1.0, total_factor,
                    )
                )
                previous = nav
        conn.executemany("INSERT INTO otf_fund_nav VALUES (?, ?, ?, ?, ?, ?, ?, ?)", nav_rows)
        conn.commit()
    return str(db_path)


def test_continuous_account_carries_fifo_pending_and_receivable_across_year(tmp_path):
    engine = OTFBacktestEngine(
        db_path=_create_cross_year_db(tmp_path),
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=3,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        initial_cash=1_000_000.0,
    )
    targets = pd.DataFrame(
        [
            {"F001": 0.30, "F003": 0.0},
            {"F001": 0.60, "F003": 0.0},
            {"F001": 0.0, "F003": 0.30},
        ],
        index=pd.to_datetime(["2020-12-22", "2020-12-24", "2020-12-30"]),
    )
    continuous = engine.run_backtest(
        targets, start="2020-12-21", end="2021-01-06", rebalance_every=1
    )
    year_end = continuous.loc[continuous["date"] == pd.Timestamp("2020-12-31")].iloc[0]
    new_year = continuous.loc[continuous["date"] == pd.Timestamp("2021-01-04")].iloc[0]
    # The account carries both the confirmed-but-unsettled redemption and the
    # still-pending QDII subscription over the calendar boundary.
    assert year_end["pending_order_count"] == 2
    assert year_end["receivable_cash"] > 0.0
    assert year_end["frozen_cash"] > 0.0
    assert new_year["receivable_cash"] > 0.0
    assert new_year["pending_order_count"] == 1

    redemption = next(
        order for order in engine.last_orders if order.side.value == "redeem"
    )
    assert len(redemption.lot_allocations) == 2
    assert [lot.acquired_date for lot in redemption.lot_allocations] == [
        pd.Timestamp("2020-12-23"), pd.Timestamp("2020-12-25")
    ]
    assert redemption.confirmation_date == pd.Timestamp("2020-12-31")

    restarted_engine = OTFBacktestEngine(
        db_path=_create_cross_year_db(tmp_path / "restart"),
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=3,
        initial_cash=1_000_000.0,
    )
    restart_targets = pd.DataFrame(
        [{"F001": 0.0, "F003": 0.0}], index=pd.to_datetime(["2021-01-04"])
    )
    restarted = restarted_engine.run_backtest(
        restart_targets, start="2021-01-04", end="2021-01-06", rebalance_every=1
    )
    restarted_first = restarted.iloc[0]
    assert new_year["equity"] != pytest.approx(restarted_first["equity"])
    assert restarted_first["receivable_cash"] == 0.0
    assert restarted_first["pending_order_count"] == 0

    compounded = float(continuous["equity"].iloc[0]) * float(
        (1.0 + continuous["daily_return"]).prod()
    )
    assert continuous["equity"].iloc[0] == pytest.approx(1_000_000.0, abs=0.01)
    assert continuous["equity"].iloc[-1] == pytest.approx(compounded, abs=0.10)


def test_same_small_scenario_continuous_vs_annual_restart_and_compounding(tmp_path):
    """Annual restart is a distinct sensitivity, not the continuous account."""
    continuous_engine = OTFBacktestEngine(
        db_path=_create_cross_year_db(tmp_path / "continuous"),
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=3,
        subscription_fee_rate=0.001,
        initial_cash=1_000_000.0,
    )
    restart_engine = OTFBacktestEngine(
        db_path=_create_cross_year_db(tmp_path / "annual_restart"),
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=3,
        subscription_fee_rate=0.001,
        initial_cash=1_000_000.0,
    )
    targets = pd.DataFrame(
        [{"F001": 0.25, "F003": 0.0}],
        index=pd.to_datetime(["2020-12-22"]),
    )
    continuous = continuous_engine.run_backtest(
        targets, start="2020-12-21", end="2021-01-06", rebalance_every=1
    )
    annual_restart = restart_engine.run_backtest(
        pd.DataFrame(
            [{"F001": 0.0, "F003": 0.0}],
            index=pd.to_datetime(["2021-01-04"]),
        ),
        start="2021-01-04",
        end="2021-01-06",
        rebalance_every=1,
    )

    continuous_boundary = continuous.loc[
        continuous["date"] == pd.Timestamp("2021-01-04")
    ].iloc[0]
    assert continuous_boundary["equity"] != pytest.approx(
        annual_restart.iloc[0]["equity"]
    )
    expected_final_equity = float(continuous["equity"].iloc[0]) * float(
        (1.0 + continuous["daily_return"]).prod()
    )
    assert continuous["equity"].iloc[-1] == pytest.approx(
        expected_final_equity, abs=0.10
    )


def test_walkforward_s1_wrapper_matches_formal_state_rotation_signal():
    direct = StateRotationSignal(nav_df=pd.DataFrame())
    runner_signal = make_s1_signal(
        rule_book=None, engine=SimpleNamespace(_nav_df=pd.DataFrame())
    )
    date = pd.Timestamp("2021-01-29")
    direct_weights = direct(date)
    runner_weights = runner_signal(date)
    assert runner_weights == pytest.approx(direct_weights)
    assert set(runner_weights).isdisjoint({"cash_mgt", "MONEY_MARKET", "CD_NCD"})


def test_artifact_damage_fails_content_gate(tmp_path):
    result = validate_artifact_bundle(tmp_path, strategy_name="B2_Static_EW_4Asset")
    assert result["passed"] is False
    assert result["present_count"] == 0
    assert any(item.startswith("missing_or_empty:") for item in result["errors"])


def test_manifest_inventory_and_not_applicable_schema_are_content_checked(tmp_path):
    bundle = _write_valid_b2_bundle(tmp_path)
    valid = validate_artifact_bundle(
        bundle,
        strategy_name="B2_Static_EW_4Asset",
        expected_run_id="test_run",
        expected_start="2021-01-04",
        expected_end="2021-01-04",
    )
    assert valid["passed"] is True
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == set(REQUIRED_ARTIFACTS) - {"manifest.json"}
    assert manifest["artifacts"]["market_states.csv"]["status"] == "NOT_APPLICABLE"

    pd.DataFrame([{"signal_date": "2021-01-04", "market_state": "RISK_ON"}]).to_csv(
        bundle / "market_states.csv", index=False, encoding="utf-8-sig"
    )
    damaged = validate_artifact_bundle(
        bundle,
        strategy_name="B2_Static_EW_4Asset",
        expected_run_id="test_run",
        expected_start="2021-01-04",
        expected_end="2021-01-04",
    )
    assert damaged["passed"] is False
    assert any("market_states.csv:must_be_empty" in item for item in damaged["errors"])
    assert any("artifact_sha256_mismatch:market_states.csv" in item for item in damaged["errors"])


def test_corrupted_artifact_is_translated_to_a_failed_candidate_gate(tmp_path):
    bundle = _write_valid_b2_bundle(tmp_path)
    (bundle / "target_weights.csv").write_text("date\n2021-01-04\n", encoding="utf-8")
    validation = validate_artifact_bundle(
        bundle,
        strategy_name="B2_Static_EW_4Asset",
        expected_run_id="test_run",
        expected_start="2021-01-04",
        expected_end="2021-01-04",
    )
    assert validation["passed"] is False
    status, _ = derive_research_status(
        {"B2_Static_EW_4Asset": {
            "gate_passed": validation["passed"],
            "failed_checks": ["artifact_content"],
        }},
        historical_rules_passed=True,
        test_result={"all_passed": True},
    )
    assert status == "OOS_GATE_FAILED"


def test_formal_s1_keeps_sleeve_and_fund_key_spaces_separate(monkeypatch):
    signal = StateRotationSignal(nav_df=pd.DataFrame())
    sleeve = next(iter(signal._selector._fixed_products))
    monkeypatch.setattr(
        signal._mse,
        "get_state",
        lambda date, prev_state=None: ("NEUTRAL", {}, {}),
    )
    monkeypatch.setattr(
        signal._selector,
        "select_weights",
        lambda date, state, prev_weights=None: pd.Series({sleeve: 1.0}),
    )
    weights = signal(pd.Timestamp("2021-01-04"))
    assert weights
    assert all(key not in {"cash_mgt", "MONEY_MARKET", "CD_NCD"} for key in weights)
    assert all(key in signal._selector._fixed_products.values() for key in weights)
    assert all(key in signal.last_signal_audit["sleeve_weights"] for key in [sleeve])
