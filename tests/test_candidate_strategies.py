"""Behavioral tests for the frozen B2-LT and D1 candidates."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from otf_rotation.artifact_validation import artifact_status_for_strategy
from otf_rotation.candidate_strategies import (
    B2LTSignal,
    D1Signal,
    build_candidate_targets,
    natural_drift_weights,
)
from otf_rotation.schedule import build_quarter_end_schedule, build_signal_submit_map
from run_candidate_strategies import (
    B2LT_NAME,
    D1_NAME,
    aggregate_candidate_status,
    apply_candidate_baseline_only_correction,
    sync_candidate_gate_fields,
)


def _nav_rows(code: str, dates: pd.DatetimeIndex, values: list[float]) -> list[dict[str, object]]:
    return [
        {"fund_code": code, "nav_date": date, "unit_nav": value}
        for date, value in zip(dates, values)
    ]


def _rules_mapping(codes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = pd.DataFrame(
        [
            {
                "fund_code": code,
                "rule_status": "DISTRIBUTOR_VERIFIED",
                "effective_from": "",
                "effective_to": "",
            }
            for code in codes
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "fund_code": code,
                "mapping_confidence": "HIGH",
                "review_status": "APPROVED",
                "effective_from": "",
                "effective_to": "",
            }
            for code in codes
        ]
    )
    return rules, mapping


def _d1_products() -> dict[str, dict[str, object]]:
    return {
        "000001": {"asset_class": "domestic_equity", "budget": 0.90, "trend_filter": True},
        "260102": {"asset_class": "cash", "budget": 0.10, "trend_filter": False},
    }


def test_quarter_end_observations_and_next_day_submit_are_distinct():
    dates = pd.bdate_range("2021-01-04", "2021-12-31")
    signals = build_quarter_end_schedule(dates, "2021-01-01", "2021-12-31")
    assert [date.month for date in signals] == [3, 6, 9, 12]
    mapping = build_signal_submit_map(dates, signals, "2021-12-31")
    # The December signal has no next trading day inside the requested end.
    assert len(mapping) == 3
    assert all(pd.Timestamp(submit) > pd.Timestamp(signal) for submit, signal in mapping.items())


def test_natural_drift_weights_are_normalised_from_published_navs():
    drift = natural_drift_weights(
        {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
        {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0},
        {"a": 1.1, "b": 1.0, "c": 1.0, "d": 1.0},
    )
    assert sum(drift.values()) == pytest.approx(1.0)
    assert drift["a"] == pytest.approx(1.1 / 4.1)


def test_b2_lt_skips_no_trigger_and_triggers_inclusive_five_point_boundary():
    first = pd.Timestamp("2021-03-31")
    second = pd.Timestamp("2021-06-30")
    boundary_nav = 9.0 / 7.0
    rows = []
    for code, value in {
        "160706": [1.0, boundary_nav],
        "000218": [1.0, 1.0],
        "001512": [1.0, 1.0],
        "260102": [1.0, 1.0],
    }.items():
        rows.extend(_nav_rows(code, pd.DatetimeIndex([first, second]), value))
    signal = B2LTSignal(pd.DataFrame(rows), threshold=0.05)
    assert signal(first) == pytest.approx({code: 0.25 for code in signal.assets})
    # The resulting 160706 weight is exactly 30%; equality at 5pp must trigger.
    assert signal(second) == pytest.approx({code: 0.25 for code in signal.assets})
    assert signal.last_signal_audit["decision"] == "THRESHOLD_TRIGGER"


def test_candidate_target_builder_omits_empty_no_rebalance_signals():
    dates = pd.bdate_range("2021-03-31", "2021-06-30")
    signal_dates = [dates[0], dates[-1]]
    signal_map = {dates[1].strftime("%Y-%m-%d"): dates[0].strftime("%Y-%m-%d")}

    def signal(_date):
        signal.last_signal_audit = {"signal_date": _date, "rebalance": False}
        return {}

    targets, audits = build_candidate_targets(signal, signal_dates, signal_map)
    assert targets.empty
    assert len(audits) == 1


def test_d1_ma200_uses_only_history_before_and_on_signal_date():
    dates = pd.bdate_range("2020-01-01", periods=205)
    rows = _nav_rows("000001", dates, [1.0] * 205)
    rows += _nav_rows("260102", dates, [1.0] * 205)
    rules, mapping = _rules_mapping(["000001", "260102"])
    signal = D1Signal(
        pd.DataFrame(rows), rules, mapping, _d1_products(),
        {"domestic_equity": 0.90, "cash": 0.10},
        fallback_fund="260102", ma_days=200, threshold=0.05,
    )
    at_signal = signal(dates[199])
    assert at_signal["000001"] == pytest.approx(0.90)

    future_rows = pd.concat(
        [pd.DataFrame(rows), pd.DataFrame(_nav_rows("000001", pd.DatetimeIndex([dates[200]]), [100.0]))],
        ignore_index=True,
    )
    future_signal = D1Signal(
        future_rows, rules, mapping, _d1_products(),
        {"domestic_equity": 0.90, "cash": 0.10},
        fallback_fund="260102", ma_days=200, threshold=0.05,
    )
    assert future_signal(dates[199]) == pytest.approx(at_signal)


def test_d1_trend_failure_transfers_product_budget_to_cash():
    dates = pd.bdate_range("2020-01-01", periods=200)
    rows = _nav_rows("000001", dates, [1.0] * 199 + [0.5])
    rows += _nav_rows("260102", dates, [1.0] * 200)
    rules, mapping = _rules_mapping(["000001", "260102"])
    signal = D1Signal(
        pd.DataFrame(rows), rules, mapping, _d1_products(),
        {"domestic_equity": 0.90, "cash": 0.10}, fallback_fund="260102",
    )
    target = signal(dates[-1])
    assert target == pytest.approx({"260102": 1.0})
    assert signal.last_signal_audit["evaluations"][0]["trend_state"] == "BELOW_MA200"


def test_d1_insufficient_pit_history_transfers_budget_to_cash():
    dates = pd.bdate_range("2020-01-01", periods=199)
    rows = _nav_rows("000001", dates, [1.0] * 199)
    rows += _nav_rows("260102", dates, [1.0] * 199)
    rules, mapping = _rules_mapping(["000001", "260102"])
    signal = D1Signal(
        pd.DataFrame(rows), rules, mapping, _d1_products(),
        {"domestic_equity": 0.90, "cash": 0.10}, fallback_fund="260102",
    )
    target = signal(dates[-1])
    assert target == pytest.approx({"260102": 1.0})
    assert "PIT_NAV_INSUFFICIENT_FOR_MA200" in signal.last_signal_audit["evaluations"][0]["reason"]


def test_d1_uses_last_published_pit_nav_when_signal_day_has_no_publication():
    dates = pd.bdate_range("2020-01-01", periods=201)
    signal_date = dates[-1]
    product_dates = dates[:-1]
    rows = _nav_rows("000001", product_dates, [1.0] * 200)
    rows += _nav_rows("260102", dates, [1.0] * 201)
    rules, mapping = _rules_mapping(["000001", "260102"])
    signal = D1Signal(
        pd.DataFrame(rows), rules, mapping, _d1_products(),
        {"domestic_equity": 0.90, "cash": 0.10}, fallback_fund="260102",
    )
    target = signal(signal_date)
    assert target["000001"] == pytest.approx(0.90)
    assert signal.last_signal_audit["evaluations"][0]["nav_date"] == product_dates[-1]


def test_candidate_ordered_signal_does_not_execute_on_signal_day():
    dates = pd.bdate_range("2021-03-31", "2021-04-05")
    rows = []
    for code in B2LTSignal.assets:
        rows.extend(_nav_rows(code, dates, [1.0] * len(dates)))
    signal = B2LTSignal(pd.DataFrame(rows))
    signal_dates = [dates[0]]
    signal_map = {dates[1].strftime("%Y-%m-%d"): dates[0].strftime("%Y-%m-%d")}
    targets, _ = build_candidate_targets(signal, signal_dates, signal_map)
    assert targets.index[0] == dates[1]
    assert targets.index[0] > signal_dates[0]


def test_candidate_validator_extension_keeps_canonical_static_schema():
    assert artifact_status_for_strategy("D1_MultiAsset_Trend_Defensive", "market_states.csv") == "REQUIRED"
    assert artifact_status_for_strategy("D1_MultiAsset_Trend_Defensive", "product_selection_audit.csv") == "REQUIRED"
    assert artifact_status_for_strategy("B2_Static_EW_4Asset", "market_states.csv") == "NOT_APPLICABLE"
    assert artifact_status_for_strategy("B2_LT_Static_EW_4Asset", "risk_contributions.csv") == "NOT_APPLICABLE"


def test_candidate_gate_alias_syncs_to_final_absolute_gate():
    passed = sync_candidate_gate_fields({"gate_passed": True, "failed_checks": []})
    failed = sync_candidate_gate_fields({"gate_passed": False, "failed_checks": ["turnover"]})
    assert passed["candidate_gate_passed"] is True
    assert failed["candidate_gate_passed"] is False
    assert passed["candidate_gate_semantics"] == "ALIAS_OF_ABSOLUTE_GATE_PASSED"


def test_candidate_status_selects_passing_candidate_independently():
    gates = {
        B2LT_NAME: {"gate_passed": True},
        D1_NAME: {"gate_passed": False},
    }
    decisions = {
        B2LT_NAME: {"relative_upgrade_check_passed": True},
        D1_NAME: {"relative_upgrade_check_passed": True},
    }
    status, selected, rejected = aggregate_candidate_status(gates, decisions)
    assert status == "PARTIAL_PAPER_TRADE_CANDIDATE"
    assert selected == [B2LT_NAME]
    assert rejected == [D1_NAME]


def test_candidate_status_is_oos_failed_when_no_candidate_passes():
    gates = {
        B2LT_NAME: {"gate_passed": False},
        D1_NAME: {"gate_passed": False},
    }
    decisions = {
        B2LT_NAME: {"relative_upgrade_check_passed": False},
        D1_NAME: {"relative_upgrade_check_passed": False},
    }
    status, selected, rejected = aggregate_candidate_status(gates, decisions)
    assert status == "OOS_GATE_FAILED"
    assert selected == []
    assert rejected == [B2LT_NAME, D1_NAME]


def test_correction_demotes_only_the_historical_candidate_run_without_touching_metrics():
    status, selected, rejected, decisions = apply_candidate_baseline_only_correction(
        "candidate_20260729_141050",
        selected_candidates=[B2LT_NAME],
        rejected_candidates=[D1_NAME],
        decisions={
            B2LT_NAME: {
                "decision": "UPGRADE",
                "classification": "EXECUTION_OPTIMIZATION_CANDIDATE_NOT_ALPHA",
                "relative_upgrade_check_passed": True,
            },
            D1_NAME: {"decision": "ELIMINATE_GATE_FAILED"},
        },
    )
    assert status == "RESEARCH_BASELINE_ONLY"
    assert selected == []
    assert rejected == [B2LT_NAME, D1_NAME]
    assert decisions[B2LT_NAME]["relative_upgrade_check_passed"] is False
    assert "5.88% CAGR" in decisions[B2LT_NAME]["reason"]


def test_c1_artifacts_are_required_while_static_candidate_schema_is_unchanged():
    assert artifact_status_for_strategy("C1_CORE_SATELLITE_MOMENTUM", "market_states.csv") == "REQUIRED"
    assert artifact_status_for_strategy("C1_CORE_SATELLITE_MOMENTUM", "risk_contributions.csv") == "REQUIRED"
    assert artifact_status_for_strategy("B2_LT_Static_EW_4Asset", "risk_contributions.csv") == "NOT_APPLICABLE"
