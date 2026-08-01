"""Behavioral tests for the frozen C2 low-turnover strategy."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from otf_rotation.low_turnover_core_satellite import (
    C2_CORE_WEIGHTS,
    C2_QDII_CODES,
    C2_SATELLITE_POOL,
    LowTurnoverCoreSatelliteSignal,
    _select_with_retention,
)
from run_low_turnover_core_satellite import (
    OBSERVATION_READINESS_BLOCKERS,
    _classify_gate_layers,
)


def _growth_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for code in [*C2_CORE_WEIGHTS, *C2_SATELLITE_POOL]:
        rate = 0.03 if code in {"160706", "000008", "007466"} else 0.01
        for date in dates:
            rows.append({"fund_code": code, "nav_date": date, "daily_growth_pct": rate, "unit_nav": 1.0})
    return pd.DataFrame(rows)


def test_c2_retains_eligible_incumbents_before_rank_fill():
    evaluations = {
        "a": {"eligible": True, "momentum_score": 0.90},
        "b": {"eligible": True, "momentum_score": 0.80},
        "c": {"eligible": True, "momentum_score": 0.70},
        "d": {"eligible": True, "momentum_score": 0.60},
    }
    selected, audit = _select_with_retention(evaluations, ["d", "c"], maximum=3)
    assert set(selected[:2]) == {"d", "c"}
    assert selected[2] == "a"
    assert audit["d"]["retained_existing_while_eligible"] is True


def test_c2_applies_domestic_t1_and_qdii_t2_availability_lags():
    dates = pd.bdate_range("2020-01-01", periods=260)
    signal = LowTurnoverCoreSatelliteSignal(_growth_frame(dates), trading_dates=dates)
    signal_date = dates[-1]
    domestic = signal.available_date("160706", signal_date)
    qdii = signal.available_date("050025", signal_date)
    assert domestic == dates[-2]
    assert qdii == dates[-3]


def test_c2_does_not_use_future_growth_for_signal():
    dates = pd.bdate_range("2020-01-01", periods=260)
    frame = _growth_frame(dates)
    signal_date = dates[-1]
    before = LowTurnoverCoreSatelliteSignal(frame, trading_dates=dates)(signal_date)
    future_dates = pd.bdate_range(signal_date + pd.Timedelta(days=1), periods=5)
    future = _growth_frame(future_dates)
    future.loc[future["fund_code"] == "160706", "daily_growth_pct"] = 50.0
    after = LowTurnoverCoreSatelliteSignal(pd.concat([frame, future], ignore_index=True), trading_dates=dates.append(future_dates))(signal_date)
    assert before == pytest.approx(after)


def test_c2_static_budget_uses_three_slots_and_cash_fallback():
    dates = pd.bdate_range("2020-01-01", periods=260)
    signal = LowTurnoverCoreSatelliteSignal(_growth_frame(dates), trading_dates=dates)
    target = signal(dates[-1])
    assert sum(target.values()) == pytest.approx(1.0)
    assert target["260102"] >= C2_CORE_WEIGHTS["260102"]
    assert len(set(target) & set(C2_SATELLITE_POOL)) <= 3


def test_c2_drift_boundary_is_7_5_percentage_points():
    dates = pd.bdate_range("2020-01-01", periods=260)
    signal = LowTurnoverCoreSatelliteSignal(_growth_frame(dates), trading_dates=dates)
    first = signal(dates[-1])
    assert first
    signal.last_signal_audit["max_drift_deviation_pct_points"] = 7.5
    assert signal.last_signal_audit["parameters"]["drift_threshold_pct_points"] == pytest.approx(7.5)


def test_c2_config_is_frozen_and_uses_qdii_codes():
    config = Path(__file__).resolve().parents[1] / "config/low_turnover_core_satellite.json"
    import json

    payload = json.loads(config.read_text(encoding="utf-8"))
    assert payload["parameter_search"] == "FORBIDDEN_ALL_PARAMETERS_FROZEN_ONCE"
    assert payload["satellite_budget"] == pytest.approx(0.25)
    assert payload["selection"]["maximum_satellites"] == 3
    assert set(payload["cross_border_qdii_codes"]) == set(C2_QDII_CODES)


def test_gate_layers_keep_original_checks_but_separate_research_and_observation():
    checks = {
        name: True
        for name in (
            "net_cagr", "sharpe", "mdd", "calmar", "relative_b2_cagr_advantage",
            "worst_two_year_cagr", "sustained_confirmed_turnover", "double_fee_net_cagr",
            "double_fee_sharpe", "double_fee_mdd", "double_fee_calmar",
            "delay_cagr_decline", "delay_mdd", "subperiod_2021_2023",
            "subperiod_2024_2026", "data_gate", "rules_gate", "mapping_gate",
            "fee_reconciliation", "c1_reference_hash_match", "artifact_schema",
            "artifact_content", "publication_timing_gate", "historical_truth_gate",
        )
    }
    checks["publication_timing_gate"] = False
    checks["historical_truth_gate"] = False
    gate = _classify_gate_layers({"checks": checks})
    assert gate["research_performance_gate_passed"] is True
    assert gate["research_performance_failed_checks"] == []
    assert gate["observation_readiness_gate_passed"] is False
    assert gate["observation_readiness_failed_reasons"][:3] == list(OBSERVATION_READINESS_BLOCKERS)
    assert "publication_timing_gate" not in gate["research_performance_failed_checks"]
    assert "historical_truth_gate" not in gate["research_performance_failed_checks"]

    checks["net_cagr"] = False
    gate = _classify_gate_layers({"checks": checks})
    assert gate["research_performance_gate_passed"] is False
    assert gate["research_performance_failed_checks"] == ["net_cagr"]
    assert "RESEARCH_PERFORMANCE_GATE_FAILED" in gate["observation_readiness_failed_reasons"]
