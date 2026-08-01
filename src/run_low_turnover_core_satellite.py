"""Run the frozen C2 low-turnover core/satellite research study."""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_core_satellite_momentum as c1
import run_b1_b2_b3_walkforward as canonical
from otf_backtest_engine import OTFBacktestEngine
from otf_rotation.artifact_validation import validate_artifact_bundle
from otf_rotation.experiment_artifacts import (
    create_run_directory,
    export_config_snapshot,
    export_json,
    export_metrics,
    export_table,
    sha256_file,
    write_manifest,
)
from otf_rotation.low_turnover_core_satellite import (
    C2_CORE_WEIGHTS,
    C2_DRIFT_THRESHOLD,
    C2_MAX_SATELLITES,
    C2_QDII_CODES,
    C2_SATELLITE_POOL,
    LowTurnoverCoreSatelliteSignal,
)
from otf_rotation.execution_calendar import load_execution_calendar
from otf_rotation.schedule import build_month_end_schedule, build_signal_submit_map
from otf_trading_rules import ProductRuleBook


DB_PATH = "data/processed/otf_expanded.sqlite"
CALENDAR_PATH = "data/processed/execution_calendar/cn_execution_calendar.csv"
RULES_PATH = "config/otf_product_rules.csv"
MAPPING_PATH = "config/otf_exposure_mapping.csv"
CONFIG_PATH = "config/low_turnover_core_satellite.json"
OUTPUT_DIR = Path("reports/strategy_research/core_satellite_low_turnover")
INITIAL_CASH = 1_000_000.0
ACCOUNT_MODE = "FROZEN_PARAMETER_CONTINUOUS_OOS"
RULE_SCENARIO = "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO"
HISTORICAL_RULE_STATUS = "NOT_ESTABLISHED"
OOS_START = "2021-01-04"
OOS_END = "2026-07-27"
C2_NAME = "C2_LOW_TURNOVER_CORE_SATELLITE"
B2_NAME = c1.B2_NAME
B2LT_NAME = c1.B2LT_NAME
PUBLICATION_TIMING_STATUS = "NAV_PUBLICATION_TIMESTAMP_NOT_AVAILABLE"
SIGNAL_PIT_TRUTH_STATUS = "NOT_ESTABLISHED"
C1_REFERENCE_DIR = ROOT / "reports/strategy_research/core_satellite/core_satellite_20260729_151927"

RESEARCH_PERFORMANCE_CHECK_NAMES = (
    "net_cagr",
    "sharpe",
    "mdd",
    "calmar",
    "relative_b2_cagr_advantage",
    "worst_two_year_cagr",
    "sustained_confirmed_turnover",
    "double_fee_net_cagr",
    "double_fee_sharpe",
    "double_fee_mdd",
    "double_fee_calmar",
    "delay_cagr_decline",
    "delay_mdd",
    "subperiod_2021_2023",
    "subperiod_2024_2026",
    "data_gate",
    "rules_gate",
    "mapping_gate",
    "fee_reconciliation",
    "c1_reference_hash_match",
    "artifact_schema",
    "artifact_content",
)
OBSERVATION_READINESS_BLOCKERS = (
    "REUSED_SAMPLE_NOT_FRESH_OOS",
    "NAV_PUBLICATION_TRUTH_NOT_ESTABLISHED",
    "HISTORICAL_RULE_TRUTH_NOT_ESTABLISHED",
)


def publication_timing_audit() -> dict[str, Any]:
    return {
        "status": PUBLICATION_TIMING_STATUS,
        "signal_point_in_time_truth": SIGNAL_PIT_TRUTH_STATUS,
        "publication_timestamp_column": None,
        "publication_timestamp_available": False,
        "nav_date_filter_is_verified_pit": False,
        "affected_or_potentially_affected_products": sorted(C2_QDII_CODES),
        "reason": (
            "The source table has nav_date and daily_growth_pct but no publication timestamp. "
            "C2 therefore applies conservative domestic T+1 and QDII T+2 availability lags; "
            "this remains an assumption until official publication timestamps are obtained."
        ),
    }


def _load_config() -> dict[str, Any]:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def _reference_hash_audit() -> dict[str, Any]:
    current = {
        "db_sha256": sha256_file(str(ROOT / DB_PATH)),
        "rules_sha256": sha256_file(str(ROOT / RULES_PATH)),
        "mapping_sha256": sha256_file(str(ROOT / MAPPING_PATH)),
    }
    reference: dict[str, Any] = {}
    path = C1_REFERENCE_DIR / "core_satellite_input_facts.json"
    if path.exists():
        try:
            reference = json.loads(path.read_text(encoding="utf-8")).get("input_hashes", {})
        except (OSError, json.JSONDecodeError):
            reference = {}
    matches = {
        key: bool(reference.get(key) and reference.get(key) == value)
        for key, value in current.items()
    }
    return {
        "reference_run_id": C1_REFERENCE_DIR.name if C1_REFERENCE_DIR.exists() else None,
        "current_hashes": current,
        "reference_hashes": {key: reference.get(key) for key in current},
        "matches": matches,
        "all_match": bool(matches) and all(matches.values()),
    }


def _facts(config: dict[str, Any]) -> dict[str, Any]:
    rules = pd.read_csv(ROOT / RULES_PATH, dtype=str).fillna("")
    mapping = pd.read_csv(ROOT / MAPPING_PATH, dtype=str).fillna("")
    return {
        "input_hashes": {
            "db_sha256": sha256_file(str(ROOT / DB_PATH)),
            "rules_sha256": sha256_file(str(ROOT / RULES_PATH)),
            "mapping_sha256": sha256_file(str(ROOT / MAPPING_PATH)),
            "config_sha256": sha256_file(str(ROOT / CONFIG_PATH)),
            "calendar_sha256": sha256_file(str(ROOT / CALENDAR_PATH)),
        },
        "rule_counts_by_status": dict(sorted(rules["rule_status"].astype(str).str.strip().value_counts().items())),
        "mapping_counts_by_status": dict(
            sorted(
                {
                    **{
                        f"review_status:{key}": int(value)
                        for key, value in mapping["review_status"].astype(str).str.strip().str.upper().value_counts().items()
                    },
                    **{
                        f"mapping_confidence:{key}": int(value)
                        for key, value in mapping["mapping_confidence"].astype(str).str.strip().str.upper().value_counts().items()
                    },
                }.items()
            )
        ),
        "rule_count": int(len(rules)),
        "mapping_count": int(len(mapping)),
        "rule_temporal_coverage": {
            "effective_from_present": int((rules["effective_from"].str.strip() != "").sum()),
            "verified_at_present": int((rules["verified_at"].str.strip() != "").sum()),
            "total_rules": int(len(rules)),
        },
        "calendar_correction_label": "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "supersedes_run_id": "low_turnover_core_satellite_20260729_163519",
        "intermediate_calendar_corrected_run_id": "low_turnover_core_satellite_20260730_115448",
        "execution_calendar": load_execution_calendar(str(ROOT / CALENDAR_PATH)).facts(),
        "publication_timing_audit": publication_timing_audit(),
        "c1_reference_hash_audit": _reference_hash_audit(),
        "config": config,
    }


def _create_engine(rule_book: ProductRuleBook) -> OTFBacktestEngine:
    return c1._create_engine(rule_book)


def _signal_targets(
    signal: LowTurnoverCoreSatelliteSignal,
    signal_dates: list[pd.Timestamp],
    signal_map: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    return c1._signal_targets(signal, signal_dates, signal_map)


def _run_account(
    engine: OTFBacktestEngine,
    targets: pd.DataFrame,
    signal_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return c1._run_account(engine, targets, signal_map)


def _metrics(
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    turnover: pd.DataFrame,
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = c1._c1_metrics(daily, orders, turnover, audits)
    metrics["strategy"] = C2_NAME
    metrics["availability_lag_model"] = {"domestic": 1, "qdii": 2}
    metrics["continuous_volatility_scaling"] = False
    metrics["maximum_satellite_count"] = C2_MAX_SATELLITES
    metrics["observation_frequency"] = "semiannual_june_december"
    return metrics


def _write_c2_audit_tables(strategy_dir: Path, audits: list[dict[str, Any]]) -> None:
    market_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for audit in audits:
        date = audit.get("signal_date")
        selected = list(audit.get("selected_satellites", []))
        market_rows.append(
            {
                "signal_date": date,
                "submit_date": audit.get("submit_date"),
                "market_state": "LOW_TURNOVER_CORE_SATELLITE",
                "rebalance": audit.get("rebalance", False),
                "decision": audit.get("decision", ""),
                "selection_changed": audit.get("selection_changed", False),
                "selected_satellites": ";".join(selected),
                "available_lag_model": "DOMESTIC_T1_QDII_T2",
                "max_drift_deviation_pct_points": audit.get("max_drift_deviation_pct_points", 0.0),
                "proposed_target_weights": json.dumps(audit.get("proposed_target_weights", {}), sort_keys=True, default=str),
                "accepted_target_weights": json.dumps(audit.get("target_weights", {}), sort_keys=True, default=str),
                "cash_transfer": audit.get("cash_transfer", 0.0),
            }
        )
        buffer_audit = audit.get("buffer_audit", {})
        for evaluation in audit.get("evaluations", []):
            code = str(evaluation.get("fund_code", ""))
            item = buffer_audit.get(code, {})
            eligible_count = sum(bool(value.get("eligible")) for value in audit.get("evaluations", []))
            row = {
                "date": date,
                "fund_code": code,
                "state": "ELIGIBLE" if evaluation.get("eligible") else evaluation.get("reason", "INELIGIBLE"),
                "score": evaluation.get("momentum_score") or 0.0,
                "available_as_of": evaluation.get("available_as_of"),
                "availability_lag_trading_days": evaluation.get("availability_lag_trading_days"),
                "published_observations": evaluation.get("published_observations"),
                "latest_published_date": evaluation.get("latest_published_date"),
                "total_return_126": evaluation.get("total_return_126"),
                "total_return_252": evaluation.get("total_return_252"),
                "ma200": evaluation.get("ma200"),
                "momentum_score": evaluation.get("momentum_score"),
                "trend_condition": evaluation.get("trend_condition"),
                "momentum_condition": evaluation.get("momentum_condition"),
                "eligible": evaluation.get("eligible"),
                "rank": item.get("rank"),
                "retained_existing_while_eligible": item.get("retained_existing_while_eligible", False),
                "selected": item.get("selected", False),
                "rebalance": audit.get("rebalance", False),
                "rebalance_reason": audit.get("decision", ""),
            }
            score_rows.append(row)
            selection_rows.append(
                {
                    "sleeve": "satellite",
                    "date": date,
                    "top_n": C2_MAX_SATELLITES,
                    "selected_count": len(selected),
                    "total_candidates": len(C2_SATELLITE_POOL),
                    "eligible_count": eligible_count,
                    "selected_funds": ";".join(selected),
                    "maximum_satellites": C2_MAX_SATELLITES,
                    **row,
                }
            )
        for code, weight in audit.get("proposed_target_weights", {}).items():
            sleeve = "core" if code in C2_CORE_WEIGHTS else "satellite" if code in C2_SATELLITE_POOL else "cash_fallback"
            budget_rows.append({"date": date, "sleeve": sleeve, "weight": weight, "fund_code": code})
            risk_rows.append(
                {
                    "date": date,
                    "fund_code": code,
                    "risk_contribution": None,
                    "predicted_volatility_pct": None,
                    "risk_model": "NOT_APPLIED_C2_STATIC_BUDGET",
                }
            )
    market = pd.DataFrame(market_rows)
    scores = pd.DataFrame(score_rows)
    budgets = pd.DataFrame(budget_rows)
    selections = pd.DataFrame(selection_rows)
    risks = pd.DataFrame(risk_rows)
    export_table(strategy_dir.as_posix(), "market_states.csv", market)
    export_table(strategy_dir.as_posix(), "state_scores.csv", scores)
    export_table(strategy_dir.as_posix(), "asset_budgets.csv", budgets)
    export_table(strategy_dir.as_posix(), "sleeve_weights.csv", budgets)
    export_table(strategy_dir.as_posix(), "fund_weights.csv", budgets)
    export_table(strategy_dir.as_posix(), "product_selection_audit.csv", selections)
    export_table(strategy_dir.as_posix(), "risk_contributions.csv", risks)


def _write_c2_bundle(
    strategy_dir: Path,
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    rejections: pd.DataFrame,
    turnover: pd.DataFrame,
    fee_reconciliation: pd.DataFrame,
    fee_order_audit: pd.DataFrame,
    audits: list[dict[str, Any]],
    engine: OTFBacktestEngine,
    config: dict[str, Any],
    facts: dict[str, Any],
    metrics: dict[str, Any],
    gate: dict[str, Any],
    parameter_freeze: dict[str, Any],
) -> dict[str, Any]:
    """Export the generic bundle, then replace C2 audit placeholders before validation."""
    canonical.export_strategy_bundle(
        strategy_dir,
        C2_NAME,
        daily,
        orders,
        rejections.to_dict(orient="records"),
        turnover,
        fee_reconciliation,
        fee_order_audit,
        audits,
        engine.last_position_lots,
        config,
        facts,
        metrics,
        gate,
        parameter_freeze,
    )
    _write_c2_audit_tables(strategy_dir, audits)
    write_manifest(
        run_dir=strategy_dir.as_posix(),
        strategy_name=C2_NAME,
        config=config,
        db_path=str(ROOT / DB_PATH),
        rules_path=str(ROOT / RULES_PATH),
        exposure_mapping_path=str(ROOT / MAPPING_PATH),
        metrics=metrics,
        gate_result=gate,
    )
    artifact_validation = canonical.finalize_artifact_gate(
        strategy_dir,
        C2_NAME,
        config["run_id"],
        config["actual_oos_period"][0],
        config["actual_oos_period"][1],
        config,
        metrics,
        gate,
    )
    gate["artifact_validation"] = {
        key: artifact_validation.get(key)
        for key in ("passed", "required_count", "present_count", "errors", "artifact_status")
    }
    return gate


def _run_stress(
    scenario: str,
    rule_book: ProductRuleBook,
    growth_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    signal_dates: list[pd.Timestamp],
    signal_map: dict[str, str],
    base_config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    used_codes = sorted(set(C2_CORE_WEIGHTS) | set(C2_SATELLITE_POOL))
    if scenario == "double_fees":
        stressed_rules = rule_book.scaled_fees(2.0)
        audit = {
            "scenario": scenario,
            "subscription_fee_multiplier": 2.0,
            "redemption_fee_multiplier": 2.0,
            "rule_override": "ProductRuleBook.scaled_fees(2.0)",
            "actual_hit_rules_required": True,
        }
    elif scenario == "delay_plus_one_trading_day":
        stressed_rules = c1.clone_rule_book_with_delay_override(rule_book, used_codes, increment=1)
        audit = {
            "scenario": scenario,
            "subscription_confirmation_days_increment": 1,
            "redemption_confirmation_days_increment": 1,
            "redemption_arrival_days_increment": 1,
            "rule_override": "cloned_actual_hit_product_rules_plus_one",
            "actual_hit_rule_codes": [code for code in used_codes if rule_book.rule_for(code) is not None],
        }
    else:
        raise ValueError(f"unknown_c2_stress:{scenario}")
    engine = _create_engine(stressed_rules)
    signal = LowTurnoverCoreSatelliteSignal(growth_df, trading_dates=trading_dates)
    targets, audits = _signal_targets(signal, signal_dates, signal_map)
    daily, orders, rejections, turnover, fees, extra = _run_account(engine, targets, signal_map)
    metrics = _metrics(daily, orders, turnover, audits)
    metrics["fee_reconciliation_passed"] = bool(extra["fee_summary"].get("passed"))
    metrics["fee_reconciliation"] = extra["fee_summary"]
    audit.update(
        {
            "metrics": metrics,
            "order_count": int(len(orders)),
            "fee_total": float(orders["fee_paid"].sum()) if not orders.empty else 0.0,
            "confirmation_dates": orders["confirmation_date"].dropna().astype(str).tolist() if not orders.empty else [],
            "redemption_arrival_dates": orders["redemption_arrival_date"].dropna().astype(str).tolist() if not orders.empty else [],
        }
    )
    pressure_dir = run_dir / "pressure_tests" / scenario
    pressure_dir.mkdir(parents=True, exist_ok=True)
    export_config_snapshot(pressure_dir.as_posix(), {**base_config, "run_id": f"{run_dir.name}_{scenario}", "stress_scenario": scenario})
    export_table(pressure_dir.as_posix(), "daily_account.csv", daily)
    export_table(pressure_dir.as_posix(), "orders.csv", orders)
    export_table(pressure_dir.as_posix(), "fees.csv", fees)
    export_table(pressure_dir.as_posix(), "turnover.csv", turnover)
    export_metrics(pressure_dir.as_posix(), metrics)
    export_json(pressure_dir.as_posix(), "stress_audit.json", audit)
    del engine
    gc.collect()
    return audit


def _classify_gate_layers(gate: dict[str, Any]) -> dict[str, Any]:
    """Add layered Gate results without changing any original check values."""
    checks = dict(gate.get("checks", {}))
    research_failed = [
        name for name in RESEARCH_PERFORMANCE_CHECK_NAMES
        if not bool(checks.get(name, False))
    ]
    research_passed = not research_failed
    observation_reasons = list(OBSERVATION_READINESS_BLOCKERS)
    if not research_passed:
        observation_reasons.append("RESEARCH_PERFORMANCE_GATE_FAILED")
    gate["research_performance_check_names"] = list(RESEARCH_PERFORMANCE_CHECK_NAMES)
    gate["research_performance_checks"] = {
        name: bool(checks.get(name, False))
        for name in RESEARCH_PERFORMANCE_CHECK_NAMES
    }
    gate["research_performance_failed_checks"] = research_failed
    gate["research_performance_gate_passed"] = research_passed
    gate["observation_readiness_blockers"] = list(OBSERVATION_READINESS_BLOCKERS)
    gate["observation_readiness_failed_reasons"] = observation_reasons
    gate["observation_readiness_gate_passed"] = bool(
        research_passed and not observation_reasons
    )
    return gate


def _gate(
    metrics: dict[str, Any],
    b2_metrics: dict[str, Any],
    double_fee_metrics: dict[str, Any],
    delay_metrics: dict[str, Any],
    static_gates: dict[str, Any],
    reference_audit: dict[str, Any],
) -> dict[str, Any]:
    thresholds = _load_config()["gate_thresholds"]
    metrics["relative_b2_cagr_advantage_pct_points"] = round(metrics["net_cagr_pct"] - b2_metrics["net_cagr_pct"], 4)
    sub_1 = metrics["subperiod_metrics"]["2021_2023"]
    sub_2 = metrics["subperiod_metrics"]["2024_2026"]
    delay_decline = metrics["net_cagr_pct"] - delay_metrics.get("net_cagr_pct", 0.0)
    sustained = metrics.get("max_sustained_annual_confirmed_turnover", np.inf)
    checks = {
        "net_cagr": metrics.get("net_cagr_pct", -np.inf) >= thresholds["net_cagr_min_pct"],
        "sharpe": metrics.get("sharpe", -np.inf) >= thresholds["sharpe_min"],
        "mdd": metrics.get("mdd_pct", -np.inf) > thresholds["mdd_min_pct_exclusive"],
        "calmar": metrics.get("calmar", -np.inf) >= thresholds["calmar_min"],
        "relative_b2_cagr_advantage": metrics["relative_b2_cagr_advantage_pct_points"] >= thresholds["relative_b2_cagr_advantage_min_pct_points"],
        "worst_two_year_cagr": metrics.get("worst_two_year_cagr_pct", -np.inf) >= thresholds["worst_two_year_cagr_min_pct"],
        "sustained_confirmed_turnover": sustained < thresholds["sustained_annual_confirmed_turnover_max_exclusive"],
        "double_fee_net_cagr": double_fee_metrics.get("net_cagr_pct", -np.inf) >= thresholds["double_fee_net_cagr_min_pct"],
        "double_fee_sharpe": double_fee_metrics.get("sharpe", -np.inf) >= thresholds["double_fee_sharpe_min"],
        "double_fee_mdd": double_fee_metrics.get("mdd_pct", -np.inf) > thresholds["double_fee_mdd_min_pct_exclusive"],
        "double_fee_calmar": c1._calmar(double_fee_metrics) >= thresholds["double_fee_calmar_min"],
        "delay_cagr_decline": delay_decline <= thresholds["delay_cagr_decline_max_pct_points"],
        "delay_mdd": delay_metrics.get("mdd_pct", -np.inf) > thresholds["delay_mdd_min_pct_exclusive"],
        "subperiod_2021_2023": sub_1.get("net_cagr_pct", -np.inf) > thresholds["subperiod_net_cagr_positive"] and sub_1.get("sharpe", -np.inf) >= thresholds["subperiod_sharpe_min"],
        "subperiod_2024_2026": sub_2.get("net_cagr_pct", -np.inf) > thresholds["subperiod_net_cagr_positive"] and sub_2.get("sharpe", -np.inf) >= thresholds["subperiod_sharpe_min"],
        "data_gate": bool(static_gates.get("data_gate")),
        "rules_gate": bool(static_gates.get("rules_gate")),
        "mapping_gate": bool(static_gates.get("mapping_gate")),
        "fee_reconciliation": bool(metrics.get("fee_reconciliation_passed")),
        "c1_reference_hash_match": bool(reference_audit.get("all_match")),
        "publication_timing_gate": False,
        "historical_truth_gate": False,
        "artifact_schema": False,
        "artifact_content": False,
    }
    gate = {
        "gate_passed": False,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "thresholds": thresholds,
        "historical_truth_gate": False,
        "historical_truth_status": HISTORICAL_RULE_STATUS,
        "sample_label": "REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "publication_timing_status": PUBLICATION_TIMING_STATUS,
        "signal_point_in_time_truth": SIGNAL_PIT_TRUTH_STATUS,
        "publication_timing_audit": publication_timing_audit(),
        "delay_cagr_decline_pct_points": round(delay_decline, 4),
        "benchmark_b2": B2_NAME,
        "benchmark_c1_reference": reference_audit,
    }
    return _classify_gate_layers(gate)


def _write_conclusion(
    run_dir: Path,
    config: dict[str, Any],
    facts: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    gate: dict[str, Any],
    stresses: dict[str, Any],
    status: str,
) -> None:
    c2 = metrics[C2_NAME]
    lines = [
        "# C2 Low-Turnover Core/Satellite Research",
        "",
        f"- Run ID: `{config['run_id']}`",
        f"- Status: `{status}`",
        f"- Sample label: `{config['sample_label']}`",
        f"- OOS interval: `{config['actual_oos_period'][0]}` to `{config['actual_oos_period'][1]}`",
        f"- Interpreter: `{config['interpreter']['path']}` / `{config['interpreter']['version'].splitlines()[0]}`",
        f"- Historical rule truth: `{HISTORICAL_RULE_STATUS}`; publication timing: `{PUBLICATION_TIMING_STATUS}`.",
        "- Parameter search: `FORBIDDEN`; the C2 configuration was frozen before the run.",
        "",
        "## Frozen C2 rules",
        "",
        "- Core: 001512 25%, 000148 20%, 000218 15%, 260102 15%.",
        "- Satellite budget: 25%; semiannual June/December observation; maximum three satellites; each filled slot 8.3333%.",
        "- Existing eligible satellites are retained; ranking changes alone do not force a sale; unused slots go to 260102.",
        "- Signal: available-date total-return index, 126/252-day equal momentum, MA200 filter, domestic T+1 and QDII T+2 availability; no continuous volatility scaling; 7.5pp natural-drift trigger.",
        "",
        "## Metrics",
        "",
        "| Strategy | Net CAGR | Sharpe | MDD | Calmar | Max sustained confirmed turnover | Fees |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (C2_NAME, B2_NAME, B2LT_NAME):
        item = metrics[name]
        lines.append(
            f"| {name} | {item.get('net_cagr_pct', 0.0):.4f}% | {item.get('sharpe', 0.0):.4f} | {item.get('mdd_pct', 0.0):.4f}% | {item.get('calmar', 0.0):.4f} | {item.get('max_sustained_annual_confirmed_turnover', item.get('max_annual_bilateral_turnover', 0.0)):.6f} | {item.get('total_fee_amount', 0.0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Research performance Gate",
            "",
            f"- Research performance Gate passed: `{gate.get('research_performance_gate_passed')}`",
            f"- Research performance failed checks: `{gate.get('research_performance_failed_checks', [])}`",
            f"- B2 CAGR advantage: `{c2.get('relative_b2_cagr_advantage_pct_points', 0.0):.4f} pp`.",
            f"- Worst rolling two-year CAGR: `{c2.get('worst_two_year_cagr_pct')}`.",
            f"- 2021-2023: CAGR `{c2['subperiod_metrics']['2021_2023'].get('net_cagr_pct'):.4f}%`, Sharpe `{c2['subperiod_metrics']['2021_2023'].get('sharpe'):.4f}`.",
            f"- 2024-2026: CAGR `{c2['subperiod_metrics']['2024_2026'].get('net_cagr_pct'):.4f}%`, Sharpe `{c2['subperiod_metrics']['2024_2026'].get('sharpe'):.4f}`.",
            "",
            "## Observation readiness Gate",
            "",
            f"- Observation readiness Gate passed: `{gate.get('observation_readiness_gate_passed')}`",
            f"- Observation readiness blockers: `{gate.get('observation_readiness_failed_reasons', [])}`",
            "- Observation eligibility: `false`.",
            "",
            "## Pressure tests",
            "",
        ]
    )
    for name, stress in stresses.items():
        sm = stress["metrics"]
        lines.append(
            f"- `{name}`: CAGR `{sm.get('net_cagr_pct', 0.0):.4f}%`, Sharpe `{sm.get('sharpe', 0.0):.4f}`, MDD `{sm.get('mdd_pct', 0.0):.4f}%`, fees `{sm.get('total_fee_amount', 0.0):.2f}`, orders `{stress.get('order_count', 0)}`; `{stress.get('rule_override')}`."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "C2 is a reused research sample and is not fresh OOS or paper-trade eligible. The conservative lags reduce signal look-ahead risk but do not establish official NAV publication-time truth.",
            "Any failed threshold remains a hard research failure; no failed strategy enters observation.",
            f"- Input hashes: `{facts['input_hashes']}`",
            f"- Artifact root: `{run_dir.as_posix()}`",
        ]
    )
    (run_dir / "low_turnover_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research() -> int:
    config_source = _load_config()
    facts = _facts(config_source)
    rule_book = ProductRuleBook.from_csv(ROOT / RULES_PATH)
    schedule_engine = _create_engine(rule_book)
    valued_dates = pd.to_datetime(schedule_engine._trading_dates)
    valued_dates = valued_dates[(valued_dates >= pd.Timestamp(OOS_START)) & (valued_dates <= pd.Timestamp(OOS_END))]
    if valued_dates.empty:
        raise RuntimeError("C2_NO_VALUED_DATES")
    actual_period = [valued_dates.min().strftime("%Y-%m-%d"), valued_dates.max().strftime("%Y-%m-%d")]
    run_dir = Path(create_run_directory(str(ROOT / OUTPUT_DIR), "low_turnover_core_satellite"))
    run_id = run_dir.name
    freeze_payload = {"config": config_source, "actual_oos_period": actual_period, "strategy": C2_NAME}
    parameter_freeze_id = hashlib.sha256(json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    config = {
        **config_source,
        "run_id": run_id,
        "db_path": DB_PATH,
        "rules_path": RULES_PATH,
        "mapping_path": MAPPING_PATH,
        "config_source_path": CONFIG_PATH,
        "initial_cash": INITIAL_CASH,
        "actual_oos_period": actual_period,
        "account_mode": ACCOUNT_MODE,
        "rule_scenario": RULE_SCENARIO,
        "historical_rule_status": HISTORICAL_RULE_STATUS,
        "parameter_freeze_id": parameter_freeze_id,
        "interpreter": {"path": sys.executable, "version": sys.version},
    }
    facts["sample_label"] = config["sample_label"]
    facts["interpreter"] = {"path": sys.executable, "version": sys.version}

    all_dates = pd.to_datetime(schedule_engine._trading_dates)
    all_month_ends = build_month_end_schedule(all_dates, OOS_START, OOS_END)
    signal_dates = [date for date in all_month_ends if pd.Timestamp(date).month in {6, 12}]
    signal_map = build_signal_submit_map(all_dates, signal_dates, OOS_END)
    growth_df = schedule_engine._nav_df[["fund_code", "nav_date", "daily_growth_pct", "unit_nav"]].copy()
    trading_dates = pd.DatetimeIndex(all_dates)
    signal = LowTurnoverCoreSatelliteSignal(growth_df, trading_dates=trading_dates)
    targets, audits = _signal_targets(signal, signal_dates, signal_map)
    if targets.empty:
        raise RuntimeError("C2_NO_ACCEPTED_TARGET_SIGNALS")
    daily, orders, rejections, turnover, fees, extra = _run_account(schedule_engine, targets, signal_map)
    c2_metrics = _metrics(daily, orders, turnover, audits)
    c2_metrics["fee_reconciliation_passed"] = bool(extra["fee_summary"].get("passed"))
    c2_metrics["fee_reconciliation"] = extra["fee_summary"]
    static_gates = c1._data_gates(schedule_engine, facts, set(targets.columns))

    stress_results = {
        scenario: _run_stress(
            scenario, rule_book, growth_df, trading_dates, signal_dates, signal_map, config, run_dir
        )
        for scenario in ("double_fees", "delay_plus_one_trading_day")
    }

    b2_engine = _create_engine(rule_book)
    monthly_dates = c1.build_month_end_schedule(b2_engine._trading_dates, OOS_START, OOS_END)
    monthly_map = c1.build_signal_submit_map(b2_engine._trading_dates, monthly_dates, OOS_END)
    quarterly_dates = c1.build_quarter_end_schedule(b2_engine._trading_dates, OOS_START, OOS_END)
    quarterly_map = c1.build_signal_submit_map(b2_engine._trading_dates, quarterly_dates, OOS_END)
    b2_targets, b2_audits, _ = c1._build_benchmark(B2_NAME, b2_engine, monthly_dates, monthly_map, growth_df)
    b2_daily, b2_orders, b2_rejections, b2_turnover, b2_fees, b2_extra = _run_account(b2_engine, b2_targets, monthly_map)
    b2_metrics = c1._benchmark_metrics(b2_daily, b2_turnover, B2_NAME, b2_extra["fee_summary"])
    b2_gate = c1._benchmark_gate(b2_metrics, c1._data_gates(b2_engine, facts, set(b2_targets.columns)))
    b2lt_targets, b2lt_audits, _ = c1._build_benchmark(B2LT_NAME, b2_engine, quarterly_dates, quarterly_map, growth_df)
    b2lt_daily, b2lt_orders, b2lt_rejections, b2lt_turnover, b2lt_fees, b2lt_extra = _run_account(b2_engine, b2lt_targets, quarterly_map)
    b2lt_metrics = c1._benchmark_metrics(b2lt_daily, b2lt_turnover, B2LT_NAME, b2lt_extra["fee_summary"])
    b2lt_gate = c1._benchmark_gate(b2lt_metrics, c1._data_gates(b2_engine, facts, set(b2lt_targets.columns)))

    gate = _gate(c2_metrics, b2_metrics, stress_results["double_fees"]["metrics"], stress_results["delay_plus_one_trading_day"]["metrics"], static_gates, facts["c1_reference_hash_audit"])
    c2_dir = run_dir / C2_NAME
    parameter_freeze = {
        "mode": ACCOUNT_MODE,
        "strategy": C2_NAME,
        "parameter_freeze_id": parameter_freeze_id,
        "sample_label": config["sample_label"],
        "parameter_search": "FORBIDDEN",
        "frozen_rules": config_source,
    }
    c2_gate = _write_c2_bundle(c2_dir, daily, orders, rejections, turnover, fees, extra["fee_order_audit"], audits, schedule_engine, config, facts, c2_metrics, gate, parameter_freeze)
    c2_gate["checks"]["artifact_schema"] = bool(c2_gate.get("artifact_validation", {}).get("passed"))
    c2_gate["checks"]["artifact_content"] = bool(c2_gate.get("artifact_validation", {}).get("passed"))
    c2_gate["failed_checks"] = [name for name, value in c2_gate["checks"].items() if not value]
    c2_gate["gate_passed"] = not c2_gate["failed_checks"]
    _classify_gate_layers(c2_gate)
    c2_gate["candidate_gate_passed"] = False
    c2_gate["observation_eligible"] = False
    export_json(c2_dir.as_posix(), "gate_result.json", c2_gate)
    export_metrics(c2_dir.as_posix(), c2_metrics)
    # The gate is finalized after the generic bundle is exported; refresh the
    # manifest so its gate hash and artifact inventory describe the final files.
    write_manifest(run_dir=c2_dir.as_posix(), strategy_name=C2_NAME, config=config, db_path=str(ROOT / DB_PATH), rules_path=str(ROOT / RULES_PATH), exposure_mapping_path=str(ROOT / MAPPING_PATH), metrics=c2_metrics, gate_result=c2_gate)
    validation = validate_artifact_bundle(c2_dir, strategy_name=C2_NAME, expected_run_id=run_id, expected_start=actual_period[0], expected_end=actual_period[1])
    if not validation.get("passed"):
        raise RuntimeError(f"C2_ARTIFACT_VALIDATION_FAILED:{validation.get('errors')}")

    status = "RESEARCH_GATE_FAILED" if not c2_gate["gate_passed"] else "RESEARCH_GATE_PASSED_CURRENT_SNAPSHOT_ONLY"
    status_payload = {
        "run_id": run_id,
        "strategy": C2_NAME,
        "status": status,
        "sample_label": config["sample_label"],
        "observation_eligible": False,
        "research_performance_gate_passed": c2_gate["research_performance_gate_passed"],
        "research_performance_failed_checks": c2_gate["research_performance_failed_checks"],
        "observation_readiness_gate_passed": c2_gate["observation_readiness_gate_passed"],
        "observation_readiness_failed_reasons": c2_gate["observation_readiness_failed_reasons"],
        "historical_truth_gate": False,
        "historical_rule_status": HISTORICAL_RULE_STATUS,
        "publication_timing_status": PUBLICATION_TIMING_STATUS,
        "publication_timing_gate": False,
        "signal_point_in_time_truth": SIGNAL_PIT_TRUTH_STATUS,
        "publication_timing_audit": publication_timing_audit(),
        "rule_scenario": RULE_SCENARIO,
        "requested_oos_period": config["requested_oos_period"],
        "actual_oos_period": actual_period,
        "account_mode": ACCOUNT_MODE,
        "parameter_freeze_id": parameter_freeze_id,
        "interpreter": facts["interpreter"],
        "input_hashes": facts["input_hashes"],
        "metrics": {C2_NAME: c2_metrics, B2_NAME: b2_metrics, B2LT_NAME: b2lt_metrics},
        "gate_result": {C2_NAME: c2_gate, B2_NAME: b2_gate, B2LT_NAME: b2lt_gate},
        "stress_tests": stress_results,
        "benchmark_rerun": {"B2": B2_NAME, "B2LT": B2LT_NAME, "C1_reference": facts["c1_reference_hash_audit"]},
        "calendar_correction_label": "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "supersedes_run_id": "low_turnover_core_satellite_20260729_163519",
        "intermediate_calendar_corrected_run_id": "low_turnover_core_satellite_20260730_115448",
        "failed_research_thresholds": c2_gate["research_performance_failed_checks"],
        "failed_observation_readiness_reasons": c2_gate["observation_readiness_failed_reasons"],
        "disclosures": [
            "This sample interval was reused across multiple observations and is not fresh OOS.",
            "Historical rule dates are not established; only the current-snapshot conservative execution scenario is reported.",
            "NAV publication timestamps are unavailable; conservative domestic T+1 and QDII T+2 lags are applied but official PIT remains unverified.",
            "Signal point-in-time truth is NOT_ESTABLISHED.",
        ],
    }
    export_json(run_dir.as_posix(), "low_turnover_input_facts.json", facts)
    export_config_snapshot(run_dir.as_posix(), config)
    export_json(run_dir.as_posix(), "low_turnover_status.json", status_payload)
    export_json(run_dir.as_posix(), "gate_result.json", c2_gate)
    _write_conclusion(run_dir, config, facts, {C2_NAME: c2_metrics, B2_NAME: b2_metrics, B2LT_NAME: b2lt_metrics}, c2_gate, stress_results, status)
    write_manifest(run_dir=run_dir.as_posix(), strategy_name=C2_NAME, config=config, db_path=str(ROOT / DB_PATH), rules_path=str(ROOT / RULES_PATH), exposure_mapping_path=str(ROOT / MAPPING_PATH), metrics=c2_metrics, gate_result=c2_gate)
    print(json.dumps(status_payload, ensure_ascii=False, indent=2, default=str))
    return 0


def refresh_final_run(run_dir: Path) -> int:
    """Refresh only audit classification/report/manifest files for an existing run."""
    run_dir = run_dir.resolve()
    status_path = run_dir / "low_turnover_status.json"
    facts_path = run_dir / "low_turnover_input_facts.json"
    config_path = run_dir / C2_NAME / "config_snapshot.json"
    child_dir = run_dir / C2_NAME
    if not status_path.exists() or not facts_path.exists() or not child_dir.exists():
        raise FileNotFoundError(f"C2_REFRESH_RUN_NOT_FOUND:{run_dir}")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    correction_metadata = {
        "calendar_correction_label": "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "supersedes_run_id": "low_turnover_core_satellite_20260729_163519",
        "intermediate_calendar_corrected_run_id": "low_turnover_core_satellite_20260730_115448",
    }
    status.update(correction_metadata)
    facts.update(correction_metadata)
    config.update(correction_metadata)
    gate = json.loads((child_dir / "gate_result.json").read_text(encoding="utf-8"))
    gate["failed_checks"] = [
        name for name, value in gate.get("checks", {}).items() if not value
    ]
    gate["gate_passed"] = not gate["failed_checks"]
    _classify_gate_layers(gate)

    export_json(child_dir.as_posix(), "gate_result.json", gate)
    export_json(run_dir.as_posix(), "gate_result.json", gate)
    status.setdefault("gate_result", {})[C2_NAME] = gate
    status["research_performance_gate_passed"] = gate["research_performance_gate_passed"]
    status["research_performance_failed_checks"] = gate["research_performance_failed_checks"]
    status["observation_readiness_gate_passed"] = gate["observation_readiness_gate_passed"]
    status["observation_readiness_failed_reasons"] = gate["observation_readiness_failed_reasons"]
    status["failed_research_thresholds"] = gate["research_performance_failed_checks"]
    status["failed_observation_readiness_reasons"] = gate["observation_readiness_failed_reasons"]
    status["observation_eligible"] = False
    export_json(run_dir.as_posix(), "low_turnover_input_facts.json", facts)
    export_json(run_dir.as_posix(), "low_turnover_status.json", status)
    export_config_snapshot(run_dir.as_posix(), config)
    export_config_snapshot(child_dir.as_posix(), config)

    _write_conclusion(
        run_dir,
        config,
        facts,
        status["metrics"],
        gate,
        status.get("stress_tests", {}),
        status.get("status", "RESEARCH_GATE_FAILED"),
    )
    # Only manifests are refreshed after the audit JSON/report writes.  This
    # preserves all account, order, daily and metrics files byte-for-byte.
    write_manifest(
        run_dir=child_dir.as_posix(),
        strategy_name=C2_NAME,
        config=config,
        db_path=str(ROOT / DB_PATH),
        rules_path=str(ROOT / RULES_PATH),
        exposure_mapping_path=str(ROOT / MAPPING_PATH),
        metrics=status["metrics"][C2_NAME],
        gate_result=gate,
    )
    write_manifest(
        run_dir=run_dir.as_posix(),
        strategy_name=C2_NAME,
        config=config,
        db_path=str(ROOT / DB_PATH),
        rules_path=str(ROOT / RULES_PATH),
        exposure_mapping_path=str(ROOT / MAPPING_PATH),
        metrics=status["metrics"][C2_NAME],
        gate_result=gate,
    )
    print(json.dumps({
        "run_id": status["run_id"],
        "research_performance_gate_passed": gate["research_performance_gate_passed"],
        "research_performance_failed_checks": gate["research_performance_failed_checks"],
        "observation_readiness_gate_passed": gate["observation_readiness_gate_passed"],
        "observation_readiness_failed_reasons": gate["observation_readiness_failed_reasons"],
        "refreshed_files": [
            "gate_result.json",
            "C2_LOW_TURNOVER_CORE_SATELLITE/gate_result.json",
            "low_turnover_status.json",
            "low_turnover_conclusion.md",
            "manifest.json",
            "C2_LOW_TURNOVER_CORE_SATELLITE/manifest.json",
        ],
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run or refresh the frozen C2 research study")
    parser.add_argument("--refresh-run", type=Path, help="refresh only audit layers for an existing final run")
    args = parser.parse_args()
    if args.refresh_run is not None:
        return refresh_final_run(args.refresh_run)
    return run_research()


if __name__ == "__main__":
    raise SystemExit(main())
