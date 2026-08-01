"""Run the frozen C1 core/satellite momentum research study.

The run is independent from the canonical and candidate run directories.  It
reruns B2 and B2-LT with the same expanded OTF database and product-rule book,
then evaluates C1 under the frozen research Gate and two required stresses.
"""

from __future__ import annotations

import hashlib
import json
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otf_backtest_engine import OTFBacktestEngine
from otf_rotation.artifact_validation import validate_artifact_bundle
from otf_rotation.core_satellite_momentum import (
    CORE_WEIGHTS,
    COVARIANCE_DAYS,
    DRIFT_THRESHOLD,
    MIN_PUBLISHED_OBSERVATIONS,
    SATELLITE_BASE_WEIGHT,
    SATELLITE_BUDGET,
    SATELLITE_POOL,
    VOLATILITY_TARGET,
    CoreSatelliteMomentumSignal,
    clone_rule_book_with_delay_override,
    estimate_total_return_covariance,
    sustained_turnover_excluding_initial,
)
from otf_rotation.experiment_artifacts import (
    create_run_directory,
    export_config_snapshot,
    export_input_hashes,
    export_json,
    export_metrics,
    export_table,
    sha256_file,
    write_manifest,
)
from otf_rotation.execution_calendar import load_execution_calendar
from otf_rotation.schedule import (
    build_month_end_schedule,
    build_quarter_end_schedule,
    build_signal_submit_map,
)
from otf_trading_rules import ProductRuleBook
import run_b1_b2_b3_walkforward as canonical
from run_candidate_strategies import B2LTSignal, build_candidate_targets


DB_PATH = "data/processed/otf_expanded.sqlite"
CALENDAR_PATH = "data/processed/execution_calendar/cn_execution_calendar.csv"
RULES_PATH = "config/otf_product_rules.csv"
MAPPING_PATH = "config/otf_exposure_mapping.csv"
CONFIG_PATH = "config/core_satellite_momentum.json"
OUTPUT_DIR = Path("reports/strategy_research/core_satellite")
INITIAL_CASH = 1_000_000.0
ACCOUNT_MODE = "FROZEN_PARAMETER_CONTINUOUS_OOS"
RULE_SCENARIO = "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO"
HISTORICAL_RULE_STATUS = "NOT_ESTABLISHED"
OOS_START = "2021-01-04"
OOS_END = "2026-07-27"
C1_NAME = "C1_CORE_SATELLITE_MOMENTUM"
B2_NAME = "B2_Static_EW_4Asset"
B2LT_NAME = "B2_LT_Static_EW_4Asset"
PUBLICATION_TIMING_STATUS = "NAV_PUBLICATION_TIMESTAMP_NOT_AVAILABLE"
SIGNAL_PIT_TRUTH_STATUS = "NOT_ESTABLISHED"


def publication_timing_audit() -> dict[str, Any]:
    """Return the immutable disclosure for missing NAV publication timestamps."""
    return {
        "status": PUBLICATION_TIMING_STATUS,
        "signal_point_in_time_truth": SIGNAL_PIT_TRUTH_STATUS,
        "publication_timestamp_column": None,
        "publication_timestamp_available": False,
        "nav_date_filter_is_verified_pit": False,
        "affected_or_potentially_affected_products": ["050025", "OTHER_CROSS_BORDER_FUNDS"],
        "reason": (
            "The source table contains nav_date and daily_growth_pct but no publication_date "
            "or publication timestamp. A NAV dated t may be disclosed at t+1/T+2 for cross-border/QDII funds."
        ),
    }


def _load_config() -> dict[str, Any]:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


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
        "rule_counts_by_status": dict(
            sorted(rules["rule_status"].astype(str).str.strip().value_counts().items())
        ),
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
        "supersedes_run_id": "core_satellite_20260729_151927",
        "intermediate_calendar_corrected_run_id": "core_satellite_20260730_114756",
        "execution_calendar": load_execution_calendar(str(ROOT / CALENDAR_PATH)).facts(),
        "publication_timing_audit": publication_timing_audit(),
        "config": config,
    }


def _create_engine(rule_book: ProductRuleBook) -> OTFBacktestEngine:
    return canonical.create_engine(rule_book)


def _signal_targets(
    signal: Any,
    signal_dates: list[pd.Timestamp],
    signal_map: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    submit_for_signal = {str(signal_date): str(submit) for submit, signal_date in signal_map.items()}
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    audits: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        date = pd.Timestamp(signal_date)
        submit = submit_for_signal.get(date.strftime("%Y-%m-%d"))
        if submit is None:
            continue
        weights = signal(date) or {}
        audit = getattr(signal, "last_signal_audit", None)
        if audit:
            row = dict(audit)
            row["submit_date"] = submit
            audits.append(row)
        if not weights:
            continue
        target = {
            str(code): float(weight)
            for code, weight in weights.items()
            if float(weight) > 1e-12
        }
        if target:
            rows[pd.Timestamp(submit)] = target
    if not rows:
        return pd.DataFrame(), audits
    target_frame = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    target_frame.index = pd.to_datetime(target_frame.index)
    target_frame = target_frame.sort_index()
    return target_frame.loc[:, (target_frame != 0).any(axis=0)], audits


def _run_account(
    engine: OTFBacktestEngine,
    targets: pd.DataFrame,
    signal_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    daily = engine.run_backtest(
        targets,
        start=OOS_START,
        end=OOS_END,
        rebalance_every=1,
        signal_dates={pd.Timestamp(key): pd.Timestamp(value) for key, value in signal_map.items()},
    )
    orders = engine.order_audit_frame()
    rejections = (
        engine.last_rejections.copy()
        if not engine.last_rejections.empty
        else pd.DataFrame(columns=["fund_code", "side", "date", "reason"])
    )
    turnover = canonical.compute_turnover(daily, orders)
    fee_reconciliation, fee_order_audit, fee_summary = canonical.build_fee_reconciliation(
        daily, orders
    )
    return daily, orders, rejections, turnover, fee_reconciliation, {
        "fee_order_audit": fee_order_audit,
        "fee_summary": fee_summary,
    }


def _calmar(metrics: dict[str, Any]) -> float:
    cagr = float(metrics.get("net_cagr_pct", 0.0))
    mdd = abs(float(metrics.get("mdd_pct", 0.0)))
    return round(cagr / mdd, 4) if mdd > 1e-12 else float("inf")


def _subperiod_metrics(daily: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    frame = daily[
        (pd.to_datetime(daily["date"]) >= pd.Timestamp(start))
        & (pd.to_datetime(daily["date"]) <= pd.Timestamp(end))
    ].copy()
    metrics = canonical.compute_metrics(frame)
    metrics["calmar"] = _calmar(metrics)
    return metrics


def _c1_metrics(
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    turnover: pd.DataFrame,
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = canonical.compute_metrics(daily, turnover)
    metrics["calmar"] = _calmar(metrics)
    first_build = next(
        (row for row in audits if row.get("decision") == "FIRST_BUILD" and row.get("rebalance")),
        None,
    )
    initial_signal_date = (
        first_build.get("signal_date") if first_build else audits[0].get("signal_date")
    )
    sustained = sustained_turnover_excluding_initial(daily, orders, initial_signal_date)
    subperiods = {
        "2021_2023": _subperiod_metrics(daily, "2021-01-04", "2023-12-31"),
        "2024_2026": _subperiod_metrics(daily, "2024-01-01", OOS_END),
    }
    metrics.update(
        {
            "strategy": C1_NAME,
            "confirmed_turnover_excludes_initial_build": True,
            "initial_build_signal_date": pd.Timestamp(initial_signal_date).strftime("%Y-%m-%d"),
            "sustained_annual_confirmed_turnover": sustained["annual_confirmed_turnover"],
            "max_sustained_annual_confirmed_turnover": sustained["max_annual_confirmed_turnover"],
            "initial_build_orders_excluded": sustained["excluded_initial_order_count"],
            "subperiod_metrics": subperiods,
            "accepted_target_count": int(sum(bool(row.get("rebalance")) for row in audits)),
            "observation_count": int(len(audits)),
            "no_rebalance_observation_count": int(sum(not bool(row.get("rebalance")) for row in audits)),
            "fee_reconciliation_passed": bool(
                metrics.get("fee_reconciliation_passed", False)
            ),
        }
    )
    return metrics


def _benchmark_metrics(
    daily: pd.DataFrame, turnover: pd.DataFrame, name: str, fee_summary: dict[str, Any]
) -> dict[str, Any]:
    metrics = canonical.compute_metrics(daily, turnover)
    metrics["strategy"] = name
    metrics["calmar"] = _calmar(metrics)
    metrics["fee_reconciliation_passed"] = bool(fee_summary.get("passed"))
    metrics["fee_reconciliation"] = fee_summary
    metrics["confirmed_turnover_gate"] = bool(
        metrics.get("max_annual_bilateral_turnover", np.inf)
        <= canonical.GATE_THRESHOLDS["annual_bilateral_turnover_max"]
    )
    return metrics


def _used_funds_from_targets(targets: pd.DataFrame) -> set[str]:
    return {str(code) for code in targets.columns}


def _data_gates(
    engine: OTFBacktestEngine, facts: dict[str, Any], used_funds: set[str]
) -> dict[str, Any]:
    gates = canonical.build_static_data_gates(engine, facts, used_funds)
    gates["historical_rule_status"] = HISTORICAL_RULE_STATUS
    gates["rule_scenario"] = RULE_SCENARIO
    return gates


def _write_c1_audit_tables(
    strategy_dir: Path,
    audits: list[dict[str, Any]],
    growth_df: pd.DataFrame,
) -> None:
    state_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    fund_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for audit in audits:
        date = audit.get("signal_date")
        selected = list(audit.get("selected_satellites", []))
        evaluations = audit.get("evaluations", [])
        summary = {
            "signal_date": date,
            "market_state": "SATELLITE_MOMENTUM",
            "submit_date": audit.get("submit_date"),
            "rebalance": audit.get("rebalance", False),
            "decision": audit.get("decision", ""),
            "selection_changed": audit.get("selection_changed", False),
            "selected_satellites": ";".join(selected),
            "covariance_coverage": audit.get("covariance_coverage", 0),
            "covariance_start": audit.get("covariance_start"),
            "covariance_end": audit.get("covariance_end"),
            "predicted_volatility_pct": audit.get("predicted_volatility_pct"),
            "satellite_scale": audit.get("satellite_scale", 0.0),
            "max_drift_deviation_pct_points": audit.get("max_drift_deviation_pct_points", 0.0),
            "drift_weights": json.dumps(audit.get("drift_weights", {}), ensure_ascii=False, sort_keys=True, default=str),
            "proposed_target_weights": json.dumps(audit.get("proposed_target_weights", {}), ensure_ascii=False, sort_keys=True, default=str),
            "accepted_target_weights": json.dumps(audit.get("target_weights", {}), ensure_ascii=False, sort_keys=True, default=str),
            "cash_transfer": audit.get("cash_transfer", 0.0),
            "cash_transfer_reasons": json.dumps(audit.get("cash_transfer_reasons", {}), ensure_ascii=False, sort_keys=True),
        }
        # A market state row is also the period-level signal audit summary.
        state_rows.append(summary)
        buffer_audit = audit.get("buffer_audit", {})
        for evaluation in evaluations:
            code = str(evaluation.get("fund_code", ""))
            buffer = buffer_audit.get(code, {})
            score_rows.append(
                {
                    "date": date,
                    "state": "ELIGIBLE" if evaluation.get("eligible") else evaluation.get("reason", "INELIGIBLE"),
                    "score": evaluation.get("momentum_score") or 0.0,
                    "fund_code": code,
                    "total_return_126": evaluation.get("total_return_126"),
                    "total_return_252": evaluation.get("total_return_252"),
                    "ma200": evaluation.get("ma200"),
                    "published_observations": evaluation.get("published_observations"),
                    "latest_published_date": evaluation.get("latest_published_date"),
                    "trend_condition": evaluation.get("trend_condition"),
                    "momentum_condition": evaluation.get("momentum_condition"),
                    "eligible": evaluation.get("eligible"),
                    "rank": buffer.get("rank"),
                    "retained_by_buffer": buffer.get("retained_by_buffer", False),
                    "buffer_margin_pct_points": buffer.get("buffer_margin_pct_points"),
                    "selected": buffer.get("selected", False),
                    "selection_reason": evaluation.get("reason"),
                    "covariance_coverage": audit.get("covariance_coverage", 0),
                    "predicted_volatility_pct": audit.get("predicted_volatility_pct"),
                    "satellite_scale": audit.get("satellite_scale", 0.0),
                    "drift_weight": audit.get("drift_weights", {}).get(code, 0.0),
                    "drift_deviation_pct_points": audit.get("max_drift_deviation_pct_points", 0.0),
                    "rebalance": audit.get("rebalance", False),
                    "rebalance_reason": audit.get("decision", ""),
                }
            )
            selection_rows.append(
                {
                    "sleeve": "satellite",
                    "date": date,
                    "top_n": 2,
                    "selected_count": len(selected),
                    "total_candidates": len(SATELLITE_POOL),
                    "eligible_count": sum(bool(item.get("eligible")) for item in evaluations),
                    "selected_funds": ";".join(selected),
                    "fund_code": code,
                    "published_observations": evaluation.get("published_observations"),
                    "total_return_126": evaluation.get("total_return_126"),
                    "total_return_252": evaluation.get("total_return_252"),
                    "ma200": evaluation.get("ma200"),
                    "momentum_score": evaluation.get("momentum_score"),
                    "eligible": evaluation.get("eligible"),
                    "rank": buffer.get("rank"),
                    "retained_by_buffer": buffer.get("retained_by_buffer", False),
                    "buffer_margin_pct_points": buffer.get("buffer_margin_pct_points"),
                    "selected": buffer.get("selected", False),
                    "covariance_coverage": audit.get("covariance_coverage", 0),
                    "predicted_volatility_pct": audit.get("predicted_volatility_pct"),
                    "satellite_scale": audit.get("satellite_scale", 0.0),
                    "drift_deviation_pct_points": audit.get("max_drift_deviation_pct_points", 0.0),
                    "rebalance": audit.get("rebalance", False),
                    "rebalance_reason": audit.get("decision", ""),
                    "cash_transfer_reason": audit.get("cash_transfer_reasons", {}).get(code, ""),
                }
            )
        for code, weight in audit.get("proposed_target_weights", {}).items():
            sleeve = "core" if code in CORE_WEIGHTS else "satellite" if code in SATELLITE_POOL else "cash_fallback"
            budget_rows.append({"date": date, "sleeve": sleeve, "weight": weight, "fund_code": code})
            sleeve_rows.append({"date": date, "sleeve": sleeve, "weight": weight, "fund_code": code})
            fund_rows.append(
                {
                    "date": date,
                    "fund_code": code,
                    "weight": weight,
                    "accepted_target": code in audit.get("target_weights", {}),
                    "selected_satellite": code in selected,
                    "satellite_scale": audit.get("satellite_scale", 0.0),
                }
            )
        cov, cov_audit = estimate_total_return_covariance(
            growth_df,
            list(audit.get("proposed_target_weights", {})),
            pd.Timestamp(date),
            window=COVARIANCE_DAYS,
        )
        proposed = audit.get("proposed_target_weights", {})
        if cov is not None and proposed:
            codes = [code for code in proposed if code in cov.index]
            vector = np.array([float(proposed[code]) for code in codes])
            matrix = cov.loc[codes, codes].to_numpy(dtype=float)
            volatility = float(np.sqrt(max(0.0, vector @ matrix @ vector)))
            marginal = matrix @ vector
            for code, weight, marginal_value in zip(codes, vector, marginal):
                contribution = max(0.0, float(weight * marginal_value / volatility)) if volatility > 0 else 0.0
                risk_rows.append(
                    {
                        "date": date,
                        "fund_code": code,
                        "risk_contribution": contribution,
                        "covariance_coverage": cov_audit.get("common_observations", 0),
                        "predicted_volatility_pct": volatility * 100.0,
                        "satellite_scale": audit.get("satellite_scale", 0.0),
                    }
                )
        else:
            for code, weight in proposed.items():
                risk_rows.append(
                    {
                        "date": date,
                        "fund_code": code,
                        "risk_contribution": 0.0,
                        "covariance_coverage": audit.get("covariance_coverage", 0),
                        "predicted_volatility_pct": audit.get("predicted_volatility_pct"),
                        "satellite_scale": audit.get("satellite_scale", 0.0),
                    }
                )
    export_table(strategy_dir.as_posix(), "market_states.csv", pd.DataFrame(state_rows))
    export_table(strategy_dir.as_posix(), "state_scores.csv", pd.DataFrame(score_rows))
    export_table(strategy_dir.as_posix(), "asset_budgets.csv", pd.DataFrame(budget_rows))
    export_table(strategy_dir.as_posix(), "sleeve_weights.csv", pd.DataFrame(sleeve_rows))
    export_table(strategy_dir.as_posix(), "fund_weights.csv", pd.DataFrame(fund_rows))
    export_table(strategy_dir.as_posix(), "product_selection_audit.csv", pd.DataFrame(selection_rows))
    export_table(strategy_dir.as_posix(), "risk_contributions.csv", pd.DataFrame(risk_rows))


def _write_bundle(
    strategy_dir: Path,
    strategy_name: str,
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
    canonical.export_strategy_bundle(
        strategy_dir,
        strategy_name,
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
    if strategy_name == C1_NAME:
        _write_c1_audit_tables(strategy_dir, audits, engine._nav_df)
    write_manifest(
        run_dir=strategy_dir.as_posix(),
        strategy_name=strategy_name,
        config=config,
        db_path=str(ROOT / DB_PATH),
        rules_path=str(ROOT / RULES_PATH),
        exposure_mapping_path=str(ROOT / MAPPING_PATH),
        metrics=metrics,
        gate_result=gate,
    )
    artifact_validation = canonical.finalize_artifact_gate(
        strategy_dir,
        strategy_name,
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
    signal_dates: list[pd.Timestamp],
    signal_map: dict[str, str],
    base_config: dict[str, Any],
    run_dir: Path,
    trading_dates: pd.DatetimeIndex | None = None,
) -> dict[str, Any]:
    if scenario == "double_fees":
        stressed_rules = rule_book.scaled_fees(2.0)
        audit = {
            "scenario": scenario,
            "subscription_fee_multiplier": 2.0,
            "redemption_fee_multiplier": 2.0,
            "rule_override": "ProductRuleBook.scaled_fees(2.0)",
        }
    elif scenario == "delay_plus_one_trading_day":
        used = list(CORE_WEIGHTS) + list(SATELLITE_POOL)
        stressed_rules = clone_rule_book_with_delay_override(rule_book, used, increment=1)
        audit = {
            "scenario": scenario,
            "subscription_confirmation_days_increment": 1,
            "redemption_confirmation_days_increment": 1,
            "redemption_arrival_days_increment": 1,
            "rule_override": "cloned_actual_hit_product_rules_plus_one",
            "actual_hit_rule_codes": sorted(
                code for code in used if rule_book.rule_for(code) is not None
            ),
            "effective_delays": {
                code: {
                    "baseline_subscription_confirmation_days": rule_book.rule_for(code).subscription_confirmation_days,
                    "stressed_subscription_confirmation_days": stressed_rules.rule_for(code).subscription_confirmation_days,
                    "baseline_redemption_confirmation_days": rule_book.rule_for(code).redemption_confirmation_days,
                    "stressed_redemption_confirmation_days": stressed_rules.rule_for(code).redemption_confirmation_days,
                    "baseline_redemption_arrival_days": rule_book.rule_for(code).redemption_settlement_days,
                    "stressed_redemption_arrival_days": stressed_rules.rule_for(code).redemption_settlement_days,
                }
                for code in used
                if rule_book.rule_for(code) is not None
            },
        }
    else:
        raise ValueError(f"unknown_c1_stress:{scenario}")
    engine = _create_engine(stressed_rules)
    signal = CoreSatelliteMomentumSignal(growth_df, trading_dates=trading_dates)
    targets, audits = _signal_targets(signal, signal_dates, signal_map)
    daily, orders, rejections, turnover, fees, extra = _run_account(engine, targets, signal_map)
    metrics = _c1_metrics(daily, orders, turnover, audits)
    metrics["fee_reconciliation_passed"] = bool(extra["fee_summary"].get("passed"))
    metrics["fee_reconciliation"] = extra["fee_summary"]
    audit.update(
        {
            "metrics": metrics,
            "order_count": int(len(orders)),
            "fee_total": float(orders["fee_paid"].sum()) if not orders.empty else 0.0,
            "confirmation_dates": (
                orders["confirmation_date"].dropna().astype(str).tolist()
                if not orders.empty else []
            ),
            "redemption_arrival_dates": (
                orders["redemption_arrival_date"].dropna().astype(str).tolist()
                if not orders.empty else []
            ),
        }
    )
    pressure_dir = run_dir / "pressure_tests" / scenario
    pressure_dir.mkdir(parents=True, exist_ok=True)
    export_config_snapshot(
        pressure_dir.as_posix(),
        {**base_config, "run_id": f"{run_dir.name}_{scenario}", "stress_scenario": scenario},
    )
    export_table(pressure_dir.as_posix(), "daily_account.csv", daily)
    export_table(pressure_dir.as_posix(), "orders.csv", orders)
    export_table(pressure_dir.as_posix(), "fees.csv", fees)
    export_table(pressure_dir.as_posix(), "turnover.csv", turnover)
    export_metrics(pressure_dir.as_posix(), metrics)
    export_json(pressure_dir.as_posix(), "stress_audit.json", audit)
    del engine
    gc.collect()
    return audit


def _build_benchmark(
    name: str,
    engine: OTFBacktestEngine,
    signal_dates: list[pd.Timestamp],
    signal_map: dict[str, str],
    nav_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    if name == B2_NAME:
        def static_signal(_date: pd.Timestamp) -> dict[str, float]:
            return canonical.static_ew_4asset_signal(_date)

        targets, audits = _signal_targets(static_signal, signal_dates, signal_map)
    else:
        signal = B2LTSignal(nav_df, threshold=0.05)
        targets, audits = build_candidate_targets(signal, signal_dates, signal_map)
    return targets, audits, []


def _benchmark_gate(
    metrics: dict[str, Any], static_gates: dict[str, Any]
) -> dict[str, Any]:
    gate = canonical.metrics_gate(metrics, static_gates)
    gate["benchmark_only"] = True
    return gate


def _write_conclusion(
    run_dir: Path,
    config: dict[str, Any],
    facts: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    c1_gate: dict[str, Any],
    stresses: dict[str, Any],
    status: str,
) -> None:
    c1 = metrics[C1_NAME]
    lines = [
        "# C1 Core/Satellite Momentum Research",
        "",
        f"- Run ID: `{config['run_id']}`",
        f"- Status: `{status}`",
        f"- Sample label: `{config['sample_label']}`",
        f"- OOS interval: `{config['actual_oos_period'][0]}` to `{config['actual_oos_period'][1]}`",
        f"- Rule scenario: `{RULE_SCENARIO}`",
        f"- Interpreter: `{config['interpreter']['path']}` / `{config['interpreter']['version'].splitlines()[0]}`",
        "- Historical rule truth: `FAIL` / `NOT_ESTABLISHED`; this is only the current-snapshot conservative execution scenario.",
        f"- NAV publication timing: `{PUBLICATION_TIMING_STATUS}`; signal point-in-time truth: `{SIGNAL_PIT_TRUTH_STATUS}`.",
        "- Parameter search: `FORBIDDEN`; all C1 parameters were frozen once before the run.",
        "",
        "## Core / satellite frozen rules",
        "",
        "- Core: 001512 15%, 000148 10%, 000218 15%, 260102 10%.",
        "- Satellite pool: 160706, 000008, 007466, 050021, 050025, 000071; quarter-end signal, next-trading-day submission.",
        "- Total-return index uses only signal-date-or-earlier `daily_growth_pct`; 126/252 return score, 200-day SMA trend filter, Top2, fixed 3pp buffer, 60-day covariance, [0,1] bisection scale and 5pp rebalance boundary.",
        "",
        "## Metrics",
        "",
        "| Strategy | Net CAGR | Sharpe | MDD | Calmar | Max sustained confirmed turnover | Fees |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (C1_NAME, B2_NAME, B2LT_NAME):
        item = metrics[name]
        lines.append(
            f"| {name} | {item.get('net_cagr_pct', 0.0):.4f}% | {item.get('sharpe', 0.0):.4f} | {item.get('mdd_pct', 0.0):.4f}% | {item.get('calmar', 0.0):.4f} | {item.get('max_sustained_annual_confirmed_turnover', item.get('max_annual_bilateral_turnover', 0.0)):.6f} | {item.get('total_fee_amount', 0.0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## C1 Gate",
            "",
            f"- Gate passed: `{c1_gate.get('gate_passed')}`",
            f"- Failed thresholds: `{c1_gate.get('failed_checks', []) or 'none'}`",
            f"- B2 CAGR advantage: `{c1.get('relative_b2_cagr_advantage_pct_points', 0.0):.4f} pp`.",
            f"- Worst rolling two-year CAGR: `{c1.get('worst_two_year_cagr_pct')}`.",
            f"- Subperiod 2021-2023: CAGR `{c1['subperiod_metrics']['2021_2023'].get('net_cagr_pct'):.4f}%`, Sharpe `{c1['subperiod_metrics']['2021_2023'].get('sharpe'):.4f}`.",
            f"- Subperiod 2024-2026: CAGR `{c1['subperiod_metrics']['2024_2026'].get('net_cagr_pct'):.4f}%`, Sharpe `{c1['subperiod_metrics']['2024_2026'].get('sharpe'):.4f}`.",
            "",
            "## Pressure tests",
            "",
        ]
    )
    for name, stress in stresses.items():
        sm = stress["metrics"]
        lines.append(
            f"- `{name}`: CAGR `{sm.get('net_cagr_pct', 0.0):.4f}%`, Sharpe `{sm.get('sharpe', 0.0):.4f}`, MDD `{sm.get('mdd_pct', 0.0):.4f}%`, fees `{sm.get('total_fee_amount', 0.0):.2f}`, orders `{stress.get('order_count', 0)}`. Audit: `{stress.get('rule_override')}`."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "C1 is not a fresh OOS result and cannot be called a paper-trade candidate. NAV publication timing is not available, so `nav_date <= signal_date` is not a verified publication-time point-in-time filter; signal point-in-time truth remains `NOT_ESTABLISHED`.",
            "Any failed research threshold is recorded as `RESEARCH_GATE_FAILED`; no failed strategy enters observation.",
            f"- Input hashes: `{facts['input_hashes']}`",
            f"- Artifact root: `{run_dir.as_posix()}`",
        ]
    )
    (run_dir / "core_satellite_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _c1_gate(
    metrics: dict[str, Any],
    b2_metrics: dict[str, Any],
    double_fee_metrics: dict[str, Any],
    delay_metrics: dict[str, Any],
    static_gates: dict[str, Any],
) -> dict[str, Any]:
    thresholds = _load_config()["gate_thresholds"]
    metrics["relative_b2_cagr_advantage_pct_points"] = round(
        metrics["net_cagr_pct"] - b2_metrics["net_cagr_pct"], 4
    )
    delay_decline = b2_metrics.get("net_cagr_pct", 0.0)  # overwritten below
    delay_decline = metrics["net_cagr_pct"] - delay_metrics.get("net_cagr_pct", 0.0)
    sub_1 = metrics["subperiod_metrics"]["2021_2023"]
    sub_2 = metrics["subperiod_metrics"]["2024_2026"]
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
        "double_fee_calmar": _calmar(double_fee_metrics) >= thresholds["double_fee_calmar_min"],
        "delay_cagr_decline": delay_decline <= thresholds["delay_cagr_decline_max_pct_points"],
        "delay_mdd": delay_metrics.get("mdd_pct", -np.inf) > thresholds["delay_mdd_min_pct_exclusive"],
        "subperiod_2021_2023": sub_1.get("net_cagr_pct", -np.inf) > thresholds["subperiod_net_cagr_positive"] and sub_1.get("sharpe", -np.inf) >= thresholds["subperiod_sharpe_min"],
        "subperiod_2024_2026": sub_2.get("net_cagr_pct", -np.inf) > thresholds["subperiod_net_cagr_positive"] and sub_2.get("sharpe", -np.inf) >= thresholds["subperiod_sharpe_min"],
        "data_gate": bool(static_gates.get("data_gate")),
        "rules_gate": bool(static_gates.get("rules_gate")),
        "mapping_gate": bool(static_gates.get("mapping_gate")),
        "fee_reconciliation": bool(metrics.get("fee_reconciliation_passed")),
        "publication_timing_gate": False,
        "artifact_schema": False,
        "artifact_content": False,
    }
    return {
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
    }


def _run_research() -> int:
    config_source = _load_config()
    facts = _facts(config_source)
    rule_book = ProductRuleBook.from_csv(ROOT / RULES_PATH)
    schedule_engine = _create_engine(rule_book)
    valued_dates = pd.to_datetime(schedule_engine._trading_dates)
    valued_dates = valued_dates[
        (valued_dates >= pd.Timestamp(OOS_START)) & (valued_dates <= pd.Timestamp(OOS_END))
    ]
    if valued_dates.empty:
        raise RuntimeError("C1_NO_VALUED_DATES")
    actual_period = [valued_dates.min().strftime("%Y-%m-%d"), valued_dates.max().strftime("%Y-%m-%d")]
    run_dir = Path(create_run_directory(str(ROOT / OUTPUT_DIR), "core_satellite"))
    run_id = run_dir.name
    freeze_payload = {"config": config_source, "actual_oos_period": actual_period, "strategy": C1_NAME}
    parameter_freeze_id = hashlib.sha256(
        json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
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
        "calendar_year_reporting_slices": True,
        "interpreter": {"path": sys.executable, "version": sys.version},
    }
    facts["sample_label"] = config["sample_label"]
    facts["interpreter"] = {"path": sys.executable, "version": sys.version}

    quarterly_dates = build_quarter_end_schedule(schedule_engine._trading_dates, OOS_START, OOS_END)
    quarterly_map = build_signal_submit_map(schedule_engine._trading_dates, quarterly_dates, OOS_END)
    monthly_dates = build_month_end_schedule(schedule_engine._trading_dates, OOS_START, OOS_END)
    monthly_map = build_signal_submit_map(schedule_engine._trading_dates, monthly_dates, OOS_END)
    growth_df = schedule_engine._nav_df[
        ["fund_code", "nav_date", "daily_growth_pct", "unit_nav"]
    ].copy()
    c1_signal = CoreSatelliteMomentumSignal(
        growth_df, trading_dates=pd.to_datetime(schedule_engine._trading_dates)
    )
    c1_targets, c1_audits = _signal_targets(c1_signal, quarterly_dates, quarterly_map)
    if c1_targets.empty:
        raise RuntimeError("C1_NO_ACCEPTED_TARGET_SIGNALS")
    # Reuse the schedule engine's immutable NAV cache for the base C1 account;
    # this avoids a second 1.2M-row SQLite load before stress testing.
    c1_engine = schedule_engine
    c1_daily, c1_orders, c1_rejections, c1_turnover, c1_fees, c1_extra = _run_account(
        c1_engine, c1_targets, quarterly_map
    )
    c1_metrics = _c1_metrics(c1_daily, c1_orders, c1_turnover, c1_audits)
    c1_metrics["fee_reconciliation_passed"] = bool(c1_extra["fee_summary"].get("passed"))
    c1_metrics["fee_reconciliation"] = c1_extra["fee_summary"]
    c1_metrics["confirmed_turnover_gate"] = bool(
        c1_metrics.get("max_sustained_annual_confirmed_turnover", np.inf) < 0.80
    )
    c1_static_gates = _data_gates(c1_engine, facts, _used_funds_from_targets(c1_targets) | set(CORE_WEIGHTS) | set(SATELLITE_POOL))

    stress_results = {
        scenario: _run_stress(
            scenario,
            rule_book,
            growth_df,
            quarterly_dates,
            quarterly_map,
            config,
            run_dir,
            trading_dates=pd.to_datetime(schedule_engine._trading_dates),
        )
        for scenario in ("double_fees", "delay_plus_one_trading_day")
    }

    b2_engine = _create_engine(rule_book)
    b2_targets, b2_audits, _ = _build_benchmark(B2_NAME, b2_engine, monthly_dates, monthly_map, growth_df)
    b2_daily, b2_orders, b2_rejections, b2_turnover, b2_fees, b2_extra = _run_account(b2_engine, b2_targets, monthly_map)
    b2_metrics = _benchmark_metrics(b2_daily, b2_turnover, B2_NAME, b2_extra["fee_summary"])
    b2_gate = _benchmark_gate(b2_metrics, _data_gates(b2_engine, facts, _used_funds_from_targets(b2_targets)))

    # B2 and B2-LT share the same base execution engine sequentially; each
    # run_backtest call resets account state while preserving the same data and
    # rule-book execution semantics.
    b2lt_engine = b2_engine
    b2lt_targets, b2lt_audits, _ = _build_benchmark(B2LT_NAME, b2lt_engine, quarterly_dates, quarterly_map, growth_df)
    b2lt_daily, b2lt_orders, b2lt_rejections, b2lt_turnover, b2lt_fees, b2lt_extra = _run_account(b2lt_engine, b2lt_targets, quarterly_map)
    b2lt_metrics = _benchmark_metrics(b2lt_daily, b2lt_turnover, B2LT_NAME, b2lt_extra["fee_summary"])
    b2lt_gate = _benchmark_gate(b2lt_metrics, _data_gates(b2lt_engine, facts, _used_funds_from_targets(b2lt_targets)))

    c1_gate = _c1_gate(
        c1_metrics,
        b2_metrics,
        stress_results["double_fees"]["metrics"],
        stress_results["delay_plus_one_trading_day"]["metrics"],
        c1_static_gates,
    )
    c1_freeze = {
        "mode": ACCOUNT_MODE,
        "strategy": C1_NAME,
        "parameter_freeze_id": parameter_freeze_id,
        "sample_label": config["sample_label"],
        "parameter_search": "FORBIDDEN",
        "frozen_rules": config_source,
    }
    benchmark_freeze = {
        "mode": ACCOUNT_MODE,
        "strategy": "BENCHMARK_RERUN",
        "parameter_freeze_id": parameter_freeze_id,
        "sample_label": config["sample_label"],
        "benchmark_execution_rerun": True,
    }
    c1_dir = run_dir / C1_NAME
    b2_dir = run_dir / B2_NAME
    b2lt_dir = run_dir / B2LT_NAME
    c1_gate = _write_bundle(c1_dir, C1_NAME, c1_daily, c1_orders, c1_rejections, c1_turnover, c1_fees, c1_extra["fee_order_audit"], c1_audits, c1_engine, config, facts, c1_metrics, c1_gate, c1_freeze)
    b2_gate = _write_bundle(b2_dir, B2_NAME, b2_daily, b2_orders, b2_rejections, b2_turnover, b2_fees, b2_extra["fee_order_audit"], b2_audits, b2_engine, config, facts, b2_metrics, b2_gate, benchmark_freeze)
    b2lt_gate = _write_bundle(b2lt_dir, B2LT_NAME, b2lt_daily, b2lt_orders, b2lt_rejections, b2lt_turnover, b2lt_fees, b2lt_extra["fee_order_audit"], b2lt_audits, b2lt_engine, config, facts, b2lt_metrics, b2lt_gate, benchmark_freeze)

    # finalise C1 research checks after artifact validation has been applied
    c1_gate["checks"]["artifact_schema"] = bool(c1_gate.get("artifact_validation", {}).get("passed"))
    c1_gate["checks"]["artifact_content"] = bool(c1_gate.get("artifact_validation", {}).get("passed"))
    c1_gate["failed_checks"] = [name for name, value in c1_gate["checks"].items() if not value]
    c1_gate["gate_passed"] = not c1_gate["failed_checks"]
    c1_gate["candidate_gate_passed"] = False
    c1_gate["observation_eligible"] = False
    export_json(c1_dir.as_posix(), "gate_result.json", c1_gate)
    export_metrics(c1_dir.as_posix(), c1_metrics)
    write_manifest(
        run_dir=c1_dir.as_posix(), strategy_name=C1_NAME, config=config,
        db_path=str(ROOT / DB_PATH), rules_path=str(ROOT / RULES_PATH),
        exposure_mapping_path=str(ROOT / MAPPING_PATH), metrics=c1_metrics, gate_result=c1_gate,
    )
    validation = validate_artifact_bundle(c1_dir, strategy_name=C1_NAME, expected_run_id=run_id, expected_start=actual_period[0], expected_end=actual_period[1])
    if not validation.get("passed"):
        raise RuntimeError(f"C1_ARTIFACT_VALIDATION_FAILED:{validation.get('errors')}")

    status = "RESEARCH_GATE_FAILED" if not c1_gate["gate_passed"] else "RESEARCH_GATE_PASSED_CURRENT_SNAPSHOT_ONLY"
    status_payload = {
        "run_id": run_id,
        "strategy": C1_NAME,
        "status": status,
        "sample_label": config["sample_label"],
        "observation_eligible": False,
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
        "metrics": {C1_NAME: c1_metrics, B2_NAME: b2_metrics, B2LT_NAME: b2lt_metrics},
        "gate_result": {C1_NAME: c1_gate, B2_NAME: b2_gate, B2LT_NAME: b2lt_gate},
        "stress_tests": stress_results,
        "benchmark_rerun": {"B2": B2_NAME, "B2LT": B2LT_NAME, "same_db": True, "same_rule_book": True},
        "calendar_correction_label": "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "supersedes_run_id": "core_satellite_20260729_151927",
        "intermediate_calendar_corrected_run_id": "core_satellite_20260730_114756",
        "failed_research_thresholds": c1_gate["failed_checks"],
        "disclosures": [
            "This sample interval was reused across multiple observations and is not fresh OOS.",
            "Historical rule dates are not established; only the current-snapshot conservative execution scenario is reported.",
            "NAV_PUBLICATION_TIMESTAMP_NOT_AVAILABLE: the source table has no publication_date or publication timestamp; nav_date <= signal_date is not a verified publication-time point-in-time filter, especially for 050025 and potentially other cross-border/QDII funds.",
            "Signal point-in-time truth is NOT_ESTABLISHED.",
        ],
    }
    export_json(run_dir.as_posix(), "core_satellite_input_facts.json", facts)
    export_config_snapshot(run_dir.as_posix(), config)
    export_json(run_dir.as_posix(), "core_satellite_status.json", status_payload)
    export_json(run_dir.as_posix(), "gate_result.json", c1_gate)
    _write_conclusion(run_dir, config, facts, {C1_NAME: c1_metrics, B2_NAME: b2_metrics, B2LT_NAME: b2lt_metrics}, c1_gate, stress_results, status)
    write_manifest(
        run_dir=run_dir.as_posix(), strategy_name=C1_NAME, config=config,
        db_path=str(ROOT / DB_PATH), rules_path=str(ROOT / RULES_PATH),
        exposure_mapping_path=str(ROOT / MAPPING_PATH), metrics=c1_metrics, gate_result=c1_gate,
    )
    print(json.dumps(status_payload, ensure_ascii=False, indent=2, default=str))
    return 0


def _refresh_existing_run(run_dir: Path) -> int:
    """Refresh only publication-timing audit fields for an existing C1 run."""
    run_dir = run_dir.resolve()
    status_path = run_dir / "core_satellite_status.json"
    facts_path = run_dir / "core_satellite_input_facts.json"
    config_path = run_dir / "config_snapshot.json"
    c1_dir = run_dir / C1_NAME
    if not status_path.exists() or not c1_dir.exists():
        raise FileNotFoundError(f"C1_REFRESH_RUN_NOT_FOUND:{run_dir}")

    status = json.loads(status_path.read_text(encoding="utf-8"))
    facts = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.exists() else {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    correction_metadata = {
        "calendar_correction_label": "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "supersedes_run_id": "core_satellite_20260729_151927",
        "intermediate_calendar_corrected_run_id": "core_satellite_20260730_114756",
    }
    status.update(correction_metadata)
    facts.update(correction_metadata)
    config.update(correction_metadata)
    metrics = status["metrics"]
    c1_gate = json.loads((c1_dir / "gate_result.json").read_text(encoding="utf-8"))
    checks = dict(c1_gate.get("checks", {}))
    checks["publication_timing_gate"] = False
    c1_gate.update(
        {
            "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
            "gate_passed": False,
            "candidate_gate_passed": False,
            "observation_eligible": False,
            "publication_timing_status": PUBLICATION_TIMING_STATUS,
            "signal_point_in_time_truth": SIGNAL_PIT_TRUTH_STATUS,
            "publication_timing_audit": publication_timing_audit(),
        }
    )

    facts["publication_timing_audit"] = publication_timing_audit()
    status["status"] = "RESEARCH_GATE_FAILED"
    status["observation_eligible"] = False
    status["publication_timing_status"] = PUBLICATION_TIMING_STATUS
    status["publication_timing_gate"] = False
    status["signal_point_in_time_truth"] = SIGNAL_PIT_TRUTH_STATUS
    status["publication_timing_audit"] = publication_timing_audit()
    status.setdefault("disclosures", [])
    for disclosure in (
        "NAV_PUBLICATION_TIMESTAMP_NOT_AVAILABLE: the source table has no publication_date or publication timestamp; nav_date <= signal_date is not a verified publication-time point-in-time filter, especially for 050025 and potentially other cross-border/QDII funds.",
        "Signal point-in-time truth is NOT_ESTABLISHED.",
    ):
        if disclosure not in status["disclosures"]:
            status["disclosures"].append(disclosure)
    status.setdefault("gate_result", {})[C1_NAME] = c1_gate
    status["failed_research_thresholds"] = list(c1_gate["failed_checks"])

    export_json(c1_dir.as_posix(), "gate_result.json", c1_gate)
    export_json(run_dir.as_posix(), "core_satellite_input_facts.json", facts)
    export_json(run_dir.as_posix(), "core_satellite_status.json", status)
    export_config_snapshot(run_dir.as_posix(), config)
    export_config_snapshot(c1_dir.as_posix(), config)
    export_json(run_dir.as_posix(), "gate_result.json", c1_gate)
    _write_conclusion(
        run_dir,
        config,
        facts,
        metrics,
        c1_gate,
        status.get("stress_tests", {}),
        status["status"],
    )

    for manifest_dir in (c1_dir, run_dir):
        write_manifest(
            run_dir=manifest_dir.as_posix(),
            strategy_name=C1_NAME,
            config=config,
            db_path=str(ROOT / DB_PATH),
            rules_path=str(ROOT / RULES_PATH),
            exposure_mapping_path=str(ROOT / MAPPING_PATH),
            metrics=metrics[C1_NAME],
            gate_result=c1_gate,
        )
    validation = validate_artifact_bundle(
        c1_dir,
        strategy_name=C1_NAME,
        expected_run_id=status["run_id"],
        expected_start=status["actual_oos_period"][0],
        expected_end=status["actual_oos_period"][1],
    )
    if not validation.get("passed"):
        raise RuntimeError(f"C1_REFRESH_ARTIFACT_VALIDATION_FAILED:{validation.get('errors')}")
    print(
        json.dumps(
            {
                "run_id": status["run_id"],
                "status": status["status"],
                "publication_timing_gate": False,
                "failed_checks": c1_gate["failed_checks"],
                "metrics_unchanged": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run or audit the frozen C1 research run")
    parser.add_argument(
        "--refresh-run",
        type=Path,
        help="refresh only publication-timing audit fields in an existing run",
    )
    args = parser.parse_args(argv)
    if args.refresh_run is not None:
        return _refresh_existing_run(args.refresh_run)
    return _run_research()


if __name__ == "__main__":
    raise SystemExit(main())
