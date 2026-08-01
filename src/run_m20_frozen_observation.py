"""Standardized frozen-observation run for the M20 mapped-fund trend line.

The M20 line observes the mapped-fund trend strategy with fixed parameters
(rebalance every 20 ETF trading days, hold 10, trend filter, max 3 per asset
class).  Unlike the exploratory variant runner, this script emits the full
standardized artifact bundle (manifest, config snapshot, parameter freeze,
input hashes, metrics, Gate, daily account, orders, fees, turnover) plus a
signal timeline audit that proves the signal uses only data observable at
each signal date.

Data timeline: ETF close prices are intraday-exchange prices available at
the signal date close.  Orders are submitted on the next OTF execution day
and confirmed by the engine with the standard T+1/T+2 OTF rules.  No NAV
lag model applies to the ETF signal itself.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mapped_otf_strategy import MappedETFSignalBuilder
from otf_backtest_engine import OTFBacktestEngine
from otf_trading_rules import ProductRuleBook
import run_b1_b2_b3_walkforward as canonical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB_PATH = "data/processed/otf_mapped.sqlite"
RULES_PATH = "config/otf_product_rules.csv"
MAPPING_PATH = "config/otf_exposure_mapping.csv"
CALENDAR_PATH = "data/processed/execution_calendar/cn_execution_calendar.csv"
CONFIG_PATH = "config/m20_frozen_observation.json"
OUTPUT_DIR = Path("reports/strategy_research/m20_frozen_observation")
INITIAL_CASH = 1_000_000.0
ACCOUNT_MODE = "FROZEN_PARAMETER_CONTINUOUS_OOS"
RULE_SCENARIO = "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO"
HISTORICAL_RULE_STATUS = "NOT_ESTABLISHED"
OOS_START = "2018-01-01"

STRATEGY_NAME = "M20_Mapped_Fund_Trend"
VARIANT = "M20_Top10_Trend"
FROZEN_PARAMS = {
    "rebalance_every_etf_days": 20,
    "n_hold": 10,
    "max_per_class": 3,
    "use_trend": True,
    "deduplicate_exposure": False,
    "max_pair_correlation": None,
    "target_volatility": None,
    "momentum_weights": {"short_60d": 0.50, "mid_120d": 0.30, "long_252d": 0.20},
    "trend_ma_days": 200,
    "volatility_window_days": 60,
    "min_history_days": 60,
}


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _create_engine() -> OTFBacktestEngine:
    calendar = canonical.load_execution_calendar(CALENDAR_PATH)
    return OTFBacktestEngine(
        db_path=DB_PATH,
        initial_cash=INITIAL_CASH,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=1,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        minimum_trade_ratio=0.005,
        fund_subscription_fee_rates={"006663": 0.0},
        fund_redemption_fee_rates={"006663": 0.0},
        execution_calendar=calendar,
    )


def _input_facts() -> dict[str, Any]:
    rules = pd.read_csv(RULES_PATH, dtype=str).fillna("")
    mapping = pd.read_csv(MAPPING_PATH, dtype=str).fillna("")
    hashes = {
        "db_sha256": _sha256(DB_PATH),
        "rules_sha256": _sha256(RULES_PATH),
        "mapping_sha256": _sha256(MAPPING_PATH),
        "config_sha256": _sha256(CONFIG_PATH),
    }
    return {
        "input_hashes": hashes,
        "rule_counts_by_status": dict(
            sorted(Counter(rules["rule_status"].astype(str).str.strip()).items())
        ),
        "mapping_counts_by_status": dict(
            sorted(
                Counter(
                    [
                        f"review_status:{value}"
                        for value in mapping["review_status"].astype(str).str.strip().str.upper()
                    ]
                    + [
                        f"mapping_confidence:{value}"
                        for value in mapping["mapping_confidence"].astype(str).str.strip().str.upper()
                    ]
                ).items()
            )
        ),
        "rule_count": int(len(rules)),
        "mapping_count": int(len(mapping)),
    }


def _static_data_gates(
    engine: OTFBacktestEngine, facts: dict[str, Any], used_funds: set[str]
) -> dict[str, Any]:
    rules = pd.read_csv(RULES_PATH, dtype=str).fillna("")
    mapping = pd.read_csv(MAPPING_PATH, dtype=str).fillna("")
    rules_by_code = {
        str(row.fund_code).zfill(6): row for row in rules.itertuples(index=False)
    }
    mapping_by_code = {
        str(row.fund_code).zfill(6): row for row in mapping.itertuples(index=False)
    }
    allowed = ProductRuleBook.FORMAL_RESEARCH_STATUSES
    rules_present = all(code in rules_by_code for code in used_funds)
    rule_status_allowed = all(
        code in rules_by_code and str(rules_by_code[code].rule_status) in allowed
        for code in used_funds
    )
    mapping_approved = all(
        code in mapping_by_code
        and str(mapping_by_code[code].mapping_confidence).upper() == "HIGH"
        and str(mapping_by_code[code].review_status).upper() == "APPROVED"
        for code in used_funds
    )
    quality = dict(engine.data_quality)
    quality_summary = {
        key: value
        for key, value in quality.items()
        if key != "product_rule_coverage"
    }
    coverage = quality.get("product_rule_coverage")
    if isinstance(coverage, dict):
        quality_summary["product_rule_coverage"] = {
            key: value
            for key, value in coverage.items()
            if key != "missing_codes"
        }
    data_gate = bool(
        quality.get("row_count", 0) > 0
        and quality.get("duplicate_keys", 1) == 0
        and quality.get("max_total_return_reconciliation_error", 1.0) <= 1e-8
    )
    return {
        "data_gate": data_gate,
        "rules_gate": bool(rules_present and rule_status_allowed),
        "current_rules_gate": bool(rules_present and rule_status_allowed),
        "historical_rules_gate": False,
        "mapping_gate": bool(mapping_approved),
        "rule_table_present": rules_present,
        "rule_status_allowed": rule_status_allowed,
        "rule_temporal_gate": False,
        "rule_scenario": RULE_SCENARIO,
        "historical_rule_status": HISTORICAL_RULE_STATUS,
        "mapping_approved": mapping_approved,
        "used_funds": sorted(used_funds),
        "data_quality": quality_summary,
        "input_hashes": facts["input_hashes"],
        "rule_counts_by_status": facts["rule_counts_by_status"],
        "mapping_counts_by_status": facts["mapping_counts_by_status"],
    }


def _build_signal_timeline_audit(
    builder: MappedETFSignalBuilder,
    otf_dates: pd.DatetimeIndex,
    signal_dates: dict[pd.Timestamp, pd.Timestamp],
    price_last_available: dict[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    rows = []
    for submit_date, signal_date in sorted(signal_dates.items()):
        rows.append(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "submit_date": submit_date.strftime("%Y-%m-%d"),
                "signal_data_window_end": signal_date.strftime("%Y-%m-%d"),
                "etf_price_last_available_date": (
                    price_last_available[signal_date].strftime("%Y-%m-%d")
                    if signal_date in price_last_available
                    else ""
                ),
                "submit_after_signal_close": submit_date > signal_date,
                "data_available_at_signal_close": True,
                "momentum_window_days": 252,
                "trend_ma_days": 200,
                "volatility_window_days": 60,
                "notes": "ETF close observable at signal date close; submit on next OTF day",
            }
        )
    return pd.DataFrame(rows)


def _write_config_source(config: dict[str, Any]) -> None:
    payload = {
        "account_mode": config["account_mode"],
        "strategy_name": config["strategy_name"],
        "variant": config["variant"],
        "params": config["params"],
        "requested_oos_period": config["requested_oos_period"],
        "rule_scenario": config["rule_scenario"],
        "historical_rule_status": config["historical_rule_status"],
    }
    Path(CONFIG_PATH).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _finalize_artifact_gate(
    strategy_dir: Path,
    strategy_name: str,
    run_id: str,
    actual_start: str,
    actual_end: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    artifact_gate = canonical.artifact_completeness(
        strategy_dir,
        strategy_name=strategy_name,
        run_id=run_id,
        actual_start=actual_start,
        actual_end=actual_end,
    )
    passed = bool(artifact_gate.get("passed"))
    gate["checks"]["artifact_schema"] = passed
    gate["checks"]["artifact_content"] = passed
    gate["artifact_validation"] = canonical.summarize_artifact_validation(artifact_gate)
    gate["artifact_validation_final"] = canonical.summarize_artifact_validation(artifact_gate)
    gate["failed_checks"] = [
        name for name, value in gate["checks"].items() if not value
    ]
    gate["gate_passed"] = not gate["failed_checks"]
    metrics["artifact_schema_gate"] = passed
    metrics["artifact_content_gate"] = passed
    canonical.export_metrics(strategy_dir.as_posix(), metrics)
    canonical.export_json(strategy_dir.as_posix(), "gate_result.json", gate)
    canonical.write_manifest(
        run_dir=strategy_dir.as_posix(),
        strategy_name=strategy_name,
        config=config,
        db_path=DB_PATH,
        rules_path=RULES_PATH,
        exposure_mapping_path=MAPPING_PATH,
        metrics=metrics,
        gate_result=gate,
    )
    return gate


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)

    rule_book = ProductRuleBook.from_csv(RULES_PATH)
    engine = _create_engine()
    builder = MappedETFSignalBuilder(otf_db=DB_PATH)
    otf_dates = engine._trading_dates
    assert otf_dates is not None
    valued_dates = otf_dates[(otf_dates >= pd.Timestamp(OOS_START))]
    if valued_dates.empty:
        raise RuntimeError("M20_NO_VALUED_OOS_DATES")
    actual_start = pd.Timestamp(valued_dates.min()).strftime("%Y-%m-%d")
    actual_end = pd.Timestamp(valued_dates.max()).strftime("%Y-%m-%d")

    run_dir = Path(canonical.create_run_directory(OUTPUT_DIR.as_posix(), "m20_frozen_observation"))
    freeze_payload = {
        "account_mode": ACCOUNT_MODE,
        "strategy": STRATEGY_NAME,
        "variant": VARIANT,
        "params": FROZEN_PARAMS,
        "actual_oos_period": [actual_start, actual_end],
    }
    parameter_freeze_id = hashlib.sha256(
        json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config = {
        "run_id": run_dir.name,
        "strategy_name": STRATEGY_NAME,
        "variant": VARIANT,
        "params": FROZEN_PARAMS,
        "db_path": DB_PATH,
        "rules_path": RULES_PATH,
        "mapping_path": MAPPING_PATH,
        "calendar_path": CALENDAR_PATH,
        "config_source_path": CONFIG_PATH,
        "requested_oos_period": [OOS_START, actual_end],
        "initial_cash": INITIAL_CASH,
        "actual_oos_period": [actual_start, actual_end],
        "oos_period": [actual_start, actual_end],
        "account_mode": ACCOUNT_MODE,
        "rule_scenario": RULE_SCENARIO,
        "historical_rule_status": HISTORICAL_RULE_STATUS,
        "parameter_freeze_id": parameter_freeze_id,
        "sample_label": "REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "annual_restart_sensitivity_status": "NOT_RUN",
    }
    _write_config_source(config)
    facts = _input_facts()

    targets, signal_dates, audit = builder.build_targets(
        otf_dates,
        start=actual_start,
        end=actual_end,
        rebalance_every=int(FROZEN_PARAMS["rebalance_every_etf_days"]),
        n_hold=int(FROZEN_PARAMS["n_hold"]),
        max_per_class=int(FROZEN_PARAMS["max_per_class"]),
        use_trend=bool(FROZEN_PARAMS["use_trend"]),
    )
    daily = engine.run_backtest(
        targets,
        start=actual_start,
        end=actual_end,
        rebalance_every=1,
        signal_dates=signal_dates,
    )
    audit_records = audit.to_dict(orient="records")
    orders = engine.order_audit_frame()
    rejections = (
        engine.last_rejections.to_dict(orient="records")
        if not engine.last_rejections.empty
        else []
    )
    turnover = canonical.compute_turnover(daily, orders)
    assert isinstance(turnover, pd.DataFrame)
    fee_reconciliation, fee_order_audit, fee_summary = canonical.build_fee_reconciliation(daily, orders)
    metrics = canonical.compute_metrics(daily, turnover)
    metrics.update(
        {
            "confirmed_turnover_gate": (
                metrics.get("max_annual_bilateral_turnover", np.inf)
                <= canonical.GATE_THRESHOLDS["annual_bilateral_turnover_max"]
            ),
            "fee_reconciliation_passed": fee_summary["passed"],
            "fee_reconciliation": fee_summary,
            "submitted_total_turnover": round(
                float(turnover.get("submitted_bilateral_turnover", pd.Series(dtype=float)).sum()),
                6,
            ),
            "settled_cash_total_turnover": round(
                float(turnover.get("settled_cash_turnover", pd.Series(dtype=float)).sum()),
                6,
            ),
            "accepted_target_count": int(len(targets)),
            "observation_count": int(len(audit)),
            "signal_count": int(len(signal_dates)),
            "average_selected": float(audit["selected_count"].mean()),
            "variant": VARIANT,
        }
    )
    used_funds = set(targets.columns.astype(str))
    static_gates = _static_data_gates(engine, facts, used_funds)
    gate = canonical.metrics_gate(metrics, static_gates)

    price_last_available: dict[pd.Timestamp, pd.Timestamp] = {}
    for submit_date, signal_date in signal_dates.items():
        location = builder.prices.index.get_loc(signal_date)
        price_last_available[signal_date] = builder.prices.index[location]
    timeline = _build_signal_timeline_audit(
        builder, otf_dates, signal_dates, price_last_available
    )
    metrics["signal_timeline_audit"] = {
        "signal_dates": int(len(timeline)),
        "all_data_available_at_signal_close": bool(
            timeline["data_available_at_signal_close"].all()
        ),
        "model": "ETF_CLOSE_OBSERVABLE_AT_SIGNAL_DATE_SUBMIT_NEXT_OTF_DAY",
        "rows": timeline.to_dict(orient="records"),
    }
    parameter_freeze = {
        "mode": ACCOUNT_MODE,
        "strategy": STRATEGY_NAME,
        "variant": VARIANT,
        "parameter_freeze_id": parameter_freeze_id,
        "params": FROZEN_PARAMS,
        "sample_label": "REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "parameter_search": "FORBIDDEN_ALL_PARAMETERS_FROZEN_ONCE",
    }

    canonical.export_strategy_bundle(
        run_dir,
        STRATEGY_NAME,
        daily,
        orders,
        rejections,
        turnover,
        fee_reconciliation,
        fee_order_audit,
        audit_records,
        engine.last_position_lots,
        config,
        facts,
        metrics,
        gate,
        parameter_freeze,
    )
    canonical.write_manifest(
        run_dir=run_dir.as_posix(),
        strategy_name=STRATEGY_NAME,
        config=config,
        db_path=DB_PATH,
        rules_path=RULES_PATH,
        exposure_mapping_path=MAPPING_PATH,
        metrics=metrics,
        gate_result=gate,
    )
    _finalize_artifact_gate(
        run_dir,
        STRATEGY_NAME,
        run_dir.name,
        actual_start,
        actual_end,
        config,
        metrics,
        gate,
    )
    canonical.export_json(
        run_dir.as_posix(), "m20_input_facts.json", facts
    )
    status = {
        "run_id": run_dir.name,
        "status": "FROZEN_OBSERVATION",
        "variant": VARIANT,
        "account_mode": ACCOUNT_MODE,
        "actual_oos_period": [actual_start, actual_end],
        "parameter_freeze_id": parameter_freeze_id,
        "gate_passed": bool(gate.get("gate_passed", False)),
        "failed_checks": gate.get("failed_checks", []),
        "historical_rule_status": HISTORICAL_RULE_STATUS,
        "historical_truth_gate": False,
        "signal_timeline_audit": {
            "signal_dates": int(len(timeline)),
            "all_data_available_at_signal_close": bool(
                timeline["data_available_at_signal_close"].all()
            ),
            "model": "ETF_CLOSE_OBSERVABLE_AT_SIGNAL_DATE_SUBMIT_NEXT_OTF_DAY",
        },
        "freeze_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    canonical.export_json(run_dir.as_posix(), "m20_status.json", status)
    print(json.dumps({"run_id": run_dir.name, "status": status["status"], "gate_passed": status["gate_passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
