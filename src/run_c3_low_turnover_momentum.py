"""Run the frozen C3 low-turnover core/satellite momentum research study."""

from __future__ import annotations

import argparse

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

import run_core_satellite_momentum as c1_runner
import run_b1_b2_b3_walkforward as canonical
from otf_backtest_engine import OTFBacktestEngine
from otf_rotation.artifact_validation import validate_artifact_bundle
from otf_rotation.candidate_strategies import B2LTSignal, build_candidate_targets
from otf_rotation.core_satellite_momentum import (
    clone_rule_book_with_delay_override,
    sustained_turnover_excluding_initial,
)
from otf_rotation.c3_low_turnover_momentum import (
    CORE_WEIGHTS,
    C3LowTurnoverMomentumSignal,
    DRIFT_THRESHOLD,
    FALLBACK_PRIMARY,
    FALLBACK_SECONDARY,
    MAX_SATELLITES,
    MIN_HOLDING_QUARTERS,
    SATELLITE_BUDGET,
    SATELLITE_POOL,
    SATELLITE_SLOT_WEIGHT,
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
    build_quarter_end_schedule,
    build_signal_submit_map,
)
from otf_trading_rules import ProductRuleBook


DB_PATH = "data/processed/otf_expanded.sqlite"
RULES_PATH = "config/otf_product_rules.csv"
MAPPING_PATH = "config/otf_exposure_mapping.csv"
CONFIG_PATH = "config/c3_low_turnover_momentum.json"
OUTPUT_DIR = Path("reports/strategy_research/c3_low_turnover_momentum")
INITIAL_CASH = 1_000_000.0
OOS_START = "2021-01-04"
OOS_END = "2026-07-27"
C3_NAME = "C3_LOW_TURNOVER_MOMENTUM"
B2_NAME = c1_runner.B2_NAME
B2LT_NAME = c1_runner.B2LT_NAME
B2LT_BUNDLE_DIR = ROOT / "reports/strategy_research/core_satellite/core_satellite_20260730_124538/B2_LT_Static_EW_4Asset"
EXPECTED_B2LT_METRICS_SHA256 = "67625410796a4ff12fa866e3786ba4fda4172758b09f1f7587bf7334f6ce9073"
ACCOUNT_MODE = "FROZEN_PARAMETER_CONTINUOUS_OOS"
RULE_SCENARIO = "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO"
HISTORICAL_RULE_STATUS = "NOT_ESTABLISHED"
SAMPLE_LABEL = "REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
PUBLICATION_TIMING_STATUS = "NAV_PUBLICATION_TIMESTAMP_NOT_AVAILABLE"
SIGNAL_PIT_TRUTH_STATUS = "NOT_ESTABLISHED"


def publication_timing_audit() -> dict[str, Any]:
    return {
        "status": PUBLICATION_TIMING_STATUS,
        "signal_point_in_time_truth": SIGNAL_PIT_TRUTH_STATUS,
        "publication_timestamp_column": None,
        "publication_timestamp_available": False,
        "nav_date_filter_is_verified_pit": False,
        "affected_or_potentially_affected_products": sorted(SATELLITE_POOL),
        "reason": "Source has nav_date and daily_growth_pct but no publication timestamp.",
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
        },
        "rule_counts_by_status": dict(
            sorted(rules["rule_status"].astype(str).str.strip().value_counts().items())
        ),
        "mapping_counts_by_status": dict(
            sorted(
                {
                    **{f"review_status:{k}": int(v) for k, v in mapping["review_status"].astype(str).str.strip().str.upper().value_counts().items()},
                    **{f"mapping_confidence:{k}": int(v) for k, v in mapping["mapping_confidence"].astype(str).str.strip().str.upper().value_counts().items()},
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
        "execution_calendar": load_execution_calendar(str(ROOT / "data/processed/execution_calendar/cn_execution_calendar.csv")).facts(),
        "publication_timing_audit": publication_timing_audit(),
        "config": config,
    }


def _create_engine(rule_book: ProductRuleBook) -> OTFBacktestEngine:
    return c1_runner._create_engine(rule_book)


def _signal_targets(
    signal: C3LowTurnoverMomentumSignal,
    signal_dates: list[pd.Timestamp],
    signal_map: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    submit_for_signal = {str(sd): str(sub) for sub, sd in signal_map.items()}
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    audits: list[dict[str, Any]] = []

    for signal_date in signal_dates:
        date = pd.Timestamp(signal_date)
        submit_str = submit_for_signal.get(date.strftime("%Y-%m-%d"))
        if submit_str is None:
            continue

        target_weights = signal.generate_signal(date)
        audit = getattr(signal, "last_signal_audit", {}) or {}
        action = audit.get("action")

        row = dict(audit)
        row["submit_date"] = submit_str
        row["market_state"] = "C3_LOW_TURNOVER_MOMENTUM"
        row["selected_satellites"] = list(getattr(signal, "last_selected_satellites", []) or [])
        audits.append(row)

        if action != "TRIGGER_REBALANCE":
            continue

        if not target_weights:
            continue

        clean_target = {str(code): float(weight) for code, weight in target_weights.items() if float(weight) > 1e-12}
        if clean_target:
            rows[pd.Timestamp(submit_str)] = clean_target

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
    return c1_runner._run_account(engine, targets, signal_map)


def _c3_metrics(
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    turnover: pd.DataFrame,
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = canonical.compute_metrics(daily, turnover)
    mdd = abs(float(metrics.get("mdd_pct", 0.0)))
    cagr = float(metrics.get("net_cagr_pct", 0.0))
    metrics["calmar"] = round(cagr / mdd, 4) if mdd > 1e-12 else float("inf")

    first_build = next(
        (row for row in audits if row.get("action") == "TRIGGER_REBALANCE" and row.get("reason") == "INITIAL_BUILD"),
        None,
    )
    initial_signal_date = first_build.get("signal_date") if first_build else None

    subperiods = {
        "2021_2023": c1_runner._subperiod_metrics(daily, "2021-01-04", "2023-12-31"),
        "2024_2026": c1_runner._subperiod_metrics(daily, "2024-01-01", OOS_END),
    }

    trigger_count = sum(1 for a in audits if a.get("action") == "TRIGGER_REBALANCE")
    no_trade_count = sum(1 for a in audits if a.get("action") == "NO_TRADE")

    # Compute sustained turnover only when INITIAL_BUILD is explicitly found
    initial_date_str = None
    sustained_annual = None
    max_sustained = None
    excluded_count = 0
    excludes_initial_build = False

    if first_build is not None:
        initial_date_str = pd.Timestamp(initial_signal_date).strftime("%Y-%m-%d")
        sustained_result = sustained_turnover_excluding_initial(daily, orders, initial_date_str)
        sustained_annual = sustained_result["annual_confirmed_turnover"]
        max_sustained = sustained_result["max_annual_confirmed_turnover"]
        excluded_count = sustained_result["excluded_initial_order_count"]
        excludes_initial_build = True

    metrics.update({
        "strategy": C3_NAME,
        "initial_build_signal_date": initial_date_str,
        "subperiod_metrics": subperiods,
        "accepted_target_count": int(trigger_count),
        "observation_count": int(len(audits)),
        "no_trade_observation_count": int(no_trade_count),
        # Sustained turnover fields (only valid when INITIAL_BUILD found)
        "confirmed_turnover_excludes_initial_build": excludes_initial_build,
        "sustained_annual_confirmed_turnover": sustained_annual,
        "max_sustained_annual_confirmed_turnover": max_sustained,
        "initial_build_orders_excluded": excluded_count,
    })
    return metrics


def _build_benchmark(
    name: str,
    engine: OTFBacktestEngine,
    signal_dates: list[pd.Timestamp],
    signal_map: dict[str, str],
    nav_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    return c1_runner._build_benchmark(name, engine, signal_dates, signal_map, nav_df)


def _c3_gate(
    metrics: dict[str, Any],
    static_gates: dict[str, Any],
    b2lt_metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    gt = config.get("gate_thresholds", {})

    cagr_adv = metrics.get("net_cagr_pct", -np.inf) - b2lt_metrics.get("net_cagr_pct", 0.0)
    sharpe_adv = metrics.get("sharpe", -np.inf) - b2lt_metrics.get("sharpe", 0.0)

    checks: dict[str, bool] = {
        "data_gate": bool(static_gates.get("data_gate")),
        "rules_gate": bool(static_gates.get("rules_gate")),
        "mapping_gate": bool(static_gates.get("mapping_gate")),
        "net_cagr": float(metrics.get("net_cagr_pct", -np.inf)) >= float(gt.get("net_cagr_min_pct", 6.0)),
        "sharpe": float(metrics.get("sharpe", -np.inf)) >= float(gt.get("sharpe_min", 0.9)),
        "mdd": float(metrics.get("mdd_pct", -np.inf)) > float(gt.get("mdd_min_pct_exclusive", -12.0)),
        "calmar": float(metrics.get("calmar", -np.inf)) >= float(gt.get("calmar_min", 0.55)),
        "relative_b2lt_gate": bool(
            cagr_adv >= float(gt.get("relative_b2lt_gate", {}).get("cagr_advantage_min_pct_points", 0.5))
            or sharpe_adv >= float(gt.get("relative_b2lt_gate", {}).get("sharpe_advantage_min", 0.1))
        ),
        "worst_two_year_cagr": (
            metrics.get("worst_two_year_cagr_pct") is not None
            and float(metrics.get("worst_two_year_cagr_pct", -np.inf)) >= float(gt.get("worst_two_year_cagr_min_pct", -1.5))
        ),
        "rolling_two_year_positive": (
            float(metrics.get("rolling_two_year_positive_ratio_pct", 0.0))
            >= float(gt.get("rolling_two_year_positive_min_pct", 90.0))
        ),
        "double_fee_net_cagr": float(metrics.get("double_fee_net_cagr_pct", -np.inf)) >= float(gt.get("double_fee_net_cagr_min_pct", 5.5)),
        "double_fee_sharpe": float(metrics.get("double_fee_sharpe", -np.inf)) >= float(gt.get("double_fee_sharpe_min", 0.85)),
        "double_fee_mdd": float(metrics.get("double_fee_mdd_pct", -np.inf)) > float(gt.get("double_fee_mdd_min_pct_exclusive", -12.0)),
        "delay_cagr_decline": float(metrics.get("delay_cagr_decline_max_pct_points", np.inf)) <= float(gt.get("delay_cagr_decline_max_pct_points", 1.0)),
        "subperiod_2021_2023_cagr_positive": (
            float(metrics.get("subperiod_metrics", {}).get("2021_2023", {}).get("net_cagr_pct", -np.inf))
            >= float(gt.get("subperiod_net_cagr_positive", 0.0))
        ),
        "subperiod_2024_2026_cagr_positive": (
            float(metrics.get("subperiod_metrics", {}).get("2024_2026", {}).get("net_cagr_pct", -np.inf))
            >= float(gt.get("subperiod_net_cagr_positive", 0.0))
        ),
        "subperiod_sharpe_min": bool(
            all(float(sp.get("sharpe", -np.inf)) >= float(gt.get("subperiod_sharpe_min", 0.5))
                for sp in metrics.get("subperiod_metrics", {}).values())
        ),
        # Sustained confirmed turnover: MUST be strictly less than threshold; fail if fields missing
        "sustained_confirmed_turnover": bool(
            "max_sustained_annual_confirmed_turnover" in metrics
            and "confirmed_turnover_excludes_initial_build" in metrics
            and metrics.get("confirmed_turnover_excludes_initial_build") is True
            and float(metrics.get("max_sustained_annual_confirmed_turnover", np.inf))
            < float(gt.get("sustained_annual_confirmed_turnover_max_exclusive", 0.8))
        ),
    }

    failed = [name for name, value in checks.items() if not value]
    return {
        "checks": checks,
        "failed_checks": failed,
        "gate_passed": bool(all(checks.values())),
        "candidate_gate_passed": False,
        "observation_eligible": False,
        "relative_b2lt_cagr_advantage_pct_points": round(cagr_adv, 4),
        "relative_b2lt_sharpe_advantage": round(sharpe_adv, 4),
        "relative_b2lt_gate_is_or": True,
        "historical_truth_gate": False,
        "publication_timing_status": PUBLICATION_TIMING_STATUS,
        "signal_point_in_time_truth": SIGNAL_PIT_TRUTH_STATUS,
    }


def _status_from_gate(gate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": config.get("run_id"),
        "strategy": C3_NAME,
        "status": SAMPLE_LABEL,
        "observation_eligible": False,
        "publication_truth": SIGNAL_PIT_TRUTH_STATUS,
        "historical_rules": HISTORICAL_RULE_STATUS,
        "gate_passed": gate.get("gate_passed", False),
        "failed_checks": gate.get("failed_checks", []),
    }


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
        audit = {"scenario": scenario, "subscription_fee_multiplier": 2.0, "redemption_fee_multiplier": 2.0, "rule_override": "ProductRuleBook.scaled_fees(2.0)"}
    elif scenario == "delay_plus_one_trading_day":
        used = list(CORE_WEIGHTS) + list(SATELLITE_POOL)
        stressed_rules = clone_rule_book_with_delay_override(rule_book, used, increment=1)
        audit = {"scenario": scenario, "subscription_confirmation_days_increment": 1, "redemption_confirmation_days_increment": 1, "redemption_arrival_days_increment": 1, "rule_override": "cloned_actual_hit_product_rules_plus_one", "actual_hit_rule_codes": sorted(code for code in used if rule_book.rule_for(code) is not None)}
    else:
        raise ValueError(f"unknown_c3_stress:{scenario}")

    engine = _create_engine(stressed_rules)
    signal = C3LowTurnoverMomentumSignal(growth_df, trading_dates=trading_dates)
    targets, audits = _signal_targets(signal, signal_dates, signal_map)
    daily, orders, rejections, turnover, fees, extra = _run_account(engine, targets, signal_map)
    metrics = _c3_metrics(daily, orders, turnover, audits)
    metrics["fee_reconciliation_passed"] = bool(extra["fee_summary"].get("passed"))
    metrics["fee_reconciliation"] = extra["fee_summary"]

    audit.update({"metrics": metrics, "order_count": int(len(orders)), "fee_total": float(orders["fee_paid"].sum()) if not orders.empty else 0.0})

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


def _build_parameter_freeze(parameter_freeze_id: str) -> dict[str, Any]:
    """Build the frozen parameter payload for C3 (pure helper for testability)."""
    return {
        "strategy": C3_NAME,
        "mode": "FROZEN_PARAMETER_CONTINUOUS_OOS",
        "parameter_freeze_id": parameter_freeze_id,
        "sample_label": SAMPLE_LABEL,
        "parameter_search": "FORBIDDEN_ALL_PARAMETERS_FROZEN_ONCE",
        "frozen_rules": {
            "core_weights": dict(CORE_WEIGHTS),
            "satellite_pool": list(SATELLITE_POOL),
            "satellite_budget": SATELLITE_BUDGET,
            "max_satellites": MAX_SATELLITES,
            "satellite_slot_weight": SATELLITE_SLOT_WEIGHT,
            "min_holding_quarters": MIN_HOLDING_QUARTERS,
            "drift_threshold_pct_points": DRIFT_THRESHOLD * 100.0,
        },
    }


def _export_c3_market_states(output_dir: str, audits: list[dict[str, Any]]) -> None:
    """Export market_states.csv with required C3 audit fields (requirement #2)."""
    rows: list[dict[str, Any]] = []
    for audit in audits:
        row = {
            "signal_date": str(audit.get("signal_date", "")),
            "market_state": str(audit.get("market_state", "C3_LOW_TURNOVER_MOMENTUM")),
            "submit_date": str(audit.get("submit_date", "")),
            "action": str(audit.get("action", "")),
            "reason": str(audit.get("reason", "")),
            "selected_satellites": audit.get("selected_satellites", []),
            "max_drift_deviation_pct_points": float(audit.get("max_drift_deviation_pct_points", 0.0)),
            "proposed_target_weights": audit.get("proposed_target_weights", {}),
            "target_weights": audit.get("target_weights", {}),
        }
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        export_table(output_dir, "market_states.csv", df)
    else:
        # Header-only for empty case (NOT_APPLICABLE pattern)
        export_table(
            output_dir,
            "market_states.csv",
            pd.DataFrame(columns=["signal_date", "market_state", "submit_date", "action", "reason",
                                   "selected_satellites", "max_drift_deviation_pct_points",
                                   "proposed_target_weights", "target_weights"]),
        )


def _run_research() -> int:
    config_source = _load_config()
    facts = _facts(config_source)
    rule_book = ProductRuleBook.from_csv(ROOT / RULES_PATH)
    schedule_engine = _create_engine(rule_book)
    trading_dates = pd.to_datetime(schedule_engine._trading_dates)
    valued_dates = trading_dates[(trading_dates >= pd.Timestamp(OOS_START)) & (trading_dates <= pd.Timestamp(OOS_END))]

    signal_dates = build_quarter_end_schedule(trading_dates, OOS_START, OOS_END)
    signal_map = build_signal_submit_map(trading_dates, signal_dates, OOS_END)

    if not signal_dates:
        raise RuntimeError("C3_NO_QUARTER_END_SIGNAL_DATES")

    actual_period = [valued_dates.min().strftime("%Y-%m-%d"), valued_dates.max().strftime("%Y-%m-%d")]
    run_dir = Path(create_run_directory(str(ROOT / OUTPUT_DIR), "c3_low_turnover_momentum"))
    run_id = run_dir.name

    freeze_payload = {"config": config_source, "actual_oos_period": actual_period, "strategy": C3_NAME}
    parameter_freeze_id = hashlib.sha256(json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    config = {
        **config_source,
        "run_id": run_id,
        "db_path": DB_PATH,
        "rules_path": RULES_PATH,
        "mapping_path": MAPPING_PATH,
        "initial_cash": INITIAL_CASH,
        "actual_oos_period": actual_period,
        "account_mode": ACCOUNT_MODE,
        "rule_scenario": RULE_SCENARIO,
        "historical_rule_status": HISTORICAL_RULE_STATUS,
        "parameter_freeze_id": parameter_freeze_id,
        "config_source_path": str(ROOT / CONFIG_PATH),
        "interpreter": {"path": sys.executable, "version": sys.version},
    }

    growth_df = schedule_engine._nav_df[["fund_code", "nav_date", "daily_growth_pct", "unit_nav"]].copy()
    signal = C3LowTurnoverMomentumSignal(growth_df, trading_dates=trading_dates)
    targets, audits = _signal_targets(signal, signal_dates, signal_map)

    if targets.empty:
        raise RuntimeError("C3_NO_ACCEPTED_TARGET_SIGNALS")

    engine = schedule_engine
    daily, orders, rejections, turnover, fee_reconciliation, extra = _run_account(engine, targets, signal_map)

    c3_metrics = _c3_metrics(daily, orders, turnover, audits)
    c3_metrics["fee_reconciliation_passed"] = bool(extra["fee_summary"].get("passed"))
    c3_metrics["fee_reconciliation"] = extra["fee_summary"]

    # B2LT benchmark: read from existing verified bundle (do NOT recompute)
    b2lt_metrics_path = B2LT_BUNDLE_DIR / "metrics.json"
    if not b2lt_metrics_path.exists():
        raise RuntimeError(f"B2LT_METRICS_NOT_FOUND:{b2lt_metrics_path}")
    b2lt_metrics_sha = sha256_file(str(b2lt_metrics_path))
    if b2lt_metrics_sha != EXPECTED_B2LT_METRICS_SHA256:
        raise RuntimeError(
            f"B2LT_METRICS_SHA_MISMATCH:expected={EXPECTED_B2LT_METRICS_SHA256},actual={b2lt_metrics_sha}"
        )
    b2lt_metrics = json.loads(b2lt_metrics_path.read_text(encoding="utf-8"))

    # Cross-check B2LT input_hashes against the baseline freeze (requirement #6)
    # The bundle is a frozen artifact: its fingerprints must equal the baseline
    # freeze manifest, NOT the current (possibly revised) inputs. Current-input
    # changes are legitimate only when recorded in the data revision registry,
    # which is verified separately before forward observation.
    b2lt_input_hashes_path = B2LT_BUNDLE_DIR / "input_hashes.json"
    if not b2lt_input_hashes_path.exists():
        raise RuntimeError(f"B2LT_INPUT_HASHES_NOT_FOUND:{b2lt_input_hashes_path}")
    b2lt_input_hashes_sha = sha256_file(str(b2lt_input_hashes_path))
    b2lt_input_hashes = json.loads(b2lt_input_hashes_path.read_text(encoding="utf-8"))

    manifest = json.loads((ROOT / "baseline_freeze_manifest.json").read_text(encoding="utf-8"))
    snapshot = manifest["data_snapshot"]
    for key in ("db_sha256", "rules_sha256", "mapping_sha256"):
        if key not in snapshot or key not in b2lt_input_hashes:
            raise RuntimeError(f"B2LT_INPUT_HASH_KEY_MISSING:{key}")
        if snapshot[key] != b2lt_input_hashes[key]:
            raise RuntimeError(
                f"B2LT_INPUT_HASH_MISMATCH:{key}:baseline={snapshot[key]},b2lt={b2lt_input_hashes[key]}"
            )

    # Record B2LT bundle reference in facts for auditability
    facts["b2lt_bundle_reference"] = {
        "bundle_dir": str(B2LT_BUNDLE_DIR),
        "metrics_sha256": b2lt_metrics_sha,
        "input_hashes_sha256": b2lt_input_hashes_sha,
    }

    # Data gates
    used_funds = set(targets.columns) | set(CORE_WEIGHTS) | set(SATELLITE_POOL)
    static_gates = canonical.build_static_data_gates(engine, facts, used_funds)
    static_gates["historical_rule_status"] = HISTORICAL_RULE_STATUS
    static_gates["rule_scenario"] = RULE_SCENARIO

    # Gate
    c3_gate = _c3_gate(c3_metrics, static_gates, b2lt_metrics, config)

    # Stress tests
    stresses: dict[str, Any] = {}
    for scenario in ("double_fees", "delay_plus_one_trading_day"):
        stresses[scenario] = _run_stress(scenario, rule_book, growth_df, signal_dates, signal_map, config, run_dir, trading_dates=trading_dates)

    # Enrich metrics with stress results
    c3_metrics["double_fee_net_cagr_pct"] = stresses["double_fees"]["metrics"].get("net_cagr_pct")
    c3_metrics["double_fee_sharpe"] = stresses["double_fees"]["metrics"].get("sharpe")
    c3_metrics["double_fee_mdd_pct"] = stresses["double_fees"]["metrics"].get("mdd_pct")
    delay_daily = stresses["delay_plus_one_trading_day"]["metrics"]
    c3_metrics["delay_cagr_decline_max_pct_points"] = round(c3_metrics.get("net_cagr_pct", 0.0) - delay_daily.get("net_cagr_pct", 0.0), 4)

    # Re-evaluate gate with stress-enriched metrics
    c3_gate = _c3_gate(c3_metrics, static_gates, b2lt_metrics, config)

    # Export bundle using canonical helpers
    parameter_freeze = _build_parameter_freeze(parameter_freeze_id)

    c3_dir = run_dir / C3_NAME
    canonical.export_strategy_bundle(
        c3_dir, C3_NAME, daily, orders, rejections.to_dict(orient="records") if not rejections.empty else [],
        turnover, fee_reconciliation, extra["fee_order_audit"], audits,
        engine.last_position_lots, config, facts, c3_metrics, c3_gate, parameter_freeze,
    )

    # Export market_states.csv with required audit fields (requirement #2)
    # NOTE: must be called AFTER canonical.export_strategy_bundle so we get complete C3 fields;
    # this overwrites the minimal market_states.csv that canonical exported.
    _export_c3_market_states(c3_dir.as_posix(), audits)

    # Refresh child manifest AFTER market_states.csv overwrite so inventory hashes are correct.
    # finalize_artifact_gate runs validation against this manifest; without this refresh,
    # the first validation would fail (stale market_states hash) and artifact checks stay false.
    write_manifest(
        run_dir=c3_dir.as_posix(),
        strategy_name=C3_NAME,
        config=config,
        db_path=str(ROOT / DB_PATH),
        rules_path=str(ROOT / RULES_PATH),
        exposure_mapping_path=str(ROOT / MAPPING_PATH),
        metrics=c3_metrics,
        gate_result=c3_gate,
    )

    # Artifact validation (must raise on failure per requirement #5)
    artifact_validation = canonical.finalize_artifact_gate(c3_dir, C3_NAME, run_id, actual_period[0], actual_period[1], config, c3_metrics, c3_gate)
    if not artifact_validation.get("passed"):
        errors = artifact_validation.get("errors", [])
        raise RuntimeError(f"C3_ARTIFACT_VALIDATION_FAILED:errors={errors}")
    c3_gate["artifact_validation"] = {k: artifact_validation.get(k) for k in ("passed", "required_count", "present_count", "errors", "artifact_status")}

    # Status
    status = _status_from_gate(c3_gate, config)
    status["metrics"] = {C3_NAME: c3_metrics, B2LT_NAME: b2lt_metrics}
    status["stress_tests"] = stresses
    status["gate_result"] = {C3_NAME: c3_gate}

    # Export root-level artifacts. config_sha256 remains the SOURCE config file hash
    # computed in _facts(); manifest has separate config_snapshot_sha256 for the snapshot.
    export_config_snapshot(run_dir.as_posix(), config)
    export_input_hashes(run_dir.as_posix(), facts["input_hashes"])
    export_json(run_dir.as_posix(), "c3_input_facts.json", facts)
    export_json(run_dir.as_posix(), "c3_status.json", status)

    # Write root manifest (requirement #5)
    write_manifest(
        run_dir=run_dir.as_posix(),
        strategy_name=C3_NAME,
        config=config,
        db_path=str(ROOT / DB_PATH),
        rules_path=str(ROOT / RULES_PATH),
        exposure_mapping_path=str(ROOT / MAPPING_PATH),
        metrics=c3_metrics,
        gate_result=c3_gate,
    )

    # Conclusion
    _write_conclusion(run_dir, config, facts, {C3_NAME: c3_metrics, B2LT_NAME: b2lt_metrics}, c3_gate, stresses)

    print(json.dumps({
        "run_id": run_id,
        "status": status["status"],
        "observation_eligible": False,
        "gate_passed": c3_gate.get("gate_passed", False),
        "failed_checks": c3_gate.get("failed_checks", []),
        "c3_metrics": {k: v for k, v in c3_metrics.items() if not isinstance(v, (pd.DataFrame, np.ndarray))},
        "b2lt_metrics": {k: v for k, v in b2lt_metrics.items() if not isinstance(v, (pd.DataFrame, np.ndarray))},
        "order_count": int(len(orders)),
        "target_count": int(len(targets)),
    }, ensure_ascii=False, indent=2, default=str))

    return 0


def _write_conclusion(run_dir: Path, config: dict[str, Any], facts: dict[str, Any], metrics: dict[str, dict[str, Any]], c3_gate: dict[str, Any], stresses: dict[str, Any]) -> None:
    c3 = metrics[C3_NAME]
    b2lt = metrics[B2LT_NAME]
    lines = [
        "# C3 Low-Turnover Momentum Research", "",
        f"- Run ID: `{config['run_id']}`",
        f"- Status: `{SAMPLE_LABEL}`",
        f"- OOS interval: `{OOS_START}` to `{OOS_END}`",
        "- Historical rule truth: `NOT_ESTABLISHED`.",
        f"- NAV publication timing: `{PUBLICATION_TIMING_STATUS}`.",
        "", "## Metrics", "",
        "| Strategy | Net CAGR | Sharpe | MDD | Calmar | Fees |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (C3_NAME, B2LT_NAME):
        item = metrics[name]
        lines.append(f"| {name} | {item.get('net_cagr_pct', 0.0):.4f}% | {item.get('sharpe', 0.0):.4f} | {item.get('mdd_pct', 0.0):.4f}% | {item.get('calmar', 0.0):.4f} | {item.get('total_fee_amount', 0.0):.2f} |")
    lines.extend(["", "## C3 Gate", "", f"- Gate passed: `{c3_gate.get('gate_passed')}`", f"- Failed thresholds: `{c3_gate.get('failed_checks', []) or 'none'}`", f"- B2LT CAGR advantage: `{c3_gate.get('relative_b2lt_cagr_advantage_pct_points', 0.0):.4f} pp`", f"- B2LT Sharpe advantage: `{c3_gate.get('relative_b2lt_sharpe_advantage', 0.0):.4f}`", ""])
    (run_dir / "c3_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finalize_run(run_path: str) -> int:
    """Finalize an existing C3 run by fixing market_states, recalculating derived metrics, and refreshing artifacts.
    
    Does NOT recompute account state, targets, or stress tests. Only recalculates derived metrics
    (sustained turnover) from existing child files. Verifies core file integrity with SHA.
    """
    from pathlib import Path
    import json
    
    run_dir = Path(run_path).resolve()
    c3_dir = run_dir / C3_NAME
    
    # Verify child directory exists with core files
    required_core = ["daily_account.csv", "orders.csv", "metrics.json", "config_snapshot.json",
                     "input_hashes.json", "gate_result.json", "parameter_freeze.json"]
    for fname in required_core:
        if not (c3_dir / fname).exists():
            raise RuntimeError(f"FINALIZE_MISSING_CORE:{fname}")
    
    # Compute SHA256 of core account files BEFORE changes
    core_files_to_protect = [
          "daily_account.csv", "orders.csv", "fees.csv", "turnover.csv",
          "actual_weights.csv", "target_weights.csv",
      ]
    pre_shas = {f: sha256_file(str(c3_dir / f)) for f in core_files_to_protect}
    
    # Fix market_states.csv: add market_state column if missing
    ms_path = c3_dir / "market_states.csv"
    if ms_path.exists():
        ms_df = pd.read_csv(ms_path)
        if "market_state" not in ms_df.columns:
            ms_df.insert(1, "market_state", "C3_LOW_TURNOVER_MOMENTUM")
            export_table(c3_dir.as_posix(), "market_states.csv", ms_df)
    else:
        # Create minimal market_states if somehow missing (should not happen)
        export_table(
            c3_dir.as_posix(),
            "market_states.csv",
            pd.DataFrame(columns=["signal_date", "market_state", "submit_date", "action", "reason",
                                   "selected_satellites", "max_drift_deviation_pct_points",
                                   "proposed_target_weights", "target_weights"]),
        )
    
    # Load existing config and metrics
    config = json.loads((c3_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    config["config_source_path"] = str(ROOT / CONFIG_PATH)
    metrics = json.loads((c3_dir / "metrics.json").read_text(encoding="utf-8"))
    gate = json.loads((c3_dir / "gate_result.json").read_text(encoding="utf-8"))
    

    # Recalculate sustained turnover from existing child files (requirement #3)
    # Derive initial_build_signal_date from market_states.csv, NOT from old metrics.
    daily_df = pd.read_csv(c3_dir / "daily_account.csv")
    orders_df = pd.read_csv(c3_dir / "orders.csv")

    # Locate INITIAL_BUILD record in market_states.csv
    ms_df = pd.read_csv(ms_path)
    initial_build_rows = ms_df[
        (ms_df["action"].astype(str).str.strip() == "TRIGGER_REBALANCE")
        & (ms_df["reason"].astype(str).str.strip() == "INITIAL_BUILD")
    ]

    if len(initial_build_rows) == 0:
        # No INITIAL_BUILD found -> cannot compute sustained turnover
        metrics["confirmed_turnover_excludes_initial_build"] = False
        metrics["sustained_annual_confirmed_turnover"] = None
        metrics["max_sustained_annual_confirmed_turnover"] = None
        metrics["initial_build_orders_excluded"] = 0
    elif len(initial_build_rows) > 1:
        # Multiple INITIAL_BUILD records -> explicit error
        raise RuntimeError(
            f"FINALIZE_MULTIPLE_INITIAL_BUILD_RECORDS:"
            f"count={len(initial_build_rows)},dates={list(initial_build_rows['signal_date'])}"
        )
    else:
        # Exactly one INITIAL_BUILD record
        ms_initial_date = str(initial_build_rows.iloc[0]["signal_date"]).strip()
        # Normalize to YYYY-MM-DD
        if "T" in ms_initial_date or " " in ms_initial_date:
            ms_initial_date = ms_initial_date.split("T")[0].split(" ")[0]

        # Consistency check against old metrics date (if present)
        old_initial_date = metrics.get("initial_build_signal_date")
        if old_initial_date is not None and str(old_initial_date).strip() != ms_initial_date:
            raise RuntimeError(
                f"FINALIZE_INITIAL_BUILD_DATE_MISMATCH:"
                f"metrics={old_initial_date},market_states={ms_initial_date}"
            )

        metrics["initial_build_signal_date"] = ms_initial_date
        sustained_result = sustained_turnover_excluding_initial(daily_df, orders_df, ms_initial_date)
        metrics["confirmed_turnover_excludes_initial_build"] = True
        metrics["sustained_annual_confirmed_turnover"] = sustained_result["annual_confirmed_turnover"]
        metrics["max_sustained_annual_confirmed_turnover"] = sustained_result["max_annual_confirmed_turnover"]
        metrics["initial_build_orders_excluded"] = sustained_result["excluded_initial_order_count"]

    # Re-run gate with updated metrics (including sustained_confirmed_turnover check)
    static_gates = {
        "data_gate": bool(gate.get("checks", {}).get("data_gate")),
        "rules_gate": bool(gate.get("checks", {}).get("rules_gate")),
        "mapping_gate": bool(gate.get("checks", {}).get("mapping_gate")),
    }
    b2lt_metrics_path = B2LT_BUNDLE_DIR / "metrics.json"
    b2lt_metrics_for_gate = json.loads(b2lt_metrics_path.read_text(encoding="utf-8"))
    new_gate = _c3_gate(metrics, static_gates, b2lt_metrics_for_gate, config)

    # Preserve artifact validation fields from old gate if present
    if "artifact_validation" in gate:
        new_gate["artifact_validation"] = gate["artifact_validation"]
    if "artifact_validation_final" in gate:
        new_gate["artifact_validation_final"] = gate["artifact_validation_final"]

    gate = new_gate

    # Write updated metrics and gate BEFORE manifest refresh
    export_metrics(c3_dir.as_posix(), metrics)
    export_json(c3_dir.as_posix(), "gate_result.json", gate)

    # Refresh child manifest AFTER market_states fix
    write_manifest(
        run_dir=c3_dir.as_posix(),
        strategy_name=C3_NAME,
        config=config,
        db_path=str(ROOT / DB_PATH),
        rules_path=str(ROOT / RULES_PATH),
        exposure_mapping_path=str(ROOT / MAPPING_PATH),
        metrics=metrics,
        gate_result=gate,
    )
    
    # Run artifact validation
    actual_period = config.get("requested_oos_period", [OOS_START, OOS_END])
    run_id = run_dir.name
    artifact_validation = canonical.finalize_artifact_gate(c3_dir, C3_NAME, run_id,
                                                           actual_period[0], actual_period[1],
                                                           config, metrics, gate)
    
    if not artifact_validation.get("passed"):
        errors = artifact_validation.get("errors", [])
        raise RuntimeError(f"C3_FINALIZE_ARTIFACT_VALIDATION_FAILED:errors={errors}")
    
    # Verify core files unchanged after finalize
    post_shas = {f: sha256_file(str(c3_dir / f)) for f in core_files_to_protect}
    for f in core_files_to_protect:
        if pre_shas[f] != post_shas[f]:
            raise RuntimeError(f"FINALIZE_CORE_FILE_MODIFIED:{f}")

    # Write finalize_audit.json to ROOT directory (not child, requirement #4)
    finalize_audit = {
        "mode": "finalize",
        "run_id": run_dir.name,
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "core_files_protected": core_files_to_protect,
        "before_sha256": {f: pre_shas[f] for f in core_files_to_protect},
        "after_sha256": {f: post_shas[f] for f in core_files_to_protect},
        "unchanged": True,
    }
    export_json(run_dir.as_posix(), "finalize_audit.json", finalize_audit)

    # B2LT metrics already loaded above for gate recalculation
    b2lt_metrics_path = B2LT_BUNDLE_DIR / "metrics.json"
    b2lt_metrics = json.loads(b2lt_metrics_path.read_text(encoding="utf-8"))
    
    # Build facts for root artifacts
    facts = _facts(config)
    b2lt_metrics_sha = sha256_file(str(b2lt_metrics_path))
    b2lt_input_hashes_path = B2LT_BUNDLE_DIR / "input_hashes.json"
    if b2lt_input_hashes_path.exists():
        b2lt_input_hashes_sha = sha256_file(str(b2lt_input_hashes_path))
        facts["b2lt_bundle_reference"] = {
            "bundle_dir": str(B2LT_BUNDLE_DIR),
            "metrics_sha256": b2lt_metrics_sha,
            "input_hashes_sha256": b2lt_input_hashes_sha,
        }
    
    # Build status from gate and metrics
    status = _status_from_gate(gate, config)
    status["metrics"] = {C3_NAME: metrics, B2LT_NAME: b2lt_metrics}
    status["stress_tests"] = {}  # Not reloaded in finalize mode
    status["gate_result"] = {C3_NAME: gate}
    
    # Export root-level artifacts BEFORE manifest (requirement: all files on disk before write_manifest)
    export_config_snapshot(run_dir.as_posix(), config)
    export_input_hashes(run_dir.as_posix(), facts["input_hashes"])
    export_json(run_dir.as_posix(), "c3_input_facts.json", facts)
    export_json(run_dir.as_posix(), "c3_status.json", status)
    _write_conclusion(run_dir, config, facts, {C3_NAME: metrics, B2LT_NAME: b2lt_metrics}, gate, {})

    # Write root manifest AFTER all other root files are on disk (finalize_audit/status/facts/config/input/conclusion)
    write_manifest(
        run_dir=run_dir.as_posix(),
        strategy_name=C3_NAME,
        config=config,
        db_path=str(ROOT / DB_PATH),
        rules_path=str(ROOT / RULES_PATH),
        exposure_mapping_path=str(ROOT / MAPPING_PATH),
        metrics=metrics,
        gate_result=gate,
    )
    
    print(json.dumps({
        "mode": "finalize",
        "run_id": run_id,
        "status": status["status"],
        "gate_passed": gate.get("gate_passed"),
        "failed_checks": gate.get("failed_checks", []),
        "artifact_validation_passed": artifact_validation.get("passed"),
        "core_files_verified": True,
    }, ensure_ascii=False, indent=2))
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C3 Low-Turnover Momentum Runner")
    parser.add_argument("--finalize-run", type=str, help="Finalize an existing run directory without re-running backtest")
    args = parser.parse_args()
    
    if args.finalize_run:
        raise SystemExit(_finalize_run(args.finalize_run))
    else:
        raise SystemExit(_run_research())
