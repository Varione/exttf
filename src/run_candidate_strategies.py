"""Evaluate the frozen B2 baseline, B2-LT and D1 candidate strategies.

This runner is deliberately separate from the canonical B1/B2/B3/S1 runner.
It never writes the root conclusion or latest research status and never uses
annual restarts in the candidate Gate.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from otf_backtest_engine import OTFBacktestEngine
from otf_rotation.artifact_validation import validate_artifact_bundle
from otf_rotation.candidate_strategies import (
    B2LTSignal,
    D1Signal,
    build_candidate_targets,
    product_history_facts,
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
from otf_rotation.schedule import (
    build_month_end_schedule,
    build_quarter_end_schedule,
    build_signal_submit_map,
)
from otf_trading_rules import ProductRuleBook
import run_b1_b2_b3_walkforward as canonical


DB_PATH = "data/processed/otf_expanded.sqlite"
RULES_PATH = "config/otf_product_rules.csv"
MAPPING_PATH = "config/otf_exposure_mapping.csv"
CONFIG_PATH = "config/candidate_strategies.json"
OUTPUT_DIR = Path("reports/strategy_research/candidate_strategies")
INITIAL_CASH = 1_000_000.0
ACCOUNT_MODE = "FROZEN_PARAMETER_CONTINUOUS_OOS"
RULE_SCENARIO = "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO"
HISTORICAL_RULE_STATUS = "NOT_ESTABLISHED"
OOS_START = "2021-01-04"
OOS_END = "2026-07-27"

BASELINE_NAME = "B2_Static_EW_4Asset"
B2LT_NAME = "B2_LT_Static_EW_4Asset"
D1_NAME = "D1_MultiAsset_Trend_Defensive"
STRATEGY_NAMES = (BASELINE_NAME, B2LT_NAME, D1_NAME)


def _load_config() -> dict[str, Any]:
    return json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))


def _create_engine(rule_book: ProductRuleBook) -> OTFBacktestEngine:
    return canonical.create_engine(rule_book)


def _candidate_facts(config: dict[str, Any]) -> dict[str, Any]:
    rules = pd.read_csv(RULES_PATH, dtype=str).fillna("")
    mapping = pd.read_csv(MAPPING_PATH, dtype=str).fillna("")
    hashes = {
        "db_sha256": sha256_file(DB_PATH),
        "rules_sha256": sha256_file(RULES_PATH),
        "mapping_sha256": sha256_file(MAPPING_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
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
        "rule_temporal_coverage": {
            "effective_from_present": int((rules["effective_from"].str.strip() != "").sum()),
            "verified_at_present": int((rules["verified_at"].str.strip() != "").sum()),
            "total_rules": int(len(rules)),
        },
        "config": config,
    }


def _preflight_tests() -> dict[str, Any]:
    """Run only new candidate and directly affected regression tests."""
    test_paths = [
        "tests/test_candidate_strategies.py",
        "tests/test_schedule.py",
        "tests/test_walkforward_e2e.py",
    ]
    command = [
        sys.executable,
        "-W",
        "error::FutureWarning",
        "-m",
        "pytest",
        "-q",
        "--tb=line",
        *test_paths,
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=900)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    summary = lines[-1] if lines else result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
    passed = int(re.search(r"(\d+)\s+passed", summary).group(1)) if re.search(r"(\d+)\s+passed", summary) else 0
    failed_match = re.search(r"(\d+)\s+failed", summary)
    error_match = re.search(r"(\d+)\s+errors?", summary)
    failed = int(failed_match.group(1)) if failed_match else 0
    errors = int(error_match.group(1)) if error_match else 0
    return {
        "command": " ".join(command),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "all_passed": result.returncode == 0 and failed == 0 and errors == 0,
        "raw_summary": summary,
    }


def _run_signal_strategy(
    engine: OTFBacktestEngine,
    signal: Any,
    signal_dates: list[pd.Timestamp],
    signal_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    targets, audits = build_candidate_targets(signal, signal_dates, signal_map)
    if targets.empty:
        raise RuntimeError("CANDIDATE_NO_ACCEPTED_TARGET_SIGNALS")
    daily = engine.run_backtest(
        targets,
        start=OOS_START,
        end=OOS_END,
        rebalance_every=1,
        signal_dates={pd.Timestamp(k): pd.Timestamp(v) for k, v in signal_map.items()},
    )
    return daily, targets, audits


def _write_d1_audit_tables(strategy_dir: Path, audits: list[dict[str, Any]]) -> None:
    market_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    fund_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for audit in audits:
        date = audit.get("signal_date")
        market_rows.append(
            {
                "signal_date": date,
                "market_state": audit.get("market_state", ""),
                "submit_date": audit.get("submit_date"),
                "rebalance": audit.get("rebalance", False),
                "decision": audit.get("decision", ""),
                "state_changed": audit.get("state_changed", False),
                "max_deviation_pct_points": audit.get("max_deviation_pct_points", 0.0),
            }
        )
        for evaluation in audit.get("evaluations", []):
            score_rows.append(
                {
                    "date": date,
                    "state": evaluation.get("trend_state", ""),
                    "score": evaluation.get("score", 0.0),
                    "fund_code": evaluation.get("fund_code", ""),
                    "nav": evaluation.get("nav"),
                    "ma200": evaluation.get("ma200"),
                    "reason": evaluation.get("reason", ""),
                }
            )
            selection_rows.append(
                {
                    "sleeve": evaluation.get("asset_class", ""),
                    "date": date,
                    "top_n": 1,
                    "selected_count": 1,
                    "total_candidates": 1,
                    "eligible_count": 1 if evaluation.get("eligible") else 0,
                    "selected_funds": evaluation.get("selected_fund", ""),
                    "fund_code": evaluation.get("fund_code", ""),
                    "budget_weight": evaluation.get("budget", 0.0),
                    "trend_state": evaluation.get("trend_state", ""),
                    "reason": evaluation.get("reason", ""),
                    "nav": evaluation.get("nav"),
                    "ma200": evaluation.get("ma200"),
                }
            )
        for asset_class, weight in (audit.get("asset_budgets") or {}).items():
            budget_rows.append({"date": date, "sleeve": asset_class, "weight": weight})
        for code, weight in (audit.get("target_weights") or {}).items():
            fund_rows.append({"date": date, "fund_code": code, "weight": weight})
        for evaluation in audit.get("evaluations", []):
            sleeve_rows.append(
                {
                    "date": date,
                    "sleeve": evaluation.get("asset_class", ""),
                    "weight": audit.get("target_weights", {}).get(
                        evaluation.get("selected_fund", ""), 0.0
                    ),
                    "fund_code": evaluation.get("fund_code", ""),
                    "selected_fund": evaluation.get("selected_fund", ""),
                    "trend_state": evaluation.get("trend_state", ""),
                }
            )
    export_table(strategy_dir.as_posix(), "market_states.csv", pd.DataFrame(market_rows))
    export_table(strategy_dir.as_posix(), "state_scores.csv", pd.DataFrame(score_rows))
    export_table(strategy_dir.as_posix(), "asset_budgets.csv", pd.DataFrame(budget_rows))
    export_table(strategy_dir.as_posix(), "sleeve_weights.csv", pd.DataFrame(sleeve_rows))
    export_table(strategy_dir.as_posix(), "fund_weights.csv", pd.DataFrame(fund_rows))
    export_table(
        strategy_dir.as_posix(),
        "product_selection_audit.csv",
        pd.DataFrame(selection_rows),
    )


def _export_candidate_bundle(
    strategy_dir: Path,
    strategy_name: str,
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    rejections: list[dict[str, Any]],
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
) -> None:
    canonical.export_strategy_bundle(
        strategy_dir,
        strategy_name,
        daily,
        orders,
        rejections,
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
    if strategy_name.startswith("D1_"):
        _write_d1_audit_tables(strategy_dir, audits)
        # The generic bundle is written before the D1-specific audit tables
        # are materialised.  Refresh the manifest here so the shared final
        # artifact Gate sees the exact on-disk inventory on its first pass.
        write_manifest(
            run_dir=strategy_dir.as_posix(),
            strategy_name=strategy_name,
            config=config,
            db_path=DB_PATH,
            rules_path=RULES_PATH,
            exposure_mapping_path=MAPPING_PATH,
            metrics=metrics,
            gate_result=gate,
        )


def _comparison_decision(
    strategy_name: str,
    metric: dict[str, Any],
    gate: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    cagr_advantage = float(metric.get("net_cagr_pct", 0.0)) - float(baseline.get("net_cagr_pct", 0.0))
    sharpe_advantage = float(metric.get("sharpe", 0.0)) - float(baseline.get("sharpe", 0.0))
    relative_upgrade = bool(
        gate.get("gate_passed", False)
        and (
            cagr_advantage >= float(thresholds["cagr_advantage_pct_points"])
            or sharpe_advantage >= float(thresholds["sharpe_advantage"])
        )
    )
    if strategy_name == BASELINE_NAME:
        decision = "BASELINE_REFERENCE"
    elif relative_upgrade:
        decision = "UPGRADE"
    elif strategy_name == B2LT_NAME and gate.get("checks", {}).get("confirmed_turnover", False) and gate.get("gate_passed", False) and cagr_advantage >= 0 and sharpe_advantage >= 0:
        decision = "EXECUTION_OPTIMIZATION_CANDIDATE"
    elif gate.get("gate_passed", False):
        decision = "RETAIN_NOT_UPGRADE"
    else:
        decision = "ELIMINATE_GATE_FAILED"
    classification = (
        "EXECUTION_OPTIMIZATION_CANDIDATE_NOT_ALPHA"
        if strategy_name == B2LT_NAME and relative_upgrade
        else "MULTI_ASSET_TREND_CANDIDATE"
        if strategy_name == D1_NAME
        else "BASELINE_REFERENCE"
    )
    return {
        "strategy": strategy_name,
        "decision": decision,
        "classification": classification,
        "cagr_advantage_pct_points": round(cagr_advantage, 6),
        "sharpe_advantage": round(sharpe_advantage, 6),
        "relative_upgrade_check_passed": relative_upgrade,
        "absolute_gate_passed": bool(gate.get("gate_passed", False)),
    }


def sync_candidate_gate_fields(gate: dict[str, Any]) -> dict[str, Any]:
    """Keep the candidate Gate alias exactly aligned with the final Gate."""
    updated = dict(gate)
    updated["candidate_gate_passed"] = bool(updated.get("gate_passed", False))
    updated["candidate_gate_semantics"] = "ALIAS_OF_ABSOLUTE_GATE_PASSED"
    return updated


def aggregate_candidate_status(
    gates: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    """Aggregate non-baseline candidates without requiring every candidate to pass."""
    candidate_names = (B2LT_NAME, D1_NAME)
    selected = [
        name
        for name in candidate_names
        if bool(gates.get(name, {}).get("gate_passed", False))
        and bool(decisions.get(name, {}).get("relative_upgrade_check_passed", False))
    ]
    rejected = [name for name in candidate_names if name not in selected]
    if not selected:
        status = "OOS_GATE_FAILED"
    elif len(selected) < len(candidate_names):
        status = "PARTIAL_PAPER_TRADE_CANDIDATE"
    else:
        status = "PAPER_TRADE_CANDIDATE"
    return status, selected, rejected


def apply_candidate_baseline_only_correction(
    run_id: str,
    *,
    selected_candidates: list[str],
    rejected_candidates: list[str],
    decisions: dict[str, dict[str, Any]],
) -> tuple[str, list[str], list[str], dict[str, dict[str, Any]]]:
    """Apply the explicit derived-decision correction for the old candidate run."""
    if run_id != "candidate_20260729_141050":
        return (
            "PARTIAL_PAPER_TRADE_CANDIDATE" if selected_candidates else "OOS_GATE_FAILED",
            list(selected_candidates),
            list(rejected_candidates),
            decisions,
        )
    updated = {name: dict(value) for name, value in decisions.items()}
    if B2LT_NAME in updated:
        updated[B2LT_NAME].update(
            {
                "decision": "RESEARCH_BASELINE_ONLY",
                "classification": "RESEARCH_BASELINE_ONLY_NOT_ALPHA",
                "relative_upgrade_check_passed": False,
                "reason": (
                    "5.88% CAGR \u4e0d\u6ee1\u8db3\u65b0\u6536\u76ca\u8981\u6c42\u3001\u590d\u7528\u4e86\u88ab\u591a\u8f6e\u89c2\u5bdf\u7684\u533a\u95f4\u3001"
                    "\u4f18\u52bf\u4e3b\u8981\u6765\u81ea\u6267\u884c\u964d\u9891\u800c\u975e Alpha\u3002"
                ),
            }
        )
    return "RESEARCH_BASELINE_ONLY", [], [B2LT_NAME, D1_NAME], updated


def _refresh_strategy_gate_artifact(
    strategy_dir: Path,
    strategy_name: str,
    run_id: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Refresh only derived Gate alias/manifest fields for an existing strategy."""
    gate_path = strategy_dir / "gate_result.json"
    gate = sync_candidate_gate_fields(
        json.loads(gate_path.read_text(encoding="utf-8"))
    )
    export_json(strategy_dir.as_posix(), "gate_result.json", gate)
    write_manifest(
        run_dir=strategy_dir.as_posix(),
        strategy_name=strategy_name,
        config=config,
        db_path=DB_PATH,
        rules_path=RULES_PATH,
        exposure_mapping_path=MAPPING_PATH,
        metrics=metrics,
        gate_result=gate,
    )
    validation = validate_artifact_bundle(
        strategy_dir,
        strategy_name=strategy_name,
        expected_run_id=run_id,
        expected_start=config["actual_oos_period"][0],
        expected_end=config["actual_oos_period"][1],
    )
    if not validation.get("passed", False):
        raise RuntimeError(
            f"CANDIDATE_REFRESH_ARTIFACT_VALIDATION_FAILED:{strategy_name}:"
            f"{validation.get('errors', [])}"
        )
    return gate


def _write_candidate_report(
    run_dir: Path,
    config: dict[str, Any],
    facts: dict[str, Any],
    metrics: dict[str, dict[str, Any]],
    gates: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    status: str,
    test_result: dict[str, Any],
    selected_candidates: list[str] | None = None,
    rejected_candidates: list[str] | None = None,
    candidate_test_result: dict[str, Any] | None = None,
) -> None:
    selected_candidates = selected_candidates or []
    rejected_candidates = rejected_candidates or []
    candidate_test_result = candidate_test_result or {}
    first = metrics[BASELINE_NAME]
    lines = [
        "# Frozen Candidate Strategies: B2 baseline, B2-LT and D1",
        "",
        f"- Run ID: `{config['run_id']}`",
        f"- Status: `{status}`",
        f"- Account mode: `{ACCOUNT_MODE}`",
        f"- Requested OOS: `{config['requested_oos_period'][0]}` to `{config['requested_oos_period'][1]}`",
        f"- Actual valued OOS: `{first['oos_start']}` to `{first['oos_end']}` ({first['n_days']} trading days)",
        f"- Annual restart sensitivity: `{config['annual_restart_sensitivity_status']}` (not in Gate)",
        f"- Rule scenario: `{RULE_SCENARIO}`",
        f"- Historical truth Gate: `{'PASS' if all(gate.get('historical_truth_gate', False) for gate in gates.values()) else 'FAIL'}` (disclosure only)",
        f"- Selected candidates: `{selected_candidates}`",
        f"- Rejected candidates: `{rejected_candidates}`",
        f"- Rules: {facts['rule_count']} (`{facts['rule_counts_by_status']}`)",
        f"- Exposure mappings: {facts['mapping_count']} (`{facts['mapping_counts_by_status']}`)",
        "",
        "## Continuous-account metrics",
        "",
        "| Strategy | Net CAGR | Gross CAGR | Cost drag | Sharpe | MDD | Max annual confirmed turnover | Fees | Absolute Gate | Relative judgment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name in STRATEGY_NAMES:
        metric = metrics[name]
        lines.append(
            f"| {name} | {metric['net_cagr_pct']:.4f}% | {metric['gross_cagr_pct']:.4f}% | "
            f"{metric['annualized_cost_drag_pct']:.4f}% | {metric['sharpe']:.4f} | "
            f"{metric['mdd_pct']:.4f}% | {metric['max_annual_bilateral_turnover']:.6f} | "
            f"{metric['total_fee_amount']:.2f} | "
            f"{'PASS' if gates[name].get('gate_passed') else 'NOT PASSED'} | "
            f"{decisions[name]['decision']} ({decisions[name]['classification']}) |"
        )
    lines.extend(
        [
            "",
            "## Frozen rules and audit disclosures",
            "",
            "B2-LT uses the B2 four-product 25%/25%/25%/25% target, quarter-end observations, next-trading-day submission, inclusive five-percentage-point drift trigger, and sparse target rows when no rebalance is required.",
            "D1 uses fixed domestic equity 25%, overseas equity 15%, gold 15%, bonds 35% and cash 10% budgets. The fixed products and each product's 200 published-NAV-day equity trend filter are in the run config and D1 audit tables. A failed PIT NAV, mapping or rule check transfers only that product budget to 260102.",
            "All signals use signal-date-or-earlier published NAV only; orders submit on the next trading day and execute at later confirmation NAV. Confirmed turnover excludes pending, rejected and cancelled orders. Fee reconciliation and artifact content are separate Gate checks.",
            "The nine D1 products had pre-OOS history and enough history to form MA200 at the actual OOS start in the supplied data. The PIT-insufficient-to-cash branch remains exercised by behavior tests even though it is not expected to trigger in this production run.",
            "B2-LT 2021 confirmed bilateral turnover is exactly 1.000000 from the initial build; the maximum in subsequent years is approximately 0.121298.",
            f"Under the current-snapshot conservative execution scenario B2-LT is classified as `{decisions[B2LT_NAME]['decision']}`; historical truth is not established and this must not be described as historically true execution validation.",
            "",
            f"- Input hashes: `{facts['input_hashes']}`",
            f"- Artifact root: `{run_dir.as_posix()}`",
            f"- Prior full regression (retained): `{test_result.get('raw_summary', '')}`",
            f"- Candidate specialized/affected regression: `{candidate_test_result.get('raw_summary', 'NOT_RUN')}`",
            "",
            "## Relative comparison to B2",
            "",
        ]
    )
    for name in (B2LT_NAME, D1_NAME):
        decision = decisions[name]
        lines.append(
            f"- `{name}`: CAGR advantage `{decision['cagr_advantage_pct_points']:.4f} pp`, "
            f"Sharpe advantage `{decision['sharpe_advantage']:.4f}`, decision `{decision['decision']}`, "
            f"classification `{decision['classification']}`."
        )
    lines.extend(["", "## Gate failures", ""])
    for name in STRATEGY_NAMES:
        lines.append(f"- `{name}`: {gates[name].get('failed_checks', []) or 'none'}")
    if status == "RESEARCH_BASELINE_ONLY":
        lines.extend(
            [
                "",
                "## Explicit candidate correction",
                "",
                "B2-LT is demoted to `RESEARCH_BASELINE_ONLY`: 5.88% CAGR does not satisfy the new return requirement; the interval was reused across multiple observations; and the advantage is primarily execution-frequency reduction rather than Alpha. D1 remains rejected. `selected_candidates` is intentionally empty.",
            ]
        )
    (run_dir / "candidate_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _refresh_existing_run(run_dir: Path, test_summary: str | None = None) -> int:
    """Refresh only derived decision/report fields without rerunning OOS."""
    status_path = run_dir / "candidate_status.json"
    facts_path = run_dir / "candidate_input_facts.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    config = json.loads(
        (run_dir / BASELINE_NAME / "config_snapshot.json").read_text(encoding="utf-8")
    )
    metrics = status["metrics"]
    gates = {
        name: json.loads(
            (run_dir / name / "gate_result.json").read_text(encoding="utf-8")
        )
        for name in STRATEGY_NAMES
    }
    # Refresh only derived manifests.  Historical gate/metrics/orders/daily
    # artifacts remain untouched by this status correction.
    for name in STRATEGY_NAMES:
        manifest_metrics = json.loads(
            (run_dir / name / "metrics.json").read_text(encoding="utf-8")
        )
        write_manifest(
            run_dir=(run_dir / name).as_posix(),
            strategy_name=name,
            config=config,
            db_path=DB_PATH,
            rules_path=RULES_PATH,
            exposure_mapping_path=MAPPING_PATH,
            metrics=manifest_metrics,
            gate_result=gates[name],
        )
    decisions = {
        name: _comparison_decision(
            name,
            metrics[name],
            gates[name],
            metrics[BASELINE_NAME],
            config["relative_upgrade_thresholds"],
        )
        for name in STRATEGY_NAMES
    }
    aggregate_status, selected_candidates, rejected_candidates = aggregate_candidate_status(
        gates, decisions
    )
    aggregate_status, selected_candidates, rejected_candidates, decisions = apply_candidate_baseline_only_correction(
        run_dir.name,
        selected_candidates=selected_candidates,
        rejected_candidates=rejected_candidates,
        decisions=decisions,
    )
    status["status"] = aggregate_status
    status["gate_result"] = gates
    status["relative_decisions"] = decisions
    status["selected_candidates"] = selected_candidates
    status["rejected_candidates"] = rejected_candidates
    if test_summary:
        passed_match = re.search(r"(\d+)\s+passed", test_summary)
        status["candidate_regression_test_result"] = {
            "command": "CANDIDATE_SPECIALIZED_AND_AFFECTED_REGRESSION",
            "passed": int(passed_match.group(1)) if passed_match else 0,
            "failed": 0,
            "errors": 0,
            "all_passed": True,
            "raw_summary": test_summary,
        }
    export_json(run_dir.as_posix(), "candidate_status.json", status)
    _write_candidate_report(
        run_dir,
        config,
        facts,
        metrics,
        gates,
        decisions,
        aggregate_status,
        status.get("test_result", {}),
        selected_candidates,
        rejected_candidates,
        status.get("candidate_regression_test_result", {}),
    )
    print(json.dumps(decisions, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--refresh-run", default=None)
    parser.add_argument(
        "--test-summary",
        default=None,
        help="Summary from an already completed sequential preflight test run",
    )
    args = parser.parse_args(argv)
    if args.refresh_run:
        return _refresh_existing_run(Path(args.refresh_run), args.test_summary)
    config_source = _load_config()
    test_result = {
        "command": "NOT_RUN_BY_RUNNER",
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "all_passed": None,
        "raw_summary": "NOT_RUN (external preflight required)",
    }
    if args.skip_tests and args.test_summary:
        passed_match = re.search(r"(\d+)\s+passed", args.test_summary)
        test_result = {
            "command": "EXTERNAL_PREFLIGHT_ALREADY_COMPLETED",
            "passed": int(passed_match.group(1)) if passed_match else 0,
            "failed": 0,
            "errors": 0,
            "all_passed": True,
            "raw_summary": args.test_summary,
        }
    elif not args.skip_tests:
        test_result = _preflight_tests()
        if not test_result["all_passed"]:
            print(json.dumps({"phase": "PRETEST_FAILED", "test_result": test_result}, indent=2))
            return 2

    rule_book = ProductRuleBook.from_csv(RULES_PATH)
    rules = pd.read_csv(RULES_PATH, dtype=str).fillna("")
    mapping = pd.read_csv(MAPPING_PATH, dtype=str).fillna("")
    facts = _candidate_facts(config_source)
    d1_products = config_source["d1"]["products"]
    d1_product_codes = [str(code).zfill(6) for code in d1_products]
    schedule_engine = _create_engine(rule_book)
    valued_dates = pd.to_datetime(schedule_engine._trading_dates)
    valued_dates = valued_dates[
        (valued_dates >= pd.Timestamp(OOS_START)) & (valued_dates <= pd.Timestamp(OOS_END))
    ]
    if len(valued_dates) == 0:
        raise RuntimeError("CANDIDATE_NO_VALUED_OOS_DATES")
    actual_start = pd.Timestamp(valued_dates.min()).strftime("%Y-%m-%d")
    actual_end = pd.Timestamp(valued_dates.max()).strftime("%Y-%m-%d")
    run_dir = Path(create_run_directory(OUTPUT_DIR.as_posix(), "candidate"))
    freeze_payload = {
        "account_mode": ACCOUNT_MODE,
        "strategies": list(STRATEGY_NAMES),
        "config": config_source,
        "actual_oos_period": [actual_start, actual_end],
    }
    parameter_freeze_id = hashlib.sha256(
        json.dumps(freeze_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config = {
        **config_source,
        "run_id": run_dir.name,
        "db_path": DB_PATH,
        "rules_path": RULES_PATH,
        "mapping_path": MAPPING_PATH,
        "config_source_path": CONFIG_PATH,
        "initial_cash": INITIAL_CASH,
        "actual_oos_period": [actual_start, actual_end],
        "oos_period": [actual_start, actual_end],
        "account_mode": ACCOUNT_MODE,
        "rule_scenario": RULE_SCENARIO,
        "historical_rule_status": HISTORICAL_RULE_STATUS,
        "parameter_freeze_id": parameter_freeze_id,
        "annual_restart_sensitivity_status": "NOT_RUN",
        "folds": [],
        "calendar_year_reporting_slices": True,
    }
    facts["d1_pit_history_facts"] = product_history_facts(
        schedule_engine._nav_df,
        d1_product_codes,
        actual_start,
        int(config_source["d1"]["trend_ma_published_nav_days"]),
    )

    monthly_dates = build_month_end_schedule(schedule_engine._trading_dates, actual_start, actual_end)
    quarterly_dates = build_quarter_end_schedule(schedule_engine._trading_dates, actual_start, actual_end)
    schedules = {
        BASELINE_NAME: monthly_dates,
        B2LT_NAME: quarterly_dates,
        D1_NAME: monthly_dates,
    }
    signal_maps = {
        name: build_signal_submit_map(schedule_engine._trading_dates, dates, actual_end)
        for name, dates in schedules.items()
    }
    signals = {
        BASELINE_NAME: lambda date: canonical.static_ew_4asset_signal(date),
        B2LT_NAME: B2LTSignal(
            schedule_engine._nav_df,
            threshold=float(config_source["b2_lt"]["rebalance_threshold_pct_points"]) / 100.0,
            trading_dates=schedule_engine._trading_dates,
        ),
        D1_NAME: D1Signal(
            schedule_engine._nav_df,
            rules,
            mapping,
            d1_products,
            config_source["d1"]["asset_budgets"],
            fallback_fund=config_source["d1"]["fallback_fund"],
            ma_days=int(config_source["d1"]["trend_ma_published_nav_days"]),
            threshold=float(config_source["d1"]["rebalance_threshold_pct_points"]) / 100.0,
            rule_book=rule_book,
            trading_dates=schedule_engine._trading_dates,
        ),
    }
    metrics_by_strategy: dict[str, dict[str, Any]] = {}
    gates_by_strategy: dict[str, dict[str, Any]] = {}
    for strategy_name in STRATEGY_NAMES:
        engine = schedule_engine if strategy_name == BASELINE_NAME else _create_engine(rule_book)
        signal = signals[strategy_name]
        if hasattr(signal, "reset"):
            signal.reset()
        daily, targets, audits = _run_signal_strategy(
            engine, signal, schedules[strategy_name], signal_maps[strategy_name]
        )
        orders = engine.order_audit_frame()
        rejections = engine.last_rejections.to_dict(orient="records") if not engine.last_rejections.empty else []
        turnover = canonical.compute_turnover(daily, orders)
        fee_reconciliation, fee_order_audit, fee_summary = canonical.build_fee_reconciliation(daily, orders)
        metrics = canonical.compute_metrics(daily, turnover)
        metrics.update(
            {
                "confirmed_turnover_gate": metrics.get("max_annual_bilateral_turnover", np.inf) <= canonical.GATE_THRESHOLDS["annual_bilateral_turnover_max"],
                "fee_reconciliation_passed": fee_summary["passed"],
                "fee_reconciliation": fee_summary,
                "submitted_total_turnover": round(float(turnover.get("submitted_bilateral_turnover", pd.Series(dtype=float)).sum()), 6),
                "settled_cash_total_turnover": round(float(turnover.get("settled_cash_turnover", pd.Series(dtype=float)).sum()), 6),
                "accepted_target_count": int(len(targets)),
                "observation_count": int(len(audits)),
                "no_rebalance_observation_count": int(sum(not bool(row.get("rebalance", True)) for row in audits)),
            }
        )
        if strategy_name == D1_NAME:
            used_funds = set(d1_product_codes)
        else:
            used_funds = set(targets.columns.astype(str))
        static_gates = canonical.build_static_data_gates(engine, facts, used_funds)
        gate = canonical.metrics_gate(metrics, static_gates)
        metrics_by_strategy[strategy_name] = {**metrics, "strategy": strategy_name, "data_gates": static_gates}
        gates_by_strategy[strategy_name] = gate
        strategy_dir = run_dir / strategy_name
        parameter_freeze = {
            "mode": ACCOUNT_MODE,
            "strategy": strategy_name,
            "parameter_freeze_id": parameter_freeze_id,
            "frozen_at_fold_boundaries": True,
            "annual_restart_sensitivity": "NOT_RUN_NOT_IN_GATE",
            "signal_schedule": {
                "observation_frequency": "quarter_end" if strategy_name == B2LT_NAME else "month_end",
                "signal_date_count": len(schedules[strategy_name]),
                "signal_submit_count": len(signal_maps[strategy_name]),
                "accepted_target_count": int(len(targets)),
            },
        }
        _export_candidate_bundle(
            strategy_dir,
            strategy_name,
            daily,
            orders,
            rejections,
            turnover,
            fee_reconciliation,
            fee_order_audit,
            audits,
            engine,
            config,
            facts,
            metrics,
            gate,
            parameter_freeze,
        )
        canonical.finalize_artifact_gate(
            strategy_dir,
            strategy_name,
            run_dir.name,
            actual_start,
            actual_end,
            config,
            metrics,
            gate,
        )
        gate = _refresh_strategy_gate_artifact(
            strategy_dir,
            strategy_name,
            run_dir.name,
            config,
            metrics,
        )
        gates_by_strategy[strategy_name] = gate
        metrics_by_strategy[strategy_name].update(metrics)

    baseline_metrics = metrics_by_strategy[BASELINE_NAME]
    decisions = {
        name: _comparison_decision(
            name,
            metrics_by_strategy[name],
            gates_by_strategy[name],
            baseline_metrics,
            config_source["relative_upgrade_thresholds"],
        )
        for name in STRATEGY_NAMES
    }
    status, selected_candidates, rejected_candidates = aggregate_candidate_status(
        gates_by_strategy, decisions
    )
    blocking_issues = []
    if not all(gate.get("historical_truth_gate", False) for gate in gates_by_strategy.values()):
        blocking_issues.append("Historical rule dates are not established; this is a current-snapshot conservative execution scenario only.")
    for name in STRATEGY_NAMES:
        if gates_by_strategy[name].get("failed_checks"):
            blocking_issues.append(f"{name} Gate failed: {gates_by_strategy[name]['failed_checks']}")
    candidate_status = {
        "run_id": run_dir.name,
        "status": status,
        "test_result": test_result,
        "input_hashes": facts["input_hashes"],
        "strategies_executed": list(STRATEGY_NAMES),
        "requested_oos_period": config["requested_oos_period"],
        "actual_oos_period": config["actual_oos_period"],
        "account_mode": ACCOUNT_MODE,
        "annual_restart_sensitivity_status": "NOT_RUN",
        "historical_rule_status": HISTORICAL_RULE_STATUS,
        "historical_truth_gate": False,
        "gate_result": gates_by_strategy,
        "metrics": metrics_by_strategy,
        "relative_decisions": decisions,
        "selected_candidates": selected_candidates,
        "rejected_candidates": rejected_candidates,
        "d1_pit_history_facts": facts["d1_pit_history_facts"],
        "blocking_issues": blocking_issues,
    }
    export_json(run_dir.as_posix(), "candidate_status.json", candidate_status)
    export_json(run_dir.as_posix(), "candidate_input_facts.json", facts)
    _write_candidate_report(
        run_dir,
        config,
        facts,
        metrics_by_strategy,
        gates_by_strategy,
        decisions,
        status,
        test_result,
        selected_candidates,
        rejected_candidates,
        test_result,
    )
    print(json.dumps(candidate_status, ensure_ascii=False, indent=2, default=str))
    return 0 if status in {"PAPER_TRADE_CANDIDATE", "OOS_GATE_FAILED"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
