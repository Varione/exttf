"""Unique source of truth for research status per P0-B.

Written atomically by experiment orchestrator after successful completion.
All conclusion.md, planning_status_check.md and summaries must derive from
this file and corresponding run artifacts; no numbers maintained separately.

Status enum:
- BUILD_FAILED
- TEST_FAILED
- DATA_GATE_FAILED
- EXPERIMENT_INVALID
- OOS_NOT_RUN
- OOS_GATE_FAILED
- PAPER_TRADE_CANDIDATE
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATUS_PATH = Path("reports/latest_research_status.json")


def _atomic_write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON atomically via temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(str(tmp), str(path))


def build_status(
    *,
    run_id: str,
    status: str,
    completed_phases: list[str] | None = None,
    failed_phases: list[str] | None = None,
    test_result: dict[str, Any] | None = None,
    input_hashes: dict[str, Any] | None = None,
    rule_counts_by_status: dict[str, int] | None = None,
    mapping_counts_by_status: dict[str, int] | None = None,
    strategies_executed: list[str] | None = None,
    oos_period: str | None = None,
    requested_oos_period: str | None = None,
    actual_oos_period: str | None = None,
    account_mode: str | None = None,
    rule_scenario: str | None = None,
    historical_rule_status: str | None = None,
    historical_truth_gate: bool | None = None,
    parameter_freeze_id: str | None = None,
    annual_restart_sensitivity_status: str | None = None,
    gate_result: dict[str, Any] | None = None,
    blocking_issues: list[str] | None = None,
    finalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical research status payload."""
    return {
        "run_id": run_id,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "completed_phases": sorted(completed_phases or []),
        "failed_phases": sorted(failed_phases or []),
        "test_result": test_result or {},
        "input_hashes": input_hashes or {},
        "rule_counts_by_status": rule_counts_by_status or {},
        "mapping_counts_by_status": mapping_counts_by_status or {},
        "strategies_executed": sorted(strategies_executed or []),
        "oos_period": oos_period,
        "requested_oos_period": requested_oos_period or oos_period,
        "actual_oos_period": actual_oos_period or oos_period,
        "account_mode": account_mode,
        "rule_scenario": rule_scenario,
        "historical_rule_status": historical_rule_status,
        "historical_truth_gate": historical_truth_gate,
        "parameter_freeze_id": parameter_freeze_id,
        "annual_restart_sensitivity_status": annual_restart_sensitivity_status,
        "gate_result": gate_result or {},
        "blocking_issues": blocking_issues or [],
        "finalization": finalization or {},
    }


def write_status(status_path: str | Path | None = None, **kwargs) -> dict[str, Any]:
    """Build and atomically write latest_research_status.json."""
    path = Path(status_path or DEFAULT_STATUS_PATH)
    payload = build_status(**kwargs)
    _atomic_write_json(path, payload)
    return payload


def read_status(status_path: str | Path | None = None) -> dict[str, Any] | None:
    """Read latest_research_status.json if it exists."""
    path = Path(status_path or DEFAULT_STATUS_PATH)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mark_stale_if_inputs_changed(
    current_input_hashes: dict[str, str],
    status_path: str | Path | None = None,
) -> bool:
    """Mark status STALE_INPUTS if input hashes differ from last run."""
    existing = read_status(status_path)
    if existing is None:
        return False

    prev_hashes = existing.get("input_hashes", {})
    for key, value in current_input_hashes.items():
        if prev_hashes.get(key) != value:
            existing["status"] = "STALE_INPUTS"
            existing["stale_since"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_json(status_path or DEFAULT_STATUS_PATH, existing)
            return True
    return False


def run_pytest_summary(
    test_dir: str | None = None,
    *,
    warning_error: bool = False,
) -> dict[str, Any]:
    """Run pytest and return a minimal summary for status embedding."""
    import subprocess

    cmd = [sys.executable, "-m", "pytest"]
    if warning_error:
        cmd[1:1] = ["-W", "error::FutureWarning"]
    if test_dir:
        cmd.extend(["-q", "--tb=line", test_dir])
    else:
        cmd.extend(["-q", "--tb=line"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        stdout = result.stdout.strip()
        last_line = stdout.splitlines()[-1] if stdout.splitlines() else ""

        def count(label: str) -> int:
            match = re.search(rf"(\d+)\s+{label}", last_line)
            return int(match.group(1)) if match else 0

        passed = count("passed")
        failed = count("failed")
        errors = count("errors?")

        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "all_passed": failed == 0 and errors == 0,
            "raw_summary": last_line,
        }
    except Exception as e:
        return {
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "all_passed": False,
            "error": str(e),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage latest_research_status.json")
    sub = parser.add_subparsers(dest="cmd")

    write_p = sub.add_parser("write", help="Write status")
    write_p.add_argument("--run-id", required=True)
    write_p.add_argument("--status", required=True)
    write_p.add_argument("--completed-phases", nargs="*", default=[])
    write_p.add_argument("--failed-phases", nargs="*", default=[])
    write_p.add_argument("--strategies-executed", nargs="*", default=[])

    sub.add_parser("read", help="Read current status")

    args = parser.parse_args()

    if args.cmd == "write":
        payload = write_status(
            run_id=args.run_id,
            status=args.status,
            completed_phases=args.completed_phases,
            failed_phases=args.failed_phases,
            strategies_executed=args.strategies_executed,
        )
        print(json.dumps(payload, indent=2))
    elif args.cmd == "read":
        existing = read_status()
        if existing:
            print(json.dumps(existing, indent=2))
        else:
            print("No status file found", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
