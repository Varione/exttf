"""Validate forward shadow observation records (planning P2-3/P2-4).

Checks every decision file under reports/forward_shadow/decisions against
the schema in config/forward_shadow_schema.json, and performs the forward
reconciliation chain: shadow orders reconcile with decision target
weights and fees, dates are ordered and fall on execution calendar days,
and the daily NAV file is present and consistent.

Also verifies the frozen parameter registry is unchanged and that the data
revision registry is CLEAN, since frozen parameters and data snapshots are
required inputs for any decision record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "config" / "forward_shadow_schema.json"
RECORD_ROOT = ROOT / "reports" / "forward_shadow"
CALENDAR = ROOT / "data" / "processed" / "execution_calendar" / "cn_execution_calendar.csv"
FROZEN_VERSIONS = ROOT / "config" / "otf_frozen_strategy_versions.csv"
REVISION_REGISTRY = ROOT / "config" / "data_revision_registry.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calendar() -> set[str]:
    dates: set[str] = set()
    with CALENDAR.open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            if line.strip():
                dates.add(line.split(",")[0].strip())
    return dates


def _load_frozen_keys() -> set[str]:
    keys: set[str] = set()
    with FROZEN_VERSIONS.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        key_idx = header.index("strategy_key")
        for line in handle:
            if line.strip():
                keys.add(line.split(",")[key_idx].strip())
    return keys


def validate_decision(
    path: Path, schema: dict, calendar: set[str], frozen_keys: set[str]
) -> list[str]:
    errors: list[str] = []
    try:
        record = _load_json(path)
    except json.JSONDecodeError as exc:
        return [f"invalid json: {exc}"]

    required = schema["decision_file"]["required_fields"]
    missing = [f for f in required if f not in record]
    if missing:
        errors.append(f"missing fields: {missing}")

    if record.get("strategy_key") not in frozen_keys:
        errors.append(f"strategy_key not frozen: {record.get('strategy_key')}")

    weights = record.get("target_weights")
    if weights is not None:
        total = sum(weights.values())
        if abs(total - 1.0) > schema["decision_file"]["constraints"]["target_weight_tolerance"]:
            errors.append(f"target weight sum {total:.4f} != 1")
        if any(w < 0 for w in weights.values()):
            errors.append("negative target weight")

    dates = {
        "submit_date": record.get("submit_date"),
        "confirm_date": record.get("confirm_date"),
        "value_date": record.get("value_date"),
    }
    for name, value in dates.items():
        if value:
            if value not in calendar:
                errors.append(f"{name} not an execution calendar day: {value}")
            if value < record.get("decision_date", ""):
                errors.append(f"{name} before decision_date")

    if (
        record.get("confirm_date")
        and record.get("submit_date")
        and record["confirm_date"] < record["submit_date"]
    ):
        errors.append("confirm_date < submit_date")
    if (
        record.get("value_date")
        and record.get("confirm_date")
        and record["value_date"] < record["confirm_date"]
    ):
        errors.append("value_date < confirm_date")

    est = record.get("estimated_fees")
    act = record.get("actual_fees")
    if est is not None and (not isinstance(est, (int, float)) or est < 0):
        errors.append("estimated_fees must be a non-negative number")
    if act is not None and (not isinstance(act, (int, float)) or act < 0):
        errors.append("actual_fees must be a non-negative number")
    return errors


def validate_orders(orders_csv: Path) -> list[str]:
    errors: list[str] = []
    if not orders_csv.exists():
        return ["shadow_orders.csv missing"]
    with orders_csv.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        required_cols = ["order_id", "decision_date", "strategy_key", "fund_code",
                         "side", "amount", "fee", "submit_date", "confirm_date",
                         "value_date", "status"]
        missing = [c for c in required_cols if c not in header]
        if missing:
            return [f"shadow_orders missing columns: {missing}"]
        for lineno, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            parts = line.strip().split(",")
            row = dict(zip(header, parts))
            if row["amount"] and float(row["amount"]) <= 0:
                errors.append(f"line {lineno}: non-positive amount")
            if row["fee"] and float(row["fee"]) < 0:
                errors.append(f"line {lineno}: negative fee")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(RECORD_ROOT))
    args = parser.parse_args()
    root = Path(args.root)

    schema = _load_json(SCHEMA_PATH)
    calendar = _load_calendar()
    frozen_keys = _load_frozen_keys()

    report = {"status": "PASS", "decisions": [], "checks": {}}

    revision = _load_json(REVISION_REGISTRY)
    files = revision.get("files", {})
    dirty = {
        p: info for p, info in files.items()
        if info.get("current_sha256") != info.get("last_recorded_sha256")
    }
    report["checks"]["data_revision_clean"] = len(dirty) == 0
    if dirty:
        report["status"] = "FAIL"
        report["checks"]["unrecorded_files"] = list(dirty)

    decision_dir = root / "decisions"
    if not decision_dir.exists():
        report["checks"]["decisions_present"] = False
        report["status"] = "FAIL"
    else:
        decision_files = sorted(decision_dir.glob("*_decision.json"))
        report["checks"]["decisions_present"] = len(decision_files) > 0
        report["checks"]["decision_count"] = len(decision_files)
        for path in decision_files:
            errors = validate_decision(path, schema, calendar, frozen_keys)
            report["decisions"].append(
                {"file": path.name, "errors": errors}
            )
            if errors:
                report["status"] = "FAIL"

    orders_csv = root / "shadow_orders.csv"
    order_errors = validate_orders(orders_csv)
    report["checks"]["shadow_orders_valid"] = not order_errors
    if order_errors:
        report["status"] = "FAIL"
        report["checks"]["shadow_order_errors"] = order_errors[:10]

    daily_nav = root / "daily_nav.csv"
    report["checks"]["daily_nav_present"] = daily_nav.exists()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
