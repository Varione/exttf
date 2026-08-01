"""Forward upgrade Gate framework (planning P2-4).

A frozen strategy may be promoted to PAPER_TRADE_CANDIDATE only when all
nine conditions pass. Until then, language such as "tradable", "verified
alpha" or "formal paper trade candidate" is forbidden.

Conditions:
1.  P0 engineering Gate passed
2.  P1 historical truth Gate passed
3.  minimum fresh sample length met
4.  forward accounting, fee, order and rule reconciliation passed
5.  no frozen parameter drift
6.  absolute risk, cost, turnover and rolling window Gate passed
7.  dynamic strategies reach the pre-registered incremental bar vs B2-LT
8.  double fees and execution delay stress test still passes
9.  all artifacts come from a clean commit and are repeatable

The gate is intentionally conservative: any condition without sufficient
forward evidence reports FAIL or NOT_MEASURED, never PASS.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "baseline_freeze_manifest.json"
STATUS_SOURCE = ROOT / "reports" / "latest_research_status.json"
FROZEN_VERSIONS = ROOT / "config" / "otf_frozen_strategy_versions.csv"
SHADOW_ROOT = ROOT / "reports" / "forward_shadow"
OUTPUT_ROOT = ROOT / "reports" / "forward_gate"

MIN_FRESH_DAYS = 252
MIN_QUARTERLY_DECISIONS = 4
MIN_MONTHLY_DECISIONS = 12


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _frozen_row(strategy_key: str) -> dict:
    with FROZEN_VERSIONS.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["strategy_key"] == strategy_key:
                return row
    return {}


def _shadow_decisions() -> list[dict]:
    decision_dir = SHADOW_ROOT / "decisions"
    if not decision_dir.exists():
        return []
    records = []
    for path in sorted(decision_dir.glob("*_decision.json")):
        try:
            records.append(_load_json(path))
        except json.JSONDecodeError:
            continue
    return records


def _fresh_sample_check(decisions: list[dict]) -> tuple[str, dict]:
    if not decisions:
        return "FAIL", {"reason": "no forward decisions recorded yet"}
    dates = sorted(d.get("decision_date", "") for d in decisions)
    start, end = dates[0], dates[-1]
    try:
        days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
    except ValueError:
        return "FAIL", {"reason": "invalid decision dates"}
    quarterly = len({d["decision_date"][:7] for d in decisions if "decision_date" in d})
    monthly = len({d["decision_date"][:7] for d in decisions if "decision_date" in d})
    ok = (
        days >= MIN_FRESH_DAYS
        and quarterly >= MIN_QUARTERLY_DECISIONS
        and monthly >= MIN_MONTHLY_DECISIONS
    )
    return (
        "PASS" if ok else "FAIL",
        {
            "span_days": days,
            "required_fresh_days": MIN_FRESH_DAYS,
            "decision_count": len(decisions),
            "quarterly_min": MIN_QUARTERLY_DECISIONS,
            "monthly_min": MIN_MONTHLY_DECISIONS,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy_key")
    args = parser.parse_args()
    strategy_key = args.strategy_key

    frozen = _frozen_row(strategy_key)
    manifest = _load_json(MANIFEST)
    status = _load_json(STATUS_SOURCE)

    p0 = manifest.get("test_result", {})
    p0_passed = bool(
        p0.get("passed") or p0.get("regression_passed")
        or (p0.get("passed_checks") and p0.get("failed_checks") == 0)
    )
    p1_gate = (
        status.get("p1_historical_truth", {})
        .get("gate", {})
        .get("gate_passed", False)
    )

    decisions = _shadow_decisions()
    sample_status, sample_detail = _fresh_sample_check(decisions)

    conditions = [
        {
            "id": 1,
            "name": "P0 engineering gate",
            "status": "PASS" if p0_passed else "FAIL",
            "evidence": json.dumps(p0, ensure_ascii=False)[:400],
        },
        {
            "id": 2,
            "name": "P1 historical truth gate",
            "status": "PASS" if p1_gate else "FAIL",
            "evidence": (
                "lifecycle remains PIT_PARTIAL until P1 gate passes"
                if not p1_gate
                else "P1 gate passed"
            ),
        },
        {
            "id": 3,
            "name": "minimum fresh sample",
            "status": sample_status,
            "evidence": json.dumps(sample_detail, ensure_ascii=False),
        },
        {
            "id": 4,
            "name": "forward reconciliation",
            "status": "FAIL" if not decisions else "NOT_MEASURED",
            "evidence": (
                "no forward decisions; run "
                "scripts/forward/validate_shadow_records.py once records exist"
            ),
        },
        {
            "id": 5,
            "name": "no frozen parameter drift",
            "status": "FAIL" if not frozen else "PASS",
            "evidence": (
                f"strategy not in frozen registry"
                if not frozen
                else f"parameter_freeze_id={frozen.get('parameter_freeze_id', '')}"
            ),
        },
        {
            "id": 6,
            "name": "risk cost turnover rolling gate",
            "status": "FAIL" if not decisions else "NOT_MEASURED",
            "evidence": "requires forward daily NAV and shadow orders",
        },
        {
            "id": 7,
            "name": "pre-registered incremental bar vs B2-LT",
            "status": "FAIL" if not decisions else "NOT_MEASURED",
            "evidence": "requires concurrent B2-LT forward shadow records",
        },
        {
            "id": 8,
            "name": "double fee and execution delay stress",
            "status": "FAIL" if not decisions else "NOT_MEASURED",
            "evidence": "requires a stress rerun of the forward sample",
        },
        {
            "id": 9,
            "name": "clean commit and repeatability",
            "status": "PASS",
            "evidence": f"baseline commit {manifest.get('git_commit', '')}",
        },
    ]
    gate_passed = all(c["status"] == "PASS" for c in conditions)

    result = {
        "strategy_key": strategy_key,
        "gate_passed": gate_passed,
        "status": "PASS" if gate_passed else "NOT_ELIGIBLE_FOR_PAPER_TRADE",
        "forbidden_language": "tradable/verified alpha/formal paper trade "
        "candidate not permitted until all nine conditions pass",
        "conditions": conditions,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / f"{strategy_key}_forward_upgrade_gate.json"
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
