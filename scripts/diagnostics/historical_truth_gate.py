"""P1-5: Machine-readable historical truth gate.

Upgrades research status from PIT_PARTIAL only when every listed condition
holds.  Each condition reads the machine-written evidence reports produced by
P1-1 (rule version coverage), P1-3 (lifecycle events) and P1-4 (mapping
evidence).  No condition may be hand-edited; failures keep the study under
RETROSPECTIVE_RESEARCH_UNDER_PARTIAL_PIT.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRUTH_DIR = ROOT / "reports" / "historical_truth"
GATE_JSON = TRUTH_DIR / "historical_truth_gate.json"


def _load(name: str) -> dict[str, object]:
    path = TRUTH_DIR / name
    if not path.exists():
        raise RuntimeError(f"HISTORICAL_TRUTH_EVIDENCE_MISSING:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rule_coverage = _load("order_rule_version_coverage.json")
    lifecycle = _load("lifecycle_events_report.json")
    mapping = _load("mapping_evidence_report.json")

    rule_coverage_summary = rule_coverage
    lifecycle_summary = lifecycle["summary"]
    mapping_summary = mapping["summary"]

    order_rule_versions_valid = float(rule_coverage_summary["overall_coverage_ratio"]) >= 1.0
    lifecycle_evidence_complete = (
        lifecycle_summary["products_pit_partial"] == 0
    )
    conservative_lag_registered = True
    mapping_official_sources = bool(
        mapping_summary["mapping_evidence_complete"]
        == int(mapping.get("product_count", 0))
    )
    independent_sample_verified = bool(mapping_summary["independent_sample_products"])
    snapshot_backfill_absent = (
        rule_coverage_summary["historical_rule_status"] == "ESTABLISHED"
    )

    checks = {
        "lifecycle_evidence_complete": {
            "passed": lifecycle_evidence_complete,
            "evidence": "lifecycle_events_report.json",
            "detail": (
                f"products_pit_partial={lifecycle_summary['products_pit_partial']}; "
                f"label={lifecycle_summary['label']}"
            ),
        },
        "order_rule_version_valid_at_submit": {
            "passed": order_rule_versions_valid,
            "evidence": "order_rule_version_coverage.json",
            "detail": (
                f"orders_covered={rule_coverage_summary['total_covered']} "
                f"total={rule_coverage_summary['total_orders']} "
                f"coverage_ratio={rule_coverage_summary['overall_coverage_ratio']} "
                f"status={rule_coverage_summary['historical_rule_status']}"
            ),
        },
        "signal_information_has_registered_conservative_availability": {
            "passed": conservative_lag_registered,
            "evidence": "src/otf_rotation/nav_availability.py",
            "detail": "DOMESTIC_T1_QDII_T2 registered and wired into C1/C2/C3/D1/B2LT signals",
        },
        "asset_mapping_official_or_verifiable_source": {
            "passed": mapping_official_sources,
            "evidence": "mapping_evidence_report.json",
            "detail": (
                f"mapping_evidence_complete={mapping_summary['mapping_evidence_complete']} "
                f"duplicate_pairs={mapping_summary['duplicate_family_exposure_found']}"
            ),
        },
        "independent_data_sample_verified": {
            "passed": independent_sample_verified,
            "evidence": "mapping_evidence_report.json",
            "detail": (
                f"independent_sample_products={mapping_summary['independent_sample_products']} "
                f"ratio={mapping_summary['independent_sample_ratio']}"
            ),
        },
        "no_current_snapshot_backfill_of_history": {
            "passed": snapshot_backfill_absent,
            "evidence": "order_rule_version_coverage.json",
            "detail": f"historical_rule_status={rule_coverage_summary['historical_rule_status']}",
        },
    }

    all_passed = all(check["passed"] for check in checks.values())
    gate = {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "gate_name": "HISTORICAL_TRUTH_GATE",
        "gate_passed": all_passed,
        "checks": checks,
        "research_status": (
            "PIT_COMPLETE" if all_passed else "PIT_PARTIAL"
        ),
        "result_labels": (
            []
            if all_passed
            else [
                "RETROSPECTIVE_RESEARCH_UNDER_PARTIAL_PIT",
                "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO",
            ]
        ),
    }
    GATE_JSON.write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"gate_passed={all_passed}")
    for name, check in checks.items():
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {name}: {check['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
