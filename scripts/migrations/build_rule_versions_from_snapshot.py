"""P1-1a: Build rule version table for frozen-strategy core products.

Each version row records the verified rule content with an explicit
effective_from/verified_at window.  Without historical fee evidence the
window starts at the snapshot verification date only, so historical orders
before that date remain UNCOVERED and the historical truth gate stays
NOT_ESTABLISHED (honest conservative semantics per planning.md P1-1).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES_CSV = ROOT / "config" / "otf_product_rules.csv"
REDEMPTION_TIERS_CSV = ROOT / "config" / "otf_redemption_fee_tiers.csv"
SUBSCRIPTION_TIERS_CSV = ROOT / "config" / "otf_subscription_fee_tiers.csv"
OUT_CSV = ROOT / "config" / "otf_rule_versions.csv"

CORE_CODES = [
    "000008", "000071", "000148", "000218", "001512", "006663", "007466",
    "016633", "021778", "050021", "050025", "160706", "260102",
]

VERSION_COLUMNS = [
    "fund_code",
    "rule_version_id",
    "effective_from",
    "effective_to",
    "channel",
    "source_type",
    "source_url",
    "verified_at",
    "rule_status",
    "subscription_confirmation_days",
    "redemption_confirmation_days",
    "redemption_settlement_days",
    "subscription_fee_rate",
    "redemption_fee_tiers_json",
    "subscription_fee_tiers_json",
    "minimum_holding_calendar_days",
    "maximum_subscription_amount_per_order",
    "subscription_open",
    "redemption_open",
    "evidence_note",
]


def load_rules() -> dict[str, dict[str, str]]:
    with RULES_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["fund_code"].strip().zfill(6): row for row in csv.DictReader(handle)}


def load_tiers(path: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row["fund_code"].strip().zfill(6)
            result.setdefault(code, []).append(row)
    return result


def main(verified_at: str) -> int:
    rules = load_rules()
    redemption_tiers = load_tiers(REDEMPTION_TIERS_CSV)
    subscription_tiers = load_tiers(SUBSCRIPTION_TIERS_CSV)

    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for code in CORE_CODES:
        rule = rules.get(code)
        if rule is None:
            missing.append(code)
            continue
        version_id = f"v{verified_at.replace('-', '')}"
        fee_tiers = redemption_tiers.get(code, [])
        sub_tiers = subscription_tiers.get(code, [])
        rows.append(
            {
                "fund_code": code,
                "rule_version_id": version_id,
                "effective_from": verified_at,
                "effective_to": "",
                "channel": rule.get("channel", "ALL") or "ALL",
                "source_type": "CURRENT_SNAPSHOT",
                "source_url": rule.get("source_url", ""),
                "verified_at": verified_at,
                "rule_status": rule.get("rule_status", "CONSERVATIVE_ASSUMPTION"),
                "subscription_confirmation_days": rule.get(
                    "subscription_confirmation_days", ""
                ),
                "redemption_confirmation_days": rule.get(
                    "redemption_confirmation_days", ""
                ),
                "redemption_settlement_days": rule.get("redemption_settlement_days", ""),
                "subscription_fee_rate": rule.get("subscription_fee_rate", ""),
                "redemption_fee_tiers_json": json.dumps(
                    fee_tiers, ensure_ascii=False, sort_keys=True
                ),
                "subscription_fee_tiers_json": json.dumps(
                    sub_tiers, ensure_ascii=False, sort_keys=True
                ),
                "minimum_holding_calendar_days": rule.get(
                    "minimum_holding_calendar_days", "0"
                ),
                "maximum_subscription_amount_per_order": rule.get(
                    "maximum_subscription_amount_per_order", ""
                ),
                "subscription_open": rule.get("subscription_open", "1"),
                "redemption_open": rule.get("redemption_open", "1"),
                "evidence_note": (
                    "CURRENT_SNAPSHOT_ONLY no historical fee evidence; "
                    "window starts at snapshot verification date"
                ),
            }
        )

    if missing:
        print(f"MISSING RULES: {missing}")
        return 1

    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VERSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} version rows to {OUT_CSV.name} (verified_at={verified_at})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "2026-08-01"))
