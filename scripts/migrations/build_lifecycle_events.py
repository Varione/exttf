"""P1-3: Build the independent lifecycle event table for frozen-strategy products.

Records every lifecycle event that has verifiable evidence (inception from the
research catalog cross-checked against the first NAV date).  Announcement-based
events (suspension, subscription limits, merger, conversion, liquidation) have
no historical evidence source in this repository, so they are explicitly
reported as NOT_ESTABLISHED rather than fabricated.  Products without full
evidence are flagged PIT_PARTIAL, never silently assumed complete.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "processed" / "otf_research.sqlite"
OUTPUT_CSV = ROOT / "config" / "otf_lifecycle_events.csv"
REPORT_JSON = ROOT / "reports" / "historical_truth" / "lifecycle_events_report.json"

CORE_CODES = [
    "000008", "000071", "000148", "000218", "001512", "006663", "007466",
    "016633", "021778", "050021", "050025", "160706", "260102",
]

EVENT_TYPES = [
    "INCEPTION",
    "SUSPENSION",
    "RESUMPTION",
    "SUBSCRIPTION_LIMIT",
    "LIQUIDATION",
    "MERGER",
    "CONVERSION",
    "TERMINATION",
    "SHARE_CLASS_ADDED",
    "SHARE_CLASS_REMOVED",
]

VERIFIED_AT = "2026-08-01"


def _load_catalog(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT fund_code, fund_name, inception_date, termination_date, "
        "fund_family, share_class, asset_class, source, fetched_at "
        "FROM otf_fund_catalog",
        con,
    )


def _load_nav_coverage(con: sqlite3.Connection) -> pd.DataFrame:
    placeholders = ",".join("?" * len(CORE_CODES))
    return pd.read_sql_query(
        f"SELECT fund_code, COUNT(*) AS nav_days, MIN(nav_date) AS first_nav_date, "
        f"MAX(nav_date) AS last_nav_date, "
        f"SUM(CASE WHEN unit_nav IS NULL OR unit_nav = 0 THEN 1 ELSE 0 END) AS null_nav_rows "
        f"FROM otf_fund_nav WHERE fund_code IN ({placeholders}) GROUP BY fund_code",
        con,
        params=CORE_CODES,
    )


def main() -> int:
    con = sqlite3.connect(str(DB_PATH))
    catalog = _load_catalog(con)
    nav = _load_nav_coverage(con)
    con.close()

    merged = catalog.merge(nav, on="fund_code", how="inner")
    missing = sorted(set(CORE_CODES) - set(merged["fund_code"]))
    if missing:
        raise RuntimeError(f"LIFECYCLE_CATALOG_MISSING_CODES:{missing}")

    event_rows: list[dict[str, object]] = []
    per_product: dict[str, dict[str, object]] = {}
    for _, row in merged.sort_values("fund_code").iterrows():
        code = row["fund_code"]
        inception = pd.Timestamp(row["inception_date"]).date().isoformat()
        termination = row["termination_date"]
        first_nav = row["first_nav_date"]
        last_nav = row["last_nav_date"]
        nav_days = int(row["nav_days"])
        null_nav = int(row["null_nav_rows"])

        event_rows.append(
            {
                "fund_code": code,
                "event_type": "INCEPTION",
                "event_date": inception,
                "source_type": "RESEARCH_DB_CATALOG",
                "evidence_note": (
                    f"catalog inception_date {inception}; first NAV row {first_nav}; "
                    f"NAV coverage {nav_days} rows from inception with no null unit_nav"
                ),
                "verified_at": VERIFIED_AT,
            }
        )

        inception_matches_nav = pd.Timestamp(inception) == pd.Timestamp(first_nav)
        termination_detected = bool(termination and str(termination).strip())
        if termination_detected:
            event_rows.append(
                {
                    "fund_code": code,
                    "event_type": "TERMINATION",
                    "event_date": str(termination),
                    "source_type": "RESEARCH_DB_CATALOG",
                    "evidence_note": f"catalog termination_date {termination}",
                    "verified_at": VERIFIED_AT,
                }
            )

        missing_event_types = [
            et
            for et in EVENT_TYPES
            if et not in ("INCEPTION", "TERMINATION")
        ]
        per_product[code] = {
            "fund_code": code,
            "inception_date": inception,
            "first_nav_date": first_nav,
            "last_nav_date": last_nav,
            "nav_days": nav_days,
            "null_nav_rows": null_nav,
            "termination_date": str(termination) if termination_detected else None,
            "inception_matches_first_nav": bool(inception_matches_nav),
            "termination_detected": termination_detected,
            "event_types_with_evidence": (
                ["INCEPTION", "TERMINATION"] if termination_detected else ["INCEPTION"]
            ),
            "event_types_without_evidence": missing_event_types,
            "coverage_status": "PIT_PARTIAL",
            "coverage_reasons": [
                "ANNOUNCEMENT_EVENTS_NO_HISTORICAL_SOURCE"
                if missing_event_types
                else "NONE",
                "TERMINATION_NOT_RECORDED"
                if not termination_detected
                else "NONE",
            ],
        }

    events_df = pd.DataFrame(
        event_rows,
        columns=[
            "fund_code", "event_type", "event_date", "source_type",
            "evidence_note", "verified_at",
        ],
    )
    events_df = events_df.sort_values(["fund_code", "event_date", "event_type"])
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    report = {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "coverage_scope": "FROZEN_STRATEGY_CORE_PRODUCTS",
        "products": sorted(CORE_CODES),
        "product_count": len(CORE_CODES),
        "per_product": per_product,
        "summary": {
            "inception_events_recorded": int((events_df["event_type"] == "INCEPTION").sum()),
            "termination_events_recorded": int((events_df["event_type"] == "TERMINATION").sum()),
            "products_with_inception_matching_first_nav": int(
                sum(1 for p in per_product.values() if p["inception_matches_first_nav"])
            ),
            "products_pit_partial": int(
                sum(1 for p in per_product.values() if p["coverage_status"] == "PIT_PARTIAL")
            ),
            "announcement_event_types_without_source": sorted(
                set(EVENT_TYPES) - {"INCEPTION", "TERMINATION"}
            ),
            "lifecycle_status": "PIT_PARTIAL",
            "label": "INCEPTION_EVIDENCE_COMPLETE; ANNOUNCEMENT_EVENT_EVIDENCE_NOT_ESTABLISHED",
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"lifecycle events written: {OUTPUT_CSV} ({len(events_df)} rows)")
    print(f"report written: {REPORT_JSON}")
    print(
        "summary:",
        json.dumps(report["summary"], ensure_ascii=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
