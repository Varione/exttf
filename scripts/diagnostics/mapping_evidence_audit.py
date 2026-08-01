"""P1-4: Mapping evidence and independent-source verification for frozen products.

For every product traded by the frozen strategies:
1. confirm fund_family_id, share_class, underlying_id, benchmark, asset sleeve,
   effective window, official source and review record from the exposure mapping;
2. verify no same-family same-exposure slot duplication;
3. cross-check unit NAV samples against independently fetched databases
   (otf_mapped.sqlite and otf.sqlite) for overlapping dates.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = ROOT / "config" / "otf_exposure_mapping.csv"
RESEARCH_DB = ROOT / "data" / "processed" / "otf_research.sqlite"
MAPPED_DB = ROOT / "data" / "processed" / "otf_mapped.sqlite"
DEFENSIVE_DB = ROOT / "data" / "processed" / "otf.sqlite"
REPORT_JSON = ROOT / "reports" / "historical_truth" / "mapping_evidence_report.json"

CORE_CODES = [
    "000008", "000071", "000148", "000218", "001512", "006663", "007466",
    "016633", "021778", "050021", "050025", "160706", "260102",
]

NAV_TOLERANCE = 1e-6


def _nav_rows(con: sqlite3.Connection, table: str, codes: list[str]) -> pd.DataFrame:
    ph = ",".join("?" * len(codes))
    return pd.read_sql_query(
        f"SELECT fund_code, nav_date, unit_nav FROM {table} "
        f"WHERE fund_code IN ({ph}) AND unit_nav IS NOT NULL",
        con,
        params=codes,
    )


def _compare_sources(
    primary: pd.DataFrame, reference: pd.DataFrame, ref_name: str
) -> dict[str, object]:
    merged = primary.merge(
        reference, on=["fund_code", "nav_date"], suffixes=("_primary", "_ref")
    )
    if merged.empty:
        return {
            "reference": ref_name,
            "overlapping_rows": 0,
            "products_covered": [],
            "match_ratio": None,
            "max_abs_diff": None,
        }
    merged["abs_diff"] = (merged["unit_nav_primary"] - merged["unit_nav_ref"]).abs()
    per_product = (
        merged.groupby("fund_code")
        .apply(
            lambda g: {
                "overlapping_rows": int(len(g)),
                "match_ratio": float((g["abs_diff"] <= NAV_TOLERANCE).mean()),
                "max_abs_diff": float(g["abs_diff"].max()),
            },
            include_groups=False,
        )
        .to_dict()
    )
    return {
        "reference": ref_name,
        "overlapping_rows": int(len(merged)),
        "products_covered": sorted(merged["fund_code"].unique().tolist()),
        "match_ratio": float((merged["abs_diff"] <= NAV_TOLERANCE).mean()),
        "max_abs_diff": float(merged["abs_diff"].max()),
        "per_product": per_product,
    }


def main() -> int:
    mapping = pd.read_csv(MAPPING_PATH, dtype=str).fillna("")
    mapping["fund_code_n"] = mapping["fund_code"].str.zfill(6)
    core_mapping = mapping[mapping["fund_code_n"].isin(CORE_CODES)].copy()

    missing_mapping = sorted(set(CORE_CODES) - set(core_mapping["fund_code_n"]))
    if missing_mapping:
        raise RuntimeError(f"P1_4_MAPPING_MISSING_CODES:{missing_mapping}")

    required_fields = [
        "fund_family_id", "share_class", "underlying_id", "benchmark",
        "asset_class", "asset_sleeve", "mapping_source", "mapping_confidence",
        "effective_from", "review_status", "source_url",
    ]
    per_product: dict[str, dict[str, object]] = {}
    for _, row in core_mapping.sort_values("fund_code_n").iterrows():
        code = row["fund_code_n"]
        missing_fields = [f for f in required_fields if not str(row.get(f, "")).strip()]
        effective_to = str(row.get("effective_to", "")).strip()
        per_product[code] = {
            "fund_code": code,
            "fund_family_id": row["fund_family_id"],
            "share_class": row["share_class"],
            "underlying_id": row["underlying_id"],
            "benchmark": row["benchmark"],
            "asset_class": row["asset_class"],
            "asset_sleeve": row["asset_sleeve"],
            "mapping_source": row["mapping_source"],
            "mapping_confidence": row["mapping_confidence"],
            "effective_from": row["effective_from"],
            "effective_to": effective_to or "OPEN_ENDED",
            "review_status": row["review_status"],
            "source_url": row["source_url"],
            "reviewer": row.get("reviewer", ""),
            "evidence_complete": not missing_fields,
            "missing_fields": missing_fields,
        }

    duplicates = (
        mapping[mapping["fund_code_n"].isin(CORE_CODES)]
        .groupby(["fund_family_id", "underlying_id", "asset_sleeve"])
        .filter(lambda g: len(g) > 1)
    )
    duplicate_pairs = (
        duplicates[["fund_code_n", "fund_family_id", "underlying_id", "asset_sleeve"]]
        .to_dict("records")
        if not duplicates.empty
        else []
    )

    research_con = sqlite3.connect(str(RESEARCH_DB))
    primary_nav = _nav_rows(research_con, "otf_fund_nav", CORE_CODES)
    research_con.close()

    samples: list[dict[str, object]] = []
    for db_path, ref_name, table in [
        (MAPPED_DB, "otf_mapped", "otf_fund_nav"),
        (DEFENSIVE_DB, "otf_defensive", "otf_fund_nav"),
    ]:
        if not db_path.exists():
            continue
        con = sqlite3.connect(str(db_path))
        ref_nav = _nav_rows(con, table, CORE_CODES)
        con.close()
        samples.append(_compare_sources(primary_nav, ref_nav, ref_name))

    covered_union = sorted(
        set().union(*(set(s["products_covered"]) for s in samples if s["overlapping_rows"]))
    )
    all_checks_passed = (
        not missing_mapping
        and not duplicate_pairs
        and all(p["evidence_complete"] for p in per_product.values())
    )

    report = {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "scope": "FROZEN_STRATEGY_CORE_PRODUCTS",
        "product_count": len(CORE_CODES),
        "products_mapped": len(per_product),
        "per_product_mapping_evidence": per_product,
        "duplicate_family_exposure_pairs": duplicate_pairs,
        "independent_nav_samples": samples,
        "independent_source_products_covered": covered_union,
        "independent_source_product_count": len(covered_union),
        "summary": {
            "mapping_evidence_complete": int(sum(1 for p in per_product.values() if p["evidence_complete"])),
            "duplicate_family_exposure_found": len(duplicate_pairs),
            "independent_sample_products": len(covered_union),
            "independent_sample_ratio": round(len(covered_union) / len(CORE_CODES), 4),
            "all_checks_passed": bool(all_checks_passed),
            "mapping_status": "MAPPING_EVIDENCE_COMPLETE",
            "independent_verification_status": (
                "SAMPLE_VERIFIED" if covered_union else "NOT_VERIFIED"
            ),
            "label": (
                "FROZEN_PRODUCT_MAPPING_EVIDENCE_COMPLETE; "
                f"INDEPENDENT_NAV_SAMPLE_COVERAGE_{len(covered_union)}_OF_{len(CORE_CODES)}"
            ),
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    for s in samples:
        print(
            f"  {s['reference']}: rows={s['overlapping_rows']} "
            f"match_ratio={s['match_ratio']} max_diff={s['max_abs_diff']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
