"""Build a versioned catalog for the full publicly listed OTC fund universe."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd


RAW_CATALOG = Path("data/raw/otf_full_catalog/fund_catalog.csv")
CATALOG_DB = Path("data/processed/otf_full_catalog.sqlite")
SUMMARY = Path("reports/data_validation/otf_full_catalog_summary.json")

MONEY_TYPES = {"货币型-普通货币", "货币型-浮动净值"}
PASSIVE_EQUITY_TYPES = {"指数型-股票", "指数型-海外股票"}
PASSIVE_FIXED_INCOME_TYPES = {"指数型-固收", "QDII-纯债"}
BOND_DEFENSIVE_TYPES = {
    "债券型-中短债", "债券型-长债", "债券型-利率债", "债券型-信用债"
}


def split_generic_share_class(name: str) -> tuple[str, str]:
    cleaned = str(name).strip()
    match = re.search(r"(?:[-－/]?)(A|B|C|D|E|F|H|I|O|R|Y)(?:类)?$", cleaned, re.I)
    if not match:
        return cleaned, ""
    return cleaned[: match.start()].rstrip("-－/ "), match.group(1).upper()


def classify_research_scope(fund_type: str) -> str:
    if fund_type in MONEY_TYPES:
        return "money"
    if fund_type in PASSIVE_FIXED_INCOME_TYPES:
        return "passive_fixed_income"
    if fund_type in BOND_DEFENSIVE_TYPES:
        return "bond_defensive"
    if fund_type in PASSIVE_EQUITY_TYPES:
        return "passive_equity"
    return "other"


def normalize_catalog(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.iloc[:, :5].copy()
    frame.columns = [
        "fund_code", "pinyin_abbr", "fund_name", "fund_type", "pinyin_full"
    ]
    frame["fund_code"] = frame["fund_code"].astype(str).str.zfill(6)
    frame["fund_name"] = frame["fund_name"].astype(str).str.strip()
    frame["fund_type"] = frame["fund_type"].fillna("").astype(str).str.strip()
    family_and_class = frame["fund_name"].map(split_generic_share_class)
    frame["fund_family"] = family_and_class.map(lambda item: item[0])
    frame["share_class"] = family_and_class.map(lambda item: item[1])
    frame["research_scope"] = frame["fund_type"].map(classify_research_scope)
    frame["data_model"] = frame["research_scope"].map(
        lambda scope: "money_yield" if scope == "money" else "unit_nav"
    )
    frame["is_current_catalog"] = True
    frame["pit_status"] = "PIT_PARTIAL"
    frame["catalog_as_of"] = datetime.now().astimezone().date().isoformat()
    if frame["fund_code"].duplicated().any():
        raise ValueError("DUPLICATE_FUND_CODES_IN_PROVIDER_CATALOG")
    return frame


def build_catalog(
    raw_catalog: Path = RAW_CATALOG,
    catalog_db: Path = CATALOG_DB,
    summary_path: Path = SUMMARY,
) -> dict:
    catalog = normalize_catalog(ak.fund_name_em())
    raw_catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog_db.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(raw_catalog, index=False, encoding="utf-8-sig")
    temporary = catalog_db.with_suffix(catalog_db.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        catalog.to_sql("otf_full_catalog", connection, index=False, if_exists="replace")
        connection.execute(
            "CREATE UNIQUE INDEX idx_full_catalog_code ON otf_full_catalog(fund_code)"
        )
        connection.execute(
            "CREATE INDEX idx_full_catalog_scope ON otf_full_catalog(research_scope)"
        )
        connection.commit()
    finally:
        connection.close()
    temporary.replace(catalog_db)

    type_counts = catalog["fund_type"].value_counts().to_dict()
    scope_counts = catalog["research_scope"].value_counts().to_dict()
    summary = {
        "catalog_as_of": catalog["catalog_as_of"].iloc[0],
        "fund_share_count": int(len(catalog)),
        "fund_family_count": int(catalog["fund_family"].nunique()),
        "type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "scope_counts": {str(k): int(v) for k, v in scope_counts.items()},
        "money_data_model": "per_10000_income_and_7d_annualized",
        "nav_data_model": "unit_nav_total_return",
        "pit_status": "PIT_PARTIAL",
        "historical_inactive_coverage": False,
        "source_independent": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-catalog", type=Path, default=RAW_CATALOG)
    parser.add_argument("--catalog-db", type=Path, default=CATALOG_DB)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    print(
        json.dumps(
            build_catalog(args.raw_catalog, args.catalog_db, args.summary),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
