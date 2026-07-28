"""Discover ETF feeder funds and map them to listed ETF signal proxies."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from difflib import SequenceMatcher
from pathlib import Path

import akshare as ak
import pandas as pd


DEFAULT_DB = Path("data/processed/etf.sqlite")
DEFAULT_CATALOG = Path("data/raw/otf_catalog/etf_feeder_catalog.csv")
DEFAULT_MAPPING = Path("config/etf_otf_mapping.csv")
DEFAULT_SUMMARY = Path("reports/data_validation/etf_otf_mapping_summary.json")

SHARE_CLASS_PRIORITY = {
    "A": 0,
    "R": 1,
    "I": 2,
    "E": 3,
    "C": 4,
    "Y": 5,
    "D": 6,
    "F": 7,
    "": 8,
}

REMOVE_TOKENS = (
    "交易型开放式指数证券投资基金联接基金",
    "交易型开放式指数证券投资基金",
    "ETF联接基金",
    "ETF联接",
    "联接基金",
    "发起式",
    "人民币",
    "QDII",
    "LOF",
)


def normalize_text(value: str) -> str:
    text = str(value).upper()
    for token in REMOVE_TOKENS:
        text = text.replace(token.upper(), "")
    return re.sub(r"[^0-9A-Z\u4e00-\u9fff]+", "", text)


def split_share_class(name: str) -> tuple[str, str]:
    """Return a stable fund-family name and the terminal share class."""
    cleaned = str(name)
    cleaned = re.sub(r"[（(](?:QDII|LOF|场外)[^）)]*[）)]", "", cleaned, flags=re.I)
    cleaned = re.sub(
        r"[（(](?:人民币|美元[^）)]*|港币[^）)]*)[）)]$", "", cleaned, flags=re.I
    )
    cleaned = re.sub(r"(?:人民币|美元现汇|美元现钞|港币)$", "", cleaned, flags=re.I)
    match = re.search(r"(?:[-－/]?)(A|C|E|I|R|Y|D|F)(?:类)?$", cleaned, flags=re.I)
    if not match:
        return cleaned.strip(), ""
    share_class = match.group(1).upper()
    family = cleaned[: match.start()].rstrip("-－/ ")
    return family, share_class


def discover_etf_feeders() -> pd.DataFrame:
    names = ak.fund_name_em().copy()
    names = names.iloc[:, :5]
    names.columns = [
        "fund_code", "pinyin_abbr", "fund_name", "fund_type", "pinyin_full"
    ]
    names["fund_code"] = names["fund_code"].astype(str).str.zfill(6)
    names["fund_name"] = names["fund_name"].astype(str).str.strip()
    feeders = names.loc[
        names["fund_name"].str.contains("ETF联接", case=False, na=False)
        & ~names["fund_name"].str.contains("后端", na=False)
    ].copy()
    family_and_class = feeders["fund_name"].map(split_share_class)
    feeders["fund_family"] = family_and_class.map(lambda item: item[0])
    feeders["share_class"] = family_and_class.map(lambda item: item[1])
    feeders["share_class_priority"] = feeders["share_class"].map(
        SHARE_CLASS_PRIORITY
    ).fillna(99)
    feeders["foreign_currency_share"] = feeders["fund_name"].str.contains(
        "美元|港币", regex=True, na=False
    )
    feeders.loc[feeders["foreign_currency_share"], "share_class_priority"] += 50
    feeders = feeders.sort_values(
        ["fund_family", "share_class_priority", "fund_code"]
    )
    feeders["selected_share"] = ~feeders.duplicated("fund_family", keep="first")
    return feeders.reset_index(drop=True)


def load_etf_catalog(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        etfs = pd.read_sql_query(
            """
            SELECT c.symbol, c.name, q.asset_class, q.rows_valid,
                   q.median_amount_60d
            FROM etf_catalog c
            LEFT JOIN etf_quality q ON q.symbol = c.symbol
            """,
            conn,
        )
    etfs["symbol"] = etfs["symbol"].astype(str).str.zfill(6)
    etfs["name"] = etfs["name"].astype(str)
    split = etfs["name"].str.split("ETF", n=1, expand=True)
    etfs["underlying_name"] = split[0].fillna("")
    etfs["issuer_hint"] = split[1].fillna("") if split.shape[1] > 1 else ""
    etfs["underlying_norm"] = etfs["underlying_name"].map(normalize_text)
    etfs["issuer_norm"] = etfs["issuer_hint"].map(normalize_text)
    etfs["name_norm"] = etfs["name"].map(normalize_text)
    etfs["rows_valid"] = pd.to_numeric(etfs["rows_valid"], errors="coerce").fillna(0)
    etfs["median_amount_60d"] = pd.to_numeric(
        etfs["median_amount_60d"], errors="coerce"
    ).fillna(0.0)
    return etfs


def score_mapping(fund_family: str, etf: pd.Series) -> float:
    fund_norm = normalize_text(fund_family)
    underlying = str(etf["underlying_norm"])
    issuer = str(etf["issuer_norm"])
    if not underlying:
        return 0.0
    underlying_exact = underlying in fund_norm
    issuer_exact = bool(issuer) and issuer in fund_norm
    if underlying_exact and issuer_exact:
        return 100.0
    if underlying_exact:
        return 88.0
    sequence = SequenceMatcher(None, fund_norm, str(etf["name_norm"])).ratio()
    partial = SequenceMatcher(None, fund_norm, underlying).ratio()
    issuer_bonus = 8.0 if issuer_exact else 0.0
    return min(87.0, 55.0 * sequence + 35.0 * partial + issuer_bonus)


def build_mapping(feeders: pd.DataFrame, etfs: pd.DataFrame) -> pd.DataFrame:
    selected = feeders.loc[feeders["selected_share"]].copy()
    rows: list[dict] = []
    for fund in selected.itertuples(index=False):
        scored = etfs.copy()
        scored["mapping_score"] = scored.apply(
            lambda row: score_mapping(fund.fund_family, row), axis=1
        )
        scored = scored.sort_values(
            ["mapping_score", "rows_valid", "median_amount_60d"],
            ascending=[False, False, False],
        )
        best = scored.iloc[0]
        score = float(best["mapping_score"])
        confidence = "HIGH" if score >= 99 else "MEDIUM" if score >= 80 else "LOW"
        rows.append(
            {
                "fund_code": fund.fund_code,
                "fund_name": fund.fund_name,
                "fund_family": fund.fund_family,
                "share_class": fund.share_class,
                "fund_type": fund.fund_type,
                "etf_symbol": best["symbol"],
                "etf_name": best["name"],
                "underlying_name": best["underlying_name"],
                "asset_class": best["asset_class"],
                "mapping_score": round(score, 3),
                "mapping_confidence": confidence,
                "mapping_method": "name_underlying_issuer_v1",
                "etf_rows_valid": int(best["rows_valid"]),
                "etf_median_amount_60d": float(best["median_amount_60d"]),
                "executable": confidence in {"HIGH", "MEDIUM"},
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mapping_confidence", "mapping_score", "fund_code"],
        ascending=[True, False, True],
    )


def run(
    db_path: Path = DEFAULT_DB,
    catalog_path: Path = DEFAULT_CATALOG,
    mapping_path: Path = DEFAULT_MAPPING,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict:
    feeders = discover_etf_feeders()
    etfs = load_etf_catalog(db_path)
    mapping = build_mapping(feeders, etfs)

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    feeders.to_csv(catalog_path, index=False, encoding="utf-8-sig")
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")

    counts = mapping["mapping_confidence"].value_counts().to_dict()
    summary = {
        "catalog_share_count": int(len(feeders)),
        "fund_family_count": int(feeders["fund_family"].nunique()),
        "selected_share_count": int(feeders["selected_share"].sum()),
        "mapped_count": int(len(mapping)),
        "high_confidence": int(counts.get("HIGH", 0)),
        "medium_confidence": int(counts.get("MEDIUM", 0)),
        "low_confidence": int(counts.get("LOW", 0)),
        "executable_count": int(mapping["executable"].sum()),
        "unique_etf_signals": int(
            mapping.loc[mapping["executable"], "etf_symbol"].nunique()
        ),
        "mapping_method": "name_underlying_issuer_v1",
        "requires_manual_review": True,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    print(json.dumps(run(args.db, args.catalog, args.mapping, args.summary), indent=2))


if __name__ == "__main__":
    main()
