"""Auditable China execution calendar for OTC fund simulations.

The calendar is intentionally not a weekday calendar and is not derived from
the union of OTC NAV dates.  Historical dates come from the distinct dates in
``etf.sqlite.etf_daily``.  Only the post-ETF-source tail is extended, and then
only when all four domestic equity reference funds publish the same NAV date.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_PRIMARY_DB = Path("data/processed/etf.sqlite")
DEFAULT_OTF_DB = Path("data/processed/otf_expanded.sqlite")
DEFAULT_CALENDAR_PATH = Path("data/processed/execution_calendar/cn_execution_calendar.csv")
DEFAULT_EXTENSION_CODES = ("160706", "000008", "050021", "007466")
EXPECTED_PRIMARY_MAX_DATE = pd.Timestamp("2026-07-17")
EXPECTED_EXTENSION_START = pd.Timestamp("2026-07-20")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExecutionCalendar:
    dates: pd.DatetimeIndex
    path: str
    source: str
    consensus_count: int
    primary_source_max_date: str
    input_hashes: dict[str, str]
    extension_codes: tuple[str, ...]

    @property
    def min_date(self) -> str:
        return self.dates.min().strftime("%Y-%m-%d")

    @property
    def max_date(self) -> str:
        return self.dates.max().strftime("%Y-%m-%d")

    def facts(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": sha256_file(self.path),
            "source": self.source,
            "consensus_count": self.consensus_count,
            "primary_source_max_date": self.primary_source_max_date,
            "coverage": {"min_date": self.min_date, "max_date": self.max_date, "count": len(self.dates)},
            "extension_codes": list(self.extension_codes),
            "input_hashes": dict(self.input_hashes),
        }


def _distinct_dates(db_path: str | Path, query: str, params: tuple[Any, ...] = ()) -> pd.DatetimeIndex:
    with sqlite3.connect(str(db_path)) as connection:
        frame = pd.read_sql_query(query, connection, params=params)
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce").dropna()
    return pd.DatetimeIndex(sorted(dates.unique()))


def _consensus_dates(
    otf_db_path: str | Path,
    extension_codes: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    codes = tuple(str(code).zfill(6) for code in extension_codes)
    placeholders = ",".join("?" for _ in codes)
    query = f"""
        SELECT nav_date, fund_code
        FROM otf_fund_nav
        WHERE fund_code IN ({placeholders})
          AND nav_date >= ? AND nav_date <= ?
        GROUP BY nav_date, fund_code
        ORDER BY nav_date, fund_code
    """
    with sqlite3.connect(str(otf_db_path)) as connection:
        frame = pd.read_sql_query(query, connection, params=(*codes, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    if frame.empty:
        audit = pd.DataFrame(columns=["execution_date", "consensus_count", "reference_codes", "source"])
        return audit, pd.DatetimeIndex([])
    frame["nav_date"] = pd.to_datetime(frame["nav_date"], errors="coerce")
    grouped = frame.groupby("nav_date")["fund_code"].agg(lambda values: sorted({str(v).zfill(6) for v in values})).reset_index()
    grouped["consensus_count"] = grouped["fund_code"].map(len)
    grouped["reference_codes"] = grouped["fund_code"].map(lambda values: ";".join(values))
    grouped["source"] = "OTF_DOMESTIC_EQUITY_CONSENSUS_4_OF_4"
    accepted = grouped[grouped["consensus_count"] == len(codes)].copy()
    return grouped.drop(columns=["fund_code"]), pd.DatetimeIndex(sorted(accepted["nav_date"].unique()))


def build_cn_execution_calendar(
    primary_db_path: str | Path = DEFAULT_PRIMARY_DB,
    otf_db_path: str | Path = DEFAULT_OTF_DB,
    output_path: str | Path = DEFAULT_CALENDAR_PATH,
    extension_codes: Iterable[str] = DEFAULT_EXTENSION_CODES,
    extension_end: str | pd.Timestamp | None = None,
) -> ExecutionCalendar:
    primary_db_path = Path(primary_db_path)
    otf_db_path = Path(otf_db_path)
    output_path = Path(output_path)
    primary_dates = _distinct_dates(primary_db_path, "SELECT DISTINCT date FROM etf_daily")
    if primary_dates.empty:
        raise RuntimeError("CN_EXECUTION_CALENDAR_PRIMARY_EMPTY")
    primary_max = pd.Timestamp(primary_dates.max())
    if primary_max != EXPECTED_PRIMARY_MAX_DATE:
        raise RuntimeError(f"CN_EXECUTION_CALENDAR_PRIMARY_MAX_UNEXPECTED:{primary_max.date()}")
    ext_end = pd.Timestamp(extension_end) if extension_end is not None else pd.Timestamp("2026-07-27")
    audit, consensus_dates = _consensus_dates(otf_db_path, extension_codes, EXPECTED_EXTENSION_START, ext_end)
    extension_codes = tuple(str(code).zfill(6) for code in extension_codes)
    if consensus_dates.empty:
        raise RuntimeError("CN_EXECUTION_CALENDAR_NO_4_OF_4_EXTENSION_DATES")
    dates = pd.DatetimeIndex(sorted(set(primary_dates) | set(consensus_dates)))
    rows: list[dict[str, Any]] = []
    primary_set = set(primary_dates)
    consensus_set = set(consensus_dates)
    for date in dates:
        if date in primary_set:
            source = "ETF_DAILY_DISTINCT_DATE"
            count = pd.NA
        elif date in consensus_set:
            source = "OTF_DOMESTIC_EQUITY_CONSENSUS_4_OF_4"
            count = len(extension_codes)
        else:
            raise RuntimeError(f"CN_EXECUTION_CALENDAR_UNCLASSIFIED_DATE:{date.date()}")
        rows.append({
            "execution_date": date.strftime("%Y-%m-%d"),
            "source": source,
            "consensus_count": count,
            "reference_codes": ";".join(extension_codes) if source.endswith("4_OF_4") else "",
            "primary_source_max_date": primary_max.strftime("%Y-%m-%d"),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calendar_frame = pd.DataFrame(rows)
    calendar_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    hashes = {
        "primary_db_sha256": sha256_file(primary_db_path),
        "otf_db_sha256": sha256_file(otf_db_path),
    }
    audit_payload = {
        "schema_version": "cn_execution_calendar_v1",
        "calendar_path": str(output_path),
        "source": "ETF_DAILY_DISTINCT_DATE_PLUS_4_OF_4_DOMESTIC_EQUITY_NAV_CONSENSUS",
        "consensus_count": len(extension_codes),
        "primary_source_max_date": primary_max.strftime("%Y-%m-%d"),
        "extension_start": EXPECTED_EXTENSION_START.strftime("%Y-%m-%d"),
        "extension_end": ext_end.strftime("%Y-%m-%d"),
        "extension_codes": list(extension_codes),
        "accepted_extension_dates": [d.strftime("%Y-%m-%d") for d in consensus_dates],
        "input_hashes": hashes,
        "calendar_sha256": sha256_file(output_path),
        "coverage": {"min_date": dates.min().strftime("%Y-%m-%d"), "max_date": dates.max().strftime("%Y-%m-%d"), "count": len(dates)},
    }
    (output_path.with_suffix(".audit.json")).write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return ExecutionCalendar(
        dates=dates,
        path=str(output_path),
        source=audit_payload["source"],
        consensus_count=len(extension_codes),
        primary_source_max_date=primary_max.strftime("%Y-%m-%d"),
        input_hashes=hashes,
        extension_codes=extension_codes,
    )


def load_execution_calendar(path: str | Path) -> ExecutionCalendar:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CN_EXECUTION_CALENDAR_MISSING:{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"execution_date", "source", "consensus_count", "primary_source_max_date"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"CN_EXECUTION_CALENDAR_SCHEMA_MISSING:{sorted(missing)}")
    dates = pd.to_datetime(frame["execution_date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise RuntimeError("CN_EXECUTION_CALENDAR_DATES_INVALID")
    if not (frame["source"].astype(str).str.len() > 0).all():
        raise RuntimeError("CN_EXECUTION_CALENDAR_SOURCE_MISSING")
    audit_path = path.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    if audit.get("calendar_sha256") and audit["calendar_sha256"] != sha256_file(path):
        raise RuntimeError("CN_EXECUTION_CALENDAR_HASH_MISMATCH")
    return ExecutionCalendar(
        dates=pd.DatetimeIndex(dates),
        path=str(path),
        source=str(audit.get("source", "UNKNOWN_CALENDAR_SOURCE")),
        consensus_count=int(audit.get("consensus_count", 0)),
        primary_source_max_date=str(audit.get("primary_source_max_date", frame["primary_source_max_date"].iloc[0])),
        input_hashes=dict(audit.get("input_hashes", {})),
        extension_codes=tuple(audit.get("extension_codes", [])),
    )


def calendar_facts(calendar: ExecutionCalendar) -> dict[str, Any]:
    return calendar.facts()

