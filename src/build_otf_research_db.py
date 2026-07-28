"""Build a non-destructive OTC research database with defensive funds."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


MAPPED_DB = Path("data/processed/otf_mapped.sqlite")
DEFENSIVE_DB = Path("data/processed/otf.sqlite")
OUTPUT_DB = Path("data/processed/otf_research.sqlite")
EXTENDED_DB = Path("data/processed/otf_extended_assets.sqlite")
DEFAULT_DEFENSIVE_CODES = ("006663",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_research_db(
    mapped_db: Path = MAPPED_DB,
    defensive_db: Path = DEFENSIVE_DB,
    output_db: Path = OUTPUT_DB,
    defensive_codes: tuple[str, ...] = DEFAULT_DEFENSIVE_CODES,
    extended_db: Path | None = None,
) -> dict:
    """Copy mapped data and append explicitly approved direct OTC funds."""
    for source in (mapped_db, defensive_db):
        if not source.exists():
            raise FileNotFoundError(source)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_db.with_suffix(output_db.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(mapped_db, temporary)

    inserted: list[dict] = []
    target: sqlite3.Connection | None = None
    source: sqlite3.Connection | None = None
    try:
        with sqlite3.connect(temporary) as target, sqlite3.connect(
            defensive_db
        ) as source:
            for code in defensive_codes:
                catalog = source.execute(
                    """
                    SELECT fund_code,share_class,fund_name,asset_class,benchmark,
                           inception_date,termination_date,source,fetched_at
                    FROM otf_fund_catalog WHERE fund_code=?
                    """,
                    (code,),
                ).fetchone()
                if catalog is None:
                    raise ValueError(f"DEFENSIVE_FUND_NOT_FOUND:{code}")
                nav_rows = source.execute(
                    """
                    SELECT fund_code,nav_date,unit_nav,cumulative_nav,
                           daily_growth_pct,cumulative_nav_imputed,
                           distribution_per_share,share_adjustment_factor,
                           total_return_factor
                    FROM otf_fund_nav WHERE fund_code=? ORDER BY nav_date
                    """,
                    (code,),
                ).fetchall()
                if not nav_rows:
                    raise ValueError(f"DEFENSIVE_NAV_NOT_FOUND:{code}")
                (
                    fund_code, share_class, fund_name, asset_class, benchmark,
                    inception_date, termination_date, source_name, fetched_at,
                ) = catalog
                target.execute(
                    """
                    INSERT OR REPLACE INTO otf_fund_catalog (
                        fund_code,fund_name,fund_family,share_class,fund_type,
                        etf_symbol,etf_name,underlying_name,asset_class,
                        mapping_score,mapping_confidence,mapping_method,
                        inception_date,termination_date,source,fetched_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        fund_code, fund_name, fund_name, share_class,
                        "direct_otf_defensive", "", "", benchmark,
                        "bond_short", None, "DIRECT", "approved_direct_otf_v1",
                        inception_date, termination_date, source_name, fetched_at,
                    ),
                )
                target.execute(
                    "DELETE FROM otf_fund_nav WHERE fund_code=?", (code,)
                )
                target.executemany(
                    "INSERT INTO otf_fund_nav VALUES (?,?,?,?,?,?,?,?,?)",
                    nav_rows,
                )
                inserted.append(
                    {
                        "fund_code": code,
                        "fund_name": fund_name,
                        "source_asset_class": asset_class,
                        "research_asset_class": "bond_short",
                        "nav_rows": len(nav_rows),
                        "first_nav": nav_rows[0][1],
                        "last_nav": nav_rows[-1][1],
                    }
                )
            if extended_db is not None:
                if not extended_db.exists():
                    raise FileNotFoundError(extended_db)
                with sqlite3.connect(extended_db) as extended:
                    extended_catalog = extended.execute(
                        """
                        SELECT fund_code,fund_name,fund_family,share_class,
                               fund_type,etf_symbol,etf_name,underlying_name,
                               asset_class,mapping_score,mapping_confidence,
                               mapping_method,inception_date,termination_date,
                               source,fetched_at
                        FROM otf_fund_catalog
                        """
                    ).fetchall()
                    extended_nav = extended.execute(
                        """
                        SELECT fund_code,nav_date,unit_nav,cumulative_nav,
                               daily_growth_pct,cumulative_nav_imputed,
                               distribution_per_share,share_adjustment_factor,
                               total_return_factor FROM otf_fund_nav
                        """
                    ).fetchall()
                extended_codes = {row[0] for row in extended_catalog}
                extended_nav_counts = Counter(row[0] for row in extended_nav)
                target.executemany(
                    """
                    INSERT OR REPLACE INTO otf_fund_catalog (
                        fund_code,fund_name,fund_family,share_class,fund_type,
                        etf_symbol,etf_name,underlying_name,asset_class,
                        mapping_score,mapping_confidence,mapping_method,
                        inception_date,termination_date,source,fetched_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    extended_catalog,
                )
                target.executemany(
                    "DELETE FROM otf_fund_nav WHERE fund_code=?",
                    [(code,) for code in extended_codes],
                )
                target.executemany(
                    "INSERT INTO otf_fund_nav VALUES (?,?,?,?,?,?,?,?,?)",
                    extended_nav,
                )
                inserted.extend(
                    {
                        "fund_code": row[0],
                        "fund_name": row[1],
                        "research_asset_class": row[8],
                        "nav_rows": extended_nav_counts[row[0]],
                        "source": "extended_otf_assets",
                    }
                    for row in extended_catalog
                )
            target.execute("DROP TABLE IF EXISTS research_db_manifest")
            target.execute(
                """
                CREATE TABLE research_db_manifest (
                    generated_at TEXT, mapped_db_sha256 TEXT,
                    defensive_db_sha256 TEXT, defensive_codes TEXT,
                    extended_db_sha256 TEXT, source_independent INTEGER,
                    pit_status TEXT
                )
                """
            )
            target.execute(
                "INSERT INTO research_db_manifest VALUES (?,?,?,?,?,?,?)",
                (
                    datetime.now().astimezone().isoformat(),
                    _sha256(mapped_db),
                    _sha256(defensive_db),
                    ";".join(defensive_codes),
                    _sha256(extended_db) if extended_db is not None else "",
                    0,
                    "PIT_PARTIAL",
                ),
            )
            target.commit()
        target.close()
        source.close()
        target = None
        source = None
        os.replace(temporary, output_db)
    except Exception:
        if target is not None:
            target.close()
        if source is not None:
            source.close()
        if temporary.exists():
            temporary.unlink()
        raise

    with sqlite3.connect(output_db) as connection:
        fund_count = connection.execute(
            "SELECT COUNT(*) FROM otf_fund_catalog"
        ).fetchone()[0]
        nav_count = connection.execute(
            "SELECT COUNT(*) FROM otf_fund_nav"
        ).fetchone()[0]
    return {
        "output_db": str(output_db),
        "fund_count": int(fund_count),
        "nav_count": int(nav_count),
        "defensive_funds": inserted,
        "pit_status": "PIT_PARTIAL",
        "source_independent": False,
        "extended_db": str(extended_db) if extended_db is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapped-db", type=Path, default=MAPPED_DB)
    parser.add_argument("--defensive-db", type=Path, default=DEFENSIVE_DB)
    parser.add_argument("--output-db", type=Path, default=OUTPUT_DB)
    parser.add_argument(
        "--defensive-codes", default=",".join(DEFAULT_DEFENSIVE_CODES)
    )
    parser.add_argument("--extended-db", type=Path, default=EXTENDED_DB)
    args = parser.parse_args()
    codes = tuple(code.strip() for code in args.defensive_codes.split(",") if code.strip())
    print(
        json.dumps(
            build_research_db(
                args.mapped_db, args.defensive_db, args.output_db, codes,
                args.extended_db,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
