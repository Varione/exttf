"""Tests for the defensive OTC database merge."""

from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_otf_research_db import build_research_db


ROOT = Path(__file__).resolve().parents[1]
MAPPED = ROOT / "data" / "processed" / "otf_mapped.sqlite"
DEFENSIVE = ROOT / "data" / "processed" / "otf.sqlite"
EXTENDED = ROOT / "data" / "processed" / "otf_extended_assets.sqlite"


@pytest.mark.skipif(
    not MAPPED.exists() or not DEFENSIVE.exists(), reason="production OTC data absent"
)
def test_build_research_db_adds_defensive_fund_without_mutating_source(tmp_path):
    mapped_copy = tmp_path / "mapped.sqlite"
    defensive_copy = tmp_path / "defensive.sqlite"
    output = tmp_path / "research.sqlite"
    shutil.copy2(MAPPED, mapped_copy)
    shutil.copy2(DEFENSIVE, defensive_copy)
    before_size = mapped_copy.stat().st_size

    result = build_research_db(
        mapped_copy, defensive_copy, output, ("006663",)
    )

    assert output.exists()
    assert mapped_copy.stat().st_size == before_size
    assert result["defensive_funds"][0]["fund_code"] == "006663"
    with sqlite3.connect(output) as connection:
        catalog = connection.execute(
            "SELECT asset_class,mapping_method FROM otf_fund_catalog "
            "WHERE fund_code='006663'"
        ).fetchone()
        nav_count = connection.execute(
            "SELECT COUNT(*) FROM otf_fund_nav WHERE fund_code='006663'"
        ).fetchone()[0]
        manifest_count = connection.execute(
            "SELECT COUNT(*) FROM research_db_manifest"
        ).fetchone()[0]
    assert catalog == ("bond_short", "approved_direct_otf_v1")
    assert nav_count > 1000
    assert manifest_count == 1


def test_missing_source_database_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_research_db(
            tmp_path / "missing.sqlite",
            tmp_path / "also_missing.sqlite",
            tmp_path / "output.sqlite",
        )


@pytest.mark.skipif(
    not MAPPED.exists() or not DEFENSIVE.exists() or not EXTENDED.exists(),
    reason="production extended OTC data absent",
)
def test_build_research_db_merges_extended_asset_classes(tmp_path):
    output = tmp_path / "research.sqlite"
    result = build_research_db(
        MAPPED, DEFENSIVE, output, (), extended_db=EXTENDED
    )
    with sqlite3.connect(output) as connection:
        classes = dict(
            connection.execute(
                "SELECT fund_code,asset_class FROM otf_fund_catalog "
                "WHERE mapping_method='approved_direct_otf_extended_v1'"
            ).fetchall()
        )
    assert classes["260102"] == "money"
    assert classes["001512"] == "bond_government_3_5"
    assert result["extended_db"] == str(EXTENDED)
