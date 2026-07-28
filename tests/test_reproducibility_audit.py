"""Tests for reproducibility audit module."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reproducibility_audit import (
    Finding,
    audit_project,
    main,
    verify_manifest_reproducibility,
    verify_summary_reproducibility,
    _check_stale_factors_references,
    _check_direct_raw_close_loading,
    _check_missing_transaction_cost_gross_return,
    _check_missing_pit_price_mode_metadata,
    _check_db_schema_for_metadata,
    _check_otc_nav_availability,
    _check_arbitrary_return_clipping,
    _check_db_candidate_consistency,
    _check_factor_artifact_readiness,
    _is_excluded_path,
    _db_fingerprint,
)


def make_temp_project(tmp_path: Path) -> Path:
    """Create a temporary mini project structure for testing."""
    src_dir = tmp_path / "src"
    docs_dir = tmp_path / "docs"
    data_dir = tmp_path / "data" / "processed"
    tests_dir = tmp_path / "tests"

    src_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    return src_dir


def create_db_with_schema(
    tmp_path: Path,
    has_tables: list[str] | None = None,
    price_modes_schema: str = "schema_a",
) -> Path:
    """Create a temporary SQLite database with specified tables.

    Args:
        tmp_path: Temporary directory
        has_tables: List of table names to create
        price_modes_schema: 'schema_a' (price_mode) or 'schema_b' (raw_close/selected_price_mode)
    """
    db_path = tmp_path / "test_etf.sqlite"

    if has_tables is None:
        has_tables = ["etf_daily", "etf_daily_price_modes"]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for table_name in has_tables:
        if table_name == "etf_daily":
            cursor.execute(
                """CREATE TABLE etf_daily (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    amount REAL)"""
            )
        elif table_name == "etf_daily_price_modes":
            if price_modes_schema == "schema_a":
                cursor.execute(
                    """CREATE TABLE etf_daily_price_modes (
                        symbol TEXT,
                        date TEXT,
                        price_mode TEXT,
                        validation_status TEXT)"""
                )
            else:
                cursor.execute(
                    """CREATE TABLE etf_daily_price_modes (
                        symbol TEXT,
                        date TEXT,
                        raw_close REAL,
                        hfq_reference REAL,
                        legacy_proxy REAL,
                        total_return_proxy REAL,
                        cumulative_dividend REAL,
                        split_factor REAL,
                        factor_f REAL,
                        selected_price_mode TEXT,
                        validation_status TEXT)"""
                )
        elif table_name == "fund_nav":
            cursor.execute(
                """CREATE TABLE fund_nav (
                    fund_code TEXT,
                    nav_date TEXT,
                    total_return_nav REAL)"""
            )

    conn.commit()
    conn.close()

    return db_path


# ============================================================================
# Finding dataclass tests
# ============================================================================


class TestFindingDataclass:
    """Tests for Finding dataclass."""

    def test_finding_creation(self):
        f = Finding(
            file_path="test.py",
            issue_type="TEST_ISSUE",
            severity="P1",
            description="Test description",
            recommended_fix="Fix this",
        )
        assert f.file_path == "test.py"
        assert f.issue_type == "TEST_ISSUE"
        assert f.severity == "P1"
        assert f.description == "Test description"
        assert f.recommended_fix == "Fix this"

    def test_finding_to_dict(self):
        f = Finding(
            file_path="test.py",
            issue_type="TEST_ISSUE",
            severity="P2",
            description="Test",
            recommended_fix="Fix",
        )
        d = f.to_dict()
        assert isinstance(d, dict)
        assert "file_path" in d
        assert "issue_type" in d
        assert "severity" in d
        assert "description" in d
        assert "recommended_fix" in d


# ============================================================================
# Path-aware exclusion tests
# ============================================================================


class TestPathExclusion:
    """Tests for _is_excluded_path filtering."""

    def test_excludes_audit_module(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        audit_file = src_dir / "reproducibility_audit.py"
        audit_file.write_text("# audit module content")
        assert _is_excluded_path(audit_file, src_dir) is True

    def test_excludes_test_files(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        test_file = src_dir / "test_something.py"
        test_file.write_text("# test file")
        assert _is_excluded_path(test_file, src_dir) is True

    def test_excludes_files_in_tests_dir(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        tests_subdir = src_dir / "tests"
        tests_subdir.mkdir(exist_ok=True)
        test_file = tests_subdir / "helper.py"
        test_file.write_text("# helper")
        assert _is_excluded_path(test_file, src_dir) is True

    def test_does_not_exclude_deprecated_scripts(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        deprecated_dir = src_dir / "deprecated"
        deprecated_dir.mkdir(exist_ok=True)
        script_file = deprecated_dir / "run_targeted.py"
        script_file.write_text("# deprecated script")
        assert _is_excluded_path(script_file, src_dir) is False

    def test_does_not_exclude_regular_source(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        normal_file = src_dir / "factor_engine.py"
        normal_file.write_text("# normal source")
        assert _is_excluded_path(normal_file, src_dir) is False


# ============================================================================
# False-positive exclusion tests
# ============================================================================


class TestFalsePositiveExclusions:
    """Tests that the audit module and test files are not scanned for findings."""

    def test_audit_module_not_scanned_for_stale_refs(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        # The audit module contains strings like 'factors_all.csv' as detector patterns
        audit_file = src_dir / "reproducibility_audit.py"
        audit_file.write_text(
            '''# This file contains detector patterns
pattern = "factors_all.csv"  # used to test detection
'''
        )
        findings = _check_stale_factors_references(src_dir)
        assert len(findings) == 0

    def test_audit_module_not_scanned_for_clipping(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        audit_file = src_dir / "reproducibility_audit.py"
        audit_file.write_text(
            '''# Detector patterns include np.clip references
clip_patterns = [r'np\\.clip\\s*\\(']
'''
        )
        findings = _check_arbitrary_return_clipping(src_dir)
        assert len(findings) == 0

    def test_test_fixture_not_scanned(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        tests_dir = src_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        fixture_file = tests_dir / "conftest.py"
        fixture_file.write_text(
            '''factors = pd.read_csv("data/processed/factors_all.csv")
'''
        )
        findings = _check_stale_factors_references(src_dir)
        assert len(findings) == 0

    def test_string_literal_not_flagged(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        py_file = src_dir / "detector_test.py"
        # Line inside a multi-line string should not be flagged
        py_file.write_text(
            '''"""Test docstring mentioning factors_all.csv as test data."""
x = 1
'''
        )
        findings = _check_stale_factors_references(src_dir)
        assert len(findings) == 0


# ============================================================================
# Stale factors reference tests
# ============================================================================


class TestStaleFactorsReferences:
    """Tests for stale factors_all.csv reference detection."""

    def test_detects_stale_reference(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "bad_loader.py"
        py_file.write_text(
            'factors = pd.read_csv("data/processed/factors_all.csv", parse_dates=["date"])'
        )

        findings = _check_stale_factors_references(src_dir)

        assert len(findings) == 1
        assert findings[0].file_path == str(py_file)
        assert findings[0].issue_type == "STALE_FACTORS_REFERENCE"
        assert findings[0].severity == "P0"
        assert "factors_all.csv" in findings[0].description

    def test_ignores_repaired_reference(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "good_loader.py"
        py_file.write_text(
            'factors = pd.read_csv("data/processed/factors_all_repaired.csv", parse_dates=["date"])'
        )

        findings = _check_stale_factors_references(src_dir)
        assert len(findings) == 0

    def test_ignores_both_references(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "mixed_loader.py"
        py_file.write_text(
            """
# Old reference in comment
factors = pd.read_csv("data/processed/factors_all_repaired.csv", parse_dates=["date"])
"""
        )

        findings = _check_stale_factors_references(src_dir)
        assert len(findings) == 0


# ============================================================================
# Direct raw close loading tests
# ============================================================================


class TestDirectRawCloseLoading:
    """Tests for direct raw close loading detection."""

    def test_detects_potential_raw_close_loading(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "bad_loader.py"
        py_file.write_text(
            """
df = pd.read_csv("data/raw/etf_daily.csv")
prices = df[df['close'] > 0]
"""
        )

        findings = _check_direct_raw_close_loading(src_dir)

        assert len(findings) >= 1
        for f in findings:
            assert f.issue_type == "DIRECT_RAW_CLOSE_LOADING"

    def test_ignores_price_mode_usage(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "good_loader.py"
        py_file.write_text(
            """
from data_loader import load_price_series
prices = load_price_series(db_path, price_mode="total_return_proxy")
"""
        )

        findings = _check_direct_raw_close_loading(src_dir)
        assert len(findings) == 0


# ============================================================================
# Missing cost accounting tests
# ============================================================================


class TestMissingCostAccounting:
    """Tests for missing transaction_cost/gross_return detection."""

    def test_detects_missing_cost_accounting(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "backtest.py"
        py_file.write_text(
            """
def calculate_return(ret):
    return ret * 1.0
"""
        )

        findings = _check_missing_transaction_cost_gross_return(src_dir)

        assert len(findings) >= 1
        for f in findings:
            assert f.issue_type == "MISSING_COST_ACCOUNTING"

    def test_ignores_proper_cost_tracking(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "good_backtest.py"
        py_file.write_text(
            """
gross_return = sum(weight * ret for weight, ret in positions.items())
transaction_cost = traded_notional * fee_rate
net_return = gross_return - transaction_cost
"""
        )

        findings = _check_missing_transaction_cost_gross_return(src_dir)
        assert len(findings) == 0


# ============================================================================
# Missing PIT metadata tests
# ============================================================================


class TestMissingPITMetadata:
    """Tests for missing PIT/price_mode metadata detection."""

    def test_detects_missing_pit_metadata(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "factor_loader.py"
        py_file.write_text(
            """
factors = pd.read_csv("data/processed/factors_all.csv")
"""
        )

        findings = _check_missing_pit_price_mode_metadata(src_dir)

        assert len(findings) >= 1
        for f in findings:
            assert f.issue_type == "MISSING_PIT_METADATA"

    def test_ignores_pit_metadata_usage(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "good_loader.py"
        py_file.write_text(
            """
factors = pd.read_csv("data/processed/factors_all_repaired.csv")
pit_eligible = factors["pit_eligible"]
"""
        )

        findings = _check_missing_pit_price_mode_metadata(src_dir)
        assert len(findings) == 0


# ============================================================================
# DB schema tests - both valid schemas accepted
# ============================================================================


class TestDBSchemaChecks:
    """Tests for database schema validation."""

    def test_detects_missing_etf_daily_table(self, tmp_path: Path):
        db_path = create_db_with_schema(tmp_path, has_tables=[])

        findings = _check_db_schema_for_metadata(db_path)

        assert any(f.issue_type == "MISSING_ETF_DAILY_TABLE" for f in findings)
        assert any(
            f.severity == "P0"
            for f in findings
            if f.issue_type == "MISSING_ETF_DAILY_TABLE"
        )

    def test_detects_missing_etf_daily_price_modes_table(self, tmp_path: Path):
        db_path = create_db_with_schema(tmp_path, has_tables=["etf_daily"])

        findings = _check_db_schema_for_metadata(db_path)

        assert any(f.issue_type == "MISSING_PRICE_MODES_TABLE" for f in findings)

    def test_accepts_schema_a(self, tmp_path: Path):
        """Schema A: price_mode + validation_status should NOT trigger MISSING_PRICE_MODES_COLUMNS."""
        db_path = create_db_with_schema(
            tmp_path,
            has_tables=["etf_daily", "etf_daily_price_modes"],
            price_modes_schema="schema_a",
        )

        findings = _check_db_schema_for_metadata(db_path)
        assert not any(f.issue_type == "MISSING_PRICE_MODES_COLUMNS" for f in findings)

    def test_accepts_schema_b(self, tmp_path: Path):
        """Schema B: raw_close/hfq_reference/total_return_proxy/selected_price_mode + validation_status
        should NOT trigger MISSING_PRICE_MODES_COLUMNS."""
        db_path = create_db_with_schema(
            tmp_path,
            has_tables=["etf_daily", "etf_daily_price_modes"],
            price_modes_schema="schema_b",
        )

        findings = _check_db_schema_for_metadata(db_path)
        assert not any(f.issue_type == "MISSING_PRICE_MODES_COLUMNS" for f in findings)

    def test_rejects_invalid_schema(self, tmp_path: Path):
        """A schema missing both price_mode and selected_price_mode should be flagged."""
        db_path = tmp_path / "bad_schema.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE etf_daily (symbol TEXT, date TEXT, close REAL)"""
        )
        cursor.execute(
            """CREATE TABLE etf_daily_price_modes (symbol TEXT, date TEXT, some_col TEXT)"""
        )
        conn.commit()
        conn.close()

        findings = _check_db_schema_for_metadata(db_path)
        assert any(f.issue_type == "MISSING_PRICE_MODES_COLUMNS" for f in findings)

    def test_does_not_flag_missing_pit_columns(self, tmp_path: Path):
        db_path = create_db_with_schema(tmp_path, has_tables=["etf_daily"])

        findings = _check_db_schema_for_metadata(db_path)

        assert not any(f.issue_type == "MISSING_PIT_COLUMNS" for f in findings)


# ============================================================================
# DB candidate consistency tests
# ============================================================================


class TestDBCandidateConsistency:
    """Tests for DB candidate consistency check."""

    def test_detects_inconsistent_dbs(self, tmp_path: Path):
        """When two configured DBs exist but differ, should report P0 finding."""
        # Create DB 1 with schema A
        db1 = tmp_path / "db1.sqlite"
        conn1 = sqlite3.connect(db1)
        c1 = conn1.cursor()
        c1.execute(
            """CREATE TABLE etf_daily_price_modes (
                symbol TEXT, date TEXT, price_mode TEXT, validation_status TEXT)"""
        )
        c1.execute("INSERT INTO etf_daily_price_modes VALUES ('SH510300', '2024-01-01', 'raw_close', 'ok')")
        conn1.commit()
        conn1.close()

        # Create DB 2 with schema B (different)
        db2 = tmp_path / "db2.sqlite"
        conn2 = sqlite3.connect(db2)
        c2 = conn2.cursor()
        c2.execute(
            """CREATE TABLE etf_daily_price_modes (
                symbol TEXT, date TEXT, raw_close REAL, selected_price_mode TEXT, validation_status TEXT)"""
        )
        c2.execute("INSERT INTO etf_daily_price_modes VALUES ('SH510300', '2024-01-01', 4.5, 'raw_close', 'ok')")
        conn2.commit()
        conn2.close()

        findings = _check_db_candidate_consistency([str(db1), str(db2)])

        assert len(findings) >= 1
        assert any(f.issue_type == "DB_CANDIDATE_INCONSISTENCY" for f in findings)
        assert any(f.severity == "P0" for f in findings if f.issue_type == "DB_CANDIDATE_INCONSISTENCY")

    def test_no_finding_for_identical_dbs(self, tmp_path: Path):
        """When two configured DBs exist and are identical, no finding."""
        db1 = tmp_path / "db1.sqlite"
        conn1 = sqlite3.connect(db1)
        c1 = conn1.cursor()
        c1.execute("CREATE TABLE test (x INTEGER)")
        c1.execute("INSERT INTO test VALUES (1)")
        conn1.commit()
        conn1.close()

        # Copy db1 to db2
        import shutil
        db2 = tmp_path / "db2.sqlite"
        shutil.copy(str(db1), str(db2))

        findings = _check_db_candidate_consistency([str(db1), str(db2)])
        assert not any(f.issue_type == "DB_CANDIDATE_INCONSISTENCY" for f in findings)

    def test_no_finding_when_only_one_db_exists(self, tmp_path: Path):
        db1 = tmp_path / "db1.sqlite"
        conn1 = sqlite3.connect(db1)
        conn1.execute("CREATE TABLE test (x INTEGER)")
        conn1.commit()
        conn1.close()

        findings = _check_db_candidate_consistency([str(db1), str(tmp_path / "nonexistent.sqlite")])
        assert not any(f.issue_type == "DB_CANDIDATE_INCONSISTENCY" for f in findings)

    def test_no_finding_when_no_dbs_exist(self, tmp_path: Path):
        findings = _check_db_candidate_consistency(
            [str(tmp_path / "a.sqlite"), str(tmp_path / "b.sqlite")]
        )
        assert not any(f.issue_type == "DB_CANDIDATE_INCONSISTENCY" for f in findings)


# ============================================================================
# Factor artifact readiness tests
# ============================================================================


class TestFactorArtifactReadiness:
    """Tests for factor artifact manifest checks."""

    def _make_csv(self, data_dir: Path, num_rows: int = 10, symbols: list[str] | None = None) -> Path:
        csv_path = data_dir / "factors_all_repaired.csv"
        if symbols is None:
            symbols = [f"SH{i:06d}" for i in range(1, num_rows + 1)]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["symbol", "date", "value"])
            for i, sym in enumerate(symbols):
                writer.writerow([sym, f"2024-01-{i+1:02d}", float(i)])
        return csv_path

    def test_manifest_missing_is_p0(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        data_dir = tmp_path / "data" / "processed"
        self._make_csv(data_dir)

        findings = _check_factor_artifact_readiness(src_dir)

        assert any(f.issue_type == "FACTOR_MANIFEST_MISSING" for f in findings)
        assert any(
            f.severity == "P0" for f in findings if f.issue_type == "FACTOR_MANIFEST_MISSING"
        )

    def test_manifest_incomplete_completed_false_is_p0(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        data_dir = tmp_path / "data" / "processed"
        self._make_csv(data_dir, num_rows=5)

        manifest_path = data_dir / "factors_all_repaired.csv.manifest.json"
        manifest_path.write_text(json.dumps({"completed": False, "row_count": 5}))

        findings = _check_factor_artifact_readiness(src_dir)

        assert any(f.issue_type == "FACTOR_MANIFEST_INCOMPLETE" for f in findings)
        assert any(
            f.severity == "P0"
            for f in findings
            if f.issue_type == "FACTOR_MANIFEST_INCOMPLETE"
        )

    def test_manifest_missing_completed_field_is_p0(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        data_dir = tmp_path / "data" / "processed"
        self._make_csv(data_dir, num_rows=5)

        manifest_path = data_dir / "factors_all_repaired.csv.manifest.json"
        manifest_path.write_text(json.dumps({"row_count": 5}))

        findings = _check_factor_artifact_readiness(src_dir)

        assert any(f.issue_type == "FACTOR_MANIFEST_INCOMPLETE" for f in findings)
        assert any(
            f.severity == "P0"
            for f in findings
            if f.issue_type == "FACTOR_MANIFEST_INCOMPLETE"
        )

    def test_manifest_row_count_mismatch(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        data_dir = tmp_path / "data" / "processed"
        self._make_csv(data_dir, num_rows=10)

        manifest_path = data_dir / "factors_all_repaired.csv.manifest.json"
        manifest_path.write_text(json.dumps({"completed": True, "row_count": 5}))

        findings = _check_factor_artifact_readiness(src_dir)

        assert any(f.issue_type == "FACTOR_MANIFEST_MISMATCH" for f in findings)

    def test_manifest_symbol_mismatch(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        data_dir = tmp_path / "data" / "processed"
        actual_symbols = ["SH510300", "SH510500"]
        self._make_csv(data_dir, num_rows=2, symbols=actual_symbols)

        manifest_path = data_dir / "factors_all_repaired.csv.manifest.json"
        manifest_path.write_text(
            json.dumps({"completed": True, "row_count": 2, "symbols": ["SH510300", "SH510000"]})
        )

        findings = _check_factor_artifact_readiness(src_dir)

        assert any(f.issue_type == "FACTOR_MANIFEST_MISMATCH" for f in findings)

    def test_valid_manifest_no_findings(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        data_dir = tmp_path / "data" / "processed"
        actual_symbols = ["SH510300", "SH510500"]
        self._make_csv(data_dir, num_rows=2, symbols=actual_symbols)

        manifest_path = data_dir / "factors_all_repaired.csv.manifest.json"
        manifest_path.write_text(
            json.dumps({"completed": True, "row_count": 2, "symbols": actual_symbols})
        )

        findings = _check_factor_artifact_readiness(src_dir)
        assert len(findings) == 0

    def test_no_csv_no_findings(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        # Don't create the CSV at all

        findings = _check_factor_artifact_readiness(src_dir)
        assert len(findings) == 0


# ============================================================================
# OTC NAV availability tests
# ============================================================================


class TestOTCNavAvailability:
    """Tests for OTC NAV table availability check."""

    def test_detects_missing_fund_nav_table(self, tmp_path: Path):
        db_path = create_db_with_schema(tmp_path, has_tables=["etf_daily"])

        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "otc_backtest.py"
        py_file.write_text(
            """from data_loader import load_price_series
prices = load_price_series(db_path, data_mode="otc_nav")"""
        )

        findings = _check_otc_nav_availability(src_dir)

        assert len(findings) == 1
        assert findings[0].issue_type == "OTC_NAV_UNAVAILABLE"
        assert findings[0].severity == "P0"

    def test_no_findings_when_no_otc_mode(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "etf_backtest.py"
        py_file.write_text(
            """
from data_loader import load_price_series
prices = load_price_series(db_path, data_mode="etf")
"""
        )

        findings = _check_otc_nav_availability(src_dir)
        assert len(findings) == 0

    def test_no_findings_when_fund_nav_exists(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "otc_backtest.py"
        py_file.write_text(
            """
from data_loader import load_price_series
prices = load_price_series(db_path, data_mode="otc_nav")
"""
        )

        db_path = create_db_with_schema(tmp_path, has_tables=["fund_nav"])

        findings = _check_otc_nav_availability(src_dir)
        assert len(findings) == 0


# ============================================================================
# Arbitrary return clipping tests
# ============================================================================


class TestArbitraryReturnClipping:
    """Tests for np.clip detection in return calculations."""

    def test_detects_np_clip(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "clipper.py"
        py_file.write_text(
            """
import numpy as np
returns = np.clip(returns, -0.1, 0.1)
"""
        )

        findings = _check_arbitrary_return_clipping(src_dir)

        assert len(findings) >= 1
        assert any(f.issue_type == "ARBITRARY_RETURN_CLIPPING" for f in findings)

    def test_detects_series_clip(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "clipper2.py"
        py_file.write_text(
            """
returns = returns.clip(lower=-0.05, upper=0.05)
"""
        )

        findings = _check_arbitrary_return_clipping(src_dir)

        assert len(findings) >= 1
        assert any(f.issue_type == "ARBITRARY_RETURN_CLIPPING" for f in findings)

    def test_ignores_commented_clip(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "clean_clip.py"
        py_file.write_text(
            """
# returns = np.clip(returns, -0.1, 0.1)
returns = returns * 1.0
"""
        )

        findings = _check_arbitrary_return_clipping(src_dir)
        assert len(findings) == 0

    def test_no_clip_has_no_findings(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "no_clip.py"
        py_file.write_text(
            """
returns = prices.pct_change()
signal = returns.mean()
"""
        )

        findings = _check_arbitrary_return_clipping(src_dir)
        assert len(findings) == 0


# ============================================================================
# OTC NAV configured DB paths tests
# ============================================================================


class TestOTCNavConfiguredDBPaths:
    """Tests that OTC NAV check uses configured db_paths, not arbitrary sqlite files."""

    def test_uses_configured_db_paths(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "otc_backtest.py"
        py_file.write_text(
            """from data_loader import load_price_series
prices = load_price_series(db_path, data_mode="otc_nav")"""
        )

        db_path = tmp_path / "configured.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE etf_daily (symbol TEXT)")
        conn.commit()
        conn.close()

        findings = _check_otc_nav_availability(src_dir, db_paths=[str(db_path)])

        assert len(findings) == 1
        assert findings[0].issue_type == "OTC_NAV_UNAVAILABLE"

    def test_no_findings_when_fund_nav_in_configured_db(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "otc_backtest.py"
        py_file.write_text(
            """from data_loader import load_price_series
prices = load_price_series(db_path, data_mode="otc_nav")"""
        )

        db_path = tmp_path / "configured.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE fund_nav (fund_code TEXT, nav_date TEXT, total_return_nav REAL)"
        )
        conn.commit()
        conn.close()

        findings = _check_otc_nav_availability(src_dir, db_paths=[str(db_path)])
        assert len(findings) == 0


# ============================================================================
# audit_project integration tests
# ============================================================================


class TestAuditProject:
    """Integration tests for audit_project function."""

    def test_audit_project_returns_findings_list(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "bad.py"
        py_file.write_text('factors = pd.read_csv("data/processed/factors_all.csv")')

        findings = audit_project(root=src_dir)

        assert isinstance(findings, list)
        assert len(findings) > 0

    def test_audit_project_outputs_json_when_path_provided(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "bad.py"
        py_file.write_text('factors = pd.read_csv("data/processed/factors_all.csv")')

        output_path = tmp_path / "audit_results.json"
        findings = audit_project(root=src_dir, output_path=str(output_path))

        assert output_path.exists()

        with open(output_path) as fh:
            data = json.load(fh)

        assert isinstance(data, list)
        assert len(data) > 0

        finding = data[0]
        assert "file_path" in finding
        assert "issue_type" in finding
        assert "severity" in finding
        assert "description" in finding
        assert "recommended_fix" in finding

    def test_audit_project_no_json_without_explicit_path(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "bad.py"
        py_file.write_text('factors = pd.read_csv("data/processed/factors_all.csv")')

        findings = audit_project(root=src_dir)

        assert not (tmp_path / "audit_results.json").exists()

    def test_findings_sorted_by_severity(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file1 = src_dir / "p0.py"
        py_file1.write_text('factors = pd.read_csv("data/processed/factors_all.csv")')

        findings = audit_project(root=src_dir)

        if len(findings) >= 2:
            assert findings[0]["severity"] <= findings[1]["severity"]


# ============================================================================
# DB schema integration in audit_project
# ============================================================================


class TestDBSchemaIntegration:
    """Tests that both DB schemas work correctly in full audit."""

    def test_schema_a_no_missing_columns_finding(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        db_path = create_db_with_schema(
            tmp_path,
            has_tables=["etf_daily", "etf_daily_price_modes"],
            price_modes_schema="schema_a",
        )

        findings = audit_project(root=src_dir, db_paths=[str(db_path)])
        assert not any(f["issue_type"] == "MISSING_PRICE_MODES_COLUMNS" for f in findings)

    def test_schema_b_no_missing_columns_finding(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        db_path = create_db_with_schema(
            tmp_path,
            has_tables=["etf_daily", "etf_daily_price_modes"],
            price_modes_schema="schema_b",
        )

        findings = audit_project(root=src_dir, db_paths=[str(db_path)])
        assert not any(f["issue_type"] == "MISSING_PRICE_MODES_COLUMNS" for f in findings)


# ============================================================================
# DB consistency integration
# ============================================================================


class TestDBConsistencyIntegration:
    """Tests that DB consistency check is included in audit_project."""

    def test_db_consistency_included(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        db1 = tmp_path / "db1.sqlite"
        conn1 = sqlite3.connect(db1)
        c1 = conn1.cursor()
        c1.execute("CREATE TABLE t1 (x INTEGER)")
        c1.execute("INSERT INTO t1 VALUES (1)")
        conn1.commit()
        conn1.close()

        db2 = tmp_path / "db2.sqlite"
        conn2 = sqlite3.connect(db2)
        c2 = conn2.cursor()
        c2.execute("CREATE TABLE t2 (y TEXT)")
        c2.execute("INSERT INTO t2 VALUES ('hello')")
        conn2.commit()
        conn2.close()

        findings = audit_project(root=src_dir, db_paths=[str(db1), str(db2)])
        assert any(f["issue_type"] == "DB_CANDIDATE_INCONSISTENCY" for f in findings)


# ============================================================================
# Factor artifact integration
# ============================================================================


class TestFactorArtifactIntegration:
    """Tests that factor artifact check is included in audit_project."""

    def test_manifest_missing_included(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        data_dir = tmp_path / "data" / "processed"

        csv_path = data_dir / "factors_all_repaired.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["symbol", "date", "value"])
            writer.writerow(["SH510300", "2024-01-01", 1.0])

        findings = audit_project(root=src_dir)
        assert any(f["issue_type"] == "FACTOR_MANIFEST_MISSING" for f in findings)


# ============================================================================
# Main CLI tests
# ============================================================================


class TestMainCLI:
    """Tests for main() CLI entry point."""

    def test_main_returns_zero_on_success(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "clean.py"
        py_file.write_text(
            """from data_loader import load_price_series
prices = load_price_series("data/processed/test_etf.sqlite")"""
        )

        result = main(argv=["--root", str(src_dir)])
        assert result == 0

    def test_main_returns_nonzero_on_findings(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "bad.py"
        py_file.write_text('factors = pd.read_csv("data/processed/factors_all.csv")')

        result = main(argv=["--root", str(src_dir)])
        assert result == 1


# ============================================================================
# Deprecated scripts still scanned
# ============================================================================


class TestDeprecatedScriptsScanned:
    """Verify deprecated directory scripts are still scanned."""

    def test_deprecated_scripts_included(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)
        deprecated_dir = src_dir / "deprecated"
        deprecated_dir.mkdir(exist_ok=True)

        script = deprecated_dir / "old_runner.py"
        script.write_text('factors = pd.read_csv("data/processed/factors_all.csv")')

        findings = _check_stale_factors_references(src_dir)
        assert len(findings) >= 1
        assert any("deprecated" in f.file_path for f in findings)


# ============================================================================
# Partial file skip tests
# ============================================================================


class TestPartialFileSkip:
    """Tests that files with both old and new references are not skipped entirely."""

    def test_file_with_both_references_flags_stale(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "mixed.py"
        py_file.write_text(
            """
# This file uses the repaired version for most things
factors_new = pd.read_csv("data/processed/factors_all_repaired.csv")
# But this line still references the old file
factors_old = pd.read_csv("data/processed/factors_all.csv")
"""
        )

        findings = _check_stale_factors_references(src_dir)

        assert len(findings) >= 1
        assert any("factors_all.csv" in f.description for f in findings)

    def test_file_with_only_repaired_has_no_findings(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "clean.py"
        py_file.write_text(
            """
factors = pd.read_csv("data/processed/factors_all_repaired.csv")
"""
        )

        findings = _check_stale_factors_references(src_dir)
        assert len(findings) == 0


# ============================================================================
# Audit project integration with new checks
# ============================================================================


class TestAuditProjectIntegration:
    """Integration tests for audit_project with new checks."""

    def test_clip_check_included_in_audit(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "clipper.py"
        py_file.write_text(
            """
import numpy as np
returns = np.clip(returns, -0.1, 0.1)
"""
        )

        findings = audit_project(root=src_dir)

        assert any(f["issue_type"] == "ARBITRARY_RETURN_CLIPPING" for f in findings)

    def test_otc_nav_uses_configured_paths(self, tmp_path: Path):
        src_dir = make_temp_project(tmp_path)

        py_file = src_dir / "otc_backtest.py"
        py_file.write_text(
            """from data_loader import load_price_series
prices = load_price_series(db_path, data_mode="otc_nav")"""
        )

        db_path = tmp_path / "test.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE etf_daily (symbol TEXT)")
        conn.commit()
        conn.close()

        findings = audit_project(root=src_dir, db_paths=[str(db_path)])

        assert any(f["issue_type"] == "OTC_NAV_UNAVAILABLE" for f in findings)


# ============================================================================
# Manifest reproducibility verification tests
# ============================================================================


class TestVerifyManifestReproducibility:
    """Tests for manifest comparison and reproducibility verification."""

    def _make_manifest(self, path: Path, overrides: dict | None = None) -> None:
        manifest = {
            "run_id": "test_run_001",
            "timestamp": "2026-07-28T10:00:00+08:00",
            "gate_passed": True,
            "failed_reasons": [],
            "config": {"fee_rate_per_side": 0.0003},
            "db_info": {"db_sha256": "abc123"},
            "factor_manifest": {"completed": True, "row_count": 100},
            "full_period_results": {
                "S04_VolTarget_Trend": {"CAGR%": 4.26, "Sharpe": 0.5}
            },
            "oos_results": {},
            "artifacts": {
                "data/processed/factors_all_repaired.csv": {
                    "sha256": "deadbeef",
                    "size_bytes": 1000,
                }
            },
            "regime_training_end": "2018-01-01",
            "python_version": "3.11.5",
            "dependencies": {"pandas": "2.0.0", "numpy": "1.24.0"},
            "git_commit": "abcdef1234567890",
            "git_status": "clean",
            "code_fingerprint": "fingerprint123",
        }
        if overrides:
            manifest.update(overrides)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def test_identical_manifests_are_reproducible(self, tmp_path: Path):
        m1 = tmp_path / "m1.json"
        m2 = tmp_path / "m2.json"
        self._make_manifest(m1)
        self._make_manifest(m2)

        result = verify_manifest_reproducibility(m1, m2)

        assert result["reproducible"] is True
        assert result["num_differences"] == 0
        assert not result["missing_protocol_fields"]

    def test_different_timestamps_are_ignored(self, tmp_path: Path):
        m1 = tmp_path / "m1.json"
        m2 = tmp_path / "m2.json"
        self._make_manifest(m1)
        self._make_manifest(
            m2,
            overrides={"timestamp": "2026-07-28T11:00:00+08:00"},
        )

        result = verify_manifest_reproducibility(m1, m2)

        assert result["reproducible"] is True
        assert result["num_differences"] == 0

    def test_different_metrics_are_detected(self, tmp_path: Path):
        m1 = tmp_path / "m1.json"
        m2 = tmp_path / "m2.json"
        self._make_manifest(m1)
        self._make_manifest(
            m2,
            overrides={
                "full_period_results": {
                    "S04_VolTarget_Trend": {"CAGR%": 5.00, "Sharpe": 0.6}
                }
            },
        )

        result = verify_manifest_reproducibility(m1, m2)

        assert result["reproducible"] is False
        assert result["num_differences"] >= 2

    def test_different_artifact_sha256_detected(self, tmp_path: Path):
        m1 = tmp_path / "m1.json"
        m2 = tmp_path / "m2.json"
        self._make_manifest(m1)
        self._make_manifest(
            m2,
            overrides={
                "artifacts": {
                    "data/processed/factors_all_repaired.csv": {
                        "sha256": "different_hash",
                        "size_bytes": 1000,
                    }
                }
            },
        )

        result = verify_manifest_reproducibility(m1, m2)

        assert result["reproducible"] is False
        assert "artifacts.data/processed/factors_all_repaired.csv.sha256" in result["differences"]

    def test_missing_protocol_fields_detected(self, tmp_path: Path):
        m1 = tmp_path / "m1.json"
        m2 = tmp_path / "m2.json"
        self._make_manifest(m1)
        old_style = {
            "run_id": "old_run",
            "timestamp": "2026-07-28T10:00:00+08:00",
            "gate_passed": True,
            "config": {},
            "db_info": {},
            "factor_manifest": {},
            "full_period_results": {},
            "oos_results": {},
            "artifacts": {},
            "regime_training_end": "2018-01-01",
        }
        m2.write_text(json.dumps(old_style, indent=2), encoding="utf-8")

        result = verify_manifest_reproducibility(m1, m2)

        assert result["reproducible"] is False
        assert len(result["missing_protocol_fields"]) > 0

    def test_missing_manifest_returns_error(self, tmp_path: Path):
        result = verify_manifest_reproducibility(
            tmp_path / "nonexistent1.json",
            tmp_path / "nonexistent2.json",
        )
        assert result["reproducible"] is False
        assert "error" in result


class TestVerifySummaryReproducibility:
    """Tests for summary.csv comparison."""

    def test_identical_summaries_are_reproducible(self, tmp_path: Path):
        r1 = tmp_path / "run1"
        r2 = tmp_path / "run2"
        r1.mkdir()
        r2.mkdir()

        csv_content = "strategy,CAGR%,Sharpe\nS04,4.26,0.5\n"
        (r1 / "summary.csv").write_text(csv_content, encoding="utf-8")
        (r2 / "summary.csv").write_text(csv_content, encoding="utf-8")

        result = verify_summary_reproducibility(r1, r2)

        assert result["reproducible"] is True
        assert result["columns_match"] is True
        assert result["shapes_match"] is True

    def test_different_values_detected(self, tmp_path: Path):
        r1 = tmp_path / "run1"
        r2 = tmp_path / "run2"
        r1.mkdir()
        r2.mkdir()

        (r1 / "summary.csv").write_text(
            "strategy,CAGR%,Sharpe\nS04,4.26,0.5\n", encoding="utf-8"
        )
        (r2 / "summary.csv").write_text(
            "strategy,CAGR%,Sharpe\nS04,5.00,0.6\n", encoding="utf-8"
        )

        result = verify_summary_reproducibility(r1, r2)

        assert result["reproducible"] is False
        assert "different_columns" in result

    def test_different_columns_detected(self, tmp_path: Path):
        r1 = tmp_path / "run1"
        r2 = tmp_path / "run2"
        r1.mkdir()
        r2.mkdir()

        (r1 / "summary.csv").write_text(
            "strategy,CAGR%,Sharpe\nS04,4.26,0.5\n", encoding="utf-8"
        )
        (r2 / "summary.csv").write_text(
            "strategy,CAGR%,MaxDD\nS04,4.26,-20\n", encoding="utf-8"
        )

        result = verify_summary_reproducibility(r1, r2)

        assert result["reproducible"] is False
        assert not result["columns_match"]

    def test_missing_summary_returns_error(self, tmp_path: Path):
        result = verify_summary_reproducibility(
            tmp_path / "run1", tmp_path / "run2"
        )
        assert result["reproducible"] is False
        assert "error" in result


class TestVerifyManifestCLI:
    """Tests for CLI manifest verification commands."""

    def test_verify_manifest_command(self, tmp_path: Path):
        m1 = tmp_path / "m1.json"
        m2 = tmp_path / "m2.json"
        manifest = {
            "run_id": "r1",
            "timestamp": "2026-07-28T10:00:00+08:00",
            "gate_passed": True,
            "config": {},
            "db_info": {},
            "factor_manifest": {},
            "full_period_results": {"S04": {"CAGR%": 4.26}},
            "oos_results": {},
            "artifacts": {},
            "regime_training_end": "2018-01-01",
            "python_version": "3.11.5",
            "dependencies": {},
            "git_commit": "abc",
            "git_status": "clean",
            "code_fingerprint": "fp1",
        }
        m1.write_text(json.dumps(manifest), encoding="utf-8")
        manifest["timestamp"] = "2026-07-28T11:00:00+08:00"
        m2.write_text(json.dumps(manifest), encoding="utf-8")

        result = main(argv=["verify-manifest", str(m1), str(m2)])
        assert result == 0

    def test_verify_manifest_different_returns_nonzero(self, tmp_path: Path):
        m1 = tmp_path / "m1.json"
        m2 = tmp_path / "m2.json"
        manifest = {
            "run_id": "r1",
            "timestamp": "2026-07-28T10:00:00+08:00",
            "gate_passed": True,
            "config": {},
            "db_info": {},
            "factor_manifest": {},
            "full_period_results": {"S04": {"CAGR%": 4.26}},
            "oos_results": {},
            "artifacts": {},
            "regime_training_end": "2018-01-01",
            "python_version": "3.11.5",
            "dependencies": {},
            "git_commit": "abc",
            "git_status": "clean",
            "code_fingerprint": "fp1",
        }
        m1.write_text(json.dumps(manifest), encoding="utf-8")
        manifest["full_period_results"] = {"S04": {"CAGR%": 5.00}}
        m2.write_text(json.dumps(manifest), encoding="utf-8")

        result = main(argv=["verify-manifest", str(m1), str(m2)])
        assert result == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
