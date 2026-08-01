"""Tests for research_status module per P0-B."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import json
import tempfile
import os
from otf_rotation.research_status import (
    build_status,
    write_status,
    read_status,
    mark_stale_if_inputs_changed,
    DEFAULT_STATUS_PATH,
)


class TestBuildStatus:
    def test_structure(self):
        status = build_status(run_id="test_001", status="BUILD_FAILED")
        assert status["run_id"] == "test_001"
        assert status["status"] == "BUILD_FAILED"
        assert "timestamp" in status
        assert status["completed_phases"] == []
        assert status["failed_phases"] == []

    def test_all_fields(self):
        status = build_status(
            run_id="test_002",
            status="PAPER_TRADE_CANDIDATE",
            completed_phases=["DATA_GATE", "BACKTEST"],
            failed_phases=[],
            test_result={"passed": 100},
            input_hashes={"db_sha256": "abc"},
            rule_counts_by_status={"OFFICIAL_VERIFIED": 3},
            mapping_counts_by_status={"HIGH_APPROVED": 10},
            strategies_executed=["B1", "S1"],
            oos_period="2021-2026",
            gate_result={"passed": True},
            blocking_issues=[],
        )
        assert status["strategies_executed"] == ["B1", "S1"]
        assert status["rule_counts_by_status"]["OFFICIAL_VERIFIED"] == 3


class TestWriteAndReadStatus:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "status.json")
            write_status(path, run_id="test_003", status="TEST_FAILED")
            result = read_status(path)
            assert result is not None
            assert result["run_id"] == "test_003"

    def test_read_missing(self):
        assert read_status("/nonexistent/status.json") is None

    def test_atomic_write_creates_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "status.json")
            write_status(path, run_id="test_004", status="DATA_GATE_FAILED")
            assert os.path.exists(path)


class TestMarkStale:
    def test_no_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "status.json")
            write_status(
                path,
                run_id="test_005",
                status="PAPER_TRADE_CANDIDATE",
                input_hashes={"db": "aaa"},
            )
            changed = mark_stale_if_inputs_changed({"db": "aaa"}, path)
            assert not changed

    def test_inputs_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "status.json")
            write_status(
                path,
                run_id="test_006",
                status="PAPER_TRADE_CANDIDATE",
                input_hashes={"db": "aaa"},
            )
            changed = mark_stale_if_inputs_changed({"db": "bbb"}, path)
            assert changed
            result = read_status(path)
            assert result["status"] == "STALE_INPUTS"
