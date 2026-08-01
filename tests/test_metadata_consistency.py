"""Test metadata consistency and old manifest read-only integrity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


EXPECTED_OLD_C1_SHA = "6d685522854aff88cdb97cd528c769563e8f7cc6d7e78519621f89c3ea5f44ce"
EXPECTED_OLD_C2_SHA = "f9f8877646b68c4c7af4f5f785d71b3fd72991610447e021835e50bbf8e269d4"


def _latest_c1_run() -> Path:
    """Return the latest core_satellite run directory with calendar_correction_label."""
    base = ROOT / "reports/strategy_research/core_satellite"
    if not base.exists():
        pytest.skip("No C1 runs found")
    dirs = sorted(base.iterdir(), key=lambda d: d.name, reverse=True)
    for d in dirs:
        status_path = d / "core_satellite_status.json"
        if status_path.exists():
            try:
                s = json.loads(status_path.read_text(encoding="utf-8"))
                if s.get("calendar_correction_label") == "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS":
                    return d
            except (json.JSONDecodeError, OSError):
                continue
    pytest.skip("No calendar-corrected C1 run found")


def _latest_c2_run() -> Path:
    """Return the latest low_turnover_core_satellite run directory with calendar_correction_label."""
    base = ROOT / "reports/strategy_research/core_satellite_low_turnover"
    if not base.exists():
        pytest.skip("No C2 runs found")
    dirs = sorted(base.iterdir(), key=lambda d: d.name, reverse=True)
    for d in dirs:
        status_path = d / "low_turnover_status.json"
        if status_path.exists():
            try:
                s = json.loads(status_path.read_text(encoding="utf-8"))
                if s.get("calendar_correction_label") == "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS":
                    return d
            except (json.JSONDecodeError, OSError):
                continue
    pytest.skip("No calendar-corrected C2 run found")


class TestOldManifestIntegrity:
    """Verify old manifests are read-only and match recorded SHA."""

    def test_old_c1_manifest_sha_recorded(self):
        """SHA of old C1 manifest is recorded in old_run_sha_records.json."""
        records = json.loads((ROOT / "data/processed/old_run_sha_records.json").read_text(encoding="utf-8"))
        assert "C1" in records
        assert "C1_main_manifest_sha" in records["C1"]

    def test_old_c2_manifest_sha_recorded(self):
        """SHA of old C2 manifest is recorded in old_run_sha_records.json."""
        records = json.loads((ROOT / "data/processed/old_run_sha_records.json").read_text(encoding="utf-8"))
        assert "C2" in records
        assert "C2_main_manifest_sha" in records["C2"]

    def test_old_c1_manifest_no_superseded_field(self):
        """Old C1 manifest must NOT contain SUPERSEDED_BY_EXECUTION_CALENDAR_FIX."""
        path = ROOT / "reports/strategy_research/core_satellite/core_satellite_20260729_151927/manifest.json"
        m = json.loads(path.read_text(encoding="utf-8"))
        assert "SUPERSEDED_BY_EXECUTION_CALENDAR_FIX" not in m, (
            "Old C1 manifest must not be modified with SUPERSEDED field"
        )

    def test_old_c2_manifest_no_superseded_field(self):
        """Old C2 manifest must NOT contain SUPERSEDED_BY_EXECUTION_CALENDAR_FIX."""
        path = ROOT / "reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260729_163519/manifest.json"
        m = json.loads(path.read_text(encoding="utf-8"))
        assert "SUPERSEDED_BY_EXECUTION_CALENDAR_FIX" not in m, (
            "Old C2 manifest must not be modified with SUPERSEDED field"
        )


class TestNewRunMetadata:
    """Verify new calendar-corrected runs have required metadata fields."""

    def test_new_c1_status_has_calendar_label(self):
        path = ROOT / "reports/strategy_research/core_satellite/core_satellite_20260730_114756/core_satellite_status.json"
        s = json.loads(path.read_text(encoding="utf-8"))
        assert s.get("calendar_correction_label") == "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
        assert s.get("supersedes_run_id") == "core_satellite_20260729_151927"
        # Original sample_label is preserved, not overwritten
        assert "sample_label" in s

    def test_new_c1_input_facts_has_calendar_label(self):
        path = ROOT / "reports/strategy_research/core_satellite/core_satellite_20260730_114756/core_satellite_input_facts.json"
        f = json.loads(path.read_text(encoding="utf-8"))
        assert f.get("calendar_correction_label") == "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
        assert f.get("supersedes_run_id") == "core_satellite_20260729_151927"

    def test_new_c2_status_has_calendar_label(self):
        path = ROOT / "reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_115448/low_turnover_status.json"
        s = json.loads(path.read_text(encoding="utf-8"))
        assert s.get("calendar_correction_label") == "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
        assert s.get("supersedes_run_id") == "low_turnover_core_satellite_20260729_163519"

    def test_new_c2_input_facts_has_calendar_label(self):
        path = ROOT / "reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_115448/low_turnover_input_facts.json"
        f = json.loads(path.read_text(encoding="utf-8"))
        assert f.get("calendar_correction_label") == "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
        assert f.get("supersedes_run_id") == "low_turnover_core_satellite_20260729_163519"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])