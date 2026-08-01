"""Tests for the P2 frozen-forward mechanisms (planning P2-1..P2-4).

Covers: frozen version registry, T0 registry, forward data revision
mechanism, shadow observation schema and validator, and the forward
upgrade Gate framework. These tests are deliberately lightweight and do
not rerun the frozen strategy engines.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, str(ROOT / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def frozen_rows():
    path = ROOT / "config" / "otf_frozen_strategy_versions.csv"
    assert path.exists(), "frozen version registry missing"
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestFrozenVersionRegistry:
    def test_all_frozen_strategies_registered(self, frozen_rows):
        keys = {r["strategy_key"] for r in frozen_rows}
        assert {"B2LT", "C1", "C3", "M20", "C2"} <= keys

    def test_freeze_content_complete(self, frozen_rows):
        required = [
            "git_commit", "data_cutoff_date", "parameter_freeze_id",
            "trading_rule_scenario", "execution_calendar_sha256",
            "config_sha256", "gate_threshold_ref", "product_pool",
            "run_id",
        ]
        for row in frozen_rows:
            for field in required:
                assert row.get(field), f"{row['strategy_key']} missing {field}"

    def test_c2_archived_status(self, frozen_rows):
        c2 = next(r for r in frozen_rows if r["strategy_key"] == "C2")
        assert c2["freeze_status"] == "ARCHIVED_PARAMETER_SEARCH_FORBIDDEN"

    def test_m20_pool_nonempty(self, frozen_rows):
        m20 = next(r for r in frozen_rows if r["strategy_key"] == "M20")
        assert m20["product_pool"], "M20 product pool should be nonempty"

    def test_generator_report_clean(self):
        report = json.loads(
            (ROOT / "reports" / "historical_truth"
             / "frozen_strategy_versions_report.json").read_text(encoding="utf-8")
        )
        assert report["freeze_content_complete"] is True
        assert report["problems"] == []


class TestT0Registry:
    def test_registry_exists_and_consistent(self):
        path = ROOT / "config" / "otf_t0_registry.json"
        assert path.exists()
        reg = json.loads(path.read_text(encoding="utf-8"))
        assert reg["t0_status"] in {"REGISTERED", "PENDING_CALENDAR_EXTENSION"}
        assert reg["data_cutoff_date"] == "2026-07-27"
        assert reg["baseline_git_commit"]

    def test_rerun_is_idempotent(self):
        result = _run_script("scripts/migrations/build_t0_registry.py")
        assert result.returncode == 0, result.stderr
        reg = json.loads(
            (ROOT / "config" / "otf_t0_registry.json").read_text(encoding="utf-8")
        )
        assert reg["data_cutoff_date"] == "2026-07-27"


class TestDataRevision:
    def test_verify_clean(self):
        result = _run_script("scripts/migrations/record_data_revision.py", "--verify")
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(result.stdout)
        assert report["status"] == "CLEAN"

    def test_registry_tracks_all_files(self):
        reg = json.loads(
            (ROOT / "config" / "data_revision_registry.json").read_text(encoding="utf-8")
        )
        assert len(reg["files"]) == 7
        assert "data/processed/otf_mapped.sqlite" in reg["files"]
        assert "data/processed/execution_calendar/cn_execution_calendar.csv" in reg["files"]

    def test_record_requires_reason(self):
        result = _run_script("scripts/migrations/record_data_revision.py", "--record")
        assert result.returncode == 1
        assert "reason" in result.stdout


class TestShadowSchema:
    def test_schema_valid(self):
        schema = json.loads(
            (ROOT / "config" / "forward_shadow_schema.json").read_text(encoding="utf-8")
        )
        required = schema["decision_file"]["required_fields"]
        for field in [
            "data_snapshot_sha256", "signal_observed_at", "target_weights",
            "submit_date", "confirm_date", "value_date", "rule_version_id",
            "estimated_fees", "actual_fees", "rejections", "purchase_limits",
            "deferrals", "target_vs_actual_deviation",
        ]:
            assert field in required

    def test_validator_fails_honestly_without_records(self):
        result = _run_script("scripts/forward/validate_shadow_records.py")
        assert result.returncode == 1
        report = json.loads(result.stdout)
        assert report["status"] == "FAIL"
        assert report["checks"]["data_revision_clean"] is True


class TestForwardUpgradeGate:
    def test_c3_gate_fails_honestly(self):
        result = _run_script("scripts/forward/forward_upgrade_gate.py", "C3")
        assert result.returncode == 1
        gate = json.loads(result.stdout)
        assert gate["gate_passed"] is False
        assert gate["status"] == "NOT_ELIGIBLE_FOR_PAPER_TRADE"
        assert len(gate["conditions"]) == 9

    def test_p1_condition_reflects_reality(self):
        result = _run_script("scripts/forward/forward_upgrade_gate.py", "C3")
        gate = json.loads(result.stdout)
        p1 = next(c for c in gate["conditions"] if c["id"] == 2)
        assert p1["status"] == "FAIL"
        p0 = next(c for c in gate["conditions"] if c["id"] == 1)
        assert p0["status"] == "PASS"
