"""Tests verifying experiment artifacts are wired into runners per P0-A #5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest


class TestRunnerWiring:
    """Verify artifact export functions are imported and called in runner source."""

    def test_run_s1_imports_artifacts(self):
        src = (Path(__file__).resolve().parent.parent / "src/run_s1_experiment.py").read_text(encoding="utf-8")
        assert "from otf_rotation.experiment_artifacts import" in src
        assert "create_run_directory" in src
        assert "write_manifest" in src
        assert "export_daily_nav" in src
        assert "export_orders" in src
        assert "export_rejections" in src
        assert "export_metrics" in src

    def test_run_s1_calls_create_run_directory(self):
        src = (Path(__file__).resolve().parent.parent / "src/run_s1_experiment.py").read_text(encoding="utf-8")
        assert "create_run_directory(" in src

    def test_run_s1_calls_export_daily_nav(self):
        src = (Path(__file__).resolve().parent.parent / "src/run_s1_experiment.py").read_text(encoding="utf-8")
        assert "export_daily_nav(" in src

    def test_run_s1_calls_write_manifest(self):
        src = (Path(__file__).resolve().parent.parent / "src/run_s1_experiment.py").read_text(encoding="utf-8")
        assert "write_manifest(" in src

    def test_run_s1_exports_orders_and_rejections(self):
        src = (Path(__file__).resolve().parent.parent / "src/run_s1_experiment.py").read_text(encoding="utf-8")
        assert "export_orders(" in src
        assert "export_rejections(" in src

    def test_run_unified_imports_research_status(self):
        src = (Path(__file__).resolve().parent.parent / "src/run_unified_experiment.py").read_text(encoding="utf-8")
        assert "from otf_rotation.research_status import" in src
        assert "write_research_status(" in src
