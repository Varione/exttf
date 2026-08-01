"""Tests for Experiment Artifacts exporter per P0-A."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import json
import os
import tempfile
import pandas as pd
from otf_rotation.experiment_artifacts import (
    sha256_file,
    gather_environment,
    create_run_directory,
    write_manifest,
    export_daily_nav,
    export_orders,
    export_rejections,
    export_selection_audits,
    export_metrics,
)


@pytest.fixture
def tmp_root():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_db(tmp_root):
    path = os.path.join(tmp_root, "test.db")
    with open(path, "w") as f:
        f.write("dummy content")
    return path


class TestSha256File:
    def test_existing_file(self, sample_db):
        h = sha256_file(sample_db)
        assert isinstance(h, str) and len(h) == 64

    def test_missing_file(self):
        assert sha256_file("/nonexistent/path") is None


class TestGatherEnvironment:
    def test_structure(self):
        env = gather_environment(".")
        assert "python_version" in env
        assert "platform" in env
        assert "dependencies" in env

    def test_git_fields_present(self):
        env = gather_environment(".")
        assert "git_commit" in env
        assert "git_dirty" in env


class TestCreateRunDirectory:
    def test_creates_directory(self, tmp_root):
        run_dir = create_run_directory(tmp_root, "test_run")
        assert os.path.isdir(run_dir)
        assert run_dir.startswith(os.path.join(tmp_root, "test_run"))


class TestWriteManifest:
    def test_writes_manifest(self, tmp_root, sample_db):
        run_dir = create_run_directory(tmp_root, "manifest_test")
        config = {"strategy": "S1"}
        manifest = write_manifest(
            run_dir=run_dir,
            strategy_name="S1_Test",
            config=config,
            db_path=sample_db,
            rules_path=sample_db,
            exposure_mapping_path=sample_db,
        )
        assert os.path.exists(os.path.join(run_dir, "manifest.json"))
        with open(os.path.join(run_dir, "manifest.json")) as f:
            loaded = json.load(f)
        assert loaded["strategy_name"] == "S1_Test"
        assert loaded["input_hashes"]["db_sha256"] is not None
        assert loaded["config_snapshot"] == config


class TestExportDailyNav:
    def test_exports_csv(self, tmp_root):
        run_dir = create_run_directory(tmp_root, "nav_test")
        df = pd.DataFrame({"date": ["2021-01-01"], "net_nav": [100.0]})
        export_daily_nav(run_dir, df)
        assert os.path.exists(os.path.join(run_dir, "daily_nav.csv"))


class TestExportOrders:
    def test_exports_orders(self, tmp_root):
        run_dir = create_run_directory(tmp_root, "orders_test")
        df = pd.DataFrame({"fund_code": ["123456"], "side": ["BUY"]})
        export_orders(run_dir, df)
        assert os.path.exists(os.path.join(run_dir, "orders.csv"))


class TestExportRejections:
    def test_exports_rejections(self, tmp_root):
        run_dir = create_run_directory(tmp_root, "rej_test")
        rejections = [{"fund_code": "123456", "reason": "RULE_MISSING"}]
        export_rejections(run_dir, rejections)
        path = os.path.join(run_dir, "order_rejections.csv")
        assert os.path.exists(path)

    def test_empty_rejections_header(self, tmp_root):
        run_dir = create_run_directory(tmp_root, "rej_empty_test")
        export_rejections(run_dir, [])
        with open(os.path.join(run_dir, "order_rejections.csv"), encoding="utf-8-sig") as f:
            header = f.readline().strip()
        assert "fund_code" in header


class TestExportSelectionAudits:
    def test_exports_audits(self, tmp_root):
        run_dir = create_run_directory(tmp_root, "audit_test")
        audits = [{
            "sleeve": "EQUITY_CN",
            "date": "2021-06-30",
            "top_n": 1,
            "selected": ["123456"],
            "candidates": [{"fund_code": "123456"}],
        }]
        export_selection_audits(run_dir, audits)
        assert os.path.exists(os.path.join(run_dir, "product_selection_audit.csv"))


class TestExportMetrics:
    def test_exports_metrics(self, tmp_root):
        run_dir = create_run_directory(tmp_root, "metrics_test")
        metrics = {"cagr": 0.05, "sharpe": 0.6}
        export_metrics(run_dir, metrics)
        with open(os.path.join(run_dir, "metrics.json")) as f:
            loaded = json.load(f)
        assert loaded["cagr"] == 0.05
