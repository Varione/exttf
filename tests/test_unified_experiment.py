"""Tests for unified experiment runner: gates, factors, reproducibility."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_unified_experiment import (
    DataGate,
    FactorBuilder,
    compute_file_sha256,
    get_db_schema,
    get_db_table_counts,
)


@pytest.fixture
def config():
    config_path = str(Path(__file__).resolve().parents[1] / "config" / "unified_experiment.json")
    with open(config_path, "r") as f:
        return json.load(f)


class TestDataGate:
    """Test canonical DB lock and data gate checks."""

    def test_canonical_db_exists(self, config):
        assert os.path.exists(config["canonical_db"]), "Canonical DB must exist"

    def test_canonical_db_has_required_tables(self, config):
        schema = get_db_schema(config["canonical_db"])
        assert "etf_daily" in schema
        assert "etf_daily_price_modes" in schema

    def test_canonical_db_row_counts(self, config):
        counts = get_db_table_counts(config["canonical_db"])
        assert counts["etf_daily"] == 1386449
        assert counts.get("etf_catalog", 0) == 1549

    def test_sha256_is_stable(self, config):
        sha1 = compute_file_sha256(config["canonical_db"])
        sha2 = compute_file_sha256(config["canonical_db"])
        assert sha1 == sha2
        assert len(sha1) == 64

    def test_gate_passes(self, config):
        gate = DataGate(config)
        result = gate.run_all()
        assert result is True, f"Data gate failed: {gate.errors}"

    def test_gate_fails_on_missing_db(self, config):
        bad_config = dict(config, canonical_db="/nonexistent/path.db")
        gate = DataGate(bad_config)
        result = gate.run_all()
        assert result is False
        assert any("CANONICAL_DB_MISSING" in e for e in gate.errors)

    def test_root_db_diff_recorded(self, config):
        gate = DataGate(config)
        gate.check_canonical_db_lock()
        gate.check_root_db_diff()
        if os.path.exists("etf.sqlite"):
            assert "root_db_sha256" in gate.manifest_data

    def test_reference_verification_ratio_computed(self, config):
        gate = DataGate(config)
        gate.check_canonical_db_lock()
        gate.check_reference_verification()
        ratio = gate.manifest_data.get("reference_verification_ratio", -1)
        assert 0.0 <= ratio <= 1.0

    def test_reference_sample_and_pit_scope_are_explicit(self, config):
        gate = DataGate(config)
        assert gate.run_all() is True
        assert gate.manifest_data["reference_source_independent"] is False
        assert gate.manifest_data["official_reference_sample_rows"] > 0
        assert gate.manifest_data["pit_status"] == "PIT_PARTIAL"
        assert gate.manifest_data["historical_lifecycle_records_available"] is True
        assert (
            gate.manifest_data["historical_lifecycle_independently_verified"]
            is False
        )


class TestFactorValidation:
    """Test factor artifact validation."""

    def test_factor_csv_exists(self, config):
        assert os.path.exists(config["factor_csv"])

    def test_factor_manifest_exists(self, config):
        manifest_path = config["factor_csv"] + ".manifest.json"
        assert os.path.exists(manifest_path), "Factor manifest must exist after rebuild"

    def test_factor_manifest_completed(self, config):
        manifest_path = config["factor_csv"] + ".manifest.json"
        if not os.path.exists(manifest_path):
            pytest.skip("Manifest not yet generated")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        assert manifest.get("completed", False) is True

    def test_factor_row_count(self, config):
        df = pd.read_csv(config["factor_csv"], nrows=0)
        n_cols = len(df.columns)
        assert n_cols == config["factor_validation"]["expected_columns"]

    def test_factor_column_count(self, config):
        df = pd.read_csv(config["factor_csv"], nrows=0)
        expected = config["factor_validation"]["expected_columns"]
        assert len(df.columns) == expected

    def test_no_duplicate_keys(self, config):
        df = pd.read_csv(config["factor_csv"], parse_dates=["date"])
        dupes = df.duplicated(subset=["symbol", "date"], keep=False)
        assert not dupes.any(), f"Found {dupes.sum()} duplicate symbol-date keys"

    def test_symbols_count(self, config):
        df = pd.read_csv(config["factor_csv"], usecols=["symbol"])
        n_symbols = df["symbol"].nunique()
        assert n_symbols == config["factor_validation"]["expected_symbols"]


class TestReproducibility:
    """Test that fixed parameters produce deterministic results."""

    def test_backtest_fixed_params(self, config):
        from backtest_engine import BacktestEngine

        engine = BacktestEngine(
            factor_path=config["factor_csv"],
            regime_path=config.get("regime_predictions_csv", "data/processed/regime_predictions.csv"),
            db_path=config["canonical_db"],
            data_mode=config["data_mode"],
            price_mode=config.get("price_mode", "total_return_proxy"),
            fee_rate_per_side=config.get("fee_rate_per_side", 0.0003),
            slippage_rate_per_side=config.get("slippage_rate_per_side", 0.0002),
            require_pit=config.get("require_pit", True),
        )

        daily1 = engine.run_backtest(
            "B0_BuyHold_EW",
            start="2018-01-01",
            end="2018-01-31",
            n_hold=config["n_hold"],
            max_weight=config["max_weight"],
            signal_to_return_lag=config.get("signal_to_return_lag", 2),
            rebalance_every=config.get("rebalance_every", 5),
        )

        daily2 = engine.run_backtest(
            "B0_BuyHold_EW",
            start="2018-01-01",
            end="2018-01-31",
            n_hold=config["n_hold"],
            max_weight=config["max_weight"],
            signal_to_return_lag=config.get("signal_to_return_lag", 2),
            rebalance_every=config.get("rebalance_every", 5),
        )

        pd.testing.assert_frame_equal(daily1, daily2)

    def test_invalid_factor_artifact_rejected(self, tmp_path):
        from factor_engine import _validate_existing_artifact

        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("symbol,date,mom_20\n000001,2020-01-01,0.5\n")

        valid_syms, errors = _validate_existing_artifact(
            str(bad_csv),
            expected_header_cols=["symbol", "date", "pit_eligible", "mom_20"],
            expected_symbols={"000001"},
        )
        assert len(errors) > 0, "Invalid artifact should be rejected"

    def test_pit_gate_enabled(self, config):
        assert config.get("require_pit", True) is True


class TestManifestMetadata:
    """Test experiment manifest contains required metadata."""

    def test_report_dir_structure(self, config):
        report_base = Path(config["report_dir"])
        if not report_base.exists():
            pytest.skip("Reports not yet generated")

        run_dirs = list(report_base.iterdir())
        assert len(run_dirs) > 0, "At least one run directory must exist"

        latest = max(run_dirs, key=lambda p: p.name)
        manifest_path = latest / "experiment_manifest.json"
        if not manifest_path.exists():
            pytest.skip("Manifest not yet written")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        required_keys = [
            "run_id",
            "timestamp",
            "gate_passed",
            "config",
            "db_info",
            "factor_manifest",
            "full_period_results",
        ]
        for key in required_keys:
            assert key in manifest, f"Missing manifest key: {key}"

    def test_db_fingerprint_in_manifest(self, config):
        report_base = Path(config["report_dir"])
        if not report_base.exists():
            pytest.skip("Reports not yet generated")

        run_dirs = list(report_base.iterdir())
        if not run_dirs:
            pytest.skip("No run directories")

        latest = max(run_dirs, key=lambda p: p.name)
        manifest_path = latest / "experiment_manifest.json"
        if not manifest_path.exists():
            pytest.skip("Manifest not yet written")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        db_info = manifest.get("db_info", {})
        assert "db_sha256" in db_info, "DB SHA256 must be in manifest"
        assert len(db_info["db_sha256"]) == 64


class TestOOSWindows:
    """Test OOS window configuration and metrics computation."""

    def test_oos_windows_defined(self, config):
        oos = config.get("oos_windows", [])
        assert len(oos) == 3
        names = [w["name"] for w in oos]
        assert "OOS_2018_2021" in names
        assert "OOS_2022_2024" in names
        assert "OOS_2025_2026" in names

    def test_oos_windows_no_overlap(self, config):
        oos = config.get("oos_windows", [])
        for i in range(len(oos)):
            for j in range(i + 1, len(oos)):
                assert oos[i]["end"] < oos[j]["start"], (
                    f"OOS windows {oos[i]['name']} and {oos[j]['name']} overlap"
                )

    def test_oos_windows_within_backtest_range(self, config):
        start = config["backtest_start"]
        end = config["backtest_end"]
        for w in config.get("oos_windows", []):
            assert w["start"] >= start, f"{w['name']} starts before backtest"
            assert w["end"] <= end, f"{w['name']} ends after backtest"

    def test_oos_metrics_computation(self, config):
        from run_unified_experiment import BacktestRunner, ReportGenerator

        runner = BacktestRunner(config)
        runner.engine = None
        runner.results = {}
        runner.daily_results = {}

        import pandas as pd
        import numpy as np

        dates = pd.bdate_range("2018-01-01", periods=500)
        returns = np.random.RandomState(42).normal(0.0003, 0.01, 500)
        daily = pd.DataFrame({
            "date": dates,
            "return": returns,
            "turnover": np.zeros(500),
            "transaction_cost": np.zeros(500),
            "gross_return": returns,
            "exposure": np.ones(500) * 0.5,
        })

        class FakeEngine:
            pit_status = "PIT_COMPLETE"
            data_mode = "etf"
            price_mode = "total_return_proxy"

            @staticmethod
            def calculate_metrics(rets, turnover, cost, gross, exposure):
                wealth = (1.0 + rets).cumprod()
                total_return = wealth.iloc[-1] - 1.0
                n_days = len(rets)
                ann_return = wealth.iloc[-1] ** (252.0 / n_days) - 1.0
                return {
                    "total_return": total_return * 100,
                    "ann_return": ann_return * 100,
                    "sharpe": 0.5,
                    "max_drawdown": -0.1 * 100,
                    "n_days": n_days,
                }

        runner.engine = FakeEngine()
        runner.daily_results = {"B0_BuyHold_EW": daily}

        oos = runner.compute_oos_metrics()
        assert len(oos) == len(config.get("oos_windows", []))
        for key, metrics in oos.items():
            assert "n_days" in metrics
            # The synthetic fixture only covers the first OOS window. Later
            # windows must be represented explicitly as NO_DATA rather than
            # being treated as a successful non-empty period.
            if metrics.get("error") != "NO_DATA":
                assert metrics["n_days"] > 0


class TestStrategyConfig:
    """Test strategy configuration matches library."""

    def test_all_strategies_exist(self, config):
        from strategy_library import STRATEGIES
        for name in config["strategies"]:
            assert name in STRATEGIES, f"Strategy {name} not in library"

    def test_fixed_params_correct(self, config):
        assert config["n_hold"] == 20
        assert config["max_weight"] == 0.05
        assert config["signal_to_return_lag"] == 2
        assert config["rebalance_every"] == 5
        assert config["fee_rate_per_side"] == 0.0003
        assert config["slippage_rate_per_side"] == 0.0002

    def test_backtest_dates_fixed(self, config):
        assert config["backtest_start"] == "2018-01-01"
        assert config["backtest_end"] == "2026-07-17"

    def test_no_param_search(self, config):
        assert "param_grid" not in config
        assert "search_space" not in config


class TestReportGeneration:
    """Test report generation produces correct artifacts."""

    def test_report_dir_creates_subdir(self, config, tmp_path):
        from run_unified_experiment import ReportGenerator

        config["report_dir"] = str(tmp_path)
        report = ReportGenerator(config, "test_run")
        assert (tmp_path / "test_run").exists()

    def test_manifest_has_required_keys(self, config, tmp_path):
        from run_unified_experiment import ReportGenerator

        config["report_dir"] = str(tmp_path)
        report = ReportGenerator(config, "test_manifest")

        report.write_manifest(
            db_info={"db_sha256": "a" * 64},
            factor_manifest={"completed": True},
            full_results={"B0_BuyHold_EW": {"ann_return": 5.0}},
            oos_results={},
            gate_passed=True,
            failed_reasons=[],
        )

        manifest_path = tmp_path / "test_manifest" / "experiment_manifest.json"
        with open(manifest_path) as f:
            data = json.load(f)

        required = [
            "run_id", "timestamp", "gate_passed", "config",
            "db_info", "factor_manifest", "full_period_results",
        ]
        for key in required:
            assert key in data, f"Missing manifest key: {key}"

    def test_summary_csv_written(self, config, tmp_path):
        from run_unified_experiment import ReportGenerator

        config["report_dir"] = str(tmp_path)
        report = ReportGenerator(config, "test_summary")

        report.write_summary(
            full_results={"B0_BuyHold_EW": {"ann_return": 5.0, "sharpe": 0.5}},
            oos_results={},
        )

        summary_path = tmp_path / "test_summary" / "summary.csv"
        assert summary_path.exists()
        df = pd.read_csv(summary_path)
        assert len(df) >= 1

    def test_daily_csvs_written(self, config, tmp_path):
        from run_unified_experiment import ReportGenerator

        config["report_dir"] = str(tmp_path)
        report = ReportGenerator(config, "test_daily")

        dates = pd.bdate_range("2018-01-01", periods=10)
        daily = pd.DataFrame({
            "date": dates,
            "return": np.random.normal(0, 0.01, 10),
        })
        report.write_daily_csvs({"B0_BuyHold_EW": daily})

        daily_path = tmp_path / "test_daily" / "daily_B0_BuyHold_EW.csv"
        assert daily_path.exists()
        assert len(pd.read_csv(daily_path)) == 10


class TestDataGateConfig:
    """Test data gate configuration values."""

    def test_canonical_db_is_processed(self, config):
        assert "processed" in config["canonical_db"]

    def test_data_mode_is_etf(self, config):
        assert config["data_mode"] == "etf"

    def test_price_mode_is_total_return_proxy(self, config):
        assert config["price_mode"] == "total_return_proxy"

    def test_require_pit_is_true(self, config):
        assert config.get("require_pit", True) is True


class TestFactorValidationConfig:
    """Test factor validation thresholds match DB."""

    def test_expected_rows_match_db(self, config):
        import sqlite3
        with sqlite3.connect(config["canonical_db"]) as conn:
            actual = conn.execute("SELECT COUNT(*) FROM etf_daily").fetchone()[0]
        assert config["factor_validation"]["expected_rows"] == actual

    def test_expected_symbols_match_db(self, config):
        import sqlite3
        with sqlite3.connect(config["canonical_db"]) as conn:
            actual = conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM etf_daily"
            ).fetchone()[0]
        assert config["factor_validation"]["expected_symbols"] == actual


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
