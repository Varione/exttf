"""Unified experiment runner with data gates, factor rebuild, and reproducible backtesting."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger("unified_experiment")


def get_git_info() -> dict:
    """Get current git commit hash and clean/dirty status."""
    info = {"git_commit": "unknown", "git_status": "unknown"}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            info["git_commit"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    try:
        result = subprocess.run(
            ["git", "diff-index", "--quiet", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            info["git_status"] = "clean"
        else:
            info["git_status"] = "dirty"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return info


def compute_code_fingerprint(src_dir: str = "src") -> str:
    """Compute SHA256 of all .py files in src directory sorted by path."""
    h = hashlib.sha256()
    py_files = sorted(Path(src_dir).rglob("*.py"))
    for py_file in py_files:
        try:
            content = py_file.read_bytes()
            h.update(py_file.relative_to(src_dir).as_posix().encode())
            h.update(content)
        except (OSError, PermissionError):
            pass
    return h.hexdigest()


def get_dependency_summary() -> dict[str, str]:
    """Get version summary of key dependencies."""
    deps = {}
    for pkg in ("pandas", "numpy", "scipy", "sklearn", "torch"):
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[pkg] = "not-installed"
    return deps


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_db_schema(db_path: str) -> dict[str, list[str]]:
    schema = {}
    with sqlite3.connect(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        for (tname,) in tables:
            cols = conn.execute(f"PRAGMA table_info([{tname}])").fetchall()
            schema[tname] = [c[1] for c in cols]
    return schema


def get_db_table_counts(db_path: str) -> dict[str, int]:
    counts = {}
    with sqlite3.connect(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        for (tname,) in tables:
            counts[tname] = conn.execute(
                f"SELECT COUNT(*) FROM [{tname}]"
            ).fetchone()[0]
    return counts


def compare_dbs(canonical: str, root: str) -> dict:
    canon_counts = get_db_table_counts(canonical)
    root_counts = get_db_table_counts(root)
    diffs = {}
    all_tables = set(canon_counts.keys()) | set(root_counts.keys())
    for t in sorted(all_tables):
        c = canon_counts.get(t, 0)
        r = root_counts.get(t, 0)
        if c != r:
            diffs[t] = {"canonical": c, "root": r}
    return diffs


class DataGate:
    """Fail-closed data gate for canonical database."""

    def __init__(self, config: dict):
        self.config = config
        self.db_path = config["canonical_db"]
        self.errors: list[str] = []
        self.manifest_data: dict = {}

    def check_canonical_db_lock(self) -> bool:
        if not os.path.exists(self.db_path):
            self.errors.append(f"CANONICAL_DB_MISSING: {self.db_path}")
            return False

        sha256 = compute_file_sha256(self.db_path)
        schema = get_db_schema(self.db_path)
        counts = get_db_table_counts(self.db_path)
        self.manifest_data["db_sha256"] = sha256
        self.manifest_data["db_schema"] = {k: v for k, v in schema.items()}
        self.manifest_data["db_table_counts"] = counts

        required_tables = {"etf_daily", "etf_daily_price_modes"}
        missing = required_tables - set(schema.keys())
        if missing:
            self.errors.append(f"MISSING_TABLES: {sorted(missing)}")
            return False

        if counts.get("etf_daily", 0) == 0:
            self.errors.append("EMPTY_ETF_DAILY")
            return False

        logger.info(
            f"Canonical DB locked: {self.db_path} "
            f"SHA256={sha256[:16]}... "
            f"etf_daily={counts['etf_daily']} rows, "
            f"{counts.get('etf_catalog', 0)} symbols"
        )
        return True

    def check_root_db_diff(self) -> bool:
        root_path = "etf.sqlite"
        if os.path.exists(root_path):
            diffs = compare_dbs(self.db_path, root_path)
            root_sha = compute_file_sha256(root_path)
            self.manifest_data["root_db_sha256"] = root_sha
            self.manifest_data["root_db_diffs"] = diffs
            logger.info(f"Root DB differs from canonical: {diffs}")
        return True

    def check_data_mode(self) -> bool:
        required_mode = self.config.get("data_mode", "etf")
        if required_mode != "etf":
            self.errors.append(f"UNSUPPORTED_DATA_MODE: {required_mode}")
            return False
        logger.info(f"data_mode={required_mode} OK")
        return True

    def check_price_mode(self) -> bool:
        required_mode = self.config.get("price_mode", "total_return_proxy")
        with sqlite3.connect(self.db_path) as conn:
            tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        if required_mode != "raw_close" and "etf_daily_price_modes" not in tables:
            self.errors.append("PRICE_MODE_TABLE_MISSING")
            return False
        if required_mode != "raw_close":
            with sqlite3.connect(self.db_path) as conn:
                columns = {
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(etf_daily_price_modes)"
                    ).fetchall()
                }
            required_columns = {"symbol", "date", required_mode}
            missing = required_columns - columns
            if missing:
                self.errors.append(
                    f"PRICE_MODE_COLUMN_MISSING: {sorted(missing)}"
                )
                return False
        logger.info(f"price_mode={required_mode} OK")
        return True

    def check_pit(self) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            lifecycle_columns = (
                {
                    row[1]
                    for row in conn.execute(
                        "PRAGMA table_info(etf_lifecycle)"
                    ).fetchall()
                }
                if "etf_lifecycle" in tables
                else set()
            )
            lifecycle_complete = False
            if {"source_independent", "universe_scope"}.issubset(
                lifecycle_columns
            ):
                incomplete = conn.execute(
                    """
                    SELECT COUNT(*) FROM etf_lifecycle
                    WHERE COALESCE(source_independent, 0) != 1
                       OR universe_scope != 'historical_including_inactive'
                    """
                ).fetchone()[0]
                lifecycle_complete = incomplete == 0
        lifecycle_candidates = {
            "etf_lifecycle",
            "etf_listing_history",
            "etf_termination",
            "etf_delisting",
        }
        lifecycle_available = bool(tables & lifecycle_candidates)
        self.manifest_data["historical_lifecycle_records_available"] = lifecycle_available
        self.manifest_data["historical_lifecycle_independently_verified"] = (
            lifecycle_complete
        )
        self.manifest_data["pit_status"] = (
            "PIT_COMPLETE" if lifecycle_complete else "PIT_PARTIAL"
        )
        self.manifest_data["pit_validation_scope"] = (
            "independent_history_including_inactive"
            if lifecycle_complete
            else "current_catalog_with_observed_price_lifecycle"
            if lifecycle_available
            else "history_and_liquidity_only"
        )
        if self.config.get("require_full_pit", False) and not lifecycle_complete:
            self.errors.append("FULL_PIT_LIFECYCLE_NOT_INDEPENDENT_OR_INCOMPLETE")
            return False
        if self.config.get("require_pit", True):
            logger.info(
                "PIT requirement: enabled; status=%s", self.manifest_data["pit_status"]
            )
        return True

    def check_reference_verification(self) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            total = conn.execute(
                "SELECT COUNT(*) FROM etf_daily_price_modes"
            ).fetchone()[0]
            verified = conn.execute(
                "SELECT COUNT(*) FROM etf_daily_price_modes WHERE validation_status='PASS'"
            ).fetchone()[0]
        ratio = verified / total if total > 0 else 0.0
        self.manifest_data["reference_verification_ratio"] = round(ratio, 4)
        # The current price-mode table is internally consistent but is not an
        # independent fund-company or exchange reference feed.
        self.manifest_data["reference_source_independent"] = False
        self.manifest_data["reference_validation_scope"] = "self_consistency_only"
        if "etf_daily_external_reference" in tables:
            with sqlite3.connect(self.db_path) as conn:
                ext = conn.execute(
                    """
                    SELECT COUNT(*), COUNT(DISTINCT symbol),
                           COALESCE(SUM(CASE WHEN source_independent=1 THEN 1 ELSE 0 END), 0),
                           COALESCE(SUM(CASE WHEN official_or_exchange=1 THEN 1 ELSE 0 END), 0),
                           COUNT(DISTINCT CASE WHEN source_independent=1 THEN symbol END),
                           COUNT(DISTINCT CASE WHEN official_or_exchange=1 THEN symbol END)
                    FROM etf_daily_external_reference
                    """
                ).fetchone()
            (
                ext_rows,
                ext_symbols,
                independent_rows,
                official_rows,
                independent_symbols,
                official_symbols,
            ) = ext
            self.manifest_data.update(
                {
                    "independent_reference_sample_rows": int(independent_rows),
                    "independent_reference_sample_symbols": int(independent_symbols),
                    "official_reference_sample_rows": int(official_rows),
                    "official_reference_sample_symbols": int(official_symbols),
                    "independent_reference_sample_coverage": round(
                        int(independent_rows) / total if total else 0.0, 6
                    ),
                }
            )
            self.manifest_data["reference_validation_scope"] = (
                "independent_sample_only"
            )
        logger.info(
            f"Reference verification: {verified}/{total} "
            f"({ratio:.1%})"
        )
        return True

    def check_otf_data(self) -> bool:
        """Fail closed when configured OTC NAV inputs are incomplete."""
        if not self.config.get("otf_strategies"):
            return True
        otf = self.config.get("otf_data", {})
        db_path = otf.get("canonical_db", "data/processed/otf.sqlite")
        gate_path = otf.get(
            "data_gate_report", "reports/data_validation/otf_data_gate.json"
        )
        manifest_path = otf.get("manifest")
        if not os.path.exists(db_path):
            self.errors.append(f"OTF_CANONICAL_DB_MISSING: {db_path}")
            return False
        if not os.path.exists(gate_path):
            self.errors.append(f"OTF_DATA_GATE_REPORT_MISSING: {gate_path}")
            return False
        with open(gate_path, "r", encoding="utf-8") as handle:
            gate = json.load(handle)
        if not gate.get("data_gate_passed", False):
            self.errors.append("OTF_DATA_GATE_FAILED")
            return False
        schema = get_db_schema(db_path)
        counts = get_db_table_counts(db_path)
        required_tables = {"otf_fund_catalog", "otf_fund_nav"}
        if required_tables - set(schema):
            self.errors.append("OTF_REQUIRED_TABLES_MISSING")
            return False
        required_nav_columns = {
            "fund_code", "nav_date", "unit_nav", "cumulative_nav",
            "daily_growth_pct", "share_adjustment_factor",
            "total_return_factor",
        }
        if required_nav_columns - set(schema["otf_fund_nav"]):
            self.errors.append("OTF_NAV_SCHEMA_INCOMPLETE")
            return False
        min_funds = int(otf.get("minimum_fund_count", 18))
        fund_count = int(counts.get("otf_fund_catalog", 0))
        if fund_count < min_funds:
            self.errors.append(
                f"OTF_FUND_COUNT_BELOW_MINIMUM: {fund_count} < {min_funds}"
            )
            return False
        if manifest_path and os.path.exists(manifest_path):
            manifest = pd.read_csv(manifest_path, dtype={"fund_code": str})
            failures = manifest.loc[manifest["status"] != "success"]
            if not failures.empty:
                self.errors.append(
                    f"OTF_FETCH_FAILURES_PRESENT: {len(failures)}"
                )
                return False
        self.manifest_data["otf_data"] = {
            "db_path": db_path,
            "db_sha256": compute_file_sha256(db_path),
            "table_counts": counts,
            "fund_count": fund_count,
            "nav_row_count": int(counts.get("otf_fund_nav", 0)),
            "price_mode": gate.get("price_mode"),
            "source_independent": bool(gate.get("source_independent", False)),
            "quality_gate": gate,
        }
        return True

    def run_all(self) -> bool:
        checks = [
            self.check_canonical_db_lock,
            self.check_root_db_diff,
            self.check_data_mode,
            self.check_price_mode,
            self.check_pit,
            self.check_reference_verification,
            self.check_otf_data,
        ]
        for check in checks:
            if not check():
                return False
        return len(self.errors) == 0


class FactorBuilder:
    """Rebuild factors from canonical DB with strict validation."""

    def __init__(self, config: dict):
        self.config = config
        self.db_path = config["canonical_db"]
        self.csv_path = config["factor_csv"]
        self.price_mode = config.get("price_mode", "total_return_proxy")
        self.expected_rows = config["factor_validation"]["expected_rows"]
        self.expected_symbols = config["factor_validation"]["expected_symbols"]
        self.expected_columns = config["factor_validation"]["expected_columns"]

    def rebuild(self) -> bool:
        from factor_engine import compute_all_factors, _compute_db_fingerprint
        from data_loader import get_valid_symbols

        logger.info("=== FACTOR REBUILD ===")
        logger.info(f"DB: {self.db_path}")
        logger.info(
            f"Expected: {self.expected_rows} rows, "
            f"{self.expected_symbols} symbols, "
            f"{self.expected_columns} columns"
        )

        t0 = time.time()
        symbols = get_valid_symbols(self.db_path)
        logger.info(f"Valid symbols from DB: {len(symbols)}")

        try:
            compute_all_factors(
                db_path=self.db_path,
                output_path=self.csv_path,
                symbols=symbols,
                price_mode=self.price_mode,
                resume_from_checkpoint=False,
            )
        except Exception as exc:
            logger.exception("Factor rebuild raised an exception")
            self.error = f"FACTOR_REBUILD_EXCEPTION: {type(exc).__name__}: {exc}"
            return False
        elapsed = time.time() - t0
        logger.info(f"Factor rebuild completed in {elapsed:.1f}s")

        return self.validate()

    def validate(self) -> bool:
        manifest_path = self.csv_path + ".manifest.json"
        if not os.path.exists(self.csv_path):
            logger.error("FACTOR_CSV_MISSING after rebuild")
            return False

        if not os.path.exists(manifest_path):
            logger.error("FACTOR_MANIFEST_MISSING after rebuild")
            return False

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        errors = []
        if not manifest.get("completed", False):
            errors.append("MANIFEST_NOT_COMPLETED")

        from factor_engine import _compute_db_fingerprint
        current_db_fingerprint = _compute_db_fingerprint(self.db_path)
        manifest_db_fingerprint = manifest.get(
            "source_db_fingerprint", manifest.get("db_fingerprint")
        )
        if manifest_db_fingerprint != current_db_fingerprint:
            errors.append(
                "FACTOR_DB_FINGERPRINT_MISMATCH: "
                f"manifest={manifest_db_fingerprint}, current={current_db_fingerprint}"
            )

        actual_rows = manifest.get("row_count", 0)
        if actual_rows != self.expected_rows:
            errors.append(
                f"ROW_COUNT_MISMATCH: expected {self.expected_rows}, "
                f"got {actual_rows}"
            )

        actual_symbols = manifest.get("symbol_count", 0)
        if actual_symbols != self.expected_symbols:
            errors.append(
                f"SYMBOL_COUNT_MISMATCH: expected {self.expected_symbols}, "
                f"got {actual_symbols}"
            )

        header = pd.read_csv(self.csv_path, nrows=0)
        actual_cols = len(header.columns)
        if actual_cols != self.expected_columns:
            errors.append(
                f"COLUMN_COUNT_MISMATCH: expected {self.expected_columns}, "
                f"got {actual_cols}"
            )

        if errors:
            for e in errors:
                logger.error(f"Factor validation: {e}")
            return False

        db_fingerprint = _compute_db_fingerprint(self.db_path)
        logger.info(
            f"Factor validation PASSED: "
            f"{actual_rows} rows, {actual_symbols} symbols, "
            f"{actual_cols} columns, DB fingerprint={db_fingerprint}"
        )
        self.manifest = manifest
        return True

    def get_manifest(self) -> dict:
        if not hasattr(self, "manifest"):
            manifest_path = self.csv_path + ".manifest.json"
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    self.manifest = json.load(f)
            else:
                self.manifest = {}
        return self.manifest


class RegimeBuilder:
    """Rebuild regime labels and predictions from fresh factors."""

    def __init__(self, config: dict):
        self.config = config
        self.csv_path = config["factor_csv"]
        self.regime_path = config.get(
            "regime_predictions_csv",
            "data/processed/regime_predictions.csv",
        )

    def rebuild(self) -> bool:
        from detect_regimes import (
            load_factors,
            get_selected_factors,
            compute_daily_factors,
            detect_regimes_train_only,
        )
        from factor_engine import compute_correlation_matrix

        logger.info("=== REGIME REBUILD ===")
        t0 = time.time()

        df = load_factors(self.csv_path)

        # The correlation matrix is a model-selection input. It must be fit on
        # the pre-backtest training sample, not on the full period.
        train_end = pd.Timestamp("2018-01-01")
        training_df = df.loc[df["date"] < train_end].copy()
        if training_df.empty:
            logger.error("REGIME_TRAINING_EMPTY")
            return False

        # Recompute correlation matrix from training-only factors
        logger.info("Recomputing factor correlation matrix from fresh factors...")
        corr_df = compute_correlation_matrix(factors_df=training_df, top_n=100)
        corr_df.to_csv("data/processed/factor_correlation.csv", index=True)
        logger.info("Factor correlation matrix saved")

        factor_cols = get_selected_factors(0.5)
        logger.info(f"Selected {len(factor_cols)} uncorrelated factors for regime detection")

        daily_means = compute_daily_factors(df, factor_cols)
        regime_series, _, scaler, kmeans = detect_regimes_train_only(
            daily_means, train_end="2018-01-01", n_clusters=3
        )

        predictions_df = pd.DataFrame({
            "date": regime_series.index.strftime("%Y-%m-%d"),
            "regime": regime_series.values,
        })
        predictions_df.to_csv(self.regime_path, index=False)

        elapsed = time.time() - t0
        logger.info(
            f"Regime rebuild completed in {elapsed:.1f}s: "
            f"{len(predictions_df)} dates saved to {self.regime_path}"
        )
        return True


class BacktestRunner:
    """Run fixed-strategy backtests with OOS window metrics."""

    def __init__(self, config: dict):
        self.config = config
        self.db_path = config["canonical_db"]
        self.csv_path = config["factor_csv"]
        self.regime_path = config.get(
            "regime_predictions_csv",
            "data/processed/regime_predictions.csv",
        )

    def run(self) -> dict[str, pd.DataFrame]:
        from backtest_engine import BacktestEngine

        logger.info("=== BACKTEST EXECUTION ===")
        engine = BacktestEngine(
            factor_path=self.csv_path,
            regime_path=self.regime_path,
            db_path=self.db_path,
            data_mode=self.config["data_mode"],
            price_mode=self.config.get("price_mode", "total_return_proxy"),
            fee_rate_per_side=self.config.get("fee_rate_per_side", 0.0003),
            slippage_rate_per_side=self.config.get("slippage_rate_per_side", 0.0002),
            require_pit=self.config.get("require_pit", True),
            require_full_pit=self.config.get("require_full_pit", False),
        )

        results = {}
        daily_results = {}
        strategies = self.config["strategies"]

        for name in strategies:
            logger.info(f"Running strategy: {name}")
            t0 = time.time()

            daily = engine.run_backtest(
                strategy_name=name,
                start=self.config["backtest_start"],
                end=self.config["backtest_end"],
                n_hold=self.config["n_hold"],
                max_weight=self.config["max_weight"],
                signal_to_return_lag=self.config.get("signal_to_return_lag", 2),
                rebalance_every=self.config.get("rebalance_every", 5),
            )

            metrics = engine.calculate_metrics(
                daily["return"],
                daily["turnover"],
                daily["transaction_cost"],
                daily["gross_return"],
                daily["exposure"],
            )
            metrics.update({
                "strategy": name,
                "period": "FULL",
                "start_date": self.config["backtest_start"],
                "end_date": self.config["backtest_end"],
                "n_days": len(daily),
                "pit_status": engine.pit_status,
                "data_mode": engine.data_mode,
                "price_mode": engine.price_mode,
            })

            results[name] = metrics
            daily_results[name] = daily
            logger.info(
                f"  {name}: CAGR={metrics.get('ann_return', 0):.2f}%, "
                f"Sharpe={metrics.get('sharpe', 0):.3f}, "
                f"MDD={metrics.get('max_drawdown', 0):.2f}%, "
                f"time={time.time()-t0:.1f}s"
            )

        self.engine = engine
        self.results = results
        self.daily_results = daily_results
        return results

    def compute_oos_metrics(self) -> dict[str, dict]:
        oos_windows = self.config.get("oos_windows", [])
        all_oos: dict[str, dict] = {}

        for window in oos_windows:
            window_name = window["name"]
            logger.info(f"OOS window: {window_name} ({window['start']} ~ {window['end']})")

            # Use the completed daily result set. This keeps the reporting
            # helper robust when a caller supplies a reduced test fixture,
            # while the production runner still contains every configured
            # strategy.
            for name, daily in self.daily_results.items():
                mask = (
                    (daily["date"] >= pd.Timestamp(window["start"]))
                    & (daily["date"] <= pd.Timestamp(window["end"]))
                )
                window_daily = daily.loc[mask]

                if len(window_daily) == 0:
                    all_oos[f"{name}_{window_name}"] = {
                        "strategy": name,
                        "period": window_name,
                        "start_date": window["start"],
                        "end_date": window["end"],
                        "n_days": 0,
                        "error": "NO_DATA",
                    }
                    continue

                metrics = self.engine.calculate_metrics(
                    window_daily["return"],
                    window_daily["turnover"],
                    window_daily["transaction_cost"],
                    window_daily["gross_return"],
                    window_daily["exposure"],
                )
                metrics.update({
                    "strategy": name,
                    "period": window_name,
                    "start_date": window["start"],
                    "end_date": window["end"],
                    "n_days": len(window_daily),
                    "pit_status": self.engine.pit_status,
                    "data_mode": self.engine.data_mode,
                    "price_mode": self.engine.price_mode,
                })
                all_oos[f"{name}_{window_name}"] = metrics

        return all_oos


class OTFBacktestRunner:
    """Run OTC fund NAV-based backtests."""

    def __init__(self, config: dict):
        self.config = config
        otf_config = config.get("otf_data", {})
        self.otf_db_path = otf_config.get(
            "canonical_db", "data/processed/otf.sqlite"
        )
        self.otf_strategies = config.get("otf_strategies", [])

    def run(self) -> dict[str, dict]:
        from otf_backtest_engine import (
            OTFBacktestEngine,
            OTFStrategySignal,
        )

        logger.info("=== OTF BACKTEST EXECUTION ===")
        otf_config = self.config.get("otf_data", {})

        engine = OTFBacktestEngine(
            db_path=self.otf_db_path,
            confirmation_days_subscribe=otf_config.get(
                "confirmation_days_subscribe", 1
            ),
            confirmation_days_redeem=otf_config.get(
                "confirmation_days_redeem", 1
            ),
            settlement_days_redeem=otf_config.get(
                "settlement_days_redeem", 1
            ),
            subscription_fee_rate=otf_config.get(
                "fee_rate_subscription", 0.001
            ),
            redemption_fee_rate=otf_config.get(
                "fee_rate_redemption", 0.0015
            ),
            initial_cash=self.config.get("initial_cash", 1_000_000.0),
        )

        signal = OTFStrategySignal(engine)
        results: dict[str, dict] = {}
        daily_results: dict[str, pd.DataFrame] = {}

        for strategy_name in self.otf_strategies:
            logger.info(f"Running OTF strategy: {strategy_name}")
            t0 = time.time()

            if strategy_name == "OTF_EW":
                sig_func = lambda d: signal.equal_weight_signal()
            elif strategy_name == "OTF_Momentum":
                sig_func = lambda d: signal.momentum_signal(d, n_hold=5)
            else:
                logger.warning(f"Unknown OTF strategy: {strategy_name}")
                continue

            target_weights = signal.generate_target_weights(
                sig_func,
                start=self.config.get("backtest_start", "2018-01-01"),
                end=self.config.get("backtest_end", "2026-07-17"),
            )

            daily = engine.run_backtest(
                target_weights,
                start=self.config.get("backtest_start", "2018-01-01"),
                end=self.config.get("backtest_end", "2026-07-17"),
                rebalance_every=self.config.get("rebalance_every", 5),
            )

            metrics = engine.calculate_metrics(daily)
            metrics.update({
                "strategy": strategy_name,
                "mode": "otf_nav",
                "period": "FULL",
                "start_date": self.config.get("backtest_start", "2018-01-01"),
                "end_date": self.config.get("backtest_end", "2026-07-17"),
                "n_days": len(daily),
                "confirmation_rule": otf_config.get(
                    "confirmation_rule", "T+1"
                ),
            })

            results[strategy_name] = metrics
            daily_results[strategy_name] = daily
            logger.info(
                f"  {strategy_name}: CAGR={metrics.get('CAGR%', 0):.2f}%, "
                f"Sharpe={metrics.get('Sharpe', 0):.3f}, "
                f"MDD={metrics.get('Max_Drawdown%', 0):.2f}%, "
                f"time={time.time()-t0:.1f}s"
            )

        self.engine = engine
        self.results = results
        self.daily_results = daily_results
        return results


class ReportGenerator:
    """Generate experiment reports with manifest."""

    def __init__(self, config: dict, run_id: str):
        self.config = config
        self.run_id = run_id
        self.report_dir = Path(config["report_dir"]) / run_id
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(
        self,
        db_info: dict,
        factor_manifest: dict,
        full_results: dict[str, dict],
        oos_results: dict[str, dict],
        gate_passed: bool,
        failed_reasons: list[str],
    ) -> None:
        manifest = {
            "run_id": self.run_id,
            "timestamp": datetime.now().astimezone().isoformat(),
            "gate_passed": gate_passed,
            "failed_reasons": failed_reasons,
            "config": self.config,
            "db_info": db_info,
            "factor_manifest": factor_manifest,
            "full_period_results": {},
            "oos_results": {},
            "artifacts": {},
            "regime_training_end": "2018-01-01",
            "python_version": sys.version,
            "dependencies": get_dependency_summary(),
            **get_git_info(),
            "code_fingerprint": compute_code_fingerprint(),
        }

        for raw_path in (
            self.config.get("factor_csv"),
            f"{self.config.get('factor_csv')}.manifest.json",
            self.config.get("regime_predictions_csv"),
            "data/processed/factor_correlation.csv",
        ):
            if raw_path and os.path.exists(raw_path):
                manifest["artifacts"][raw_path] = {
                    "sha256": compute_file_sha256(raw_path),
                    "size_bytes": os.path.getsize(raw_path),
                }

        for name, metrics in full_results.items():
            manifest["full_period_results"][name] = {k: self._serialize(v) for k, v in metrics.items()}

        for key, metrics in oos_results.items():
            manifest["oos_results"][key] = {k: self._serialize(v) for k, v in metrics.items()}

        manifest_path = self.report_dir / "experiment_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Manifest written to {manifest_path}")

    def write_summary(self, full_results: dict, oos_results: dict) -> None:
        rows = []
        for name, metrics in full_results.items():
            row = {"strategy": name, "period": "FULL"}
            for k, v in metrics.items():
                if k not in ("strategy",):
                    row[k] = self._serialize(v)
            rows.append(row)

        for key, metrics in oos_results.items():
            row = {}
            for k, v in metrics.items():
                row[k] = self._serialize(v)
            rows.append(row)

        summary_df = pd.DataFrame(rows)
        summary_path = self.report_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary written to {summary_path}")

    def write_daily_csvs(self, daily_results: dict[str, pd.DataFrame]) -> None:
        for name, daily in daily_results.items():
            safe_name = name.replace(" ", "_").replace("/", "_")
            path = self.report_dir / f"daily_{safe_name}.csv"
            daily.to_csv(path, index=False)
            logger.info(f"Daily CSV: {path} ({len(daily)} rows)")

    def write_run_log(self, log_lines: list[str]) -> None:
        log_path = self.report_dir / "run.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        logger.info(f"Run log written to {log_path}")

    @staticmethod
    def _serialize(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v


def run_experiment():
    log_lines: list[str] = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    log("=" * 60)
    log("UNIFIED ETF STRATEGY EXPERIMENT")
    log("=" * 60)
    log(f"Python: {sys.version}")
    git_info = get_git_info()
    log(f"Git commit: {git_info['git_commit']} ({git_info['git_status']})")
    log(f"Code fingerprint: {compute_code_fingerprint()}")

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "unified_experiment.json",
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    gate_passed = True
    failed_reasons: list[str] = []

    # Phase 1: Data Gate
    log("\n--- PHASE 1: DATA GATE ---")
    t0 = time.time()
    gate = DataGate(config)
    gate_ok = gate.run_all()
    if not gate_ok:
        gate_passed = False
        failed_reasons.extend(gate.errors)
        log(f"DATA GATE FAILED: {gate.errors}")
    else:
        log("DATA GATE PASSED")
    log(f"Phase 1 time: {time.time()-t0:.1f}s")

    if not gate_ok:
        report = ReportGenerator(config, run_id)
        report.write_manifest(
            db_info=gate.manifest_data,
            factor_manifest={},
            full_results={},
            oos_results={},
            gate_passed=False,
            failed_reasons=failed_reasons,
        )
        (report.report_dir / "FAILED").write_text(
            "EXPERIMENT FAILED - data gate did not pass\n"
            + "\n".join(failed_reasons),
            encoding="utf-8",
        )
        report.write_run_log(log_lines)
        log("FAILED manifest written; factor and backtest phases suppressed")
        sys.exit(1)

    # Phase 2: Factor Rebuild (or validate an already-completed artifact).
    log("\n--- PHASE 2: FACTOR REBUILD ---")
    t0 = time.time()
    factor_builder = FactorBuilder(config)
    if "--skip-factor-rebuild" in sys.argv:
        log("Existing factor artifact requested; validating without recomputation")
        factor_ok = factor_builder.validate()
    else:
        factor_ok = factor_builder.rebuild()
    if not factor_ok:
        gate_passed = False
        failed_reasons.append(getattr(factor_builder, "error", "FACTOR_REBUILD_FAILED"))
        log("FACTOR REBUILD FAILED - aborting experiment")
        report = ReportGenerator(config, run_id)
        report.write_manifest(
            db_info=gate.manifest_data,
            factor_manifest={},
            full_results={},
            oos_results={},
            gate_passed=False,
            failed_reasons=failed_reasons,
        )
        report.write_run_log(log_lines)
        log("FAILED manifest written. Experiment aborted.")
        sys.exit(1)
    factor_manifest = factor_builder.get_manifest()
    log(f"Phase 2 time: {time.time()-t0:.1f}s")

    # Phase 3: Regime Rebuild
    log("\n--- PHASE 3: REGIME REBUILD ---")
    t0 = time.time()
    regime_builder = RegimeBuilder(config)
    try:
        regime_ok = regime_builder.rebuild()
    except Exception as exc:
        logger.exception("Regime rebuild raised an exception")
        regime_builder.error = (
            f"REGIME_REBUILD_EXCEPTION: {type(exc).__name__}: {exc}"
        )
        regime_ok = False
    if not regime_ok:
        gate_passed = False
        failed_reasons.append(
            getattr(regime_builder, "error", "REGIME_REBUILD_FAILED")
        )
        log("REGIME REBUILD FAILED - aborting experiment")
        report = ReportGenerator(config, run_id)
        report.write_manifest(
            db_info=gate.manifest_data,
            factor_manifest=factor_manifest,
            full_results={},
            oos_results={},
            gate_passed=False,
            failed_reasons=failed_reasons,
        )
        (report.report_dir / "FAILED").write_text(
            "EXPERIMENT FAILED - regime rebuild did not pass\n"
            + "\n".join(failed_reasons),
            encoding="utf-8",
        )
        report.write_run_log(log_lines)
        sys.exit(1)
    log(f"Phase 3 time: {time.time()-t0:.1f}s")

    # Phase 4: Backtest Execution (ETF)
    log("\n--- PHASE 4: BACKTEST EXECUTION (ETF) ---")
    t0 = time.time()
    bt_runner = BacktestRunner(config)
    full_results = bt_runner.run()
    oos_results = bt_runner.compute_oos_metrics()
    log(f"Phase 4 time: {time.time()-t0:.1f}s")

    # Phase 4b: OTF Backtest Execution (if configured)
    otf_results: dict[str, dict] = {}
    otf_daily_results: dict[str, pd.DataFrame] = {}
    if config.get("otf_strategies"):
        log("\n--- PHASE 4b: BACKTEST EXECUTION (OTF NAV) ---")
        t0 = time.time()
        try:
            otf_runner = OTFBacktestRunner(config)
            otf_results = otf_runner.run()
            otf_daily_results = otf_runner.daily_results
        except Exception as exc:
            logger.exception("OTF backtest raised an exception")
            reason = f"OTF_BACKTEST_FAILED: {type(exc).__name__}: {exc}"
            gate_passed = False
            failed_reasons.append(reason)
            log(reason)
        log(f"Phase 4b time: {time.time()-t0:.1f}s")

    # Phase 5: Report Generation
    log("\n--- PHASE 5: REPORT GENERATION ---")
    t0 = time.time()
    report = ReportGenerator(config, run_id)
    all_full_results = {**full_results, **otf_results}
    report.write_manifest(
        db_info=gate.manifest_data,
        factor_manifest=factor_manifest,
        full_results=all_full_results,
        oos_results=oos_results,
        gate_passed=gate_passed,
        failed_reasons=failed_reasons,
    )
    if gate_passed:
        report.write_summary(all_full_results, oos_results)
        report.write_daily_csvs(bt_runner.daily_results)
        if otf_daily_results:
            report.write_daily_csvs(otf_daily_results)
    else:
        failed_marker = report.report_dir / "FAILED"
        failed_marker.write_text(
            "EXPERIMENT FAILED - see experiment_manifest.json for details\n"
            + "\n".join(failed_reasons),
            encoding="utf-8",
        )
        log("FAILED marker written; official summary suppressed")

    report.write_run_log(log_lines)
    log(f"Phase 5 time: {time.time()-t0:.1f}s")

    # Final summary
    log("\n" + "=" * 60)
    log("EXPERIMENT COMPLETE")
    log(f"Run ID: {run_id}")
    log(f"Report directory: {report.report_dir}")
    log(f"Gate passed: {gate_passed}")
    if failed_reasons:
        log(f"Failures: {failed_reasons}")
    log("=" * 60)


def verify_repeat_runs(
    run1_dir: str, run2_dir: str
) -> dict:
    """Compare two experiment runs for reproducibility.

    Returns a dict with match status and per-field differences.
    Excludes timestamp fields as specified in the experiment protocol.
    """
    r1_path = Path(run1_dir)
    r2_path = Path(run2_dir)

    m1_file = r1_path / "experiment_manifest.json"
    m2_file = r2_path / "experiment_manifest.json"

    if not m1_file.exists():
        return {"error": f"Manifest not found: {m1_file}"}
    if not m2_file.exists():
        return {"error": f"Manifest not found: {m2_file}"}

    with open(m1_file, "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open(m2_file, "r", encoding="utf-8") as f:
        m2 = json.load(f)

    TIMESTAMP_FIELDS = {"timestamp"}
    diffs: dict[str, dict] = {}

    def compare_dicts(d1: dict, d2: dict, prefix: str = "") -> None:
        all_keys = set(d1.keys()) | set(d2.keys())
        for key in sorted(all_keys):
            full_key = f"{prefix}.{key}" if prefix else key
            if key in TIMESTAMP_FIELDS:
                continue
            v1 = d1.get(key)
            v2 = d2.get(key)
            if v1 is None and v2 is None:
                continue
            if isinstance(v1, dict) and isinstance(v2, dict):
                compare_dicts(v1, v2, full_key)
            elif v1 != v2:
                diffs[full_key] = {"run1": v1, "run2": v2}

    compare_dicts(m1, m2)

    # Compare summary.csv if both exist
    s1_file = r1_path / "summary.csv"
    s2_file = r2_path / "summary.csv"
    if s1_file.exists() and s2_file.exists():
        s1 = pd.read_csv(s1_file)
        s2 = pd.read_csv(s2_file)
        if not s1.equals(s2):
            diffs["summary.csv"] = {
                "run1_shape": list(s1.shape),
                "run2_shape": list(s2.shape),
                "columns_match": list(s1.columns) == list(s2.columns),
            }

    # Compare artifact SHA256
    a1 = m1.get("artifacts", {})
    a2 = m2.get("artifacts", {})
    for artifact_name in sorted(set(a1.keys()) | set(a2.keys())):
        sha1 = a1.get(artifact_name, {}).get("sha256")
        sha2 = a2.get(artifact_name, {}).get("sha256")
        if sha1 != sha2:
            diffs[f"artifacts.{artifact_name}.sha256"] = {"run1": sha1, "run2": sha2}

    return {
        "run1_id": m1.get("run_id"),
        "run2_id": m2.get("run_id"),
        "reproducible": len(diffs) == 0,
        "num_differences": len(diffs),
        "differences": diffs,
    }


def main():
    log_lines: list[str] = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Check for repeat-verification mode
    if "--verify-repeat" in sys.argv:
        idx = sys.argv.index("--verify-repeat")
        if idx + 2 >= len(sys.argv):
            print(
                "Usage: run_unified_experiment.py --verify-repeat <run1_dir> <run2_dir>",
                file=sys.stderr,
            )
            sys.exit(1)
        run1_dir = sys.argv[idx + 1]
        run2_dir = sys.argv[idx + 2]
        result = verify_repeat_runs(run1_dir, run2_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        if not result["reproducible"]:
            print(
                f"REPRODUCIBILITY CHECK FAILED: {result['num_differences']} differences found",
                file=sys.stderr,
            )
            sys.exit(1)
        print("REPRODUCIBILITY CHECK PASSED: runs are identical (excluding timestamps)")
        sys.exit(0)

    # The experiment implementation lives in one place.  Keeping a second
    # copied main flow caused fixes to apply to an inactive definition.
    return run_experiment()

    log("=" * 60)
    log("UNIFIED ETF STRATEGY EXPERIMENT")
    log("=" * 60)

    # Print environment info
    log(f"Python: {sys.version}")
    git_info = get_git_info()
    log(f"Git commit: {git_info['git_commit']} ({git_info['git_status']})")
    log(f"Code fingerprint: {compute_code_fingerprint()}")
    deps = get_dependency_summary()
    log(f"Dependencies: {', '.join(f'{k}={v}' for k, v in deps.items())}")

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "unified_experiment.json",
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    gate_passed = True
    failed_reasons: list[str] = []

    # Phase 1: Data Gate
    log("\n--- PHASE 1: DATA GATE ---")
    t0 = time.time()
    gate = DataGate(config)
    gate_ok = gate.run_all()
    if not gate_ok:
        gate_passed = False
        failed_reasons.extend(gate.errors)
        log(f"DATA GATE FAILED: {gate.errors}")
    else:
        log("DATA GATE PASSED")
    log(f"Phase 1 time: {time.time()-t0:.1f}s")

    if not gate_ok:
        report = ReportGenerator(config, run_id)
        report.write_manifest(
            db_info=gate.manifest_data,
            factor_manifest={},
            full_results={},
            oos_results={},
            gate_passed=False,
            failed_reasons=failed_reasons,
        )
        (report.report_dir / "FAILED").write_text(
            "EXPERIMENT FAILED - data gate did not pass\n"
            + "\n".join(failed_reasons),
            encoding="utf-8",
        )
        report.write_run_log(log_lines)
        log("FAILED manifest written; factor and backtest phases suppressed")
        sys.exit(1)

    # Phase 2: Factor Rebuild (or validate an already-completed artifact).
    log("\n--- PHASE 2: FACTOR REBUILD ---")
    t0 = time.time()
    factor_builder = FactorBuilder(config)
    if "--skip-factor-rebuild" in sys.argv:
        log("Existing factor artifact requested; validating without recomputation")
        factor_ok = factor_builder.validate()
    else:
        factor_ok = factor_builder.rebuild()
    if not factor_ok:
        gate_passed = False
        failed_reasons.append(getattr(factor_builder, "error", "FACTOR_REBUILD_FAILED"))
        log("FACTOR REBUILD FAILED - aborting experiment")
        report = ReportGenerator(config, run_id)
        report.write_manifest(
            db_info=gate.manifest_data,
            factor_manifest={},
            full_results={},
            oos_results={},
            gate_passed=False,
            failed_reasons=failed_reasons,
        )
        report.write_run_log(log_lines)
        log("FAILED manifest written. Experiment aborted.")
        sys.exit(1)
    factor_manifest = factor_builder.get_manifest()
    log(f"Phase 2 time: {time.time()-t0:.1f}s")

    # Phase 3: Regime Rebuild
    log("\n--- PHASE 3: REGIME REBUILD ---")
    t0 = time.time()
    regime_builder = RegimeBuilder(config)
    try:
        regime_ok = regime_builder.rebuild()
    except Exception as exc:
        logger.exception("Regime rebuild raised an exception")
        regime_builder.error = (
            f"REGIME_REBUILD_EXCEPTION: {type(exc).__name__}: {exc}"
        )
        regime_ok = False
    if not regime_ok:
        gate_passed = False
        failed_reasons.append(
            getattr(regime_builder, "error", "REGIME_REBUILD_FAILED")
        )
        log("REGIME REBUILD FAILED - aborting experiment")
        report = ReportGenerator(config, run_id)
        report.write_manifest(
            db_info=gate.manifest_data,
            factor_manifest=factor_manifest,
            full_results={},
            oos_results={},
            gate_passed=False,
            failed_reasons=failed_reasons,
        )
        (report.report_dir / "FAILED").write_text(
            "EXPERIMENT FAILED - regime rebuild did not pass\n"
            + "\n".join(failed_reasons),
            encoding="utf-8",
        )
        report.write_run_log(log_lines)
        sys.exit(1)
    log(f"Phase 3 time: {time.time()-t0:.1f}s")

    # Phase 4: Backtest Execution (ETF)
    log("\n--- PHASE 4: BACKTEST EXECUTION (ETF) ---")
    t0 = time.time()
    bt_runner = BacktestRunner(config)
    full_results = bt_runner.run()
    oos_results = bt_runner.compute_oos_metrics()
    log(f"Phase 4 time: {time.time()-t0:.1f}s")

    # Phase 4b: OTF Backtest Execution (if configured)
    otf_results: dict[str, dict] = {}
    otf_daily_results: dict[str, pd.DataFrame] = {}
    if config.get("otf_strategies"):
        log("\n--- PHASE 4b: BACKTEST EXECUTION (OTF NAV) ---")
        t0 = time.time()
        try:
            otf_runner = OTFBacktestRunner(config)
            otf_results = otf_runner.run()
            otf_daily_results = otf_runner.daily_results
        except Exception as exc:
            logger.exception("OTF backtest raised an exception")
            log(f"OTF_BACKTEST_FAILED: {type(exc).__name__}: {exc}")
        log(f"Phase 4b time: {time.time()-t0:.1f}s")

    # Phase 5: Report Generation
    log("\n--- PHASE 5: REPORT GENERATION ---")
    t0 = time.time()
    report = ReportGenerator(config, run_id)
    all_full_results = {**full_results, **otf_results}
    report.write_manifest(
        db_info=gate.manifest_data,
        factor_manifest=factor_manifest,
        full_results=all_full_results,
        oos_results=oos_results,
        gate_passed=gate_passed,
        failed_reasons=failed_reasons,
    )
    if gate_passed:
        report.write_summary(all_full_results, oos_results)
        report.write_daily_csvs(bt_runner.daily_results)
        if otf_daily_results:
            report.write_daily_csvs(otf_daily_results)
    else:
        failed_marker = report.report_dir / "FAILED"
        failed_marker.write_text(
            "EXPERIMENT FAILED - see experiment_manifest.json for details\n"
            + "\n".join(failed_reasons),
            encoding="utf-8",
        )
        log("FAILED marker written; official summary suppressed")

    report.write_run_log(log_lines)
    log(f"Phase 5 time: {time.time()-t0:.1f}s")

    # Final summary
    log("\n" + "=" * 60)
    log("EXPERIMENT COMPLETE")
    log(f"Run ID: {run_id}")
    log(f"Report directory: {report.report_dir}")
    log(f"Gate passed: {gate_passed}")
    if failed_reasons:
        log(f"Failures: {failed_reasons}")
    log("=" * 60)


if __name__ == "__main__":
    main()
