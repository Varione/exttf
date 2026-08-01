"""Experiment artifact exporter for OTF strategies per P0-5.

Each run produces a timestamped directory with:
- manifest.json (Git commit, DB/config SHA256, environment)
- daily_nav.csv, daily_returns.csv
- orders.csv, order_rejections.csv
- product_selection_audit.csv
- metrics.json
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from otf_rotation.artifact_validation import artifact_status_for_strategy


def sha256_file(path: str) -> str | None:
    """Compute SHA256 hash of a file."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except (FileNotFoundError, OSError):
        return None


def gather_environment(root_dir: str = ".") -> dict[str, Any]:
    """Collect environment information for manifest."""
    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": sys.platform,
        "dependencies": {},
    }

    try:
        import pandas; env["dependencies"]["pandas"] = pandas.__version__
        import numpy; env["dependencies"]["numpy"] = numpy.__version__
    except Exception:
        pass

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root_dir, stderr=subprocess.DEVNULL
        ).decode().strip()
        env["git_commit"] = commit
        env["git_dirty"] = len(status) > 0
    except Exception:
        env["git_commit"] = None
        env["git_dirty"] = None

    return env


def create_run_directory(base_dir: str, prefix: str = "otf_experiment") -> str:
    """Create timestamped run directory."""
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{prefix}_{ts}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(
    run_dir: str,
    strategy_name: str,
    config: dict[str, Any],
    db_path: str,
    rules_path: str,
    exposure_mapping_path: str,
    metrics: dict[str, Any] | None = None,
    gate_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write manifest.json with input hashes and an artifact inventory."""
    root_dir = os.path.dirname(os.path.abspath(db_path))

    def artifact_inventory() -> dict[str, dict[str, Any]]:
        inventory: dict[str, dict[str, Any]] = {}
        for path in sorted(Path(run_dir).iterdir()):
            if not path.is_file() or path.name == "manifest.json":
                continue
            entry: dict[str, Any] = {
                "sha256": sha256_file(path.as_posix()),
                "bytes": path.stat().st_size,
                "status": "PRESENT",
                "generated_at": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
            entry["status"] = artifact_status_for_strategy(strategy_name, path.name)
            if path.suffix.lower() == ".csv":
                try:
                    frame = pd.read_csv(path, encoding="utf-8-sig")
                    entry["rows"] = int(len(frame))
                    entry["columns"] = list(frame.columns)
                except Exception as exc:
                    entry["parse_error"] = str(exc)
            elif path.suffix.lower() == ".json":
                try:
                    with path.open("r", encoding="utf-8-sig") as handle:
                        payload = json.load(handle)
                    entry["keys"] = sorted(payload) if isinstance(payload, dict) else []
                except Exception as exc:
                    entry["parse_error"] = str(exc)
            inventory[path.name] = entry
        return inventory

    manifest = {
        "run_id": config.get("run_id", Path(run_dir).name),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy_name": strategy_name,
        "artifact_schema_version": "1.0",
        "environment": gather_environment(root_dir),
        "input_hashes": {
            "db_sha256": sha256_file(db_path),
            "rules_sha256": sha256_file(rules_path),
            "exposure_mapping_sha256": sha256_file(exposure_mapping_path),
            "config_sha256": sha256_file(
                config.get("config_source_path", "config/otf_walkforward.json")
            ),
            "config_snapshot_sha256": sha256_file(
                os.path.join(run_dir, "config_snapshot.json")
            ),
        },
        "config_snapshot": config,
        "metrics": metrics or {},
        "gate_result": gate_result or {},
        "artifacts": artifact_inventory(),
    }

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    return manifest


def export_daily_nav(
    run_dir: str,
    daily_data: pd.DataFrame,
    filename: str = "daily_nav.csv",
) -> None:
    """Export daily NAV/returns to CSV."""
    path = os.path.join(run_dir, filename)
    daily_data.to_csv(path, index=False, encoding="utf-8-sig")


def export_orders(
    run_dir: str,
    orders_df: pd.DataFrame,
) -> None:
    """Export order audit frame to CSV."""
    path = os.path.join(run_dir, "orders.csv")
    orders_df.to_csv(path, index=False, encoding="utf-8-sig")


def export_rejections(
    run_dir: str,
    rejections: list[dict[str, Any]],
) -> None:
    """Export order rejections to CSV."""
    path = os.path.join(run_dir, "order_rejections.csv")
    if rejections:
        pd.DataFrame(rejections).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("fund_code,side,date,reason\n")


def export_selection_audits(
    run_dir: str,
    audits: list[dict[str, Any]],
) -> None:
    """Export product selection audit records to CSV."""
    path = os.path.join(run_dir, "product_selection_audit.csv")

    # Flatten nested candidate lists
    rows = []
    for audit in audits:
        sleeve = audit.get("sleeve", "")
        date = audit.get("date", "")
        top_n = audit.get("top_n", 0)
        selected = audit.get("selected", [])

        # Summary row
        rows.append({
            "sleeve": sleeve,
            "date": date,
            "top_n": top_n,
            "selected_count": len(selected),
            "total_candidates": len(audit.get("candidates", [])),
            "eligible_count": sum(
                1 for c in audit.get("candidates", []) if not c.get("ineligible")
            ),
            "selected_funds": ";".join(selected),
        })

    columns = [
        "sleeve", "date", "top_n", "selected_count", "total_candidates",
        "eligible_count", "selected_funds",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def export_metrics(
    run_dir: str,
    metrics: dict[str, Any],
) -> None:
    """Export metrics to JSON."""
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)


def export_json(run_dir: str, filename: str, payload: Any) -> None:
    """Export a JSON audit payload using the same deterministic formatting."""
    path = os.path.join(run_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def export_config_snapshot(run_dir: str, config: dict[str, Any]) -> None:
    """Persist the exact configuration used by a run."""
    export_json(run_dir, "config_snapshot.json", config)


def export_input_hashes(run_dir: str, hashes: dict[str, str | None]) -> None:
    """Persist input hashes separately for simple audit tooling."""
    export_json(run_dir, "input_hashes.json", hashes)


def export_position_lots(
    run_dir: str,
    lots_by_fund: dict[str, list[Any]],
) -> None:
    """Export the final FIFO lot state in a flat, re-computable table."""
    rows: list[dict[str, Any]] = []
    for fund_code, lots in lots_by_fund.items():
        for lot in lots:
            rows.append(
                {
                    "fund_code": fund_code,
                    "lot_id": getattr(lot, "lot_id", ""),
                    "acquired_date": getattr(lot, "acquired_date", None),
                    "shares": getattr(lot, "shares", None),
                    "reserved_shares": getattr(lot, "reserved_shares", None),
                }
            )
    columns = ["fund_code", "lot_id", "acquired_date", "shares", "reserved_shares"]
    pd.DataFrame(rows, columns=columns).to_csv(
        os.path.join(run_dir, "position_lots.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def export_table(run_dir: str, filename: str, frame: pd.DataFrame) -> None:
    """Export an arbitrary tabular audit artifact."""
    frame.to_csv(os.path.join(run_dir, filename), index=False, encoding="utf-8-sig")
