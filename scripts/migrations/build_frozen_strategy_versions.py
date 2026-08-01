"""Build the frozen strategy version registry from baseline freeze artifacts.

Registers every strategy that is frozen for forward shadow observation
(B2-LT, C1, C3, M20) plus the archived C2 line, with the full freeze
content required by the planning P2-1 section: strategy code commit, data
cutoff, product pool, parameters, Gate thresholds, trading rule scenario,
execution calendar, configuration hashes and parameter freeze ID.

Outputs:
- config/otf_frozen_strategy_versions.csv
- reports/historical_truth/frozen_strategy_versions_report.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "baseline_freeze_manifest.json"
OUTPUT_CSV = ROOT / "config" / "otf_frozen_strategy_versions.csv"
OUTPUT_REPORT = ROOT / "reports" / "historical_truth" / "frozen_strategy_versions_report.json"

RUN_BASE = ROOT / "reports" / "strategy_research"

COLUMNS = [
    "strategy_key",
    "strategy_name",
    "role",
    "freeze_status",
    "git_commit",
    "freeze_timestamp",
    "data_cutoff_date",
    "parameter_freeze_id",
    "trading_rule_scenario",
    "historical_rule_status",
    "execution_calendar_sha256",
    "config_sha256",
    "gate_threshold_ref",
    "product_pool",
    "run_id",
    "output_dir",
    "notes",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _product_pool_from_snapshot(config: dict[str, Any]) -> str:
    products: set[str] = set()
    core = config.get("core_weights") or {}
    products.update(str(code).zfill(6) for code in core)
    products.update(
        str(code).zfill(6) for code in (config.get("satellite_pool") or [])
    )
    d1 = config.get("products") or {}
    products.update(str(code).zfill(6) for code in d1)
    b2 = config.get("products") or {}
    products.update(str(code).zfill(6) for code in b2)
    if "b2_baseline" in config:
        products.update(str(code).zfill(6) for code in (config["b2_baseline"].get("products") or {}))
    if "b2_lt" in config:
        products.update(str(code).zfill(6) for code in (config["b2_lt"].get("products") or {}))
    return ";".join(sorted(products))


def _pool_from_run_dir(run_dir: Path) -> str:
    snapshot = run_dir / "config_snapshot.json"
    if snapshot.exists():
        pool = _product_pool_from_snapshot(_read_json(snapshot))
        if pool:
            return pool
    weights = run_dir / "actual_weights.csv"
    if weights.exists():
        import pandas as pd
        frame = pd.read_csv(weights)
        code_col = next(
            (c for c in frame.columns if "code" in c.lower() and c != "date"), None
        )
        if code_col is None and "date" in frame.columns:
            codes = [c for c in frame.columns if c != "date"]
            if codes:
                return ";".join(sorted(str(c).zfill(6) for c in codes))
        elif code_col is not None:
            return ";".join(
                sorted({str(c).zfill(6) for c in frame[code_col].dropna().unique()})
            )
    return ""


def main() -> int:
    manifest = _read_json(MANIFEST)
    frozen_runs = manifest.get("frozen_runs", {})
    data_snapshot = manifest.get("data_snapshot", {})
    freeze_timestamp = manifest.get("freeze_timestamp", "")
    git_commit = manifest.get("git_commit", "")

    rows: list[dict[str, str]] = []
    problems: list[str] = []

    def register(
        strategy_key: str,
        strategy_name: str,
        role: str,
        freeze_status: str,
        run_id: str,
        output_dir: Path,
        parameter_freeze_id: str,
        data_cutoff: str,
        rule_scenario: str,
        historical_rule_status: str,
        pool: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                "strategy_key": strategy_key,
                "strategy_name": strategy_name,
                "role": role,
                "freeze_status": freeze_status,
                "git_commit": git_commit,
                "freeze_timestamp": freeze_timestamp,
                "data_cutoff_date": data_cutoff,
                "parameter_freeze_id": parameter_freeze_id,
                "trading_rule_scenario": rule_scenario,
                "historical_rule_status": historical_rule_status,
                "execution_calendar_sha256": data_snapshot.get("calendar_sha256", ""),
                "config_sha256": data_snapshot.get("config_sha256", ""),
                "gate_threshold_ref": f"gate_result.json@{run_id}",
                "product_pool": pool,
                "run_id": run_id,
                "output_dir": output_dir.as_posix(),
                "notes": notes,
            }
        )

    c1_run = ROOT / "reports" / "strategy_research" / "core_satellite" / "core_satellite_20260801_114514"
    c1_dir = c1_run / "C1_CORE_SATELLITE_MOMENTUM"
    c1_freeze = _read_json(c1_dir / "parameter_freeze.json")
    register(
        "C1",
        "C1_CORE_SATELLITE_MOMENTUM",
        "high_return_high_turnover_diagnostic_control",
        "FROZEN_OBSERVATION_CONTROL",
        frozen_runs["C1_core_satellite"]["run_id"],
        c1_dir,
        c1_freeze["parameter_freeze_id"],
        "2026-07-27",
        "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO",
        "NOT_ESTABLISHED",
        _pool_from_run_dir(c1_dir),
        "diagnostic control; not a candidate",
    )

    b2lt_dir = c1_run / "B2_LT_Static_EW_4Asset"
    b2lt_freeze = _read_json(b2lt_dir / "parameter_freeze.json")
    register(
        "B2LT",
        "B2_LT_Static_EW_4Asset",
        "best_static_baseline_control",
        "FROZEN",
        frozen_runs["C1_core_satellite"]["run_id"],
        b2lt_dir,
        b2lt_freeze["parameter_freeze_id"],
        "2026-07-27",
        "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO",
        "NOT_ESTABLISHED",
        _pool_from_run_dir(b2lt_dir),
        "low turnover static control; candidate gate passed in candidate run",
    )

    c3_run = ROOT / "reports" / "strategy_research" / "c3_low_turnover_momentum" / "c3_low_turnover_momentum_20260801_120449"
    c3_dir = c3_run / "C3_LOW_TURNOVER_MOMENTUM"
    c3_freeze = _read_json(c3_dir / "parameter_freeze.json")
    register(
        "C3",
        "C3_LOW_TURNOVER_MOMENTUM",
        "frozen_observation_control_not_candidate",
        "FROZEN_OBSERVATION_CONTROL",
        frozen_runs["C3_low_turnover_momentum"]["run_id"],
        c3_dir,
        c3_freeze["parameter_freeze_id"],
        "2026-07-27",
        "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO",
        "NOT_ESTABLISHED",
        _pool_from_run_dir(c3_dir),
        "frozen observation only; sustained turnover gate failed",
    )

    m20_dir = max(
        (ROOT / "reports" / "strategy_research" / "m20_frozen_observation").glob(
            "m20_frozen_observation_*"
        ),
        key=lambda p: p.stat().st_mtime,
        default=None,
    )
    if m20_dir is None:
        problems.append("M20 run directory missing")
    else:
        m20_freeze = _read_json(m20_dir / "parameter_freeze.json")
        m20_metrics = _read_json(m20_dir / "metrics.json")
        register(
            "M20",
            "M20_Mapped_Fund_Trend",
            "retrospective_observation_line",
            "FROZEN_OBSERVATION",
            m20_dir.name,
            m20_dir,
            m20_freeze["parameter_freeze_id"],
            m20_metrics["oos_end"],
            "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO",
            "NOT_ESTABLISHED",
            _pool_from_run_dir(m20_dir),
            "frozen observation only; gate failed (mdd/cost/turnover/rules/mapping)",
        )

    c2_run = ROOT / "reports" / "strategy_research" / "core_satellite_low_turnover" / "low_turnover_core_satellite_20260801_115521"
    c2_dir = c2_run / "C2_LOW_TURNOVER_CORE_SATELLITE"
    c2_freeze = _read_json(c2_dir / "parameter_freeze.json")
    register(
        "C2",
        "C2_Low_Turnover_Core_Satellite",
        "archived_failed_research_line",
        "ARCHIVED_PARAMETER_SEARCH_FORBIDDEN",
        frozen_runs["C2_low_turnover_core_satellite"]["run_id"],
        c2_dir,
        c2_freeze["parameter_freeze_id"],
        "2026-07-27",
        "CURRENT_SNAPSHOT_CONSERVATIVE_EXECUTION_SCENARIO",
        "NOT_ESTABLISHED",
        _pool_from_run_dir(c2_dir),
        "archived; no further tuning permitted",
    )

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "generated_by": "scripts/migrations/build_frozen_strategy_versions.py",
        "git_commit": git_commit,
        "freeze_timestamp": freeze_timestamp,
        "registered": [row["strategy_key"] for row in rows],
        "freeze_content_complete": not problems,
        "problems": problems,
        "output": OUTPUT_CSV.as_posix(),
    }
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
