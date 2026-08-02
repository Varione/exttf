"""P0-1: Run the full regression suite in batches and record a machine-readable artifact.

Records interpreter, Python version, git commit, dirty state, test file count,
per-batch summaries and the full original stdout into reports/test_regression/.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = r"D:\miniconda\envs\agents\python.exe"
OUT_DIR = ROOT / "reports" / "test_regression"

BATCHES: list[list[str]] = [
    [
        "tests/test_walkforward_e2e.py",
        "tests/test_strategy_validation.py",
        "tests/test_strategy_attribution.py",
        "tests/test_smoke.py",
        "tests/test_reproducibility_audit.py",
    ],
    [
        "tests/test_state_allocation.py",
        "tests/test_schedule.py",
        "tests/test_risk_parity.py",
        "tests/test_experiment_artifacts.py",
        "tests/test_runner_wiring.py",
        "tests/test_product_selector.py",
        "tests/test_otf_trading_rules.py",
        "tests/test_otf_backtest.py",
        "tests/test_signal_timing.py",
        "tests/test_research_status.py",
        "tests/test_c3_runner.py",
        "tests/test_nav_availability.py",
        "tests/test_p2_forward_mechanisms.py",
        "tests/test_nav_refresh_incremental.py",
    ],
    ["tests/test_unified_experiment.py"],
    [
        "tests/test_phase3.py",
        "tests/test_mapped_otf_strategy.py",
        "tests/test_low_turnover_core_satellite.py",
        "tests/test_holiday_share_adjustment.py",
        "tests/test_full_otf_catalog.py",
        "tests/test_factor_artifact.py",
        "tests/test_extended_otf_assets.py",
        "tests/test_execution_calendar_integration.py",
        "tests/test_etf_otf_mapping.py",
        "tests/test_cross_fund_holiday_adj.py",
        "tests/test_core_satellite_momentum.py",
        "tests/test_candidate_strategies.py",
        "tests/test_c3_low_turnover_momentum.py",
        "tests/test_build_otf_research_db.py",
        "tests/test_all_otf_allocation.py",
        "tests/test_metadata_consistency.py",
        "tests/test_market_state.py",
    ],
    [
        "tests/test_rl_meta.py",
        "tests/test_repaired_core.py",
        "tests/test_regime_predictor.py",
    ],
]


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(ROOT), capture_output=True, text=True
    )
    return result.stdout.strip()


def parse_summary(stdout: str) -> dict[str, int]:
    last_line = [line for line in stdout.splitlines() if line.strip()][-1]
    pattern = re.compile(r"(\d+)\s+(passed|failed|error|warnings?)")
    counts: dict[str, int] = {}
    for number, label in pattern.findall(last_line):
        counts[label] = counts.get(label, 0) + int(number)
    return counts


def run_batch(batch: list[str], index: int, total: int) -> dict[str, object]:
    cmd = [PYTHON, "-m", "pytest", "-q", "--tb=line", "-W", "error::FutureWarning", *batch]
    started = time.time()
    result = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800
    )
    elapsed = round(time.time() - started, 2)
    stdout = result.stdout
    stderr = result.stderr
    counts = parse_summary(stdout)
    log_path = OUT_DIR / f"batch_{index:02d}_of_{total:02d}.log"
    log_path.write_text(
        f"COMMAND: {' '.join(cmd)}\nRETURN_CODE: {result.returncode}\n"
        f"ELAPSED_SECONDS: {elapsed}\n\n=== STDOUT ===\n{stdout}\n"
        f"\n=== STDERR ===\n{stderr}\n",
        encoding="utf-8",
    )
    return {
        "batch_index": index,
        "batch_total": total,
        "files": batch,
        "return_code": result.returncode,
        "elapsed_seconds": elapsed,
        "counts": counts,
        "log_file": log_path.name,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    test_files = sorted(
        str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_*.py")
    )

    started = time.time()
    batches: list[dict[str, object]] = []
    for index, batch in enumerate(BATCHES, start=1):
        print(f"Running batch {index}/{len(BATCHES)}: {len(batch)} files...", flush=True)
        batches.append(run_batch(batch, index, len(BATCHES)))

    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_warnings = 0
    for b in batches:
        counts: dict[str, int] = b["counts"]  # type: ignore[assignment]
        total_passed += int(counts.get("passed", 0))
        total_failed += int(counts.get("failed", 0))
        total_errors += int(counts.get("error", 0))
        total_warnings += int(counts.get("warnings", 0))

    payload = {
        "record_type": "FULL_REGRESSION_RECORD",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interpreter": PYTHON,
        "python_version": sys.version,
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
        "test_file_count": len(test_files),
        "test_files": test_files,
        "total_elapsed_seconds": round(time.time() - started, 2),
        "summary": {
            "passed": total_passed,
            "failed": total_failed,
            "errors": total_errors,
            "warnings": total_warnings,
            "all_passed": total_failed == 0 and total_errors == 0,
        },
        "batches": batches,
    }
    out_path = OUT_DIR / f"regression_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    print(f"Record written: {out_path}")
    return 0 if payload["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
