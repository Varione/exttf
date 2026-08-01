"""Forward data revision mechanism (planning P2-E).

Tracks hashes of every data artifact that feeds frozen strategies. Any
change to a tracked file must be recorded as a revision BEFORE the data is
used for forward observation; otherwise the registry reports
UNRECORDED_CHANGES and forward observation is considered paused.

Commands:
- --init   : create config/data_revision_registry.json from the baseline
             freeze manifest (idempotent)
- --verify : compare current hashes with the baseline and the last
             recorded revision; print status
- --record : record a revision for every file whose hash differs from the
             last recorded hash. Requires --reason
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "baseline_freeze_manifest.json"
REGISTRY = ROOT / "config" / "data_revision_registry.json"

TRACKED_FILES = [
    ("data/processed/otf_expanded.sqlite", "db_expanded", "db_sha256"),
    ("data/processed/otf_mapped.sqlite", "db_mapped", None),
    ("data/processed/etf.sqlite", "db_etf", None),
    ("config/otf_product_rules.csv", "rules", "rules_sha256"),
    ("config/otf_exposure_mapping.csv", "mapping", "mapping_sha256"),
    ("config/otf_walkforward.json", "config", "config_sha256"),
    ("data/processed/execution_calendar/cn_execution_calendar.csv", "calendar", "calendar_sha256"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _init_registry() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    snapshot = manifest["data_snapshot"]
    files = {}
    for rel_path, role, manifest_key in TRACKED_FILES:
        path = ROOT / rel_path
        baseline = snapshot.get(manifest_key, "") if manifest_key else ""
        current = _sha256(path) if path.exists() else ""
        if not baseline and current:
            baseline = current
        files[rel_path] = {
            "role": role,
            "baseline_sha256": baseline,
            "last_recorded_sha256": baseline,
            "current_sha256": current,
            "baseline_source": (
                "freeze_manifest" if manifest_key else "initialized_at_first_run"
            ),
        }
    return {
        "mechanism": "forward_data_revision",
        "rule": "Any change to a tracked file must be recorded via "
        "--record BEFORE the data is used for forward observation. "
        "UNRECORDED_CHANGES pauses forward observation.",
        "baseline_freeze_timestamp": manifest["freeze_timestamp"],
        "baseline_git_commit": manifest["git_commit"],
        "revisions": [],
        "files": files,
    }


def _verify(registry: dict) -> tuple[dict, int]:
    report = {"status": "CLEAN", "files": {}}
    exit_code = 0
    for rel_path, entry in registry["files"].items():
        path = ROOT / rel_path
        current = _sha256(path) if path.exists() else ""
        entry["current_sha256"] = current
        if current != entry["last_recorded_sha256"]:
            report["status"] = "UNRECORDED_CHANGES"
            exit_code = 1
            report["files"][rel_path] = {
                "status": "CHANGED_UNRECORDED",
                "last_recorded_sha256": entry["last_recorded_sha256"],
                "current_sha256": current,
                "baseline_sha256": entry["baseline_sha256"],
            }
        elif current != entry["baseline_sha256"]:
            report["files"][rel_path] = {"status": "RECORDED_REVISION"}
        else:
            report["files"][rel_path] = {"status": "MATCHES_BASELINE"}
    return report, exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    if args.init:
        registry = _init_registry()
        REGISTRY.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"registry initialized: {REGISTRY}")
        return 0

    if not REGISTRY.exists():
        print("ERROR: registry missing; run --init first")
        return 1
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    if args.verify or (not args.record):
        report, exit_code = _verify(registry)
        REGISTRY.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return exit_code

    if args.record:
        if not args.reason:
            print("ERROR: --record requires --reason")
            return 1
        report, _ = _verify(registry)
        if report["status"] == "CLEAN":
            print("no changes to record")
            return 0
        revision = {
            "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
            "reason": args.reason,
            "git_commit": _current_commit(),
            "changes": {},
        }
        for rel_path, info in report["files"].items():
            entry = registry["files"][rel_path]
            revision["changes"][rel_path] = {
                "role": entry["role"],
                "old_sha256": entry["last_recorded_sha256"],
                "new_sha256": entry["current_sha256"],
            }
            entry["last_recorded_sha256"] = entry["current_sha256"]
        registry["revisions"].append(revision)
        REGISTRY.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(revision, indent=2, ensure_ascii=False))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
