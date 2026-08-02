"""Build and maintain the T0 registry for forward shadow observation.

T0 is defined (planning P2-2) as the first available execution trading day
after the clean baseline freeze.  Two constraints must hold simultaneously
(audit 2026-08-02):

1. The candidate day must be strictly later than the freeze COMPLETION time
   (freeze_timestamp, converted to Asia/Shanghai), not just later than the
   baseline data cutoff.  Days between the data cutoff and the freeze time
   (2026-07-28 through 2026-07-31) were backfilled on 2026-08-02 and are
   therefore marked POST_FREEZE_BACKFILLED_VALIDATION_WINDOW: usable for
   validation, never countable as fresh forward observations.
2. The candidate day must have NAV data collected forward (db max nav_date
   >= candidate).  Until that holds the registry stays
   PENDING_POST_FREEZE_OBSERVATION; no date is hardcoded.

This script is idempotent: rerun it after a forward data refresh and it
will promote the registry to REGISTERED automatically once the true T0 has
real data.

Outputs:
- config/otf_t0_registry.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "baseline_freeze_manifest.json"
CALENDAR = ROOT / "data" / "processed" / "execution_calendar" / "cn_execution_calendar.csv"
DB_EXPANDED = ROOT / "data" / "processed" / "otf_expanded.sqlite"
OUTPUT = ROOT / "config" / "otf_t0_registry.json"

TZ_SHANGHAI = dt.timezone(dt.timedelta(hours=8))
BACKFILL_WINDOW = "POST_FREEZE_BACKFILLED_VALIDATION_WINDOW"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_local_date(freeze_timestamp: str) -> dt.date:
    parsed = dt.datetime.fromisoformat(freeze_timestamp)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(TZ_SHANGHAI)
    return parsed.date()


def _calendar_dates(calendar: Path) -> list[dt.date]:
    dates: list[dt.date] = []
    with calendar.open(encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            if not line.strip():
                continue
            dates.append(dt.date.fromisoformat(line.split(",")[0].strip()))
    return dates


def _db_max_nav_date(db_path: Path) -> dt.date | None:
    if not db_path.exists():
        return None
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("SELECT MAX(nav_date) FROM otf_fund_nav").fetchone()
    return dt.date.fromisoformat(row[0]) if row and row[0] else None


def compute_t0_registry(
    freeze_timestamp: str,
    git_commit: str,
    calendar_file: Path,
    db_max_date: dt.date | None,
) -> dict:
    """Pure T0 computation; returns the full registry payload."""
    freeze_date = _freeze_local_date(freeze_timestamp)
    dates = _calendar_dates(calendar_file)
    calendar_last = max(dates)

    candidates = [d for d in dates if d > freeze_date]
    candidate = min(candidates) if candidates else None

    # Backfill window between the baseline data cutoff and the freeze time
    oos_period = ""
    manifest_path = ROOT / "baseline_freeze_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        oos_period = str(manifest.get("oos_period", ""))
    data_cutoff = None
    if oos_period and "~" in oos_period:
        try:
            data_cutoff = dt.date.fromisoformat(oos_period.split("~")[1].strip())
        except ValueError:
            data_cutoff = None

    backfill_window = None
    if data_cutoff is not None and freeze_date > data_cutoff:
        backfill_window = {
            "start": data_cutoff.isoformat(),
            "end_exclusive": freeze_date.isoformat(),
            "classification": BACKFILL_WINDOW,
            "note": (
                "Dates in this window were backfilled during the post-freeze "
                "refresh; they validate tooling but are NOT fresh forward "
                "observations and cannot be counted as fresh OOS."
            ),
        }

    if candidate is None:
        status = "PENDING_CALENDAR_EXTENSION"
        note = (
            "No execution calendar day exists after the freeze completion "
            f"time ({freeze_date.isoformat()}). Rerun after the calendar "
            "extends beyond the freeze."
        )
    elif db_max_date is None or db_max_date < candidate:
        status = "PENDING_POST_FREEZE_OBSERVATION"
        note = (
            f"First post-freeze execution day is {candidate.isoformat()} but "
            "its NAV data has not yet been collected forward. T0 stays "
            "PENDING until the candidate day has data (db max nav_date >= "
            "candidate). No date is hardcoded."
        )
    else:
        status = "REGISTERED"
        note = (
            f"T0={candidate.isoformat()} is the first execution calendar day "
            f"strictly after the freeze completion time "
            f"({freeze_date.isoformat()}) and its NAV data exists in the DB; "
            "forward shadow observation may begin."
        )

    return {
        "t0_date": candidate.isoformat() if candidate and status == "REGISTERED" else None,
        "t0_status": status,
        "baseline_freeze_timestamp": freeze_timestamp,
        "baseline_freeze_local_date": freeze_date.isoformat(),
        "baseline_git_commit": git_commit,
        "baseline_data_cutoff": data_cutoff.isoformat() if data_cutoff else None,
        "post_freeze_backfill_window": backfill_window,
        "execution_calendar_last_date": calendar_last.isoformat(),
        "db_max_nav_date": db_max_date.isoformat() if db_max_date else None,
        "execution_calendar_sha256": _sha256(calendar_file),
        "candidate_next_business_day": candidate.isoformat() if candidate else None,
        "rule_reference": "planning P2-2: T0 is the first available "
        "execution trading day after the clean baseline freeze; no "
        "backfitting with post-T0 data; modifications generate new versions; "
        "data revisions must be recorded; forward uses then-visible data and "
        "then-rules.",
        "note": note,
    }


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry = compute_t0_registry(
        freeze_timestamp=manifest["freeze_timestamp"],
        git_commit=manifest["git_commit"],
        calendar_file=CALENDAR,
        db_max_date=_db_max_nav_date(DB_EXPANDED),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(registry, indent=2, ensure_ascii=False))
    return 0 if registry["t0_status"] == "REGISTERED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
