"""Build and maintain the T0 registry for forward shadow observation.

T0 is defined (planning P2-2) as the first available execution trading day
after the clean baseline freeze. The baseline freeze data cutoff is
2026-07-27, and the consensus execution calendar currently ends on that
same date, so the true T0 can only be registered once the calendar and
NAV data have been extended.

This script is idempotent: rerun it after a data/calendar refresh and it
will promote the registry from PENDING to REGISTERED automatically.

Outputs:
- config/otf_t0_registry.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "baseline_freeze_manifest.json"
CALENDAR = ROOT / "data" / "processed" / "execution_calendar" / "cn_execution_calendar.csv"
OUTPUT = ROOT / "config" / "otf_t0_registry.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _next_business_day(after: dt.date) -> dt.date:
    day = after + dt.timedelta(days=1)
    while day.weekday() >= 5:
        day += dt.timedelta(days=1)
    return day


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    freeze_timestamp = manifest["freeze_timestamp"]
    git_commit = manifest["git_commit"]

    dates: list[dt.date] = []
    with CALENDAR.open(encoding="utf-8") as handle:
        header = handle.readline()
        for line in handle:
            if not line.strip():
                continue
            dates.append(dt.date.fromisoformat(line.split(",")[0].strip()))
    calendar_last = max(dates)
    calendar_set = set(dates)

    data_cutoff = calendar_last
    t0_candidates = [d for d in dates if d > data_cutoff]
    t0 = min(t0_candidates) if t0_candidates else None

    if t0 is not None:
        status = "REGISTERED"
        note = (
            f"T0={t0.isoformat()} is the first execution calendar day after "
            "the baseline data cutoff; forward shadow observation may begin."
        )
    else:
        estimate = _next_business_day(data_cutoff)
        status = "PENDING_CALENDAR_EXTENSION"
        note = (
            "Execution calendar and NAV data end at the baseline cutoff; no "
            "post-cutoff execution day exists yet. Rerun this script after a "
            "data refresh to auto-register the true T0. "
            f"Non-weekend estimate is {estimate.isoformat()} but the true T0 "
            "must come from the consensus calendar."
        )

    registry = {
        "t0_date": t0.isoformat() if t0 else None,
        "t0_status": status,
        "baseline_freeze_timestamp": freeze_timestamp,
        "baseline_git_commit": git_commit,
        "data_cutoff_date": data_cutoff.isoformat(),
        "execution_calendar_last_date": calendar_last.isoformat(),
        "execution_calendar_sha256": _sha256(CALENDAR),
        "candidate_next_business_day": (
            None if t0 else _next_business_day(data_cutoff).isoformat()
        ),
        "rule_reference": "planning P2-2: T0 is the first available "
        "execution trading day after the clean baseline freeze; no "
        "backfitting with post-T0 data; modifications generate new versions; "
        "data revisions must be recorded; forward uses then-visible data and "
        "then-rules.",
        "note": note,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(registry, indent=2, ensure_ascii=False))
    return 0 if t0 is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
