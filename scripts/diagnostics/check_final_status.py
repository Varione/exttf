import json
from pathlib import Path

for label, status_path in [
    ("C1", Path("reports/strategy_research/core_satellite/core_satellite_20260730_124538/core_satellite_status.json")),
    ("C2", Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_125734/low_turnover_status.json")),
]:
    print(f"\n=== {label} status ===")
    s = json.loads(status_path.read_text(encoding="utf-8"))
    print(f"calendar_correction_label: {s.get('calendar_correction_label', 'MISSING')}")
    print(f"supersedes_run_id: {s.get('supersedes_run_id', 'MISSING')}")