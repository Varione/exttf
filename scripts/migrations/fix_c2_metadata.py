import json
from pathlib import Path

new_c2_dir = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_115448")

# Update with correct filenames for C2
for name, path in [
    ("status", new_c2_dir / "low_turnover_status.json"),
    ("input_facts", new_c2_dir / "low_turnover_input_facts.json"),
    ("manifest", new_c2_dir / "manifest.json"),
]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["calendar_correction_label"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
        data["supersedes_run_id"] = "low_turnover_core_satellite_20260729_163519"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Updated C2 {name}.json")

# Verify metadata was added
for name in ["low_turnover_status.json", "low_turnover_input_facts.json", "manifest.json"]:
    path = new_c2_dir / name
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"\n{name} has:")
        print(f"  calendar_correction_label: {data.get('calendar_correction_label', 'MISSING')}")
        print(f"  supersedes_run_id: {data.get('supersedes_run_id', 'MISSING')}")
        print(f"  sample_label: {data.get('sample_label', 'N/A')}")