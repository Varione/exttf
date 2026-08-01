import json
from pathlib import Path

# Update new C1 root metadata
new_c1_dir = Path("reports/strategy_research/core_satellite/core_satellite_20260730_114756")

# 1. Update status.json
status_path = new_c1_dir / "core_satellite_status.json"
if status_path.exists():
    s = json.loads(status_path.read_text(encoding="utf-8"))
    s["calendar_correction_label"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
    s["supersedes_run_id"] = "core_satellite_20260729_151927"
    status_path.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Updated C1 status.json with calendar_correction_label and supersedes_run_id")

# 2. Update input_facts.json
facts_path = new_c1_dir / "core_satellite_input_facts.json"
if facts_path.exists():
    f = json.loads(facts_path.read_text(encoding="utf-8"))
    f["calendar_correction_label"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
    f["supersedes_run_id"] = "core_satellite_20260729_151927"
    facts_path.write_text(json.dumps(f, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Updated C1 input_facts.json")

# 3. Update conclusion.json if exists
conc_path = new_c1_dir / "conclusion.json"
if conc_path.exists():
    c = json.loads(conc_path.read_text(encoding="utf-8"))
    c["calendar_correction_label"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
    c["supersedes_run_id"] = "core_satellite_20260729_151927"
    conc_path.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Updated C1 conclusion.json")

# 4. Update root manifest.json
manifest_path = new_c1_dir / "manifest.json"
if manifest_path.exists():
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    m["calendar_correction_label"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
    m["supersedes_run_id"] = "core_satellite_20260729_151927"
    manifest_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Updated C1 manifest.json")

# Same for new C2
new_c2_dir = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_115448")

for name, path in [
    ("status", new_c2_dir / "low_turnover_core_satellite_status.json"),
    ("input_facts", new_c2_dir / "core_satellite_input_facts.json"),
    ("conclusion", new_c2_dir / "conclusion.json"),
    ("manifest", new_c2_dir / "manifest.json"),
]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["calendar_correction_label"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
        data["supersedes_run_id"] = "low_turnover_core_satellite_20260729_163519"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Updated C2 {name}.json")

print("\nMetadata update complete.")