import json, sys
from pathlib import Path

# Mark old C1 manifest as SUPERSEDED
old_c1_manifest = Path("reports/strategy_research/core_satellite/core_satellite_20260729_151927/manifest.json")
if old_c1_manifest.exists():
    m = json.loads(old_c1_manifest.read_text())
    m["SUPERSEDED_BY_EXECUTION_CALENDAR_FIX"] = "core_satellite_20260730_114756"
    old_c1_manifest.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Marked old C1 manifest as SUPERSEDED")

# Mark old C2 manifest as SUPERSEDED  
old_c2_manifest = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260729_163519/manifest.json")
if old_c2_manifest.exists():
    m = json.loads(old_c2_manifest.read_text())
    m["SUPERSEDED_BY_EXECUTION_CALENDAR_FIX"] = "low_turnover_core_satellite_20260730_115448"
    old_c2_manifest.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Marked old C2 manifest as SUPERSEDED")

# Add CALENDAR_CORRECTED tag to new C1 manifest
new_c1_manifest = Path("reports/strategy_research/core_satellite/core_satellite_20260730_114756/manifest.json")
if new_c1_manifest.exists():
    m = json.loads(new_c1_manifest.read_text())
    m["CALENDAR_CORRECTED"] = True
    m["SAMPLE_TAG"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
    new_c1_manifest.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Tagged new C1 manifest as CALENDAR_CORRECTED")

# Add CALENDAR_CORRECTED tag to new C2 manifest
new_c2_manifest = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_115448/manifest.json")
if new_c2_manifest.exists():
    m = json.loads(new_c2_manifest.read_text())
    m["CALENDAR_CORRECTED"] = True
    m["SAMPLE_TAG"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
    new_c2_manifest.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Tagged new C2 manifest as CALENDAR_CORRECTED")

print("\nDone!")