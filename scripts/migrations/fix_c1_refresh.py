src = "src/run_core_satellite_momentum.py"
with open(src, "r", encoding="utf-8") as f:
    content = f.read()

# Add calendar_correction_label and supersedes fields to _refresh_existing_run
old_refresh = '''    facts["publication_timing_audit"] = publication_timing_audit()
    status["status"] = "RESEARCH_GATE_FAILED"'''

new_refresh = '''    facts["publication_timing_audit"] = publication_timing_audit()
    facts["calendar_correction_label"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
    facts["supersedes_run_id"] = "core_satellite_20260729_151927"
    facts["intermediate_calendar_corrected_run_id"] = "core_satellite_20260730_114756"
    status["status"] = "RESEARCH_GATE_FAILED"'''

content = content.replace(old_refresh, new_refresh)

old_status_set = '''    status["publication_timing_audit"] = publication_timing_audit()
    status.setdefault("disclosures", [])'''

new_status_set = '''    status["publication_timing_audit"] = publication_timing_audit()
    status["calendar_correction_label"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
    status["supersedes_run_id"] = "core_satellite_20260729_151927"
    status["intermediate_calendar_corrected_run_id"] = "core_satellite_20260730_114756"
    status.setdefault("disclosures", [])'''

content = content.replace(old_status_set, new_status_set)

with open(src, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed C1 refresh function")