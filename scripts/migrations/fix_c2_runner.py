src = "src/run_low_turnover_core_satellite.py"
with open(src, "r", encoding="utf-8") as f:
    content = f.read()

# Add calendar_correction_label and supersedes_run_id to status_payload
old_status_c2 = '''        "failed_research_thresholds": c2_gate["research_performance_failed_checks"],
        "failed_observation_readiness_reasons": c2_gate["observation_readiness_failed_reasons"],
        "disclosures": ['''

new_status_c2 = '''        "calendar_correction_label": "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "supersedes_run_id": "low_turnover_core_satellite_20260730_115448",
        "failed_research_thresholds": c2_gate["research_performance_failed_checks"],
        "failed_observation_readiness_reasons": c2_gate["observation_readiness_failed_reasons"],
        "disclosures": ['''

content = content.replace(old_status_c2, new_status_c2)

# Add to _facts function return dict
old_tc_end_c2 = '''            "total_rules": int(len(rules)),
        },
        "calendar_correction_label": "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "supersedes_run_id": "low_turnover_core_satellite_20260730_115448",
        "execution_calendar": load_execution_calendar(str(ROOT / CALENDAR_PATH)).facts(),'''

# Check if already present from previous patch attempt, otherwise add
if '"calendar_correction_label"' not in content.split('"rule_temporal_coverage"')[1].split("},\n        \"publication_timing_audit\"")[0]:
    old_tc_end_c2 = '''            "total_rules": int(len(rules)),
        },
        "execution_calendar": load_execution_calendar(str(ROOT / CALENDAR_PATH)).facts(),'''

    new_tc_end_c2 = '''            "total_rules": int(len(rules)),
        },
        "calendar_correction_label": "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "supersedes_run_id": "low_turnover_core_satellite_20260730_115448",
        "execution_calendar": load_execution_calendar(str(ROOT / CALENDAR_PATH)).facts(),'''

    content = content.replace(old_tc_end_c2, new_tc_end_c2)

with open(src, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed run_low_turnover_core_satellite.py. New length:", len(content))