src = "src/run_core_satellite_momentum.py"
with open(src, "r", encoding="utf-8") as f:
    content = f.read()

# Add calendar_correction_label and supersedes_run_id to status_payload
old_status = '''        "failed_research_thresholds": c1_gate["failed_checks"],
        "disclosures": ['''

new_status = '''        "calendar_correction_label": "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "supersedes_run_id": "core_satellite_20260730_114756",
        "failed_research_thresholds": c1_gate["failed_checks"],
        "disclosures": ['''

content = content.replace(old_status, new_status)

# Add to facts (input_facts.json) - in _facts function return dict
old_tc_end = '''            "total_rules": int(len(rules)),
        },
        "execution_calendar": load_execution_calendar(str(ROOT / CALENDAR_PATH)).facts(),'''

new_tc_end = '''            "total_rules": int(len(rules)),
        },
        "calendar_correction_label": "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "supersedes_run_id": "core_satellite_20260730_114756",
        "execution_calendar": load_execution_calendar(str(ROOT / CALENDAR_PATH)).facts(),'''

content = content.replace(old_tc_end, new_tc_end)

with open(src, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed run_core_satellite_momentum.py. New length:", len(content))