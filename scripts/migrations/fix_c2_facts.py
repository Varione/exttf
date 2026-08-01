src = "src/run_low_turnover_core_satellite.py"
with open(src, "r", encoding="utf-8") as f:
    content = f.read()

# Add calendar_correction_label to _facts return dict
old_facts = '''        },
        "execution_calendar": load_execution_calendar(str(ROOT / CALENDAR_PATH)).facts(),'''

new_facts = '''        },
        "calendar_correction_label": "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS",
        "supersedes_run_id": "low_turnover_core_satellite_20260729_163519",
        "intermediate_calendar_corrected_run_id": "low_turnover_core_satellite_20260730_115448",
        "execution_calendar": load_execution_calendar(str(ROOT / CALENDAR_PATH)).facts(),'''

content = content.replace(old_facts, new_facts)

with open(src, "w", encoding="utf-8") as f:
    f.write(content)
print("Added calendar_correction_label to C2 _facts")