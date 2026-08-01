src = "src/run_low_turnover_core_satellite.py"
with open(src, "r", encoding="utf-8") as f:
    content = f.read()

# Fix supersedes_run_id to point to original buggy run
content = content.replace(
    '"supersedes_run_id": "low_turnover_core_satellite_20260730_115448"',
    '"supersedes_run_id": "low_turnover_core_satellite_20260729_163519"'
)

# Add intermediate_calendar_corrected_run_id
content = content.replace(
    '"supersedes_run_id": "low_turnover_core_satellite_20260729_163519",',
    '"supersedes_run_id": "low_turnover_core_satellite_20260729_163519",\n        "intermediate_calendar_corrected_run_id": "low_turnover_core_satellite_20260730_115448",'
)

with open(src, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed C2 runner supersedes_run_id")