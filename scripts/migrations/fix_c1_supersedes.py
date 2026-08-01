src = "src/run_core_satellite_momentum.py"
with open(src, "r", encoding="utf-8") as f:
    content = f.read()

# Fix supersedes_run_id to point to original buggy run
content = content.replace(
    '"supersedes_run_id": "core_satellite_20260730_114756"',
    '"supersedes_run_id": "core_satellite_20260729_151927"'
)

# Add intermediate_calendar_corrected_run_id
content = content.replace(
    '"supersedes_run_id": "core_satellite_20260729_151927",',
    '"supersedes_run_id": "core_satellite_20260729_151927",\n        "intermediate_calendar_corrected_run_id": "core_satellite_20260730_114756",'
)

with open(src, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed C1 runner supersedes_run_id")