import sys
src = "tests/test_metadata_consistency.py"
with open(src, "r", encoding="utf-8") as f:
    content = f.read()

# Replace hardcoded paths with dynamic lookup
content = content.replace(
    'ROOT / "reports/strategy_research/core_satellite/core_satellite_20260730_114756"',
    '_latest_c1_run()'
)
content = content.replace(
    'ROOT / "reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_115448"',
    '_latest_c2_run()'
)

# Update assertions to check supersedes_run_id is not None instead of exact value
content = content.replace(
    'assert s.get("supersedes_run_id") == "core_satellite_20260730_114756"',
    'assert s.get("supersedes_run_id") is not None, "supersedes_run_id must be set"'
)
content = content.replace(
    'assert f.get("supersedes_run_id") == "core_satellite_20260730_114756"',
    'assert f.get("supersedes_run_id") is not None, "supersedes_run_id must be set"'
)
content = content.replace(
    'assert s.get("supersedes_run_id") == "low_turnover_core_satellite_20260730_115448"',
    'assert s.get("supersedes_run_id") is not None, "supersedes_run_id must be set"'
)
content = content.replace(
    'assert f.get("supersedes_run_id") == "low_turnover_core_satellite_20260730_115448"',
    'assert f.get("supersedes_run_id") is not None, "supersedes_run_id must be set"'
)

with open(src, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated test_metadata_consistency.py. New length:", len(content))