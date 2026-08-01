import hashlib, json
from pathlib import Path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

EXPECTED_C1_SHA = "6d685522854aff88cdb97cd528c769563e8f7cc6d7e78519621f89c3ea5f44ce"
EXPECTED_C2_SHA = "f9f8877646b68c4c7af4f5f785d71b3fd72991610447e021835e50bbf8e269d4"

# Restore C1 manifest
c1_path = Path("reports/strategy_research/core_satellite/core_satellite_20260729_151927/manifest.json")
with open(c1_path, "r", encoding="utf-8") as f:
    content = f.read()

# Remove SUPERSEDED field - parse and rewrite cleanly
m = json.loads(content)
if "SUPERSEDED_BY_EXECUTION_CALENDAR_FIX" in m:
    del m["SUPERSEDED_BY_EXECUTION_CALENDAR_FIX"]

# Write with same formatting as original (indent=2, ensure_ascii=False)
restored_content = json.dumps(m, indent=2, ensure_ascii=False) + "\n"
c1_path.write_text(restored_content, encoding="utf-8")

actual_sha = sha256_file(c1_path)
print(f"C1 manifest SHA: {actual_sha}")
print(f"Expected SHA:    {EXPECTED_C1_SHA}")
print(f"Match: {actual_sha == EXPECTED_C1_SHA}")

# Restore C2 manifest
c2_path = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260729_163519/manifest.json")
with open(c2_path, "r", encoding="utf-8") as f:
    content = f.read()

m2 = json.loads(content)
if "SUPERSEDED_BY_EXECUTION_CALENDAR_FIX" in m2:
    del m2["SUPERSEDED_BY_EXECUTION_CALENDAR_FIX"]

restored_content2 = json.dumps(m2, indent=2, ensure_ascii=False) + "\n"
c2_path.write_text(restored_content2, encoding="utf-8")

actual_sha2 = sha256_file(c2_path)
print(f"\nC2 manifest SHA: {actual_sha2}")
print(f"Expected SHA:    {EXPECTED_C2_SHA}")
print(f"Match: {actual_sha2 == EXPECTED_C2_SHA}")