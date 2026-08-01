import hashlib, json
from pathlib import Path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Check current state of old manifests
c1_path = Path("reports/strategy_research/core_satellite/core_satellite_20260729_151927/manifest.json")
c2_path = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260729_163519/manifest.json")

print("Old C1 manifest SHA:", sha256_file(c1_path))
print("Expected C1 SHA:    ", "6d685522854aff88cdb97cd528c769563e8f7cc6d7e78519621f89c3ea5f44ce")
print("Match:", sha256_file(c1_path) == "6d685522854aff88cdb97cd528c769563e8f7cc6d7e78519621f89c3ea5f44ce")

print("\nOld C2 manifest SHA:", sha256_file(c2_path))
print("Expected C2 SHA:    ", "f9f8877646b68c4c7af4f5f785d71b3fd72991610447e021835e50bbf8e269d4")
print("Match:", sha256_file(c2_path) == "f9f8877646b68c4c7af4f5f785d71b3fd72991610447e021835e50bbf8e269d4")

# Check if SUPERSEDED field is still present
m1 = json.loads(c1_path.read_text(encoding="utf-8"))
m2 = json.loads(c2_path.read_text(encoding="utf-8"))
print("\nC1 has SUPERSEDED:", "SUPERSEDED_BY_EXECUTION_CALENDAR_FIX" in m1)
print("C2 has SUPERSEDED:", "SUPERSEDED_BY_EXECUTION_CALENDAR_FIX" in m2)