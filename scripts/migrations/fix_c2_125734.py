import json, hashlib
from pathlib import Path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Fix C2 run low_turnover_core_satellite_20260730_125734
c2_dir = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_125734")

for name, path in [
    ("status", c2_dir / "low_turnover_status.json"),
    ("input_facts", c2_dir / "low_turnover_input_facts.json"),
    ("manifest", c2_dir / "manifest.json"),
]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["calendar_correction_label"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
        data["supersedes_run_id"] = "low_turnover_core_satellite_20260730_115448"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"C2 fixed {name}.json")

# Verify
status = json.loads((c2_dir / "low_turnover_status.json").read_text(encoding="utf-8"))
print(f"\nC2 status calendar_correction_label: {status.get('calendar_correction_label')}")
print(f"C2 status supersedes_run_id: {status.get('supersedes_run_id')}")