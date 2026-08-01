import json, hashlib
from pathlib import Path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Fix C1 run core_satellite_20260730_124538
c1_dir = Path("reports/strategy_research/core_satellite/core_satellite_20260730_124538")

for name, path in [
    ("status", c1_dir / "core_satellite_status.json"),
    ("input_facts", c1_dir / "core_satellite_input_facts.json"),
    ("manifest", c1_dir / "manifest.json"),
]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["calendar_correction_label"] = "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS"
        data["supersedes_run_id"] = "core_satellite_20260730_114756"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"C1 fixed {name}.json")

# Verify
status = json.loads((c1_dir / "core_satellite_status.json").read_text(encoding="utf-8"))
print(f"\nC1 status calendar_correction_label: {status.get('calendar_correction_label')}")
print(f"C1 status supersedes_run_id: {status.get('supersedes_run_id')}")
print(f"C1 manifest SHA: {sha256_file(c1_dir / 'manifest.json')[:32]}...")