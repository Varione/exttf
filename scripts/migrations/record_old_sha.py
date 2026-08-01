import hashlib, json, os
from pathlib import Path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Old C1 run
c1_dir = Path("reports/strategy_research/core_satellite/core_satellite_20260729_151927")
# Old C2 runs (latest)
c2_dir = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260729_163519")

old_runs = {}

for name, d in [("C1", c1_dir), ("C2", c2_dir)]:
    hashes = {}
    for sub in ["C1_CORE_SATELLITE_MOMENTUM", "B2_Static_EW_4Asset", "B2_LT_Static_EW_4Asset"]:
        metrics_path = d / sub / "metrics.json"
        if metrics_path.exists():
            hashes[f"{name}_{sub}_metrics_sha"] = sha256_file(metrics_path)
        manifest_path = d / sub / "manifest.json"
        if manifest_path.exists():
            hashes[f"{name}_{sub}_manifest_sha"] = sha256_file(manifest_path)
    # Main manifest
    main_manifest = d / "manifest.json"
    if main_manifest.exists():
        hashes[f"{name}_main_manifest_sha"] = sha256_file(main_manifest)
    old_runs[name] = hashes

print("Old run SHA records:")
for name, hashes in old_runs.items():
    print(f"\n{name}:")
    for k, v in hashes.items():
        print(f"  {k}: {v[:16]}...")

# Save to file
with open("data/processed/old_run_sha_records.json", "w", encoding="utf-8") as f:
    json.dump(old_runs, f, indent=2, ensure_ascii=False)
print("\nSaved to data/processed/old_run_sha_records.json")