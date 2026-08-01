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

# Try different formatting options for C1
c1_path = Path("reports/strategy_research/core_satellite/core_satellite_20260729_151927/manifest.json")
m = json.loads(c1_path.read_text(encoding="utf-8"))

for variant in [
    ("no_newline", lambda: json.dumps(m, indent=2, ensure_ascii=False)),
    ("with_newline", lambda: json.dumps(m, indent=2, ensure_ascii=False) + "\n"),
    ("sort_keys_no_nl", lambda: json.dumps(m, indent=2, ensure_ascii=False, sort_keys=True)),
    ("sort_keys_nl", lambda: json.dumps(m, indent=2, ensure_ascii=False, sort_keys=True) + "\n"),
    ("compact_newline", lambda: json.dumps(m, indent=2, ensure_ascii=False) + "\n\n"),
]:
    name, content_fn = variant
    try:
        content = content_fn()
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        match = "MATCH" if sha == EXPECTED_C1_SHA else ""
        print(f"C1 {name}: {sha[:32]}... {match}")
    except Exception as e:
        print(f"C1 {name}: ERROR {e}")

# Same for C2
c2_path = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260729_163519/manifest.json")
m2 = json.loads(c2_path.read_text(encoding="utf-8"))

for variant in [
    ("no_newline", lambda: json.dumps(m2, indent=2, ensure_ascii=False)),
    ("with_newline", lambda: json.dumps(m2, indent=2, ensure_ascii=False) + "\n"),
    ("sort_keys_no_nl", lambda: json.dumps(m2, indent=2, ensure_ascii=False, sort_keys=True)),
    ("sort_keys_nl", lambda: json.dumps(m2, indent=2, ensure_ascii=False, sort_keys=True) + "\n"),
]:
    name, content_fn = variant
    try:
        content = content_fn()
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        match = "MATCH" if sha == EXPECTED_C2_SHA else ""
        print(f"C2 {name}: {sha[:32]}... {match}")
    except Exception as e:
        print(f"C2 {name}: ERROR {e}")