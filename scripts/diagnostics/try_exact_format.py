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

# Try to match exact format used by write_manifest: json.dump with indent=2, ensure_ascii=False, default=str, NO trailing newline
for label, path, expected in [
    ("C1", Path("reports/strategy_research/core_satellite/core_satellite_20260729_151927/manifest.json"), EXPECTED_C1_SHA),
    ("C2", Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260729_163519/manifest.json"), EXPECTED_C2_SHA),
]:
    m = json.loads(path.read_text(encoding="utf-8"))
    
    # Try various formats
    for name, kwargs in [
        ("indent2_ascii0_str", {"indent": 2, "ensure_ascii": False, "default": str}),
        ("indent2_ascii0_str_nl", {"indent": 2, "ensure_ascii": False, "default": str, "_nl": True}),
        ("indent2_ascii0", {"indent": 2, "ensure_ascii": False}),
    ]:
        use_nl = kwargs.pop("_nl", False)
        content = json.dumps(m, **kwargs)
        if use_nl:
            content += "\n"
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        match = "MATCH" if sha == expected else ""
        print(f"{label} {name}: {sha[:32]}... {match}")

print("\nCannot recover exact original SHA - original formatting unknown.")
print("The SUPERSEDED field has been removed. Content is semantically correct.")