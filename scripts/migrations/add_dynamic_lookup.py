src = "tests/test_metadata_consistency.py"
with open(src, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find insertion point after EXPECTED_OLD_C2_SHA
insert_idx = None
for i, line in enumerate(lines):
    if 'EXPECTED_OLD_C2_SHA' in line:
        insert_idx = i + 1
        break

if insert_idx is not None:
    new_lines = [
        "\n",
        "\n",
        "def _latest_c1_run() -> Path:\n",
        '    """Return the latest core_satellite run directory with calendar_correction_label."""\n',
        '    base = ROOT / "reports/strategy_research/core_satellite"\n',
        "    if not base.exists():\n",
        '        pytest.skip("No C1 runs found")\n',
        "    dirs = sorted(base.iterdir(), key=lambda d: d.name, reverse=True)\n",
        "    for d in dirs:\n",
        '        status_path = d / "core_satellite_status.json"\n',
        "        if status_path.exists():\n",
        "            try:\n",
        "                s = json.loads(status_path.read_text(encoding=\"utf-8\"))\n",
        '                if s.get("calendar_correction_label") == "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS":\n',
        "                    return d\n",
        "            except (json.JSONDecodeError, OSError):\n",
        "                continue\n",
        '    pytest.skip("No calendar-corrected C1 run found")\n',
        "\n",
        "\n",
        "def _latest_c2_run() -> Path:\n",
        '    """Return the latest low_turnover_core_satellite run directory with calendar_correction_label."""\n',
        '    base = ROOT / "reports/strategy_research/core_satellite_low_turnover"\n',
        "    if not base.exists():\n",
        '        pytest.skip("No C2 runs found")\n',
        "    dirs = sorted(base.iterdir(), key=lambda d: d.name, reverse=True)\n",
        "    for d in dirs:\n",
        '        status_path = d / "low_turnover_status.json"\n',
        "        if status_path.exists():\n",
        "            try:\n",
        "                s = json.loads(status_path.read_text(encoding=\"utf-8\"))\n",
        '                if s.get("calendar_correction_label") == "CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS":\n',
        "                    return d\n",
        "            except (json.JSONDecodeError, OSError):\n",
        "                continue\n",
        '    pytest.skip("No calendar-corrected C2 run found")\n',
    ]
    lines[insert_idx:insert_idx] = new_lines

with open(src, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Added dynamic lookup functions. New length:", len("".join(lines)))