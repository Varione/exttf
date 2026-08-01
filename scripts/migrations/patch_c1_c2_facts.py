import sys

# Patch run_core_satellite_momentum.py
src = 'src/run_core_satellite_momentum.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# Add calendar import after schedule import
content = content.replace(
    'from otf_rotation.schedule import (\n',
    'from otf_rotation.execution_calendar import load_execution_calendar\nfrom otf_rotation.schedule import (\n'
)

# Add CALENDAR_PATH constant after DB_PATH
content = content.replace(
    'DB_PATH = "data/processed/otf_expanded.sqlite"\n',
    'DB_PATH = "data/processed/otf_expanded.sqlite"\nCALENDAR_PATH = "data/processed/execution_calendar/cn_execution_calendar.csv"\n'
)

# Update _facts to include calendar hash and facts
old_hashes = '''        "input_hashes": {
            "db_sha256": sha256_file(str(ROOT / DB_PATH)),
            "rules_sha256": sha256_file(str(ROOT / RULES_PATH)),
            "mapping_sha256": sha256_file(str(ROOT / MAPPING_PATH)),
            "config_sha256": sha256_file(str(ROOT / CONFIG_PATH)),
        },'''

new_hashes = '''        "input_hashes": {
            "db_sha256": sha256_file(str(ROOT / DB_PATH)),
            "rules_sha256": sha256_file(str(ROOT / RULES_PATH)),
            "mapping_sha256": sha256_file(str(ROOT / MAPPING_PATH)),
            "config_sha256": sha256_file(str(ROOT / CONFIG_PATH)),
            "calendar_sha256": sha256_file(str(ROOT / CALENDAR_PATH)),
        },'''

content = content.replace(old_hashes, new_hashes)

# Add calendar facts after rule_temporal_coverage
old_tc_end = '''            "total_rules": int(len(rules)),
        },
        "publication_timing_audit": publication_timing_audit(),'''

new_tc_end = '''            "total_rules": int(len(rules)),
        },
        "execution_calendar": load_execution_calendar(str(ROOT / CALENDAR_PATH)).facts(),
        "publication_timing_audit": publication_timing_audit(),'''

content = content.replace(old_tc_end, new_tc_end)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("run_core_satellite_momentum.py patched. New length:", len(content))

# Patch run_low_turnover_core_satellite.py similarly
src2 = 'src/run_low_turnover_core_satellite.py'
with open(src2, 'r', encoding='utf-8') as f:
    content2 = f.read()

content2 = content2.replace(
    'from otf_rotation.schedule import build_month_end_schedule, build_signal_submit_map\n',
    'from otf_rotation.execution_calendar import load_execution_calendar\nfrom otf_rotation.schedule import build_month_end_schedule, build_signal_submit_map\n'
)

content2 = content2.replace(
    'DB_PATH = "data/processed/otf_expanded.sqlite"\n',
    'DB_PATH = "data/processed/otf_expanded.sqlite"\nCALENDAR_PATH = "data/processed/execution_calendar/cn_execution_calendar.csv"\n'
)

content2 = content2.replace(
    '''        "input_hashes": {
            "db_sha256": sha256_file(str(ROOT / DB_PATH)),
            "rules_sha256": sha256_file(str(ROOT / RULES_PATH)),
            "mapping_sha256": sha256_file(str(ROOT / MAPPING_PATH)),
            "config_sha256": sha256_file(str(ROOT / CONFIG_PATH)),
        },''',
    '''        "input_hashes": {
            "db_sha256": sha256_file(str(ROOT / DB_PATH)),
            "rules_sha256": sha256_file(str(ROOT / RULES_PATH)),
            "mapping_sha256": sha256_file(str(ROOT / MAPPING_PATH)),
            "config_sha256": sha256_file(str(ROOT / CONFIG_PATH)),
            "calendar_sha256": sha256_file(str(ROOT / CALENDAR_PATH)),
        },'''
)

# Add calendar facts after rule_temporal_coverage in C2 runner
old_tc_end2 = '''            "total_rules": int(len(rules)),
        },
        "publication_timing_audit": publication_timing_audit(),'''

new_tc_end2 = '''            "total_rules": int(len(rules)),
        },
        "execution_calendar": load_execution_calendar(str(ROOT / CALENDAR_PATH)).facts(),
        "publication_timing_audit": publication_timing_audit(),'''

content2 = content2.replace(old_tc_end2, new_tc_end2)

with open(src2, 'w', encoding='utf-8') as f:
    f.write(content2)

print("run_low_turnover_core_satellite.py patched. New length:", len(content2))