src = 'src/run_b1_b2_b3_walkforward.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# Update collect_input_facts to include calendar facts
old_collect = '''def collect_input_facts() -> dict[str, Any]:
    rules = pd.read_csv(RULES_PATH, dtype=str).fillna("")
    mapping = pd.read_csv(MAPPING_PATH, dtype=str).fillna("")
    input_paths = {
        "db_sha256": DB_PATH,
        "rules_sha256": RULES_PATH,
        "mapping_sha256": MAPPING_PATH,
        "config_sha256": CONFIG_PATH,
    }
    hashes = {key: sha256_file(path) for key, path in input_paths.items()}'''

new_collect = '''def collect_input_facts() -> dict[str, Any]:
    rules = pd.read_csv(RULES_PATH, dtype=str).fillna("")
    mapping = pd.read_csv(MAPPING_PATH, dtype=str).fillna("")
    calendar = load_execution_calendar(CALENDAR_PATH)
    input_paths = {
        "db_sha256": DB_PATH,
        "rules_sha256": RULES_PATH,
        "mapping_sha256": MAPPING_PATH,
        "config_sha256": CONFIG_PATH,
        "calendar_sha256": CALENDAR_PATH,
    }
    hashes = {key: sha256_file(path) for key, path in input_paths.items()}'''

content = content.replace(old_collect, new_collect)

# Add calendar facts to the return dict
old_return = '''        "rule_temporal_coverage": {
            "effective_from_present": int((rules["effective_from"].str.strip() != "").sum()),
            "verified_at_present": int((rules["verified_at"].str.strip() != "").sum()),
            "total_rules": int(len(rules)),
        },
    }'''

new_return = '''        "rule_temporal_coverage": {
            "effective_from_present": int((rules["effective_from"].str.strip() != "").sum()),
            "verified_at_present": int((rules["verified_at"].str.strip() != "").sum()),
            "total_rules": int(len(rules)),
        },
        "execution_calendar": calendar.facts(),
    }'''

content = content.replace(old_return, new_return)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("collect_input_facts patched. New length:", len(content))