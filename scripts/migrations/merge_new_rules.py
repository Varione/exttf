"""
合并新候选规则到 config/otf_product_rules.csv
"""
import pandas as pd

# Load existing rules
existing = pd.read_csv("config/otf_product_rules.csv", dtype=str)
print(f"现有规则: {len(existing)}")

# Load new rules to add
new_rules = pd.read_csv("reports/data_validation/otf_new_core_rules_to_add.csv", dtype=str)
print(f"新规则: {len(new_rules)}")

# Filter out duplicates
new_codes = set(new_rules["fund_code"])
existing_codes = set(existing["fund_code"])
actually_new = new_rules[~new_rules["fund_code"].isin(existing_codes)]
print(f"实际新增（去重后）: {len(actually_new)}")

# Append and sort
combined = pd.concat([existing, actually_new], ignore_index=True)
combined = combined.drop_duplicates(subset=["fund_code"], keep="last")
combined = combined.sort_values(["fund_code"]).reset_index(drop=True)

# Save back
combined.to_csv("config/otf_product_rules.csv", index=False)
print(f"最终规则总数: {len(combined)}")

# Print fund codes and names
for _, r in actually_new.iterrows():
    print(f"  + {r['fund_code']} {r['fund_name']} ({r['rule_status']})")
