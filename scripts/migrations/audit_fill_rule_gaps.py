"""
§5-3: 补齐核心袖套候选的产品级执行规则
目标：每个核心袖套至少有2个可交易备选
"""
import sqlite3
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("reports/data_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load existing rules
existing_rules = pd.read_csv("config/otf_product_rules.csv", dtype=str)
existing_codes = set(existing_rules["fund_code"].tolist())
print(f"当前已有产品规则: {len(existing_rules)} 条")

# Load 448 research catalog
conn = sqlite3.connect("data/processed/otf_research.sqlite")
catalog = pd.read_sql("SELECT * FROM otf_fund_catalog", conn)
conn.close()
print(f"448候选池: {len(catalog)} 只基金")

# ================================================================
# §5-3a: 按核心袖套分析规则缺口
# ================================================================

CORE_SLEEVE_CONFIGS = {
    "CSI300": {"keywords": ["沪深300"], "exclude": ["红利", "低波", "增强"], "min": 3},
    "CSI500": {"keywords": ["中证500"], "exclude": ["红利", "增强"], "min": 3},
    "CSI1000": {"keywords": ["中证1000"], "exclude": [], "min": 2},
    "CHINEXT": {"keywords": ["创业板"], "exclude": ["红利", "价值"], "min": 2},
    "DIVIDEND": {"keywords": ["红利", "低波"], "exclude": ["恒生", "港股"], "min": 2},
    "GOLD": {"keywords": ["黄金"], "exclude": [], "min": 3},
    "NASDAQ": {"keywords": ["纳斯达克", "纳指"], "exclude": [], "min": 2},
    "SP500": {"keywords": ["标普500", "标普500"], "exclude": [], "min": 2},
    "HANGSENG": {"keywords": ["恒生"], "exclude": ["科技", "红利", "消费", "医药", "互联网"], "min": 2},
    "MONEY_MARKET": {"keywords": ["货币"], "exclude": [], "min": 3},
    "ULTRA_SHORT_BOND": {"keywords": ["超短债"], "exclude": [], "min": 2},
    "GOV_BOND_1_3Y": {"keywords": ["1-3年", "1_3年", "1~3年"], "exclude": [], "min": 2},
    "GOV_BOND_3_5Y": {"keywords": ["3-5年", "3_5年", "3~5年", "国债", "国开", "政金", "农发"], "exclude": ["1-3", "1_3", "1~3", "信用", "企业", "短债"], "min": 2},
    "CREDIT_BOND": {"keywords": ["信用债", "信用"], "exclude": ["国债", "政金", "国开", "短债"], "min": 2},
    "CD_NCD": {"keywords": ["同业存单"], "exclude": [], "min": 1},
}

def match_core_sleeve(name):
    name_l = str(name).lower()
    matched = []
    for sleeve, cfg in CORE_SLEEVE_CONFIGS.items():
        kw_match = any(kw.lower() in name_l for kw in cfg["keywords"])
        exclude_match = any(ex.lower() in name_l for ex in cfg["exclude"])
        if kw_match and not exclude_match:
            matched.append(sleeve)
    return matched

# Build sleeve mapping for all 448 candidates
catalog["matched_sleeves"] = catalog["fund_name"].apply(match_core_sleeve)

# Check gaps: for each sleeve, which candidates already have rules?
sleeve_gap_report = []
for sleeve, cfg in CORE_SLEEVE_CONFIGS.items():
    candidates = catalog[catalog["matched_sleeves"].apply(lambda x: sleeve in x)]
    candidates_with_rules = candidates[candidates["fund_code"].isin(existing_codes)]
    candidates_without_rules = candidates[~candidates["fund_code"].isin(existing_codes)]

    sleeve_gap_report.append({
        "sleeve": sleeve,
        "total_candidates": len(candidates),
        "with_rules": len(candidates_with_rules),
        "without_rules": len(candidates_without_rules),
        "target_min": cfg["min"],
        "gap": max(0, cfg["min"] - len(candidates_with_rules)),
        "candidates_without_rules_codes": ",".join(candidates_without_rules.head(5)["fund_code"].tolist()),
        "candidates_without_rules_names": ",".join(candidates_without_rules.head(5)["fund_name"].tolist()),
    })

gap_df = pd.DataFrame(sleeve_gap_report)
gap_path = OUTPUT_DIR / "otf_448_rule_gap_analysis.csv"
gap_df.to_csv(gap_path, index=False)
print(f"\n规则缺口分析: {gap_path}")
print(f"{'袖套':25s} {'候选':>4s} {'已有规则':>6s} {'缺口':>4s} {'目标':>4s}")
for _, row in gap_df.iterrows():
    flag = " ***GAP***" if row["gap"] > 0 else ""
    print(f"  {row['sleeve']:25s} {int(row['total_candidates']):4d} {int(row['with_rules']):6d} {int(row['gap']):4d} {int(row['target_min']):4d}{flag}")

# ================================================================
# §5-3b: 为缺口袖套生成产品规则
# ================================================================

# Default rule values by asset type
SLEEVE_RULE_DEFAULTS = {
    "CSI300": dict(sub_fee=0.012, confirm=1, settle=7, min_hold=0, max_amt=""),
    "CSI500": dict(sub_fee=0.012, confirm=1, settle=7, min_hold=0, max_amt=""),
    "CSI1000": dict(sub_fee=0.012, confirm=1, settle=7, min_hold=0, max_amt=""),
    "CHINEXT": dict(sub_fee=0.008, confirm=1, settle=7, min_hold=0, max_amt=""),
    "DIVIDEND": dict(sub_fee=0.012, confirm=1, settle=7, min_hold=0, max_amt=""),
    "GOLD": dict(sub_fee=0.007, confirm=1, settle=7, min_hold=0, max_amt=""),
    "NASDAQ": dict(sub_fee=0.012, confirm=2, settle=7, min_hold=0, max_amt=""),
    "SP500": dict(sub_fee=0.012, confirm=2, settle=7, min_hold=0, max_amt=""),
    "HANGSENG": dict(sub_fee=0.012, confirm=1, settle=7, min_hold=0, max_amt=""),
    "MONEY_MARKET": dict(sub_fee=0, confirm=1, settle=1, min_hold=0, max_amt=""),
    "ULTRA_SHORT_BOND": dict(sub_fee=0, confirm=1, settle=7, min_hold=0, max_amt=""),
    "GOV_BOND_1_3Y": dict(sub_fee=0, confirm=1, settle=7, min_hold=0, max_amt=""),
    "GOV_BOND_3_5Y": dict(sub_fee=0.008, confirm=1, settle=7, min_hold=0, max_amt=""),
    "CREDIT_BOND": dict(sub_fee=0, confirm=1, settle=7, min_hold=0, max_amt=""),
    "CD_NCD": dict(sub_fee=0, confirm=1, settle=7, min_hold=7, max_amt=""),
}

new_rules = []
for _, row in gap_df.iterrows():
    sleeve = row["sleeve"]
    gap = int(row["gap"])
    if gap <= 0:
        continue
    # Pick best candidates without rules
    candidates_in_sleeve = catalog[catalog["matched_sleeves"].apply(lambda x: sleeve in x)]
    candidates_needed = candidates_in_sleeve[~candidates_in_sleeve["fund_code"].isin(existing_codes)].head(gap)

    defaults = SLEEVE_RULE_DEFAULTS.get(sleeve, {})
    for _, fund in candidates_needed.iterrows():
        existing_codes.add(fund["fund_code"])
        new_rules.append({
            "fund_code": fund["fund_code"],
            "fund_name": fund["fund_name"],
            "subscription_fee_rate": defaults.get("sub_fee", 0.01),
            "subscription_confirmation_days": defaults.get("confirm", 1),
            "redemption_confirmation_days": defaults.get("confirm", 1),
            "redemption_settlement_days": defaults.get("settle", 7),
            "minimum_holding_calendar_days": defaults.get("min_hold", 0),
            "maximum_subscription_amount_per_order": defaults.get("max_amt", ""),
            "subscription_open": 1,
            "redemption_open": 1,
            "rule_status": "CONSERVATIVE_ASSUMPTION",
            "official_source": "",
        })

if new_rules:
    new_rules_df = pd.DataFrame(new_rules)
    new_rules_path = OUTPUT_DIR / "otf_new_core_rules_to_add.csv"
    new_rules_df.to_csv(new_rules_path, index=False)
    print(f"\n新规则生成: {new_rules_path}")
    print(f"  新增规则数量: {len(new_rules_df)}")
    for _, r in new_rules_df.iterrows():
        print(f"  {r['fund_code']} {r['fund_name']} ({r['subscription_fee_rate']}费率, {r['subscription_confirmation_days']}确认)")
else:
    print("\n  没有需要新增的规则")

# ================================================================
# §5-3c: 最终覆盖统计
# ================================================================
print(f"\n最终规则覆盖统计:")
combined_codes = set(existing_codes) | set(r["fund_code"] for r in new_rules)
for sleeve, cfg in CORE_SLEEVE_CONFIGS.items():
    candidates = catalog[catalog["matched_sleeves"].apply(lambda x: sleeve in x)]
    covered = candidates[candidates["fund_code"].isin(combined_codes)]
    families = candidates[candidates["fund_code"].isin(combined_codes)]["fund_family"].nunique()
    status = "OK" if (len(covered) >= cfg["min"] and families >= 2) else "GAP"
    print(f"  [{status}] {sleeve:25s}: {len(covered)}/{cfg['min']} 有规则, {families} 个家族")

print(f"\n§5-3 完成!")
