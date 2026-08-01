"""
§5-1: 14只成交基金来源审计表
§5-2: 448候选家族/暴露去重 + 规则完整度矩阵
"""
import sqlite3
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("reports/data_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# §5-1: 14只成交基金审计表
# ================================================================

TRADED_FUNDS = {
    "160706": ("嘉实沪深300ETF联接A", "CSI300", "进攻型·国内权益"),
    "000008": ("嘉实中证500ETF联接A", "CSI500", "进攻型·国内权益"),
    "016633": ("富国中证1000ETF联接A", "CSI1000", "进攻型·国内权益"),
    "050021": ("博时创业板ETF联接A", "CHINEXT", "进攻型·国内权益"),
    "007466": ("华泰柏瑞中证红利低波ETF联接A", "DIVIDEND", "进攻型·国内权益"),
    "000218": ("国泰黄金ETF联接A", "GOLD", "进攻型·实物资产"),
    "021778": ("广发纳指100ETF联接F", "NASDAQ", "进攻型·海外权益"),
    "050025": ("博时标普500ETF联接A", "SP500", "进攻型·海外权益"),
    "000071": ("华夏恒生ETF联接A", "HANGSENG", "进攻型·海外权益"),
    "260102": ("景顺长城货币A", "MONEY", "防御型·现金管理"),
    "006663": ("易方达安悦超短债C", "ULTRA_SHORT_BOND", "防御型·短债"),
    "001512": ("易方达中债3-5年国债指数", "GOV_BOND_3_5Y", "防御型·利率债"),
    "005839": ("创金合信中债1-3年政金债C", "POLICY_BANK_BOND_1_3Y", "防御型·利率债"),
    "000148": ("易方达高等级信用债债券C", "CREDIT_BOND", "防御型·信用债"),
}

# Load product rules
rules_df = pd.read_csv("config/otf_product_rules.csv", dtype=str)
rules_index = rules_df.set_index("fund_code")

# Load OTF db for NAV stats
conn = sqlite3.connect("data/processed/otf.sqlite")
nav_df = pd.read_sql("SELECT fund_code, nav_date FROM otf_fund_nav WHERE fund_code IS NOT NULL", conn)
conn.close()
nav_df.columns = ["code", "trade_date"]
nav_df["trade_date"] = pd.to_datetime(nav_df["trade_date"])

rows = []
for code, (name, sleeve, sleeve_type) in TRADED_FUNDS.items():
    fund_nav = nav_df[nav_df["code"] == code]
    first_trade = fund_nav["trade_date"].min()
    last_trade = fund_nav["trade_date"].max()
    trade_days = len(fund_nav)
    rule_info = rules_index.loc[code] if code in rules_index.index else None
    rule_status = rule_info["rule_status"] if rule_info is not None else "NO_RULES"
    official_source = rule_info["official_source"] if rule_info is not None else ""
    sub_fee = rule_info["subscription_fee_rate"] if rule_info is not None else "N/A"

    rows.append({
        "fund_code": code,
        "fund_name": name,
        "asset_sleeve": sleeve,
        "sleeve_category": sleeve_type,
        "first_nav_date": first_trade.strftime("%Y-%m-%d") if pd.notna(first_trade) else "N/A",
        "last_nav_date": last_trade.strftime("%Y-%m-%d") if pd.notna(last_trade) else "N/A",
        "total_nav_days": trade_days,
        "rule_status": rule_status,
        "subscription_fee_rate": sub_fee,
        "rule_source_official": official_source[:80] if official_source else "",
        "strategy_source": "mapped_otf_strategy._select_core_sleeves()" if sleeve_type.startswith("进攻") else "all_otf_allocation.DEFENSIVE_SLEEVES",
    })

audit_df = pd.DataFrame(rows)
audit_path = OUTPUT_DIR / "otf_14funds_audit_table.csv"
audit_df.to_csv(audit_path, index=False)
print(f"§5-1 完成: {audit_path}")
print(f"  共 {len(audit_df)} 只基金")
print(f"  进攻型: {len(audit_df[audit_df['sleeve_category'].str.startswith('进攻')])} 只")
print(f"  防御型: {len(audit_df[audit_df['sleeve_category'].str.startswith('防御')])} 只")
print(f"  规则覆盖率: {(audit_df['rule_status'] != 'NO_RULES').sum()}/{len(audit_df)}")
print()

# ================================================================
# §5-2: 448候选基金家族/暴露去重矩阵
# ================================================================

print("=" * 60)
print("§5-2: 448候选基金家族/暴露去重")
print("=" * 60)

conn = sqlite3.connect("data/processed/otf_research.sqlite")
catalog_df = pd.read_sql("SELECT * FROM otf_fund_catalog", conn)
nav_info = pd.read_sql("""
    SELECT fund_code, COUNT(*) as nav_records,
           MIN(nav_date) as first_date, MAX(nav_date) as last_date
    FROM otf_fund_nav GROUP BY fund_code
""", conn)
conn.close()

# Merge catalog with nav stats
merged = catalog_df.merge(nav_info, left_on="fund_code", right_on="fund_code", how="left")

# Determine fund family (基金家族) from fund_name
def extract_family(name):
    if pd.isna(name):
        return "UNKNOWN"
    name = str(name)
    for kw in [
        "易方达", "华夏", "嘉实", "博时", "南方", "广发", "富国",
        "华安", "天弘", "汇添富", "招商", "景顺长城", "大成",
        "创金合信", "华泰柏瑞", "国泰", "鹏华", "工银", "建信",
        "交银", "中欧", "兴全", "万家", "浦银安盛", "上投摩根",
        "摩根", "平安", "国联安", "国投瑞银", "长城", "银华",
        "长信", "光大保德信", "申万菱信", "前海开源", "民生加银",
        "中银", "银河", "诺安", "海富通", "华宝", "国寿安保",
    ]:
        if name.startswith(kw):
            return kw
    return name[:4]

merged["fund_family"] = merged["fund_name"].apply(extract_family)

# Build sleeve mapping based on asset_category and fund_name
def map_sleeve(row):
    cat = str(row.get("asset_class", "")).lower()
    name = str(row.get("fund_name", ""))
    idx_name = str(row.get("underlying_name", ""))

    if "货币" in cat or "money" in cat:
        return "MONEY_MARKET"
    if "超短债" in name or "短债" in cat:
        return "ULTRA_SHORT_BOND"
    if "黄金" in name or "gold" in cat:
        return "GOLD"
    if "纳斯达克" in name or "纳指" in name or "nasdaq" in cat:
        return "NASDAQ"
    if "标普" in name or "sp500" in cat or "s&p" in cat:
        return "SP500"
    if "恒生" in name and "科技" not in name and "红利" not in name:
        return "HANGSENG"
    if "沪深300" in name or "csi300" in cat:
        return "CSI300"
    if "中证500" in name or "csi500" in cat:
        return "CSI500"
    if "中证1000" in name or "csi1000" in cat:
        return "CSI1000"
    if "创业板" in name or "chinext" in cat:
        return "CHINEXT"
    if "红利" in name or "dividend" in cat or "低波" in name:
        return "DIVIDEND"
    if "上证50" in name or "中证100" in name:
        return "CSI100_LARGE_CAP"
    if "国债" in cat or "国开" in cat or "农发" in cat or "口行" in cat or "政金" in cat:
        return "GOVERNMENT_BOND"
    if "信用" in cat or "企业" in cat or "公司" in cat:
        return "CREDIT_BOND"
    if "同业存单" in name:
        return "CD_NCD"
    if "消费" in name or "医药" in name or "医疗" in name or "科技" in name or "信息" in name:
        return "THEME_SECTOR"
    if "银行" in name or "金融" in name or "地产" in name or "证券" in name:
        return "THEME_SECTOR"
    if "中短债" in name or "纯债" in name:
        return "GOVERNMENT_BOND"
    return f"OTHER_{cat[:20]}"

merged["asset_sleeve"] = merged.apply(map_sleeve, axis=1)

# Load product rules to check rule completeness
rules_codes = set(rules_df["fund_code"].tolist())
merged["has_product_rules"] = merged["fund_code"].isin(rules_codes)

# ================================================================
# §5-2a: 家族去重统计
# ================================================================
family_stats = merged.groupby("fund_family").agg(
    fund_count=("fund_code", "count"),
    funds_with_rules=("has_product_rules", "sum"),
    sleeves_covered=("asset_sleeve", lambda x: x.nunique()),
).sort_values("fund_count", ascending=False)
family_stats_path = OUTPUT_DIR / "otf_448_family_dedup.csv"
family_stats.to_csv(family_stats_path)
print(f"家族去重: {family_stats_path}")
print(f"  总基金家族数: {len(family_stats)}")
print(f"  前10大家族:")
for fam, row in family_stats.head(10).iterrows():
    print(f"    {fam}: {int(row['fund_count'])}只基金, {int(row['funds_with_rules'])}只有规则, {int(row['sleeves_covered'])}个袖套")
print()

# ================================================================
# §5-2b: 资产袖套—候选产品—规则完整度矩阵
# ================================================================
sleeve_matrix = merged.groupby("asset_sleeve").agg(
    total_candidates=("fund_code", "count"),
    unique_families=("fund_family", "nunique"),
    candidates_with_rules=("has_product_rules", "sum"),
    candidates_no_rules=("has_product_rules", lambda x: (~x).sum()),
    sample_funds=("fund_name", lambda x: ", ".join(x.head(5).tolist())),
).sort_values("total_candidates", ascending=False)

sleeve_matrix["rule_coverage_pct"] = (sleeve_matrix["candidates_with_rules"] / sleeve_matrix["total_candidates"] * 100).round(1)
sleeve_matrix_path = OUTPUT_DIR / "otf_448_sleeve_matrix.csv"
sleeve_matrix.to_csv(sleeve_matrix_path)
print(f"袖套矩阵: {sleeve_matrix_path}")
print(f"  总袖套数: {len(sleeve_matrix)}")
for sleeve, row in sleeve_matrix.iterrows():
    print(f"  {sleeve:30s}: {int(row['total_candidates']):3d}候选, {int(row['candidates_with_rules']):2d}有规则, {int(row['unique_families']):2d}家族 → {row['rule_coverage_pct']:.0f}%")
print()

# ================================================================
# §5-2c: 核心袖套（进攻型9个+防御型5个）候选与规则分析
# ================================================================
CORE_SLEEVES = [
    "CSI300", "CSI500", "CSI1000", "CHINEXT", "DIVIDEND",
    "GOLD", "NASDAQ", "SP500", "HANGSENG",
    "MONEY_MARKET", "ULTRA_SHORT_BOND", "GOVERNMENT_BOND",
    "CREDIT_BOND", "CD_NCD",
]

core_analysis = merged[merged["asset_sleeve"].isin(CORE_SLEEVES)].copy()
core_matrix = core_analysis.groupby("asset_sleeve").agg(
    total_candidates=("fund_code", "count"),
    unique_families=("fund_family", "nunique"),
    funds_with_rules=("has_product_rules", "sum"),
).sort_values("total_candidates", ascending=False)

core_matrix["need_more_candidates"] = core_matrix["total_candidates"] < 3
core_matrix["need_more_rules"] = core_matrix["funds_with_rules"] < 2
core_matrix_path = OUTPUT_DIR / "otf_448_core_sleeve_analysis.csv"
core_matrix.to_csv(core_matrix_path)
print(f"核心袖套矩阵: {core_matrix_path}")
print(f"  核心袖套数: {len(core_matrix)}")
for sleeve, row in core_matrix.iterrows():
    flag = ""
    if row["need_more_candidates"]:
        flag += " ⚠候选<3"
    if row["need_more_rules"]:
        flag += " ⚠规则<2"
    print(f"  {sleeve:30s}: {int(row['total_candidates']):3d}候选, {int(row['funds_with_rules']):2d}有规则, {int(row['unique_families']):2d}家族{flag}")

print()
print("=" * 60)
print("完成! 所有审计文件输出到:", OUTPUT_DIR.resolve())
print("=" * 60)
