src = 'src/run_b1_b2_b3_walkforward.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# Add import for execution_calendar after existing imports from otf_rotation
content = content.replace(
    'from otf_rotation.schedule import build_month_end_schedule, build_signal_submit_map\n',
    'from otf_rotation.execution_calendar import load_execution_calendar\nfrom otf_rotation.schedule import build_month_end_schedule, build_signal_submit_map\n'
)

# Add CALENDAR_PATH constant after DB_PATH
content = content.replace(
    'DB_PATH = "data/processed/otf_expanded.sqlite"\n',
    'DB_PATH = "data/processed/otf_expanded.sqlite"\nCALENDAR_PATH = "data/processed/execution_calendar/cn_execution_calendar.csv"\n'
)

# Modify create_engine to load and pass calendar
old_create_engine = '''def create_engine(rule_book: ProductRuleBook) -> OTFBacktestEngine:
    return OTFBacktestEngine(
        db_path=DB_PATH,
        initial_cash=INITIAL_CASH,
        strict_product_rules=True,
        product_rule_book=rule_book,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=7,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        minimum_trade_ratio=0.005,
    )'''

new_create_engine = '''def create_engine(rule_book: ProductRuleBook) -> OTFBacktestEngine:
    calendar = load_execution_calendar(CALENDAR_PATH)
    return OTFBacktestEngine(
        db_path=DB_PATH,
        initial_cash=INITIAL_CASH,
        strict_product_rules=True,
        product_rule_book=rule_book,
        confirmation_days_subscribe=1,
        confirmation_days_redeem=1,
        settlement_days_redeem=7,
        subscription_fee_rate=0.001,
        redemption_fee_rate=0.0015,
        minimum_trade_ratio=0.005,
        execution_calendar=calendar,
    )'''

content = content.replace(old_create_engine, new_create_engine)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("run_b1_b2_b3_walkforward.py patched. New length:", len(content))