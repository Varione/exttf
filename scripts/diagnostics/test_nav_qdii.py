import sys, pandas as pd
sys.path.insert(0, "src")
from otf_backtest_engine import OTFBacktestEngine
from otf_rotation.execution_calendar import load_execution_calendar
from otf_trading_rules import ProductRuleBook

calendar = load_execution_calendar("data/processed/execution_calendar/cn_execution_calendar.csv")
rule_book = ProductRuleBook.from_csv("config/otf_product_rules.csv")
engine = OTFBacktestEngine(
    db_path="data/processed/otf_expanded.sqlite",
    initial_cash=1_000_000.0,
    strict_product_rules=True,
    product_rule_book=rule_book,
    confirmation_days_subscribe=1,
    confirmation_days_redeem=1,
    settlement_days_redeem=7,
    subscription_fee_rate=0.001,
    redemption_fee_rate=0.0015,
    minimum_trade_ratio=0.005,
    execution_calendar=calendar,
)

# Find a QDII fund and check NAV around 2022-12-31
qdii_funds = [f for f in engine.available_fund_codes if engine.is_qdii(f)]
print("QDII funds found:", len(qdii_funds))
if qdii_funds:
    fund = qdii_funds[0]
    print(f"Testing QDII fund: {fund}")
    
    # Get indices for dates around 2022-12-31
    dti = engine._date_to_index
    dec_30_idx = dti.get(pd.Timestamp("2022-12-30"))
    jan_3_idx = dti.get(pd.Timestamp("2023-01-03"))
    
    print(f"2022-12-30 idx: {dec_30_idx}, NAV: {engine.get_nav(fund, dec_30_idx)}, ValNAV: {engine.get_valuation_nav(fund, dec_30_idx)}")
    print(f"2023-01-03 idx: {jan_3_idx}, NAV: {engine.get_nav(fund, jan_3_idx)}, ValNAV: {engine.get_valuation_nav(fund, jan_3_idx)}")
    
    # Check if there is NAV on 2022-12-31 in valuation source dates
    nav_dates = set(engine._valuation_source_dates.strftime('%Y-%m-%d'))
    print(f"2022-12-31 in valuation_source_dates: {'YES' if '2022-12-31' in nav_dates else 'NO'}")

print("NAV lookup verification done!")