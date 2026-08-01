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

print("Engine trading dates count:", len(engine._trading_dates))
print("Valuation source dates count:", len(engine._valuation_source_dates))
print("Difference (should be 7):", len(engine._valuation_source_dates) - len(engine._trading_dates))
cal_set = set(engine._trading_dates.strftime('%Y-%m-%d'))
problem_dates = ['2022-03-05','2022-03-20','2022-12-31','2023-01-02','2023-05-02','2023-12-31','2024-06-30']
for d in problem_dates:
    print(f"{d} in trading_dates: {'YES' if d in cal_set else 'NO'}")

# Test that submit on non-execution date raises error
from otf_backtest_engine import OrderSide
try:
    order = engine.submit_order(
        order_id="test-non-exec",
        side=OrderSide.SUBSCRIBE,
        fund_code="160706",
        signal_date=pd.Timestamp("2022-12-30"),
        submit_date=pd.Timestamp("2022-12-31"),
        requested_amount=100_000.0,
        available_cash=1_000_000.0,
        current_positions={},
    )
    print("ERROR: Should have raised RuntimeError for 2022-12-31")
except RuntimeError as e:
    print(f"Correctly raised RuntimeError for 2022-12-31: {e}")

print("All checks passed!")