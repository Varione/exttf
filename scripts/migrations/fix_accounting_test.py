src = 'tests/test_holiday_share_adjustment.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the accounting test - insufficient cash for fee
old_test = '''        engine = OTFBacktestEngine(
            db_path=holiday_gap_db,
            execution_calendar=calendar_excluding_holiday,
            initial_cash=100_000.0,
        )

        fund_code = "H001"
        idx_t = engine._date_to_index[pd.Timestamp("2024-06-28")]
        idx_tp1 = engine._date_to_index[pd.Timestamp("2024-06-30")]

        # Subscribe on T at NAV=1.0, get 100_000 shares
        order = engine.submit_order(
            order_id="test-hol-adj",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=pd.Timestamp("2024-06-28"),
            submit_date=pd.Timestamp("2024-06-28"),
            requested_amount=100_000.0,
            available_cash=100_000.0,'''

new_test = '''        engine = OTFBacktestEngine(
            db_path=holiday_gap_db,
            execution_calendar=calendar_excluding_holiday,
            initial_cash=200_000.0,
            subscription_fee_rate=0.0,  # No fee to simplify test
        )

        fund_code = "H001"
        idx_t = engine._date_to_index[pd.Timestamp("2024-06-28")]
        idx_tp1 = engine._date_to_index[pd.Timestamp("2024-06-30")]

        # Subscribe on T at NAV=1.0, get 100_000 shares (fee=0)
        order = engine.submit_order(
            order_id="test-hol-adj",
            side=OrderSide.SUBSCRIBE,
            fund_code=fund_code,
            signal_date=pd.Timestamp("2024-06-28"),
            submit_date=pd.Timestamp("2024-06-28"),
            requested_amount=100_000.0,
            available_cash=200_000.0,'''

content = content.replace(old_test, new_test)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed accounting test")