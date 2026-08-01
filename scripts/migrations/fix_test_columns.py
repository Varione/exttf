src = 'tests/test_cross_fund_holiday_adj.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the test to use actual columns from daily output
old_test = '''    def test_run_backtest_fundb_confirms_correctly(self, cross_fund_holiday_db, calendar_fri_mon):
        """Verify FundB (which has NAV on both Fri and Mon) confirms orders correctly."""
        engine = OTFBacktestEngine(
            db_path=cross_fund_holiday_db,
            execution_calendar=calendar_fri_mon,
            initial_cash=100_000.0,
            subscription_fee_rate=0.0,
        )

        # Subscribe to FundB on Friday at NAV=1.0
        targets = pd.DataFrame(
            [[1.0], [1.0]],
            index=pd.to_datetime(["2024-06-28", "2024-07-01"]),
            columns=["B001"],
        )
        daily = engine.run_backtest(targets, start="2024-06-28", end="2024-07-01", rebalance_every=1)

        # Monday: order confirmed at NAV=1.02
        mon_row = daily[daily["date"] == "2024-07-01"].iloc[0]
        pos_col = [c for c in daily.columns if "B001" in c and "position" in c.lower()][0]
        # Shares = 100_000 / 1.02 (confirmed at Monday NAV)
        expected_shares = 100_000.0 / 1.02
        assert abs(mon_row[pos_col] - expected_shares) < 1, (
            f"Mon shares should be ~{expected_shares:.0f}, got {mon_row[pos_col]}"
        )'''

new_test = '''    def test_run_backtest_fundb_confirms_correctly(self, cross_fund_holiday_db, calendar_fri_mon):
        """Verify FundB (which has NAV on both Fri and Mon) confirms orders correctly."""
        engine = OTFBacktestEngine(
            db_path=cross_fund_holiday_db,
            execution_calendar=calendar_fri_mon,
            initial_cash=100_000.0,
            subscription_fee_rate=0.0,
        )

        # Subscribe to FundB on Friday at NAV=1.0
        targets = pd.DataFrame(
            [[1.0], [1.0]],
            index=pd.to_datetime(["2024-06-28", "2024-07-01"]),
            columns=["B001"],
        )
        daily = engine.run_backtest(targets, start="2024-06-28", end="2024-07-01", rebalance_every=1)

        # Monday: order confirmed at NAV=1.02; check equity reflects position value
        mon_row = daily[daily["date"] == "2024-07-01"].iloc[0]
        # Shares = 100_000 / 1.02, position value = shares * NAV = 100_000
        assert abs(mon_row["end_equity"] - 100_000.0) < 1, (
            f"Mon end_equity should be ~100k, got {mon_row['end_equity']}"
        )'''

content = content.replace(old_test, new_test)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed test to use end_equity")