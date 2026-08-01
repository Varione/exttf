src = 'tests/test_execution_calendar_integration.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# Update error name from C2_NOT_EXECUTION_DATE to OTF_NOT_EXECUTION_DATE
content = content.replace('C2_NOT_EXECUTION_DATE.*2022-12-31', 'OTF_NOT_EXECUTION_DATE.*2022-12-31')

# Add new test class for run_backtest target validation
new_tests = '''

class TestRunBacktestTargetValidation:
    """Verify run_backtest blocks targets on non-execution dates."""

    def test_target_on_holiday_blocked(self, engine_with_calendar):
        """Target with non-NaN weights on 2022-12-31 must raise RuntimeError."""
        fund_code = engine_with_calendar.available_fund_codes[0]
        targets = pd.DataFrame(
            [[1.0]],
            index=pd.to_datetime(["2022-12-31"]),
            columns=[fund_code],
        )
        with pytest.raises(RuntimeError, match="OTF_TARGET_NOT_EXECUTION_DATE.*2022-12-31"):
            engine_with_calendar.run_backtest(
                targets,
                start="2022-12-01",
                end="2023-01-15",
                rebalance_every=1,
            )

    def test_target_on_valid_execution_date_allowed(self, engine_with_calendar):
        """Target on valid execution date should not raise."""
        fund_code = engine_with_calendar.available_fund_codes[0]
        targets = pd.DataFrame(
            [[1.0]],
            index=pd.to_datetime(["2023-01-03"]),
            columns=[fund_code],
        )
        daily = engine_with_calendar.run_backtest(
            targets,
            start="2022-12-01",
            end="2023-01-15",
            rebalance_every=1,
        )
        assert len(daily) > 0

    def test_target_nan_on_holiday_allowed(self, engine_with_calendar):
        """Target with all-NaN on holiday is allowed (no signal)."""
        fund_code = engine_with_calendar.available_fund_codes[0]
        targets = pd.DataFrame(
            [[float("nan")]],
            index=pd.to_datetime(["2022-12-31"]),
            columns=[fund_code],
        )
        # Should not raise since all weights are NaN (no signal on that date)
        daily = engine_with_calendar.run_backtest(
            targets,
            start="2022-12-01",
            end="2023-01-15",
            rebalance_every=1,
        )
        assert len(daily) > 0

'''

# Insert new tests before the final if __name__ block
content = content.replace(
    '\nif __name__ == "__main__":',
    new_tests + '\nif __name__ == "__main__":'
)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("Test file updated. New length:", len(content))