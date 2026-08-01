import sys
sys.path.insert(0, "src")
from otf_rotation.execution_calendar import load_execution_calendar
calendar = load_execution_calendar("data/processed/execution_calendar/cn_execution_calendar.csv")
print("Calendar loaded:", len(calendar.dates), "dates")
print("Coverage:", calendar.min_date, "~", calendar.max_date)

# Check that problem dates are NOT in calendar
problem_dates = ['2022-03-05','2022-03-20','2022-12-31','2023-01-02','2023-05-02','2023-12-31','2024-06-30']
cal_set = set(calendar.dates.strftime('%Y-%m-%d'))
for d in problem_dates:
    print(f"{d}: {'PRESENT' if d in cal_set else 'ABSENT'}")
print("Calendar facts keys:", list(calendar.facts().keys()))