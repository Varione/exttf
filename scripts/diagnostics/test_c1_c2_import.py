import sys
sys.path.insert(0, "src")
import run_core_satellite_momentum as c1
import run_low_turnover_core_satellite as c2
print("C1 import OK, has CALENDAR_PATH:", hasattr(c1, 'CALENDAR_PATH'))
print("C2 import OK, has CALENDAR_PATH:", hasattr(c2, 'CALENDAR_PATH'))