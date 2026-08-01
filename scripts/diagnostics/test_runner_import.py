import sys
sys.path.insert(0, "src")
import run_b1_b2_b3_walkforward as canonical
print("Import OK")
print("Has CALENDAR_PATH:", hasattr(canonical, 'CALENDAR_PATH'))
print("CALENDAR_PATH:", canonical.CALENDAR_PATH)