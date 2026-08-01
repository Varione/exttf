import sys
sys.path.insert(0, "src")
from otf_backtest_engine import OTFBacktestEngine
print("Import OK")
print("Has execution_calendar param:", "execution_calendar" in str(OTFBacktestEngine.__init__.__annotations__))