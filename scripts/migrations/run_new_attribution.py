from __future__ import annotations
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from otf_rotation.strategy_attribution import ROOT, load_spec, run_attribution

root = Path(__file__).resolve().parent
# New calendar-corrected runs
c1_root = root / "reports/strategy_research/core_satellite/core_satellite_20260730_114756"
c2_root = root / "reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_115448"
db_path = root / "data/processed/otf_expanded.sqlite"

specs = [
    load_spec("C1", c1_root / "C1_CORE_SATELLITE_MOMENTUM", db_path),
    load_spec("B2", c1_root / "B2_Static_EW_4Asset", db_path),
    load_spec("B2LT", c1_root / "B2_LT_Static_EW_4Asset", db_path),
    load_spec("C2", c2_root / "C2_LOW_TURNOVER_CORE_SATELLITE", db_path),
]

run_id = datetime.now(timezone.utc).astimezone().strftime("phase_a_calendar_corrected_%Y%m%d_%H%M%S")
output = root / "reports/strategy_research/attribution" / run_id

result = run_attribution(specs, output.resolve(), root=root)
print(f"output_dir={result['output_dir']}")
print(f"state={result['status']['state']}")
print(f"accounting_identity_passed={result['status']['accounting_identity_gate']['passed']}")
print(f"warnings={len(result['status']['warnings'])}")
for w in result['status'].get('warnings', []):
    print(f"  WARNING: {w}")