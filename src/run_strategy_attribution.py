"""Run Phase A attribution for the frozen C1/C2/B2/B2-LT bundles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from otf_rotation.strategy_attribution import AttributionSpec, ROOT, load_spec, run_attribution


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--c1-run", type=Path, default=None)
    parser.add_argument("--c2-run", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    c1_root = (
        args.c1_run.resolve()
        if args.c1_run is not None
        else root / "reports/strategy_research/core_satellite/core_satellite_20260729_151927"
    )
    c2_root = (
        args.c2_run.resolve()
        if args.c2_run is not None
        else root / "reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260729_163519"
    )
    db_path = root / "data/processed/otf_expanded.sqlite"
    specs = [
        load_spec("C1", c1_root / "C1_CORE_SATELLITE_MOMENTUM", db_path),
        load_spec("B2", c1_root / "B2_Static_EW_4Asset", db_path),
        load_spec("B2LT", c1_root / "B2_LT_Static_EW_4Asset", db_path),
        load_spec("C2", c2_root / "C2_LOW_TURNOVER_CORE_SATELLITE", db_path),
    ]
    output = args.output
    if output is None:
        run_id = datetime.now(timezone.utc).astimezone().strftime("phase_a_%Y%m%d_%H%M%S")
        output = root / "reports/strategy_research/attribution" / run_id
    result = run_attribution(specs, output.resolve(), root=root)
    status = result["status"]
    print(f"output_dir={result['output_dir']}")
    print(f"state={status['state']}")
    print(f"accounting_identity_passed={status['accounting_identity_gate']['passed']}")
    print(f"warnings={len(status['warnings'])}")
    return 0 if status["state"] != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
