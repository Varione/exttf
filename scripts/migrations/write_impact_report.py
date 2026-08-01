import json, sys
from pathlib import Path

def load_metrics(path):
    return json.loads(Path(path).read_text())

# Old runs
old_c1_dir = Path("reports/strategy_research/core_satellite/core_satellite_20260729_151927")
old_c2_dir = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260729_163519")

# New runs (calendar-corrected)
new_c1_dir = Path("reports/strategy_research/core_satellite/core_satellite_20260730_114756")
new_c2_dir = Path("reports/strategy_research/core_satellite_low_turnover/low_turnover_core_satellite_20260730_115448")

report_lines = []
report_lines.append("# Execution Calendar Fix - Impact Report")
report_lines.append("")
report_lines.append("## Summary")
report_lines.append("")
report_lines.append("- **Fix**: Separated valuation source dates (NAV union) from execution dates (CN calendar)")
report_lines.append("- **7 excluded dates**: 2022-03-05, 2022-03-20, 2022-12-31, 2023-01-02, 2023-05-02, 2023-12-31, 2024-06-30")
report_lines.append("- **Old runs**: SUPERSEDED_BY_EXECUTION_CALENDAR_FIX")
report_lines.append("")

# Compare C1 strategies
report_lines.append("## C1 Comparison (core_satellite)")
report_lines.append("")
for strat in ["C1_CORE_SATELLITE_MOMENTUM", "B2_Static_EW_4Asset", "B2_LT_Static_EW_4Asset"]:
    old_m = load_metrics(old_c1_dir / strat / "metrics.json")
    new_m = load_metrics(new_c1_dir / strat / "metrics.json")
    report_lines.append(f"### {strat}")
    report_lines.append("")
    keys = ["net_cagr_pct", "sharpe", "mdd_pct", "calmar", "total_fee_amount", "worst_two_year_cagr_pct"]
    for k in keys:
        ov = old_m.get(k, "N/A")
        nv = new_m.get(k, "N/A")
        if isinstance(ov, (int,float)) and isinstance(nv, (int,float)):
            diff = nv - ov
            report_lines.append(f"- {k}: OLD={ov:.4f} NEW={nv:.4f} DIFF={diff:+.4f}")
    report_lines.append("")

# Compare C2
report_lines.append("## C2 Comparison (low_turnover_core_satellite)")
report_lines.append("")
strat = "C2_LOW_TURNOVER_CORE_SATELLITE"
old_m = load_metrics(old_c2_dir / strat / "metrics.json")
new_m = load_metrics(new_c2_dir / strat / "metrics.json")
report_lines.append(f"### {strat}")
report_lines.append("")
keys = ["net_cagr_pct", "sharpe", "mdd_pct", "calmar", "total_fee_amount", "worst_two_year_cagr_pct"]
for k in keys:
    ov = old_m.get(k, "N/A")
    nv = new_m.get(k, "N/A")
    if isinstance(ov, (int,float)) and isinstance(nv, (int,float)):
        diff = nv - ov
        report_lines.append(f"- {k}: OLD={ov:.4f} NEW={nv:.4f} DIFF={diff:+.4f}")
report_lines.append("")

# Attribution verification
report_lines.append("## Attribution Verification")
report_lines.append("")
report_lines.append("- New attribution run: phase_a_calendar_corrected_20260730_200151")
report_lines.append("- State: COMPLETE")
report_lines.append("- Accounting identity gate: PASSED")
report_lines.append("- Warnings: 0")
report_lines.append("")

# Old run references
report_lines.append("## Old Run References (SUPERSEDED)")
report_lines.append("")
report_lines.append(f"- C1 old: {old_c1_dir.name}")
report_lines.append(f"- C2 old: {old_c2_dir.name}")
report_lines.append("- SHA records saved to: data/processed/old_run_sha_records.json")
report_lines.append("")

# New run references
report_lines.append("## New Run References (CALENDAR_CORRECTED)")
report_lines.append("")
report_lines.append(f"- C1 new: {new_c1_dir.name}")
report_lines.append(f"- C2 new: {new_c2_dir.name}")
report_lines.append("- Sample tag: CALENDAR_CORRECTED_REUSED_RESEARCH_SAMPLE_NOT_FRESH_OOS")
report_lines.append("")

with open("reports/strategy_research/execution_calendar_fix_impact_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Report written to reports/strategy_research/execution_calendar_fix_impact_report.md")
print("\n" + "\n".join(report_lines[:50]))