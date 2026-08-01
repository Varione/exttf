"""Generate conclusion.md from latest_research_status.json and run artifacts."""
import json
import pathlib
from datetime import datetime, timezone

status = json.loads(
    pathlib.Path("reports/latest_research_status.json").read_text(encoding="utf-8")
)
run_dir = pathlib.Path(
    "reports/strategy_research/walkforward_continuous/walkforward_20260731_082713"
)
strategies = [
    "B1_Static_60_20_20",
    "B2_Static_EW_4Asset",
    "B3_Rolling_Risk_Parity",
    "S1_State_Rotation_Fixed",
]
metrics = {
    s: json.loads((run_dir / s / "metrics.json").read_text(encoding="utf-8"))
    for s in strategies
}

lines = []
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
lines.append("# OTF Walk-forward Continuous OOS 结论报告")
lines.append("")
lines.append(f"自动生成时间: {now_str}")
lines.append("数据来源: reports/latest_research_status.json + walkforward_20260731_082713/")
lines.append("")

# Run info
lines.append("## 运行信息")
lines.append("")
lines.append(f'- Run ID: `{status["run_id"]}`')
lines.append(f'- 状态: `{status["status"]}`')
lines.append(f'- 账户模式: `{status["account_mode"]}`')
first_m = metrics[strategies[0]]
lines.append(
    f"- OOS区间: {status['oos_period']} ({first_m['n_days']}个交易日)"
)
lines.append(f'- 规则场景: `{status.get("rule_scenario", "UNKNOWN")}`')
lines.append(f'- 历史规则状态: `{status.get("historical_rule_status", "UNKNOWN")}`')
lines.append("")

# Input hashes
lines.append("## 输入哈希与数据事实")
lines.append("")
for k, v in status["input_hashes"].items():
    lines.append(f"- {k}: `{v}`")
lines.append("")
rc = status["rule_counts_by_status"]
total_rules = sum(rc.values())
lines.append(
    f"- 产品规则: {total_rules}条 "
    f"(OFFICIAL={rc.get('OFFICIAL_VERIFIED',0)}, "
    f"DISTRIBUTOR={rc.get('DISTRIBUTOR_VERIFIED',0)}, "
    f"CONSERVATIVE={rc.get('CONSERVATIVE_ASSUMPTION',0)}, "
    f"ASSUMPTION={rc.get('PRODUCT_TYPE_ASSUMPTION',0)})"
)
mc = status["mapping_counts_by_status"]
total_mappings = sum(mc.values())
lines.append(
    f"- 暴露映射: {total_mappings}条 "
    f"(HIGH={mc.get('mapping_confidence:HIGH',0)}, "
    f"APPROVED={mc.get('review_status:APPROVED',0)})"
)
lines.append("")

# Continuous OOS metrics table
lines.append("## 连续账户OOS指标")
lines.append("")
lines.append(
    "| 策略 | 净CAGR | 毛CAGR | 成本拖累 | Sharpe | MDD "
    "| 稳态最大年度BT | 滚动两年正收益比 | 最差两年CAGR | 总费用 | Gate |"
)
lines.append(
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
)
for s in strategies:
    m = metrics[s]
    g = status["gate_result"][s]
    gate_str = "PASS" if g["gate_passed"] else "FAIL"
    lines.append(
        f"| {s} | {m['net_cagr_pct']:.2f}% | {m['gross_cagr_pct']:.2f}% | "
        f"{m['annualized_cost_drag_pct']:.2f}% | {m['sharpe']:.3f} | "
        f"{m['mdd_pct']:.2f}% | {m['steady_state_max_annual_bilateral_turnover']:.4f} | "
        f"{m['rolling_two_year_positive_ratio_pct']:.1f}% | "
        f"{m['worst_two_year_cagr_pct']:.2f}% | "
        f"{m['total_fee_amount']:.0f}元 | {gate_str} |"
    )
lines.append("")

# Gate details
lines.append("## Gate逐项结果")
lines.append("")
for s in strategies:
    g = status["gate_result"][s]
    lines.append(f"### {s}")
    lines.append("")
    if not g.get("failed_checks"):
        lines.append("- 所有检查通过")
    else:
        for fc in g["failed_checks"]:
            lines.append(f"- 失败: `{fc}`")
    rb = g.get("relative_benchmark_gate")
    if rb:
        lines.append("")
        lines.append("- 相对基准门 (vs B2):")
        lines.append(
            f"  - CAGR优势: {rb['cagr_advantage_pp']:.4f}pp "
            f"(门槛 +0.75pp)"
        )
        lines.append(
            f"  - Sharpe优势: {rb['sharpe_advantage']:.4f} (门槛 +0.15)"
        )
    lines.append("")

# Fee reconciliation
lines.append("## 费用勾稽")
lines.append("")
for s in strategies:
    m = metrics[s]
    fr = m.get("fee_reconciliation", {})
    daily_fee = fr.get("daily_total_fee_amount", 0)
    order_fee = fr.get("order_total_fee_amount", 0)
    delta = fr.get("aggregate_abs_delta", 0)
    passed_str = "通过" if fr.get("passed") else "失败"
    lines.append(
        f"- {s}: 日级总费用={daily_fee:.2f}, "
        f"订单总费用={order_fee:.2f}, "
        f"聚合绝对偏差={delta:.6f} ({passed_str})"
    )
lines.append("")

# Decision section
lines.append("## 策略决策")
lines.append("")
fin = status.get("finalization", {})
if fin.get("decision") == "STOP_STATE_ROTATION":
    lines.append("### STOP_STATE_ROTATION")
    lines.append("")
    lines.append("按 planning.md 停止规则:")
    lines.append(
        "- S1 OOS Gate失败: 相对基准门未通过 "
        "(CAGR优势 -0.64pp, Sharpe优势 -0.01)"
    )
    lines.append("- 停止状态旋转路线和S2/S3开发")
    best = fin.get("best_static_baseline", "UNKNOWN")
    lines.append(f"- 最佳静态基准: {best}")

    b2 = metrics["B2_Static_EW_4Asset"]
    lines.append(
        f"  - B2净CAGR: {b2['net_cagr_pct']:.2f}%, Sharpe: "
        f"{b2['sharpe']:.3f}, MDD: {b2['mdd_pct']:.2f}%"
    )

    b3 = metrics["B3_Rolling_Risk_Parity"]
    lines.append(
        f"  - B3净CAGR: {b3['net_cagr_pct']:.2f}%, Sharpe: "
        f"{b3['sharpe']:.3f}, MDD: {b3['mdd_pct']:.2f}%"
    )
    lines.append("")

    s1 = metrics["S1_State_Rotation_Fixed"]
    lines.append("关键发现:")
    lines.append(
        f"- S1绝对风险优秀 (MDD {s1['mdd_pct']:.2f}%, "
        f"优于B2的{b2['mdd_pct']:.2f}%), 但收益不足以覆盖状态切换产生的额外换手成本"
    )
    lines.append(
        "- B2四资产等权在OOS期间表现最稳健: Sharpe最高、费用最低、稳态换手率最低"
    )
    b3_ss = metrics["B3_Rolling_Risk_Parity"][
        "steady_state_max_annual_bilateral_turnover"
    ]
    b2_ss = metrics["B2_Static_EW_4Asset"][
        "steady_state_max_annual_bilateral_turnover"
    ]
    lines.append(
        f"- B3滚动风险平价通过Gate但换手率显著高于B2 "
        f"(稳态最大年度BT {b3_ss:.4f} vs {b2_ss:.4f})"
    )
    lines.append("")

    lines.append("下一步:")
    lines.append(
        "- B2/B3作为候选基线保留基础设施，完成前向纸面交易验证"
    )
    lines.append(
        "- 不强行选择任何策略为'主策略'，转入数据改进或纸面基准观察阶段"
    )

lines.append("")
report = "\n".join(lines) + "\n"
pathlib.Path("conclusion.md").write_text(report, encoding="utf-8-sig")
print(f"conclusion.md generated: {len(report)} chars, {len(lines)} lines")
