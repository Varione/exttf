# planning.md 执行状态检查报告

## P0-A：修复时间点产品选择 ✅ 完成

| 要求 | 状态 | 位置 |
|------|------|------|
| select()必须使用date参数 | ✅ | product_selector.py:283-285, date=None raises ValueError |
| NAV历史长度只统计到选择日 | ✅ | ProductSelector._build_candidates()用WHERE nav_date <= date过滤 |
| 产品评分只用当时可见数据 | ✅ | 使用截至date的NAV记录数、规则状态等 |
| 未来成立/已终止产品排除 | ✅ | inception_date <= date AND (termination_date IS NULL OR >= date) |
| PIT_PARTIAL标记 | ✅ | 文档注释中说明当前快照为PIT_PARTIAL |

## P0-B：修复信号、提交和确认时序 ✅ 完成

| 要求 | 状态 | 位置 |
|------|------|------|
| signal_observed_at与order_submit_date分离 | ✅ | run_s1_s2_experiment.py:build_month_end_schedule()构建signal→submit映射 |
| 月末信号→下一可申购日提交 | ✅ | build_month_end_schedule()取当月最后一个交易日作为信号日，下一交易日为提交日 |
| signal_dates审计关系保留 | ✅ | otf_backtest_engine.py:725,943-944记录signal_dates映射 |
| 无同日信号提交 | ✅ | 103个信号日→103个提交日，严格signal_date < submit_date |

## P0-C：统一产品规则与严格执行 ✅ 完成

| 要求 | 状态 | 位置 |
|------|------|------|
| ProductSelector与Engine共享ProductRuleBook | ✅ | run_s1_s2_experiment.py:48-50, selector和engine共用同一rule_book实例 |
| strict_product_rules=True | ✅ | run_s1_s2_experiment.py:49, engine严格模式启用 |
| 数据库自动规则不计入真实覆盖率 | ✅ | otf_trading_rules.py区分PRODUCT_TYPE_ASSUMPTION与OFFICIAL_VERIFIED/DISTRIBUTOR_VERIFIED |
| 缺少费用数据不填0 | ✅ | strict模式下未知费用产品被拒单 |
| 40-60只高质量候选池 | ✅ | otf_product_rules.csv现有55条规则，覆盖15袖套 |
| 每袖套≥2只正式可交易候选 | ✅ | 验证通过，所有15袖套≥2只合格产品 |

## P0-D：修复成本与绩效会计 ✅ 完成

| 要求 | 状态 | 位置 |
|------|------|------|
| Gross/Net CAGR分别计算 | ✅ | run_s1_s2_experiment.py:223-228, gross_cagr = (gross_eq[-1]/gross_eq[0])^(1/Y) - 1 |
| 年化成本拖累从毛净终值差推导 | ✅ | cost_drag = gross_cagr - net_cagr |
| 不再将每日成本比例除以初始资金 | ✅ | otf_backtest_engine.py按当日权益比例计算费用 |
| B1/B2/S1费用指标不再为0 | ✅ | S1: 0.39%, B2: 0.19%, B1: 0.28% |
| 日报/订单审计/汇总报告费用勾稽 | ✅ | total_fees = sum(order fees) |

## P1-A：落实状态滞回、调仓带和风险控制 ✅ 完成

| 要求 | 状态 | 位置 |
|------|------|------|
| confirm_months=2连续确认 | ✅ | market_state.py:454,474-475 |
| min_state_duration_days=30 | ✅ | market_state.py:455 |
| fast_switch_threshold仅降风险 | ✅ | market_state.py:453 |
| 调仓带过滤小额订单 | ✅ | run_s1_s2_experiment.py中rebalance_band=0.005 |

## P1-B：修复调仓日历 ⚠️ 基本完成（遗留小问题）

| 要求 | 状态 | 位置 |
|------|------|------|
| 废除freq="21D"月频近似 | ✅ | run_s1_s2_experiment.py使用build_month_end_schedule()基于真实交易日历 |
| market_state.py遗留freq="21D" | ⚠️ | market_state.py:521 simulate_monthly()未使用，不影响实验 |

## P1-C：重建资产暴露映射 ✅ 基本完成

| 要求 | 状态 | 位置 |
|------|------|------|
| underlying_id/underlying_name | ✅ | otf_fund_catalog.underlying_name |
| asset_sleeve/asset_class | ✅ | otf_fund_catalog.asset_class |
| mapping_source/mapping_confidence | ⚠️ | 未单独建模，通过SLEEVE_PATTERNS正则和TYPE_TO_SLEEVE映射隐式实现 |

## P2：可复现重跑协议 ✅ 完成

| 要求 | 状态 | 位置 |
|------|------|------|
| B1/B2/S1/S2命名与定义 | ✅ | run_s1_s2_experiment.py标准命名 |
| 审计报告保存 | ✅ | reports/strategy_research/s1_s2_experiment/ |
| 无硬编码历史指标 | ✅ | 所有报告字段从本次运行自动计算 |

## P3：重新实验与停止条件 ✅ 完成

| 步骤 | 状态 | 结果 |
|------|------|------|
| B1/B2会计对照 | ✅ | 通过，费用勾稽正确 |
| S1固定产品验证 | ✅ | NetCAGR=4.51%, Sharpe=0.661, MDD=-8.37% |
| S2动态产品验证 | ✅ | NetCAGR=4.51%, Sharpe=0.667, MDD=-8.33% |
| S2不优于S1→保留固定产品 | ✅ | 按计划执行，停止动态切换 |

## P4：Walk-forward与最终 Gate ⏸️ 按计划暂停

| 要求 | 状态 | 说明 |
|------|------|------|
| Walk-forward扩展窗口 | ⏸️ | S2无增值，按规划"不进入S3" |
| Gate门槛验证 | ⏸️ | CAGR gate未通过（4.51% < 5.05%），用户override后仍未能弥补缺口 |

## 执行顺序（§11）核对

| # | 任务 | 状态 |
|---|------|------|
| 1 | Phase 5产物标记INVALID | ✅ |
| 2 | ProductSelector时间点资格修复 | ✅ |
| 3 | 信号日→下一申购日映射 | ✅ |
| 4 | 统一规则簿严格模式 | ✅ |
| 5 | 成本会计Gross/Net报告 | ✅ |
| 6 | 状态确认、最短持续期、调仓带 | ✅ |
| 7 | 月末调仓日历修复 | ✅ |
| 8 | 资产暴露映射重建 | ✅基本完成 |
| 9 | 清理一次性脚本 | ⚠️未系统归档（quick_*/run_phase*仍存） |
| 10 | 端到端测试 | ✅ |
| 11 | 重跑B1/B2会计勾稽 | ✅ |
| 12 | 重跑固定产品S1 | ✅ |
| 13 | S1 Gate决策→允许S2 | ✅用户override |
| 14 | Walk-forward重写conclusion.md | ⏸️按"不进入S3"暂停 |

## 遗留事项

1. market_state.py:521 simulate_monthly()仍用freq="21D"（未使用，低优先级）
2. otf_fund_catalog缺少mapping_source/mapping_confidence字段（隐式实现，非阻塞）
3. 一次性脚本未系统归档（quick_*/run_phase*等仍在src/目录）

## 总体结论

**P0-A/B/C/D全部完成，P1-A/B基本完成，P2/P3按规划完成。**

S2验证后按规划决策：保留状态配置+固定低费产品，停止动态产品切换，不进入S3。
conclusion.md已重写反映最终状态。
