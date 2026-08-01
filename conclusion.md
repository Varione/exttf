# OTF Walk-forward Continuous OOS 结论报告

自动生成时间: 2026-08-01 11:34:21 UTC
数据来源: reports/latest_research_status.json + walkforward_20260731_082713/

## 运行信息

- Run ID: `walkforward_20260731_082713`
- 状态: `OOS_GATE_FAILED`
- 账户模式: `FROZEN_PARAMETER_CONTINUOUS_OOS`
- OOS区间: 2021-01-04~2026-07-27 (1347个交易日)
- 规则场景: `CONSERVATIVE_STANDARD`
- 历史规则状态: `CURRENT_SNAPSHOT_ONLY`

## 输入哈希与数据事实

- db_sha256: `dcc1d865998942948a4650957ae49e26357949a1b7f296b527986d1eafc40e79`
- rules_sha256: `fe133a27c9f06c7e3e95f12cce3c2157ff4ef2834fa432f95f3c60f44a229a98`
- mapping_sha256: `5342f4cd400a6cebd812f9c8735377d0b99faddec1560bced7eceff9e5090251`
- config_sha256: `22b8b1342d4decd8e35c8bbe153996e53ce244c69d57413631071e1a1fa93aa4`
- calendar_sha256: `66bdb882ffe5edd342773310b57c53ffe6970f46ba6321c6f96b0033f1771ec7`

- 产品规则: 55条 (OFFICIAL=3, DISTRIBUTOR=31, CONSERVATIVE=20, ASSUMPTION=1)
- 暴露映射: 62条 (HIGH=31, APPROVED=31)

## 连续账户OOS指标

| 策略 | 净CAGR | 毛CAGR | 成本拖累 | Sharpe | MDD | 稳态最大年度BT | 滚动两年正收益比 | 最差两年CAGR | 总费用 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B1_Static_60_20_20 | 3.44% | 3.78% | 0.34% | 0.359 | -21.80% | 0.0533 | 58.5% | -7.59% | 17149元 | FAIL |
| B2_Static_EW_4Asset | 5.13% | 5.37% | 0.24% | 0.815 | -8.71% | 0.0368 | 94.3% | -1.46% | 12665元 | PASS |
| B3_Rolling_Risk_Parity | 4.41% | 4.89% | 0.48% | 0.647 | -9.54% | 0.2117 | 75.5% | -2.34% | 24895元 | PASS |
| S1_State_Rotation_Fixed | 4.49% | 4.72% | 0.24% | 0.810 | -5.95% | 0.1958 | 85.4% | -2.47% | 12647元 | FAIL |

## Gate逐项结果

### B1_Static_60_20_20

- 失败: `sharpe`
- 失败: `mdd`
- 失败: `rolling_two_year_positive_ratio`
- 失败: `worst_two_year_cagr`

### B2_Static_EW_4Asset

- 所有检查通过

### B3_Rolling_Risk_Parity

- 所有检查通过

### S1_State_Rotation_Fixed

- 失败: `relative_benchmark`

- 相对基准门 (vs B2):
  - CAGR优势: -0.6447pp (门槛 +0.75pp)
  - Sharpe优势: -0.0054 (门槛 +0.15)

## 费用勾稽

- B1_Static_60_20_20: 日级总费用=17149.46, 订单总费用=17149.46, 聚合绝对偏差=0.000000 (通过)
- B2_Static_EW_4Asset: 日级总费用=12665.16, 订单总费用=12665.16, 聚合绝对偏差=0.000000 (通过)
- B3_Rolling_Risk_Parity: 日级总费用=24895.14, 订单总费用=24895.14, 聚合绝对偏差=0.000000 (通过)
- S1_State_Rotation_Fixed: 日级总费用=12647.09, 订单总费用=12647.09, 聚合绝对偏差=0.000000 (通过)

## 策略决策

### STOP_STATE_ROTATION

按 planning.md 停止规则:
- S1 OOS Gate失败: 相对基准门未通过 (CAGR优势 -0.64pp, Sharpe优势 -0.01)
- 停止状态旋转路线和S2/S3开发
- 最佳静态基准: B2_Static_EW_4Asset
  - B2净CAGR: 5.13%, Sharpe: 0.815, MDD: -8.71%
  - B3净CAGR: 4.41%, Sharpe: 0.647, MDD: -9.54%

关键发现:
- S1绝对风险优秀 (MDD -5.95%, 优于B2的-8.71%), 但收益不足以覆盖状态切换产生的额外换手成本
- B2四资产等权在OOS期间表现最稳健: Sharpe最高、费用最低、稳态换手率最低
- B3滚动风险平价通过Gate但换手率显著高于B2 (稳态最大年度BT 0.2117 vs 0.0368)

下一步:
- B2/B3作为候选基线保留基础设施，完成前向纸面交易验证
- 不强行选择任何策略为'主策略'，转入数据改进或纸面基准观察阶段

