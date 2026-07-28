# Findings Log

## P0 修复记录

### 因子文件重建
- 原始 factors_all.csv 在构建过程中中断，仅覆盖 808 ETFs（截至 2019-04-04）
- 使用完整 repair pipeline 重建为 factors_all_repaired.csv
- 最终规模: 1,386,449 rows, 1,549 symbols, 131 columns
- Manifest 验证通过, completed=true, validation_errors=[]

### 旧脚本隔离
- 识别出绕过新 backtest rules 的旧脚本
- 旧脚本未使用 total_return_proxy price mode
- 旧脚本缺少 transaction cost 和 gross return 计算

### Regime 重建
- 基于完整 factor artifact 重新训练 regime model
- 输出: 4,948 days (2006-03-10 ~ 2026-07-17)
- 重建耗时: 100.7s

## P1 修复记录

### Win Rate 修复
- S04_VolTarget_Trend win rate 从异常值修正为合理范围
- 修复了信号计算中的 look-ahead bias

### 协方差波动率修复
- 修正 covariance-based volatility 计算中的维度不匹配
- 确保截面标准化一致性

### 截面标准化修复
- 统一所有因子的 cross-sectional z-score 标准化
- 修复 rolling window 边界条件

### 资产类别上限
- 设置 max_weight=0.05 per ETF
- 限制单只 ETF 最大持仓比例

## 数据质量审计结果 (Phase 4)

### 日期连续性缺口
- 1,542 symbols 存在日期缺口（共 1,549 个）
- 平均缺口: 3.38 天
- 最大缺口: 175 天
- 缺口最大的 ETF:
  - 511660: max gap 175 days
  - 511820: max gap 175 days
  - 511930: max gap 175 days
  - 510680: max gap 90 days
  - 511950: max gap 28 days

### 价格模式覆盖
- 全部 1,549 ETFs 仅支持 total_return_proxy 模式
- 无 external reference 交叉验证数据（覆盖率 1.3%）

### 外部参考覆盖
- etf_daily_external_reference 表: 20 symbols, 31,175 rows
- 覆盖率: 1.3% (20/1,549)

## 回测关键发现

### Unified Experiment (20260727_163752)
- Data gate: PASSED
- 5 strategies tested over 2018-01-01 ~ 2026-07-17 (2,071 trading days)

### B0_BuyHold_EW (基准)
- Full period: Total return 40.95%, Sharpe 0.396, Max DD -25.13%
- OOS_2018_2021: Total return 29.06%, Sharpe 0.572 (牛市表现优异)
- OOS_2022_2024: Total return -5.31%, Sharpe -0.106 (熊市回撤)
- OOS_2025_2026: Total return 15.34%, Sharpe 0.809 (反弹行情)

### S04_VolTarget_Trend
- Full period: Total return -0.99%, Sharpe 0.063, Max DD -34.93%
- 最大优势窗口 OOS_2025_2026: Total return 18.20%, Sharpe 0.737
- 最大劣势窗口 OOS_2022_2024: Total return -15.60%, Sharpe -0.360
- DD 改善有限，全周期收益低于基准

### S12_Low_Volatility (P1 backtest)
- Sharpe 0.539 为所有策略最高
- Max DD -6.52% 为所有策略最优
- 年化收益 1.74% 偏低，但风险调整后表现最佳

### S01_CS_Momentum
- Full period: Total return 2.88%, Sharpe 0.092, Max DD -25.56%
- OOS_2018_2021: Total return 9.08%, Sharpe 0.240
- 动量策略在熊市中回撤严重

### B1_Momentum_Agnostic
- Full period: Total return 21.56%, Sharpe 0.231, Max DD -28.61%
- 换手率极高 (total_turnover 130.62)，成本拖累显著 (annualized_cost_drag 1.64%)

### S23_Oversold_Long
- Full period: Total return -2.74%, Sharpe -0.028, Max DD -16.18%
- OOS_2018_2021: Total return 7.62%, Sharpe 0.546 (均值回归在牛市有效)
- OOS_2022_2024: Total return -9.50%, Sharpe -0.348

## 数据一致性验证

以下四处信息已确认一致:
- config/unified_experiment.json (factor_validation: 1,386,449 rows, 1,549 symbols, 131 columns)
- progress.md (DB: 1,549 symbols, 1,386,449 rows)
- findings.md (本文档)
- reports/unified_experiment/20260727_163752/experiment_manifest.json (factor_validation: 1,386,449 rows, 1,549 symbols, 131 columns)

## Fingerprint Reference

- DB SHA256: 3441d394e7cc8cb0f7b5f45d45e3a5329c208ee9203c2c2e95e0c679f5a46d8d
- Factor schema hash: dd4fa3c6616a54b8
- Latest run ID: 20260727_163752
- PIT status: PIT_PARTIAL
- Classification samples: 4,885
