# 动态波动率目标+趋势过滤策略实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 开发 S04_VolTargetTrend 策略，通过波动率目标和趋势过滤降低最大回撤

**Architecture:** 在现有策略框架中添加新策略类，无需修改回测引擎或 RL 环境

**Tech Stack:** Python, pandas, numpy

## Global Constraints

- 使用 `D:\miniconda\envs\agents\python.exe` 作为 Python 环境
- 所有策略必须继承 `Strategy` 基类
- 策略信号必须只使用历史数据（T+1 延迟）
- 回测周期：2018-01-01 ~ 2026-07-17

---

### Task 1: 添加 S04_VolTargetTrend 策略类

**Files:**
- Modify: `src/strategy_library.py` (添加新策略类)
- Modify: `src/strategy_library.py:274-288` (注册新策略)

**Interfaces:**
- Consumes: `Strategy` 基类定义
- Produces: `S04_VolTargetTrend` 策略类，包含 `compute_signal` 和 `get_positions` 方法

- [ ] **Step 1: 实现 S04_VolTargetTrend 策略类**

```python
class S04_VolTargetTrend(Strategy):
    """Qian & Basso (2018): Volatility targeting with trend filter."""
    name = "S04_VolTarget_Trend"
    regime = 0
    description = "Dynamic volatility targeting + MA60 trend filter"
    literature = "Qian & Basso, Risk Management 2018"

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 60:
            return pd.Series(dtype=float)

        # 趋势过滤：仅当 price > MA60 时考虑
        ma_dist = day_data.set_index("symbol")["ma_dist_60"]
        trend_filter = (ma_dist > 0).astype(float)

        # 波动率调整动量信号
        mom = day_data.set_index("symbol")["mom_20"]
        vol = day_data.set_index("symbol")["real_vol_10"]

        # 避免除零
        vol_safe = vol.clip(lower=1e-8)

        # 风险调整后动量：低波 ETF 给予更高权重
        signal = (mom / vol_safe) * trend_filter
        return signal

    def get_positions(self, date: pd.Timestamp, n_hold: int = 20,
                      max_weight: float = 0.05, target_vol: float = 0.005) -> dict[str, float]:
        signals = self.compute_signal(date)
        if len(signals) == 0:
            return {}

        signals = signals.dropna().sort_values(ascending=False)
        if len(signals) < n_hold:
            n_hold = max(1, len(signals))

        top_n = signals.head(n_hold)

        # 计算当前组合波动率（近似）
        # 使用选中 ETF 的平均波动率作为组合波动率代理
        day_data = self._get_day_data(date)
        if len(day_data) == 0:
            return {}

        vol_series = day_data.set_index("symbol")["real_vol_10"]
        selected_vols = [vol_series[sym] for sym in top_n.index if sym in vol_series.index]

        if not selected_vols:
            return {}

        avg_vol = np.mean(selected_vols)

        # 动态仓位控制：根据波动率调整总仓位
        position_ratio = min(1.0, target_vol / (avg_vol + 1e-8))

        # 计算实际权重
        weight = min(max_weight, 1.0 / n_hold) * position_ratio
        return {sym: weight for sym in top_n.index}
```

- [ ] **Step 2: 注册新策略**

修改 `STRATEGIES` 字典：

```python
STRATEGIES = {
    s.name: s for s in [
        S01_CrossSectionalMomentum,
        S02_TrendFollowing,
        S03_DualThrust,
        S04_VolTargetTrend,  # 添加新策略
        S11_MeanReversion,
        S12_LowVolatility,
        S13_QualityFactor,
        S21_LowVolDefense,
        S22_TailRiskDefense,
        S23_OversoldLong,
        B0_BuyHoldEW,
        B1_RegimeAgnosticMomentum,
    ]
}
```

- [ ] **Step 3: 测试策略类导入**

运行：`D:\miniconda\envs\agents\python.exe -c "from src.strategy_library import S04_VolTargetTrend; print('OK')"`

预期输出：`OK`

- [ ] **Step 4: Commit**

```bash
git add src/strategy_library.py
git commit -m "feat: add S04_VolTargetTrend strategy with volatility targeting"
```

### Task 2: 运行回测验证新策略

**Files:**
- Modify: `src/backtest_engine.py` (无需修改，直接使用现有引擎)

**Interfaces:**
- Consumes: `S04_VolTargetTrend` 策略类
- Produces: 回测结果数据，包含 S04 策略的性能指标

- [ ] **Step 1: 运行全量回测**

运行：`D:\miniconda\envs\agents\python.exe src/backtest_engine.py`

预期输出：包含 S04_VolTarget_Trend 的回测结果表格

- [ ] **Step 2: 分析回测结果**

检查：
- S04 的年化收益是否 >5%（跑赢基准）
- 最大回撤是否 <15%（目标）
- Sharpe 是否 >0.5

- [ ] **Step 3: Commit 回测结果**

```bash
git add data/processed/backtest_results.csv
git commit -m "data: backtest results with S04_VolTargetTrend"
```

## 预期效果对比

| 策略 | 年化收益 | 最大回撤 | Sharpe |
|------|---------|---------|--------|
| S23_Oversold_Long | +8.31% | -39.39% | 0.42 |
| B0_BuyHold_EW | +5.92% | -29.80% | 0.34 |
| **S04_VolTarget_Trend (目标)** | **>5%** | **<15%** | **>0.5** |
