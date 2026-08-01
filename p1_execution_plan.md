# P1 Phase Execution Plan

## Overview
P0 phase complete (6/6 tasks). This plan details the implementation of P1-1, P1-2, and P1-3.

---

## P1-1: 执行日历修复 — 删除freq=21D，统一月末调度器

### Current Issues
1. `MarketStateEngine.simulate_monthly()` uses `freq="21D"` which generates ~4 signals per month (every 21 days), not monthly.
2. `build_month_end_schedule()` in `run_s1_experiment.py` and `run_s1_s2_experiment.py` is correct but needs verification:
   - First rebalance at start date itself (pseudo month-end signal violation)
   - Multiple signals potentially mapping to same submit date

### Implementation Steps

#### Step 1.1: Fix MarketStateEngine.simulate_monthly()
- **File**: `src/otf_rotation/market_state.py`
- **Change**: Replace `freq="21D"` with proper month-end schedule
- **Implementation**: Use the same logic as `build_month_end_schedule()` or create a shared utility function

```python
# Before:
dates = pd.date_range(start, end, freq="21D")

# After:
dates = self._get_month_end_dates(start, end)
```

#### Step 1.2: Create shared month-end scheduler
- **File**: `src/otf_rotation/schedule.py` (new)
- **Function**: `build_month_end_schedule(trading_dates, start, end)` returns list of last trading days of each month
- **Rules**:
  - Each calendar month has exactly one signal date (last trading day before month end)
  - No pseudo month-end at backtest start
  - First signal is the first natural month-end after start date

#### Step 1.3: Fix run_s1_experiment.py and run_s1_s2_experiment.py
- **Files**: `src/run_s1_experiment.py`, `src/run_s1_s2_experiment.py`
- **Change**: Remove first rebalance at start date; use shared scheduler
- **Rule**: Initial allocation happens on first month-end signal, not backtest start

#### Step 1.4: Add tests
- **File**: `tests/test_schedule.py` (new) or add to existing test file
- **Tests**:
  - Each calendar month has exactly one signal date
  - No signal at arbitrary backtest start date
  - Signal dates are valid trading days

### Acceptance Criteria
- Every calendar month has at most one regular signal
- No pseudo month-end signal at backtest start date
- `freq="21D"` completely removed from codebase
- Domestic and QDII timing tests pass

---

## P1-2: 状态风险控制接线 — fast_switch只降风险、prev_weights调仓带、9%目标波动率

### Current Issues
1. `fast_switch_threshold` allows switching to ANY state (including higher risk) when score gap is extreme.
2. `prev_weights` is used in `ExposureSelector._apply_min_trade()` but not properly tested.
3. Target volatility (9%) exists in config but not applied to portfolio risk scaling.
4. No distinction between target weight changes and actual position drift.
5. No explicit `reset()` method on MarketStateEngine for independent experiments.

### Implementation Steps

#### Step 2.1: State risk ordering and fast_switch constraint
- **File**: `src/otf_rotation/market_state.py`
- **Add**: Static mapping of state to risk level:
  ```python
  STATE_RISK_LEVEL = {
      "RISK_ON": 5,
      "NEUTRAL": 4,
      "INFLATION_REAL_ASSET": 3,
      "DEFLATION_RATE_DOWN": 2,
      "STRESS": 1,
  }
  ```
- **Change**: `_map_state()` fast_switch logic:
  - Only allow fast switch if `STATE_RISK_LEVEL[best_state] <= STATE_RISK_LEVEL[prev_state]`
  - Higher risk switches must go through normal confirmation

#### Step 2.2: Add MarketStateEngine.reset()
- **File**: `src/otf_rotation/market_state.py`
- **Add**:
  ```python
  def reset(self):
      self._state_entry_date = None
      self._current_state = None
      self._consecutive_counts = {}
  ```

#### Step 2.3: Target volatility scaling (9% annualized)
- **File**: `src/otf_rotation/experiment_artifacts.py` or new file `src/otf_rotation/volatility_scaler.py`
- **Add**: Portfolio-level volatility scaling function:
  - Compute rolling portfolio volatility (e.g., 60-day lookback)
  - Scale factor = min(1.0, target_vol / realized_vol)
  - Apply to all non-cash weights (only reduce risk, never leverage)
- **Integration**: Apply after `ExposureSelector.get_fund_weights()` in signal functions

#### Step 2.4: Verify prev_weights usage in AssetBudgetEngine/ExposureSelector
- **File**: `src/otf_rotation/asset_budget.py`
- **Check**: `_apply_min_trade()` properly uses prev_weights
- **Fix if needed**: Ensure min_weight_change_pct filter works correctly
- **Add test**: Verify that small weight changes are filtered out

#### Step 2.5: Turnover decomposition reporting
- **File**: Output enhancement in experiment runners
- **Track**: Separate turnover contributions from:
  - State switches
  - Rebalancing band triggers (prev_weights filter)
  - Risk scaling adjustments
  - Product replacements

#### Step 2.6: Add tests
- **Tests**:
  - fast_switch only allows down-risk transitions
  - reset() clears all internal state
  - Target volatility scaling reduces weights when vol > target
  - prev_weights min_trade filter prevents churn

### Acceptance Criteria
- All config parameters have behavioral tests
- Fast switch only to lower-risk states
- Target volatility (9%) applied as downside-only scaling
- Each component can be disabled and observed via ablation experiment
- State engine provides explicit `reset()` called before each independent experiment

---

## P1-3: 修正基准定义 — B2更名，实现B3滚动风险平价

### Current Issues
1. `B2_Risk_Parity` is actually static equal weight (25%/25%/25%/25%) across 4 assets — not risk parity.
2. No true rolling risk parity benchmark exists.

### Implementation Steps

#### Step 3.1: Rename B2
- **Files**: `src/run_s1_experiment.py`, `src/run_s1_s2_experiment.py`, `src/audit_phase5.py`
- **Change**:
  ```python
  # Before:
  ("B2_Risk_Parity", lambda d: static_risk_parity_signal(d))

  # After:
  ("B2_Static_EW_4Asset", lambda d: static_equal_weight_4asset_signal(d))
  ```
- **Rename function**: `static_risk_parity_signal()` → `static_equal_weight_4asset_signal()`

#### Step 3.2: Implement B3 Rolling Risk Parity
- **File**: `src/otf_rotation/risk_parity.py` (new)
- **Class**: `RollingRiskParityEngine`
- **Algorithm**:
  ```python
  class RollingRiskParityEngine:
      def __init__(self, 
                   assets: list[str],             # fund codes or ETF proxies
                   lookback_days: int = 252,       # rolling window for covariance
                   min_lookback: int = 60,         # minimum data required
                   max_weight: float = 0.40,       # per-asset cap
                   fallback_weights: dict[str, float],  # used when lookback insufficient
                   nav_source):                    # NAV/price data source

      def get_weights(self, date: pd.Timestamp) -> dict[str, float]:
          # 1. Get rolling returns for each asset
          # 2. Compute sample covariance matrix
          # 3. Apply risk parity optimization (equal risk contribution)
          #    - Use scipy.optimize.minimize with risk contribution constraint
          #    - Weights >= 0, sum <= 1.0
          #    - No leverage
          # 4. Apply max_weight cap and renormalize
          # 5. If lookback < min_lookback or optimization fails → use fallback_weights
          # 6. Return weights + marginal risk contributions for audit
  ```

- **Risk Parity Implementation Details**:
  - Objective: minimize sum of (risk_contribution_i / target_risk - 1)^2
  - Where risk_contribution_i = w_i * (Sigma @ w)_i / portfolio_vol
  - Target risk contribution = 1/N for each asset
  - Regularize covariance matrix if needed (Ledoit-Wolf shrinkage)
  - Constraint: w_i >= 0, sum(w) <= 1.0

#### Step 3.3: Integrate B3 into experiment runners
- **Files**: `src/run_s1_experiment.py`, `src/run_s1_s2_experiment.py`
- **Add**:
  ```python
  b3_engine = RollingRiskParityEngine(
      assets=["160706", "000218", "001512", "260102"],  # CSI300/GOLD/GOV_BOND/MM
      lookback_days=252,
      min_lookback=60,
      max_weight=0.40,
      fallback_weights={"160706": 0.25, "000218": 0.25, "001512": 0.25, "260102": 0.25},
      nav_source=engine._nav_df,
  )
  strategies.append(("B3_Rolling_Risk_Parity", b3_engine))
  ```

#### Step 3.4: Add tests
- **Tests**:
  - B3 risk contributions are approximately equal (within tolerance)
  - B3 does not look ahead (weights at date T only use data up to T)
  - B3 falls back to static weights when insufficient history
  - B3 weights sum to <= 1.0 and all >= 0

### Acceptance Criteria
- B2 renamed to `B2_Static_EW_4Asset` everywhere
- B3 implemented with rolling covariance-based risk parity
- B3 risk contributions approximately equal per asset
- B3 does not read future data
- B2 no longer appears with "risk parity" in its name

---

## Execution Order

1. **P1-1 first** (calendar fix) because it affects all subsequent experiments
2. **P1-3 second** (benchmark rename + B3) because it's a clean, independent change
3. **P1-2 third** (risk control wiring) because it depends on correct calendar and has most moving parts

## Testing Strategy
- After each P1 task: run `pytest tests/test_otf_backtest.py -v`
- After all P1 tasks: re-run S1 experiment to verify integration
- Add regression tests to prevent backsliding

## Deliverables
- Updated source files with changes
- New files: `schedule.py`, `risk_parity.py` (if needed)
- Test coverage for all new behavior
- Clean git history with descriptive commits per task
