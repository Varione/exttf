"""Asset Budget Engine and Exposure Selector.

Usage:
    engine = MarketStateEngine()
    budget_engine = AssetBudgetEngine()
    selector = ExposureSelector(engine, budget_engine)
    
    state, scores, features = engine.get_state("2024-01-15")
    weights = selector.select_weights("2024-01-15", state)
    print(weights)  # Series[fund_code -> target_weight]
"""

from __future__ import annotations
import logging
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from otf_rotation.product_selector import ProductSelector

log = logging.getLogger(__name__)


# ================================================================
# AssetBudgetEngine: maps market state to asset class budget ranges
# ================================================================

class AssetBudgetEngine:
    """Maps market state to budget ranges per asset class.

    Budgets are defined in config/state_allocation.yaml with min/max/default
    per state. The engine selects a specific weight within each range based
    on optional feature signals, then normalizes.
    """

    def __init__(self, config_path: str = "config/state_allocation.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)
        self._budgets = self._cfg["state_budgets"]
        self._asset_classes = self._cfg["asset_classes"]

    def get_budget_range(self, state: str) -> dict[str, dict[str, float]]:
        """Get min/max/default budget ranges for a given state."""
        state = state.upper()
        if state not in self._budgets:
            state = "NEUTRAL"
        return self._budgets[state]

    def get_default_weights(self, state: str) -> dict[str, float]:
        """Get default class-level weights (midpoint of ranges)."""
        ranges = self.get_budget_range(state)
        return {cls: info["default"] for cls, info in ranges.items()}

    def allocate_class_weight(self, state: str, class_name: str, signal: float = 0.5) -> float:
        """Allocate a specific weight within the min-max range.

        signal: 0 = min, 1 = max, 0.5 = default (midpoint)
        """
        ranges = self.get_budget_range(state)
        if class_name not in ranges:
            return 0.0
        info = ranges[class_name]
        sig = np.clip(signal, 0.0, 1.0)
        return info["min"] + sig * (info["max"] - info["min"])

    def normalize(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize class weights to sum to 1.0, residual -> cash_mgt."""
        total = sum(weights.values())
        if abs(total - 1.0) < 1e-6:
            return dict(weights)

        if total > 1.0:
            scale = 1.0 / total
            return {k: v * scale for k, v in weights.items()}
        else:
            result = dict(weights)
            result["cash_mgt"] = result.get("cash_mgt", 0.0) + (1.0 - total)
            return result

    def get_class_sleeves(self, class_name: str) -> list[str]:
        """Get sleeve names for an asset class."""
        info = self._asset_classes.get(class_name)
        return list(info["sleeves"]) if info else []

    def get_max_sleeve_weight(self, class_name: str) -> float:
        info = self._asset_classes.get(class_name, {})
        return info.get("max_sleeve_pct", 1.0)

    def get_selection_method(self, class_name: str) -> str:
        info = self._asset_classes.get(class_name, {})
        return info.get("selection_method", "equal_weight")

    @property
    def target_vol(self) -> float:
        return self._cfg.get("volatility_target", {}).get("target_pct", 0.09)

    def vol_scale_weights(
        self,
        weights: dict[str, float],
        estimated_port_vol: float,
        cash_class: str = "cash_mgt",
    ) -> dict[str, float]:
        """Scale portfolio weights to meet target volatility (per P1-2).

        Only reduces risk, never adds leverage. If estimated portfolio
        volatility exceeds target, shrink risky weights proportionally
        and move excess to cash.

        Args:
            weights: current portfolio weights by asset class
            estimated_port_vol: estimated annualized portfolio volatility
            cash_class: cash management asset class name

        Returns:
            Scaled weights summing to 1.0
        """
        target = self.target_vol
        if estimated_port_vol <= 0 or target <= 0:
            return dict(weights)

        # Only scale down, never leverage up
        if estimated_port_vol <= target:
            return dict(weights)

        scale = target / estimated_port_vol
        scaled = {}

        for cls, w in weights.items():
            if cls == cash_class:
                continue
            else:
                scaled[cls] = w * scale

        risky_sum = sum(scaled.values())
        scaled[cash_class] = max(0.0, 1.0 - risky_sum)

        return scaled


# ================================================================
# ExposureSelector: selects specific sleeve weights within budgets
# ================================================================

class ExposureSelector:
    """Selects sleeve-level weights within asset class budgets.

    Uses momentum/trend signals to allocate within each asset class.
    Uses ProductSelector for dynamic multi-fund selection per sleeve.
    """

    def __init__(self, market_state_engine, budget_engine: AssetBudgetEngine,
                  product_selector: ProductSelector | None = None,
                  top_n_funds: int = 2):
        self._mse = market_state_engine
        self._budget = budget_engine
        self._cfg = budget_engine._cfg
        self._fixed_products = self._cfg.get("fixed_products", {})
        self._product_selector = product_selector
        self._top_n = top_n_funds
        self._prev_sleeve_weights: dict[str, float] | None = None

    def select_weights(self, date: str | pd.Timestamp,
                       market_state: str,
                       prev_weights: dict[str, float] | None = None
                       ) -> pd.Series:
        """Compute target weights for all sleeves.

        Returns: pd.Series with MultiIndex (asset_class, sleeve) -> target_weight
        """
        dt = pd.Timestamp(date) if isinstance(date, str) else date

        # 1. Get class-level budget
        class_weights = self._budget.get_default_weights(market_state)

        # 2. Allocate within each class
        sleeve_weights = {}
        for class_name, class_weight in class_weights.items():
            if class_weight <= 0:
                continue
            sleeves = self._budget.get_class_sleeves(class_name)
            if not sleeves:
                continue

            method = self._budget.get_selection_method(class_name)
            alloc = self._allocate_sleeves(sleeves, class_name, class_weight,
                                           dt, method, market_state)
            sleeve_weights.update(alloc)

        # 3. Apply constraints (per-sleeve caps, class budgets)
        sleeve_weights = self._apply_constraints(sleeve_weights, market_state)

        # 4. Normalize: redistribute residual to cash, then scale to 1.0
        sleeve_weights = self._normalize_weights(dict(sleeve_weights), market_state)

        # 5. Apply min trade filter (avoid churn)
        if prev_weights:
            sleeve_weights = self._apply_min_trade(sleeve_weights, prev_weights)

        return sleeve_weights

    def _allocate_sleeves(self, sleeves: list[str], class_name: str,
                          class_weight: float, date: pd.Timestamp,
                          method: str, state: str) -> dict[str, float]:
        """Allocate class weight among sleeves using selection method."""
        if method == "fixed":
            # Full weight to first sleeve
            return {sleeves[0]: class_weight}

        elif method == "equal_weight":
            n = len(sleeves)
            return {s: class_weight / max(n, 1) for s in sleeves}

        elif method.startswith("momentum"):
            # Rank sleeves by momentum, pick top N
            lookback = self._cfg.get("selection", {}).get("momentum_lookback", 120)
            top_n = self._get_top_n(class_name)

            momentum_scores = {}
            for sleeve in sleeves:
                # Get representative ETF for this sleeve
                proxy_etfs = self._get_proxy_etfs(sleeve)
                mom = self._compute_momentum(proxy_etfs, date, lookback)
                if mom is not None:
                    momentum_scores[sleeve] = mom

            if not momentum_scores:
                return {s: class_weight / max(len(sleeves), 1) for s in sleeves}

            # Sort by momentum descending, pick top N
            ranked = sorted(momentum_scores.items(), key=lambda x: -x[1])
            selected = [s for s, _ in ranked[:top_n]]
            max_sleeve = self._budget.get_max_sleeve_weight(class_name)

            if len(selected) == 0:
                return {}

            # Inverse volatility weighting among selected
            weights = {}
            for s in selected:
                proxy = self._get_proxy_etfs(s)
                vol = self._compute_vol(proxy, date, 60)
                weights[s] = 1.0 / max(vol, 0.05) if vol else 1.0

            total_inv = sum(weights.values())
            allocated = {}
            for s in selected:
                raw = (weights[s] / max(total_inv, 0.01)) * class_weight
                allocated[s] = min(raw, max_sleeve)

            return allocated

        return {s: class_weight / max(len(sleeves), 1) for s in sleeves}

    def _apply_constraints(self, weights: dict[str, float],
                           state: str) -> dict[str, float]:
        """Apply per-sleeve caps within each class budget.

        Uses iterative redistribution: if capping a sleeve creates excess,
        it is redistributed to non-capped sleeves. If all sleeves hit cap,
        remaining excess returns to class residual (lost).
        """
        budget_ranges = self._budget.get_budget_range(state)
        result = {}
        residual = 0.0

        for class_name, class_info in self._budget._asset_classes.items():
            sleeves = class_info["sleeves"]
            max_sleeve = class_info.get("max_sleeve_pct", 1.0)
            class_budget = budget_ranges.get(class_name, {}).get("default", 0.0)

            if class_budget <= 0:
                continue

            class_raw = {s: weights.get(s, 0.0) for s in sleeves if s in weights}
            if not class_raw:
                residual += class_budget
                continue

            raw_total = sum(class_raw.values())
            if raw_total <= 0:
                residual += class_budget
                continue

            scale = class_budget / raw_total

            # Iterative capping: redistribute excess until stable
            alloc = {s: w * scale for s, w in class_raw.items()}
            max_iter = 10
            for _ in range(max_iter):
                capped_any = False
                excess = 0.0
                uncapped_count = 0
                for s in list(alloc.keys()):
                    if alloc[s] > max_sleeve:
                        excess += alloc[s] - max_sleeve
                        alloc[s] = max_sleeve
                        capped_any = True
                    elif alloc[s] < max_sleeve:
                        uncapped_count += 1

                if not capped_any or excess < 0.0001 or uncapped_count == 0:
                    break

                # Distribute excess among uncapped
                uncapped_total = sum(alloc[s] for s in alloc if alloc[s] < max_sleeve)
                if uncapped_total > 0:
                    for s in alloc:
                        if alloc[s] < max_sleeve and alloc[s] > 0:
                            alloc[s] += excess * (alloc[s] / uncapped_total)

            # After capping, remaining excess goes to residual
            used = sum(alloc.values())
            if used < class_budget:
                residual += class_budget - used

            result.update(alloc)

        return result

    def _normalize_weights(self, weights: dict[str, float],
                            state: str | None = None) -> pd.Series:
        """Normalize weights: redistribute residual to cash, scale to 1.0.

        After class-level constraints, weights may not sum to 1.0 due to
        per-sleeve caps. Residual goes to money_market.
        """
        total = sum(weights.values())
        if total <= 0:
            return pd.Series(dtype=float)

        # Add residual to cash_mgt sleeves
        if total < 1.0:
            cash_sleeves = self._budget.get_class_sleeves("cash_mgt")
            if cash_sleeves:
                first_cash = cash_sleeves[0]
                weights[first_cash] = weights.get(first_cash, 0.0) + (1.0 - total)

        # Scale to 1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001 and total > 0:
            for k in weights:
                weights[k] /= total

        # Filter tiny weights
        return pd.Series({k: v for k, v in weights.items() if v > 0.001})

    def _apply_min_trade(self, new_weights: pd.Series,
                         prev_weights: dict[str, float]) -> pd.Series:
        """Filter out changes below minimum trade threshold."""
        min_change = self._cfg.get("selection", {}).get("min_weight_change_pct", 0.005)
        result = new_weights.copy()
        for k in result.index:
            prev = prev_weights.get(k, 0.0)
            if abs(result[k] - prev) < min_change:
                result[k] = prev
        # Re-normalize after filtering
        return result / max(result.sum(), 0.01)

    # ----------------------------------------------------------------
    # Helper: momentum / vol from ETF data
    # ----------------------------------------------------------------

    def _get_proxy_etfs(self, sleeve: str) -> list[str]:
        """Get ETF symbols for a sleeve from MSE config."""
        proxy = self._mse._config.get("proxy_etfs", {})
        for group in proxy.values():
            for s_name, symbols in group.items():
                if s_name.upper() == sleeve.upper():
                    return symbols
        return []

    def _compute_momentum(self, symbols: list[str], date: pd.Timestamp,
                          lookback: int) -> float | None:
        """Average momentum across ETF proxies."""
        vals = []
        for sym in symbols:
            s = self._mse._prices.get(sym)
            if s is None:
                continue
            idx_arr = s.index.get_indexer([date], method="ffill")
            idx = idx_arr[0]
            if idx < lookback or idx >= len(s):
                continue
            p_now = float(s.iloc[idx])
            p_prev = float(s.iloc[idx - lookback])
            if p_prev > 0 and np.isfinite(p_now):
                vals.append(p_now / p_prev - 1.0)
        return np.mean(vals) if vals else None

    def _compute_vol(self, symbols: list[str], date: pd.Timestamp,
                     lookback: int) -> float | None:
        """Annualized vol from daily returns."""
        all_rets = []
        for sym in symbols:
            s = self._mse._prices.get(sym)
            if s is None:
                continue
            idx_arr = s.index.get_indexer([date], method="ffill")
            idx = idx_arr[0]
            if idx < lookback + 5 or idx >= len(s):
                continue
            prices = s.iloc[idx - lookback: idx + 1].values.astype(np.float64)
            rets = np.diff(prices) / prices[:-1]
            rets = rets[np.isfinite(rets)]
            if len(rets) > 10:
                all_rets.append(np.std(rets, ddof=1) * np.sqrt(252))
        return np.mean(all_rets) if all_rets else None

    def _get_top_n(self, class_name: str) -> int:
        if class_name == "domestic_equity":
            return self._cfg.get("selection", {}).get("top_n_domestic", 3)
        elif class_name == "overseas_equity":
            return self._cfg.get("selection", {}).get("top_n_overseas", 2)
        return len(self._budget.get_class_sleeves(class_name))

    def reset(self):
        """Reset internal state for new experiment or fold."""
        self._prev_sleeve_weights = None

    @property
    def previous_sleeve_weights(self) -> dict[str, float] | None:
        """Return the last sleeve-level target used by the rebalance band."""
        return dict(self._prev_sleeve_weights) if self._prev_sleeve_weights else None

    def set_previous_sleeve_weights(self, weights: dict[str, float]) -> None:
        """Store a sleeve-level target after any post-selection risk scaling.

        The rebalance band operates in sleeve space.  Keeping this setter
        explicit prevents fund-code keys from leaking back into
        ``select_weights`` and makes the ordering (select -> scale -> map)
        auditable.
        """
        self._prev_sleeve_weights = {
            str(key): float(value)
            for key, value in weights.items()
            if float(value) > 0.0
        }

    # ----------------------------------------------------------------
    # Public API: get target weights as fund_code -> weight
    # Strict S1/S2 isolation per P0-1 of planning.md
    # ----------------------------------------------------------------

    def map_sleeves_to_fixed_products(self, sleeve_weights: dict[str, float]) -> pd.Series:
        """Map sleeve weights to fixed verified products.

        Never calls ProductSelector. Reads only from _fixed_products dict.

        Args:
            sleeve_weights: sleeve_name -> weight dict (after vol scaling)

        Returns:
            pd.Series[fund_code -> target_weight]
        """
        fund_weights: dict[str, float] = {}
        for sleeve, weight in sleeve_weights.items():
            if weight <= 0:
                continue

            fund_code = self._fixed_products.get(sleeve)
            if fund_code is None:
                raise ValueError(
                    f"S1 fixed mapping missing for sleeve={sleeve}. "
                    "Do not silently switch to dynamic selection."
                )

            fund_weights[fund_code] = fund_weights.get(fund_code, 0.0) + weight

        return pd.Series(fund_weights)

    def map_to_fixed_products(
        self,
        date: str | pd.Timestamp,
        market_state: str,
        prev_fund_weights: dict[str, float] | None = None,
    ) -> pd.Series:
        """S1: Map sleeve weights to fixed verified products.

        Never calls ProductSelector. Reads only from _fixed_products dict.
        Raises if a sleeve has no mapped product or product lacks NAV on date.

        Uses prev_sleeve_weights (not prev_fund_weights) for rebalance band in select_weights,
        so that key space matches: sleeve names vs sleeve names.  This method does NOT
        update _prev_sleeve_weights internally; callers must invoke
        set_previous_sleeve_weights() after any post-selection risk scaling to keep the
        rebalance band comparing scaled-vs-scaled weights.

        Returns: pd.Series[fund_code -> target_weight]
        """
        sleeve_weights = self.select_weights(
            date, market_state, prev_weights=self._prev_sleeve_weights
        )

        return self.map_sleeves_to_fixed_products(sleeve_weights.to_dict())

    def map_to_dynamic_products(self, date: str | pd.Timestamp,
                                market_state: str,
                                fallback_sleeves: list[str] | None = None) -> pd.Series:
        """S2: Map sleeve weights to dynamically selected products.

        Requires self._product_selector to be set. Never falls back to
        _fixed_products when no candidates are found; instead transfers
        the budget to fallback cash/short-term sleeves (e.g. MONEY_MARKET,
        ULTRA_SHORT_BOND). Raises if product_selector is None.

        Returns: pd.Series[fund_code -> target_weight]
        """
        if self._product_selector is None:
            raise RuntimeError(
                "S2 dynamic selection requires ProductSelector. "
                "It was not injected into ExposureSelector."
            )

        fallback_sleeves = fallback_sleeves or ["MONEY_MARKET"]

        sleeve_weights = self.select_weights(date, market_state)

        fund_weights: dict[str, float] = {}
        orphan_budget: dict[str, float] = {}

        for sleeve, weight in sleeve_weights.items():
            if weight <= 0:
                continue

            funds = self._product_selector.select(sleeve, top_n=self._top_n, date=date)
            if funds:
                per_fund = weight / len(funds)
                for f in funds:
                    fund_weights[f] = fund_weights.get(f, 0.0) + per_fund
            else:
                orphan_budget[sleeve] = weight

        # Transfer orphan budget to fallback sleeves via dynamic selection
        if orphan_budget:
            log.warning(
                "S2 no eligible product for sleeves=%s on %s; transferring budget %.4f to fallback",
                list(orphan_budget.keys()), date, sum(orphan_budget.values())
            )
            for fb_sleeve in fallback_sleeves:
                fb_funds = self._product_selector.select(
                    fb_sleeve, top_n=self._top_n, date=date
                )
                if fb_funds:
                    per_fund = sum(orphan_budget.values()) / len(fb_funds)
                    for f in fb_funds:
                        fund_weights[f] = fund_weights.get(f, 0.0) + per_fund
                    break
            else:
                log.warning(
                    "S2 fallback sleeves also empty on %s; %.4f budget remains unallocated",
                    date, sum(orphan_budget.values())
                )

        return pd.Series(fund_weights)
