"""Market State Engine — transparent, interpretable state classifier.

Usage:
    engine = MarketStateEngine()
    state, scores, features = engine.get_state("2024-01-15")
    print(f"State: {state}, Scores: {scores}")
"""

from __future__ import annotations
import sqlite3
import yaml
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

from .schedule import build_month_end_schedule


class MarketStateEngine:
    """Transparent market state classifier using ETF price data.

    Computes 6 feature groups → maps to 5 state scores → applies hysteresis.
    All features computed from publicly available ETF total_return_proxy prices.
    """

    STATES = ["RISK_ON", "NEUTRAL", "INFLATION_REAL_ASSET", "DEFLATION_RATE_DOWN", "STRESS"]

    # Risk level per state: 1=lowest (high cash), 5=highest (full equity)
    STATE_RISK_LEVEL: dict[str, int] = {
        "RISK_ON": 5,
        "INFLATION_REAL_ASSET": 4,
        "NEUTRAL": 3,
        "DEFLATION_RATE_DOWN": 2,
        "STRESS": 1,
    }

    def __init__(self, config_path: str = "config/market_state.yaml"):
        self._config = self._load_config(config_path)
        self._etf_db = "data/processed/etf.sqlite"
        self._otf_db = "data/processed/otf_expanded.sqlite"
        self._prices: dict[str, pd.Series] = {}
        self._bond_nav: dict[str, pd.Series] = {}
        self._all_dates: pd.DatetimeIndex | None = None
        self._state_entry_date: pd.Timestamp | None = None
        self._current_state: str | None = None
        self._consecutive_counts: dict[str, int] = {}
        self._load_data()

    # ----------------------------------------------------------------
    # Data Loading
    # ----------------------------------------------------------------

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_data(self):
        """Load ETF total_return_proxy prices and OTF bond NAV data."""
        proxy = self._config["proxy_etfs"]
        all_symbols = set()
        for group in proxy.values():
            for sleeve_symbols in group.values():
                all_symbols.update(sleeve_symbols)

        conn = sqlite3.connect(self._etf_db)
        for sym in all_symbols:
            df = pd.read_sql(
                "SELECT date, total_return_proxy FROM etf_daily_price_modes "
                "WHERE symbol=? AND total_return_proxy IS NOT NULL ORDER BY date",
                conn, params=(sym,)
            )
            if len(df) > 0:
                s = pd.Series(df["total_return_proxy"].values.astype(np.float64),
                              index=pd.DatetimeIndex(pd.to_datetime(df["date"]), name="date"),
                              name=sym)
                s.index = s.index.normalize()
                self._prices[sym] = s
        conn.close()

        # Load OTF bond NAV data
        bond = self._config.get("bond_proxies", {})
        for sleeve, fund_code in bond.items():
            conn = sqlite3.connect(self._otf_db)
            try:
                df = pd.read_sql(
                    "SELECT nav_date, cumulative_nav FROM otf_fund_nav "
                    "WHERE fund_code=? AND cumulative_nav IS NOT NULL ORDER BY nav_date",
                    conn, params=(fund_code,)
                )
                if len(df) > 0:
                    s = pd.Series(df["cumulative_nav"].values.astype(np.float64),
                                  index=pd.DatetimeIndex(pd.to_datetime(df["nav_date"]), name="date"),
                                  name=f"bond_{sleeve}")
                    s.index = s.index.normalize()
                    self._bond_nav[f"bond_{sleeve}"] = s
            except Exception:
                pass
            conn.close()

        # Build unified date axis
        all_dates = set()
        for s in list(self._prices.values()) + list(self._bond_nav.values()):
            all_dates.update(s.index)
        self._all_dates = pd.DatetimeIndex(sorted(all_dates))

    def _get_price(self, symbol: str, date: pd.Timestamp) -> float | None:
        s = self._prices.get(symbol)
        if s is None or date not in s.index:
            return None
        return float(s.loc[date])

    def _get_returns(self, symbols: list[str], date: pd.Timestamp, period: int) -> float | None:
        """Compute average period return for a basket of symbols."""
        vals = []
        for sym in symbols:
            s = self._prices.get(sym)
            if s is None:
                continue
            idx_arr = s.index.get_indexer([date], method="ffill")
            idx = idx_arr[0]
            if idx < period or idx >= len(s):
                continue
            p_now = float(s.iloc[idx])
            p_before = float(s.iloc[idx - period])
            if np.isfinite(p_now) and np.isfinite(p_before) and p_before > 0:
                vals.append(p_now / p_before - 1.0)
        return np.mean(vals) if vals else None

    def _get_vol(self, symbols: list[str], date: pd.Timestamp, period: int) -> float | None:
        """Compute realized volatility (std of daily returns) for a basket."""
        all_returns = []
        for sym in symbols:
            s = self._prices.get(sym)
            if s is None:
                continue
            idx_arr = s.index.get_indexer([date], method="ffill")
            idx = idx_arr[0]
            if idx < period + 5 or idx >= len(s):
                continue
            prices = s.iloc[idx - period: idx + 1].values.astype(np.float64)
            rets = np.diff(prices) / prices[:-1]
            rets = rets[np.isfinite(rets)]
            if len(rets) > 5:
                all_returns.append(np.std(rets, ddof=1) * np.sqrt(252))
        return np.mean(all_returns) if all_returns else None

    def _get_rolling_corr(self, sym_a: str, sym_b: str, date: pd.Timestamp,
                          window: int) -> float | None:
        """Rolling correlation between two symbols."""
        sa = self._prices.get(sym_a)
        sb = self._prices.get(sym_b)
        if sa is None or sb is None:
            return None
        idx_arr_a = sa.index.get_indexer([date], method="ffill")
        idx_arr_b = sb.index.get_indexer([date], method="ffill")
        idx_a, idx_b = idx_arr_a[0], idx_arr_b[0]
        if idx_a < window + 2 or idx_b < window + 2:
            return None
        pa = sa.iloc[idx_a - window: idx_a + 1].values.astype(np.float64)
        pb = sb.iloc[idx_b - window: idx_b + 1].values.astype(np.float64)
        ra = np.diff(pa) / pa[:-1]
        rb = np.diff(pb) / pb[:-1]
        ra = ra[np.isfinite(ra) & np.isfinite(rb)]
        rb = rb[np.isfinite(ra) & np.isfinite(rb)]
        if len(ra) < 10:
            return None
        corr = np.corrcoef(ra, rb)[0, 1]
        return float(corr) if np.isfinite(corr) else None

    # ----------------------------------------------------------------
    # Feature Computation
    # ----------------------------------------------------------------

    def _compute_features(self, date: pd.Timestamp) -> dict[str, float]:
        """Compute all features for a given date."""
        f = {}

        # --- Trend features ---
        # Domestic equity trend
        for period in self._config["feature"]["trend_periods"]:
            dom_ret = self._get_returns(
                ["510300", "159919", "510500", "159845", "159915"], date, period)
            f[f"trend_equity_{period}"] = dom_ret if dom_ret is not None else 0.0

            ov_ret = self._get_returns(
                ["513500", "513100", "159920"], date, period)
            f[f"trend_overseas_{period}"] = ov_ret if ov_ret is not None else 0.0

            gold_ret = self._get_returns(["518880"], date, period)
            f[f"trend_gold_{period}"] = gold_ret if gold_ret is not None else 0.0

        # Bond trend (from OTF NAV)
        for bond_name, bond_series in self._bond_nav.items():
            idx_arr = bond_series.index.get_indexer([date], method="ffill")
            idx = idx_arr[0]
            for period in [60, 120]:
                if idx >= period and idx < len(bond_series):
                    p_now = float(bond_series.iloc[idx])
                    p_before = float(bond_series.iloc[idx - period])
                    if p_before > 0:
                        f[f"trend_{bond_name}_{period}"] = p_now / p_before - 1.0

        # Composite bond trend
        bond_fields = [k for k in f if k.startswith("trend_bond_")]
        if bond_fields:
            f["trend_bond_60"] = np.mean([f[k] for k in bond_fields if "60" in str(k)])
            f["trend_bond_120"] = np.mean([f[k] for k in bond_fields if "120" in str(k)])
        else:
            f["trend_bond_60"] = 0.0
            f["trend_bond_120"] = 0.0

        # --- Volatility features ---
        for period in self._config["feature"]["vol_periods"]:
            vol_equity = self._get_vol(
                ["510300", "159919", "510500", "159915"], date, period)
            f[f"vol_equity_{period}"] = vol_equity if vol_equity is not None else 0.0

        # Vol expansion (ratio of short/medium vol)
        vol_20 = f.get("vol_equity_20", 0.0)
        vol_60 = f.get("vol_equity_60", 0.0)
        f["vol_expansion"] = (vol_20 / vol_60 - 1.0) if vol_60 > 0.001 else 0.0

        # --- Breadth features ---
        breadth_ok = 0
        breadth_total = 0
        for sleeve_syms in self._config["proxy_etfs"]["domestic_equity"].values():
            for sym in sleeve_syms:
                s = self._prices.get(sym)
                if s is None:
                    continue
                idx_arr = s.index.get_indexer([date], method="ffill")
                idx = idx_arr[0]
                if idx < 200 or idx >= len(s):
                    continue
                price_now = float(s.iloc[idx])
                price_200d = float(s.iloc[idx - 200])
                price_120d = float(s.iloc[idx - 120])
                if price_200d > 0 and price_120d > 0:
                    above_ma = price_now > price_200d
                    positive_mom = (price_now / price_120d - 1.0) > 0
                    if above_ma and positive_mom:
                        breadth_ok += 1
                    breadth_total += 1
        f["breadth_120_200"] = breadth_ok / max(breadth_total, 1)

        # --- Correlation features ---
        f["corr_equity_gold_60"] = self._get_rolling_corr(
            "510300", "518880", date, self._config["feature"]["corr_window"]) or 0.0
        f["corr_equity_bond_60"] = self._get_rolling_corr(
            "510300", "511660", date, self._config["feature"]["corr_window"]) or 0.0

        # --- Max drawdown ---
        for period in [60, 120]:
            sym = "510300"
            s = self._prices.get(sym)
            if s is not None:
                idx_arr = s.index.get_indexer([date], method="ffill")
                idx = idx_arr[0]
                if idx >= period and idx < len(s):
                    prices = s.iloc[idx - period: idx + 1].values.astype(np.float64)
                    rolling_max = np.maximum.accumulate(prices)
                    dd = (prices - rolling_max) / rolling_max
                    f[f"max_drawdown_{period}"] = float(np.min(dd))
                else:
                    f[f"max_drawdown_{period}"] = 0.0

        # --- Dividend vs Growth (value vs growth proxy) ---
        div_ret = self._get_returns(["510880"], date, 60)
        growth_ret = self._get_returns(["159915", "159949"], date, 60)
        if div_ret is not None and growth_ret is not None and growth_ret != 0:
            f["dividend_vs_growth_60"] = div_ret - growth_ret
        else:
            f["dividend_vs_growth_60"] = 0.0

        # --- Gold vs Equity (inflation proxy) ---
        gold_ret = self._get_returns(["518880"], date, 60)
        equity_ret = self._get_returns(["510300", "159919"], date, 60)
        if gold_ret is not None and equity_ret is not None:
            f["gold_vs_equity_60"] = gold_ret - equity_ret
        else:
            f["gold_vs_equity_60"] = 0.0

        # --- Rate trend (money market as proxy) ---
        mm_ret = self._get_returns(["511660", "511880"], date, 60)
        if mm_ret is not None:
            f["rate_trend_60"] = mm_ret * 252 / 60
        else:
            f["rate_trend_60"] = 0.0

        # Fill any remaining None
        for k, v in f.items():
            if v is None or not np.isfinite(v):
                f[k] = 0.0

        return f

    # ----------------------------------------------------------------
    # Score Computation
    # ----------------------------------------------------------------

    def _compute_scores(self, features: dict[str, float],
                        hist_features: list[dict[str, float]] | None = None) -> dict[str, float]:
        """Map features to 5 state scores with hard gates + continuous scoring.

        Each state gets a base continuous score AND two hard gates:
        - Exclusion gate: if TRUE, state score is floored to 20 (cannot dominate)
        - Boost gate: if TRUE, state score gets a +20 boost

        This ensures states only trigger when their defining conditions are met.
        """
        f = features

        # ---- Continuous sub-scores (0-100 each) ----
        def score_equity_trend(period=120):
            v = f.get(f"trend_equity_{period}", 0.0)
            return np.clip((v + 0.30) / 0.60 * 100, 0, 100) if v > -0.30 else 0.0

        def score_equity_weakness(period=120):
            v = f.get(f"trend_equity_{period}", 0.0)
            return np.clip((-v) / 0.20 * 100, 0, 100)

        def score_high_vol():
            v = f.get("vol_equity_60", 0.0)
            return np.clip((v - 0.10) / 0.30 * 100, 0, 100)

        def score_low_vol():
            v = f.get("vol_equity_60", 0.0)
            return np.clip((0.30 - v) / 0.20 * 100, 0, 100) if v < 0.30 else 0.0

        def score_breadth():
            v = f.get("breadth_120_200", 0.0)
            return v * 100

        def score_gold_up():
            v = f.get("trend_gold_120", 0.0)
            return np.clip(v / 0.25 * 100, 0, 100) if v > 0 else 0.0

        def score_gold_weak():
            v = f.get("trend_gold_120", 0.0)
            return np.clip((-v) / 0.15 * 100, 0, 100) if v < 0 else 0.0

        def score_gold_beat_equity():
            g = f.get("trend_gold_120", 0.0)
            e = f.get("trend_equity_120", 0.0)
            spread = g - e
            return np.clip(spread / 0.30 * 100, 0, 100)

        def score_bond_up():
            v = f.get("trend_bond_60", 0.0)
            return np.clip(v / 0.05 * 100, 0, 100) if v > 0 else 0.0

        def score_drawdown():
            v = f.get("max_drawdown_60", 0.0)
            return np.clip((-v) / 0.15 * 100, 0, 100)

        # ---- RISK_ON ----
        risk_on = (
            0.30 * score_equity_trend(120) +
            0.15 * score_equity_trend(60) +
            0.10 * (f.get("trend_overseas_120", 0.0) + 0.20) / 0.40 * 100 +
            0.20 * score_breadth() +
            0.15 * score_low_vol() +
            0.10 * (1.0 - f.get("corr_equity_gold_60", 0.0) + 1.0) / 2.0 * 100
        )
        risk_on_gate = f.get("trend_equity_120", -1.0) > -0.05 and f.get("breadth_120_200", 0) > 0.15

        # ---- INFLATION_REAL_ASSET ----
        inflation = (
            0.35 * score_gold_up() +
            0.25 * score_gold_beat_equity() +
            0.20 * (f.get("dividend_vs_growth_60", 0.0) + 0.20) / 0.40 * 100 +
            0.20 * (f.get("rate_trend_60", 0.0)) / 0.02 * 100
        )
        inflation_gate = f.get("trend_gold_120", 0.0) > 0.03

        # ---- DEFLATION_RATE_DOWN ----
        deflation = (
            0.30 * score_bond_up() +
            0.20 * score_equity_weakness(120) +
            0.15 * score_gold_weak() +
            0.10 * (0.03 - f.get("rate_trend_60", 0.0)) / 0.03 * 100 +
            0.15 * (1.0 - f.get("breadth_120_200", 0.0)) * 100 +
            0.10 * score_high_vol()
        )
        deflation_gate = f.get("trend_equity_120", 0.0) < -0.03 or (
            f.get("trend_equity_60", 0.0) < -0.01 and f.get("vol_equity_60", 0.0) < 0.20
        )

        # ---- STRESS ----
        stress = (
            0.25 * score_high_vol() +
            0.20 * score_equity_weakness(60) +
            0.15 * score_equity_weakness(120) +
            0.15 * (1.0 - f.get("breadth_120_200", 0.0)) * 100 +
            0.15 * score_drawdown() +
            0.10 * np.clip((f.get("vol_expansion", 0.0) + 0.50) / 1.0 * 100, 0, 100)
        )
        stress_gate = f.get("vol_equity_60", 0.0) > 0.15 and f.get("trend_equity_60", 0.0) < -0.03

        # ---- Mutual exclusion for competing states ----
        # STRESS and DEFLATION_RATE_DOWN are competing: prefer STRESS when both gates fire
        if stress_gate and deflation_gate:
            deflation *= 0.5  # halve deflation score when stress gate also fires
        # STRESS and RISK_ON are competing: prefer STRESS when its gate fires
        if stress_gate:
            risk_on *= 0.3  # severely reduce RISK_ON during stress

        # ---- Apply gates ----
        scores = {}
        for name, raw_score, gate in [
            ("RISK_ON", risk_on, risk_on_gate),
            ("INFLATION_REAL_ASSET", inflation, inflation_gate),
            ("DEFLATION_RATE_DOWN", deflation, deflation_gate),
            ("STRESS", stress, stress_gate),
        ]:
            score = max(raw_score, (20.0 if gate else 0.0))
            scores[name] = round(float(np.clip(score, 0, 100)), 1)

        # ---- NEUTRAL ----
        other_max = max(scores.values())
        neutral = max(50.0 - max(0, other_max - 30.0), 5.0)
        scores["NEUTRAL"] = round(neutral, 1)

        return scores

    # ----------------------------------------------------------------
    # State Mapping with Hysteresis
    # ----------------------------------------------------------------

    def _map_state(self, scores: dict[str, float], prev_state: str | None = None,
                   prev_scores: dict[str, float] | None = None,
                   date: pd.Timestamp | None = None) -> str:
        """Map scores to discrete state with hysteresis and temporal constraints.

        Implements:
        - confirm_months: require N consecutive observations confirming switch
        - min_state_duration_days: prevent switching out before minimum tenure
        - fast_switch_threshold: bypass both checks for extreme score gaps
        """
        if not scores:
            return "NEUTRAL"

        best_state = max(scores, key=lambda k: (scores[k], k == "NEUTRAL"))

        # Update consecutive observation counts
        for s in self.STATES:
            self._consecutive_counts[s] = (
                self._consecutive_counts.get(s, 0) + 1
                if s == best_state else 0
            )

        # No previous state → initialise and return best
        if prev_state is None:
            self._state_entry_date = date
            self._current_state = best_state
            return best_state

        best_score = scores[best_state]
        prev_best = scores.get(prev_state, 0.0)

        hysteresis = self._config["hysteresis"]
        fast_threshold = hysteresis["fast_switch_threshold"]
        confirm_months = hysteresis.get("confirm_months", 1)
        min_duration = hysteresis.get("min_state_duration_days", 0)

        # Fast switch: only allowed when moving to LOWER risk state (per P1-2)
        if best_score - prev_best >= fast_threshold:
            current_risk = self.STATE_RISK_LEVEL.get(prev_state, 3)
            new_risk = self.STATE_RISK_LEVEL.get(best_state, 3)
            if new_risk < current_risk:
                self._state_entry_date = date
                self._current_state = best_state
                return best_state

        # Staying in same state
        if best_state == prev_state:
            return prev_state

        # Min duration check: prevent switching out of a state too soon
        if self._state_entry_date is not None and date is not None and min_duration > 0:
            days_in_state = (date - self._state_entry_date).days
            if days_in_state < min_duration:
                return prev_state

        # Confirmation check: require N consecutive observations
        if confirm_months > 1:
            if self._consecutive_counts.get(best_state, 0) < confirm_months:
                return prev_state

        # Normal switch: need best > prev by margin
        switch_margin = 10.0
        if best_score > prev_best + switch_margin:
            self._state_entry_date = date
            self._current_state = best_state
            return best_state

        return prev_state

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def get_state(self, date: str | pd.Timestamp, prev_state: str | None = None,
                  prev_features: dict[str, float] | None = None) -> tuple[str, dict[str, float], dict[str, float]]:
        """End-to-end: compute features → scores → state.

        Returns:
            (state_name, scores_dict, features_dict)
        """
        dt = pd.Timestamp(date).normalize() if isinstance(date, str) else date.normalize()

        features = self._compute_features(dt)
        scores = self._compute_scores(features)

        state = self._map_state(scores, prev_state, date=dt)

        return state, scores, features

    def get_state_series(self, dates: list[pd.Timestamp]) -> pd.DataFrame:
        """Compute state for a series of dates with hysteresis."""
        results = []
        prev_state = None
        for dt in sorted(dates):
            state, scores, features = self.get_state(dt, prev_state)
            row = {"date": dt, "state": state, **scores}
            results.append(row)
            prev_state = state
        return pd.DataFrame(results).set_index("date")

    def simulate_monthly(self, start: str = "2018-01-01",
                          end: str = "2026-07-17") -> pd.DataFrame:
        """Simulate monthly state observations using proper month-end schedule."""
        if self._all_dates is None or len(self._all_dates) == 0:
            return pd.DataFrame()
        dates = build_month_end_schedule(self._all_dates, start, end)
        if not dates:
            return pd.DataFrame()
        return self.get_state_series(dates)

    def reset(self):
        """Reset internal state for independent experiments (per P1-2).

        Clears hysteresis history so each experiment starts fresh without
        contamination from previous runs.
        """
        self._state_entry_date = None
        self._current_state = None
        self._consecutive_counts = {}
