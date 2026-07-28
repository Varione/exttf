"""Strategy library: literature-based strategies for each regime."""

import os
import numpy as np
import pandas as pd


class Strategy:
    """Base strategy class."""
    name: str = ""
    regime: int = -1
    description: str = ""
    literature: str = ""
    positive_only: bool = False
    buy_and_hold: bool = False

    def __init__(self, factor_df: pd.DataFrame):
        self.factor_df = factor_df
        date_keys = pd.to_datetime(factor_df["date"]).dt.strftime("%Y-%m-%d")
        self._factor_by_date = date_keys.groupby(date_keys, sort=False).indices

        # 加载资产类别映射（从数据库）
        self._asset_class_map = {}
        try:
            import sqlite3
            db_path = "data/processed/etf.sqlite"
            if os.path.exists(db_path):
                with sqlite3.connect(db_path) as conn:
                    rows = conn.execute(
                        "SELECT symbol, asset_class FROM etf_quality"
                    ).fetchall()
                    self._asset_class_map = {
                        str(r[0]).zfill(6): r[1] for r in rows
                    }
        except Exception:
            pass

    @staticmethod
    def _cs_zscore(series: pd.Series) -> pd.Series:
        """Cross-sectional z-score normalization."""
        mu = series.mean()
        sigma = series.std(ddof=0)
        if sigma < 1e-10:
            return series - mu
        return (series - mu) / sigma

    def _get_day_data(self, date: pd.Timestamp) -> pd.DataFrame:
        """Get factor data for a specific date using O(1) lookup."""
        date_str = date.strftime("%Y-%m-%d")
        row_locs = self._factor_by_date.get(date_str)
        if row_locs is None or len(row_locs) == 0:
            return pd.DataFrame()
        day_data = self.factor_df.iloc[row_locs]
        if "pit_eligible" in day_data.columns:
            eligible = day_data["pit_eligible"]
            if eligible.dtype != bool:
                eligible = eligible.astype(str).str.lower().isin({"1", "true", "yes"})
            day_data = day_data.loc[eligible.fillna(False)]
        return day_data.drop_duplicates(subset="symbol", keep="last")

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        """Return signal for all ETFs at given date. Higher = stronger buy."""
        raise NotImplementedError

    def get_positions(
        self, date: pd.Timestamp, n_hold: int = 20,
        max_weight: float = 0.1, max_per_class: float = 0.3,
    ) -> dict[str, float]:
        """Return portfolio weights for a given date."""
        signals = self.compute_signal(date)
        signals = signals.replace([np.inf, -np.inf], np.nan).dropna()
        if self.positive_only:
            signals = signals[signals > 0]
        signals = signals.sort_values(ascending=False)

        if signals.empty or n_hold <= 0:
            return {}

        top_n = signals.head(min(n_hold, len(signals)))
        weight = min(max_weight, 1.0 / len(top_n))
        positions = {sym: weight for sym in top_n.index}

        # 资产类别上限检查
        if self._asset_class_map and max_per_class < 1.0:
            class_weights: dict[str, float] = {}
            for sym, w in positions.items():
                ac = self._asset_class_map.get(sym, "unknown")
                class_weights[ac] = class_weights.get(ac, 0) + w
            for ac, total_w in class_weights.items():
                if total_w > max_per_class:
                    scale = max_per_class / total_w
                    for sym, w in positions.items():
                        if self._asset_class_map.get(sym) == ac:
                            positions[sym] = w * scale

        return positions


class S01_CrossSectionalMomentum(Strategy):
    """Jegadeesh & Titman (1993): Cross-sectional momentum."""
    name = "S01_CS_Momentum"
    regime = 0
    description = "Buy top decile of 20-day momentum"
    literature = "Jegadeesh & Titman, JF 1993"
    positive_only = True

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)
        return day_data.set_index("symbol")["mom_20"]


class S02_TrendFollowing(Strategy):
    """Welles Wilder (1978): Trend following with ADX filter."""
    name = "S02_Trend_Following"
    regime = 0
    description = "Buy when ADX > threshold and slope positive"
    literature = "Welles Wilder, New Concepts in Technical Trading Systems 1978"
    positive_only = True

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)

        adx = day_data.set_index("symbol")["adx_proxy_14"]
        slope = day_data.set_index("symbol")["slope_20"]
        mom = day_data.set_index("symbol")["mom_20"]

        # Combine: ADX strength * positive direction
        signal = adx.clip(lower=0) * (slope > 0).astype(float) * mom.clip(lower=0)
        return signal


class S03_DualThrust(Strategy):
    """Kaufman (1987): Dual Thrust breakout strategy."""
    name = "S03_Dual_Thrust"
    regime = 0
    description = "Buy recent breakouts using intraday range"
    literature = "Kaufman, Smart Money 1987"
    positive_only = True

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)

        # Use intraday_range and close_pos as breakout proxies
        intraday = day_data.set_index("symbol")["intraday_range"]
        close_pos = day_data.set_index("symbol")["close_pos"]
        mom_5 = day_data.set_index("symbol")["mom_5"]

        signal = intraday * close_pos.clip(0, 1) + mom_5
        return signal


class S04_VolTargetTrend(Strategy):
    """Qian & Basso (2018): Volatility targeting with trend filter."""
    name = "S04_VolTarget_Trend"
    regime = 0
    description = "Dynamic volatility targeting + MA60 trend filter"
    literature = "Qian & Basso, Risk Management 2018"
    positive_only = True

    @staticmethod
    def _estimate_portfolio_vol_daily(
        selected_vols: list[float], rho: float = 0.3
    ) -> float:
        """Estimate equal-weight portfolio daily volatility.

        The pairwise covariance terms appear twice in a variance expansion:
        ``w'Σw = Σ wi²σi² + 2Σ(i<j) wi wj covij``.  The previous
        implementation omitted that factor of two and understated risk.
        """
        values = np.asarray(selected_vols, dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size == 0:
            return 0.0

        w = 1.0 / values.size
        diagonal = float(np.sum((w * values) ** 2))
        pairwise = float(
            sum(values[i] * values[j] for i in range(values.size) for j in range(i + 1, values.size))
        )
        variance = diagonal + 2.0 * float(rho) * (w**2) * pairwise
        return float(np.sqrt(max(variance, 0.0)))

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
                      max_weight: float = 0.05,
                      target_vol_annual: float = 0.10) -> dict[str, float]:
        signals = self.compute_signal(date)
        signals = signals.replace([np.inf, -np.inf], np.nan).dropna()
        signals = signals[signals > 0].sort_values(ascending=False)
        if signals.empty or n_hold <= 0:
            return {}

        top_n = signals.head(min(n_hold, len(signals)))

        # 获取各ETF的日波动率
        day_data = self._get_day_data(date)
        if len(day_data) == 0:
            return {}

        vol_series = day_data.set_index("symbol")["real_vol_10"]
        selected_vols = [vol_series[sym] for sym in top_n.index if sym in vol_series.index]

        if not selected_vols:
            return {}

        n = len(selected_vols)

        rho = 0.3
        port_vol_daily = self._estimate_portfolio_vol_daily(selected_vols, rho=rho)

        if not np.isfinite(port_vol_daily) or port_vol_daily <= 0:
            return {}

        # 波动率目标缩放
        target_vol_daily = target_vol_annual / np.sqrt(252)
        scale = min(1.0, target_vol_daily / port_vol_daily)

        # 应用 max_weight 限制
        weight = min(max_weight, scale / n)
        return {sym: weight for sym in top_n.index}


class S11_MeanReversion(Strategy):
    """De Long et al. (1990): Mean reversion in efficient markets."""
    name = "S11_Mean_Reversion"
    regime = 1
    description = "Buy oversold ETFs (price below MA)"
    literature = "De Long et alal, JPE 1990"
    positive_only = True

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)

        ibias = day_data.set_index("symbol")["ibias_20"]
        mr_speed = day_data.set_index("symbol")["mr_speed_20"]
        rsi = day_data.set_index("symbol")["rsi_14"]

        # Negative ibias = price below MA (oversold)
        # Low RSI = oversold
        signal = -ibias + (-mr_speed) + (50 - rsi).clip(lower=0) / 50
        return signal


class S12_LowVolatility(Strategy):
    """Ang et al. (2006): Low volatility anomaly."""
    name = "S12_Low_Volatility"
    regime = 1
    description = "Buy lowest volatility ETFs"
    literature = "Ang et al, JF 2006"

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)

        vol = day_data.set_index("symbol")["real_vol_10"]
        vol = vol[vol > 0]

        # Inverse volatility ranking
        signal = -vol
        return signal


class S13_QualityFactor(Strategy):
    """Fama-French (2015): Quality/Profitability factor."""
    name = "S13_Quality"
    regime = 1
    description = "Buy high quality (high omega, positive skew) ETFs"
    literature = "Fama-French, JFE 2015"

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)

        idx = day_data.set_index("symbol")
        omega_z = self._cs_zscore(idx["omega_20"])
        skew_z = self._cs_zscore(idx["skew_20"])
        win_z = self._cs_zscore(idx["win_rate_20"])

        return omega_z + skew_z.clip(-3, 3) + win_z


class S21_LowVolDefense(Strategy):
    """Ang et al. (2006): Low volatility defense in bear markets."""
    name = "S21_LowVol_Defense"
    regime = 2
    description = "Defensive: lowest volatility ETFs"
    literature = "Ang et al, JF 2006"

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)

        idx = day_data.set_index("symbol")
        vol_z = self._cs_zscore(idx["real_vol_10"])
        omega_z = self._cs_zscore(idx["omega_20"])

        valid = vol_z[
            vol_z.index.intersection(
                idx["real_vol_10"][idx["real_vol_10"] > 0].index
            )
        ]
        return -valid + omega_z.loc[valid.index] / 2


class S22_TailRiskDefense(Strategy):
    """Kelly & Jiang (2014): Tail risk hedging."""
    name = "S22_Tail_Risk"
    regime = 2
    description = "Defensive: high tail_ratio, low kurtosis ETFs"
    literature = "Kelly & Jiang, RFS 2014"

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)

        tail_ratio = day_data.set_index("symbol")["tail_ratio_60"]
        kurt = day_data.set_index("symbol")["kurt_20"]
        skew = day_data.set_index("symbol")["skew_20"]

        # High tail ratio, low kurtosis, positive skew = safer
        signal = tail_ratio - (kurt - 1) / 2 + skew.clip(-1, 1)
        return signal


class S23_OversoldLong(Strategy):
    """Oversold mean-reversion long-only strategy."""
    name = "S23_Oversold_Long"
    regime = 2
    description = "Long oversold ETFs based on RSI and stochastic divergence"
    literature = "Gatev et al, JBF 2006"
    positive_only = True

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)

        # Use RSI divergence as mean-reversion signal
        rsi = day_data.set_index("symbol")["rsi_14"]
        stoch = day_data.set_index("symbol")["stoch_k_14"]

        # Combine oversold signals (normalized to 0-1)
        rsi_signal = (30 - rsi).clip(lower=0) / 30
        stoch_signal = (20 - stoch).clip(lower=0) / 20

        signal = rsi_signal + stoch_signal
        return signal


class B0_BuyHoldEW(Strategy):
    """Buy and hold equal weight benchmark."""
    name = "B0_BuyHold_EW"
    regime = -1
    description = "Equal weight buy and hold all ETFs"
    literature = "Benchmark"
    buy_and_hold = True

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)
        return pd.Series(1.0, index=day_data["symbol"].astype(str))

    def get_positions(self, date: pd.Timestamp, n_hold: int = 20,
                      max_weight: float = 0.1) -> dict[str, float]:
        signals = self.compute_signal(date)
        if len(signals) < 1:
            return {}

        # True equal weight: all available ETFs with equal weight (no n_hold limit)
        n_etfs = len(signals)
        weight = min(max_weight, 1.0 / n_etfs)
        return {sym: weight for sym in signals.index}


class B1_RegimeAgnosticMomentum(Strategy):
    """Always follow momentum regardless of regime."""
    name = "B1_Momentum_Agnostic"
    regime = -1
    description = "Cross-sectional momentum in all regimes"
    literature = "Benchmark"
    positive_only = True

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 20:
            return pd.Series(dtype=float)
        return day_data.set_index("symbol")["mom_20"]


class B2_Cash(Strategy):
    """Cash benchmark: zero exposure, earns cash_daily_return."""
    name = "B2_Cash"
    regime = -1
    description = "Cash benchmark (daily risk-free rate)"
    literature = "Benchmark"
    buy_and_hold = True

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        return pd.Series(dtype=float)

    def get_positions(self, date: pd.Timestamp, n_hold: int = 20,
                      max_weight: float = 0.1) -> dict[str, float]:
        return {}


class B3_IndexProxy(Strategy):
    """Broad market index proxy: equal weight of large-cap stock ETFs."""
    name = "B3_Index_Proxy"
    regime = -1
    description = "Equal-weight broad market index proxy (large-cap stock ETFs)"
    literature = "Benchmark"
    buy_and_hold = True

    _INDEX_PROXY_SYMBOLS = {
        "159919", "159949", "510300", "510050", "510500",
        "159915", "159920", "510030", "510180", "510880",
    }

    def compute_signal(self, date: pd.Timestamp) -> pd.Series:
        day_data = self._get_day_data(date)
        if len(day_data) < 5:
            return pd.Series(dtype=float)
        idx = day_data["symbol"].astype(str)
        proxy = day_data[idx.isin(self._INDEX_PROXY_SYMBOLS)]
        if proxy.empty:
            proxy = day_data.head(10)
        return pd.Series(1.0, index=proxy["symbol"].astype(str))

    def get_positions(self, date: pd.Timestamp, n_hold: int = 20,
                      max_weight: float = 0.1) -> dict[str, float]:
        signals = self.compute_signal(date)
        if len(signals) < 1:
            return {}
        n_etfs = len(signals)
        weight = min(max_weight, 1.0 / n_etfs)
        return {sym: weight for sym in signals.index}


# Registry
STRATEGIES = {
    s.name: s for s in [
        S01_CrossSectionalMomentum,
        S02_TrendFollowing,
        S03_DualThrust,
        S04_VolTargetTrend,
        S11_MeanReversion,
        S12_LowVolatility,
        S13_QualityFactor,
        S21_LowVolDefense,
        S22_TailRiskDefense,
        S23_OversoldLong,
        B0_BuyHoldEW,
        B1_RegimeAgnosticMomentum,
        B2_Cash,
        B3_IndexProxy,
    ]
}


def get_strategies_for_regime(regime: int) -> list[str]:
    """Get strategy names for a specific regime."""
    return [name for name, s in STRATEGIES.items() if s.regime == regime]


def get_all_strategy_names() -> list[str]:
    return list(STRATEGIES.keys())
