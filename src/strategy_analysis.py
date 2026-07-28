"""Regime-based strategy analysis using best LSTM model."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from train_classifier import RegimeLSTM, RegimeClassifier, RegimeTransformer
from detect_regimes import load_factors, get_selected_factors, compute_daily_factors, detect_regimes_train_only


# ── Load model and data ──────────────────────────────────────────────

LOOKBACK = 20
N_FACTORS = 52

factor_cols = get_selected_factors(0.5)
df = load_factors()


def load_model(model_type: str = "LSTM"):
    """Load saved model checkpoint."""
    ckpt = torch.load(f"data/processed/regime_model_{model_type}.pt", weights_only=False)
    n_classes = ckpt["n_classes"]

    if model_type == "LSTM":
        model = RegimeLSTM(input_dim=N_FACTORS, n_classes=n_classes, hidden_dim=128,
                           num_layers=2, dropout=0.3, bidirectional=True)
    elif model_type == "MLP":
        model = RegimeClassifier(input_dim=LOOKBACK * N_FACTORS, n_classes=n_classes)
    else:
        model = RegimeTransformer(input_dim=N_FACTORS, n_classes=n_classes, d_model=64,
                                  nhead=8, num_layers=3, dim_feedforward=256, dropout=0.2)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def build_sequences(factor_cols: list[str], lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Build factor sequences for all dates."""
    sampled = df[df["symbol"].isin(df["symbol"].value_counts().head(100).index)]
    daily = sampled.groupby("date")[factor_cols].mean().sort_index()

    sequences = []
    dates = []
    n = len(daily)
    for i in range(lookback, n):
        seq = daily.iloc[i - lookback:i + 1].values
        if not np.any(np.isnan(seq)) and not np.any(np.isinf(seq)):
            sequences.append(seq)
            dates.append(daily.index[i])

    X = np.array(sequences, dtype=np.float64)
    # Clip
    for col in range(X.shape[2]):
        flat = X[:, :, col].flatten()
        q1, q3 = np.percentile(flat, [1, 99])
        X[:, :, col] = np.clip(X[:, :, col], q1, q3)

    return X, dates


@torch.no_grad()
def predict_regimes(model, X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Predict regime for all sequences."""
    X_norm = (X - mean.reshape(1, -1, N_FACTORS)) / std.reshape(1, -1, N_FACTORS)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    preds = []
    probs_list = []
    for i in range(0, len(X_tensor), 256):
        batch = X_tensor[i:i+256]
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).numpy()
        preds.append(logits.argmax(dim=1).numpy())
        probs_list.append(probs)

    return np.concatenate(preds), np.vstack(probs_list)


# ── Strategy definitions ─────────────────────────────────────────────

class RegimeStrategy:
    """Regime-dependent strategy with different rules per regime."""

    def __init__(self, regime_labels: np.ndarray, factor_data: pd.DataFrame):
        self.regime = regime_labels
        self.factor_data = factor_data

    def momentum_signal(self) -> np.ndarray:
        """Momentum-based signal: buy high mom, sell low mom."""
        mom = self.factor_data["mom_20"].values.copy()
        # Cross-sectional rank
        mom_rank = pd.Series(mom).rank(pct=True).values
        return mom_rank

    def mean_reversion_signal(self) -> np.ndarray:
        """Mean reversion signal: buy when price is below MA."""
        ibias = self.factor_data["ibias_20"].values.copy()
        # Negative ibias = price below MA → buy signal
        mr_rank = pd.Series(-ibias).rank(pct=True).values
        return mr_rank

    def volatility_signal(self) -> np.ndarray:
        """Low vol anomaly: buy low volatility stocks."""
        vol = self.factor_data["real_vol_10"].values.copy()
        vol[vol <= 0] = np.nan
        vol_rank = pd.Series(-vol).rank(pct=True, na_option="skip").values
        return vol_rank

    def volume_signal(self) -> np.ndarray:
        """Volume surge signal."""
        vol_ma = self.factor_data["vol_ma_5"].values.copy()
        vol_rank = pd.Series(vol_ma).rank(pct=True).values
        return vol_rank

    def composite_signal(self, regime: int) -> str:
        """Return recommended strategy for a given regime."""
        strategies = {
            0: "mean_reversion",   # 低波动/横盘 → 均值回归
            1: "momentum",         # 常态/温和上涨 → 动量跟随
            2: "defensive",        # 下跌趋势 → 防御（低波动+空仓）
        }
        return strategies.get(regime, "momentum")


# ── Analysis ─────────────────────────────────────────────────────────

def analyze_regime_performance(
    dates: list,
    regime_preds: np.ndarray,
    probs: np.ndarray,
    factor_cols: list[str],
):
    """Analyze characteristics and optimal strategy per regime."""

    # Get daily cross-sectional data for analysis dates
    sampled = df[df["symbol"].isin(df["symbol"].value_counts().head(100).index)]
    daily = sampled.groupby("date")[factor_cols].mean().sort_index()

    # Map predictions to daily dates
    pred_dates = pd.Series(regime_preds, index=pd.DatetimeIndex(dates), name="predicted_regime")

    print("=" * 70)
    print("REGIME-BASED STRATEGY ANALYSIS")
    print("=" * 70)
    print(f"Period: {dates[0]} ~ {dates[-1]} ({len(dates)} trading days)")
    print(f"Model: LSTM (lookback={LOOKBACK})")
    print()

    for regime_id in sorted(np.unique(regime_preds)):
        mask = regime_preds == regime_id
        n_days = mask.sum()
        avg_confidence = probs[mask].max(axis=1).mean()

        # Get dates for this regime
        regime_dates = [dates[i] for i in range(len(dates)) if mask[i]]

        print(f"{'─'*60}")
        print(f"REGIME {regime_id}: {n_days} days ({n_days/len(dates)*100:.1f}%)")
        print(f"Average confidence: {avg_confidence:.3f}")
        print()

        # Factor characteristics
        regime_daily = daily.loc[regime_dates] if pd.DatetimeIndex(regime_dates).intersection(daily.index).size > 0 else daily.iloc[:0]

        if len(regime_daily) > 0:
            means = regime_daily.mean()
            print(f"Key factor characteristics:")

            key_groups = {
                "Momentum": ["mom_5", "mom_20", "mom_60", "win_rate_20"],
                "Trend": ["r2_20", "slope_20", "hull_dist_20"],
                "Volatility": ["real_vol_10", "tr_vol_20", "vol_ratio_5_20"],
                "Volume": ["vol_ma_5", "vp_corr_20", "amount_vol_ratio"],
                "Risk": ["mdd_60", "omega_20", "down_capture_20", "tail_ratio_60"],
                "Distribution": ["skew_20", "kurt_20", "autocorr_20"],
                "Trend Strength": ["adx_proxy_14", "choppiness_14", "trend_str_10"],
            }

            for group_name, factors in key_groups.items():
                vals = [(f, means[f]) for f in factors if f in means.index]
                if vals:
                    top = sorted(vals, key=lambda x: -abs(x[1]))[:3]
                    print(f"  {group_name:20s} | ", end="")
                    for f, v in top:
                        print(f"{f}={v:+.4f} ", end="")
                    print()

            # Recommended strategy
            strat = RegimeStrategy(regime_preds, regime_daily)
            rec = strat.composite_signal(regime_id)
            print(f"\n  Recommended Strategy: {rec.upper()}")

            if regime_id == 0:
                print(f"    → 横盘震荡市：均值回归策略")
                print(f"    → 买入偏离均线过远的标的，等待回归")
                print(f"    → 信号因子：ibias_20, mr_speed_20, boll_pos_10")
            elif regime_id == 1:
                print(f"    → 温和上涨市：动量跟随策略")
                print(f"    → 买入动量强的标的，持有至趋势衰竭")
                print(f"    → 信号因子：mom_20, mom_60, win_rate_20, slope_20")
            else:
                print(f"    → 下跌趋势市：防御策略")
                print(f"    → 降低仓位，持有低波动+高Omega标的")
                print(f"    → 信号因子：real_vol_10(低), omega_20(高), tail_ratio_60")

        print()

    # ── Regime transition analysis ────────────────────────────────
    print(f"{'='*60}")
    print("REGIME TRANSITION ANALYSIS")
    print(f"{'='*60}")

    transitions = np.diff(regime_preds)
    n_transitions = (transitions != 0).sum()
    print(f"Total regime switches: {n_transitions} over {len(dates)} days")
    print(f"Average regime duration: {len(dates)/(n_transitions+1):.1f} days")

    # Transition matrix
    n_classes = len(np.unique(regime_preds))
    trans_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(1, len(regime_preds)):
        trans_matrix[regime_preds[i-1], regime_preds[i]] += 1

    print(f"\nTransition matrix (from → to):")
    header = "         " + "".join(f"→R{j:>5}" for j in range(n_classes))
    print(header)
    for i in range(n_classes):
        row = f"From R{i}:"
        for j in range(n_classes):
            row += f"{trans_matrix[i,j]:6d}"
        print(row)

    # Transition probabilities
    print(f"\nTransition probabilities:")
    for i in range(n_classes):
        row = f"From R{i}:"
        total = trans_matrix[i].sum()
        if total > 0:
            for j in range(n_classes):
                row += f" →R{j}: {trans_matrix[i,j]/total*100:5.1f}%"
        print(row)

    return regime_preds, probs, dates


def analyze_strategy_returns(
    dates: list,
    regime_preds: np.ndarray,
    factor_cols: list[str],
):
    """Compute forward returns for each strategy in each regime."""

    import sqlite3
    conn = sqlite3.connect("data/processed/etf.sqlite")
    prices = pd.read_sql("SELECT symbol, date, close FROM etf_daily WHERE close > 0", conn)
    prices["date"] = pd.to_datetime(prices["date"])
    conn.close()

    # Top 100 ETFs by data completeness
    top_etfs = df["symbol"].value_counts().head(100).index
    prices = prices[prices["symbol"].isin(top_etfs)]

    print(f"\n{'='*60}")
    print("FORWARD RETURN ANALYSIS BY REGIME & STRATEGY")
    print(f"{'='*60}\n")

    sampled = df[df["symbol"].isin(top_etfs)]

    for regime_id in sorted(np.unique(regime_preds)):
        mask = regime_preds == regime_id
        regime_dates = [dates[i] for i in range(len(dates)) if mask[i]]
        regime_date_set = set(pd.DatetimeIndex(regime_dates))

        # Factor characteristics per symbol during this regime
        regime_data = sampled[sampled["date"].isin(list(regime_date_set))]

        if len(regime_data) > 100:
            mom_mean = regime_data.groupby("symbol")["mom_20"].mean()
            vol_mean = regime_data.groupby("symbol")["real_vol_10"].mean()
            omega_mean = regime_data.groupby("symbol")["omega_20"].mean()

            print(f"Regime {regime_id}:")

            if regime_id == 0:
                ibias_mean = regime_data.groupby("symbol")["ibias_20"].mean()
                top5 = ibias_mean.nsmallest(5)
                print(f"  [Mean Reversion] Top buy signals (most oversold):")
                for sym, val in top5.items():
                    print(f"    {sym}: ibias_20={val:.4f}")

            elif regime_id == 1:
                top5 = mom_mean.nlargest(5)
                print(f"  [Momentum] Top buy signals (strongest momentum):")
                for sym, val in top5.items():
                    print(f"    {sym}: mom_20={val:.4f}")

            else:
                valid_vol = vol_mean[vol_mean > 0]
                score = -valid_vol + (omega_mean.loc[valid_vol.index] - omega_mean.mean()) / (omega_mean.std() + 1e-8)
                top5 = score.nlargest(5)
                print(f"  [Defensive] Top holdings (low vol + high omega):")
                for sym, val in top5.items():
                    print(f"    {sym}: score={val:.4f} (vol={vol_mean[sym]:.4f}, omega={omega_mean[sym]:.3f})")

            # Forward return analysis
            regime_dates_str = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d)[:10] for d in regime_dates]
            regime_prices = prices[prices["date"].dt.strftime("%Y-%m-%d").isin(regime_dates_str)]
            if len(regime_prices) > 100:
                returns_1d = []
                returns_5d = []
                for sym in top_etfs:
                    sym_p = regime_prices[regime_prices["symbol"] == sym].sort_values("date")
                    if len(sym_p) > 20:
                        close = sym_p["close"].values
                        rets = np.diff(close) / close[:-1]
                        returns_1d.extend(rets[~np.isnan(rets)])
                        for i in range(len(close) - 5):
                            fwd_ret = (close[i+5] - close[i]) / close[i]
                            if not np.isnan(fwd_ret):
                                returns_5d.append(fwd_ret)

                if returns_1d:
                    print(f"\n  Forward returns during this regime:")
                    print(f"    1-day:  mean={np.mean(returns_1d)*100:+.3f}% median={np.median(returns_1d)*100:+.3f}%")
                if returns_5d:
                    print(f"    5-day:  mean={np.mean(returns_5d)*100:+.3f}% median={np.median(returns_5d)*100:+.3f}%")

            print()


if __name__ == "__main__":
    import time
    t0 = time.time()

    # Load model
    model, ckpt = load_model("LSTM")
    mean = np.array(ckpt["mean"])
    std = np.array(ckpt["std"])
    print(f"Loaded LSTM model (test_acc={ckpt['test_acc']:.3f}, macro_f1={ckpt['macro_f1']:.3f})\n")

    # Build sequences
    X, dates = build_sequences(factor_cols, LOOKBACK)
    print(f"Sequences: {X.shape}, dates: {dates[0]} ~ {dates[-1]}\n")

    # Predict regimes
    preds, probs = predict_regimes(model, X, mean, std)

    # Analysis
    regime_preds, probs_arr, date_list = analyze_regime_performance(
        dates, preds, probs, factor_cols
    )
    analyze_strategy_returns(date_list, regime_preds, factor_cols)

    # Save predictions
    pred_df = pd.DataFrame({
        "date": date_list,
        "regime": regime_preds,
        "confidence": probs_arr.max(axis=1),
        **{f"prob_r{i}": probs_arr[:, i] for i in range(3)},
    })
    pred_df.to_csv("data/processed/regime_predictions.csv", index=False)

    print(f"\nPredictions saved to data/processed/regime_predictions.csv")
    print(f"Total time: {time.time()-t0:.1f}s")
