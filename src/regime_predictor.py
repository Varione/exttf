"""Probability-based market regime prediction using trained classification models."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from train_classifier import RegimeClassifier, RegimeLSTM, RegimeTransformer


class RegimePredictor:
    """Outputs regime probabilities instead of hard labels.

    Loads a trained classification model (MLP/LSTM/Transformer) and the
    corresponding normalization parameters, then produces per-date
    probability vectors P(regime=0), P(regime=1), P(regime=2).

    Falls back to KMeans distance-based probabilities if no model is
    available.
    """

    def __init__(
        self,
        model_type: str = "LSTM",
        model_path: str | None = None,
        scaler_path: str = "data/processed/regime_model_cluster.joblib",
        classification_data_path: str = "data/processed/classification_data.npz",
    ):
        self.model_type = model_type.upper()
        self.scaler_path = scaler_path
        self.classification_data_path = classification_data_path

        if model_path is None:
            model_path = f"data/processed/regime_model_{self.model_type}.pt"
        self.model_path = model_path

        self.model = None
        self.n_classes = 3
        self.lookback = 20
        self.n_factors = 52
        self.mean = None
        self.std = None
        self.factor_cols = None
        self.kmeans = None
        self.scaler = None

        self._load_model()

    def _load_model(self):
        """Load the trained model and normalization parameters."""
        if os.path.exists(self.model_path):
            ckpt = torch.load(self.model_path, weights_only=False)
            self.n_classes = int(ckpt.get("n_classes", 3))
            self.lookback = int(ckpt.get("lookback", 20))
            self.n_factors = int(ckpt.get("n_factors", 52))
            self.mean = ckpt.get("mean")
            self.std = ckpt.get("std")
            self.factor_cols = ckpt.get("factor_cols")

            input_dim = self.n_factors
            if self.model_type == "MLP":
                self.model = RegimeClassifier(
                    input_dim=input_dim * self.lookback,
                    n_classes=self.n_classes,
                )
            elif self.model_type == "LSTM":
                self.model = RegimeLSTM(
                    input_dim=input_dim,
                    n_classes=self.n_classes,
                    hidden_dim=128,
                    num_layers=2,
                    dropout=0.3,
                    bidirectional=False,
                )
            elif self.model_type == "TRANSFORMER":
                self.model = RegimeTransformer(
                    input_dim=input_dim,
                    n_classes=self.n_classes,
                    d_model=64,
                    nhead=8,
                    num_layers=3,
                    dim_feedforward=256,
                    dropout=0.2,
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            print(f"Loaded {self.model_type} regime model from {self.model_path}")
        else:
            print(f"Model file not found: {self.model_path}, falling back to KMeans")
            self._load_kmeans_fallback()

    def _load_kmeans_fallback(self):
        """Load KMeans scaler and cluster model as fallback."""
        if os.path.exists(self.scaler_path):
            self.scaler, self.kmeans = joblib.load(self.scaler_path)
            print(f"Loaded KMeans fallback from {self.scaler_path}")
        else:
            raise FileNotFoundError(
                f"No model found at {self.model_path} and no KMeans fallback at {self.scaler_path}"
            )

    def _compute_daily_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Compute daily cross-sectional mean factors from the PIT-eligible universe."""
        sampled = factor_df.copy()
        if "pit_eligible" in sampled.columns:
            eligible = sampled["pit_eligible"]
            if eligible.dtype != bool:
                eligible = eligible.astype(str).str.lower().isin({"1", "true", "yes"})
            sampled = sampled.loc[eligible.fillna(False)]

        factor_cols = (
            list(self.factor_cols)
            if self.factor_cols is not None
            else self._auto_detect_factors(sampled)
        )
        available = [f for f in factor_cols if f in sampled.columns]
        if len(available) < self.n_factors:
            print(
                f"Warning: only {len(available)}/{self.n_factors} factors available, "
                f"using {available}"
            )

        daily = sampled.groupby("date")[available].mean().sort_index()
        daily = daily.rolling(20, min_periods=5).mean()
        return daily.dropna()

    def _auto_detect_factors(self, df: pd.DataFrame) -> list[str]:
        """Auto-detect factor columns from the data."""
        from detect_regimes import get_selected_factors

        return get_selected_factors(0.5)

    def predict_probabilities(
        self,
        factor_df: pd.DataFrame | None = None,
        factor_path: str = "data/processed/factors_all_repaired.csv",
    ) -> pd.DataFrame:
        """Return DataFrame with columns: date, prob_r0, prob_r1, prob_r2.

        Args:
            factor_df: Optional factor DataFrame. If None, loads from factor_path.
            factor_path: Path to factors CSV if factor_df is None.

        Returns:
            DataFrame with date index and probability columns.
        """
        if factor_df is None:
            factor_df = pd.read_csv(factor_path, parse_dates=["date"])

        if self.model is not None:
            return self._predict_with_model(factor_df)
        else:
            return self._predict_with_kmeans(factor_df)

    def _predict_with_model(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Predict using the trained neural network model."""
        daily_means = self._compute_daily_factors(factor_df)

        n_fac_actual = daily_means.shape[1]
        if n_fac_actual != self.n_factors:
            print(
                f"Factor count mismatch: expected {self.n_factors}, got {n_fac_actual}. "
                f"Padding/truncating."
            )
            pad_cols = ["pad_01", "pad_02"]
            while daily_means.shape[1] < self.n_factors:
                daily_means[pad_cols[daily_means.shape[1] - n_fac_actual]] = 0.0
            daily_means = daily_means.iloc[:, : self.n_factors]

        X_raw = daily_means.values.astype(np.float64)
        X_seq = self._build_sequences(X_raw, self.lookback)

        for col in range(self.n_factors):
            X_seq[:, :, col] = np.clip(
                X_seq[:, :, col],
                np.percentile(X_seq[:, :, col].reshape(-1), 1),
                np.percentile(X_seq[:, :, col].reshape(-1), 99),
            )

        mean_arr = np.asarray(self.mean, dtype=np.float64).reshape(1, 1, self.n_factors)
        std_arr = np.asarray(self.std, dtype=np.float64).reshape(1, 1, self.n_factors) + 1e-8
        X_norm = (X_seq - mean_arr) / std_arr

        X_tensor = torch.tensor(X_norm, dtype=torch.float32)

        if self.model_type == "MLP":
            X_tensor = X_tensor.reshape(X_tensor.size(0), -1)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).numpy()

        dates = daily_means.index[self.lookback - 1 :]
        result = pd.DataFrame(probs, index=dates, columns=[f"prob_r{i}" for i in range(self.n_classes)])
        result["regime"] = probs.argmax(axis=1)
        return result

    def _build_sequences(self, X: np.ndarray, lookback: int) -> np.ndarray:
        """Build sliding window sequences: output shape (n_samples, lookback, n_features).

        numpy's sliding_window_view places the window axis at the end,
        so we transpose to get (n_samples, lookback, n_features).
        """
        n_samples = X.shape[0] - lookback + 1
        view = np.lib.stride_tricks.sliding_window_view(X, lookback, axis=0)
        return view[:n_samples].transpose(0, 2, 1).copy()

    def _predict_with_kmeans(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Predict using KMeans distance-based probabilities as fallback.

        Probability approximated by inverse distance normalization:
            P(regime=k) = (1 / dist_k) / sum(1 / dist_j)
        """
        daily_means = self._compute_daily_factors(factor_df)

        # Use scaler's feature names if available, else auto-detect
        if hasattr(self.scaler, "feature_names_in_"):
            scaler_features = list(self.scaler.feature_names_in_)
        else:
            from detect_regimes import get_selected_factors
            scaler_features = get_selected_factors(0.5)

        available = [f for f in scaler_features if f in daily_means.columns]
        if len(available) < len(scaler_features):
            missing = set(scaler_features) - set(available)
            print(f"KMeans warning: {len(missing)} factors missing, padding with zeros: {missing}")
            for f in scaler_features:
                if f not in daily_means.columns:
                    daily_means[f] = 0.0

        X_df = daily_means[scaler_features]
        X_scaled = self.scaler.transform(X_df)
        distances = self.kmeans.transform(X_scaled)

        inv_dist = 1.0 / (distances + 1e-8)
        probs = inv_dist / inv_dist.sum(axis=1, keepdims=True)

        result = pd.DataFrame(
            probs, index=daily_means.index, columns=[f"prob_r{i}" for i in range(self.n_classes)]
        )
        result["regime"] = probs.argmax(axis=1)
        return result

    def get_regime_weighted_strategy_weights(
        self,
        probs: pd.Series | np.ndarray,
        strategy_regimes: dict[str, int],
        temperature: float = 1.0,
    ) -> dict[str, float]:
        """Compute w_strategy = sum_k P(regime=k) * w_strategy|k.

        For each regime k, strategies assigned to that regime get equal weight.
        The final weight is a probability-weighted average across regimes.

        Args:
            probs: Probability vector [P(r0), P(r1), P(r2)].
            strategy_regimes: Mapping from strategy name to preferred regime (-1 = agnostic).
            temperature: Softmax temperature for smoothing (1.0 = no extra smoothing).

        Returns:
            Dictionary mapping strategy name to weight.
        """
        if isinstance(probs, pd.Series):
            prob_vals = probs.values.astype(float)
        else:
            prob_vals = np.asarray(probs, dtype=float)

        n_regimes = len(prob_vals)
        weights: dict[str, float] = {s: 0.0 for s in strategy_regimes}

        regime_strategies: dict[int, list[str]] = {}
        agnostic_strategies: list[str] = []

        for strat_name, regime_id in strategy_regimes.items():
            if regime_id < 0:
                agnostic_strategies.append(strat_name)
            else:
                regime_strategies.setdefault(regime_id, []).append(strat_name)

        for regime_k in range(n_regimes):
            p_k = prob_vals[regime_k]
            if p_k <= 0 or regime_k not in regime_strategies:
                continue

            strats_in_regime = regime_strategies[regime_k]
            w_per_strat = p_k / len(strats_in_regime) if strats_in_regime else 0.0
            for s in strats_in_regime:
                weights[s] += w_per_strat

        if agnostic_strategies:
            agnostic_share = float(np.sum(prob_vals))
            w_per_agnostic = agnostic_share / len(agnostic_strategies) if agnostic_strategies else 0.0
            for s in agnostic_strategies:
                weights[s] += w_per_agnostic

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_smoothed_regime_weights(
        self,
        probs: pd.Series | np.ndarray,
        strategy_regimes: dict[str, int],
        strategy_returns_in_regime: dict[int, dict[str, float]],
        temperature: float = 3.0,
    ) -> dict[str, float]:
        """Compute regime-weighted strategy weights using historical performance.

        For each regime k, weight strategies by their historical performance
        in that regime, then average across regimes weighted by P(regime=k).

        Args:
            probs: Probability vector [P(r0), P(r1), P(r2)].
            strategy_regimes: Mapping from strategy name to preferred regime.
            strategy_returns_in_regime: Dict mapping regime_id -> {strategy_name: avg_return}.
            temperature: Temperature for softmax-like weighting (higher = more uniform).

        Returns:
            Dictionary mapping strategy name to weight.
        """
        if isinstance(probs, pd.Series):
            prob_vals = probs.values.astype(float)
        else:
            prob_vals = np.asarray(probs, dtype=float)

        all_strategies = set(strategy_regimes.keys())
        weights: dict[str, float] = {s: 0.0 for s in all_strategies}

        for regime_k, p_k in enumerate(prob_vals):
            if p_k <= 1e-8:
                continue

            ret_in_regime = strategy_returns_in_regime.get(regime_k, {})
            if not ret_in_regime:
                continue

            perf = np.array([ret_in_regime.get(s, 0.0) for s in all_strategies])
            perf_shifted = perf - perf.min() + 1e-8
            perf_weighted = np.exp(perf_shifted / max(temperature, 1e-8))
            perf_normalized = perf_weighted / perf_weighted.sum()

            for i, s in enumerate(all_strategies):
                weights[s] += p_k * perf_normalized[i]

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


def save_regime_probabilities(
    predictor: RegimePredictor,
    output_path: str = "data/processed/regime_predictions.csv",
    factor_path: str = "data/processed/factors_all_repaired.csv",
):
    """Compute and save regime probabilities to CSV."""
    probs_df = predictor.predict_probabilities(factor_path=factor_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    probs_df.to_csv(output_path)
    print(f"Saved regime probabilities ({len(probs_df)} dates) to {output_path}")
    return probs_df


if __name__ == "__main__":
    import time

    t0 = time.time()

    for model_type in ["LSTM", "MLP", "Transformer"]:
        print(f"\n{'='*60}")
        print(f"Model: {model_type}")
        print(f"{'='*60}")

        predictor = RegimePredictor(model_type=model_type)
        probs_df = save_regime_probabilities(
            predictor,
            output_path=f"data/processed/regime_predictions_{model_type}.csv",
        )

        print(f"\nProbability distribution (last 10 dates):")
        print(probs_df[["prob_r0", "prob_r1", "prob_r2", "regime"]].tail(10))

        regime_counts = probs_df["regime"].value_counts().sort_index()
        print(f"\nRegime distribution:")
        for r, cnt in regime_counts.items():
            print(f"  Regime {r}: {cnt} days ({cnt/len(probs_df)*100:.1f}%)")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
