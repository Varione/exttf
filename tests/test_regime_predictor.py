"""Tests for regime predictor module."""

import numpy as np
import pandas as pd
import pytest


class TestRegimePredictorInit:
    """Test RegimePredictor initialization."""

    def test_lstm_model_loads(self):
        from src.regime_predictor import RegimePredictor

        p = RegimePredictor(model_type="LSTM")
        assert p.model is not None
        assert p.n_classes == 3
        assert p.lookback == 20
        assert p.n_factors == 52

    def test_mlp_model_loads(self):
        from src.regime_predictor import RegimePredictor

        p = RegimePredictor(model_type="MLP")
        assert p.model is not None

    def test_transformer_model_loads(self):
        from src.regime_predictor import RegimePredictor

        p = RegimePredictor(model_type="Transformer")
        assert p.model is not None

    def test_kmeans_fallback(self):
        from src.regime_predictor import RegimePredictor

        p = RegimePredictor(
            model_type="INVALID",
            model_path="nonexistent_model.pt",
        )
        assert p.model is None
        assert p.kmeans is not None


class TestRegimePredictorProbabilities:
    """Test probability prediction."""

    @pytest.fixture
    def predictor(self):
        from src.regime_predictor import RegimePredictor
        return RegimePredictor(model_type="LSTM")

    def test_probabilities_sum_to_one(self, predictor):
        probs = predictor.predict_probabilities()
        prob_sum = probs[["prob_r0", "prob_r1", "prob_r2"]].sum(axis=1)
        np.testing.assert_allclose(prob_sum, 1.0, atol=1e-6)

    def test_probabilities_non_negative(self, predictor):
        probs = predictor.predict_probabilities()
        assert (probs["prob_r0"] >= 0).all()
        assert (probs["prob_r1"] >= 0).all()
        assert (probs["prob_r2"] >= 0).all()

    def test_output_shape(self, predictor):
        probs = predictor.predict_probabilities()
        assert "prob_r0" in probs.columns
        assert "prob_r1" in probs.columns
        assert "prob_r2" in probs.columns
        assert "regime" in probs.columns
        assert len(probs) > 4000

    def test_regime_is_argmax(self, predictor):
        probs = predictor.predict_probabilities()
        expected = probs[["prob_r0", "prob_r1", "prob_r2"]].values.argmax(axis=1)
        np.testing.assert_array_equal(probs["regime"].values, expected)


class TestRegimeWeightedWeights:
    """Test regime-weighted strategy weight computation."""

    @pytest.fixture
    def predictor(self):
        from src.regime_predictor import RegimePredictor
        return RegimePredictor(model_type="LSTM")

    def test_weights_sum_to_one(self, predictor):
        probs = np.array([0.5, 0.3, 0.2])
        strategy_regimes = {
            "S01": 0, "S04": 0,
            "S11": 1, "S12": 1,
            "S21": 2, "S22": 2,
        }
        weights = predictor.get_regime_weighted_strategy_weights(probs, strategy_regimes)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_weights_non_negative(self, predictor):
        probs = np.array([0.5, 0.3, 0.2])
        strategy_regimes = {"S01": 0, "S11": 1, "S21": 2}
        weights = predictor.get_regime_weighted_strategy_weights(probs, strategy_regimes)
        assert all(v >= 0 for v in weights.values())

    def test_agnostic_strategies_included(self, predictor):
        probs = np.array([0.5, 0.3, 0.2])
        strategy_regimes = {"S01": 0, "B0": -1}
        weights = predictor.get_regime_weighted_strategy_weights(probs, strategy_regimes)
        assert "S01" in weights
        assert "B0" in weights

    def test_deterministic(self, predictor):
        probs = np.array([0.5, 0.3, 0.2])
        strategy_regimes = {"S01": 0, "S04": 0, "S11": 1}
        w1 = predictor.get_regime_weighted_strategy_weights(probs, strategy_regimes)
        w2 = predictor.get_regime_weighted_strategy_weights(probs, strategy_regimes)
        assert w1 == w2


class TestBuildSequences:
    """Test sequence building utility."""

    def test_sequence_shape(self):
        from src.regime_predictor import RegimePredictor

        p = RegimePredictor(model_type="LSTM")
        X = np.random.randn(100, 52)
        seq = p._build_sequences(X, 20)
        assert seq.shape == (81, 20, 52)

    def test_sequence_values(self):
        from src.regime_predictor import RegimePredictor

        p = RegimePredictor(model_type="LSTM")
        X = np.arange(100 * 5).reshape(100, 5).astype(float)
        seq = p._build_sequences(X, 3)
        # First sequence should contain rows 0, 1, 2
        np.testing.assert_array_equal(seq[0, 0], X[0])
        np.testing.assert_array_equal(seq[0, 1], X[1])
        np.testing.assert_array_equal(seq[0, 2], X[2])
