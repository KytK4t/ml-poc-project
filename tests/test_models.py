"""Tests for model loading and prediction."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import MODELS
from data import load_dataset_split, FEATURES


class TestModelFiles:
    """Verify that trained model files exist."""

    @pytest.mark.parametrize("model_key", list(MODELS.keys()))
    def test_model_file_exists(self, model_key):
        """Each registered model should have a file on disk."""
        model_path = MODELS[model_key]["path"]
        assert Path(model_path).exists(), f"Model file missing: {model_path}"

    @pytest.mark.parametrize("model_key", list(MODELS.keys()))
    def test_model_has_required_keys(self, model_key):
        """Each model config should have name, description, path."""
        cfg = MODELS[model_key]
        assert "name" in cfg
        assert "description" in cfg
        assert "path" in cfg


class TestModelPredictions:
    """Verify that models can make predictions."""

    @pytest.fixture(scope="class")
    def test_data(self):
        """Load test data once for the class."""
        _, X_test, _, y_test = load_dataset_split()
        return X_test, y_test

    @pytest.mark.parametrize("model_key", list(MODELS.keys()))
    def test_model_can_predict(self, model_key, test_data):
        """Each model should produce predictions without error."""
        import joblib
        X_test, _ = test_data
        model = joblib.load(MODELS[model_key]["path"])
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    @pytest.mark.parametrize("model_key", list(MODELS.keys()))
    def test_predictions_are_numeric(self, model_key, test_data):
        """Predictions should be numeric values."""
        import joblib
        X_test, _ = test_data
        model = joblib.load(MODELS[model_key]["path"])
        predictions = model.predict(X_test)
        assert np.issubdtype(predictions.dtype, np.number)

    @pytest.mark.parametrize("model_key", list(MODELS.keys()))
    def test_predictions_no_nan(self, model_key, test_data):
        """No prediction should be NaN."""
        import joblib
        X_test, _ = test_data
        model = joblib.load(MODELS[model_key]["path"])
        predictions = model.predict(X_test)
        assert not np.any(np.isnan(predictions))

    def test_single_sample_prediction(self):
        """Models should work with a single sample."""
        import joblib
        sample = pd.DataFrame(
            [[2010, 1, 3, 15000, 0.75, 500000, 2, 12.0]],
            columns=FEATURES,
        )
        model = joblib.load(MODELS["random_forest"]["path"])
        pred = model.predict(sample)
        assert len(pred) == 1
        assert np.isfinite(pred[0])
