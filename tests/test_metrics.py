"""Tests for metric computation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics import compute_metrics


class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_returns_all_keys(self):
        """Should return mae, rmse, r2, mape."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 2.2, 2.9, 4.1, 5.0]
        result = compute_metrics(y_true, y_pred)
        assert set(result.keys()) == {"mae", "rmse", "r2", "mape"}

    def test_values_are_floats(self):
        """All values should be Python floats."""
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        result = compute_metrics(y_true, y_pred)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not float: {type(val)}"

    def test_perfect_prediction(self):
        """Perfect predictions should give MAE=0, RMSE=0, R2=1, MAPE=0."""
        y_true = [10, 20, 30]
        y_pred = [10, 20, 30]
        result = compute_metrics(y_true, y_pred)
        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0
        assert result["r2"] == 1.0
        assert result["mape"] == 0.0

    def test_mae_positive(self):
        """MAE should always be non-negative."""
        y_true = [1, 2, 3]
        y_pred = [3, 2, 1]
        result = compute_metrics(y_true, y_pred)
        assert result["mae"] >= 0

    def test_rmse_greater_or_equal_mae(self):
        """RMSE should always be >= MAE."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.5, 2.5, 2.5, 4.5, 4.5]
        result = compute_metrics(y_true, y_pred)
        assert result["rmse"] >= result["mae"]

    def test_r2_range(self):
        """R2 should be <= 1."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]
        result = compute_metrics(y_true, y_pred)
        assert result["r2"] <= 1.0

    def test_mape_with_zeros_in_true(self):
        """MAPE should handle zeros in y_true gracefully."""
        y_true = [0, 0, 10]
        y_pred = [1, 1, 11]
        result = compute_metrics(y_true, y_pred)
        # Should use only the non-zero values
        assert np.isfinite(result["mape"])

    def test_mape_all_zeros(self):
        """MAPE should be inf when all y_true are zero."""
        y_true = [0, 0, 0]
        y_pred = [1, 2, 3]
        result = compute_metrics(y_true, y_pred)
        assert result["mape"] == float("inf")
