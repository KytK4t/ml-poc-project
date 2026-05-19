"""Tests for data loading and preprocessing."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import load_dataset_split, _preprocess, FEATURES, TARGET


class TestPreprocess:
    """Tests for the _preprocess function."""

    def test_column_renaming(self):
        """Verify that columns are correctly renamed."""
        df = pd.DataFrame({
            "country": ["France", "France"],
            "year": [2000, 2000],
            "sex": ["male", "female"],
            "age": ["25-34 years", "25-34 years"],
            "suicides/100k pop": [10.0, 5.0],
            "HDI for year": [0.85, 0.85],
            " gdp_for_year ($) ": [1_000_000, 1_000_000],
            "gdp_per_capita ($)": [30_000, 30_000],
            "population": [10_000, 10_000],
            "generation": ["Generation X", "Generation X"],
        })
        result = _preprocess(df)
        assert "suicides_per_100k" in result.columns
        assert "HDI_for_year" in result.columns
        assert "gdp_per_capita" in result.columns

    def test_sex_encoding(self):
        """Male should be 1, female should be 0."""
        df = pd.DataFrame({
            "country": ["A", "A"],
            "year": [2000, 2000],
            "sex": ["male", "female"],
            "age": ["25-34 years", "25-34 years"],
            "suicides/100k pop": [10.0, 5.0],
            "HDI for year": [0.8, 0.8],
            " gdp_for_year ($) ": [100, 100],
            "gdp_per_capita ($)": [100, 100],
            "population": [1000, 1000],
            "generation": ["Generation X", "Generation X"],
        })
        result = _preprocess(df)
        assert result["sex"].tolist() == [1, 0]

    def test_age_ordinal_encoding(self):
        """Age groups should be mapped to 0-5."""
        df = pd.DataFrame({
            "country": ["A"] * 6,
            "year": [2000] * 6,
            "sex": ["male"] * 6,
            "age": ["5-14 years", "15-24 years", "25-34 years",
                    "35-54 years", "55-74 years", "75+ years"],
            "suicides/100k pop": [1.0] * 6,
            "HDI for year": [0.8] * 6,
            " gdp_for_year ($) ": [100] * 6,
            "gdp_per_capita ($)": [100] * 6,
            "population": [1000] * 6,
            "generation": ["Generation X"] * 6,
        })
        result = _preprocess(df)
        assert result["age"].tolist() == [0, 1, 2, 3, 4, 5]

    def test_drops_nan_hdi(self):
        """Rows with NaN HDI should be dropped."""
        df = pd.DataFrame({
            "country": ["A", "A"],
            "year": [2000, 2001],
            "sex": ["male", "male"],
            "age": ["25-34 years", "25-34 years"],
            "suicides/100k pop": [10.0, 5.0],
            "HDI for year": [0.8, float("nan")],
            " gdp_for_year ($) ": [100, 100],
            "gdp_per_capita ($)": [100, 100],
            "population": [1000, 1000],
            "generation": ["Generation X", "Generation X"],
        })
        result = _preprocess(df)
        assert len(result) == 1

    def test_country_encoded_is_numeric(self):
        """country_encoded should be a float (target encoding)."""
        df = pd.DataFrame({
            "country": ["A", "B", "A"],
            "year": [2000, 2000, 2001],
            "sex": ["male", "male", "female"],
            "age": ["25-34 years", "25-34 years", "25-34 years"],
            "suicides/100k pop": [10.0, 20.0, 10.0],
            "HDI for year": [0.8, 0.7, 0.8],
            " gdp_for_year ($) ": [100, 100, 100],
            "gdp_per_capita ($)": [100, 100, 100],
            "population": [1000, 1000, 1000],
            "generation": ["Generation X", "Generation X", "Generation X"],
        })
        result = _preprocess(df)
        assert result["country_encoded"].dtype in [np.float64, np.float32]


class TestLoadDatasetSplit:
    """Tests for the full dataset loading pipeline."""

    def test_returns_four_elements(self):
        """load_dataset_split should return 4 arrays."""
        result = load_dataset_split()
        assert len(result) == 4

    def test_shapes_consistent(self):
        """Train and test sets should have matching dimensions."""
        X_train, X_test, y_train, y_test = load_dataset_split()
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert X_train.shape[1] == len(FEATURES)

    def test_no_nan_in_features(self):
        """Features should not contain NaN values."""
        X_train, X_test, _, _ = load_dataset_split()
        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()

    def test_target_is_numeric(self):
        """Target should be numeric."""
        _, _, y_train, y_test = load_dataset_split()
        assert np.issubdtype(y_train.dtype, np.number)
        assert np.issubdtype(y_test.dtype, np.number)
