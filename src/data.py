"""Dataset loading and preprocessing for suicide rate prediction."""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import DATA_DIR


FEATURES = [
    "year",
    "sex",
    "age",
    "gdp_per_capita",
    "HDI_for_year",
    "population",
    "generation",
    "country_encoded",
]

TARGET = "suicides_per_100k"


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to the raw dataframe."""

    df = df.copy()

    # Rename columns
    df.rename(
        columns={
            "suicides/100k pop": "suicides_per_100k",
            "HDI for year": "HDI_for_year",
            " gdp_for_year ($) ": "gdp_for_year",
            "gdp_per_capita ($)": "gdp_per_capita",
        },
        inplace=True,
    )

    # Drop rows where HDI_for_year is NaN
    df.dropna(subset=["HDI_for_year"], inplace=True)

    # Encode sex: male=1, female=0
    df["sex"] = df["sex"].map({"male": 1, "female": 0})

    # Encode age as ordinal
    age_mapping = {
        "5-14 years": 0,
        "15-24 years": 1,
        "25-34 years": 2,
        "35-54 years": 3,
        "55-74 years": 4,
        "75+ years": 5,
    }
    df["age"] = df["age"].map(age_mapping)

    # Encode generation with label encoding
    generation_labels = sorted(df["generation"].unique())
    gen_mapping = {g: i for i, g in enumerate(generation_labels)}
    df["generation"] = df["generation"].map(gen_mapping)

    # Target encoding for country (mean suicides_per_100k per country)
    country_means = df.groupby("country")["suicides_per_100k"].mean()
    df["country_encoded"] = df["country"].map(country_means)

    return df


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return (X_train, X_test, y_train, y_test) for model evaluation."""

    df = pd.read_csv(DATA_DIR / "master.csv")
    df = _preprocess(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
