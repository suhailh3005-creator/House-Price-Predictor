from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

REQUIRED_COLUMNS = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "location_score",
    "parking_spaces",
    "house_price",
]


def load_dataset(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load the housing dataset and perform a small amount of validation."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset is empty. Please check the CSV file.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing_columns)
        )

    df = df.copy()
    df = df.drop_duplicates()

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)

    if (df["bedrooms"] <= 0).any():
        raise ValueError("Bedrooms must be greater than zero.")
    if (df["bathrooms"] <= 0).any():
        raise ValueError("Bathrooms must be greater than zero.")
    if (df["house_price"] <= 0).any():
        raise ValueError("House prices must be greater than zero.")

    return df
