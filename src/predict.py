from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd

from src.features import engineer_features

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
FEATURE_COLUMNS = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "location_score",
    "parking_spaces",
    "area_per_bedroom",
    "house_age_group_code",
]


def load_model(model_path: str | Path = MODEL_PATH):
    """Load a saved trained model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Train the model first with: python src/train.py"
        )
    return joblib.load(model_path)


def _build_feature_row(values: dict[str, float | int]) -> pd.DataFrame:
    required_fields = [
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "age_years",
        "location_score",
        "parking_spaces",
    ]

    missing = [field for field in required_fields if field not in values]
    if missing:
        raise ValueError(f"Missing values for: {', '.join(missing)}")

    for field in required_fields:
        values[field] = float(values[field])

    if values["bedrooms"] <= 0:
        raise ValueError("Bedrooms must be greater than zero.")
    if values["bathrooms"] <= 0:
        raise ValueError("Bathrooms must be greater than zero.")
    if values["age_years"] < 0:
        raise ValueError("House age cannot be negative.")

    row = {field: values[field] for field in required_fields}
    feature_df = pd.DataFrame([row])
    engineered = engineer_features(feature_df)
    return engineered[FEATURE_COLUMNS]


def predict_house_price(values: dict[str, float | int], model_path: str | Path = MODEL_PATH) -> float:
    """Predict a price from house details."""
    model = load_model(model_path)
    feature_row = _build_feature_row(values)
    prediction = float(model.predict(feature_row)[0])
    return prediction
