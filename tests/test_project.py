from __future__ import annotations

from pathlib import Path

from src.data import load_dataset
from src.features import engineer_features
from src.predict import predict_house_price
from src.train import train_model


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "houses.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "test_model.joblib"


def test_dataset_loads_successfully():
    df = load_dataset(DATA_PATH)
    assert not df.empty
    for column in [
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "age_years",
        "location_score",
        "parking_spaces",
        "house_price",
    ]:
        assert column in df.columns


def test_feature_engineering_creates_expected_columns():
    df = load_dataset(DATA_PATH)
    engineered = engineer_features(df.head())
    assert "area_per_bedroom" in engineered.columns
    assert "house_age_group" in engineered.columns
    assert "house_age_group_code" in engineered.columns


def test_training_script_creates_model_file():
    summary = train_model(DATA_PATH, MODEL_PATH)
    assert summary["model"] is not None
    assert MODEL_PATH.exists()
    assert summary["mae"] >= 0
    assert summary["r2"] <= 1


def test_prediction_returns_numeric_price():
    house = {
        "area_sqft": 2100,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 12,
        "location_score": 78,
        "parking_spaces": 2,
    }
    prediction = predict_house_price(house, MODEL_PATH)
    assert isinstance(prediction, float)
    assert prediction > 0


def test_invalid_user_input_is_handled_gracefully():
    try:
        predict_house_price(
            {
                "area_sqft": 1000,
                "bedrooms": 0,
                "bathrooms": 2,
                "age_years": 12,
                "location_score": 70,
                "parking_spaces": 1,
            },
            MODEL_PATH,
        )
    except ValueError:
        return
    raise AssertionError("Invalid bedroom input should raise a ValueError.")
