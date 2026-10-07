from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.data import load_dataset
from src.features import engineer_features

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "houses.csv"
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


def _prepare_split(data_path: str | Path = DATA_PATH):
    df = load_dataset(data_path)
    prepared = engineer_features(df)

    X = prepared[FEATURE_COLUMNS]
    y = prepared["house_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def _evaluate_model(model, X_train, X_test, y_train, y_test) -> dict[str, Any]:
    start = perf_counter()
    model.fit(X_train, y_train)
    training_time_ms = (perf_counter() - start) * 1000

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "training_time_ms": training_time_ms,
    }


def train_model(data_path: str | Path = DATA_PATH, model_path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Train the baseline linear regression model and save it to disk."""
    X_train, X_test, y_train, y_test = _prepare_split(data_path)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "model_path": str(model_path),
    }


def compare_models(data_path: str | Path = DATA_PATH) -> list[dict[str, Any]]:
    """Compare a baseline linear model, a decision tree, and a random forest."""
    X_train, X_test, y_train, y_test = _prepare_split(data_path)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
    }

    results = []
    for name, model in models.items():
        model_result = _evaluate_model(model, X_train, X_test, y_train, y_test)
        model_result["name"] = name
        results.append(model_result)

    header = (
        f"{'Model':<18} {'MAE ($)':>12} {'RMSE ($)':>13} {'R² Score':>10} {'Training Time (ms)':>18}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for result in results:
        print(
            f"{result['name']:<18} "
            f"${result['mae']:>10,.0f} "
            f"${result['rmse']:>10,.0f} "
            f"{result['r2']:>9.3f} "
            f"{result['training_time_ms']:>14.2f}"
        )
    print(separator)

    return results


if __name__ == "__main__":
    result = train_model()
    print(f"Training complete. Model saved to {result['model_path']}")
    print(f"MAE: ${result['mae']:,.0f}")
    print(f"RMSE: ${result['rmse']:,.0f}")
    print(f"R2: {result['r2']:.3f}")
    print("\nModel comparison on the shared train/test split:")
    compare_models()
