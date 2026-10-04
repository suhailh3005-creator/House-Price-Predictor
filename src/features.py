from __future__ import annotations

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a few simple, meaningful features for the learning project."""
    engineered = df.copy()

    engineered["area_per_bedroom"] = (
        engineered["area_sqft"] / engineered["bedrooms"].replace(0, 1)
    )

    age_groups = pd.cut(
        engineered["age_years"],
        bins=[-1, 5, 20, float("inf")],
        labels=["new", "mid_age", "older"],
        right=False,
    )
    engineered["house_age_group"] = age_groups.astype(str)
    engineered["house_age_group_code"] = engineered["house_age_group"].map(
        {"new": 0, "mid_age": 1, "older": 2}
    )

    return engineered
