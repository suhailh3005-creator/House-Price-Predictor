# System Context & Architecture Overview

The House Price Predictor is an end-to-end, modular, supervised machine learning pipeline and web application designed for educational regression tasks.

```text
                  ┌──────────────────────┐
                  │      houses.csv      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │       data.py        │  <-- Ingestion & Cleaning
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │     features.py      │  <-- Feature Engineering
                  └──────────┬───────────┘
                             │
             ┌───────────────┴───────────────┐
             │                               │
             ▼                               ▼
  ┌─────────────────────┐         ┌─────────────────────┐
  │      train.py       │         │ house_price_eda.    │
  │  (Model Training &  │         │      ipynb          │
  │     Evaluation)     │         │ (Exploratory Analysis│
  └──────────┬──────────┘         └─────────────────────┘
             │
             ▼
  ┌─────────────────────┐
  │  house_price_model. │  <-- Serialized Model Artifact
  │       joblib        │
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │     predict.py      │  <-- Inference & Validation Layer
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │       app.py        │  <-- Streamlit UI Layer
  └─────────────────────┘
```

## 1. Module Responsibilities & Component API

| File / Component | Responsibility | Core Operations & Dependencies |
| --- | --- | --- |
| `houses.csv` | Primary Data Store | Synthetic tabular dataset containing raw features (`area_sqft`, `bedrooms`, `bathrooms`, `age_years`, `location_score`, `parking_spaces`) and target (`house_price`). |
| `data.py` | Ingestion & Validation | Reads CSV, handles type casting, checks numeric boundaries (e.g., `bedrooms > 0`, `house_price > 0`), drops missing/duplicate records via `pandas`. |
| `features.py` | Preprocessing & Feature Engineering | Creates synthetic features: `area_per_bedroom = area_sqft / bedrooms` and `house_age_group_code` via `pd.cut` binning. |
| `train.py` | Training Pipeline & Persistence | Performs 80/20 `train_test_split`, trains `sklearn.linear_model.LinearRegression`, evaluates (MAE, RMSE, $R^2$), serializes model to `joblib`. |
| `house_price_model.joblib` | Model Artifact | Exported binary representation of the fitted `LinearRegression` estimator. |
| `predict.py` | Inference Engine | Receives raw input dictionary/DataFrame, validates boundary limits, executes `features.py` transformations, invokes `joblib.load()`, outputs float prediction. |
| `app.py` | UI/UX Entry Point | Renders input forms, handles button events, passes payload to `predict.py`, displays formatted regression results using Streamlit. |
| `test_project.py` | Integration & Unit Tests | Runs assertions on data loading, transformation schemas, model execution non-nullity, and input validation bounds. |

## 2. Machine Learning Pipeline & Data Flow

### Data Transformation Matrix

```text
[Raw Features] ──> [data.py Cleaning] ──> [features.py Transformations] ──> [Final Vector]
- area_sqft          - Validated            - area_sqft                     - area_sqft
- bedrooms           - Non-null             - bedrooms                      - bedrooms
- bathrooms          - Numeric-cast         - bathrooms                     - bathrooms
- age_years                                 - age_years                     - age_years
- location_score                            - location_score                - location_score
- parking_spaces                            - parking_spaces                - parking_spaces
                                            - area_per_bedroom (Derived)   - area_per_bedroom
                                            - house_age_group_code (Derived)- house_age_group_code
```

### Training & Inference Workflow

1. Training Phase (`train.py`):

$$
\text{Dataset} \xrightarrow{\text{Clean}} X, y \xrightarrow{80/20 \text{ Split}} (X_{\text{train}}, y_{\text{train}}), (X_{\text{test}}, y_{\text{test}})
$$

$$
X_{\text{train}} \xrightarrow{\text{LinearRegression().fit()}} f(X) \xrightarrow{\text{Joblib Export}} \text{house\_price\_model.joblib}
$$

2. Evaluation Metrics:

- MAE (Mean Absolute Error):

$$
\frac{1}{n} \sum \lvert y_i - \hat{y}_i \rvert
$$

- RMSE (Root Mean Squared Error):

$$
\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}
$$

- $R^2$ Score:

$$
1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

3. Inference Phase (`predict.py` → `app.py`):

$$
\text{User Payload} \xrightarrow{\text{Validation Checks}} X_{\text{raw}} \xrightarrow{\text{engineer\_features()}} X_{\text{vector}} \xrightarrow{\text{model.predict()}} \hat{y}_{\text{price}}
$$

## 3. Key Design Constraints & Guarantees

- Problem Framing: Single-target tabular regression.
- Deterministic Inference: Prediction outputs are fully deterministic given the static model weights saved in `house_price_model.joblib`.
- Input Schema Constraints:
  - Numeric inputs must be strictly bounded (`bedrooms > 0`, `bathrooms > 0`, `age_years >= 0`).
  - Inputs missing required columns or containing invalid data types raise structured validation errors before model execution.
- Scope Boundary: Synthetic educational pipeline; unsuited for live commercial real estate valuations without recalibration on real-world market datasets.
