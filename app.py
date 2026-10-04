from __future__ import annotations

import streamlit as st

from src.predict import predict_house_price
from src.train import MODEL_PATH, train_model


def add_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Source+Sans+3:wght@400;600;700&display=swap');

        :root {
            --background: #F4EFE9;
            --panel: #F8F4F0;
            --surface: #F5F1ED;
            --input: #2D3A48;
            --text: #2F2B2A;
            --muted: #7D736B;
            --accent: #E28A67;
            --border: #E7E0D8;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--background);
            color: var(--text);
            font-family: 'Source Sans 3', sans-serif;
        }

        [data-testid="stMain"] {
            background: var(--background);
        }

        .block-container {
            max-width: 840px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1 {
            font-family: 'Fraunces', serif !important;
            font-weight: 600 !important;
            color: var(--text) !important;
            letter-spacing: -0.04em !important;
            margin-bottom: 0.35rem !important;
        }

        .subtitle {
            font-size: 1.05rem;
            color: var(--muted);
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }

        .panel {
            background: rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 0.35rem 0;
            margin-top: 1rem;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text);
            margin: 0 0 1rem 0;
        }

        .result-box {
            background: #FFF3EE;
            border: 1px solid rgba(226, 138, 103, 0.32);
            border-radius: 18px;
            padding: 1.1rem 1rem;
            text-align: center;
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--text);
            margin-top: 0.9rem;
            margin-bottom: 0.55rem;
        }

        .helper-note {
            font-size: 0.96rem;
            color: var(--muted);
            margin: 0.15rem 0 1.2rem 0;
        }

        div[data-testid="stNumberInput"] {
            margin-bottom: 0.6rem;
        }

        div[data-testid="stNumberInput"] > div {
            background: var(--input);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 14px;
            padding: 0.3rem 0.7rem;
            color: white;
        }

        div[data-testid="stNumberInput"] label {
            color: var(--text) !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
        }

        div[data-testid="stNumberInput"] input {
            background: transparent !important;
            color: white !important;
            font-size: 1.05rem !important;
            border: none !important;
            box-shadow: none !important;
        }

        div[data-testid="stNumberInput"] button {
            background: rgba(255,255,255,0.08) !important;
            border: none !important;
            color: white !important;
        }

        div[data-testid="stFormSubmitButton"] > button {
            background: var(--accent) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            padding: 0.8rem 1.2rem !important;
            width: auto !important;
            min-width: 180px;
            margin-top: 0.35rem;
            box-shadow: none !important;
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #d97653 !important;
        }

        .stAlert {
            margin-top: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if not MODEL_PATH.exists():
    with st.spinner("Training the model for the first time..."):
        train_model()

add_styles()

st.title("Let's make a sensible guess.")
st.markdown(
    "<div class='subtitle'>Tell us a little about the house, and our small machine-learning model will estimate what it might be worth based on the homes it learned from.</div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='panel'>", unsafe_allow_html=True)

with st.form("house_form"):
    st.markdown("<div class='section-title'>Tell us about the house</div>", unsafe_allow_html=True)

    area_sqft = st.number_input("Area (sq ft)", min_value=300.0, value=1800.0, step=50.0)
    bedrooms = st.number_input("Bedrooms", min_value=1, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1.0, value=2.0, step=0.5)
    age_years = st.number_input("House age", min_value=0.0, value=12.0, step=1.0)
    location_score = st.number_input("Location score", min_value=0.0, max_value=100.0, value=72.0, step=1.0)
    parking_spaces = st.number_input("Parking spaces", min_value=0, value=2, step=1)

    st.markdown(
        "<div class='helper-note'>A rough estimate based on the examples the model learned from.</div>",
        unsafe_allow_html=True,
    )

    submitted = st.form_submit_button("Estimate the price")

    if submitted:
        try:
            prediction = predict_house_price(
                {
                    "area_sqft": area_sqft,
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "age_years": age_years,
                    "location_score": location_score,
                    "parking_spaces": parking_spaces,
                }
            )
            st.markdown(
                f"<div class='result-box'>${prediction:,.0f}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='helper-note'>This is a learning project, not a professional property valuation.</div>",
                unsafe_allow_html=True,
            )
        except ValueError as exc:
            st.error(f"Please check your values: {exc}")

st.markdown("</div>", unsafe_allow_html=True)
