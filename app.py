
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Marketing Campaign Prediction",
    page_icon="📊",
    layout="wide"
)

# Load saved models and feature names
folder = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    revenue_model = joblib.load(folder / "revenue_model.pkl")
    profit_model = joblib.load(folder / "profit_model.pkl")
    features = joblib.load(folder / "model_features.pkl")
    return revenue_model, profit_model, features

revenue_model, profit_model, features = load_models()

st.title("📊 Marketing Campaign Performance Prediction")
st.write("Predict campaign revenue and profitability using Machine Learning.")

st.sidebar.header("Campaign Inputs")

impressions = st.sidebar.number_input(
    "Impressions", min_value=0, value=100000, step=1000
)
clicks = st.sidebar.number_input(
    "Clicks", min_value=0, value=10000, step=100
)
leads = st.sidebar.number_input(
    "Leads", min_value=0, value=3000, step=100
)
conversions = st.sidebar.number_input(
    "Conversions", min_value=0, value=500, step=10
)
acquisition_cost = st.sidebar.number_input(
    "Acquisition Cost", min_value=0.01, value=300.0
)
engagement_score = st.sidebar.number_input(
    "Engagement Score", min_value=0.0, value=5.0
)
duration = st.sidebar.number_input(
    "Duration", min_value=0.0, value=30.0
)

# Build input DataFrame in training feature order
input_data = pd.DataFrame([{
    "Impressions": impressions,
    "Clicks": clicks,
    "Leads": leads,
    "Conversions": conversions,
    "Acquisition_Cost": acquisition_cost,
    "Engagement_Score": engagement_score,
    "Duration": duration
}])[features]

st.subheader("Campaign Input Summary")
st.dataframe(input_data, use_container_width=True)

if st.button("Predict Campaign Performance", type="primary"):
    revenue_prediction = revenue_model.predict(input_data)[0]
    profit_prediction = profit_model.predict(input_data)[0]
    profit_probability = profit_model.predict_proba(input_data)[0, 1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Revenue",
            f"{revenue_prediction:,.2f}"
        )

    with col2:
        status = "Profitable" if profit_prediction == 1 else "Non-profitable"
        st.metric("Predicted Status", status)

    st.write(
        f"Estimated probability of profitability: "
        f"{profit_probability:.1%}"
    )

    st.info(
        "Predictions are estimates based on historical campaign data. "
        "Profitability labels are based on the dataset's ROI values, "
        "which contain some identified inconsistencies."
    )
