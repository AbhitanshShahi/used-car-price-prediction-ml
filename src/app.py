import streamlit as st
import pandas as pd
import joblib
import altair as alt

model = joblib.load("models/linear_model.joblib")
scaler = joblib.load("models/scaler.joblib")
columns = joblib.load("models/columns.joblib")

st.set_page_config(
    page_title="🚗 Used Car Price Predictor",
    layout="wide"
)

st.title("🚗 Used Car Price Predictor")
st.write(
    """
Predict the market value of a used car using a Linear Regression model.
"""
)

brand = st.text_input("Brand")
model_name = st.text_input("Model")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (Years)",
        min_value=0,
        max_value=40,
        value=5
    )

with col2:
    km_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=50000
    )

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid/CNG"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Owner",
    ["first", "second"]
)

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "Age": [age],
        "kmDriven": [km_driven]
    })

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    brand_col = f"Brand_{brand}"
    model_col = f"model_{model_name}"
    fuel_col = f"FuelType_{fuel_type}"
    transmission_col = f"Transmission_{transmission}"
    owner_col = f"Owner_{owner}"

    if brand_col in input_df.columns:
        input_df[brand_col] = 1

    if model_col in input_df.columns:
        input_df[model_col] = 1

    if fuel_col in input_df.columns:
        input_df[fuel_col] = 1

    if transmission_col in input_df.columns:
        input_df[transmission_col] = 1

    if owner_col in input_df.columns:
        input_df[owner_col] = 1

    input_df = input_df[columns]

    input_df[["Age", "kmDriven"]] = scaler.transform(
        input_df[["Age", "kmDriven"]]
    )

    prediction = model.predict(input_df)[0]

    st.success(
        f"Estimated Price: ₹{prediction:,.0f}"
    )

    st.markdown("---")

    st.subheader("Feature Importance")

    feature_importance = pd.DataFrame({
        "Feature": columns,
        "Importance": model.coef_
    })

    feature_importance["Absolute Importance"] = (
        feature_importance["Importance"].abs()
    )

    feature_importance = (
        feature_importance
        .sort_values(
            by="Absolute Importance",
            ascending=False
        )
        .head(15)
    )

    chart = (
        alt.Chart(feature_importance)
        .mark_bar()
        .encode(
            x=alt.X(
                "Importance:Q",
                title="Coefficient"
            ),
            y=alt.Y(
                "Feature:N",
                sort="-x"
            ),
            color=alt.condition(
                alt.datum.Importance > 0,
                alt.value("#00C853"),
                alt.value("#FF5252")
            )
        )
        .properties(
            title="Top 15 Most Important Features",
            width=800,
            height=500
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True
    )

    st.write(
        """
🟢 Green bars increase the predicted price.

🔴 Red bars decrease the predicted price.
"""
    )