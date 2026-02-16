"""
Streamlit Web Application for Rental Price Prediction
"""

import streamlit as st
import pandas as pd
import pickle
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Rental Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        st.error("Model file not found!")
        return None

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    return model


# ---------------- LOAD METRICS ---------------- #
@st.cache_data
def load_metrics():
    metrics = {}

    if os.path.exists("model_metrics.txt"):
        try:
            with open("model_metrics.txt", "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for line in lines:
                if "MAE" in line:
                    metrics["MAE"] = float(line.split(":")[1].strip().replace(",", ""))
                elif "RMSE" in line:
                    metrics["RMSE"] = float(line.split(":")[1].strip().replace(",", ""))
                elif "R2" in line or "R²" in line:
                    metrics["R2"] = float(line.split(":")[1].strip())

        except Exception as e:
            st.warning(f"Metrics load error: {e}")

    return metrics


# ---------------- LOAD DATASET ---------------- #
@st.cache_data
def load_dataset():
    try:
        try:
            df = pd.read_csv("estate_rent_dataset.csv", encoding="utf-8")
        except:
            df = pd.read_csv("estate_rent_dataset.csv", encoding="latin1")

        return df

    except Exception as e:
        st.error(f"Dataset loading error: {e}")
        return None


# ---------------- MAIN APP ---------------- #
def main():

    st.title("🏠 Rental Price Predictor")
    st.markdown("---")

    model = load_model()
    if model is None:
        st.stop()

    metrics = load_metrics()
    df = load_dataset()

    if df is None:
        st.stop()

    # ---------------- SIDEBAR ---------------- #
    st.sidebar.header("📋 Property Details")

    # Dynamic City selection
    city = st.sidebar.selectbox(
        "City",
        sorted(df["City"].unique())
    )

    # Filter locations based on city
    filtered_locations = df[df["City"] == city]["Location"].unique()

    location = st.sidebar.selectbox(
        "Location",
        sorted(filtered_locations)
    )

    bhk = st.sidebar.number_input("BHK", 1, 10, 2)
    size_sqft = st.sidebar.number_input("Size (sqft)", 100, 10000, 1200, step=50)
    bathrooms = st.sidebar.number_input("Bathrooms", 1, 10, 2)
    floor = st.sidebar.number_input("Floor", 0, 50, 3)
    total_floors = st.sidebar.number_input("Total Floors", 1, 50, 10)

    furnishing = st.sidebar.selectbox(
        "Furnishing Status",
        sorted(df["Furnishing"].unique())
    )

    property_age = st.sidebar.number_input("Property Age (years)", 0, 50, 5)
    parking = st.sidebar.number_input("Parking Spaces", 0, 5, 1)

    predict_button = st.sidebar.button("🔮 Predict Rental Price")

    # ---------------- MAIN LAYOUT ---------------- #
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Prediction Results")

        if predict_button:
            input_data = pd.DataFrame({
                "City": [city],
                "Location": [location],
                "BHK": [bhk],
                "Size_sqft": [size_sqft],
                "Bathrooms": [bathrooms],
                "Floor": [floor],
                "Total_Floors": [total_floors],
                "Furnishing": [furnishing],
                "Property_Age": [property_age],
                "Parking": [parking]
            })

            try:
                prediction = model.predict(input_data)[0]

                st.success(f"💰 Predicted Rent: ₹{prediction:,.0f} per month")

                # -------- Input Summary -------- #
                st.markdown("### 📝 Input Summary")

                summary_df = pd.DataFrame({
                    "Feature": [
                        "City", "Location", "BHK", "Size (sqft)", "Bathrooms",
                        "Floor", "Total Floors", "Furnishing",
                        "Property Age", "Parking"
                    ],
                    "Value": [
                        city, location, bhk, size_sqft, bathrooms,
                        floor, total_floors, furnishing,
                        property_age, parking
                    ]
                })

                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

        else:
            st.info("👈 Fill details and click Predict")

    with col2:
        st.subheader("📈 Model Performance")

        if metrics:
            st.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
            st.metric("MAE", f"₹{metrics.get('MAE', 0):,.0f}")
            st.metric("RMSE", f"₹{metrics.get('RMSE', 0):,.0f}")
        else:
            st.warning("Metrics not available")

    st.markdown("---")
    st.caption("Built using Streamlit & Machine Learning")


if __name__ == "__main__":
    main()
