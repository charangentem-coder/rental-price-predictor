"""
Rental Price Prediction - College Presentation Version
Clean & Professional UI
"""

import streamlit as st
import pandas as pd
import pickle
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Rental Price Predictor",
    page_icon="🏠",
    layout="centered"
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

    st.title("🏠 Rental Price Prediction System")
    st.markdown("### AI-Based Monthly Rent Estimator")
    st.markdown("---")

    model = load_model()
    df = load_dataset()

    if model is None or df is None:
        st.stop()

    st.sidebar.header("📋 Enter Property Details")

    # Dynamic City Dropdown
    city = st.sidebar.selectbox(
        "City",
        sorted(df["City"].unique())
    )

    # Filter locations based on selected city
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

    st.markdown("## 📊 Prediction Result")

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

            # BIG Professional Prediction Box
            st.markdown("""
                <div style='text-align:center; padding:40px;
                            background-color:#f8f9fa;
                            border-radius:20px;
                            box-shadow:0 6px 18px rgba(0,0,0,0.15);'>
                    <h3 style='color:#555;'>Estimated Monthly Rent</h3>
                    <h1 style='font-size:70px; color:#2E8B57; margin:15px 0;'>
                        ₹{:,.0f}
                    </h1>
                    <p style='color:gray;'>Predicted using Machine Learning Model</p>
                </div>
            """.format(prediction), unsafe_allow_html=True)

            # Input Summary
            st.markdown("### 📝 Selected Property Details")

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
        st.info("👈 Enter details from the sidebar and click Predict.")

    st.markdown("---")
    st.caption("Developed using Streamlit & Machine Learning")


if __name__ == "__main__":
    main()
