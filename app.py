"""
Rental Price Prediction - Premium Soft Blue Version
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

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

/* Soft Shaded Blue Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e3f2fd, #f5f7fa);
}

/* Main card container */
.main-card {
    background-color: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-top: 20px;
}

/* Title */
.title-text {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #1f2d3d;
}

/* Subtitle */
.subtitle-text {
    text-align: center;
    color: #5f6c7b;
    margin-bottom: 30px;
    font-size: 16px;
}

/* Prediction card */
.prediction-box {
    text-align: center;
    padding: 50px;
    border-radius: 20px;
    background: linear-gradient(135deg, #ffffff, #f1f5f9);
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}

/* Prediction value */
.prediction-value {
    font-size: 70px;
    font-weight: bold;
    color: #2e7d32;
    margin: 15px 0;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #f8fafc;
}

/* Custom Button */
.stButton>button {
    background: linear-gradient(135deg, #4e73df, #6c8ff5);
    color: white;
    font-size: 17px;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 20px;
    border: none;
    transition: 0.3s ease;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #3b5edb, #5a7df0);
    color: white;
    transform: scale(1.02);
}

.footer {
    text-align: center;
    color: #6c757d;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

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

    st.markdown("<div class='title-text'>🏠 Rental Price Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-text'>AI Powered Monthly Rent Estimator</div>", unsafe_allow_html=True)

    model = load_model()
    df = load_dataset()

    if model is None or df is None:
        st.stop()

    # ---------------- SIDEBAR ---------------- #
    st.sidebar.header("📋 Property Details")

    city = st.sidebar.selectbox(
        "City",
        sorted(df["City"].unique())
    )

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

    # ---------------- PREDICTION ---------------- #
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

            st.markdown(f"""
            <div class='main-card'>
                <div class='prediction-box'>
                    <h3>Estimated Monthly Rent</h3>
                    <div class='prediction-value'>₹{prediction:,.0f}</div>
                    <p>Predicted using Machine Learning Model</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

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

    st.markdown("<div class='footer'>Developed using Streamlit & Machine Learning</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
