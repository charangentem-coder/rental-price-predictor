import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Rental Price Predictor", page_icon="🏠", layout="centered")

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>

body {
    background-color: #f4f6f9;
}

.main {
    background: linear-gradient(to right, #f8f9fa, #eef2f7);
}

.stButton>button {
    background: linear-gradient(to right, #1f4037, #99f2c8);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
}

.stButton>button:hover {
    background: linear-gradient(to right, #0f2027, #2c5364);
    color: white;
}

.result-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-top: 20px;
}

.summary-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("rent_model.pkl", "rb"))

# -----------------------------
# Title Section
# -----------------------------
st.markdown("<h1 style='text-align: center;'>🏠 Smart Rental Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>AI Powered Rental Estimation System</p>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("📋 Enter Property Details")

location = st.selectbox("Select Location", ["Hyderabad", "Kolkata", "Mumbai", "Delhi"])
bhk = st.selectbox("Select BHK", [1, 2, 3, 4])
bathroom = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])
area = st.number_input("Area (in square feet)", min_value=200, max_value=5000, value=1000)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔮 Predict Rental Price"):

    input_df = pd.DataFrame({
        "location": [location],
        "bhk": [bhk],
        "bathroom": [bathroom],
        "area": [area]
    })

    prediction = model.predict(input_df)[0]

    # -------- RESULT DISPLAY --------
    st.markdown(f"""
    <div class="result-box">
        <h2 style="text-align:center; color:#1f4037;">
        💰 Estimated Monthly Rent: ₹ {round(prediction,2)}
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # -------- INPUT SUMMARY DISPLAY --------
    st.markdown(f"""
    <div class="summary-box">
        <b>Selected Property Details:</b><br><br>
        📍 Location: {location}<br>
        🏢 BHK: {bhk}<br>
        🚿 Bathrooms: {bathroom}<br>
        📐 Area: {area} sq.ft
    </div>
    """, unsafe_allow_html=True)
