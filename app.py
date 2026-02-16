import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Rental Price Predictor", page_icon="🏠", layout="centered")

# ---------------------------
# Custom CSS
# ---------------------------
st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #f8f9fa, #eef2f7);
}

.stButton>button {
    background: linear-gradient(to right, #1f4037, #99f2c8);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
}

.result-box {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    margin-top: 25px;
    text-align: center;
}

.summary-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------
model = pickle.load(open("rent_model.pkl", "rb"))

# Optional R2 score (write your actual R2 here)
R2_SCORE = 0.89

# ---------------------------
# Title
# ---------------------------
st.markdown("<h1 style='text-align:center;'>🏠 Smart Rental Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:grey;'>AI Powered Rental Estimation System</p>", unsafe_allow_html=True)

st.write("")

# ---------------------------
# Inputs
# ---------------------------
st.subheader("📋 Enter Property Details")

location = st.selectbox("Select Location",
                        ["Hyderabad", "Kolkata", "Mumbai", "Delhi"])

bhk = st.selectbox("Number of BHK", [1, 2, 3, 4, 5])
bathroom = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])
area = st.number_input("Area (in square feet)", min_value=200, max_value=5000, value=1000)

# ---------------------------
# Prediction Logic
# ---------------------------
if st.button("🔮 Predict Rental Price"):

    # Create base dataframe
    input_dict = {
        "area": area,
        "bathroom": bathroom,
        "bhk": bhk,
        "location_Delhi": 0,
        "location_Hyderabad": 0,
        "location_Kolkata": 0,
        "location_Mumbai": 0
    }

    # Activate selected location
    input_dict[f"location_{location}"] = 1

    input_df = pd.DataFrame([input_dict])

    # Prediction
    prediction = model.predict(input_df)[0]

    # ---------------------------
    # Display Result
    # ---------------------------
    st.markdown(f"""
    <div class="result-box">
        <h2 style="color:#1f4037;">💰 Estimated Monthly Rent</h2>
        <h1 style="color:#0f2027;">₹ {round(prediction,2)}</h1>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------
    # Display Summary
    # ---------------------------
    st.markdown(f"""
    <div class="summary-box">
        <b>Selected Property Details:</b><br><br>
        📍 Location: {location}<br>
        🏢 BHK: {bhk}<br>
        🚿 Bathrooms: {bathroom}<br>
        📐 Area: {area} sq.ft
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------
    # Optional Model Performance
    # ---------------------------
    st.markdown(f"<p style='text-align:center; color:grey; margin-top:20px;'>Model R² Score: {R2_SCORE}</p>", unsafe_allow_html=True)
