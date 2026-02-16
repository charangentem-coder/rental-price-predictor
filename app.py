"""
Streamlit Web Application for Rental Price Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Rental Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


# ------------------ LOAD MODEL ------------------ #
@st.cache_resource
def load_model():
    if not os.path.exists('model.pkl'):
        st.error("Model file not found! Please run train_model.py first.")
        return None

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    return model


# ------------------ LOAD METRICS (FIXED) ------------------ #
@st.cache_data
def load_metrics():
    metrics = {}

    if os.path.exists('model_metrics.txt'):
        try:
            # FIX: Explicit UTF-8 encoding
            with open('model_metrics.txt', 'r', encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            for line in lines:
                if 'MAE' in line:
                    metrics['MAE'] = float(line.split(':')[1].strip().replace(',', ''))
                elif 'RMSE' in line:
                    metrics['RMSE'] = float(line.split(':')[1].strip().replace(',', ''))
                elif 'R²' in line or 'R2' in line:
                    metrics['R2'] = float(line.split(':')[1].strip())

        except Exception as e:
            st.warning(f"Error reading metrics file: {e}")

    return metrics


# ------------------ LOAD UNIQUE VALUES ------------------ #
def get_unique_values_from_dataset():
    try:
        # FIX: Added encoding fallback
        try:
            df = pd.read_csv('estate_rent_dataset.csv', encoding="utf-8")
        except:
            df = pd.read_csv('estate_rent_dataset.csv', encoding="latin1")

        return {
            'cities': sorted(df['City'].unique().tolist()),
            'locations': sorted(df['Location'].unique().tolist()),
            'furnishing': sorted(df['Furnishing'].unique().tolist())
        }

    except Exception as e:
        st.warning(f"Could not load dataset: {e}")

        return {
            'cities': ['Bangalore', 'Mumbai', 'Hyderabad', 'Chennai', 'Delhi', 'Pune', 'Kolkata'],
            'locations': ['Whitefield', 'Andheri', 'Gachibowli', 'Velachery', 'Dwarka',
                          'Hinjewadi', 'New Town', 'Electronic City', 'Powai', 'Madhapur'],
            'furnishing': ['Furnished', 'Semi-Furnished', 'Unfurnished']
        }


# ------------------ MAIN APP ------------------ #
def main():

    st.markdown('<h1 class="main-header">🏠 Rental Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")

    model = load_model()
    if model is None:
        st.stop()

    metrics = load_metrics()

    st.sidebar.header("📋 Property Details")
    st.sidebar.markdown("Enter the property features below:")

    unique_values = get_unique_values_from_dataset()

    city = st.sidebar.selectbox("City", unique_values['cities'])
    location = st.sidebar.selectbox("Location", unique_values['locations'])

    bhk = st.sidebar.number_input("BHK", 1, 10, 2)
    size_sqft = st.sidebar.number_input("Size (sqft)", 100, 10000, 1200, step=50)
    bathrooms = st.sidebar.number_input("Bathrooms", 1, 10, 2)
    floor = st.sidebar.number_input("Floor", 0, 50, 3)
    total_floors = st.sidebar.number_input("Total Floors", 1, 50, 10)
    furnishing = st.sidebar.selectbox("Furnishing Status", unique_values['furnishing'])
    property_age = st.sidebar.number_input("Property Age (years)", 0, 50, 5)
    parking = st.sidebar.number_input("Parking Spaces", 0, 5, 1)

    predict_button = st.sidebar.button("🔮 Predict Rental Price", type="primary")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Prediction Results")

        if predict_button:
            input_data = pd.DataFrame({
                'City': [city],
                'Location': [location],
                'BHK': [bhk],
                'Size_sqft': [size_sqft],
                'Bathrooms': [bathrooms],
                'Floor': [floor],
                'Total_Floors': [total_floors],
                'Furnishing': [furnishing],
                'Property_Age': [property_age],
                'Parking': [parking]
            })

            try:
                prediction = model.predict(input_data)[0]

                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Rental Price</h2>
                    <h1 style="color: #2ecc71;">₹{prediction:,.0f}</h1>
                    <p>per month</p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

        else:
            st.info("👈 Enter details in sidebar and click Predict")

    with col2:
        st.subheader("📈 Model Performance")

        if metrics:
            st.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
            st.metric("MAE", f"₹{metrics.get('MAE', 0):,.0f}")
            st.metric("RMSE", f"₹{metrics.get('RMSE', 0):,.0f}")
        else:
            st.warning("Metrics not available.")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:gray;'>Built with using Streamlit</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
