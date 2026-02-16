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

# Custom CSS for better styling
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

@st.cache_resource
def load_model():
    """Load the trained model"""
    if not os.path.exists('model.pkl'):
        st.error("Model file not found! Please run train_model.py first.")
        return None
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_metrics():
    """Load model metrics"""
    metrics = {}
    if os.path.exists('model_metrics.txt'):
        with open('model_metrics.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'MAE' in line:
                    metrics['MAE'] = float(line.split(':')[1].strip().replace(',', ''))
                elif 'RMSE' in line:
                    metrics['RMSE'] = float(line.split(':')[1].strip().replace(',', ''))
                elif 'R²' in line:
                    metrics['R2'] = float(line.split(':')[1].strip())
    return metrics

def get_unique_values_from_dataset():
    """Get unique values for categorical features from dataset"""
    try:
        df = pd.read_csv('estate_rent_dataset.csv')
        return {
            'cities': sorted(df['City'].unique().tolist()),
            'locations': sorted(df['Location'].unique().tolist()),
            'furnishing': sorted(df['Furnishing'].unique().tolist())
        }
    except Exception as e:
        st.warning(f"Could not load dataset for unique values: {e}")
        return {
            'cities': ['Bangalore', 'Mumbai', 'Hyderabad', 'Chennai', 'Delhi', 'Pune', 'Kolkata'],
            'locations': ['Whitefield', 'Andheri', 'Gachibowli', 'Velachery', 'Dwarka', 'Hinjewadi', 'New Town', 'Electronic City', 'Powai', 'Madhapur'],
            'furnishing': ['Furnished', 'Semi-Furnished', 'Unfurnished']
        }

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏠 Rental Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Load metrics
    metrics = load_metrics()
    
    # Sidebar for user inputs
    st.sidebar.header("📋 Property Details")
    st.sidebar.markdown("Enter the property features below:")
    
    # Get unique values
    unique_values = get_unique_values_from_dataset()
    
    # User inputs
    city = st.sidebar.selectbox(
        "City",
        options=unique_values['cities'],
        index=0
    )
    
    location = st.sidebar.selectbox(
        "Location",
        options=unique_values['locations'],
        index=0
    )
    
    bhk = st.sidebar.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )
    
    size_sqft = st.sidebar.number_input(
        "Size (sqft)",
        min_value=100,
        max_value=10000,
        value=1200,
        step=50
    )
    
    bathrooms = st.sidebar.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )
    
    floor = st.sidebar.number_input(
        "Floor",
        min_value=0,
        max_value=50,
        value=3,
        step=1
    )
    
    total_floors = st.sidebar.number_input(
        "Total Floors",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )
    
    furnishing = st.sidebar.selectbox(
        "Furnishing Status",
        options=unique_values['furnishing'],
        index=1
    )
    
    property_age = st.sidebar.number_input(
        "Property Age (years)",
        min_value=0,
        max_value=50,
        value=5,
        step=1
    )
    
    parking = st.sidebar.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=5,
        value=1,
        step=1
    )
    
    st.sidebar.markdown("---")
    
    # Predict button
    predict_button = st.sidebar.button(
        "🔮 Predict Rental Price",
        type="primary",
        use_container_width=True
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Prediction Results")
        
        if predict_button:
            # Prepare input data
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
            
            # Make prediction
            try:
                prediction = model.predict(input_data)[0]
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: #1f77b4; margin-bottom: 1rem;">Predicted Rental Price</h2>
                    <h1 style="font-size: 3rem; color: #2ecc71; margin: 0;">₹{prediction:,.0f}</h1>
                    <p style="color: #7f8c8d; margin-top: 0.5rem;">per month</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display input summary
                st.markdown("### 📝 Input Summary")
                summary_df = pd.DataFrame({
                    'Feature': ['City', 'Location', 'BHK', 'Size (sqft)', 'Bathrooms', 
                               'Floor', 'Total Floors', 'Furnishing', 'Property Age', 'Parking'],
                    'Value': [city, location, bhk, size_sqft, bathrooms, floor, 
                             total_floors, furnishing, property_age, parking]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        else:
            st.info("👈 Fill in the property details in the sidebar and click 'Predict Rental Price' to get started!")
    
    with col2:
        st.subheader("📈 Model Performance")
        
        if metrics:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
            st.metric("MAE", f"₹{metrics.get('MAE', 0):,.0f}")
            st.metric("RMSE", f"₹{metrics.get('RMSE', 0):,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("""
            **Metrics Explanation:**
            - **R² Score**: Higher is better (max 1.0)
            - **MAE**: Average prediction error
            - **RMSE**: Penalizes larger errors more
            """)
        else:
            st.warning("Model metrics not available. Run train_model.py to generate metrics.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>"
        "Built with ❤️ using Streamlit and Random Forest Regression"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

