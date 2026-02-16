# 🏠 Rental Price Prediction Web Application

A production-ready machine learning web application for predicting rental prices of properties using Random Forest Regression.
🔗 Try it live on Streamlit:[Rental Price Predictor](https://rental-price-predictor-l8dqncgsxqju7fvjksypxc.streamlit.app/#selected-property-details
)
✨ Try it now: Rental Price Predictor
## 📋 Features

- **Interactive Web Interface**: Clean and professional Streamlit-based UI
- **Real-time Predictions**: Get instant rental price predictions based on property features
- **Model Performance Metrics**: View MAE, RMSE, and R² scores
- **Production-Ready Pipeline**: Complete preprocessing pipeline with categorical encoding and feature scaling

## 🗂️ Project Structure

```
rental-price-app/
├── estate_rent_dataset.csv      # Dataset file
├── train_model.py                # Model training script
├── app.py                        # Streamlit web application
├── model.pkl                     # Trained model (generated after training)
├── model_metrics.txt             # Model evaluation metrics (generated after training)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Before running the web app, you need to train the model:

```bash
python train_model.py
```

This will:
- Load the dataset from `estate_rent_dataset.csv`
- Preprocess the data (categorical encoding, feature scaling)
- Train a RandomForestRegressor model
- Evaluate the model and save metrics
- Save the trained model as `model.pkl`

### 3. Run the Web Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 Dataset Features

The model uses the following features:

- **City**: Property location city
- **Location**: Specific area/locality
- **BHK**: Number of bedrooms/halls/kitchen
- **Size_sqft**: Property size in square feet
- **Bathrooms**: Number of bathrooms
- **Floor**: Floor number
- **Total_Floors**: Total floors in the building
- **Furnishing**: Furnishing status (Furnished/Semi-Furnished/Unfurnished)
- **Property_Age**: Age of the property in years
- **Parking**: Number of parking spaces

**Target Variable**: `Rent` (monthly rental price in ₹)

## 🎯 Model Details

- **Algorithm**: Random Forest Regressor
- **Preprocessing**:
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
- **Evaluation Metrics**: MAE, RMSE, R² Score

## ☁️ Deployment to Streamlit Cloud

### Prerequisites

1. A GitHub account
2. Your code pushed to a GitHub repository
3. A Streamlit Cloud account (free at [streamlit.io/cloud](https://streamlit.io/cloud))

### Deployment Steps

#### Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Rental price prediction app"
   ```

2. **Create a GitHub repository**:
   - Go to [github.com](https://github.com) and create a new repository
   - Name it (e.g., `rental-price-app`)
   - **Do NOT** initialize with README, .gitignore, or license

3. **Push your code**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/rental-price-app.git
   git branch -M main
   git push -u origin main
   ```

#### Step 2: Deploy to Streamlit Cloud

1. **Sign in to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**:
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/rental-price-app`
   - Set the branch: `main`
   - Set the main file path: `app.py`
   - Click "Deploy!"

3. **Important Notes**:
   - **Model File**: Make sure `model.pkl` is committed to your repository. If the file is too large (>100MB), consider using Git LFS or hosting it elsewhere.
   - **Dataset**: The `estate_rent_dataset.csv` file should be in the repository for the app to work properly.
   - **Requirements**: Streamlit Cloud will automatically install packages from `requirements.txt`

#### Step 3: Post-Deployment

After deployment:
- Streamlit Cloud will automatically rebuild your app when you push changes
- Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`
- You can share this URL with others

### Alternative: Manual Model Training on Streamlit Cloud

If you prefer not to commit the model file, you can modify `app.py` to train the model on first run:

```python
if not os.path.exists('model.pkl'):
    st.info("Training model... This may take a few minutes.")
    import subprocess
    subprocess.run(['python', 'train_model.py'])
    model = load_model()
```

**Note**: This will slow down the first app load significantly.

## 🔧 Configuration

### Model Parameters

You can modify model hyperparameters in `train_model.py`:

```python
RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=5,  # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples in leaf
    random_state=42        # Random seed
)
```

### App Customization

Modify `app.py` to customize:
- UI colors and styling (CSS in the `st.markdown` section)
- Input ranges and defaults
- Display format

## 📝 Usage

1. **Training**: Run `python train_model.py` to train/retrain the model
2. **Prediction**: Use the Streamlit app to input property features and get predictions
3. **Metrics**: View model performance metrics in the app sidebar

## 🐛 Troubleshooting

### Model file not found
- **Error**: "Model file not found! Please run train_model.py first."
- **Solution**: Run `python train_model.py` to generate `model.pkl`

### Import errors
- **Error**: ModuleNotFoundError
- **Solution**: Install dependencies with `pip install -r requirements.txt`

### Streamlit Cloud deployment issues
- **Issue**: App fails to deploy
- **Solution**: 
  - Check that `requirements.txt` includes all dependencies
  - Ensure `app.py` is in the root directory
  - Verify all files are committed to GitHub

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Built as a production-ready ML application for rental price prediction.

---

**Happy Predicting! 🎉**


