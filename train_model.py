"""
Model Training Script for Rental Price Prediction
Trains a RandomForestRegressor with proper preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

def load_data(filepath='estate_rent_dataset.csv'):
    """Load the rental dataset"""
    df = pd.read_csv(filepath)
    return df

def prepare_features(df):
    """Prepare features and target variable"""
    # Drop Property_ID as it's not a feature
    X = df.drop(['Property_ID', 'Rent'], axis=1)
    y = df['Rent']
    return X, y

def create_preprocessing_pipeline():
    """Create preprocessing pipeline for categorical and numerical features"""
    # Define categorical and numerical columns
    categorical_features = ['City', 'Location', 'Furnishing']
    numerical_features = ['BHK', 'Size_sqft', 'Bathrooms', 'Floor', 'Total_Floors', 'Property_Age', 'Parking']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def train_model(X_train, y_train, preprocessor):
    """Train RandomForestRegressor model"""
    # Create full pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train the model
    print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return mae, rmse, r2, y_pred

def save_model(model, filepath='model.pkl'):
    """Save trained model to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def save_metrics(mae, rmse, r2, filepath='model_metrics.txt'):
    """Save model metrics to text file"""
    with open(filepath, 'w') as f:
        f.write("Model Evaluation Metrics\n")
        f.write("=" * 30 + "\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R² Score: {r2:.4f}\n")
    print(f"Metrics saved to {filepath}")

def main():
    """Main training function"""
    print("=" * 50)
    print("Rental Price Prediction - Model Training")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading dataset...")
    df = load_data('estate_rent_dataset.csv')
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Prepare features
    print("\n2. Preparing features...")
    X, y = prepare_features(df)
    print(f"   Features: {list(X.columns)}")
    print(f"   Target: Rent")
    
    # Split data
    print("\n3. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Create preprocessing pipeline
    print("\n4. Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline()
    
    # Train model
    print("\n5. Training model...")
    model = train_model(X_train, y_train, preprocessor)
    
    # Evaluate model
    print("\n6. Evaluating model...")
    mae, rmse, r2, y_pred = evaluate_model(model, X_test, y_test)
    
    print("\n" + "=" * 50)
    print("Model Evaluation Results")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print("=" * 50)
    
    # Save model and metrics
    print("\n7. Saving model and metrics...")
    save_model(model, 'model.pkl')
    save_metrics(mae, rmse, r2, 'model_metrics.txt')
    
    print("\n✓ Training completed successfully!")

if __name__ == "__main__":
    main()

