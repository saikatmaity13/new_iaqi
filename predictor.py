import joblib
import numpy as np
import pandas as pd

# ===================================================================
# LOAD ALL MODELS AND TRANSFORMERS ONCE
# ===================================================================

try:
    # Load PM2.5 components
    model_low_pm25 = joblib.load('saved_models/model_low_pm25.pkl')
    model_high_pm25 = joblib.load('saved_models/model_high_pm25.pkl')
    poly_pm25 = joblib.load('saved_models/poly_pm25.pkl')
    rf_pm25 = joblib.load('saved_models/rf_pm25.pkl')

    # Load PM10 components
    model_low_pm10 = joblib.load('saved_models/model_low_pm10.pkl')
    model_high_pm10 = joblib.load('saved_models/model_high_pm10.pkl')
    poly_pm10 = joblib.load('saved_models/poly_pm10.pkl')
    rf_pm10 = joblib.load('saved_models/rf_pm10.pkl')

    # Load CO components
    model_low_co = joblib.load('saved_models/model_low_co.pkl')
    model_high_co = joblib.load('saved_models/model_high_co.pkl')
    poly_co = joblib.load('saved_models/poly_co.pkl')
    rf_co = joblib.load('saved_models/rf_co.pkl')

except FileNotFoundError:
    print("Error: One or more model files (.pkl) not found. Please run train_models.py first.")
    exit()

# Define the thresholds used during training
BEST_THRESH_PM25 = 10.83
BEST_THRESH_PM10 = 16.94
BEST_THRESH_CO = 150.06


# ===================================================================
# PREDICTION FUNCTIONS
# ===================================================================

def predict_pm25(input_df: pd.DataFrame):
    """Predicts a single PM2.5 value from a DataFrame row."""
    required_cols = rf_pm25.feature_names_in_
    input_data = input_df[required_cols]

    # 1. TAR Prediction
    X_poly = poly_pm25.transform(input_data)
    tar_pred = (model_low_pm25.predict(X_poly) if input_df['PM2.5_lag1'].iloc[0] <= BEST_THRESH_PM25
                else model_high_pm25.predict(X_poly))

    # 2. RF Residual Correction
    residual_pred = rf_pm25.predict(input_data)

    return tar_pred[0] + residual_pred[0]


def predict_pm10(input_df: pd.DataFrame):
    """Predicts a single PM10 value from a DataFrame row."""
    required_cols = rf_pm10.feature_names_in_
    input_data = input_df[required_cols]

    X_poly = poly_pm10.transform(input_data)
    tar_pred = (model_low_pm10.predict(X_poly) if input_df['PM10_lag1'].iloc[0] <= BEST_THRESH_PM10
                else model_high_pm10.predict(X_poly))

    residual_pred = rf_pm10.predict(input_data)

    return tar_pred[0] + residual_pred[0]


def predict_co(input_df: pd.DataFrame):
    """Predicts a single CO value from a DataFrame row."""
    required_cols = rf_co.feature_names_in_
    input_data = input_df[required_cols]

    X_poly = poly_co.transform(input_data)
    tar_pred = (model_low_co.predict(X_poly) if input_df['CO_lag1'].iloc[0] <= BEST_THRESH_CO
                else model_high_co.predict(X_poly))

    residual_pred = rf_co.predict(input_data)

    return tar_pred[0] + residual_pred[0]