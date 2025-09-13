import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# --- Helper Function ---
def create_lagged_data(df, lags=1):
    """Creates lagged features for all columns in the DataFrame."""
    df_lagged = df.copy()
    for col in df.columns:
        for lag in range(1, lags + 1):
            df_lagged[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df_lagged.dropna()

def fit_tar_model_poly(df_lagged, threshold, target_col, features, alpha=1.0):
    """Fits a Threshold Autoregressive (TAR) model with two regimes."""
    poly = PolynomialFeatures(degree=1, include_bias=False)
    
    regime_low = df_lagged[df_lagged[f'{target_col}_lag1'] <= threshold]
    regime_high = df_lagged[df_lagged[f'{target_col}_lag1'] > threshold]
    
    if regime_low.empty or regime_high.empty:
        raise ValueError(f"For {target_col}, one regime is empty. Adjust threshold.")
        
    X_low, y_low = regime_low[features], regime_low[target_col]
    X_high, y_high = regime_high[features], regime_high[target_col]
    
    X_low_poly = poly.fit_transform(X_low)
    X_high_poly = poly.transform(X_high)
    
    model_low = Ridge(alpha=alpha).fit(X_low_poly, y_low)
    model_high = Ridge(alpha=alpha).fit(X_high_poly, y_high)
    
    return model_low, model_high, poly

# --- Data Loading and Preparation ---
print("Loading and preparing base data...")
os.makedirs('data', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

try:
    # Read the CSV without parsing dates automatically
    df_base = pd.read_csv('data/CPCB.csv')
    
    # Manually convert 'Datetime' using the specific format to avoid errors
    df_base['Datetime'] = pd.to_datetime(df_base['Datetime'], format='%d-%m-%Y %H:%M', errors='coerce')
    
    # Drop any rows that couldn't be converted
    df_base.dropna(subset=['Datetime'], inplace=True)
    
    # Now, set the DatetimeIndex
    df_base.set_index('Datetime', inplace=True)

except FileNotFoundError:
    print("Error: 'data/CPCB.csv' not found. Please ensure it is in the 'data' subfolder.")
    exit()

# Add time-based features (this will now work correctly)
df_base['hour'] = df_base.index.hour
df_base['dayofweek'] = df_base.index.dayofweek
print("Base data loaded and time features added.")
print("-" * 50)


# ===================================================================
# 1. TRAIN AND SAVE PM2.5 MODEL
# ===================================================================
print("🚀 Starting PM2.5 Model Training...")
df_pm25 = df_base.copy()
df_pm25['PM2.5_roll3'] = df_pm25['PM2.5'].rolling(9).mean()
df_pm25['Temp_roll3'] = df_pm25['Temp'].rolling(9).mean()
df_pm25 = df_pm25.dropna()

features_pm25 = ['SO2', 'CO', 'PM10', 'Temp', 'RH', 'hour', 'dayofweek', 'PM2.5_roll3', 'Temp_roll3']
train_pm25 = df_pm25.iloc[:int(len(df_pm25) * 0.7)]
train_lagged_pm25 = create_lagged_data(train_pm25, lags=1)

# Add lagged versions of the features to the feature list
features_lagged_pm25 = [f'{col}_lag1' for col in train_pm25.columns]
X_train_pm25 = train_lagged_pm25[features_lagged_pm25]
y_train_pm25 = train_lagged_pm25['PM2.5']

# -- Stage 1: Train TAR Model --
best_thresh_pm25 = 10.83
model_low_pm25, model_high_pm25, poly_pm25 = fit_tar_model_poly(
    train_lagged_pm25, best_thresh_pm25, 'PM2.5', features_lagged_pm25, alpha=0.1
)

# -- Stage 2: Train RF on TAR Residuals --
X_train_poly = poly_pm25.transform(X_train_pm25)
condition = train_lagged_pm25['PM2.5_lag1'] <= best_thresh_pm25
yhat_train_pm25 = np.where(condition, model_low_pm25.predict(X_train_poly), model_high_pm25.predict(X_train_poly))
residuals_train_pm25 = y_train_pm25 - yhat_train_pm25

rf_pm25 = RandomForestRegressor(n_estimators=30, random_state=42)
rf_pm25.fit(X_train_pm25, residuals_train_pm25)

# -- Save all components --
joblib.dump(model_low_pm25, 'saved_models/model_low_pm25.pkl')
joblib.dump(model_high_pm25, 'saved_models/model_high_pm25.pkl')
joblib.dump(poly_pm25, 'saved_models/poly_pm25.pkl')
joblib.dump(rf_pm25, 'saved_models/rf_pm25.pkl')
print("✅ PM2.5 model components saved successfully!")
print("-" * 50)


# ===================================================================
# 2. TRAIN AND SAVE PM10 MODEL
# ===================================================================
print("🚀 Starting PM10 Model Training...")
df_pm10 = df_base.copy()
df_pm10['PM10_roll3'] = df_pm10['PM10'].rolling(9).mean()
df_pm10['Temp_roll3'] = df_pm10['Temp'].rolling(9).mean()
df_pm10 = df_pm10.dropna()

train_pm10 = df_pm10.iloc[:int(len(df_pm10) * 0.7)]
train_lagged_pm10 = create_lagged_data(train_pm10, lags=1)

features_lagged_pm10 = [f'{col}_lag1' for col in train_pm10.columns]
X_train_pm10 = train_lagged_pm10[features_lagged_pm10]
y_train_pm10 = train_lagged_pm10['PM10']

best_thresh_pm10 = 16.94
model_low_pm10, model_high_pm10, poly_pm10 = fit_tar_model_poly(
    train_lagged_pm10, best_thresh_pm10, 'PM10', features_lagged_pm10, alpha=0.01
)

X_train_poly = poly_pm10.transform(X_train_pm10)
condition = train_lagged_pm10['PM10_lag1'] <= best_thresh_pm10
yhat_train_pm10 = np.where(condition, model_low_pm10.predict(X_train_poly), model_high_pm10.predict(X_train_poly))
residuals_train_pm10 = y_train_pm10 - yhat_train_pm10

rf_pm10 = RandomForestRegressor(n_estimators=30, random_state=42)
rf_pm10.fit(X_train_pm10, residuals_train_pm10)

joblib.dump(model_low_pm10, 'saved_models/model_low_pm10.pkl')
joblib.dump(model_high_pm10, 'saved_models/model_high_pm10.pkl')
joblib.dump(poly_pm10, 'saved_models/poly_pm10.pkl')
joblib.dump(rf_pm10, 'saved_models/rf_pm10.pkl')
print("✅ PM10 model components saved successfully!")
print("-" * 50)


# ===================================================================
# 3. TRAIN AND SAVE CO MODEL
# ===================================================================
print("🚀 Starting CO Model Training...")
df_co = df_base.copy()
df_co['CO_roll_3_day'] = df_co['CO'].rolling(window=3).mean()
df_co = df_co.dropna()

train_co = df_co.iloc[:int(len(df_co) * 0.7)]
train_lagged_co = create_lagged_data(train_co, lags=3)

features_lagged_co = [f'{c}_lag{l}' for c in train_co.columns for l in [1, 2, 3]]
X_train_co = train_lagged_co[features_lagged_co]
y_train_co = train_lagged_co['CO']

best_thresh_co = 150.06
model_low_co, model_high_co, poly_co = fit_tar_model_poly(
    train_lagged_co, best_thresh_co, 'CO', features_lagged_co, alpha=1.0
)

X_train_poly = poly_co.transform(X_train_co)
condition = train_lagged_co['CO_lag1'] <= best_thresh_co
yhat_train_co = np.where(condition, model_low_co.predict(X_train_poly), model_high_co.predict(X_train_poly))
residuals_train_co = y_train_co - yhat_train_co

rf_co = RandomForestRegressor(n_estimators=100, random_state=42)
rf_co.fit(X_train_co, residuals_train_co)

joblib.dump(model_low_co, 'saved_models/model_low_co.pkl')
joblib.dump(model_high_co, 'saved_models/model_high_co.pkl')
joblib.dump(poly_co, 'saved_models/poly_co.pkl')
joblib.dump(rf_co, 'saved_models/rf_co.pkl')
print("✅ CO model components saved successfully!")
print("-" * 50)

print("🎉 All models have been trained and saved successfully.")
