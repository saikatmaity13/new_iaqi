from flask import Flask, render_template, request, jsonify
import pandas as pd
import warnings
from datetime import timedelta

# --- Import your predictor and helper functions ---
from predictor import predict_pm25, predict_pm10, predict_co
from train_models import create_lagged_data

warnings.filterwarnings('ignore')

app = Flask(__name__)


def get_starting_data_for_forecast():
    """
    Loads and prepares the CPCB.csv data to create the initial input for the forecast.
    """
    try:
        df_base = pd.read_csv('data/CPCB.csv')
        df_base['Datetime'] = pd.to_datetime(df_base['Datetime'], format='%d-%m-%Y %H:%M', errors='coerce')
        df_base.dropna(subset=['Datetime'], inplace=True)
        df_base.set_index('Datetime', inplace=True)

        # Add features
        df_base['hour'] = df_base.index.hour
        df_base['dayofweek'] = df_base.index.dayofweek
        df_base['PM2.5_roll3'] = df_base['PM2.5'].rolling(9, min_periods=1).mean()
        df_base['Temp_roll3'] = df_base['Temp'].rolling(9, min_periods=1).mean()
        df_base['PM10_roll3'] = df_base['PM10'].rolling(9, min_periods=1).mean()
        df_base['CO_roll_3_day'] = df_base['CO'].rolling(3, min_periods=1).mean()

        df_base.dropna(inplace=True)

        # Create lagged data
        df_lagged = create_lagged_data(df_base, lags=3)
        if df_lagged.empty:
            raise ValueError("Not enough data in CPCB.csv to create starting features.")
            
        return df_lagged.iloc[[-1]]

    except FileNotFoundError:
        raise FileNotFoundError("Error: 'data/CPCB.csv' not found.")
    except Exception as e:
        raise Exception(f"An error occurred while preparing data: {e}")


@app.route('/')
def home():
    """Serve the HTML frontend"""
    return render_template('index.html')


@app.route('/upload_and_forecast', methods=['POST'])
def upload_and_forecast():
    """
    Handle CSV upload and return 7-day forecast for all metrics as JSON.
    """
    try:
        if 'file' in request.files:
            file = request.files['file']
            file.save('data/CPCB.csv')

        last_known_data = get_starting_data_for_forecast()
        current_input = last_known_data.copy()
        forecasts = []

        last_date = pd.to_datetime(current_input.index[0])

        for day in range(7):
            pred_pm25 = predict_pm25(current_input)
            pred_pm10 = predict_pm10(current_input)
            pred_co = predict_co(current_input)

            forecast = {
                "Day": day + 1,
                "PM2.5": round(pred_pm25, 2),
                "PM10": round(pred_pm10, 2),
                "CO": round(pred_co, 2),
            }
            forecasts.append(forecast)

            all_feature_cols = [col for col in last_known_data.columns if '_lag' not in col]
            for col_name in all_feature_cols:
                for i in range(3, 1, -1):
                    if f'{col_name}_lag{i}' in current_input.columns and f'{col_name}_lag{i-1}' in current_input.columns:
                        current_input[f'{col_name}_lag{i}'] = current_input[f'{col_name}_lag{i-1}']

            current_input['PM2.5_lag1'] = pred_pm25
            current_input['PM10_lag1'] = pred_pm10
            current_input['CO_lag1'] = pred_co

            next_day_datetime = last_date + timedelta(days=day + 1)
            current_input['hour'] = next_day_datetime.hour
            current_input['dayofweek'] = next_day_datetime.dayofweek

        return jsonify({"forecasts": forecasts})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/forecast/<metric>', methods=['POST'])
def forecast_metric(metric):
    """
    Handle a request to forecast a single metric for 7 days.
    """
    try:
        valid_metrics = ['PM2.5', 'PM10', 'CO']
        if metric not in valid_metrics:
            return jsonify({"error": "Invalid metric specified. Choose from 'PM2.5', 'PM10', 'CO'."}), 400

        last_known_data = get_starting_data_for_forecast()
        current_input = last_known_data.copy()
        forecasts = []

        last_date = pd.to_datetime(current_input.index[0])

        for day in range(7):
            pred = None
            if metric == 'PM2.5':
                pred = predict_pm25(current_input)
            elif metric == 'PM10':
                pred = predict_pm10(current_input)
            elif metric == 'CO':
                pred = predict_co(current_input)

            if pred is None:
                raise Exception(f"Prediction failed for {metric}")

            forecast = {
                "Day": day + 1,
                metric: round(pred, 2)
            }
            forecasts.append(forecast)

            all_feature_cols = [col for col in last_known_data.columns if '_lag' not in col]
            for col_name in all_feature_cols:
                for i in range(3, 1, -1):
                    if f'{col_name}_lag{i}' in current_input.columns and f'{col_name}_lag{i-1}' in current_input.columns:
                        current_input[f'{col_name}_lag{i}'] = current_input[f'{col_name}_lag{i-1}']

            current_input[f'{metric}_lag1'] = pred

            next_day_datetime = last_date + timedelta(days=day + 1)
            current_input['hour'] = next_day_datetime.hour
            current_input['dayofweek'] = next_day_datetime.dayofweek

        return jsonify({"forecasts": forecasts})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
