import pandas as pd
import warnings
from predictor import predict_pm25, predict_pm10, predict_co

warnings.filterwarnings('ignore')

def get_last_known_data():
    """
    Loads the base CSV, prepares all necessary features including time and lags,
    and returns the last row of data as the starting point for the forecast.
    """
    try:
        from train_models import create_lagged_data

        # Read the CSV and explicitly parse the Datetime column
        df_base = pd.read_csv('data/CPCB.csv')
        df_base['Datetime'] = pd.to_datetime(df_base['Datetime'], format='%d-%m-%Y %H:%M', errors='coerce')
        df_base.dropna(subset=['Datetime'], inplace=True)
        df_base.set_index('Datetime', inplace=True)
        
        # Add time-based features
        df_base['hour'] = df_base.index.hour
        df_base['dayofweek'] = df_base.index.dayofweek
        
        # Add rolling features consistent with training
        df_base['PM2.5_roll3'] = df_base['PM2.5'].rolling(9, min_periods=1).mean()
        df_base['Temp_roll3'] = df_base['Temp'].rolling(9, min_periods=1).mean()
        df_base['PM10_roll3'] = df_base['PM10'].rolling(9, min_periods=1).mean()
        df_base['CO_roll_3_day'] = df_base['CO'].rolling(3, min_periods=1).mean()

        # Drop any rows with NaN values that were created by rolling means
        df_base.dropna(inplace=True)

        # Create lagged data (up to 3 lags for the CO model)
        df_lagged = create_lagged_data(df_base, lags=3)
        if df_lagged.empty:
            print("Error: Not enough data in CPCB.csv to create starting features after processing.")
            return None
            
        return df_lagged.iloc[[-1]]

    except FileNotFoundError:
        print("Error: 'data/CPCB.csv' not found. Please make sure the file exists.")
        return None
    except Exception as e:
        print(f"An error occurred while preparing data: {e}")
        return None

def run_command_line_forecast(metric):
    """
    Main function to run the 7-day forecast for the specified metric and print the results.
    
    Args:
        metric (str): The air quality metric to forecast ('PM2.5', 'PM10', or 'CO').
    """
    print(f"Initializing command-line forecast for {metric}...")
    
    # 1. Get the starting data point
    last_known_data = get_last_known_data()
    if last_known_data is None:
        return  # Stop if data couldn't be loaded

    current_input = last_known_data.copy()
    future_forecasts = []
    
    # Get the last known date to increment from
    last_date = pd.to_datetime(current_input.index[0])

    print("Generating 7-day forecast recursively...")
    # 2. Loop for 7 days, feeding predictions back as inputs
    for day in range(7):
        pred = None
        
        # Choose the correct prediction function based on the user's choice
        if metric == 'PM2.5':
            pred = predict_pm25(current_input)
        elif metric == 'PM10':
            pred = predict_pm10(current_input)
        elif metric == 'CO':
            pred = predict_co(current_input)
        
        if pred is None:
            print(f"Error: Could not predict for metric '{metric}'.")
            return

        # Store the rounded prediction
        forecast = {
            'Day': day + 1,
            metric: round(pred, 2)
        }
        future_forecasts.append(forecast)
        print(f"  - Day {day+1} forecast for {metric} complete.")

        # --- UPDATE INPUT FOR THE NEXT LOOP ITERATION ---
        
        # 1. Shift all existing lag features down by one.
        all_feature_cols = [col for col in last_known_data.columns if '_lag' not in col]
        for col_name in all_feature_cols:
            for i in range(3, 1, -1): # Shift lag3 -> lag2, lag2 -> lag1
                if f'{col_name}_lag{i}' in current_input.columns and f'{col_name}_lag{i-1}' in current_input.columns:
                    current_input[f'{col_name}_lag{i}'] = current_input[f'{col_name}_lag{i-1}']

        # 2. Update lag1 with the new prediction for the chosen metric
        current_input[f'{metric}_lag1'] = pred
        # Note: Weather lags are shifted but not updated with new forecasts.

        # 3. Increment time features for the next day
        next_day_datetime = last_date + pd.Timedelta(days=day + 1)
        current_input['hour'] = next_day_datetime.hour
        current_input['dayofweek'] = next_day_datetime.dayofweek
    
    # 3. Display the final result in a clean table
    forecast_df = pd.DataFrame(future_forecasts)
    print("\n" + "="*50)
    print(f"✅ Final 7-Day {metric} Air Quality Forecast:")
    print("="*50)
    print(forecast_df.to_string())

def main():
    """
    Prompts the user for a metric and starts the forecast.
    """
    while True:
        print("\nSelect the air quality metric you want to forecast:")
        print("1. PM2.5")
        print("2. PM10")
        print("3. CO")
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == '1':
            run_command_line_forecast('PM2.5')
            break
        elif choice == '2':
            run_command_line_forecast('PM10')
            break
        elif choice == '3':
            run_command_line_forecast('CO')
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main()
