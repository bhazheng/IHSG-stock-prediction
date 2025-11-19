import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# --- IMPORTS ---
from data_loader import download_data
from feature_engineering import preprocess_data

app = Flask(__name__)

# --- CONFIGURATION ---
TICKER = "^JKSE"
MODEL_DIR = "output/model"
START_DATE_DATA = "2023-01-01" # Fetch enough history for feature engineering

# Global Model Container
models = {}

def load_all_models():
    """Loads all models into memory on startup."""
    print("⏳ Loading models...")
    global models
    
    # 1. Load XGBoost
    try:
        with open(os.path.join(MODEL_DIR, "XGBoost_IHSG.pkl"), "rb") as f:
            models['xgboost'] = pickle.load(f)
        print("✅ XGBoost loaded.")
    except Exception as e:
        print(f"❌ Error loading XGBoost: {e}")

    # 2. Load Prophet
    try:
        with open(os.path.join(MODEL_DIR, "Prophet_IHSG.pkl"), "rb") as f:
            models['prophet'] = pickle.load(f)
        print("✅ Prophet loaded.")
    except Exception as e:
        print(f"❌ Error loading Prophet: {e}")

    # 3. Load LSTM
    try:
        models['lstm'] = load_model(os.path.join(MODEL_DIR, "LSTM_IHSG.keras"))
        print("✅ LSTM loaded.")
    except Exception as e:
        print(f"❌ Error loading LSTM: {e}")

# Load on startup
load_all_models()

@app.route('/')
def home():
    return jsonify({
        "message": "IHSG Prediction API is Running",
        "usage": "/predict?model=[xgboost|prophet|lstm]"
    })

@app.route('/predict', methods=['GET'])
def predict():
    model_type = request.args.get('model', default='xgboost').lower()
    
    # 1. Download Latest Data
    today = datetime.today().strftime('%Y-%m-%d')
    data = download_data(TICKER, START_DATE_DATA, today)
    
    if data is None or data.empty:
        return jsonify({"error": "Failed to download latest data"}), 500

    last_date = data.index[-1].strftime('%Y-%m-%d')
    last_price = float(data['Close'].iloc[-1])

    try:
        result = {}
        
        # === XGBOOST ===
        if model_type == 'xgboost':
            if 'xgboost' not in models: return jsonify({"error": "XGBoost model not available"}), 500
            
            # Feature Engineering
            df_processed = preprocess_data(data) 
            
            if df_processed.empty:
                return jsonify({"error": "Not enough data for feature engineering"}), 400

            features = ['RSI', 'MACD', 'MACD_Signal', 'Volume', 'High', 'Low', 'BB_Upper', 'BB_Lower']
            X_latest = df_processed[features].iloc[[-1]] # Get last row

            preds = {}
            xgb_models = models['xgboost']
            
            # 1. UpDown
            preds['direction'] = "UP" if xgb_models['Target_UpDown'].predict(X_latest)[0] == 1 else "DOWN"
            
            # 2. Daily Return
            pred_return = xgb_models['Target_DailyReturn'].predict(X_latest)[0]
            preds['predicted_return_pct'] = float(pred_return * 100)
            
            # 3. Volatility
            pred_vol = xgb_models['Target_Volatility'].predict(X_latest)[0]
            preds['predicted_volatility'] = float(pred_vol)

            result = {
                "model": "XGBoost",
                "last_date": last_date,
                "last_price": last_price,
                "prediction": preds
            }

        # === PROPHET ===
        elif model_type == 'prophet':
            if 'prophet' not in models: return jsonify({"error": "Prophet model not available"}), 500
            
            model = models['prophet']
            
            # Forecast 1 day ahead
            future = model.make_future_dataframe(periods=1)
            forecast = model.predict(future)
            
            next_day = forecast.iloc[-1]
            
            result = {
                "model": "Prophet",
                "last_date": last_date,
                "prediction_date": next_day['ds'].strftime('%Y-%m-%d'),
                "predicted_price": float(next_day['yhat']),
                "lower_bound": float(next_day['yhat_lower']),
                "upper_bound": float(next_day['yhat_upper'])
            }

        # === LSTM ===
        elif model_type == 'lstm':
            if 'lstm' not in models: return jsonify({"error": "LSTM model not available"}), 500
            
            TIME_STEP = 60
            dataset = data.filter(['Close']).values
            
            if len(dataset) < TIME_STEP:
                 return jsonify({"error": f"Not enough data (< {TIME_STEP} days)"}), 400

            # Fit Scaler (on latest data)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
            
            # Get last 60 days
            last_60_days = scaled_data[-TIME_STEP:]
            X_input = last_60_days.reshape(1, TIME_STEP, 1)
            
            # Predict
            pred_scaled = models['lstm'].predict(X_input)
            pred_inv = scaler.inverse_transform(pred_scaled)
            
            result = {
                "model": "LSTM",
                "last_date": last_date,
                "last_price": last_price,
                "predicted_price": float(pred_inv[0][0])
            }

        else:
            return jsonify({"error": "Unknown model. Choose: xgboost, prophet, or lstm"}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)