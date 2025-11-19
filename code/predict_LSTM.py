import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime
import os
from data_loader import download_data

TICKER = "^JKSE"
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')
TIME_STEP = 60

os.makedirs("output/prediction", exist_ok=True)

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def main():
    print("--- PREDICT & EVALUATE LSTM ---")
    
    # 1. Load Data & Model
    data = download_data(TICKER, START_DATE, END_DATE)
    model = load_model("output/model/LSTM_IHSG.keras")
    
    # 2. Preprocess
    dataset = data.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # 3. Split Test Data (Last 20%)
    test_len = int(len(dataset) * 0.2)
    test_data = scaled_data[-(test_len + TIME_STEP):] # Ambil buffer time_step
    
    X_test, y_test = create_dataset(test_data, TIME_STEP)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # 4. Predict
    preds = model.predict(X_test)
    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 5. Evaluate
    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
    print(f"RMSE: {rmse:.2f}")

    # 6. Plot
    plt.figure(figsize=(12,6))
    plt.plot(y_test_inv, label='Actual Price', color='blue')
    plt.plot(preds_inv, label='Predicted Price', color='red')
    plt.title(f'LSTM Evaluation (RMSE: {rmse:.2f})')
    plt.legend()
    plt.savefig("output/prediction/LSTM_Evaluation.png")
    print("Plot Saved -> output/prediction/LSTM_Evaluation.png")

if __name__ == '__main__':
    main()