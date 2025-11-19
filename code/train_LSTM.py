import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import os
from data_loader import download_data

TICKER = "^JKSE"
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')
TIME_STEP = 60

os.makedirs("output/model", exist_ok=True)

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    print("--- TRAIN LSTM ---")
    data = download_data(TICKER, START_DATE, END_DATE)
    if data is None: return

    dataset = data.filter(['Close']).values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    X, y = create_dataset(scaled_data, TIME_STEP)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_model((X.shape[1], 1))
    model.fit(X, y, batch_size=32, epochs=20, verbose=1)

    model.save("output/model/LSTM_IHSG.keras")
    print("Model Saved -> output/model/LSTM_IHSG.keras")

if __name__ == '__main__':
    main()