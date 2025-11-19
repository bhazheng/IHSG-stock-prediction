import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
from data_loader import download_data

TICKER = "^JKSE"
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

os.makedirs("output/prediction", exist_ok=True)

def main():
    print("--- PREDICT & EVALUATE PROPHET ---")
    
    # 1. Load Data & Model
    data = download_data(TICKER, START_DATE, END_DATE)
    with open("output/model/Prophet_IHSG.pkl", "rb") as f:
        model = pickle.load(f)

    # 2. Prepare Actuals
    df_actual = data.reset_index()[['Date', 'Close']]
    df_actual.columns = ['ds', 'y']
    df_actual['ds'] = df_actual['ds'].dt.tz_localize(None)

    # 3. Predict (on history to compare)
    future = model.make_future_dataframe(periods=0) 
    forecast = model.predict(future)

    # 4. Evaluate
    # Merge to ensure alignment
    cmp = pd.merge(df_actual, forecast[['ds', 'yhat']], on='ds')
    rmse = np.sqrt(mean_squared_error(cmp['y'], cmp['yhat']))
    print(f"RMSE: {rmse:.2f}")

    # 5. Plot
    plt.figure(figsize=(12,6))
    plt.plot(cmp['ds'], cmp['y'], label='Actual', color='blue')
    plt.plot(cmp['ds'], cmp['yhat'], label='Predicted', color='orange', alpha=0.7)
    plt.title(f'Prophet Evaluation (RMSE: {rmse:.2f})')
    plt.legend()
    plt.savefig("output/prediction/Prophet_Evaluation.png")
    print("Plot Saved -> output/prediction/Prophet_Evaluation.png")

if __name__ == "__main__":
    main()