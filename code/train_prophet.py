import pandas as pd
from prophet import Prophet
import pickle
import os
from datetime import datetime
from data_loader import download_data

TICKER = "^JKSE"
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

os.makedirs("output/model", exist_ok=True)

def main():
    print("--- TRAIN PROPHET ---")
    data = download_data(TICKER, START_DATE, END_DATE)
    if data is None: return

    df_p = data.reset_index()[['Date', 'Close']]
    df_p.columns = ['ds', 'y']
    df_p['ds'] = df_p['ds'].dt.tz_localize(None)

    model = Prophet(daily_seasonality=True)
    model.fit(df_p)

    with open("output/model/Prophet_IHSG.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model Saved -> output/model/Prophet_IHSG.pkl")

if __name__ == '__main__':
    main()