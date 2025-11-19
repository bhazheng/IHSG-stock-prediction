import yfinance as yf
import pandas as pd
import logging

def download_data(ticker, start_date, end_date):
    try:
        print(f"⬇️ Downloading data for {ticker} ({start_date} to {end_date})...")
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(-1)
        return data
    except Exception as e:
        logging.error(f"Error downloading data: {e}")
        return None