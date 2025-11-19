import pandas as pd
import numpy as np

def preprocess_data(df):
    df = df.copy()
    # Indikator
    window = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)

    # Target
    df['Target_UpDown'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Target_DailyReturn'] = df['Close'].pct_change().shift(-1)
    df['Target_Volatility'] = df['Close'].pct_change().rolling(window=30).std().shift(-1)
    
    df.dropna(inplace=True)
    return df