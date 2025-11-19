import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import pickle
import os
from datetime import datetime
from data_loader import download_data
from feature_engineering import preprocess_data

TICKER = "^JKSE"
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

os.makedirs("output/model", exist_ok=True)

def tune_and_train(X, y, mode):
    tscv = TimeSeriesSplit(n_splits=3)
    if mode == 'classifier':
        model = xgb.XGBClassifier(eval_metric='logloss', n_jobs=-1)
        scoring = 'accuracy'
    else:
        model = xgb.XGBRegressor(n_jobs=-1)
        scoring = 'neg_root_mean_squared_error'

    params = {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3, 5]}
    gs = GridSearchCV(model, params, cv=tscv, scoring=scoring, n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_

def main():
    print("--- TRAIN XGBOOST ---")
    data = download_data(TICKER, START_DATE, END_DATE)
    if data is None: return
    
    df = preprocess_data(data)
    features = ['RSI', 'MACD', 'MACD_Signal', 'Volume', 'High', 'Low', 'BB_Upper', 'BB_Lower']
    X = df[features]
    
    models = {}
    targets = [('Target_UpDown', 'classifier'), 
               ('Target_DailyReturn', 'regressor'), 
               ('Target_Volatility', 'regressor')]

    for target, mode in targets:
        print(f"Training {target}...")
        models[target] = tune_and_train(X, df[target], mode)

    with open("output/model/XGBoost_IHSG.pkl", "wb") as f:
        pickle.dump(models, f)
    print("Models Saved -> output/model/XGBoost_IHSG.pkl")

if __name__ == '__main__':
    main()