import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from datetime import datetime
import numpy as np
from data_loader import download_data
from feature_engineering import preprocess_data

TICKER = "^JKSE"
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

os.makedirs("output/prediction", exist_ok=True)

def main():
    print("--- PREDICT & EVALUATE XGBOOST ---")
    
    # 1. Load
    data = download_data(TICKER, START_DATE, END_DATE)
    with open("output/model/XGBoost_IHSG.pkl", "rb") as f:
        models = pickle.load(f)
    
    # 2. Preprocess & Split
    df = preprocess_data(data)
    features = ['RSI', 'MACD', 'MACD_Signal', 'Volume', 'High', 'Low', 'BB_Upper', 'BB_Lower']
    
    # Ambil 20% data terakhir sebagai Test Set
    split_idx = int(len(df) * 0.8)
    X_test = df[features].iloc[split_idx:]
    
    # 3. Evaluate Loop
    targets = [('Target_UpDown', 'classifier'), 
               ('Target_DailyReturn', 'regressor'), 
               ('Target_Volatility', 'regressor')]

    for target, mode in targets:
        y_test = df[target].iloc[split_idx:]
        model = models[target]
        preds = model.predict(X_test)

        print(f"\nEvaluasi {target}:")
        
        if mode == 'classifier':
            acc = accuracy_score(y_test, preds)
            print(f"Accuracy: {acc:.4f}")
            
            plt.figure(figsize=(5,4))
            sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
            plt.title(f'CM - {target} (Acc: {acc:.2f})')
            plt.savefig(f"output/prediction/XGBoost_{target}_CM.png")
            plt.close()
            
        else:
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            print(f"RMSE: {rmse:.4f}")
            
            plt.figure(figsize=(10,4))
            plt.plot(y_test.values, label='Actual', alpha=0.7)
            plt.plot(preds, label='Predicted', alpha=0.7)
            plt.title(f'Pred vs Act - {target} (RMSE: {rmse:.4f})')
            plt.legend()
            plt.savefig(f"output/prediction/XGBoost_{target}_Plot.png")
            plt.close()

    print("\nAll evaluations saved to output/prediction/")

if __name__ == '__main__':
    main()