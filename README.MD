# IHSG Stock Prediction: Comprehensive Comparison of Time Series Models

This repository implements and compares multiple time series forecasting approaches to predict the Jakarta Composite Index (IHSG / ^JKSE). It is intended as an educational portfolio showcasing deep learning (LSTM), statistical (Prophet), and ensemble (XGBoost) approaches for financial forecasting.

## 1. Problem Description

**Objective:** Predict future trends and price movements of the IHSG index using historical data. The project answers questions such as: "Will the market go up or down tomorrow?" and "What is the expected volatility?".

**Context:** Financial market prediction is difficult due to high volatility and many external drivers. This project compares different algorithmic philosophies:

- **Trend Following (Prophet):** Can we capture long-term growth and structural changes?
- **Sequence Learning (LSTM):** Can we learn short- to medium-term sequential patterns?
- **Feature-Based (XGBoost):** Can technical indicators (RSI, MACD, Bollinger Bands) predict short-term movements?

**Solution Overview:** The codebase provides a modular pipeline that:

- Downloads and cleans historical data from Yahoo Finance.
- Performs extensive Exploratory Data Analysis (EDA).
- Engineers technical indicators (RSI, MACD, Bollinger Bands).
- Trains and tunes multiple models (LSTM, Prophet, XGBoost).
- Exposes a Flask API to serve predictions.

## 2. Exploratory Data Analysis (EDA)

Extensive EDA was performed to understand IHSG behavior. Key analyses include:

- Trend analysis and notable market events (e.g., 2020 pandemic crash).
- Volatility analysis via daily returns.
- Seasonality checks and decomposition.
- Correlation between engineered technical indicators and price movements.

See `notebooks/notebook.ipynb` for full EDA and visualizations.

## 3. Models and Training

We trained and compared three modeling paradigms. Hyperparameters were tuned using appropriate validation strategies.

### A. Long Short-Term Memory (LSTM)

- **Type:** Recurrent Neural Network (Deep Learning)
- **Configuration:** Lookback window = 60 days; 2 LSTM layers with 50 units each; Dropout = 0.2
- **Goal:** Capture non-linear sequential dependencies in price history.

### B. Facebook Prophet

- **Type:** Additive regression model
- **Configuration:** Daily seasonality enabled; tuned `changepoint_prior_scale` for trend flexibility
- **Goal:** Model strong seasonal effects and robust trend forecasting.

### C. XGBoost

- **Type:** Gradient-boosted decision trees (ensemble)
- **Feature Engineering:** RSI, MACD (and MACD signal), Bollinger Bands, and other technical features
- **Targets:**
  - Classification: `Target_UpDown` (will price close higher tomorrow?)
  - Regression: `Target_DailyReturn` (predicted percent change) and `Target_Volatility`
- **Tuning:** Grid search with `TimeSeriesSplit` for `learning_rate`, `max_depth`, `n_estimators`.

## 4. Project Structure

Key files and directories:

- `app/main.py` — Flask API for serving model predictions.
- `code/data_loader.py` — Data download and cleaning utilities.
- `code/feature_engineering.py` — Technical indicator calculations.
- `code/train_LSTM.py`, `code/train_prophet.py`, `code/train_xgboost.py` — Training scripts that save models into `output/model/`.
- `code/predict_LSTM.py`, `code/predict_prophet.py`, `code/predict_xgboost.py` — Prediction/evaluation scripts.
- `notebooks/notebook.ipynb` — EDA and exploratory analysis.
- `output/` — Generated models and prediction outputs.

## 5. Reproducibility / Quick Start

Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

Create and activate a virtual environment, then install dependencies:

```bash
# Create env
python -m venv venv

# Activate env (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Run training scripts (this will save models to `output/model/`):

```bash
python code/train_LSTM.py
python code/train_prophet.py
python code/train_xgboost.py
```

Run prediction/evaluation scripts (outputs saved to `output/prediction/`):

```bash
python code/predict_LSTM.py
python code/predict_prophet.py
python code/predict_xgboost.py
```

## 6. API / Model Deployment

The repository includes a Flask app at `app/main.py`.

Endpoints:

- `GET /` — API status
- `GET /predict?model=[xgboost|prophet|lstm]` — Returns a prediction from the specified model

Example request:

```
http://localhost:5000/predict?model=xgboost
```

Example JSON response:

```json
{
  "model": "XGBoost",
  "last_date": "2023-10-27",
  "prediction": {
    "direction": "UP",
    "predicted_return_pct": 0.45
  }
}
```

## 7. Dependencies

All main dependencies are listed in `requirements.txt`.

Example `requirements.txt` contents:

- pandas
- numpy
- yfinance
- scikit-learn
- tensorflow
- xgboost
- prophet
- flask
- matplotlib
- seaborn

Install them via `pip install -r requirements.txt` as shown in Quick Start.

## 8. Containerization

A `Dockerfile` is provided to containerize the API service.

Build the image:

```bash
docker build -t ihsg-predictor .
```

Run the container:

```bash
docker run -p 5000:5000 ihsg-predictor
```

## 9. Cloud Deployment

The app is ready for deployment on platforms such as Render or Google Cloud Run. Example (Render):

1. Push the repository to GitHub.
2. Create a new "Web Service" in Render and connect your repository.
3. Choose Docker as the environment and deploy.

Test the live API (replace with your service URL):

```bash
curl "https://your-app-url.com/predict?model=lstm"
```

## Notes & Next Steps

- Add README badges (build, license) if desired.
- Consider adding a `CONTRIBUTING.md` and license file.

If you'd like, I can also:

- Add quick usage examples for each model's output plots.
- Create a minimal Docker Compose for local development.

---

_If you want any wording changes, extra badges, or a CONTRIBUTING section, tell me what to include._
