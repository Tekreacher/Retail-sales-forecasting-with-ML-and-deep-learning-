# Retail Sales Forecasting (XGBoost + LSTM)

Forecasts daily retail sales and compares a seasonal-naive baseline, XGBoost (lag/rolling/calendar features), and a PyTorch LSTM. Evaluated on a time-based holdout (last 90 days) with MAE, RMSE, and MAPE.

## Setup
pip install -r requirements.txt

## Data
Put your data at `data/retail_sales.csv` with columns `date`, `sales` (optional `promo`). Multiple stores are summed per day. If the file is missing, generated sample data is used.

## Run
python forecast.py

Outputs `results/metrics.csv` and `results/forecast.png`.

## Results
(paste your real metrics table here after running)

## Notes
- Time-based split, no shuffling, so there is no data leakage.
- Scaler is fit on training data only.
- XGBoost and LSTM do one-step-ahead forecasting using past actual values.
