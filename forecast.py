import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATA_PATH = "data/retail_sales.csv"   # needs columns: date, sales (optional: promo)
WINDOW = 28
TEST_DAYS = 90
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


def make_fake_data():
    """Used only if data/retail_sales.csv is missing, so the project still runs."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    t = np.arange(len(dates))
    dow = dates.dayofweek.values
    promo = (rng.random(len(dates)) < 0.12).astype(int)
    sales = (
        1000 + 0.6 * t
        + np.array([0, 20, 30, 50, 120, 260, 200])[dow]
        + 140 * np.sin(2 * np.pi * dates.dayofyear.values / 365.25)
        + 250 * promo
        + rng.normal(0, 40, len(dates))
    )
    return pd.DataFrame({"date": dates, "sales": sales.round(2), "promo": promo})


def load_data():
    if os.path.exists(DATA_PATH):
        print("Loaded", DATA_PATH)
        raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
        if "promo" not in raw.columns:
            raw["promo"] = 0
        df = raw.groupby("date").agg(sales=("sales", "sum"), promo=("promo", "max")).reset_index()
    else:
        print("No data/retail_sales.csv found, using generated data")
        df = make_fake_data()

    df = df.sort_values("date").set_index("date").asfreq("D")
    df["sales"] = df["sales"].interpolate()
    df["promo"] = df["promo"].fillna(0)
    df = df.reset_index()

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["mon_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["mon_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df):
    d = df.copy()
    for lag in (1, 7, 14, 28):
        d[f"lag_{lag}"] = d["sales"].shift(lag)
    d["roll_7"] = d["sales"].shift(1).rolling(7).mean()
    d["roll_28"] = d["sales"].shift(1).rolling(28).mean()
    return d.dropna().reset_index(drop=True)


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) * 100
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE_%": round(mape, 2)}


# ---------------- XGBoost ----------------
XGB_FEATURES = ["promo", "dow", "month", "is_weekend",
                "lag_1", "lag_7", "lag_14", "lag_28", "roll_7", "roll_28"]


def run_xgboost(df):
    feat = add_lag_features(df)
    train, test = feat.iloc[:-TEST_DAYS], feat.iloc[-TEST_DAYS:]
    model = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
    )
    model.fit(train[XGB_FEATURES], train["sales"])
    preds = model.predict(test[XGB_FEATURES])
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_sales.joblib")
    return test["date"].values, test["sales"].values, preds


# ---------------- LSTM (deep learning) ----------------
class SalesLSTM(nn.Module):
    def __init__(self, n_feat, n_cov, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_cov, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, seq, cov):
        out, _ = self.lstm(seq)
        h = out[:, -1]
        return self.head(torch.cat([h, cov], dim=1)).squeeze(1)


def run_lstm(df, epochs=60, batch_size=64):
    n = len(df)
    split = n - TEST_DAYS

    scaler = StandardScaler()
    scaler.fit(df[["sales"]].iloc[:split])
    sales_scaled = scaler.transform(df[["sales"]]).ravel()

    cov_cols = ["promo", "dow_sin", "dow_cos", "mon_sin", "mon_cos"]
    cov = df[cov_cols].values.astype("float32")
    arr = np.column_stack([sales_scaled, cov]).astype("float32")  # (n, 6)

    X_seq, X_cov, y, idx = [], [], [], []
    for i in range(WINDOW, n):
        X_seq.append(arr[i - WINDOW:i])
        X_cov.append(cov[i])
        y.append(sales_scaled[i])
        idx.append(i)
    X_seq = torch.tensor(np.array(X_seq))
    X_cov = torch.tensor(np.array(X_cov))
    y = torch.tensor(np.array(y), dtype=torch.float32)
    idx = np.array(idx)

    train_mask = idx < split
    test_mask = idx >= split

    tr_seq, tr_cov, tr_y = X_seq[train_mask], X_cov[train_mask], y[train_mask]
    te_seq, te_cov = X_seq[test_mask], X_cov[test_mask]

    # last 10% of train = validation (for picking the best epoch)
    v = int(len(tr_y) * 0.9)
    va_seq, va_cov, va_y = tr_seq[v:], tr_cov[v:], tr_y[v:]
    tr_seq, tr_cov, tr_y = tr_seq[:v], tr_cov[:v], tr_y[:v]

    model = SalesLSTM(n_feat=arr.shape[1], n_cov=cov.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_y))
        for start in range(0, len(perm), batch_size):
            b = perm[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(tr_seq[b], tr_cov[b]), tr_y[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(va_seq, va_cov), va_y).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"LSTM epoch {epoch + 1}/{epochs}  val loss {val_loss:.4f}")

    model.load_state_dict(best_state)
    os.makedirs("models", exist_ok=True)
    torch.save(best_state, "models/lstm_sales.pt")

    model.eval()
    with torch.no_grad():
        preds_scaled = model(te_seq, te_cov).numpy()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    return preds


def main():
    df = load_data()

    # XGBoost
    dates, y_true, xgb_pred = run_xgboost(df)

    # Seasonal naive baseline: same weekday last week
    naive_pred = df["sales"].shift(7).iloc[-TEST_DAYS:].values

    # LSTM (same last TEST_DAYS dates)
    lstm_pred = run_lstm(df)

    results = pd.DataFrame({
        "SeasonalNaive": metrics(y_true, naive_pred),
        "XGBoost": metrics(y_true, xgb_pred),
        "LSTM": metrics(y_true, lstm_pred),
    }).T
    print("\n=== Test results (last", TEST_DAYS, "days) ===")
    print(results)

    os.makedirs("results", exist_ok=True)
    results.to_csv("results/metrics.csv")

    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual", linewidth=2)
    plt.plot(dates, naive_pred, label="Seasonal naive", alpha=0.6)
    plt.plot(dates, xgb_pred, label="XGBoost", alpha=0.8)
    plt.plot(dates, lstm_pred, label="LSTM", alpha=0.8)
    plt.title("Retail sales forecast vs actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/forecast.png", dpi=150)
    print("Saved results/metrics.csv and results/forecast.png")


if __name__ == "__main__":
    main()
