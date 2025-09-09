import numpy as np
import pandas as pd

# ---------- Feature engineering ----------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dayofweek"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df

def add_lag_features(df: pd.DataFrame, lags=(7, 28), roll_windows=(7, 28)) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["store", "item", "date"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store", "item"])["sales"].shift(lag)

    for w in roll_windows:
        df[f"rmean_{w}"] = (
            df.groupby(["store", "item"])["sales"]
              .shift(1)                         # avoid leakage
              .rolling(window=w, min_periods=1)
              .mean()
              .reset_index(level=[0,1], drop=True)
        )
    return df

def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df)
    # drop rows where lags are NaN (the initial periods)
    lag_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("rmean_")]
    df = df.dropna(subset=lag_cols)
    return df

# ---------- Recursive forecasting for one (store,item) ----------
def make_future_df(history_si: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    history_si: history for one store-item with columns: date, store, item, sales
    Returns a dataframe of future dates with placeholders for features.
    """
    history_si = history_si.sort_values("date").copy()
    last_date = history_si["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.DataFrame({
        "date": future_dates,
        "store": history_si["store"].iloc[0],
        "item": history_si["item"].iloc[0],
        "sales": np.nan,
    })
    return future

def recursive_forecast(model, meta: dict, history_si: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    model: trained sklearn pipeline (with OneHotEncoder inside meta or model)
    meta: dict with "feature_cols" and categories
    history_si: must include all columns required to compute lags
    """
    # Work on a copy
    df_hist = history_si.copy()
    forecasts = []

    for _ in range(horizon):
        # Build one-step feature row using current history
        tmp = add_time_features(df_hist.tail(60))  # smaller slice is enough
        tmp = add_lag_features(tmp)
        row = tmp.tail(1).copy()  # last available row (yesterday)
        row.loc[:, "date"] = row["date"] + pd.Timedelta(days=1)  # predict next day

        # recompute features for that next day
        row = add_time_features(row)
        # lags/rolling for the next day use existing df_hist (the true history)
        # We'll rebuild a temp frame with the new date and use last known lags
        # Create a single-row "next" frame using latest known values
        next_row = {
            "date": row["date"].iloc[0],
            "store": df_hist["store"].iloc[-1],
            "item": df_hist["item"].iloc[-1],
            # sales unknown
        }
        sim = pd.concat([df_hist, pd.DataFrame([next_row])], ignore_index=True)
        sim = add_time_features(sim)
        sim = add_lag_features(sim)

        feat = sim.tail(1).copy()
        # Prepare features
        X = feat[meta["feature_cols"]]
        y_pred = model.predict(X)[0]

        # Append prediction to history for next iteration
        sim.loc[sim.index[-1], "sales"] = max(0.0, float(y_pred))  # no negatives
        df_hist = sim.copy()
        forecasts.append({"date": sim["date"].iloc[-1], "forecast": float(y_pred)})

    return pd.DataFrame(forecasts)
