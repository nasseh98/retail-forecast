import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# -------------------------
# 1. Load Data
# -------------------------
DATA_PATH = "retail_sales.csv"

print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("✅ Data loaded:", df.shape)
print(df.head())

# -------------------------
# 2. Feature Engineering
# -------------------------
# Convert date column
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# Drop non-numeric/unused columns
X = df.drop(columns=["sales", "date"])
y = df["sales"]

# -------------------------
# 3. Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 4. Train Model (XGBoost)
# -------------------------
print("🛠 Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# 5. Evaluate Model
# -------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"📊 RMSE on test set: {rmse:.2f}")

# -------------------------
# 6. Save Model
# -------------------------
joblib.dump(model, "demand_forecast_model.pkl")
print("✅ Model saved as demand_forecast_model.pkl")
