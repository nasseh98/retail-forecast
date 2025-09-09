import pandas as pd
import numpy as np

# Parameters
n_stores = 5
n_items = 20
n_days = 730  # ~2 years

# Date range
dates = pd.date_range(start="2021-01-01", periods=n_days, freq="D")

# Generate dataset
data = []
for store in range(1, n_stores + 1):
    for item in range(1, n_items + 1):
        base_sales = np.random.randint(20, 200)  # baseline demand
        price = np.random.uniform(5, 50)  # constant price per item
        for date in dates:
            day_of_week = date.dayofweek
            is_weekend = 1 if day_of_week >= 5 else 0
            is_holiday = 1 if np.random.rand() < 0.05 else 0
            promotion = 1 if np.random.rand() < 0.1 else 0

            # Sales = base + noise + effects
            sales = base_sales \
                    + (promotion * np.random.randint(10, 50)) \
                    + (is_weekend * np.random.randint(5, 20)) \
                    + (is_holiday * np.random.randint(20, 80)) \
                    + np.random.normal(0, 10)

            sales = max(0, int(sales))  # no negative sales

            data.append([date, store, item, price, promotion, is_holiday, day_of_week, sales])

# Create DataFrame
df = pd.DataFrame(data, columns=["date", "store_id", "item_id", "price", 
                                 "promotion", "holiday", "day_of_week", "sales"])

# Save to CSV
df.to_csv("retail_sales.csv", index=False)

print("✅ Synthetic retail_sales.csv generated with shape:", df.shape)
