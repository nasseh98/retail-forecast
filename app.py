from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import datetime
import plotly.express as px
import plotly.io as pio
import os

# Load trained model & dataset
model = joblib.load("demand_forecast_model.pkl")
df = pd.read_csv("retail_sales.csv")
df["date"] = pd.to_datetime(df["date"])

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        store_id = int(request.form["store_id"])
        item_id = int(request.form["item_id"])
        price = float(request.form["price"])
        promotion = int(request.form["promotion"])
        holiday = int(request.form["holiday"])
        date_str = request.form["date"]

        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        month = date.month
        year = date.year
        day_of_week = date.weekday()

        input_df = pd.DataFrame([{
            "store_id": store_id,
            "item_id": item_id,
            "price": price,
            "promotion": promotion,
            "holiday": holiday,
            "day_of_week": day_of_week,
            "month": month,
            "year": year
        }])

        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template("index.html",
                               prediction_text=f"Predicted Sales: {prediction} units")

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"❌ Error: {str(e)}")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    try:
        store_ids = sorted(df["store_id"].unique())

        selected_store = int(request.form.get("store_id", store_ids[0]))
        item_ids = sorted(df[df["store_id"] == selected_store]["item_id"].unique())
        selected_item = int(request.form.get("item_id", item_ids[0]))

        # Filtered dataset
        sales_data = df[(df["store_id"] == selected_store) &
                        (df["item_id"] == selected_item)]
        sales_trend = sales_data.groupby("date")["sales"].sum().reset_index()

        # Chart 1: Item trend
        fig1 = px.line(sales_trend, x="date", y="sales",
                       title=f"Sales Trend - Store {selected_store}, Item {selected_item}",
                       labels={"sales": "Units Sold", "date": "Date"},
                       template="plotly_white")
        chart_item = pio.to_html(fig1, full_html=False)

        # Chart 2: Store breakdown
        store_data = df[df["store_id"] == selected_store]
        store_trend = store_data.groupby(["date", "item_id"])["sales"].sum().reset_index()

        fig2 = px.area(store_trend, x="date", y="sales", color="item_id",
                       title=f"Total Sales Breakdown by Item - Store {selected_store}",
                       labels={"sales": "Units Sold", "date": "Date", "item_id": "Item"},
                       template="plotly_white")
        chart_store = pio.to_html(fig2, full_html=False)

        return render_template("dashboard.html",
                               chart_item=chart_item,
                               chart_store=chart_store,
                               store_ids=store_ids,
                               item_ids=item_ids,
                               selected_store=selected_store,
                               selected_item=selected_item)

    except Exception as e:
        return f"❌ Error generating dashboard: {str(e)}"

@app.route("/download_csv/<int:store_id>/<int:item_id>")
def download_csv(store_id, item_id):
    try:
        # Filter dataset
        export_data = df[(df["store_id"] == store_id) &
                         (df["item_id"] == item_id)]

        # Save temporary file
        file_path = f"export_store{store_id}_item{item_id}.csv"
        export_data.to_csv(file_path, index=False)

        return send_file(file_path, as_attachment=True)

    except Exception as e:
        return f"❌ Error exporting CSV: {str(e)}"
    finally:
        # Clean up temp file after sending
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

if __name__ == "__main__":
    app.run(debug=True)
