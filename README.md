# 🛒 Retail Demand Forecasting

A Flask web app that predicts and visualizes retail sales using **Machine Learning (XGBoost + Scikit-learn)**.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20ScikitLearn-green)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Features
- ✨ Animated landing homepage
- 🔮 Demand prediction form (store, item, price, promo, holiday, date)
- 📊 Interactive dashboard (filters, charts, breakdowns)
- ⬇ Export filtered sales data as CSV
- Trained **ML model** for demand forecasting

---

## 📸 Screenshots

### 🏠 Home
![Home Page](screenshots/home.png)

### 🔮 Prediction Form
![Prediction](screenshots/prediction.png)

### 📊 Dashboard
![Dashboard](screenshots/dashboard.png)

---

## ⚡ Tech Stack
- **Backend**: Flask, Pandas, Scikit-learn, XGBoost
- **Frontend**: HTML, CSS, Bootstrap 5, Plotly
- **Environment**: Python 3.10+

---

## ▶️ Run Locally

```bash
# Clone repo
git clone https://github.com/nasseh98/retail-forecast.git
cd retail-forecast

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Run app
python app.py


