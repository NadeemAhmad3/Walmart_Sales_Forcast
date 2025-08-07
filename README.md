# 📈 SalesForecaster Pro: A Walmart Sales Prediction Tool

![python-shield](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![pandas-shield](https://img.shields.io/badge/pandas-2.2-blue)
![sklearn-shield](https://img.shields.io/badge/scikit--learn-1.6-blue)
![xgboost-shield](https://img.shields.io/badge/XGBoost-2.1-blue)
![streamlit-shield](https://img.shields.io/badge/Streamlit-1.3-ff69b4)

A **sophisticated machine learning project** that forecasts weekly sales for Walmart stores. This repository contains the complete workflow—from data exploration and feature engineering in a Jupyter Notebook to a final, interactive web application built with Streamlit.

> 💡 The core of the project is an **XGBoost Regressor model**, which outperformed a deep learning model on this structured dataset, achieving a high R² score.


---

## 🌟 Features

- 🎯 **Dynamic Forecasting**: Adjust parameters like Store ID, Date, Fuel Price, CPI, etc. and get real-time predictions.
- 📊 **Confidence Score**: A gauge meter reflects model reliability using the R² score.
- 🌊 **Feature Contribution**: Waterfall chart to show how each feature influences the final prediction.
- 🔍 **What-If Analysis**: Test scenarios by adjusting input features to see their impact on sales.
- 🧭 **Contextual Gauges**: See if your inputs are within low, average, or high range vs training data.
- 📈 **Model Insights**: View residual plots, prediction errors, and model diagnostics.

---

## 🛠️ Tech Stack

| Category              | Tools & Libraries                                 |
|-----------------------|---------------------------------------------------|
| **Data Processing**   | Pandas, NumPy                                     |
| **Visualization**     | Matplotlib, Seaborn, Plotly                       |
| **Machine Learning**  | Scikit-learn, XGBoost, TensorFlow (experimental)  |
| **Web App**           | Streamlit                                         |
| **Dev Environment**   | JupyterLab, Python 3.9+                           |

---

## 📁 Project Structure

```bash
.
├── input/
│   ├── features.csv
│   ├── stores.csv
│   ├── train.csv
│   └── test.csv
├── .ipynb    # Jupyter Notebook for model training
├── app.py                  # Main Streamlit web app
├── style.css                         # Custom CSS styling
├── requirements.txt                  # Dependencies
├── xgb_model.joblib                  # Trained XGBoost model
├── scaler.joblib                     # Feature scaler
└── training_columns.joblib           # Column names used in training

```

## ⚙️ Installation & Setup
**1. Clone the Repository** 
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```


