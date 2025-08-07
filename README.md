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
**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/Mac
source venv/bin/activate
```
**3. Install Dependencies**
**Method 1:** From requirements.txt
```bash
pip install -r requirements.txt
```
**Method 2:** Manual Installation
```bash
# Core Data Stack
pip install pandas numpy scipy

# Visualization
pip install matplotlib seaborn plotly

# ML & Modeling
pip install scikit-learn xgboost tensorflow

# Jupyter Environment
pip install jupyterlab ipykernel

```
## ▶️ How to Run the Project
**1. Run the Jupyter Notebook** 
```bash
jupyter lab
```
Then open .ipynb to walk through data analysis and model training.

**2. Launch the Streamlit App** 
```bash
streamlit run streamlit_app.py
```
Your browser will automatically open the app at http://localhost:8501.

🧠 Modeling & Results
Several models were evaluated:
| Model             | Performance (R² Score) | Notes                |
| ----------------- | ---------------------- | -------------------- |
| **XGBoost**       | **0.92**               | Best performing      |
| Deep Learning     | \~0.84                 | Good, but not better |
| Linear Regression | \~0.70                 | Poor generalization  |





