# Demand Intelligence & Sales Forecasting Dashboard

An end-to-end Sales Forecasting and Stocking Strategy Optimization system designed for retail inventory management. This application transitions a business from reactive stock management to proactive demand forecasting using advanced statistical, machine learning, and time-series models.

The live application is built with **Streamlit** and can be deployed directly to the web.

---

## 🚀 Key Features

1. **Interactive Sales Overview Dashboard**
   - Explore four years of historical sales transactions across categories (Technology, Furniture, Office Supplies) and geographic regions.
   - Live KPI tracking (Revenue, Orders, Avg Order Value, Profit) with custom multi-parameter filters.
   
2. **Three-Month Forecast Explorer**
   - High-fidelity time series forecasting comparisons.
   - Interactive forecasting dropdowns to drill down by **Product Category** or **Geographic Region**.
   - Model options: **Facebook Prophet**, **SARIMA (1,1,1)(1,1,1)₁₂**, and **XGBoost Regressor**.

3. **Fulfillment Anomaly Report**
   - Detects operational sales surges and deviations using a dual-method approach: **Isolation Forest (Machine Learning)** and **Z-Score Rolling Standard Deviation (Statistical)**.
   - Flags high-confidence anomaly weeks (where both models agree) to isolate events like Black Friday, year-end procurement, and school district cycles.

4. **Product Demand Segmentation & Stocking Strategy**
   - Automatically segments the 17 product sub-categories into 4 demand clusters using **K-Means Clustering** based on total volume, volatility, average order value, and year-over-year growth.
   - Outlines actionable inventory stocking strategies (EOQ models, safety stock adjustments, Just-In-Time replenishment) tailored to each cluster's unique pattern.

5. **Executive PDF Business Report**
   - Compiles all business insights, forecast projections, detected anomalies, and cluster strategies into a professional typeset executive PDF (`summary.pdf`) utilizing **ReportLab**.

---

## 📊 Model Performance Comparison

The models were evaluated on a 3-month holdout test horizon:

| Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Absolute Percentage Error (MAPE) |
| :--- | :--- | :--- | :--- |
| **🏆 Facebook Prophet** | **$7,429** | **$8,103** | **11.2%** |
| **SARIMA** | $9,410 | $10,880 | 14.5% |
| **XGBoost** | $12,150 | $14,920 | 18.9% |

*Facebook Prophet is configured as the recommended production forecasting model due to its low error margins and robust seasonality handling.*

---

## 📁 Repository Structure

```text
├── charts/                   # Exported visual analysis and forecast plots
├── app.py                    # Streamlit dashboard application source code
├── train.csv                 # Historical retail transaction sales dataset
├── summary.pdf               # Executive Business Report in PDF format
├── requirements.txt          # Python library dependencies
└── README.md                 # Project documentation
```

---

## 🛠️ Local Installation & Running Guide

### Prerequisites
- Python 3.10 or 3.11 (Recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/VenkataTejaP9587/SalesForecasting.git
cd SalesForecasting
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application
```bash
streamlit run app.py
```
This will start a local web server, and the dashboard will automatically open at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

To host this dashboard online:
1. Push this repository to your GitHub account.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and point it to this repository (`VenkataTejaP9587/SalesForecasting`), branch `main`, and file `app.py`.
4. **Important Settings:** Go to **Advanced Settings** and set the **Python version to 3.11** or **3.12** to ensure standard wheel compatibility for Prophet and compiler binaries, then click **Reboot**.